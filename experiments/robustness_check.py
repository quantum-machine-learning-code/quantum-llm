"""稳健性复核（2026-11-13，REVIEW 第 9 节）：第 8 节的结论是稳态还是欠拟合假象？

背景
----
第 8 节在句级 disjoint 语料（k=4 覆盖率 38.7%）上用 5 seed × 300 步得到：
量子层 train 显著更好（Δ=-0.1069），独立 val 上优势消失且方向反转（Δ=+0.0974，
CI 跨 0），**泛化间隙显著更大**（Δ=+0.2044，Bonferroni 后仍显著）。
但该结论有两个未排除的替代解释：

1. **300 步欠拟合**：量子线路优化更慢，300 步时可能还没收敛；泛化间隙更大
   有可能只是「训练不足」的瞬时现象，延长训练后会消失。
2. **单一语料依赖**：结论可能只对这一份自写英文成立。

本实验用两个互补阶段检验：

* Stage A（收敛性 + 多 seed）：同一份 ``independent`` 语料，20 seed × **1200 步**，
  ``eval_every=100`` 记录确定性全语料 train/val 曲线。一次训练同时给出
  ① 20 seed 下第 8 节结论是否稳健；② gap 随训练步数的轨迹（持续扩大 = 稳态
  过拟合倾向；收敛/反转 = 欠拟合假象）。
* Stage B（外部效度）：``synthetic`` 受控合成伪词语料（k=4 覆盖率 0.24%，
  字符 unigram JS 散度 0.2538，约为 A 的 27 倍），10 seed × 300 步，
  检验结论在极端分布偏移下是否复现。

方法学约束
----------
* 两臂规格、初始化、lr、seq_len、batch、seed 与第 8 节逐字一致；
  逐 seed 断言共享模块初始权重哈希相同（TorchLayer 的 torch 全局 RNG 顺序）。
* 检查点很多（Stage A 12 个 × 3 指标），因此预先指定 6 个 **primary endpoints**
  并做 Holm 校正；其余检查点只作为描述性轨迹，同时报告全部检验的
  Bonferroni 校正后显著数（REVIEW 第 15.3 节约定）。
* 不预设结论：判定逻辑写在 ``interpretation``，由实测数字决定。

资源提示
--------
每个 worker 是独立的 torch + PennyLane 进程（约 300–600 MB 常驻）。本机 31 GB
物理内存中 WSL ``vmmemWSL`` 常驻约 13 GB，因此默认只开 8 个 worker；本次正式运行
曾显式用 ``--workers 16`` 抢时间，峰值接近内存上限，不作为复现推荐值。
内存受限时用 ``--workers 4``，只影响墙钟时间，不改变任何数值结果。

用法：
    python experiments/robustness_check.py --quick
    python experiments/robustness_check.py                      # 正式，归档 docs/
    python experiments/robustness_check.py --stage a --no-archive
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from experiments.diagnose_llm import (  # noqa: E402
    _build_model_from_spec,
    _group_specs,
    _paired_ci,
    _tensor_sha256,
)
from qml_lab.data import (  # noqa: E402
    build_char_data,
    corpus_entropy_floors,
    corpus_overlap,
    iter_batches,
    load_split,
)
from qml_lab.corpus_independent import word_bigram_novelty  # noqa: E402
from qml_lab.train import train_lm  # noqa: E402
from qml_lab.utils import count_parameters, ensure_dir  # noqa: E402

GROUP_IDS = ("caus_q_rich48", "caus_classical")
METRICS = ("train", "val", "gap")
CAVEAT = (
    "本实验是 4-qubit 状态向量模拟的玩具级字符 LM。synthetic 语料是人工伪词、"
    "分布偏移远大于自然文本，只能用于压力测试，不能外推为自然语言泛化能力。"
    "跨语料绝对 loss 不可逐位比较（词表与任务难度不同）；比较只在同一语料内的"
    "配对差值上进行。延长训练未给两臂各自调参， lr 与第 8 节逐字一致。"
)


def _specs(qubits: int = 4) -> list[dict[str, Any]]:
    return [s for s in _group_specs(qubits=qubits) if s["id"] in GROUP_IDS]


def _verify_weight_control(specs: list[dict[str, Any]], vocab_size: int,
                           seeds: list[int]) -> dict[str, Any]:
    """逐 seed 断言两臂共享模块初始权重逐元素一致（TorchLayer RNG 顺序约定）。"""
    checks: dict[str, Any] = {"per_seed": {}, "all_passed": True}
    for seed in seeds:
        models = {
            s["id"]: _build_model_from_spec(s, vocab_size=vocab_size, seed=seed)
            for s in specs
        }
        q, c = models["caus_q_rich48"], models["caus_classical"]
        same = {
            "embed": _tensor_sha256(q.embed.weight) == _tensor_sha256(c.embed.weight),
            "causal_conv": (
                _tensor_sha256(q.causal_encoder.causal_conv.weight)
                == _tensor_sha256(c.causal_encoder.causal_conv.weight)
            ),
            "head_first_layer": (
                _tensor_sha256(q.head[0].weight) == _tensor_sha256(c.head[0].weight)
            ),
        }
        checks["per_seed"][str(seed)] = same
        if not all(same.values()):
            checks["all_passed"] = False
    if not checks["all_passed"]:
        raise AssertionError(f"受控初始权重断言失败: {checks['per_seed']}")
    return checks


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    """子进程：单个 (stage, group, seed) 训练，返回确定性全语料曲线。"""
    torch.set_num_threads(1)
    train_data, vocab_size, stoi, _ = build_char_data(task["train_text"])
    val_data, _, _, _ = build_char_data(task["val_text"], stoi=stoi)
    model = _build_model_from_spec(task["spec"], vocab_size=int(vocab_size),
                                   seed=int(task["seed"]))
    gen = iter_batches(
        data_tensor=train_data,
        seq_len=int(task["seq_len"]),
        batch_size=int(task["batch_size"]),
        seed=int(task["seed"]),
    )
    hist = train_lm(
        model=model, batch_gen=gen, steps=int(task["steps"]),
        lr=float(task["lr"]), seed=int(task["seed"]),
        verbose=False, log_every=0,
        data_tensor=train_data, val_tensor=val_data,
        seq_len=int(task["seq_len"]), eval_every=int(task["eval_every"]),
    )
    return {
        "stage": task["stage"],
        "group_id": task["spec"]["id"],
        "seed": int(task["seed"]),
        "params": int(count_parameters(model)),
        "readout_dim": int(model.readout_dim),
        "eval_steps": [int(v) for v in hist.get("eval_steps", [])],
        "eval_curve": [float(v) for v in hist.get("eval_loss_curve", [])],
        "val_curve": [float(v) for v in hist.get("val_loss_curve", [])],
        "eval_loss": float(hist["eval_loss"]),
        "val_loss": float(hist["val_loss"]),
        "val_perplexity": float(hist["val_perplexity"]),
        "final_loss": float(hist["final_loss"]),
        "total_seconds": float(hist["total_seconds"]),
    }


def _p_value(p: dict[str, Any]) -> float:
    n = len(p["per_seed_diff"])
    if n < 2 or p["std_diff"] <= 0:
        return 1.0
    t = abs(p["mean_diff"]) / (p["std_diff"] / math.sqrt(n))
    return float(2 * stats.t.sf(t, df=n - 1))


def _holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down：p 升序，一旦 p_i > alpha/(m-i) 则其后全部不拒绝。"""
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(order)
    out: dict[str, bool] = {}
    still_reject = True
    for i, (name, p) in enumerate(order):
        still_reject = still_reject and (p <= alpha / (m - i))
        out[name] = bool(still_reject)
    return out


def _analyse_stage(stage: str, raw: dict[str, list[dict[str, Any]]],
                   seeds: list[int]) -> dict[str, Any]:
    """把逐 seed 曲线对齐成逐检查点的配对检验。"""
    q = sorted(raw["caus_q_rich48"], key=lambda r: r["seed"])
    c = sorted(raw["caus_classical"], key=lambda r: r["seed"])
    if [r["seed"] for r in q] != seeds or [r["seed"] for r in c] != seeds:
        raise AssertionError(f"{stage}: seed 未对齐")
    checkpoints = q[0]["eval_steps"]
    for r in q + c:
        if r["eval_steps"] != checkpoints or len(r["val_curve"]) != len(checkpoints):
            raise AssertionError(f"{stage}: 检查点不一致 seed={r['seed']}")

    out: dict[str, Any] = {
        "checkpoints": checkpoints,
        "n_seeds": len(seeds),
        "points": {},
        "per_seed_final": {
            gid: {str(r["seed"]): {
                "train": r["eval_curve"][-1], "val": r["val_curve"][-1],
                "gap": r["val_curve"][-1] - r["eval_curve"][-1],
                "val_perplexity": r["val_perplexity"],
                "seconds": r["total_seconds"],
            } for r in runs}
            for gid, runs in (("caus_q_rich48", q), ("caus_classical", c))
        },
        "wallclock_seconds_mean": {
            gid: round(float(np.mean([r["total_seconds"] for r in runs])), 2)
            for gid, runs in (("caus_q_rich48", q), ("caus_classical", c))
        },
    }
    for i, step in enumerate(checkpoints):
        qtr = [r["eval_curve"][i] for r in q]
        ctr = [r["eval_curve"][i] for r in c]
        qva = [r["val_curve"][i] for r in q]
        cva = [r["val_curve"][i] for r in c]
        qgap = [v - t for v, t in zip(qva, qtr)]
        cgap = [v - t for v, t in zip(cva, ctr)]
        pairs = {
            "train": _paired_ci(qtr, ctr),
            "val": _paired_ci(qva, cva),
            "gap": _paired_ci(qgap, cgap),
        }
        for p in pairs.values():
            p["p_value"] = _p_value(p)
        out["points"][str(step)] = {
            "step": step,
            "q_train_mean": round(float(np.mean(qtr)), 5),
            "c_train_mean": round(float(np.mean(ctr)), 5),
            "q_val_mean": round(float(np.mean(qva)), 5),
            "c_val_mean": round(float(np.mean(cva)), 5),
            "q_gap_mean": round(float(np.mean(qgap)), 5),
            "c_gap_mean": round(float(np.mean(cgap)), 5),
            "q_val_se": round(float(np.std(qva, ddof=1) / math.sqrt(len(qva))), 5),
            "c_val_se": round(float(np.std(cva, ddof=1) / math.sqrt(len(cva))), 5),
            "paired": pairs,
        }
    return out


def _plot(report: dict[str, Any], out_path: Path) -> None:
    a = report["stages"]["independent"]
    b = report["stages"]["synthetic"]
    steps = a["checkpoints"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))

    # (1) 泛化间隙轨迹
    qg = [a["points"][str(s)]["q_gap_mean"] for s in steps]
    cg = [a["points"][str(s)]["c_gap_mean"] for s in steps]
    ax = axes[0]
    ax.plot(steps, qg, "o-", color="#2e86c1", label="quantum (caus_q_rich48)")
    ax.plot(steps, cg, "s-", color="#e67e22", label="classical (caus_classical)")
    ax.axvline(300, ls="--", color="#888", lw=1)
    ax.text(300, ax.get_ylim()[1], " 300 steps (R8)", fontsize=7, va="top", color="#666")
    ax.set_xlabel("training steps")
    ax.set_ylabel("generalization gap  val - train (nats)")
    ax.set_title("Stage A: gap trajectory (20 seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (2) val loss 轨迹（看是否 U 型 / 是否收敛）
    qv = [a["points"][str(s)]["q_val_mean"] for s in steps]
    cv = [a["points"][str(s)]["c_val_mean"] for s in steps]
    qse = [a["points"][str(s)]["q_val_se"] for s in steps]
    cse = [a["points"][str(s)]["c_val_se"] for s in steps]
    ax = axes[1]
    ax.plot(steps, qv, "o-", color="#2e86c1", label="quantum val")
    ax.plot(steps, cv, "s-", color="#e67e22", label="classical val")
    ax.fill_between(steps, np.array(qv) - np.array(qse), np.array(qv) + np.array(qse),
                    color="#2e86c1", alpha=0.16)
    ax.fill_between(steps, np.array(cv) - np.array(cse), np.array(cv) + np.array(cse),
                    color="#e67e22", alpha=0.16)
    ax.axvline(300, ls="--", color="#888", lw=1)
    ax.set_xlabel("training steps")
    ax.set_ylabel("independent val loss (nats)")
    ax.set_title("Stage A: does quantum catch up?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (3) 跨阶段 gap 差值（量子 − 经典）+ 95% CI
    def gp(stage, step):
        return stage["points"][str(step)]["paired"]["gap"]

    a300 = min(a["checkpoints"], key=lambda s: abs(s - 300))
    afinal = a["checkpoints"][-1]
    bfinal = b["checkpoints"][-1]
    sel = [(f"A @{a300}", gp(a, a300)), (f"A @{afinal}", gp(a, afinal)),
           (f"B @{bfinal}", gp(b, bfinal))]
    ax = axes[2]
    means = [p["mean_diff"] for _, p in sel]
    errs = [p["ci95_half"] for _, p in sel]
    colors = ["#7fb3d5", "#2e86c1", "#8e44ad"]
    bars = ax.bar(range(3), means, 0.6, yerr=errs, capsize=5,
                  color=colors, alpha=0.92)
    for bar, m, p in zip(bars, means, [p for _, p in sel]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + (0.02 if m >= 0 else -0.045),
                f"{m:+.3f}{'*' if p.get('holm_significant') else ''}",
                ha="center", fontsize=8.5)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for n, _ in sel], fontsize=9)
    ax.set_ylabel("paired gap difference (nats)")
    ax.set_title("Quantum - classical gap  (* Holm sig.)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=140, format="png")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("a", "b", "both"), default="both")
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--a-seeds", type=int, default=20)
    ap.add_argument("--b-seeds", type=int, default=10)
    ap.add_argument("--a-steps", type=int, default=1200)
    ap.add_argument("--b-steps", type=int, default=300)
    ap.add_argument("--a-eval-every", type=int, default=100)
    ap.add_argument("--b-eval-every", type=int, default=60)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                    help="并行 worker 数；每个 torch/PennyLane 进程约 300-600 MB，"
                         "内存紧张或 WSL vmmem 占用高时请调低（只影响耗时，不改数值）")
    ap.add_argument("--quick", action="store_true", help="冒烟，不写 docs/")
    ap.add_argument("--no-archive", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.a_seeds = args.b_seeds = 2
        args.a_steps = args.b_steps = 10
        args.a_eval_every = args.b_eval_every = 5
        args.workers = 2
        args.no_archive = True

    stage_cfg: dict[str, dict[str, Any]] = {}
    if args.stage in ("a", "both"):
        stage_cfg["independent"] = {
            "which": "independent", "seeds": list(range(args.a_seeds)),
            "steps": args.a_steps, "eval_every": args.a_eval_every,
        }
    if args.stage in ("b", "both"):
        stage_cfg["synthetic"] = {
            "which": "synthetic", "seeds": list(range(args.b_seeds)),
            "steps": args.b_steps, "eval_every": args.b_eval_every,
        }

    corpora: dict[str, Any] = {}
    specs = _specs(qubits=args.qubits)
    stage_texts: dict[str, tuple[str, str]] = {}
    for name, cfg in stage_cfg.items():
        tr, va = load_split(cfg["which"])
        stage_texts[name] = (tr, va)
        ov = corpus_overlap(tr, va)
        ov["word_bigram_novelty"] = word_bigram_novelty(tr, va)
        fl = corpus_entropy_floors(tr, max_context=8, seq_len=args.seq_len)
        ov["train_entropy_floors_nats"] = {
            str(k): round(fl[k]["loss"], 4) for k in (0, 1, 2, 4, 8) if k in fl
        }
        corpora[name] = ov
        print(f"[{name}] train {len(tr)} / val {len(va)} 字符 | k=4 覆盖 "
              f"{ov['val_context_covered_by_train']['4']['val_token_coverage']:.4f} | "
              f"JS {ov['char_distribution_divergence']['js_unigram']:.4f} | "
              f"新 4-gram {ov['val_new_4gram_ratio']:.4f}", flush=True)

    weight_checks: dict[str, Any] = {}
    for name, cfg in stage_cfg.items():
        _, vocab_size, _, _ = build_char_data(stage_texts[name][0])
        weight_checks[name] = _verify_weight_control(specs, int(vocab_size), cfg["seeds"])
        print(f"[受控] {name} 初始权重一致: {weight_checks[name]['all_passed']}", flush=True)

    # ---- 训练 ----
    tasks: list[dict[str, Any]] = []
    for name, cfg in stage_cfg.items():
        tr, va = stage_texts[name]
        for spec in specs:
            for seed in cfg["seeds"]:
                tasks.append({
                    "stage": name, "spec": spec, "seed": seed,
                    "train_text": tr, "val_text": va,
                    "steps": cfg["steps"], "eval_every": max(1, cfg["eval_every"]),
                    "seq_len": args.seq_len, "batch_size": args.batch_size,
                    "lr": args.lr,
                })
    print(f"[训练] {len(tasks)} 任务 | workers={args.workers} | "
          + " | ".join(f"{n}: {c['steps']}步×{len(c['seeds'])}seed×2组"
                       for n, c in stage_cfg.items()), flush=True)
    raw: dict[str, dict[str, list[dict[str, Any]]]] = {
        n: {g: [] for g in GROUP_IDS} for n in stage_cfg
    }
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_task, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            raw[r["stage"]][r["group_id"]].append(r)
            done += 1
            print(f"  [{done:>2}/{len(tasks)}] {r['stage']:<11s} {r['group_id']:<15s} "
                  f"seed={r['seed']:>2} train={r['eval_loss']:.4f} "
                  f"val={r['val_loss']:.4f} {r['total_seconds']:.1f}s", flush=True)

    stages = {n: _analyse_stage(n, raw[n], cfg["seeds"])
              for n, cfg in stage_cfg.items()}

    # ---- 多重比较：primary endpoints 用 Holm，全部检验另报 Bonferroni ----
    primary: dict[str, float] = {}
    all_tests: dict[str, float] = {}
    for name, st in stages.items():
        for s in st["checkpoints"]:
            for m in METRICS:
                key = f"{name}@{s}:{m}"
                all_tests[key] = st["points"][str(s)]["paired"][m]["p_value"]
        # 预指定主检验：最接近 300 步与最终步的 gap / val
        near300 = min(st["checkpoints"], key=lambda s: abs(s - 300))
        for s in sorted({near300, st["checkpoints"][-1]}):
            for m in ("gap", "val"):
                key = f"{name}@{s}:{m}"
                primary[key] = st["points"][str(s)]["paired"][m]["p_value"]
    holm = _holm(primary)
    n_all = len(all_tests)
    for name, st in stages.items():
        for s in st["checkpoints"]:
            for m in METRICS:
                key = f"{name}@{s}:{m}"
                p = st["points"][str(s)]["paired"][m]
                p["is_primary"] = key in primary
                p["holm_significant"] = bool(holm.get(key, False))
                p["bonferroni_significant"] = bool(
                    p["p_value"] <= 0.05 / n_all
                )
    print(f"\n[多重比较] 全部检验 {n_all} 个，Bonferroni 后显著 "
          f"{sum(1 for v in all_tests.values() if v <= 0.05 / n_all)}/{n_all}；"
          f"主检验 {len(primary)} 个，Holm 后显著 "
          f"{sum(holm.values())}/{len(primary)}", flush=True)
    for k in sorted(primary, key=lambda x: primary[x]):
        print(f"  primary {k:<28s} p={primary[k]:.5f} Holm显著={holm[k]}", flush=True)

    for name, st in stages.items():
        print(f"\n=== {name} 检查点摘要 ===", flush=True)
        for s in st["checkpoints"]:
            pt = st["points"][str(s)]
            g = pt["paired"]["gap"]
            v = pt["paired"]["val"]
            print(f"  step {s:>4}: gap(Q−C) {g['mean_diff']:+.4f} "
                  f"CI[{g['ci95_low']:+.4f},{g['ci95_high']:+.4f}] "
                  f"p={g['p_value']:.4f} Holm={g['holm_significant']} | "
                  f"val Δ {v['mean_diff']:+.4f} | Qval {pt['q_val_mean']:.4f} "
                  f"Cval {pt['c_val_mean']:.4f}", flush=True)

    # ---- 与第 8 节归档的可复现性核对（seed 0-4, step 300）----
    repro: dict[str, Any] = {"checked": False}
    if "independent" in stages and not args.quick:
        try:
            old = json.loads((ROOT / "docs" / "independent_corpus_metrics.json")
                             .read_text(encoding="utf-8"))
            a = stages["independent"]
            ck = min(a["checkpoints"], key=lambda s: abs(s - 300))
            idx = a["checkpoints"].index(ck)
            diffs = []
            for gid in GROUP_IDS:
                runs = sorted(raw["independent"][gid], key=lambda r: r["seed"])
                for r in runs:
                    if str(r["seed"]) in old["groups"][gid]["per_seed"]:
                        old_v = old["groups"][gid]["per_seed"][str(r["seed"])]["val_loss"]
                        diffs.append(abs(r["val_curve"][idx] - old_v))
            repro = {
                "checked": True, "step": ck,
                "max_abs_diff": round(max(diffs), 8) if diffs else None,
                "n_compared": len(diffs),
                "identical": bool(diffs) and max(diffs) < 1e-6,
                "note": "旧归档为 300 步训练；新实验训练 1200 步，"
                        "第 300 步检查点应与之逐位一致（同配置同 seed）。",
            }
            print(f"\n[可复现] step={ck} 对比 seed 0-4: "
                  f"max|Δ|={repro['max_abs_diff']}, identical={repro['identical']}",
                  flush=True)
        except Exception as exc:
            repro = {"checked": False, "error": repr(exc)}

    # ---- 结构化判定（不预设结论）----
    interp: dict[str, Any] = {
        "primary_question": "量子层泛化间隙更大，是稳态过拟合、300 步欠拟合假象，"
                            "还是只对单一语料成立？",
    }
    if not args.quick and "independent" in stages and "synthetic" in stages:
        a, b = stages["independent"], stages["synthetic"]
        s300 = min(a["checkpoints"], key=lambda s: abs(s - 300))
        sf = a["checkpoints"][-1]
        g300 = a["points"][str(s300)]["paired"]["gap"]
        gf = a["points"][str(sf)]["paired"]["gap"]
        gb = b["points"][str(b["checkpoints"][-1])]["paired"]["gap"]
        qv = [a["points"][str(s)]["q_val_mean"] for s in a["checkpoints"]]
        trough = min(range(len(qv)), key=lambda i: qv[i])
        interp.update({
            "gap_at_300": round(g300["mean_diff"], 5),
            "gap_at_final": round(gf["mean_diff"], 5),
            "gap_holm_significant_at_300": g300["holm_significant"],
            "gap_holm_significant_at_final": gf["holm_significant"],
            "gap_synthetic": round(gb["mean_diff"], 5),
            "gap_holm_significant_synthetic": gb["holm_significant"],
            "gap_change_ratio": round(
                (gf["mean_diff"] - g300["mean_diff"])
                / max(abs(g300["mean_diff"]), 1e-9), 4),
            "quantum_val_min_at_step": a["checkpoints"][trough],
            "quantum_val_u_shape": bool(0 < trough < len(qv) - 1),
            "verdict": (
                "稳态：延长训练后量子层泛化间隙仍显著更大"
                if gf["holm_significant"] and gf["mean_diff"] > 0 else
                "欠拟合假象：延长训练后泛化间隙不再显著"
                if g300["holm_significant"] and not gf["holm_significant"] else
                "未达显著：需更多 seed / 调参"
            ),
            "corpus_dependence": (
                "跨语料复现" if gb["holm_significant"] and gb["mean_diff"] > 0
                else "结论对语料敏感：synthetic 上未复现"
            ),
        })

    report = {
        "purpose": "检验 REVIEW 第 8 节『量子层泛化间隙更大』是稳态过拟合、"
                   "300 步欠拟合假象，还是单一语料依赖",
        "config": {
            "qubits": args.qubits, "seq_len": args.seq_len,
            "batch_size": args.batch_size, "lr": args.lr,
            "workers": args.workers, "quick": args.quick,
            "stages": {n: {**{k: v for k, v in c.items() if k != "which"}}
                       for n, c in stage_cfg.items()},
            "groups": [s["id"] for s in specs],
        },
        "corpora": corpora,
        "weight_control_checks": weight_checks,
        "stages": stages,
        "multiple_comparisons": {
            "n_all_tests": n_all,
            "bonferroni_alpha_per_test": round(0.05 / n_all, 6),
            "bonferroni_significant_count": sum(
                1 for v in all_tests.values() if v <= 0.05 / n_all),
            "primary_endpoints": sorted(primary),
            "holm_significant_count": sum(holm.values()),
            "holm_significant": holm,
            "all_test_pvalues": all_tests,
        },
        "reproducibility_check": repro,
        "interpretation": interp,
        "caveat": CAVEAT,
    }

    if not args.quick and not args.no_archive:
        out_json = ROOT / "docs" / "robustness_metrics.json"
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        _plot(report, ROOT / "docs" / "figures" / "robustness_check.png")
        print(f"\narchive -> {out_json}")
        print(f"figure  -> {ROOT / 'docs' / 'figures' / 'robustness_check.png'}")
    print("ROBUSTNESS OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
