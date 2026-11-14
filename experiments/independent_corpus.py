"""独立语料泛化重跑（2026-11-13）：修复 docs/REVIEW.md 第 7 节留下的证据缺口。

背景
----
第八轮把 ``_HELDOUT_VAL_CORPUS`` 的 ``val PPL 4.29`` 降级为「同分布重组句迁移」，
因为验证集 k=4 上下文 90.2% 在训练集出现过。本实验换用
``qml_lab.corpus_independent`` 中句级 disjoint、主题分离的新语料，
用**与 diagnose_llm.py 完全相同的两组配置和 5 个 seed** 重跑量子层与其经典替身：

* ``caus_q_rich48``  — 4 qubits, 48-Pauli readout, causal k=4, MLP head
* ``caus_classical`` — 同结构经典 Linear48+Tanh 替身

除语料外，qubits / seq_len / batch / steps / lr / seeds / head / 初始化全部不变，
因此两臂之间、以及「同组旧语料 → 新语料」之间的差值都可解释。

诚实边界
--------
1. 字符级英文文本不可能零共享 4-gram，本实验报告逐 k 覆盖率，不宣称完全 OOD。
2. n=5 的配对 CI 未做多重比较校正；本报告只有 2 个主检验（train / val），
   并同时给出 Bonferroni 校正后的判定。
3. 仍是 4-qubit 状态向量模拟的玩具字符 LM，不是可用 LLM。

用法：
    python experiments/independent_corpus.py --quick     # 冒烟，不写 docs/
    python experiments/independent_corpus.py             # 正式 5-seed，归档 docs/
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

from experiments.diagnose_llm import (  # noqa: E402
    _build_model_from_spec,
    _group_specs,
    _paired_ci,
    _tensor_sha256,
)
from qml_lab.data import (  # noqa: E402
    build_char_data,
    corpus_entropy_floors,
    iter_batches,
    load_independent_split,
    corpus_overlap,
)
from qml_lab.corpus_independent import word_bigram_novelty  # noqa: E402
from qml_lab.train import train_lm  # noqa: E402
from qml_lab.utils import count_parameters, ensure_dir  # noqa: E402

GROUP_IDS = ("caus_q_rich48", "caus_classical")
CAVEAT = (
    "本实验是 4-qubit 状态向量模拟的玩具级字符 LM；新语料只是句级 disjoint + 主题分离，"
    "字符级英文仍共享功能 4-gram，val loss 应读作「主题外新句迁移」而非严格 OOD 泛化。"
    "配对 CI 为 n=5 双侧 95% Student-t，另报 Bonferroni(2 检验) 校正判定。"
)


def _specs(qubits: int = 4) -> list[dict[str, Any]]:
    """直接复用 diagnose_llm 的规格表，保证配置逐字一致。"""
    all_specs = _group_specs(qubits=qubits)
    return [s for s in all_specs if s["id"] in GROUP_IDS]


def _verify_weight_control(specs: list[dict[str, Any]], vocab_size: int,
                           seeds: list[int]) -> dict[str, Any]:
    """逐 seed 断言两臂共享模块（embed / causal_conv / head[0]）初始权重逐元素一致。

    依据项目方法学约定：qml.qnn.TorchLayer 用 torch 全局 RNG 初始化，
    必须在构造模型之前 set_seed；_build_model_from_spec 已保证这一点。
    """
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
    """子进程：单个 (group, seed) 训练 + 确定性全语料评估。"""
    torch.set_num_threads(1)
    train_text = task["train_text"]
    val_text = task["val_text"]
    train_data, vocab_size, stoi, _ = build_char_data(train_text)
    val_data, _, _, _ = build_char_data(val_text, stoi=stoi)

    model = _build_model_from_spec(task["spec"], vocab_size=int(vocab_size),
                                   seed=int(task["seed"]))
    gen = iter_batches(
        data_tensor=train_data,
        seq_len=int(task["seq_len"]),
        batch_size=int(task["batch_size"]),
        seed=int(task["seed"]),
    )
    hist = train_lm(
        model=model,
        batch_gen=gen,
        steps=int(task["steps"]),
        lr=float(task["lr"]),
        seed=int(task["seed"]),
        verbose=False,
        log_every=0,
        data_tensor=train_data,
        val_tensor=val_data,
        seq_len=int(task["seq_len"]),
        eval_every=int(task["eval_every"]),
    )
    return {
        "group_id": task["spec"]["id"],
        "seed": int(task["seed"]),
        "params": int(count_parameters(model)),
        "readout_dim": int(model.readout_dim),
        "eval_loss": float(hist["eval_loss"]),
        "eval_perplexity": float(hist["eval_perplexity"]),
        "val_loss": float(hist["val_loss"]),
        "val_perplexity": float(hist["val_perplexity"]),
        "final_loss": float(hist["final_loss"]),
        "total_seconds": float(hist["total_seconds"]),
        "eval_curve": [float(v) for v in hist.get("eval_loss_curve", [])],
        "val_curve": [float(v) for v in hist.get("val_loss_curve", [])],
    }


def _summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def stat(key: str) -> tuple[float, float]:
        vals = [float(r[key]) for r in runs]
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    out: dict[str, Any] = {}
    for key in ("eval_loss", "eval_perplexity", "val_loss", "val_perplexity",
                "total_seconds", "final_loss"):
        m, s = stat(key)
        out[f"{key}_mean"] = m
        out[f"{key}_std"] = s
    out["params"] = int(runs[0]["params"])
    out["readout_dim"] = int(runs[0]["readout_dim"])
    out["per_seed"] = {str(r["seed"]): {
        "eval_loss": r["eval_loss"],
        "val_loss": r["val_loss"],
        "val_perplexity": r["val_perplexity"],
        "seconds": r["total_seconds"],
    } for r in runs}
    return out


def _plot(report: dict[str, Any], out_path: Path) -> None:
    ks = [int(k) for k in report["corpus"]["val_context_covered_by_train"]]
    new_cov = [report["corpus"]["val_context_covered_by_train"][str(k)]["val_token_coverage"]
               for k in ks]
    legacy = report["legacy_corpus"]["val_context_covered_by_train"]
    old_cov = [legacy.get(str(k), {}).get("val_token_coverage", 0.0) for k in ks]

    g = report["groups"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    x = np.arange(len(ks))
    ax1.bar(x - 0.2, old_cov, 0.4, label="legacy held-out (k=4: 90.2%)",
            color="#c0392b", alpha=0.85)
    ax1.bar(x + 0.2, new_cov, 0.4, label="independent split (k=4: 38.7%)",
            color="#2e86c1", alpha=0.9)
    for i, (a, b) in enumerate(zip(old_cov, new_cov)):
        ax1.text(i - 0.2, a + 0.015, f"{a:.2f}", ha="center", fontsize=7.5)
        ax1.text(i + 0.2, b + 0.015, f"{b:.2f}", ha="center", fontsize=7.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"k={k}" for k in ks])
    ax1.set_ylabel("val context coverage by train")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Corpus overlap: legacy vs independent")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=8)

    labels = ["Q train", "C train", "Q val", "C val"]
    means = [
        g["caus_q_rich48"]["eval_loss_mean"], g["caus_classical"]["eval_loss_mean"],
        g["caus_q_rich48"]["val_loss_mean"], g["caus_classical"]["val_loss_mean"],
    ]
    errs = [
        g["caus_q_rich48"]["eval_loss_std"], g["caus_classical"]["eval_loss_std"],
        g["caus_q_rich48"]["val_loss_std"], g["caus_classical"]["val_loss_std"],
    ]
    colors = ["#2e86c1", "#7fb3d5", "#e67e22", "#f5b041"]
    bars = ax2.bar(range(4), means, 0.62, yerr=errs, capsize=4, color=colors, alpha=0.92)
    for b, m in zip(bars, means):
        ax2.text(b.get_x() + b.get_width() / 2, m + 0.035, f"{m:.3f}",
                 ha="center", va="bottom", fontsize=8.5)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("cross-entropy loss (nats)")
    ax2.set_ylim(0, max(means) + max(errs) + 0.5)
    ax2.set_title("Independent corpus: quantum vs classical (5 seeds)")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=140, format="png")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--eval-every", type=int, default=60)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    ap.add_argument("--quick", action="store_true", help="冒烟运行，不写 docs/")
    ap.add_argument("--no-archive", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.steps = 5
        args.seeds = [0, 1]
        args.eval_every = 0
        args.workers = 2

    train_text, val_text = load_independent_split()
    train_data, vocab_size, stoi, _ = build_char_data(train_text)
    val_data, _, _, _ = build_char_data(val_text, stoi=stoi)
    vocab_size = int(vocab_size)

    # ---- 语料统计 ----
    corpus = corpus_overlap(train_text, val_text)
    corpus["word_bigram_novelty"] = word_bigram_novelty(train_text, val_text)
    floors = corpus_entropy_floors(train_text, max_context=8, seq_len=args.seq_len)
    val_floors = corpus_entropy_floors(val_text, max_context=8, seq_len=args.seq_len)
    corpus["train_entropy_floors_nats"] = {
        str(k): round(floors[k]["loss"], 4) for k in (0, 1, 2, 4, 8) if k in floors
    }
    corpus["val_entropy_floors_nats"] = {
        str(k): round(val_floors[k]["loss"], 4) for k in (0, 1, 2, 4, 8) if k in val_floors
    }
    corpus["train_window16_floor_nats"] = round(floors[4]["window16_loss"], 4)
    print(f"[语料] train {len(train_text)} 字符 / val {len(val_text)} 字符 / 词表 {vocab_size}")
    print(f"[语料] k=4 覆盖率 {corpus['val_context_covered_by_train']['4']['val_token_coverage']:.4f}"
          f" (旧语料 {corpus['legacy_k4_coverage']:.4f})；"
          f"新 4-gram {corpus['val_new_4gram_ratio']:.4f}")
    print(f"[语料] train k=4 条件熵 {floors[4]['loss']:.4f} nats"
          f"（旧语料 0.1000），窗口16下界 {corpus['train_window16_floor_nats']:.4f}")

    # ---- 受控性断言 ----
    specs = _specs(qubits=args.qubits)
    weight_checks = _verify_weight_control(specs, vocab_size, args.seeds)
    print(f"[受控] 初始权重逐元素一致: {weight_checks['all_passed']}", flush=True)

    # ---- 训练 ----
    tasks = []
    for spec in specs:
        for seed in args.seeds:
            tasks.append({
                "spec": spec, "seed": int(seed),
                "train_text": train_text, "val_text": val_text,
                "steps": args.steps, "seq_len": args.seq_len,
                "batch_size": args.batch_size, "lr": args.lr,
                "eval_every": args.eval_every,
            })
    print(f"[训练] {len(specs)} 组 × {len(args.seeds)} seeds = {len(tasks)} 任务，"
          f"workers={args.workers}, steps={args.steps}", flush=True)
    raw: dict[str, list[dict[str, Any]]] = {s["id"]: [] for s in specs}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_task, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            raw[r["group_id"]].append(r)
            done += 1
            print(f"  [{done:>2}/{len(tasks)}] {r['group_id']:<15s} seed={r['seed']} "
                  f"train={r['eval_loss']:.4f} val={r['val_loss']:.4f} "
                  f"ppl={r['val_perplexity']:.2f} {r['total_seconds']:.1f}s", flush=True)

    groups = {}
    for gid, runs in raw.items():
        runs.sort(key=lambda r: r["seed"])
        groups[gid] = _summarise(runs)
        m = groups[gid]
        print(f"[汇总] {gid:<15s} train {m['eval_loss_mean']:.4f}±{m['eval_loss_std']:.4f} | "
              f"val {m['val_loss_mean']:.4f}±{m['val_loss_std']:.4f} "
              f"(ppl {m['val_perplexity_mean']:.3f}) | {m['total_seconds_mean']:.1f}s")

    # ---- 配对检验（量子 − 经典）----
    q = [r for r in raw["caus_q_rich48"]]
    c = [r for r in raw["caus_classical"]]
    q.sort(key=lambda r: r["seed"])
    c.sort(key=lambda r: r["seed"])
    paired = {
        "train_eval_loss": _paired_ci([r["eval_loss"] for r in q],
                                      [r["eval_loss"] for r in c]),
        "val_loss": _paired_ci([r["val_loss"] for r in q],
                               [r["val_loss"] for r in c]),
        "val_perplexity": _paired_ci([r["val_perplexity"] for r in q],
                                     [r["val_perplexity"] for r in c]),
        # 泛化间隙 (val - train)：检验量子层是否比经典层更容易过拟合
        "generalization_gap": _paired_ci(
            [r["val_loss"] - r["eval_loss"] for r in q],
            [r["val_loss"] - r["eval_loss"] for r in c],
        ),
    }
    from scipy import stats as _stats
    n_tests = len(paired)
    for name, p in paired.items():
        n = len(p["per_seed_diff"])
        se = p["std_diff"] / math.sqrt(n) if n > 1 else 0.0
        crit = float(_stats.t.ppf(1 - (0.05 / n_tests) / 2, df=n - 1)) if n > 1 else 0.0
        half = crit * se
        lo, hi = p["mean_diff"] - half, p["mean_diff"] + half
        p["bonferroni_alpha_per_test"] = round(0.05 / n_tests, 4)
        p["bonferroni_familywise_level"] = 0.95
        p["bonferroni_ci_low"] = lo
        p["bonferroni_ci_high"] = hi
        p["bonferroni_significant"] = bool(lo > 0.0 or hi < 0.0)
        print(f"[配对] {name:<18s} Δ={p['mean_diff']:+.4f} "
              f"95%CI [{p['ci95_low']:+.4f},{p['ci95_high']:+.4f}] "
              f"| Bonferroni 95%FWI CI [{lo:+.4f},{hi:+.4f}] "
              f"显著={p['bonferroni_significant']}")

    # ---- 旧语料并列数字（只读，不重跑）----
    legacy: dict[str, Any] = {}
    try:
        d = json.loads((ROOT / "docs" / "llm_diagnosis_metrics.json").read_text(encoding="utf-8"))
        legacy["corpus"] = d.get("config", {})
        legacy["groups"] = {
            gid: {k: v for k, v in d["groups"][gid].items()
                  if k.endswith("_mean") or k.endswith("_std")}
            for gid in GROUP_IDS
        }
        _pc = d["paired_comparisons"]["caus_q_rich48_minus_caus_classical"]
        legacy["paired_q_minus_c"] = {
            "train_eval_loss_mean_diff": _pc["train_eval_loss"]["mean_diff"],
            "heldout_val_loss_mean_diff": _pc["heldout_val_loss"]["mean_diff"],
            "heldout_val_ci95": [
                _pc["heldout_val_loss"]["ci95_low"],
                _pc["heldout_val_loss"]["ci95_high"],
            ],
        }
    except Exception as exc:
        legacy["error"] = repr(exc)
    legacy_corpus = {}
    try:
        ca = json.loads((ROOT / "docs" / "corpus_audit.json").read_text(encoding="utf-8"))
        legacy_corpus = ca.get("val_context_covered_by_train", {})
        legacy["k4_coverage"] = legacy_corpus.get("4", {}).get("val_token_coverage")
        legacy["train_k4_entropy_nats"] = 0.1000
    except Exception as exc:
        legacy["corpus_error"] = repr(exc)

    report = {
        "purpose": "修复 REVIEW 第 7 节证据缺口：句级 disjoint、主题分离语料上的 5-seed 受控重跑",
        "config": {
            "qubits": args.qubits, "seq_len": args.seq_len,
            "batch_size": args.batch_size, "steps": args.steps,
            "lr": args.lr, "seeds": args.seeds, "eval_every": args.eval_every,
            "quick": args.quick,
            "train_corpus_chars": len(train_text),
            "val_corpus_chars": len(val_text),
            "vocab_size": vocab_size,
            "groups": [s["id"] for s in specs],
        },
        "corpus": corpus,
        "legacy_corpus": {"val_context_covered_by_train": legacy_corpus,
                          "legacy_k4_coverage": 0.9024},
        "legacy_results": legacy,
        "weight_control_checks": weight_checks,
        "groups": groups,
        "generalization_gap_per_group": {
            gid: round(groups[gid]["val_loss_mean"] - groups[gid]["eval_loss_mean"], 4)
            for gid in GROUP_IDS
        },
        "paired_comparisons": paired,
        "sign_flip_vs_legacy": {
            "legacy_val_diff_q_minus_c": legacy.get("paired_q_minus_c", {}).get(
                "heldout_val_loss_mean_diff"),
            "independent_val_diff_q_minus_c": paired["val_loss"]["mean_diff"],
            "sign_flipped": bool(
                legacy.get("paired_q_minus_c", {}).get("heldout_val_loss_mean_diff", 0) < 0
                < paired["val_loss"]["mean_diff"]
            ),
            "note": (
                "旧语料（k=4 覆盖率 90.2%）上量子−经典的 val 差值为负（量子数值更好但不显著）；"
                "独立语料（38.7%）上差值转正。两次 CI 都跨 0，因此只能说『优势消失且方向反转』，"
                "不能宣称量子在独立语料上显著更差。"
            ),
        },
        "caveat": CAVEAT,
    }

    if not args.quick and not args.no_archive:
        out_json = ROOT / "docs" / "independent_corpus_metrics.json"
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        _plot(report, ROOT / "docs" / "figures" / "independent_corpus.png")
        print(f"archive -> {out_json}")
        print(f"figure  -> {ROOT / 'docs' / 'figures' / 'independent_corpus.png'}")
    print("INDEPENDENT CORPUS OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
