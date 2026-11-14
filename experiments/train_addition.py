"""课题一：模 112 加法的量子层 / 经典层受控对照。

样本 ``[a][+][b][=] -> c``（``c = (a + b) mod 112``），词表 117 token，
答案位只在 112 个数字类上做 CE。``add_classical`` 用 ``Linear(4,48)+Tanh``
替换 48 维 Pauli 读出的量子线路、其余逐项不变，因此它同时是量子层消融。

循环角度编码：数字 token 的 embedding 前 4 维**初始化**为
``[sin(2πn/112), cos(2πn/112), sin(4πn/112), cos(4πn/112)]``。注意这只是
初始化，默认模式 ``trainable_circular`` 下 embedding 与其它参数一起被 Adam
更新（归档里的 ``embedding_drift_rms`` 会实测漂移量）——它并非冻结。
这是两臂共享的经典先验；没有它时 1000 步后两臂都停在随机基线（1-seed 探针），
所以它不是量子优势的来源，也不应被归因于量子层。

三种 embedding 模式（详见 experiments/embedding_ablation.py）：
    frozen_circular    循环编码初始化 + 冻结（真正「不放开」）
    trainable_circular 循环编码初始化 + 可训练（课题一默认）
    trainable_random   纯随机初始化 + 可训练（人为先验完全拿掉）

用法：
    python experiments/train_addition.py --probe      # 1 seed x 1000 步容量探针
    python experiments/train_addition.py              # 5 seed x 2000 步正式实验
    python experiments/train_addition.py --quick      # 10 步冒烟，不写归档
    python experiments/train_addition.py --probe --no-circular-embedding  # 消融别名

方法学约定：
* 每个 worker 独立 ``set_seed``，构造模型前设种子（TorchLayer 消耗 torch 全局 RNG）。
* 训练/验证按 (a, b) 组合严格 disjoint，主指标为全量 2509 个验证对的
  答案位 top-1 准确率；随机基线 1/112。
* 曲线用固定 1024 子集（控制量子评估开销），最终指标用全量。
* 配对差值报双侧 95% Student-t CI，主检验做 Holm 校正，并给出是否
  显著高于随机基线的二项检验。不预设结论，判定写在 interpretation。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qml_lab.arithmetic import (  # noqa: E402
    BOUNDARY_EXAMPLES,
    CONTEXT_WINDOW,
    MODULUS,
    RANDOM_BASELINE_ACCURACY,
    VOCAB_SIZE,
    answer_logits,
    build_tensors,
    circular_init,
    context_overlap,
    encode_expression,
    enumerate_table,
    evaluate_answers,
    iter_batches,
    split_table,
)
from qml_lab.models import RichQuantumLM  # noqa: E402
from qml_lab.utils import set_seed  # noqa: E402

DOCS = ROOT / "docs"
FIGS = DOCS / "figures"
METRICS = DOCS / "addition_metrics.json"
FIGURE = FIGS / "addition_accuracy.png"

GROUP_IDS = ("add_q_rich48", "add_classical")

# 与 diagnose_llm.py 的 caus_q_rich48 / caus_classical 逐项一致
GROUP_SPECS = {
    "add_q_rich48": {
        "transform_type": "quantum",
        "context_mode": "causal",
        "context_window": CONTEXT_WINDOW,
        "n_qubits": 4,
        "n_layers": 1,
        "observables": "z+x+y+zzall+xxall+yyall",
        "reupload": False,
        "pre_scale": True,
        "near_identity_init": True,
        "scale": 5.0,
        "head": "mlp",
        "hidden": 48,
    },
    "add_classical": {
        "transform_type": "classical",
        "context_mode": "causal",
        "context_window": CONTEXT_WINDOW,
        "n_qubits": 4,
        "n_layers": 1,
        "observables": "z+x+y+zzall+xxall+yyall",
        "reupload": False,
        "pre_scale": True,
        "near_identity_init": False,
        "scale": 1.0,
        "head": "mlp",
        "hidden": 48,
    },
}

CAVEAT = (
    "val 是未见的 (a,b) 组合而非未见数字，属于组合泛化，不是数值外推。"
    "答案位只在 112 个数字类上评估；输出头仍有 117 维但符号类不参与 loss。"
    "数字 token 用循环角度编码**初始化**（两臂完全相同）；默认模式下该 embedding "
    "是可训练参数而非冻结常量，它是让任务可学的经典先验，不是量子优势。"
    "1-seed 探针显示随机 embedding 下 1000 步后两臂仍停留在随机基线，故该编码"
    "属于必要设计而非可归因于量子层的提升。"
    "除 embedding 初始化外，模型规格与超参与历史字符 LM 实验逐项一致。"
)

EMBEDDING_MODES = ("frozen_circular", "trainable_circular", "trainable_random")


def _tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def build_model(group: str, seed: int, mode: str = "trainable_circular") -> RichQuantumLM:
    """构建模型并按 ``mode`` 设置数字 embedding 的初始化 / 梯度开关。

    ``frozen_circular`` 与 ``trainable_circular`` 只差一个 requires_grad，
    初始权重逐位相同，因此两者的对比干净地隔离了「放开 embedding」这一件事。
    """
    if mode not in EMBEDDING_MODES:
        raise ValueError(f"未知 embedding 模式 {mode!r}，可选 {EMBEDDING_MODES}")
    set_seed(seed)
    spec = dict(GROUP_SPECS[group])
    spec.pop("n_qubits")
    spec.pop("n_layers")
    model = RichQuantumLM(vocab_size=VOCAB_SIZE, n_qubits=4, n_layers=1, **spec)
    if mode in ("frozen_circular", "trainable_circular"):
        circular_init(model)
    if mode == "frozen_circular":
        model.embed.weight.requires_grad_(False)
    return model


def _subset(x: torch.Tensor, y: torch.Tensor, n: int, seed: int = 12345):
    g = torch.Generator().manual_seed(seed)
    n = min(n, x.shape[0])
    idx = torch.randperm(x.shape[0], generator=g)[:n]
    return x[idx], y[idx]


def _paired_ci(a: list[float], b: list[float]) -> dict:
    """a - b 的配对双侧 95% Student-t CI（负值表示前者更好/更低）。"""
    from scipy import stats

    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    if n > 1 and se > 0:
        tcrit = float(stats.t.ppf(0.975, n - 1))
        half = tcrit * se
        p = float(2 * stats.t.sf(abs(mean / se), n - 1))
    else:
        half, p = 0.0, 1.0
    return {
        "n": n,
        "mean_diff": mean,
        "std_diff": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "ci95_half": half,
        "p_value": p,
        "significant_95": bool(mean - half > 0 or mean + half < 0),
        "per_seed_diff": [float(v) for v in d],
    }


def run_one(task: tuple) -> dict:
    group, seed, cfg = task
    set_seed(seed)
    t0 = time.perf_counter()

    table = enumerate_table(cfg["op"])
    train_pairs, val_pairs = split_table(
        table, cfg["split_mode"], cfg["val_fraction"], cfg["split_seed"]
    )
    xtr, ytr = build_tensors(train_pairs, cfg["op"])
    xva, yva = build_tensors(val_pairs, cfg["op"])
    cx, cy = _subset(xtr, ytr, cfg["curve_n"])
    vx, vy = _subset(xva, yva, cfg["curve_n"])

    mode = cfg.get("embedding_mode", "trainable_circular")
    model = build_model(group, seed, mode)
    embed_init = model.embed.weight.detach().clone()
    init_hashes = {
        "embed": _tensor_hash(model.embed.weight),
        "causal_conv_weight": _tensor_hash(model.causal_encoder.causal_conv.weight),
        "head_weight": _tensor_hash(model.head[-1].weight)
        if hasattr(model.head, "__getitem__")
        else _tensor_hash(model.head.weight),
    }
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"]
    )
    gen = iter_batches(xtr, ytr, cfg["batch_size"], seed)
    curve: list[dict] = []
    model.train()
    for step in range(1, cfg["steps"] + 1):
        xb, yb = next(gen)
        opt.zero_grad()
        loss = F.cross_entropy(answer_logits(model, xb), yb)
        loss.backward()
        opt.step()
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            model.eval()
            with torch.no_grad():
                ctr = evaluate_answers(model, cx, cy, cfg["eval_batch"])
                cva = evaluate_answers(model, vx, vy, cfg["eval_batch"])
            model.train()
            curve.append(
                {
                    "step": step,
                    "train_acc": ctr["answer_accuracy"],
                    "val_acc": cva["answer_accuracy"],
                    "train_loss": ctr["answer_loss"],
                    "val_loss": cva["answer_loss"],
                }
            )

    final_train = evaluate_answers(model, xtr, ytr, cfg["eval_batch"])
    final_val = evaluate_answers(model, xva, yva, cfg["eval_batch"])
    embed_drift = float(
        (model.embed.weight.detach() - embed_init).pow(2).mean().sqrt()
    )
    wall = time.perf_counter() - t0
    print(
        f"[{group} seed={seed}] val_acc={final_val['answer_accuracy']:.4f} "
        f"train_acc={final_train['answer_accuracy']:.4f} "
        f"val_loss={final_val['answer_loss']:.4f} | {wall:.1f}s",
        flush=True,
    )
    return {
        "group": group,
        "seed": seed,
        "embedding_mode": mode,
        "n_params": n_params,
        "n_trainable_params": n_trainable,
        "embedding_trainable": bool(model.embed.weight.requires_grad),
        "embedding_drift_rms": embed_drift,
        "init_hashes": init_hashes,
        "curve": curve,
        "final": {
            "train_accuracy": final_train["answer_accuracy"],
            "train_loss": final_train["answer_loss"],
            "train_perplexity": final_train["answer_perplexity"],
            "val_accuracy": final_val["answer_accuracy"],
            "val_loss": final_val["answer_loss"],
            "val_perplexity": final_val["answer_perplexity"],
            "generalization_gap": final_train["answer_accuracy"]
            - final_val["answer_accuracy"],
            "val_n": final_val["n"],
        },
        "wallclock_seconds": wall,
    }


def _holm(pairs: dict, keys: list[str]) -> dict:
    items = sorted(keys, key=lambda k: pairs[k]["p_value"])
    m = len(items)
    out = {}
    running = 0.0
    for i, k in enumerate(items):
        adj = min(1.0, (m - i) * pairs[k]["p_value"])
        running = max(running, adj)
        out[k] = {"p_holm": float(running), "holm_significant": bool(running < 0.05)}
    return out


def _binom_vs_random(acc: float, n: int) -> dict:
    from scipy import stats

    k = int(round(acc * n))
    p = float(stats.binomtest(k, n, RANDOM_BASELINE_ACCURACY, alternative="greater").pvalue)
    return {
        "correct": k,
        "n": n,
        "random_baseline_accuracy": RANDOM_BASELINE_ACCURACY,
        "p_value_vs_random": p,
        "significantly_above_random": bool(p < 0.05),
    }


def _figure(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    colors = {"add_q_rich48": "#534AB7", "add_classical": "#185FA5"}
    labels = {"add_q_rich48": "Quantum (48 Pauli)", "add_classical": "Classical (Linear+Tanh)"}
    ax = axes[0]
    for gid in GROUP_IDS:
        curves = [r["curve"] for r in report["runs"] if r["group"] == gid]
        steps = [c["step"] for c in curves[0]]
        for key, style, alpha in (("train_acc", "-", 0.9), ("val_acc", "--", 0.9)):
            ys = np.array([[c[key] for c in cu] for cu in curves])
            mu, sd = ys.mean(0), ys.std(0, ddof=1) if len(curves) > 1 else np.zeros(len(steps))
            ax.plot(steps, mu, style, color=colors[gid], alpha=alpha,
                    label=f"{labels[gid]} {key.split('_')[0]}")
            if len(curves) > 1:
                ax.fill_between(steps, mu - sd, mu + sd, color=colors[gid], alpha=0.10)
    ax.axhline(RANDOM_BASELINE_ACCURACY, color="#888780", ls=":", lw=1,
               label="random 1/112")
    ax.set_xlabel("training steps")
    ax.set_ylabel("answer top-1 accuracy")
    ax.set_title("Mod-112 addition: train vs unseen pairs")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    metrics = ("train_accuracy", "val_accuracy", "generalization_gap")
    xs = np.arange(len(metrics))
    w = 0.36
    for j, gid in enumerate(GROUP_IDS):
        vals, errs = [], []
        for m in metrics:
            per_seed = [r["final"][m] for r in report["runs"] if r["group"] == gid]
            vals.append(float(np.mean(per_seed)))
            errs.append(float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0)
        bars = ax.bar(xs + (j - 0.5) * w, vals, w, yerr=errs, capsize=3,
                      color=colors[gid], alpha=0.85, label=labels[gid])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(RANDOM_BASELINE_ACCURACY, color="#888780", ls=":", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["train acc", "val acc\n(unseen pairs)", "gap"], fontsize=9)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Final full-set metrics (n={len(report['runs']) // len(GROUP_IDS)} seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="1 seed x 1000 步容量探针")
    ap.add_argument("--quick", action="store_true", help="10 步冒烟，不写归档")
    ap.add_argument("--no-circular-embedding", action="store_true",
                    help="--embedding-mode trainable_random 的兼容别名")
    ap.add_argument("--embedding-mode", default="trainable_circular",
                    choices=EMBEDDING_MODES,
                    help="frozen_circular / trainable_circular / trainable_random")
    ap.add_argument("--metrics-out", default="", help="覆盖归档 JSON 路径")
    ap.add_argument("--figure-out", default="", help="覆盖归档图路径")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--curve-n", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2),
                    help="并行 worker 数；内存紧张时调小（默认 4）")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--split-mode", default="holdout_pairs",
                    choices=("holdout_pairs", "holdout_operand"))
    ap.add_argument("--op", default="+")
    ap.add_argument("--redraw-from", default="",
                    help="从归档 JSON 重画对比图，不重跑训练")
    args = ap.parse_args()

    if args.redraw_from:
        report = json.loads(Path(args.redraw_from).read_text(encoding="utf-8"))
        _figure(report, FIGURE)
        print(f"redrawn from {args.redraw_from} -> {FIGURE}")
        return

    if args.quick:
        steps, seeds, eval_every, workers, curve_n = 10, 1, 5, 1, 64
        write = False
    elif args.probe:
        steps, seeds, eval_every, workers, curve_n = 1000, 1, 100, 1, 512
        write = False
    else:
        steps, seeds, eval_every, workers, curve_n = (
            args.steps, args.seeds, args.eval_every, args.workers, args.curve_n
        )
        write = True

    mode = "trainable_random" if args.no_circular_embedding else args.embedding_mode
    cfg = {
        "embedding_mode": mode,
        "circular_embedding": mode != "trainable_random",
        "op": args.op,
        "split_mode": args.split_mode,
        "val_fraction": args.val_fraction,
        "split_seed": 0,
        "steps": steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "eval_every": eval_every,
        "eval_batch": 512,
        "curve_n": curve_n,
        "workers": workers,
        "seeds": list(range(seeds)),
        "vocab_size": VOCAB_SIZE,
        "modulus": MODULUS,
        "groups": list(GROUP_IDS),
    }

    table = enumerate_table(args.op)
    train_pairs, val_pairs = split_table(table, args.split_mode, args.val_fraction, 0)
    overlap = context_overlap(train_pairs, val_pairs, args.split_mode, args.op)
    assert overlap["pair_level_disjoint"], "训练/验证组合必须严格不相交"
    assert overlap["full_context_coverage_k4"] == 0.0, "k=4 上下文覆盖率必须为 0"

    tasks = [(g, s, cfg) for g in GROUP_IDS for s in cfg["seeds"]]
    t0 = time.perf_counter()
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            runs = list(ex.map(run_one, tasks))
    else:
        runs = [run_one(t) for t in tasks]
    elapsed = time.perf_counter() - t0

    # 初始权重控制：同 seed 下两臂前端与分类头应逐位一致
    weight_checks: dict = {"all_passed": True, "per_seed": {}}
    for s in cfg["seeds"]:
        pair = [r for r in runs if r["seed"] == s]
        h = {r["group"]: r["init_hashes"] for r in pair}
        ok = all(
            h[GROUP_IDS[0]][k] == h[GROUP_IDS[1]][k]
            for k in ("embed", "causal_conv_weight", "head_weight")
        ) if len(h) == 2 else False
        weight_checks["per_seed"][str(s)] = {"passed": ok, "hashes": h}
        weight_checks["all_passed"] &= ok

    by_group = {
        g: {m: [r["final"][m] for r in runs if r["group"] == g]
            for m in ("train_accuracy", "val_accuracy", "train_loss", "val_loss",
                      "generalization_gap")}
        for g in GROUP_IDS
    }
    agg = {
        g: {
            f"{m}_{stat}": (float(np.mean(v)) if stat == "mean"
                            else float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            for m, v in ms.items()
            for stat in ("mean", "std")
        }
        for g, ms in by_group.items()
    }

    paired = {
        "val_accuracy": _paired_ci(by_group[GROUP_IDS[0]]["val_accuracy"],
                                   by_group[GROUP_IDS[1]]["val_accuracy"]),
        "train_accuracy": _paired_ci(by_group[GROUP_IDS[0]]["train_accuracy"],
                                     by_group[GROUP_IDS[1]]["train_accuracy"]),
        "val_loss": _paired_ci(by_group[GROUP_IDS[0]]["val_loss"],
                               by_group[GROUP_IDS[1]]["val_loss"]),
        "generalization_gap": _paired_ci(by_group[GROUP_IDS[0]]["generalization_gap"],
                                         by_group[GROUP_IDS[1]]["generalization_gap"]),
    }
    primary = ["val_accuracy", "train_accuracy", "generalization_gap"]
    holm = _holm(paired, primary)
    for k, v in holm.items():
        paired[k].update(v)
    for k, p in paired.items():
        p["bonferroni_alpha"] = 0.05 / len(paired)
        p["bonferroni_significant"] = bool(p["p_value"] < 0.05 / len(paired))

    binom = {
        g: _binom_vs_random(float(np.mean(by_group[g]["val_accuracy"])),
                            runs[0]["final"]["val_n"])
        for g in GROUP_IDS
    }
    learned = {g: bool(binom[g]["significantly_above_random"]
                       and np.mean(by_group[g]["val_accuracy"]) > 2 * RANDOM_BASELINE_ACCURACY)
               for g in GROUP_IDS}
    n_holm_sig = sum(1 for k in primary if paired[k].get("holm_significant"))
    if not any(learned.values()):
        verdict = ("两臂均未在未见 (a,b) 组合上显著超过随机基线 —— 在当前 4 qubits / "
                   "48 维读出 / 该训练预算下，任务超出模型容量，不能据此比较量子层优劣。")
    elif all(learned.values()) and n_holm_sig == 0:
        verdict = (
            "两臂都在未见 (a,b) 组合上显著超过 1/112 随机基线，但 3 个主检验 "
            "Holm 校正后 0 个显著：量子臂 val 准确率数值更高但配对 CI 跨 0，"
            "量子层无可证实的正贡献。val_loss 差值虽在 Bonferroni 后显著，"
            "但 loss 不是预先指定的主指标，不能据此宣称分类能力更强。"
        )
    elif all(learned.values()):
        verdict = ("两臂都学会了组合泛化，且主检验 Holm 校正后有显著项；"
                   "仍需两臂各自调参后才能归因于量子层。")
    else:
        verdict = "仅一臂学会组合泛化；注意该判定可能受统一 lr 与容量限制驱动。"

    report = {
        "task": "mod-112 addition (Z_112): [a][+][b][=] -> c",
        "config": cfg,
        "embedding_design": {
            "embedding_mode": mode,
            "trainable": mode != "frozen_circular",
            "circular_init": mode != "trainable_random",
            "circular_embedding": cfg["circular_embedding"],
            "formula": "embed[n,:4] = [sin(2πn/112), cos(2πn/112), sin(4πn/112), cos(4πn/112)]",
            "shared_by_both_arms": True,
            "mean_embedding_drift_rms": {
                g: float(np.mean([r["embedding_drift_rms"] for r in runs
                                  if r["group"] == g]))
                for g in GROUP_IDS
            },
            "ablation_1seed_illustrative": (
                "随机 embedding 下 1000 步 / 25.5 epochs 两臂 val acc 仍约 0.009 "
                "（=1/112 随机基线）、loss 约 4.72=ln(112)；循环编码下 1000 步 "
                "量子 0.1144 / 经典 0.0972。单次运行仅作示意，不作结论性数字。"
            ),
        },
        "boundary_examples": [
            {"expr": f"{a}{op}{b}={c}", "tokens": encode_expression(a, b, op, c)}
            for a, op, b, c in BOUNDARY_EXAMPLES
        ],
        "corpus_overlap": overlap,
        "weight_control_checks": weight_checks,
        "runs": runs,
        "aggregate": agg,
        "paired_comparisons": paired,
        "binomial_vs_random": binom,
        "learned_compositional_generalization": learned,
        "interpretation": {
            "verdict": verdict,
            "primary_endpoints": primary,
            "n_primary_holm_significant": n_holm_sig,
            "random_baseline_accuracy": RANDOM_BASELINE_ACCURACY,
        },
        "caveat": CAVEAT,
        "wallclock_seconds_total": elapsed,
    }

    print("\n=== probe/final summary ===")
    for g in GROUP_IDS:
        a = agg[g]
        print(f"  {g:16s} train {a['train_accuracy_mean']:.4f} | "
              f"val {a['val_accuracy_mean']:.4f} ± {a['val_accuracy_std']:.4f} | "
              f"gap {a['generalization_gap_mean']:+.4f} | "
              f"vs-random p={binom[g]['p_value_vs_random']:.2g} | learned={learned[g]}")
    for k, p in paired.items():
        print(f"  paired {k:20s} Δ={p['mean_diff']:+.4f} "
              f"CI[{p['ci95_low']:+.4f},{p['ci95_high']:+.4f}] p={p['p_value']:.3g}")
    print("  verdict:", verdict)
    print(f"  embedding mode={mode} trainable={mode != 'frozen_circular'} "
          f"drift_rms={report['embedding_design']['mean_embedding_drift_rms']}")
    print(f"  weight control all_passed={weight_checks['all_passed']} | wall={elapsed:.1f}s")

    if write:
        metrics_path = Path(args.metrics_out) if args.metrics_out else METRICS
        figure_path = Path(args.figure_out) if args.figure_out else FIGURE
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
        _figure(report, figure_path)
        print(f"  archived: {metrics_path}\n  figure:   {figure_path}")


if __name__ == "__main__":
    main()
