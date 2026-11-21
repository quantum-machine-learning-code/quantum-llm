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

# 与 diagnose_llm.py 的 caus_q_rich48 / caus_classical 逐项一致（显式标为 legacy_rich48）
GROUP_SPECS = {
    "add_q_rich48": {
        "transform_type": "quantum",
        "context_mode": "causal",
        "context_window": CONTEXT_WINDOW,
        "n_qubits": 4,
        "n_layers": 1,
        "observables": "legacy_rich48",
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
        "observables": "legacy_rich48",
        "reupload": False,
        "pre_scale": True,
        "near_identity_init": False,
        "scale": 1.0,
        "head": "mlp",
        "hidden": 48,
    },
}

# --------------------------------------------------------------------------- #
# 严格对照配置（2026-11-16，v3）：两臂 spec 除 transform_type 外逐项完全相同。
#
# 历史 GROUP_SPECS 混了三个变量：transform_type、scale(5.0 vs 1.0)、
# near_identity_init(True vs False)，且两臂中间层容量 12 vs 240；
# 因此历史「量子 vs 经典」差值不能归因于量子层。这里消除所有人为引入的
# 不一致，并把无法消除的结构性容量差异显式记录、显式声明为限制。
#
# v1（ni=True, norm=False）：字段一致但行为失衡。量子/经典初始输出 std
#   0.390 / 0.024（16×），经典臂 3/5 seed 卡在随机基线。
# v2（ni=True, norm=True）：想用 LayerNorm 校准，结果两臂一起退化到 0.0062。
#   实测 LayerNorm 沿 48 维标准化时，对量子臂有效增益 2.5×、对经典臂 40×，
#   根本不是对称操作（见 docs/matched_calibration.json 的装置诊断）。
# v3（本版，ni=False, norm=False）：两臂都退回各自框架的默认初始化，
#   不加任何只在单边生效的补丁。实测初始特征 std 0.258 / 0.286（差 10%）、
#   成对距离 0.687 / 0.908，是三者中唯一量级接近的起跑线；也不新增层或参数。
# --------------------------------------------------------------------------- #
MATCHED_GROUP_SPECS = {
    g: {
        **GROUP_SPECS["add_q_rich48"],
        "transform_type": t,
        "scale": "learnable",       # 两臂同初值、同为可训练标量，不再一边放大一边不放大
        "scale_init": 5.0,          # 可由 --scale-init 标定，两臂始终同值
        "near_identity_init": False,  # 两臂都不加小初始化补丁（v3 的关键修复）
        "readout_norm": False,       # 不再用增益不等的 LayerNorm（v2 已证伪）
        "n_layers": 1,              # 容量轴；两臂同值，量子臂按语义使用
    }
    for g, t in zip(GROUP_IDS, ("quantum", "classical"))
}


def assert_matched(specs: dict, label: str = "matched") -> dict:
    """硬约束：两臂除 transform_type 外必须逐项相等，否则直接拒绝跑实验。

    这是把方法学约定「量子层消融 = 换成同结构经典层、其余不变」写进代码，
    防止以后再靠手工看配置来保证对照干净。
    """
    a, b = (specs[g] for g in GROUP_IDS)
    keys = sorted(set(a) | set(b))
    diff = {k: (a.get(k, "<missing>"), b.get(k, "<missing>"))
            for k in keys
            if k != "transform_type" and a.get(k, "<missing>") != b.get(k, "<missing>")}
    if diff:
        raise AssertionError(f"[{label}] 两臂除 transform_type 外还有差异，拒绝运行: {diff}")
    if a["transform_type"] == b["transform_type"]:
        raise AssertionError(f"[{label}] 两臂 transform_type 相同，对照无意义")
    return {"label": label, "differing_keys": ["transform_type"],
           "other_keys_identical": True, "n_compared_keys": len(keys)}


def annotate_matched(report: dict) -> dict:
    """装置有效性检查：两臂都必须真的学动，否则任何「谁更好」都没有意义。

    与项目里「报 0 相变前先确认 train≥0.95」同一条方法学约定：
    报「两组无差异 / 有差异」之前，先确认两组都离开了随机基线。
    """
    agg = report.get("aggregate", {})
    learned = report.get("learned_compositional_generalization", {})
    runs = report.get("runs", [])
    base = report.get("interpretation", {}).get("random_baseline_accuracy", 1 / 112)
    per_arm = {}
    for g in GROUP_IDS:
        vals = [r["final"]["val_accuracy"] for r in runs if r["group"] == g]
        n_ok = sum(1 for v in vals if v > 2 * base)
        need = (min(len(vals), max(3, (4 * len(vals) + 4) // 5))
                if vals else 3)  # ≥80% 的 seed 都要离开基线
        per_arm[g] = {
            "val_mean": agg.get(g, {}).get("val_accuracy_mean"),
            "above_random": bool(learned.get(g, False)),
            "seeds_above_2x_baseline": n_ok,
            "n_seeds": len(vals), "required_seeds": need,
        }
    valid = all(v["above_random"]
                and v["seeds_above_2x_baseline"] >= v["required_seeds"]
                for v in per_arm.values())
    note = ("两臂均显著超过随机基线且多数 seed 都离开基线，装置有效，"
            "可用于比较两种变换实现" if valid else
            "至少一臂有多数 seed 停在随机基线附近，装置无效："
            "本归档只能用于诊断配置，不得据此比较两臂优劣，"
            "更不得写成「量子更好 / 无量子差异」")
    report.setdefault("matched_control", {})["device_validity"] = {
        "valid": valid, "per_arm": per_arm, "note": note,
        "rule": "两臂各自 ≥80% 的 seed 的 val 需超过 2×随机基线(1/112)，"
                "且二项检验显著，否则对照不产生信息",
    }
    if not valid and not str(report["interpretation"]["verdict"]).startswith("【装置无效"):
        report["interpretation"]["verdict"] = (
            "【装置无效，不得比较】" + report["interpretation"]["verdict"]
        )
    # 顺带把早期归档里的措辞与代码最新口径对齐（不改任何训练数据）
    c = report.get("caveat", "").replace(
        "三处人为不一致已移除", "两处人为不一致已移除（容量差为结构固有，见下）")
    specs = report.get("matched_control", {}).get("specs", {})
    has_norm = bool(specs) and all(s.get("readout_norm") for s in specs.values())
    has_ni = bool(specs) and all(s.get("near_identity_init") for s in specs.values())
    if has_ni and not has_norm:
        device_note = ("【v1 装置】两臂小初始化补丁造成初始输出 std 0.390/0.024（16×），"
                       "字段一致但行为失衡，经典臂多数 seed 学不动")
    elif has_ni and has_norm:
        device_note = ("【v2 装置】LayerNorm 沿 48 维标准化对两臂有效增益约 2.5×/40×，"
                       "不是对称操作，两臂一起退化到随机基线")
    else:
        device_note = ("【v3 装置】两臂均用各自框架默认初始化、不加校准补丁，"
                       "初始特征 std 0.258/0.286 量级接近")
    if "【v1 装置】" not in c and "【v2 装置】" not in c and "【v3 装置】" not in c:
        c += " " + device_note
    report["caveat"] = c
    if "【后注】本归档经" not in report["caveat"]:
        report["caveat"] += (
            " 【后注】本归档经 --annotate-from 补注装置有效性，训练数据未改动。")
    return report


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


def build_model(group: str, seed: int, mode: str = "trainable_circular",
                matched: bool = False,
                spec_overrides: dict | None = None) -> RichQuantumLM:
    """构建模型并按 ``mode`` 设置数字 embedding 的初始化 / 梯度开关。

    ``frozen_circular`` 与 ``trainable_circular`` 只差一个 requires_grad，
    初始权重逐位相同，因此两者的对比干净地隔离了「放开 embedding」这一件事。
    ``matched=True`` 使用 MATCHED_GROUP_SPECS：两臂仅差 transform_type。
    ``spec_overrides`` 供装置标定 / 容量扫描注入**两臂同值**的配置覆盖，
    调用方须自行保证覆盖项对两臂一致（见 matched_calibration.py）。
    """
    if mode not in EMBEDDING_MODES:
        raise ValueError(f"未知 embedding 模式 {mode!r}，可选 {EMBEDDING_MODES}")
    set_seed(seed)
    specs = MATCHED_GROUP_SPECS if matched else GROUP_SPECS
    spec = dict(specs[group])
    if spec_overrides:
        spec.update(spec_overrides)
    n_qubits = int(spec.pop("n_qubits", 4))
    n_layers = int(spec.pop("n_layers", 1))
    model = RichQuantumLM(vocab_size=VOCAB_SIZE, n_qubits=n_qubits,
                          n_layers=n_layers, **spec)
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
    matched = bool(cfg.get("matched", False))
    model = build_model(group, seed, mode, matched=matched,
                        spec_overrides=cfg.get("spec_overrides"))
    embed_init = model.embed.weight.detach().clone()
    with torch.no_grad():
        z0 = model.bottleneck_features(xtr[:16])
        zn = model.readout_norm(z0) if model.readout_norm is not None else z0

    def _stats(t: torch.Tensor) -> dict:
        return {"mean": float(t.mean()), "std": float(t.std()),
                "min": float(t.min()), "max": float(t.max())}

    init_output_stats = {
        "raw": _stats(z0), "into_head": _stats(zn), "shape": list(z0.shape),
        "readout_norm": model.readout_norm is not None,
    }
    init_hashes = {
        "embed": _tensor_hash(model.embed.weight),
        "causal_conv_weight": _tensor_hash(model.causal_encoder.causal_conv.weight),
        "head_weight": _tensor_hash(model.head[-1].weight)
        if hasattr(model.head, "__getitem__")
        else _tensor_hash(model.head.weight),
        "out_scale": _tensor_hash(model.out_scale.detach()),
    }
    if model.readout_norm is not None:
        init_hashes["readout_norm_weight"] = _tensor_hash(model.readout_norm.weight)
    n_params_by_component = {
        "embed": sum(p.numel() for p in model.embed.parameters()),
        "causal_conv": sum(p.numel() for p in model.causal_encoder.causal_conv.parameters()),
        "transform": sum(p.numel() for p in model.transform.parameters()),
        "head": sum(p.numel() for p in model.head.parameters()),
        "out_scale": model.out_scale.numel(),
        "readout_norm": (sum(p.numel() for p in model.readout_norm.parameters())
                         if model.readout_norm is not None else 0),
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
        "n_params_by_component": n_params_by_component,
        "init_output_stats": init_output_stats,
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
    ap.add_argument("--no-figure", action="store_true", help="只归档 JSON，不画图")
    ap.add_argument("--matched", action="store_true",
                    help="严格对照：两臂除 transform_type 外配置逐项相同（含硬断言）")
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
    ap.add_argument("--scale-init", type=float, default=None,
                    help="覆盖两臂共同的可学习 out_scale 初值（装置标定用）")
    ap.add_argument("--n-layers", type=int, default=None,
                    help="覆盖两臂共同的 n_layers（容量匹配扫描用；量子臂按语义使用）")
    ap.add_argument("--near-identity-init", action="store_true",
                    help="复现 v1：两臂都开小初始化补丁（诊断用，非默认）")
    ap.add_argument("--readout-norm", action="store_true",
                    help="复现 v2：两臂都开 LayerNorm（诊断用，非默认）")
    ap.add_argument("--redraw-from", default="",
                    help="从归档 JSON 重画对比图，不重跑训练")
    ap.add_argument("--annotate-from", default="",
                    help="给已有 matched 归档补注装置有效性/口径说明，不重跑训练")
    args = ap.parse_args()

    if args.annotate_from:
        p = Path(args.annotate_from)
        rep = json.loads(p.read_text(encoding="utf-8"))
        annotate_matched(rep)
        p.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
        dv = rep["matched_control"]["device_validity"]
        print(f"annotated {p} | device_valid={dv['valid']} | {dv['note']}")
        return

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

    overrides: dict = {}
    if args.scale_init is not None:
        overrides["scale_init"] = float(args.scale_init)
    if args.n_layers is not None:
        overrides["n_layers"] = int(args.n_layers)
    if args.near_identity_init:
        overrides["near_identity_init"] = True
    if args.readout_norm:
        overrides["readout_norm"] = True

    specs = MATCHED_GROUP_SPECS if args.matched else GROUP_SPECS
    if overrides and args.matched:
        # 覆盖项对两臂同值，assert_matched 仍只应看到 transform_type 一个差异
        specs = {g: {**specs[g], **overrides} for g in GROUP_IDS}
    spec_check = (assert_matched(specs) if args.matched else
                  {"label": "legacy", "differing_keys": [
                      k for k in specs[GROUP_IDS[0]]
                      if specs[GROUP_IDS[0]].get(k) != specs[GROUP_IDS[1]].get(k)],
                   "other_keys_identical": False,
                   "n_compared_keys": len(specs[GROUP_IDS[0]])})
    if args.matched:
        print(f"[matched] 两臂仅差 transform_type；其余 "
              f"{spec_check['n_compared_keys'] - 1} 项已断言逐项相同", flush=True)

    mode = "trainable_random" if args.no_circular_embedding else args.embedding_mode
    cfg = {
        "matched": bool(args.matched),
        "group_specs": {g: dict(specs[g]) for g in GROUP_IDS},
        "spec_check": spec_check,
        "spec_overrides": overrides,
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
        keys = ("embed", "causal_conv_weight", "head_weight")
        if args.matched:
            keys += ("out_scale", "readout_norm_weight")
        ok = all(
            h[GROUP_IDS[0]].get(k, "<missing>") == h[GROUP_IDS[1]].get(k, "<missing>")
            for k in keys
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

    if args.matched:
        by_comp = {
            g: {c: next(r["n_params_by_component"][c] for r in runs if r["group"] == g)
                for c in ("embed", "causal_conv", "transform", "head",
                          "out_scale", "readout_norm")}
            for g in GROUP_IDS
        }
        init_stats = {
            g: next(r["init_output_stats"] for r in runs if r["group"] == g)
            for g in GROUP_IDS
        }
        report["matched_control"] = {
            "specs": {g: dict(specs[g]) for g in GROUP_IDS},
            "spec_check": spec_check,
            "n_params_by_component": by_comp,
            "transform_param_gap": abs(by_comp[GROUP_IDS[0]]["transform"]
                                       - by_comp[GROUP_IDS[1]]["transform"]),
            "initial_output_stats": init_stats,
            "unresolved_structural_difference": (
                "中间层参数量不等：量子线路 12 个旋转角 vs 经典 Linear(4,48) 240 个权重。"
                "这是「4 维输入映射到 48 维读出」这一接口下两种参数化的结构固有差异，"
                "无法在保持同输入/同输出维度的前提下消除；因此本对照测量的是"
                "「两种变换实现」的差值，仍不是纯粹的量子层因果效应。"
                "要消除它需做容量匹配扫描（量子臂 n_layers 递增 / 经典臂宽度递减）。"
            ),
        }
        report["interpretation"]["verdict"] = (
            f"严格对照（两臂仅差 transform_type，其余 {spec_check['n_compared_keys'] - 1} "
            f"项断言相同）：量子 {agg[GROUP_IDS[0]]['val_accuracy_mean']:.4f}±"
            f"{agg[GROUP_IDS[0]]['val_accuracy_std']:.4f} vs 经典 "
            f"{agg[GROUP_IDS[1]]['val_accuracy_mean']:.4f}±"
            f"{agg[GROUP_IDS[1]]['val_accuracy_std']:.4f}；"
            f"Holm 后显著主检验 {n_holm_sig} 个。中间层容量 "
            f"{by_comp[GROUP_IDS[0]]['transform']} vs "
            f"{by_comp[GROUP_IDS[1]]['transform']} 的差异未消除，"
            "结论仅适用于「两种变换实现」的比较，不得表述为量子层因果效应。"
        )
        s0 = specs[GROUP_IDS[0]]
        scale_desc = (f"scale 为同初值 {s0.get('scale_init', 5.0):g} 的可学习标量"
                      if s0.get("scale") == "learnable"
                      else f"scale 固定 {s0.get('scale')}")
        init_desc = ("两臂均开小初始化补丁" if s0.get("near_identity_init")
                     else "两臂均用各自框架默认初始化（不加小初始化补丁）")
        norm_desc = ("两臂共用同一 LayerNorm" if s0.get("readout_norm")
                     else "不启用 LayerNorm")
        report["caveat"] = (
            f"【严格对照】两臂 spec 除 transform_type 外逐项相同：{scale_desc}、"
            f"{init_desc}、{norm_desc}、初始化/优化器/lr/batch/步数/seed/切分/"
            "评估口径全部一致。历史 GROUP_SPECS 的 scale 5.0 vs 1.0、"
            "near_identity_init True vs False 两处人为不一致已移除；"
            "v1/v2 装置（小初始化补丁 / LayerNorm）已证伪，见 "
            "docs/matched_calibration.json。"
            + CAVEAT + " " + report["matched_control"]["unresolved_structural_difference"]
        )
        verdict = report["interpretation"]["verdict"]

    if args.matched:
        annotate_matched(report)
        verdict = report["interpretation"]["verdict"]

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
        if not args.no_figure:
            _figure(report, figure_path)
            print(f"  archived: {metrics_path}\n  figure:   {figure_path}")
        else:
            print(f"  archived: {metrics_path} (按 --no-figure 跳过出图)")


if __name__ == "__main__":
    main()
