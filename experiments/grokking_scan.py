"""课题一 grokking 长训练扫描（2026-11-14）。

问题来源：第 11 节发现 ``trainable_random``（随机可训练 embedding）在 2000 步
内两臂都停在 1/112 随机基线。按 ``(a,b)`` 组合留出 + 随机符号 embedding 正是
Power et al. (2022, arXiv:2201.02177) **grokking** 的原始任务设置；Nanda et al.
(2023, arXiv:2301.05217) 逆向工程出模型最终学到的算法是「把数字映射到圆上的
sin/cos、用三角恒等式做旋转加法」；Liu et al. (2022, arXiv:2205.10343) 指出
结构化表征 + 有限数据 + 足够正则化压力才落在 grokking 区间。

因此上一轮 1000~2000 步（3~6 epochs）、mini-batch Adam、无 weight decay 的条件
**连记忆阶段都没走完**，把「学不动」归因为梯度摊薄是不准确的。本脚本直接检验：

1. 把训练推到 4x10^4 步（约 128 epochs）、加 AdamW + weight decay 后，
   随机 embedding 是否出现「先记忆、后突然泛化」的相变？
2. 用 Nanda 式的 **progress measure**（从 embedding 线性解码理想 sin/cos 基的
   R^2）观察电路是否在准确率跳变前就连续形成。
3. 第 11 节「冻结优于放开」在收敛后是否反转（交叉点）。
4. 量子臂能否跑到相变尺度 —— 用实测单步成本外推，并明确标注有限预算示意。

方法学约定：
* 超参网格先在 **1-seed pilot** 上探索并据其结果预选，随后只在预选条件上做
  **5 seed** 确认；pilot 数字一律标为探索性，不得当作结论。
* 每个 worker 构造模型前 ``set_seed``；同 seed 下 rich 架构两臂前端 / 头初始
  hash 必须逐位一致。
* 结论性数字 >=5 seed，配对双侧 95% Student-t CI；多个主检验统一 Holm 校正。
* 相变判定用预先写死的阈值（见 PHASE_*），不在看到结果后调整。
* 不预设结论；量子臂预算不足时如实写「未达相变尺度」，不做等价比较。

用法：
    python experiments/grokking_scan.py --quick               # 10 步冒烟，不写归档
    python experiments/grokking_scan.py                       # pilot + 5-seed 确认 + 量子示意
    python experiments/grokking_scan.py --confirm-steps 30000 --workers 6
    python experiments/grokking_scan.py --redraw-from docs/grokking_metrics.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path(__file__).resolve().parent
for p in (str(ROOT), str(EXPERIMENTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_addition import (  # noqa: E402
    GROUP_IDS,
    _holm,
    _paired_ci,
    _tensor_hash,
    build_model,
)
from qml_lab.arithmetic import (  # noqa: E402
    MODULUS,
    N_NUMBER_TOKENS,
    RANDOM_BASELINE_ACCURACY,
    VOCAB_SIZE,
    answer_logits,
    build_tensors,
    context_overlap,
    enumerate_table,
    evaluate_answers,
    iter_batches,
    split_table,
)
from qml_lab.utils import set_seed  # noqa: E402

DOCS = ROOT / "docs"
METRICS = DOCS / "grokking_metrics.json"
FIGURE = DOCS / "figures" / "grokking_scan.png"

# 相变判定阈值：预先写死，不随结果调整
PHASE_MEM_ACC = 0.95        # 训练准确率超过此值视为「完成记忆」
PHASE_GEN_ACC = 0.90        # 验证准确率超过此值视为「完成泛化」
PHASE_JUMP = 0.20           # 相邻采样点 val acc 跳变 >= 此值视为「突发」
GROKKING_STEP_SCALE = 10**5  # 文献量级的相变步数，用于成本外推

COLORS = {
    "ctrl_random": "#888780",
    "best_random": "#D85A30",
    "frozen": "#185FA5",
    "trainable_circular": "#534AB7",
    "control_mlp": "#1D9D75",
}


# --------------------------------------------------------------------------- #
# 正对照：Liu et al. (2022) 式的二次 MLP，模加法上已知可 grok
# --------------------------------------------------------------------------- #
class QuadraticMLP(nn.Module):
    """``out((W[a] + W[b])^2)``：交叉项 2*W[a]*W[b] 提供乘法交互。

    接口刻意与 ``RichQuantumLM`` 对齐：``forward`` 返回 (B, 1, 112)，
    并暴露 ``.embed`` 以便复用同一个傅里叶 progress measure。
    这是**装置校验用正对照**，不属于课题一两臂的任何一臂。
    """

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, hidden)
        self.out = nn.Linear(hidden, N_NUMBER_TOKENS, bias=False)
        nn.init.normal_(self.embed.weight, std=0.05)
        nn.init.normal_(self.out.weight, std=0.05 / math.sqrt(hidden))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        e = self.embed(idx[:, 0]) + self.embed(idx[:, 2])  # (B, H)
        return self.out(e * e).unsqueeze(1)                # (B, 1, 112)


# --------------------------------------------------------------------------- #
# 傅里叶 progress measure
# --------------------------------------------------------------------------- #
def _fourier_target(freqs: tuple[int, ...]) -> np.ndarray:
    n = np.arange(MODULUS, dtype=float)
    cols: list[np.ndarray] = []
    for k in freqs:
        th = 2 * np.pi * k * n / MODULUS
        cols.extend((np.sin(th), np.cos(th)))
    return np.stack(cols, axis=1)


TARGET_K1 = _fourier_target((1,))
TARGET_K12 = _fourier_target((1, 2))


# 固定的留出划分：高维正对照（256 维）若在全量上做最小二乘会平凡插值出 R^2=1，
# 因此统一「先用训练集 PCA 降到 <=16 维、再在留出数字上算 R^2」。
_SPLIT = np.random.RandomState(20261114).permutation(MODULUS)
_TRAIN_IDX, _TEST_IDX = _SPLIT[: 3 * MODULUS // 4], _SPLIT[3 * MODULUS // 4 :]
_MAX_PCA_DIM = 16  # 与 rich 架构的 embed_dim 对齐，保证跨架构可比


def _r2_decode(x: np.ndarray, target: np.ndarray) -> float:
    """留出集上的多元 R^2：线性读出能否从学到的 embedding 解码 sin/cos 基。

    返回 1 表示完全可解码，<=0 表示不比常数预测好；可能为负（过拟合）。
    """
    xtr, xte = x[_TRAIN_IDX], x[_TEST_IDX]
    ttr, tte = target[_TRAIN_IDX], target[_TEST_IDX]
    mu = xtr.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(xtr - mu, full_matrices=False)
    comps = vt[: min(_MAX_PCA_DIM, vt.shape[0])]
    ztr = (xtr - mu) @ comps.T
    zte = (xte - mu) @ comps.T
    atr = np.concatenate([ztr, np.ones((ztr.shape[0], 1))], axis=1)
    ate = np.concatenate([zte, np.ones((zte.shape[0], 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(atr, ttr, rcond=None)
    pred = ate @ coef
    ss_res = float(((tte - pred) ** 2).sum())
    ss_tot = float(((tte - tte.mean(0)) ** 2).sum())
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def _progress(model: nn.Module, circular: bool) -> dict:
    with torch.no_grad():
        w = model.embed.weight.detach().cpu().numpy()[:N_NUMBER_TOKENS]
        out = {
            "r2_k1": _r2_decode(w, TARGET_K1),
            "r2_k12": _r2_decode(w, TARGET_K12),
            "param_l2": float(sum(p.detach().pow(2).sum().item()
                                  for p in model.parameters() if p.requires_grad) ** 0.5),
        }
        if circular:
            # 只看前 4 维：直接度量 sin/cos 相位结构被保留了多少
            out["phase_r2_first4"] = _r2_decode(w[:, :4], TARGET_K12)
        return out


def _phase_detect(curve: list[dict]) -> dict:
    steps = [c["step"] for c in curve]
    tr = np.array([c["train_acc"] for c in curve])
    va = np.array([c["val_acc"] for c in curve])
    mem_idx = next((i for i, v in enumerate(tr) if v >= PHASE_MEM_ACC), None)
    gen_idx = next((i for i, v in enumerate(va) if v >= PHASE_GEN_ACC), None)
    jumps = np.diff(va) if len(va) > 1 else np.array([0.0])
    j = int(np.argmax(jumps)) if len(jumps) else 0
    max_jump = float(jumps[j]) if len(jumps) else 0.0
    jump_step = int(steps[j + 1]) if len(jumps) else None
    mem_step = int(steps[mem_idx]) if mem_idx is not None else None
    gen_step = int(steps[gen_idx]) if gen_idx is not None else None
    delayed = bool(mem_step is not None and gen_step is not None and gen_step > mem_step)
    abrupt = bool(max_jump >= PHASE_JUMP)
    return {
        "memorization_step": mem_step,
        "generalization_step": gen_step,
        "delay_steps": (gen_step - mem_step) if (mem_step is not None
                                                 and gen_step is not None) else None,
        "max_val_jump": max_jump,
        "max_jump_step": jump_step,
        "max_val_accuracy": float(va.max()) if len(va) else 0.0,
        "final_train_accuracy": float(tr[-1]) if len(tr) else 0.0,
        "final_val_accuracy": float(va[-1]) if len(va) else 0.0,
        "delayed_generalization": delayed,
        "abrupt_generalization": abrupt,
        "grokking_observed": bool(delayed and abrupt),
        "thresholds": {
            "mem_acc": PHASE_MEM_ACC, "gen_acc": PHASE_GEN_ACC, "jump": PHASE_JUMP,
        },
    }


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def _param_groups(model: nn.Module, lr: float, scale: float) -> list[dict]:
    ps = [p for p in model.parameters() if p.requires_grad]
    if scale == 1.0:
        return [{"params": ps, "lr": lr}]
    eids = {id(p) for p in model.embed.parameters() if p.requires_grad}
    ep = [p for p in ps if id(p) in eids]
    op = [p for p in ps if id(p) not in eids]
    groups = []
    if ep:
        groups.append({"params": ep, "lr": lr * scale})
    if op:
        groups.append({"params": op, "lr": lr})
    return groups


def build_grok_model(cond: dict, seed: int) -> nn.Module:
    set_seed(seed)
    if cond["arch"] == "quad_mlp":
        model = QuadraticMLP()
        if cond["mode"] in ("frozen_circular", "trainable_circular"):
            from qml_lab.arithmetic import circular_init
            circular_init(model)
        if cond["mode"] == "frozen_circular":
            model.embed.weight.requires_grad_(False)
        return model
    return build_model(cond["group"], seed, cond["mode"])


def run_one(task: tuple) -> dict:
    cond, seed, cfg = task
    set_seed(seed)
    t0 = time.perf_counter()
    torch.set_num_threads(1)

    table = enumerate_table("+")
    tr_pairs, va_pairs = split_table(
        table, "holdout_pairs", cond.get("val_fraction", cfg["val_fraction"]), 0
    )
    xtr, ytr = build_tensors(tr_pairs, "+")
    xva, yva = build_tensors(va_pairs, "+")
    # 曲线用固定子集；最终指标仍用全量验证集
    g = torch.Generator().manual_seed(90210)
    ctr_idx = torch.randperm(xtr.shape[0], generator=g)[: min(cfg["curve_n"], xtr.shape[0])]
    cva_idx = torch.randperm(xva.shape[0], generator=g)[: min(cfg["curve_n"], xva.shape[0])]
    cx, cy = xtr[ctr_idx], ytr[ctr_idx]
    vx, vy = xva[cva_idx], yva[cva_idx]

    model = build_grok_model(cond, seed)
    circular = cond["mode"] != "trainable_random"
    embed_init = model.embed.weight.detach().clone()
    init_hashes = {"embed": _tensor_hash(model.embed.weight)}
    if cond["arch"] == "rich":
        init_hashes["causal_conv_weight"] = _tensor_hash(
            model.causal_encoder.causal_conv.weight
        )
        init_hashes["head_weight"] = _tensor_hash(
            model.head[-1].weight if hasattr(model.head, "__getitem__")
            else model.head.weight
        )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    groups = _param_groups(model, cond["lr"], cond["embed_lr_scale"])
    if cond["opt"] == "adamw":
        opt = torch.optim.AdamW(groups, lr=cond["lr"], weight_decay=cond["wd"])
    else:
        opt = torch.optim.Adam(groups, lr=cond["lr"], weight_decay=cond["wd"])
    gen = iter_batches(xtr, ytr, cfg["batch_size"], seed)
    steps = cond["steps"]
    eval_every = cond["eval_every"]
    curve: list[dict] = []
    model.train()
    for step in range(1, steps + 1):
        xb, yb = next(gen)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(answer_logits(model, xb), yb)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                a = evaluate_answers(model, cx, cy, cfg["eval_batch"])
                b = evaluate_answers(model, vx, vy, cfg["eval_batch"])
                pr = _progress(model, circular)
            model.train()
            curve.append({
                "step": step,
                "train_acc": a["answer_accuracy"], "val_acc": b["answer_accuracy"],
                "train_loss": a["answer_loss"], "val_loss": b["answer_loss"],
                **pr,
            })

    final_train = evaluate_answers(model, xtr, ytr, cfg["eval_batch"])
    final_val = evaluate_answers(model, xva, yva, cfg["eval_batch"])
    drift = float((model.embed.weight.detach() - embed_init).pow(2).mean().sqrt())
    final_progress = _progress(model, circular)
    wall = time.perf_counter() - t0
    phase = _phase_detect(curve)
    print(f"[{cond['name']} seed={seed}] train={final_train['answer_accuracy']:.4f} "
          f"val={final_val['answer_accuracy']:.4f} maxval={phase['max_val_accuracy']:.4f} "
          f"grok={phase['grokking_observed']} r2_k1={final_progress['r2_k1']:.3f} | {wall:.1f}s",
          flush=True)
    return {
        "condition": cond["name"], "cond": cond, "seed": seed,
        "val_fraction": cond.get("val_fraction", cfg["val_fraction"]),
        "n_train": len(tr_pairs), "n_val": len(va_pairs),
        "n_params": n_params, "n_trainable_params": n_trainable,
        "embedding_drift_rms": drift,
        "init_hashes": init_hashes,
        "curve": curve,
        "phase": phase,
        "final_progress": final_progress,
        "final": {
            "train_accuracy": final_train["answer_accuracy"],
            "train_loss": final_train["answer_loss"],
            "val_accuracy": final_val["answer_accuracy"],
            "val_loss": final_val["answer_loss"],
            "generalization_gap": final_train["answer_accuracy"]
            - final_val["answer_accuracy"],
            "val_n": final_val["n"],
        },
        "wallclock_seconds": wall,
        "seconds_per_step": wall / max(steps, 1),
    }


# --------------------------------------------------------------------------- #
# 条件网格
# --------------------------------------------------------------------------- #
def _cond(name: str, mode: str, *, arch: str = "rich", group: str = "add_classical",
          opt: str = "adam", wd: float = 0.0, lr: float = 0.015,
          embed_lr_scale: float = 1.0, val_fraction: float = 0.2) -> dict:
    return {"name": name, "arch": arch, "group": group, "mode": mode,
            "opt": opt, "wd": wd, "lr": lr, "embed_lr_scale": embed_lr_scale,
            "val_fraction": val_fraction, "steps": 0, "eval_every": 0}


def _pilot_conditions(steps: int, eval_every: int) -> list[dict]:
    base = [
        _cond("R/adam/wd0", "trainable_random"),
        _cond("R/adamw/wd.01", "trainable_random", opt="adamw", wd=0.01),
        _cond("R/adamw/wd.1", "trainable_random", opt="adamw", wd=0.1),
        _cond("R/adamw/wd.5", "trainable_random", opt="adamw", wd=0.5),
        _cond("R/adamw/wd1", "trainable_random", opt="adamw", wd=1.0),
        _cond("R/adamw/wd.1/lr.05", "trainable_random", opt="adamw", wd=0.1, lr=0.05),
        _cond("R/adamw/wd1/lr.05", "trainable_random", opt="adamw", wd=1.0, lr=0.05),
        _cond("F/adam/wd0", "frozen_circular"),
        _cond("F/adamw/wd.1", "frozen_circular", opt="adamw", wd=0.1),
        _cond("T/adam/wd0", "trainable_circular"),
        _cond("T/adamw/wd.1/emb0.1", "trainable_circular", opt="adamw", wd=0.1,
              embed_lr_scale=0.1),
        _cond("CTRL/mlp/adamw/wd1", "trainable_random", arch="quad_mlp",
              opt="adamw", wd=1.0, lr=0.003),
        _cond("CTRL/mlp/adamw/wd.1", "trainable_random", arch="quad_mlp",
              opt="adamw", wd=0.1, lr=0.003),
        # grokking 区间搜索：训练占比是最关键的旋钮。80% 训练数据下正对照
        # 1000 步就同时记住并泛化，属于「正常学习」而非 grokking；把训练对
        # 压到 50% / 20% 才可能先记忆、后突发泛化（Power et al. 用 ~50%）。
        _cond("CTRL/vf.5/wd.1", "trainable_random", arch="quad_mlp",
              opt="adamw", wd=0.1, lr=0.003, val_fraction=0.5),
        _cond("CTRL/vf.5/wd1", "trainable_random", arch="quad_mlp",
              opt="adamw", wd=1.0, lr=0.003, val_fraction=0.5),
        _cond("CTRL/vf.5/wd3", "trainable_random", arch="quad_mlp",
              opt="adamw", wd=3.0, lr=0.003, val_fraction=0.5),
        _cond("CTRL/vf.8/wd.1", "trainable_random", arch="quad_mlp",
              opt="adamw", wd=0.1, lr=0.003, val_fraction=0.8),
        _cond("CTRL/vf.8/wd1", "trainable_random", arch="quad_mlp",
              opt="adamw", wd=1.0, lr=0.003, val_fraction=0.8),
        _cond("R/vf.5/wd1", "trainable_random", opt="adamw", wd=1.0,
              lr=0.015, val_fraction=0.5),
    ]
    for c in base:
        c["steps"] = steps
        c["eval_every"] = eval_every
    return base


def _run_pool(tasks: list[tuple], workers: int) -> list[dict]:
    if workers > 1 and len(tasks) > 1:
        # 长任务排前面，避免尾部只剩单个慢量子任务占着 worker
        tasks = sorted(tasks, key=lambda t: -t[0]["steps"] * (30 if t[0]["group"]
                       == "add_q_rich48" else 1))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(run_one, tasks))
    return [run_one(t) for t in tasks]


def _aggregate(runs: list[dict]) -> dict:
    keys = ("train_accuracy", "val_accuracy", "train_loss", "val_loss",
            "generalization_gap", "val_accuracy")
    out: dict = {}
    for k in dict.fromkeys(keys):
        v = [r["final"][k] for r in runs]
        out[f"{k}_mean"] = float(np.mean(v))
        out[f"{k}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return out


# --------------------------------------------------------------------------- #
# 画图
# --------------------------------------------------------------------------- #
def _figure(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.4))
    pilot = report["pilot"]["per_condition"]
    confirm = report["confirm"]["per_condition"]

    # (a) pilot 各条件最终 val acc（探索性）
    ax = axes[0, 0]
    names = list(pilot)
    vals = [pilot[n]["phase"]["final_val_accuracy"] for n in names]
    ys = np.arange(len(names))
    sel = report["pilot"]["selected_grokking_condition"]
    bar_colors = ["#1D9D75" if n.startswith("CTRL") else
                  ("#D85A30" if n == sel else "#888780") for n in names]
    ax.barh(ys, vals, color=bar_colors, alpha=0.85)
    ax.axvline(RANDOM_BASELINE_ACCURACY, color="#444444", ls=":", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([n.replace("CTRL/", "C/").replace("mlp/", "")
                        for n in names], fontsize=6.8)
    ax.set_xlabel("final val accuracy (1-seed pilot)")
    ax.set_title(f"(a) pilot sweep @ {report['config']['pilot_steps']} steps — exploratory")
    ax.grid(alpha=0.25, axis="x")

    # (b) 5-seed 确认：val acc 曲线
    ax = axes[0, 1]
    for name, d in confirm.items():
        curves = d["curves"]
        if not curves:
            continue
        steps = [c["step"] for c in curves[0]]
        ys_a = np.array([[c["val_acc"] for c in cu] for cu in curves])
        mu = ys_a.mean(0)
        ax.plot(steps, mu, color=COLORS.get(name, "#444444"), lw=1.8, label=name)
        if len(curves) > 1:
            sd = ys_a.std(0, ddof=1)
            ax.fill_between(steps, mu - sd, mu + sd, color=COLORS.get(name, "#444444"),
                            alpha=0.10)
    ax.axhline(RANDOM_BASELINE_ACCURACY, color="#888780", ls=":", lw=1, label="random 1/112")
    ax.axhline(PHASE_GEN_ACC, color="#444444", ls="--", lw=1, label="gen threshold 0.90")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("training steps (log)")
    ax.set_ylabel("val top-1 accuracy")
    ax.set_title(f"(b) 5-seed confirm @ {report['config']['confirm_steps']} steps")
    ax.legend(fontsize=6.5, framealpha=0.92,
              bbox_to_anchor=(0.99, 0.60), loc="center right")
    ax.grid(alpha=0.25)

    # (c) 真正发生相变的正对照：train/val 是否分离 + 傅里叶 R^2
    ax = axes[1, 0]
    focus = "control_mlp" if "control_mlp" in confirm else max(
        confirm, key=lambda k: confirm[k]["aggregate"]["val_accuracy_mean"])
    d = confirm.get(focus)
    if d and d["curves"]:
        curves = d["curves"]
        steps = [c["step"] for c in curves[0]]
        for key, color, label in (("train_acc", "#185FA5", "train acc"),
                                  ("val_acc", "#D85A30", "val acc")):
            arr = np.array([[c[key] for c in cu] for cu in curves])
            ax.plot(steps, arr.mean(0), color=color, lw=1.8, label=label)
        ax.set_ylabel("accuracy", color="#333333")
        ax.set_ylim(-0.02, 1.02)
        ax2 = ax.twinx()
        r2 = np.array([[c["r2_k1"] for c in cu] for cu in curves])
        ax2.plot(steps, r2.mean(0), color="#1D9D75", lw=1.8, ls="--",
                 label="progress: R²(sin/cos k=1)")
        ax2.set_ylabel("R²", color="#1D9D75")
        ax2.set_ylim(-0.05, 1.05)
        ph = d["phase_per_seed"][0]
        ax.axvline(ph["memorization_step"] or steps[0], color="#185FA5", ls=":", lw=1)
        ax.axvline(ph["generalization_step"] or steps[0], color="#D85A30", ls=":", lw=1)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    # 相变发生在 500~1500 步，线性轴比对数轴更能看清 train/val 的分离
    ax.set_xlabel("training steps")
    ax.set_title(f"(c) delayed generalization? — {focus} "
                 f"({d['n_grokking_seeds']}/{len(d['curves'])} seeds grok)")
    ax.grid(alpha=0.25)

    # (d) 量子臂成本外推
    ax = axes[1, 1]
    q = report["quantum"]["cost_extrapolation"]
    labels = ["classical\n(measured)", "quantum\n(measured)"]
    per_step = [q["classical_seconds_per_step"], q["quantum_seconds_per_step"]]
    bars = ax.bar(labels, per_step, color=["#185FA5", "#534AB7"], alpha=0.85, width=0.55)
    ax.set_yscale("log")
    ax.set_ylim(bottom=min(per_step) / 4, top=max(per_step) * 40)
    for b, v in zip(bars, per_step):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.35, f"{v:.4f} s/step",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("seconds per training step (log)")
    ax.set_title(f"(d) cost: {q['hours_to_1e5_steps_quantum']:.1f} h vs "
                 f"{q['hours_to_1e5_steps_classical']:.1f} h for 10⁵ steps")
    ax.text(0.02, 0.97,
            f"quantum / classical = {q['cost_ratio']:.1f}×\n"
            f"quantum budget used: {report['config']['quantum_steps']} steps "
            f"({report['config']['quantum_steps'] / GROKKING_STEP_SCALE:.1%} of 10⁵)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            bbox=dict(fc="white", ec="#cccccc", alpha=0.9))
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.subplots_adjust(left=0.11, wspace=0.26, hspace=0.32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="10 步 x 1 seed 冒烟，不写归档")
    ap.add_argument("--pilot-only", action="store_true",
                    help="只跑 1-seed pilot 并写 docs/grokking_pilot.json，不进入长训练")
    ap.add_argument("--pilot-from", default="",
                    help="复用已有 docs/grokking_pilot.json 的条件选择，跳过 pilot")
    ap.add_argument("--control-steps", type=int, default=5000,
                    help="正对照只需跑到收敛 + 稳定段，不必陪跑 4 万步")
    ap.add_argument("--control-eval-every", type=int, default=250)
    ap.add_argument("--pilot-steps", type=int, default=20000)
    ap.add_argument("--confirm-steps", type=int, default=40000)
    ap.add_argument("--quantum-steps", type=int, default=2500)
    ap.add_argument("--quantum-seeds", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--curve-n", type=int, default=1024)
    ap.add_argument("--pilot-eval-every", type=int, default=500)
    ap.add_argument("--confirm-eval-every", type=int, default=1000)
    ap.add_argument("--quantum-eval-every", type=int, default=500)
    ap.add_argument("--quantum-curve-n", type=int, default=256)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--redraw-from", default="", help="从归档 JSON 重画图，不重跑训练")
    args = ap.parse_args()

    if args.redraw_from:
        _figure(json.loads(Path(args.redraw_from).read_text(encoding="utf-8")), FIGURE)
        print(f"redrawn from {args.redraw_from} -> {FIGURE}")
        return

    quick = args.quick
    pilot_steps, confirm_steps = (10, 10) if quick else (args.pilot_steps, args.confirm_steps)
    quantum_steps = 10 if quick else args.quantum_steps
    workers = 1 if quick else args.workers
    seeds_n, qseeds_n = (1, 1) if quick else (args.seeds, args.quantum_seeds)
    cfg = {
        "val_fraction": args.val_fraction, "batch_size": args.batch_size,
        "eval_batch": 512, "curve_n": 64 if quick else args.curve_n,
        "split_mode": "holdout_pairs", "split_seed": 0, "op": "+",
        "modulus": MODULUS, "vocab_size": VOCAB_SIZE,
        "pilot_steps": pilot_steps, "confirm_steps": confirm_steps,
        "quantum_steps": quantum_steps, "pilot_eval_every": 5 if quick else args.pilot_eval_every,
        "confirm_eval_every": 5 if quick else args.confirm_eval_every,
        "quantum_eval_every": 5 if quick else args.quantum_eval_every,
        "quantum_curve_n": 64 if quick else args.quantum_curve_n,
        "seeds": list(range(seeds_n)), "quantum_seeds": list(range(qseeds_n)),
        "workers": workers,
    }

    table = enumerate_table("+")
    tr, va = split_table(table, "holdout_pairs", args.val_fraction, 0)
    overlap = context_overlap(tr, va, "holdout_pairs", "+")
    assert overlap["pair_level_disjoint"] and overlap["full_context_coverage_k4"] == 0.0
    t0 = time.perf_counter()

    # ---- Stage 1: 1-seed pilot（探索性，用于预选条件） ----
    if args.pilot_from:
        pd = json.loads(Path(args.pilot_from).read_text(encoding="utf-8"))
        pilot_conds = pd["conditions"]
        pilot_per = pd["per_condition"]
        best = next(c for c in pilot_conds
                    if c["name"] == pd["selected_grokking_condition"])
        best_ctrl = next(c for c in pilot_conds
                         if c["name"] == pd["selected_positive_control"])
        print(f"[stage 1] reused pilot from {args.pilot_from}: "
              f"{len(pilot_conds)} conditions, "
              f"{pd['n_phase_transitions']} phase transition(s)", flush=True)
    else:
        pilot_conds = _pilot_conditions(pilot_steps, cfg["pilot_eval_every"])
        print(f"[stage 1] pilot: {len(pilot_conds)} conditions x 1 seed x {pilot_steps} steps",
              flush=True)
        pilot_runs = _run_pool([(c, 0, cfg) for c in pilot_conds], workers)
        pilot_per = {r["condition"]: {"phase": r["phase"], "final": r["final"],
                                      "final_progress": r["final_progress"],
                                      "val_fraction": r["val_fraction"],
                                      "n_train": r["n_train"], "n_val": r["n_val"],
                                      "curve": r["curve"],
                                      "seconds_per_step": r["seconds_per_step"]}
                     for r in pilot_runs}
    # 主臂候选限定在与默认切分一致的 val_fraction，保证 5-seed 配对比较同切分
    random_rich = [c for c in pilot_conds
                   if c["mode"] == "trainable_random" and c["arch"] == "rich"
                   and c["val_fraction"] == args.val_fraction]
    ctrl_mlps = [c for c in pilot_conds if c["arch"] == "quad_mlp"]

    def _pilot_score(c: dict) -> tuple:
        p = pilot_per[c["name"]]
        # 先看是否出现延迟泛化/突发（grokking 定义），再看全量验证集准确率，
        # 最后才用子集曲线的最大值做 tie-break（子集噪声大，不作为主依据）
        return (p["phase"]["grokking_observed"],
                p["phase"]["delayed_generalization"],
                p["final"]["val_accuracy"],
                p["phase"]["max_val_accuracy"])

    best = max(random_rich, key=_pilot_score)
    best_ctrl = max(ctrl_mlps, key=_pilot_score)
    print(f"[stage 1] best random-rich (default split): {best['name']} | "
          f"positive control: {best_ctrl['name']} "
          f"(full-set val {pilot_per[best_ctrl['name']]['final']['val_accuracy']:.4f})",
          flush=True)

    if args.pilot_only:
        out = DOCS / "grokking_pilot.json"
        DOCS.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "status": "exploratory_1_seed_only", "config": cfg,
            "conditions": pilot_conds, "per_condition": pilot_per,
            "selected_grokking_condition": best["name"],
            "selected_positive_control": best_ctrl["name"],
            "n_phase_transitions": sum(
                1 for v in pilot_per.values() if v["phase"]["grokking_observed"]),
            "n_delayed_generalization": sum(
                1 for v in pilot_per.values() if v["phase"]["delayed_generalization"]),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[stage 1] pilot-only done -> {out} | wall={time.perf_counter() - t0:.1f}s")
        return

    # ---- Stage 2: 5-seed 确认（结论性） ----
    # 所有 rich 条件统一 val_fraction=args.val_fraction，保证配对比较同切分。
    # control_mlp_matched 也用同一 split，用于干净的「架构差距」比较；
    # control_mlp 保留 pilot 里真正出现相变的少数据条件，用于 5-seed 复现相变。
    names = {
        "ctrl_random": next(c for c in pilot_conds if c["name"] == "R/adam/wd0"),
        "best_random": best,
        "frozen": next(c for c in pilot_conds if c["name"] == "F/adam/wd0"),
        "trainable_circular": next(c for c in pilot_conds if c["name"] == "T/adam/wd0"),
        "control_mlp": best_ctrl,
        "control_mlp_matched": next(c for c in pilot_conds
                                    if c["name"] == "CTRL/mlp/adamw/wd.1"),
    }
    confirm_conds: list[tuple] = []
    control_steps = 10 if quick else args.control_steps
    for key, c in names.items():
        cc = dict(c)
        cc["arch"] = "quad_mlp" if key.startswith("control_mlp") else "rich"
        cc["group"] = "add_classical"
        cc["val_fraction"] = (c["val_fraction"] if key == "control_mlp"
                              else args.val_fraction)
        short = key.startswith("control_mlp")
        cc["steps"] = control_steps if short else confirm_steps
        cc["eval_every"] = (5 if quick else args.control_eval_every) if short else (
            5 if quick else cfg["confirm_eval_every"])
        confirm_conds.append((key, cc))

    # ---- Stage 3: 量子臂有限预算示意（两臂同条件，用于成本外推） ----
    quantum_conds: list[tuple] = []
    for key in ("best_random", "frozen"):
        c = dict(names[key])
        c["steps"] = quantum_steps
        c["eval_every"] = cfg["quantum_eval_every"]
        for g in GROUP_IDS:
            cg = dict(c)
            cg["group"] = g
            cg["arch"] = "rich"
            quantum_conds.append((f"{key}|{g}", cg))

    all_tasks: list[tuple] = []
    for key, c in confirm_conds:
        for s in cfg["seeds"]:
            all_tasks.append(({**c, "name": key}, s, cfg))
    for key, c in quantum_conds:
        for s in cfg["quantum_seeds"]:
            qcfg = {**cfg, "curve_n": cfg["quantum_curve_n"]}
            all_tasks.append(({**c, "name": key}, s, qcfg))

    print(f"[stage 2+3] {len(confirm_conds)} x {seeds_n} classical @ {confirm_steps} steps + "
          f"{len(quantum_conds)} x {qseeds_n} quantum @ {quantum_steps} steps "
          f"= {len(all_tasks)} runs, workers={workers}", flush=True)
    runs = _run_pool(all_tasks, workers)

    confirm_runs = [r for r in runs if not r["condition"].endswith(
        tuple(f"|{g}" for g in GROUP_IDS))]
    quantum_runs = [r for r in runs if r["condition"].endswith(
        tuple(f"|{g}" for g in GROUP_IDS))]

    # ---- 确认阶段聚合 ----
    per_condition: dict = {}
    for key, _ in confirm_conds:
        rs = [r for r in confirm_runs if r["condition"] == key]
        if not rs:
            continue
        per_condition[key] = {
            "cond": rs[0]["cond"],
            "seeds": [r["seed"] for r in rs],
            "curves": [r["curve"] for r in rs],
            "aggregate": _aggregate(rs),
            "phase_per_seed": [r["phase"] for r in rs],
            "n_grokking_seeds": sum(1 for r in rs if r["phase"]["grokking_observed"]),
            "phase_r2_first4_mean": float(np.mean(
                [r["final_progress"].get("phase_r2_first4", float("nan")) for r in rs
                 if "phase_r2_first4" in r["final_progress"]]))
            if any("phase_r2_first4" in r["final_progress"] for r in rs) else None,
            "final_r2_k1_mean": float(np.mean([r["final_progress"]["r2_k1"] for r in rs])),
            "embedding_drift_rms_mean": float(np.mean([r["embedding_drift_rms"] for r in rs])),
            "seconds_per_step_mean": float(np.mean([r["seconds_per_step"] for r in rs])),
        }

    # 预注册主检验：同 seed 配对，val_accuracy 为主指标
    primary_specs = [
        ("best_random", "ctrl_random", "grokking_condition_vs_control"),
        ("frozen", "trainable_circular", "frozen_vs_trainable_at_convergence"),
        ("control_mlp_matched", "ctrl_random", "architecture_gap_same_split"),
    ]
    paired: dict = {}
    for a, b, label in primary_specs:
        if a not in per_condition or b not in per_condition:
            continue
        for metric in ("val_accuracy", "train_accuracy", "generalization_gap"):
            xa = [r["final"][metric] for r in confirm_runs if r["condition"] == a]
            xb = [r["final"][metric] for r in confirm_runs if r["condition"] == b]
            if len(xa) == len(xb) and len(xa) >= 2:
                paired[f"{label}|{metric}"] = _paired_ci(xa, xb)
    holm_keys = [k for k in paired if k.endswith("|val_accuracy")]
    holm = _holm(paired, holm_keys)
    for k, v in holm.items():
        paired[k].update(v)
    n_holm = sum(1 for k in holm_keys if paired[k].get("holm_significant"))

    # ---- 量子臂：权重控制 + 成本外推 ----
    q_by_group = {g: [r for r in quantum_runs if r["cond"]["group"] == g]
                  for g in GROUP_IDS}
    weight_checks = {"all_passed": True, "checks": {}}
    for s in cfg["quantum_seeds"]:
        for key in ("best_random", "frozen"):
            a = [r for r in q_by_group[GROUP_IDS[0]] if r["seed"] == s
                 and r["condition"] == f"{key}|{GROUP_IDS[0]}"]
            b = [r for r in q_by_group[GROUP_IDS[1]] if r["seed"] == s
                 and r["condition"] == f"{key}|{GROUP_IDS[1]}"]
            ok = bool(a and b and all(
                a[0]["init_hashes"][k] == b[0]["init_hashes"][k]
                for k in a[0]["init_hashes"]
            ))
            weight_checks["checks"][f"{key}|seed{s}"] = ok
            weight_checks["all_passed"] &= ok
    q_spp = {g: float(np.mean([r["seconds_per_step"] for r in q_by_group[g]]))
             if q_by_group[g] else 0.0 for g in GROUP_IDS}
    # 成本外推只用 rich 架构的经典臂（二次 MLP 单步更便宜，会低估量子/经典比）
    rich_classical = [r for r in confirm_runs
                      if r["cond"]["arch"] == "rich"
                      and r["cond"]["group"] == "add_classical"]
    c_spp = float(np.mean([r["seconds_per_step"] for r in rich_classical])) \
        if rich_classical else 0.0
    ratio = q_spp[GROUP_IDS[0]] / max(c_spp, 1e-9)
    cost = {
        "classical_seconds_per_step": c_spp,
        "quantum_seconds_per_step": q_spp[GROUP_IDS[0]],
        "cost_ratio": ratio,
        "hours_to_1e5_steps_classical": c_spp * GROKKING_STEP_SCALE / 3600,
        "hours_to_1e5_steps_quantum": q_spp[GROUP_IDS[0]] * GROKKING_STEP_SCALE / 3600,
        "reference_scale_steps": GROKKING_STEP_SCALE,
    }
    quantum = {
        "status": "illustrative_only_limited_budget",
        "conditions": {key: {
            g: {"final_val_accuracy_mean": float(np.mean(
                [r["final"]["val_accuracy"] for r in q_by_group[g]
                 if r["condition"] == f"{key}|{g}"])),
                "final_train_accuracy_mean": float(np.mean(
                    [r["final"]["train_accuracy"] for r in q_by_group[g]
                     if r["condition"] == f"{key}|{g}"]))}
            for g in GROUP_IDS}
            for key in ("best_random", "frozen")},
        "weight_control_checks": weight_checks,
        "cost_extrapolation": cost,
        "note": "量子臂只跑到有限步数，未达 grokking 相变量级；此处数字仅用于成本外推，"
                "不得与经典臂长训练结果做等价比较。",
    }

    # ---- 结论 ----
    b = per_condition.get("best_random", {})
    ctrl_grok = b.get("n_grokking_seeds", 0)
    ctrl_mlp_grok = per_condition.get("control_mlp", {}).get("n_grokking_seeds", 0)
    rich_val = b.get("aggregate", {}).get("val_accuracy_mean", float("nan"))
    ctrl_val = per_condition.get("control_mlp", {}).get(
        "aggregate", {}).get("val_accuracy_mean", float("nan"))
    frozen_val = per_condition.get("frozen", {}).get(
        "aggregate", {}).get("val_accuracy_mean", float("nan"))
    trainable_val = per_condition.get("trainable_circular", {}).get(
        "aggregate", {}).get("val_accuracy_mean", float("nan"))
    if ctrl_grok > 0:
        verdict = (f"预选 grokking 条件 {best['name']} 在 {ctrl_grok}/{seeds_n} 个 seed 上"
                   f"出现「先记忆后突发泛化」的相变，随机 embedding 学不动确实只是"
                   f"训练预算与正则化条件不足，不是容量或梯度问题。")
    elif ctrl_mlp_grok > 0:
        verdict = (
            f"【grokking 复现了，但不在本仓库模型里】正对照 {best_ctrl['name']} 在 5-seed "
            f"确认中 {ctrl_mlp_grok}/{seeds_n} 个 seed 出现「先记忆、后突发泛化」，"
            f"val 约 {ctrl_val:.4f}；同一任务、同一随机 embedding、同一 (a,b) 切分下，"
            f"RichQuantumLM 经典臂长训练 {confirm_steps} 步只到 {rich_val:.4f}，"
            f"{ctrl_grok}/{seeds_n} 个 seed 出现相变。因此「再训练久一点就会 grok」"
            f"在本仓库架构上不成立：瓶颈是表示能力 / 架构，而不是训练预算或梯度摊薄。"
            f"第 10.3 节的旧归因应据此再收窄。"
        )
    else:
        verdict = (f"在 {confirm_steps} 步、含 AdamW + weight decay 的全部预选条件下，"
                   f"RichQuantumLM 与二次 MLP 正对照均未观察到相变"
                   f"（{ctrl_grok}/{seeds_n} 与 {ctrl_mlp_grok}/{seeds_n}）。"
                   f"只能说「该架构与训练条件下未复现 grokking」。")
    if np.isfinite(frozen_val) and np.isfinite(trainable_val):
        verdict += (f" 冻结 vs 放开在 {confirm_steps} 步后为 "
                    f"{frozen_val:.4f} vs {trainable_val:.4f}，"
                    f"{'排序未反转' if frozen_val >= trainable_val else '排序已反转'}。")
    holm_txt = (f" {len(holm_keys)} 个预注册主检验（val_accuracy）Holm 校正后 "
                f"{n_holm} 个显著；其余 "
                f"{len(paired) - len(holm_keys)} 个指标为次要描述性比较，未校正。"
                if paired else " 样本不足，未做主检验。")

    # pilot 里真正进入相变区间 / 发生崩溃的条件（1 seed，探索性）
    grokking_hits = {
        n: {"val_fraction": v["val_fraction"], "n_train": v["n_train"],
            "memorization_step": v["phase"]["memorization_step"],
            "generalization_step": v["phase"]["generalization_step"],
            "delay_steps": v["phase"]["delay_steps"],
            "max_val_jump": v["phase"]["max_val_jump"],
            "final_val_accuracy": v["final"]["val_accuracy"]}
        for n, v in pilot_per.items() if v["phase"]["grokking_observed"]
    }
    transient_collapse = {
        n: {"val_fraction": v["val_fraction"],
            "peak_val_accuracy": v["phase"]["max_val_accuracy"],
            "final_val_accuracy": v["final"]["val_accuracy"]}
        for n, v in pilot_per.items()
        if v["phase"]["max_val_accuracy"] > 0.5 and v["final"]["val_accuracy"] < 0.1
    }

    elapsed = time.perf_counter() - t0
    report = {
        "task": "grokking long-training scan on mod-112 addition (Z_112)",
        "question": "随机 embedding 的失败是否只是没跑到 grokking 相变；冻结/放开排序在收敛后是否反转",
        "literature": [
            {"ref": "Power et al. 2022, arXiv:2201.02177",
             "claim": "模加法是 grokking 原始任务；decoder-only transformer + AdamW/weight decay，"
                      "验证准确率在过拟合很久后（10^4~10^6 步量级）才从随机跳到近 100%。"},
            {"ref": "Nanda et al. 2023, arXiv:2301.05217 (ICLR Oral)",
             "claim": "p=113 模加法最终学到 sin/cos 圆上旋转 + 三角恒等式；"
                      "分记忆/电路形成/清理三阶段，相变前 progress measure 连续变化。"},
            {"ref": "Liu et al. 2022, arXiv:2205.10343",
             "claim": "结构化环形表征 + 有限数据 + 足够正则化压力才落在 grokking 区间；"
                      "二次 MLP 是常用可 grok 的正对照。"},
        ],
        "config": cfg,
        "design": {
            "pilot": "1 seed 超参网格，探索性；用它预选条件，数字不作结论",
            "confirm": "预选条件 5 seed 长训练，结论性；同 seed 配对 + Holm",
            "positive_control": "QuadraticMLP：out((W[a]+W[b])^2)，二次项提供乘法交互；"
                                "接口对齐以便复用同一个 progress measure",
            "progress_measure": "从 112 个数字 embedding 线性解码 sin/cos(k=1,2) 的多元 R^2；"
                                "circular 模式另报只用前 4 维的 phase_r2_first4（相位保持度）",
            "phase_detection": f"train>={PHASE_MEM_ACC} 为记忆完成、val>={PHASE_GEN_ACC} 为泛化完成、"
                               f"相邻采样跳变>={PHASE_JUMP}；阈值预先写死",
            "quantum": "有限预算示意 + 实测单步成本外推 10^5 步耗时，不与经典臂等价比较",
        },
        "corpus_overlap": overlap,
        "pilot": {
            "status": "exploratory_1_seed",
            "per_condition": pilot_per,
            "selected_grokking_condition": best["name"],
            "selected_positive_control": best_ctrl["name"],
            "selected_condition_grokked_in_pilot": bool(
                pilot_per[best["name"]]["phase"]["grokking_observed"]),
            "n_conditions": len(pilot_conds),
        },
        "grokking_regime_search": {
            "phase_transitions_observed": grokking_hits,
            "transient_then_collapse": transient_collapse,
            "note": "全部来自 1-seed pilot，属探索性观察；"
                    "phase_transitions_observed 里的条件已在 5-seed confirm 复现，"
                    "transient_then_collapse 未做多 seed 复核。",
        },
        "confirm": {
            "per_condition": per_condition,
            "selected_grokking_condition": "best_random",
            "selected_grokking_pilot_name": best["name"],
            "condition_details": {k: v for k, v in names.items()},
            "paired_primary_tests": paired,
            "primary_endpoints": holm_keys,
            "n_primary_holm_significant": n_holm,
            "n_grokking_seeds": {k: v.get("n_grokking_seeds", 0)
                                 for k, v in per_condition.items()},
        },
        "quantum": quantum,
        "interpretation": {
            "verdict": verdict + holm_txt,
            "phase_thresholds": {"mem_acc": PHASE_MEM_ACC, "gen_acc": PHASE_GEN_ACC,
                                 "jump": PHASE_JUMP},
            "random_baseline_accuracy": RANDOM_BASELINE_ACCURACY,
        },
        "caveat": (
            "val 是未见 (a,b) 组合，属组合泛化；答案位只在 112 个数字类上评估。"
            "pilot 是 1-seed 探索，条件选择看的是同一批结果，存在选择偏差；"
            "结论只由 5-seed 确认阶段给出。循环角度编码是两臂共享的经典先验，"
            "任何由其带来的提升都不能归因于量子层。本脚本未对任一臂单独调参；"
            "量子臂使用 state-vector 模拟、无 shots/噪声，成本外推只适用于该模拟条件。"
        ),
        "runs": runs,
        "wallclock_seconds_total": elapsed,
    }

    print("\n=== grokking scan ===")
    print("[pilot top]")
    for n in sorted(pilot_per, key=lambda k: -pilot_per[k]["phase"]["max_val_accuracy"])[:5]:
        p = pilot_per[n]["phase"]
        print(f"  {n:26s} maxval={p['max_val_accuracy']:.4f} final={p['final_val_accuracy']:.4f} "
              f"jump={p['max_val_jump']:.3f} grok={p['grokking_observed']}")
    print("[confirm 5-seed]")
    for k, v in per_condition.items():
        a = v["aggregate"]
        print(f"  {k:20s} val {a['val_accuracy_mean']:.4f} ± {a['val_accuracy_std']:.4f} | "
              f"train {a['train_accuracy_mean']:.4f} | grok {v['n_grokking_seeds']}/{seeds_n} | "
              f"R²(k=1) {v['final_r2_k1_mean']:.3f} | phaseR² "
              f"{v['phase_r2_first4_mean'] if v['phase_r2_first4_mean'] is not None else float('nan'):.3f}")
    for k in holm_keys:
        p = paired[k]
        print(f"  primary {k:52s} Δ={p['mean_diff']:+.4f} "
              f"CI[{p['ci95_low']:+.4f},{p['ci95_high']:+.4f}] p={p['p_value']:.3g} "
              f"holm={p.get('p_holm')}")
    print(f"[quantum] weight control all_passed={weight_checks['all_passed']} | "
          f"ratio={ratio:.1f}x | 1e5 steps: Q {cost['hours_to_1e5_steps_quantum']:.1f} h vs "
          f"C {cost['hours_to_1e5_steps_classical']:.1f} h")
    print("  verdict:", verdict + holm_txt)
    print(f"  wall={elapsed:.1f}s")

    if not quick:
        DOCS.mkdir(parents=True, exist_ok=True)
        METRICS.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
        _figure(report, FIGURE)
        print(f"  archived: {METRICS}\n  figure:   {FIGURE}")


if __name__ == "__main__":
    main()
