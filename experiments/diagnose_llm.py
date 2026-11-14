"""多随机种子系统诊断实验（5 seeds）：信息论熵下界、逐字符瓶颈消融与因果上下文破局。

运行方式：
  * 快速冒烟验证（不覆盖 docs/）：
      python experiments/diagnose_llm.py --quick
  * 完整 5-seed 诊断实验（更新 docs/llm_diagnosis_metrics.json 与 docs/figures/）：
      python experiments/diagnose_llm.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import stats
import torch

from qml_lab.data import (
    build_char_data,
    corpus_entropy_floors,
    iter_batches,
    load_corpus_split,
)
from qml_lab.models import RichQuantumLM
from qml_lab.train import greedy_sample, train_lm
from qml_lab.utils import (
    count_parameters,
    ensure_dir,
    fmt_table,
    save_json,
    set_seed,
)
from qml_lab.viz import plot_bars, plot_curves

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except Exception:
    plt = None
    _MPL_OK = False

CAVEAT = (
    "本实验是玩具级量子增强语言模型（字符级词表 31、序列长 16、4 qubits 状态向量模拟）"
    "的受控方法学诊断，不是能对话的大语言模型（LLM）。"
)


def _tensor_sha256(t: torch.Tensor) -> str:
    arr = np.ascontiguousarray(t.detach().cpu().numpy())
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _oracle_greedy_samples(text: str, prompt: str = "A qubit can hold", length: int = 104) -> dict[int, str]:
    """用精确经验 k-gram 频率表从 prompt 出发做贪心 argmax 续写，证明 k=1/2 的必然死循环。"""
    samples: dict[int, str] = {}
    for k in (1, 2, 3, 4, 6, 8):
        counts: dict[str, Counter] = defaultdict(Counter)
        for i in range(len(text) - k):
            counts[text[i:i + k]][text[i + k]] += 1
        chars = list(prompt)
        for _ in range(length):
            ctx = "".join(chars[-k:])
            if ctx in counts and counts[ctx]:
                # 频次相同时按字符字典序确定性打破平局
                best_ch = sorted(counts[ctx].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            else:
                best_ch = " "
            chars.append(best_ch)
        samples[k] = "".join(chars)
    return samples


def _group_specs(qubits: int = 4) -> list[dict[str, Any]]:
    """返回 Part 2（Token-wise, k=1）与 Part 3（Causal, k=4）共 11 个受控实验组规格。"""
    return [
        # ---------------- Part 2: Token-Wise Regime (k=1) ----------------
        {
            "id": "tok_q_legacy",
            "regime": "token",
            "label": "Token Q-Legacy (4q Z, scale=1)",
            "transform_type": "quantum",
            "context_mode": "token",
            "context_window": 1,
            "n_qubits": qubits,
            "n_layers": 1,
            "observables": "z",
            "reupload": False,
            "pre_scale": False,
            "near_identity_init": False,
            "scale": 1.0,
            "head": "linear",
            "hidden": 32,
        },
        {
            "id": "tok_q_scaled",
            "regime": "token",
            "label": "Token Q-Scaled (4q Z, tanh+scale=5)",
            "transform_type": "quantum",
            "context_mode": "token",
            "context_window": 1,
            "n_qubits": qubits,
            "n_layers": 1,
            "observables": "z",
            "reupload": False,
            "pre_scale": True,
            "near_identity_init": False,
            "scale": 5.0,
            "head": "linear",
            "hidden": 32,
        },
        {
            "id": "tok_q_rich24",
            "regime": "token",
            "label": "Token Q-Rich24 (4q Z+X+Y+ZZall)",
            "transform_type": "quantum",
            "context_mode": "token",
            "context_window": 1,
            "n_qubits": qubits,
            "n_layers": 1,
            "observables": "z+x+y+zzall",
            "reupload": False,
            "pre_scale": True,
            "near_identity_init": False,
            "scale": 5.0,
            "head": "linear",
            "hidden": 32,
        },
        {
            "id": "tok_q_rich48",
            "regime": "token",
            "label": "Token Q-Rich48 (4q 48-Pauli)",
            "transform_type": "quantum",
            "context_mode": "token",
            "context_window": 1,
            "n_qubits": qubits,
            "n_layers": 1,
            "observables": "z+x+y+zzall+xxall+yyall",
            "reupload": False,
            "pre_scale": True,
            "near_identity_init": False,
            "scale": 5.0,
            "head": "linear",
            "hidden": 32,
        },
        {
            "id": "tok_identity",
            "regime": "token",
            "label": "Token Identity (no transform)",
            "transform_type": "identity",
            "context_mode": "token",
            "context_window": 1,
            "n_qubits": qubits,
            "n_layers": 1,
            "observables": "z",
            "reupload": False,
            "pre_scale": True,
            "near_identity_init": False,
            "scale": 5.0,
            "head": "linear",
            "hidden": 32,
        },
        {
            "id": "tok_classical",
            "regime": "token",
            "label": "Token Classical (Linear+Tanh)",
            "transform_type": "classical",
            "context_mode": "token",
            "context_window": 1,
            "n_qubits": qubits,
            "n_layers": 1,
            "observables": "z",
            "reupload": False,
            "pre_scale": True,
            "near_identity_init": False,
            "scale": 5.0,
            "head": "linear",
            "hidden": 32,
            "classical_width": qubits,
        },
        # ---------------- Part 3: Causal Context Regime (k=4) ----------------
        {
            "id": "caus_q_z",
            "regime": "causal",
            "label": "Causal Q-Z (k=4, 4q Z, Linear)",
            "transform_type": "quantum",
            "context_mode": "causal",
            "context_window": 4,
            "n_qubits": qubits,
            "n_layers": 1,
            "observables": "z",
            "reupload": False,
            "pre_scale": True,
            "near_identity_init": True,
            "scale": 5.0,
            "head": "linear",
            "hidden": 48,
        },
        {
            "id": "caus_q_reup",
            "regime": "causal",
            "label": "Causal Q-Reup (k=4, 2L trainable, 12-Pauli)",
            "transform_type": "quantum",
            "context_mode": "causal",
            "context_window": 4,
            "n_qubits": qubits,
            "n_layers": 2,
            "observables": "z+x+y",
            "reupload": "trainable",
            "pre_scale": True,
            "near_identity_init": True,
            "scale": 5.0,
            "head": "linear",
            "hidden": 48,
        },
        {
            "id": "caus_q_rich48",
            "regime": "causal",
            "label": "Causal Q-Rich48 (k=4, 48-Pauli, MLP head)",
            "transform_type": "quantum",
            "context_mode": "causal",
            "context_window": 4,
            "n_qubits": qubits,
            "n_layers": 1,
            "observables": "z+x+y+zzall+xxall+yyall",
            "reupload": False,
            "pre_scale": True,
            "near_identity_init": True,
            "scale": 5.0,
            "head": "mlp",
            "hidden": 48,
        },
        {
            "id": "caus_identity",
            "regime": "causal",
            "label": "Causal Identity (k=4, no transform)",
            "transform_type": "identity",
            "context_mode": "causal",
            "context_window": 4,
            "n_qubits": qubits,
            "n_layers": 1,
            "observables": "z",
            "reupload": False,
            "pre_scale": True,
            "near_identity_init": False,
            "scale": 5.0,
            "head": "linear",
            "hidden": 48,
        },
        {
            "id": "caus_classical",
            "regime": "causal",
            "label": "Causal Classical (k=4, Linear48+Tanh, MLP head)",
            "transform_type": "classical",
            "context_mode": "causal",
            "context_window": 4,
            "n_qubits": qubits,
            "n_layers": 1,
            "observables": "z+x+y+zzall+xxall+yyall",
            "reupload": False,
            "pre_scale": True,
            "near_identity_init": False,
            "scale": 1.0,
            "head": "mlp",
            "hidden": 48,
        },
    ]


def _build_model_from_spec(spec: dict[str, Any], vocab_size: int, seed: int) -> RichQuantumLM:
    set_seed(seed)
    return RichQuantumLM(
        vocab_size=vocab_size,
        n_qubits=spec["n_qubits"],
        n_layers=spec["n_layers"],
        transform_type=spec["transform_type"],
        context_mode=spec["context_mode"],
        context_window=spec["context_window"],
        observables=spec["observables"],
        reupload=spec["reupload"],
        pre_scale=spec["pre_scale"],
        near_identity_init=spec["near_identity_init"],
        scale=spec["scale"],
        head=spec["head"],
        hidden=spec["hidden"],
        classical_width=spec.get("classical_width"),
    )


def _verify_initial_weight_control(
    specs: list[dict[str, Any]], vocab_size: int, seeds: list[int]
) -> dict[str, Any]:
    """在训练前严格校验所有受控组在每个 seed 下的初始权重 SHA256 一致性。"""
    spec_map = {s["id"]: s for s in specs}
    checks = {
        "token_embed_identical": True,
        "token_head4_identical": True,
        "token_q_weights_identical": True,
        "causal_frontend_identical": True,
        "causal_head48_identical": True,
        "causal_q_weights_identical": True,
        "per_seed_hashes": {},
    }
    for seed in seeds:
        models = {
            gid: _build_model_from_spec(spec, vocab_size, seed)
            for gid, spec in spec_map.items()
        }
        tok_ids = [s["id"] for s in specs if s["regime"] == "token"]
        caus_ids = [s["id"] for s in specs if s["regime"] == "causal"]

        # 1. Token-wise embed 权重全部一致
        tok_emb_hashes = {_tensor_sha256(models[g].embed.weight) for g in tok_ids}
        if len(tok_emb_hashes) != 1:
            checks["token_embed_identical"] = False

        # 2. Token-wise readout=4 线性头一致
        tok_h4_ids = ["tok_q_legacy", "tok_q_scaled", "tok_identity", "tok_classical"]
        tok_h4_hashes = {
            _tensor_sha256(models[g].head.weight) for g in tok_h4_ids
        }
        if len(tok_h4_hashes) != 1:
            checks["token_head4_identical"] = False

        # 3. Token-wise 量子权重一致
        tok_q_ids = ["tok_q_legacy", "tok_q_scaled", "tok_q_rich24", "tok_q_rich48"]
        tok_q_hashes = {
            _tensor_sha256(models[g].transform.weights) for g in tok_q_ids
        }
        if len(tok_q_hashes) != 1:
            checks["token_q_weights_identical"] = False

        # 4. Causal 前端 (embed + causal_conv) 权重全部一致
        caus_emb_hashes = {_tensor_sha256(models[g].embed.weight) for g in caus_ids}
        caus_conv_hashes = {
            _tensor_sha256(models[g].causal_encoder.causal_conv.weight)
            for g in caus_ids
        }
        if len(caus_emb_hashes) != 1 or len(caus_conv_hashes) != 1:
            checks["causal_frontend_identical"] = False

        # 5. Causal 48维 MLP 分类头一致 (caus_q_rich48 vs caus_classical)
        caus_h48_hashes = {
            _tensor_sha256(models[g].head[0].weight)
            for g in ("caus_q_rich48", "caus_classical")
        }
        if len(caus_h48_hashes) != 1:
            checks["causal_head48_identical"] = False

        # 6. Causal 1层量子权重一致 (caus_q_z vs caus_q_rich48)
        caus_q1_hashes = {
            _tensor_sha256(models[g].transform.weights)
            for g in ("caus_q_z", "caus_q_rich48")
        }
        if len(caus_q1_hashes) != 1:
            checks["causal_q_weights_identical"] = False

        checks["per_seed_hashes"][str(seed)] = {
            "tok_embed": next(iter(tok_emb_hashes)),
            "tok_head4": next(iter(tok_h4_hashes)),
            "tok_q_weights": next(iter(tok_q_hashes)),
            "caus_embed": next(iter(caus_emb_hashes)),
            "caus_conv": next(iter(caus_conv_hashes)),
            "caus_head48": next(iter(caus_h48_hashes)),
            "caus_q1_weights": next(iter(caus_q1_hashes)),
        }

    all_ok = all(
        checks[k]
        for k in (
            "token_embed_identical",
            "token_head4_identical",
            "token_q_weights_identical",
            "causal_frontend_identical",
            "causal_head48_identical",
            "causal_q_weights_identical",
        )
    )
    if not all_ok:
        raise AssertionError(f"受控初始权重断言失败: {checks}")
    checks["all_passed"] = True
    return checks


def _run_single_task(task: dict[str, Any]) -> dict[str, Any]:
    """子进程执行单个 (group_spec, seed) 训练与确定性评估任务。"""
    torch.set_num_threads(1)
    spec = task["spec"]
    seed = int(task["seed"])
    steps = int(task["steps"])
    seq_len = int(task["seq_len"])
    batch_size = int(task["batch_size"])
    lr = float(task["lr"])
    eval_every = int(task["eval_every"])

    train_text, val_text = load_corpus_split()
    train_data, vocab_size, stoi, itos = build_char_data(train_text)
    val_data, _, _, _ = build_char_data(val_text, stoi=stoi)

    model = _build_model_from_spec(spec, vocab_size=int(vocab_size), seed=seed)
    gen = iter_batches(
        data_tensor=train_data,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=seed,
    )
    hist = train_lm(
        model=model,
        batch_gen=gen,
        steps=steps,
        lr=lr,
        seed=seed,
        verbose=False,
        log_every=0,
        data_tensor=train_data,
        val_tensor=val_data,
        seq_len=seq_len,
        eval_every=eval_every,
    )

    prompt = train_text[:seq_len]
    sample_str = greedy_sample(
        model=model,
        vocab=(stoi, itos),
        prompt=prompt,
        length=120 - len(prompt),
        seq_len=seq_len,
    )

    # 计算末 20 步单 batch loss 的标准差，量化单 batch 评价噪声
    tail_losses = hist["loss"][-min(20, len(hist["loss"])):]
    tail_batch_std = float(np.std(tail_losses, ddof=1)) if len(tail_losses) > 1 else 0.0
    tail_batch_span = (
        float(max(tail_losses) - min(tail_losses)) if tail_losses else 0.0
    )

    return {
        "group_id": spec["id"],
        "seed": seed,
        "params": int(count_parameters(model)),
        "readout_dim": int(model.readout_dim),
        "final_loss": float(hist["final_loss"]),
        "final_perplexity": float(hist["final_perplexity"]),
        "eval_loss": float(hist["eval_loss"]),
        "eval_perplexity": float(hist["eval_perplexity"]),
        "val_loss": float(hist["val_loss"]),
        "val_perplexity": float(hist["val_perplexity"]),
        "tail_batch_std": tail_batch_std,
        "tail_batch_span": tail_batch_span,
        "total_seconds": float(hist["total_seconds"]),
        "sample": sample_str,
        "history": {
            "step": hist["step"],
            "loss": hist["loss"],
            "perplexity": hist["perplexity"],
            "seconds": hist["seconds"],
            "eval_steps": hist.get("eval_steps", []),
            "eval_loss_curve": hist.get("eval_loss_curve", []),
            "eval_perplexity_curve": hist.get("eval_perplexity_curve", []),
            "val_loss_curve": hist.get("val_loss_curve", []),
            "val_perplexity_curve": hist.get("val_perplexity_curve", []),
        },
    }


def _paired_ci(
    vals_a: list[float], vals_b: list[float]
) -> dict[str, Any]:
    """计算配对差值 A - B 的均值、样本标准差与双侧 95% Student-t 置信区间。"""
    diffs = [float(a - b) for a, b in zip(vals_a, vals_b)]
    n = len(diffs)
    mean = float(np.mean(diffs))
    if n < 2:
        return {
            "per_seed_diff": diffs,
            "mean_diff": mean,
            "std_diff": 0.0,
            "ci95_half": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
            "significant_95": False,
        }
    std = float(np.std(diffs, ddof=1))
    se = std / math.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    half = t_crit * se
    low, high = mean - half, mean + half
    sig = bool(low > 0.0 or high < 0.0)
    return {
        "per_seed_diff": [round(d, 6) for d in diffs],
        "mean_diff": mean,
        "std_diff": std,
        "ci95_half": half,
        "ci95_low": low,
        "ci95_high": high,
        "significant_95": sig,
    }


def _plot_diagnosis_figure(
    agg: dict[str, dict[str, Any]],
    floors: dict[int, dict[str, float]],
    out_path: Path,
) -> None:
    if not _MPL_OK:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=False)

    # Panel A: Token-wise (k=1)
    tok_order = [
        ("tok_q_legacy", "Q-Legacy\n(4q Z, s=1)"),
        ("tok_q_scaled", "Q-Scaled\n(4q Z, s=5)"),
        ("tok_q_rich24", "Q-Rich24\n(24-Pauli)"),
        ("tok_q_rich48", "Q-Rich48\n(48-Pauli)"),
        ("tok_identity", "Identity\n(No-Q)"),
        ("tok_classical", "Classical\n(Linear+Tanh)"),
    ]
    ax0 = axes[0]
    x0 = np.arange(len(tok_order))
    means0 = [agg[k]["eval_loss_mean"] for k, _ in tok_order]
    stds0 = [agg[k]["eval_loss_std"] for k, _ in tok_order]
    colors0 = ["#9b59b6", "#8e44ad", "#6c3483", "#4a235a", "#7f8c8d", "#2980b9"]
    bars0 = ax0.bar(x0, means0, yerr=stds0, capsize=4, color=colors0, alpha=0.88)
    floor_k1 = floors[1]["loss"]
    ax0.axhline(floor_k1, color="#c0392b", linestyle="--", linewidth=1.6,
                label=f"k=1 Bigram Entropy Floor ({floor_k1:.4f} nats)")
    for b, m in zip(bars0, means0):
        ax0.text(b.get_x() + b.get_width() / 2, m + 0.03, f"{m:.3f}",
                 ha="center", va="bottom", fontsize=8.5)
    ax0.set_xticks(x0)
    ax0.set_xticklabels([lbl for _, lbl in tok_order], fontsize=8.5)
    ax0.set_ylabel("Deterministic Full-Corpus Loss (nats)")
    ax0.set_title("Part 2: Token-Wise Regime (k=1 Context)")
    ax0.set_ylim(1.75, max(means0) + 0.25)
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.legend(loc="upper right", fontsize=8.5)

    # Panel B: Causal (k=4) Train & Held-out Val
    caus_order = [
        ("caus_q_z", "Causal Q-Z\n(4-Pauli)"),
        ("caus_identity", "Causal Identity\n(No-Q)"),
        ("caus_q_reup", "Causal Q-Reup\n(12-Pauli, 2L)"),
        ("caus_classical", "Causal Classical\n(48-Tanh+MLP)"),
        ("caus_q_rich48", "Causal Q-Rich48\n(48-Pauli+MLP)"),
    ]
    ax1 = axes[1]
    x1 = np.arange(len(caus_order))
    w = 0.36
    tr_means = [agg[k]["eval_loss_mean"] for k, _ in caus_order]
    tr_stds = [agg[k]["eval_loss_std"] for k, _ in caus_order]
    val_means = [agg[k]["val_loss_mean"] for k, _ in caus_order]
    val_stds = [agg[k]["val_loss_std"] for k, _ in caus_order]

    b_tr = ax1.bar(x1 - w / 2, tr_means, w, yerr=tr_stds, capsize=3,
                   label="Train Corpus Eval Loss", color="#2e86c1", alpha=0.9)
    b_vl = ax1.bar(x1 + w / 2, val_means, w, yerr=val_stds, capsize=3,
                   label="Held-Out Val Sentences Loss", color="#e67e22", alpha=0.85)
    ax1.axhline(floor_k1, color="#c0392b", linestyle="--", linewidth=1.3,
                label=f"k=1 Bigram Wall ({floor_k1:.2f} nats)")
    floor_k4_win = floors[4]["window16_loss"]
    ax1.axhline(floor_k4_win, color="#27ae60", linestyle=":", linewidth=1.5,
                label=f"k=4 Window-16 Floor ({floor_k4_win:.2f} nats)")
    for b, m in zip(b_tr, tr_means):
        ax1.text(b.get_x() + b.get_width() / 2, m + 0.04, f"{m:.2f}",
                 ha="center", va="bottom", fontsize=8)
    for b, m in zip(b_vl, val_means):
        ax1.text(b.get_x() + b.get_width() / 2, m + 0.04, f"{m:.2f}",
                 ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([lbl for _, lbl in caus_order], fontsize=8.5)
    ax1.set_ylabel("Cross-Entropy Loss (nats)")
    ax1.set_title("Part 3: Causal Context Regime (k=4 Window)")
    ax1.set_ylim(0.0, max(val_means + tr_means) + 0.35)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=140, format="png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5-Seed 系统方法学诊断与消融实验")
    p.add_argument("--qubits", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.015)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--eval-every", type=int, default=60)
    p.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    p.add_argument("--quick", action="store_true", help="快速冒烟运行（仅写 results/，不改 docs/）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.qubits = 2
        args.seq_len = 8
        args.batch_size = 4
        args.steps = 4
        args.seeds = [0, 1]
        args.eval_every = 2
        args.workers = 2

    train_text, val_text = load_corpus_split()
    train_data, vocab_size, stoi, itos = build_char_data(train_text)
    val_data, _, _, _ = build_char_data(val_text, stoi=stoi)
    vocab_size = int(vocab_size)

    # Part 1: 信息论 N-Gram 条件熵下界与精确经验 Oracle 轨迹
    floors = corpus_entropy_floors(train_text, max_context=16, seq_len=args.seq_len)
    val_floors = corpus_entropy_floors(val_text, max_context=8, seq_len=args.seq_len)
    oracle_samples = _oracle_greedy_samples(train_text)

    print("=" * 78)
    print("Part 1: 语料经验条件熵下界 H(C_{t+1} | C_{t-k+1..t})")
    print("=" * 78)
    floor_rows = []
    for k in (-1, 0, 1, 2, 3, 4, 8, 16):
        if k in floors:
            f = floors[k]
            label = "Uniform ln(V)" if k == -1 else f"k={k} ({k + 1}-gram)"
            floor_rows.append([
                label,
                int(f["unique_contexts"]),
                f"{f['loss']:.4f}",
                f"{f['perplexity']:.2f}",
                f"{f['window16_loss']:.4f}",
                f"{f['window16_perplexity']:.2f}",
            ])
    print(
        fmt_table(
            floor_rows,
            ["上下文阶数", "唯一上下文数", "条件熵 H (nats)", "理论最低 PPL", "窗口(T=16)下界", "窗口最低 PPL"],
        )
    )
    print("\n精确经验 k-gram Oracle 贪心解码轨迹：")
    for k in (1, 2, 4, 8):
        print(f"  k={k}: {oracle_samples[k][:88]}")

    # 校验初始权重受控性
    specs = _group_specs(qubits=args.qubits)
    weight_checks = _verify_initial_weight_control(specs, vocab_size, args.seeds)
    print(f"\n[受控性校验] 初始权重 SHA256 对齐断言全部通过: {weight_checks['all_passed']}")

    # 组装 11 组 × 5 seeds 任务；优先调度耗时较长的多观测量量子组以最小化总体墙钟耗时
    priority_order = {
        "caus_q_rich48": 0,
        "tok_q_rich48": 1,
        "tok_q_rich24": 2,
        "caus_q_reup": 3,
        "caus_q_z": 4,
        "tok_q_legacy": 5,
        "tok_q_scaled": 6,
    }
    tasks: list[dict[str, Any]] = []
    for spec in sorted(specs, key=lambda s: priority_order.get(s["id"], 10)):
        for seed in args.seeds:
            tasks.append(
                {
                    "spec": spec,
                    "seed": int(seed),
                    "steps": args.steps,
                    "seq_len": args.seq_len,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "eval_every": args.eval_every,
                }
            )

    print(
        f"\n开始并行执行 {len(specs)} 组 × {len(args.seeds)} seeds = {len(tasks)} 个训练任务 "
        f"(workers={args.workers}, steps={args.steps})..."
    )
    raw_results: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fut_map = {ex.submit(_run_single_task, t): t for t in tasks}
        done_cnt = 0
        for fut in as_completed(fut_map):
            res = fut.result()
            done_cnt += 1
            gid = res["group_id"]
            sd = res["seed"]
            raw_results[gid][sd] = res
            if done_cnt % len(args.seeds) == 0 or done_cnt == len(tasks):
                print(
                    f"  [{done_cnt:>2}/{len(tasks)}] 完成 {gid:<15s} seed={sd} | "
                    f"eval_loss={res['eval_loss']:.4f} (ppl {res['eval_perplexity']:.2f}) | "
                    f"val_loss={res['val_loss']:.4f} | {res['total_seconds']:.1f}s"
                )

    # 按 seed 排序汇总统计
    agg: dict[str, dict[str, Any]] = {}
    for spec in specs:
        gid = spec["id"]
        runs = [raw_results[gid][s] for s in args.seeds]
        eval_losses = [r["eval_loss"] for r in runs]
        eval_ppls = [r["eval_perplexity"] for r in runs]
        val_losses = [r["val_loss"] for r in runs]
        val_ppls = [r["val_perplexity"] for r in runs]
        final_losses = [r["final_loss"] for r in runs]
        tail_stds = [r["tail_batch_std"] for r in runs]
        secs = [r["total_seconds"] for r in runs]

        agg[gid] = {
            "id": gid,
            "label": spec["label"],
            "regime": spec["regime"],
            "transform_type": spec["transform_type"],
            "context_mode": spec["context_mode"],
            "context_window": spec["context_window"],
            "observables": spec["observables"],
            "reupload": spec["reupload"],
            "pre_scale": spec["pre_scale"],
            "near_identity_init": spec["near_identity_init"],
            "scale": spec["scale"],
            "head": spec["head"],
            "params": runs[0]["params"],
            "readout_dim": runs[0]["readout_dim"],
            "eval_loss_mean": float(np.mean(eval_losses)),
            "eval_loss_std": float(np.std(eval_losses, ddof=1)) if len(runs) > 1 else 0.0,
            "eval_ppl_mean": float(np.mean(eval_ppls)),
            "eval_ppl_std": float(np.std(eval_ppls, ddof=1)) if len(runs) > 1 else 0.0,
            "val_loss_mean": float(np.mean(val_losses)),
            "val_loss_std": float(np.std(val_losses, ddof=1)) if len(runs) > 1 else 0.0,
            "val_ppl_mean": float(np.mean(val_ppls)),
            "val_ppl_std": float(np.std(val_ppls, ddof=1)) if len(runs) > 1 else 0.0,
            "final_batch_loss_mean": float(np.mean(final_losses)),
            "final_batch_loss_std": float(np.std(final_losses, ddof=1)) if len(runs) > 1 else 0.0,
            "within_run_tail20_batch_std": float(np.mean(tail_stds)),
            "seconds_mean": float(np.mean(secs)),
            "seconds_std": float(np.std(secs, ddof=1)) if len(runs) > 1 else 0.0,
            "per_seed_eval_loss": [round(x, 6) for x in eval_losses],
            "per_seed_val_loss": [round(x, 6) for x in val_losses],
            "seed0_sample": runs[0]["sample"],
            "per_seed": [
                {k: v for k, v in r.items() if k != "history"} for r in runs
            ],
            "seed0_history": runs[0]["history"],
        }

    # 配对统计检验（负值表示前者 loss 更低 / 更优）
    pairs = [
        ("tok_q_scaled", "tok_q_legacy", "Token: 角度预缩放 + 5x 缩放 vs 原版基线"),
        ("tok_q_rich24", "tok_q_scaled", "Token: 24维读出 vs 4维 Z 读出"),
        ("tok_q_rich48", "tok_q_scaled", "Token: 48维全两体读出 vs 4维 Z 读出"),
        ("tok_q_rich48", "tok_classical", "Token: 48维量子 vs 经典 Linear(4,4)+Tanh"),
        ("tok_q_scaled", "tok_identity", "Token: 4维量子 Z vs 无量子恒等映射"),
        ("caus_identity", "tok_identity", "因果窗口 (k=4) vs 逐字符 (k=1) [Identity 对照]"),
        ("caus_q_reup", "caus_q_z", "Causal: 2层可训练重上传(12维) vs 4维 Z 读出"),
        ("caus_q_rich48", "caus_q_z", "Causal: 48维两体读出+MLP vs 4维 Z 读出"),
        ("caus_q_rich48", "caus_identity", "Causal: 48维两体量子 vs 无量子恒等瓶颈"),
        ("caus_q_rich48", "caus_classical", "Causal: 48维两体量子 vs 同结构48维经典瓶颈"),
        ("caus_q_rich48", "tok_q_legacy", "总体演进: Causal Q-Rich48 vs 初版 Token Q-Legacy"),
    ]
    paired_stats: dict[str, Any] = {}
    for a, b, desc in pairs:
        key = f"{a}_minus_{b}"
        paired_stats[key] = {
            "comparison": f"{a} - {b}",
            "description": desc,
            "train_eval_loss": _paired_ci(
                agg[a]["per_seed_eval_loss"], agg[b]["per_seed_eval_loss"]
            ),
            "heldout_val_loss": _paired_ci(
                agg[a]["per_seed_val_loss"], agg[b]["per_seed_val_loss"]
            ),
        }

    # 打印汇总表格
    print("\n" + "=" * 78)
    print("Part 2 & Part 3: 5-Seed 受控实验结果汇总（均值 ± 标准差）")
    print("=" * 78)
    table_rows = []
    for spec in specs:
        g = agg[spec["id"]]
        table_rows.append([
            g["id"],
            g["regime"],
            g["params"],
            g["readout_dim"],
            f"{g['eval_loss_mean']:.4f} ± {g['eval_loss_std']:.4f}",
            f"{g['eval_ppl_mean']:.2f} ± {g['eval_ppl_std']:.2f}",
            f"{g['val_loss_mean']:.4f} ± {g['val_loss_std']:.4f}",
            f"{g['val_ppl_mean']:.2f}",
            f"{g['seconds_mean']:.2f}s",
        ])
    print(
        fmt_table(
            table_rows,
            ["组 ID", "模式", "参数量", "读出维", "Train Eval Loss", "Train PPL", "Held-out Val Loss", "Val PPL", "平均耗时"],
        )
    )

    print("\n配对差值检验（A - B，负数代表 A 的 Loss 更低/更优，双侧 95% Student-t CI）：")
    ci_rows = []
    for key, item in paired_stats.items():
        tr = item["train_eval_loss"]
        vl = item["heldout_val_loss"]
        ci_rows.append([
            item["comparison"],
            f"{tr['mean_diff']:+.4f} ± {tr['std_diff']:.4f}",
            f"[{tr['ci95_low']:+.4f}, {tr['ci95_high']:+.4f}]",
            "显著" if tr["significant_95"] else "不显著",
            f"{vl['mean_diff']:+.4f} [{vl['ci95_low']:+.4f}, {vl['ci95_high']:+.4f}]",
        ])
    print(
        fmt_table(
            ci_rows,
            ["对比 (A - B)", "Train ΔLoss", "Train 95% CI", "显著性", "Val ΔLoss [95% CI]"],
        )
    )

    print("\nSeed 0 代表性贪心生成样本（前 96 字符）：")
    for gid in ("tok_q_legacy", "tok_q_rich48", "caus_q_z", "caus_q_rich48", "caus_classical"):
        print(f"  {gid:<15s}: {agg[gid]['seed0_sample'][:96]}")

    payload = {
        "config": {
            "qubits": args.qubits,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "lr": args.lr,
            "seeds": args.seeds,
            "eval_every": args.eval_every,
            "quick": args.quick,
            "train_corpus_chars": len(train_text),
            "val_corpus_chars": len(val_text),
            "vocab_size": vocab_size,
        },
        "entropy_floors": {
            "train": {str(k): v for k, v in floors.items()},
            "val": {str(k): v for k, v in val_floors.items()},
            "oracle_greedy_samples": {str(k): s for k, s in oracle_samples.items()},
        },
        "weight_control_checks": weight_checks,
        "groups": {k: {kk: vv for kk, vv in v.items() if kk != "seed0_history"} for k, v in agg.items()},
        "paired_comparisons": paired_stats,
        "caveat": CAVEAT,
    }

    if args.quick:
        out_json = Path("results/quick_llm_metrics.json")
        fig_dir = ensure_dir("results/figures")
        save_json(payload, str(out_json))
        _plot_diagnosis_figure(agg, floors, fig_dir / "lm_diagnosis.png")
        print(f"\n[--quick] 已写入 {out_json}（未修改 docs/）")
    else:
        out_json = Path("docs/llm_diagnosis_metrics.json")
        fig_dir = ensure_dir("docs/figures")
        save_json(payload, str(out_json))
        _plot_diagnosis_figure(agg, floors, fig_dir / "lm_diagnosis.png")

        # 同步更新主图 docs/figures/lm_{loss,perplexity,time}.png
        curve_histories = {
            "Token Q-Legacy (k=1)": agg["tok_q_legacy"]["seed0_history"],
            "Token Q-Rich48 (k=1)": agg["tok_q_rich48"]["seed0_history"],
            "Token Classical (k=1)": agg["tok_classical"]["seed0_history"],
            "Causal Q-Rich48 (k=4)": agg["caus_q_rich48"]["seed0_history"],
            "Causal Classical (k=4)": agg["caus_classical"]["seed0_history"],
        }
        plot_curves(
            histories=curve_histories,
            metric="loss",
            ylabel="Mini-batch Loss (nats)",
            path=str(fig_dir / "lm_loss.png"),
            title="Quantum-LLM: Training Loss Across Regimes (Seed 0)",
        )
        plot_curves(
            histories=curve_histories,
            metric="perplexity",
            ylabel="Mini-batch Perplexity",
            path=str(fig_dir / "lm_perplexity.png"),
            title="Quantum-LLM: Training Perplexity Across Regimes (Seed 0)",
        )
        plot_bars(
            values={
                "Tok Q-Legacy": agg["tok_q_legacy"]["seconds_mean"],
                "Tok Q-Rich48": agg["tok_q_rich48"]["seconds_mean"],
                "Tok Classical": agg["tok_classical"]["seconds_mean"],
                "Caus Q-Reup": agg["caus_q_reup"]["seconds_mean"],
                "Caus Q-Rich48": agg["caus_q_rich48"]["seconds_mean"],
                "Caus Classical": agg["caus_classical"]["seconds_mean"],
            },
            ylabel="Mean Training Time (s, 300 steps)",
            path=str(fig_dir / "lm_time.png"),
            title="Quantum-LLM: 5-Seed Mean Training Wall Time",
        )
        print(f"\n[完成] 已归档完整 5-seed 指标至 {out_json} 并更新 {fig_dir}/ 下全部图表。")


if __name__ == "__main__":
    main()
