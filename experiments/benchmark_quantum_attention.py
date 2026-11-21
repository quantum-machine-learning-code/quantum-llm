"""Quantum Self-Attention Three-Regime Benchmark (SWAP-Test vs LCU+QSVT Quixer vs Classical).

Evaluates strictly parameter-matched (~924-946 params) attention models across all 3 regimes on
zero-overlap Key-Value Associative Recall (T_train=6 -> OOD T_OOD=10, 4 keys x 4 values, chance=0.25):

  * Regime I:  ``classical_dot_attn``, ``classical_overlap_attn``, ``classical_linear_mixer``
  * Regime II: ``quantum_swap_test_attn`` (analytic + finite-shot Bernoulli sampling S in {100,500,2000})
  * Regime III: ``quantum_quixer_lcu_qsvt`` (0 MLP crutch, learned PREPARE + degree-3 QSVT + POVM)
                plus ``quantum_quixer_uniform_lcu`` ablation (fixed uniform alpha_t = 1/N).

Usage:
    python experiments/benchmark_quantum_attention.py --quick   # fast smoke -> results/
    python experiments/benchmark_quantum_attention.py           # full 5-seed -> docs/ + figures
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F

from qml_lab.quantum_attention import (
    ClassicalDotAttention,
    ClassicalLinearMixer,
    ClassicalOverlapAttention,
    QuantumQuixerAttention,
    QuantumSwapTestAttention,
    attention_parameter_breakdown,
    build_kv_recall_dataset,
)
from qml_lab.utils import set_seed

DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
RESULTS = ROOT / "results"

SHOT_BUDGETS = (100, 500, 2000)


def _paired_ci(a: list[float], b: list[float]) -> dict[str, Any]:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    if n > 1 and se > 0:
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        half = tcrit * se
        pval = float(2 * stats.t.sf(abs(mean / se), df=n - 1))
    elif n > 1 and abs(mean) > 1e-12:
        half, pval = 0.0, 0.0
    else:
        half, pval = 0.0, 1.0
    return {
        "n": n,
        "mean_diff": mean,
        "std_diff": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "p_value": pval,
        "significant_95_uncorrected": bool(mean - half > 0 or mean + half < 0),
        "per_seed_diff": [round(float(x), 6) for x in d],
    }


def _holm(tests: dict[str, dict[str, Any]], keys: list[str]) -> None:
    order = sorted(keys, key=lambda k: tests[k]["p_value"])
    m = len(order)
    running = 0.0
    for idx, key in enumerate(order):
        adj = min(1.0, (m - idx) * tests[key]["p_value"])
        running = max(running, adj)
        tests[key]["p_holm"] = float(running)
        tests[key]["holm_significant"] = bool(running < 0.05)


def _mean_std(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def _build_arm(arm: str, vocab_size: int, n_classes: int) -> tuple[torch.nn.Module, bool]:
    if arm == "classical_dot_attn":
        return ClassicalDotAttention(vocab_size=vocab_size, n_classes=n_classes, embed_dim=20, key_dim=4, val_dim=14), False
    if arm == "classical_overlap_attn":
        return ClassicalOverlapAttention(vocab_size=vocab_size, n_classes=n_classes, embed_dim=20, key_dim=4, val_dim=14), False
    if arm == "classical_linear_mixer":
        return ClassicalLinearMixer(vocab_size=vocab_size, n_classes=n_classes, embed_dim=20, key_dim=4, val_dim=13), False
    if arm == "quantum_swap_test_attn":
        return QuantumSwapTestAttention(vocab_size=vocab_size, n_classes=n_classes, n_qubits=2, embed_dim=20, val_dim=14), False
    if arm == "quantum_quixer_lcu_qsvt":
        return QuantumQuixerAttention(vocab_size=vocab_size, n_classes=n_classes, n_qubits=3, povm_rank=2, learned_prepare=True, use_qsvt=True), True
    if arm == "quantum_quixer_uniform_lcu":
        return QuantumQuixerAttention(vocab_size=vocab_size, n_classes=n_classes, n_qubits=3, povm_rank=2, learned_prepare=False, use_qsvt=True), True
    raise ValueError(f"Unknown arm: {arm}")


def _eval_arm(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, is_log_probs: bool) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        if isinstance(model, QuantumQuixerAttention):
            logp, _, p_succ = model.forward_with_stats(x)
            p_succ_mean = float(p_succ.mean().item())
            p_succ_min = float(p_succ.min().item())
        else:
            out = model(x)
            logp = out if is_log_probs else F.log_softmax(out, dim=-1)
            p_succ_mean = None
            p_succ_min = None
        loss = float(F.nll_loss(logp, y).item())
        acc = float((logp.argmax(dim=-1) == y).float().mean().item())
    res: dict[str, float] = {"loss": loss, "acc": acc}
    if p_succ_mean is not None:
        res["p_succ_mean"] = p_succ_mean
        res["p_succ_min"] = p_succ_min
    return res


def _run_one(task: tuple[str, dict[str, Any], int, float, int]) -> tuple[str, dict[str, Any]]:
    arm, ds, steps, lr, seed = task
    torch.set_num_threads(1)
    set_seed(seed)
    model, is_log_probs = _build_arm(arm, ds["vocab_size"], ds["n_classes"])
    p_info = attention_parameter_breakdown(model)

    x_tr, y_tr = ds["train"]
    x_va, y_va = ds["val"]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(30_000 + seed)
    best_val = float("inf")
    best_state = None
    eval_every = max(1, steps // 10)
    batch_size = 32
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, x_tr.shape[0], (min(batch_size, x_tr.shape[0]),), generator=gen)
        opt.zero_grad(set_to_none=True)
        out = model(x_tr[idx])
        logp = out if is_log_probs else F.log_softmax(out, dim=-1)
        loss = F.nll_loss(logp, y_tr[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % eval_every == 0 or step == steps:
            ev = _eval_arm(model, x_va, y_va, is_log_probs)
            if ev["loss"] < best_val:
                best_val = ev["loss"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    wall = time.perf_counter() - t0

    id_ev = _eval_arm(model, *ds["test_id"], is_log_probs)
    ood_ev = _eval_arm(model, *ds["test_ood"], is_log_probs)

    finite_shots: dict[str, Any] | None = None
    pl_verify: dict[str, float] | None = None

    if isinstance(model, QuantumSwapTestAttention):
        finite_shots = {}
        for s_idx, shots in enumerate(SHOT_BUDGETS):
            rng_id = np.random.default_rng(70_000 + seed * 100 + s_idx)
            logits_id, st_id = model.forward_finite_shots(ds["test_id"][0], shots=shots, rng=rng_id)
            acc_id_s = float((logits_id.argmax(dim=-1) == ds["test_id"][1]).float().mean().item())
            rng_ood = np.random.default_rng(80_000 + seed * 100 + s_idx)
            logits_ood, st_ood = model.forward_finite_shots(ds["test_ood"][0], shots=shots, rng=rng_ood)
            acc_ood_s = float((logits_ood.argmax(dim=-1) == ds["test_ood"][1]).float().mean().item())
            finite_shots[str(shots)] = {
                "acc_id": acc_id_s,
                "acc_ood": acc_ood_s,
                "shots_last_token_id": st_id["shots_last_token_query"],
                "shots_full_causal_id": st_id["shots_full_causal_matrix"],
                "shots_last_token_ood": st_ood["shots_last_token_query"],
                "shots_full_causal_ood": st_ood["shots_full_causal_matrix"],
                "overlap_mae_id": st_id["overlap_mae"],
                "overlap_mae_ood": st_ood["overlap_mae"],
            }
        if seed == 0:
            pl_verify = model.verify_against_pennylane(ds["test_id"][0][0])

    if isinstance(model, QuantumQuixerAttention) and seed == 0:
        pl_verify = model.verify_against_pennylane(ds["test_id"][0][0])

    return arm, {
        "seed": seed,
        "params": p_info,
        "best_val_loss": best_val,
        "test_id": id_ev,
        "test_ood": ood_ev,
        "finite_shots": finite_shots,
        "pennylane_verification": pl_verify,
        "seconds": wall,
    }


def plot_attention_figures(report: dict[str, Any], fig_dir: Path) -> list[str]:
    """Generate two 300-dpi publication-quality English-label figures from verified metrics."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    arms_order = [
        ("classical_dot_attn", "Regime I\nDot Softmax", "#3b82f6"),
        ("classical_overlap_attn", "Regime I\nOverlap Surr.", "#60a5fa"),
        ("classical_linear_mixer", "Regime I\nLinear Mixer", "#93c5fd"),
        ("quantum_swap_test_attn", "Regime II\nSWAP-Test (∞)", "#f59e0b"),
        ("quantum_quixer_lcu_qsvt", "Regime III\nQuixer LCU+QSVT", "#10b981"),
        ("quantum_quixer_uniform_lcu", "Regime III\nQuixer Uniform", "#6ee7b7"),
    ]
    arms_data = report["arms"]

    # Figure 1: ID vs OOD accuracy across all 3 regimes
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    xs = np.arange(len(arms_order))
    width = 0.36
    id_means = [arms_data[k]["test_id_acc"]["mean"] for k, _, _ in arms_order]
    id_stds = [arms_data[k]["test_id_acc"]["std"] for k, _, _ in arms_order]
    ood_means = [arms_data[k]["test_ood_acc"]["mean"] for k, _, _ in arms_order]
    ood_stds = [arms_data[k]["test_ood_acc"]["std"] for k, _, _ in arms_order]
    colors = [c for _, _, c in arms_order]

    b1 = ax.bar(xs - width / 2, id_means, width, yerr=id_stds, capsize=3, color=colors, edgecolor="#1f2937", linewidth=0.8, label="Test-ID (T=6)")
    b2 = ax.bar(xs + width / 2, ood_means, width, yerr=ood_stds, capsize=3, color=colors, alpha=0.58, hatch="//", edgecolor="#1f2937", linewidth=0.8, label="Test-OOD (T=10)")
    ax.axhline(report["config"]["chance_accuracy"], color="#ef4444", linestyle="--", linewidth=1.2, label="Random chance (0.25)")
    for bar, m in zip(b1, id_means):
        ax.text(bar.get_x() + bar.get_width() / 2, min(1.03, m + 0.025), f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, m in zip(b2, ood_means):
        ax.text(bar.get_x() + bar.get_width() / 2, min(1.03, m + 0.025), f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl, _ in arms_order], fontsize=8.5)
    ax.set_ylim(0.0, 1.14)
    ax.set_ylabel("Retrieval Accuracy (mean ± SD, 5 seeds)")
    ax.set_title("Quantum Self-Attention Benchmark: KV Associative Recall (T=6 -> OOD T=10)")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8.5)
    fig.tight_layout()
    p1 = fig_dir / "quantum_attention_accuracy.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)

    # Figure 2: Left = SWAP-Test finite-shot accuracy + overlap MAE / shot count; Right = LCU+QSVT p_succ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.5))
    sw = arms_data["quantum_swap_test_attn"]
    shot_labels = ["S=100\n(5.5k)", "S=500\n(27.5k)", "S=2000\n(110k)", "Analytic\n(S=∞)"]
    id_s_m = [sw["finite_shots"][str(s)]["acc_id"]["mean"] for s in SHOT_BUDGETS] + [sw["test_id_acc"]["mean"]]
    id_s_sd = [sw["finite_shots"][str(s)]["acc_id"]["std"] for s in SHOT_BUDGETS] + [sw["test_id_acc"]["std"]]
    ood_s_m = [sw["finite_shots"][str(s)]["acc_ood"]["mean"] for s in SHOT_BUDGETS] + [sw["test_ood_acc"]["mean"]]
    ood_s_sd = [sw["finite_shots"][str(s)]["acc_ood"]["std"] for s in SHOT_BUDGETS] + [sw["test_ood_acc"]["std"]]
    mae_ood = [sw["finite_shots"][str(s)]["overlap_mae_ood"]["mean"] for s in SHOT_BUDGETS] + [0.0]
    sx = np.arange(len(shot_labels))
    l1 = ax1.errorbar(sx, id_s_m, yerr=id_s_sd, marker="o", linewidth=2, capsize=4, color="#d97706", label="Acc ID (T=6)")
    l2 = ax1.errorbar(sx, ood_s_m, yerr=ood_s_sd, marker="s", linestyle="--", linewidth=2, capsize=4, color="#92400e", label="Acc OOD (T=10)")
    for xi, ym, yo in zip(sx, id_s_m, ood_s_m):
        ax1.text(xi, ym + 0.018, f"{ym:.3f}", ha="center", va="bottom", fontsize=7.8, color="#b45309")
        ax1.text(xi, yo - 0.042, f"{yo:.3f}", ha="center", va="top", fontsize=7.8, color="#78350f")
    ax1.set_xticks(sx)
    ax1.set_xticklabels(shot_labels, fontsize=8.0)
    ax1.set_xlabel("Shots / pair (OOD T=10 full causal shots = 55·S)")
    ax1.set_ylim(0.75, 1.08)
    ax1.set_ylabel("Retrieval Accuracy (mean ± SD)", color="#92400e")
    ax1.set_title("Regime II SWAP-Test: Shots, Accuracy & Overlap MAE")
    ax1.grid(True, alpha=0.25)

    ax1_r = ax1.twinx()
    l3 = ax1_r.plot(sx, mae_ood, marker="^", linestyle=":", linewidth=1.8, color="#2563eb", label="Overlap MAE (OOD)")
    for xi, mv in zip(sx[:-1], mae_ood[:-1]):
        ax1_r.text(xi, mv + 0.004, f"MAE {mv:.3f}", ha="center", va="bottom", fontsize=7.3, color="#1d4ed8")
    ax1_r.set_ylim(0.0, 0.11)
    ax1_r.set_ylabel("SWAP-Test Overlap MAE [O(1/√S)]", color="#1d4ed8")
    handles = [l1, l2, l3[0]]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="center right", fontsize=7.6, framealpha=0.92)

    qx_l = arms_data["quantum_quixer_lcu_qsvt"]
    qx_u = arms_data["quantum_quixer_uniform_lcu"]
    q_labels = ["Learned PREP\nT=6 (ID)", "Learned PREP\nT=10 (OOD)", "Uniform LCU\nT=6 (ID)", "Uniform LCU\nT=10 (OOD)"]
    ps_m = [
        qx_l["p_succ_id"]["mean"],
        qx_l["p_succ_ood"]["mean"],
        qx_u["p_succ_id"]["mean"],
        qx_u["p_succ_ood"]["mean"],
    ]
    ps_sd = [
        qx_l["p_succ_id"]["std"],
        qx_l["p_succ_ood"]["std"],
        qx_u["p_succ_id"]["std"],
        qx_u["p_succ_ood"]["std"],
    ]
    qx_x = np.arange(len(q_labels))
    bars = ax2.bar(qx_x, ps_m, yerr=ps_sd, capsize=4, color=["#10b981", "#059669", "#6ee7b7", "#34d399"], edgecolor="#1f2937", linewidth=0.8)
    for bar, m, sd in zip(bars, ps_m, ps_sd):
        aa_rounds = 1.0 / math.sqrt(max(m, 1e-6))
        ax2.text(bar.get_x() + bar.get_width() / 2, m + sd + 0.02, f"{m:.3f}\n(~{aa_rounds:.1f}× AA)", ha="center", va="bottom", fontsize=7.6)
    ax2.set_xticks(qx_x)
    ax2.set_xticklabels(q_labels, fontsize=8.0)
    ax2.set_ylim(0.0, max(m + sd for m, sd in zip(ps_m, ps_sd)) + 0.16)
    ax2.set_ylabel("Ancilla Post-Selection Probability $p_{\\mathrm{succ}}$")
    ax2.set_title("Regime III Quixer LCU+QSVT: Post-Selection Cost")
    ax2.grid(axis="y", alpha=0.28)

    fig.tight_layout()
    p2 = fig_dir / "quantum_attention_shots_and_lcu.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    return [str(p1), str(p2)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--lr", type=float, default=0.025)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.seeds = [0, 1]
        args.steps = 30
        args.workers = 1

    torch.set_num_threads(1)
    t0 = time.perf_counter()

    arms = (
        "classical_dot_attn",
        "classical_overlap_attn",
        "classical_linear_mixer",
        "quantum_swap_test_attn",
        "quantum_quixer_lcu_qsvt",
        "quantum_quixer_uniform_lcu",
    )
    datasets = {
        seed: build_kv_recall_dataset(
            n_keys=4,
            n_values=4,
            n_distractors=2,
            train_len=6,
            ood_len=10,
            n_train=160 if args.quick else 640,
            n_val=64 if args.quick else 160,
            n_test_id=64 if args.quick else 200,
            n_test_ood=64 if args.quick else 200,
            seed=400 + seed,
        )
        for seed in args.seeds
    }
    tasks = [(arm, datasets[seed], args.steps, args.lr, seed) for arm in arms for seed in args.seeds]
    per_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for arm, res in ex.map(_run_one, tasks):
                per_arm[arm].append(res)
    else:
        for task in tasks:
            arm, res = _run_one(task)
            per_arm[arm].append(res)
    for arm in arms:
        per_arm[arm].sort(key=lambda r: r["seed"])

    print("\n=== Quantum Self-Attention Benchmark: KV Associative Recall (T=6 -> OOD T=10) ===", flush=True)
    agg: dict[str, Any] = {}
    for arm in arms:
        runs = per_arm[arm]
        entry: dict[str, Any] = {
            "params": runs[0]["params"],
            "test_id_acc": _mean_std([r["test_id"]["acc"] for r in runs]),
            "test_id_loss": _mean_std([r["test_id"]["loss"] for r in runs]),
            "test_ood_acc": _mean_std([r["test_ood"]["acc"] for r in runs]),
            "test_ood_loss": _mean_std([r["test_ood"]["loss"] for r in runs]),
            "seconds_mean": float(np.mean([r["seconds"] for r in runs])),
        }
        if "p_succ_mean" in runs[0]["test_id"]:
            entry["p_succ_id"] = _mean_std([r["test_id"]["p_succ_mean"] for r in runs])
            entry["p_succ_ood"] = _mean_std([r["test_ood"]["p_succ_mean"] for r in runs])
        if runs[0]["finite_shots"] is not None:
            fs_agg: dict[str, Any] = {}
            for s in SHOT_BUDGETS:
                sk = str(s)
                fs_agg[sk] = {
                    "acc_id": _mean_std([r["finite_shots"][sk]["acc_id"] for r in runs]),
                    "acc_ood": _mean_std([r["finite_shots"][sk]["acc_ood"] for r in runs]),
                    "overlap_mae_id": _mean_std([r["finite_shots"][sk]["overlap_mae_id"] for r in runs]),
                    "overlap_mae_ood": _mean_std([r["finite_shots"][sk]["overlap_mae_ood"] for r in runs]),
                    "shots_last_token_id": runs[0]["finite_shots"][sk]["shots_last_token_id"],
                    "shots_full_causal_id": runs[0]["finite_shots"][sk]["shots_full_causal_id"],
                    "shots_last_token_ood": runs[0]["finite_shots"][sk]["shots_last_token_ood"],
                    "shots_full_causal_ood": runs[0]["finite_shots"][sk]["shots_full_causal_ood"],
                }
            entry["finite_shots"] = fs_agg
        if runs[0]["pennylane_verification"] is not None:
            entry["pennylane_verification"] = runs[0]["pennylane_verification"]
        agg[arm] = entry
        extra = ""
        if "p_succ_id" in entry:
            extra = f" | p_succ ID={entry['p_succ_id']['mean']:.3f} OOD={entry['p_succ_ood']['mean']:.3f}"
        print(
            f"  {arm:28s} (p={entry['params']['total']:3d}, q={entry['params']['quantum_params']:3d}) | "
            f"ID={entry['test_id_acc']['mean']:.4f}±{entry['test_id_acc']['std']:.4f} | "
            f"OOD={entry['test_ood_acc']['mean']:.4f}±{entry['test_ood_acc']['std']:.4f}{extra} | "
            f"{entry['seconds_mean']:.2f}s",
            flush=True,
        )

    sw_runs = per_arm["quantum_swap_test_attn"]
    paired: dict[str, dict[str, Any]] = {
        "swap_analytic_minus_overlap|acc_id": _paired_ci(
            [r["test_id"]["acc"] for r in sw_runs],
            [r["test_id"]["acc"] for r in per_arm["classical_overlap_attn"]],
        ),
        "swap_analytic_minus_dot|acc_ood": _paired_ci(
            [r["test_ood"]["acc"] for r in sw_runs],
            [r["test_ood"]["acc"] for r in per_arm["classical_dot_attn"]],
        ),
        "swap_s100_minus_analytic|acc_ood": _paired_ci(
            [r["finite_shots"]["100"]["acc_ood"] for r in sw_runs],
            [r["test_ood"]["acc"] for r in sw_runs],
        ),
        "swap_s500_minus_analytic|acc_ood": _paired_ci(
            [r["finite_shots"]["500"]["acc_ood"] for r in sw_runs],
            [r["test_ood"]["acc"] for r in sw_runs],
        ),
        "quixer_minus_dot|acc_ood": _paired_ci(
            [r["test_ood"]["acc"] for r in per_arm["quantum_quixer_lcu_qsvt"]],
            [r["test_ood"]["acc"] for r in per_arm["classical_dot_attn"]],
        ),
        "quixer_minus_linear_mixer|acc_ood": _paired_ci(
            [r["test_ood"]["acc"] for r in per_arm["quantum_quixer_lcu_qsvt"]],
            [r["test_ood"]["acc"] for r in per_arm["classical_linear_mixer"]],
        ),
        "quixer_minus_uniform_lcu|acc_ood": _paired_ci(
            [r["test_ood"]["acc"] for r in per_arm["quantum_quixer_lcu_qsvt"]],
            [r["test_ood"]["acc"] for r in per_arm["quantum_quixer_uniform_lcu"]],
        ),
    }
    _holm(paired, list(paired.keys()))

    ds0 = datasets[args.seeds[0]]
    report = {
        "benchmark": "Quantum Self-Attention Three-Regime Prototype Benchmark (KV Associative Recall)",
        "config": {
            "seeds": list(args.seeds),
            "steps": args.steps,
            "lr": args.lr,
            "batch_size": 32,
            "n_keys": ds0["n_keys"],
            "n_values": ds0["n_values"],
            "n_distractors": ds0["n_distractors"],
            "vocab_size": ds0["vocab_size"],
            "n_classes": ds0["n_classes"],
            "chance_accuracy": ds0["chance_accuracy"],
            "train_len": ds0["train_len"],
            "ood_len": ds0["ood_len"],
            "n_train": int(ds0["train"][0].shape[0]),
            "n_val": int(ds0["val"][0].shape[0]),
            "n_test_id": int(ds0["test_id"][0].shape[0]),
            "n_test_ood": int(ds0["test_ood"][0].shape[0]),
            "shot_budgets": list(SHOT_BUDGETS),
            "quick": args.quick,
        },
        "arms": agg,
        "paired_comparisons": paired,
        "per_seed": per_arm,
        "caveat": (
            "All training runs execute on exact CPU statevector/linear-algebra simulation. "
            "Regime II SWAP-Test attention is analytically identical to ClassicalOverlapAttention and "
            "requires O(T*S) (last-token query) or O(T^2*S) (full causal matrix) circuits under finite shots. "
            "Regime III Quixer LCU+QSVT eliminates classical Softmax and MLP crutches via block-encoding, "
            "but post-selection succeeds with probability p_succ < 1, requiring O(1/sqrt(p_succ)) amplitude "
            "amplification on fault-tolerant hardware (FTQC) and QRAM for scaling."
        ),
        "wallclock_seconds_total": time.perf_counter() - t0,
    }

    out_path = Path(args.out) if args.out else ((RESULTS if args.quick else DOCS) / "quantum_attention_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.no_figures and not args.quick:
        figs = plot_attention_figures(report, FIGURES)
        print(f"  saved figures: {figs}", flush=True)

    print(f"[DONE] wrote {out_path} in {report['wallclock_seconds_total']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
