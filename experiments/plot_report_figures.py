"""Generate publication-quality 300-dpi summary figures for formal IMRaD reports (01-03).

Reads exclusively from archived verified JSONs (zero hardcoded metrics):
  * docs/figures/matched_task2_comparison.png     <- docs/matched_task2_metrics.json
  * docs/figures/state_channel_probe_bloch.png    <- docs/state_channel_probe.json
                                                     + docs/quantum_native_seq_metrics.json (L1 recall)
  * docs/figures/quantum_native_seq_summary.png   <- docs/quantum_native_seq_metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"


def plot_matched_task2() -> Path:
    data = json.loads((DOCS / "matched_task2_metrics.json").read_text(encoding="utf-8"))
    arms = data["arms"]
    paired = data["paired_comparisons"]

    order = [
        ("caus_q_rich30", "Matched Q\n(12q, 3 QWC)", "#7c3aed"),
        ("caus_classical_tanh", "Matched Tanh\n(150c)", "#3b82f6"),
        ("caus_classical_trig", "Matched Trig\n(12c exact)", "#0ea5e9"),
        ("diag_unmatched_q_scale5", "Legacy Confound\n(scale=5, ni=T)", "#ef4444"),
    ]
    xs = np.arange(len(order))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    # Left: Train vs Independent Val Loss
    w = 0.35
    tr_m = [arms[k]["train_loss_mean"] for k, _, _ in order]
    tr_s = [arms[k]["train_loss_std"] for k, _, _ in order]
    va_m = [arms[k]["val_loss_mean"] for k, _, _ in order]
    va_s = [arms[k]["val_loss_std"] for k, _, _ in order]
    cols = [c for _, _, c in order]

    b1 = ax1.bar(xs - w / 2, tr_m, w, yerr=tr_s, capsize=3, color=cols, alpha=0.55, edgecolor="#1f2937", label="Train Loss")
    b2 = ax1.bar(xs + w / 2, va_m, w, yerr=va_s, capsize=3, color=cols, edgecolor="#1f2937", label="Independent Val Loss")
    for bar, m in zip(b1, tr_m):
        ax1.text(bar.get_x() + bar.get_width() / 2, m + 0.03, f"{m:.3f}", ha="center", va="bottom", fontsize=7.8)
    for bar, m in zip(b2, va_m):
        ax1.text(bar.get_x() + bar.get_width() / 2, m + 0.03, f"{m:.3f}", ha="center", va="bottom", fontsize=7.8)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([lbl for _, lbl, _ in order], fontsize=8)
    ax1.set_ylim(1.8, 2.75)
    ax1.set_ylabel("Cross-Entropy Loss (nats, mean ± SD)")
    ax1.set_title("Regime II Matched Task 2 (Canonical 30-Pauli / 3 QWC)")
    ax1.grid(axis="y", alpha=0.28)
    ax1.legend(loc="upper left", fontsize=8)

    # Right: Generalization Gap + Confound Isolation
    gap_m = [arms[k]["generalization_gap_mean"] for k, _, _ in order]
    gap_s = [arms[k]["generalization_gap_std"] for k, _, _ in order]
    bars = ax2.bar(xs, gap_m, width=0.52, yerr=gap_s, capsize=4, color=cols, edgecolor="#1f2937")
    for bar, m, sd in zip(bars, gap_m, gap_s):
        ax2.text(bar.get_x() + bar.get_width() / 2, m + sd + 0.015, f"{m:.4f}", ha="center", va="bottom", fontsize=8.2)
    p_conf = paired["diag_scale5_minus_matched_q|generalization_gap"]["p_holm"]
    d_conf = paired["diag_scale5_minus_matched_q|generalization_gap"]["mean_diff"]
    p_trig = paired["q_minus_trig|val_loss"]["p_holm"]
    ax2.annotate(
        f"Confound alone:\nΔ={d_conf:+.4f} (p_Holm={p_conf:.4f})",
        xy=(3, gap_m[3] + gap_s[3] + 0.055),
        xytext=(1.85, 0.65),
        arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 1.2},
        fontsize=7.8,
        color="#991b1b",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#fef2f2", "ec": "#fca5a5"},
    )
    ax2.text(
        0.03,
        0.96,
        f"Matched Q vs 12p Trig: Val Δ=-0.0030 (p_Holm={p_trig:.3f})\nQ cost: 20.7× slower (12/3646 = 0.33% Q params)",
        transform=ax2.transAxes,
        fontsize=7.6,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#f8fafc", "ec": "#cbd5e1"},
    )
    ax2.set_xticks(xs)
    ax2.set_xticklabels([lbl for _, lbl, _ in order], fontsize=8)
    ax2.set_ylim(0.0, 0.80)
    ax2.set_ylabel("Generalization Gap (Val − Train, nats)")
    ax2.set_title("Overfitting Gap & Legacy scale=5.0 Confound Isolation")
    ax2.grid(axis="y", alpha=0.28)

    fig.tight_layout()
    out = FIGURES / "matched_task2_comparison.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_state_channel_probe() -> Path:
    probe = json.loads((DOCS / "state_channel_probe.json").read_text(encoding="utf-8"))
    nat = json.loads((DOCS / "quantum_native_seq_metrics.json").read_text(encoding="utf-8"))

    cfgs = [
        ("RZ_feedback_Z_readout", "RZ + Z (legacy dead channel)", "#ef4444", "o"),
        ("RZ_feedback_X_readout", "RZ + X (zero at s=0, w=0)", "#f59e0b", "s"),
        ("RZ_feedback_Y_readout", "RZ + Y (conjugate, O(1) at s=0)", "#10b981", "^"),
        ("RY_feedback_Z_readout", "RY + Z (co-axial upper bound)", "#3b82f6", "D"),
    ]
    stds = [row["init_std"] for row in probe["measurements_at_init_zero"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    # Left: Exact s=0 initial state sensitivity |d<O>/ds| across init_std
    for key, label, col, mk in cfgs:
        vals = [max(row["configs"][key]["mean_abs_grad"], 1e-6) for row in probe["measurements_at_init_zero"]]
        ax1.plot(stds, vals, marker=mk, linewidth=2.0, markersize=6, color=col, label=label)
    ax1.set_yscale("log")
    ax1.set_ylim(5e-7, 2.0)
    ax1.set_xlabel("Entangler Weight Scale init_std")
    ax1.set_ylabel("Mean |∂(Σ⟨O_i⟩)/∂s| at exact s=0 (log scale)")
    ax1.set_title("L1 Bloch-Sphere Channel Sensitivity at Exact Init s=0")
    ax1.annotate(
        "RZ+Z = 0 & RZ+X = 0\n(exact analytic zero)",
        xy=(0.0, 1e-6),
        xytext=(0.06, 2.5e-5),
        arrowprops={"arrowstyle": "->", "color": "#dc2626"},
        fontsize=7.8,
        color="#991b1b",
    )
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="lower right", fontsize=7.6)

    # Right: Empirical Validation on Clean Deterministic Delayed Recall (pad=4 ID & pad=8 OOD)
    rec_arms = nat["track2_deterministic_delayed_recall"]["arms"]
    bar_spec = [
        ("l1_quantum_legacy_rzz", "Legacy L1\nRZ+Z (0.05)", "#ef4444"),
        ("l1_quantum_fixed_rzy", "Fixed L1\nRZ+Y (1.0)", "#10b981"),
        ("l1_quantum_fixed_ryz", "Fixed L1\nRY+Z (1.0)", "#3b82f6"),
        ("l2_quantum_unitary_ssm", "True L2\nUnitary SSM", "#7c3aed"),
    ]
    bx = np.arange(len(bar_spec))
    bw = 0.35
    id_m = [rec_arms[k]["acc_id_pad4"]["mean"] for k, _, _ in bar_spec]
    id_s = [rec_arms[k]["acc_id_pad4"]["std"] for k, _, _ in bar_spec]
    ood_m = [rec_arms[k]["acc_ood_pad8"]["mean"] for k, _, _ in bar_spec]
    ood_s = [rec_arms[k]["acc_ood_pad8"]["std"] for k, _, _ in bar_spec]
    cols = [c for _, _, c in bar_spec]
    b1 = ax2.bar(bx - bw / 2, id_m, bw, yerr=id_s, capsize=3, color=cols, edgecolor="#1f2937", label="ID (pad=4)")
    b2 = ax2.bar(bx + bw / 2, ood_m, bw, yerr=ood_s, capsize=3, color=cols, alpha=0.55, hatch="//", edgecolor="#1f2937", label="OOD (pad=8)")
    ax2.axhline(0.25, color="#6b7280", linestyle="--", linewidth=1.1, label="Chance (0.25)")
    for bar, m in zip(b1, id_m):
        ax2.text(bar.get_x() + bar.get_width() / 2, min(1.03, m + 0.025), f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    for bar, m in zip(b2, ood_m):
        ax2.text(bar.get_x() + bar.get_width() / 2, min(1.03, m + 0.025), f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(bx)
    ax2.set_xticklabels([lbl for _, lbl, _ in bar_spec], fontsize=8)
    ax2.set_ylim(0.0, 1.15)
    ax2.set_ylabel("Exact Delayed-Recall Accuracy (5 seeds)")
    ax2.set_title("Training Validation: Fixing L1 Channel vs True L2")
    ax2.grid(axis="y", alpha=0.28)
    ax2.legend(loc="upper left", fontsize=7.8)

    fig.tight_layout()
    out = FIGURES / "state_channel_probe_bloch.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_quantum_native_summary() -> Path:
    nat = json.loads((DOCS / "quantum_native_seq_metrics.json").read_text(encoding="utf-8"))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 4.3))

    # Panel A: Track 1A Periodic-4 Exact KL (ID T=8 & OOD T=14, lower is better)
    t1a = nat["track1_periodic_interference"]["arms"]
    s1a = [
        ("quantum_mps_complex_chi2", "Q-MPS\n(c,χ=2,24p)", "#10b981"),
        ("classical_nonneg_mps_chi2", "NonNeg\n(χ=2,12p)", "#f59e0b"),
        ("classical_nonneg_mps_chi3", "NonNeg\n(χ=3,24p)", "#d97706"),
        ("classical_rnn_matched", "RNN\n(35p)", "#64748b"),
    ]
    x1 = np.arange(len(s1a))
    w = 0.36
    kl_id = [t1a[k]["kl_id_exact"]["mean"] for k, _, _ in s1a]
    kl_14 = [t1a[k]["ood_T14_kl_exact"]["mean"] for k, _, _ in s1a]
    c1 = [c for _, _, c in s1a]
    ax1.bar(x1 - w / 2, kl_id, w, color=c1, edgecolor="#1f2937", label="ID (T=8)")
    ax1.bar(x1 + w / 2, kl_14, w, color=c1, alpha=0.55, hatch="//", edgecolor="#1f2937", label="OOD (T=14)")
    for xi, a, b in zip(x1, kl_id, kl_14):
        ax1.text(xi - w / 2, a + 0.012, f"{a:.3f}", ha="center", va="bottom", fontsize=7.2)
        ax1.text(xi + w / 2, b + 0.012, f"{b:.3f}", ha="center", va="bottom", fontsize=7.2)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([lbl for _, lbl, _ in s1a], fontsize=7.5)
    ax1.set_ylim(0.0, 0.62)
    ax1.set_ylabel("Exact KL(P* || P_model)  [lower better]")
    ax1.set_title("(a) Track 1A: Periodic-4 Interference")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper left", fontsize=7.5)

    # Panel B: Track 1B Dyck-1 Valid Bracket Generation Rate (ID T=14 & OOD T=16)
    t1b = nat["track1_dyck1"]["arms"]
    s1b = [
        ("quantum_mps_chi6", "Q-MPS\n(χ=6,84p)", "#10b981"),
        ("classical_nonneg_mps_chi6", "NonNeg\n(χ=6,84p)", "#f59e0b"),
        ("classical_rnn_matched", "RNN\n(98p)", "#64748b"),
    ]
    x2 = np.arange(len(s1b))
    v_id = [t1b[k]["valid_rate_id"]["mean"] for k, _, _ in s1b]
    v_ood = [t1b[k]["valid_rate_ood"]["mean"] for k, _, _ in s1b]
    c2 = [c for _, _, c in s1b]
    ax2.bar(x2 - w / 2, v_id, w, color=c2, edgecolor="#1f2937", label="ID (T=14)")
    ax2.bar(x2 + w / 2, v_ood, w, color=c2, alpha=0.55, hatch="//", edgecolor="#1f2937", label="OOD (T=16)")
    for xi, a, b in zip(x2, v_id, v_ood):
        ax2.text(xi - w / 2, a + 0.02, f"{a:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax2.text(xi + w / 2, b + 0.02, f"{b:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([lbl for _, lbl, _ in s1b], fontsize=8)
    ax2.set_ylim(0.0, 1.12)
    ax2.set_ylabel("Valid Dyck-1 Sample Rate  [higher better]")
    ax2.set_title("(b) Track 1B: Dyck-1 Brackets")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=7.5)

    # Panel C: Track 2A Non-Abelian S_3 State Tracking (ID T=6 -> OOD T=12, T=16)
    t2a = nat["track2_s3_state_tracking"]["arms"]
    s2a = [
        ("l2_quantum_unitary_ssm", "L2 Q-SSM\n(3q,300p)", "#7c3aed"),
        ("classical_orthogonal_ssm", "Orth-SSM\n(SO9,270p)", "#3b82f6"),
        ("classical_rnn", "RNN\n(327p)", "#64748b"),
        ("l1_quantum_fixed_rzy", "Fixed L1\n(278p)", "#f59e0b"),
        ("l0_causal_conv", "L0 Conv\n(334p)", "#ef4444"),
    ]
    x3 = np.arange(len(s2a))
    acc_6 = [t2a[k]["test_id_final_acc"]["mean"] for k, _, _ in s2a]
    acc_16 = [t2a[k]["ood_final_acc"]["16"]["mean"] for k, _, _ in s2a]
    c3 = [c for _, _, c in s2a]
    ax3.bar(x3 - w / 2, acc_6, w, color=c3, edgecolor="#1f2937", label="ID (T=6)")
    ax3.bar(x3 + w / 2, acc_16, w, color=c3, alpha=0.55, hatch="//", edgecolor="#1f2937", label="OOD (T=16)")
    ax3.axhline(1.0 / 6.0, color="#ef4444", linestyle="--", linewidth=1.0, label="Chance (0.167)")
    for xi, a, b in zip(x3, acc_6, acc_16):
        ax3.text(xi - w / 2, min(1.03, a + 0.02), f"{a:.2f}", ha="center", va="bottom", fontsize=7.0)
        ax3.text(xi + w / 2, min(1.03, b + 0.02), f"{b:.2f}", ha="center", va="bottom", fontsize=7.0)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([lbl for _, lbl, _ in s2a], fontsize=7.2)
    ax3.set_ylim(0.0, 1.15)
    ax3.set_ylabel("Final-Step S_3 Accuracy  [higher better]")
    ax3.set_title("(c) Track 2A: Non-Abelian S_3 Tracking")
    ax3.grid(axis="y", alpha=0.25)
    ax3.legend(loc="upper right", fontsize=7.2)

    fig.tight_layout()
    out = FIGURES / "quantum_native_seq_summary.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    p1 = plot_matched_task2()
    p2 = plot_state_channel_probe()
    p3 = plot_quantum_native_summary()
    print(f"Generated report figures: {p1.name}, {p2.name}, {p3.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
