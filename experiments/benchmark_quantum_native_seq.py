"""Regime III Quantum-Native Sequence Modeling Benchmark (Track 1 + Track 2).

Track 1 — Autoregressive Tensor-Network Born Machine (``qml_lab/born_machine.py``):
  * Task 1A: Periodic-4 Quantum Interference Language P*(x) ∝ cos^2(pi/4 * sum x_t)
    (Train T=8, Val T=8, Test-ID T=8, OOD T=12, 14). Tests Perron-Frobenius separation
    between signed ``QuantumMPSBornMachine(chi=2)`` (12 params, destructive interference)
    and its exact single-variable control ``ClassicalNonNegMPS(chi=2)`` (12 params, no
    destructive interference) + ``ClassicalCausalRNNLM``.
  * Task 1B: Dyck-1 Balanced Brackets (strict disjoint Train/Val/Test at T=14 + OOD T=16)
    comparing ``QuantumMPSBornMachine(chi=6)`` (84 params) vs ``ClassicalNonNegMPS(chi=6)``
    (84 params) vs ``ClassicalCausalRNNLM`` (98 params).

Track 2 — True L2 Persistent Quantum-State Recurrence / Unitary SSM (``qml_lab/quantum_ssm.py``):
  * Task 2A: Non-Abelian S_3 Group Word Problem (strict disjoint Train/Val/Test-ID at T=6,
    OOD length generalization at T=12 and T=16). Compares True L2 ``QuantumUnitarySSM``
    (n_H=3 qubits, 100% quantum unitary + POVM, 0 MLP params, PennyLane verified!) against
    matched ``ClassicalOrthogonalSSM``, ``ClassicalRNNTracker``, L0 ``CausalConvTracker``,
    and Fixed L1 ``QuantumRecurrentLM(rz+y, init_std=1.0)``.
  * Task 2B: Clean Deterministic Delayed Recall (all K=4 exact sequences evaluated with zero
    sampling variance, pad=4 ID + pad=8 OOD), directly comparing Legacy L1 ``RZ+Z (0.05)`` vs
    Fixed L1 ``RZ+Y (1.0)`` & ``RY+Z (1.0)`` vs True L2 ``QuantumUnitarySSM`` vs classical controls.

Usage:
    python experiments/benchmark_quantum_native_seq.py --quick   # smoke test -> results/
    python experiments/benchmark_quantum_native_seq.py           # 5 seeds -> docs/
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

import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F

from qml_lab.born_machine import (
    ClassicalCausalRNNLM,
    ClassicalNonNegMPS,
    QuantumMPSBornMachine,
    generate_dyck1_dataset,
    generate_periodic_interference_dataset,
    is_valid_dyck1,
)
from qml_lab.quantum_ssm import (
    CausalConvTracker,
    ClassicalOrthogonalSSM,
    ClassicalRNNTracker,
    QuantumUnitarySSM,
    build_deterministic_delayed_recall,
    build_s3_dataset,
)
from qml_lab.recurrent import QuantumRecurrentLM
from qml_lab.utils import count_parameters, set_seed

DOCS = ROOT / "docs"
RESULTS = ROOT / "results"


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
    for i, k in enumerate(order):
        adj = min(1.0, (m - i) * tests[k]["p_value"])
        running = max(running, adj)
        tests[k]["p_holm"] = float(running)
        tests[k]["holm_significant"] = bool(running < 0.05)


def _mean_std(vals: list[float]) -> dict[str, float]:
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Track 1: Autoregressive Tensor-Network Born Machine
# --------------------------------------------------------------------------- #
def _exact_kl(model: torch.nn.Module, all_x: torch.Tensor, target_probs: torch.Tensor) -> float:
    """Exact KL(P* || P_model) over the full discrete support {0,1}^T (chunked for speed)."""
    model.eval()
    nz_mask = target_probs > 1e-12
    x_nz = all_x[nz_mask]
    p_nz = target_probs[nz_mask].to(torch.float64)
    log_q_chunks = []
    with torch.no_grad():
        for i in range(0, x_nz.shape[0], 512):
            log_q_chunks.append(model.log_prob(x_nz[i : i + 512]).to(torch.float64))
    log_q = torch.cat(log_q_chunks, dim=0)
    kl = float((p_nz * (torch.log(p_nz) - log_q)).sum().item())
    return max(0.0, kl)


def _train_lm_gen(
    model: torch.nn.Module,
    x_tr: torch.Tensor | dict[int, torch.Tensor],
    x_va: torch.Tensor | dict[int, torch.Tensor],
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(10_000 + seed)
    best_val = float("inf")
    best_state = None
    t0 = time.perf_counter()
    tr_pools = x_tr if isinstance(x_tr, dict) else {int(x_tr.shape[1]): x_tr}
    va_pools = x_va if isinstance(x_va, dict) else {int(x_va.shape[1]): x_va}
    tr_lens = sorted(tr_pools.keys())
    eval_every = max(1, steps // 10)
    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = []
        for L in tr_lens:
            pool = tr_pools[L]
            idx = torch.randint(0, pool.shape[0], (min(batch_size, pool.shape[0]),), generator=g)
            losses.append(model.nll_loss(pool[idx]))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vloss = float(np.mean([float(model.nll_loss(v).item()) for v in va_pools.values()]))
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_nll": best_val, "seconds": time.perf_counter() - t0}


def _train_periodic_arm(task: tuple[str, dict, int, tuple[int, ...], int]) -> tuple[str, dict[str, Any]]:
    arm, ds, steps, ood_lens, seed = task
    torch.set_num_threads(1)
    set_seed(seed)
    if arm == "quantum_mps_complex_chi2":
        model: torch.nn.Module = QuantumMPSBornMachine(vocab_size=2, bond_dim=2, complex_amplitudes=True)
    elif arm == "classical_nonneg_mps_chi2":
        model = ClassicalNonNegMPS(vocab_size=2, bond_dim=2)
    elif arm == "classical_nonneg_mps_chi3":
        model = ClassicalNonNegMPS(vocab_size=2, bond_dim=3)
    elif arm == "classical_rnn_matched":
        model = ClassicalCausalRNNLM(vocab_size=2, embed_dim=2, hidden_dim=3)
    else:
        raise ValueError(arm)

    tr_info = _train_lm_gen(
        model, ds["train_by_len"], ds["val_by_len"], steps=steps, batch_size=64, lr=0.03, seed=seed
    )
    model.eval()
    with torch.no_grad():
        test_id_nll = float(model.nll_loss(ds["test_id"]).item())
        kl_id = _exact_kl(model, ds["id_all_x"], ds["id_probs"])
        gen = torch.Generator().manual_seed(500 + seed)
        samp_id = model.sample(400, ds["train_len"], generator=gen)
        forb_id = float(((samp_id.sum(dim=-1) % 4) == 2).float().mean().item())
        ood_res = {}
        for L in ood_lens:
            od = ds["ood"][L]
            nll_l = float(model.nll_loss(od["x"]).item())
            kl_l = _exact_kl(model, od["all_x"], od["probs"]) if L <= 14 else None
            samp_l = model.sample(400, L, generator=gen)
            forb_l = float(((samp_l.sum(dim=-1) % 4) == 2).float().mean().item())
            ood_res[str(L)] = {"nll": nll_l, "kl_exact": kl_l, "forbidden_rate": forb_l}
    return arm, {
        "seed": seed,
        "n_params": count_parameters(model),
        "best_val_nll": tr_info["best_val_nll"],
        "test_id_nll": test_id_nll,
        "kl_id_exact": kl_id,
        "forbidden_rate_id": forb_id,
        "ood": ood_res,
        "seconds": tr_info["seconds"],
    }


def run_track1_periodic(seeds: list[int], steps: int, quick: bool, workers: int) -> dict[str, Any]:
    print("\n=== Track 1A: Periodic-4 Quantum Interference (train T in {4,6,8} -> OOD T=12,14) ===", flush=True)
    arms = (
        "quantum_mps_complex_chi2",
        "classical_nonneg_mps_chi2",
        "classical_nonneg_mps_chi3",
        "classical_rnn_matched",
    )
    train_lens = (4, 6, 8)
    ood_lens = (10,) if quick else (12, 14)
    datasets = {
        seed: generate_periodic_interference_dataset(
            modulus=4,
            train_len=8,
            train_lens=train_lens,
            ood_lens=ood_lens,
            n_train=450 if quick else 1800,
            n_val=150 if quick else 450,
            n_test=200 if quick else 600,
            seed=100 + seed,
        )
        for seed in seeds
    }
    tasks = [(arm, datasets[seed], steps, ood_lens, seed) for arm in arms for seed in seeds]
    per_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for arm, r in ex.map(_train_periodic_arm, tasks):
                per_arm[arm].append(r)
    else:
        for t in tasks:
            arm, r = _train_periodic_arm(t)
            per_arm[arm].append(r)
    for arm in arms:
        per_arm[arm].sort(key=lambda r: r["seed"])

    agg = {}
    max_ood = str(max(ood_lens))
    for arm in arms:
        runs = per_arm[arm]
        agg[arm] = {
            "n_params": runs[0]["n_params"],
            "test_id_nll": _mean_std([r["test_id_nll"] for r in runs]),
            "kl_id_exact": _mean_std([r["kl_id_exact"] for r in runs]),
            "forbidden_rate_id": _mean_std([r["forbidden_rate_id"] for r in runs]),
            f"ood_T{max_ood}_nll": _mean_std([r["ood"][max_ood]["nll"] for r in runs]),
            f"ood_T{max_ood}_kl_exact": _mean_std([r["ood"][max_ood]["kl_exact"] for r in runs]),
            f"ood_T{max_ood}_forbidden_rate": _mean_std([r["ood"][max_ood]["forbidden_rate"] for r in runs]),
            "seconds_mean": float(np.mean([r["seconds"] for r in runs])),
        }
        print(
            f"  {arm:26s} (p={agg[arm]['n_params']:2d}) | KL_ID={agg[arm]['kl_id_exact']['mean']:.4f}±{agg[arm]['kl_id_exact']['std']:.4f} "
            f"forb_ID={agg[arm]['forbidden_rate_id']['mean']:.3f} | "
            f"KL_T{max_ood}={agg[arm][f'ood_T{max_ood}_kl_exact']['mean']:.4f}±{agg[arm][f'ood_T{max_ood}_kl_exact']['std']:.4f} "
            f"forb_T{max_ood}={agg[arm][f'ood_T{max_ood}_forbidden_rate']['mean']:.3f}",
            flush=True,
        )

    paired = {
        "q_mps_minus_nonneg_chi2|kl_id": _paired_ci(
            [r["kl_id_exact"] for r in per_arm["quantum_mps_complex_chi2"]],
            [r["kl_id_exact"] for r in per_arm["classical_nonneg_mps_chi2"]],
        ),
        "q_mps_minus_nonneg_chi3|kl_id": _paired_ci(
            [r["kl_id_exact"] for r in per_arm["quantum_mps_complex_chi2"]],
            [r["kl_id_exact"] for r in per_arm["classical_nonneg_mps_chi3"]],
        ),
        f"q_mps_minus_nonneg_chi3|kl_T{max_ood}": _paired_ci(
            [r["ood"][max_ood]["kl_exact"] for r in per_arm["quantum_mps_complex_chi2"]],
            [r["ood"][max_ood]["kl_exact"] for r in per_arm["classical_nonneg_mps_chi3"]],
        ),
        f"q_mps_minus_rnn|kl_T{max_ood}": _paired_ci(
            [r["ood"][max_ood]["kl_exact"] for r in per_arm["quantum_mps_complex_chi2"]],
            [r["ood"][max_ood]["kl_exact"] for r in per_arm["classical_rnn_matched"]],
        ),
    }
    _holm(paired, list(paired.keys()))
    ds_ref = generate_periodic_interference_dataset(modulus=4, train_len=8, train_lens=train_lens, ood_lens=ood_lens, n_train=10, n_val=10, n_test=10, seed=0)
    return {
        "task": "Periodic-4 Quantum Interference P*(x) ∝ cos^2(pi/4 * sum x_t)",
        "train_len": 8,
        "train_lens": list(train_lens),
        "ood_lens": list(ood_lens),
        "bayes_nll_id": ds_ref["bayes_nll_id"],
        "bayes_nll_ood": {str(L): ds_ref["ood"][L]["bayes_nll_per_token"] for L in ood_lens},
        "arms": agg,
        "paired_comparisons": paired,
        "per_seed": per_arm,
    }


def _train_dyck1_arm(task: tuple[str, dict, int, int, int, int]) -> tuple[str, dict[str, Any]]:
    arm, ds, steps, train_len, ood_len, seed = task
    torch.set_num_threads(1)
    set_seed(seed)
    if arm == "quantum_mps_chi6":
        model: torch.nn.Module = QuantumMPSBornMachine(vocab_size=2, bond_dim=6, complex_amplitudes=False)
    elif arm == "classical_nonneg_mps_chi6":
        model = ClassicalNonNegMPS(vocab_size=2, bond_dim=6)
    elif arm == "classical_rnn_matched":
        model = ClassicalCausalRNNLM(vocab_size=2, embed_dim=4, hidden_dim=6)
    else:
        raise ValueError(arm)
    tr_info = _train_lm_gen(
        model, ds["train"], ds["val"], steps=steps, batch_size=64, lr=0.03, seed=seed
    )
    model.eval()
    with torch.no_grad():
        nll_id = float(model.nll_loss(ds["test_id"]).item())
        nll_ood = float(model.nll_loss(ds["test_ood"]).item())
        gen = torch.Generator().manual_seed(700 + seed)
        s_id = model.sample(200, train_len, generator=gen).tolist()
        s_ood = model.sample(200, ood_len, generator=gen).tolist()
        valid_id = float(np.mean([is_valid_dyck1(seq) for seq in s_id]))
        valid_ood = float(np.mean([is_valid_dyck1(seq) for seq in s_ood]))
    return arm, {
        "seed": seed,
        "n_params": count_parameters(model),
        "best_val_nll": tr_info["best_val_nll"],
        "test_id_nll": nll_id,
        "test_ood_nll": nll_ood,
        "valid_rate_id": valid_id,
        "valid_rate_ood": valid_ood,
        "seconds": tr_info["seconds"],
    }


def run_track1_dyck1(seeds: list[int], steps: int, quick: bool, workers: int) -> dict[str, Any]:
    print("\n=== Track 1B: Dyck-1 Balanced Brackets (T=14 -> OOD T=16, chi=6) ===", flush=True)
    arms = ("quantum_mps_chi6", "classical_nonneg_mps_chi6", "classical_rnn_matched")
    train_len, ood_len = (10, 12) if quick else (14, 16)
    datasets = {seed: generate_dyck1_dataset(train_len=train_len, ood_len=ood_len, seed=200 + seed) for seed in seeds}
    tasks = [(arm, datasets[seed], steps, train_len, ood_len, seed) for arm in arms for seed in seeds]
    per_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for arm, r in ex.map(_train_dyck1_arm, tasks):
                per_arm[arm].append(r)
    else:
        for t in tasks:
            arm, r = _train_dyck1_arm(t)
            per_arm[arm].append(r)
    for arm in arms:
        per_arm[arm].sort(key=lambda r: r["seed"])

    agg = {}
    for arm in arms:
        runs = per_arm[arm]
        agg[arm] = {
            "n_params": runs[0]["n_params"],
            "test_id_nll": _mean_std([r["test_id_nll"] for r in runs]),
            "test_ood_nll": _mean_std([r["test_ood_nll"] for r in runs]),
            "valid_rate_id": _mean_std([r["valid_rate_id"] for r in runs]),
            "valid_rate_ood": _mean_std([r["valid_rate_ood"] for r in runs]),
            "seconds_mean": float(np.mean([r["seconds"] for r in runs])),
        }
        print(
            f"  {arm:26s} (p={agg[arm]['n_params']:2d}) | NLL_ID={agg[arm]['test_id_nll']['mean']:.4f} "
            f"valid_ID={agg[arm]['valid_rate_id']['mean']:.3f} | "
            f"NLL_OOD={agg[arm]['test_ood_nll']['mean']:.4f} valid_OOD={agg[arm]['valid_rate_ood']['mean']:.3f}",
            flush=True,
        )

    paired = {
        "q_mps_minus_nonneg_mps|test_id_nll": _paired_ci(
            [r["test_id_nll"] for r in per_arm["quantum_mps_chi6"]],
            [r["test_id_nll"] for r in per_arm["classical_nonneg_mps_chi6"]],
        ),
        "q_mps_minus_nonneg_mps|valid_rate_ood": _paired_ci(
            [r["valid_rate_ood"] for r in per_arm["quantum_mps_chi6"]],
            [r["valid_rate_ood"] for r in per_arm["classical_nonneg_mps_chi6"]],
        ),
        "q_mps_minus_rnn|test_ood_nll": _paired_ci(
            [r["test_ood_nll"] for r in per_arm["quantum_mps_chi6"]],
            [r["test_ood_nll"] for r in per_arm["classical_rnn_matched"]],
        ),
        "q_mps_minus_rnn|valid_rate_ood": _paired_ci(
            [r["valid_rate_ood"] for r in per_arm["quantum_mps_chi6"]],
            [r["valid_rate_ood"] for r in per_arm["classical_rnn_matched"]],
        ),
    }
    _holm(paired, list(paired.keys()))
    ds_ref = generate_dyck1_dataset(train_len=train_len, ood_len=ood_len, seed=0)
    return {
        "task": f"Dyck-1 balanced brackets (strict disjoint T={train_len} -> OOD T={ood_len})",
        "train_len": train_len,
        "ood_len": ood_len,
        "bayes_nll_id": ds_ref["bayes_nll_id"],
        "bayes_nll_ood": ds_ref["bayes_nll_ood"],
        "arms": agg,
        "paired_comparisons": paired,
        "per_seed": per_arm,
    }


# --------------------------------------------------------------------------- #
# Track 2: True L2 QuantumUnitarySSM vs Matched Classical + Fixed L1
# --------------------------------------------------------------------------- #
def _eval_tracker(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, is_log_probs: bool) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        out = model(x)  # (B, T, C)
        logp = out if is_log_probs else F.log_softmax(out, dim=-1)
        C = logp.shape[-1]
        loss = float(F.nll_loss(logp.reshape(-1, C), y.reshape(-1)).item())
        all_acc = float((logp.argmax(dim=-1) == y).float().mean().item())
        last_acc = float((logp[:, -1, :].argmax(dim=-1) == y[:, -1]).float().mean().item())
    return {"loss": loss, "all_pos_acc": all_acc, "final_pos_acc": last_acc}


def _train_s3_arm(task: tuple[str, dict, int, int]) -> tuple[str, dict[str, Any]]:
    arm, ds, steps, seed = task
    torch.set_num_threads(1)
    set_seed(seed)
    V, C = ds["vocab_size"], ds["n_classes"]
    is_log_probs = True
    lr = 0.03  # Unified LR across all 5 arms (no per-arm LR confound)
    if arm == "l2_quantum_unitary_ssm":
        model: torch.nn.Module = QuantumUnitarySSM(vocab_size=V, n_classes=C, n_qubits=3, povm_rank=2)
    elif arm == "classical_orthogonal_ssm":
        model = ClassicalOrthogonalSSM(vocab_size=V, n_classes=C, state_dim=9, readout_rank=3)
    elif arm == "classical_rnn":
        model = ClassicalRNNTracker(vocab_size=V, n_classes=C, embed_dim=8, hidden_dim=11)
    elif arm == "l0_causal_conv":
        model = CausalConvTracker(vocab_size=V, n_classes=C, embed_dim=6, hidden_dim=10, context_window=4)
    elif arm == "l1_quantum_fixed_rzy":
        model = QuantumRecurrentLM(
            vocab_size=C, n_qubits=4, n_layers=1, embed_dim=8, hidden=16,
            state_feedback="rz", readout_basis="y", init_std=1.0,
        )
        is_log_probs = False
    else:
        raise ValueError(arm)

    x_tr, y_tr = ds["train"]
    x_va, y_va = ds["val"]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(20_000 + seed)
    best_val = float("inf")
    best_state = None
    t0 = time.perf_counter()
    eval_every = max(1, steps // 10)
    batch = 16

    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, x_tr.shape[0], (min(batch, x_tr.shape[0]),), generator=g)
        opt.zero_grad(set_to_none=True)
        out = model(x_tr[idx])
        logp = out if is_log_probs else F.log_softmax(out, dim=-1)
        loss = F.nll_loss(logp.reshape(-1, C), y_tr[idx].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            ev = _eval_tracker(model, x_va, y_va, is_log_probs)
            if ev["loss"] < best_val:
                best_val = ev["loss"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    wall = time.perf_counter() - t0

    id_ev = _eval_tracker(model, *ds["test_id"], is_log_probs)
    ood_ev = {str(L): _eval_tracker(model, *pair, is_log_probs) for L, pair in ds["ood"].items()}
    pl_check = None
    if arm == "l2_quantum_unitary_ssm" and seed == 0:
        pl_check = model.verify_against_pennylane(ds["test_id"][0][0])

    return arm, {
        "seed": seed,
        "n_params": count_parameters(model),
        "test_id": id_ev,
        "ood": ood_ev,
        "pennylane_verification": pl_check,
        "seconds": wall,
    }


def run_track2_s3(seeds: list[int], steps: int, quick: bool, workers: int) -> dict[str, Any]:
    print("\n=== Track 2A: Non-Abelian S_3 Group State-Tracking (T=6 -> OOD T=12, 16) ===", flush=True)
    arms = (
        "l2_quantum_unitary_ssm",
        "classical_orthogonal_ssm",
        "classical_rnn",
        "l0_causal_conv",
        "l1_quantum_fixed_rzy",
    )
    ood_lens = (10,) if quick else (12, 16)
    datasets = {
        seed: build_s3_dataset(
            train_len=6,
            ood_lens=ood_lens,
            n_train=180 if quick else 360,
            n_val=60 if quick else 120,
            n_test_id=60 if quick else 160,
            n_test_ood=60 if quick else 200,
            seed=300 + seed,
        )
        for seed in seeds
    }
    # Schedule slower PennyLane L1 arm first for parallel load balance
    ordered_arms = ("l1_quantum_fixed_rzy", "l2_quantum_unitary_ssm", "classical_orthogonal_ssm", "classical_rnn", "l0_causal_conv")
    tasks = [(arm, datasets[seed], steps, seed) for arm in ordered_arms for seed in seeds]
    per_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for arm, r in ex.map(_train_s3_arm, tasks):
                per_arm[arm].append(r)
    else:
        for t in tasks:
            arm, r = _train_s3_arm(t)
            per_arm[arm].append(r)
    for arm in arms:
        per_arm[arm].sort(key=lambda r: r["seed"])

    max_ood = str(max(ood_lens))
    agg: dict[str, Any] = {}
    for arm in arms:
        runs = per_arm[arm]
        agg[arm] = {
            "n_params": runs[0]["n_params"],
            "test_id_final_acc": _mean_std([r["test_id"]["final_pos_acc"] for r in runs]),
            "test_id_all_acc": _mean_std([r["test_id"]["all_pos_acc"] for r in runs]),
            "ood_final_acc": {
                str(L): _mean_std([r["ood"][str(L)]["final_pos_acc"] for r in runs]) for L in ood_lens
            },
            "seconds_mean": float(np.mean([r["seconds"] for r in runs])),
        }
        if runs[0]["pennylane_verification"] is not None:
            agg[arm]["pennylane_verification"] = runs[0]["pennylane_verification"]
        ood_str = " ".join(
            f"T{L}={agg[arm]['ood_final_acc'][str(L)]['mean']:.3f}±{agg[arm]['ood_final_acc'][str(L)]['std']:.3f}"
            for L in ood_lens
        )
        print(
            f"  {arm:26s} (p={agg[arm]['n_params']:3d}) | "
            f"ID(T6)={agg[arm]['test_id_final_acc']['mean']:.3f}±{agg[arm]['test_id_final_acc']['std']:.3f} | "
            f"{ood_str} | {agg[arm]['seconds_mean']:.1f}s",
            flush=True,
        )

    paired = {
        f"l2_qssm_minus_orth_ssm|ood_T{max_ood}": _paired_ci(
            [r["ood"][max_ood]["final_pos_acc"] for r in per_arm["l2_quantum_unitary_ssm"]],
            [r["ood"][max_ood]["final_pos_acc"] for r in per_arm["classical_orthogonal_ssm"]],
        ),
        f"l2_qssm_minus_rnn|ood_T{max_ood}": _paired_ci(
            [r["ood"][max_ood]["final_pos_acc"] for r in per_arm["l2_quantum_unitary_ssm"]],
            [r["ood"][max_ood]["final_pos_acc"] for r in per_arm["classical_rnn"]],
        ),
        f"l2_qssm_minus_l0_conv|ood_T{max_ood}": _paired_ci(
            [r["ood"][max_ood]["final_pos_acc"] for r in per_arm["l2_quantum_unitary_ssm"]],
            [r["ood"][max_ood]["final_pos_acc"] for r in per_arm["l0_causal_conv"]],
        ),
        f"l2_qssm_minus_l1_fixed|ood_T{max_ood}": _paired_ci(
            [r["ood"][max_ood]["final_pos_acc"] for r in per_arm["l2_quantum_unitary_ssm"]],
            [r["ood"][max_ood]["final_pos_acc"] for r in per_arm["l1_quantum_fixed_rzy"]],
        ),
    }
    _holm(paired, list(paired.keys()))
    return {
        "task": "Non-Abelian S_3 word problem (6 classes, random baseline 1/6=0.1667)",
        "train_len": 6,
        "ood_lens": list(ood_lens),
        "arms": agg,
        "paired_comparisons": paired,
        "per_seed": per_arm,
    }


def _train_recall_arm(task: tuple[str, dict, int, int]) -> tuple[str, dict[str, Any]]:
    arm, ds, steps, seed = task
    torch.set_num_threads(1)
    set_seed(seed)
    V, C = ds["vocab_size"], ds["n_classes"]
    x_id, y_id = ds["id"]
    x_ood, y_ood = ds["ood"]
    train_batches = ds["train_batches"]
    is_log_probs = True
    lr = 0.03  # Unified LR across all 7 arms

    if arm == "l2_quantum_unitary_ssm":
        model: torch.nn.Module = QuantumUnitarySSM(vocab_size=V, n_classes=C, n_qubits=3, povm_rank=2)
    elif arm == "classical_orthogonal_ssm":
        model = ClassicalOrthogonalSSM(vocab_size=V, n_classes=C, state_dim=9, readout_rank=3)
    elif arm == "classical_rnn":
        model = ClassicalRNNTracker(vocab_size=V, n_classes=C, embed_dim=8, hidden_dim=8)
    elif arm == "l0_causal_conv":
        model = CausalConvTracker(vocab_size=V, n_classes=C, embed_dim=8, hidden_dim=8, context_window=4)
    elif arm == "l1_quantum_legacy_rzz":
        model = QuantumRecurrentLM(
            vocab_size=V, n_qubits=4, n_layers=1, embed_dim=8, hidden=16,
            state_feedback="rz", readout_basis="z", init_std=0.05,
        )
        is_log_probs = False
    elif arm == "l1_quantum_fixed_rzy":
        model = QuantumRecurrentLM(
            vocab_size=V, n_qubits=4, n_layers=1, embed_dim=8, hidden=16,
            state_feedback="rz", readout_basis="y", init_std=1.0,
        )
        is_log_probs = False
    elif arm == "l1_quantum_fixed_ryz":
        model = QuantumRecurrentLM(
            vocab_size=V, n_qubits=4, n_layers=1, embed_dim=8, hidden=16,
            state_feedback="ry", readout_basis="z", init_std=1.0,
        )
        is_log_probs = False
    else:
        raise ValueError(arm)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.perf_counter()
    for step in range(steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        # Cycle deterministically over pad_len in (2, 3, 4) so models must learn delay-invariant memory
        xb, yb = train_batches[step % len(train_batches)]
        out = model(xb)[:, -1, :C]
        logp = out if is_log_probs else F.log_softmax(out, dim=-1)
        loss = F.nll_loss(logp, yb[:, -1])
        loss.backward()
        opt.step()
    wall = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        o_id = model(x_id)[:, -1, :C]
        lp_id = o_id if is_log_probs else F.log_softmax(o_id, dim=-1)
        acc_id = float((lp_id.argmax(-1) == y_id[:, -1]).float().mean().item())
        loss_id = float(F.nll_loss(lp_id, y_id[:, -1]).item())

        o_ood = model(x_ood)[:, -1, :C]
        lp_ood = o_ood if is_log_probs else F.log_softmax(o_ood, dim=-1)
        acc_ood = float((lp_ood.argmax(-1) == y_ood[:, -1]).float().mean().item())

    return arm, {
        "seed": seed,
        "n_params": count_parameters(model),
        "acc_id_pad4": acc_id,
        "loss_id_pad4": loss_id,
        "acc_ood_pad8": acc_ood,
        "seconds": wall,
    }


def run_track2_delayed_recall(seeds: list[int], steps: int, workers: int) -> dict[str, Any]:
    print("\n=== Track 2B: Clean Deterministic Delayed Recall (train pad=2..4, eval pad=4 ID & pad=8 OOD) ===", flush=True)
    ds = build_deterministic_delayed_recall(k=4, pad_len=4, ood_pad_len=8, train_pad_lens=(2, 3, 4))
    arms = (
        "l2_quantum_unitary_ssm",
        "classical_orthogonal_ssm",
        "classical_rnn",
        "l0_causal_conv",
        "l1_quantum_legacy_rzz",
        "l1_quantum_fixed_rzy",
        "l1_quantum_fixed_ryz",
    )
    ordered = (
        "l1_quantum_legacy_rzz", "l1_quantum_fixed_rzy", "l1_quantum_fixed_ryz",
        "l2_quantum_unitary_ssm", "classical_orthogonal_ssm", "classical_rnn", "l0_causal_conv",
    )
    tasks = [(arm, ds, steps, seed) for arm in ordered for seed in seeds]
    per_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for arm, r in ex.map(_train_recall_arm, tasks):
                per_arm[arm].append(r)
    else:
        for t in tasks:
            arm, r = _train_recall_arm(t)
            per_arm[arm].append(r)
    for arm in arms:
        per_arm[arm].sort(key=lambda r: r["seed"])

    agg = {}
    for arm in arms:
        runs = per_arm[arm]
        agg[arm] = {
            "n_params": runs[0]["n_params"],
            "acc_id_pad4": _mean_std([r["acc_id_pad4"] for r in runs]),
            "loss_id_pad4": _mean_std([r["loss_id_pad4"] for r in runs]),
            "acc_ood_pad8": _mean_std([r["acc_ood_pad8"] for r in runs]),
            "seconds_mean": float(np.mean([r["seconds"] for r in runs])),
        }
        print(
            f"  {arm:26s} (p={agg[arm]['n_params']:3d}) | "
            f"ID(pad=4)={agg[arm]['acc_id_pad4']['mean']:.3f}±{agg[arm]['acc_id_pad4']['std']:.3f} "
            f"(loss={agg[arm]['loss_id_pad4']['mean']:.3f}) | "
            f"OOD(pad=8)={agg[arm]['acc_ood_pad8']['mean']:.3f}±{agg[arm]['acc_ood_pad8']['std']:.3f} | "
            f"{agg[arm]['seconds_mean']:.2f}s",
            flush=True,
        )

    paired = {
        "l1_fixed_rzy_minus_l1_legacy_rzz|acc_id": _paired_ci(
            [r["acc_id_pad4"] for r in per_arm["l1_quantum_fixed_rzy"]],
            [r["acc_id_pad4"] for r in per_arm["l1_quantum_legacy_rzz"]],
        ),
        "l1_fixed_ryz_minus_l1_legacy_rzz|acc_id": _paired_ci(
            [r["acc_id_pad4"] for r in per_arm["l1_quantum_fixed_ryz"]],
            [r["acc_id_pad4"] for r in per_arm["l1_quantum_legacy_rzz"]],
        ),
        "l2_qssm_minus_l1_legacy_rzz|acc_id": _paired_ci(
            [r["acc_id_pad4"] for r in per_arm["l2_quantum_unitary_ssm"]],
            [r["acc_id_pad4"] for r in per_arm["l1_quantum_legacy_rzz"]],
        ),
        "l2_qssm_minus_l0_conv|acc_id": _paired_ci(
            [r["acc_id_pad4"] for r in per_arm["l2_quantum_unitary_ssm"]],
            [r["acc_id_pad4"] for r in per_arm["l0_causal_conv"]],
        ),
    }
    _holm(paired, list(paired.keys()))
    return {
        "task": "Clean Deterministic Delayed Recall (all K=4 sequences evaluated exactly; chance=0.2500)",
        "k": 4,
        "pad_len_id": 4,
        "pad_len_ood": 8,
        "arms": agg,
        "paired_comparisons": paired,
        "per_seed": per_arm,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--t1-steps", type=int, default=450)
    ap.add_argument("--s3-steps", type=int, default=250)
    ap.add_argument("--recall-steps", type=int, default=150)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.quick:
        args.seeds = [0, 1]
        args.t1_steps = 30
        args.s3_steps = 20
        args.recall_steps = 20
        args.workers = 1

    torch.set_num_threads(1)
    t0 = time.perf_counter()
    t1a = run_track1_periodic(args.seeds, args.t1_steps, args.quick, args.workers)
    t1b = run_track1_dyck1(args.seeds, args.t1_steps, args.quick, args.workers)
    t2a = run_track2_s3(args.seeds, args.s3_steps, args.quick, args.workers)
    t2b = run_track2_delayed_recall(args.seeds, args.recall_steps, args.workers)

    report = {
        "benchmark": "Regime III Quantum-Native Sequence Modeling (Track 1 Born Machine + Track 2 True L2 Unitary SSM)",
        "config": {
            "seeds": list(args.seeds),
            "t1_steps": args.t1_steps,
            "s3_steps": args.s3_steps,
            "recall_steps": args.recall_steps,
            "quick": args.quick,
        },
        "track1_periodic_interference": t1a,
        "track1_dyck1": t1b,
        "track2_s3_state_tracking": t2a,
        "track2_deterministic_delayed_recall": t2b,
        "caveat": (
            "All runs use exact CPU statevector / tensor-network simulation (no hardware shot noise). "
            "MPS Born Machines scale polynomially O(T*V*chi^3) classically and compile to shallow sequential "
            "quantum circuits; scaling QuantumUnitarySSM to exponentially large Hilbert spaces requires FTQC."
        ),
        "wallclock_seconds_total": time.perf_counter() - t0,
    }
    out_path = Path(args.out) if args.out else ((RESULTS if args.quick else DOCS) / "quantum_native_seq_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[DONE] wrote {out_path} in {report['wallclock_seconds_total']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
