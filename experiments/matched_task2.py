"""Regime II Clean Matched Task 2 Benchmark (resolving the uncaught scale=5 vs 1 confound).

Previous Task 2 archives (``diagnose_llm.py``, ``independent_corpus.py``, ``robustness_check.py``)
inherited three confounds:
1. ``scale=5.0`` (quantum) vs ``1.0`` (classical);
2. ``near_identity_init=True`` (quantum) vs ``False`` (classical);
3. ``12`` quantum params vs ``240`` additive ``Linear+Tanh`` params, with NO fair classical
   trigonometric/bilinear surrogate matching the single-encoding PQC's function class.

This script runs a strictly matched 3-arm comparison on the disjoint ``independent`` character
corpus (V=29, k=4 overlap 38.67%) using canonical symmetric observables
``z+x+y+zzall+xxall+yyall`` (30 ops, **3 QWC groups**):
* ``caus_q_rich30``       : 4-qubit L0 PQC (12 quantum params, 0.33% share, 3 QWC groups)
* ``caus_classical_tanh`` : matched-scale additive ``Linear(4,30)+Tanh`` (150 transform params)
* ``caus_classical_trig`` : fair ``ClassicalTrigBilinearLayer`` (**12 trainable params** — exact
  parameter AND degree-1 trigonometric-product function-class surrogate!)
plus a single-variable confound-diagnostic arm ``diag_unmatched_q_scale5`` (``scale=5.0, ni=True``)
to measure directly how much of the legacy ``+0.3263`` overfitting gap was caused by the scale
confound.

Usage:
    python experiments/matched_task2.py --quick           # smoke test, writes results/
    python experiments/matched_task2.py                   # 5 seeds x 500 steps -> docs/
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
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

from qml_lab.data import (
    build_char_data,
    corpus_entropy_floors,
    corpus_overlap,
    iter_batches,
    load_independent_split,
)
from qml_lab.models import RichQuantumLM, parameter_breakdown
from qml_lab.quantum import count_qwc_groups, n_readout
from qml_lab.train import train_lm
from qml_lab.utils import set_seed

DOCS = ROOT / "docs"
RESULTS = ROOT / "results"
CANONICAL_OBS = "z+x+y+zzall+xxall+yyall"  # 30 ops on 4q, 3 QWC groups

MATCHED_ARMS = ("caus_q_rich30", "caus_classical_tanh", "caus_classical_trig")
DIAG_ARMS = ("diag_unmatched_q_scale5",)
ALL_ARMS = MATCHED_ARMS + DIAG_ARMS


def _specs(qubits: int = 4) -> dict[str, dict[str, Any]]:
    base = {
        "context_mode": "causal",
        "context_window": 4,
        "n_qubits": int(qubits),
        "n_layers": 1,
        "observables": CANONICAL_OBS,
        "reupload": False,
        "pre_scale": True,
        "near_identity_init": False,
        "scale": "learnable",
        "scale_init": 1.0,
        "head": "mlp",
        "hidden": 48,
        "readout_norm": False,
    }
    return {
        "caus_q_rich30": {**base, "transform_type": "quantum"},
        "caus_classical_tanh": {**base, "transform_type": "classical"},
        "caus_classical_trig": {**base, "transform_type": "classical_trig"},
        # Diagnostic arm: reproduces the legacy scale=5.0 + near_identity_init=True confound
        "diag_unmatched_q_scale5": {
            **base,
            "transform_type": "quantum",
            "scale": 5.0,
            "near_identity_init": True,
        },
    }


def assert_matched_task2(specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Hard assertion: the 3 matched arms differ ONLY in ``transform_type``."""
    ref = specs[MATCHED_ARMS[0]]
    keys = sorted(ref.keys())
    for arm in MATCHED_ARMS[1:]:
        diff = {
            k: (ref.get(k), specs[arm].get(k))
            for k in keys
            if k != "transform_type" and ref.get(k) != specs[arm].get(k)
        }
        if diff:
            raise AssertionError(f"Matched Task 2 violation ({arm}): {diff}")
    return {
        "matched_arms": list(MATCHED_ARMS),
        "differing_key": "transform_type",
        "n_identical_fields": len(keys) - 1,
        "all_other_fields_identical": True,
    }


def _sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()[:16]


def _build(spec: dict[str, Any], vocab_size: int, seed: int) -> RichQuantumLM:
    set_seed(seed)
    return RichQuantumLM(vocab_size=vocab_size, **spec)


def _verify_weights_and_init_std(
    specs: dict[str, dict[str, Any]],
    vocab_size: int,
    train_data: torch.Tensor,
    seeds: list[int],
) -> dict[str, Any]:
    """Verify identical embed/conv/head/scale init hashes and measure initial readout std."""
    checks: dict[str, Any] = {"all_passed": True, "per_seed": {}, "init_readout_std_seed0": {}}
    probe_x = torch.stack([train_data[i : i + 16] for i in range(0, 128, 16)])

    for s_idx, seed in enumerate(seeds):
        models = {arm: _build(specs[arm], vocab_size, seed) for arm in MATCHED_ARMS}
        hashes = {
            arm: {
                "embed": _sha256(m.embed.weight),
                "causal_conv": _sha256(m.causal_encoder.causal_conv.weight),
                "head_0": _sha256(m.head[0].weight),
                "head_2": _sha256(m.head[2].weight),
                "out_scale": _sha256(m.out_scale.detach()),
            }
            for arm, m in models.items()
        }
        ok = all(
            len({hashes[a][k] for a in MATCHED_ARMS}) == 1
            for k in ("embed", "causal_conv", "head_0", "head_2", "out_scale")
        )
        checks["per_seed"][str(seed)] = {"passed": ok}
        checks["all_passed"] &= ok

        if s_idx == 0:
            for arm, m in models.items():
                m.eval()
                with torch.no_grad():
                    z = m.bottleneck_features(probe_x)
                    checks["init_readout_std_seed0"][arm] = {
                        "raw_std": float(z.std()),
                        "into_head_std": float((z * m.out_scale).std()),
                        "min": float(z.min()),
                        "max": float(z.max()),
                    }
    if not checks["all_passed"]:
        raise AssertionError("Matched Task 2 initial weight hashes do not match!")
    return checks


def _run_task(task: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    train_data, vocab_size, stoi, _ = build_char_data(task["train_text"])
    val_data, _, _, _ = build_char_data(task["val_text"], stoi=stoi)
    model = _build(task["spec"], int(vocab_size), int(task["seed"]))
    breakdown = parameter_breakdown(model)
    gen = iter_batches(train_data, task["seq_len"], task["batch_size"], int(task["seed"]))
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
        "arm": task["arm"],
        "seed": int(task["seed"]),
        "params": breakdown,
        "readout_dim": int(model.readout_dim),
        "final_out_scale": float(model.out_scale.detach().item()),
        "eval_steps": [int(s) for s in hist.get("eval_steps", [])],
        "train_curve": [float(v) for v in hist.get("eval_loss_curve", [])],
        "val_curve": [float(v) for v in hist.get("val_loss_curve", [])],
        "train_loss": float(hist["eval_loss"]),
        "val_loss": float(hist["val_loss"]),
        "val_perplexity": float(hist["val_perplexity"]),
        "generalization_gap": float(hist["val_loss"] - hist["eval_loss"]),
        "total_seconds": float(hist["total_seconds"]),
    }


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.quick:
        args.steps = 10
        args.eval_every = 5
        args.seeds = [0, 1]
        args.workers = 1

    train_text, val_text = load_independent_split()
    train_data, vocab_size, _, _ = build_char_data(train_text)
    vocab_size = int(vocab_size)
    overlap = corpus_overlap(train_text, val_text)
    floors = corpus_entropy_floors(train_text, max_context=8, seq_len=args.seq_len)

    specs = _specs(args.qubits)
    match_assert = assert_matched_task2(specs)
    qwc = count_qwc_groups(args.qubits, CANONICAL_OBS)
    rdim = n_readout(args.qubits, CANONICAL_OBS)
    w_checks = _verify_weights_and_init_std(specs, vocab_size, train_data, args.seeds)

    print(
        f"[matched_task2] canonical obs={CANONICAL_OBS} -> dim={rdim}, QWC={qwc} | "
        f"seeds={args.seeds}, steps={args.steps}, lr={args.lr}, workers={args.workers}",
        flush=True,
    )
    for arm, st in w_checks["init_readout_std_seed0"].items():
        print(f"  init std ({arm:20s}): raw={st['raw_std']:.4f} into_head={st['into_head_std']:.4f}", flush=True)

    tasks = []
    # Run slower quantum tasks first for better multi-process load balancing
    order = ("caus_q_rich30", "diag_unmatched_q_scale5", "caus_classical_tanh", "caus_classical_trig")
    for arm in order:
        for seed in args.seeds:
            tasks.append({
                "arm": arm,
                "spec": specs[arm],
                "seed": int(seed),
                "train_text": train_text,
                "val_text": val_text,
                "steps": args.steps,
                "eval_every": args.eval_every,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "lr": args.lr,
            })

    t0 = time.perf_counter()
    raw: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ALL_ARMS}
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_run_task, t) for t in tasks]
            for fut in as_completed(futs):
                r = fut.result()
                raw[r["arm"]].append(r)
                print(
                    f"  done {r['arm']:24s} seed={r['seed']} | "
                    f"tr={r['train_loss']:.4f} val={r['val_loss']:.4f} "
                    f"gap={r['generalization_gap']:.4f} scale={r['final_out_scale']:.2f} "
                    f"({r['total_seconds']:.1f}s)",
                    flush=True,
                )
    else:
        for t in tasks:
            r = _run_task(t)
            raw[r["arm"]].append(r)

    summary: dict[str, Any] = {}
    for arm in ALL_ARMS:
        runs = sorted(raw[arm], key=lambda r: r["seed"])
        raw[arm] = runs

        def _ms(key: str) -> tuple[float, float]:
            vals = [float(r[key]) for r in runs]
            return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        s: dict[str, Any] = {
            "params": runs[0]["params"],
            "readout_dim": runs[0]["readout_dim"],
            "qwc_groups": qwc if specs[arm]["transform_type"] == "quantum" else 0,
            "is_matched_arm": arm in MATCHED_ARMS,
        }
        for m in ("train_loss", "val_loss", "val_perplexity", "generalization_gap", "final_out_scale", "total_seconds"):
            mu, sd = _ms(m)
            s[f"{m}_mean"] = mu
            s[f"{m}_std"] = sd
        # Step-300 checkpoint summary if present (direct comparison to legacy independent_corpus.py)
        if 300 in runs[0]["eval_steps"]:
            i300 = runs[0]["eval_steps"].index(300)
            tr3 = [r["train_curve"][i300] for r in runs]
            va3 = [r["val_curve"][i300] for r in runs]
            gp3 = [v - t for v, t in zip(va3, tr3)]
            s["at_step_300"] = {
                "train_loss_mean": float(np.mean(tr3)),
                "val_loss_mean": float(np.mean(va3)),
                "generalization_gap_mean": float(np.mean(gp3)),
            }
        s["per_seed"] = [
            {
                "seed": r["seed"],
                "train_loss": r["train_loss"],
                "val_loss": r["val_loss"],
                "generalization_gap": r["generalization_gap"],
                "final_out_scale": r["final_out_scale"],
                "seconds": r["total_seconds"],
            }
            for r in runs
        ]
        summary[arm] = s

    # Paired comparisons across matched arms + confound isolation (diag vs matched quantum)
    pairs_def = [
        ("q_minus_tanh", "caus_q_rich30", "caus_classical_tanh"),
        ("q_minus_trig", "caus_q_rich30", "caus_classical_trig"),
        ("trig_minus_tanh", "caus_classical_trig", "caus_classical_tanh"),
        ("diag_scale5_minus_matched_q", "diag_unmatched_q_scale5", "caus_q_rich30"),
    ]
    paired: dict[str, dict[str, Any]] = {}
    primary_keys: list[str] = []
    for pname, a, b in pairs_def:
        for metric in ("train_loss", "val_loss", "generalization_gap"):
            key = f"{pname}|{metric}"
            paired[key] = _paired_ci(
                [r[metric] for r in raw[a]],
                [r[metric] for r in raw[b]],
            )
            if pname in ("q_minus_tanh", "q_minus_trig", "trig_minus_tanh"):
                primary_keys.append(key)
    _holm(paired, primary_keys)
    # Also Holm-correct the diagnostic scale-5 vs matched-Q family separately
    diag_keys = [f"diag_scale5_minus_matched_q|{m}" for m in ("train_loss", "val_loss", "generalization_gap")]
    _holm(paired, diag_keys)

    q_s = summary["caus_q_rich30"]
    t_s = summary["caus_classical_tanh"]
    g_s = summary["caus_classical_trig"]
    d_s = summary["diag_unmatched_q_scale5"]

    report = {
        "task": "Regime II Clean Matched Task 2 (independent char corpus, canonical 30-Pauli / 3 QWC)",
        "question": (
            "1) Does the legacy +0.3263 quantum overfitting gap disappear or persist once scale=5 vs 1 "
            "and near_identity_init confounds are removed? "
            "2) Does the L0 4-qubit PQC outperform a fair 12-param ClassicalTrigBilinearLayer surrogate?"
        ),
        "config": {
            "qubits": args.qubits,
            "observables": CANONICAL_OBS,
            "readout_dim": rdim,
            "qwc_groups": qwc,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "eval_every": args.eval_every,
            "lr": args.lr,
            "seeds": list(args.seeds),
            "vocab_size": vocab_size,
            "train_chars": len(train_text),
            "val_chars": len(val_text),
            "val_context_k4_coverage": overlap["val_context_covered_by_train"]["4"]["val_token_coverage"],
            "train_k4_entropy_floor": round(floors[4]["loss"], 4),
        },
        "matched_control_assertion": match_assert,
        "weight_control_checks": w_checks,
        "arms": summary,
        "paired_comparisons": paired,
        "interpretation": {
            "confound_resolution": (
                f"Matched L0 quantum gap = {q_s['generalization_gap_mean']:.4f} vs "
                f"matched classical tanh = {t_s['generalization_gap_mean']:.4f} "
                f"(Δ={paired['q_minus_tanh|generalization_gap']['mean_diff']:+.4f}, "
                f"p_holm={paired['q_minus_tanh|generalization_gap']['p_holm']:.4g}) and "
                f"fair 12-param classical trig = {g_s['generalization_gap_mean']:.4f} "
                f"(Δ={paired['q_minus_trig|generalization_gap']['mean_diff']:+.4f}, "
                f"p_holm={paired['q_minus_trig|generalization_gap']['p_holm']:.4g}). "
                f"Re-introducing the legacy scale=5.0 + ni=True confound alone shifts quantum gap to "
                f"{d_s['generalization_gap_mean']:.4f} "
                f"(Δ vs matched Q = {paired['diag_scale5_minus_matched_q|generalization_gap']['mean_diff']:+.4f}, "
                f"p_holm={paired['diag_scale5_minus_matched_q|generalization_gap']['p_holm']:.4g})."
            ),
            "fair_surrogate_verdict": (
                f"Matched Q val={q_s['val_loss_mean']:.4f}±{q_s['val_loss_std']:.4f} vs "
                f"12-param ClassicalTrig val={g_s['val_loss_mean']:.4f}±{g_s['val_loss_std']:.4f} "
                f"(Δ={paired['q_minus_trig|val_loss']['mean_diff']:+.4f}, "
                f"p_holm={paired['q_minus_trig|val_loss']['p_holm']:.4g}) at "
                f"{q_s['total_seconds_mean'] / max(g_s['total_seconds_mean'], 1e-6):.1f}× wall-clock cost; "
                f"quantum params are only {q_s['params']['transform_quantum']}/{q_s['params']['total']} "
                f"({100 * q_s['params']['quantum_param_share']:.2f}%)."
            ),
        },
        "wallclock_seconds_total": time.perf_counter() - t0,
    }

    out_path = Path(args.out) if args.out else ((RESULTS if args.quick else DOCS) / "matched_task2_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[summary] wrote {out_path} ({report['wallclock_seconds_total']:.1f}s)")
    for arm in ALL_ARMS:
        s = summary[arm]
        print(
            f"  {arm:24s} | params={s['params']['total']} (q={s['params']['transform_quantum']}, "
            f"c_tf={s['params']['transform_classical']}) | tr={s['train_loss_mean']:.4f} "
            f"val={s['val_loss_mean']:.4f}±{s['val_loss_std']:.4f} gap={s['generalization_gap_mean']:.4f} "
            f"scale={s['final_out_scale_mean']:.2f} | {s['total_seconds_mean']:.1f}s"
        )
    print("  confound_resolution:", report["interpretation"]["confound_resolution"])
    print("  fair_surrogate_verdict:", report["interpretation"]["fair_surrogate_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
