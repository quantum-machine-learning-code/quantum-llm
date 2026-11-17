# Changelog

All notable changes to `quantum-llm` will be documented in this file.

## [Unreleased] - 2026-11-17

### Added
- L1 short-budget pilot ([`experiments/recurrent_pilot.py`](experiments/recurrent_pilot.py)): standard next-token prediction on the independent character split (k=4 coverage 0.3867), a delayed-recall memory task, and a 2x2 `{quantum, classical} x {feedback on/off}` device calibration. Quantifies P1-P4 contracts, asserts the state ablation is single-variable at the code level, and measures per-step cost plus peak working set (333 MB, single process).
- `check_recurrent_pilot()` offline contract test; `--recompute-from` / `--redraw-from` for recomputing the verdict or redrawing the Chinese figure without retraining.
- Archives [`docs/recurrent_pilot.json`](docs/recurrent_pilot.json) and [`docs/figures/recurrent_pilot.png`](docs/figures/recurrent_pilot.png); documentation in README 1f, REVIEW section 16, and ARCHITECTURE_PROPOSAL section 10.

### Notes
- **P1-P4 structural contracts pass 4/4**; disabling state feedback zeroes prefix sensitivity and cross-step gradients exactly. **Functional state use is NOT demonstrated**: on delayed recall both quantum arms stay at 0.297 (random baseline 0.25) while the classical positive control reaches 0.594. No quantum-vs-classical comparison, paired CI, or Holm claim is made; A2/A3 long runs remain unstarted pending a budget/memory window.

### Cleanup (post-pilot audit)
- Published the delayed-recall criterion (`last-token acc > 0.25 + 0.10`; a device-calibration threshold, **not** a significance test) in `verdict.functional_state_use.criterion`.
- Made the peak working-set number reproducible: `--probe-memory` rechecks it in a fresh process (334.8 MB vs 332.7 MB at run time, 2.1 MB baseline drift), and `runtime.memory_probe` records both readings plus the reproduce command. No hand-written metric patches remain.
- `--recompute-from` / `--redraw-from` now honour `--out`, so a redraw cannot silently overwrite the committed figure; removed quick-mode smoke artifacts from `results/` (gitignored and regenerable).

### Skill-based audit (all 5 research skills + project guardrails)
- **New mechanistic diagnostic** `experiments/state_channel_probe.py` + `docs/state_channel_probe.json`: analytic/numeric gradient check on the L1 QNode. With a zero-weight entangler, `RZ` phase feedback is invisible to `expval(Z)` (`2.29e-9`) while `expval(X)` sees it (`4.13e-2`); at the default `init_std=0.05` the `RZ+Z` channel is ~163x weaker than `RY+Z` (`5.20e-3` vs `8.48e-1`). This is a **testable candidate cause** for the zero recall gain, not a training result; section 16's "functional state use unproven" is unchanged. `check_state_channel_probe()` locks three structural properties as contracts.
- **Statistical disclosure** (`reporting_scope` in `docs/recurrent_pilot.json`, README 1f, REVIEW section 16): independent replicate = seed (n=2); all tabulated numbers are 2-seed illustrative means without SD/CI; delayed recall has only k=4 distinct input sequences with full train/eval input overlap (not held-out generalisation); 32 samples/seed are not 32 independent replicates.
- Chinese figure footer now states the n=2 / no-CI / non-held-out limits (redrawn with `--redraw-from`, no retraining).
- Next-step priority changed: a single-variable A/B on readout basis / state gate comes before longer training.

## [0.2.0] - 2026-11-14

### Added
- Packaging metadata via [`pyproject.toml`](pyproject.toml) (`qml_lab` package, Python `>=3.11`, optional `[dev]` dependencies, pytest and ruff configuration).
- GitHub Actions CI workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) running `ruff check .` and `pytest -v tests/smoke_test.py` on Python 3.11 and 3.12 with CPU-only PyTorch.
- Monthly Dependabot automation ([`.github/dependabot.yml`](.github/dependabot.yml)) for `pip` and `github-actions`.
- English documentation [`README_EN.md`](README_EN.md) with bidirectional language switcher in [`README.md`](README.md), plus [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Task 1 modulo-112 addition (`experiments/train_addition.py`), three-way embedding ablation (`experiments/embedding_ablation.py`), and negative grokking scan (`experiments/grokking_scan.py`).
- Task 2 disjoint independent-corpus benchmark (`experiments/independent_corpus.py`) and 20-seed robustness audit (`experiments/robustness_check.py`).

### Changed
- Exported `test_* = check_*` aliases for all 10 offline contract/smoke suites in [`tests/smoke_test.py`](tests/smoke_test.py), and isolated `check_quick_cli()` output to a temporary directory via `--out`.
- Clarified directed two-body Pauli pairing (`Z_i Z_j` and `Z_i X_j`, $n(n-1)$ operators per family, 48 total on 4 qubits with single-qubit `X/Y/Z`) in [`qml_lab/quantum.py`](qml_lab/quantum.py).
- Cleaned up unused imports across `qml_lab/models.py` and `experiments/`.

## [0.1.0] - 2026-11-07

### Added
- Initial controlled methodological audit of toy quantum character-level language models (`TinyLM`, `RichQuantumLM`, `CausalContextEncoder`), corpus $k$-gram overlap auditing, and 11-group × 5-seed readout bottleneck benchmark.
