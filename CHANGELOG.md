# Changelog

All notable changes to `quantum-llm` will be documented in this file.

## [Unreleased] - 2026-11-21

### Added (Phase 3 — Quantum Self-Attention Benchmark & Formal IMRaD Reports)
- **Quantum Self-Attention Three-Regime Prototype (`qml_lab/quantum_attention.py`, `experiments/benchmark_quantum_attention.py`, `docs/quantum_attention_metrics.json`)**:
  - Zero-overlap Key-Value Associative Recall benchmark ($K=4, V=4$, 2 distractors, $T_{\text{train}}=6 \to T_{\text{OOD}}=10$, 5 seeds, strictly matched `924–946` params) across 6 arms: Regime I (`ClassicalDotAttention`, `ClassicalOverlapAttention`, `ClassicalLinearMixer`), Regime II (`QuantumSwapTestAttention` verified against real 5-qubit PennyLane `CSWAP` circuit to `7.77e-16` + finite-shot Bernoulli sampling $S \in \{100, 500, 2000\}$ exposing `MAE 0.0737..0.0164` and `5.5k..110k` full-causal shots), and Regime III (`QuantumQuixerAttention`: 3q $U(8)$ + 6q `PREPARE/SELECT/PREPARE†` LCU + exact degree-3 QSVT `PCPhase` + complete POVM $\sum M_c = I_8$, **946 quantum / 0 classical MLP params**, PennyLane-verified `<2.9e-7`, `0.9910 ± 0.0074` ID & `0.9910 ± 0.0124` OOD, $p_{\text{succ}}=0.423 / 0.395$, plus uniform-LCU ablation `0.8300` OOD).
  - Generated 5 publication-quality 300-dpi figures: `docs/figures/quantum_attention_accuracy.png`, `docs/figures/quantum_attention_shots_and_lcu.png`, plus `experiments/plot_report_figures.py` generating `matched_task2_comparison.png`, `state_channel_probe_bloch.png`, and `quantum_native_seq_summary.png`.
  - Extended `tests/smoke_test.py` to **16 offline contract & regression tests** (`test_quantum_attention`).
- **4 Standalone Formal IMRaD Experimental Reports (`docs/reports/`)**:
  - [`01_regime2_l0_vqc_confound_report.md`](docs/reports/01_regime2_l0_vqc_confound_report.md), [`02_l1_bloch_channel_probe_report.md`](docs/reports/02_l1_bloch_channel_probe_report.md), [`03_regime3_born_and_unitary_ssm_report.md`](docs/reports/03_regime3_born_and_unitary_ssm_report.md), [`04_quantum_attention_benchmark_report.md`](docs/reports/04_quantum_attention_benchmark_report.md) — each with formal hypotheses, circuit/Hilbert-space derivations, verbatim JSON metrics, embedded PNG figures, QML reproducibility checklists, and verified citations.
- Updated synced ZH/EN [`README.md`](README.md) & [`README_EN.md`](README_EN.md) (102 lines each) with Quantum Self-Attention headline results and formal report index.

### Fixed (P0 Audit Fixes & Confound Resolution)
- **`qml_lab/recurrent.py`**:
  - Fixed `detach_state=True` bug that detached `next_state` before computing `self.head(...)` logits (cutting off head gradients on the current step) in both `QuantumRecurrentLM` and `ClassicalRecurrentLM`.
  - Fixed `_RecurrentBase.generate()` double-feeding `prefix[:, -1]` (now matches `forward(prefix)[:, -1].argmax(-1)` on 100% of batches) and implemented true `temperature` sampling via `torch.multinomial`.
  - Added configurable `state_feedback` (`"rz"` / `"ry"`) and `readout_basis` (`"z"` / `"x"` / `"y"` / `"xyz"`) and switched defaults to conjugate `RZ + PauliY` (`init_std=1.0`) to eliminate the `s=0` dead-channel bug.
- **`qml_lab/quantum.py`**: Aligned canonical `"z+x+y+zzall+xxall+yyall"` with `quantum-mnist` (symmetric same-basis 30-Pauli readout, **3 QWC groups**), added programmatic `count_qwc_groups()`, preserved legacy 24/48-Pauli readouts under explicit `"legacy_rich24"` / `"legacy_rich48"` opt-ins (10 QWC groups) for `diagnose_llm.py` and `train_addition.py`, and removed dead MNIST `make_amplitude_layer`.
- **`qml_lab/models.py`**: Fixed `near_identity_init=True` on `transform_type="classical"` (semi-orthogonal tiled init instead of near-zero collapse), added exact 12-param `ClassicalTrigBilinearLayer` surrogate (`transform_type="classical_trig"`), and added `parameter_breakdown(model)`.
- **`qml_lab/arithmetic.py`**: Optimized `answer_logits()` to run `model.transform` only on the final causal position `h_last`, eliminating 75% redundant QNode evaluations.
- **`experiments/train_llm.py`**: Passed `near_identity_init` symmetrically to the classical arm.
- **`experiments/state_channel_probe.py` & `docs/state_channel_probe.json`**: Added `PauliY` conjugate basis and exact initial-state `s=0` regime (`RZ+Z = 0`, `RZ+X = 0`, `RZ+Y = 1.140e-01`, `RY+Z = 7.951e-01` at `s=0, w=0`).
- **`docs/REVIEW.md`**: Fixed §5.2 `caus_q_reup` parameter-count typo (`20 -> 32` quantum params) and appended §17 audit closure.

### Added (Clean Matched Task 2 & Regime III Quantum-Native Sequence Models)
- **`experiments/matched_task2.py` & `docs/matched_task2_metrics.json`**: Strictly matched 3-arm Task 2 benchmark (`caus_q_rich30`, `caus_classical_tanh`, `caus_classical_trig`, 30-Pauli / 3 QWC) plus diagnostic confound arm (`diag_unmatched_q_scale5`). Demonstrates that legacy `scale=5.0 + ni=True` alone inflated the quantum overfitting gap by `+0.1249` ($p_{\text{Holm}}=0.0097$) and that matched L0 quantum ties the 12-param `ClassicalTrig` surrogate (`val 2.4804` vs `2.4834`, $p_{\text{Holm}}=0.895$) at 20.7× wall-clock cost (`12/3646 = 0.33%` quantum params).
- **`qml_lab/born_machine.py`** (Regime III Track 1): Autoregressive `QuantumMPSBornMachine` (exact transfer-matrix normalization & ancestral sampling), exact single-variable control `ClassicalNonNegMPS` (fair `softplus^{-1}` sparse init), `ClassicalCausalRNNLM`, and Periodic-4 / disjoint Dyck-1 generators.
- **`qml_lab/quantum_ssm.py`** (Regime III Track 2): True L2 persistent coherent `QuantumUnitarySSM` (3q $d_H=8$, 1+2-body Pauli Hamiltonians, complete Born-rule POVM $\sum M_c = I_8$, 0 MLP crutch, PennyLane-verified `<2.4e-7`), matched `ClassicalOrthogonalSSM`, `ClassicalRNNTracker`, `CausalConvTracker`, and $S_3$ / deterministic Delayed Recall datasets.
- **`experiments/benchmark_quantum_native_seq.py` & `docs/quantum_native_seq_metrics.json`**: 5-seed Holm-corrected benchmark across Periodic-4, Dyck-1, $S_3$ OOD state tracking ($T=6 \to 16$), and clean Delayed Recall (`pad=4 -> 8`).
- **`tests/smoke_test.py`**: Expanded to 15 offline regression & contract tests covering QWC counts, `near_identity_init` parity, 12-param trig surrogate, `answer_logits` 75%-compute savings, `generate()` parity + temperature sampling, `detach_state=True` gradients, `state_channel_probe` `s=0` physics, Born Machine exact normalization, and `QuantumUnitarySSM` unitarity/POVM/PennyLane equivalence.

### Changed (Documentation Overhaul)
- Rewrote [`README.md`](README.md) and [`README_EN.md`](README_EN.md) to be concise (~130 lines each), 100% synced ZH/EN, opening with a 1-screen RQ ↔ Experiment ↔ Verdict table, verbatim JSON metrics, and explicit FTQC/QRAM boundaries.

## [0.3.0] - 2026-11-17

### Added
- **2026-11-16 (`REVIEW.md` §§13–15)**:
  - Matched-hyperparameter $10^5$-step no-prior grokking audit (`experiments/grokking_no_prior.py`, `docs/grokking_no_prior.json`), withdrawing §12's premature architecture-bottleneck attribution.
  - Strict `--matched` control mode + calibration (`experiments/matched_calibration.py`, `docs/matched_v4_metrics.json`, `docs/capacity_matched_pilot.json`) achieving first valid matched Task 1 calibration (`device_validity.valid = true`, quantum `0.3251` vs classical `0.2767`, $p=0.311$).
  - Architecture eligibility audit (`docs/ARCHITECTURE_PROPOSAL.md`) and initial L1 measurement-feedback prototype (`qml_lab/recurrent.py`).
- **2026-11-17 (`REVIEW.md` §16)**:
  - L1 short-budget pilot (`experiments/recurrent_pilot.py`, `docs/recurrent_pilot.json`): passed P1–P4 structural contracts 4/4, showed functional state use unproven on delayed recall (`0.297` vs classical `0.594`), added reproducible `--probe-memory` and `reporting_scope` disclosures.
  - Initial analytic gradient diagnostic (`experiments/state_channel_probe.py`, `docs/state_channel_probe.json`).

## [0.2.0] - 2026-11-14

### Added
- Packaging metadata via [`pyproject.toml`](pyproject.toml), GitHub Actions CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)), Dependabot, English [`README_EN.md`](README_EN.md), and [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Task 1 modulo-112 addition (`experiments/train_addition.py`), three-way embedding ablation (`experiments/embedding_ablation.py`), and negative grokking scan (`experiments/grokking_scan.py`).
- Task 2 disjoint independent-corpus benchmark (`experiments/independent_corpus.py`) and 20-seed robustness audit (`experiments/robustness_check.py`).

### Changed
- Exported `test_* = check_*` aliases in [`tests/smoke_test.py`](tests/smoke_test.py) and isolated `check_quick_cli()` outputs to temporary directories.
- Clarified directed two-body Pauli pairing (`Z_i Z_j`, `Z_i X_j`) in [`qml_lab/quantum.py`](qml_lab/quantum.py).

## [0.1.0] - 2026-11-07

### Added
- Initial controlled methodological audit of toy quantum character-level language models (`TinyLM`, `RichQuantumLM`, `CausalContextEncoder`), corpus $k$-gram overlap auditing, and 11-group × 5-seed readout bottleneck benchmark.
