# quantum-llm · Methodological Audit & Quantum-Native Sequence Benchmarks

English | [简体中文](README.md)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Scope](https://img.shields.io/badge/scope-toy%20%2F%20methodological%20benchmark-orange)

> **Scope disclaimer**: This repository is **not** a conversational LLM. It is a controlled CPU statevector / tensor-network benchmark separating **Regime II (tiny VQC bottlenecks / hybrid SWAP-Test swaps inside classical backbones)** from **Regime III (quantum-native sequence models: autoregressive MPS Born Machines + true L2 Unitary SSMs + coherent LCU+QSVT Quixer Attention)**. Formal IMRaD reports: [`docs/reports/`](docs/reports/); audit log: [`docs/REVIEW.md`](docs/REVIEW.md); sister repo: [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist).

---

## Research Questions ↔ Experiments ↔ Verdicts

| Research Question (RQ) | Paradigm / Script & Archive | Controlled Verdict (5 Seeds, Holm-Bonferroni) |
| --- | --- | --- |
| **RQ1**: On zero-overlap mod-112 addition, does a 4q L0 VQC beat a strictly matched classical layer? | Regime II · `train_addition.py --matched` ([`matched_v4_metrics.json`](docs/matched_v4_metrics.json)) | **No**. Quantum `0.3251 ± 0.0464` vs Classical `0.2767 ± 0.0917` ($\Delta=+0.0485$, CI `[-0.0677, +0.1647]`, $p=0.311$); learnability comes from the shared circular prior (`0.1835` vs `0.0062`), quantum params `12/10230 (0.12%)`. |
| **RQ2**: Was the legacy char-LM gap (`+0.3263`) a confound, and does L0 VQC beat a 12-param trig surrogate? | Regime II · `matched_task2.py` ([`matched_task2_metrics.json`](docs/matched_task2_metrics.json)) | **Confound-driven + parity**. Legacy `scale=5.0 + ni=True` alone inflates gap `+0.1249` ($p_{\text{Holm}}=0.0097$); matched Q ties 12-param `ClassicalTrig` (`val 2.4804` vs `2.4834`, $\Delta=-0.0030$, $p_{\text{Holm}}=0.895$) at **20.7×** cost (`12/3646 = 0.33%` Q). |
| **RQ3**: Why did the legacy L1 quantum RNN stay near chance on delayed recall? | L1 probe · `state_channel_probe.py` ([`state_channel_probe.json`](docs/state_channel_probe.json)) | **Dead-channel bug**. At $s=0, w=0$, `RZ+Z` and `RZ+X` vanish (`0.000e+00`) while conjugate `RZ+Y` (`1.140e-01`) and `RY+Z` (`7.951e-01`) stay $O(1)$; fixing this lifts L1 recall from `0.500` to `0.750` (`RZ+Y`) / `0.900` (`RY+Z`). |
| **RQ4**: Does MPS Born Machine destructive interference beat parameter-matched non-negative MPS & RNN? | Regime III Track 1 · `benchmark_quantum_native_seq.py` ([`quantum_native_seq_metrics.json`](docs/quantum_native_seq_metrics.json)) | **Yes (interference representation)**. On Periodic-4 OOD $T=14$, 24p complex $\chi=2$ MPS hits exact `KL=0.0949` vs 24p non-negative $\chi=3$ MPS `0.3929` ($p_{\text{Holm}}=0.0247$) and RNN `0.5144` ($p_{\text{Holm}}=0.00573$); on Dyck-1, 84p signed MPS `NLL_ID=0.4415` beats 84p non-negative `0.4732` ($p_{\text{Holm}}=0.000382$). |
| **RQ5**: Does true L2 persistent Unitary SSM achieve OOD non-Abelian state tracking? | Regime III Track 2 · `benchmark_quantum_native_seq.py` ([`quantum_native_seq_metrics.json`](docs/quantum_native_seq_metrics.json)) | **Yes (vs L0/L1/RNN), tied with $SO(9)$**. Hits **`1.000 ± 0.000`** on $S_3$ (OOD $T=16$) and Delayed Recall (OOD `pad=8`, 0 MLP, PennyLane `<2.4e-7`), beating L0 (`0.156`), L1 (`0.291`), RNN (`0.270`), tied with `OrthogonalSSM` (`1.000`). |
| **RQ6**: What are the true boundaries of Quantum Self-Attention (SWAP-Test vs LCU+QSVT Quixer) on KV Recall? | Regime I/II/III · `benchmark_quantum_attention.py` ([`quantum_attention_metrics.json`](docs/quantum_attention_metrics.json)) | **SWAP-Test equals classical + $O(T^2 S)$ shot wall; Quixer hits `0.9910` with 0 Softmax/MLP**. Analytic SWAP-Test ties `ClassicalOverlap` (`1.0000`, CSWAP `7.77e-16`), but $S=100..2000$ adds `MAE 0.0737..0.0164` & up to `110,000` circuits; Quixer LCU+QSVT (`946q/0c`, PennyLane `<2.9e-7`) hits **`0.9910`** ID/OOD with $p_{\text{succ}}=0.423 / 0.395$. |

---

## Formal Experimental Reports (`docs/reports/` · IMRaD + QML Checklists)

- **Report 01**: [`docs/reports/01_regime2_l0_vqc_confound_report.md`](docs/reports/01_regime2_l0_vqc_confound_report.md) — Regime II L0 VQC confound resolution, Task 1 `matched_v4`, and 12-param `ClassicalTrigBilinearLayer` benchmark.
- **Report 02**: [`docs/reports/02_l1_bloch_channel_probe_report.md`](docs/reports/02_l1_bloch_channel_probe_report.md) — L1 measurement-feedback Bloch-sphere proof ($\langle Y\rangle=\sin x\sin s$) and delayed-recall validation.
- **Report 03**: [`docs/reports/03_regime3_born_and_unitary_ssm_report.md`](docs/reports/03_regime3_born_and_unitary_ssm_report.md) — Regime III MPS Born Machine interference (Periodic-4 / Dyck-1) & True L2 `QuantumUnitarySSM` ($S_3$ / Delayed Recall).
- **Report 04**: [`docs/reports/04_quantum_attention_benchmark_report.md`](docs/reports/04_quantum_attention_benchmark_report.md) — Quantum Self-Attention three-regime benchmark: SWAP-Test shot wall, coherent LCU+QSVT Quixer $p_{\text{succ}}$, and FTQC/QRAM limits.

---

## 1. Regime II: Strictly Matched L0 VQC & L1 State-Channel Diagnosis

### 1.1 Task 2 (Independent Char Corpus, Canonical 30-Pauli / 3 QWC, [`docs/matched_task2_metrics.json`](docs/matched_task2_metrics.json))

| Arm | Total Params (Q / C-tf) | QWC | Init Std | Train Loss | Indep. Val Loss | Gen Gap | Final Scale | Time / seed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `caus_q_rich30` (matched) | `3,646` (**12** = `0.33%` / 0) | **3** | `0.3261` | `2.0992 ± 0.0074` | `2.4804 ± 0.0291` | `0.3812 ± 0.0354` | `1.57` | `31.6 s` |
| `caus_classical_tanh` (matched) | `3,784` (0 / 150) | 0 | `0.3951` | `2.1435 ± 0.0115` | `2.4425 ± 0.0400` | `0.2990 ± 0.0399` | `0.93` | `1.2 s` |
| `caus_classical_trig` (12-param) | `3,646` (0 / **12**) | 0 | `0.3765` | `2.0498 ± 0.0213` | `2.4834 ± 0.0335` | `0.4335 ± 0.0275` | `1.30` | `1.5 s` |
| `diag_unmatched_q_scale5` (diag) | `3,645` (12 / 0) | 3 | — | `2.0315 ± 0.0271` | `2.5376 ± 0.0361` | `0.5062 ± 0.0415` | `5.00` | `31.2 s` |

Re-introducing legacy `scale=5.0 + ni=True` alone inflates the gap by **`+0.1249`** ($p_{\text{Holm}}=0.009705$); once matched, Quantum ties 12-param `ClassicalTrig` (`Δ=-0.0030`, $p_{\text{Holm}}=0.895$). Task 1 `matched_v4` ([`matched_v4_metrics.json`](docs/matched_v4_metrics.json)) likewise shows no accuracy gain (`0.3251 ± 0.0464` vs `0.2767 ± 0.0917`, $p=0.311$), and [`grokking_no_prior.json`](docs/grokking_no_prior.json) formally withdraws the §12 architecture-bottleneck claim.

### 1.2 L1 Bloch-Sphere Channel Probe ([`docs/state_channel_probe.json`](docs/state_channel_probe.json))

| Entangler `init_std` | `RZ + Z` ($s=0$ / $s\sim\mathcal{N}$) | `RZ + X` ($s=0$ / $s\sim\mathcal{N}$) | `RZ + Y` conj. ($s=0$ / $s\sim\mathcal{N}$) | `RY + Z` ($s=0$ / $s\sim\mathcal{N}$) |
| --- | --- | --- | --- | --- |
| `0.00` | `0.000e+00` / `2.056e-09` | `0.000e+00` / `4.125e-02` | **`1.140e-01`** / **`1.135e-01`** | `7.951e-01` / `8.389e-01` |
| `0.05` (legacy default) | `1.837e-03` / `5.200e-03` | `4.276e-02` / `4.063e-02` | **`1.084e-01`** / **`1.093e-01`** | `7.998e-01` / `8.476e-01` |
| `0.50` | `8.608e-02` / `9.492e-02` | `1.368e-01` / `1.302e-01` | `1.242e-01` / `1.185e-01` | `5.024e-01` / `5.030e-01` |

`qml_lab/recurrent.py` now defaults to `RZ+Y (init_std=1.0)` and fixes `generate()` boundary double-feeding, true `temperature` sampling, and `detach_state=True` head-gradient detachment.

---

## 2. Regime III: Quantum-Native Sequence & Self-Attention Benchmarks

### 2.1 Tracks 1 & 2: MPS Born Machine & True L2 Unitary SSM ([`docs/quantum_native_seq_metrics.json`](docs/quantum_native_seq_metrics.json))

| Track & Model | Params | ID Metric | OOD Length Metric | Paired Test (Holm-Corrected) |
| --- | ---: | --- | --- | --- |
| **Periodic-4** ($T\in\{4,6,8\} \to 14$) `quantum_mps_complex_chi2` | **24** | `KL=0.0729` / forb `0.048` | **`KL=0.0949`** / forb **`0.057`** | vs 24p NonNeg $\chi=3$ (`0.3929`): $p_{\text{Holm}}=0.0247$ ✓; vs RNN (`0.5144`): $p_{\text{Holm}}=0.00573$ ✓ |
| **Dyck-1** (disjoint $T=14 \to 16$) `quantum_mps_chi6` | **84** | **`NLL=0.4415`** / valid `0.939` | `NLL=0.4704` / valid **`0.856`** | vs 84p NonNeg (`0.4732`, `0.592`): `NLL_ID` $p_{\text{Holm}}=0.000382$ ✓; vs RNN (`0.224`): $p_{\text{Holm}}=0.0144$ ✓ |
| **$S_3$ Tracking** ($T=6 \to 16$) `l2_quantum_unitary_ssm` (0 MLP) | 300 | **`1.000 ± 0.000`** | **`1.000 ± 0.000`** ($T=12, 16$) | Ties $SO(9)$ (`1.000`); beats RNN (`0.270`, $0.000912$ ✓), L1 (`0.291`, $3.32\times10^{-5}$ ✓), L0 (`0.156`) |
| **Delayed Recall** (`pad=4 -> 8`) L2 vs Fixed L1 (`RY+Z`/`RZ+Y`) vs Leg L1 | 344 / 278 | **`1.000`** / `0.900` / `0.750` / `0.500` | **`1.000`** / `0.700` / `0.550` / `0.400` | L2 vs L0 (`0.250`): `+0.750`, $p_{\text{Holm}}=0.0$ ✓; $SO(9)$ OOD `0.950` |

### 2.2 Track 3: Quantum Self-Attention Benchmark (KV Recall $T=6 \to 10$, [`docs/quantum_attention_metrics.json`](docs/quantum_attention_metrics.json))

| Arm (Params Matched `924–946`) | Total (Q) | Test-ID ($T=6$) | Test-OOD ($T=10$) | Physical Cost / Circuit Verification |
| --- | ---: | ---: | ---: | --- |
| Regime I `classical_dot_attn` / `classical_overlap_attn` | 940 / 941 (0) | `0.9630` / `1.0000` | `0.9710` / `1.0000` | Classical Softmax baselines (`Overlap` = exact SWAP-Test surrogate) |
| Regime I `classical_linear_mixer` (Softmax-free) | 938 (0) | `1.0000 ± 0.0000` | `0.9950 ± 0.0112` | Fair linear-superposition control (same `PREPARE`-style gating) |
| Regime II `quantum_swap_test_attn` (5q `CSWAP`) | 941 (0) | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | CSWAP diff `7.77e-16`; $S=100..2000$ OOD `0.997..0.999` (`MAE 0.0737..0.0164`, full causal `5.5k..110k` shots) |
| **Regime III `quantum_quixer_lcu_qsvt` (0 Softmax / 0 MLP)** | **946 (946)** | **`0.9910 ± 0.0074`** | **`0.9910 ± 0.0124`** | 6q `PREPARE/SELECT/PREPARE†+PCPhase` diff `<2.9e-7`; $p_{\text{succ}} = 0.423 / 0.395$ (~1.6× AA) |
| Regime III `quantum_quixer_uniform_lcu` (uniform ablation) | 924 (924) | `0.9870 ± 0.0057` | `0.8300 ± 0.0894` | $p_{\text{succ}}=0.247 / 0.434$; learned `PREPARE` curbs OOD dilution (`+0.1610`, $p=0.0209$) |

> **Honest Hardware & FTQC Boundaries**: 1D MPS scales classically as $O(TV\chi^3)$ (quantum-inspired interference representation, not exponential speedup). SWAP-Test is analytically identical to classical overlap and hits an $O(T^2 S)$ measurement wall. Scaling `QuantumUnitarySSM` or `QuantumQuixerAttention` to exponential Hilbert spaces and long contexts requires fault-tolerant quantum computing (FTQC), $O(1/\sqrt{p_{\text{succ}}})$ amplitude amplification, and QRAM.

---

## 3. Layout & Quickstart

- `qml_lab/`: `quantum.py` (30-Pauli / 3 QWC), `models.py` (`ClassicalTrigBilinearLayer`, `parameter_breakdown`), `recurrent.py` (fixed L1), `born_machine.py`, `quantum_ssm.py`, `quantum_attention.py`, `arithmetic.py`, `data.py`.
- `experiments/`: `state_channel_probe.py`, `matched_task2.py`, `benchmark_quantum_native_seq.py`, `benchmark_quantum_attention.py`, `plot_report_figures.py`, `train_addition.py`.
- `tests/smoke_test.py`: 16 offline contract & regression tests (QWC counts, CSWAP `<1e-10`, LCU+QSVT `<1e-6`, POVM completeness, and `--quick` CLI).

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -v tests/smoke_test.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/matched_task2.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/benchmark_quantum_native_seq.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/benchmark_quantum_attention.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/plot_report_figures.py
```

*Note: On CPU, set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` and `--workers <= 4`; all scripts support `--quick` smoke runs into `results/` without overwriting `docs/`.*
