# quantum-llm · Methodological Audit & Quantum-Native Sequence Benchmarks

English | [简体中文](README.md)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Scope](https://img.shields.io/badge/scope-toy%20%2F%20methodological%20benchmark-orange)

> **Scope disclaimer**: This repository is **not** a conversational LLM. It is a controlled CPU statevector / tensor-network benchmark separating **Regime II (tiny VQC bottlenecks embedded in classical backbones)** from **Regime III (quantum-native sequence models: autoregressive MPS Born Machines + true L2 persistent Unitary SSMs)**. Full audit log: [`docs/REVIEW.md`](docs/REVIEW.md); architecture spec: [`docs/ARCHITECTURE_PROPOSAL.md`](docs/ARCHITECTURE_PROPOSAL.md); sister repo: [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist).

---

## Research Questions ↔ Experiments ↔ Verdicts

| Research Question (RQ) | Paradigm / Script & Archive | Controlled Verdict (5 Seeds, Holm-Bonferroni) |
| --- | --- | --- |
| **RQ1**: On zero-overlap mod-112 compositional addition, does a 4q L0 VQC beat a strictly matched classical layer? | Regime II · `train_addition.py --matched` ([`matched_v4_metrics.json`](docs/matched_v4_metrics.json)) | **No**. Quantum `0.3251 ± 0.0464` vs Classical `0.2767 ± 0.0917` ($\Delta=+0.0485$, CI `[-0.0677, +0.1647]`, $p=0.311$). Learnability comes from the shared circular prior (`0.1835` vs `0.0062`); quantum weights are `12/10230 (0.12%)`. |
| **RQ2**: Was the legacy char-LM quantum overfitting gap (`+0.3263`) a confound artifact, and does L0 VQC beat a parameter-matched classical trig surrogate? | Regime II · `matched_task2.py` ([`matched_task2_metrics.json`](docs/matched_task2_metrics.json)) | **Confound-driven + parity**. Re-introducing legacy `scale=5.0 + ni=True` alone inflates the gap by `+0.1249` ($p_{\text{Holm}}=0.0097$). Once matched, Quantum ties the 12-param `ClassicalTrig` surrogate (`val 2.4804` vs `2.4834`, $\Delta=-0.0030$, $p_{\text{Holm}}=0.895$) at **20.7×** wall-clock cost (`12/3646 = 0.33%` quantum params). |
| **RQ3**: Why did the legacy L1 quantum RNN stay near chance on delayed recall? | L1 probe · `state_channel_probe.py` ([`state_channel_probe.json`](docs/state_channel_probe.json)) | **Dead-channel bug**. At exact initial state $s=0, w=0$, both `RZ+Z` and `RZ+X` have `0.000e+00` gradient, whereas conjugate `RZ+Y` (`1.140e-01`) and `RY+Z` (`7.951e-01`) remain $O(1)$. Fixing this lifts L1 recall from `0.500` to `0.750` (`RZ+Y`) / `0.900` (`RY+Z`). |
| **RQ4**: Does quantum destructive interference (MPS Born Machine) outperform parameter-matched non-negative MPS and RNN controls? | Regime III Track 1 · `benchmark_quantum_native_seq.py` ([`quantum_native_seq_metrics.json`](docs/quantum_native_seq_metrics.json)) | **Yes (representational interference advantage)**. On Periodic-4 OOD $T=14$, 24-param complex $\chi=2$ MPS reaches exact `KL=0.0949` vs parameter-matched 24-param non-negative $\chi=3$ MPS `0.3929` ($p_{\text{Holm}}=0.0247$) and RNN `0.5144` ($p_{\text{Holm}}=0.00573$). On Dyck-1, 84-param signed $\chi=6$ MPS beats 84-param non-negative MPS on `NLL_ID` (`0.4415` vs `0.4732`, $p_{\text{Holm}}=0.000382$) and RNN on OOD validity (`+63.2 pt`, $p_{\text{Holm}}=0.0144$). |
| **RQ5**: Does true L2 persistent coherent unitary recurrence (`QuantumUnitarySSM`) achieve OOD non-Abelian state tracking? | Regime III Track 2 · `benchmark_quantum_native_seq.py` ([`quantum_native_seq_metrics.json`](docs/quantum_native_seq_metrics.json)) | **Yes (vs L0/L1/RNN), tied with matched $SO(d)$**. Hits **`1.000 ± 0.000`** on both $S_3$ (train $T=6 \to$ OOD $T=16$) and Delayed Recall (OOD `pad=8`) with 0 MLP crutch (PennyLane verified `<2.4e-7`), decisively beating L0 (`0.156`), L1 (`0.291`), and RNN (`0.270`), while matching `ClassicalOrthogonalSSM` (`1.000`). |

---

## 1. Regime II: Strictly Matched L0 Classical-Backbone + VQC Benchmarks

### 1.1 Task 2 (Independent Character Corpus, Canonical 30-Pauli / 3 QWC Readout)

Legacy `independent_corpus.py` / `robustness_check.py` had an uncaught confound: `scale=5.0, near_identity_init=True` on the quantum arm vs `scale=1.0, near_identity_init=False` on the classical arm, and no trigonometric classical surrogate. `experiments/matched_task2.py` asserts 13 identical fields across arms (learnable `out_scale` initialized at `1.0`, default init, `steps=400, lr=0.01, seeds=0..4`), aligns with `quantum-mnist` on the canonical 30-Pauli readout (`z+x+y+zzall+xxall+yyall`, **3 QWC groups** vs 10 in legacy 48-Pauli), and adds a 12-param `ClassicalTrigBilinearLayer` ([`docs/matched_task2_metrics.json`](docs/matched_task2_metrics.json)):

| Arm | Total Params (Q / C-transform) | QWC | Init Std | Train Loss | Independent Val Loss | Gen Gap (Val−Train) | Final Scale | Time / seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `caus_q_rich30` (matched) | `3,646` (**12** = `0.33%` / 0) | **3** | `0.3261` | `2.0992 ± 0.0074` | `2.4804 ± 0.0291` | `0.3812 ± 0.0354` | `1.57` | `31.6 s` |
| `caus_classical_tanh` (matched) | `3,784` (0 / 150) | 0 | `0.3951` | `2.1435 ± 0.0115` | `2.4425 ± 0.0400` | `0.2990 ± 0.0399` | `0.93` | `1.2 s` |
| `caus_classical_trig` (12-param) | `3,646` (0 / **12**) | 0 | `0.3765` | `2.0498 ± 0.0213` | `2.4834 ± 0.0335` | `0.4335 ± 0.0275` | `1.30` | `1.5 s` |
| `diag_unmatched_q_scale5` (diag) | `3,645` (12 / 0) | 3 | — | `2.0315 ± 0.0271` | `2.5376 ± 0.0361` | `0.5062 ± 0.0415` | `5.00` | `31.2 s` |

- **Confound resolution**: Re-introducing `scale=5.0 + ni=True` alone increases the quantum overfitting gap by **`+0.1249`** ($p_{\text{Holm}}=0.009705$) and degrades val loss by `+0.0572` ($p_{\text{Holm}}=0.0413$).
- **No unique quantum effect**: Once matched, `Q − Tanh` val loss (`+0.0379`, $p_{\text{Holm}}=0.289$) and gap (`+0.0822`, $p_{\text{Holm}}=0.095$) are not Holm-significant. Against the exact 12-param `ClassicalTrig` surrogate, val loss is indistinguishable (`Δ=-0.0030`, $p_{\text{Holm}}=0.895$), confirming the historical behavior came from **degree-1+2 trig monomials + fixed 5× scaling**, not quantum mechanics.

### 1.2 Task 1 (Zero-Overlap Mod-112 Compositional Addition)

- **Matched v4 ([`docs/matched_v4_metrics.json`](docs/matched_v4_metrics.json), 5 seeds × 2000 steps, `lr=0.003, scale_init=5.0`)**: Quantum `0.3251 ± 0.0464` vs Classical `0.2767 ± 0.0917` (5/5 seeds pass device validity on both arms; paired $\Delta=+0.0485$, CI `[-0.0677, +0.1647]`, $p=0.311$; only `generalization_gap` `+0.0156` is Holm-significant — larger gap on quantum, not an accuracy gain).
- **Prior & grokking boundaries**: Embedding ablation ([`embedding_ablation_metrics.json`](docs/embedding_ablation_metrics.json)) confirms random init stalls at `0.0062` (≈ $1/112$), while the shared circular prior adds `+0.1773` / `+0.1400`. Matched-hyperparameter $10^5$-step no-prior reruns ([`grokking_no_prior.json`](docs/grokking_no_prior.json)) show `RICH wd=0` mid-memorization (`train 0.6565 / val 0.0076`) vs `MLP wd=1` (`val 0.9918`); the earlier §12 "architecture bottleneck" claim was formally withdrawn in §13.

---

## 2. L1 Recurrent State-Channel Diagnostic

`experiments/state_channel_probe.py` ([`docs/state_channel_probe.json`](docs/state_channel_probe.json)) evaluates $|\partial(\sum_i \langle O_i \rangle)/\partial s|$ across 4 gate/basis configs at both random $s \sim \mathcal{N}(0, 0.1)$ and exact initial state $s = 0$:

| Entangler `init_std` | `RZ + Z` ($s=0$ / $s\sim\mathcal{N}$) | `RZ + X` ($s=0$ / $s\sim\mathcal{N}$) | `RZ + Y` conjugate ($s=0$ / $s\sim\mathcal{N}$) | `RY + Z` ($s=0$ / $s\sim\mathcal{N}$) |
| --- | --- | --- | --- | --- |
| `0.00` | `0.000e+00` / `2.056e-09` | `0.000e+00` / `4.125e-02` | **`1.140e-01`** / **`1.135e-01`** | `7.951e-01` / `8.389e-01` |
| `0.05` (legacy default) | `1.837e-03` / `5.200e-03` | `4.276e-02` / `4.063e-02` | **`1.084e-01`** / **`1.093e-01`** | `7.998e-01` / `8.476e-01` |
| `0.50` | `8.608e-02` / `9.492e-02` | `1.368e-01` / `1.302e-01` | `1.242e-01` / `1.185e-01` | `5.024e-01` / `5.030e-01` |

Mechanism: `RY(x)|0⟩` lies in the $XZ$ plane, so `RZ(s)` first rotates the Bloch vector toward the $Y$ axis ($\langle Y \rangle = \sin x \sin s$). Hence both `Z` and `X` readouts have zero first-order sensitivity at $s=0, w=0$, whereas `PauliY` is naturally $O(1)$. `qml_lab/recurrent.py` now defaults to `RZ+Y (init_std=1.0)` and fixes the `generate()` boundary-token double-feed bug, real `temperature` sampling, and `detach_state=True` head-gradient detachment.

---

## 3. Regime III: Quantum-Native Sequence Modeling Benchmarks

Archived in [`docs/quantum_native_seq_metrics.json`](docs/quantum_native_seq_metrics.json) (5 seeds, strictly disjoint Train/Val/Test + OOD length generalization, Holm-Bonferroni corrected).

### 3.1 Track 1: Autoregressive Tensor-Network Born Machine (`qml_lab/born_machine.py`)

Exact transfer-matrix contraction gives exact Born probabilities $P(x) = |\langle x|\psi\rangle|^2 / \langle\psi|\psi\rangle$ ($\sum_x P(x)=1$ identically, exact ancestral sampling), tested against its **single-variable classical control `ClassicalNonNegMPS`** (same topology, strictly non-negative entries disabling destructive interference, fair `softplus^{-1}` sparse init) and `ClassicalCausalRNNLM`:

| Task & Model | Params | ID Metric | OOD Length Metric | Paired Test (Holm-Corrected) |
| --- | --- | --- | --- | --- |
| **Periodic-4** ($P^* \propto \cos^2(\frac{\pi}{4}\sum x_t)$, train $T\in\{4,6,8\}$) | | Exact $\text{KL}_{\text{ID}}$ ($T=8$) / Forb. | Exact $\text{KL}_{\text{OOD}}$ ($T=14$) / Forb. | |
| `quantum_mps_complex_chi2` (complex $\chi=2$) | **24** | **`0.0729 ± 0.1526`** / **`0.048`** | **`0.0949 ± 0.1521`** / **`0.057`** | — (4/5 seeds hit `KL_ID ≤ 0.0066`, `forb_T14 ≤ 0.010`) |
| `classical_nonneg_mps_chi2` (same $\chi=2$) | 12 | `0.3667 ± 0.0478` / `0.220` | `0.4527 ± 0.0739` / `0.255` | `Q − NonNeg(χ=2)` $\text{KL}_{\text{ID}} = -0.2938$ ($p_{\text{Holm}}=0.0192$ ✓) |
| `classical_nonneg_mps_chi3` (**exact 24p**, $\chi=3$) | **24** | `0.3066 ± 0.0489` / `0.195` | `0.3929 ± 0.0505` / `0.248` | `Q − NonNeg(χ=3)` $\text{KL}_{\text{OOD}} = -0.2981$ ($p_{\text{Holm}}=0.0247$ ✓) |
| `classical_rnn_matched` | 35 | `0.4310 ± 0.0374` / `0.230` | `0.5144 ± 0.0854` / `0.249` | `Q − RNN` $\text{KL}_{\text{OOD}} = -0.4195$ ($p_{\text{Holm}}=0.00573$ ✓) |
| **Dyck-1 Brackets** (disjoint $T=14 \to 16$; Bayes `0.4330 / 0.4541`) | | Test-ID NLL / Valid | OOD $T=16$ NLL / Valid | |
| `quantum_mps_chi6` (signed real $\chi=6$) | **84** | **`0.4415 ± 0.0070`** / **`0.939`** | **`0.4704 ± 0.0186`** / **`0.856`** | — |
| `classical_nonneg_mps_chi6` (same $\chi=6$, 84p) | **84** | `0.4732 ± 0.0046` / `0.650` | `0.5036 ± 0.0030` / `0.592` | `Q − NonNeg` $\text{NLL}_{\text{ID}} = -0.0317$ ($p_{\text{Holm}}=0.000382$ ✓) |
| `classical_rnn_matched` | 98 | `0.4741 ± 0.0158` / `0.718` | `0.5928 ± 0.0598` / `0.224` | `Q − RNN` OOD valid `+0.632` ($p_{\text{Holm}}=0.0144$ ✓) |

> **Honest boundary**: 1D MPS contraction scales classically as $O(TV\chi^3)$. Track 1 demonstrates a **quantum-inspired destructive-interference / representational separation** (Perron-Frobenius theorem; compiles to shallow sequential circuits), **not** an exponential quantum-hardware speedup.

### 3.2 Track 2: True L2 Coherent Quantum-State Recurrence / Unitary SSM (`qml_lab/quantum_ssm.py`)

`QuantumUnitarySSM` evolves a persistent, uncollapsed 3-qubit coherent state $|\psi_t\rangle = U(x_t)|\psi_{t-1}\rangle \in \mathbb{C}^8$ generated by 1+2-body Pauli Hamiltonians ($U(x_t)=e^{-iH(x_t)}$) with exact complete POVM Born readout ($\sum_c M_c = I_8$), **zero classical MLP crutch**, verified against PennyLane `default.qubit` (`max|Δψ|=1.21e-7`, `max|Δp|=2.38e-7`):

| Model | $S_3$ Word Problem (Params / ID $T=6$ / OOD $T=12$ / **OOD $T=16$**, chance `0.167`) | Deterministic Delayed Recall (Params / ID `pad=4` / **OOD `pad=8`**, chance `0.250`) |
| --- | --- | --- |
| **`l2_quantum_unitary_ssm`** (True L2, 0 MLP) | `300` / **`1.000 ± 0.000`** / **`1.000 ± 0.000`** / **`1.000 ± 0.000`** | `344` / **`1.000 ± 0.000`** / **`1.000 ± 0.000`** |
| `classical_orthogonal_ssm` (matched $SO(9)$) | `270` / `1.000 ± 0.000` / `1.000 ± 0.000` / `1.000 ± 0.000` | `324` / `1.000 ± 0.000` / `0.950 ± 0.112` |
| `classical_rnn` (unconstrained RNN) | `327` / `0.595 ± 0.134` / `0.321 ± 0.153` / `0.270 ± 0.155` | `228` / `0.900 ± 0.137` / `0.500 ± 0.468` |
| `l1_quantum_fixed_rzy` (fixed L1 `RZ+Y`) | `278` / `0.390 ± 0.126` / `0.331 ± 0.070` / `0.291 ± 0.059` | `278` / `0.750 ± 0.306` / `0.550 ± 0.326` |
| `l1_quantum_fixed_ryz` (fixed L1 `RY+Z`) | — | `278` / `0.900 ± 0.137` / `0.700 ± 0.209` |
| `l1_quantum_legacy_rzz` (legacy `RZ+Z`) | — | `278` / `0.500 ± 0.306` / `0.400 ± 0.137` |
| `l0_causal_conv` (fixed window $k=4$) | `334` / `0.180 ± 0.018` / `0.176 ± 0.016` / `0.156 ± 0.018` | `348` / `0.250 ± 0.000` / `0.250 ± 0.000` |

- **$S_3$ OOD $T=16$ paired tests**: L2 decisively beats RNN (`+0.730`, $p_{\text{Holm}}=0.000912$), L1 (`+0.709`, $p_{\text{Holm}}=3.32\times10^{-5}$), and L0 (`+0.844`, $p_{\text{Holm}}=2.06\times10^{-7}$), while matching `ClassicalOrthogonalSSM` ($\Delta=0.000, p_{\text{Holm}}=1.0$).
- **Physical attribution & FTQC boundary**: At $d_H=8$, 100% OOD generalization comes from **norm-preserving coherent group evolution** (unitary/orthogonal vs dissipative RNNs or collapsing L1). Scaling `QuantumUnitarySSM` to exponentially large $2^n$ Hilbert spaces — or implementing **QSVT / block-encoded Quantum Attention** — requires fault-tolerant quantum hardware (FTQC) and QRAM and cannot be claimed as an exponential speedup on classical simulators.

---

## 4. Layout & Quickstart

- `qml_lab/`: `quantum.py` (canonical 30-Pauli / 3 QWC readout + legacy opt-ins), `models.py` (`RichQuantumLM`, `ClassicalTrigBilinearLayer`, `parameter_breakdown`), `recurrent.py` (fixed L1), `born_machine.py` (Regime III Track 1), `quantum_ssm.py` (Regime III Track 2), `arithmetic.py`, `data.py`, `corpus_independent.py`.
- `experiments/`: `state_channel_probe.py`, `matched_task2.py`, `benchmark_quantum_native_seq.py`, `train_addition.py`, `grokking_no_prior.py`, `diagnose_llm.py`.
- `tests/smoke_test.py`: 15 offline contract & regression tests (QWC counts, strict causality, POVM completeness, PennyLane statevector parity, and CLI smoke tests).

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -v tests/smoke_test.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/state_channel_probe.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/matched_task2.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/benchmark_quantum_native_seq.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/train_addition.py --matched --lr 0.003 --scale-init 5 --workers 4 --no-figure
```

*Note: On CPU, set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` and `--workers <= 4`; all experiment scripts support `--quick` smoke runs writing to `results/` without overwriting `docs/`.*
