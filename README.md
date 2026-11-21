# quantum-llm · 量子序列建模方法学审查与原生基准

[English](README_EN.md) | 简体中文

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Scope](https://img.shields.io/badge/scope-toy%20%2F%20methodological%20benchmark-orange)

> **边界声明**：本仓库**不是**可对话的大语言模型，而是纯 CPU 状态向量 / 张量网络模拟器上的受控方法学基准，系统区分 **Regime II（经典主干中嵌入小参数 VQC 瓶颈 / SWAP-Test 混合替换）** 与 **Regime III（量子原生序列建模：MPS 玻恩机 + 真正 L2 幺正 SSM + 相干 LCU+QSVT Quixer 注意力）**。独立 IMRaD 正式实验报告见 [`docs/reports/`](docs/reports/)，完整审查日志见 [`docs/REVIEW.md`](docs/REVIEW.md)，姊妹仓库见 [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist)。

---

## 核心研究问题 ↔ 实验 ↔ 结论一览

| 研究问题 (RQ) | 范式 / 脚本与归档 | 受控结论（5 Seeds，Holm-Bonferroni 校正） |
| --- | --- | --- |
| **RQ1**：模 112 组合泛化加法中，4q L0 VQC 能否超越严格对齐经典层？ | Regime II · `train_addition.py --matched` ([`matched_v4_metrics.json`](docs/matched_v4_metrics.json)) | **否**。量子 `0.3251 ± 0.0464` vs 经典 `0.2767 ± 0.0917`（$\Delta=+0.0485$，CI `[-0.0677, +0.1647]`，$p=0.311$）；可学性由共享循环先验驱动（`0.1835` vs `0.0062`），量子参数仅 `12/10230 (0.12%)`。 |
| **RQ2**：历史字符 LM 过拟合间隙（`+0.3263`）是否为混杂？对齐后能否击败等参三角替身？ | Regime II · `matched_task2.py` ([`matched_task2_metrics.json`](docs/matched_task2_metrics.json)) | **混杂主导 + 无优势**。历史 `scale=5.0 + ni=True` 单独拉大间隙 `+0.1249`（$p_{\text{Holm}}=0.0097$）；对齐后与 12 参 `ClassicalTrig` 验证集无差异（`2.4804` vs `2.4834`，$\Delta=-0.0030$，$p_{\text{Holm}}=0.895$），耗时慢 **20.7×**（量子 `12/3646 = 0.33%`）。 |
| **RQ3**：旧版 L1 量子循环为何在延迟回忆上停在随机基线？ | L1 诊断 · `state_channel_probe.py` ([`state_channel_probe.json`](docs/state_channel_probe.json)) | **死通道缺陷**。$s=0, w=0$ 时 `RZ+Z` 与 `RZ+X` 导数严格为 `0.000e+00`，唯独共轭 `RZ+Y`（`1.140e-01`）与 `RY+Z`（`7.951e-01`）为 $O(1)$；修复后 L1 延迟回忆由 `0.500` 升至 `0.750`（`RZ+Y`）/ `0.900`（`RY+Z`）。 |
| **RQ4**：MPS 玻恩机相消干涉能否在等参条件下击败非负 MPS 与 RNN？ | Regime III 轨道 1 · `benchmark_quantum_native_seq.py` ([`quantum_native_seq_metrics.json`](docs/quantum_native_seq_metrics.json)) | **是（干涉表征优势）**。Periodic-4 OOD $T=14$ 上 24 参复数 $\chi=2$ MPS 精确 `KL=0.0949` vs 等 24 参非负 $\chi=3$ MPS `0.3929`（$p_{\text{Holm}}=0.0247$）与 RNN `0.5144`（$p_{\text{Holm}}=0.00573$）；Dyck-1 上 84 参有符号 MPS `NLL_ID=0.4415` 优于等参非负 `0.4732`（$p_{\text{Holm}}=0.000382$）。 |
| **RQ5**：真正 L2 幺正 SSM 能否实现非阿贝尔群 OOD 长度外推？ | Regime III 轨道 2 · `benchmark_quantum_native_seq.py` ([`quantum_native_seq_metrics.json`](docs/quantum_native_seq_metrics.json)) | **是（相对 L0/L1/RNN），与同维 $SO(9)$ 持平**。$S_3$（OOD $T=16$）与延迟回忆（OOD `pad=8`）均达 **`1.000 ± 0.000`**（0 MLP，PennyLane `<2.4e-7`），击败 L0（`0.156`）、L1（`0.291`）、RNN（`0.270`），持平 `OrthogonalSSM`（`1.000`）。 |
| **RQ6**：量子自注意力（SWAP-Test vs LCU+QSVT Quixer）在零重叠 KV 联想回忆上的真实边界是什么？ | Regime I/II/III · `benchmark_quantum_attention.py` ([`quantum_attention_metrics.json`](docs/quantum_attention_metrics.json)) | **SWAP-Test 解析等价经典 + 有限 shot $O(T^2 S)$ 墙；Quixer 零 Softmax/MLP 达 `0.9910`**。SWAP-Test 解析与 `ClassicalOverlap` 逐位一致（`1.0000`，CSWAP 校验 `7.77e-16`），但 $S=100..2000$ 带来 `MAE 0.0737..0.0164` 与最高 `110,000` 电路开销；Quixer LCU+QSVT（`946q/0c`，PennyLane `<3.7e-7`）ID/OOD 均 **`0.9910`**，实测后选择 $p_{\text{succ}}=0.423 / 0.395$。 |

---

## 正式实验报告（`docs/reports/` · IMRaD + QML 可复现核查表）

- **报告 01**：[`docs/reports/01_regime2_l0_vqc_confound_report.md`](docs/reports/01_regime2_l0_vqc_confound_report.md) — Regime II L0 VQC 混杂清算、课题一 `matched_v4` 与等 12 参 `ClassicalTrigBilinearLayer` 基准。
- **报告 02**：[`docs/reports/02_l1_bloch_channel_probe_report.md`](docs/reports/02_l1_bloch_channel_probe_report.md) — L1 测量反馈循环 Bloch 球状态通道几何证明（$\langle Y\rangle=\sin x\sin s$）与延迟回忆修复验证。
- **报告 03**：[`docs/reports/03_regime3_born_and_unitary_ssm_report.md`](docs/reports/03_regime3_born_and_unitary_ssm_report.md) — Regime III MPS 玻恩机相消干涉（Periodic-4 / Dyck-1）与 L2 `QuantumUnitarySSM`（$S_3$ / 延迟回忆）。
- **报告 04**：[`docs/reports/04_quantum_attention_benchmark_report.md`](docs/reports/04_quantum_attention_benchmark_report.md) — 量子自注意力三范式对照：SWAP-Test 有限 shot 瓶颈、相干 LCU+QSVT Quixer 后选择概率与 FTQC/QRAM 边界。

---

## 一、Regime II：严格对齐的 L0 VQC 与 L1 状态通道诊断

### 1.1 课题二（独立字符语料，规范 30-Pauli / 3 QWC，[`docs/matched_task2_metrics.json`](docs/matched_task2_metrics.json)）

| 臂 | 总参 (量子 / 经典变换) | QWC 组 | 初始 std | Train Loss | 独立 Val Loss | 泛化间隙 | 最终 scale | 耗时 / seed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `caus_q_rich30`（对齐） | `3,646` (**12** = `0.33%` / 0) | **3** | `0.3261` | `2.0992 ± 0.0074` | `2.4804 ± 0.0291` | `0.3812 ± 0.0354` | `1.57` | `31.6 s` |
| `caus_classical_tanh`（对齐） | `3,784` (0 / 150) | 0 | `0.3951` | `2.1435 ± 0.0115` | `2.4425 ± 0.0400` | `0.2990 ± 0.0399` | `0.93` | `1.2 s` |
| `caus_classical_trig`（等 12 参） | `3,646` (0 / **12**) | 0 | `0.3765` | `2.0498 ± 0.0213` | `2.4834 ± 0.0335` | `0.4335 ± 0.0275` | `1.30` | `1.5 s` |
| `diag_unmatched_q_scale5`（诊断） | `3,645` (12 / 0) | 3 | — | `2.0315 ± 0.0271` | `2.5376 ± 0.0361` | `0.5062 ± 0.0415` | `5.00` | `31.2 s` |

仅恢复旧版 `scale=5.0 + ni=True` 就拉大间隙 **`+0.1249`**（$p_{\text{Holm}}=0.009705$）；对齐后量子与等 12 参 `ClassicalTrig` 验证集无差异（`Δ=-0.0030`, $p_{\text{Holm}}=0.895$）。课题一严格对齐 v4（[`matched_v4_metrics.json`](docs/matched_v4_metrics.json)）亦无准确率优势（`0.3251 ± 0.0464` vs `0.2767 ± 0.0917`, $p=0.311$），无先验复验（[`grokking_no_prior.json`](docs/grokking_no_prior.json)）已正式撤回 §12「架构缺陷」归因。

### 1.2 L1 Bloch 球状态通道探针（[`docs/state_channel_probe.json`](docs/state_channel_probe.json)）

| 纠缠层 `init_std` | `RZ + Z` ($s=0$ / $s\sim\mathcal{N}$) | `RZ + X` ($s=0$ / $s\sim\mathcal{N}$) | `RZ + Y` 共轭 ($s=0$ / $s\sim\mathcal{N}$) | `RY + Z` ($s=0$ / $s\sim\mathcal{N}$) |
| --- | --- | --- | --- | --- |
| `0.00` | `0.000e+00` / `2.056e-09` | `0.000e+00` / `4.125e-02` | **`1.140e-01`** / **`1.135e-01`** | `7.951e-01` / `8.389e-01` |
| `0.05`（旧版默认） | `1.837e-03` / `5.200e-03` | `4.276e-02` / `4.063e-02` | **`1.084e-01`** / **`1.093e-01`** | `7.998e-01` / `8.476e-01` |
| `0.50` | `8.608e-02` / `9.492e-02` | `1.368e-01` / `1.302e-01` | `1.242e-01` / `1.185e-01` | `5.024e-01` / `5.030e-01` |

`qml_lab/recurrent.py` 已修复默认配置为 `RZ+Y (init_std=1.0)`、修正 `generate()` 重复喂入与 `temperature` 采样及 `detach_state=True` 梯度截断。

---

## 二、Regime III：量子原生序列与自注意力基准

### 2.1 轨道 1 & 2：MPS 玻恩机与 L2 幺正 SSM（[`docs/quantum_native_seq_metrics.json`](docs/quantum_native_seq_metrics.json)）

| 轨道与模型 | 参数量 | ID 指标 | OOD 长度外推指标 | 配对检验（Holm 校正） |
| --- | ---: | --- | --- | --- |
| **Periodic-4**（训 $T\in\{4,6,8\} \to 14$）`quantum_mps_complex_chi2` | **24** | `KL=0.0729` / 禁态 `0.048` | **`KL=0.0949`** / 禁态 **`0.057`** | vs 24p 非负 $\chi=3$ (`0.3929`): $p_{\text{Holm}}=0.0247$ ✓; vs RNN (`0.5144`): $p_{\text{Holm}}=0.00573$ ✓ |
| **Dyck-1**（零重叠 $T=14 \to 16$）`quantum_mps_chi6` | **84** | **`NLL=0.4415`** / 合法 `0.939` | `NLL=0.4704` / 合法 **`0.856`** | vs 84p 非负 (`0.4732`, `0.592`): `NLL_ID` $p_{\text{Holm}}=0.000382$ ✓; vs RNN (`0.224`): $p_{\text{Holm}}=0.0144$ ✓ |
| **$S_3$ 追踪**（$T=6 \to 16$）`l2_quantum_unitary_ssm`（0 MLP） | 300 | **`1.000 ± 0.000`** | **`1.000 ± 0.000`** ($T=12, 16$) | 持平 $SO(9)$ (`1.000`); 击败 RNN (`0.270`, $0.000912$ ✓)、L1 (`0.291`, $3.32\times10^{-5}$ ✓)、L0 (`0.156`) |
| **延迟回忆**（`pad=4 -> 8`）L2 vs 修复 L1 (`RY+Z`/`RZ+Y`) vs 旧 L1 | 344 / 278 | **`1.000`** / `0.900` / `0.750` / `0.500` | **`1.000`** / `0.700` / `0.550` / `0.400` | L2 vs L0 (`0.250`): `+0.750`, $p_{\text{Holm}}=0.0$ ✓；$SO(9)$ OOD `0.950` |

### 2.2 轨道 3：量子自注意力三范式基准（KV 联想回忆 $T=6 \to 10$，[`docs/quantum_attention_metrics.json`](docs/quantum_attention_metrics.json)）

| 模型臂（参数严格匹配 `924–946`） | 总参 (量子) | Test-ID ($T=6$) | Test-OOD ($T=10$) | 关键物理开销 / 校验 |
| --- | ---: | ---: | ---: | --- |
| Regime I `classical_dot_attn` / `classical_overlap_attn` | 940 / 941 (0) | `0.9630` / `1.0000` | `0.9710` / `1.0000` | 经典 Softmax 基线（`Overlap` 为 SWAP-Test 解析等价替身） |
| Regime I `classical_linear_mixer`（无 Softmax） | 938 (0) | `1.0000 ± 0.0000` | `0.9950 ± 0.0112` | 公平线性叠加基线（与 Quixer `PREPARE` 同权重机制） |
| Regime II `quantum_swap_test_attn`（5q `CSWAP`） | 941 (0) | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | CSWAP 校验 `7.77e-16`；$S=100..2000$ OOD `0.997..0.999` (`MAE 0.0737..0.0164`，完整因果 `5.5k..110k` shots) |
| **Regime III `quantum_quixer_lcu_qsvt`（0 Softmax / 0 MLP）** | **946 (946)** | **`0.9910 ± 0.0074`** | **`0.9910 ± 0.0124`** | 6q `PREPARE/SELECT/PREPARE†+PCPhase` 校验 `<3.7e-7`；后选择 $p_{\text{succ}} = 0.423 / 0.395$（~1.6× AA） |
| Regime III `quantum_quixer_uniform_lcu`（均匀消融） | 924 (924) | `0.9870 ± 0.0057` | `0.8300 ± 0.0894` | $p_{\text{succ}}=0.247 / 0.434$；证明学习 `PREPARE` 抑制 OOD 稀释（`+0.1610`, $p=0.0209$） |

> **诚实硬件与 FTQC 边界**：1D MPS 为经典多项式 $O(TV\chi^3)$ 的量子启发干涉压缩优势；SWAP-Test 解析等价经典且受 $O(T^2 S)$ 测量墙制约；将 `QuantumUnitarySSM` 或 `QuantumQuixerAttention` 扩展至指数维希尔伯特空间与长序列均依赖容错量子计算（FTQC）、振幅放大 $O(1/\sqrt{p_{\text{succ}}})$ 与 QRAM，不能在当前经典模拟器上声称指数级实用加速。

---

## 三、目录结构与快速复现

- `qml_lab/`：`quantum.py`（30-Pauli / 3 QWC）、`models.py`（`ClassicalTrigBilinearLayer`, `parameter_breakdown`）、`recurrent.py`（L1 修复版）、`born_machine.py`、`quantum_ssm.py`、`quantum_attention.py`、`arithmetic.py`、`data.py`。
- `experiments/`：`state_channel_probe.py`、`matched_task2.py`、`benchmark_quantum_native_seq.py`、`benchmark_quantum_attention.py`、`plot_report_figures.py`、`train_addition.py`。
- `tests/smoke_test.py`：16 项离线契约与回归测试（含 QWC 计数、CSWAP `<1e-10`、LCU+QSVT `<1e-6`、POVM 完备性及 `--quick` CLI）。

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -v tests/smoke_test.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/matched_task2.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/benchmark_quantum_native_seq.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/benchmark_quantum_attention.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/plot_report_figures.py
```

*注：CPU 复现建议设 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` 且 `--workers <= 4`；各脚本均支持 `--quick` 冒烟（写入 `results/`，不覆盖 `docs/`）。*
