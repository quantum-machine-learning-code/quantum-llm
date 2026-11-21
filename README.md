# quantum-llm · 量子序列建模方法学审查与原生基准

[English](README_EN.md) | 简体中文

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Scope](https://img.shields.io/badge/scope-toy%20%2F%20methodological%20benchmark-orange)

> **边界声明**：本仓库**不是**可对话的大语言模型，而是纯 CPU 状态向量 / 张量网络模拟器上的受控方法学基准，系统区分 **Regime II（经典主干中嵌入小参数 VQC 瓶颈）** 与 **Regime III（量子原生序列建模：MPS 玻恩机 + 真正 L2 幺正状态空间模型）**。完整审查日志见 [`docs/REVIEW.md`](docs/REVIEW.md)，架构规范见 [`docs/ARCHITECTURE_PROPOSAL.md`](docs/ARCHITECTURE_PROPOSAL.md)，姊妹仓库见 [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist)。

---

## 核心研究问题 ↔ 实验 ↔ 结论一览

| 研究问题 (RQ) | 范式 / 脚本与归档 | 受控结论（5 Seeds，Holm-Bonferroni 校正） |
| --- | --- | --- |
| **RQ1**：模 112 组合泛化加法中，4q L0 VQC 能否在严格同口径下超越经典变换层？ | Regime II · `train_addition.py --matched` ([`matched_v4_metrics.json`](docs/matched_v4_metrics.json)) | **否**。量子 `0.3251 ± 0.0464` vs 经典 `0.2767 ± 0.0917`（$\Delta=+0.0485$，CI `[-0.0677, +0.1647]`，$p=0.311$）；任务可学性由共享循环先验驱动（`0.1835` vs 无先验 `0.0062`），量子参数仅占 `12/10230 (0.12%)`。 |
| **RQ2**：历史字符 LM 的量子过拟合间隙（`+0.3263`）是否为混杂伪影？对齐后 L0 VQC 能否超越等参三角替身？ | Regime II · `matched_task2.py` ([`matched_task2_metrics.json`](docs/matched_task2_metrics.json)) | **混杂主导 + 无优势**。历史 `scale=5.0 + ni=True` 单独拉大间隙 `+0.1249`（$p_{\text{Holm}}=0.0097$）；严格对齐后量子与 12 参 `ClassicalTrig` 验证集无差异（`2.4804` vs `2.4834`，$\Delta=-0.0030$，$p_{\text{Holm}}=0.895$），耗时却慢 **20.7×**（量子仅 `12/3646 = 0.33%`）。 |
| **RQ3**：旧版 L1 量子循环为何在延迟回忆上停在随机基线？ | L1 诊断 · `state_channel_probe.py` ([`state_channel_probe.json`](docs/state_channel_probe.json)) | **死通道缺陷**。真实初态 $s=0, w=0$ 时 `RZ+Z` 与 `RZ+X` 导数严格为 `0.000e+00`，唯独共轭 `RZ+Y`（`1.140e-01`）与 `RY+Z`（`7.951e-01`）保持 $O(1)$；修复后 L1 延迟回忆从 `0.500` 升至 `0.750`（`RZ+Y`）/ `0.900`（`RY+Z`）。 |
| **RQ4**：量子振幅相消干涉（MPS 玻恩机）能否在等参条件下击败禁用负振幅的经典非负 MPS 与 RNN？ | Regime III 轨道 1 · `benchmark_quantum_native_seq.py` ([`quantum_native_seq_metrics.json`](docs/quantum_native_seq_metrics.json)) | **是（表征/干涉优势）**。Periodic-4 上 24 参复数 $\chi=2$ MPS 在 OOD $T=14$ 的精确 KL 为 `0.0949` vs 等 24 参非负 $\chi=3$ MPS `0.3929`（$p_{\text{Holm}}=0.0247$）与 RNN `0.5144`（$p_{\text{Holm}}=0.00573$）；Dyck-1 上 84 参有符号 $\chi=6$ MPS `NLL_ID=0.4415` 显著优于等 84 参非负 MPS `0.4732`（$p_{\text{Holm}}=0.000382$）与 RNN OOD 合法率 `+63.2 pt`（$p_{\text{Holm}}=0.0144$）。 |
| **RQ5**：真正跨步不坍缩的 L2 幺正状态空间模型（`QuantumUnitarySSM`）能否实现 OOD 非阿贝尔群状态追踪？ | Regime III 轨道 2 · `benchmark_quantum_native_seq.py` ([`quantum_native_seq_metrics.json`](docs/quantum_native_seq_metrics.json)) | **是（相对 L0/L1/RNN），与同维经典 $SO(d)$ 持平**。$S_3$（训练 $T=6 \to$ OOD $T=16$）与延迟回忆（OOD `pad=8`）均达 **`1.000 ± 0.000`**（0 MLP 拐杖，PennyLane 校验误差 `<2.4e-7`），显著击败 L0（`0.156`）、L1（`0.291`）与 RNN（`0.270`），并与经典 `OrthogonalSSM`（`1.000`）一致。 |

---

## 一、Regime II：严格对齐的 L0 经典主干 + VQC 基准

### 1.1 课题二（独立字符语料，规范 30-Pauli / 3 QWC 读出）

历史 `independent_corpus.py` / `robustness_check.py` 存在未察觉混杂：量子臂固定 `scale=5.0, near_identity_init=True`，经典臂 `scale=1.0, near_identity_init=False`，且缺少三角特征经典替身。`experiments/matched_task2.py` 将 13 项配置逐项断言一致（统一可学习 `out_scale` 初值 `1.0`、默认初始化、`steps=400, lr=0.01, seeds=0..4`），并与 `quantum-mnist` 对齐的规范 30-Pauli 读出（`z+x+y+zzall+xxall+yyall`，**3 个 QWC 对易组**，旧版 48-Pauli 为 10 组）及 12 参 `ClassicalTrigBilinearLayer` 严格对照（[`docs/matched_task2_metrics.json`](docs/matched_task2_metrics.json)）：

| 臂 | 总参 (量子 / 经典变换) | QWC 组 | 初始读出 std | Train Loss | 独立 Val Loss | 泛化间隙 (Val−Train) | 最终 scale | 耗时 / seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `caus_q_rich30`（严格对齐） | `3,646` (**12** = `0.33%` / 0) | **3** | `0.3261` | `2.0992 ± 0.0074` | `2.4804 ± 0.0291` | `0.3812 ± 0.0354` | `1.57` | `31.6 s` |
| `caus_classical_tanh`（对齐） | `3,784` (0 / 150) | 0 | `0.3951` | `2.1435 ± 0.0115` | `2.4425 ± 0.0400` | `0.2990 ± 0.0399` | `0.93` | `1.2 s` |
| `caus_classical_trig`（等 12 参） | `3,646` (0 / **12**) | 0 | `0.3765` | `2.0498 ± 0.0213` | `2.4834 ± 0.0335` | `0.4335 ± 0.0275` | `1.30` | `1.5 s` |
| `diag_unmatched_q_scale5`（诊断） | `3,645` (12 / 0) | 3 | — | `2.0315 ± 0.0271` | `2.5376 ± 0.0361` | `0.5062 ± 0.0415` | `5.00` | `31.2 s` |

- **混杂归因证伪**：仅恢复历史 `scale=5.0 + ni=True` 就使同一量子线路的过拟合间隙上升 **`+0.1249`**（$p_{\text{Holm}}=0.009705$）、验证 loss 恶化 `+0.0572`（$p_{\text{Holm}}=0.0413$）。
- **对齐后无量子特性优势**：对齐后 `Q − Tanh` 验证 loss（`+0.0379`, $p_{\text{Holm}}=0.289$）与间隙（`+0.0822`, $p_{\text{Holm}}=0.095$）均不再显著；与精确 12 参 `ClassicalTrig` 相比，验证 loss 几乎完全一致（`Δ=-0.0030`, $p_{\text{Holm}}=0.895$），说明历史差异源自**一/二阶三角单项式基与固定 5× 缩放**，而非量子效应。

### 1.2 课题一（模 112 组合泛化加法，零 $k=4$ 重叠）

- **严格对齐 v4（[`docs/matched_v4_metrics.json`](docs/matched_v4_metrics.json)，5 seeds × 2000 步，`lr=0.003, scale_init=5.0`）**：量子 `0.3251 ± 0.0464` vs 经典 `0.2767 ± 0.0917`（两臂 5/5 seed 通过装置有效性；配对 $\Delta=+0.0485$，CI `[-0.0677, +0.1647]`，$p=0.311$ 不显著；仅 `generalization_gap` 差 `+0.0156` 显著，量子过拟合间隙更大）。
- **先验与 Grokking 边界**：三模式消融（[`embedding_ablation_metrics.json`](docs/embedding_ablation_metrics.json)）表明随机 embedding 停在 `0.0062`（≈ $1/112$），循环编码先验贡献 `+0.1773` / `+0.1400`。不加先验同超参 $10^5$ 步复验（[`grokking_no_prior.json`](docs/grokking_no_prior.json)）显示 `RICH wd=0` 处于记忆中段（`train 0.6565 / val 0.0076`）而 `MLP wd=1` 达 `val 0.9918`；早期 §12「架构缺陷」归因已在 §13 正式撤回。

---

## 二、L1 量子循环状态通道解析诊断

`experiments/state_channel_probe.py`（[`docs/state_channel_probe.json`](docs/state_channel_probe.json)）同时在随机扰动 $s \sim \mathcal{N}(0, 0.1)$ 与真实首步初态 $s = 0$ 上检验 4 种门/读出基组合的 $|\partial(\sum_i \langle O_i \rangle)/\partial s|$：

| 纠缠层 `init_std` | `RZ + Z` ($s=0$ / $s\sim\mathcal{N}$) | `RZ + X` ($s=0$ / $s\sim\mathcal{N}$) | `RZ + Y` 共轭 ($s=0$ / $s\sim\mathcal{N}$) | `RY + Z` ($s=0$ / $s\sim\mathcal{N}$) |
| --- | --- | --- | --- | --- |
| `0.00` | `0.000e+00` / `2.056e-09` | `0.000e+00` / `4.125e-02` | **`1.140e-01`** / **`1.135e-01`** | `7.951e-01` / `8.389e-01` |
| `0.05`（旧版默认） | `1.837e-03` / `5.200e-03` | `4.276e-02` / `4.063e-02` | **`1.084e-01`** / **`1.093e-01`** | `7.998e-01` / `8.476e-01` |
| `0.50` | `8.608e-02` / `9.492e-02` | `1.368e-01` / `1.302e-01` | `1.242e-01` / `1.185e-01` | `5.024e-01` / `5.030e-01` |

物理机制：`RY(x)|0⟩` 位于 Bloch 球 $XZ$ 平面，`RZ(s)` 首先将矢量推向 $Y$ 轴（$\langle Y \rangle = \sin x \sin s$），故 $s=0, w=0$ 时 $Z$ 与 $X$ 的一阶导数均恒为 0，唯独 `PauliY` 读出保持 $O(1)$ 灵敏度。`qml_lab/recurrent.py` 已修复默认配置为 `RZ+Y (init_std=1.0)`、修正 `generate()` 首尾 token 重复喂入与 `temperature` 采样、以及 `detach_state=True` 阻断末步头梯度的问题。

---

## 三、Regime III：量子原生序列建模基准

归档于 [`docs/quantum_native_seq_metrics.json`](docs/quantum_native_seq_metrics.json)（5 seeds，严格独立 Train/Val/Test + OOD 长度外推，Holm 校正）。

### 3.1 轨道 1：自回归张量网络玻恩机（`qml_lab/born_machine.py`）

以精确转移矩阵收缩计算玻恩概率 $P(x) = |\langle x|\psi\rangle|^2 / \langle\psi|\psi\rangle$（$\sum_x P(x) = 1$ 精确归一，支持精确祖先采样），与**单变量受控经典基线 `ClassicalNonNegMPS`**（相同拓扑、非负实数张量，禁用相消干涉，采用公平 `softplus^{-1}` 稀疏初始化）及 `ClassicalCausalRNNLM` 对照：

| 任务与模型 | 参数量 | ID 指标 | OOD 长度外推指标 | 相对基线配对检验（Holm 校正） |
| --- | --- | --- | --- | --- |
| **Periodic-4** ($P^* \propto \cos^2(\frac{\pi}{4}\sum x_t)$，训 $T\in\{4,6,8\}$) | | 精确 $\text{KL}_{\text{ID}}$ ($T=8$) / 禁态率 | 精确 $\text{KL}_{\text{OOD}}$ ($T=14$) / 禁态率 | |
| `quantum_mps_complex_chi2`（复振幅 $\chi=2$） | **24** | **`0.0729 ± 0.1526`** / **`0.048`** | **`0.0949 ± 0.1521`** / **`0.057`** | —（4/5 seed 达 `KL_ID ≤ 0.0066`、`forb_T14 ≤ 0.010`） |
| `classical_nonneg_mps_chi2`（同 $\chi=2$） | 12 | `0.3667 ± 0.0478` / `0.220` | `0.4527 ± 0.0739` / `0.255` | `Q − NonNeg(χ=2)` $\text{KL}_{\text{ID}} = -0.2938$ ($p_{\text{Holm}}=0.0192$ ✓) |
| `classical_nonneg_mps_chi3`（**等 24 参**，更高 $\chi=3$） | **24** | `0.3066 ± 0.0489` / `0.195` | `0.3929 ± 0.0505` / `0.248` | `Q − NonNeg(χ=3)` $\text{KL}_{\text{OOD}} = -0.2981$ ($p_{\text{Holm}}=0.0247$ ✓) |
| `classical_rnn_matched` | 35 | `0.4310 ± 0.0374` / `0.230` | `0.5144 ± 0.0854` / `0.249` | `Q − RNN` $\text{KL}_{\text{OOD}} = -0.4195$ ($p_{\text{Holm}}=0.00573$ ✓) |
| **Dyck-1 括号平衡**（零重叠 $T=14 \to 16$；Bayes `0.4330 / 0.4541`） | | Test-ID NLL / 合法率 | OOD $T=16$ NLL / 合法率 | |
| `quantum_mps_chi6`（有符号实振幅 $\chi=6$） | **84** | **`0.4415 ± 0.0070`** / **`0.939`** | **`0.4704 ± 0.0186`** / **`0.856`** | — |
| `classical_nonneg_mps_chi6`（同 $\chi=6$、同参） | **84** | `0.4732 ± 0.0046` / `0.650` | `0.5036 ± 0.0030` / `0.592` | `Q − NonNeg` $\text{NLL}_{\text{ID}} = -0.0317$ ($p_{\text{Holm}}=0.000382$ ✓) |
| `classical_rnn_matched` | 98 | `0.4741 ± 0.0158` / `0.718` | `0.5928 ± 0.0598` / `0.224` | `Q − RNN` 合法率 `+0.632` ($p_{\text{Holm}}=0.0144$ ✓) |

> **诚实边界**：1D MPS 精确收缩经典复杂度为 $O(TV\chi^3)$，因此本轨道验证的是**量子相消干涉带来的表征/压缩优势**（Perron-Frobenius 分离，且可编译为浅层序列量子电路），**不是**指数级量子硬件加速。

### 3.2 轨道 2：真正 L2 相干量子态循环 / 幺正 SSM（`qml_lab/quantum_ssm.py`）

`QuantumUnitarySSM` 维护跨步**不坍缩**的 3-qubit 相干隐态 $|\psi_t\rangle = U(x_t)|\psi_{t-1}\rangle \in \mathbb{C}^8$（由 1+2 体 Pauli 哈密顿量生成，$U(x_t) = e^{-iH(x_t)}$），仅通过完备 POVM（$\sum_c M_c = I_8$）做精确玻恩规则读出，**零经典 MLP 拐杖**，经 PennyLane `default.qubit` 逐元素校验（`max|Δψ|=1.21e-7`，`max|Δp|=2.38e-7`）：

| 模型 | $S_3$ 群词问题（参 / ID $T=6$ / OOD $T=12$ / **OOD $T=16$**，随机 `0.167`） | 确定性延迟回忆（参 / ID `pad=4` / **OOD `pad=8`**，随机 `0.250`） |
| --- | --- | --- |
| **`l2_quantum_unitary_ssm`**（真正 L2，0 MLP） | `300` / **`1.000 ± 0.000`** / **`1.000 ± 0.000`** / **`1.000 ± 0.000`** | `344` / **`1.000 ± 0.000`** / **`1.000 ± 0.000`** |
| `classical_orthogonal_ssm`（同维 $SO(9)$ 几何对照） | `270` / `1.000 ± 0.000` / `1.000 ± 0.000` / `1.000 ± 0.000` | `324` / `1.000 ± 0.000` / `0.950 ± 0.112` |
| `classical_rnn`（无约束经典 RNN） | `327` / `0.595 ± 0.134` / `0.321 ± 0.153` / `0.270 ± 0.155` | `228` / `0.900 ± 0.137` / `0.500 ± 0.468` |
| `l1_quantum_fixed_rzy`（修复后 L1 `RZ+Y`） | `278` / `0.390 ± 0.126` / `0.331 ± 0.070` / `0.291 ± 0.059` | `278` / `0.750 ± 0.306` / `0.550 ± 0.326` |
| `l1_quantum_fixed_ryz`（修复后 L1 `RY+Z`） | — | `278` / `0.900 ± 0.137` / `0.700 ± 0.209` |
| `l1_quantum_legacy_rzz`（旧版死通道 `RZ+Z`） | — | `278` / `0.500 ± 0.306` / `0.400 ± 0.137` |
| `l0_causal_conv`（固定窗口 $k=4$） | `334` / `0.180 ± 0.018` / `0.176 ± 0.016` / `0.156 ± 0.018` | `348` / `0.250 ± 0.000` / `0.250 ± 0.000` |

- **$S_3$ OOD $T=16$ 配对检验**：L2 显著击败 RNN（`+0.730`, $p_{\text{Holm}}=0.000912$）、L1（`+0.709`, $p_{\text{Holm}}=3.32\times10^{-5}$）与 L0（`+0.844`, $p_{\text{Holm}}=2.06\times10^{-7}$），并与经典 `OrthogonalSSM` 完全持平（$\Delta=0.000, p_{\text{Holm}}=1.0$）。
- **物理归因与 FTQC 边界**：在 $d_H=8$ 尺度上，100% OOD 外推来自**保范群表示（幺正/正交相干演化 vs 有耗散 RNN / 测量坍缩 L1）**；将 `QuantumUnitarySSM` 扩展至指数级希尔伯特空间 $2^n$ 或实现基于 **QSVT / 块编码的量子自注意力**均需要容错量子计算（FTQC）与 QRAM，**无法**在当前经典状态向量模拟器上声称指数级加速。

---

## 四、目录结构与快速复现

- `qml_lab/`：`quantum.py`（规范 30-Pauli / 3 QWC 读出与旧版 opt-in）、`models.py`（`RichQuantumLM`, `ClassicalTrigBilinearLayer`, `parameter_breakdown`）、`recurrent.py`（L1 修复版）、`born_machine.py`（Regime III 轨道 1）、`quantum_ssm.py`（Regime III 轨道 2）、`arithmetic.py`、`data.py`、`corpus_independent.py`。
- `experiments/`：`state_channel_probe.py`、`matched_task2.py`、`benchmark_quantum_native_seq.py`、`train_addition.py`、`grokking_no_prior.py`、`diagnose_llm.py`。
- `tests/smoke_test.py`：15 项离线契约与回归测试（含 QWC 计数、因果严格性、POVM 完备性、PennyLane 校验与 CLI 冒烟）。

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -v tests/smoke_test.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/state_channel_probe.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/matched_task2.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/benchmark_quantum_native_seq.py --workers 4
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python experiments/train_addition.py --matched --lr 0.003 --scale-init 5 --workers 4 --no-figure
```

*注：本机纯 CPU 复现建议固定 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` 且 `--workers <= 4`；各实验脚本均支持 `--quick` 秒级冒烟（产物写 `results/`，不覆盖 `docs/`）。*
