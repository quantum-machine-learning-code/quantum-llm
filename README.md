# 量子 LLM · 玩具级量子增强语言模型

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.45-9b59b6)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Toy scale](https://img.shields.io/badge/scale-toy%20%2F%20not%20an%20LLM-orange)

> **先把边界说清楚：这不是能对话的大语言模型。** 本仓库把参数化量子电路当作**一个可微分层**嵌进字符级自回归模型，验证「量子层能不能被训练起来」，并与同规模经典层对比 loss / 困惑度 / 耗时。

当前硬件与模拟器条件下，不存在可行的十亿参数级量子 LLM：状态向量维度是 2ⁿ，每多一个 qubit 内存翻倍。这里做的是**玩具级 QNLP 可行性验证**，词表是字符、语料 1668 字符、序列长 16、4 qubits，生成的是短文本。仓库沿用「量子 LLM」这个名字记录研究方向，不代表它是 LLM。

姊妹仓库 [quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist) 做量子手写数字识别；两个仓库代码各自自包含，可独立 clone 与运行。

## 快速结论（实跑结果）

Python 3.13 / PennyLane 0.45.1 / torch 2.15 CPU，`seed=0`，单次运行，200 步。复现命令见[快速开始](#快速开始)。

| 模型 | 参数量 | 最终 loss | 困惑度 | 训练耗时 |
| --- | --- | --- | --- | --- |
| 量子层 | 291 | **2.5403** | 12.68 | 5.37 s |
| 经典层（Linear + Tanh） | 299 | **2.5083** | 12.28 | 0.42 s |

**量子层略逊，慢约 13 倍。** 两组数字都不代表模型质量好 —— 200 步只够学到局部字符模式；本实验的产出是「量子层可训练 + 一份公平的对照度量」。完整解读见[怎么读这些数字](#怎么读这些数字)。

## 能做什么，不能做什么

- **不是 LLM**：字符级词表 31、序列 16、语料 1668 字符，没有注意力机制、没有预训练、不能对话。贪心生成的样本是 `A qubit can hold the the the the ...`。
- **模拟器的内存墙**：`default.qubit` 状态向量 2ⁿ，4–12 qubits 是笔记本舒适区；把 qubits 推到 20+ 需要 GPU 模拟器，但这也不会让它变成 LLM。
- **不宣称量子优势**：本仓库的目的是给出可复现的对照实验与度量，不是证明量子更好。
- **不需要真机凭证**：纯模拟无需 IBM Quantum 等账号。
- **不需要 MNIST / torchvision / scikit-learn**：语料是内置的自写文本，完全离线。

## 目录结构

```
quantum-llm/
├── README.md
├── LICENSE                     # MIT
├── requirements.txt
├── qml_lab/
│   ├── utils.py                # 随机种子、路径、JSON、参数量、终端表格
│   ├── data.py                 # 内置字符语料、char↔id 词表、可复现 batch 生成器
│   ├── quantum.py              # PennyLane 设备、QNode、AngleEmbedding + ansatz
│   ├── models.py               # TinyLM：Embedding -> 可微变换层 -> 词表头
│   ├── train.py                # 自回归训练循环（loss / ppl / 耗时）
│   └── viz.py                  # 指标图绘制
├── experiments/train_llm.py    # 实验入口（含贪心生成）
├── tests/smoke_test.py         # 离线冒烟测试
├── docs/DEV_NOTES.md           # 本机环境备注
├── docs/figures/               # 本次实测对比图（已归档）
└── results/                    # 运行产物（gitignore）
```

## 安装

要求 Python 3.11+，纯 CPU 即可运行默认配置，依赖比图像实验少（不需要 torchvision / scikit-learn）。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows 下把 `.venv/bin/python` 换成 `.venv\Scripts\python.exe`、路径改用反斜杠。

## 快速开始

```bash
.venv/bin/python tests/smoke_test.py
.venv/bin/python experiments/train_llm.py --quick
```

`--quick` 把 qubits 降到 2、只跑 3 步，用于验证流程跑通，**不用于报告结论**。

下面这条正是本 README 实测结果的复现命令：

```bash
.venv/bin/python experiments/train_llm.py \
    --qubits 4 --layers 1 --seq-len 16 --batch-size 8 --steps 200 --seed 0
```

想换语料用 `--corpus path/to/text.txt`（任意 UTF-8 纯文本，词表自动按字符构建）。全部产物写入 `results/`。

## 实验设计

- **词表**：字符级 `char ↔ id`，基于 `--corpus`（默认内置 1668 字符自写英文语料）构建，实测词表 31。
- **序列**：固定长度 `seq-len` 的字符窗口，预测下一个字符，loss 为词表交叉熵。batch 生成器用固定 seed 的 `torch.Generator` 保证可复现。
- **量子层位置**：`Embedding(vocab, qubits)` → 量子层（参数化量子电路）→ `Linear(qubits, vocab)` 输出词表 logits。对照组把中间换成 `Linear + Tanh`，其余结构与超参一致。
- **量子电路**：`AngleEmbedding(RY)` → `StronglyEntanglingLayers` → 每个 qubit 的 `PauliZ` 期望，输出仍是 `n_qubits` 维实数向量，因此可以直接接线性头。
- **对比方式**：同一 `steps / lr / batch-size / seed` 下分别训练（各用独立 seed 的 batch 生成器），对比 loss、困惑度（`exp(loss)`，clamp 防溢出）和耗时。
- **生成样本**：训练后用贪心解码生成 120 个字符写入 `results/lm_sample.txt`，仅用于直观检查是否学到字符分布。
- **采样与求导**：`--shots 0` 解析模拟（默认）；正整数启用有限采样并自动切 `parameter-shift`。

## 参数速查

> 若与脚本实现不一致，以脚本内 `argparse` 默认值为准。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--qubits` | `4` | qubit 数，也等于 Embedding 输出维 |
| `--layers` | `1` | `StronglyEntanglingLayers` 层数 |
| `--seq-len` | `16` | 自回归窗口长度（字符数） |
| `--batch-size` | `8` | 批大小 |
| `--steps` | `200` | 训练步数 |
| `--lr` | `0.01` | 学习率 |
| `--seed` | `0` | 随机种子 |
| `--corpus` | 内置示例语料 | 自定义 UTF-8 纯文本语料 |
| `--diff-method` | `backprop` | PennyLane 求导方式（解析模拟下） |
| `--shots` | `0` | `0` 解析模拟；正整数为有限采样 |
| `--out` | `results` | 指标、图片、生成文本的输出目录 |
| `--quick` | 关闭 | 极小规模验证 |

## 实测结果与复现

配置：内置 1668 字符语料、词表 31、序列长 16、batch 8、200 steps、4 qubits / 1 层。

![量子层与经典层的训练 loss 曲线](docs/figures/lm_loss.png)

困惑度与耗时对比见 [`docs/figures/lm_perplexity.png`](docs/figures/lm_perplexity.png) 与 [`docs/figures/lm_time.png`](docs/figures/lm_time.png)。

### 怎么读这些数字

- 量子层**略逊于经典层**，且慢约 13 倍。loss 2.5 左右对应困惑度约 12，两组都只学到很局部的字符模式 —— 这是 200 步、字符级小语料的预期结果，不是模型能力的比较。
- 量子层慢是因为每个 token 位置都要跑一遍状态向量模拟；参数量（291 vs 299）已经基本对齐，所以差距主要来自计算方式而非容量。
- 准确的表述是：**在这组配置下，参数化量子电路可以作为可微分层接入自回归语言模型并正常收敛，但训练代价明显更高，且没有带来更好的 loss。**
- 要更可靠的结论需要：多 `--seed` 重复取均值、更长训练、更丰富的编码与量子层位置、以及把「量子层」与「等 FLOPs 的经典层」对齐。

### 运行产物

`results/` 被 `.gitignore` 忽略，以下均为运行后本地生成；`docs/figures/` 归档了本次实测的对比图。

| 文件 | 内容 |
| --- | --- |
| `results/lm_metrics.json` | 两组 loss / 困惑度 / 耗时、`config`、`vocab_size`、`caveat` |
| `results/lm_loss.png` / `lm_perplexity.png` / `lm_time.png` | loss 曲线、困惑度曲线、耗时对比 |
| `results/lm_sample.txt` | 贪心解码生成的样本文本 |

## 技术栈与下一步

**PennyLane**（可微分量子电路）+ **PyTorch**（混合训练），辅以 NumPy / Matplotlib；默认 `default.qubit` 状态向量模拟。

下一步：多 seed 重复与统计检验、等计算预算的经典对照、量子层放在不同位置（替代 attention 的一个头 / 门控）、amplitude encoding 与数据重上传、噪声模型与有限 `--shots`、更大的 qubits 与 Lightning GPU 后端。

## 相关仓库与许可

- 姊妹仓库：[quantum-mnist](https://github.com/quantum-machine-learning-code/quantum-mnist) · 量子手写数字识别
- 组织主页：[quantum-machine-learning-code](https://github.com/quantum-machine-learning-code)

本项目采用 [MIT License](LICENSE)。
