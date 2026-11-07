# 开发备注（quantum-llm）

主 `README.md` 与 `docs/REVIEW.md` 面向仓库读者；本文件记录开发与复现环境细节。路径用占位符表示，不含凭证。

## 已验证的环境组合

| 项 | 值 |
| --- | --- |
| Python | 3.13（Linux / WSL & Windows 侧均验证） |
| PennyLane | 0.45.1 |
| torch | 2.15.0（CPU 版） |
| numpy / scipy / scikit-learn | 2.5.4 / 1.18.1 / 1.9.1 |
| 模拟器 | `default.qubit`，解析 `backprop`，无真机 |

本仓库不需要 `torchvision`：训练语料与留出验证语料均内置在 `qml_lab/data.py` 中，全部实验完全离线可跑。`scipy` 与 `scikit-learn` 用于多随机种子 Student-$t$ 置信区间计算与特征分析。

## 1. pip 源问题

若默认镜像源缺少新版 `torch` CPU wheel，可使用官方 PyTorch CPU 源配合 PyPI 安装：

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt
```

## 2. 复现检查清单

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 1. 离线契约测试（验证末字符采样、条件熵下界、48维读出、因果零泄漏、权重对齐、确定性评估）
.venv/bin/python tests/smoke_test.py

# 2. 快速冒烟运行（仅写 results/，不覆盖 docs/）
.venv/bin/python experiments/train_llm.py --quick
.venv/bin/python experiments/diagnose_llm.py --quick

# 3. 正式受控实验与 5-Seed 诊断
.venv/bin/python experiments/train_llm.py --steps 200 --seed 0
.venv/bin/python experiments/diagnose_llm.py --steps 300 --seeds 0 1 2 3 4 --workers 12
```

## 3. 关键工程不变量

1. **受控初始化对齐**：`RichQuantumLM` 在构造时先初始化前端（`embed` 与 `causal_encoder`），随后快照 RNG 状态以分别派生 `head` 与 `transform`。不要随意调整 `RichQuantumLM.__init__` 内的子模块构造顺序，否则会破坏 `diagnose_llm.py` 与 `smoke_test.py` 中的跨组权重 SHA256 一致性断言。
2. **全语料确定性评估**：报告任何模型 loss / 困惑度时，必须使用 `evaluate_lm(model, data_tensor)` 返回的 `eval_loss` / `eval_perplexity`（及留出集 `val_loss`），不得使用单步随机 mini-batch 的 `final_loss`。
3. **关于「量子 LLM」名称**：本仓库实现的是字符级玩具自回归模型 + 可微分量子电路层，README 开头、`REVIEW.md` 与输出 JSON 的 `caveat` 字段均已明确声明边界。
