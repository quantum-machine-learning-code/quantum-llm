# 开发备注（quantum-llm）

主 `README.md` 给仓库读者看；本文件记录只在开发机上成立的环境细节。路径用占位符表示，不含凭证。

## 已验证的环境组合

| 项 | 值 |
| --- | --- |
| Python | 3.13（Windows 侧） |
| PennyLane | 0.45.1 |
| torch | 2.15.0（CPU 版） |
| 模拟器 | `default.qubit`，解析 `backprop`，无真机 |

本仓库**不需要** torchvision 或 scikit-learn：语料是内置的自写英文文本，全部实验离线可跑。

## 1. pip 源问题

全局 pip 配置指向清华镜像，但该环境里连 `torch` 都取不到。改用阿里云镜像 + 官方 PyPI 兜底即可：

```bash
python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://pypi.org/simple --timeout 60 --retries 3 \
    -r requirements.txt
```

## 2. Windows 与 WSL 双副本

开发副本在 Windows 工作区，git 仓库在 WSL Ubuntu 的 home 目录，靠复制同步。同步时排除 `.venv/` 与 `results/`：

```bash
rsync -a --exclude '.venv/' --exclude 'results/' \
    /mnt/c/Users/<WIN_USER>/WorkBuddy/quantum-machine-learning-code/<repo>/ \
    /home/<WSL_USER>/quantum-machine-learning-code/<repo>/
```

从 Windows 侧也可以通过 UNC 路径 `\\wsl$\Ubuntu\home\<WSL_USER>\...` 直接读写 WSL 文件（实测读写正常），不必启动 WSL 进程。但要在 WSL **内部执行**命令仍需 `wsl.exe`；该程序若被安全策略「程序黑名单」拦截，需使用者手动在「安全中心 → 命令安全 → 程序黑名单」移除。

## 3. 复现检查清单

```bash
python -m venv .venv
# 按第 1 节换源安装 requirements.txt
.venv/Scripts/python.exe tests/smoke_test.py                     # 或 .venv/bin/python
.venv/Scripts/python.exe experiments/train_llm.py --quick
.venv/Scripts/python.exe experiments/train_llm.py --steps 200 --seed 0
```

冒烟测试全程离线。若失败，优先检查 PennyLane 版本：`qml.qnn.TorchLayer` 的 QNode 接口、batch broadcasting、`TinyLM` 里 transform 输出的 dtype 转换（float64 → float32，且必须保留梯度）在 0.45 下已验证可用，升级大版本时需重新确认。

## 4. 关于「量子 LLM」这个名字

仓库名沿用用户指定的方向称呼，但实现是字符级玩具模型 + 一层可微分量子电路。README 开头与输出 JSON 的 `caveat` 字段都明确写了这一点；引用本仓库的结果时请不要把它描述成「量子大语言模型」。
