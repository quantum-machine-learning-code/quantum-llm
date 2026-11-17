"""L1 状态通道的**解析/数值梯度诊断**（不含训练，不是效果结论）。

背景：`qml_lab/recurrent.py` 把上一步测量期望以 ``RZ`` 相位重上传，
读出却只有 ``expval(PauliZ)``。在 Z 基下，局部 RZ 只改变相对相位，
**不改变振幅模方**，因此纯 Z 读出原则上观测不到它；只有后面
`StronglyEntanglingLayers` 的非对易旋转能把相位转成可测的振幅。
纠缠层权重按 ``init_std=0.05`` 初始化时接近恒等，这条转换通路可能很弱。

Pennylane 技能的失败检查明确写着：「零/平坦梯度可能来自未使用的参数、
对称性、饱和编码或微分失败，它本身不能诊断 barren plateau」。
`recurrent_pilot.py` 观察到量子臂 P1 前缀敏感度 `1.117e-3`
（经典正对照 `3.818e-2`）、延迟回忆反馈增益 0；本脚本把
「状态通路在初始化附近是否被结构性压制」变成可复现的定量问题。

三种配置的单一变量是对照：

* ``RZ + expval(Z)``：L1 现状；
* ``RZ + expval(X)``：同状态编码，只换读出基（相位对 X 期望直接可见）；
* ``RY + expval(Z)``：同读出，只把状态改成与输入同轴的加性角度（对照上界）。

**边界（必须随结果一起读）**：这是随机输入上的解析梯度，不是训练梯度；
它不能证明「量子循环不能记忆」，也不能证明「换读出就一定学得会」。
它只给出一个可检验的候选原因，用于决定下一步先改读出还是先延长训练。
§16 的结论「功能有效性未证明」不因本诊断改变。

用法：
    python experiments/state_channel_probe.py                # 写 docs/state_channel_probe.json
    python experiments/state_channel_probe.py --out <path>   # 隔离输出（冒烟用）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import pennylane as qml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT_NAME = "state_channel_probe.json"

Q = 4                      # 与 recurrent.py / pilot 的 n_qubits 一致
L = 1                      # StronglyEntanglingLayers 层数
STDS = [0.0, 0.05, 0.2, 0.5]   # 0.05 是 QuantumRecurrentLM 的默认 init_std
SEED = 20261117

CONFIGS = {
    "RZ_feedback_Z_readout": ("RZ", "Z"),
    "RZ_feedback_X_readout": ("RZ", "X"),
    "RY_feedback_Z_readout": ("RY", "Z"),
}


def make_qnode(feedback_gate: str, readout: str, n_qubits: int = Q):
    """与 recurrent.py 同构的批量 QNode；仅状态门与读出基可切换。"""
    dev = qml.device("default.qubit", wires=n_qubits)
    gate = qml.RZ if feedback_gate == "RZ" else qml.RY
    obs = qml.PauliZ if readout == "Z" else qml.PauliX

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x_ang: torch.Tensor, s_ang: torch.Tensor,
                weights: torch.Tensor):
        qml.AngleEmbedding(x_ang, wires=range(n_qubits), rotation="Y")
        for i in range(n_qubits):
            gate(s_ang[:, i], wires=i)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(obs(i)) for i in range(n_qubits)]

    return circuit


def sensitivity(circuit, x: torch.Tensor, s0: torch.Tensor,
                w: torch.Tensor) -> dict:
    """随机输入上 ``|∂(Σ_i expval_i)/∂s|`` 的均值与 RMS。"""
    s = s0.clone().requires_grad_(True)
    out = circuit(x, s, w)
    torch.stack(list(out), dim=-1).sum().backward()
    g = s.grad.detach()
    return {
        "mean_abs_grad": float(g.abs().mean()),
        "rms_grad": float(g.pow(2).mean().sqrt()),
        "max_abs_grad": float(g.abs().max()),
    }


def run_probe(n_inputs: int = 8) -> dict:
    """对每组纠缠层权重尺度，测三种配置的敏感度并返回结构化结果。"""
    torch.manual_seed(SEED)
    x = (torch.rand(n_inputs, Q) * 2 - 1) * (np.pi / 2)   # 与 tanh*π/2 同范围
    s0 = torch.randn(n_inputs, Q) * 0.1
    circuits = {name: make_qnode(g, r) for name, (g, r) in CONFIGS.items()}

    rows = []
    for std in STDS:
        torch.manual_seed(SEED + int(std * 1000) + 1)
        w = torch.randn(L, Q, 3) * std
        row = {"init_std": std, "configs": {}}
        for name, circuit in circuits.items():
            row["configs"][name] = sensitivity(circuit, x, s0, w)
        rz_z = row["configs"]["RZ_feedback_Z_readout"]["mean_abs_grad"]
        rz_x = row["configs"]["RZ_feedback_X_readout"]["mean_abs_grad"]
        ry_z = row["configs"]["RY_feedback_Z_readout"]["mean_abs_grad"]
        row["ratio_rz_z_over_ry_z"] = float(rz_z / ry_z) if ry_z else None
        row["ratio_rz_x_over_rz_z"] = float(rz_x / rz_z) if rz_z else None
        rows.append(row)

    zero = rows[0]["configs"]["RZ_feedback_Z_readout"]["mean_abs_grad"]
    default = next(r for r in rows if r["init_std"] == 0.05)
    return {
        "scope": "解析/数值梯度诊断；无训练、无准确率、无显著性检验",
        "question": "L1 的状态相位反馈在 Z 读出下是否被结构性压制？",
        "config": {
            "n_qubits": Q, "n_layers": L, "n_random_inputs": n_inputs,
            "input_range": "uniform(-π/2, π/2)，与 recurrent.py 的 tanh 编码一致",
            "state_range": "N(0, 0.1)", "seed": SEED,
            "weight_scales": STDS,
            "default_init_std_in_recurrent_py": 0.05,
        },
        "configurations": {
            "RZ_feedback_Z_readout": "L1 现状：RZ 相位反馈 + expval(Z)",
            "RZ_feedback_X_readout": "只换读出基到 expval(X)，状态编码不变",
            "RY_feedback_Z_readout": "只把状态门换成 RY（与输入同轴），读出不变",
        },
        "measurements": rows,
        "findings": [
            f"纠缠层权重为 0 时，RZ+Z 的 |∂Z/∂s| = {zero:.3e}（数值残差级），"
            "符合「Z 基下局部相位不改变振幅模方」的解析预期："
            "相位只能靠后续非对易旋转转成可测振幅。",
            "默认初始化 init_std=0.05 下，RZ+Z 相对 RY+Z 被压制 "
            f"{1 / default['ratio_rz_z_over_ry_z']:.0f} 倍"
            f"（{default['configs']['RZ_feedback_Z_readout']['mean_abs_grad']:.3e} vs "
            f"{default['configs']['RY_feedback_Z_readout']['mean_abs_grad']:.3e}）；"
            "这与 pilot 里量子臂 P1 前缀敏感度 1.117e-3 远小于经典 3.818e-2 同向。",
            "同状态下把读出换成 expval(X)，即使权重为 0 也能直接看到相位，"
            "说明「改读出基」比「先延长训练」是更便宜、可单变量检验的下一步。",
        ],
        "interpretation": (
            "这给出 §16「延迟回忆反馈增益 0」的一个**可检验候选机制**："
            "状态通路在默认初始化附近接近死通道，120 步短训练可能根本来不及把它打开。"
            "但它不改变 §16 的结论——功能有效性仍未证明；"
            "要验证这个机制，需要重训练做读出基 / 状态门的单变量 A/B，"
            "而不是靠本诊断下结论。"
        ),
        "claims_forbidden": [
            "量子循环不能记忆 / 经典更好",
            "换读出基就能学会延迟回忆（未重训验证）",
            "任何准确率、CI、显著性或 ≥5 seed 的结论",
            "把解析梯度当成训练梯度或 barren plateau 证据",
        ],
        "reproduce": "python experiments/state_channel_probe.py",
        "related": {
            "pilot": "docs/recurrent_pilot.json",
            "review": "docs/REVIEW.md §16",
            "circuit": "qml_lab/recurrent.py:make_recurrent_qnode",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="输出 JSON 路径，默认 docs/")
    ap.add_argument("--n-inputs", type=int, default=8)
    args = ap.parse_args()
    report = run_probe(args.n_inputs)
    out = Path(args.out) if args.out else DOCS / OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    for row in report["measurements"]:
        c = row["configs"]
        print(f"  init_std={row['init_std']:.2f}  "
              f"RZ+Z={c['RZ_feedback_Z_readout']['mean_abs_grad']:.3e}  "
              f"RZ+X={c['RZ_feedback_X_readout']['mean_abs_grad']:.3e}  "
              f"RY+Z={c['RY_feedback_Z_readout']['mean_abs_grad']:.3e}",
              flush=True)
    print(f"  written: {out}", flush=True)


if __name__ == "__main__":
    main()
