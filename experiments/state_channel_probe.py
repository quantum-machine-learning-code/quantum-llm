"""L1 状态通道的解析/数值梯度诊断（含 Z / X / Y 三基与实际初态 s=0 检查）。

背景：``qml_lab/recurrent.py`` 原版以 ``RZ`` 相位重上传状态，却仅用 ``expval(PauliZ)``
读出且默认 ``init_std=0.05``。因 ``RY(x)|0>`` 位于 Bloch 球 XZ 平面，施加 ``RZ(s)`` 后
Bloch 向量为 ``(sin(x)cos(s), sin(x)sin(s), cos(x))``：

* ``expval(Z)`` 对局部相位 ``s`` 在 ``w=0`` 时解析导数恒为 0；
* ``expval(X)`` 在实际初始状态 ``s=0``（``t=1``）且 ``w=0`` 时导数 ``-sin(x)sin(0) = 0`` 同样消失；
* **``expval(Y)``** 是 ``RZ`` 作用于 ``RY|0>`` 的共轭读出基：``d<Y>/ds|_{s=0} = sin(x) != 0``，
  即使 ``w=0, s=0`` 仍有 O(1) 一阶敏感度！

本脚本同时在 ``s=0``（真实 ``init_state``）与 ``s ~ N(0, 0.1)`` 两种状态下对比
``RZ+Z`` / ``RZ+X`` / ``RZ+Y`` / ``RY+Z`` 四种配置，验证解析机制。

用法：
    python experiments/state_channel_probe.py                # 写 docs/state_channel_probe.json
    python experiments/state_channel_probe.py --out <path>   # 隔离输出（冒烟用）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT_NAME = "state_channel_probe.json"

Q = 4
L = 1
STDS = [0.0, 0.05, 0.2, 0.5]
SEED = 20261117

CONFIGS = {
    "RZ_feedback_Z_readout": ("RZ", "Z"),
    "RZ_feedback_X_readout": ("RZ", "X"),
    "RZ_feedback_Y_readout": ("RZ", "Y"),
    "RY_feedback_Z_readout": ("RY", "Z"),
}

_OBS_MAP = {
    "Z": qml.PauliZ,
    "X": qml.PauliX,
    "Y": qml.PauliY,
}


def make_qnode(feedback_gate: str, readout: str, n_qubits: int = Q):
    """与 recurrent.py 同构的批量 QNode；可切换状态门 (RZ/RY) 与读出基 (Z/X/Y)。"""
    dev = qml.device("default.qubit", wires=n_qubits)
    gate = qml.RZ if feedback_gate == "RZ" else qml.RY
    obs = _OBS_MAP[readout]

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x_ang: torch.Tensor, s_ang: torch.Tensor, weights: torch.Tensor):
        qml.AngleEmbedding(x_ang, wires=range(n_qubits), rotation="Y")
        for i in range(n_qubits):
            gate(s_ang[:, i], wires=i)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(obs(i)) for i in range(n_qubits)]

    return circuit


def sensitivity(circuit, x: torch.Tensor, s0: torch.Tensor, w: torch.Tensor) -> dict:
    """随机输入上 ``|d(sum_i expval_i)/ds|`` 的均值、RMS 与最大值。"""
    s = s0.clone().requires_grad_(True)
    out = circuit(x, s, w)
    torch.stack(list(out), dim=-1).sum().backward()
    g = s.grad.detach()
    return {
        "mean_abs_grad": float(g.abs().mean()),
        "rms_grad": float(g.pow(2).mean().sqrt()),
        "max_abs_grad": float(g.abs().max()),
    }


def _sweep_regime(circuits: dict, x: torch.Tensor, s0: torch.Tensor) -> list[dict]:
    rows = []
    for std in STDS:
        torch.manual_seed(SEED + int(std * 1000) + 1)
        w = torch.randn(L, Q, 3) * std
        row: dict = {"init_std": std, "configs": {}}
        for name, circuit in circuits.items():
            row["configs"][name] = sensitivity(circuit, x, s0, w)
        rz_z = row["configs"]["RZ_feedback_Z_readout"]["mean_abs_grad"]
        rz_x = row["configs"]["RZ_feedback_X_readout"]["mean_abs_grad"]
        rz_y = row["configs"]["RZ_feedback_Y_readout"]["mean_abs_grad"]
        ry_z = row["configs"]["RY_feedback_Z_readout"]["mean_abs_grad"]
        row["ratio_rz_z_over_ry_z"] = float(rz_z / ry_z) if ry_z else None
        row["ratio_rz_x_over_rz_z"] = float(rz_x / rz_z) if rz_z > 1e-12 else None
        row["ratio_rz_y_over_rz_z"] = float(rz_y / rz_z) if rz_z > 1e-12 else None
        row["ratio_rz_y_over_rz_x"] = float(rz_y / rz_x) if rz_x > 1e-12 else None
        rows.append(row)
    return rows


def run_probe(n_inputs: int = 8) -> dict:
    """同时在 s~N(0,0.1) 与真实初态 s=0 上测四种配置的解析梯度敏感度。"""
    torch.manual_seed(SEED)
    x = (torch.rand(n_inputs, Q) * 2 - 1) * (np.pi / 2)
    s_rand = torch.randn(n_inputs, Q) * 0.1
    s_zero = torch.zeros(n_inputs, Q)
    circuits = {name: make_qnode(g, r) for name, (g, r) in CONFIGS.items()}

    rows_rand = _sweep_regime(circuits, x, s_rand)
    rows_zero = _sweep_regime(circuits, x, s_zero)

    z0_rand = rows_rand[0]["configs"]["RZ_feedback_Z_readout"]["mean_abs_grad"]
    z0_zero = rows_zero[0]["configs"]["RZ_feedback_Z_readout"]["mean_abs_grad"]
    x0_zero = rows_zero[0]["configs"]["RZ_feedback_X_readout"]["mean_abs_grad"]
    y0_zero = rows_zero[0]["configs"]["RZ_feedback_Y_readout"]["mean_abs_grad"]
    def_rand = next(r for r in rows_rand if r["init_std"] == 0.05)
    def_zero = next(r for r in rows_zero if r["init_std"] == 0.05)

    return {
        "scope": "解析/数值梯度诊断（含 Z/X/Y 三基与 s=0 真实初态）；无训练、无准确率结论",
        "question": "L1 的 RZ 相位反馈在 Z / X / Y 读出基与真实初态 s=0 下的敏感度对比",
        "config": {
            "n_qubits": Q,
            "n_layers": L,
            "n_random_inputs": n_inputs,
            "input_range": "uniform(-pi/2, pi/2)，与 recurrent.py 的 tanh 编码一致",
            "state_regimes": {"s_random": "N(0, 0.1)", "s_zero": "exact zeros (actual t=1 init_state)"},
            "seed": SEED,
            "weight_scales": STDS,
            "legacy_init_std": 0.05,
            "fixed_default_init_std": 1.0,
        },
        "configurations": {
            "RZ_feedback_Z_readout": "旧版 L1：RZ 相位反馈 + expval(Z)（死通道）",
            "RZ_feedback_X_readout": "只换到 expval(X)：s=0,w=0 时 d(cos s)/ds = 0 仍为零",
            "RZ_feedback_Y_readout": "共轭读出 expval(Y)：s=0,w=0 时 d(sin s)/ds = 1 保持 O(1) 敏感度",
            "RY_feedback_Z_readout": "同轴 RY 状态门 + expval(Z)（对照上界）",
        },
        "measurements": rows_rand,
        "measurements_at_init_zero": rows_zero,
        "findings": [
            f"在真实初始状态 s=0 且 w=0 时：RZ+Z = {z0_zero:.3e}、RZ+X = {x0_zero:.3e}（两者解析导数均恒为 0），"
            f"而共轭基 RZ+Y = {y0_zero:.3e}（一阶非零，验证 <Y>=sin(x)sin(s) 机制）。",
            f"在 s~N(0,0.1)、旧版 init_std=0.05 下：RZ+Z = {def_rand['configs']['RZ_feedback_Z_readout']['mean_abs_grad']:.3e} "
            f"相对 RY+Z ({def_rand['configs']['RY_feedback_Z_readout']['mean_abs_grad']:.3e}) 被压制 "
            f"{1 / def_rand['ratio_rz_z_over_ry_z']:.0f} 倍；RZ+Y ({def_rand['configs']['RZ_feedback_Y_readout']['mean_abs_grad']:.3e}) "
            f"比 RZ+Z 强 {def_rand['ratio_rz_y_over_rz_z']:.1f} 倍、比 RZ+X ({def_rand['configs']['RZ_feedback_X_readout']['mean_abs_grad']:.3e}) "
            f"强 {def_rand['ratio_rz_y_over_rz_x']:.1f} 倍。",
            f"在 s=0、init_std=0.05 下：RZ+Z = {def_zero['configs']['RZ_feedback_Z_readout']['mean_abs_grad']:.3e}、"
            f"RZ+X = {def_zero['configs']['RZ_feedback_X_readout']['mean_abs_grad']:.3e}、"
            f"RZ+Y = {def_zero['configs']['RZ_feedback_Y_readout']['mean_abs_grad']:.3e}；"
            f"此外 s~N(0,0.1), w=0 时 RZ+Z = {z0_rand:.3e}（仅数值舍入残差）。",
        ],
        "interpretation": (
            "旧版 state_channel_probe 遗漏了 PauliY 共轭读出基：RY(x)|0> 位于 XZ 平面，RZ(s) 绕 Z 轴旋转"
            "首先把 Bloch 矢量推向 Y 轴，因此 s=0 时 X 与 Z 均无一阶导数，唯独 Y 读出天然非零。"
            "已据此修复 qml_lab/recurrent.py 默认配置为 RZ+Y (init_std=1.0) 并支持 RY+Z 与 XYZ 读出；"
            "实际训练效果验证见 docs/quantum_native_seq_metrics.json 的确定性延迟回忆与 S_3 基准。"
        ),
        "claims_forbidden": [
            "量子循环不能记忆 / 经典更好",
            "仅靠解析梯度替代重训验证",
            "把解析梯度当成 barren plateau 证据",
        ],
        "reproduce": "python experiments/state_channel_probe.py",
        "related": {
            "pilot": "docs/recurrent_pilot.json",
            "native_benchmark": "docs/quantum_native_seq_metrics.json",
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
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for title, key in (("s~N(0,0.1)", "measurements"), ("s=0 (init)", "measurements_at_init_zero")):
        print(f"=== {title} ===")
        for row in report[key]:
            c = row["configs"]
            print(
                f"  std={row['init_std']:.2f}  "
                f"RZ+Z={c['RZ_feedback_Z_readout']['mean_abs_grad']:.3e}  "
                f"RZ+X={c['RZ_feedback_X_readout']['mean_abs_grad']:.3e}  "
                f"RZ+Y={c['RZ_feedback_Y_readout']['mean_abs_grad']:.3e}  "
                f"RY+Z={c['RY_feedback_Z_readout']['mean_abs_grad']:.3e}",
                flush=True,
            )
    print(f"  written: {out}", flush=True)


if __name__ == "__main__":
    main()
