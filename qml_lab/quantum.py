# -*- coding: utf-8 -*-
"""可微分的量子变分线路层（PennyLane + PyTorch）。"""
from __future__ import annotations

import pennylane as qml
import torch
from torch import nn


def n_quantum_params(n_qubits: int, n_layers: int) -> int:
    """StronglyEntanglingLayers 每个量子比特、每层有 3 个旋转角。"""
    return n_layers * n_qubits * 3


class _Float32VQC(nn.Module):
    """薄封装：统一输入/输出为 float32；cast 是可微算子，不破坏梯度图。"""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer  # 注册为子模块，参数可被 model.parameters() 收集

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        out = self.layer(x)
        out = out.to(torch.float32)  # TorchLayer 常返回 float64
        # 少数 PennyLane 版本广播时返回 (n_qubits, B)，纠正为契约要求的 (B, n_qubits)
        if out.dim() == 2 and out.shape[0] != x.shape[0] and out.shape[1] == x.shape[0]:
            out = out.transpose(0, 1)
        return out


def make_vqc_layer(
    n_qubits: int = 4,
    n_layers: int = 2,
    diff_method: str = "backprop",
    shots: int | None = None,
) -> nn.Module:
    """构造线路层：输入 (B, n_qubits)，输出 (B, n_qubits) 的 PauliZ 期望值。"""
    analytic = shots is None or shots == 0
    # 有限 shots 是采样估计，不支持解析 backprop，改用 parameter-shift
    if not analytic and diff_method == "backprop":
        diff_method = "parameter-shift"
        print("[quantum] 有限 shots 不支持 backprop，diff_method 已自动切换为 parameter-shift")

    # 解析模拟不传 shots 参数；有限采样才把 shots 交给设备
    if analytic:
        dev = qml.device("default.qubit", wires=n_qubits)
    else:
        dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    # 新版 PennyLane 默认 interface="auto"，TorchLayer 会按 torch 张量自动推断接口；
    # 旧版若强制要求 interface="torch"，回退显式声明以兼容。
    try:
        qnode = qml.qnode(dev, diff_method=diff_method)(circuit)
        layer = qml.qnn.TorchLayer(qnode, weight_shapes)
    except Exception:
        qnode = qml.qnode(dev, interface="torch", diff_method=diff_method)(circuit)
        layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    layer = layer.to(torch.float32)  # 统一权重精度，避免与 float32 输入混用
    return _Float32VQC(layer)
