# -*- coding: utf-8 -*-
"""可微分的量子变分线路层（PennyLane + PyTorch）：支持多可观测量读出与可训练数据重上传。"""
from __future__ import annotations

import pennylane as qml
import torch
from torch import nn

__all__ = [
    "n_quantum_params",
    "make_observables",
    "n_readout",
    "make_vqc_layer",
    "make_amplitude_layer",
    "make_rich_layer",
]


def n_quantum_params(
    n_qubits: int, n_layers: int, reupload: bool | str = False
) -> int:
    """StronglyEntanglingLayers 每个量子比特、每层有 3 个旋转角；trainable 重上传额外含每层每比特缩放系数。"""
    base = n_layers * n_qubits * 3
    if isinstance(reupload, str) and reupload.lower() == "trainable":
        return base + n_layers * n_qubits
    return base


class _Float32VQC(nn.Module):
    """薄封装：统一输入/输出为 float32；cast 是可微算子，不破坏梯度图。"""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer  # 注册为子模块，参数可被 model.parameters() 收集

    @property
    def weights(self) -> torch.nn.Parameter:
        return self.layer.weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        out = self.layer(x)
        out = out.to(torch.float32)  # TorchLayer 常返回 float64
        # 少数 PennyLane 版本广播时返回 (n_obs, B)，纠正为契约要求的 (B, n_obs)
        if out.dim() == 2 and out.shape[0] != x.shape[0] and out.shape[1] == x.shape[0]:
            out = out.transpose(0, 1)
        return out


def _apply_near_identity_init(layer: nn.Module, init_std: float = 0.05) -> None:
    """将量子旋转角初始化为近似恒等的小高斯分布 N(0, init_std^2)，抑制初始梯度发散。"""
    with torch.no_grad():
        # 从当前全局 RNG 采样与默认初始化相同元素数的正态随机数
        noise = torch.randn_like(layer.weights) * float(init_std)
        layer.weights.copy_(noise)


def make_vqc_layer(
    n_qubits: int = 4,
    n_layers: int = 2,
    diff_method: str = "backprop",
    shots: int | None = None,
    near_identity_init: bool = False,
    init_std: float = 0.05,
) -> nn.Module:
    """构造线路层：输入 (B, n_qubits)，输出 (B, n_qubits) 的 PauliZ 期望值。"""
    return make_rich_layer(
        n_qubits=n_qubits,
        n_layers=n_layers,
        encoding="angle",
        observables="z",
        reupload=False,
        diff_method=diff_method,
        shots=shots,
        near_identity_init=near_identity_init,
        init_std=init_std,
    )


def make_amplitude_layer(
    n_qubits: int = 10,
    n_layers: int = 2,
    diff_method: str = "backprop",
    shots: int | None = None,
    n_features: int = 784,
) -> nn.Module:
    """振幅编码层：输入 (B, n_features) 的 L2 归一化向量，输出 (B, n_qubits)。"""
    return make_rich_layer(
        n_qubits=n_qubits,
        n_layers=n_layers,
        encoding="amplitude",
        observables="z",
        n_features=n_features,
        reupload=False,
        diff_method=diff_method,
        shots=shots,
    )


_READOUT_PARTS = ("z", "x", "y", "zz", "zzall", "xxall", "yyall")


def make_observables(n_qubits: int, spec: str):
    """按 ``+`` 分隔的规格构造正交可观测测量表。

    * ``z`` / ``x`` / ``y`` 各贡献 ``n`` 个单比特 Pauli 期望；
    * ``zz`` 贡献 ``n`` 个相邻环 ``Z_i Z_{i+1}`` 关联；
    * ``zzall`` 贡献所有 ``n(n-1)`` 个含 Z 主导的有向两体正交关联（``i < j`` 的 ``Z_i Z_j`` 与 ``Z_i X_j``，共 ``2 * n(n-1)/2 = n(n-1)`` 项，区别于单向全对的 ``n(n-1)/2``）；
    * ``xxall`` 贡献所有 ``n(n-1)`` 个含 X 主导的有向两体正交关联（``i < j`` 的 ``X_i X_j`` 与 ``X_i Y_j``）；
    * ``yyall`` 贡献所有 ``n(n-1)`` 个含 Y 主导的有向两体正交关联（``i < j`` 的 ``Y_i Y_j`` 与 ``Y_i Z_j``）。

    在 ``n_qubits = 4`` 下：
    * ``"z"`` -> 4 维
    * ``"z+x"`` -> 8 维
    * ``"z+x+y"`` -> 12 维
    * ``"z+x+zz"`` -> 12 维
    * ``"z+x+y+zz"`` -> 16 维
    * ``"z+x+y+zzall"`` -> 24 维
    * ``"z+x+y+zzall+xxall+yyall"`` -> 48 维（12 个单比特 ``X/Y/Z`` + 36 个有向正交两体关联，首次超过 31 维词表秩限制）
    """
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    unknown = [p for p in parts if p not in _READOUT_PARTS]
    if unknown:
        raise ValueError(f"未知可观测量 {unknown}，可选 {_READOUT_PARTS}")
    ops = []
    if "z" in parts:
        ops.extend(qml.PauliZ(i) for i in range(n_qubits))
    if "x" in parts:
        ops.extend(qml.PauliX(i) for i in range(n_qubits))
    if "y" in parts:
        ops.extend(qml.PauliY(i) for i in range(n_qubits))
    if "zz" in parts:
        ops.extend(
            qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits)
            for i in range(n_qubits)
        )
    if "zzall" in parts:
        ops.extend(
            qml.PauliZ(i) @ qml.PauliZ(j)
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        )
        ops.extend(
            qml.PauliZ(i) @ qml.PauliX(j)
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        )
    if "xxall" in parts:
        ops.extend(
            qml.PauliX(i) @ qml.PauliX(j)
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        )
        ops.extend(
            qml.PauliX(i) @ qml.PauliY(j)
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        )
    if "yyall" in parts:
        ops.extend(
            qml.PauliY(i) @ qml.PauliY(j)
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        )
        ops.extend(
            qml.PauliY(i) @ qml.PauliZ(j)
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        )
    return ops


def n_readout(n_qubits: int, spec: str) -> int:
    """与 :func:`make_observables` 对应的读出维度，供分类头构造使用。"""
    return len(make_observables(n_qubits, spec))


def make_rich_layer(
    n_qubits: int = 4,
    n_layers: int = 2,
    encoding: str = "angle",
    observables: str = "z",
    n_features: int = 784,
    reupload: bool | str = False,
    diff_method: str = "backprop",
    shots: int | None = None,
    near_identity_init: bool = False,
    init_std: float = 0.05,
) -> nn.Module:
    """通用可变量子线路层：支持角度/振幅编码、多可观测量读出与固定/可训练数据重上传。

    参数
    ----
    encoding : ``"angle"`` 用 ``AngleEmbedding(RY)``（输入维度等于 ``n_qubits``）；
        ``"amplitude"`` 用 ``AmplitudeEmbedding``（要求 ``2**n_qubits >= n_features``）。
    observables : 传给 :func:`make_observables` 的规格，例如 ``"z+x+y+zzall+xxall+yyall"``。
    reupload :
        * ``False``：单次编码 + ``n_layers`` 层 ``StronglyEntanglingLayers``；
        * ``True``：每层固定重嵌入输入 ``AngleEmbedding(inputs)``；
        * ``"trainable"``：Pérez-Salinas et al. 可训练输入缩放重上传
          ``AngleEmbedding(inputs * in_scales[layer_i])``，``in_scales`` 初始化为全 1，
          且保持 ``weights`` 的随机初始化流与 ``reupload=False`` 完全一致。
    near_identity_init : 若为 True，将 ``weights`` 初始化为 ``N(0, init_std^2)``。
    """
    if encoding not in ("amplitude", "angle"):
        raise ValueError(f"未知编码 {encoding!r}，可选 'amplitude' / 'angle'")
    trainable_reup = isinstance(reupload, str) and reupload.lower() == "trainable"
    if isinstance(reupload, str) and not trainable_reup:
        raise ValueError(f"未知 reupload 模式 {reupload!r}，可选 False / True / 'trainable'")
    use_reupload = bool(reupload) or trainable_reup

    if encoding == "amplitude":
        if use_reupload:
            raise ValueError("AmplitudeEmbedding 会重置量子态，不支持数据重上传")
        if 2 ** n_qubits < n_features:
            raise ValueError(
                f"2**{n_qubits}={2 ** n_qubits} < n_features={n_features}"
            )
    ops = make_observables(n_qubits, observables)

    analytic = shots is None or shots == 0
    if not analytic and diff_method == "backprop":
        diff_method = "parameter-shift"
        print("[quantum] 有限 shots 不支持 backprop，diff_method 已自动切换为 parameter-shift")
    dev = (
        qml.device("default.qubit", wires=n_qubits)
        if analytic
        else qml.device("default.qubit", wires=n_qubits, shots=shots)
    )

    def _embed(inputs):
        if encoding == "amplitude":
            qml.AmplitudeEmbedding(
                inputs, wires=range(n_qubits), normalize=True, pad_with=0.0
            )
        else:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

    if trainable_reup:
        def circuit(inputs, weights, in_scales):
            for layer_i in range(n_layers):
                _embed(inputs * in_scales[layer_i])
                qml.StronglyEntanglingLayers(
                    weights[layer_i:layer_i + 1], wires=range(n_qubits)
                )
            return [qml.expval(op) for op in ops]

        weight_shapes = {
            "weights": (n_layers, n_qubits, 3),
            "in_scales": (n_layers, n_qubits),
        }
    else:
        def circuit(inputs, weights):
            if use_reupload:
                for layer_i in range(n_layers):
                    _embed(inputs)
                    qml.StronglyEntanglingLayers(
                        weights[layer_i:layer_i + 1], wires=range(n_qubits)
                    )
            else:
                _embed(inputs)
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(op) for op in ops]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

    try:
        qnode = qml.qnode(dev, diff_method=diff_method)(circuit)
        layer = qml.qnn.TorchLayer(qnode, weight_shapes)
    except Exception:
        qnode = qml.qnode(dev, interface="torch", diff_method=diff_method)(circuit)
        layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    layer = layer.to(torch.float32)
    if trainable_reup:
        with torch.no_grad():
            layer.in_scales.fill_(1.0)
    if near_identity_init:
        _apply_near_identity_init(layer, init_std=init_std)
    return _Float32VQC(layer)
