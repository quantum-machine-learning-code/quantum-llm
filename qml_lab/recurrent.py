"""量子态参与序列依赖的最小原型（架构改造 L1，2026-11-16）。

现有 ``RichQuantumLM`` 的量子层把 ``(B*T, 4)`` 每一行**独立**送进 QNode：
不接收上一步状态、不向下一步传递状态、不跨位置纠缠。自回归链条完全在
经典侧，量子部分只是逐位置的特征变换。本模块给出最小改造，让自回归链条
真正经过量子系统。

每步 t（``QuantumRecurrentLM``）：

    h_t     = tanh(W_in · Embedding(idx_t)) · π/2      经典输入编码，(B, Q)
    s_ang_t = s_{t-1} · π/2                           上一步测量期望直接重上传
    s_t     = QNode[ RY(h_t) → RZ(s_ang_t) → StronglyEntanglingLayers(w)
                     → expval(Z_i) ]                   (B, Q)，既是新状态也是 logits 输入
    logits_t = head(s_t)

与现状的三点实质差别：

1. **状态反馈**：``s_{t-1}`` 以 RZ 相位形式重上传（RZ 与输入 RY 不交换），
   再经纠缠层把相位信息转成可测的 Z 期望；不是简单的角度相加。
2. **梯度跨步**：``s_t`` 不 detach，反向传播会穿过量子线路回到前一步，
   量子层因此参与序列依赖的优化，而不只是逐位置特征映射。
3. **逐步生成**：提供 ``step()`` / ``generate()``，预测下一个 token 时
   逐个时间步推进状态，而非把完整序列一次性喂进去做分类。

层级说明（避免把 L1 说成「真·量子态跨步」）：
    L0 现状          无状态，逐位置变换
    L1 本模块        测量反馈型量子循环：状态由测量期望携带并重上传
    L2 更彻底        保留隐 qubit 不测量、量子态在时间步间延续（需更多 qubit，
                     见 docs/ARCHITECTURE_PROPOSAL.md）

``ClassicalRecurrentLM`` 是同接口的经典 RNN 替身，用于后续消融；本模块
**不做任何效果对比、不宣称性能优势**，只提供结构可验证的实现。
"""
from __future__ import annotations

import math

import pennylane as qml
import torch
from torch import nn

__all__ = ["QuantumRecurrentLM", "ClassicalRecurrentLM", "make_recurrent_qnode"]


def make_recurrent_qnode(n_qubits: int):
    """构造带状态重上传的批量 QNode；权重作为参数传入，不闭包在模型里。

    ``x_ang`` / ``s_ang`` 形状均为 ``(B, Q)``，``weights`` 为 ``(L, Q, 3)``。
    返回 list of ``(B,)`` torch 张量，调用方用 ``torch.stack(..., dim=-1)``
    还原成 ``(B, Q)``；PennyLane 0.45 下直接 ``qml.math.stack`` 会退回
    pennylane.numpy.tensor 并破坏 torch autograd，故在 QNode 外部 stack。
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x_ang: torch.Tensor, s_ang: torch.Tensor,
                weights: torch.Tensor):
        qml.AngleEmbedding(x_ang, wires=range(n_qubits), rotation="Y")
        # 上一步测量期望作为相位反馈；RZ 与 RY 不交换，之后的纠缠层
        # 才能把相位信息转成可测的 Z 期望（纯角度相加会退化成线性偏置）。
        for i in range(n_qubits):
            qml.RZ(s_ang[:, i], wires=i)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class _RecurrentBase(nn.Module):
    """共用接口：``forward -> (B,T,V)``、``step``、``initial_state``、``generate``。"""

    def __init__(self, vocab_size: int, state_dim: int = 4,
                 embed_dim: int = 16, hidden: int = 32,
                 feedback: bool = True, learnable_init: bool = False):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.state_dim = int(state_dim)
        self.embed_dim = int(embed_dim)
        self.feedback = bool(feedback)
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.head = nn.Sequential(
            nn.Linear(self.state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.vocab_size),
        )
        if learnable_init:
            self.init_state = nn.Parameter(torch.zeros(1, self.state_dim))
        else:
            self.register_buffer("init_state", torch.zeros(1, self.state_dim))

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        dev = device if device is not None else self.init_state.device
        return self.init_state.expand(batch, -1).to(dev)

    def step(self, idx_t: torch.Tensor, state: torch.Tensor):
        raise NotImplementedError

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """严格因果：位置 t 的 logits 只依赖 idx[:, :t+1] 与初始状态。"""
        B, T = idx.shape
        state = self.initial_state(B, idx.device)
        out = []
        for t in range(T):
            logits_t, state = self.step(idx[:, t], state)
            out.append(logits_t)
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, n_new: int,
                 temperature: float = 1.0) -> torch.Tensor:
        """逐步自回归生成：每步取 argmax 拼回输入，再推进状态。"""
        self.eval()
        cur = prefix.clone()
        state = self.initial_state(cur.shape[0], cur.device)
        # 先跑完 prefix，建立状态
        for t in range(cur.shape[1]):
            _, state = self.step(cur[:, t], state)
        for _ in range(n_new):
            logits_t, state = self.step(cur[:, -1], state)
            probs = (logits_t / max(temperature, 1e-6)).softmax(-1)
            nxt = probs.argmax(-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
        return cur


class QuantumRecurrentLM(_RecurrentBase):
    """L1：测量反馈型量子循环语言模型。

    线路权重 ``(n_layers, Q, 3)`` 是可训练参数；``feedback=False`` 时把
    状态角度置零，退化为逐位置独立的 L0 量子变换，用于状态消融。
    """

    def __init__(self, vocab_size: int, n_qubits: int = 4, n_layers: int = 1,
                 embed_dim: int = 16, hidden: int = 32, feedback: bool = True,
                 learnable_init: bool = False, init_std: float = 0.05,
                 detach_state: bool = False):
        super().__init__(vocab_size, state_dim=n_qubits, embed_dim=embed_dim,
                         hidden=hidden, feedback=feedback,
                         learnable_init=learnable_init)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.detach_state = bool(detach_state)  # True = 截断 BPTT，省内存
        self.input_proj = nn.Linear(self.embed_dim, self.n_qubits)
        self.weights = nn.Parameter(
            torch.empty(self.n_layers, self.n_qubits, 3)
        )
        nn.init.normal_(self.weights, std=init_std)
        self.qnode = make_recurrent_qnode(self.n_qubits)

    def step(self, idx_t: torch.Tensor, state: torch.Tensor):
        h = self.embed(idx_t)                                   # (B, E)
        x_ang = torch.tanh(self.input_proj(h)) * (math.pi / 2)  # (B, Q)
        s_ang = ((state if self.feedback else torch.zeros_like(state))
                 * (math.pi / 2))
        raw = self.qnode(x_ang, s_ang, self.weights)
        # PennyLane 默认 float64；cast 保留 autograd 链，与 head 的 float32 对齐
        new_state = torch.stack(list(raw), dim=-1).to(torch.float32)  # (B, Q)
        if self.detach_state:
            new_state = new_state.detach()
        return self.head(new_state), new_state


class ClassicalRecurrentLM(_RecurrentBase):
    """同接口经典 RNN 替身（状态消融 / 后续容量匹配的对照组）。

    ``feedback=False`` 时同样把上一步状态置零，与量子臂的消融口径一致。
    """

    def __init__(self, vocab_size: int, state_dim: int = 4,
                 embed_dim: int = 16, hidden: int = 32, feedback: bool = True,
                 learnable_init: bool = False, detach_state: bool = False):
        super().__init__(vocab_size, state_dim=state_dim, embed_dim=embed_dim,
                         hidden=hidden, feedback=feedback,
                         learnable_init=learnable_init)
        self.detach_state = bool(detach_state)
        self.cell = nn.RNNCell(self.embed_dim, self.state_dim)

    def step(self, idx_t: torch.Tensor, state: torch.Tensor):
        h = self.embed(idx_t)                                   # (B, E)
        s_in = state if self.feedback else torch.zeros_like(state)
        new_state = self.cell(h, s_in)                          # (B, Q), 含 tanh
        if self.detach_state:
            new_state = new_state.detach()
        return self.head(new_state), new_state
