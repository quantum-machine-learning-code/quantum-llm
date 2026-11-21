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


def make_recurrent_qnode(
    n_qubits: int,
    state_feedback: str = "rz",
    readout_basis: str = "y",
):
    """构造带状态重上传的批量 QNode；权重作为参数传入，不闭包在模型里。

    * ``state_feedback``: ``"rz"``（非对易相位反馈）或 ``"ry"``（同轴角度反馈）；
    * ``readout_basis``: ``"z"`` / ``"x"`` / ``"y"`` / ``"xyz"``。
      默认 ``rz + y``：``RY(x)|0>`` 落在 XZ 平面，``RZ(s)`` 产生 ``<Y> = sin(x)sin(s)``，
      即使 ``s=0, w=0`` 一阶导数仍非零，彻底修复旧版 ``rz + z`` 的死通道问题。
    """
    fb = str(state_feedback).lower()
    if fb not in ("rz", "ry"):
        raise ValueError(f"未知 state_feedback {state_feedback!r}，可选 'rz' / 'ry'")
    rb = str(readout_basis).lower()
    if rb not in ("z", "x", "y", "xyz"):
        raise ValueError(f"未知 readout_basis {readout_basis!r}，可选 'z' / 'x' / 'y' / 'xyz'")

    dev = qml.device("default.qubit", wires=n_qubits)
    fb_gate = qml.RZ if fb == "rz" else qml.RY
    if rb == "z":
        obs = [qml.PauliZ(i) for i in range(n_qubits)]
    elif rb == "x":
        obs = [qml.PauliX(i) for i in range(n_qubits)]
    elif rb == "y":
        obs = [qml.PauliY(i) for i in range(n_qubits)]
    else:
        obs = (
            [qml.PauliZ(i) for i in range(n_qubits)]
            + [qml.PauliX(i) for i in range(n_qubits)]
            + [qml.PauliY(i) for i in range(n_qubits)]
        )

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x_ang: torch.Tensor, s_ang: torch.Tensor,
                weights: torch.Tensor):
        qml.AngleEmbedding(x_ang, wires=range(n_qubits), rotation="Y")
        for i in range(n_qubits):
            fb_gate(s_ang[:, i], wires=i)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(op) for op in obs]

    return circuit


class _RecurrentBase(nn.Module):
    """共用接口：``forward -> (B,T,V)``、``step``、``initial_state``、``generate``。"""

    def __init__(self, vocab_size: int, state_dim: int = 4,
                 embed_dim: int = 16, hidden: int = 32,
                 feedback: bool = True, learnable_init: bool = False,
                 readout_dim: int | None = None):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.state_dim = int(state_dim)
        self.readout_dim = int(readout_dim) if readout_dim is not None else int(state_dim)
        self.embed_dim = int(embed_dim)
        self.feedback = bool(feedback)
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.head = nn.Sequential(
            nn.Linear(self.readout_dim, hidden),
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
    def generate(
        self,
        prefix: torch.Tensor,
        n_new: int,
        temperature: float = 1.0,
        *,
        greedy: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """逐步自回归生成：精确推进 prefix 状态，不再重复送入末位 prefix token。

        * ``greedy=True`` 或 ``temperature <= 0``：确定性贪心 argmax，
          保证 ``generate(prefix, 1, greedy=True)[:, -1] == forward(prefix)[:, -1].argmax(-1)``；
        * ``greedy=False`` 且 ``temperature > 0``：按 ``softmax(logits / temperature)`` 多项采样。
        """
        self.eval()
        cur = prefix.clone()
        if n_new <= 0:
            return cur
        state = self.initial_state(cur.shape[0], cur.device)
        logits_t: torch.Tensor | None = None
        for t in range(cur.shape[1]):
            logits_t, state = self.step(cur[:, t], state)
        assert logits_t is not None, "prefix 长度必须 >= 1"
        for step_i in range(int(n_new)):
            if greedy or temperature <= 0.0:
                nxt = logits_t.argmax(dim=-1, keepdim=True)
            else:
                probs = (logits_t / float(temperature)).softmax(dim=-1)
                nxt = torch.multinomial(probs, num_samples=1, generator=generator)
            cur = torch.cat([cur, nxt], dim=1)
            if step_i + 1 < int(n_new):
                logits_t, state = self.step(nxt.squeeze(-1), state)
        return cur


class QuantumRecurrentLM(_RecurrentBase):
    """L1：测量反馈型量子循环语言模型（状态为经典测量期望，非 L2 持久量子态）。

    默认 ``state_feedback="rz", readout_basis="y", init_std=1.0`` 修复了旧版
    ``rz + z + init_std=0.05`` 状态通道在初始化附近被压制 ~163× 的结构缺陷。
    """

    def __init__(self, vocab_size: int, n_qubits: int = 4, n_layers: int = 1,
                 embed_dim: int = 16, hidden: int = 32, feedback: bool = True,
                 learnable_init: bool = False, init_std: float = 1.0,
                 detach_state: bool = False, state_feedback: str = "rz",
                 readout_basis: str = "y"):
        rb = str(readout_basis).lower()
        rdim = 3 * int(n_qubits) if rb == "xyz" else int(n_qubits)
        super().__init__(vocab_size, state_dim=n_qubits, embed_dim=embed_dim,
                         hidden=hidden, feedback=feedback,
                         learnable_init=learnable_init, readout_dim=rdim)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.detach_state = bool(detach_state)  # True = 截断 BPTT（仅截断传向下一步的状态）
        self.state_feedback = str(state_feedback).lower()
        self.readout_basis = rb
        self.input_proj = nn.Linear(self.embed_dim, self.n_qubits)
        self.weights = nn.Parameter(
            torch.empty(self.n_layers, self.n_qubits, 3)
        )
        nn.init.normal_(self.weights, std=float(init_std))
        self.qnode = make_recurrent_qnode(
            self.n_qubits,
            state_feedback=self.state_feedback,
            readout_basis=self.readout_basis,
        )

    def step(self, idx_t: torch.Tensor, state: torch.Tensor):
        h = self.embed(idx_t)                                   # (B, E)
        x_ang = torch.tanh(self.input_proj(h)) * (math.pi / 2)  # (B, Q)
        s_ang = ((state if self.feedback else torch.zeros_like(state))
                 * (math.pi / 2))
        raw = self.qnode(x_ang, s_ang, self.weights)
        # PennyLane 默认 float64；cast 保留 autograd 链，与 head 的 float32 对齐
        readout = torch.stack(list(raw), dim=-1).to(torch.float32)  # (B, rdim)
        if self.readout_basis == "xyz":
            # rz 相位反馈对 Y 基一阶敏感；ry 同轴反馈对 Z 基一阶敏感
            new_state = (
                readout[:, 2 * self.n_qubits :]
                if self.state_feedback == "rz"
                else readout[:, : self.n_qubits]
            )
        else:
            new_state = readout
        logits = self.head(readout)
        # 截断 BPTT：必须在 head(readout) 之后再 detach 传向 t+1 的状态，
        # 否则当前步 t 的 embed / input_proj / weights 将全部失去梯度！
        next_state = new_state.detach() if self.detach_state else new_state
        return logits, next_state


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
        logits = self.head(new_state)
        next_state = new_state.detach() if self.detach_state else new_state
        return logits, next_state
