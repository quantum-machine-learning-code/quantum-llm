"""玩具级字符级自回归语言模型：支持逐字符（token-wise）与严格因果上下文（causal window）、多观测量量子层及消融对照。"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

try:
    from .quantum import make_rich_layer, make_vqc_layer, n_readout
except ImportError:
    from quantum import make_rich_layer, make_vqc_layer, n_readout

__all__ = [
    "CausalContextEncoder",
    "TinyLM",
    "RichQuantumLM",
]


def _build_head(n_in: int, vocab_size: int, head: str = "linear", hidden: int = 32) -> nn.Module:
    """构建词表分类头：``linear`` 为单层线性投影，``mlp`` 为带 GELU 的双层非线性头。"""
    head_key = str(head).lower()
    if head_key == "linear":
        return nn.Linear(n_in, vocab_size)
    if head_key == "mlp":
        return nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, vocab_size),
        )
    raise ValueError(f"未知分类头 {head!r}，可选 'linear' / 'mlp'")


def _set_scale(module: nn.Module, scale: float | str = 1.0) -> None:
    """设置读出缩放系数以缓解测量导致的 logit 收缩（measurement-induced logit contraction）。"""
    if isinstance(scale, str) and scale.lower() == "learnable":
        module.out_scale = nn.Parameter(torch.tensor(5.0, dtype=torch.float32))
    else:
        module.register_buffer(
            "out_scale", torch.tensor(float(scale), dtype=torch.float32)
        )


def _infer_readout_dim(transform: nn.Module, n_qubits: int, explicit_dim: int | None = None) -> int:
    """自动推断 transform 输出特征维度，兼容扩展可观测量与经典 Linear/Sequential。"""
    if explicit_dim is not None:
        return int(explicit_dim)
    if hasattr(transform, "readout_dim"):
        return int(getattr(transform, "readout_dim"))
    if isinstance(transform, nn.Linear):
        return int(transform.out_features)
    if isinstance(transform, nn.Sequential):
        for sub in reversed(list(transform.modules())):
            if isinstance(sub, nn.Linear):
                return int(sub.out_features)
    # 用单样本干跑推断输出维（无副作用）
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, n_qubits, dtype=torch.float32)
            out = transform(dummy)
            return int(out.shape[-1])
    except Exception:
        return int(n_qubits)


class CausalContextEncoder(nn.Module):
    """严格因果局部上下文编码器：左填充 ``window - 1`` 个时间步后做因果一维卷积。

    位置 ``t`` 的输出仅依赖 ``idx[:, max(0, t - window + 1) : t + 1]``，
    未来时间步 ``t' > t`` 的梯度与数值贡献恒为 0。
    """

    def __init__(
        self,
        vocab_size: int,
        n_qubits: int,
        context_window: int = 4,
        embed_dim: int | None = None,
    ):
        super().__init__()
        if context_window < 1:
            raise ValueError(f"context_window 必须 >= 1，当前 {context_window}")
        self.vocab_size = int(vocab_size)
        self.n_qubits = int(n_qubits)
        self.context_window = int(context_window)
        self.embed_dim = int(embed_dim) if embed_dim is not None else max(n_qubits * 4, 16)
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.causal_conv = nn.Conv1d(
            self.embed_dim, self.n_qubits, kernel_size=self.context_window
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)  # (B, T, E)
        x_t = x.transpose(1, 2)  # (B, E, T)
        if self.context_window > 1:
            x_t = F.pad(x_t, (self.context_window - 1, 0))
        h = self.causal_conv(x_t).transpose(1, 2)  # (B, T, Q)
        return h


class TinyLM(nn.Module):
    """字符级自回归模型：embedding(/因果编码器) -> 可微变换层(量子/经典/恒等) -> 词表分类头。

    默认参数（``context_mode="token"``, ``pre_scale=False``, ``scale=1.0``, ``head="linear"``）
    与初版 ``TinyLM(vocab_size, n_qubits, transform)`` 100% 向后兼容；
    同时修复了 ``z.reshape(B, T, -1)`` 以支持读出维度 ``readout_dim != n_qubits`` 的多可观测量量子层。
    """

    def __init__(
        self,
        vocab_size: int,
        n_qubits: int,
        transform: nn.Module | None = None,
        *,
        context_mode: str = "token",
        context_window: int = 4,
        embed_dim: int | None = None,
        pre_scale: bool = False,
        scale: float | str = 1.0,
        head: str = "linear",
        hidden: int = 32,
        readout_dim: int | None = None,
    ):
        super().__init__()
        if context_mode not in ("token", "causal"):
            raise ValueError(f"未知 context_mode {context_mode!r}，可选 'token' / 'causal'")
        self.vocab_size = int(vocab_size)
        self.n_qubits = int(n_qubits)
        self.context_mode = context_mode
        self.context_window = int(context_window) if context_mode == "causal" else 1
        self.pre_scale = bool(pre_scale)
        self.head_type = str(head)

        if context_mode == "token":
            self.embed_dim = int(n_qubits)
            self.embed = nn.Embedding(self.vocab_size, self.n_qubits)
            self.causal_encoder = None
        else:
            self.causal_encoder = CausalContextEncoder(
                vocab_size=self.vocab_size,
                n_qubits=self.n_qubits,
                context_window=self.context_window,
                embed_dim=embed_dim,
            )
            self.embed_dim = self.causal_encoder.embed_dim
            self.embed = self.causal_encoder.embed

        self.transform = transform if transform is not None else nn.Identity()
        self.readout_dim = _infer_readout_dim(self.transform, self.n_qubits, readout_dim)
        self.head = _build_head(self.readout_dim, self.vocab_size, head=head, hidden=hidden)
        _set_scale(self, scale)

    def encode_context(self, idx: torch.Tensor) -> torch.Tensor:
        """将输入 token 序列编码为进入量子/经典瓶颈层的 (B, T, n_qubits) 特征。"""
        if self.context_mode == "token":
            h = self.embed(idx)
        else:
            h = self.causal_encoder(idx)
        if self.pre_scale:
            h = torch.tanh(h) * (math.pi / 2.0)
        return h

    def bottleneck_features(self, idx: torch.Tensor) -> torch.Tensor:
        """返回经量子/经典/恒等变换后的 (B, T, readout_dim) 读出特征张量。"""
        h = self.encode_context(idx)
        B, T, Q = h.shape
        z = self.transform(h.reshape(B * T, Q))
        z = z.to(torch.float32)
        return z.reshape(B, T, -1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck_features(idx)  # (B, T, readout_dim)
        return self.head(z * self.out_scale)  # (B, T, V) logits


class RichQuantumLM(TinyLM):
    """受控实验用自回归语言模型：统一支持量子多观测量、重上传、经典基线与无变换（Identity）消融。

    构造顺序严格受控：
    1. 先在给定随机种子下初始化前端（``embed`` 及 ``causal_conv``），保证同 ``context_mode`` 下
       所有模型（Quantum / Classical / Identity）前端初始权重逐位一致；
    2. 从前端之后的同一 RNG 快照派生 ``head``，保证相同 ``(readout_dim, head)`` 的模型分类头初始权重逐位一致；
    3. 从同一 RNG 快照初始化中间变换层 ``transform``，保证同 ``(n_qubits, n_layers)`` 的不同读出量子模型
       共享完全相同的初始量子旋转角 ``weights``。
    """

    def __init__(
        self,
        vocab_size: int,
        n_qubits: int = 4,
        n_layers: int = 1,
        *,
        transform_type: str = "quantum",
        context_mode: str = "token",
        context_window: int = 4,
        embed_dim: int | None = None,
        observables: str = "z",
        reupload: bool | str = False,
        pre_scale: bool = True,
        near_identity_init: bool = False,
        init_std: float = 0.05,
        scale: float | str = 5.0,
        head: str = "linear",
        hidden: int = 32,
        classical_width: int | None = None,
        diff_method: str = "backprop",
        shots: int | None = None,
    ):
        nn.Module.__init__(self)
        if context_mode not in ("token", "causal"):
            raise ValueError(f"未知 context_mode {context_mode!r}，可选 'token' / 'causal'")
        ttype = str(transform_type).lower()
        if ttype not in ("quantum", "classical", "identity"):
            raise ValueError(
                f"未知 transform_type {transform_type!r}，可选 'quantum' / 'classical' / 'identity'"
            )

        self.vocab_size = int(vocab_size)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.transform_type = ttype
        self.context_mode = context_mode
        self.context_window = int(context_window) if context_mode == "causal" else 1
        self.observables = str(observables)
        self.reupload = reupload
        self.pre_scale = bool(pre_scale)
        self.near_identity_init = bool(near_identity_init)
        self.head_type = str(head)

        # 1. 初始化前端（同 seed + 同 context_mode 下所有模型逐元素一致）
        if context_mode == "token":
            self.embed_dim = int(n_qubits)
            self.embed = nn.Embedding(self.vocab_size, self.n_qubits)
            self.causal_encoder = None
        else:
            self.causal_encoder = CausalContextEncoder(
                vocab_size=self.vocab_size,
                n_qubits=self.n_qubits,
                context_window=self.context_window,
                embed_dim=embed_dim,
            )
            self.embed_dim = self.causal_encoder.embed_dim
            self.embed = self.causal_encoder.embed

        rng_after_frontend = torch.random.get_rng_state()

        # 2. 计算读出维度并从确定性偏移 RNG 初始化分类头（同 readout_dim + head_type 下逐元素一致）
        if ttype == "quantum":
            rdim = n_readout(self.n_qubits, self.observables)
        elif ttype == "classical":
            rdim = (
                int(classical_width)
                if classical_width is not None
                else n_readout(self.n_qubits, self.observables)
            )
        else:
            rdim = self.n_qubits
        self.readout_dim = int(rdim)

        # 用独立派生种子初始化 head，使其仅依赖前端之后的 RNG 状态与 head 形状
        _head_probe = torch.randint(0, 2**31 - 1, (1,)).item()
        torch.manual_seed(int(_head_probe))
        self.head = _build_head(self.readout_dim, self.vocab_size, head=head, hidden=hidden)

        # 3. 恢复前端之后的 RNG 快照以初始化 transform（同 qubits/layers 的量子模型权重逐元素一致）
        torch.random.set_rng_state(rng_after_frontend)
        if ttype == "quantum":
            self.transform = make_rich_layer(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                encoding="angle",
                observables=self.observables,
                reupload=self.reupload,
                diff_method=diff_method,
                shots=shots,
                near_identity_init=self.near_identity_init,
                init_std=init_std,
            )
            self.q = self.transform
        elif ttype == "classical":
            self.transform = nn.Sequential(
                nn.Linear(self.n_qubits, self.readout_dim),
                nn.Tanh(),
            )
            self.q = None
        else:
            self.transform = nn.Identity()
            self.q = None

        _set_scale(self, scale)
