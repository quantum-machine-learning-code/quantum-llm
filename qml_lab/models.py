"""玩具级字符级自回归语言模型：支持逐字符（token-wise）与严格因果上下文（causal window）、多观测量量子层及消融对照。"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

try:
    from .quantum import make_rich_layer, n_readout
except ImportError:
    from quantum import make_rich_layer, n_readout

__all__ = [
    "CausalContextEncoder",
    "ClassicalTrigBilinearLayer",
    "TinyLM",
    "RichQuantumLM",
    "parameter_breakdown",
]


class ClassicalTrigBilinearLayer(nn.Module):
    """公平的经典函数类替身：显式构造 1-body/2-body 三角乘积特征，匹配单次 RY 角度编码 PQC。

    对 ``Q`` 维瓶颈输入 ``h`` 计算 ``u = in_scale * h + in_shift``，展开为全部
    ``2Q + 4*Q*(Q-1)/2`` 个度-1 三角单项式
    ``[cos u_i, sin u_i, cos u_i cos u_j, sin u_i sin u_j, cos u_i sin u_j, sin u_i cos u_j]``，
    再通过固定正交投影（或可学习线性投影）输出 ``out_features`` 维特征。

    * ``param_mode="matched"``（默认）：仅训练 ``in_scale`` (Q) + ``in_shift`` (Q) +
      ``out_gain`` (Q) = **``3 * Q`` 个参数**（``Q=4`` 时精确等于 1 层 PQC 的 12 个参数），
      固定正交投影矩阵为 buffer，输出尺度与单层量子读出自然对齐。
    * ``param_mode="linear"``：在三角单项式基上接 ``nn.Linear(D_trig, out_features)``。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        param_mode: str = "matched",
        near_identity_init: bool = False,
        init_std: float = 0.05,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.readout_dim = int(out_features)
        self.param_mode = str(param_mode).lower()
        if self.param_mode not in ("matched", "linear"):
            raise ValueError(f"未知 param_mode {param_mode!r}，可选 'matched' / 'linear'")

        q = self.in_features
        self._pairs = [(i, j) for i in range(q) for j in range(i + 1, q)]
        self.trig_dim = 2 * q + 4 * len(self._pairs)

        self.in_scale = nn.Parameter(torch.ones(q, dtype=torch.float32))
        self.in_shift = nn.Parameter(torch.zeros(q, dtype=torch.float32))
        if near_identity_init:
            with torch.no_grad():
                self.in_scale.add_(torch.randn(q) * float(init_std))
                self.in_shift.add_(torch.randn(q) * float(init_std))
        else:
            with torch.no_grad():
                self.in_shift.uniform_(-math.pi / 4.0, math.pi / 4.0)

        if self.param_mode == "matched":
            self.out_gain = nn.Parameter(torch.ones(q, dtype=torch.float32))
            # 固定正交投影（buffer，不计入可训练参数）：使输出保持单位范数尺度
            raw = torch.randn(max(self.trig_dim, self.out_features),
                              max(self.trig_dim, self.out_features))
            q_mat, _ = torch.linalg.qr(raw)
            proj = q_mat[:self.trig_dim, :self.out_features].contiguous()
            # 缩放使其与 Pauli 期望读出的初始 std (~0.25-0.35) 量级对齐
            proj = proj * math.sqrt(self.trig_dim / max(self.out_features, 1)) * 0.65
            self.register_buffer("proj", proj.to(torch.float32))
            self.linear_out = None
        else:
            self.out_gain = None
            self.register_buffer("proj", torch.empty(0))
            self.linear_out = nn.Linear(self.trig_dim, self.out_features)

    def _trig_features(self, x: torch.Tensor) -> torch.Tensor:
        u = x * self.in_scale + self.in_shift
        c = torch.cos(u)
        s = torch.sin(u)
        feats = [c, s]
        if self._pairs:
            idx_i = [i for i, _ in self._pairs]
            idx_j = [j for _, j in self._pairs]
            ci, cj = c[:, idx_i], c[:, idx_j]
            si, sj = s[:, idx_i], s[:, idx_j]
            feats.extend([ci * cj, si * sj, ci * sj, si * cj])
        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self._trig_features(x.to(torch.float32))
        if self.param_mode == "matched":
            out = phi @ self.proj
            # 按周期索引广播 Q 维可学习通道增益（总可训练参数 = 3*Q）
            gain_idx = torch.arange(self.out_features, device=out.device) % self.in_features
            return out * self.out_gain[gain_idx]
        return torch.tanh(self.linear_out(phi))


def parameter_breakdown(model: nn.Module) -> dict[str, float | int]:
    """按组件分解模型参数量，显式分离量子参数与经典封装参数。"""
    embed_p = sum(p.numel() for p in getattr(model, "embed").parameters()) if hasattr(model, "embed") else 0
    ce = getattr(model, "causal_encoder", None)
    conv_p = sum(p.numel() for p in ce.causal_conv.parameters()) if ce is not None else 0
    head_p = sum(p.numel() for p in getattr(model, "head").parameters()) if hasattr(model, "head") else 0
    ttype = getattr(model, "transform_type", None)
    tf = getattr(model, "transform", None)
    tf_p = sum(p.numel() for p in tf.parameters()) if tf is not None else 0
    if ttype == "quantum":
        q_p, c_tf_p = tf_p, 0
    elif hasattr(model, "weights") and hasattr(model, "qnode"):
        q_p = int(model.weights.numel())
        c_tf_p = (
            sum(p.numel() for p in model.input_proj.parameters())
            if hasattr(model, "input_proj")
            else 0
        )
    else:
        q_p, c_tf_p = 0, tf_p
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    other = total - (embed_p + conv_p + q_p + c_tf_p + head_p)
    return {
        "embed": int(embed_p),
        "causal_conv": int(conv_p),
        "transform_quantum": int(q_p),
        "transform_classical": int(c_tf_p),
        "head": int(head_p),
        "other": int(other),
        "total": int(total),
        "trainable": int(trainable),
        "quantum_param_share": float(q_p / max(total, 1)),
    }


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


def _set_scale(module: nn.Module, scale: float | str = 1.0,
               init: float = 5.0) -> None:
    """设置读出缩放系数以缓解测量导致的 logit 收缩。

    ``scale="learnable"`` 时两臂都拿到同一初值的可训练标量（初值由
    ``init`` 指定，供装置标定扫描使用），避免「只给量子臂放大」的历史补丁。
    """
    if isinstance(scale, str) and scale.lower() == "learnable":
        module.out_scale = nn.Parameter(torch.tensor(float(init), dtype=torch.float32))
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
        # 对称读出归一化：仅 matched 对照开启，两臂同一层、同一初值。
        # 量子层近恒等小初始化与经典层小权重初始化的输出尺度天然差一个量级，
        # 光让配置字段相同并不能让两臂真的站在同一起跑线上。
        norm = getattr(self, "readout_norm", None)
        if norm is not None:
            z = norm(z)
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
        scale_init: float = 5.0,
        head: str = "linear",
        hidden: int = 32,
        classical_width: int | None = None,
        diff_method: str = "backprop",
        shots: int | None = None,
        readout_norm: bool = False,
    ):
        nn.Module.__init__(self)
        if context_mode not in ("token", "causal"):
            raise ValueError(f"未知 context_mode {context_mode!r}，可选 'token' / 'causal'")
        ttype = str(transform_type).lower()
        if ttype not in ("quantum", "classical", "classical_trig", "identity"):
            raise ValueError(
                f"未知 transform_type {transform_type!r}，可选 'quantum' / 'classical' / 'classical_trig' / 'identity'"
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
        elif ttype in ("classical", "classical_trig"):
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
            # 修复历史 near_identity_init 16× 失衡 bug：
            # 量子臂近恒等初始化（旋转角 ~ N(0, init_std^2) => U ≈ I）产生 O(1) 输出（std ~ 0.3）；
            # 旧版经典臂把权重设成 N(0, init_std^2) 实际是「近零初始化」（std ~ 0.024）。
            # 此处改为半正交恒等平铺 + 微小扰动，使两臂在 near_identity_init=True 下初始输出尺度对齐。
            if self.near_identity_init:
                with torch.no_grad():
                    w = self.transform[0].weight  # (readout_dim, n_qubits)
                    rows = []
                    n_blocks = (self.readout_dim + self.n_qubits - 1) // self.n_qubits
                    for _ in range(n_blocks):
                        blk = torch.empty(self.n_qubits, self.n_qubits)
                        nn.init.orthogonal_(blk)
                        rows.append(blk)
                    base = torch.cat(rows, dim=0)[:self.readout_dim]
                    w.copy_(base * 0.55 + torch.randn_like(w) * float(init_std))
                    nn.init.zeros_(self.transform[0].bias)
            self.q = None
        elif ttype == "classical_trig":
            self.transform = ClassicalTrigBilinearLayer(
                in_features=self.n_qubits,
                out_features=self.readout_dim,
                param_mode="matched",
                near_identity_init=self.near_identity_init,
                init_std=init_std,
            )
            self.q = None
        else:
            self.transform = nn.Identity()
            self.q = None

        # matched 对照专用：两臂完全相同的 LayerNorm（同结构、同初值），
        # 消除「量子近恒等线路 vs 经典小权重线性层」初始输出尺度差一个量级
        # 带来的起跑线不一致。历史配置 readout_norm=False，不会进入 forward。
        self.readout_norm = (
            nn.LayerNorm(self.readout_dim) if readout_norm else None
        )
        _set_scale(self, scale, scale_init)
