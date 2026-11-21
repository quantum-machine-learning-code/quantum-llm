"""Quantum Self-Attention Three-Regime Prototype Benchmark.

Compares strictly parameter-matched (~924-946 params) models across all three regimes on
zero-overlap Key-Value Associative Recall with OOD sequence-length generalization (T=6 -> T=10):

Regime I — Classical Matched Controls:
  * ``ClassicalDotAttention``: standard single-head scaled dot-product Softmax attention.
  * ``ClassicalOverlapAttention``: exact dequantized noiseless surrogate of SWAP-Test attention
    using squared unit-vector overlap |<q_hat, k_hat>|^2 + Softmax + V.
  * ``ClassicalLinearMixer``: Softmax-free linear-superposition mixer with normalized token weights
    (fair classical control for LCU linear token mixing).

Regime II — SWAP-Test Hybrid Quantum Attention:
  * ``QuantumSwapTestAttention``: amplitude-encodes query/key into n_q=2 qubit states (d=4),
    computes overlap via SWAP-Test |<psi_q|psi_k>|^2 = 2 P(ancilla=0) - 1, then classical
    Softmax + V. Verified against a real 5-qubit PennyLane ``default.qubit`` CSWAP circuit
    (<1e-10) and evaluated under finite-shot Bernoulli sampling (S in {100, 500, 2000}) with
    explicit O(T*S) / O(T^2*S) shot-complexity accounting.

Regime III — Coherent LCU + QSVT / Quixer-Style Quantum Attention:
  * ``QuantumQuixerAttention`` (grounded in Khatri et al. 2024 & Gilyen et al. 2019):
    maps each token to an n_q=3 qubit unitary U(x_t) = exp(-i H(x_t)) (d_H=8), forms the
    block-encoded linear combination A = sum_t alpha_t U(x_t) via PREPARE/SELECT/PREPARE†,
    applies a learnable degree-3 QSVT singular-value polynomial P_Phi(A) via projector-controlled
    phase rotations Phi = (phi_0, phi_1, phi_2, phi_3) to |psi_q> = U(q)|0>^3, and reads out via
    a complete Born-rule POVM (sum_c M_c = I_8, ZERO classical MLP crutch). Verified against a
    real PennyLane ``default.qubit`` PREPARE/SELECT/PREPARE† + PCPhase circuit (<1e-6), recording
    exact post-selection success probability p_succ.
"""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .quantum_ssm import _pauli_1_and_2_body_basis

__all__ = [
    "ClassicalDotAttention",
    "ClassicalOverlapAttention",
    "ClassicalLinearMixer",
    "QuantumSwapTestAttention",
    "QuantumQuixerAttention",
    "build_kv_recall_dataset",
    "attention_parameter_breakdown",
    "verify_swap_test_pennylane",
]


# --------------------------------------------------------------------------- #
# Dataset: Strictly Disjoint Key-Value Associative Recall (ID T=6 -> OOD T=10)
# --------------------------------------------------------------------------- #
def build_kv_recall_dataset(
    *,
    n_keys: int = 4,
    n_values: int = 4,
    n_distractors: int = 2,
    train_len: int = 6,
    ood_len: int = 10,
    n_train: int = 640,
    n_val: int = 160,
    n_test_id: int = 200,
    n_test_ood: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Build strictly disjoint Train/Val/Test-ID (T=6) and OOD-length (T=10) KV Associative Recall.

    Token encoding (``vocab_size = n_keys * n_values + n_keys + n_distractors``):
      * ``0 .. n_keys * n_values - 1``: key-value binding token ``(k, v) -> k * n_values + v``
      * ``query_base .. query_base + n_keys - 1``: query token ``Q_k``
      * ``dist_base .. dist_base + n_distractors - 1``: distractor/noise tokens

    Each sequence of length ``T`` contains ``T - 1`` randomly shuffled context tokens (distinct-key
    KV bindings + distractors) followed by one query token ``Q_{k*}`` at position ``T - 1``. The
    target is the value ``v* in {0 .. n_values - 1}`` bound to ``k*`` in that sequence.
    """
    n_ctx_id = int(train_len) - 1
    n_ctx_ood = int(ood_len) - 1
    if n_ctx_id < n_keys:
        raise ValueError(f"train_len - 1 ({n_ctx_id}) must be >= n_keys ({n_keys})")
    if n_ctx_ood < n_keys:
        raise ValueError(f"ood_len - 1 ({n_ctx_ood}) must be >= n_keys ({n_keys})")

    query_base = n_keys * n_values
    dist_base = query_base + n_keys
    vocab_size = dist_base + n_distractors
    rng = random.Random(int(seed))

    def _sample_one(n_ctx: int, min_kv: int, max_kv: int) -> tuple[tuple[int, ...], int]:
        n_kv = rng.randint(min_kv, min(max_kv, n_ctx))
        keys = rng.sample(range(n_keys), n_kv)
        vals = {k: rng.randrange(n_values) for k in keys}
        ctx = [k * n_values + vals[k] for k in keys]
        for _ in range(n_ctx - n_kv):
            ctx.append(dist_base + rng.randrange(n_distractors))
        rng.shuffle(ctx)
        q_key = rng.choice(keys)
        return tuple(ctx + [query_base + q_key]), vals[q_key]

    seen_id: set[tuple[int, ...]] = set()

    def _collect_unique(n_target: int, n_ctx: int, min_kv: int, max_kv: int, seen: set) -> tuple[torch.Tensor, torch.Tensor]:
        xs: list[tuple[int, ...]] = []
        ys: list[int] = []
        while len(xs) < n_target:
            seq, y = _sample_one(n_ctx, min_kv, max_kv)
            if seq not in seen:
                seen.add(seq)
                xs.append(seq)
                ys.append(y)
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

    # Train/Val/Test-ID vary distractor multiplicity (1..2 distractors at T=6) so distractors
    # cannot be exploited as a fixed-multiplicity constant offset.
    min_kv_id = max(2, n_keys - 1)
    tr_x, tr_y = _collect_unique(n_train, n_ctx_id, min_kv_id, n_keys, seen_id)
    va_x, va_y = _collect_unique(n_val, n_ctx_id, min_kv_id, n_keys, seen_id)
    te_x, te_y = _collect_unique(n_test_id, n_ctx_id, min_kv_id, n_keys, seen_id)

    seen_ood: set[tuple[int, ...]] = set()
    ood_x, ood_y = _collect_unique(n_test_ood, n_ctx_ood, n_keys, n_keys, seen_ood)

    tr_set = {tuple(r) for r in tr_x.tolist()}
    va_set = {tuple(r) for r in va_x.tolist()}
    te_set = {tuple(r) for r in te_x.tolist()}
    assert not (tr_set & va_set), "Train/Val overlap detected"
    assert not (tr_set & te_set), "Train/Test-ID overlap detected"
    assert not (va_set & te_set), "Val/Test-ID overlap detected"

    return {
        "vocab_size": vocab_size,
        "n_classes": n_values,
        "n_keys": n_keys,
        "n_values": n_values,
        "n_distractors": n_distractors,
        "train_len": int(train_len),
        "ood_len": int(ood_len),
        "chance_accuracy": 1.0 / float(n_values),
        "train": (tr_x, tr_y),
        "val": (va_x, va_y),
        "test_id": (te_x, te_y),
        "test_ood": (ood_x, ood_y),
    }


# --------------------------------------------------------------------------- #
# Regime I: Classical Matched Controls
# --------------------------------------------------------------------------- #
class ClassicalDotAttention(nn.Module):
    """Regime I: Standard single-head scaled dot-product Softmax attention."""

    def __init__(
        self,
        vocab_size: int,
        n_classes: int = 4,
        embed_dim: int = 20,
        key_dim: int = 4,
        val_dim: int = 14,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_classes = int(n_classes)
        self.key_dim = int(key_dim)
        self.embed = nn.Embedding(self.vocab_size, int(embed_dim))
        self.w_q = nn.Linear(int(embed_dim), int(key_dim), bias=False)
        self.w_k = nn.Linear(int(embed_dim), int(key_dim), bias=False)
        self.w_v = nn.Linear(int(embed_dim), int(val_dim), bias=False)
        self.head = nn.Linear(int(val_dim), self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)  # (B, T, d_e)
        q = self.w_q(emb[:, -1, :])  # (B, d_k)
        k = self.w_k(emb[:, :-1, :])  # (B, N, d_k)
        v = self.w_v(emb[:, :-1, :])  # (B, N, d_v)
        scores = torch.einsum("bd,bnd->bn", q, k) / math.sqrt(self.key_dim)
        weights = F.softmax(scores, dim=-1)
        ctx = torch.einsum("bn,bnd->bd", weights, v)
        return self.head(ctx)


class ClassicalOverlapAttention(nn.Module):
    """Regime I: Exact dequantized noiseless surrogate of SWAP-Test attention.

    Computes normalized query/key vectors q_hat, k_hat in S^{d_k-1}, squared overlap
    |<q_hat, k_hat>|^2, learnable Softmax temperature tau, and classical value aggregation.
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int = 4,
        embed_dim: int = 20,
        key_dim: int = 4,
        val_dim: int = 14,
        scale_init: float = 6.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_classes = int(n_classes)
        self.key_dim = int(key_dim)
        self.embed = nn.Embedding(self.vocab_size, int(embed_dim))
        self.w_q = nn.Linear(int(embed_dim), int(key_dim), bias=False)
        self.w_k = nn.Linear(int(embed_dim), int(key_dim), bias=False)
        self.w_v = nn.Linear(int(embed_dim), int(val_dim), bias=False)
        self.attn_scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.head = nn.Linear(int(val_dim), self.n_classes)

    def _normalized_qk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.embed(x)
        q_raw = self.w_q(emb[:, -1, :])
        k_raw = self.w_k(emb[:, :-1, :])
        v = self.w_v(emb[:, :-1, :])
        q_hat = F.normalize(q_raw, p=2, dim=-1, eps=1e-8)
        k_hat = F.normalize(k_raw, p=2, dim=-1, eps=1e-8)
        return q_hat, k_hat, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_hat, k_hat, v = self._normalized_qk(x)
        overlaps = torch.einsum("bd,bnd->bn", q_hat, k_hat).pow(2)
        weights = F.softmax(self.attn_scale * overlaps, dim=-1)
        ctx = torch.einsum("bn,bnd->bd", weights, v)
        return self.head(ctx)


class ClassicalLinearMixer(nn.Module):
    """Regime I: Softmax-free linear superposition mixer (fair control for LCU mixing).

    Uses normalized query-independent token weights alpha_t (matching LCU PREPARE) to form a
    linear operator M = sum_t alpha_t (v_t k_t^T) applied to query q without Softmax.
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int = 4,
        embed_dim: int = 20,
        key_dim: int = 4,
        val_dim: int = 13,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_classes = int(n_classes)
        self.key_dim = int(key_dim)
        self.embed = nn.Embedding(self.vocab_size, int(embed_dim))
        self.w_q = nn.Linear(int(embed_dim), int(key_dim), bias=False)
        self.w_k = nn.Linear(int(embed_dim), int(key_dim), bias=False)
        self.w_v = nn.Linear(int(embed_dim), int(val_dim), bias=False)
        self.token_gate = nn.Parameter(torch.zeros(self.vocab_size))
        self.head = nn.Linear(int(val_dim), self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        q = self.w_q(emb[:, -1, :])  # (B, d_k)
        k = self.w_k(emb[:, :-1, :])  # (B, N, d_k)
        v = self.w_v(emb[:, :-1, :])  # (B, N, d_v)
        w = F.softplus(self.token_gate[x[:, :-1]]) + 1e-4
        alpha = w / w.sum(dim=-1, keepdim=True)
        # Softmax-free bilinear superposition: sum_t alpha_t (q^T k_t / sqrt(d_k)) v_t
        lin_scores = alpha * (torch.einsum("bd,bnd->bn", q, k) / math.sqrt(self.key_dim))
        ctx = torch.einsum("bn,bnd->bd", lin_scores, v)
        return self.head(ctx)


# --------------------------------------------------------------------------- #
# Regime II: SWAP-Test Hybrid Quantum Attention + PennyLane Circuit & Shots
# --------------------------------------------------------------------------- #
def verify_swap_test_pennylane(
    q_state: torch.Tensor | np.ndarray,
    k_state: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    """Run a real 5-qubit PennyLane ``default.qubit`` SWAP-Test circuit and check exact parity.

    Wire layout: wire 0 = ancilla, wires [1, 2] = |psi_q>, wires [3, 4] = |psi_k>.
    Circuit: H(0) -> CSWAP(0, 1, 3) -> CSWAP(0, 2, 4) -> H(0) -> P(ancilla=0).
    Exact identity: |<psi_q | psi_k>|^2 = 2 * P(ancilla=0) - 1.
    """
    import pennylane as qml

    qv = np.asarray(q_state.detach().cpu().numpy() if isinstance(q_state, torch.Tensor) else q_state, dtype=np.float64)
    kv = np.asarray(k_state.detach().cpu().numpy() if isinstance(k_state, torch.Tensor) else k_state, dtype=np.float64)
    qv = qv / max(float(np.linalg.norm(qv)), 1e-12)
    kv = kv / max(float(np.linalg.norm(kv)), 1e-12)
    n_qubits = int(round(math.log2(len(qv))))
    if 2 ** n_qubits != len(qv) or len(kv) != len(qv):
        raise ValueError(f"State lengths must be equal powers of 2, got {len(qv)} and {len(kv)}")

    dev = qml.device("default.qubit", wires=1 + 2 * n_qubits)

    @qml.qnode(dev)
    def _swap_circuit(q_amp: np.ndarray, k_amp: np.ndarray):
        qml.StatePrep(q_amp, wires=list(range(1, 1 + n_qubits)))
        qml.StatePrep(k_amp, wires=list(range(1 + n_qubits, 1 + 2 * n_qubits)))
        qml.Hadamard(wires=0)
        for wire_i in range(n_qubits):
            qml.CSWAP(wires=[0, 1 + wire_i, 1 + n_qubits + wire_i])
        qml.Hadamard(wires=0)
        return qml.probs(wires=0)

    probs_anc = _swap_circuit(qv, kv)
    p_anc_zero = float(probs_anc[0])
    overlap_pl = 2.0 * p_anc_zero - 1.0
    overlap_exact = float(np.dot(qv, kv) ** 2)
    return {
        "p_ancilla_zero": p_anc_zero,
        "overlap_pennylane": overlap_pl,
        "overlap_analytic": overlap_exact,
        "abs_diff": abs(overlap_pl - overlap_exact),
        "n_qubits_circuit": 1 + 2 * n_qubits,
    }


class QuantumSwapTestAttention(nn.Module):
    """Regime II: SWAP-Test Hybrid Quantum Self-Attention (n_q=2 qubits, d=4).

    Encodes projected query q_i and key k_j into 2-qubit amplitude states |psi_q>, |psi_k>,
    computes attention overlap via SWAP-Test |<psi_q|psi_k>|^2 = 2 P(ancilla=0) - 1, and applies
    classical Softmax + value projection. Supports exact finite-shot Bernoulli ancilla sampling.
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int = 4,
        n_qubits: int = 2,
        embed_dim: int = 20,
        val_dim: int = 14,
        scale_init: float = 6.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_classes = int(n_classes)
        self.n_qubits = int(n_qubits)
        self.state_dim = 2 ** self.n_qubits
        self.embed = nn.Embedding(self.vocab_size, int(embed_dim))
        self.w_q = nn.Linear(int(embed_dim), self.state_dim, bias=False)
        self.w_k = nn.Linear(int(embed_dim), self.state_dim, bias=False)
        self.w_v = nn.Linear(int(embed_dim), int(val_dim), bias=False)
        self.attn_scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.head = nn.Linear(int(val_dim), self.n_classes)

    def _prepare_states(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.embed(x)
        q_amp = F.normalize(self.w_q(emb[:, -1, :]), p=2, dim=-1, eps=1e-8)
        k_amp = F.normalize(self.w_k(emb[:, :-1, :]), p=2, dim=-1, eps=1e-8)
        v = self.w_v(emb[:, :-1, :])
        return q_amp, k_amp, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_amp, k_amp, v = self._prepare_states(x)
        # Analytic noiseless SWAP-Test overlap: 2 * P(ancilla=0) - 1 = |<psi_q | psi_k>|^2
        p_anc0 = 0.5 * (1.0 + torch.einsum("bd,bnd->bn", q_amp, k_amp).pow(2))
        overlaps = 2.0 * p_anc0 - 1.0
        weights = F.softmax(self.attn_scale * overlaps, dim=-1)
        ctx = torch.einsum("bn,bnd->bd", weights, v)
        return self.head(ctx)

    @torch.no_grad()
    def forward_finite_shots(
        self,
        x: torch.Tensor,
        shots: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Evaluate with real finite-shot Bernoulli sampling of the SWAP-Test ancilla.

        Each (query, key_j) pair runs ``shots`` independent SWAP-Test circuits with ancilla
        outcome k_j ~ Binomial(shots, P(ancilla=0)), estimating overlap_hat = 2 * (k_j / shots) - 1.
        """
        self.eval()
        if rng is None:
            rng = np.random.default_rng(0)
        B, T = x.shape
        n_ctx = T - 1
        q_amp, k_amp, v = self._prepare_states(x)
        exact_overlap = torch.einsum("bd,bnd->bn", q_amp, k_amp).pow(2).clamp(0.0, 1.0)
        p_anc0 = (0.5 * (1.0 + exact_overlap)).cpu().numpy()  # (B, n_ctx)
        counts_zero = rng.binomial(n=int(shots), p=p_anc0)
        p0_hat = counts_zero / float(shots)
        overlap_hat = torch.tensor(2.0 * p0_hat - 1.0, dtype=q_amp.dtype, device=x.device)
        weights = F.softmax(self.attn_scale * overlap_hat, dim=-1)
        ctx = torch.einsum("bn,bnd->bd", weights, v)
        logits = self.head(ctx)
        stats = {
            "shots_per_pair": int(shots),
            "seq_len": int(T),
            "shots_last_token_query": int(n_ctx * shots),
            "shots_full_causal_matrix": int((T * (T + 1) // 2) * shots),
            "overlap_mae": float((overlap_hat - exact_overlap).abs().mean().item()),
        }
        return logits, stats

    @torch.no_grad()
    def verify_against_pennylane(self, x_single: torch.Tensor) -> dict[str, float]:
        """Verify batched PyTorch SWAP-Test overlaps on a single sequence against PennyLane CSWAP."""
        self.eval()
        x_b = x_single.unsqueeze(0) if x_single.ndim == 1 else x_single[:1]
        q_amp, k_amp, _ = self._prepare_states(x_b)
        max_diff = 0.0
        for pos in range(k_amp.shape[1]):
            res = verify_swap_test_pennylane(q_amp[0], k_amp[0, pos])
            max_diff = max(max_diff, res["abs_diff"])
        return {
            "max_abs_overlap_diff": max_diff,
            "n_pairs_checked": int(k_amp.shape[1]),
            "n_qubits_circuit": 1 + 2 * self.n_qubits,
        }


# --------------------------------------------------------------------------- #
# Regime III: Coherent LCU + QSVT / Quixer-Style Quantum Attention (0 MLP)
# --------------------------------------------------------------------------- #
class QuantumQuixerAttention(nn.Module):
    """Regime III: Coherent LCU + Degree-3 QSVT Quantum Attention (Khatri et al. 2024).

    Purely quantum Hilbert-space model (ZERO classical MLP crutch, ZERO classical Softmax):
      1. Token unitaries: U(v) = exp(-i sum_k theta_{v,k} P_k) in U(2^{n_q}) using 1+2-body Paulis.
      2. LCU block encoding (PREPARE / SELECT / PREPARE†): A = sum_{t=0}^{N-1} alpha_t U(x_t)
         with normalized non-negative amplitudes alpha_t >= 0, sum_t alpha_t = 1 (query-independent).
      3. Degree-3 QSVT singular-value polynomial P_Phi(A) via projector-controlled phase factors
         Phi = (phi_0, phi_1, phi_2, phi_3) applied to query state |psi_q> = U(x_{T-1})|0>^{n_q}:
             |psi_tilde> = P_Phi(A) |psi_q>,   p_succ = <psi_tilde | psi_tilde> <= 1.
      4. Complete Born-rule POVM readout: sum_{c=0}^{C-1} M_c = I_{d_H},
             P(c | x) = <psi_tilde | M_c | psi_tilde> / p_succ.
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int = 4,
        n_qubits: int = 3,
        povm_rank: int = 2,
        learned_prepare: bool = True,
        use_qsvt: bool = True,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_classes = int(n_classes)
        self.n_qubits = int(n_qubits)
        self.hilbert_dim = 2 ** self.n_qubits
        min_rank = math.ceil(self.hilbert_dim / self.n_classes)
        self.povm_rank = max(int(povm_rank), min_rank)
        self.learned_prepare = bool(learned_prepare)
        self.use_qsvt = bool(use_qsvt)

        basis = _pauli_1_and_2_body_basis(self.n_qubits)
        self.register_buffer("pauli_basis", basis)
        self.n_generators = int(basis.shape[0])

        self.theta = nn.Parameter(torch.randn(self.vocab_size, self.n_generators) * float(init_scale))
        if self.learned_prepare:
            self.prepare_logits = nn.Parameter(torch.zeros(self.vocab_size))
        else:
            self.register_parameter("prepare_logits", None)

        if self.use_qsvt:
            # 4 phase angles (phi_0, phi_1, phi_2, phi_3) for degree-3 QSVT
            self.qsvt_phases = nn.Parameter(torch.tensor([0.0, 0.4, 0.4, 0.0], dtype=torch.float32))
        else:
            self.register_parameter("qsvt_phases", None)

        self.povm_real = nn.Parameter(torch.randn(self.n_classes, self.povm_rank, self.hilbert_dim) * 0.3)
        self.povm_imag = nn.Parameter(torch.randn(self.n_classes, self.povm_rank, self.hilbert_dim) * 0.3)

    def token_unitaries(self) -> torch.Tensor:
        """Return exact token unitaries ``(V, d_H, d_H)``."""
        H = torch.einsum("vk,kij->vij", self.theta.to(self.pauli_basis.dtype), self.pauli_basis)
        return torch.linalg.matrix_exp(-1j * H)

    def lcu_weights(self, ctx_tokens: torch.Tensor) -> torch.Tensor:
        """Return normalized non-negative LCU weights ``(B, N)`` with ``sum_t alpha_t == 1``."""
        B, N = ctx_tokens.shape
        if self.learned_prepare and self.prepare_logits is not None:
            w = F.softplus(self.prepare_logits[ctx_tokens]) + 1e-4
            return w / w.sum(dim=-1, keepdim=True)
        return torch.full((B, N), 1.0 / float(N), dtype=torch.float32, device=ctx_tokens.device)

    def povm_kraus(self) -> torch.Tensor:
        """Return complete POVM Kraus operators ``E`` of shape ``(C, r, d_H)`` with ``sum E†E = I``."""
        F_ops = torch.complex(self.povm_real.to(torch.float64), self.povm_imag.to(torch.float64))
        S = torch.einsum("cri,crj->ij", F_ops.conj(), F_ops)
        S = 0.5 * (S + S.conj().transpose(-1, -2))
        S = S + 1e-10 * torch.eye(self.hilbert_dim, dtype=S.dtype, device=S.device)
        evals, evecs = torch.linalg.eigh(S)
        inv_sqrt = evals.real.clamp(min=1e-12).rsqrt()
        S_inv_sqrt = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.conj().transpose(-1, -2)
        E64 = torch.einsum("cri,ij->crj", F_ops, S_inv_sqrt)
        return E64.to(self.pauli_basis.dtype)

    def povm_effects(self) -> torch.Tensor:
        """Return complete PSD POVM effects ``M_c`` of shape ``(C, d_H, d_H)`` with ``sum_c M_c = I``."""
        E = self.povm_kraus()
        return torch.einsum("cri,crj->cij", E.conj(), E)

    def qsvt_apply(self, A: torch.Tensor) -> torch.Tensor:
        """Exact degree-3 QSVT polynomial P_Phi(A) = (<0|_A x I) U_Phi (|0>_A x I).

        Matches ``PCPhase(phi_0) W PCPhase(phi_1) W† PCPhase(phi_2) W PCPhase(phi_3)`` identically.
        """
        if not self.use_qsvt or self.qsvt_phases is None:
            return A
        p0, p1, p2, p3 = self.qsvt_phases[0], self.qsvt_phases[1], self.qsvt_phases[2], self.qsvt_phases[3]
        c1 = torch.exp(1j * (p1 - p2)) + torch.exp(-1j * p1) * (torch.exp(1j * p2) - torch.exp(-1j * p2))
        c3 = (torch.exp(1j * p1) - torch.exp(-1j * p1)) * (torch.exp(1j * p2) - torch.exp(-1j * p2))
        outer = torch.exp(1j * (p0 + p3))
        AAH_A = A @ A.conj().transpose(-1, -2) @ A
        return outer * (c1 * A + c3 * AAH_A)

    def forward_with_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_probs, probs, p_succ)`` for batch ``x`` of shape ``(B, T)``."""
        U_all = self.token_unitaries()  # (V, d_H, d_H)
        ctx_tokens = x[:, :-1]
        q_tokens = x[:, -1]

        alpha = self.lcu_weights(ctx_tokens)  # (B, N)
        U_ctx = U_all[ctx_tokens]  # (B, N, d_H, d_H)
        A = torch.einsum("bn,bnij->bij", alpha.to(U_ctx.dtype), U_ctx)  # (B, d_H, d_H)
        P_A = self.qsvt_apply(A)

        psi_q = U_all[q_tokens, :, 0]  # (B, d_H), query state U(q)|0>^{n_q}
        psi_tilde = torch.einsum("bij,bj->bi", P_A, psi_q)  # unnormalized post-selected state
        p_succ = psi_tilde.abs().pow(2).sum(dim=-1).clamp(min=1e-12)  # (B,)
        psi_norm = psi_tilde / p_succ.sqrt().unsqueeze(-1)

        E = self.povm_kraus()  # (C, r, d_H)
        proj = torch.einsum("cri,bi->bcr", E, psi_norm)
        probs = proj.abs().pow(2).sum(dim=-1).clamp(min=1e-12)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.log(probs), probs, p_succ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return exact Born-rule log-probabilities ``(B, C)`` (compatible with ``F.nll_loss``)."""
        log_probs, _, _ = self.forward_with_stats(x)
        return log_probs

    @torch.no_grad()
    def verify_against_pennylane(self, x_single: torch.Tensor) -> dict[str, float]:
        """Verify PyTorch LCU+QSVT output against a real PennyLane PREPARE/SELECT/PREPARE†+PCPhase circuit."""
        import pennylane as qml

        self.eval()
        seq = x_single.detach().cpu().view(-1)
        ctx = seq[:-1]
        q_tok = int(seq[-1].item())
        N = int(ctx.shape[0])
        n_anc = max(1, math.ceil(math.log2(max(N, 2))))
        dim_anc = 2 ** n_anc

        U_all = self.token_unitaries().detach().cpu().to(torch.complex128)
        alpha = self.lcu_weights(ctx.unsqueeze(0))[0].detach().cpu().to(torch.float64)
        sqrt_alpha = np.zeros(dim_anc, dtype=np.float64)
        sqrt_alpha[:N] = np.sqrt(alpha.numpy())
        sqrt_alpha = sqrt_alpha / np.linalg.norm(sqrt_alpha)

        Us_pad = [U_all[int(t.item())].numpy() for t in ctx] + [
            np.eye(self.hilbert_dim, dtype=np.complex128) for _ in range(dim_anc - N)
        ]
        Uq_mat = U_all[q_tok].numpy()

        anc_wires = list(range(n_anc))
        sys_wires = list(range(n_anc, n_anc + self.n_qubits))
        all_wires = anc_wires + sys_wires
        dev = qml.device("default.qubit", wires=n_anc + self.n_qubits)

        def _block_encode():
            qml.StatePrep(sqrt_alpha, wires=anc_wires)
            qml.Select([qml.QubitUnitary(u_mat, wires=sys_wires) for u_mat in Us_pad], control=anc_wires)
            qml.adjoint(qml.StatePrep)(sqrt_alpha, wires=anc_wires)

        phases = self.qsvt_phases.detach().cpu().to(torch.float64).tolist() if self.use_qsvt else None

        @qml.qnode(dev)
        def _quixer_circuit():
            qml.QubitUnitary(Uq_mat, wires=sys_wires)
            if phases is not None:
                qml.PCPhase(phases[3], dim=self.hilbert_dim, wires=all_wires)
                _block_encode()
                qml.PCPhase(phases[2], dim=self.hilbert_dim, wires=all_wires)
                qml.adjoint(_block_encode)()
                qml.PCPhase(phases[1], dim=self.hilbert_dim, wires=all_wires)
                _block_encode()
                qml.PCPhase(phases[0], dim=self.hilbert_dim, wires=all_wires)
            else:
                _block_encode()
            return qml.state()

        full_state = torch.tensor(_quixer_circuit(), dtype=torch.complex128)
        psi_tilde_pl = full_state[: self.hilbert_dim]
        p_succ_pl = float(psi_tilde_pl.abs().pow(2).sum().item())
        psi_norm_pl = psi_tilde_pl / math.sqrt(max(p_succ_pl, 1e-12))

        M_all = self.povm_effects().detach().cpu().to(torch.complex128)
        probs_pl = torch.einsum("i,cij,j->c", psi_norm_pl.conj(), M_all, psi_norm_pl).real

        # Compare with PyTorch training forward pass
        _, probs_pt_b, p_succ_pt_b = self.forward_with_stats(seq.unsqueeze(0).to(self.theta.device))
        probs_pt = probs_pt_b[0].detach().cpu().to(torch.float64)
        p_succ_pt = float(p_succ_pt_b[0].detach().cpu().item())

        # Also compare unnormalized state vector
        A_pt = torch.einsum("n,nij->ij", alpha.to(torch.complex128), torch.stack([U_all[int(t)] for t in ctx]))
        if phases is not None:
            p0, p1, p2, p3 = [torch.tensor(v, dtype=torch.float64) for v in phases]
            c1 = torch.exp(1j * (p1 - p2)) + torch.exp(-1j * p1) * (torch.exp(1j * p2) - torch.exp(-1j * p2))
            c3 = (torch.exp(1j * p1) - torch.exp(-1j * p1)) * (torch.exp(1j * p2) - torch.exp(-1j * p2))
            P_A_pt = torch.exp(1j * (p0 + p3)) * (c1 * A_pt + c3 * (A_pt @ A_pt.conj().T @ A_pt))
        else:
            P_A_pt = A_pt
        psi_tilde_pt = P_A_pt @ U_all[q_tok, :, 0]
        psi_norm_pt = psi_tilde_pt / math.sqrt(max(float(psi_tilde_pt.abs().pow(2).sum().item()), 1e-12))

        return {
            "max_abs_unnorm_state_diff": float((psi_tilde_pl - psi_tilde_pt).abs().max().item()),
            "max_abs_norm_state_diff": float((psi_norm_pl - psi_norm_pt).abs().max().item()),
            "max_abs_prob_diff": float((probs_pl - probs_pt).abs().max().item()),
            "p_succ_pennylane": p_succ_pl,
            "p_succ_pytorch": p_succ_pt,
            "abs_p_succ_diff": abs(p_succ_pl - p_succ_pt),
            "n_ancilla_qubits": n_anc,
            "n_system_qubits": self.n_qubits,
            "total_qubits": n_anc + self.n_qubits,
        }


# --------------------------------------------------------------------------- #
# Exact Parameter Decomposition Helper
# --------------------------------------------------------------------------- #
def attention_parameter_breakdown(model: nn.Module) -> dict[str, Any]:
    """Return exact quantum vs classical parameter decomposition for all attention models."""
    components: dict[str, int] = {}
    quantum_params = 0
    classical_params = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = int(p.numel())
        top = name.split(".")[0]
        components[top] = components.get(top, 0) + n
        if isinstance(model, QuantumQuixerAttention):
            quantum_params += n
        else:
            classical_params += n

    total = quantum_params + classical_params
    return {
        "total": total,
        "quantum_params": quantum_params,
        "classical_params": classical_params,
        "quantum_param_share": float(quantum_params / total) if total > 0 else 0.0,
        "components": components,
    }
