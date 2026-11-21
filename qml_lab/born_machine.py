"""Regime III Track 1: Autoregressive Tensor-Network Born Machine (MPS).

Compares genuine quantum amplitude interference (signed / complex Matrix Product State
wave function with Born-rule probabilities P(x) = |<x|psi>|^2 / <psi|psi>) against:
1. ``ClassicalNonNegMPS``: strict single-variable control with IDENTICAL bond dimension
   ``chi``, tensor topology, and parameter count, but strictly non-negative tensor entries
   (disabling destructive interference).
2. ``ClassicalCausalRNNLM``: parameter-matched classical autoregressive RNN baseline.

Exact transfer-matrix contraction yields exact normalization (sum_x P(x) == 1), exact
autoregressive conditionals P(x_t | x_{<t}), exact ancestral sampling, and OOD length
generalization to arbitrary sequence lengths T.
"""
from __future__ import annotations

import math
import random
from itertools import product

import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "QuantumMPSBornMachine",
    "ClassicalNonNegMPS",
    "ClassicalCausalRNNLM",
    "generate_periodic_interference_dataset",
    "generate_dyck1_dataset",
    "is_valid_dyck1",
]


class QuantumMPSBornMachine(nn.Module):
    """Translation-invariant Matrix Product State Born Machine (exact normalization).

    Site-shared tensor ``A`` of shape ``(V, chi, chi)`` plus boundary vectors
    ``left (chi,)`` and ``right (chi,)`` define an unnormalized quantum state
    ``<x_1...x_T | psi_T> = left^T A[x_1] ... A[x_T] right`` for ANY length ``T``.
    Exact Born probabilities ``P(x) = |<x|psi_T>|^2 / <psi_T|psi_T>`` are computed
    via exact right-environment transfer-matrix contraction in O(T * V * chi^3).
    """

    def __init__(
        self,
        vocab_size: int,
        bond_dim: int = 4,
        *,
        complex_amplitudes: bool = False,
        init_scale: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.bond_dim = int(bond_dim)
        self.complex_amplitudes = bool(complex_amplitudes)

        chi = self.bond_dim
        V = self.vocab_size
        # Initialize near scaled identity matrices so the transfer operator starts well-conditioned
        a_init = torch.randn(V, chi, chi) * (float(init_scale) / math.sqrt(chi))
        eye = torch.eye(chi).unsqueeze(0).expand(V, -1, -1) / math.sqrt(V)
        self.A_real = nn.Parameter(eye + a_init)
        left_init = torch.zeros(chi)
        left_init[0] = 1.0
        right_init = torch.zeros(chi)
        right_init[0] = 1.0
        self.left_real = nn.Parameter(left_init + 0.05 * torch.randn(chi))
        self.right_real = nn.Parameter(right_init + 0.05 * torch.randn(chi))

        if self.complex_amplitudes:
            self.A_imag = nn.Parameter(torch.randn(V, chi, chi) * (float(init_scale) * 0.25 / math.sqrt(chi)))
            self.left_imag = nn.Parameter(0.05 * torch.randn(chi))
            self.right_imag = nn.Parameter(0.05 * torch.randn(chi))
        else:
            self.register_parameter("A_imag", None)
            self.register_parameter("left_imag", None)
            self.register_parameter("right_imag", None)

    def _effective_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.complex_amplitudes:
            A = torch.complex(self.A_real, self.A_imag)
            left = torch.complex(self.left_real, self.left_imag)
            right = torch.complex(self.right_real, self.right_imag)
            return A, left, right
        return self.A_real, self.left_real, self.right_real

    def right_environments(self, seq_len: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Compute normalized right Gram matrices R_0..R_T and exact log Z_T = log <psi_T|psi_T>.

        R_T = r r^dagger; R_{k} = sum_s A[s] R_{k+1} A[s]^dagger, rescaled by Tr(R_k) at each step.
        """
        A, left, right = self._effective_tensors()
        # R_T = right outer right*
        R = torch.outer(right, right.conj())
        tr_T = R.diagonal().real.sum().clamp(min=1e-12)
        R = R / tr_T
        log_scales = [torch.log(tr_T)]
        envs = [R]  # will store [R_T, R_{T-1}, ..., R_0] then reverse

        A_conj = A.conj()
        for _ in range(int(seq_len)):
            # R_prev[i, j] = sum_{s, a, b} A[s, i, a] R[a, b] A*[s, j, b]
            AR = torch.einsum("sia,ab->sib", A, R)
            R_prev = torch.einsum("sib,sjb->ij", AR, A_conj)
            # Enforce Hermiticity numerically
            R_prev = 0.5 * (R_prev + R_prev.conj().transpose(-1, -2))
            tr_k = R_prev.diagonal().real.sum().clamp(min=1e-12)
            R = R_prev / tr_k
            log_scales.append(torch.log(tr_k))
            envs.append(R)

        envs.reverse()  # envs[t] = R_t for t = 0..T
        # Z_T = (left^dagger R_0 left) * prod_k tr_k  (note left is row vector: l^T A[x1]... => l^T R_0 l*)
        # Because <x|psi> = l^T A[x1]...A[xT] r, |<x|psi>|^2 = sum l_i A[x1]_{i,a} ... l*_j A*[x1]_{j,b}
        quad = torch.einsum("i,ij,j->", left, envs[0], left.conj()).real.clamp(min=1e-12)
        log_z = torch.log(quad) + torch.stack(log_scales).sum()
        return envs, log_z

    def autoregressive_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Return exact conditional log-probabilities ``(B, T, V)``: ``log P(x_t = s | x_{<t})``."""
        B, T = x.shape
        A, left, _ = self._effective_tensors()
        envs, _ = self.right_environments(T)

        v = left.unsqueeze(0).expand(B, -1)  # (B, chi)
        v = v / v.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-12)

        cond_log_probs = []
        for t in range(T):
            # Candidate next vectors for every symbol s in 0..V-1: (B, V, chi)
            v_cand = torch.einsum("bi,sij->bsj", v, A)
            R_next = envs[t + 1]  # (chi, chi) PSD
            # Unnormalized probabilities: (v_cand R_next v_cand^*) >= 0
            probs_unnorm = torch.einsum("bsi,ij,bsj->bs", v_cand, R_next, v_cand.conj()).real
            probs_unnorm = probs_unnorm.clamp(min=1e-12)
            probs = probs_unnorm / probs_unnorm.sum(dim=-1, keepdim=True)
            cond_log_probs.append(torch.log(probs.clamp(min=1e-12)))

            # Advance left state along actual observed token x[:, t]
            idx = x[:, t]  # (B,)
            v = v_cand[torch.arange(B, device=x.device), idx]
            v = v / v.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-12)

        return torch.stack(cond_log_probs, dim=1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Exact joint Born-rule log probability ``log P(x_{1:T})`` of shape ``(B,)``."""
        log_cond = self.autoregressive_log_probs(x)  # (B, T, V)
        picked = log_cond.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1)  # (B, T)
        return picked.sum(dim=-1)

    def nll_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Exact mean per-token negative log-likelihood."""
        return -self.log_prob(x).mean() / float(x.shape[1])

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Exact ancestral autoregressive sampling from the Born-rule distribution."""
        self.eval()
        A, left, _ = self._effective_tensors()
        envs, _ = self.right_environments(seq_len)

        device = A.device
        v = left.unsqueeze(0).expand(batch_size, -1)
        v = v / v.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-12)
        tokens = []
        for t in range(int(seq_len)):
            v_cand = torch.einsum("bi,sij->bsj", v, A)
            R_next = envs[t + 1]
            probs_unnorm = torch.einsum("bsi,ij,bsj->bs", v_cand, R_next, v_cand.conj()).real
            probs_unnorm = probs_unnorm.clamp(min=1e-12)
            probs = probs_unnorm / probs_unnorm.sum(dim=-1, keepdim=True)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
            tokens.append(nxt)
            v = v_cand[torch.arange(batch_size, device=device), nxt]
            v = v / v.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-12)
        return torch.stack(tokens, dim=1)


class ClassicalNonNegMPS(QuantumMPSBornMachine):
    """Strict single-variable classical control for ``QuantumMPSBornMachine``.

    IDENTICAL bond dimension ``chi``, IDENTICAL tensor topology, IDENTICAL parameter count,
    and IDENTICAL exact transfer-matrix contraction — ONLY enforces strictly non-negative
    entries via ``softplus``, disabling quantum destructive amplitude interference.
    """

    def __init__(self, vocab_size: int, bond_dim: int = 4, *, init_scale: float = 0.5):
        super().__init__(
            vocab_size=vocab_size,
            bond_dim=bond_dim,
            complex_amplitudes=False,
            init_scale=init_scale,
        )
        # Fair initialization via exact softplus^{-1} so effective non-negative tensors
        # start near sparse scaled identity (NOT dense softplus(0)=0.693 all-ones!).
        def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
            y = y.clamp(min=1e-4)
            return y + torch.log(-torch.expm1(-y))

        chi = self.bond_dim
        V = self.vocab_size
        with torch.no_grad():
            eye = torch.eye(chi).unsqueeze(0).expand(V, -1, -1) / math.sqrt(V)
            a_pos = (eye + torch.rand(V, chi, chi) * (0.25 / math.sqrt(chi))).clamp(min=0.02)
            l_pos = torch.full((chi,), 0.05)
            l_pos[0] = 1.0
            r_pos = torch.full((chi,), 0.05)
            r_pos[0] = 1.0
            self.A_real.copy_(_inv_softplus(a_pos))
            self.left_real.copy_(_inv_softplus(l_pos))
            self.right_real.copy_(_inv_softplus(r_pos))

    def _effective_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return F.softplus(self.A_real), F.softplus(self.left_real), F.softplus(self.right_real)


class ClassicalCausalRNNLM(nn.Module):
    """Parameter-matched classical autoregressive RNN baseline for Track 1 formal languages."""

    def __init__(self, vocab_size: int, embed_dim: int = 4, hidden_dim: int = 8):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.bos_id = self.vocab_size
        self.embed = nn.Embedding(self.vocab_size + 1, int(embed_dim))
        self.cell = nn.RNNCell(int(embed_dim), int(hidden_dim))
        self.head = nn.Linear(int(hidden_dim), self.vocab_size)
        self.hidden_dim = int(hidden_dim)

    def autoregressive_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device, dtype=torch.float32)
        inp = torch.full((B,), self.bos_id, dtype=torch.long, device=x.device)
        out = []
        for t in range(T):
            h = self.cell(self.embed(inp), h)
            out.append(F.log_softmax(self.head(h), dim=-1))
            inp = x[:, t]
        return torch.stack(out, dim=1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_cond = self.autoregressive_log_probs(x)
        return log_cond.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

    def nll_loss(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x).mean() / float(x.shape[1])

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        self.eval()
        device = self.head.weight.device
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        inp = torch.full((batch_size,), self.bos_id, dtype=torch.long, device=device)
        tokens = []
        for _ in range(int(seq_len)):
            h = self.cell(self.embed(inp), h)
            probs = F.softmax(self.head(h), dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
            tokens.append(nxt)
            inp = nxt
        return torch.stack(tokens, dim=1)


# --------------------------------------------------------------------------- #
# Formal-language & quantum-interference datasets (strict Train/Val/Test/OOD)
# --------------------------------------------------------------------------- #
def _periodic_exact_distribution(seq_len: int, modulus: int = 4) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Exact Born distribution P*(x) proportional to cos^2(pi * sum(x) / modulus) on {0,1}^T."""
    all_x = torch.tensor(list(product((0, 1), repeat=seq_len)), dtype=torch.long)
    sums = all_x.sum(dim=-1).to(torch.float64)
    weights = torch.cos(math.pi * sums / float(modulus)).pow(2)
    probs = weights / weights.sum()
    # Exact Bayes entropy per token (nats)
    nz = probs[probs > 1e-15]
    bayes_nll = float(-(nz * torch.log(nz)).sum().item() / seq_len)
    return all_x, probs.to(torch.float32), bayes_nll


def generate_periodic_interference_dataset(
    modulus: int = 4,
    train_len: int = 8,
    train_lens: tuple[int, ...] | None = None,
    ood_lens: tuple[int, ...] = (12, 14),
    n_train: int = 1200,
    n_val: int = 400,
    n_test: int = 600,
    seed: int = 0,
) -> dict:
    """Quantum-native periodic interference sequence distribution P*(x) ∝ cos^2(pi/M * sum x_t).

    Exact destructive interference sets P*(x) = 0 whenever sum(x) == M/2 (mod M).
    By the Perron-Frobenius theorem, a bond_dim=2 signed Quantum MPS represents this
    process EXACTLY for all lengths T (via SO(2) rotation R(pi/M)), whereas a bond_dim=2
    strictly non-negative MPS cannot sustain period-M >= 3 oscillation.

    When ``train_lens`` has multiple lengths (e.g. ``(7, 8)``), the global phase gauge on
    symbol ``0`` is broken so any translation-invariant model can generalize cleanly to
    arbitrary OOD lengths ``T``.
    """
    g = torch.Generator().manual_seed(int(seed))
    all_x, probs, bayes_id = _periodic_exact_distribution(train_len, modulus)

    def _draw(n: int, x_pool: torch.Tensor, p_pool: torch.Tensor) -> torch.Tensor:
        idx = torch.multinomial(p_pool, num_samples=n, replacement=True, generator=g)
        return x_pool[idx]

    train_x = _draw(n_train, all_x, probs)
    val_x = _draw(n_val, all_x, probs)
    test_id_x = _draw(n_test, all_x, probs)

    train_by_len: dict[int, torch.Tensor] = {}
    val_by_len: dict[int, torch.Tensor] = {}
    if train_lens is not None:
        n_each_tr = max(1, n_train // len(train_lens))
        n_each_va = max(1, n_val // len(train_lens))
        for L in train_lens:
            lx, lp, _ = _periodic_exact_distribution(int(L), modulus)
            train_by_len[int(L)] = _draw(n_each_tr, lx, lp)
            val_by_len[int(L)] = _draw(n_each_va, lx, lp)
    else:
        train_by_len[int(train_len)] = train_x
        val_by_len[int(train_len)] = val_x

    ood: dict[int, dict] = {}
    for L in ood_lens:
        ox, op, bayes_l = _periodic_exact_distribution(L, modulus)
        ood[int(L)] = {
            "x": _draw(n_test, ox, op),
            "all_x": ox,
            "probs": op,
            "bayes_nll_per_token": bayes_l,
        }

    return {
        "vocab_size": 2,
        "modulus": modulus,
        "train_len": train_len,
        "train_lens": list(train_by_len.keys()),
        "train": train_x,
        "val": val_x,
        "train_by_len": train_by_len,
        "val_by_len": val_by_len,
        "test_id": test_id_x,
        "id_all_x": all_x,
        "id_probs": probs,
        "bayes_nll_id": bayes_id,
        "ood": ood,
    }


def is_valid_dyck1(seq: list[int] | tuple[int, ...] | torch.Tensor) -> bool:
    """Return True iff binary sequence (0='(', 1=')') is a balanced Dyck-1 string."""
    bal = 0
    for tok in seq:
        bal += 1 if int(tok) == 0 else -1
        if bal < 0:
            return False
    return bal == 0


def _enumerate_dyck1(length: int) -> list[tuple[int, ...]]:
    """Enumerate all valid Dyck-1 strings of even length ``length``."""
    if length % 2 != 0:
        raise ValueError(f"Dyck-1 长度必须为偶数，收到 {length}")
    out: list[tuple[int, ...]] = []
    half = length // 2

    def dfs(prefix: list[int], n_open: int, n_close: int) -> None:
        if len(prefix) == length:
            out.append(tuple(prefix))
            return
        if n_open < half:
            prefix.append(0)
            dfs(prefix, n_open + 1, n_close)
            prefix.pop()
        if n_close < n_open:
            prefix.append(1)
            dfs(prefix, n_open, n_close + 1)
            prefix.pop()

    dfs([], 0, 0)
    return out


def generate_dyck1_dataset(
    train_len: int = 14,
    ood_len: int = 16,
    seed: int = 0,
) -> dict:
    """Strictly disjoint Train/Val/Test split of Dyck-1 balanced brackets + OOD length split.

    At ``train_len=14``, there are C_7 = 429 unique valid strings (split 260 / 84 / 85 with
    ZERO sequence overlap!). At ``ood_len=16``, there are C_8 = 1,430 valid strings (200 held
    out for OOD length generalization).
    """
    rng = random.Random(int(seed))
    id_all = _enumerate_dyck1(train_len)
    rng.shuffle(id_all)
    n_tr = int(round(len(id_all) * 0.60))
    n_va = int(round(len(id_all) * 0.20))
    train_s = id_all[:n_tr]
    val_s = id_all[n_tr : n_tr + n_va]
    test_s = id_all[n_tr + n_va :]
    assert not (set(train_s) & set(val_s))
    assert not (set(train_s) & set(test_s))
    assert not (set(val_s) & set(test_s))

    ood_all = _enumerate_dyck1(ood_len)
    rng.shuffle(ood_all)
    ood_s = ood_all[:200]

    return {
        "vocab_size": 2,
        "train_len": train_len,
        "ood_len": ood_len,
        "n_unique_id": len(id_all),
        "bayes_nll_id": math.log(len(id_all)) / train_len,
        "bayes_nll_ood": math.log(len(ood_all)) / ood_len,
        "train": torch.tensor(train_s, dtype=torch.long),
        "val": torch.tensor(val_s, dtype=torch.long),
        "test_id": torch.tensor(test_s, dtype=torch.long),
        "test_ood": torch.tensor(ood_s, dtype=torch.long),
    }
