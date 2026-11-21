"""Regime III Track 2: True L2 Persistent Quantum-State Recurrence (QuantumUnitarySSM).

Unlike L0 (pointwise PQC) and L1 (measurement-collapsed classical scalar feedback),
``QuantumUnitarySSM`` maintains a coherent persistent quantum state |psi_t> in C^{2^{n_H}}
updated by token-conditioned unitaries U(x_t) = exp(-i H(x_t)) with ZERO intermediate
measurement collapse, emitting exact Born-rule POVM probabilities P(y|psi_t) = <psi_t|M_y|psi_t>
with NO classical MLP wrapper.

Includes matched controls:
* ``ClassicalOrthogonalSSM``: real norm-preserving SO(d_R) state evolution + squared readout
  (isolates complex quantum unitary + POVM structure from pure norm preservation);
* ``ClassicalRNNTracker``: parameter-matched classical ``nn.RNNCell`` baseline;
* ``CausalConvTracker``: fixed-window (k=4) L0 classical baseline;
plus zero-leakage benchmarks: Non-Abelian S_3 group word problem (OOD length generalization
T<=6 -> T=12, 16) and clean deterministic Delayed Recall (all K=4 sequences evaluated exactly).
"""
from __future__ import annotations

import itertools
import math
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "QuantumUnitarySSM",
    "ClassicalOrthogonalSSM",
    "ClassicalRNNTracker",
    "CausalConvTracker",
    "S3_ELEMENTS",
    "S3_GENERATORS",
    "build_s3_dataset",
    "build_deterministic_delayed_recall",
]


def _pauli_1_and_2_body_basis(n_qubits: int) -> torch.Tensor:
    """Return tensor ``(K, d, d)`` of all 1-body and 2-body traceless Hermitian Pauli strings."""
    I2 = np.eye(2, dtype=np.complex64)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    paulis = (X, Y, Z)

    def kron_list(mats: list[np.ndarray]) -> np.ndarray:
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    ops: list[np.ndarray] = []
    # 1-body Paulis: 3 * n
    for i in range(n_qubits):
        for p in paulis:
            mats = [I2] * n_qubits
            mats[i] = p
            ops.append(kron_list(mats))
    # 2-body Paulis: 9 * n * (n - 1) / 2
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            for p1 in paulis:
                for p2 in paulis:
                    mats = [I2] * n_qubits
                    mats[i] = p1
                    mats[j] = p2
                    ops.append(kron_list(mats))
    return torch.from_numpy(np.stack(ops, axis=0))  # (K, 2^n, 2^n)


class QuantumUnitarySSM(nn.Module):
    """True L2 coherent quantum state-space model with exact Born-rule POVM readout.

    * Persistent state ``|psi_t> in C^{2^{n_H}}``, initialized to ``|0...0>``.
    * Token update: ``|psi_t> = U(x_t) |psi_{t-1}>``, where
      ``U(x) = exp(-i * sum_k theta[x, k] P_k)`` is strictly unitary (1+2-body Pauli Hamiltonian).
    * Readout: exact POVM ``{M_c = E_c^dagger E_c}`` with ``sum_c M_c = I``, giving exact
      Born probabilities ``P(y=c | psi_t) = ||E_c |psi_t>||^2`` (ZERO softmax, ZERO MLP!).
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        n_qubits: int = 3,
        povm_rank: int = 1,
        init_std: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_classes = int(n_classes)
        self.n_qubits = int(n_qubits)
        self.hilbert_dim = 2 ** self.n_qubits
        # Complete POVM sum_c E_c^H E_c = I_d requires C * r >= d_H
        min_rank = math.ceil(self.hilbert_dim / self.n_classes)
        self.povm_rank = max(int(povm_rank), min_rank)

        basis = _pauli_1_and_2_body_basis(self.n_qubits)
        self.register_buffer("pauli_basis", basis)  # (K, d, d) complex64
        self.n_generators = int(basis.shape[0])

        # Per-token 1+2-body Pauli Hamiltonian coefficients (real)
        self.theta = nn.Parameter(torch.randn(self.vocab_size, self.n_generators) * float(init_std))
        # Learnable raw POVM Kraus operators F_c in C^{r x d} (real + imag)
        scale = 1.0 / (self.hilbert_dim ** 0.5)
        self.povm_real = nn.Parameter(
            torch.randn(self.n_classes, self.povm_rank, self.hilbert_dim) * scale
        )
        self.povm_imag = nn.Parameter(
            torch.randn(self.n_classes, self.povm_rank, self.hilbert_dim) * scale
        )

    def token_unitaries(self) -> torch.Tensor:
        """Compute exact unitary matrices ``U(x)`` of shape ``(V, d, d)`` (complex64)."""
        # H[v] = sum_k theta[v, k] * pauli_basis[k]
        H = torch.einsum("vk,kij->vij", self.theta.to(torch.complex64), self.pauli_basis)
        return torch.linalg.matrix_exp((-1j) * H)

    def povm_kraus(self) -> torch.Tensor:
        """Return normalized POVM Kraus operators ``E`` of shape ``(C, r, d)`` with ``sum_c E_c^H E_c = I``."""
        F_ops = torch.complex(self.povm_real, self.povm_imag)  # (C, r, d)
        # S = sum_c F_c^H F_c + eps I (strictly positive definite Hermitian)
        S = torch.einsum("cri,crj->ij", F_ops.conj(), F_ops)
        S = 0.5 * (S + S.conj().transpose(-1, -2))
        S = S + 1e-6 * torch.eye(self.hilbert_dim, dtype=S.dtype, device=S.device)
        evals, evecs = torch.linalg.eigh(S)
        inv_sqrt = (evals.clamp(min=1e-7).rsqrt()).to(S.dtype)
        S_inv_sqrt = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.conj().transpose(-1, -2)
        return torch.einsum("cri,ij->crj", F_ops, S_inv_sqrt)

    def evolve_states(self, x: torch.Tensor) -> torch.Tensor:
        """Evolve persistent coherent state ``|psi_t>`` for all ``t=1..T``, shape ``(B, T, d)``."""
        B, T = x.shape
        U = self.token_unitaries()  # (V, d, d)
        psi = torch.zeros(B, self.hilbert_dim, dtype=torch.complex64, device=x.device)
        psi[:, 0] = 1.0 + 0.0j  # |0...0>
        states = []
        for t in range(T):
            U_t = U[x[:, t]]  # (B, d, d)
            psi = torch.einsum("bij,bj->bi", U_t, psi)
            states.append(psi)
        return torch.stack(states, dim=1)  # (B, T, d)

    def forward_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Return exact Born-rule POVM probabilities ``(B, T, C)``, summing to 1."""
        states = self.evolve_states(x)  # (B, T, d)
        E = self.povm_kraus()  # (C, r, d)
        proj = torch.einsum("cri,bti->btcr", E, states)
        probs = (proj.real.pow(2) + proj.imag.pow(2)).sum(dim=-1)
        return probs.clamp(min=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return exact log-probabilities ``(B, T, C)`` (compatible with NLLLoss)."""
        return torch.log(self.forward_probs(x))

    @torch.no_grad()
    def verify_against_pennylane(self, seq: torch.Tensor) -> dict[str, float]:
        """Verify statevector & POVM probabilities against PennyLane ``default.qubit``."""
        import pennylane as qml

        seq_1d = seq.reshape(-1).long()
        U_all = self.token_unitaries().detach().cpu().numpy()
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev)
        def circuit():
            for tok in seq_1d.tolist():
                qml.QubitUnitary(U_all[tok], wires=range(self.n_qubits))
            return qml.state()

        psi_pl = np.asarray(circuit(), dtype=np.complex64)
        psi_pt = self.evolve_states(seq_1d.unsqueeze(0))[0, -1].detach().cpu().numpy()
        state_diff = float(np.max(np.abs(psi_pl - psi_pt)))

        E = self.povm_kraus().detach().cpu().numpy()
        probs_pl = np.array([np.sum(np.abs(E[c] @ psi_pl) ** 2) for c in range(self.n_classes)])
        probs_pt = self.forward_probs(seq_1d.unsqueeze(0))[0, -1].detach().cpu().numpy()
        prob_diff = float(np.max(np.abs(probs_pl - probs_pt)))
        return {
            "max_abs_state_diff": state_diff,
            "max_abs_prob_diff": prob_diff,
            "norm_final_state": float(np.linalg.norm(psi_pt)),
            "prob_sum": float(probs_pt.sum()),
        }


class ClassicalOrthogonalSSM(nn.Module):
    """Real orthogonal state-space control: h_t = O(x_t) h_{t-1}, O(x) in SO(d_R).

    Preserves state norm identically to ``QuantumUnitarySSM`` using real SO(d_R) rotations
    and an exact real orthogonal POVM / squared-projection readout, isolating complex quantum
    phase structure from classical norm preservation.
    At ``state_dim=9``, each skew-symmetric generator has ``9*8/2 = 36`` parameters —
    EXACTLY matching 3-qubit 1+2-body Pauli generators (36 params/token).
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        state_dim: int = 9,
        readout_rank: int = 2,
        init_std: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_classes = int(n_classes)
        self.state_dim = int(state_dim)
        min_rank = math.ceil(self.state_dim / self.n_classes)
        self.readout_rank = max(int(readout_rank), min_rank)

        idx = torch.triu_indices(self.state_dim, self.state_dim, offset=1)
        self.register_buffer("triu_i", idx[0])
        self.register_buffer("triu_j", idx[1])
        self.n_skew = int(idx.shape[1])

        self.skew_params = nn.Parameter(
            torch.randn(self.vocab_size, self.n_skew) * float(init_std)
        )
        scale = 1.0 / (self.state_dim ** 0.5)
        self.readout_raw = nn.Parameter(
            torch.randn(self.n_classes, self.readout_rank, self.state_dim) * scale
        )

    def token_orthogonals(self) -> torch.Tensor:
        K = torch.zeros(
            self.vocab_size, self.state_dim, self.state_dim,
            dtype=self.skew_params.dtype, device=self.skew_params.device,
        )
        K[:, self.triu_i, self.triu_j] = self.skew_params
        K[:, self.triu_j, self.triu_i] = -self.skew_params
        return torch.linalg.matrix_exp(K)

    def readout_ops(self) -> torch.Tensor:
        F_ops = self.readout_raw  # (C, r, d)
        S = torch.einsum("cri,crj->ij", F_ops, F_ops)
        S = 0.5 * (S + S.transpose(-1, -2))
        S = S + 1e-6 * torch.eye(self.state_dim, dtype=S.dtype, device=S.device)
        evals, evecs = torch.linalg.eigh(S)
        inv_sqrt = evals.clamp(min=1e-7).rsqrt()
        S_inv_sqrt = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.transpose(-1, -2)
        return torch.einsum("cri,ij->crj", F_ops, S_inv_sqrt)

    def forward_probs(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        orth = self.token_orthogonals()
        h = torch.zeros(B, self.state_dim, dtype=orth.dtype, device=x.device)
        h[:, 0] = 1.0
        states = []
        for t in range(T):
            h = torch.einsum("bij,bj->bi", orth[x[:, t]], h)
            states.append(h)
        S_all = torch.stack(states, dim=1)  # (B, T, d)
        E = self.readout_ops()  # (C, r, d)
        proj = torch.einsum("cri,bti->btcr", E, S_all)
        return proj.pow(2).sum(dim=-1).clamp(min=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.forward_probs(x))


class ClassicalRNNTracker(nn.Module):
    """Parameter-matched classical ``nn.RNNCell`` state-tracking baseline."""

    def __init__(self, vocab_size: int, n_classes: int, embed_dim: int = 8, hidden_dim: int = 9):
        super().__init__()
        self.embed = nn.Embedding(int(vocab_size), int(embed_dim))
        self.cell = nn.RNNCell(int(embed_dim), int(hidden_dim))
        self.head = nn.Linear(int(hidden_dim), int(n_classes))
        self.hidden_dim = int(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        out = []
        for t in range(T):
            h = self.cell(self.embed(x[:, t]), h)
            out.append(F.log_softmax(self.head(h), dim=-1))
        return torch.stack(out, dim=1)


class CausalConvTracker(nn.Module):
    """Fixed-window (L0, k=4) classical causal Conv1d baseline — cannot track state beyond k."""

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        embed_dim: int = 6,
        hidden_dim: int = 8,
        context_window: int = 4,
    ):
        super().__init__()
        self.context_window = int(context_window)
        self.embed = nn.Embedding(int(vocab_size), int(embed_dim))
        self.conv = nn.Conv1d(int(embed_dim), int(hidden_dim), kernel_size=self.context_window)
        self.head = nn.Linear(int(hidden_dim), int(n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embed(x).transpose(1, 2)  # (B, E, T)
        if self.context_window > 1:
            e = F.pad(e, (self.context_window - 1, 0))
        h = torch.tanh(self.conv(e)).transpose(1, 2)
        return F.log_softmax(self.head(h), dim=-1)


# --------------------------------------------------------------------------- #
# Benchmarks: Non-Abelian S_3 Word Problem & Clean Deterministic Delayed Recall
# --------------------------------------------------------------------------- #
S3_ELEMENTS: tuple[tuple[int, int, int], ...] = tuple(itertools.permutations((0, 1, 2)))
_S3_TO_ID = {p: i for i, p in enumerate(S3_ELEMENTS)}
# Generators: 0 = identity (0,1,2), 1 = swap s=(1,0,2), 2 = 3-cycle r=(1,2,0)
S3_GENERATORS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (1, 0, 2),
    (1, 2, 0),
)


def _compose_perm(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    """Group composition (a o b)(i) = a[b[i]]."""
    return (a[b[0]], a[b[1]], a[b[2]])


def _s3_targets(seqs: list[tuple[int, ...]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.tensor(seqs, dtype=torch.long)
    ys = []
    for s in seqs:
        g = (0, 1, 2)
        row = []
        for tok in s:
            g = _compose_perm(S3_GENERATORS[tok], g)
            row.append(_S3_TO_ID[g])
        ys.append(row)
    return xs, torch.tensor(ys, dtype=torch.long)


def build_s3_dataset(
    train_len: int = 6,
    ood_lens: tuple[int, ...] = (12, 16),
    n_train: int = 360,
    n_val: int = 120,
    n_test_id: int = 160,
    n_test_ood: int = 200,
    seed: int = 0,
) -> dict:
    """Zero-leakage Non-Abelian S_3 group state-tracking dataset with OOD length splits.

    Input tokens in {0=e, 1=s, 2=r}; target at every position t is exact group product
    g_t = x_t o ... o x_1 in S_3 (6 classes).
    All 3^6 = 729 unique length-6 sequences are partitioned disjointly into Train / Val / Test-ID;
    OOD test sets evaluate length generalization at T=12 and T=16 (2x-2.7x training length).
    """
    rng = random.Random(int(seed))
    all_len6 = list(itertools.product((0, 1, 2), repeat=train_len))
    rng.shuffle(all_len6)
    assert n_train + n_val + n_test_id <= len(all_len6)
    tr_s = all_len6[:n_train]
    va_s = all_len6[n_train : n_train + n_val]
    te_s = all_len6[n_train + n_val : n_train + n_val + n_test_id]
    assert not (set(tr_s) & set(va_s))
    assert not (set(tr_s) & set(te_s))
    assert not (set(va_s) & set(te_s))

    x_tr, y_tr = _s3_targets(tr_s)
    x_va, y_va = _s3_targets(va_s)
    x_te, y_te = _s3_targets(te_s)

    ood: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for L in ood_lens:
        seen: set[tuple[int, ...]] = set()
        seqs: list[tuple[int, ...]] = []
        while len(seqs) < n_test_ood:
            s = tuple(rng.randrange(3) for _ in range(L))
            if s not in seen:
                seen.add(s)
                seqs.append(s)
        ood[int(L)] = _s3_targets(seqs)

    return {
        "vocab_size": 3,
        "n_classes": 6,
        "train_len": train_len,
        "train": (x_tr, y_tr),
        "val": (x_va, y_va),
        "test_id": (x_te, y_te),
        "ood": ood,
    }


def build_deterministic_delayed_recall(
    k: int = 4,
    pad_len: int = 4,
    ood_pad_len: int = 8,
    train_pad_lens: tuple[int, ...] = (2, 3, 4),
) -> dict:
    """Clean deterministic Delayed Recall dataset evaluating ALL K distinct sequences.

    Fixes ``recurrent_pilot.py``'s 32-random-draw eval imbalance (which caused constant
    predictors to fluctuate between 0.1875 and 0.40625). Here every constant predictor
    scores EXACTLY ``1/k = 0.2500`` with zero sampling variance.
    Training over ``train_pad_lens=(2, 3, 4)`` tests genuine delay-invariant memory when
    evaluated on ``pad_len=4`` (ID) and ``ood_pad_len=8`` (OOD longer delay).
    """
    vocab_size = k + 2
    pad_id, cue_id = k, k + 1

    def _make(p_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = p_len + 2
        x = torch.full((k, seq_len), pad_id, dtype=torch.long)
        y = torch.full((k, seq_len), pad_id, dtype=torch.long)
        for tgt in range(k):
            x[tgt, 0] = tgt
            x[tgt, -1] = cue_id
            y[tgt, -1] = tgt
        return x, y

    train_batches = [_make(p) for p in train_pad_lens]
    x_id, y_id = _make(pad_len)
    x_ood, y_ood = _make(ood_pad_len)
    return {
        "vocab_size": vocab_size,
        "n_classes": k,
        "k": k,
        "pad_len": pad_len,
        "ood_pad_len": ood_pad_len,
        "train_pad_lens": list(train_pad_lens),
        "train_batches": train_batches,
        "id": (x_id, y_id),
        "ood": (x_ood, y_ood),
    }
