"""字符级语言模型语料：内置自写英文语料、留出验证集、经验条件熵下界与可复现 batch 生成器。"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path

import torch

# 独立语料（2026-11-13，见 docs/REVIEW.md 第 8 节）：只依赖标准库，无循环依赖。
from .corpus_independent import (  # noqa: E402
    corpus_overlap,
    load_independent_split,
)

__all__ = [
    "load_corpus",
    "load_corpus_split",
    "build_char_data",
    "encode_with_vocab",
    "iter_batches",
    "corpus_entropy_floors",
    "load_independent_split",
    "corpus_overlap",
]

# 自写英文语料：技术短句重复若干遍，无版权问题，含重复结构且长度 >= 800（1668 字符，31 字符词表）
_CORPUS_BLOCK = (
    "A qubit can hold a superposition of zero and one until we measure it. "
    "We rotate the state with a small angle and then read the result again. "
    "The circuit learns a pattern by tuning each gate one step at a time. "
    "Gradients flow back through the layer and update the weights slowly. "
    "Noise makes the signal weak but the model still finds a simple rule. "
    "We repeat the same short sentence to give the tokenizer enough data. "
)
_BUILTIN_CORPUS = _CORPUS_BLOCK * 4

# 留出验证语料（Held-out validation corpus）：
# 严格由 _BUILTIN_CORPUS 的同一个 31 字符词表与同一批领域词汇重新组合成未见过的完整句子，
# 用于检验模型是学到了局部子词/词法转移规律，还是仅仅死记硬背了训练集的 6 条固定句界。
_HELDOUT_VAL_CORPUS = (
    "A qubit can hold the state until we rotate it with a small angle and read the result. "
    "The model learns a simple rule by tuning each gate one step at a time. "
    "Gradients flow back through the circuit and update the layer weights slowly. "
    "Noise makes the signal weak but we still measure a superposition of zero and one. "
    "We repeat the short sentence again to give the model enough data. "
)
_EXTENDED_CORPUS = _BUILTIN_CORPUS + _HELDOUT_VAL_CORPUS


def load_corpus(path=None) -> str:
    """有 path 就读文件；否则返回内置自写英文语料（长度 1668，词表 31）。"""
    if path is None:
        return _BUILTIN_CORPUS
    if str(path) == "heldout_val":
        return _HELDOUT_VAL_CORPUS
    if str(path) == "extended":
        return _EXTENDED_CORPUS
    return Path(path).read_text(encoding="utf-8")


def load_corpus_split(path=None, val_fraction: float = 0.2) -> tuple[str, str]:
    """返回 (train_text, val_text)。

    * ``path is None`` 时：训练集保持与 :func:`load_corpus` 完全一致的 ``_BUILTIN_CORPUS``
      （1668 字符，31 词表），验证集返回同词表重组的新句集合 ``_HELDOUT_VAL_CORPUS``（382 字符）。
    * ``path is not None`` 时：按 ``1 - val_fraction`` 切分指定文本。
    """
    if path is None:
        return _BUILTIN_CORPUS, _HELDOUT_VAL_CORPUS
    raw = load_corpus(path)
    frac = max(0.01, min(0.9, float(val_fraction)))
    split_idx = max(1, min(len(raw) - 1, int(round(len(raw) * (1.0 - frac)))))
    return raw[:split_idx], raw[split_idx:]


def encode_with_vocab(text: str, stoi: dict[str, int]) -> torch.Tensor:
    """用已有词表 ``stoi`` 将文本编码为 1D long 张量。"""
    unk_id = 0
    return torch.tensor([stoi.get(ch, unk_id) for ch in text], dtype=torch.long)


def build_char_data(
    text, stoi: dict[str, int] | None = None
) -> tuple[torch.Tensor, int, dict, dict]:
    """字符级编码：返回 (long 一维张量, vocab_size, stoi, itos)。

    字符表默认用 sorted(set(text))；若传入已有 ``stoi`` 则复用该词表编码。
    """
    if stoi is None:
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = encode_with_vocab(text, stoi)
    return data, len(stoi), stoi, itos


def iter_batches(data_tensor, seq_len, batch_size, seed):
    """无限生成器：每次 yield (x, y)，shape 均为 (batch_size, seq_len)。

    y 为 x 右移一位；用 torch.Generator().manual_seed(seed) 保证可复现。
    注意：``torch.randint(0, max_start, ...)`` 上界为开区间，故 ``max_start = n - seq_len``
    使得最大可采样起点为 ``s = n - seq_len - 1``，此时 ``y`` 恰好覆盖最后一个字符 ``data[n - 1]``。
    """
    data = data_tensor.reshape(-1).long()
    if data.numel() < seq_len + 2:
        # 语料过短时循环拼接，保证能取出等长片段
        data = data.repeat((seq_len + 2) // data.numel() + 2)
    g = torch.Generator().manual_seed(int(seed))
    n = data.numel()
    max_start = n - seq_len  # randint 上界不含 max_start，最大起点 s = n - seq_len - 1
    while True:
        starts = torch.randint(0, max_start, (batch_size,), generator=g).tolist()
        x = torch.stack([data[s:s + seq_len] for s in starts])
        y = torch.stack([data[s + 1:s + 1 + seq_len] for s in starts])
        yield x, y


def corpus_entropy_floors(
    text_or_tensor,
    max_context: int = 8,
    seq_len: int = 16,
) -> dict[int, dict[str, float]]:
    """计算语料在不同上下文长度 k 下的精确经验条件熵 H(C_{t+1} | C_{t-k+1..t})（单位 nats）与困惑度。

    * ``k = -1``：均匀分布基线 ``ln(V)``（无任何先验）。
    * ``k = 0``：单字符（unigram）边际分布熵 ``H(C_{t+1})``（无条件上下文）。
    * ``k = 1``：逐位置（token-wise / bigram）模型的理论下界 ``H(C_{t+1} | C_t)``。
    * ``k >= 2``：因果窗口长度为 ``k``（看前 ``k`` 个字符预测第 ``k+1`` 个字符）的经验条件熵下界。

    同时返回 ``window16_loss`` / ``window16_perplexity``：在有限序列窗口 ``seq_len`` 内，
    由于前 ``k-1`` 个位置只能看到 ``1, 2, ..., k-1`` 个上下文字符，无跨窗口记忆的因果模型
    在 ``seq_len`` 窗口上的精确平均交叉熵理论下界。
    """
    if isinstance(text_or_tensor, torch.Tensor):
        seq = [int(v) for v in text_or_tensor.reshape(-1).tolist()]
    elif isinstance(text_or_tensor, str):
        seq = list(text_or_tensor)
    else:
        seq = list(text_or_tensor)

    n = len(seq)
    if n < 2:
        raise ValueError("语料长度至少为 2 才能计算条件熵")

    vocab_size = len(set(seq))
    uniform_loss = math.log(max(vocab_size, 1))
    uniform_ppl = float(max(vocab_size, 1))

    max_k = min(n - 1, max(int(max_context), 16, int(seq_len)))
    ks = list(range(0, max_k + 1))

    raw_h: dict[int, tuple[float, int]] = {}
    for k in ks:
        if k == 0:
            # 自回归目标为 seq[1:]（共 n - 1 个预测位置）
            targets = seq[1:]
            total = len(targets)
            counts = Counter(targets)
            h = -sum((c / total) * math.log(c / total) for c in counts.values())
            raw_h[0] = (h, 1)
        else:
            ctx_counts: dict[tuple, Counter] = defaultdict(Counter)
            total = n - k
            for i in range(total):
                ctx = tuple(seq[i:i + k])
                nxt = seq[i + k]
                ctx_counts[ctx][nxt] += 1
            h = 0.0
            for nxt_counter in ctx_counts.values():
                c_ctx = sum(nxt_counter.values())
                h_ctx = -sum(
                    (c / c_ctx) * math.log(c / c_ctx) for c in nxt_counter.values()
                )
                h += (c_ctx / total) * h_ctx
            raw_h[k] = (h, len(ctx_counts))

    result: dict[int, dict[str, float]] = {
        -1: {
            "entropy_nats": float(uniform_loss),
            "loss": float(uniform_loss),
            "perplexity": float(uniform_ppl),
            "unique_contexts": 1,
            "window16_loss": float(uniform_loss),
            "window16_perplexity": float(uniform_ppl),
            "uniform_nats": float(uniform_loss),
            "uniform_perplexity": float(uniform_ppl),
        }
    }

    for k in ks:
        h, n_ctx = raw_h[k]
        if k <= 1:
            w_loss = h
        else:
            # 在长度为 seq_len 的独立窗口中，位置 t=0..seq_len-1 实际可见的因果上下文长度为 min(t+1, k)
            pos_losses = [
                raw_h[min(t + 1, k)][0] for t in range(seq_len)
            ]
            w_loss = sum(pos_losses) / len(pos_losses)
        result[k] = {
            "entropy_nats": float(h),
            "loss": float(h),
            "perplexity": float(math.exp(h)),
            "unique_contexts": int(n_ctx),
            "window16_loss": float(w_loss),
            "window16_perplexity": float(math.exp(w_loss)),
            "uniform_nats": float(uniform_loss),
            "uniform_perplexity": float(uniform_ppl),
        }

    return result
