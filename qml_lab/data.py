"""字符级语言模型语料：内置自写英文语料与可复现 batch 生成器。"""
from __future__ import annotations

from pathlib import Path

import torch

__all__ = ["load_corpus", "build_char_data", "iter_batches"]

# 自写英文语料：技术短句重复若干遍，无版权问题，含重复结构且长度 >= 800
_CORPUS_BLOCK = (
    "A qubit can hold a superposition of zero and one until we measure it. "
    "We rotate the state with a small angle and then read the result again. "
    "The circuit learns a pattern by tuning each gate one step at a time. "
    "Gradients flow back through the layer and update the weights slowly. "
    "Noise makes the signal weak but the model still finds a simple rule. "
    "We repeat the same short sentence to give the tokenizer enough data. "
)
_BUILTIN_CORPUS = _CORPUS_BLOCK * 4


def load_corpus(path=None) -> str:
    """有 path 就读文件；否则返回内置自写英文语料（长度 >= 800）。"""
    if path is not None:
        return Path(path).read_text(encoding="utf-8")
    return _BUILTIN_CORPUS


def build_char_data(text) -> tuple[torch.Tensor, int, dict, dict]:
    """字符级编码：返回 (long 一维张量, vocab_size, stoi, itos)。

    字符表用 sorted(set(text))；itos 为 {int: str} 映射。
    """
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return data, len(chars), stoi, itos


def iter_batches(data_tensor, seq_len, batch_size, seed):
    """无限生成器：每次 yield (x, y)，shape 均为 (batch_size, seq_len)。

    y 为 x 右移一位；用 torch.Generator().manual_seed(seed) 保证可复现。
    """
    data = data_tensor.reshape(-1).long()
    if data.numel() < seq_len + 2:
        # 语料过短时循环拼接，保证能取出等长片段
        data = data.repeat((seq_len + 2) // data.numel() + 2)
    g = torch.Generator().manual_seed(int(seed))
    n = data.numel()
    max_start = n - seq_len - 1  # y 需要多取一位
    while True:
        starts = torch.randint(0, max_start, (batch_size,), generator=g).tolist()
        x = torch.stack([data[s:s + seq_len] for s in starts])
        y = torch.stack([data[s + 1:s + 1 + seq_len] for s in starts])
        yield x, y
