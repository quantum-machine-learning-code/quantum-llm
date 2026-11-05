"""玩具级字符级自回归语言模型：Embedding -> 可微变换层 -> 词表头。"""
from __future__ import annotations

import torch
from torch import nn

class TinyLM(nn.Module):
    """字符级自回归模型：embedding -> 可微变换层(量子/经典) -> 词表分类头。"""

    def __init__(self, vocab_size: int, n_qubits: int, transform: nn.Module):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_qubits)
        self.transform = transform
        self.head = nn.Linear(n_qubits, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        h = self.embed(idx)  # (B, T, Q) float32
        B, T, Q = h.shape
        z = self.transform(h.reshape(B * T, Q))  # 展平后逐位置过变换层
        z = z.to(torch.float32)  # 量子 transform 可能返回 float64；cast 可微，保留梯度
        return self.head(z.reshape(B, T, Q))  # (B, T, V) logits
