"""离线冒烟测试：直接 python tests/smoke_test.py 运行，不依赖 pytest / 不联网。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

from qml_lab.data import build_char_data, iter_batches, load_corpus
from qml_lab.models import TinyLM
from qml_lab.quantum import make_vqc_layer
from qml_lab.train import train_lm


def _finite_all(values: object) -> bool:
    """判断一组 loss / 指标是否全部为有限数。"""
    return bool(np.isfinite(np.asarray(values, dtype=float)).all())


def check_lm() -> None:
    text = load_corpus()  # 内置自写语料：不联网、不下载
    data, vocab_size, _stoi, _itos = build_char_data(text=text)
    vocab_size = int(vocab_size)
    seq_len, batch_size = 8, 4

    q_lm = TinyLM(
        vocab_size=vocab_size,
        n_qubits=2,
        transform=make_vqc_layer(
            n_qubits=2, n_layers=1, diff_method="backprop", shots=None
        ),
    )
    c_lm = TinyLM(
        vocab_size=vocab_size,
        n_qubits=2,
        transform=nn.Sequential(nn.Linear(2, 2), nn.Tanh()),
    )

    for name, model, seed in [("量子 TinyLM", q_lm, 0), ("经典 TinyLM", c_lm, 1)]:
        gen = iter_batches(
            data_tensor=data, seq_len=seq_len, batch_size=batch_size, seed=seed
        )
        hist = train_lm(
            model=model, batch_gen=gen, steps=3, lr=0.05,
            seed=seed, verbose=False, log_every=1,
        )
        assert _finite_all(hist["loss"]), f"{name} loss 不是有限数: {hist['loss']}"
        assert math.isfinite(float(hist["final_loss"])), f"{name} final_loss 不是有限数"
        assert math.isfinite(float(hist["final_perplexity"])), f"{name} final_perplexity 不是有限数"

        x, _ = next(iter_batches(
            data_tensor=data, seq_len=seq_len, batch_size=batch_size, seed=999
        ))
        with torch.no_grad():
            logits = model(x)
        assert tuple(logits.shape) == (batch_size, seq_len, vocab_size), \
            f"{name} logits 形状应为 {(batch_size, seq_len, vocab_size)}，实际 {tuple(logits.shape)}"
        assert bool(torch.isfinite(logits).all()), f"{name} logits 含 NaN/Inf"


def main() -> None:
    check_lm()
    print("SMOKE OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
