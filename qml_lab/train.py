"""字符语言模型训练循环：记录 loss / 困惑度 / 累计耗时。"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

try:
    from .utils import set_seed
except ImportError:
    from utils import set_seed

def _finite(value, label: str) -> float:
    """保证记录值是有限 Python float；非有限值打印警告并回退为 0.0。"""
    f = float(value)
    if not math.isfinite(f):
        print(f"[警告] {label} 出现非有限值 {f}，已回退为 0.0")
        return 0.0
    return f


def train_lm(model, batch_gen, steps: int = 200, lr: float = 0.01, seed: int = 0,
             verbose: bool = True, log_every: int = 20) -> dict:
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    history = {"step": [], "loss": [], "perplexity": [], "seconds": []}
    t0 = time.perf_counter()

    for step, (x, y) in zip(range(steps), batch_gen):
        s = step + 1
        opt.zero_grad()
        logits = model(x)                       # (B, T, V)
        V = logits.shape[-1]
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        loss.backward()
        opt.step()

        lf = _finite(loss.item(), "loss")
        pf = _finite(math.exp(min(lf, 20.0)), "perplexity")  # clamp 后再 exp，防溢出
        sec = time.perf_counter() - t0         # 累计耗时，便于画 loss~时间曲线

        history["step"].append(s)
        history["loss"].append(lf)
        history["perplexity"].append(pf)
        history["seconds"].append(_finite(sec, "seconds"))
        if verbose and log_every > 0 and (s % log_every == 0 or s == steps):
            print(f"step {s:>4}/{steps} | loss {lf:.4f} | ppl {pf:.2f} | {sec:.1f}s")

    history["final_loss"] = history["loss"][-1] if history["loss"] else 0.0
    history["final_perplexity"] = history["perplexity"][-1] if history["perplexity"] else 1.0
    history["total_seconds"] = _finite(time.perf_counter() - t0, "total_seconds")
    return history
