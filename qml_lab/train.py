"""字符语言模型训练循环、全语料确定性评估与贪心文本生成。"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

try:
    from .utils import set_seed
except ImportError:
    from utils import set_seed

__all__ = ["evaluate_lm", "greedy_sample", "train_lm"]


def _finite(value, label: str) -> float:
    """保证记录值是有限 Python float；非有限值打印警告并回退为 0.0。"""
    f = float(value)
    if not math.isfinite(f):
        print(f"[警告] {label} 出现非有限值 {f}，已回退为 0.0")
        return 0.0
    return f


def evaluate_lm(
    model: torch.nn.Module,
    data_tensor: torch.Tensor,
    seq_len: int = 16,
    batch_size: int = 64,
    stride: int | None = None,
) -> dict[str, float]:
    """在全语料连续窗口上做确定性评估（no_grad），消除单步随机 mini-batch 评估噪声。

    默认 ``stride=None`` 取不重叠的连续窗口 ``0, seq_len, 2*seq_len, ...``，并补齐最后一个
    末尾窗口 ``n - seq_len - 1``，确保语料所有字符位置（含末字符 ``data[-1]``）均被覆盖。
    """
    data = data_tensor.reshape(-1).long()
    if data.numel() < seq_len + 2:
        data = data.repeat((seq_len + 2) // data.numel() + 2)
    n = data.numel()
    max_start = n - seq_len  # 最大合法起点为 max_start - 1 = n - seq_len - 1
    step = seq_len if stride is None else max(1, int(stride))
    starts = list(range(0, max_start, step))
    if starts[-1] != max_start - 1:
        starts.append(max_start - 1)

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for b_start in range(0, len(starts), batch_size):
            chunk = starts[b_start:b_start + batch_size]
            x = torch.stack([data[s:s + seq_len] for s in chunk])
            y = torch.stack([data[s + 1:s + 1 + seq_len] for s in chunk])
            logits = model(x)
            V = logits.shape[-1]
            loss = F.cross_entropy(
                logits.reshape(-1, V), y.reshape(-1), reduction="sum"
            )
            total_loss += float(loss.item())
            total_tokens += int(y.numel())
    if was_training:
        model.train()

    mean_loss = _finite(
        total_loss / total_tokens if total_tokens else 0.0, "eval_loss"
    )
    ppl = _finite(math.exp(min(mean_loss, 20.0)), "eval_perplexity")
    return {
        "eval_loss": mean_loss,
        "eval_perplexity": ppl,
        "loss": mean_loss,
        "perplexity": ppl,
        "n_windows": len(starts),
        "n_tokens": total_tokens,
    }


def _resolve_vocab(vocab) -> tuple[dict[str, int], dict[int, str]]:
    """支持传入 stoi、itos、(stoi, itos) 或原始语料字符串。"""
    if isinstance(vocab, tuple) and len(vocab) == 2:
        first, second = vocab
        if isinstance(first, dict) and isinstance(second, dict):
            return first, second
    if isinstance(vocab, str):
        chars = sorted(set(vocab))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return stoi, itos
    if isinstance(vocab, dict):
        if not vocab:
            raise ValueError("vocab 字典不能为空")
        k0 = next(iter(vocab.keys()))
        if isinstance(k0, str):
            stoi = dict(vocab)
            itos = {int(i): str(ch) for ch, i in stoi.items()}
            return stoi, itos
        itos = {int(i): str(ch) for i, ch in vocab.items()}
        stoi = {ch: i for i, ch in itos.items()}
        return stoi, itos
    raise TypeError(f"不支持的 vocab 类型: {type(vocab)!r}")


def greedy_sample(
    model: torch.nn.Module,
    vocab,
    prompt: str = "A qubit ",
    length: int = 120,
    seq_len: int = 16,
) -> str:
    """从给定 ``prompt`` 出发，用贪心解码（最后时间步 argmax）生成 ``length`` 个字符。"""
    stoi, itos = _resolve_vocab(vocab)
    if not prompt:
        prompt = itos[0]
    ids = [stoi.get(ch, 0) for ch in prompt]
    context = torch.tensor([ids[-seq_len:]], dtype=torch.long)
    chars = list(prompt)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(int(length)):
            logits = model(context)
            next_id = int(logits[0, -1].argmax().item())
            chars.append(itos.get(next_id, ""))
            nxt = torch.tensor([[next_id]], dtype=torch.long)
            context = torch.cat([context, nxt], dim=1)
            if context.shape[1] > seq_len:
                context = context[:, -seq_len:]
    if was_training:
        model.train()
    return "".join(chars)


def _extract_quantum_param_ids(model: torch.nn.Module) -> set[int]:
    """提取模型中量子电路层的参数 id 集合，供 quantum_lr 独立分组使用。"""
    qids: set[int] = set()
    for attr in ("q", "transform"):
        sub = getattr(model, attr, None)
        if sub is not None and (hasattr(sub, "layer") or hasattr(sub, "weights")):
            qids.update(id(p) for p in sub.parameters())
    return qids


def train_lm(
    model,
    batch_gen,
    steps: int = 200,
    lr: float = 0.01,
    seed: int = 0,
    verbose: bool = True,
    log_every: int = 20,
    data_tensor: torch.Tensor | None = None,
    val_tensor: torch.Tensor | None = None,
    seq_len: int = 16,
    eval_batch_size: int = 64,
    eval_every: int = 0,
    quantum_lr: float | None = None,
) -> dict:
    """训练字符级自回归语言模型。

    保留 ``final_loss`` / ``final_perplexity``（最末步 mini-batch 值）以完全向后兼容；
    当传入 ``data_tensor``（及可选 ``val_tensor``）时，额外计算确定性的全语料
    ``eval_loss`` / ``eval_perplexity``（及 ``val_loss`` / ``val_perplexity``）。
    """
    set_seed(seed)
    qids = _extract_quantum_param_ids(model) if quantum_lr is not None else set()
    if quantum_lr is not None and qids:
        qparams = [p for p in model.parameters() if id(p) in qids]
        cparams = [p for p in model.parameters() if id(p) not in qids]
        opt = torch.optim.Adam(
            [
                {"params": qparams, "lr": float(quantum_lr)},
                {"params": cparams, "lr": float(lr)},
            ]
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    history: dict = {
        "step": [],
        "loss": [],
        "perplexity": [],
        "seconds": [],
    }
    if eval_every and eval_every > 0 and data_tensor is not None:
        history["eval_steps"] = []
        history["eval_loss_curve"] = []
        history["eval_perplexity_curve"] = []
        if val_tensor is not None:
            history["val_loss_curve"] = []
            history["val_perplexity_curve"] = []

    t0 = time.perf_counter()

    for step, (x, y) in zip(range(steps), batch_gen):
        s = step + 1
        opt.zero_grad()
        logits = model(x)  # (B, T, V)
        V = logits.shape[-1]
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        loss.backward()
        opt.step()

        lf = _finite(loss.item(), "loss")
        pf = _finite(math.exp(min(lf, 20.0)), "perplexity")  # clamp 后再 exp，防溢出
        sec = time.perf_counter() - t0  # 累计耗时，便于画 loss~时间曲线

        history["step"].append(s)
        history["loss"].append(lf)
        history["perplexity"].append(pf)
        history["seconds"].append(_finite(sec, "seconds"))

        if (
            eval_every
            and eval_every > 0
            and data_tensor is not None
            and (s % eval_every == 0 or s == steps)
        ):
            ev = evaluate_lm(
                model, data_tensor, seq_len=seq_len, batch_size=eval_batch_size
            )
            history["eval_steps"].append(s)
            history["eval_loss_curve"].append(ev["eval_loss"])
            history["eval_perplexity_curve"].append(ev["eval_perplexity"])
            if val_tensor is not None:
                vev = evaluate_lm(
                    model, val_tensor, seq_len=seq_len, batch_size=eval_batch_size
                )
                history["val_loss_curve"].append(vev["eval_loss"])
                history["val_perplexity_curve"].append(vev["eval_perplexity"])

        if verbose and log_every > 0 and (s % log_every == 0 or s == steps):
            print(f"step {s:>4}/{steps} | loss {lf:.4f} | ppl {pf:.2f} | {sec:.1f}s")

    history["final_loss"] = history["loss"][-1] if history["loss"] else 0.0
    history["final_perplexity"] = (
        history["perplexity"][-1] if history["perplexity"] else 1.0
    )
    if data_tensor is not None:
        ev_final = evaluate_lm(
            model, data_tensor, seq_len=seq_len, batch_size=eval_batch_size
        )
        history["eval_loss"] = ev_final["eval_loss"]
        history["eval_perplexity"] = ev_final["eval_perplexity"]
    else:
        history["eval_loss"] = history["final_loss"]
        history["eval_perplexity"] = history["final_perplexity"]

    if val_tensor is not None:
        val_final = evaluate_lm(
            model, val_tensor, seq_len=seq_len, batch_size=eval_batch_size
        )
        history["val_loss"] = val_final["eval_loss"]
        history["val_perplexity"] = val_final["eval_perplexity"]

    history["total_seconds"] = _finite(time.perf_counter() - t0, "total_seconds")
    return history
