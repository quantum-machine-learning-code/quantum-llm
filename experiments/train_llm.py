"""实验2：玩具级量子增强字符语言模型（非真正 LLM），与同规模经典层对照。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# 保证从任意 cwd 运行都能 import qml_lab（须在导入 qml_lab 之前）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

from qml_lab.data import build_char_data, iter_batches, load_corpus
from qml_lab.models import TinyLM
from qml_lab.quantum import make_vqc_layer
from qml_lab.train import train_lm
from qml_lab.utils import (
    count_parameters,
    ensure_dir,
    fmt_table,
    pick_device,
    save_json,
    set_seed,
)
from qml_lab.viz import plot_bars, plot_curves

# 明确的诚实声明：同时写入 JSON 与终端提示
CAVEAT = (
    "本实验是玩具级量子增强语言模型验证（词表仅字符级、序列很短、4~8 qubits "
    "状态向量模拟），不是能对话的 LLM；真实 LLM 规模无法在状态向量模拟器上运行。"
)
GEN_CHARS = 120


def _to_builtin(obj: Any) -> Any:
    """递归把 numpy 标量/数组转成 Python 内置类型，保证 JSON 里没有 numpy 原生类型。"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="实验2：玩具级量子增强字符 LM（非真正 LLM）")
    p.add_argument("--qubits", type=int, default=4)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corpus", type=str, default=None, help="语料路径；None 用内置语料")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--quick", action="store_true", help="极小规模冒烟运行")
    p.add_argument("--diff-method", type=str, default="backprop")
    p.add_argument("--shots", type=int, default=0, help="0 表示解析模拟")
    return p.parse_args()


def _greedy_sample(
    model: TinyLM,
    text: str,
    stoi: dict[str, int],
    itos: dict[int, str],
    seq_len: int,
    n_chars: int,
) -> str:
    """从语料前 seq_len 个字符起步，贪心续写 n_chars 个字符（no_grad + 最后时间步 argmax）。"""
    start = text[:seq_len]
    ids = [stoi.get(ch, 0) for ch in start]
    context = torch.tensor([ids], dtype=torch.long)  # (1, T)
    chars = list(start)
    model.eval()
    with torch.no_grad():
        for _ in range(n_chars):
            logits = model(context)  # TinyLM 输出 (1, T, V)
            # 只取最后一个时间步的 logits 做 argmax
            next_id = int(logits[0, -1].argmax().item())
            chars.append(itos[next_id])
            nxt = torch.tensor([[next_id]], dtype=torch.long)
            context = torch.cat([context[:, 1:], nxt], dim=1)  # 滑动窗口，保持长度 T
    return "".join(chars)


def main() -> None:
    args = parse_args()

    # --quick：压到最小规模
    if args.quick:
        args.qubits = 2
        args.layers = 1
        args.seq_len = 8
        args.batch_size = 4
        args.steps = 3

    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    print(f"device: {pick_device()}")

    text = load_corpus(path=str(Path(args.corpus)) if args.corpus else None)
    data, vocab_size, stoi, itos = build_char_data(text=text)
    vocab_size = int(vocab_size)
    print(f"语料 {len(text)} 字符，词表大小 {vocab_size}")

    # 量子层 / 经典层都作为 TinyLM 内部的可微 transform
    q_layer = make_vqc_layer(
        n_qubits=args.qubits,
        n_layers=args.layers,
        diff_method=args.diff_method,
        shots=args.shots or None,
    )
    q_lm = TinyLM(vocab_size=vocab_size, n_qubits=args.qubits, transform=q_layer)
    c_lm = TinyLM(
        vocab_size=vocab_size,
        n_qubits=args.qubits,
        transform=nn.Sequential(nn.Linear(args.qubits, args.qubits), nn.Tanh()),
    )

    log_every = max(1, args.steps // 10)
    # 两个模型各用独立 seed 的无限 batch 生成器，避免共用生成器互相消耗
    gen_q = iter_batches(
        data_tensor=data, seq_len=args.seq_len, batch_size=args.batch_size, seed=args.seed
    )
    gen_c = iter_batches(
        data_tensor=data, seq_len=args.seq_len, batch_size=args.batch_size, seed=args.seed + 1
    )
    hist_q = train_lm(
        model=q_lm, batch_gen=gen_q, steps=args.steps, lr=args.lr,
        seed=args.seed, verbose=True, log_every=log_every,
    )
    hist_c = train_lm(
        model=c_lm, batch_gen=gen_c, steps=args.steps, lr=args.lr,
        seed=args.seed, verbose=True, log_every=log_every,
    )

    summary: dict[str, Any] = {}
    for name, model, hist in [
        ("Quantum", q_lm, hist_q),
        ("Classical", c_lm, hist_c),
    ]:
        summary[name] = {
            "params": int(count_parameters(model)),
            "final_loss": float(hist["final_loss"]),
            "final_perplexity": float(hist["final_perplexity"]),
            "total_seconds": float(hist["total_seconds"]),
        }

    histories = {"Quantum": hist_q, "Classical": hist_c}
    payload = {
        "config": vars(args),
        "vocab_size": vocab_size,
        "histories": histories,
        "summary": summary,
        "caveat": CAVEAT,
    }
    save_json(_to_builtin(payload), str(out_dir / "lm_metrics.json"))

    plot_curves(
        histories=histories, metric="loss", ylabel="Loss",
        path=str(out_dir / "lm_loss.png"), title="Exp2: LM loss",
    )
    plot_curves(
        histories=histories, metric="perplexity", ylabel="Perplexity",
        path=str(out_dir / "lm_perplexity.png"), title="Exp2: LM perplexity",
    )
    plot_bars(
        values={k: v["total_seconds"] for k, v in summary.items()},
        ylabel="Train Time (s)",
        path=str(out_dir / "lm_time.png"),
        title="Exp2: training time",
    )

    # 贪心生成：写入文件并打印前 120 字符
    sample = _greedy_sample(q_lm, text, stoi, itos, args.seq_len, GEN_CHARS)
    (out_dir / "lm_sample.txt").write_text(sample, encoding="utf-8")
    print(f"生成样本（前 {GEN_CHARS} 字符，已写入 lm_sample.txt）：")
    print(sample[:GEN_CHARS])

    print(fmt_table(
        [
            ["Quantum", summary["Quantum"]["params"],
             f"{summary['Quantum']['final_loss']:.4f}",
             f"{summary['Quantum']['final_perplexity']:.4f}",
             f"{summary['Quantum']['total_seconds']:.2f}"],
            ["Classical", summary["Classical"]["params"],
             f"{summary['Classical']['final_loss']:.4f}",
             f"{summary['Classical']['final_perplexity']:.4f}",
             f"{summary['Classical']['total_seconds']:.2f}"],
        ],
        ["模型", "参数量", "最终 Loss", "困惑度", "训练耗时(s)"],
    ))

    # 醒目说明：这是玩具级 QNLP 验证，不是 LLM
    line = "=" * 64
    print(line)
    print("重要说明（CAVEAT）")
    print(CAVEAT)
    print(line)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[train_llm] 实验失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
