"""实验2：玩具级量子增强字符语言模型（非真正 LLM），与同规模经典层做严格受控对照。"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

# 保证从任意 cwd 运行都能 import qml_lab（须在导入 qml_lab 之前）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

from qml_lab.data import (
    build_char_data,
    corpus_entropy_floors,
    iter_batches,
    load_corpus_split,
)
from qml_lab.models import RichQuantumLM, TinyLM
from qml_lab.quantum import make_rich_layer, n_readout
from qml_lab.train import evaluate_lm, greedy_sample, train_lm
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


def _parse_scale(val: str) -> float | str:
    if str(val).strip().lower() == "learnable":
        return "learnable"
    return float(val)


def _parse_reupload(val: str) -> bool | str:
    s = str(val).strip().lower()
    if s in ("0", "false", "none", "no"):
        return False
    if s in ("1", "true", "fixed", "yes"):
        return True
    if s == "trainable":
        return "trainable"
    raise ValueError(f"未知 --reupload 参数: {val!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="实验2：玩具级量子增强字符 LM（非真正 LLM）")
    p.add_argument("--qubits", type=int, default=4)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="多随机种子列表（如 0 1 2 3 4）")
    p.add_argument("--context-mode", type=str, choices=["token", "causal"], default="token")
    p.add_argument("--context-window", type=int, default=4)
    p.add_argument("--observables", type=str, default="z")
    p.add_argument("--reupload", type=str, default="none")
    p.add_argument("--scale", type=str, default="1.0")
    p.add_argument("--pre-scale", action="store_true", help="用 tanh(h)*pi/2 约束入线角度")
    p.add_argument("--near-identity-init", action="store_true", help="量子权重用 N(0, 0.05^2) 小初始化")
    p.add_argument("--head", type=str, choices=["linear", "mlp"], default="linear")
    p.add_argument("--corpus", type=str, default=None, help="语料路径；None 用内置语料")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--quick", action="store_true", help="极小规模冒烟运行")
    p.add_argument("--diff-method", type=str, default="backprop")
    p.add_argument("--shots", type=int, default=0, help="0 表示解析模拟")
    return p.parse_args()


def _greedy_sample(
    model: nn.Module,
    text: str,
    stoi: dict[str, int],
    itos: dict[int, str],
    seq_len: int,
    n_chars: int,
) -> str:
    """从语料前 seq_len 个字符起步，贪心续写并截取前 n_chars 个字符。"""
    prompt = text[:seq_len]
    gen_len = max(1, n_chars - len(prompt))
    full = greedy_sample(
        model=model,
        vocab=(stoi, itos),
        prompt=prompt,
        length=gen_len,
        seq_len=seq_len,
    )
    return full[:n_chars]


def _build_pair(
    args: argparse.Namespace,
    vocab_size: int,
    seed: int,
) -> tuple[RichQuantumLM, RichQuantumLM]:
    """在同一随机种子下构造严格受控的 Quantum 与 Classical 对照模型（共享 embed / causal_conv / head 初始权重）。"""
    scale_val = _parse_scale(args.scale)
    reup_val = _parse_reupload(args.reupload)
    rdim = n_readout(args.qubits, args.observables)

    set_seed(seed)
    q_lm = RichQuantumLM(
        vocab_size=vocab_size,
        n_qubits=args.qubits,
        n_layers=args.layers,
        transform_type="quantum",
        context_mode=args.context_mode,
        context_window=args.context_window,
        observables=args.observables,
        reupload=reup_val,
        pre_scale=args.pre_scale,
        near_identity_init=args.near_identity_init,
        scale=scale_val,
        head=args.head,
        diff_method=args.diff_method,
        shots=args.shots or None,
    )

    set_seed(seed)
    c_lm = RichQuantumLM(
        vocab_size=vocab_size,
        n_qubits=args.qubits,
        n_layers=args.layers,
        transform_type="classical",
        context_mode=args.context_mode,
        context_window=args.context_window,
        observables=args.observables,
        pre_scale=args.pre_scale,
        scale=scale_val,
        head=args.head,
        classical_width=rdim,
    )

    # 严格断言共享组件初始权重完全相同
    assert torch.equal(q_lm.embed.weight, c_lm.embed.weight), "embed 初始权重未对齐"
    for p_q, p_c in zip(q_lm.head.parameters(), c_lm.head.parameters()):
        assert torch.equal(p_q, p_c), "head 初始权重未对齐"
    if q_lm.causal_encoder is not None and c_lm.causal_encoder is not None:
        for p_q, p_c in zip(
            q_lm.causal_encoder.parameters(), c_lm.causal_encoder.parameters()
        ):
            assert torch.equal(p_q, p_c), "causal_encoder 初始权重未对齐"

    return q_lm, c_lm


def main() -> None:
    args = parse_args()

    # --quick：压到最小规模
    if args.quick:
        args.qubits = 2
        args.layers = 1
        args.seq_len = 8
        args.batch_size = 4
        args.steps = 3
        if args.seeds is not None:
            args.seeds = args.seeds[:2]

    seeds = args.seeds if args.seeds else [args.seed]
    set_seed(seeds[0])
    out_dir = ensure_dir(args.out)
    print(f"device: {pick_device()}")

    train_text, val_text = load_corpus_split(
        path=str(Path(args.corpus)) if args.corpus else None
    )
    data, vocab_size, stoi, itos = build_char_data(text=train_text)
    val_data, _, _, _ = build_char_data(text=val_text, stoi=stoi)
    vocab_size = int(vocab_size)
    floors = corpus_entropy_floors(train_text, max_context=8, seq_len=args.seq_len)
    print(
        f"语料 {len(train_text)} 字符（验证集 {len(val_text)} 字符），词表大小 {vocab_size} | "
        f"k=1 二元语法熵下界 {floors[1]['loss']:.4f} nats (ppl {floors[1]['perplexity']:.2f}), "
        f"k=4 熵下界 {floors[4]['loss']:.4f} nats (ppl {floors[4]['perplexity']:.2f})"
    )

    log_every = max(1, args.steps // 10)
    per_seed_runs: dict[str, list[dict[str, Any]]] = {"Quantum": [], "Classical": []}
    first_histories: dict[str, dict] = {}
    samples_by_model: dict[str, str] = {}

    for s_idx, seed in enumerate(seeds):
        if len(seeds) > 1:
            print(f"\n--- Seed {seed} ({s_idx + 1}/{len(seeds)}) ---")
        q_lm, c_lm = _build_pair(args, vocab_size=vocab_size, seed=seed)

        # 修正混淆变量：两个模型使用完全相同 seed 的独立生成器，接收逐歩完全一致的 (x, y) 序列
        gen_q = iter_batches(
            data_tensor=data,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=seed,
        )
        gen_c = iter_batches(
            data_tensor=data,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=seed,
        )

        hist_q = train_lm(
            model=q_lm,
            batch_gen=gen_q,
            steps=args.steps,
            lr=args.lr,
            seed=seed,
            verbose=(s_idx == 0),
            log_every=log_every,
            data_tensor=data,
            val_tensor=val_data,
            seq_len=args.seq_len,
        )
        hist_c = train_lm(
            model=c_lm,
            batch_gen=gen_c,
            steps=args.steps,
            lr=args.lr,
            seed=seed,
            verbose=(s_idx == 0),
            log_every=log_every,
            data_tensor=data,
            val_tensor=val_data,
            seq_len=args.seq_len,
        )

        for name, model, hist in [
            ("Quantum", q_lm, hist_q),
            ("Classical", c_lm, hist_c),
        ]:
            sample_str = _greedy_sample(
                model, train_text, stoi, itos, args.seq_len, GEN_CHARS
            )
            if s_idx == 0:
                first_histories[name] = hist
                samples_by_model[name] = sample_str
            per_seed_runs[name].append(
                {
                    "seed": int(seed),
                    "params": int(count_parameters(model)),
                    "final_loss": float(hist["final_loss"]),
                    "final_perplexity": float(hist["final_perplexity"]),
                    "eval_loss": float(hist["eval_loss"]),
                    "eval_perplexity": float(hist["eval_perplexity"]),
                    "val_loss": float(hist["val_loss"]),
                    "val_perplexity": float(hist["val_perplexity"]),
                    "total_seconds": float(hist["total_seconds"]),
                    "sample": sample_str,
                }
            )

    summary: dict[str, Any] = {}
    for name in ("Quantum", "Classical"):
        runs = per_seed_runs[name]
        r0 = runs[0]
        eval_losses = [r["eval_loss"] for r in runs]
        eval_ppls = [r["eval_perplexity"] for r in runs]
        val_losses = [r["val_loss"] for r in runs]
        val_ppls = [r["val_perplexity"] for r in runs]
        secs = [r["total_seconds"] for r in runs]
        summary[name] = {
            "params": int(r0["params"]),
            # 保留首 seed 的 final_loss / final_perplexity 供旧脚本兼容，同时汇报全语料确定性 eval_loss
            "final_loss": float(np.mean([r["final_loss"] for r in runs])),
            "final_perplexity": float(np.mean([r["final_perplexity"] for r in runs])),
            "eval_loss": float(np.mean(eval_losses)),
            "eval_loss_std": float(np.std(eval_losses, ddof=1)) if len(runs) > 1 else 0.0,
            "eval_perplexity": float(np.mean(eval_ppls)),
            "eval_perplexity_std": float(np.std(eval_ppls, ddof=1)) if len(runs) > 1 else 0.0,
            "val_loss": float(np.mean(val_losses)),
            "val_loss_std": float(np.std(val_losses, ddof=1)) if len(runs) > 1 else 0.0,
            "val_perplexity": float(np.mean(val_ppls)),
            "total_seconds": float(np.mean(secs)),
            "sample": samples_by_model[name],
            "per_seed": runs,
        }

    payload = {
        "config": vars(args),
        "vocab_size": vocab_size,
        "entropy_floors": {str(k): v for k, v in floors.items() if k in (-1, 0, 1, 2, 4, 8, 16)},
        "histories": first_histories,
        "summary": summary,
        "caveat": CAVEAT,
    }
    save_json(_to_builtin(payload), str(out_dir / "lm_metrics.json"))

    plot_curves(
        histories=first_histories,
        metric="loss",
        ylabel="Loss (nats)",
        path=str(out_dir / "lm_loss.png"),
        title="Exp2: LM training loss",
    )
    plot_curves(
        histories=first_histories,
        metric="perplexity",
        ylabel="Perplexity",
        path=str(out_dir / "lm_perplexity.png"),
        title="Exp2: LM training perplexity",
    )
    plot_bars(
        values={k: v["total_seconds"] for k, v in summary.items()},
        ylabel="Train Time (s)",
        path=str(out_dir / "lm_time.png"),
        title="Exp2: training time",
    )

    # 贪心生成：同时写入 Quantum 与 Classical 样本
    sample_q = samples_by_model["Quantum"]
    sample_c = samples_by_model["Classical"]
    combined_samples = (
        f"[Quantum ({args.context_mode}, obs={args.observables})]\n{sample_q}\n\n"
        f"[Classical ({args.context_mode})]\n{sample_c}\n"
    )
    (out_dir / "lm_sample.txt").write_text(combined_samples, encoding="utf-8")
    print(f"生成样本（前 {GEN_CHARS} 字符，已写入 lm_sample.txt）：")
    print(f"  [Quantum]   {sample_q[:GEN_CHARS]}")
    print(f"  [Classical] {sample_c[:GEN_CHARS]}")

    print(
        fmt_table(
            [
                [
                    "Quantum",
                    summary["Quantum"]["params"],
                    f"{summary['Quantum']['eval_loss']:.4f}",
                    f"{summary['Quantum']['eval_perplexity']:.2f}",
                    f"{summary['Quantum']['val_loss']:.4f}",
                    f"{summary['Quantum'][' final_loss' if False else 'final_loss']:.4f}",
                    f"{summary['Quantum']['total_seconds']:.2f}",
                ],
                [
                    "Classical",
                    summary["Classical"]["params"],
                    f"{summary['Classical']['eval_loss']:.4f}",
                    f"{summary['Classical']['eval_perplexity']:.2f}",
                    f"{summary['Classical']['val_loss']:.4f}",
                    f"{summary['Classical']['final_loss']:.4f}",
                    f"{summary['Classical']['total_seconds']:.2f}",
                ],
            ],
            ["模型", "参数量", "全语料 Eval Loss", "Eval 困惑度", "验证集 Val Loss", "末步 Batch Loss", "训练耗时(s)"],
        )
    )

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
