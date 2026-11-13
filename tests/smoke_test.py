"""离线冒烟与受控契约测试：直接 python tests/smoke_test.py 运行，不依赖 pytest / 不联网。"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from qml_lab.data import (
    build_char_data,
    corpus_entropy_floors,
    corpus_overlap,
    iter_batches,
    load_corpus,
    load_corpus_split,
    load_independent_split,
    load_synthetic_split,
)
from qml_lab.arithmetic import (
    BOUNDARY_EXAMPLES,
    MODULUS,
    RANDOM_BASELINE_ACCURACY,
    VOCAB_SIZE,
    answer_logits,
    apply_op,
    build_tensors,
    build_vocab,
    circular_init,
    context_overlap,
    decode_expression,
    encode_expression,
    enumerate_table,
    evaluate_answers,
    split_table,
)
from qml_lab.corpus_independent import assert_disjoint, word_bigram_novelty
from qml_lab.models import RichQuantumLM, TinyLM
from qml_lab.quantum import make_rich_layer, make_vqc_layer, n_readout
from qml_lab.train import evaluate_lm, greedy_sample, train_lm
from qml_lab.utils import set_seed


def _finite_all(values: object) -> bool:
    """判断一组 loss / 指标是否全部为有限数。"""
    return bool(np.isfinite(np.asarray(values, dtype=float)).all())


def check_data_and_entropy() -> None:
    """检查末字符采样覆盖修复（off-by-one bug）、留出验证集词表闭合性与经验条件熵下界。"""
    text = load_corpus()
    assert len(text) == 1668, f"内置语料长度应为 1668，实际 {len(text)}"
    data, vocab_size, stoi, _itos = build_char_data(text=text)
    assert vocab_size == 31, f"内置语料词表应为 31，实际 {vocab_size}"

    train_text, val_text = load_corpus_split()
    assert train_text == text
    assert len(val_text) > 200
    assert set(val_text) <= set(stoi.keys()), "留出验证集包含训练集词表外字符"

    # 验证 iter_batches 能够采样到最后一个字符 data[-1]
    short_data = torch.arange(20, dtype=torch.long)
    gen = iter_batches(short_data, seq_len=8, batch_size=16, seed=0)
    seen_last = False
    for _ in range(50):
        _x, y = next(gen)
        if bool((y == 19).any()):
            seen_last = True
            break
    assert seen_last, "iter_batches 未能采样到末字符 data[-1]（off-by-one bug 未修复）"

    # 验证经验条件熵计算器
    floors = corpus_entropy_floors(text, max_context=8, seq_len=16)
    assert abs(floors[0]["loss"] - 2.8493) < 1e-3, f"k=0 熵异常: {floors[0]['loss']}"
    assert abs(floors[1]["loss"] - 1.9701) < 1e-3, f"k=1 熵异常: {floors[1]['loss']}"
    assert abs(floors[2]["loss"] - 0.7938) < 1e-3, f"k=2 熵异常: {floors[2]['loss']}"
    assert abs(floors[4]["loss"] - 0.1000) < 1e-3, f"k=4 熵异常: {floors[4]['loss']}"
    assert floors[-1]["loss"] > floors[0]["loss"] > floors[1]["loss"] > floors[2]["loss"] > floors[4]["loss"]


def check_rich_quantum_layers() -> None:
    """检查 4-qubit 48 维两体 Pauli 读出与可训练数据重上传层的形状及梯度。"""
    assert n_readout(4, "z") == 4
    assert n_readout(4, "z+x+y") == 12
    assert n_readout(4, "z+x+y+zzall") == 24
    assert n_readout(4, "z+x+y+zzall+xxall+yyall") == 48

    set_seed(0)
    layer48 = make_rich_layer(
        n_qubits=4,
        n_layers=1,
        observables="z+x+y+zzall+xxall+yyall",
        near_identity_init=True,
    )
    x = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    out = layer48(x)
    assert tuple(out.shape) == (3, 48), f"48维读出形状错误: {tuple(out.shape)}"
    assert bool(torch.isfinite(out).all())
    out.sum().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    assert layer48.weights.grad is not None and float(layer48.weights.grad.abs().sum()) > 0.0

    # 可训练数据重上传层
    set_seed(1)
    reup_layer = make_rich_layer(
        n_qubits=2,
        n_layers=2,
        observables="z+x+y",
        reupload="trainable",
        near_identity_init=True,
    )
    x2 = torch.randn(4, 2, dtype=torch.float32, requires_grad=True)
    out2 = reup_layer(x2)
    assert tuple(out2.shape) == (4, 6)
    (out2 ** 2).sum().backward()
    assert reup_layer.layer.in_scales.grad is not None
    assert bool(torch.isfinite(reup_layer.layer.in_scales.grad).all())


def check_causal_strictness_and_seed_control() -> None:
    """检查因果上下文模型的严格因果掩蔽（未来 token 零泄漏）与跨模型初始权重对齐。"""
    set_seed(7)
    model = RichQuantumLM(
        vocab_size=31,
        n_qubits=2,
        n_layers=1,
        transform_type="quantum",
        context_mode="causal",
        context_window=4,
        observables="z+x+y+zzall+xxall+yyall",
        pre_scale=True,
        near_identity_init=True,
        scale=5.0,
        head="mlp",
        hidden=16,
    )
    model.eval()
    seq_a = torch.tensor([[1, 5, 9, 12, 3, 18, 7, 22]], dtype=torch.long)
    # 修改未来位置 t' = 6 > t = 4
    seq_future_changed = seq_a.clone()
    seq_future_changed[0, 6] = 29
    # 修改过去窗口内位置 t' = 2 <= t = 4
    seq_past_changed = seq_a.clone()
    seq_past_changed[0, 2] = 25

    with torch.no_grad():
        logits_a = model(seq_a)
        logits_future = model(seq_future_changed)
        logits_past = model(seq_past_changed)

    # 位置 0..5 的 logits 对位置 6 的修改必须完全不变（严格因果性）
    assert torch.allclose(logits_a[:, :6, :], logits_future[:, :6, :], atol=1e-6), (
        "Causal 模型出现未来信息泄漏：修改 t'=6 影响了 t<=5 的 logits"
    )
    # 位置 4 的 logits 对窗口内过去位置 2 的修改必须发生改变
    assert not torch.allclose(logits_a[:, 4, :], logits_past[:, 4, :], atol=1e-6), (
        "Causal 模型未能利用窗口内历史上下文：修改 t'=2 未改变 t=4 的 logits"
    )

    # 检验同 seed 下 Quantum / Classical / Identity 共享初始权重对齐
    set_seed(42)
    m_q = RichQuantumLM(
        vocab_size=31, n_qubits=2, n_layers=1, transform_type="quantum",
        context_mode="causal", context_window=4, observables="z+x", head="linear",
    )
    set_seed(42)
    m_c = RichQuantumLM(
        vocab_size=31, n_qubits=2, n_layers=1, transform_type="classical",
        context_mode="causal", context_window=4, observables="z+x", head="linear",
    )
    set_seed(42)
    m_id = RichQuantumLM(
        vocab_size=31, n_qubits=2, n_layers=1, transform_type="identity",
        context_mode="causal", context_window=4, observables="z", head="linear",
    )
    assert torch.equal(m_q.embed.weight, m_c.embed.weight)
    assert torch.equal(m_q.embed.weight, m_id.embed.weight)
    assert torch.equal(
        m_q.causal_encoder.causal_conv.weight,
        m_c.causal_encoder.causal_conv.weight,
    )
    assert torch.equal(m_q.head.weight, m_c.head.weight)


def check_lm() -> None:
    """检查 TinyLM 训练、全语料确定性评估 evaluate_lm 与 greedy_sample。"""
    text = load_corpus()
    data, vocab_size, stoi, itos = build_char_data(text=text)
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
            model=model,
            batch_gen=gen,
            steps=3,
            lr=0.05,
            seed=seed,
            verbose=False,
            log_every=1,
            data_tensor=data,
            seq_len=seq_len,
            quantum_lr=0.02,
        )
        assert _finite_all(hist["loss"]), f"{name} loss 不是有限数: {hist['loss']}"
        assert math.isfinite(float(hist["final_loss"])), f"{name} final_loss 不是有限数"
        assert math.isfinite(float(hist["eval_loss"])), f"{name} eval_loss 不是有限数"

        # 确定性全语料评估两次必须逐位一致
        ev1 = evaluate_lm(model, data, seq_len=seq_len)
        ev2 = evaluate_lm(model, data, seq_len=seq_len)
        assert ev1["eval_loss"] == ev2["eval_loss"], "evaluate_lm 结果非确定性"

        sample = greedy_sample(model, (stoi, itos), prompt="A qubit ", length=24, seq_len=seq_len)
        assert sample.startswith("A qubit ") and len(sample) == len("A qubit ") + 24

        x, _ = next(
            iter_batches(
                data_tensor=data, seq_len=seq_len, batch_size=batch_size, seed=999
            )
        )
        with torch.no_grad():
            logits = model(x)
        assert tuple(logits.shape) == (batch_size, seq_len, vocab_size), (
            f"{name} logits 形状应为 {(batch_size, seq_len, vocab_size)}，实际 {tuple(logits.shape)}"
        )
        assert bool(torch.isfinite(logits).all()), f"{name} logits 含 NaN/Inf"


def check_independent_corpus() -> None:
    """独立语料契约：句级 disjoint、零 UNK、重叠率显著低于旧留出集。

    防止以后改动语料时悄悄退化回「验证集上下文几乎都在训练集见过」。
    """
    train, val = load_independent_split()
    assert_disjoint(train, val)
    assert set(val) <= set(train), "独立验证集包含训练集词表外字符（会产生 UNK）"
    rep = corpus_overlap(train, val)
    assert rep["sentence_level_disjoint"] is True
    assert rep["val_context_covered_by_train"]["4"]["val_token_coverage"] < 0.5, (
        "独立语料 k=4 覆盖率退化到 0.5 以上，不再比旧留出集 (0.9024) 明显更独立"
    )
    assert rep["val_new_4gram_ratio"] > 0.30
    assert rep["train_unique_4gram_ratio"] > 0.50, "训练语料退化为复读文本"
    nov = word_bigram_novelty(train, val)
    assert nov["novel_ratio"] > 0.80, f"验证集新词对占比过低: {nov['novel_ratio']}"


def check_synthetic_corpus() -> None:
    """合成语料契约（REVIEW 第 9 节）：极端分布偏移、零 UNK、音节不相交。

    防止语料 B 退化成「只是另一份自然英文」，那样分布偏移压力测试就失去意义。
    """
    train, val = load_synthetic_split()
    assert_disjoint(train, val)
    assert set(val) <= set(train), "合成验证集包含训练集词表外字符（会产生 UNK）"
    rep = corpus_overlap(train, val)
    assert rep["sentence_level_disjoint"] is True
    assert rep["val_context_covered_by_train"]["4"]["val_token_coverage"] < 0.05, (
        "合成语料 k=4 覆盖率退化，不再是极端分布偏移"
    )
    assert rep["val_new_4gram_ratio"] > 0.90
    # 分布差异必须显著大于 independent 语料（其 JS ≈ 0.0094）
    js = rep["char_distribution_divergence"]["js_unigram"]
    assert js > 0.10, f"合成语料字符分布差异过小: JS={js}"
    a_tr, a_val = load_independent_split()
    a_js = corpus_overlap(a_tr, a_val)["char_distribution_divergence"]["js_unigram"]
    assert js > 5 * a_js, f"合成语料分布差异未明显超过 independent: {js} vs {a_js}"
    nov = word_bigram_novelty(train, val)
    assert nov["novel_ratio"] > 0.95


def check_arithmetic_task() -> None:
    """课题一契约：117 词表、模 112 边界样例、零重叠切分、循环编码受控。

    防止以后把词表改成字符级数字、把切分改回有重叠，或让两臂共享的
    循环编码悄悄退化成不一致的随机初始化。
    """
    stoi, _, size = build_vocab()
    assert size == VOCAB_SIZE == 117
    assert stoi["0"] == 0 and stoi["111"] == 111
    assert stoi["+"] == 112 and stoi["/"] == 115 and stoi["="] == 116

    for a, op, b, c in BOUNDARY_EXAMPLES:
        ids = encode_expression(a, b, op, c)
        assert len(ids) == 5 and ids[-1] == c
        assert decode_expression(ids) == f"{a}{op}{b}={c}"
        assert apply_op(a, b, op) == c

    table = enumerate_table("+")
    assert len(table) == MODULUS ** 2 == 12544
    for op in ("+", "-", "*"):
        assert len(enumerate_table(op)) == 12544, f"{op} 未在 Z_112 上封闭"
    # 只有与 112 互质的除数有模逆：phi(112) = 48
    assert len(enumerate_table("/")) == 48 * MODULUS

    tr, va = split_table(table, "holdout_pairs", 0.2, 0)
    ov = context_overlap(tr, va, "holdout_pairs")
    assert ov["pair_level_disjoint"] and ov["pair_intersection"] == 0
    assert ov["full_context_coverage_k4"] == 0.0, "验证集输入四元组必须全部未见"
    assert ov["coverage_by_k"]["k1"] == 1.0, "数字 token 应都见过（组合泛化而非数值外推）"
    tr2, va2 = split_table(table, "holdout_operand", 0.2, 0)
    assert context_overlap(tr2, va2, "holdout_operand")[
        "val_first_operand_novel_ratio"
    ] == 1.0

    x, y = build_tensors(tr)
    assert tuple(x.shape) == (len(tr), 4) and int(y.max()) < MODULUS
    assert abs(RANDOM_BASELINE_ACCURACY - 1.0 / MODULUS) < 1e-12

    specs = (
        dict(transform_type="quantum", observables="z+x+y+zzall+xxall+yyall", scale=5.0),
        dict(transform_type="classical", observables="z+x+y+zzall+xxall+yyall", scale=1.0),
    )
    embeddings = []
    for spec in specs:
        set_seed(0)
        model = RichQuantumLM(
            vocab_size=VOCAB_SIZE, n_qubits=4, n_layers=1,
            context_mode="causal", context_window=4,
            head="mlp", hidden=48, pre_scale=True, **spec,
        )
        circular_init(model)
        embeddings.append(model.embed.weight.detach().clone())
        e = model.embed.weight.detach()
        assert abs(float(e[0, 1]) - 1.0) < 1e-6
        assert abs(float(e[28, 0]) - 1.0) < 1e-6
        assert float(e[112, 4]) == 1.0
        assert tuple(answer_logits(model, x[:8]).shape) == (8, MODULUS)
        ev = evaluate_answers(model, x[:8], y[:8])
        assert 0.0 <= ev["answer_accuracy"] <= 1.0
        assert np.isfinite(ev["answer_loss"])
    assert torch.equal(embeddings[0], embeddings[1]), "循环编码后两臂 embed 不一致"

    cmd = [sys.executable, str(ROOT / "experiments" / "train_addition.py"), "--quick"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"train_addition.py --quick 失败:\n{proc.stderr}\n{proc.stdout}"


def check_embedding_modes() -> None:
    """课题一补：三模式 embedding 的初始化对齐、梯度开关与训练漂移契约。

    防止以后把「循环编码只是初始化」偷偷改成冻结（或反之），也防止
    frozen / trainable_circular 两模式初始权重不一致导致消融失去因果解释。
    """
    exp_dir = str(ROOT / "experiments")
    if exp_dir not in sys.path:
        sys.path.insert(0, exp_dir)
    from train_addition import EMBEDDING_MODES, build_model

    assert EMBEDDING_MODES == (
        "frozen_circular", "trainable_circular", "trainable_random"
    )
    inits = {}
    for mode in EMBEDDING_MODES:
        m_q = build_model("add_q_rich48", 0, mode)
        m_c = build_model("add_classical", 0, mode)
        assert torch.equal(m_q.embed.weight, m_c.embed.weight), (
            f"{mode}: 两臂初始 embed 不一致，受控对照失效"
        )
        assert m_q.embed.weight.requires_grad == (mode != "frozen_circular")
        assert m_c.embed.weight.requires_grad == (mode != "frozen_circular")
        if mode != "trainable_random":
            e = m_q.embed.weight.detach()
            assert abs(e[0, 1].item() - 1.0) < 1e-6
            assert abs(e[28, 0].item() - 1.0) < 1e-6
        inits[mode] = m_q.embed.weight.detach().clone()
    assert torch.equal(inits["frozen_circular"], inits["trainable_circular"]), (
        "冻结 / 放开两模式初始权重必须逐位相同，否则无法隔离 requires_grad 的净效应"
    )
    assert not torch.equal(inits["frozen_circular"], inits["trainable_random"])

    table = enumerate_table("+")
    tr, _ = split_table(table, "holdout_pairs", 0.2, 0)
    x, y = build_tensors(tr)
    for mode, should_move in (("frozen_circular", False),
                              ("trainable_circular", True),
                              ("trainable_random", True)):
        m = build_model("add_classical", 0, mode)
        before = m.embed.weight.detach().clone()
        opt = torch.optim.Adam(
            [p for p in m.parameters() if p.requires_grad], lr=0.05
        )
        loss = torch.nn.functional.cross_entropy(
            answer_logits(m, x[:16]), y[:16]
        )
        loss.backward()
        opt.step()
        moved = not torch.equal(m.embed.weight, before)
        assert moved == should_move, f"{mode}: embed 训练漂移不符合预期"
        assert np.isfinite(loss.item())

    cmds = [
        [sys.executable, str(ROOT / "experiments" / "train_addition.py"),
         "--quick", "--embedding-mode", "frozen_circular"],
        [sys.executable, str(ROOT / "experiments" / "embedding_ablation.py"),
         "--quick"],
    ]
    for cmd in cmds:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                              text=True, check=False)
        assert proc.returncode == 0, (
            f"{' '.join(cmd[1:])} 失败:\n{proc.stderr}\n{proc.stdout}"
        )


def check_quick_cli() -> None:
    """验证 train_llm.py --quick 可正常跑通。"""
    cmd = [sys.executable, str(ROOT / "experiments" / "train_llm.py"), "--quick"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"train_llm.py --quick 失败:\n{proc.stderr}\n{proc.stdout}"


def main() -> None:
    check_data_and_entropy()
    check_independent_corpus()
    check_synthetic_corpus()
    check_arithmetic_task()
    check_embedding_modes()
    check_rich_quantum_layers()
    check_causal_strictness_and_seed_control()
    check_lm()
    check_quick_cli()
    print("SMOKE OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
