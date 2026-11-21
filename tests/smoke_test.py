"""离线冒烟与受控契约测试：支持 `pytest` 或直接 `python tests/smoke_test.py` 运行，不联网。"""
from __future__ import annotations

import math
import subprocess
import sys
import tempfile
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
from qml_lab.models import ClassicalTrigBilinearLayer, RichQuantumLM, TinyLM, parameter_breakdown
from qml_lab.quantum import count_qwc_groups, make_rich_layer, make_vqc_layer, n_readout
from qml_lab.train import evaluate_lm, greedy_sample, train_lm
from qml_lab.utils import set_seed


def _finite_all(values: object) -> bool:
    """判断一组 loss / 指标是否全部为有限数。"""
    return bool(np.isfinite(np.asarray(values, dtype=float)).all())


def check_data_and_entropy() -> None:
    """检查末字符采样覆盖修复（off-by-one bug）、留出验证集词表闭合性与经验条件熵下界。"""
    text = load_corpus()
    assert len(text) == 1668, f"内置语料长度应为 1668，实际 {len(text)}"
    _data, vocab_size, stoi, _itos = build_char_data(text=text)
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
    """检查规范对称多观测量读出（3 QWC 组）、legacy_rich48（10 QWC 组）、经典近恒等尺度与公平三角替身。"""
    assert n_readout(4, "z") == 4
    assert n_readout(4, "z+x+y") == 12
    assert n_readout(4, "z+x+y+zzall") == 18
    assert n_readout(4, "z+x+y+zzall+xxall+yyall") == 30
    assert count_qwc_groups(4, "z+x+y+zzall+xxall+yyall") == 3, "规范对称两体读出应恰为 3 个 QWC 组"
    assert n_readout(4, "legacy_rich24") == 24
    assert n_readout(4, "legacy_rich48") == 48
    assert count_qwc_groups(4, "legacy_rich48") == 10, "legacy_rich48 跨基读出应为 10 个 QWC 组"

    set_seed(0)
    layer30 = make_rich_layer(
        n_qubits=4,
        n_layers=1,
        observables="z+x+y+zzall+xxall+yyall",
        near_identity_init=True,
    )
    x = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    out = layer30(x)
    assert tuple(out.shape) == (3, 30), f"规范 30 维读出形状错误: {tuple(out.shape)}"
    assert bool(torch.isfinite(out).all())
    out.sum().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    assert layer30.weights.grad is not None and float(layer30.weights.grad.abs().sum()) > 0.0

    # 验证修复后的 near_identity_init=True：量子臂与经典臂初始输出 std 量级对齐（不再差 16 倍）
    set_seed(0)
    mq = RichQuantumLM(
        vocab_size=29, n_qubits=4, n_layers=1, transform_type="quantum",
        context_mode="causal", observables="z+x+y+zzall+xxall+yyall",
        pre_scale=True, near_identity_init=True, scale=1.0, head="mlp", hidden=48,
    ).eval()
    set_seed(0)
    mc = RichQuantumLM(
        vocab_size=29, n_qubits=4, n_layers=1, transform_type="classical",
        context_mode="causal", observables="z+x+y+zzall+xxall+yyall",
        pre_scale=True, near_identity_init=True, scale=1.0, head="mlp", hidden=48,
    ).eval()
    set_seed(0)
    mt = RichQuantumLM(
        vocab_size=29, n_qubits=4, n_layers=1, transform_type="classical_trig",
        context_mode="causal", observables="z+x+y+zzall+xxall+yyall",
        pre_scale=True, near_identity_init=True, scale=1.0, head="mlp", hidden=48,
    ).eval()
    idx_probe = torch.randint(0, 29, (8, 8))
    with torch.no_grad():
        sq = float(mq.bottleneck_features(idx_probe).std())
        sc = float(mc.bottleneck_features(idx_probe).std())
        st = float(mt.bottleneck_features(idx_probe).std())
    assert 0.4 <= sq / max(sc, 1e-8) <= 2.5, f"near_identity_init 初始尺度失衡: Q={sq:.4f} vs C={sc:.4f}"
    assert 0.4 <= sq / max(st, 1e-8) <= 2.5, f"ClassicalTrigBilinearLayer 初始尺度失衡: Q={sq:.4f} vs T={st:.4f}"
    # ClassicalTrigBilinearLayer matched 模式应精确匹配 12 个可训练参数
    trig_layer = ClassicalTrigBilinearLayer(4, 30, param_mode="matched")
    assert sum(p.numel() for p in trig_layer.parameters()) == 12
    assert parameter_breakdown(mq)["transform_quantum"] == 12
    assert parameter_breakdown(mt)["transform_classical"] == 12

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
        # 验证优化后的 answer_logits 与全序列 forward()[:, -1, :112] 逐位等价，且只对末位调用 transform（省 75% 计算）
        model.eval()
        seen_batches = []
        hook = model.transform.register_forward_hook(lambda _m, inp, _out: seen_batches.append(int(inp[0].shape[0])))
        try:
            with torch.no_grad():
                fast_lg = answer_logits(model, x[:8])
                full_lg = model(x[:8])[:, -1, :MODULUS]
        finally:
            hook.remove()
        assert torch.allclose(fast_lg, full_lg, atol=1e-5), "answer_logits 优化路径与完整前向不一致"
        assert seen_batches == [8, 32], f"answer_logits 应只送 B=8 而非 4*B=32 进入 transform，实测 {seen_batches}"
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


def check_grokking_scan() -> None:
    """第 12 节契约：正对照可解任务、progress measure 能区分有无循环结构、
    相变判定对「先记忆后突发泛化」敏感，且长训练脚本 --quick 可跑通。
    """
    exp_dir = str(ROOT / "experiments")
    if exp_dir not in sys.path:
        sys.path.insert(0, exp_dir)
    from grokking_scan import (
        QuadraticMLP, TARGET_K1, TARGET_K12, _phase_detect, _r2_decode,
    )

    table = enumerate_table("+")
    tr, _ = split_table(table, "holdout_pairs", 0.2, 0)
    x, y = build_tensors(tr)

    # 正对照接口必须与 RichQuantumLM 一致，才能复用 answer_logits / evaluate_answers
    ctrl = QuadraticMLP(hidden=32)
    assert tuple(answer_logits(ctrl, x[:8]).shape) == (8, MODULUS)
    ev = evaluate_answers(ctrl, x[:8], y[:8])
    assert np.isfinite(ev["answer_loss"])
    assert ctrl.embed.weight.requires_grad

    # Fourier progress measure：循环编码应近乎完全可解码，随机 embedding 不应
    rich = RichQuantumLM(vocab_size=VOCAB_SIZE, n_qubits=4, n_layers=1,
                         context_mode="causal", context_window=4,
                         transform_type="classical", head="mlp", hidden=48)
    rng = torch.Generator().manual_seed(7)
    random_w = torch.randn(112, 16, generator=rng).numpy()
    assert _r2_decode(random_w, TARGET_K1) < 0.5
    circular_init(rich)
    circ_w = rich.embed.weight.detach().numpy()[:112]
    assert _r2_decode(circ_w, TARGET_K1) > 0.95
    assert _r2_decode(circ_w[:, :4], TARGET_K12) > 0.95, "前 4 维必须保留 sin/cos 相位"
    assert _r2_decode(circ_w[:, :4] * 0.0 + torch.randn(112, 4).numpy(),
                      TARGET_K12) < 0.5

    # 相变判定：平滑同步上升不算 grokking；先记忆后突发才算
    smooth = [{"step": s, "train_acc": s / 10, "val_acc": s / 10,
               "train_loss": 1.0, "val_loss": 1.0} for s in range(1, 11)]
    assert _phase_detect(smooth)["grokking_observed"] is False
    delay = [
        {"step": 1, "train_acc": 0.10, "val_acc": 0.01,
         "train_loss": 1.0, "val_loss": 1.0},
        {"step": 2, "train_acc": 0.98, "val_acc": 0.01,
         "train_loss": 1.0, "val_loss": 1.0},
        {"step": 3, "train_acc": 1.00, "val_acc": 0.05,
         "train_loss": 1.0, "val_loss": 1.0},
        {"step": 4, "train_acc": 1.00, "val_acc": 0.95,
         "train_loss": 1.0, "val_loss": 1.0},
    ]
    ph = _phase_detect(delay)
    assert ph["delayed_generalization"] and ph["abrupt_generalization"]
    assert ph["grokking_observed"] and ph["delay_steps"] == 2

    cmd = [sys.executable, str(ROOT / "experiments" / "grokking_scan.py"), "--quick"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert proc.returncode == 0, (
        f"grokking_scan.py --quick 失败:\n{proc.stderr}\n{proc.stdout}"
    )


def check_grokking_no_prior() -> None:
    """第 13 节契约：「不加先验 = grokking」修正实验的两架构必须完全同超参，
    网格必须覆盖上一轮缺失的 lr=0.003 / wd=1.0 / 少训练数据条件，
    且 --quick 全链路可跑通。
    """
    exp_dir = str(ROOT / "experiments")
    if exp_dir not in sys.path:
        sys.path.insert(0, exp_dir)
    from grokking_no_prior import LR_GRID, VF_GRID, WD_GRID, _grid, _select

    grid = _grid(10, 5)
    rich = {c["name"].replace("RICH/", ""): c for c in grid if c["arch"] == "rich"}
    mlp = {c["name"].replace("MLP/", ""): c for c in grid
           if c["arch"] == "quad_mlp"}
    assert set(rich) == set(mlp) and len(rich) > 0
    for tag in rich:
        a, b = rich[tag], mlp[tag]
        for k in ("lr", "wd", "val_fraction", "steps", "eval_every",
                  "mode", "opt", "group", "embed_lr_scale", "batch_size"):
            if k in a:
                assert a[k] == b[k], f"{tag} 的 {k} 未对齐，架构对照不干净"
        assert a["mode"] == "trainable_random", "修正实验只能用不加先验组"
    assert 0.003 in LR_GRID and 1.0 in WD_GRID and 0.8 in VF_GRID, (
        "网格必须覆盖上一轮正对照用过的低 lr / 强 wd / 少训练数据条件"
    )
    # 探索性预选不能把 MLP 条件当成 rich 条件选出来
    pilot = {
        "RICH/a": {"arch": "rich", "lr": 0.003, "wd": 1.0,
                   "val_fraction": 0.8,
                   "final": {"train_accuracy": 0.99, "val_accuracy": 0.99},
                   "phase": {"grokking_observed": True,
                             "delayed_generalization": True,
                             "max_val_accuracy": 0.99,
                             "final_train_accuracy": 0.99}},
        "MLP/a": {"arch": "quad_mlp", "lr": 0.003, "wd": 1.0,
                  "val_fraction": 0.8,
                  "final": {"train_accuracy": 1.0, "val_accuracy": 1.0},
                  "phase": {"grokking_observed": True,
                            "delayed_generalization": True,
                            "max_val_accuracy": 1.0,
                            "final_train_accuracy": 1.0}},
    }
    selected = _select(pilot, 1)
    assert len(selected) == 1 and selected[0]["name"] == "RICH/a"

    cmd = [sys.executable, str(ROOT / "experiments" / "grokking_no_prior.py"),
           "--quick"]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          check=False)
    assert proc.returncode == 0, (
        f"grokking_no_prior.py --quick 失败:\n{proc.stderr}\n{proc.stdout}"
    )


def check_recurrent_architecture() -> None:
    """L1 架构契约：量子态参与序列依赖（前向/反向/状态/因果/生成）。

    见 docs/ARCHITECTURE_PROPOSAL.md 的 P1–P4。这里只验证结构，不比较性能。
    """
    from qml_lab.recurrent import (
        ClassicalRecurrentLM,
        QuantumRecurrentLM,
        make_recurrent_qnode,
    )

    set_seed(0)
    V, B, T, Q = 31, 2, 4, 4
    idx = torch.randint(0, V, (B, T))

    # QNode 契约：批量输出 (B,Q)，梯度覆盖输入 / 上一步状态 / 线路权重
    qnode = make_recurrent_qnode(Q)
    x = torch.rand(B, Q, requires_grad=True)
    s = torch.rand(B, Q, requires_grad=True)
    w = torch.rand(1, Q, 3, requires_grad=True)
    out = torch.stack(list(qnode(x, s, w)), dim=-1)
    assert out.shape == (B, Q) and _finite_all(out.detach().numpy())
    out.pow(2).sum().backward()
    assert x.grad is not None and s.grad is not None and w.grad is not None, (
        "量子循环 QNode 必须对输入、上一步状态、线路权重都可微"
    )
    with torch.no_grad():
        out0 = torch.stack(list(qnode(x, torch.zeros_like(s), w)), dim=-1)
    assert (out.detach() - out0).abs().mean() > 1e-3, (
        "上一步状态置零后输出应改变，否则状态反馈没有进入线路（P2）"
    )

    for cls in (QuantumRecurrentLM, ClassicalRecurrentLM):
        set_seed(0)
        model = cls(V, n_qubits=Q) if cls is QuantumRecurrentLM else cls(V, state_dim=Q)
        model.train()
        logits = model(idx)
        assert logits.shape == (B, T, V), f"{cls.__name__} 输出应为 (B,T,V)"
        assert _finite_all(logits.detach().numpy())

        # P3 严格因果：改掉位置 >=2 的 token 不能影响位置 0/1 的 logits
        idx2 = idx.clone()
        idx2[:, 2:] = 0
        with torch.no_grad():
            logits2 = model(idx2)
        assert torch.allclose(logits.detach()[:, :2], logits2[:, :2], atol=1e-6), (
            f"{cls.__name__} 非因果：未来 token 影响了过去位置的 logits"
        )

        # 反向传播覆盖输入编码与循环层（量子臂还要覆盖线路权重）
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, V), idx[:, 1:].reshape(-1)
        )
        loss.backward()
        assert model.embed.weight.grad is not None
        if cls is QuantumRecurrentLM:
            assert model.weights.grad is not None, "量子线路权重必须收到跨步梯度"
            assert model.input_proj.weight.grad is not None

        # P4 逐步生成：前缀保持不变、长度正确，且 greedy 首 token 与 forward(prefix)[:, -1] 100% 逐位一致
        prefix = idx[:, :2]
        gen = model.generate(prefix, n_new=3, greedy=True)
        assert gen.shape == (B, 5) and torch.equal(gen[:, :2], prefix)
        with torch.no_grad():
            rand_pref = torch.randint(0, V, (16, 4))
            fwd_first = model(rand_pref)[:, -1, :].argmax(dim=-1)
            gen_first = model.generate(rand_pref, n_new=1, greedy=True)[:, -1]
        assert torch.equal(fwd_first, gen_first), (
            f"{cls.__name__}.generate() 首生成 token 与 forward()[:, -1] 不一致（重复喂入 prefix[-1] bug）"
        )
        # 温度采样分支（greedy=False）应能产生非恒定随机样本
        g_samp = torch.Generator().manual_seed(123)
        s_a = model.generate(rand_pref[:4], n_new=4, temperature=2.0, greedy=False, generator=g_samp)
        s_b = model.generate(rand_pref[:4], n_new=4, temperature=2.0, greedy=False, generator=g_samp)
        assert s_a.shape == (4, 8) and not torch.equal(s_a[:, 4:], s_b[:, 4:]), (
            f"{cls.__name__}.generate(greedy=False) 未执行真正的温度多项采样"
        )

        # 回归测试 detach_state=True：截断 BPTT 不能切断当前步 t 对 embed / weights / cell 的梯度！
        set_seed(0)
        m_det = (
            cls(V, n_qubits=Q, detach_state=True)
            if cls is QuantumRecurrentLM
            else cls(V, state_dim=Q, detach_state=True)
        )
        m_det.train()
        logits_det = m_det(idx)
        loss_det = torch.nn.functional.cross_entropy(
            logits_det[:, :-1].reshape(-1, V), idx[:, 1:].reshape(-1)
        )
        loss_det.backward()
        assert m_det.embed.weight.grad is not None and float(m_det.embed.weight.grad.abs().sum()) > 0.0, (
            f"{cls.__name__}(detach_state=True) 切断了当前步 embed 梯度"
        )
        if cls is QuantumRecurrentLM:
            assert m_det.weights.grad is not None and float(m_det.weights.grad.abs().sum()) > 0.0, (
                "QuantumRecurrentLM(detach_state=True) 切断了当前步量子线路权重梯度"
            )
        else:
            assert m_det.cell.weight_ih.grad is not None and float(m_det.cell.weight_ih.grad.abs().sum()) > 0.0

    # 状态消融：同权重下 feedback=False 应改变输出（含 RZ+Y、RY+Z、XYZ 读出模式）
    for fb, rb in (("rz", "y"), ("ry", "z"), ("rz", "xyz")):
        set_seed(0)
        m_on = QuantumRecurrentLM(V, n_qubits=Q, state_feedback=fb, readout_basis=rb).eval()
        m_off = QuantumRecurrentLM(V, n_qubits=Q, state_feedback=fb, readout_basis=rb, feedback=False).eval()
        with torch.no_grad():
            m_off.load_state_dict(m_on.state_dict())
            d = (m_on(idx) - m_off(idx)).abs().mean()
        assert d > 1e-3, f"feedback 消融 ({fb}+{rb}) 应改变输出，实测 {d:.6f}"


def check_recurrent_pilot() -> None:
    """L1 短预算 pilot 契约：状态消融是单变量、P1–P4 可量化、结论边界不被越过。

    只跑 ``--quick --out <tmp>``，不写 docs/ 归档、不重跑正式实验。
    这里验证的是**装置与口径**，不是任何模型效果。
    """
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable,
            str(ROOT / "experiments" / "recurrent_pilot.py"),
            "--quick", "--no-figure", "--out", tmpdir,
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                              text=True, check=False)
        assert proc.returncode == 0, (
            f"recurrent_pilot.py --quick 失败:\n{proc.stderr}\n{proc.stdout}"
        )
        report = json.loads(
            (Path(tmpdir) / "recurrent_pilot.json").read_text(encoding="utf-8")
        )

    # 消融单变量：A0/A1 与两经典臂的初始权重必须逐元素相同
    for pair, info in report["ablation_matched_check"].items():
        assert info["max_abs_init_weight_diff"] == 0.0, (
            f"{pair} 初始权重不一致，状态反馈消融被污染"
        )
        assert info["only_difference"] == "feedback"

    contracts = report["contracts"]
    on_q = contracts["A0_quantum_feedback"]
    off_q = contracts["A1_quantum_no_feedback"]
    on_c = contracts["CTRL_classical_feedback"]
    # P1：前缀信息只有在状态反馈打开时才传得到末位
    assert on_q["P1_prefix_logits_mean_abs_delta"] > 0
    assert off_q["P1_prefix_logits_mean_abs_delta"] < 1e-9
    assert on_c["P1_prefix_logits_mean_abs_delta"] > 0
    # P2：跨步梯度在开反馈时非空、关反馈时消失
    assert on_q["P2_state_grad_norm"] > 0
    assert off_q["P2_state_grad_norm"] == 0.0
    assert on_q["P2b_bptt_grad_norm"] > 0
    # P3/P4：严格因果、逐步生成
    for name, c in contracts.items():
        assert c["P3_future_leakage_max_abs"] < 1e-6, f"{name} 未来信息泄漏"
        assert c["P4_generate_length_ok"] and c["P4_prefix_preserved"], (
            f"{name} 逐步生成接口不正确"
        )

    # 语料泄漏口径：报验证集指标必须带 k=4 上下文重叠率
    assert 0.0 <= report["corpus"]["val_context_k4_coverage"] <= 1.0
    # 训练曲线与耗时必须是有限数
    for arm, runs in report["char_lm"]["per_arm"].items():
        for r in runs:
            assert math.isfinite(r["train_final_loss"]), f"{arm} loss 非有限数"
            assert math.isfinite(r["eval"]["val_loss"]), f"{arm} val loss 非有限数"
            assert r["seconds_per_step"] > 0
    assert report["runtime"]["single_process"] is True

    # 结论边界：必须显式禁止「量子 vs 经典优劣」与显著性结论
    verdict = report["verdict"]
    assert verdict["all_contracts_pass"] is True
    forbidden = " ".join(verdict["claims_forbidden"])
    assert "量子循环优于" in forbidden and "Holm" in forbidden
    assert "≥5 seed" in forbidden, "必须显式禁止用本 pilot 下结论性数字"
    # 结构契约与功能有效性必须分开记录，防止把 P1–P4 通过误读成「状态被用上」
    fv = verdict["functional_state_use"]
    assert "quantum_above_baseline" in fv
    assert "未证明" in fv["interpretation"]
    assert report["corpus"]["val_context_k4_coverage"] < 0.5


def check_state_channel_probe() -> None:
    """L1 状态通道解析诊断的契约：验证 s=0,w=0 下 RZ+Z 与 RZ+X 导数皆为 0 而 RZ+Y 非零。"""
    from experiments.state_channel_probe import run_probe

    report = run_probe(n_inputs=4)
    rows_rand = {row["init_std"]: row for row in report["measurements"]}
    rows_zero = {row["init_std"]: row for row in report["measurements_at_init_zero"]}

    # 真实初态 s=0, w=0：RZ+Z 与 RZ+X 皆为 0，唯有共轭基 RZ+Y 与同轴 RY+Z 非零
    z0 = rows_zero[0.0]["configs"]
    assert z0["RZ_feedback_Z_readout"]["mean_abs_grad"] < 1e-6
    assert z0["RZ_feedback_X_readout"]["mean_abs_grad"] < 1e-6, (
        "s=0, w=0 时 d(cos s)/ds = 0，RZ+X 应无一阶导数"
    )
    assert z0["RZ_feedback_Y_readout"]["mean_abs_grad"] > 1e-2, (
        "共轭基 RZ+Y 在 s=0, w=0 时应有 O(1) 一阶敏感度"
    )
    assert z0["RY_feedback_Z_readout"]["mean_abs_grad"] > 1e-1

    # s~N(0,0.1), w=0.05：RZ+Y > RZ+X > RZ+Z，且 RZ+Z 远弱于 RY+Z
    default = rows_rand[0.05]["configs"]
    assert default["RZ_feedback_Y_readout"]["mean_abs_grad"] > default["RZ_feedback_X_readout"]["mean_abs_grad"]
    assert default["RZ_feedback_X_readout"]["mean_abs_grad"] > default["RZ_feedback_Z_readout"]["mean_abs_grad"]
    assert (
        default["RZ_feedback_Z_readout"]["mean_abs_grad"]
        < 0.05 * default["RY_feedback_Z_readout"]["mean_abs_grad"]
    )
    forbidden = " ".join(report["claims_forbidden"])
    assert "量子循环不能记忆" in forbidden


def check_quantum_native_models() -> None:
    """Regime III 契约：验证 MPS Born Machine 精确归一化与 True L2 Unitary SSM 酉演化 + PennyLane 等价性。"""
    from itertools import product

    from qml_lab.born_machine import (
        ClassicalCausalRNNLM,
        ClassicalNonNegMPS,
        QuantumMPSBornMachine,
        generate_dyck1_dataset,
        generate_periodic_interference_dataset,
    )
    from qml_lab.quantum_ssm import (
        ClassicalOrthogonalSSM,
        QuantumUnitarySSM,
        build_deterministic_delayed_recall,
        build_s3_dataset,
    )

    # 1. Track 1: QuantumMPSBornMachine & ClassicalNonNegMPS 精确概率归一化 sum_x P(x) == 1
    T_check = 5
    all_x = torch.tensor(list(product((0, 1), repeat=T_check)), dtype=torch.long)
    set_seed(0)
    q_mps = QuantumMPSBornMachine(vocab_size=2, bond_dim=3, complex_amplitudes=False)
    set_seed(0)
    c_mps = ClassicalNonNegMPS(vocab_size=2, bond_dim=3)
    assert sum(p.numel() for p in q_mps.parameters()) == sum(p.numel() for p in c_mps.parameters())
    for m in (q_mps, c_mps):
        with torch.no_grad():
            probs = torch.exp(m.log_prob(all_x))
            assert abs(float(probs.sum().item()) - 1.0) < 1e-4, f"{type(m).__name__} 概率未归一化: {probs.sum()}"
            # 确认采样输出形状与词表合法
            samp = m.sample(8, T_check)
            assert samp.shape == (8, T_check) and int(samp.min()) >= 0 and int(samp.max()) <= 1
        # 确认梯度正常回传
        loss = m.nll_loss(all_x[:8])
        loss.backward()
        assert m.A_real.grad is not None and float(m.A_real.grad.abs().sum()) > 0.0

    # 验证 complex_amplitudes=True 同样精确归一化
    q_mps_c = QuantumMPSBornMachine(vocab_size=2, bond_dim=2, complex_amplitudes=True)
    with torch.no_grad():
        assert abs(float(torch.exp(q_mps_c.log_prob(all_x)).sum().item()) - 1.0) < 1e-4

    # 验证数据集零重叠与 ClassicalCausalRNNLM 归一化
    dyck = generate_dyck1_dataset(train_len=10, ood_len=12, seed=0)
    assert dyck["train"].shape[1] == 10 and dyck["test_ood"].shape[1] == 12
    per = generate_periodic_interference_dataset(modulus=4, train_len=6, ood_lens=(8,), n_train=32, n_val=16, n_test=16)
    assert per["train"].shape == (32, 6)
    rnn = ClassicalCausalRNNLM(vocab_size=2, embed_dim=2, hidden_dim=3)
    with torch.no_grad():
        assert abs(float(torch.exp(rnn.log_prob(all_x)).sum().item()) - 1.0) < 1e-4

    # 2. Track 2: True L2 QuantumUnitarySSM 严格酉性、精确 POVM 完备性、零中间塌缩与 PennyLane 等价性
    set_seed(0)
    qssm = QuantumUnitarySSM(vocab_size=3, n_classes=6, n_qubits=3, povm_rank=2)
    with torch.no_grad():
        U = qssm.token_unitaries()
        eye8 = torch.eye(8, dtype=U.dtype).unsqueeze(0).expand(3, -1, -1)
        assert torch.allclose(U.conj().transpose(-1, -2) @ U, eye8, atol=1e-5), "token_unitaries 未严格酉"
        E = qssm.povm_kraus()
        M_sum = torch.einsum("cri,crj->ij", E.conj(), E)
        assert torch.allclose(M_sum, torch.eye(8, dtype=M_sum.dtype), atol=1e-4), "POVM 不满足 sum M_c = I"

    s3 = build_s3_dataset(train_len=6, ood_lens=(10,), n_train=60, n_val=20, n_test_id=20, n_test_ood=20, seed=0)
    x_sample, y_sample = s3["train"][0][:4], s3["train"][1][:4]
    probs = qssm.forward_probs(x_sample)
    assert probs.shape == (4, 6, 6)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4, 6), atol=1e-4), "POVM Born 概率之和必须为 1"
    pl_chk = qssm.verify_against_pennylane(x_sample[0])
    assert pl_chk["max_abs_state_diff"] < 1e-5, f"与 PennyLane 态矢量不一致: {pl_chk}"
    assert pl_chk["max_abs_prob_diff"] < 1e-5, f"与 PennyLane POVM 概率不一致: {pl_chk}"
    loss_q = torch.nn.functional.nll_loss(qssm(x_sample).reshape(-1, 6), y_sample.reshape(-1))
    loss_q.backward()
    assert qssm.theta.grad is not None and float(qssm.theta.grad.abs().sum()) > 0.0

    # ClassicalOrthogonalSSM 严格正交性与概率和为 1
    ossm = ClassicalOrthogonalSSM(vocab_size=3, n_classes=6, state_dim=9, readout_rank=2)
    with torch.no_grad():
        orth = ossm.token_orthogonals()
        eye9 = torch.eye(9).unsqueeze(0).expand(3, -1, -1)
        assert torch.allclose(orth.transpose(-1, -2) @ orth, eye9, atol=1e-5)
        assert torch.allclose(ossm.forward_probs(x_sample).sum(dim=-1), torch.ones(4, 6), atol=1e-4)

    # 确定性延迟回忆契约：精确 4 条序列
    rec = build_deterministic_delayed_recall(k=4, pad_len=4, ood_pad_len=8)
    assert rec["id"][0].shape == (4, 6) and rec["ood"][0].shape == (4, 10)


def check_quantum_attention() -> None:
    """验证量子自注意力三范式契约：SWAP-Test (<1e-10), Quixer LCU+QSVT (<1e-6), POVM, --quick CLI。"""
    from qml_lab.quantum_attention import (
        ClassicalOverlapAttention,
        QuantumQuixerAttention,
        QuantumSwapTestAttention,
        attention_parameter_breakdown,
        build_kv_recall_dataset,
    )

    set_seed(0)
    ds = build_kv_recall_dataset(n_train=60, n_val=20, n_test_id=20, n_test_ood=20, seed=0)
    x_id, y_id = ds["test_id"]
    assert x_id.shape == (20, 6) and ds["test_ood"][0].shape == (20, 10)

    # 1. QuantumSwapTestAttention: exact parity with PennyLane CSWAP (<1e-10) & ClassicalOverlapAttention
    sw = QuantumSwapTestAttention(vocab_size=ds["vocab_size"], n_classes=ds["n_classes"])
    ov = ClassicalOverlapAttention(vocab_size=ds["vocab_size"], n_classes=ds["n_classes"])
    ov.load_state_dict(sw.state_dict())
    with torch.no_grad():
        max_surr_diff = float((sw(x_id[:8]) - ov(x_id[:8])).abs().max().item())
        assert max_surr_diff < 1e-6, f"SWAP-Test 与 ClassicalOverlapAttention 应解析等价，diff={max_surr_diff}"
    pl_sw = sw.verify_against_pennylane(x_id[0])
    assert pl_sw["max_abs_overlap_diff"] < 1e-10, f"SWAP-Test 与 PennyLane CSWAP 偏差过大: {pl_sw}"
    logits_fs, fs_stats = sw.forward_finite_shots(x_id[:4], shots=200)
    assert logits_fs.shape == (4, 4) and fs_stats["shots_last_token_query"] == 5 * 200
    assert fs_stats["shots_full_causal_matrix"] == 21 * 200

    # 2. QuantumQuixerAttention: exact parity with PennyLane PREPARE/SELECT/PREPARE†+PCPhase (<1e-6),
    #    POVM completeness sum M_c = I_8, zero classical MLP params
    qx = QuantumQuixerAttention(vocab_size=ds["vocab_size"], n_classes=ds["n_classes"], n_qubits=3, povm_rank=2)
    pb = attention_parameter_breakdown(qx)
    assert pb["classical_params"] == 0 and pb["quantum_params"] == pb["total"] > 0, f"Quixer 应零经典参数: {pb}"
    with torch.no_grad():
        M_sum = qx.povm_effects().sum(dim=0)
        assert torch.allclose(M_sum, torch.eye(8, dtype=M_sum.dtype), atol=1e-5), "Quixer POVM 完备性失败"
        _logp, probs, p_succ = qx.forward_with_stats(x_id[:4])
        assert probs.shape == (4, 4)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
        assert bool((p_succ > 0).all() and (p_succ <= 1.0 + 1e-5).all())
    pl_qx = qx.verify_against_pennylane(x_id[0])
    assert pl_qx["max_abs_unnorm_state_diff"] < 1e-6, f"Quixer 态矢量与 PennyLane 不一致: {pl_qx}"
    assert pl_qx["max_abs_prob_diff"] < 1e-6, f"Quixer POVM 概率与 PennyLane 不一致: {pl_qx}"
    loss_qx = torch.nn.functional.nll_loss(qx(x_id[:4]), y_id[:4])
    loss_qx.backward()
    assert qx.theta.grad is not None and float(qx.theta.grad.abs().sum()) > 0.0
    assert qx.qsvt_phases.grad is not None and float(qx.qsvt_phases.grad.abs().sum()) > 0.0

    # 3. benchmark_quantum_attention.py --quick CLI isolation test
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = Path(tmpdir) / "qa_quick.json"
        cmd = [
            sys.executable,
            str(ROOT / "experiments" / "benchmark_quantum_attention.py"),
            "--quick",
            "--out",
            str(out_json),
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"benchmark_quantum_attention.py --quick 失败:\n{proc.stderr}\n{proc.stdout}"
        assert out_json.exists()


def check_quick_cli() -> None:
    """验证 train_llm.py --quick 可正常跑通且不污染默认 results/。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable,
            str(ROOT / "experiments" / "train_llm.py"),
            "--quick",
            "--out",
            tmpdir,
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"train_llm.py --quick 失败:\n{proc.stderr}\n{proc.stdout}"


# Export pytest-discoverable test_* aliases for all check_* suites.
test_data_and_entropy = check_data_and_entropy
test_rich_quantum_layers = check_rich_quantum_layers
test_causal_strictness_and_seed_control = check_causal_strictness_and_seed_control
test_lm = check_lm
test_independent_corpus = check_independent_corpus
test_synthetic_corpus = check_synthetic_corpus
test_arithmetic_task = check_arithmetic_task
test_embedding_modes = check_embedding_modes
test_grokking_scan = check_grokking_scan
test_grokking_no_prior = check_grokking_no_prior
test_recurrent_architecture = check_recurrent_architecture
test_recurrent_pilot = check_recurrent_pilot
test_state_channel_probe = check_state_channel_probe
test_quantum_native_models = check_quantum_native_models
test_quantum_attention = check_quantum_attention
test_quick_cli = check_quick_cli


def main() -> None:
    check_data_and_entropy()
    check_independent_corpus()
    check_synthetic_corpus()
    check_arithmetic_task()
    check_embedding_modes()
    check_grokking_scan()
    check_grokking_no_prior()
    check_rich_quantum_layers()
    check_causal_strictness_and_seed_control()
    check_recurrent_architecture()
    check_recurrent_pilot()
    check_state_channel_probe()
    check_quantum_native_models()
    check_quantum_attention()
    check_lm()
    check_quick_cli()
    print("SMOKE OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
