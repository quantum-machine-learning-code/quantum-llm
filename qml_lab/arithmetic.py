"""课题一：模 112 算术（Z_112）的词表、数据与答案位评估工具。

与历史字符级语料不同，这里 0~111 各自是一个独立 token：

    词表 = 112 个数字 + ``+ - * /`` 四个运算符 + ``=`` 一个等号 = 117

一个样本是定长 5 token 的等式 ``[a][op][b][=][c]``。模型只看前 4 个
token（恰好等于 ``context_window=4``），在答案位预测 ``c``，即 112 类
单 token 分类；因此主指标可以直接是 top-1 准确率，随机基线为 1/112。

方法学要点：

* 全空间 112 x 112 = 12544 个有序对可穷举，训练/验证按 ``(a, b)``
  组合严格 disjoint，验证集输入四元组在训练集中的覆盖率恒为 0。
  这直接补上了历史语料「k=4 上下文重叠 90.2% / 38.7%」的漏洞。
* Z_112 不是域：加、减、乘封闭；除法仅在 ``gcd(b, 112) = 1`` 时有唯一解。
  课题一只生成加法样本，其余三个运算符仅占词表位，供后续课题复用。
* 输出头仍是 117 维（不改动 ``RichQuantumLM``），但 loss 与准确率只在
  前 112 个数字类上计算 —— 答案恒为数字，让符号类参与 softmax 只会
  引入无意义的概率质量。
"""
from __future__ import annotations

import math
import random
from typing import Iterator, Sequence

import torch

MODULUS = 112
N_NUMBER_TOKENS = MODULUS
OPERATORS = ("+", "-", "*", "/")
EQUALS = "="
OP_OFFSET = N_NUMBER_TOKENS
EQ_TOKEN_ID = N_NUMBER_TOKENS + len(OPERATORS)
VOCAB_SIZE = N_NUMBER_TOKENS + len(OPERATORS) + 1  # 117
INPUT_LEN = 4  # [a][op][b][=]
SEQ_LEN = 5  # [a][op][b][=][c]
CONTEXT_WINDOW = 4
RANDOM_BASELINE_ACCURACY = 1.0 / N_NUMBER_TOKENS

# 用户给定的两个边界样例：111+1=0（回绕）、0+0=0（加法恒等元）
BOUNDARY_EXAMPLES = ((111, "+", 1, 0), (0, "+", 0, 0))

__all__ = [
    "MODULUS",
    "N_NUMBER_TOKENS",
    "OPERATORS",
    "EQUALS",
    "EQ_TOKEN_ID",
    "VOCAB_SIZE",
    "INPUT_LEN",
    "SEQ_LEN",
    "CONTEXT_WINDOW",
    "RANDOM_BASELINE_ACCURACY",
    "BOUNDARY_EXAMPLES",
    "build_vocab",
    "operator_token",
    "decode_token",
    "modular_inverse",
    "apply_op",
    "enumerate_table",
    "encode_expression",
    "decode_expression",
    "split_table",
    "ngram_coverage",
    "context_overlap",
    "build_tensors",
    "iter_batches",
    "answer_logits",
    "evaluate_answers",
    "circular_init",
]


def build_vocab() -> tuple[dict[str, int], dict[int, str], int]:
    """返回 ``(stoi, itos, vocab_size)``，数字 token 的 id 恰为数值本身。"""
    stoi = {str(n): n for n in range(N_NUMBER_TOKENS)}
    for i, op in enumerate(OPERATORS):
        stoi[op] = OP_OFFSET + i
    stoi[EQUALS] = EQ_TOKEN_ID
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos, len(stoi)


def operator_token(op: str) -> int:
    if op not in OPERATORS:
        raise ValueError(f"未知运算符 {op!r}，可选 {OPERATORS}")
    return OP_OFFSET + OPERATORS.index(op)


def decode_token(token_id: int) -> str:
    if not 0 <= token_id < VOCAB_SIZE:
        raise ValueError(f"token id 越界: {token_id}")
    if token_id < N_NUMBER_TOKENS:
        return str(token_id)
    if token_id == EQ_TOKEN_ID:
        return EQUALS
    return OPERATORS[token_id - OP_OFFSET]


def modular_inverse(b: int) -> int | None:
    """Z_112 中 b 的乘法逆元；不存在时返回 None。"""
    b %= MODULUS
    if math.gcd(b, MODULUS) != 1:
        return None
    return pow(b, -1, MODULUS)


def apply_op(a: int, b: int, op: str) -> int | None:
    """计算 ``a op b (mod 112)``；除法无唯一解时返回 None。"""
    if not 0 <= a < MODULUS or not 0 <= b < MODULUS:
        raise ValueError(f"操作数必须在 [0, {MODULUS}) 内: {(a, b)}")
    if op == "+":
        return (a + b) % MODULUS
    if op == "-":
        return (a - b) % MODULUS
    if op == "*":
        return (a * b) % MODULUS
    if op == "/":
        inv = modular_inverse(b)
        return None if inv is None else (a * inv) % MODULUS
    raise ValueError(f"未知运算符 {op!r}，可选 {OPERATORS}")


def enumerate_table(op: str = "+") -> list[tuple[int, int, int]]:
    """穷举全部 ``(a, b, c)``；除法会剔除无唯一解的样本。"""
    rows: list[tuple[int, int, int]] = []
    for a in range(MODULUS):
        for b in range(MODULUS):
            c = apply_op(a, b, op)
            if c is not None:
                rows.append((a, b, c))
    return rows


def encode_expression(a: int, b: int, op: str = "+", c: int | None = None) -> list[int]:
    """把等式编码为 5 个 token id；不传 ``c`` 时自动计算。"""
    if c is None:
        c = apply_op(a, b, op)
    if c is None:
        raise ValueError(f"{(a, op, b)} 在 Z_{MODULUS} 中无唯一解")
    if not 0 <= c < MODULUS:
        raise ValueError(f"结果越界: {c}")
    return [a, operator_token(op), b, EQ_TOKEN_ID, c]


def decode_expression(ids: Sequence[int]) -> str:
    """把 4 或 5 个 token id 还原成可读等式，如 ``111+1=0``。"""
    if len(ids) not in (INPUT_LEN, SEQ_LEN):
        raise ValueError(f"等式长度必须是 {INPUT_LEN} 或 {SEQ_LEN}，得到 {len(ids)}")
    return "".join(decode_token(int(i)) for i in ids)


def split_table(
    table: Sequence[tuple[int, int, int]],
    mode: str = "holdout_pairs",
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """按组合或按操作数切分，保证 train/val 的 ``(a, b)`` 严格不相交。

    ``holdout_pairs``   : 随机留出 ``val_fraction`` 的 ``(a, b)`` 组合（grokking 标准设置）。
    ``holdout_operand`` : 整行留出部分 ``a`` 值，验证时这些 a 从未作为第一操作数出现。
    """
    if mode not in ("holdout_pairs", "holdout_operand"):
        raise ValueError(f"未知切分模式 {mode!r}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction 必须在 (0, 1) 内: {val_fraction}")
    rng = random.Random(seed)
    if mode == "holdout_pairs":
        rows = list(table)
        rng.shuffle(rows)
        n_val = max(1, int(round(len(rows) * val_fraction)))
        return rows[n_val:], rows[:n_val]

    a_values = sorted({a for a, _, _ in table})
    rng.shuffle(a_values)
    n_hold_a = max(1, int(round(len(a_values) * val_fraction)))
    held = set(a_values[:n_hold_a])
    train = [r for r in table if r[0] not in held]
    val = [r for r in table if r[0] in held]
    return train, val


def ngram_coverage(
    train_pairs: Sequence[tuple[int, int, int]],
    val_pairs: Sequence[tuple[int, int, int]],
    k: int,
    op: str = "+",
) -> float:
    """验证集输入里长度 ``k`` 的上下文片段在训练集出现过的比例。

    输入前缀固定为 ``[a][op][b][=]``。k=4 即模型实际使用的完整上下文，
    主口径；k<4 用来说明「数字都见过、但组合没见过」这一区别。
    """
    if not 1 <= k <= INPUT_LEN:
        raise ValueError(f"k 必须在 [1, {INPUT_LEN}] 内: {k}")
    op_id = operator_token(op)

    def prefix_ids(rows: Sequence[tuple[int, int, int]]) -> list[list[int]]:
        return [[a, op_id, b, EQ_TOKEN_ID][:k] for a, b, _ in rows]

    train_grams = {tuple(g) for g in prefix_ids(train_pairs)}
    val_grams = prefix_ids(val_pairs)
    if not val_grams:
        return 0.0
    return sum(1 for g in val_grams if tuple(g) in train_grams) / len(val_grams)


def context_overlap(
    train_pairs: Sequence[tuple[int, int, int]],
    val_pairs: Sequence[tuple[int, int, int]],
    mode: str = "holdout_pairs",
    op: str = "+",
) -> dict:
    """量化训练/验证独立性，口径与历史语料审查保持一致。"""
    train_keys = {(a, b) for a, b, _ in train_pairs}
    val_keys = {(a, b) for a, b, _ in val_pairs}
    cov = {
        f"k{k}": ngram_coverage(train_pairs, val_pairs, k, op)
        for k in range(1, INPUT_LEN + 1)
    }
    train_a = {a for a, _, _ in train_pairs}
    val_a = {a for a, _, _ in val_pairs}
    train_numbers = train_a | {b for _, b, _ in train_pairs}
    val_numbers = val_a | {b for _, b, _ in val_pairs}
    return {
        "mode": mode,
        "modulus": MODULUS,
        "operator": op,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "pair_intersection": len(train_keys & val_keys),
        "pair_level_disjoint": not (train_keys & val_keys),
        "full_context_coverage_k4": cov["k4"],
        "coverage_by_k": cov,
        "val_first_operand_novel_ratio": len(val_a - train_a) / max(1, len(val_a)),
        "val_number_token_coverage_by_train": len(val_numbers & train_numbers)
        / max(1, len(val_numbers)),
    }


def build_tensors(
    pairs: Sequence[tuple[int, int, int]], op: str = "+"
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 ``(x, y)``：x 为 (N, 4) 输入前缀，y 为 (N,) 答案 token id。"""
    op_id = operator_token(op)
    x = torch.tensor(
        [[a, op_id, b, EQ_TOKEN_ID] for a, b, _ in pairs], dtype=torch.long
    )
    y = torch.tensor([c for _, _, c in pairs], dtype=torch.long)
    if x.numel() and (x.max() >= VOCAB_SIZE or y.max() >= N_NUMBER_TOKENS):
        raise ValueError("token id 越界")
    return x, y


def iter_batches(
    x: torch.Tensor, y: torch.Tensor, batch_size: int, seed: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """无限生成器，用独立 ``torch.Generator`` 保证可复现。"""
    g = torch.Generator().manual_seed(int(seed))
    n = x.shape[0]
    while True:
        idx = torch.randint(0, n, (batch_size,), generator=g)
        yield x[idx], y[idx]


def answer_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """取最后位置的 logits 并裁到 112 个数字类，返回 (B, 112)。"""
    logits = model(x)  # (B, T, V)
    return logits[:, -1, :N_NUMBER_TOKENS]


def circular_init(model: torch.nn.Module) -> None:
    """把 112 个数字 token 的 embedding 前 4 维设为循环角度编码。

    ``[sin(2πn/112), cos(2πn/112), sin(4πn/112), cos(4πn/112)]`` 让
    Z_112 的群结构在输入侧可见；5 个符号 token 占第 4~8 维的独热位。

    两臂调用完全相同的函数，因此这是公平共享的**经典先验**，不是量子优势。
    必须使用独立 generator：量子层 ``TorchLayer`` 初始化会额外消耗 torch
    全局 RNG，若用全局 RNG 会让两臂的噪声部分不一致、破坏受控对照。
    """
    with torch.no_grad():
        w = model.embed.weight
        g = torch.Generator(device=w.device).manual_seed(20261113)
        w.normal_(0, 0.02, generator=g)
        for n in range(MODULUS):
            th = 2 * math.pi * n / MODULUS
            w[n, 0] = math.sin(th)
            w[n, 1] = math.cos(th)
            w[n, 2] = math.sin(2 * th)
            w[n, 3] = math.cos(2 * th)
        for k in range(MODULUS, VOCAB_SIZE):
            if 4 + k - MODULUS < w.shape[1]:
                w[k, 4 + k - MODULUS] = 1.0


def evaluate_answers(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 512,
) -> dict:
    """确定性全量评估：答案位 112 类 top-1 准确率 / CE loss / PPL。"""
    model.eval()
    total_loss, correct, n = 0.0, 0, int(y.shape[0])
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            lg = answer_logits(model, xb)
            total_loss += torch.nn.functional.cross_entropy(
                lg, yb, reduction="sum"
            ).item()
            correct += int((lg.argmax(dim=-1) == yb).sum().item())
    loss = total_loss / max(1, n)
    return {
        "n": n,
        "answer_accuracy": correct / max(1, n),
        "answer_correct": correct,
        "answer_loss": loss,
        "answer_perplexity": math.exp(min(loss, 20.0)),
        "random_baseline_accuracy": RANDOM_BASELINE_ACCURACY,
    }
