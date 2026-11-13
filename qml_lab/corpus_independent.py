"""独立语料（2026-11-13）：句级 disjoint 的字符级训练 / 验证集，提供两份口径。

动机见 ``docs/REVIEW.md`` 第 7 节：旧 ``_HELDOUT_VAL_CORPUS`` 虽是重新组合的新句，
但在模型实际使用的因果窗口 k=4 上，验证集 90.2% 的上下文在训练集中出现过，
``val PPL 4.29`` 只能读作同分布重组句迁移，不能作为泛化证据。

本模块给出两份语料：

A. ``independent``（默认，REVIEW 第 8 节）
   44 句自写英文训练 / 14 句主题分离验证（量子线路 / 线代 / 优化 vs 航海 / 烘焙 /
   地质 / 天气 / 园艺），句级 disjoint、零 UNK；k=4 覆盖率 38.7%，字符 unigram
   JS 散度 0.0094。用于把「同分布重组」升级为「主题外新句迁移」。

B. ``synthetic``（REVIEW 第 9 节）
   固定 seed 生成的**受控合成伪词**，专门用于分布偏移压力测试：训练/验证音节集合
   严格不相交，验证集只用常见辅音而训练集偏向生僻辅音，因而 k=4 覆盖率降到
   0.24%、字符 unigram JS 散度升到 0.2538（约为 A 的 27 倍）；验证集字符仍全部
   落在训练集词表内（零 UNK）。它不是自然语言，只用于检验第 8 节的
   「量子层泛化间隙更大」结论是否依赖某一特定语料。

诚实边界：字符级英文文本**不可能**做到完全零共享 4-gram（``" the "``、``"and "``
等功能片段必然重合）；A 只把 k=4 覆盖率相对旧语料显著压低。B 是人工伪词，
分布偏移远大于自然文本，结论只能读作「在极端字符分布偏移下两臂的相对表现」，
不能外推为自然语言泛化能力。两份语料都报告逐 k 覆盖率与分布散度，
不宣称「完全独立分布」。

用法：
    from qml_lab.corpus_independent import load_split, corpus_overlap
    train, val = load_split("independent")     # 或 "synthetic"
    report = corpus_overlap(train, val)
"""
from __future__ import annotations

from collections import Counter
import math
import random

__all__ = [
    "INDEPENDENT_TRAIN_SENTENCES",
    "INDEPENDENT_VAL_SENTENCES",
    "SYNTHETIC_TRAIN_SENTENCES",
    "SYNTHETIC_VAL_SENTENCES",
    "load_independent_split",
    "load_synthetic_split",
    "load_split",
    "ngrams",
    "corpus_overlap",
    "char_distribution_divergence",
    "assert_disjoint",
]

INDEPENDENT_TRAIN_SENTENCES: tuple[str, ...] = (
    "a qubit keeps a phase that a plain binary unit cannot store in the same way.",
    "we rotate the state by a tiny angle and then read the final answer again.",
    "each gate adds one more turn to the path the amplitude will follow.",
    "the gradient tells us which parameter to nudge and by how much to nudge it.",
    "a loss curve that falls too fast often means the batch was far too small.",
    "the optimizer keeps a running mean of every past squared gradient term.",
    "we normalise each input vector before it enters the first linear map.",
    "a matrix with orthonormal columns preserves the norm of any real vector.",
    "the expectation of a pauli string is a single real number between minus one and one.",
    "shot noise shrinks only as the square root of the number of repeats.",
    "two observables that commute can be measured in one common basis.",
    "the readout head maps a fixed width feature into one score per token.",
    "a deeper circuit has more angles to tune but also a flatter landscape.",
    "we freeze the encoder for the first epoch and then release every weight.",
    "the learning rate is cut by half whenever the validation curve stops falling.",
    "a kernel built from inner products needs no explicit feature coordinates.",
    "the training set and the test set must never share a single labelled image.",
    "we clip each gradient so that no single step can destroy the whole run.",
    "an eigenbasis diagonalises the operator and makes the spectrum easy to read.",
    "the amplitude of a basis state is a complex number with a squared modulus.",
    "a unitary map preserves total probability across the entire state space.",
    "we group the pauli terms by qubit wise commuting sets before any measurement.",
    "the projector onto a subspace is idempotent by construction.",
    "a shallow ansatz may fail to represent the target function at all.",
    "we record the seed, the version and the full command line for every run.",
    "pairing the seeds turns a noisy comparison into a much tighter interval.",
    "a confidence interval that crosses zero cannot support a strong claim.",
    "we report the corrected count when many pairs are tested at once.",
    "the norm of the residual is a cheap proxy for how well the fit converged.",
    "a validation split drawn from the same paragraph leaks the context.",
    "we shuffle at the sentence level rather than at the character level.",
    "the entropy of a nearly deterministic corpus is close to zero nats.",
    "a model that memorises short contexts will fail on a fresh vocabulary.",
    "we scale the logits by a constant before the softmax to keep gradients alive.",
    "the backprop pass through a simulated circuit is exact but not free.",
    "a finite difference check can catch a wrong gradient in a few minutes.",
    "the batch generator samples a start index and slices a fixed window.",
    "we pad the final window so the last character is never silently dropped.",
    "a held out sentence is only useful if its contexts were never seen.",
    "the weight decay term pulls large coefficients back toward the origin.",
    "we compare a quantum layer against a classical layer of the same width.",
    "an ablation that removes the circuit is the only fair test of its value.",
    "the wall clock cost of a state vector grows as two to the qubit count.",
    "we keep the head, the seed and the step budget identical across both arms.",
)

INDEPENDENT_VAL_SENTENCES: tuple[str, ...] = (
    "the tide pulled a little wooden boat away from the quiet stone pier.",
    "warm bread rose slowly beneath a clean linen cloth beside the old oven.",
    "river silt settled in thin brown layers along the gravel of the valley.",
    "frost silvered the cedar branches before the morning cloud lifted.",
    "a lantern swung from the mast while the crew coiled a wet rope.",
    "cinnamon and yeast gave the kitchen a sweet and heavy smell.",
    "granite cliffs hid a small fossil under a shelf of pale marble.",
    "rain tapped the greenhouse roof and filled every clay pot to the brim.",
    "the orchard keeper tied each young root to a short bamboo stake.",
    "petals drifted across the meadow and stuck to the damp garden soil.",
    "a pebble skipped twice and vanished into the dark green harbor water.",
    "the kettle whistled as the wind pushed a cold squall over the hill.",
    "ferns curled under the hedge where the morning shadow stayed longest.",
    "harvest baskets waited in a row by the gate of the sleepy village.",
)

# ---------------------------------------------------------------------------
# 语料 B：受控合成伪词（分布偏移压力测试，REVIEW 第 9 节）
# ---------------------------------------------------------------------------
# 音节都是 CV 结构。训练集音节 = 生僻辅音×全部元音 + 常见辅音×'a'（后者只为
# 覆盖全部 21 个辅音，保证验证集零 UNK）；验证集音节 = 常见辅音×(e,i,o,u)，
# 与训练集音节集合严格不相交。因此验证集字符 ⊆ 训练集字符，但 4-gram 几乎全新。
_SYNTH_SEED = 20261113
_VOWELS = "aeiou"
_C_RARE = "zqxjkwvbgpcfy"          # 13 个：生僻辅音 + 剩余辅音
_C_COMMON = "tnshrldm"             # 8 个常见辅音
_TRAIN_SYLLABLES: tuple[str, ...] = tuple(sorted(
    {c + v for c in _C_RARE for v in _VOWELS} | {c + "a" for c in _C_COMMON}
))
_VAL_SYLLABLES: tuple[str, ...] = tuple(sorted(
    {c + v for c in _C_COMMON for v in "eiou"}
))


def _make_synthetic_sentences(pool: tuple[str, ...], count: int, salt: int,
                              words_per_sentence: tuple[int, int] = (9, 12)) -> tuple[str, ...]:
    """固定 seed 生成伪词句子；每个词由 2–3 个音节拼成，句末句点。"""
    r = random.Random(_SYNTH_SEED + salt)
    out: list[str] = []
    for _ in range(count):
        n_words = r.randint(*words_per_sentence)
        words = [
            "".join(r.choice(pool) for _ in range(r.choice((2, 2, 3))))
            for _ in range(n_words)
        ]
        out.append(" ".join(words) + ".")
    return tuple(out)


SYNTHETIC_TRAIN_SENTENCES = _make_synthetic_sentences(_TRAIN_SYLLABLES, 44, 1)
SYNTHETIC_VAL_SENTENCES = _make_synthetic_sentences(_VAL_SYLLABLES, 14, 2)

_CORPUS_REGISTRY: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "independent": (INDEPENDENT_TRAIN_SENTENCES, INDEPENDENT_VAL_SENTENCES),
    "synthetic": (SYNTHETIC_TRAIN_SENTENCES, SYNTHETIC_VAL_SENTENCES),
}


def load_split(which: str = "independent") -> tuple[str, str]:
    """返回 (train_text, val_text)；两者由互不重叠的完整句子拼接而成。"""
    if which not in _CORPUS_REGISTRY:
        raise ValueError(f"未知语料 {which!r}，可选 {sorted(_CORPUS_REGISTRY)}")
    train_s, val_s = _CORPUS_REGISTRY[which]
    train, val = " ".join(train_s), " ".join(val_s)
    assert_disjoint(train, val, train_sentences=train_s, val_sentences=val_s)
    return train, val


def load_independent_split() -> tuple[str, str]:
    """语料 A（自写英文、主题分离），REVIEW 第 8 节。"""
    return load_split("independent")


def load_synthetic_split() -> tuple[str, str]:
    """语料 B（受控合成伪词、极端分布偏移），REVIEW 第 9 节。"""
    return load_split("synthetic")


def assert_disjoint(train: str, val: str, *,
                    train_sentences: tuple[str, ...] | None = None,
                    val_sentences: tuple[str, ...] | None = None) -> None:
    """断言：句级 disjoint、验证集无训练集之外的字符、验证集确有新 4-gram。"""
    tr_s = train_sentences if train_sentences is not None else INDEPENDENT_TRAIN_SENTENCES
    va_s = val_sentences if val_sentences is not None else INDEPENDENT_VAL_SENTENCES
    overlap = set(tr_s) & set(va_s)
    if overlap:
        raise AssertionError(f"训练/验证集出现重复句子: {sorted(overlap)[:3]}")
    extra = set(val) - set(train)
    if extra:
        raise AssertionError(f"验证集引入训练集词表之外的字符（会产生 UNK）: {sorted(extra)}")
    fresh = set(ngrams(val, 4)) - set(ngrams(train, 4))
    if not fresh:
        raise AssertionError("验证集没有任何新 4-gram，独立性不足")
    if len(fresh) / max(len(set(ngrams(val, 4))), 1) < 0.30:
        raise AssertionError("验证集新 4-gram 占比过低，独立性不足")


def ngrams(text: str, k: int) -> list[str]:
    """返回文本的全部字符 k-gram（有序，含重复），与 audit_corpus.py 口径一致。"""
    return [text[i:i + k] for i in range(len(text) - k + 1)]


def char_distribution_divergence(train: str, val: str) -> dict:
    """量化训练/验证集的字符 unigram 分布差异（JS 散度等）。

    JS 用两分布的混合作为参照，天然支持「验证集出现训练集没有的字符」的情形；
    该情形同时由 ``val_chars_outside_train`` 显式报告（应为空，否则会产生 UNK）。
    """
    ct, cv = Counter(train), Counter(val)
    pt = {k: v / len(train) for k, v in ct.items()}
    pv = {k: v / len(val) for k, v in cv.items()}
    keys = set(pt) | set(pv)
    mix = {k: (pt.get(k, 0.0) + pv.get(k, 0.0)) / 2 for k in keys}

    def kl(a: dict[str, float], b: dict[str, float]) -> float:
        return sum(a[k] * math.log(a[k] / b[k])
                   for k in keys if a.get(k, 0.0) > 0 and b.get(k, 0.0) > 0)

    kl_vt = kl(pv, pt) if all(pt.get(k, 0.0) > 0 for k in pv) else None
    return {
        "js_unigram": round(0.5 * kl(pt, mix) + 0.5 * kl(pv, mix), 6),
        "kl_val_train": round(kl_vt, 6) if kl_vt is not None else None,
        "train_vocab_size": len(ct),
        "val_vocab_size": len(cv),
        "val_chars_outside_train": sorted(set(val) - set(train)),
        "train_top5": [(k, round(v, 4)) for k, v in ct.most_common(5)],
        "val_top5": [(k, round(v, 4)) for k, v in cv.most_common(5)],
    }


def corpus_overlap(train: str, val: str, ks: tuple[int, ...] = (4, 6, 8, 12, 20),
                   *, train_sentences: tuple[str, ...] | None = None,
                   val_sentences: tuple[str, ...] | None = None) -> dict:
    """量化验证集上下文在训练集中出现过的比例，并给出相对旧语料的对照口径。

    ``val_token_coverage`` 与 ``audit_corpus.py`` 完全一致：逐位置统计验证集的
    k-gram 是否落在训练集的 k-gram 集合里（token 级，非 unique 级）。
    传入 ``train_sentences`` / ``val_sentences`` 可统计非默认语料的句子数。
    """
    tr_s = train_sentences if train_sentences is not None else INDEPENDENT_TRAIN_SENTENCES
    va_s = val_sentences if val_sentences is not None else INDEPENDENT_VAL_SENTENCES
    out: dict = {
        "train_chars": len(train),
        "val_chars": len(val),
        "train_sentences": len(tr_s),
        "val_sentences": len(va_s),
        "sentence_level_disjoint": len(set(tr_s) & set(va_s)) == 0,
        "val_charset_outside_train": sorted(set(val) - set(train)),
        "train_unique_4gram_ratio": round(
            len(set(ngrams(train, 4))) / max(len(ngrams(train, 4)), 1), 4),
        "val_new_4gram_ratio": round(
            len(set(ngrams(val, 4)) - set(ngrams(train, 4)))
            / max(len(set(ngrams(val, 4))), 1), 4),
        "val_context_covered_by_train": {},
        "char_distribution_divergence": char_distribution_divergence(train, val),
    }
    for k in ks:
        g_train = set(ngrams(train, k))
        g_val = ngrams(val, k)
        covered = sum(1 for g in g_val if g in g_train) / max(len(g_val), 1)
        out["val_context_covered_by_train"][str(k)] = {
            "k": k,
            "train_unique": len(g_train),
            "val_unique": len(set(g_val)),
            "shared": len(g_train & set(g_val)),
            "val_token_coverage": round(covered, 4),
        }
    # 旧语料（_HELDOUT_VAL_CORPUS）的公开审查值，用于并列展示，数字来自 docs/corpus_audit.json
    out["legacy_k4_coverage"] = 0.9024
    out["caveat"] = (
        "字符级英文文本无法做到零共享 4-gram（功能片段必然重合）；independent 语料只把 "
        "k=4 覆盖率相对旧语料显著压低，并改用句级 disjoint + 不同主题实词。"
        "synthetic 语料是人工伪词、分布偏移远大于自然文本，只能用于压力测试。"
        "两份语料的 val loss 都不是严格的未见分布自然语言泛化。"
    )
    return out


def word_bigram_novelty(train: str, val: str) -> dict:
    """补充口径：验证集中未在训练集出现过的 word bigram 占比。"""
    tr_words = train.replace(".", " ").replace(",", " ").split()
    tr = Counter(zip(tr_words, tr_words[1:]))
    val_words = val.replace(".", " ").replace(",", " ").split()
    pairs = list(zip(val_words, val_words[1:]))
    fresh = sum(1 for p in pairs if p not in tr)
    return {
        "val_word_bigrams": len(pairs),
        "novel_word_bigrams": fresh,
        "novel_ratio": round(fresh / max(len(pairs), 1), 4),
    }
