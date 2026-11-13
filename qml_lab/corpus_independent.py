"""独立语料（2026-11-13）：句级 disjoint、主题分离的字符级训练 / 验证集。

动机见 ``docs/REVIEW.md`` 第 7 节：旧 ``_HELDOUT_VAL_CORPUS`` 虽是重新组合的新句，
但在模型实际使用的因果窗口 k=4 上，验证集 90.2% 的上下文在训练集中出现过，
``val PPL 4.29`` 只能读作同分布重组句迁移，不能作为泛化证据。

本模块给出：

* ``INDEPENDENT_TRAIN_SENTENCES``：44 句自写英文，主题为量子线路 / 线性代数 /
  优化与测量，句子互不重复，训练文本由它们各出现一次拼接而成（不复读整段）。
* ``INDEPENDENT_VAL_SENTENCES``：14 句自写英文，主题换成航海 / 烘焙 / 地质 /
  天气 / 园艺，实词与训练语料不重合，且句子级严格 disjoint。
* 只用小写字母、空格、句点与逗号，保证验证集字符不超出训练集词表（零 UNK）。

诚实边界：字符级英文文本**不可能**做到完全零共享 4-gram（``" the "``、``"and "``
等功能片段必然重合）。因此本模块报告逐 k 覆盖率与「验证集新 4-gram 比例」，
并把它当作相对于旧语料 90.2% 的**显著改善**，而不宣称「完全独立分布」。

用法：
    from qml_lab.corpus_independent import load_independent_split, corpus_overlap
    train, val = load_independent_split()
    report = corpus_overlap(train, val)
"""
from __future__ import annotations

from collections import Counter

__all__ = [
    "INDEPENDENT_TRAIN_SENTENCES",
    "INDEPENDENT_VAL_SENTENCES",
    "load_independent_split",
    "ngrams",
    "corpus_overlap",
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


def load_independent_split() -> tuple[str, str]:
    """返回 (train_text, val_text)；两者由互不重叠的完整句子拼接而成。"""
    train = " ".join(INDEPENDENT_TRAIN_SENTENCES)
    val = " ".join(INDEPENDENT_VAL_SENTENCES)
    assert_disjoint(train, val)
    return train, val


def assert_disjoint(train: str, val: str) -> None:
    """断言：句级 disjoint、验证集无训练集之外的字符、验证集确有新 4-gram。"""
    train_sentences = set(INDEPENDENT_TRAIN_SENTENCES)
    val_sentences = set(INDEPENDENT_VAL_SENTENCES)
    overlap = train_sentences & val_sentences
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


def corpus_overlap(train: str, val: str, ks: tuple[int, ...] = (4, 6, 8, 12, 20)) -> dict:
    """量化验证集上下文在训练集中出现过的比例，并给出相对旧语料的对照口径。

    ``val_token_coverage`` 与 ``audit_corpus.py`` 完全一致：逐位置统计验证集的
    k-gram 是否落在训练集的 k-gram 集合里（token 级，非 unique 级）。
    """
    out: dict = {
        "train_chars": len(train),
        "val_chars": len(val),
        "train_sentences": len(INDEPENDENT_TRAIN_SENTENCES),
        "val_sentences": len(INDEPENDENT_VAL_SENTENCES),
        "sentence_level_disjoint": True,
        "val_charset_outside_train": sorted(set(val) - set(train)),
        "train_unique_4gram_ratio": round(
            len(set(ngrams(train, 4))) / max(len(ngrams(train, 4)), 1), 4),
        "val_new_4gram_ratio": round(
            len(set(ngrams(val, 4)) - set(ngrams(train, 4)))
            / max(len(set(ngrams(val, 4))), 1), 4),
        "val_context_covered_by_train": {},
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
        "字符级英文文本无法做到零共享 4-gram（功能片段必然重合）；本语料只把 "
        "k=4 覆盖率相对旧语料显著压低，并改用句级 disjoint + 不同主题实词。"
        "因此 val loss 仍应读作「主题外新句迁移」，不是严格的未见分布泛化。"
    )
    return out


def word_bigram_novelty(train: str, val: str) -> dict:
    """补充口径：验证集中未在训练集出现过的 word bigram 占比。"""
    tr = Counter(
        zip(train.replace(".", " ").replace(",", " ").split(),
            train.replace(".", " ").replace(",", " ").split()[1:])
    )
    val_words = val.replace(".", " ").replace(",", " ").split()
    pairs = list(zip(val_words, val_words[1:]))
    fresh = sum(1 for p in pairs if p not in tr)
    return {
        "val_word_bigrams": len(pairs),
        "novel_word_bigrams": fresh,
        "novel_ratio": round(fresh / max(len(pairs), 1), 4),
    }
