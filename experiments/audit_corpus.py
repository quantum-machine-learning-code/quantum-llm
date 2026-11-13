"""语料审查（2026-11-07）：量化「留出验证集」与训练语料的 n-gram 重叠。

`0480629` 用 ``_HELDOUT_VAL_CORPUS``（382 字符）作为「独立未见句验证集」，
并以 ``caus_q_rich48`` 的 val loss 1.4494 / PPL 4.29 作为泛化证据。
本脚本不改任何实验，只复算：

* 训练集是同一段 417 字符文本重复 4 遍，unique 4-gram 占比极低；
* 验证集虽是「新句子」，但在模型实际使用的因果上下文长度 k=4 上，
  绝大多数 token 的上下文在训练集中出现过。

这决定 val PPL 应读作「同分布重组句的迁移」，不能读作严格意义的未见分布泛化。

用法：
    python experiments/audit_corpus.py                 # 归档 docs/corpus_audit.json
    python experiments/audit_corpus.py --no-archive   # 只打印
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qml_lab.data import _CORPUS_BLOCK, corpus_entropy_floors, load_corpus_split  # noqa: E402


def grams(text: str, k: int) -> list[str]:
    return [text[i:i + k] for i in range(len(text) - k + 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-archive", action="store_true")
    args = ap.parse_args()

    train, val = load_corpus_split()
    result = {
        "purpose": "审查 0480629 的留出验证集是否真正「未见」",
        "train_chars": len(train),
        "val_chars": len(val),
        "corpus_block_chars": len(_CORPUS_BLOCK),
        "train_block_repeats": round(len(train) / len(_CORPUS_BLOCK), 3),
        "train_unique_4gram_ratio": round(
            len(set(grams(train, 4))) / max(len(grams(train, 4)), 1), 4),
        "val_context_covered_by_train": {},
    }
    for k in (4, 6, 8, 12, 20):
        g_train = set(grams(train, k))
        g_val = grams(val, k)
        covered = sum(1 for g in g_val if g in g_train) / max(len(g_val), 1)
        result["val_context_covered_by_train"][str(k)] = {
            "k": k,
            "train_unique": len(g_train),
            "val_unique": len(set(g_val)),
            "shared": len(g_train & set(g_val)),
            "val_token_coverage": round(covered, 4),
        }
        print(f"k={k:2d}: val token 的 {k}-gram 上下文在训练集中出现过的比例 = {covered:.3f}",
              flush=True)
    print(f"训练集 {len(train)} 字符 = 同一段 {len(_CORPUS_BLOCK)} 字符重复 "
          f"{result['train_block_repeats']} 次；unique 4-gram 占比 "
          f"{result['train_unique_4gram_ratio']:.3f}", flush=True)

    try:
        floors = corpus_entropy_floors(train, max_context=8)
        result["train_corpus_entropy_floors"] = floors
    except Exception as exc:  # 诊断性字段，失败不影响主结论
        result["train_corpus_entropy_floors_error"] = repr(exc)

    result["caveat"] = (
        "n-gram 覆盖率是必要条件而非充分条件：即使上下文见过，"
        "模型仍需组合出新的句子级续写；但 k=4 覆盖 90% 以上意味着"
        "「留出验证集」无法排除记忆/查表式解法，val PPL 4.29 不能作为"
        "量子层带来泛化能力的证据。"
    )
    if not args.no_archive:
        out = ROOT / "docs" / "corpus_audit.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
                       encoding="utf-8")
        print(f"archive -> {out}", flush=True)
    print("CORPUS AUDIT OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
