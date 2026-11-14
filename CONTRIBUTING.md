# Contributing to quantum-llm

Thank you for helping keep `quantum-llm` reproducible and methodologically sound!

## Development Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e ".[dev]"
```

## Verification Before Opening a Pull Request

Run both lint checks and the full offline smoke/contract test suite:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -v tests/smoke_test.py
```

## Methodological & Coding Standards

1. **Controlled comparisons**: Any comparison between quantum and classical layers must keep random seeds, batch ordering, shared front-end/head initial weights, and train/validation splits identical across arms.
2. **Pre-specified statistics**: Report deterministic full-corpus or unseen-split evaluations with mean ± std, paired confidence intervals, and multiple-testing corrections (Holm or Bonferroni) rather than single-step mini-batch losses.
3. **Offline reproducibility**: Unit/contract tests in `tests/` must run offline without external downloads or hardware credentials, and temporary test outputs should not pollute `docs/` or `results/`.
4. **Honest negative results**: Document negative or null findings (such as non-grokking, overfitting on disjoint splits, and simulator runtime ceilings) alongside any positive fits.
