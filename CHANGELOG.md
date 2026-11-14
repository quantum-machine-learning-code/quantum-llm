# Changelog

All notable changes to `quantum-llm` will be documented in this file.

## [0.2.0] - 2026-11-14

### Added
- Packaging metadata via [`pyproject.toml`](pyproject.toml) (`qml_lab` package, Python `>=3.11`, optional `[dev]` dependencies, pytest and ruff configuration).
- GitHub Actions CI workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) running `ruff check .` and `pytest -v tests/smoke_test.py` on Python 3.11 and 3.12 with CPU-only PyTorch.
- Monthly Dependabot automation ([`.github/dependabot.yml`](.github/dependabot.yml)) for `pip` and `github-actions`.
- English documentation [`README_EN.md`](README_EN.md) with bidirectional language switcher in [`README.md`](README.md), plus [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Task 1 modulo-112 addition (`experiments/train_addition.py`), three-way embedding ablation (`experiments/embedding_ablation.py`), and negative grokking scan (`experiments/grokking_scan.py`).
- Task 2 disjoint independent-corpus benchmark (`experiments/independent_corpus.py`) and 20-seed robustness audit (`experiments/robustness_check.py`).

### Changed
- Exported `test_* = check_*` aliases for all 10 offline contract/smoke suites in [`tests/smoke_test.py`](tests/smoke_test.py), and isolated `check_quick_cli()` output to a temporary directory via `--out`.
- Clarified directed two-body Pauli pairing (`Z_i Z_j` and `Z_i X_j`, $n(n-1)$ operators per family, 48 total on 4 qubits with single-qubit `X/Y/Z`) in [`qml_lab/quantum.py`](qml_lab/quantum.py).
- Cleaned up unused imports across `qml_lab/models.py` and `experiments/`.

## [0.1.0] - 2026-11-07

### Added
- Initial controlled methodological audit of toy quantum character-level language models (`TinyLM`, `RichQuantumLM`, `CausalContextEncoder`), corpus $k$-gram overlap auditing, and 11-group × 5-seed readout bottleneck benchmark.
