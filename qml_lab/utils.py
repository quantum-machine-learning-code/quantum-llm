"""qml_lab 通用工具：随机种子、路径、JSON、设备、参数量与终端表格。

仅依赖标准库 / numpy / torch，不引入其它项目模块。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "set_seed",
    "ensure_dir",
    "save_json",
    "pick_device",
    "count_parameters",
    "fmt_table",
]


def set_seed(seed: int = 0) -> None:
    """统一设置 random / numpy / torch / cuda 随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN 确定性，避免卷积类算子引入非确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path) -> Path:
    """确保目录存在并返回 pathlib.Path。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(obj: Any) -> Any:
    # 实验指标常含 numpy / torch 标量或 Path，统一转成可序列化类型
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return str(obj)


def save_json(obj, path) -> Path:
    """以 UTF-8 / indent=2 / ensure_ascii=False 写入 JSON，返回文件路径。"""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    return p


def pick_device(prefer_gpu: bool = False) -> torch.device:
    """选择计算设备；默认 CPU，prefer_gpu=True 时优先 CUDA / MPS。"""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu:
        try:  # MPS 仅在 macOS 新版 torch 上存在
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def count_parameters(module: torch.nn.Module) -> int:
    """返回模块全部参数的元素总数。"""
    return sum(p.numel() for p in module.parameters())


def _looks_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def fmt_table(rows: list[list], headers: list[str]) -> str:
    """把二维数据渲染为纯字符串对齐表格，用于终端打印；数值列右对齐。"""
    str_rows = [[str(c) for c in row] for row in rows]
    cols = max([len(headers)] + [len(r) for r in str_rows])
    head = [str(h) for h in headers] + [""] * (cols - len(headers))

    widths = [len(head[i]) for i in range(cols)]
    for row in str_rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(row[i]) if i < len(row) else 0)

    right = []
    for i in range(cols):
        vals = [row[i] for row in str_rows if i < len(row) and row[i] != ""]
        right.append(bool(vals) and all(_looks_number(v) for v in vals))

    def render(cells: list[str]) -> str:
        parts = []
        for i in range(cols):
            c = cells[i] if i < len(cells) else ""
            parts.append(c.rjust(widths[i]) if right[i] else c.ljust(widths[i]))
        return " | ".join(parts)

    lines = [render(head), "-+-".join("-" * w for w in widths)]
    lines.extend(render(row) for row in str_rows)
    return "\n".join(lines)
