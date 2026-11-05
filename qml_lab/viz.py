# -*- coding: utf-8 -*-
"""matplotlib 绘图工具：训练曲线与指标柱状图（Agg，无需显示环境）。"""
from __future__ import annotations

import warnings
from pathlib import Path

try:
    from .utils import ensure_dir
except ImportError:  # 兼容把 qml_lab/ 直接加入 sys.path 后按顶层模块导入
    from utils import ensure_dir

# 必须在任何 pyplot 导入前设置 Agg；导入/初始化失败也不阻断实验
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 中文字体回退链 + 关闭 unicode minus，缺字体不因警告/异常中断
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "DejaVu Sans", "sans-serif"
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message="findfont", category=UserWarning)
    warnings.filterwarnings("ignore", message="Glyph", category=UserWarning)
    _MPL_OK = True
except Exception:  # matplotlib 不可用或初始化异常
    plt = None
    _MPL_OK = False


def plot_curves(histories: dict[str, dict], metric: str, ylabel: str, path, title: str = "") -> Path | None:
    if not _MPL_OK:
        print(f"[警告] matplotlib 不可用，跳过曲线图：{path}")
        return None
    fig = None
    try:
        p = Path(path)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for label, hist in histories.items():
            ys = hist.get(metric)
            if ys is None or isinstance(ys, (int, float)):
                continue
            ys = [float(v) for v in ys]  # 统一为普通 float，兼容 torch/numpy 标量
            if ys:
                ax.plot(range(1, len(ys) + 1), ys,
                        marker="o", markersize=3, linewidth=1.5, label=label)
        ax.set_xlabel("epoch / step")
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        fig.tight_layout()
        ensure_dir(p.parent)
        fig.savefig(p, dpi=120, format="png")
        return p
    except Exception as e:
        print(f"[警告] 曲线图绘制失败（{metric}）：{e}，跳过 {path}")
        return None
    finally:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass


def plot_bars(values: dict[str, float], ylabel: str, path, title: str = "") -> Path | None:
    if not _MPL_OK:
        print(f"[警告] matplotlib 不可用，跳过柱状图：{path}")
        return None
    fig = None
    try:
        p = Path(path)
        labels = list(values.keys())
        nums = [float(v) for v in values.values()]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(labels, nums)
        for bar, v in zip(bars, nums):  # 顶部标数值
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4g}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        ensure_dir(p.parent)
        fig.savefig(p, dpi=120, format="png")
        return p
    except Exception as e:
        print(f"[警告] 柱状图绘制失败：{e}，跳过 {path}")
        return None
    finally:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass
