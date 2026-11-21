"""容量匹配 pilot：把「12 vs 240 参数」从隐藏混杂变成被测量的轴。

REVIEW §14 里 `--matched` 仍有一处结构性差异：量子层的 12 个旋转角 vs
经典替身 Linear(4,48) 的 240 个权重。本脚本扫描量子线路层数 L，使量子层
参数量 12×L 在 L=20 时与经典臂的 240 精确相等，并记录：

* 两臂变换层参数量、初始输出尺度（是否因层数变化而失衡）；
* 单步前向+反向耗时，外推正式训练预算下的成本；
* 可选的短训练（默认关闭，20 层状态向量模拟极慢）。

重要限定：**提高层数不只改变参数量**，还同时改变线路深度、表达力与梯度
尺度，所以「参数量相等」不等于「唯一自变量已隔离」。本扫描的目的就是把
这一混杂显式测量出来，而不是声称消除了它。

用法：
    python experiments/capacity_matched.py
    python experiments/capacity_matched.py --layers 1,5,10,20 --timing-steps 8
    python experiments/capacity_matched.py --short-train 200   # 谨慎：20 层很慢
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_addition import GROUP_IDS, build_model  # noqa: E402
from qml_lab.arithmetic import (  # noqa: E402
    RANDOM_BASELINE_ACCURACY,
    answer_logits,
    build_tensors,
    enumerate_table,
    split_table,
)

DOCS = ROOT / "docs"
OUT_JSON = DOCS / "capacity_matched_pilot.json"
OUT_FIG = DOCS / "figures" / "capacity_matched_pilot.png"


def _timing(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
            steps: int) -> float:
    """单步前向+反向平均耗时（不更新参数，只测计算成本）。"""
    model.train()
    ts = []
    for _ in range(steps):
        t0 = time.perf_counter()
        loss = F.cross_entropy(answer_logits(model, x), y)
        loss.backward()
        ts.append(time.perf_counter() - t0)
        model.zero_grad(set_to_none=True)
    return float(np.median(ts))


def _short_train(group: str, L: int, steps: int, seed: int,
                 x: torch.Tensor, y: torch.Tensor) -> dict:
    """可选短训练，只回答「该容量下能否跑通 / 是否离开基线」。"""
    model = build_model(group, seed=seed, matched=True,
                        spec_overrides={"n_layers": L})
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=0.015)
    t0 = time.perf_counter()
    model.train()
    for i in range(steps):
        idx = torch.randint(0, x.shape[0], (32,))
        opt.zero_grad()
        F.cross_entropy(answer_logits(model, x[idx]), y[idx]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = float((answer_logits(model, x).argmax(-1) == y).float().mean())
    return {"val_accuracy": acc, "steps": steps,
            "wallclock_seconds": time.perf_counter() - t0}


def _figure(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cn = next((f.name for f in font_manager.fontManager.ttflist
               if "YaHei" in f.name or "Microsoft YaHei" in f.name), None)
    with plt.rc_context({"font.sans-serif": [cn or "DejaVu Sans"],
                         "axes.unicode_minus": False}):
        fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
        Q, C = GROUP_IDS
        C_Q, C_C, C_OK, C_BAD = "#534AB7", "#185FA5", "#1D9E75", "#C0392B"
        rows = report["per_layer"]
        Ls = [r["n_layers"] for r in rows]
        qp = [r["per_arm"][Q]["transform_params"] for r in rows]
        cp = [r["per_arm"][C]["transform_params"] for r in rows]

        ax = axes[0][0]
        ax.plot(Ls, qp, "o-", color=C_Q, lw=2, label="量子臂 12×L 个旋转角")
        ax.axhline(cp[0], color=C_C, lw=2, ls="--", label="经典臂 240 个权重")
        hit = next((r for r in rows if r["transform_param_gap"] == 0), None)
        if hit:
            ax.scatter([hit["n_layers"]], [hit["per_arm"][Q]["transform_params"]],
                       s=110, facecolors="none", edgecolors=C_OK, lw=2.4, zorder=5)
            ax.annotate(f"L={hit['n_layers']} 时参数量相等",
                        (hit["n_layers"], hit["per_arm"][Q]["transform_params"]),
                        xytext=(0.35, 0.55), textcoords="axes fraction",
                        fontsize=9.6, color=C_OK,
                        arrowprops=dict(arrowstyle="->", color=C_OK, lw=1.3))
        ax.set_xlabel("量子线路层数 L（两臂 spec 同值）", fontsize=9.5)
        ax.set_ylabel("变换层可训练参数量", fontsize=9.5)
        ax.set_title("(a) 参数量在 L=20 精确相等，但层数同时改变表达力",
                     fontsize=11.2, fontweight="bold", loc="left", pad=8)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(alpha=0.22)

        ax = axes[0][1]
        tq = [r["per_arm"][Q]["seconds_per_step"] for r in rows]
        tc = [r["per_arm"][C]["seconds_per_step"] for r in rows]
        ax.plot(Ls, tq, "o-", color=C_Q, lw=2, label="量子臂")
        ax.plot(Ls, tc, "s-", color=C_C, lw=2, label="经典臂")
        ax.set_yscale("log")
        ax.set_xlabel("量子线路层数 L", fontsize=9.5)
        ax.set_ylabel("单步前向+反向耗时（秒，对数轴）", fontsize=9.5)
        ax.set_title("(b) 容量匹配的计算成本：2000 步外推",
                     fontsize=11.2, fontweight="bold", loc="left", pad=8)
        if hit:
            h = hit["per_arm"][Q]["seconds_per_step"] * 2000 / 3600
            ax.text(0.03, 0.93, f"L={hit['n_layers']}：约 {h:.1f} 小时 / seed",
                    transform=ax.transAxes, va="top", fontsize=9.6, color=C_BAD)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(alpha=0.22)

        ax = axes[1][0]
        qs = [r["per_arm"][Q]["initial_output_std"] for r in rows]
        cs = [r["per_arm"][C]["initial_output_std"] for r in rows]
        xs = np.arange(len(Ls))
        w = 0.34
        ax.bar(xs - w / 2, qs, w, color=C_Q, label="量子臂")
        ax.bar(xs + w / 2, cs, w, color=C_C, label="经典臂")
        for i, (a, b) in enumerate(zip(qs, cs)):
            ax.text(i, max(a, b) * 1.06, f"{a / max(b, 1e-9):.1f}×",
                    ha="center", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"L={L}" for L in Ls], fontsize=9.5)
        ax.set_ylabel("初始读出特征标准差", fontsize=9.5)
        ax.set_title("(c) 加层数会不会再次破坏初始尺度对齐",
                     fontsize=11.2, fontweight="bold", loc="left", pad=8)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(alpha=0.22, axis="y")

        ax = axes[1][1]
        ax.axis("off")
        lines = [
            "容量匹配扫描的结论边界", "",
            f"默认 L=1：量子 {qp[0]} 参数 vs 经典 {cp[0]} 参数（差 {cp[0] // qp[0]}×）",
            "L=20：两者同为 240 参数，参数量差 = 0", "",
            "但提高层数同时改变了线路深度、表达力与梯度尺度，",
            "所以「参数量相等」不等于「只差量子/经典这一件事」。",
            "本扫描把隐藏混杂变成被测量的轴，而非消除它。", "",
            "更靠前的限制依然存在：量子层逐位置独立，不承载跨步状态。",
            "即使容量完全匹配，测到的仍是两种 pointwise 变换，",
            "回答不了「量子对语言建模有何用」。",
        ]
        for i, t in enumerate(lines):
            ax.text(0.02, 0.96 - i * 0.068, t, fontsize=9.5, va="top",
                    color="#222222", transform=ax.transAxes)
        ax.set_title("(d) 结论：容量是轴，不是终点",
                     fontsize=11.2, fontweight="bold", loc="left", pad=8)

        fig.suptitle("容量匹配 pilot：12 vs 240 到底能不能对齐",
                     fontsize=15, fontweight="bold", x=0.012, ha="left", y=0.975)
        fig.text(0.012, 0.925,
                 "两臂 spec 逐项相同（含 n_layers 同值）；量子臂按语义使用层数，经典臂忽略。"
                 "计时为 headless 状态向量模拟，无 shots / 噪声。",
                 fontsize=9.2, color="#555555", ha="left")
        fig.text(0.012, 0.018,
                 "pilot 只测参数量、初始行为与计算成本；未做 5 seed 训练，不得据此比较两臂准确率。",
                 fontsize=9, color="#555555", ha="left")
        fig.subplots_adjust(left=0.075, right=0.975, top=0.865, bottom=0.115,
                            hspace=0.42, wspace=0.27)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1,5,10,20")
    ap.add_argument("--timing-steps", type=int, default=6)
    ap.add_argument("--short-train", type=int, default=0,
                    help="每臂跑 N 步短训练（仅 seed=0），20 层极慢，默认关闭")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    Q, C = GROUP_IDS

    tr, _ = split_table(enumerate_table("+"), "holdout_pairs", 0.2, 0)
    x, y = build_tensors(tr, "+")
    xt, yt = x[:32], y[:32]
    xp, _yp = x[:128], y[:128]

    rows = []
    for L in [int(v) for v in args.layers.split(",")]:
        per_arm = {}
        for g in GROUP_IDS:
            t0 = time.perf_counter()
            model = build_model(g, seed=0, matched=True,
                                spec_overrides={"n_layers": L})
            with torch.no_grad():
                z = model.bottleneck_features(xp)[:, -1, :]
            sec = _timing(model, xt, yt, args.timing_steps)
            entry = {
                "transform_params": sum(p.numel() for p in model.transform.parameters()),
                "total_params": sum(p.numel() for p in model.parameters()),
                "initial_output_std": float(z.std()),
                "initial_output_median_feature_std": float(z.std(0).median()),
                "seconds_per_step": sec,
                "construction_seconds": time.perf_counter() - t0,
            }
            if args.short_train:
                entry["short_train"] = _short_train(g, L, args.short_train, 0, x[:512], y[:512])
            per_arm[g] = entry
        gap = abs(per_arm[Q]["transform_params"] - per_arm[C]["transform_params"])
        rows.append({
            "n_layers": L, "per_arm": per_arm,
            "transform_param_gap": gap,
            "transform_param_ratio": (per_arm[Q]["transform_params"]
                                      / max(per_arm[C]["transform_params"], 1)),
        })
        print(f"  L={L:2d}  量子 {per_arm[Q]['transform_params']:4d} 参数 "
              f"({per_arm[Q]['seconds_per_step']:.4f}s/步) | "
              f"经典 {per_arm[C]['transform_params']:4d} 参数 "
              f"({per_arm[C]['seconds_per_step']:.4f}s/步) | 差 {gap}",
              flush=True)

    report = {
        "task": "capacity-matched pilot: transform parameter count as a measured axis",
        "per_layer": rows,
        "classical_params_fixed": rows[0]["per_arm"][GROUP_IDS[1]]["transform_params"],
        "quantum_params_per_layer": 12,
        "exact_match_layer": next((r["n_layers"] for r in rows
                                   if r["transform_param_gap"] == 0), None),
        "timing_steps": args.timing_steps,
        "short_train_steps": args.short_train,
        "random_baseline_accuracy": RANDOM_BASELINE_ACCURACY,
        "caveat": ("提高量子层数同时改变线路深度、表达力与梯度尺度；参数量相等"
                   "不等于唯一自变量已隔离。本 pilot 只把容量差显式测量出来，"
                   "未做多 seed 训练，不得据此比较两臂准确率。量子层仍是逐位置"
                   "变换，不参与序列依赖。"),
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"  archived: {OUT_JSON}", flush=True)
    if not args.no_figure:
        _figure(report, OUT_FIG)
        print(f"  figure:   {OUT_FIG}", flush=True)


if __name__ == "__main__":
    main()
