"""严格对照的**装置标定与诊断**（2026-11-16，REVIEW §14.6 的可复现证据）。

「两臂 spec 逐项相同」只保证字段一致，不保证两臂真的站在同一起跑线上。
本脚本量化三版装置的初始行为与最终学习结果，并公开记录最小调参网格。

三个装置版本（都满足 assert_matched）：
    v1  near_identity_init=True,  readout_norm=False  字段一致但行为失衡
    v2  near_identity_init=True,  readout_norm=True   LayerNorm 增益不等，两臂一起退化
    v3  near_identity_init=False, readout_norm=False  两臂均用默认初始化、不加补丁

用法：
    python experiments/matched_calibration.py                 # 诊断 + 汇总归档 + 出中文图
    python experiments/matched_calibration.py --grid         # 追加公开记录的最小调参网格
    python experiments/matched_calibration.py --grid-steps 600 --grid-seeds 2

不重跑正式训练：归档数值全部从 docs/*.json 读取；--grid 是独立的小预算 pilot，
结果只能用于「装置标定」，不得当作量子层效应的结论。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_addition import (  # noqa: E402
    GROUP_IDS,
    MATCHED_GROUP_SPECS,
    build_model,
    run_one,
)
from qml_lab.arithmetic import (  # noqa: E402
    enumerate_table,
    split_table,
    build_tensors,
    RANDOM_BASELINE_ACCURACY,
)

DOCS = ROOT / "docs"
CALIBRATION_JSON = DOCS / "matched_calibration.json"
CALIBRATION_FIG = DOCS / "figures" / "matched_calibration.png"

ARCHIVES = {
    "v1": ("matched_v1_no_norm_metrics.json", "字段一致 / 无归一化"),
    "v2": ("matched_metrics.json", "两臂共用 LayerNorm"),
    "v3": ("matched_v3_metrics.json", "默认初始化 / 无补丁，lr=0.015"),
    "v4": ("matched_v4_metrics.json", "默认初始化 / 共同可行域 lr=0.003"),
}

DEVICE_OVERRIDES = {
    "v1": {"near_identity_init": True, "readout_norm": False},
    "v2": {"near_identity_init": True, "readout_norm": True},
    "v3": {"near_identity_init": False, "readout_norm": False},
    "v2b": {"near_identity_init": False, "readout_norm": True},
}
DEVICE_LABELS = {
    "v1": "v1 小初始化补丁",
    "v2": "v2 补丁+LayerNorm",
    "v3": "v3 默认初始化",
    "v4": "v4 共同可行域",
    "v2b": "默认初始化+LayerNorm（对照）",
}


# --------------------------------------------------------------------------- #
# Stage A：初始行为诊断（不训练）
# --------------------------------------------------------------------------- #
def _feature_stats(z: torch.Tensor, y: torch.Tensor) -> dict:
    d = torch.cdist(z, z)
    n = d.shape[0]
    pair = (d.sum() - d.trace()) / (n * (n - 1))
    d2 = d.clone().fill_diagonal_(float("inf"))
    return {
        "global_std": float(z.std()),
        "median_per_feature_std": float(z.std(0).median()),
        "mean_pairwise_l2": float(pair),
        "cos_adjacent": float(F.cosine_similarity(z[:-1], z[1:], dim=-1).mean()),
        "nn1_label_accuracy": float((y[d2.argmin(1)] == y).float().mean()),
    }


def diagnose_initial_behavior(n_probe: int = 256) -> dict:
    """对每个装置版本，实测两臂初始读出特征的尺度与区分度。"""
    tr, _ = split_table(enumerate_table("+"), "holdout_pairs", 0.2, 0)
    x, y = build_tensors(tr, "+")
    x, y = x[:n_probe], y[:n_probe]

    out: dict = {"n_probe": int(x.shape[0]), "devices": {}}
    for key, overrides in DEVICE_OVERRIDES.items():
        per_arm = {}
        for g in GROUP_IDS:
            model = build_model(g, seed=0, matched=True,
                                spec_overrides=overrides).eval()
            with torch.no_grad():
                z = model.bottleneck_features(x)[:, -1, :]
                zn = (model.readout_norm(z) if model.readout_norm is not None
                      else z)
                logits = model.head(zn * model.out_scale)[:, :112]
            raw, head = _feature_stats(z, y), _feature_stats(zn, y)
            per_arm[g] = {
                "raw": raw,
                "into_head": head,
                "effective_gain": head["global_std"] / max(raw["global_std"], 1e-12),
                "initial_logits_std": float(logits.std()),
                "initial_ce": float(F.cross_entropy(logits, y)),
                "transform_params": sum(p.numel() for p in model.transform.parameters()),
            }
        q, c = (per_arm[g] for g in GROUP_IDS)
        out["devices"][key] = {
            "label": DEVICE_LABELS[key],
            "overrides": overrides,
            "per_arm": per_arm,
            "initial_std_ratio": q["raw"]["global_std"] / max(c["raw"]["global_std"], 1e-12),
            "gain_ratio": q["effective_gain"] / max(c["effective_gain"], 1e-12),
        }
        print(f"  {DEVICE_LABELS[key]:28s} std 量子/经典 = "
              f"{q['raw']['global_std']:.4f}/{c['raw']['global_std']:.4f} "
              f"(比 {out['devices'][key]['initial_std_ratio']:.1f}×)；"
              f"LayerNorm 有效增益 {q['effective_gain']:.2f}×/{c['effective_gain']:.2f}×",
              flush=True)
    return out


# --------------------------------------------------------------------------- #
# Stage B：汇总正式归档的装置有效性
# --------------------------------------------------------------------------- #
def summarize_archives() -> dict:
    out: dict = {}
    for key, (fname, label) in ARCHIVES.items():
        p = DOCS / fname
        if not p.exists():
            out[key] = {"label": label, "archive": fname, "available": False}
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        dv = d.get("matched_control", {}).get("device_validity", {})
        agg = d.get("aggregate", {})
        per_seed = {
            g: [r["final"]["val_accuracy"] for r in d.get("runs", [])
                if r["group"] == g]
            for g in GROUP_IDS
        }
        out[key] = {
            "label": label, "archive": fname, "available": True,
            "steps": d.get("config", {}).get("steps"),
            "n_seeds": len(d.get("config", {}).get("seeds", [])),
            "specs": d.get("matched_control", {}).get("specs", {}),
            "device_valid": dv.get("valid"),
            "device_note": dv.get("note"),
            "per_arm": dv.get("per_arm", {}),
            "val_mean": {g: agg.get(g, {}).get("val_accuracy_mean")
                         for g in GROUP_IDS},
            "val_std": {g: agg.get(g, {}).get("val_accuracy_std")
                        for g in GROUP_IDS},
            "per_seed_val": per_seed,
            "verdict": d.get("interpretation", {}).get("verdict"),
        }
    return out


# --------------------------------------------------------------------------- #
# Stage C：公开记录的最小调参网格（可选）
# --------------------------------------------------------------------------- #
def _grid_cfg(scale_init: float, lr: float, steps: int, seeds: int,
              eval_every: int, curve_n: int, workers: int) -> dict:
    return {
        "matched": True,
        "group_specs": {g: dict(MATCHED_GROUP_SPECS[g]) for g in GROUP_IDS},
        "spec_overrides": {"scale_init": scale_init},
        "embedding_mode": "trainable_circular",
        "circular_embedding": True,
        "op": "+", "split_mode": "holdout_pairs", "val_fraction": 0.2,
        "split_seed": 0, "steps": steps, "batch_size": 32, "lr": lr,
        "eval_every": eval_every, "eval_batch": 512, "curve_n": curve_n,
        "workers": workers, "seeds": list(range(seeds)),
        "vocab_size": 117, "modulus": 112, "groups": list(GROUP_IDS),
    }


def run_grid(scale_inits: list[float], lrs: list[float], steps: int,
             seeds: int, eval_every: int, curve_n: int, workers: int,
             arms: list[str] | None = None) -> dict:
    """两臂共用同一组超参的小预算扫描；只用于装置标定，不作效应结论。

    ``arms`` 可只跑经典臂：经典臂 600 步约 3 s/seed、量子臂约 90 s/seed，
    先用便宜的一臂定位可行超参区域，再验证量子臂，避免无谓的内存与算力开销。
    """
    arms = list(arms) if arms else list(GROUP_IDS)
    combos = [(s, lr) for s in scale_inits for lr in lrs]
    tasks, meta = [], []
    for scale_init, lr in combos:
        cfg = _grid_cfg(scale_init, lr, steps, seeds, eval_every, curve_n, workers)
        for g in arms:
            for s in range(seeds):
                tasks.append((g, s, cfg))
                meta.append((scale_init, lr, g, s))
    print(f"[grid] {len(combos)} 组超参 × {len(arms)} 臂 × {seeds} seed "
          f"= {len(tasks)} 次 × {steps} 步；workers={workers}；臂={','.join(arms)}",
          flush=True)
    t0 = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            runs = list(ex.map(run_one, tasks))
    else:
        runs = [run_one(t) for t in tasks]

    cells = []
    for scale_init, lr in combos:
        per_arm = {}
        for g in arms:
            vals = [r["final"]["val_accuracy"] for (sc, lr_v, gg, s), r
                    in zip(meta, runs) if sc == scale_init and lr_v == lr and gg == g]
            per_arm[g] = {
                "val_mean": float(np.mean(vals)), "val_std": float(np.std(vals, ddof=1)),
                "per_seed": vals,
                "seeds_above_2x_baseline": sum(
                    1 for v in vals if v > 2 * RANDOM_BASELINE_ACCURACY),
                "n_seeds": len(vals),
            }
        both_learned = (set(arms) == set(GROUP_IDS)
                        and all(v["seeds_above_2x_baseline"] == v["n_seeds"]
                                for v in per_arm.values()))
        cells.append({"scale_init": scale_init, "lr": lr, "arms": arms,
                      "per_arm": per_arm, "both_arms_learned": both_learned})
        print(f"  scale_init={scale_init:g} lr={lr:g} | "
              + " / ".join(f"{g.split('_')[-1]} {v['val_mean']:.4f}"
                           for g, v in per_arm.items())
              + f" | 两臂都学动={both_learned}", flush=True)
    ok = [c for c in cells if c["both_arms_learned"]]
    return {
        "steps": steps, "seeds": seeds, "eval_every": eval_every,
        "curve_n": curve_n, "workers": workers, "arms": arms,
        "grid": list(scale_inits), "lrs": list(lrs), "cells": cells,
        "n_configs_with_both_arms_learned": len(ok),
        "selected": (max(ok, key=lambda c: min(v["val_mean"]
                                               for v in c["per_arm"].values()))
                     if ok else None),
        "wallclock_seconds": time.perf_counter() - t0,
        "caveat": ("pilot 网格仅用于装置标定（两臂是否都离开随机基线），"
                   "seed 数与步数均少于正式实验，不得据此比较两臂优劣。"),
    }


# --------------------------------------------------------------------------- #
# 中文结论式图
# --------------------------------------------------------------------------- #
def _figure(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cn = next((f.name for f in font_manager.fontManager.ttflist
               if "YaHei" in f.name or "Microsoft YaHei" in f.name), None)
    with plt.rc_context({"font.sans-serif": [cn or "DejaVu Sans"],
                         "axes.unicode_minus": False}):
        fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4))
        C_Q, C_C, C_GREY, C_OK = "#534AB7", "#185FA5", "#9A9A93", "#1D9E75"
        C_BAD = "#C0392B"
        Q, C = GROUP_IDS

        # (a) 三版装置的初始输出尺度
        ax = axes[0][0]
        keys = [k for k in ("v1", "v2", "v3") if k in report["diagnosis"]["devices"]]
        xs = np.arange(len(keys))
        w = 0.34
        qv = [report["diagnosis"]["devices"][k]["per_arm"][Q]["raw"]["global_std"]
              for k in keys]
        cv = [report["diagnosis"]["devices"][k]["per_arm"][C]["raw"]["global_std"]
              for k in keys]
        ax.bar(xs - w / 2, qv, w, color=C_Q, label="量子臂")
        ax.bar(xs + w / 2, cv, w, color=C_C, label="经典臂")
        for i, k in enumerate(keys):
            r = report["diagnosis"]["devices"][k]["initial_std_ratio"]
            ax.text(i, max(qv[i], cv[i]) * 1.12, f"{r:.1f}×", ha="center",
                    va="bottom", fontsize=10, fontweight="bold",
                    color=C_OK if 0.5 <= r <= 2 else C_BAD)
        ax.set_yscale("log")
        ax.set_ylim(1.5e-2, 7e-1)
        ax.set_xticks(xs)
        ax.set_xticklabels([DEVICE_LABELS[k].split(" ")[0] for k in keys], fontsize=10)
        ax.set_ylabel("初始读出特征标准差（对数轴）", fontsize=9.5)
        ax.set_title("(a) 只有不加补丁的 v3，两臂初始尺度才接近",
                     fontsize=11.5, fontweight="bold", loc="left", pad=8)
        ax.grid(alpha=0.22, axis="y")

        # (b) LayerNorm 的有效增益
        ax = axes[0][1]
        gq = [report["diagnosis"]["devices"][k]["per_arm"][Q]["effective_gain"]
              for k in keys]
        gc = [report["diagnosis"]["devices"][k]["per_arm"][C]["effective_gain"]
              for k in keys]
        ax.bar(xs - w / 2, gq, w, color=C_Q, label="量子臂")
        ax.bar(xs + w / 2, gc, w, color=C_C, label="经典臂")
        ax.axhline(1.0, color="#444444", ls=":", lw=1.2)
        for i, (a, b) in enumerate(zip(gq, gc)):
            ax.text(i - w / 2, a + 0.8, f"{a:.1f}×", ha="center", fontsize=9)
            ax.text(i + w / 2, b + 0.8, f"{b:.1f}×", ha="center", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([DEVICE_LABELS[k].split(" ")[0] for k in keys], fontsize=10)
        ax.set_ylabel("进分类头前 / 原始特征的标准差倍数", fontsize=9.5)
        ax.set_title("(b) LayerNorm 对两臂增益不等，并非对称操作",
                     fontsize=11.5, fontweight="bold", loc="left", pad=8)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(alpha=0.22, axis="y")

        # (c) 三版装置 5 seed 验证准确率
        ax = axes[1][0]
        arch = report["archives"]
        have = [k for k in ("v1", "v2", "v3", "v4")
                if arch.get(k, {}).get("available")]
        for j, k in enumerate(have):
            for off, g, col in ((-0.16, Q, C_Q), (0.16, C, C_C)):
                vals = arch[k]["per_seed_val"].get(g, [])
                if not vals:
                    continue
                ax.scatter([j + off] * len(vals), vals, s=26, color=col,
                           alpha=0.8, zorder=3, edgecolors="white", linewidths=0.6)
                mu = float(np.mean(vals))
                ax.hlines(mu, j + off - 0.07, j + off + 0.07, color=col,
                          lw=2.4, zorder=4)
        ax.axhline(RANDOM_BASELINE_ACCURACY, color=C_GREY, ls=":", lw=1.2)
        ax.axhline(2 * RANDOM_BASELINE_ACCURACY, color=C_BAD, ls="--", lw=1.2)
        ax.text(0.97, 0.95, "红色虚线：装置门槛 2×随机基线",
                transform=ax.transAxes, ha="right", va="top",
                color=C_BAD, fontsize=8.6)
        ax.set_xticks(range(len(have)))
        ax.set_xticklabels([DEVICE_LABELS[k].split(" ")[0] for k in have], fontsize=10)
        ax.set_ylabel("全量验证集答案准确率", fontsize=9.5)
        ax.set_ylim(-0.01, max(0.30, max(
            [v for k in have for g in GROUP_IDS
             for v in arch[k]["per_seed_val"].get(g, [])] + [0.25]) * 1.25))
        ax.set_title("(c) 前两版装置失效；后两版看两臂能否同时越过门槛",
                     fontsize=11.5, fontweight="bold", loc="left", pad=8)
        ax.grid(alpha=0.22, axis="y")

        # (d) 结论卡
        ax = axes[1][1]
        ax.axis("off")
        lines = ["装置有效性（两臂各自 ≥80% seed 超过 2×随机基线）", ""]
        for k in ("v1", "v2", "v3", "v4"):
            a = arch.get(k, {})
            if not a.get("available"):
                lines.append(f"{k}  归档未生成")
                continue
            qm = a["val_mean"].get(Q)
            cm = a["val_mean"].get(C)
            flag = "有效" if a["device_valid"] else "无效"
            if qm is not None and cm is not None:
                lines.append(f"{k}  {flag}   量子 {qm:.4f} / 经典 {cm:.4f}")
            else:
                lines.append(f"{k}  {flag}")
        lines += ["",
                  "v1：字段一致但初始输出 std 差 16 倍，经典臂 3/5 学不动",
                  "v2：LayerNorm 把两臂一起洗回随机基线 0.0062",
                  "v3：两臂都退回默认初始化，无单边补丁、无新增层",
                  "",
                  "标定网格（两臂共用同一组超参）只回答「装置能不能用」，",
                  "不回答「哪种变换更好」。容量差 12 vs 240 仍未消除。"]
        for i, t in enumerate(lines):
            ax.text(0.02, 0.96 - i * 0.062, t, fontsize=9.6, va="top",
                    color=(C_BAD if t.startswith("v1") or t.startswith("v2")
                           else C_OK if t.startswith("v3") else "#222222"),
                    transform=ax.transAxes, wrap=True)
        ax.set_title("(d) 结论：先修装置，再谈对照",
                     fontsize=11.5, fontweight="bold", loc="left", pad=8)

        fig.suptitle("严格对照装置标定：字段一致 ≠ 行为一致",
                     fontsize=15, fontweight="bold", x=0.012, ha="left", y=0.975)
        fig.text(0.012, 0.925,
                 "初始诊断 256 个训练样本；柱状/散点为归档全量评估值（2509 个未见 (a,b) 组合）；"
                 "随机基线 = 1/112",
                 fontsize=9.2, color="#555555", ha="left")
        fig.text(0.012, 0.018,
                 "注意：本图只做装置有效性诊断。即使装置有效，逐位置的量子变换仍不参与序列依赖，"
                 "不能据此讨论量子层对语言建模的作用。",
                 fontsize=9, color="#555555", ha="left")
        fig.subplots_adjust(left=0.075, right=0.975, top=0.865, bottom=0.115,
                            hspace=0.42, wspace=0.27)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="追加公开记录的最小调参网格")
    ap.add_argument("--grid-steps", type=int, default=600)
    ap.add_argument("--grid-seeds", type=int, default=2)
    ap.add_argument("--grid-eval-every", type=int, default=200)
    ap.add_argument("--grid-curve-n", type=int, default=256)
    ap.add_argument("--grid-workers", type=int, default=2,
                    help="标定网格的并行 worker 数；本机内存常紧，默认 2，"
                         "不要与其它训练任务同时开 6 workers")
    ap.add_argument("--scale-inits", default="1,2,5")
    ap.add_argument("--lrs", default="0.003,0.015")
    ap.add_argument("--arms", default="both",
                    choices=("both", "classical", "quantum"),
                    help="先只跑便宜的经典臂定位可行超参，再验证两臂")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    prev = (json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
            if CALIBRATION_JSON.exists() else {})
    report = {
        "task": "matched-control device calibration / diagnosis",
        "diagnosis": diagnose_initial_behavior(),
        "archives": summarize_archives(),
        "random_baseline_accuracy": RANDOM_BASELINE_ACCURACY,
        "device_validity_rule": "两臂各自 ≥80% 的 seed 的 val 需超过 2×随机基线，且二项检验显著",
        "caveat": ("本归档用于诊断「对照装置是否可用」，不是量子层效应结论。"
                   "即使两臂都学动，量子层仍是逐位置特征变换，不承载跨步状态。"),
    }
    grids = list(prev.get("grids", []))
    if args.grid:
        arms = {"both": list(GROUP_IDS),
                "classical": [GROUP_IDS[1]],
                "quantum": [GROUP_IDS[0]]}[args.arms]
        grid = run_grid(
            [float(s) for s in args.scale_inits.split(",")],
            [float(s) for s in args.lrs.split(",")],
            args.grid_steps, args.grid_seeds, args.grid_eval_every,
            args.grid_curve_n, args.grid_workers, arms=arms)
        grids.append({"phase": f"{args.arms}-only" if args.arms != "both" else "both",
                      **grid})
        report["grids"], report["grid"] = grids, grid
    elif "grid" in prev:
        report["grids"], report["grid"] = grids, prev["grid"]
        print("[grid] 沿用已有标定网格（本次未重跑）", flush=True)

    DOCS.mkdir(parents=True, exist_ok=True)
    CALIBRATION_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  archived: {CALIBRATION_JSON}", flush=True)
    if not args.no_figure:
        _figure(report, CALIBRATION_FIG)
        print(f"  figure:   {CALIBRATION_FIG}", flush=True)


if __name__ == "__main__":
    main()
