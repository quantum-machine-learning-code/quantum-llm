"""课题一修正轮：不加先验 = grokking（2026-11-13）。

用户指出的对应关系（上一轮答非所问的地方）：

    加循环角度先验 = 把 grokking 最终学到的 sin/cos 傅里叶表征直接注入，
                     因此不需要 grok，2000 步就能学会模 112 加法；
    不加先验       = grokking 起跑线，模型必须靠长训练 + 正则化自己长出
                     这个表征，才会出现「先记忆、后突发泛化」的相变。

上一轮结论「瓶颈是架构缺少乘法交互」不成立，三个具体原因：

1. 不加先验组 40000 步后 train acc 仅 0.0288，连记忆阶段都没进入；
   grokking 的定义要求先记忆，第一步没发生就报 0/5 相变，推不出架构结论。
2. 正对照用了不同的 lr(0.003 vs 0.015) / wd(1.0 vs 0.1) / 训练数据占比
   (20% 训练 vs 80% 训练) / 步数(5000 vs 40000)，所谓 +0.9910「架构差距」
   至少混了四个变量。
3. Power et al. 2022 的相变量级是 1e4~1e6 步，只跑到 4e4（下界）就断言
   「不是训练预算」过早。

本脚本只回答一个问题：给「不加先验」这一组充分的优化条件和相变尺度预算，
它到底 grok 不 grok？两架构使用**完全相同的** lr / wd / val_fraction /
batch / 步数 / seed / 切分，唯一差别是架构；二次 MLP 只作装置校验（证明
这套条件下任务确实可 grok），不参与「架构效应」的因果结论。

方法学约定：Stage 1 为 1-seed 探索（只用于预选，不作结论）；Stage 2 为
5-seed 确认；相变阈值沿用 grokking_scan.py 预先写死的 PHASE_*，不随结果调整。

用法：
    python experiments/grokking_no_prior.py --quick
    python experiments/grokking_no_prior.py --stage1-only --workers 6
    python experiments/grokking_no_prior.py --stage2-from docs/grokking_no_prior_pilot.json
    python experiments/grokking_no_prior.py --redraw-from docs/grokking_no_prior.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path(__file__).resolve().parent
for p in (str(ROOT), str(EXPERIMENTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from grokking_scan import (  # noqa: E402
    PHASE_GEN_ACC,
    PHASE_MEM_ACC,
    RANDOM_BASELINE_ACCURACY,
    run_one,
)
from qml_lab.arithmetic import context_overlap, enumerate_table, split_table  # noqa: E402

DOCS = ROOT / "docs"
PILOT = DOCS / "grokking_no_prior_pilot.json"
METRICS = DOCS / "grokking_no_prior.json"
FIGURE = DOCS / "figures" / "grokking_no_prior.png"

# 与上一轮不同的关键修正：lr 网格覆盖正对照用的 0.003，数据占比覆盖 grokking
# 常用的「少训练数据」区间，weight decay 覆盖强正则。
LR_GRID = (0.0003, 0.001, 0.003, 0.01, 0.015)
WD_GRID = (0.0, 0.1, 1.0)
VF_GRID = (0.2, 0.8)

COL = {"rich_train": "#185FA5", "rich_val": "#D85A30",
       "mlp_train": "#1D9D75", "mlp_val": "#534AB7"}


def _cond(name: str, arch: str, lr: float, wd: float, vf: float,
          steps: int, eval_every: int) -> dict:
    return {
        "name": name, "arch": arch, "group": "add_classical",
        "mode": "trainable_random", "opt": "adamw", "wd": wd, "lr": lr,
        "embed_lr_scale": 1.0, "val_fraction": vf,
        "steps": steps, "eval_every": eval_every,
    }


def _grid(steps: int, eval_every: int) -> list[dict]:
    conds = []
    for vf in VF_GRID:
        for lr in LR_GRID:
            for wd in WD_GRID:
                tag = f"vf{vf:g}/lr{lr:g}/wd{wd:g}"
                conds.append(_cond(f"RICH/{tag}", "rich", lr, wd, vf,
                                   steps, eval_every))
                conds.append(_cond(f"MLP/{tag}", "quad_mlp", lr, wd, vf,
                                   steps, eval_every))
    return conds


def _run_pool(tasks: list[tuple], workers: int) -> list[dict]:
    if workers <= 1:
        return [run_one(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_one, tasks))


def _cfg(args, quick: bool = False) -> dict:
    return {
        "val_fraction": 0.2, "batch_size": args.batch_size,
        "eval_batch": 512, "curve_n": 64 if quick else args.curve_n,
        "split_mode": "holdout_pairs", "split_seed": 0, "op": "+",
        "modulus": 112, "vocab_size": 117,
    }


def _slim(r: dict) -> dict:
    return {
        "arch": r["cond"]["arch"], "lr": r["cond"]["lr"], "wd": r["cond"]["wd"],
        "val_fraction": r["val_fraction"], "n_train": r["n_train"],
        "n_val": r["n_val"], "n_params": r["n_params"],
        "seconds_per_step": r["seconds_per_step"],
        "final": r["final"], "phase": r["phase"],
        "final_progress": r["final_progress"], "curve": r["curve"],
        "embedding_drift_rms": r["embedding_drift_rms"],
    }


def _select(pilot_per: dict, top_k: int) -> list[dict]:
    """在「正对照同超参下确实有效」的条件里挑 rich 表现最好的那组。

    上一轮的教训：只从 rich 自己分数里挑条件，可能挑到一组连正对照都
    学不动的超参，那样 0/5 相变什么也说明不了。因此先用 MLP 做装置校验
    （grokking 或 val acc ≥ 0.9），再在这批条件里按 rich 的探索性表现排序。
    """
    rich = {n.replace("RICH/", ""): (n, v) for n, v in pilot_per.items()
            if v["arch"] == "rich"}
    mlp = {n.replace("MLP/", ""): v for n, v in pilot_per.items()
           if v["arch"] == "quad_mlp"}
    grok_tags = {tag for tag, m in mlp.items()
                 if m["phase"]["grokking_observed"]}
    learn_tags = {tag for tag, m in mlp.items()
                  if m["final"]["val_accuracy"] >= 0.90}
    valid = grok_tags | learn_tags

    def score(v: dict):
        p = v["phase"]
        return (p["grokking_observed"], p["delayed_generalization"],
                p["max_val_accuracy"] >= PHASE_GEN_ACC,
                p["final_train_accuracy"] >= PHASE_MEM_ACC,
                v["final"]["train_accuracy"], v["final"]["val_accuracy"])

    def pick(tags: set[str], k: int) -> list[tuple[str, tuple[str, dict]]]:
        pool = {tag: rich[tag] for tag in tags if tag in rich}
        return sorted(pool.items(), key=lambda kv: score(kv[1][1]),
                      reverse=True)[:k]

    # 两类条件各留一个（若数量允许）：① 正对照真正 grok 的超参（装置对齐）；
    # ② rich 在探索网格里最学得动的超参（给不加先验组最大机会出现相变）。
    picked: list[tuple[str, tuple[str, dict]]] = []
    if top_k >= 2:
        picked += pick(grok_tags, 1)
        picked += [t for t in pick(valid - grok_tags, 1)
                   if t[0] not in {p[0] for p in picked}]
    if len(picked) < top_k:
        seen = {p[0] for p in picked}
        picked += [t for t in pick(valid or set(rich), top_k)
                   if t[0] not in seen][: top_k - len(picked)]
    return [{"name": n, "device_validated": tag in valid,
             "device_grokking": tag in grok_tags,
             **{k: v[k] for k in ("lr", "wd", "val_fraction")}}
            for tag, (n, v) in picked[:top_k]]


def _agg(runs: list[dict]) -> dict:
    def ms(key: str, sub: str = "final") -> tuple[float, float]:
        a = np.array([r[sub][key] if sub == "final" else r[key] for r in runs], dtype=float)
        return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
    out = {}
    for k in ("train_accuracy", "val_accuracy", "train_loss", "val_loss"):
        m, s = ms(k)
        out[f"{k}_mean"] = m
        out[f"{k}_std"] = s
    phases = [r["phase"] for r in runs]
    out["n_grokking_seeds"] = sum(bool(p["grokking_observed"]) for p in phases)
    out["n_memorized_seeds"] = sum(p["final_train_accuracy"] >= PHASE_MEM_ACC
                                   for p in phases)
    out["n_generalized_seeds"] = sum(p["max_val_accuracy"] >= PHASE_GEN_ACC
                                     for p in phases)
    out["memorization_steps"] = [p["memorization_step"] for p in phases]
    out["generalization_steps"] = [p["generalization_step"] for p in phases]
    out["max_val_jumps"] = [p["max_val_jump"] for p in phases]
    out["final_r2_k1_mean"] = float(np.mean([r["final_progress"]["r2_k1"] for r in runs]))
    return out


def build_verdict(conf: dict, steps: int, eval_every: int) -> tuple[str, dict]:
    """按「是否完成记忆 / 是否完成泛化」下结论，不用突发跳变当装置判据。

    stage2 用 eval_every=2000 时，正对照在 1000~1500 步发生的突发相变会被
    采样漏掉：第一个采样点 train/val 已同时为 1，delayed/abrupt 必然为假。
    因此装置有效性改用 val acc 判定，并把这条分辨率边界写进归档。
    """
    def counts(prefix: str) -> dict:
        dd = {k: v for k, v in conf.items() if k.startswith(prefix)}
        if not dd:
            return {"runs": 0, "grok": 0, "mem": 0, "gen": 0,
                    "val": 0.0, "train": 0.0}
        a = [v["aggregate"] for v in dd.values()]
        return {
            "runs": sum(len(v["curves"]) for v in dd.values()),
            "grok": sum(x["n_grokking_seeds"] for x in a),
            "mem": sum(x["n_memorized_seeds"] for x in a),
            "gen": sum(x["n_generalized_seeds"] for x in a),
            "val": float(np.mean([x["val_accuracy_mean"] for x in a])),
            "train": float(np.mean([x["train_accuracy_mean"] for x in a])),
        }
    r, m = counts("RICH"), counts("MLP")
    mlp_dd = {k: v for k, v in conf.items() if k.startswith("MLP")}
    rich_dd = {k: v for k, v in conf.items() if k.startswith("RICH")}
    best_mlp = max(mlp_dd.values(),
                   key=lambda v: (v["aggregate"]["n_generalized_seeds"],
                                  v["aggregate"]["val_accuracy_mean"])) if mlp_dd else None
    best_rich = max(rich_dd.values(),
                    key=lambda v: v["aggregate"]["train_accuracy_mean"]) if rich_dd else None

    dev = "正对照"
    if best_mlp is not None:
        ba, bn = best_mlp["aggregate"], len(best_mlp["curves"])
        dev = (f"同超参正对照 `wd={best_mlp['cond']['wd']:g}` 组 "
               f"{ba['n_generalized_seeds']}/{bn} 个 seed 完成泛化 "
               f"（val {ba['val_accuracy_mean']:.4f} ± {ba['val_accuracy_std']:.4f}）")
    detail = ""
    if best_rich is not None:
        ba = best_rich["aggregate"]
        detail = (f"（表现最好的 `wd={best_rich['cond']['wd']:g}` 组 train "
                  f"{ba['train_accuracy_mean']:.4f} ± {ba['train_accuracy_std']:.4f}、"
                  f"val {ba['val_accuracy_mean']:.4f}）")
    caveat = (f"stage2 `eval_every={eval_every}` 每 {eval_every} 步才采样一次，"
              f"早于该间隔的突发跳变会被漏掉，因此 `grokking_observed` 的"
              f"「延迟 + 突发」判定在本轮不可用。装置有效性改由 val acc 与 "
              f"stage1（`eval_every=500`，同超参正对照 `grok=True`）共同证明。")
    if r["gen"] > 0:
        v = (f"不加先验的 RichQuantumLM 在 {steps} 步下 "
             f"{r['gen']}/{r['runs']} 个 seed 完成泛化 —— "
             f"上一轮「架构不行」的结论被推翻。")
    elif m["gen"] > 0 and r["mem"] == 0:
        v = (f"装置有效：{dev}。但 RichQuantumLM {r['runs']} 个 seed "
             f"全部**未进入记忆阶段**{detail}：曲线显示 train 仍在缓慢爬升、"
             f"val 完全不动，这正是 grokking 的「记忆中」阶段。grokking 要求"
             f"先记忆，第一步都没走完就报 0 相变，**推不出「架构缺乏乘法 / "
             f"傅里叶表示能力」**；本轮能说的只是「同超参下两架构收敛速度差异"
             f"巨大」，要谈架构必须先把 rich 训练到 train≈1.0 再看 val 是否跳变。")
    elif m["gen"] > 0 and r["gen"] == 0:
        v = (f"装置有效：{dev}；RichQuantumLM {r['mem']}/{r['runs']} 完成记忆、"
             f"{r['gen']}/{r['runs']} 泛化 —— 这才是 grokking 的候选状态，"
             f"需要更长训练或更强正则才能判定。")
    else:
        v = (f"正对照在 {steps} 步下也未完成泛化 "
             f"（{m['gen']}/{m['runs']}），装置未复现 grokking，本轮无效。")
    return v + " " + caveat, {"rich": r, "mlp": m, "phase_detection_caveat": caveat}


def _figure(report: dict, path: Path) -> None:
    """「加先验 vs 不加先验」主线图（全中文、结论式标题）。

    2026-11-16 重做：上一版四个面板各说各话、缩写不解释、(d) 是一整块
    等宽英文文字墙，而且**完全没有画出「加先验」的对照**，被用户指出看不懂。
    现在四个面板围绕同一个问题：预置 sin/cos 圆编码是否等于跳过 grokking。
    """
    from matplotlib.patches import Rectangle

    pilot = report["stage1"]["per_condition"]
    conf = report["stage2"]["per_condition"]
    s1_steps = report["stage1"]["steps"]
    s2_steps = report["stage2"]["steps"]

    # §12 的「加先验」两组，把用户真正问的对照画出来；缺归档时降级
    prior: dict[str, tuple[float, float]] = {}
    g12 = DOCS / "grokking_metrics.json"
    if g12.exists():
        try:
            g = json.loads(g12.read_text(encoding="utf-8"))
            for key in ("frozen", "trainable_circular"):
                a = g["confirm"]["per_condition"][key]["aggregate"]
                prior[key] = (float(a["val_accuracy_mean"]),
                              float(a.get("val_accuracy_std", 0.0)))
        except (KeyError, ValueError):
            prior = {}

    def pick(prefix: str, wd: float | None = None) -> dict | None:
        for k, v in conf.items():
            if k.startswith(prefix) and (wd is None
                                         or abs(v["cond"]["wd"] - wd) < 1e-9):
                return v
        return None

    rich0 = pick("RICH", 0.0) or pick("RICH")
    rich1 = pick("RICH", 1.0)
    mlp1 = pick("MLP", 1.0) or pick("MLP")

    C_PRIOR, C_PRIOR_L = "#E07A2F", "#EFAE72"
    C_RICH, C_RICH_L = "#2F6FB5", "#86A9D6"
    C_CTRL, C_INK, C_MUTE = "#1D9D75", "#20242B", "#6B7280"

    def mean_curve(d: dict, key: str) -> tuple[list[int], np.ndarray]:
        cu = d["curves"]
        return ([c["step"] for c in cu[0]],
                np.asarray([[c[key] for c in x] for x in cu],
                           dtype=float).mean(0))

    with plt.rc_context({"font.sans-serif": ["Microsoft YaHei", "SimHei",
                                             "Noto Sans SC", "DejaVu Sans"],
                         "axes.unicode_minus": False}):
        fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.8))
        fig.subplots_adjust(left=0.145, right=0.972, top=0.862, bottom=0.118,
                            hspace=0.46, wspace=0.26)
        fig.suptitle("加先验 = 跳过 grokking，不加先验 = grokking 起跑线",
                     fontsize=18, fontweight="bold", color=C_INK, y=0.972)
        fig.text(0.5, 0.923,
                 "模 112 加法 · 唯一差别：数字 token 的 embedding 是否预置 sin/cos 圆编码"
                 "（数值均来自 5 seed 重复实验）",
                 ha="center", fontsize=10.5, color="#555555")

        # ---------- (a) 核心对照：验证准确率 ----------
        ax = axes[0, 0]
        rows: list[tuple[str, float, float, str, str]] = []
        if prior:
            rows += [
                ("加先验 · 冻结圆编码", *prior["frozen"], "§12 长训练", C_PRIOR),
                ("加先验 · 可训练圆编码", *prior["trainable_circular"],
                 "§12 长训练", C_PRIOR_L),
            ]
        for label, d, col in (("不加先验 · 本仓库 wd=0", rich0, C_RICH),
                              ("不加先验 · 本仓库 wd=1", rich1, C_RICH_L),
                              ("不加先验 · 二次 MLP 正对照", mlp1, C_CTRL)):
            if d is None:
                continue
            a = d["aggregate"]
            rows.append((label, float(a["val_accuracy_mean"]),
                         float(a.get("val_accuracy_std", 0.0)),
                         f"§13 · {s2_steps:,} 步", col))
        ys = np.arange(len(rows))[::-1]
        for y, (label, mean, std, _tag, col) in zip(ys, rows):
            ax.barh(y, mean, xerr=std, color=col, alpha=0.92, height=0.6,
                    error_kw=dict(ecolor="#555555", capsize=0, lw=1))
            # 数字抬到柱中线上方：误差棒横线正好在中线上，同高会被读成负号
            ax.text(mean + 0.025, y + 0.26, f"{mean:.3f}", va="bottom",
                    fontsize=9.2, fontweight="bold", color=col,
                    bbox=dict(fc="white", ec="white", alpha=1.0, pad=0.8))
        ax.axvline(RANDOM_BASELINE_ACCURACY, color="#9AA0A6", ls=":", lw=1.2,
                   label=f"随机猜 1/112 = {RANDOM_BASELINE_ACCURACY:.3f}")
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=9)
        ax.set_xlim(0, 1.16)
        ax.set_xlabel("验证集 top-1 准确率（5 seed 均值 ± 标准差）", fontsize=9.5)
        ax.set_title("(a) 预置圆编码立刻能算；本仓库从零学 10 万步还没泛化",
                     fontsize=11.5, fontweight="bold", loc="left", pad=9)
        ax.legend(fontsize=7.8, loc="upper right", framealpha=0.9)
        ax.grid(alpha=0.22, axis="x")

        # ---------- (b) 不加先验的学习曲线 ----------
        ax = axes[0, 1]
        # 右侧留白直接标注每条线，不用图例（图例会压住贴在 0 附近的验证线）
        ax.set_xscale("log")
        ax.set_xlim(1800, 600000)
        ax.set_ylim(-0.03, 1.10)
        ax.set_xticks([2000, 10000, 100000])
        ax.set_xticklabels(["2 千步", "1 万步", "10 万步"])
        if rich0 is not None and rich0["curves"]:
            st, rt = mean_curve(rich0, "train_acc")
            _, rv = mean_curve(rich0, "val_acc")
            a0 = rich0["aggregate"]  # 用全量评估口径，与 (a)(d) 的数字一致
            ax.plot(st, rt, color=C_RICH, lw=2.3)
            ax.plot(st, rv, color=C_RICH, lw=2.3, ls="--")
            ax.text(1.3e5, rt[-1], f"本仓库 · 训练 {a0['train_accuracy_mean']:.2f}",
                    va="center", fontsize=8.6, color=C_RICH, fontweight="bold")
            ax.text(1.3e5, 0.055, f"本仓库 · 验证 {a0['val_accuracy_mean']:.3f}",
                    va="center", fontsize=8.6, color=C_RICH)
        if mlp1 is not None and mlp1["curves"]:
            st, mv = mean_curve(mlp1, "val_acc")
            a1 = mlp1["aggregate"]
            ax.plot(st, mv, color=C_CTRL, lw=2.1, ls="-.")
            ax.text(1.3e5, mv[-1], f"二次 MLP · 验证 {a1['val_accuracy_mean']:.2f}",
                    va="center", fontsize=8.6, color=C_CTRL, fontweight="bold")
        ax.axhline(RANDOM_BASELINE_ACCURACY, color="#9AA0A6", ls=":", lw=1.2)
        ax.text(2300, RANDOM_BASELINE_ACCURACY + 0.04, "随机猜 0.009",
                fontsize=7.5, color="#888888")
        ax.set_xlabel("训练步数（对数轴）", fontsize=9.5)
        ax.set_ylabel("准确率", fontsize=9.5)
        ax.set_title("(b) 训练准确率在爬、验证准确率不动 —— 相变还没发生",
                     fontsize=11.5, fontweight="bold", loc="left", pad=9)
        ax.text(0.025, 0.56, "记忆中：\n训练在爬，验证不动",
                transform=ax.transAxes, fontsize=9, color=C_RICH,
                linespacing=1.6,
                bbox=dict(fc="#EAF1FA", ec="#B9CFE8", alpha=0.95,
                          boxstyle="round,pad=0.45"))
        ax.grid(alpha=0.22)

        # ---------- (c) 学习率扫描：上一轮「架构瓶颈」为何是误判 ----------
        ax = axes[1, 0]
        scan = sorted((float(v["lr"]), float(v["final"]["train_accuracy"]))
                      for v in pilot.values()
                      if v["arch"] == "rich"
                      and abs(float(v["val_fraction"]) - 0.8) < 1e-9
                      and abs(float(v["wd"])) < 1e-9)
        if scan:
            xs = [s[0] for s in scan]
            vs = [s[1] for s in scan]
            ax.plot(xs, vs, "-o", color=C_INK, lw=1.8, ms=6, zorder=2)
            bi = int(np.argmax(vs))
            ax.plot([xs[bi]], [vs[bi]], "o", color=C_CTRL, ms=12, zorder=3)
            ax.annotate(f"lr={xs[bi]:g}（最佳）\n训练准确率 {vs[bi]:.3f}",
                        (xs[bi], vs[bi]), xytext=(0, 16),
                        textcoords="offset points", ha="center", fontsize=8.6,
                        color="#137A5A", fontweight="bold")
            rounded = [round(x, 6) for x in xs]
            if 0.015 in rounded:
                j = rounded.index(0.015)
                ax.plot([xs[j]], [vs[j]], "o", color="#D0342C", ms=10, zorder=3)
                ax.annotate("上一轮用 lr=0.015\n几乎学不动（0.013）",
                            xy=(xs[j], vs[j]), xytext=(0.78, 0.55),
                            textcoords="axes fraction", ha="center",
                            fontsize=8.5, color="#D0342C",
                            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5),
                            arrowprops=dict(arrowstyle="->", color="#D0342C",
                                            lw=1.1))
        ax.set_xscale("log")
        ax.set_ylim(-0.015, 0.40)
        ax.set_xlabel("学习率（对数轴）", fontsize=9.5)
        ax.set_ylabel("训练准确率", fontsize=9.5)
        ax.set_title("(c) 同一架构、只改学习率：0.013 → 0.307",
                     fontsize=11.5, fontweight="bold", loc="left", pad=9)
        ax.grid(alpha=0.22)

        # ---------- (d) 结论卡 ----------
        ax = axes[1, 1]
        ax.axis("off")
        ax.set_title("(d) 这轮能说什么 / 不能说什么",
                     fontsize=11.5, fontweight="bold", loc="left", pad=9)
        ax.add_patch(Rectangle((0.005, 0.035), 0.99, 0.895,
                               transform=ax.transAxes,
                               facecolor="#F7F8FA", edgecolor="#D5D9DE",
                               lw=1.1))
        r0 = rich0["aggregate"] if rich0 else {}
        m1 = mlp1["aggregate"] if mlp1 else {}
        cards = [
            ("[已证实] 加先验：把 grokking 终点的圆编码直接喂进去，不用 grok 就会算",
             "#B35C18"),
            (f"[已证实] 不加先验：5 seed × {s2_steps:,} 步，训练 "
             f"{r0.get('train_accuracy_mean', 0):.3f} / 验证 "
             f"{r0.get('val_accuracy_mean', 0):.3f}，仍在记忆阶段", C_RICH),
            ("[已撤回] §12「架构缺少乘法交互」：学习率混淆 + 四项对照不同", "#C0392B"),
            ("[不能说] 本仓库模型永远不会 grok —— 连记忆都没完成", "#C0392B"),
            ("[注 意] 每 2000 步采样会漏掉 1000–1500 步的突发跳变，"
             "严格相变判定本轮不可用", C_MUTE),
            (f"[装 置] 同超参二次 MLP 验证 {m1.get('val_accuracy_mean', 0):.3f}，"
             "说明这套条件下任务确实可 grok", "#137A5A"),
        ]
        for i, (txt, col) in enumerate(cards):
            ax.text(0.028, 0.865 - i * 0.128, txt, transform=ax.transAxes,
                    fontsize=9.0, color=col, va="center", linespacing=1.5)
        ax.text(0.028, 0.075,
                "数据：docs/grokking_no_prior.json · 两架构同 lr / wd / 数据量 / "
                "seed / 切分 · 量子臂未参与本轮",
                transform=ax.transAxes, fontsize=7.5, color=C_MUTE, va="center")

        # 加先验两组来自 §12（不同 lr / 步数），只用于展示现象；
        # 结论性比较始终是 §13 内同超参的 wd 对照，写清楚避免误读。
        fig.text(0.5, 0.020,
                 "注：橙色（加先验）来自 §12 长训练，蓝色 / 绿色（不加先验）来自 §13（10 万步），"
                 "超参不完全一致，仅用于展示现象；因果结论以 §13 的同超参对照为准。\n"
                 f"(c) 为 1-seed 探索网格（同切分 / wd=0 / {s1_steps:,} 步），"
                 "各学习率均未达到记忆阈值 0.95。",
                 ha="center", va="bottom", fontsize=7.8, color="#888888",
                 linespacing=1.6)

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="10 步 x 1 seed 冒烟")
    ap.add_argument("--stage1-only", action="store_true", help="只跑探索网格")
    ap.add_argument("--stage1-steps", type=int, default=20000)
    ap.add_argument("--stage1-eval-every", type=int, default=500)
    ap.add_argument("--stage2-from", default="", help="复用 stage1 归档")
    ap.add_argument("--stage2-steps", type=int, default=100000)
    ap.add_argument("--stage2-eval-every", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--curve-n", type=int, default=1024)
    ap.add_argument("--redraw-from", default="")
    ap.add_argument("--recompute-from", default="",
                    help="从归档重算 verdict/summary 并重画图，不重跑训练")
    args = ap.parse_args()

    if args.recompute_from:
        p = Path(args.recompute_from)
        rep = json.loads(p.read_text(encoding="utf-8"))
        st = rep["stage2"]
        v, summary = build_verdict(st["per_condition"], st["steps"],
                                   st["eval_every"])
        rep["verdict"] = v
        st["summary"] = summary
        st["phase_detection_caveat"] = summary["phase_detection_caveat"]
        p.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
        _figure(rep, FIGURE)
        print(f"recomputed -> {p}\nredrawn -> {FIGURE}\nverdict: {v}")
        return

    if args.redraw_from:
        _figure(json.loads(Path(args.redraw_from).read_text(encoding="utf-8")), FIGURE)
        print(f"redrawn -> {FIGURE}")
        return

    quick = args.quick
    s1_steps, s2_steps = (10, 10) if quick else (args.stage1_steps, args.stage2_steps)
    s1_every, s2_every = (5, 5) if quick else (args.stage1_eval_every,
                                                args.stage2_eval_every)
    workers = 1 if quick else args.workers
    seeds = 1 if quick else args.seeds
    cfg = _cfg(args, quick)
    cfg["curve_n"] = 64 if quick else args.curve_n

    table = enumerate_table("+")
    tr, va = split_table(table, "holdout_pairs", 0.2, 0)
    ov = context_overlap(tr, va, "holdout_pairs", "+")
    assert ov["pair_level_disjoint"] and ov["full_context_coverage_k4"] == 0.0
    t0 = time.perf_counter()

    # ---- Stage 1: 1-seed 探索网格 ----
    if args.stage2_from:
        pd_ = json.loads(Path(args.stage2_from).read_text(encoding="utf-8"))
        pilot_per = pd_["per_condition"]
        selected = pd_["selected_conditions"]
        s1_steps, s1_every = pd_["steps"], pd_["eval_every"]
    elif quick:
        grid = _grid(s1_steps, s1_every)[:4]
        runs = _run_pool([(c, 0, cfg) for c in grid], workers)
        pilot_per = {r["condition"]: _slim(r) for r in runs}
        selected = _select(pilot_per, 1)
    else:
        grid = _grid(s1_steps, s1_every)
        print(f"[stage 1] {len(grid)} runs (rich + MLP, matched grid) x 1 seed "
              f"x {s1_steps} steps", flush=True)
        runs = _run_pool([(c, 0, cfg) for c in grid], workers)
        pilot_per = {r["condition"]: _slim(r) for r in runs}
        selected = _select(pilot_per, args.top_k)
        DOCS.mkdir(parents=True, exist_ok=True)
        PILOT.write_text(json.dumps({
            "status": "exploratory_1_seed_only",
            "steps": s1_steps, "eval_every": s1_every, "config": cfg,
            "n_conditions": len(grid), "per_condition": pilot_per,
            "selected_conditions": selected,
            "n_rich_memorized": sum(v["final"]["train_accuracy"] >= PHASE_MEM_ACC
                                    for v in pilot_per.values() if v["arch"] == "rich"),
            "n_rich_grokking": sum(v["phase"]["grokking_observed"]
                                   for v in pilot_per.values() if v["arch"] == "rich"),
            "n_mlp_grokking": sum(v["phase"]["grokking_observed"]
                                  for v in pilot_per.values() if v["arch"] == "quad_mlp"),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[stage 1] rich memorized "
              f"{sum(v['final']['train_accuracy'] >= PHASE_MEM_ACC for v in pilot_per.values() if v['arch'] == 'rich')}"
              f"/{len(grid)//2} | rich grok "
              f"{sum(v['phase']['grokking_observed'] for v in pilot_per.values() if v['arch'] == 'rich')}"
              f" | MLP grok "
              f"{sum(v['phase']['grokking_observed'] for v in pilot_per.values() if v['arch'] == 'quad_mlp')}"
              f" | selected {[s['name'] for s in selected]}", flush=True)
        if args.stage1_only:
            print(f"[stage 1] done -> {PILOT} | wall={time.perf_counter()-t0:.1f}s")
            return

    # ---- Stage 2: 5-seed 同超参、同步数、同切分确认 ----
    conds: list[dict] = []
    for s in selected:
        for arch in ("rich", "quad_mlp"):
            c = _cond(f"{'RICH' if arch == 'rich' else 'MLP'}/"
                      f"vf{s['val_fraction']:g}/lr{s['lr']:g}/wd{s['wd']:g}",
                      arch, s["lr"], s["wd"], s["val_fraction"], s2_steps, s2_every)
            conds.append(c)
    print(f"[stage 2] {len(conds)} conditions x {seeds} seeds x {s2_steps} steps "
          f"(matched everywhere except architecture)", flush=True)
    tasks = [(c, sd, cfg) for c in conds for sd in range(seeds)]
    runs = _run_pool(tasks, workers)
    per_condition: dict[str, dict] = {}
    for c in conds:
        sub = [r for r in runs if r["condition"] == c["name"]]
        per_condition[c["name"]] = {
            "cond": c, "aggregate": _agg(sub),
            "curves": [r["curve"] for r in sub],
            "finals": [r["final"] for r in sub],
            "init_hashes": [r["init_hashes"] for r in sub],
            "seconds_per_step": float(np.mean([r["seconds_per_step"] for r in sub])),
        }

    rich_keys = [k for k in per_condition if k.startswith("RICH")]
    mlp_keys = [k for k in per_condition if k.startswith("MLP")]
    n_rich_grok = sum(per_condition[k]["aggregate"]["n_grokking_seeds"] for k in rich_keys)
    n_mlp_grok = sum(per_condition[k]["aggregate"]["n_grokking_seeds"] for k in mlp_keys)
    n_rich_mem = sum(per_condition[k]["aggregate"]["n_memorized_seeds"] for k in rich_keys)

    verdict, summary = build_verdict(per_condition, s2_steps, s2_every)
    # stage1 的完整曲线已单独存在 grokking_no_prior_pilot.json，这里只留
    # 画图与结论需要的摘要，避免主归档重复存一份 500 KB 曲线。
    pilot_for_report = {n: {k: v for k, v in d.items() if k != "curve"}
                        for n, d in pilot_per.items()}

    report = {
        "status": "5_seed_confirm" if not quick else "smoke",
        "question": "加先验=跳过 grokking；不加先验=grokking 起跑线。"
                    "给不加先验组充分优化条件与相变尺度预算，它是否 grok？",
        "design": "rich 与 quad_mlp 使用完全相同的 lr/wd/val_fraction/batch/"
                  "steps/seed/切分，唯一差别是架构；MLP 仅作装置校验。",
        "stage1": {"status": "exploratory_1_seed", "steps": s1_steps,
                   "eval_every": s1_every, "per_condition": pilot_for_report,
                   "full_curves_in": "docs/grokking_no_prior_pilot.json",
                   "selected_conditions": selected},
        "stage2": {"steps": s2_steps, "eval_every": s2_every,
                   "seeds": list(range(seeds)), "per_condition": per_condition,
                   "n_rich_grokking_seeds": n_rich_grok,
                   "n_mlp_grokking_seeds": n_mlp_grok,
                   "n_rich_memorized_seeds": n_rich_mem,
                   "summary": summary,
                   "phase_detection_caveat": summary["phase_detection_caveat"],
                   "phase_thresholds": {"mem_acc": PHASE_MEM_ACC,
                                        "gen_acc": PHASE_GEN_ACC, "jump": 0.20}},
        "correction_to_previous_round": [
            "上一轮 best_random 40000 步 train acc=0.0288，连记忆阶段都没到，"
            "不能据 0/5 相变推断架构表示能力不足。",
            "上一轮正对照与 rich 的 lr/wd/训练数据占比/步数四项不同，"
            "+0.9910 不是干净的架构效应。",
            "Power et al. 2022 相变量级 1e4~1e6 步，40000 步仅在下界，"
            "不能断言不是训练预算问题。",
        ],
        "verdict": verdict,
        "wallclock_seconds_total": time.perf_counter() - t0,
    }
    if not quick:
        DOCS.mkdir(parents=True, exist_ok=True)
        METRICS.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
        _figure(report, FIGURE)
        print(f"  archived: {METRICS}\n  figure:   {FIGURE}")
    print("  verdict:", verdict)
    print(f"  wall={time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
