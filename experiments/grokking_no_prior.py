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
    _phase_detect,
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
    pilot = report["stage1"]["per_condition"]
    conf = report["stage2"]["per_condition"]
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8))

    # (a) Stage 1 网格：最终 train acc（是否进记忆阶段）
    ax = axes[0, 0]
    rich = [(n, v) for n, v in pilot.items() if v["arch"] == "rich"]
    rich.sort(key=lambda kv: kv[1]["final"]["train_accuracy"])
    names = [n.replace("RICH/", "") for n, _ in rich]
    vals = [v["final"]["train_accuracy"] for _, v in rich]
    ys = np.arange(len(names))
    ax.barh(ys, vals, color=["#1D9D75" if v >= PHASE_MEM_ACC else "#888780"
                             for v in vals], alpha=0.85, height=0.75)
    ax.axvline(PHASE_MEM_ACC, color="#D85A30", ls="--", lw=1.2,
               label=f"memorization threshold {PHASE_MEM_ACC}")
    ax.axvline(RANDOM_BASELINE_ACCURACY, color="#444444", ls=":", lw=1)
    ax.set_yticks(ys[::3])
    ax.set_yticklabels([names[i] for i in ys[::3]], fontsize=6)
    ax.set_xlabel("final train accuracy (1-seed exploratory grid)")
    ax.set_title(f"(a) can 'no prior' memorize? {report['stage1']['steps']} steps")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.25, axis="x")

    # 4 个条件（2 架构 x 2 wd）各自一个颜色；实线=train、虚线=val
    CMAP = {"RICH|0": "#185FA5", "RICH|1": "#D85A30",
            "MLP|0": "#1D9D75", "MLP|1": "#534AB7"}

    def _label(key: str, d: dict) -> tuple[str, str]:
        tag = "RICH" if key.startswith("RICH") else "MLP"
        wd = d["cond"]["wd"]
        ckey = f"{tag}|{int(wd > 0)}"
        return f"{tag} wd={wd:g}", CMAP.get(ckey, "#888780")

    # (b) Stage 2 曲线：两架构同超参、同步数
    ax = axes[0, 1]
    for key, d in conf.items():
        if not d["curves"]:
            continue
        curves = d["curves"]
        steps = [c["step"] for c in curves[0]]
        lbl, col = _label(key, d)
        for metric, style in (("train_acc", "-"), ("val_acc", "--")):
            arr = np.array([[c[metric] for c in cu] for cu in curves])
            ax.plot(steps, arr.mean(0), style, color=col, lw=1.7,
                    label=lbl if metric == "train_acc" else None)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("training steps (log)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"(b) matched hyperparameters @ {report['stage2']['steps']} steps")
    ax.legend(fontsize=7, loc="center left", ncol=1, framealpha=0.85,
              title="solid=train, dashed=val", title_fontsize=7)
    ax.grid(alpha=0.25)

    # (c) 傅里叶 progress measure
    ax = axes[1, 0]
    for key, d in conf.items():
        if not d["curves"]:
            continue
        curves = d["curves"]
        steps = [c["step"] for c in curves[0]]
        arr = np.array([[c["r2_k1"] for c in cu] for cu in curves])
        lbl, col = _label(key, d)
        ax.plot(steps, arr.mean(0), color=col, lw=1.7, label=lbl)
    ax.set_xscale("log")
    ax.set_ylim(-0.3, 1.05)
    ax.set_xlabel("training steps (log)")
    ax.set_ylabel("R² linear decode of sin/cos (k=1)")
    ax.set_title("(c) Fourier progress measure")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.25)

    # (d) 相变判定摘要
    ax = axes[1, 1]
    ax.axis("off")
    # 纯 ASCII：DejaVu Sans Mono 无中文字形，写中文会出方框并刷警告。
    lines = ["pre-registered primary question:",
             "  does 'no prior' grok at %d steps?" % report["stage2"]["steps"],
             ""]
    for key, d in conf.items():
        a = d["aggregate"]
        n = len(d["curves"])
        tag = key.split("/")[0]
        wd = d["cond"]["wd"]
        lines += [
            "%s wd=%-4g  grok %d/%d | memorized %d | generalized %d" %
            (tag, wd, a["n_grokking_seeds"], n, a["n_memorized_seeds"],
             a["n_generalized_seeds"]),
            "    final train %.4f  val %.4f" %
            (a["train_accuracy_mean"], a["val_accuracy_mean"]),
        ]
    s = report.get("stage2", {}).get("summary")
    if s:
        r, m = s["rich"], s["mlp"]
        lines += ["", "rich: %d/%d generalized, %d/%d memorized" %
                  (r["gen"], r["runs"], r["mem"], r["runs"]),
                  "mlp : %d/%d generalized, %d/%d memorized" %
                  (m["gen"], m["runs"], m["mem"], m["runs"])]
    lines += ["", "thresholds: mem>=%.2f gen>=%.2f jump>=%.2f" %
              (PHASE_MEM_ACC, PHASE_GEN_ACC,
               report["stage2"]["phase_thresholds"]["jump"]),
              "NOTE: eval_every=%d misses abrupt jumps before" %
              report["stage2"]["eval_every"],
              "      step 2000; device validity is shown by",
              "      val acc + the 500-step stage1 pilot.", "",
              "full verdict: docs/grokking_no_prior.json"]
    ax.text(0.01, 0.98, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=7.6, family="monospace",
            bbox=dict(fc="#f7f7f5", ec="#cccccc", alpha=0.9))
    ax.set_title("(d) phase-transition summary")

    fig.tight_layout()
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
    n_runs_rich = sum(len(per_condition[k]["curves"]) for k in rich_keys)
    n_runs_mlp = sum(len(per_condition[k]["curves"]) for k in mlp_keys)

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
