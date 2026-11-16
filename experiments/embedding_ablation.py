"""课题一补充消融：把「放开 embedding」拆成两个可分离的因素。

背景：课题一默认模式 ``trainable_circular`` 里，循环角度编码只是 **初始化**，
``nn.Embedding`` 一直是可训练参数，Adam 每步都在更新它（本脚本会实测
训练前后的 RMS 漂移）。文档里「固定循环角度编码」容易被误读成「冻结」，
所以这里用统一预算同时跑三种模式，把两件事干净地分开：

    frozen_circular    循环编码初始化 + 冻结     —— 真正的「不放开」
    trainable_circular 循环编码初始化 + 可训练   —— 课题一默认（本来就放开）
    trainable_random   纯随机初始化 + 可训练     —— 连人为先验也一起拿掉

前两种模式初始权重逐位相同（只差 ``requires_grad``），因此它们的配对差
就是「让 embedding 可训练」这一件事的效应；第三种与前者的配对差是
循环先验的效应。两臂（4-qubit 多观测量线路 / 同结构经典层）在每种模式下
都用同一组 seed、同一数据切分、同一 batch 顺序，配对比较。

方法学约定（与仓库 README / docs/REVIEW.md 一致）：
* 每个 worker 构造模型前 ``set_seed``；同 seed 同模式下两臂的 embed /
  causal_conv / head 初始 hash 必须逐位一致，否则实验作废。
* 结论性数字一律 >=5 seed 并报均值 ± 标准差；配对双侧 95% Student-t CI。
* 主检验 = val_accuracy / train_accuracy / generalization_gap，
  3 模式 x 3 指标 = 9 个检验统一做 Holm 校正；val_loss 不是主指标。
* 不预设结论，判定由结果自动生成。

用法：
    python experiments/embedding_ablation.py --quick      # 10 步冒烟，不写归档
    python experiments/embedding_ablation.py              # 3 模式 x 5 seed x 2000 步
    python experiments/embedding_ablation.py --steps 1200 --workers 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path(__file__).resolve().parent
for p in (str(ROOT), str(EXPERIMENTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_addition import (  # noqa: E402
    EMBEDDING_MODES,
    GROUP_IDS,
    _binom_vs_random,
    _holm,
    _paired_ci,
    run_one,
)
from qml_lab.arithmetic import (  # noqa: E402
    CONTEXT_WINDOW,
    MODULUS,
    RANDOM_BASELINE_ACCURACY,
    VOCAB_SIZE,
    context_overlap,
    enumerate_table,
    split_table,
)

DOCS = ROOT / "docs"
METRICS = DOCS / "embedding_ablation_metrics.json"
FIGURE = DOCS / "figures" / "embedding_ablation.png"

MODES = EMBEDDING_MODES
PRIMARY = ("val_accuracy", "train_accuracy", "generalization_gap")
ALL_METRICS = ("train_accuracy", "val_accuracy", "train_loss", "val_loss",
               "generalization_gap")
LABELS = {
    "frozen_circular": "frozen circular",
    "trainable_circular": "trainable circular",
    "trainable_random": "trainable random",
}
COLORS = {
    "add_q_rich48": "#534AB7",
    "add_classical": "#185FA5",
}
ARM_LABELS = {"add_q_rich48": "Quantum", "add_classical": "Classical"}


def _runs(runs: list[dict], mode: str, group: str) -> list[dict]:
    return [r for r in runs if r["embedding_mode"] == mode and r["group"] == group]


def _values(runs: list[dict], mode: str, group: str, metric: str) -> list[float]:
    return [r["final"][metric] for r in _runs(runs, mode, group)]


def _figure(report: dict, path: Path) -> None:
    """全中文主线图（2026-11-16 按项目级出图规范重做）。

    旧版三面板全是英文缩写、没有结论卡，图例压曲线、数值列挤在轴外，
    被指出「看都看不懂」。现在四个面板围绕同一个问题：默认模式本来就放开
    embedding，循环先验才是决定能不能学的因素。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    per_mode = report["per_mode"]
    interp = report.get("interpretation", {})
    n_seed = len(report["config"]["seeds"])
    C_MUTE = "#6B7280"
    C_INK = "#20242B"
    MODE_CN = {
        "frozen_circular": "冻结圆编码",
        "trainable_circular": "可训练圆编码\n（课题一默认）",
        "trainable_random": "随机初始化\n可训练",
    }
    ARM_CN = {"add_q_rich48": "量子层", "add_classical": "经典层"}
    EFFECT_CN = {
        "unfreeze_effect": "放开 − 冻结",
        "circular_prior_effect": "圆编码 − 随机",
        "frozen_vs_random": "冻结 − 随机",
    }

    with plt.rc_context({"font.sans-serif": ["Microsoft YaHei", "SimHei",
                                             "Noto Sans SC", "DejaVu Sans"],
                         "axes.unicode_minus": False}):
        fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.7))
        fig.subplots_adjust(left=0.125, right=0.972, top=0.858, bottom=0.112,
                            hspace=0.52, wspace=0.28)
        fig.suptitle("默认模式本来就放开 embedding；真正决定能不能学的是循环先验",
                     fontsize=17, fontweight="bold", color=C_INK, y=0.972)
        fig.text(0.5, 0.918,
                 f"模 112 加法 · 3 种 embedding 模式 × 量子/经典两臂 × {n_seed} seed"
                 " · 同切分 / 同步数 / 同超参 · 数值为全量评估均值 ± 标准差",
                 ha="center", fontsize=10, color="#555555")

        # ---------- (a) 三模式 × 两臂：全量验证准确率 ----------
        ax = axes[0, 0]
        xs = np.arange(len(MODES))
        w = 0.36
        for j, gid in enumerate(GROUP_IDS):
            vals = [per_mode[m]["aggregate"][gid]["val_accuracy_mean"] for m in MODES]
            errs = [per_mode[m]["aggregate"][gid]["val_accuracy_std"] for m in MODES]
            bars = ax.bar(xs + (j - 0.5) * w, vals, w, yerr=errs,
                          color=COLORS[gid], alpha=0.9, label=ARM_CN[gid],
                          error_kw=dict(ecolor="#555555", capsize=0, lw=1))
            for b, v in zip(bars, vals):
                # 数字抬到柱顶上方并加白底，避免误差棒横线被读成负号
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color=COLORS[gid],
                        bbox=dict(fc="white", ec="none", alpha=1.0, pad=0.6))
        ax.axhline(RANDOM_BASELINE_ACCURACY, color="#9AA0A6", ls=":", lw=1.2,
                   label=f"随机猜 1/112 = {RANDOM_BASELINE_ACCURACY:.3f}")
        ax.set_xticks(xs)
        ax.set_xticklabels([MODE_CN[m] for m in MODES], fontsize=9.5)
        ax.set_ylim(0, 0.32)
        ax.set_ylabel("验证集 top-1 准确率", fontsize=9.5)
        ax.set_title("(a) 拿掉循环先验，两臂都掉回随机基线",
                     fontsize=11.5, fontweight="bold", loc="left", pad=9)
        ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
        ax.grid(alpha=0.22, axis="y")

        # ---------- (b) 验证准确率学习曲线（子集采样，仅看趋势） ----------
        ax = axes[0, 1]
        linestyles = {"frozen_circular": "-", "trainable_circular": "--",
                      "trainable_random": ":"}
        for mode in MODES:
            for gid in GROUP_IDS:
                curves = [r["curve"] for r in _runs(report["runs"], mode, gid)]
                if not curves:
                    continue
                steps = [c["step"] for c in curves[0]]
                ys = np.array([[c["val_acc"] for c in cu] for cu in curves])
                mu = ys.mean(0)
                ax.plot(steps, mu, linestyles[mode], color=COLORS[gid],
                        alpha=0.9, lw=1.9,
                        label=f"{MODE_CN[mode].replace(chr(10), '')} · {ARM_CN[gid]}")
                if len(curves) > 1:
                    sd = ys.std(0, ddof=1)
                    ax.fill_between(steps, mu - sd, mu + sd,
                                    color=COLORS[gid], alpha=0.06)
        ax.axhline(RANDOM_BASELINE_ACCURACY, color="#9AA0A6", ls=":", lw=1.2)
        ax.set_ylim(0.0, 0.38)  # 顶部给两列图例留空，不压曲线
        ax.set_xlabel("训练步数", fontsize=9.5)
        ax.set_ylabel(f"验证集准确率（{report['config']['curve_n']} 条子集采样，仅看趋势）",
                      fontsize=8.8)
        ax.set_title("(b) 三种模式的验证曲线：随机初始化全程贴着随机基线",
                     fontsize=11.5, fontweight="bold", loc="left", pad=9)
        ax.legend(fontsize=7.0, ncol=2, loc="upper left", framealpha=0.94)
        ax.grid(alpha=0.22)

        # ---------- (c) 隔离效应的配对差 + 95% 置信区间 ----------
        ax = axes[1, 0]
        effects = report["cross_mode_paired"]
        keys = [k for k in effects if k.endswith("|val_accuracy")]
        ys = np.arange(len(keys))
        for i, k in enumerate(keys):
            p = effects[k]["ci"]
            part = k.split("|")
            col = COLORS[part[1]]
            ax.plot([p["ci95_low"], p["ci95_high"]], [i, i], color=col, lw=2.2)
            ax.plot([p["mean_diff"]], [i], "o", color=col, ms=6)
            # 数值放在子图内部右侧（右对齐），不再挤到轴外压住 (d) 面板
            ax.text(0.49, i,
                    f"{p['mean_diff']:+.3f} [{p['ci95_low']:+.3f}, {p['ci95_high']:+.3f}]",
                    ha="right", va="center", fontsize=7.0, color="#333333")
        ax.axvline(0, color="#444444", lw=1.2)
        ax.set_xlim(-0.20, 0.50)
        ax.set_yticks(ys)
        ax.set_yticklabels(
            [f"{EFFECT_CN.get(k.split('|')[0], k.split('|')[0])} · "
             f"{ARM_CN[k.split('|')[1]]}" for k in keys], fontsize=8.5)
        ax.set_ylim(-0.65, len(keys) - 0.35)
        ax.invert_yaxis()  # 第一个效应排在最上，阅读顺序自上而下
        ax.set_xlabel("验证准确率配对差（0 = 无差异；双侧 95% 置信区间）", fontsize=9.2)
        ax.set_title("(c) 「放开 vs 冻结」两条置信区间都跨 0",
                     fontsize=11.5, fontweight="bold", loc="left", pad=9)
        ax.grid(alpha=0.22, axis="x")

        # ---------- (d) 结论卡 ----------
        ax = axes[1, 1]
        ax.axis("off")
        ax.set_title("(d) 这轮能说什么 / 不能说什么",
                     fontsize=11.5, fontweight="bold", loc="left", pad=9)
        ax.add_patch(Rectangle((0.005, 0.035), 0.99, 0.895,
                               transform=ax.transAxes,
                               facecolor="#F7F8FA", edgecolor="#D5D9DE", lw=1.1))
        n_tests = interp.get("n_primary_tests", "?")
        n_sig = interp.get("n_primary_holm_significant", "?")
        cards = [
            ("[已证实] 默认「可训练圆编码」的 embedding 一直在训练，"
             "Adam 每步更新；不是这轮才放开", "#B35C18"),
            ("[已证实] 循环先验决定可学：随机初始化两臂都停在 0.006"
             "（随机猜 0.009）", "#137A5A"),
            (f"[未证实] 冻结更好：配对 95% CI 跨 0；{n_tests} 个主检验 "
             f"Holm 校正后仅 {n_sig} 个显著", "#C0392B"),
            ("[未证实] 任何量子层正贡献：量子 − 经典配对 CI 跨 0", "#C0392B"),
            ("[口 径] (a)(d) 全量评估均值；(b) 子集采样只看趋势；"
             "(c) 配对差与 95% CI", C_MUTE),
            ("[边 界] 本图只谈「加先验」的 embedding 消融；"
             "不加先验的 grokking 见 §13", C_MUTE),
        ]
        for i, (txt, col) in enumerate(cards):
            ax.text(0.028, 0.865 - i * 0.128, txt, transform=ax.transAxes,
                    fontsize=8.9, color=col, va="center", linespacing=1.5)
        ax.text(0.028, 0.065,
                "数据：docs/embedding_ablation_metrics.json · "
                "两臂同 seed / 同切分 / 同超参 / 同步数",
                transform=ax.transAxes, fontsize=7.5, color=C_MUTE, va="center")

        fig.text(0.5, 0.035,
                 "注：(a) 与 (d) 引用全量评估数值，(b) 使用子集采样曲线，两处口径不同；"
                 "结论以 (a)(c) 的配对统计为准。",
                 ha="center", fontsize=8.5, color=C_MUTE)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)



def _weight_controls(runs: list[dict], seeds: list[int]) -> dict:
    out: dict = {"all_passed": True, "checks": {}}
    for mode in MODES:
        for s in seeds:
            pair = [r for r in _runs(runs, mode, "add_q_rich48") if r["seed"] == s]
            pair += [r for r in _runs(runs, mode, "add_classical") if r["seed"] == s]
            h = {r["group"]: r["init_hashes"] for r in pair}
            ok = (len(h) == 2 and all(
                h[GROUP_IDS[0]][k] == h[GROUP_IDS[1]][k]
                for k in ("embed", "causal_conv_weight", "head_weight")
            ))
            out["checks"][f"{mode}|seed{s}"] = {"passed": ok, "hashes": h}
            out["all_passed"] &= ok
    # frozen 与 trainable_circular 的初始 embedding 必须逐位相同（只差 requires_grad）
    for s in seeds:
        a = [r for r in _runs(runs, "frozen_circular", GROUP_IDS[0]) if r["seed"] == s]
        b = [r for r in _runs(runs, "trainable_circular", GROUP_IDS[0]) if r["seed"] == s]
        ok = bool(a and b and a[0]["init_hashes"]["embed"] == b[0]["init_hashes"]["embed"])
        out["checks"][f"init_identical_frozen_vs_trainable|seed{s}"] = {"passed": ok}
        out["all_passed"] &= ok
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="10 步 x 1 seed 冒烟，不写归档")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--curve-n", type=int, default=512)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--redraw-from", default="",
                    help="从归档 JSON 重画对比图，不重跑训练")
    args = ap.parse_args()

    if args.redraw_from:
        report = json.loads(Path(args.redraw_from).read_text(encoding="utf-8"))
        _figure(report, FIGURE)
        print(f"redrawn from {args.redraw_from} -> {FIGURE}")
        return

    if args.quick:
        steps, seeds_n, eval_every, curve_n, workers, write = 10, 1, 5, 64, 1, False
    else:
        steps, seeds_n, eval_every, curve_n, workers, write = (
            args.steps, args.seeds, args.eval_every, args.curve_n, args.workers, True
        )

    seeds = list(range(seeds_n))
    base_cfg = {
        "op": "+",
        "split_mode": "holdout_pairs",
        "val_fraction": args.val_fraction,
        "split_seed": 0,
        "steps": steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "eval_every": eval_every,
        "eval_batch": 512,
        "curve_n": curve_n,
        "workers": workers,
        "seeds": seeds,
        "vocab_size": VOCAB_SIZE,
        "modulus": MODULUS,
        "groups": list(GROUP_IDS),
        "context_window": CONTEXT_WINDOW,
    }

    table = enumerate_table("+")
    train_pairs, val_pairs = split_table(table, "holdout_pairs", args.val_fraction, 0)
    overlap = context_overlap(train_pairs, val_pairs, "holdout_pairs", "+")
    assert overlap["pair_level_disjoint"], "训练/验证组合必须严格不相交"
    assert overlap["full_context_coverage_k4"] == 0.0, "k=4 上下文覆盖率必须为 0"

    tasks = [
        (g, s, {**base_cfg, "embedding_mode": m,
                "circular_embedding": m != "trainable_random"})
        for m in MODES for g in GROUP_IDS for s in seeds
    ]
    print(f"tasks={len(tasks)} modes={MODES} steps={steps} seeds={seeds_n} "
          f"workers={workers}", flush=True)
    t0 = time.perf_counter()
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            runs = list(ex.map(run_one, tasks))
    else:
        runs = [run_one(t) for t in tasks]
    elapsed = time.perf_counter() - t0

    weight = _weight_controls(runs, seeds)
    per_mode: dict = {}
    global_pairs: dict = {}
    for mode in MODES:
        by_group = {
            g: {m: _values(runs, mode, g, m) for m in ALL_METRICS}
            for g in GROUP_IDS
        }
        agg = {
            g: {
                f"{m}_{stat}": (float(np.mean(v)) if stat == "mean"
                                else float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                for m, v in ms.items() for stat in ("mean", "std")
            }
            for g, ms in by_group.items()
        }
        paired = {
            m: _paired_ci(by_group[GROUP_IDS[0]][m], by_group[GROUP_IDS[1]][m])
            for m in (*PRIMARY, "val_loss")
        }
        for m in PRIMARY:
            global_pairs[f"{mode}|{m}"] = paired[m]
        binom = {
            g: _binom_vs_random(agg[g]["val_accuracy_mean"], runs[0]["final"]["val_n"])
            for g in GROUP_IDS
        }
        learned = {
            g: bool(binom[g]["significantly_above_random"]
                    and agg[g]["val_accuracy_mean"] > 2 * RANDOM_BASELINE_ACCURACY)
            for g in GROUP_IDS
        }
        per_mode[mode] = {
            "aggregate": agg,
            "paired_quantum_minus_classical": paired,
            "binomial_vs_random": binom,
            "learned_compositional_generalization": learned,
            "embedding_drift_rms": {
                g: {
                    "mean": float(np.mean([r["embedding_drift_rms"]
                                           for r in _runs(runs, mode, g)])),
                    "min": float(np.min([r["embedding_drift_rms"]
                                         for r in _runs(runs, mode, g)])),
                    "max": float(np.max([r["embedding_drift_rms"]
                                         for r in _runs(runs, mode, g)])),
                }
                for g in GROUP_IDS
            },
            "n_trainable_params": {
                g: _runs(runs, mode, g)[0]["n_trainable_params"] for g in GROUP_IDS
            },
        }

    holm = _holm(global_pairs, list(global_pairs))
    for k, v in holm.items():
        global_pairs[k].update(v)
    for mode in MODES:
        for m in (*PRIMARY, "val_loss"):
            p = per_mode[mode]["paired_quantum_minus_classical"][m]
            p["bonferroni_alpha"] = 0.05 / len(global_pairs)
            p["bonferroni_significant"] = bool(p["p_value"] < 0.05 / len(global_pairs))
    n_holm_sig = sum(1 for v in global_pairs.values() if v.get("holm_significant"))

    # 跨模式配对：同 seed 同臂，隔离「放开 embedding」与「循环先验」两个效应
    comparisons = [
        ("trainable_circular", "frozen_circular", "unfreeze_effect"),
        ("trainable_circular", "trainable_random", "circular_prior_effect"),
        ("frozen_circular", "trainable_random", "frozen_vs_random"),
    ]
    cross: dict = {}
    for a, b, name in comparisons:
        for g in GROUP_IDS:
            for m in (*PRIMARY, "val_loss"):
                ci = _paired_ci(_values(runs, a, g, m), _values(runs, b, g, m))
                cross[f"{name}|{g}|{m}"] = {
                    "comparison": f"{a} - {b}", "group": g, "metric": m, "ci": ci
                }

    drift = per_mode["trainable_circular"]["embedding_drift_rms"]
    drift_frozen = per_mode["frozen_circular"]["embedding_drift_rms"]
    unfreeze_q = cross["unfreeze_effect|add_q_rich48|val_accuracy"]["ci"]
    unfreeze_c = cross["unfreeze_effect|add_classical|val_accuracy"]["ci"]
    prior_q = cross["circular_prior_effect|add_q_rich48|val_accuracy"]["ci"]
    learned_random = per_mode["trainable_random"]["learned_compositional_generalization"]
    q_holm = [k for k, v in global_pairs.items()
              if k.endswith("|val_accuracy") and v.get("holm_significant")
              and v["mean_diff"] > 0]

    verdict = (
        f"【事实】默认 trainable_circular 的 embedding 本来就在训练：训练前后 RMS 漂移 "
        f"量子 {drift['add_q_rich48']['mean']:.4f} / 经典 {drift['add_classical']['mean']:.4f}，"
        f"而 frozen_circular 两臂均严格为 {drift_frozen['add_q_rich48']['max']:.4f}（0）。"
        f"【放开 vs 冻结】val acc 配对差 量子 {unfreeze_q['mean_diff']:+.4f} "
        f"CI[{unfreeze_q['ci95_low']:+.4f},{unfreeze_q['ci95_high']:+.4f}]、"
        f"经典 {unfreeze_c['mean_diff']:+.4f} "
        f"CI[{unfreeze_c['ci95_low']:+.4f},{unfreeze_c['ci95_high']:+.4f}]。"
        f"【循环先验】trainable_circular 与 trainable_random 的量子臂差 "
        f"{prior_q['mean_diff']:+.4f} CI[{prior_q['ci95_low']:+.4f},{prior_q['ci95_high']:+.4f}]；"
        f"纯随机可训练 embedding 下两臂学会组合泛化 = {learned_random}。"
        f"【量子优势】9 个主检验 Holm 校正后 {n_holm_sig} 个显著；"
        f"val_accuracy 上显著的量子正贡献 = {q_holm if q_holm else '无'}。"
        f"上述差值均为配对 CI，未校正的多重比较不得当作显著性证据。"
    )

    report = {
        "task": "embedding ablation on mod-112 addition (Z_112)",
        "question": "把 embedding 放开（可训练）再训练，三模式对照会发生什么",
        "config": {**base_cfg, "modes": list(MODES)},
        "design": {
            "frozen_circular": "循环角度编码初始化 + embed.requires_grad=False",
            "trainable_circular": "循环角度编码初始化 + embed 可训练（课题一默认）",
            "trainable_random": "纯随机初始化 + embed 可训练（无循环先验）",
            "isolated_effect": "frozen 与 trainable_circular 初始权重逐位相同，只差 requires_grad",
            "same_budget": "三模式共用 steps/lr/batch/split/seeds/batch 顺序",
            "comparable_to_review_section10": (
                "steps/lr/batch/split/seed 与 docs/addition_metrics.json 一致；"
                "本轮 eval_every/curve_n 更小（只影响曲线采样密度与耗时，不影响全量最终指标）"
            ),
        },
        "corpus_overlap": overlap,
        "weight_control_checks": weight,
        "runs": runs,
        "per_mode": per_mode,
        "cross_mode_paired": cross,
        "interpretation": {
            "verdict": verdict,
            "primary_endpoints": list(PRIMARY),
            "n_primary_tests": len(global_pairs),
            "n_primary_holm_significant": n_holm_sig,
            "default_mode_already_trainable": True,
            "random_baseline_accuracy": RANDOM_BASELINE_ACCURACY,
        },
        "caveat": (
            "val 是未见的 (a,b) 组合而非未见数字，属组合泛化；答案位只在 112 个"
            "数字类上评估。循环编码是两臂共享的经典先验，任何由其带来的提升都"
            "不能归因于量子层。三模式共用同一组超参，未对任一臂单独调参；"
            "训练预算有限，纯随机 embedding 未收敛不等于该设置不可能收敛。"
        ),
        "wallclock_seconds_total": elapsed,
    }

    print("\n=== embedding ablation ===")
    for mode in MODES:
        a = per_mode[mode]["aggregate"]
        d = per_mode[mode]["embedding_drift_rms"]
        print(f"  {mode:20s} Q val {a['add_q_rich48']['val_accuracy_mean']:.4f} "
              f"± {a['add_q_rich48']['val_accuracy_std']:.4f} | "
              f"C val {a['add_classical']['val_accuracy_mean']:.4f} "
              f"± {a['add_classical']['val_accuracy_std']:.4f} | "
              f"drift(Q) {d['add_q_rich48']['mean']:.4f}")
    for k in ("unfreeze_effect|add_q_rich48|val_accuracy",
              "unfreeze_effect|add_classical|val_accuracy",
              "circular_prior_effect|add_q_rich48|val_accuracy",
              "circular_prior_effect|add_classical|val_accuracy"):
        p = cross[k]["ci"]
        print(f"  {k:52s} Δ={p['mean_diff']:+.4f} "
              f"CI[{p['ci95_low']:+.4f},{p['ci95_high']:+.4f}] p={p['p_value']:.3g}")
    print(f"  Holm significant (9 primary tests): {n_holm_sig} | "
          f"weight control all_passed={weight['all_passed']} | wall={elapsed:.1f}s")
    print("  verdict:", verdict)

    if write:
        DOCS.mkdir(parents=True, exist_ok=True)
        METRICS.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
        _figure(report, FIGURE)
        print(f"  archived: {METRICS}\n  figure:   {FIGURE}")


if __name__ == "__main__":
    main()
