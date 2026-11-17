"""L1 量子循环短预算 A0/A1 pilot：验证架构契约与训练成本，不做效果结论。

背景见 ``docs/ARCHITECTURE_PROPOSAL.md`` 第 10 节。既有 ``RichQuantumLM`` 的
量子层逐位置独立，P1–P4 全不满足；``qml_lab/recurrent.py`` 给出 L1 测量反馈型
原型。本脚本是该原型的**第一步实测**，只回答三个问题：

1. **P1–P4 能不能在真实训练/前向里量化验证**（前缀敏感度、跨步梯度、
   严格因果、逐步生成）；
2. **状态反馈消融 A0/A1 有没有可测的行为差**（同一量子模型，只改
   ``feedback``；不是量子 vs 经典对照）；
3. **每步耗时 / 外推成本 / 内存**是多少，够不够支撑后续 A2/A3 正式实验。

刻意不回答、也不允许从本脚本读出的问题：

* 量子循环比经典 RNN 好还是差。用户已明确「不关心量子的优势……在乱比」；
  经典臂在这里只作**装置有效性正对照**（证明任务/训练代码能学动），
  不作优劣比较、不报配对显著数、不下 Holm 结论。
* 任何需要 ≥5 seed 的结论性数字。本 pilot 是 2 seed × 短步数的探索性
  成本与装置标定，所有数值只能标「示意 / 装置检查」。

任务口径按方案第 5 节换掉一次性答案分类：标准下一 token 预测
（``logits = model(x)`` 对 ``y = x 右移一位`` 展平 CE），评估用独立切分语料
并报 k=4 上下文重叠率；另加一个「延迟回忆」合成小任务，用来检验状态
是否真的承载跨步信息（无状态模型应接近随机基线）。

内存约束：单进程、``OMP_NUM_THREADS=1 MKL_NUM_THREADS=1``，默认
``batch=8 / seq_len=8 / steps=120``。计时探针实测量子臂 T=8 约 0.17 s/步。
**不要并行叠加其它训练任务**。

用法：
    python experiments/recurrent_pilot.py --quick            # 冒烟，产物写 results/
    python experiments/recurrent_pilot.py                    # 正式短 pilot，写 docs/
    python experiments/recurrent_pilot.py --probe-memory     # 只复测单进程峰值工作集
    python experiments/recurrent_pilot.py --recompute-from docs/recurrent_pilot.json
    python experiments/recurrent_pilot.py --redraw-from docs/recurrent_pilot.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qml_lab.data import (  # noqa: E402
    build_char_data,
    corpus_overlap,
    iter_batches,
    load_independent_split,
)
from qml_lab.recurrent import (  # noqa: E402
    ClassicalRecurrentLM,
    QuantumRecurrentLM,
)
from qml_lab.utils import set_seed  # noqa: E402

DOCS = ROOT / "docs"
RESULTS = ROOT / "results"
OUT_NAME = "recurrent_pilot"
FIG_NAME = "recurrent_pilot.png"
# 「延迟回忆是否学会」的判定margin：末位 acc 需超过随机基线 + 0.10。
# 该阈值是装置标定用的显式判据，不是统计显著阈值（本 pilot 只有 2 seed）。
RECALL_ABOVE_BASELINE_MARGIN = 0.10

# A0 / A1 是同一量子模型的状态消融；经典两臂仅作装置有效性正对照。
ARMS = {
    "A0_quantum_feedback": dict(impl="quantum", feedback=True),
    "A1_quantum_no_feedback": dict(impl="quantum", feedback=False),
    "CTRL_classical_feedback": dict(impl="classical", feedback=True),
    "CTRL_classical_no_feedback": dict(impl="classical", feedback=False),
}
ARM_LABEL = {
    "A0_quantum_feedback": "量子循环·状态反馈开",
    "A1_quantum_no_feedback": "量子循环·状态反馈关",
    "CTRL_classical_feedback": "经典RNN·状态反馈开",
    "CTRL_classical_no_feedback": "经典RNN·状态反馈关",
}
C_QUANTUM = "#534AB7"
C_QUANTUM_OFF = "#9A93D4"
C_CLASSICAL = "#185FA5"
C_CLASSICAL_OFF = "#9EC2E0"
C_OK = "#1D9E75"
C_BAD = "#C0392B"


def build_arm(name: str, vocab_size: int, seed: int, n_qubits: int = 4,
              hidden: int = 32) -> torch.nn.Module:
    """构造一臂；全局 RNG 在构造前 set_seed，保证 A0/A1 初始权重逐元素相同。"""
    spec = ARMS[name]
    set_seed(seed)
    if spec["impl"] == "quantum":
        return QuantumRecurrentLM(
            vocab_size, n_qubits=n_qubits, hidden=hidden,
            feedback=spec["feedback"],
        )
    return ClassicalRecurrentLM(
        vocab_size, state_dim=n_qubits, hidden=hidden,
        feedback=spec["feedback"],
    )


def assert_ablation_matched(vocab_size: int, seed: int,
                            n_qubits: int = 4) -> dict:
    """代码级硬检查：A0/A1 与两经典臂除 feedback 外初始权重必须逐元素相同。

    这是项目技能硬检查第 3 条在 L1 上的落实：字段一致不等于对照干净，
    状态消融只允许差 ``feedback`` 一个变量。
    """
    pairs = (("A0_quantum_feedback", "A1_quantum_no_feedback"),
             ("CTRL_classical_feedback", "CTRL_classical_no_feedback"))
    out = {}
    for a, b in pairs:
        ma, mb = build_arm(a, vocab_size, seed, n_qubits), build_arm(b, vocab_size, seed, n_qubits)
        sa, sb = ma.state_dict(), mb.state_dict()
        assert set(sa) == set(sb), f"{a}/{b} 参数键不一致"
        diffs = {k: float((sa[k].float() - sb[k].float()).abs().max())
                 for k in sa}
        worst = max(diffs.values(), default=0.0)
        assert worst == 0.0, f"{a}/{b} 初始权重不一致（最大差 {worst}），消融失效"
        out[f"{a}__vs__{b}"] = {
            "max_abs_init_weight_diff": worst,
            "n_tensors": len(sa),
            "only_difference": "feedback",
        }
    return out


# --------------------------------------------------------------------------
# P1–P4 架构契约量化
# --------------------------------------------------------------------------
def check_contracts(model: torch.nn.Module, vocab_size: int, seed: int,
                    seq_len: int = 8, batch: int = 8) -> dict:
    """对单臂量化 P1–P4；返回纯数值，不打印结论。"""
    g = torch.Generator().manual_seed(10_000 + seed)
    idx = torch.randint(0, vocab_size, (batch, seq_len), generator=g)
    model.eval()
    out = {}

    # P1：前缀改变后，末位 logits 必须改变（信息经状态传到 t=seq_len-1）。
    alt = idx.clone()
    alt[:, :seq_len - 1] = (alt[:, :seq_len - 1] + 7) % vocab_size
    with torch.no_grad():
        base = model(idx)
        changed = model(alt)
    out["P1_prefix_logits_mean_abs_delta"] = float(
        (base[:, -1, :] - changed[:, -1, :]).abs().mean())

    # P2：上一步状态参与演化且梯度跨步。直接量测 state 输入的雅可比，
    #     并量化置零状态后输出的平均变化。
    with torch.no_grad():
        s = torch.zeros(batch, model.state_dim, requires_grad=False)
        _, s1 = model.step(idx[:, 0], s)
    s_leaf = s1.detach().clone().requires_grad_(True)
    logits_1, s2 = model.step(idx[:, 1], s_leaf)
    (logits_1.pow(2).mean() + s2.pow(2).mean()).backward()
    out["P2_state_grad_norm"] = float(s_leaf.grad.norm()) if s_leaf.grad is not None else 0.0
    with torch.no_grad():
        _, s_zero = model.step(idx[:, 1], torch.zeros_like(s_leaf))
    out["P2_state_zeroing_mean_abs_delta"] = float(
        (s2.detach() - s_zero).abs().mean())

    # P2b：BPTT —— 末位 loss 对线路/RNNCell 权重的梯度是否非空。
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(idx)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                           idx[:, 1:].reshape(-1))
    loss.backward()
    key = "weights" if hasattr(model, "weights") else "cell.weight_hh"
    p = dict(model.named_parameters())[key]
    out["P2b_bptt_grad_norm"] = float(p.grad.norm()) if p.grad is not None else 0.0

    # P3：严格因果 —— 改掉未来 token，过去位置 logits 必须完全不变。
    model.eval()
    fut = idx.clone()
    fut[:, -1] = (fut[:, -1] + 3) % vocab_size
    with torch.no_grad():
        a = model(idx)
        b = model(fut)
    out["P3_future_leakage_max_abs"] = float(
        (a[:, :-1, :] - b[:, :-1, :]).abs().max())

    # P4：逐步生成 —— 前缀保持、长度正确、确实产生了新 token。
    prefix = idx[:, :4]
    gen = model.generate(prefix, n_new=6)
    out["P4_generate_length_ok"] = bool(gen.shape[1] == prefix.shape[1] + 6)
    out["P4_prefix_preserved"] = bool(torch.equal(gen[:, :4], prefix))
    out["P4_new_token_fraction"] = float(
        (gen[:, 4:] != prefix[:, -1:]).float().mean())
    return out


# --------------------------------------------------------------------------
# 短训练：标准下一 token 预测 + 延迟回忆装置有效性任务
# --------------------------------------------------------------------------
def _train_one(name: str, vocab_size: int, seed: int, steps: int,
               seq_len: int, batch: int, lr: float,
               batch_iter, eval_fn, log_every: int = 20,
               task: str = "char_lm") -> dict:
    model = build_arm(name, vocab_size, seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr)
    curve = []
    t0 = time.perf_counter()
    model.train()
    for step in range(steps):
        x, y = next(batch_iter)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        if task == "delayed_recall":
            # 只在末位监督复现；其余位置全是 pad，展平 CE 会把记忆信号稀释掉
            loss = F.cross_entropy(logits[:, -1, :], y[:, -1])
        else:
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{name}/{task}/seed{seed} step{step} loss 非有限数")
        loss.backward()
        opt.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            curve.append({"step": step + 1, "loss": float(loss.detach())})
    wall = time.perf_counter() - t0
    model.eval()
    ev = eval_fn(model)
    return {
        "task": task, "arm": name, "seed": seed, "steps": steps,
        "curve": curve,
        "train_final_loss": curve[-1]["loss"],
        "eval": ev,
        "seconds_per_step": wall / max(steps, 1),
        "wallclock_seconds": wall,
    }


def _char_eval_factory(val_data: torch.Tensor, vocab_size: int,
                       seq_len: int, n_batches: int = 4, batch: int = 16):
    gen = iter_batches(val_data, seq_len=seq_len, batch_size=batch, seed=2026)

    def ev(model):
        losses = []
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = next(gen)
                lg = model(x)
                losses.append(float(F.cross_entropy(
                    lg.reshape(-1, vocab_size), y.reshape(-1))))
        return {"val_loss": float(np.mean(losses)),
                "val_perplexity": float(math.exp(min(float(np.mean(losses)), 20.0))),
                "n_batches": n_batches}
    return ev


def _delayed_recall_batches(k: int = 4, pad_len: int = 4, seed: int = 0,
                            batch: int = 8):
    """延迟回忆：``[target][pad...pad][cue]`` -> 只在末位监督复现 target。

    无状态模型（feedback=False）在末位只看到 cue，理论上无法超过 1/k；
    有状态模型需要把 target 经多步传递过来。词表 = k 个 target + pad + cue。
    """
    vocab_size = k + 2
    pad_id, cue_id = k, k + 1
    g = torch.Generator().manual_seed(seed)
    seq_len = pad_len + 2

    def gen():
        while True:
            tgt = torch.randint(0, k, (batch,), generator=g)
            x = torch.full((batch, seq_len), pad_id, dtype=torch.long)
            x[:, 0] = tgt
            x[:, -1] = cue_id
            # 只在末位算 CE；其余位置的目标置为 pad 但会被 mask 掉
            y = torch.full((batch, seq_len), pad_id, dtype=torch.long)
            y[:, -1] = tgt
            yield x, y

    return gen(), vocab_size, seq_len, k


def _recall_eval_factory(k: int, pad_len: int, vocab_size: int,
                         n_batches: int = 4, batch: int = 8):
    gen, _, _, _ = _delayed_recall_batches(k, pad_len, seed=99_999, batch=batch)

    def ev(model):
        losses, accs = [], []
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = next(gen)
                lg = model(x)
                last = lg[:, -1, :]
                losses.append(float(F.cross_entropy(last, y[:, -1])))
                accs.append(float((last.argmax(-1) == y[:, -1]).float().mean()))
        return {"val_loss": float(np.mean(losses)),
                "last_token_accuracy": float(np.mean(accs)),
                "random_baseline": 1.0 / k,
                "n_batches": n_batches}
    return ev


# --------------------------------------------------------------------------
# 计时与内存
# --------------------------------------------------------------------------
def _timing(vocab_size: int, seq_len: int, batch: int,
            repeats: int, seed: int = 0) -> dict:
    """单步前向+反向耗时（不更新参数），用于外推正式训练成本。"""
    out = {}
    g = torch.Generator().manual_seed(777 + seed)
    idx = torch.randint(0, vocab_size, (batch, seq_len), generator=g)
    for name in ARMS:
        m = build_arm(name, vocab_size, seed)
        m.train()
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            lg = m(idx)
            loss = F.cross_entropy(lg[:, :-1].reshape(-1, vocab_size),
                                   idx[:, 1:].reshape(-1))
            loss.backward()
            ts.append(time.perf_counter() - t0)
            m.zero_grad(set_to_none=True)
        out[name] = float(np.median(ts))
    return out


def _peak_rss_mb() -> float | None:
    """单进程峰值工作集（MB）。优先 psutil；无第三方库时用 Windows API。"""
    try:
        import psutil  # type: ignore
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        if ok:
            return pmc.PeakWorkingSetSize / 2 ** 20
    except Exception:
        pass
    return None


def build_memory_probe(vocab_size: int = 29, batch: int = 8,
                       seq_len: int = 8, steps: int = 3) -> dict | None:
    """独立复测单进程峰值工作集（MB），供 ``runtime.memory_probe`` 复现。

    与正式 pilot 同形状（四臂各若干步前向 + 反向），只回答「这套配置占多少
    内存」；不训练、不写文件、不产生任何结论。Windows 无 psutil 时走
    ``GetProcessMemoryInfo``。进程峰值只增不减，因此基线必须在建模前读取。
    """
    baseline = _peak_rss_mb()
    g = torch.Generator().manual_seed(3)
    idx = torch.randint(0, vocab_size, (batch, seq_len), generator=g)
    for name in ARMS:
        model = build_arm(name, vocab_size, 0)
        model.train()
        for _ in range(steps):
            logits = model(idx)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                idx[:, 1:].reshape(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)
    peak = _peak_rss_mb()
    if baseline is None or peak is None:
        return None
    return {
        "baseline_peak_mb": round(baseline, 1),
        "peak_working_set_mb": round(peak, 1),
        "config": (f"{len(ARMS)} 臂各 {steps} 步前向+反向，B={batch}/T={seq_len}，"
                   "单进程，OMP=MKL=1"),
        "reproduce": "python experiments/recurrent_pilot.py --probe-memory",
        "note": "进程峰值工作集（Windows GetProcessMemoryInfo / psutil），含 torch+pennylane 基线。",
    }


def build_verdict(report: dict) -> dict:
    """只判装置有效性，不判优劣；结论措辞必须能挡住误读。"""
    qc = report["contracts"]["A0_quantum_feedback"]
    qn = report["contracts"]["A1_quantum_no_feedback"]
    cc = report["contracts"]["CTRL_classical_feedback"]

    p1_effect = (qc["P1_prefix_logits_mean_abs_delta"]
                 - qn["P1_prefix_logits_mean_abs_delta"])
    p2_ok = (qc["P2_state_grad_norm"] > 1e-8
             and qc["P2b_bptt_grad_norm"] > 1e-8
             and qn["P2_state_grad_norm"] < 1e-12)
    p3_ok = all(v["P3_future_leakage_max_abs"] < 1e-6
                for v in report["contracts"].values())
    p4_ok = all(v["P4_generate_length_ok"] and v["P4_prefix_preserved"]
                for v in report["contracts"].values())

    char = report["char_lm"]["per_arm"]
    def _mean(arm, field):
        return float(np.mean([r["eval"][field] for r in char[arm]]))
    classical_learned = (
        _mean("CTRL_classical_feedback", "val_loss")
        < math.log(report["config"]["vocab_size"]) - 0.05
    )
    quantum_learned = (
        _mean("A0_quantum_feedback", "val_loss")
        < math.log(report["config"]["vocab_size"]) - 0.05
    )

    recall = report["delayed_recall"]["per_arm"]
    def _racc(arm):
        return float(np.mean([r["eval"]["last_token_accuracy"]
                             for r in recall[arm]]))
    recall_ctrl = _racc("CTRL_classical_feedback")
    recall_base = report["delayed_recall"]["random_baseline"]
    recall_q_on = _racc("A0_quantum_feedback")
    recall_q_off = _racc("A1_quantum_no_feedback")
    recall_c_off = _racc("CTRL_classical_no_feedback")
    # 关键区分：P1–P4 是「结构上有无资格」，功能有效性是「状态真的被用上」。
    # 两者不能混为一谈；量子臂停在基线不等于架构不行，只是本预算未证明。
    functional = {
        "criterion": (
            f"末位 acc > 随机基线 {recall_base:.2f} + "
            f"{RECALL_ABOVE_BASELINE_MARGIN:.2f}（装置标定判据，非统计显著检验）"
        ),
        "quantum_above_baseline": bool(
            recall_q_on > recall_base + RECALL_ABOVE_BASELINE_MARGIN),
        "quantum_feedback_gain": recall_q_on - recall_q_off,
        "classical_feedback_gain": recall_ctrl - recall_c_off,
        "classical_positive_control_above_baseline": bool(
            recall_ctrl > recall_base + RECALL_ABOVE_BASELINE_MARGIN),
        "quantum_accuracy": recall_q_on,
        "quantum_no_feedback_accuracy": recall_q_off,
        "classical_accuracy": recall_ctrl,
        "classical_no_feedback_accuracy": recall_c_off,
        "random_baseline": recall_base,
        "interpretation": (
            "P1–P4 结构契约通过，只说明信息流与梯度路径存在；"
            "延迟回忆上量子两臂仍近随机基线，说明本短预算下"
            "尚未证明状态反馈被模型实际利用。不能推出「量子循环不能记忆」，"
            "也不能推出「经典更好」——两臂预算、容量、参数化都未对齐。"
        ),
    }

    checks = {
        "P1_前缀影响末位输出": bool(p1_effect > 1e-4),
        "P2_状态梯度跨步且关反馈后消失": bool(p2_ok),
        "P3_严格因果无未来泄漏": bool(p3_ok),
        "P4_逐步生成接口正确": bool(p4_ok),
        "装置有效_经典正对照loss低于均匀基线": bool(classical_learned),
        "装置有效_量子循环loss低于均匀基线": bool(quantum_learned),
    }
    return {
        "status": "探索性短预算 pilot；不得据此下性能结论",
        "p1_prefix_effect_mean_abs_delta": p1_effect,
        "checks": checks,
        "all_contracts_pass": all(checks[k] for k in checks
                                  if k.startswith("P")),
        "functional_state_use": functional,
        "device_validity": {
            "classical_positive_control_val_loss": _mean(
                "CTRL_classical_feedback", "val_loss"),
            "quantum_cycle_val_loss": _mean("A0_quantum_feedback", "val_loss"),
            "uniform_baseline_loss": math.log(report["config"]["vocab_size"]),
            "delayed_recall_classical_accuracy": recall_ctrl,
            "delayed_recall_random_baseline": recall_base,
        },
        "claims_allowed": [
            "L1 原型在结构上满足 P1–P4（若四条为 true）",
            "关掉状态反馈后前缀敏感度与跨步梯度严格归零（单变量消融）",
            "每步耗时与外推成本",
            "本短预算下量子臂未完成延迟回忆，装置的功能有效性未证明",
        ],
        "claims_forbidden": [
            "量子循环优于/劣于经典 RNN",
            "量子层对语言建模有/无正贡献",
            "任何准确率差值、配对 CI、Holm 校正后的显著数",
            "任何需要 ≥5 seed 的结论性数字",
        ],
        "caveats": [
            "2 seed × 短步数，仅装置标定与成本测量。",
            "延迟回忆若各臂都停在随机基线，只能说明该预算下没学会，"
            "不能推出「状态无用」或「架构不行」；需更长训练与消融才能判断。",
            "经典臂只作任务/训练代码的正对照，不构成同构消融。",
            "状态仍是经典数值（测量-反馈 L1），量子态本身不跨步延续（L2 未实现）。",
            "模 112 加法答案只有 1 个 token，不适合检验序列依赖，本 pilot 未用它。",
            "仍是 default.qubit 解析状态向量，无 shots / 噪声 / 真机。",
        ],
    }


# --------------------------------------------------------------------------
# 中文结论式图
# --------------------------------------------------------------------------
def _figure(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cn = next((f.name for f in font_manager.fontManager.ttflist
               if "YaHei" in f.name or "Microsoft YaHei" in f.name), None)
    colors = {
        "A0_quantum_feedback": C_QUANTUM,
        "A1_quantum_no_feedback": C_QUANTUM_OFF,
        "CTRL_classical_feedback": C_CLASSICAL,
        "CTRL_classical_no_feedback": C_CLASSICAL_OFF,
    }
    short = {
        "A0_quantum_feedback": "量子·开",
        "A1_quantum_no_feedback": "量子·关",
        "CTRL_classical_feedback": "经典·开",
        "CTRL_classical_no_feedback": "经典·关",
    }
    with plt.rc_context({"font.sans-serif": [cn or "DejaVu Sans"],
                         "axes.unicode_minus": False}):
        fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))
        verdict = report["verdict"]

        # (a) P1：前缀敏感度，开/关反馈的对照必须画出来
        ax = axes[0][0]
        names = list(ARMS)
        vals = [report["contracts"][n]["P1_prefix_logits_mean_abs_delta"]
                for n in names]
        xs = np.arange(len(names))
        shown = [max(v, 1e-9) for v in vals]   # 关反馈时应恰为 0，用极短柱显示
        ax.bar(xs, shown, 0.6, color=[colors[n] for n in names])
        ax.set_yscale("log")
        for i, v in enumerate(vals):
            ax.text(i, shown[i] * 2.6, f"{v:.2e}", ha="center",
                    fontsize=9.2, color="#222222",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        ax.set_xticks(xs)
        ax.set_xticklabels([short[n] for n in names], fontsize=9.5)
        ax.set_ylabel("改前缀后末位输出变化（对数轴）", fontsize=9.5)
        ax.set_title("(a) 状态反馈开，前缀才传得到末位", fontsize=11.2,
                     fontweight="bold", loc="left", pad=8)
        ax.grid(alpha=0.22, axis="y")
        ax.set_ylim(bottom=3e-10, top=max(shown) * 60)

        # (b) 字符语料短训练曲线
        ax = axes[0][1]
        for n in names:
            curves = report["char_lm"]["per_arm"][n]
            steps = curves[0]["curve"]
            xs = [p["step"] for p in steps]
            ys = [float(np.mean([c["curve"][i]["loss"] for c in curves]))
                  for i in range(len(steps))]
            ax.plot(xs, ys, lw=2, color=colors[n], label=short[n])
        uni = math.log(report["config"]["vocab_size"])
        ax.axhline(uni, color="#888888", ls="--", lw=1.4)
        # 图例置顶框内，基线文字放在同一高度的左侧，两者不重叠
        ax.set_ylim(2.32, 3.86)
        ax.text(0.02, uni + 0.035, f"均匀基线 ln(V)={uni:.2f}",
                transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=9, color="#666666")
        ax.set_xlabel("训练步数", fontsize=9.5)
        ax.set_ylabel("训练交叉熵（nats）", fontsize=9.5)
        ax.set_title("(b) 四臂 loss 都下降：训练装置没有塌", fontsize=11.2,
                     fontweight="bold", loc="left", pad=8)
        ax.legend(fontsize=9, frameon=False, ncol=4,
                  loc="upper center", bbox_to_anchor=(0.5, 1.005))
        ax.grid(alpha=0.22)

        # (c) 延迟回忆：状态是否承载跨步信息
        ax = axes[1][0]
        accs = [float(np.mean([r["eval"]["last_token_accuracy"]
                              for r in report["delayed_recall"]["per_arm"][n]]))
                for n in names]
        xs_c = np.arange(len(names))
        ax.bar(xs_c, accs, 0.6, color=[colors[n] for n in names])
        base = report["delayed_recall"]["random_baseline"]
        ax.axhline(base, color=C_BAD, ls="--", lw=1.6)
        ax.text(0.985, base + 0.012,
                f"随机基线 1/{int(round(1/base))}={base:.2f}",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=9.2, color=C_BAD,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
        for i, v in enumerate(accs):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9.4)
        ax.set_xticks(xs_c)
        ax.set_xticklabels([short[n] for n in names], fontsize=9.5)
        ax.set_ylabel("末位复现准确率", fontsize=9.5)
        ax.set_ylim(0, max(1.02, max(accs) * 1.22))
        fv = verdict["functional_state_use"]
        title_c = ("(c) 短预算下只有经典正对照学会了延迟回忆"
                   if not fv["quantum_above_baseline"]
                   else "(c) 跨步记忆：开反馈应高于随机基线")
        ax.set_title(title_c, fontsize=11.2, fontweight="bold", loc="left", pad=8)
        ax.text(0.02, 0.97,
                "量子两臂都停在基线附近 → 结构有状态通路，\n但本预算下没证明它被用上（不等于不能）",
                transform=ax.transAxes, va="top", fontsize=8.9, color=C_BAD,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        ax.grid(alpha=0.22, axis="y")

        # (d) 成本：单步耗时随序列长度
        ax = axes[1][1]
        timing = report["timing"]
        tvals = sorted(int(k.split("=")[1]) for k in timing)
        for n in ("A0_quantum_feedback", "CTRL_classical_feedback"):
            ys = [timing[f"T={t}"][n] for t in tvals]
            ax.plot(tvals, ys, "o-", lw=2, color=colors[n], label=short[n])
        ax.set_yscale("log")
        ax.set_xlabel("序列长度 T（每个时间步一次量子线路）", fontsize=9.5)
        ax.set_ylabel("单步前向+反向耗时（秒，对数轴）", fontsize=9.5)
        ax.set_title("(d) 成本随序列长度线性上升", fontsize=11.2,
                     fontweight="bold", loc="left", pad=8)
        ax.legend(fontsize=9, frameon=False, loc="upper left")
        ax.grid(alpha=0.22)
        proj = report["cost_projection"]
        ax.text(0.985, 0.04,
                f"T=8 外推 2000 步：量子 {proj['quantum_hours_per_seed']:.2f} h/seed",
                transform=ax.transAxes, fontsize=9.2, color=C_BAD,
                ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))

        passed = sum(1 for k, v in verdict["checks"].items()
                     if k.startswith("P") and v)
        fig.suptitle("L1 量子循环 pilot：结构契约过关，但状态还没被证明用上",
                     fontsize=15.2, fontweight="bold", x=0.012, ha="left", y=0.978)
        fig.text(0.012, 0.930,
                 f"短预算装置标定：{report['config']['seeds']} seed × "
                 f"{report['config']['steps']} 步，单进程；P1–P4 通过 {passed}/4。"
                 "关反馈后前缀敏感度与跨步梯度严格归零；但延迟回忆上量子两臂仍近随机基线。",
                 fontsize=9.3, color="#555555", ha="left")
        fig.text(0.012, 0.018,
                 "数字为 2 seed 均值示意（独立重复单位 n=2，未报 SD/CI）；延迟回忆不是 held-out 泛化。"
                 "状态仍是经典数值（L1 测量-反馈），L2 隐量子态延续未实现。",
                 fontsize=8.9, color="#555555", ha="left")
        fig.subplots_adjust(left=0.078, right=0.975, top=0.875, bottom=0.115,
                            hspace=0.46, wspace=0.27)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------
def _write(report: dict, out_dir: Path, make_fig: bool) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"{OUT_NAME}.json"
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                  encoding="utf-8")
    fp = None
    if make_fig:
        fp = out_dir / "figures" / FIG_NAME if out_dir == DOCS else out_dir / FIG_NAME
        _figure(report, fp)
    return jp, fp


def run(args) -> dict:
    cfg = dict(
        seeds=args.seeds, steps=args.steps, seq_len=args.seq_len,
        batch=args.batch, lr=args.lr, recall_steps=args.recall_steps,
        n_qubits=4, timing_repeats=args.timing_repeats,
    )
    train_text, val_text = load_independent_split()
    train_data, vocab_size, stoi, _ = build_char_data(train_text)
    val_data = torch.tensor([stoi.get(ch, 0) for ch in val_text],
                            dtype=torch.long)
    overlap = corpus_overlap(train_text, val_text)
    cfg["vocab_size"] = int(vocab_size)
    cfg["train_chars"] = len(train_text)
    cfg["val_chars"] = len(val_text)
    cfg["val_context_k4_coverage"] = float(
        overlap["val_context_covered_by_train"]["4"]["val_token_coverage"])

    print(f"[语料] train={len(train_text)} val={len(val_text)} V={vocab_size} "
          f"k=4 覆盖={cfg['val_context_k4_coverage']:.4f}", flush=True)

    # 0) 消融单变量硬检查
    matched = assert_ablation_matched(vocab_size, seed=0)
    print("[断言] A0/A1 与经典两臂的初始权重逐元素相同（只差 feedback）", flush=True)

    # 1) P1–P4 契约
    contracts = {}
    for name in ARMS:
        contracts[name] = check_contracts(
            build_arm(name, vocab_size, 0), vocab_size, 0,
            seq_len=cfg["seq_len"], batch=cfg["batch"])
        print(f"[契约] {short_name(name)}: "
              f"P1={contracts[name]['P1_prefix_logits_mean_abs_delta']:.3e} "
              f"P2grad={contracts[name]['P2_state_grad_norm']:.3e} "
              f"P3={contracts[name]['P3_future_leakage_max_abs']:.1e}",
              flush=True)

    # 2) 计时（放在训练前，避免训练后缓存影响）
    timing = {}
    for t in args.timing_lengths.split(","):
        t = int(t)
        timing[f"T={t}"] = _timing(vocab_size, t, cfg["batch"],
                                   cfg["timing_repeats"])
        print(f"[计时] T={t}: 量子 {timing[f'T={t}']['A0_quantum_feedback']:.4f}s "
              f"经典 {timing[f'T={t}']['CTRL_classical_feedback']:.4f}s", flush=True)

    # 3) 短训练：字符语料
    char_per_arm = {}
    for name in ARMS:
        runs = []
        for seed in range(cfg["seeds"]):
            gen = iter_batches(train_data, seq_len=cfg["seq_len"],
                               batch_size=cfg["batch"], seed=1000 + seed)
            ev = _char_eval_factory(val_data, vocab_size, cfg["seq_len"])
            runs.append(_train_one(name, vocab_size, seed, cfg["steps"],
                                   cfg["seq_len"], cfg["batch"], cfg["lr"],
                                   gen, ev, log_every=max(1, cfg["steps"] // 6)))
        char_per_arm[name] = runs
        print(f"[训练/字符] {short_name(name)}: "
              f"val_loss={np.mean([r['eval']['val_loss'] for r in runs]):.4f} "
              f"{runs[0]['seconds_per_step']:.4f}s/步", flush=True)

    # 4) 短训练：延迟回忆（装置有效性）
    K, PAD_LEN = args.recall_k, args.recall_pad
    recall_per_arm = {}
    for name in ARMS:
        runs = []
        for seed in range(cfg["seeds"]):
            gen, rv, rl, _ = _delayed_recall_batches(
                K, PAD_LEN, seed=2000 + seed, batch=cfg["batch"])
            ev = _recall_eval_factory(K, PAD_LEN, rv)
            runs.append(_train_one(name, rv, seed, cfg["recall_steps"],
                                   rl, cfg["batch"], cfg["lr"], gen, ev,
                                   log_every=max(1, cfg["recall_steps"] // 6),
                                   task="delayed_recall"))
        recall_per_arm[name] = runs
        print(f"[训练/回忆] {short_name(name)}: "
              f"末位acc={np.mean([r['eval']['last_token_accuracy'] for r in runs]):.3f}",
              flush=True)

    t8 = timing.get("T=8") or timing[sorted(timing)[len(timing) // 2]]
    proj = {
        "quantum_seconds_per_step_at_T8": t8["A0_quantum_feedback"],
        "classical_seconds_per_step_at_T8": t8["CTRL_classical_feedback"],
        "quantum_hours_per_seed": t8["A0_quantum_feedback"] * 2000 / 3600,
        "classical_hours_per_seed": t8["CTRL_classical_feedback"] * 2000 / 3600,
        "projection_steps": 2000,
        "note": "按计时探针单步耗时线性外推；正式实验还需 ≥5 seed 与更长步数。",
    }
    report = {
        "task": "L1 recurrent pilot: architecture contracts, state ablation, cost",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg,
        "corpus": {
            "name": "independent_split (字符级，句级 disjoint，零 UNK)",
            "val_context_k4_coverage": cfg["val_context_k4_coverage"],
            "note": "报验证集泛化必须带 k=4 上下文重叠率（项目既有约定）。",
        },
        "ablation_matched_check": matched,
        "contracts": contracts,
        "timing": timing,
        "char_lm": {"per_arm": char_per_arm},
        "delayed_recall": {
            "per_arm": recall_per_arm,
            "k": K, "pad_len": PAD_LEN,
            "random_baseline": 1.0 / K,
            "note": "只在末位监督复现开头 token；无状态模型理论上无法超过 1/k。",
        },
        "cost_projection": proj,
        "runtime": {
            "single_process": True,
            "omp_mkl_threads": 1,
            "peak_rss_mb": _peak_rss_mb(),
        },
        "scope": "探索性短预算 pilot；不是性能对照实验",
    }
    report["verdict"] = build_verdict(report)
    return report


def short_name(name: str) -> str:
    return ARM_LABEL[name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--recall-steps", type=int, default=120)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--recall-k", type=int, default=4)
    ap.add_argument("--recall-pad", type=int, default=4)
    ap.add_argument("--timing-repeats", type=int, default=4)
    ap.add_argument("--timing-lengths", default="4,8,16")
    ap.add_argument("--quick", action="store_true",
                    help="极小规模冒烟，产物写 results/ 不覆盖 docs/")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--out", default=None,
                    help="产物目录；--quick 默认 results/，正式默认 docs/")
    ap.add_argument("--redraw-from", default=None,
                    help="只从既有 JSON 重画图，不重跑训练")
    ap.add_argument("--recompute-from", default=None,
                    help="从既有 JSON 重算 verdict 并重画图，不重跑训练")
    ap.add_argument("--probe-memory", action="store_true",
                    help="只复测单进程峰值工作集并打印 JSON，不训练、不写文件")
    args = ap.parse_args()

    if args.probe_memory:
        print(json.dumps(build_memory_probe(), ensure_ascii=False, indent=2))
        return

    if args.redraw_from or args.recompute_from:
        raw = args.recompute_from or args.redraw_from
        src = Path(raw)
        if not src.is_absolute():
            src = ROOT / src
        report = json.loads(src.read_text(encoding="utf-8"))
        if args.recompute_from:
            report["verdict"] = build_verdict(report)
            src.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
            print(f"  verdict recomputed: {src}", flush=True)
        # --out 可把重画结果隔离到临时目录，避免误覆盖正式图。
        fig = (Path(args.out) / "figures" / FIG_NAME if args.out
               else DOCS / "figures" / FIG_NAME)
        _figure(report, fig)
        print(f"  figure: {fig}", flush=True)
        return

    if args.quick:
        args.seeds, args.steps, args.recall_steps = 1, 4, 4
        args.seq_len, args.batch = 4, 4
        args.timing_repeats, args.timing_lengths = 2, "4,8"
        args.recall_pad = 3
    out_dir = Path(args.out) if args.out else (RESULTS if args.quick else DOCS)
    report = run(args)
    jp, fp = _write(report, out_dir, make_fig=not args.no_figure)
    v = report["verdict"]
    print(f"  archived: {jp}", flush=True)
    if fp:
        print(f"  figure:   {fp}", flush=True)
    print(f"  P1–P4 通过: {sum(1 for k, x in v['checks'].items() if k.startswith('P') and x)}/4"
          f" | T=8 外推 {report['cost_projection']['quantum_hours_per_seed']:.2f} h/seed",
          flush=True)
    print(f"  {v['status']}", flush=True)


if __name__ == "__main__":
    main()
