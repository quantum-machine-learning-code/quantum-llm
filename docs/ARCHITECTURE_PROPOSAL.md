# 架构改造方案：让量子部分真正参与自回归

> 2026-11-16 提出，2026-11-17 补第 3 步结果。状态：**方案 + 可运行 L1 原型
> （`qml_lab/recurrent.py`）+ 短预算 pilot（`experiments/recurrent_pilot.py`）**。
> 未做长训练、未做效果对比、未改任何既有归档结论。
> 本文只回答「结构上有无资格」，不回答「量子是否更好」。
> pilot 结论：P1–P4 结构契约 4/4 通过，但状态的功能有效性尚未证明（§16）。

---

## 1. 为什么要改：现状的三个结构缺口

现有 `RichQuantumLM`（`qml_lab/models.py`）的数据流是：

```
Embedding(117,16) → 因果 Conv1d(16→4, k=4) → tanh·π/2
   → VQC: 每个 (B*T, 4) 行独立跑一次 4-qubit 线路 → 48 个 Pauli 期望
   → Linear(48,48) → GELU → Linear(48,117) → 答案位 CE
```

1. **量子层逐位置独立**：`bottleneck_features()` 把 `(B,T,4)` reshape 成
   `(B*T,4)` 一次送进 QNode，不接收上一步状态、不向下一步传递状态、
   不跨位置纠缠。序列信息全在那层因果 Conv1d 里混合完，量子部分只看到
   一个已经压好的 4 维向量。
2. **自回归链条不经过量子系统**：`qml_lab/data.py:163` 的 `targets = seq[1:]`、
   `qml_lab/train.py` 的 teacher forcing 与 `greedy_sample` 都在经典侧。
3. **课题一甚至没用生成**：`arithmetic.py:answer_logits` 是
   `model(x)[:, -1, :112]`，把 `[a][+][b][=]` 四个 token 一次性喂进去，
   只监督答案位一次分类。它是查表／分类任务，套了自回归模型的壳。

因此现有实验无论是 0.1817 还是 0.1865，测的都是
「两种逐位置非线性变换在一次性分类上的差别」。**加 seed、加 Holm 校正、
把 spec 对齐，都改变不了这一点。**

## 2. 目标判据：什么才算改到位

新架构必须同时满足四条可验证条件，缺一条就不算「量子 LLM」：

| # | 判据 | 验证方式 |
|---|---|---|
| P1 | 位置 `t` 的量子线路输入包含 `t` 之前的信息 | 扰动词序／前缀，答案位 logits 必须改变 |
| P2 | 上一步状态参与当前线路演化，且梯度能跨步回传 | `state.grad` 非空；`feedback=False` 时性能显著下降 |
| P3 | 严格因果：位置 `t` 对 `idx[:, t+1:]` 的梯度恒为 0 | 单测断言 |
| P4 | 推理时逐个 token 推进状态，不是整段一次前向 | `generate()` 走 `step()`，给出长度正确的续写 |

现有 `RichQuantumLM` 四条全不满足（P3 由 Conv1d 左填充满足，但那是经典侧）。

## 3. 三个层级，别把 L1 说成 L2

| 层级 | 状态载体 | 量子态是否跨时间步延续 | 实现成本 |
|---|---|---|---|
| **L0 现状** | 无 | 否，每位置独立重制备 | 已有 |
| **L1 测量反馈型量子循环** | 上一步的 `expval(Z)`（经典数值）重上传为旋转角 | 否，但状态由量子测量产生、经量子线路演化，梯度与信息流都穿过量子系统 | 已实现原型 |
| **L2 隐量子态延续** | 保留一组隐 qubit **不测量**，只在输出 qubit 上测 | 是 | 需要更多 qubit、部分测量、更复杂训练 |

L1 是文献里 quantum RNN / quantum reservoir 的标准做法之一（测量–反馈），
它诚实地回答了「自回归链条是否经过量子系统」，但**状态本身以经典数值保存**。
要主张「量子态承载记忆」，必须做 L2，本文第 7 节给出路线，但不宣称已实现。

## 4. L1 的具体实现（已落地：`qml_lab/recurrent.py`）

每步 `t`：

```text
h_t     = tanh(W_in · Embedding(idx_t)) · π/2         # (B,Q) 经典输入编码
s_ang_t = s_{t-1} · π/2                                # 上一步测量期望重上传
s_t     = QNode[ RY(h_t) → RZ(s_ang_t) → StronglyEntanglingLayers(w)
                 → expval(Z_i) ]                        # (B,Q)，新状态 + logits 输入
logits_t = head(s_t)
```

设计要点：

- **状态用 `RZ` 反馈而不是 `RY`**：`RY(a)·RY(b) = RY(a+b)`，角度相加只是线性
  偏置；`RZ` 与输入 `RY` 不交换，后续纠缠层才能把相位信息转成可测的 Z 期望。
  已实测：把上一步状态改成全零，输出平均变化 0.15，反馈确实进入线路。
- **`s_t` 不 detach**：反向传播穿过量子线路回到前一步，量子层参与序列依赖的
  优化（BPTT）。内存随 `T` 线性增长，提供 `detach_state=True` 做截断 BPTT。
- **批量 QNode**：PennyLane 0.45 下 `x_ang/s_ang` 为 `(B,Q)`、`weights` 为
  `(L,Q,3)`，输出 `(B,Q)`；梯度对输入、状态、权重三者都非空。
  注意 `qml.math.stack` 在 QNode 内会退回 `pennylane.numpy.tensor` 并破坏
  torch autograd，必须在 QNode **外部** `torch.stack(...)`。
- **`feedback=False` 消融**：状态角度置零，模型退化为 L0 的逐位置独立变换，
  用来验证 P2；经典替身 `ClassicalRecurrentLM` 用 `nn.RNNCell` 保持同接口。

## 5. 任务与评估接口必须一起换

`answer_logits()` 那种「整段一次前向只取答案位」的口径要停用。改为：

- **训练**：标准下一 token 预测，`logits = model(seq[:, :-1])` 对
  `targets = seq[:, 1:]` 展平算 CE；teacher forcing。
- **评估**：`generate(prefix, n_new)` 逐 token 滚动，报续写准确率／PPL；
  验证集需给与训练集在 `k = 上下文窗口` 的 n-gram 重叠率（项目既有约定）。
- **任务选择**：模 112 加法的答案只有 1 个 token，**不适合**检验序列依赖。
  建议用 (a) 现有字符语料（独立切分版本），或 (b) 合成的多步算术／括号匹配
  等需要跨步记忆的任务，并给出「去掉状态反馈后性能下降」的装置有效性证明。

## 6. 消融设计（下任何结论之前）

| 消融 | 改动 | 回答什么 |
|---|---|---|
| A0 | 量子循环（`feedback=True`） | 基准装置 |
| A1 | 同一模型 `feedback=False` | 状态反馈是否真的被用上（P2） |
| A2 | 线路换成 `ClassicalRecurrentLM`（同状态维、同 head、同训练预算） | 量子层是否带来结构差异 |
| A3 | 容量匹配：量子 `n_layers` 递增到参数量与经典 RNN cell 相等 | 12 vs 240 那类容量混杂 |
| A4 | 状态维度 Q ∈ {2,4,6}、上下文长度 T 扫描 | 记忆容量与成本曲线 |

硬约束沿用项目技能 `qml-experiment-guardrails`：单变量、≥5 seed、配对 CI、
≥10 检验报 Holm、两臂同 lr/wd/步数/seed/切分、报差值前先过装置有效性门槛。
**但顺序是：先证明 P1–P4，再谈任何差值。**

## 7. L2 路线（未实现，需先确认方向）

把 `Q` 个 qubit 分成隐寄存器 `H` 与输出寄存器 `O`：

```text
每步：RY(输入) 作用于 H → 参数化纠缠 U(w) 耦合 H 与 O
      → 只在 O 上测 expval（得到 logits）→ H 保持不测量、延续到下一步
```

这样量子态本身跨时间步演化，不经过经典数值中转。代价：

- 需要 `|H| + |O|` 个 qubit 且状态向量模拟内存按 `2^(|H|+|O|)` 增长；
  本机内存常紧（典型可用 8–9 GB / 31 GB），`|H|+|O| ≤ 8` 比较现实。
- 部分测量 + 态延续在 PennyLane 里要靠 `qml.measure` / mid-circuit
  measurement 或自定义 tape，不能再用现成的 `TorchLayer` 一次性批处理。
- 梯度需用参数偏移或 `backprop` + 保留计算图，训练成本显著高于 L1。

## 8. 成本、内存与风险

- 现有量子臂 2000 步约 **305 s/seed**；L1 每个时间步一次 QNode 调用，
  `T=4` 时约 4 次/forward（每次 batch B，而非一次 `B*T`），预计单步成本
  上升 2–4 倍。正式实验必须先用短预算 pilot 计时再定步数。
- 内存：多进程 worker 每个约 420 MB。本机约束下 `workers ≤ 4`、
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`，不要与其它训练任务叠加。
- BPTT 计算图随 T 增长；`detach_state=True` 可截断，但会削弱 P2 的梯度证据，
  需单独报告。
- 仍是 `default.qubit` 解析状态向量、无 shots、无噪声、无真机。
  shots/噪声与 checkpoint → QASM/QIR 部署属于后续独立工作。

## 9. 对既有结论的处理

`docs/REVIEW.md` §14 已把历史 `GROUP_SPECS` 的「量子 vs 经典」全部降级为
「不同配置之间的观察」；`matched_v1/v2/v3` 三版装置诊断见
`docs/matched_calibration.json`，容量匹配 pilot 见
`docs/capacity_matched_pilot.json`。

**L1/L2 上线后，这些实验不升级、不重新解释，统一标为「经典瓶颈基线」**：
它们测的是逐位置变换 + 一次性分类，与新架构不在同一个问题空间。

## 10. 分阶段计划

1. ✅ 诊断现状（P1–P4 全不满足）、写清层级与判据。
2. ✅ L1 原型 + 单测（前向/反向/因果性/状态敏感度/逐步生成）。
3. ✅ 短预算 pilot（2026-11-17，`experiments/recurrent_pilot.py`）：
   P1–P4 结构契约 4/4 通过、关反馈后跨步梯度严格归零、成本与内存已测。
   **但延迟回忆上量子两臂仍停在随机基线，状态的功能有效性未证明。**
   详见 `docs/REVIEW.md` §16 与 `docs/recurrent_pilot.json`。
4. ⬜ 先解决第 3 步暴露的功能有效性问题（延长训练 / 扫状态维 Q、
   线路深度、lr；用延迟回忆当装置门槛），**通过之后**才做 A2/A3 同构对照
   （≥5 seed、配对 CI、Holm）。不通过就报「装置无效，不得比较」。
5. ⬜ 确认 L1 稳定后，评估是否投入 L2（qubit 数、内存、mid-circuit 测量）。
6. ⬜ shots/噪声、checkpoint → QASM/QIR、真机部署调研。

第 3 步是单进程短预算（2 seed × 120 步，峰值 333 MB，约 3 分钟），
不构成长训练。第 4 步起都是长训练，需另行确认预算与内存窗口，不擅自启动。
`T=8` 外推 2000 步约 `0.11 h/seed`（量子），正式实验还要 ≥5 seed。
