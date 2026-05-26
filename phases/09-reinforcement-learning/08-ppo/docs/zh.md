# 近端策略优化（Proximal Policy Optimization, PPO）

> A2C（Advantage Actor-Critic）在每次参数更新后都会丢弃当次轨迹采样（rollout）。PPO 将策略梯度（policy gradient）包裹在裁剪重要性比率（clipped importance ratio）中，使得你可以在同一批数据上进行 10 个以上的训练轮次（epoch），而不会导致策略发散。Schulman 等人（2017）。截至 2026 年，它仍然是默认的策略梯度算法。

**类型：** 构建
**语言：** Python
**前置知识：** 第 9 阶段 · 06（REINFORCE），第 9 阶段 · 07（Actor-Critic）
**预计耗时：** 约 75 分钟

## 核心问题

A2C（第 07 课）属于同策略（on-policy）算法：其梯度 `E_{π_θ}[A · ∇ log π_θ]` 要求数据必须从*当前*策略 `π_θ` 中采样。执行一次更新后，`π_θ` 发生变化；你之前使用的数据就变成了异策略（off-policy）数据。如果重复使用这些数据，计算出的梯度将产生偏差。

轨迹采样的成本很高。在 Atari 环境中，一次覆盖 8 个环境 × 128 步的采样会生成 1024 个状态转移（transition），并消耗十几秒的环境运行时间。仅执行一次梯度更新就将其丢弃是一种极大的浪费。

信任区域策略优化（Trust Region Policy Optimization, TRPO，Schulman 2015）是首个解决方案：它约束每次更新，使新旧策略之间的 KL 散度（KL divergence）保持在 `δ` 以下。该算法理论严谨，但每次更新都需要求解共轭梯度（conjugate-gradient）。到了 2026 年，已经没有人再使用 TRPO 了。

PPO（Schulman 等人，2017）用简单的裁剪目标函数（clipped objective）替代了严格的信任区域约束。只需多写一行代码。每次采样可进行十个训练轮次。无需共轭梯度求解。具备足够好的理论保证。九年后的今天，它依然是从 MuJoCo 到基于人类反馈的强化学习（RLHF）等各类任务中默认的策略梯度算法。

## 核心概念

![PPO 裁剪替代目标（Clipped Surrogate Objective）：在 1 ± ε 处进行比率裁剪](../assets/ppo.svg)

**重要性比率（Importance Ratio）。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略（Policy）与收集数据所用旧策略的似然比（Likelihood Ratio）。`r_t = 1` 表示策略未发生变化。`r_t = 2` 表示新策略采取动作 `a_t` 的概率是旧策略的两倍。

**裁剪替代目标（Clipped Surrogate Objective）。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

包含两项：

- 若优势（Advantage）`A_t > 0` 且比率试图超过 `1 + ε`，裁剪操作会使梯度变平——不要将优质动作的概率推高至超过旧概率 `+ε` 的范围。
- 若优势 `A_t < 0` 且比率试图超过 `1 - ε`（意味着相比裁剪后的限制，我们反而会让劣质动作的概率上升），裁剪会截断梯度——避免将劣质动作的概率过度降低（即不要使其偏离旧概率超过 `-ε`）。

`min` 函数负责处理另一侧的情况：若比率朝着*有益*的方向移动，你仍能获得梯度（在不会损害性能的一侧不进行裁剪）。

通常取 `ε = 0.2`。将目标函数绘制为 `r_t` 的函数时，会呈现为一个分段线性函数：在“有益侧”具有平坦的顶部（上限），在“有害侧”具有平坦的底部（下限）。

**完整的 PPO 损失函数（PPO Loss）。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

采用与 A2C 相同的演员-评论家（Actor-Critic）架构。包含三个系数，通常取值为 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**训练循环（Training Loop）。**

1. 在 `N` 个并行环境（Parallel Environments）中各运行 `T` 步，共收集 `N × T` 个状态转移（Transitions）。
2. 计算优势（使用 GAE 算法），并将其冻结为常量。
3. 将 `π_{θ_old}` 冻结为当前 `π_θ` 的快照。
4. 进行 `K` 个训练轮次（Epochs），针对每个包含 `(s, a, A, V_target, log π_old(a|s))` 的小批量数据（Minibatch）：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` 损失 + 价值损失（Value Loss） + 熵正则化（Entropy Regularization）。
   - 执行梯度更新（Gradient Step）。
5. 丢弃本轮轨迹数据（Rollout）。返回步骤 1。

`K = 10` 且小批量大小为 64 是标准的超参数（Hyperparameter）配置。PPO 具有很强的鲁棒性（Robustness）：具体数值在 ±50% 范围内波动通常影响不大。

**KL 惩罚变体（KL-Penalty Variant）。** 原始论文提出了一种使用自适应 KL 散度（Kullback-Leibler Divergence）惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)`，其中 `β` 会根据观测到的 KL 值进行调整。裁剪版本后来成为主流；而 KL 变体则在人类反馈强化学习（Reinforcement Learning from Human Feedback, RLHF）中得以保留（因为在该场景下，与参考策略的 KL 散度本身就是一个始终需要的独立约束条件）。

## 动手实现

### 步骤 1：在轨迹收集（rollout）阶段捕获 `log π_old(a | s)`

for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next

该快照仅在轨迹收集（rollout）阶段采集一次。在后续更新轮次（epochs）期间保持不变。

### 步骤 2：计算广义优势估计（Generalized Advantage Estimation, GAE）优势函数（第 07 课）

与优势演员-评论家（Advantage Actor-Critic, A2C）算法相同。在批次（batch）维度上进行归一化。

### 步骤 3：截断代理目标（clipped surrogate update）更新

for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]

“截断 → 零梯度”模式是近端策略优化（Proximal Policy Optimization, PPO）的核心。如果新策略在有益方向上已经偏离过远，更新就会停止。

### 步骤 4：价值与熵

为评论家网络（Critic）目标添加标准的均方误差（Mean Squared Error, MSE）损失，并为演员网络（Actor）添加熵奖励（entropy bonus），与 A2C 相同。

### 步骤 5：诊断指标

每次更新需关注以下三项指标：

- **平均 KL 散度（Mean KL）** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]` 范围内。若超过 `0.1`，请降低 `K_EPOCHS` 或 `LR`（学习率）。
- **截断比例（Clip fraction）** —— 比率（ratio）落在 `[1-ε, 1+ε]` 范围之外的样本比例。理想值约为 `~0.1-0.3`。若接近 `~0`，说明截断机制从未触发 → 提高 `LR` 或 `K_EPOCHS`。若达到 `~0.5+`，说明模型对当前轨迹过拟合 → 降低它们。
- **解释方差（Explained variance）** `1 - Var(V_target - V_pred) / Var(V_target)`。用于衡量 Critic 质量的指标。随着 Critic 的学习，该值应逐渐趋近于 1。

## 常见陷阱

- **裁剪系数（Clip coefficient）调优不当。** `ε = 0.2` 是事实上的标准。降至 `0.1` 会导致策略更新过于保守；而 `0.3+` 则容易引发不稳定。
- **训练轮数（Epochs）过多。** 当 `K > 20` 时，策略通常会偏离 `π_old` 过远，从而导致训练不稳定。应限制训练轮数，尤其是对于大型网络。
- **未进行奖励归一化（Reward normalization）。** 过大的奖励尺度会侵蚀裁剪范围。在计算优势函数（Advantage）前，应对奖励进行归一化（使用滑动标准差）。
- **忘记优势归一化（Advantage normalization）。** 按批次进行零均值/单位标准差归一化是标准做法。跳过这一步会在大多数基准测试中导致 PPO 失效。
- **学习率（Learning rate）未衰减。** PPO 通常受益于线性衰减至零的学习率。固定学习率的效果往往更差。
- **重要性比率（Importance ratio）计算错误。** 为保证数值稳定性，应始终使用 `exp(log_new - log_old)`，而非 `new / old`。
- **梯度符号错误。** 最大化代理目标函数等价于*最小化* `-L^{CLIP}`。符号弄反是 PPO 中最常见的 Bug。

## 实际应用

在令人惊讶的众多领域中，PPO 已成为 2026 年默认的强化学习（Reinforcement Learning, RL）算法：

| 应用场景 | PPO 变体 |
|----------|-------------|
| MuJoCo / 机器人控制 | 采用高斯策略（Gaussian policy）的 PPO，GAE(0.95) |
| Atari / 离散动作游戏 | 采用分类策略（Categorical policy）的 PPO，滚动 128 步轨迹（Rollouts） |
| 大语言模型（LLMs）的 RLHF | 对参考模型施加 KL 惩罚（KL penalty）的 PPO，在回复末尾使用奖励模型（Reward Model, RM）计算奖励 |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar, OpenAI Five） |
| 推理型大语言模型 | GRPO（第 12 课）—— 无价值网络（Critic）的 PPO 变体 |
| 仅偏好数据 | DPO —— PPO+KL 的闭式坍缩形式，无需在线采样 |

PPO 的*损失函数结构*（裁剪代理目标 + 价值损失 + 熵正则）构成了 DPO、GRPO 以及几乎所有 RLHF 流程的基础框架。

## 交付使用

保存为 `outputs/skill-ppo-trainer.md`：

---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.

## 练习

1. **简单。** 在 4×4 GridWorld 上运行 PPO，设置 `ε=0.2, K=4`。在相同的环境步数下，将其样本效率（Sample efficiency）与 A2C（每次轨迹更新一轮）进行对比。
2. **中等。** 对 `K ∈ {1, 4, 10, 30}` 进行参数扫描（Sweep）。绘制累积回报（Return）随环境步数变化的曲线，并跟踪每次更新的平均 KL 散度（KL divergence）。在该任务中，`K` 达到多少时 KL 会急剧发散？
3. **困难。** 将裁剪代理目标替换为自适应 KL 惩罚（若 `KL > 2·target` 则 `β` 翻倍，若 `KL < target/2` 则 `β` 减半）。对比最终回报、训练稳定性以及未触发裁剪的比例（Clip-free-ness）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 重要性比率 (Importance Ratio) | "r_t(θ)" | `π_θ(a|s) / π_old(a|s)`；衡量当前策略与收集数据所用旧策略之间的偏离程度。 |
| 截断代理目标 (Clipped Surrogate) | “PPO 的核心技巧” | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在截断边界之外且对策略更新有利的一侧，梯度保持平坦。 |
| 信任域 (Trust Region) | “TRPO / PPO 的设计初衷” | 限制每次更新的 KL 散度 (KL Divergence)，以保证策略单调改进。 |
| KL 惩罚 (KL Penalty) | “软信任域” | PPO 的替代形式：`L - β · KL(π_θ || π_old)`。其中 `β` 可自适应调整。 |
| 截断比例 (Clip Fraction) | “截断机制触发的频率” | 诊断指标——理想值应在 0.1-0.3 之间；超出此范围通常意味着超参数未调优。 |
| 多轮次训练 (Multi-epoch Training) | “数据复用” | 对每次轨迹采样 (Rollout) 的数据进行 K 个训练轮次 (Epoch) 的训练；以方差增大为代价换取样本效率的提升。 |
| 近似同策略 (On-policy-ish) | “基本属于同策略” | PPO 名义上是同策略 (On-policy) 算法，但当 K>1 时，会安全地利用轻微偏离策略 (Off-policy) 的数据。 |
| PPO-KL | “另一种 PPO” | 采用 KL 惩罚的变体；常用于 RLHF (基于人类反馈的强化学习)，因为其中相对于参考模型的 KL 散度本身就是一个约束条件。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) —— PPO 的原始论文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) —— TRPO，PPO 的前身算法。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) —— 对 PPO 的每个超参数进行了消融实验 (Ablation Study)。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) —— InstructGPT 论文；详细阐述了在 RLHF 中使用 PPO 的训练流程。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) —— 基于 PyTorch 的清晰现代版算法讲解。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) —— 被众多论文引用的单文件 PPO 参考实现。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) —— 面向大语言模型的 PPO 生产级训练方案；建议结合第 09 课（RLHF）一同阅读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) —— 著名的“37 项代码级优化”论文；剖析了 PPO 中哪些技巧是核心支撑，哪些仅是经验之谈。