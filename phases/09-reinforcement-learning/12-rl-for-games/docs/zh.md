# 游戏中的强化学习（Reinforcement Learning）— AlphaZero、MuZero 与大语言模型（Large Language Model）推理时代

> 1992年：TD-Gammon 凭借纯粹的时序差分（Temporal Difference）算法在双陆棋中击败人类冠军。2016年：AlphaGo 击败李世石。2017年：AlphaZero 从零开始称霸国际象棋、将棋和围棋。2024年：DeepSeek-R1 证明了同一套范式依然有效，仅需用组相对策略优化（Group Relative Policy Optimization）替代近端策略优化（Proximal Policy Optimization），即可应用于推理任务。游戏是推动本阶段每一次突破的基准。

**Type:** 构建
**Languages:** Python
**Prerequisites:** 第9阶段 · 05（深度Q网络, DQN）、第9阶段 · 08（近端策略优化, PPO）、第9阶段 · 09（基于人类反馈的强化学习, RLHF）、第9阶段 · 10（多智能体强化学习, MARL）
**Time:** 约120分钟

## 核心问题

游戏具备强化学习所需的一切要素。清晰的奖励信号（胜/负）。无限的回合数（自我对弈（Self-Play）可重置状态）。完美的仿真环境（游戏本身*就是*模拟器）。离散或低维连续的动作空间。以及强制要求对抗鲁棒性的多智能体结构。

而每一项重大的强化学习突破，也都是通过游戏进行验证的。TD-Gammon（双陆棋，1992年）。Atari-DQN（2013年）。AlphaGo（2016年）。AlphaZero（2017年）。OpenAI Five（Dota 2，2019年）。AlphaStar（星际争霸II，2019年）。MuZero（学习模型，2019年）。AlphaTensor（矩阵乘法，2022年）。AlphaDev（排序算法，2023年）。DeepSeek-R1（数学推理，2025年）——这是最新证明游戏强化学习技术可应用于文本领域的案例。

本综合实践将通过一个统一的视角来剖析三大里程碑式架构——AlphaZero、MuZero 与 GRPO：**自我对弈 + 搜索 + 策略改进**。每一代架构都是对前一代的泛化与扩展；特别是 GRPO，它本质上是将 AlphaZero 的成功范式应用于大语言模型推理任务中，以词元（Token）作为动作，以数学验证作为获胜信号。

## 核心概念

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../assets/rl-games.svg)

**统一的循环。**

while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output

**AlphaZero（2017）。** Silver 等人。针对规则已知的游戏（国际象棋、将棋、围棋）：

- 策略价值网络（Policy-Value Network）：单塔结构 `f_θ(s) → (p, v)`。`p` 是合法动作的先验概率分布。`v` 是预期的游戏结果。
- 蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）：在每一步，展开可能后续步骤的树。使用 `(p, v)` 作为先验与引导（bootstrap）。通过 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自我对弈（Self-play）：智能体之间进行对局。在第 `t` 步，MCTS 的访问分布 `π_t` 成为策略的训练目标。
- 损失函数（Loss）：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 为游戏结果（+1 / 0 / -1）。

零人类知识。零手工启发式规则。仅凭一套算法，在各自进行数千万局自我对弈后，便掌握了国际象棋、将棋和围棋。

**MuZero（2019）。** Schrittwieser 等人。移除了对规则必须已知的前提要求。

- 不再依赖固定环境，而是学习一个*潜在动力学模型（Latent Dynamics Model）* `(h, g, f)`：
  - `h(s)`：将观测编码为潜在状态。
  - `g(s_latent, a)`：预测下一个潜在状态与奖励。
  - `f(s_latent)`：预测策略先验与价值。
- MCTS 在*学习到的潜在空间*中运行。搜索机制与训练循环保持不变。
- 适用于围棋、国际象棋、将棋*以及* Atari 游戏——同一套算法，无需知晓规则。

**随机 MuZero（Stochastic MuZero，2022）。** 引入了随机动力学与机会节点（chance nodes）；扩展至双陆棋类游戏。

**Muesli 与 Gumbel MuZero（2022-2024）。** 在样本效率与确定性搜索方面进行了改进。

**GRPO（2024-2025）。** DeepSeek-R1 的训练方案。采用与 AlphaZero 结构相同的循环，应用于语言模型的推理任务：

- “游戏”：解答数学/编程/推理问题。“获胜”= 验证器（测试用例通过、数值答案匹配）返回 1。
- 策略（Policy）：大语言模型（LLM）。动作（Actions）：词元（tokens）。状态（State）：提示词（prompt）+ 已生成的回复。
- 无需价值网络（Critic，即 PPO 风格的 `V_φ`）。取而代之的是，针对每个提示词，从策略中采样 `G` 个续写结果。分别计算每个结果的奖励。使用**组相对优势（Group-Relative Advantage）** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- 引入相对于参考策略的 KL 散度惩罚（KL Penalty），以防止策略漂移（类似于 RLHF）。
- 完整损失函数：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

无需奖励模型（Reward Model）、无需价值网络、无需 MCTS。组相对基线（Group-Relative Baseline）一举替代了这三者。在推理基准测试中，其效果达到或超越了 PPO-RLHF 的质量，而计算成本仅为后者的一小部分。

**完整的 R1 训练方案。** DeepSeek-R1（DeepSeek 2025）在一篇论文中发布了两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基座模型出发。不进行监督微调（Supervised Fine-Tuning, SFT）。直接应用 GRPO，包含两项奖励成分：*准确率奖励*（基于规则——最终答案是否解析为正确数值/代码是否通过单元测试）与*格式奖励*（生成内容是否将思维链（Chain-of-Thought, CoT）包裹在 `<think>…</think>` 标签中）。经过数千步训练，平均回复长度从约 100 个词元增长至约 10,000 个词元，数学基准测试分数攀升至接近 o1-preview 的水平。模型从零开始学会了推理。缺点在于：其思维链往往难以阅读、混合多种语言，且缺乏风格上的打磨。
- **R1。** 通过四阶段流程解决 R1-Zero 的可读性问题：
  1. **冷启动监督微调（Cold-start SFT）。** 收集数千条格式规范的长思维链演示数据。在此数据上对基座模型进行监督微调。这提供了一个具备可读性的起点。
  2. **面向推理的 GRPO。** 应用 GRPO，在准确率与格式奖励的基础上，增加*语言一致性*奖励，以防止语码转换（code-switching）。
  3. **拒绝采样（Rejection Sampling） + 第二轮 SFT。** 从强化学习检查点中采样约 60 万条推理轨迹，仅保留最终答案正确且思维链可读的样本，并与约 20 万条非推理 SFT 样本（写作、问答、自我认知）合并。再次对基座模型进行微调。
  4. **全谱系 GRPO。** 再进行一轮强化学习，同时覆盖推理任务（基于规则的奖励）与通用对齐任务（基于有用性/无害性偏好的奖励）。

该模型在开放权重下，于 AIME 和 MATH-500 基准上的表现与 o1 持平，且规模足够小，便于进行知识蒸馏（Distillation）。同一篇论文还通过基于 R1 推理轨迹进行 SFT，发布了六个蒸馏后的稠密模型（从 Qwen-1.5B 到 Llama-70B）——学生模型端无需进行强化学习。在同等学生模型规模下，对强大的强化学习教师模型进行蒸馏，其效果始终优于从零开始进行强化学习。

**为何在推理任务中选择 GRPO 而非 PPO。** DeepSeekMath 论文（2024 年 2 月）给出了三个原因：（1）无需训练价值网络，内存占用减半；（2）组基线（Group Baseline）天然能够处理推理任务产生的稀疏轨迹末端奖励；（3）按提示词归一化（Per-Prompt Normalization）使得优势值在不同难度的问题之间具有可比性，这是 PPO 的单一价值网络无法做到的。

**无搜索 vs 基于搜索。** 游戏领域已出现分化：

- *长视距完全信息博弈*（围棋、国际象棋）：仍基于搜索。AlphaZero / MuZero 占据主导地位。
- *大语言模型推理*：生产环境中尚未引入 MCTS；采用 GRPO 进行完整轨迹展开（Rollout），推理阶段使用 Best-of-N 策略。过程奖励模型（Process Reward Models, PRMs）的出现暗示着逐步级搜索（Step-Level Search）可能会被重新引入。

## 动手实现

`code/main.py` 中的代码实现了 **GRPO（Group Relative Policy Optimization）的微型版本**——一个包含多组样本的多臂老虎机（multi-armed bandit）问题。该算法与大语言模型（LLM）上的完全一致，仅策略（policy）和环境（environment）更为简化。它重点讲解了*损失（loss）*和*组相对优势（group-relative advantage）*，后者是 2025 年的创新点。

### 步骤 1：构建微型验证器（verifier）环境

QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0

在真实的 GRPO 中，验证器会运行单元测试或检查数学等式。

### 步骤 2：策略（policy）：对每个提示词（prompt）的 K 个答案词元（token）进行 softmax 计算

def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])

这等价于大语言模型在给定提示词条件下的最后一层输出。

### 步骤 3：组采样与组相对优势

def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])

组相对优势是 DeepSeek 在 2024 年提出的技巧。该方法无需评论家网络（critic）。其“基线（baseline）”为组内均值，归一化则使用组内标准差。

### 步骤 4：与 REINFORCE 基线（无价值网络）进行对比

在相同的设置与计算量下，使用基础的 REINFORCE 算法进行对比。GRPO 的收敛速度更快且更稳定。

### 步骤 5：观察熵（entropy）与 KL 散度（KL divergence）

诊断指标与基于人类反馈的强化学习（RLHF）相同：相对于参考模型的平均 KL 散度、策略熵以及随时间变化的奖励（reward-over-time）。当这些指标趋于稳定时，训练即可结束。

## 常见陷阱

- **通过操纵验证器进行奖励欺骗（Reward hacking via verifier gaming）。** 组相对策略优化（GRPO）继承了基于人类反馈的强化学习（RLHF）的风险：如果验证器（verifier）存在缺陷或可被利用，大语言模型（LLM）就会找到漏洞。因此，构建鲁棒的验证器（如包含多组测试用例或形式化证明）至关重要。
- **分组规模过小。** 分组基线的方差与 `1/√G` 成正比。当 `G < 4` 时，优势信号（advantage signal）噪声较大；通常的标准选择是 `G = 8` 到 `64`。
- **长度偏差（Length bias）。** 不同长度的大语言模型（LLM）生成结果具有不同的对数概率（log-probabilities）。可通过 token 数量进行归一化，或使用序列级对数概率，亦或截断至最大长度。
- **纯自我对弈循环（Pure self-play cycles）。** AlphaZero 风格的训练在一般和博弈（general-sum games）中容易陷入策略支配循环（dominance loops）。可通过引入多样化的对手池（如联赛机制，参见第 10 课）来缓解该问题。
- **搜索与策略不匹配（Search-policy mismatch）。** AlphaZero 训练策略网络（policy network）以模仿搜索算法的输出。若策略网络容量过小，无法充分表征搜索产生的分布，训练便会陷入停滞。
- **算力门槛（Compute floor）。** MuZero / AlphaZero 需要海量计算资源。单次消融实验（ablation study）通常就需要消耗数百个 GPU 小时。为便于学习，社区已提供小型演示项目（例如在四子棋上运行的 AlphaZero）。
- **验证器覆盖率（Verifier coverage）。** 若单元测试未能拦截错误解法，反而会强化模型对该错误的记忆。应设计能够捕捉边界情况（edge cases）的验证器。

## 应用指南（Use It）

2026 年游戏强化学习（game-RL）领域格局，按应用场景划分：

| 领域 | 主流方法 |
|--------|-----------------|
| 双人零和棋类游戏（围棋、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 非完全信息卡牌游戏（扑克） | 反事实遗憾最小化（CFR）+ 深度学习（DeepStack, Libratus, Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略游戏（Dota、星际争霸） | 近端策略优化（PPO）+ 自我对弈 + 联赛机制（OpenAI Five, AlphaStar） |
| 大语言模型数学/代码推理 | 组相对策略优化（GRPO）（DeepSeek-R1, Qwen-RL 及开源复现版本） |
| 大语言模型对齐 | 直接偏好优化（DPO）/ RLHF-PPO（非 GRPO；验证器基于偏好而非可验证事实） |
| 机器人控制 | PPO + 域随机化（DR）（非游戏强化学习，但使用相同的策略梯度工具） |
| 组合优化问题 | AlphaZero 变体（AlphaTensor, AlphaDev） |

这套核心范式——自我对弈、搜索增强改进与策略蒸馏——已横跨文本生成、像素级游戏与物理控制领域。GRPO 是其中最新的应用实例，未来必将涌现更多衍生方法。

## 部署上线（Ship It）

保存为 `outputs/skill-game-rl-designer.md`：

---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO 多臂老虎机（GRPO Bandit）。使用 2 个提示词（prompt）× 每个 4 个答案词元（token）进行训练。在 `G=8` 的条件下，于 1,000 次更新内实现收敛。
2. **中等。** 接入 PPO（裁剪版）与原始 REINFORCE 算法。在同一老虎机任务上，对比它们与 GRPO 的样本效率（sample efficiency）和奖励方差（reward variance）。
3. **困难。** 扩展至长度为 2 的“推理链”（reasoning chain）：智能体（agent）输出两个词元，验证器（verifier）对这对词元进行奖励。评估 GRPO 如何处理两步序列中的信用分配（credit assignment）。（提示：按*完整序列*计算组优势（group advantage），并将其传播至两个词元位置。）

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| MCTS | “带学习网络的树搜索” | 蒙特卡洛树搜索（Monte Carlo Tree Search）；结合学习到的 `(p, v)` 先验进行 UCB1/PUCT 选择。 |
| AlphaZero | “自我对弈 + MCTS” | 策略价值网络（Policy-value Network），训练目标为拟合 MCTS 访问次数与游戏结果。 |
| MuZero | “学习模型的 AlphaZero” | 相同循环，但通过学习的动力学模型在隐空间（latent space）中进行。 |
| GRPO | “无评论家（Critic-free）的 PPO” | 组相对策略优化（Group Relative Policy Optimization）；带组均值基线（group-mean baseline）与 KL 散度的 REINFORCE 算法。 |
| PUCT | “AlphaZero 的 UCB” | `Q + c · p · √N / (1 + N_a)` —— 平衡价值估计与先验概率。 |
| Self-play | “智能体与过去的自己对战” | 零和博弈（zero-sum）的标准配置；提供对称的训练信号。 |
| League play | “基于群体的自我对弈” | 从过去版本、当前版本及利用型智能体（exploiters）中采样作为对手。 |
| Verifier reward | “可验证强化学习” | 奖励来源于确定性检查器（如测试通过、答案匹配）。 |
| Process reward | “PRM” | 对每个推理步骤进行评分，而非仅针对最终答案。 |

## 扩展阅读

- [Silver 等人 (2017)。无需人类知识掌握围棋（AlphaGo Zero）](https://www.nature.com/articles/nature24270)。
- [Silver 等人 (2018)。通过自我对弈（Self-play）掌握国际象棋、将棋和围棋的通用强化学习（Reinforcement Learning）算法（AlphaZero）](https://www.science.org/doi/10.1126/science.aar6404)。
- [Schrittwieser 等人 (2020)。基于学习模型规划掌握 Atari、围棋、国际象棋和将棋（MuZero）](https://www.nature.com/articles/s41586-020-03051-4)。
- [Vinyals 等人 (2019)。《星际争霸 II》中的大师级水平（AlphaStar）](https://www.nature.com/articles/s41586-019-1724-z)。
- [DeepSeek-AI (2024)。DeepSeekMath：突破开放语言模型数学推理的极限（GRPO）](https://arxiv.org/abs/2402.03300) —— 引入 GRPO（Group Relative Policy Optimization）与组相对基线（Group-relative Baseline）的论文。
- [DeepSeek-AI (2025)。DeepSeek-R1：通过强化学习激励大语言模型（Large Language Model, LLM）的推理能力](https://arxiv.org/abs/2501.12948) —— 完整的四阶段 R1 训练流程及 R1-Zero 消融实验（Ablation Study）。
- [Brown 等人 (2019)。多人扑克游戏中的超人类人工智能（Pluribus）](https://www.science.org/doi/10.1126/science.aay2400) —— 大规模结合 CFR（Counterfactual Regret Minimization）与深度学习（Deep Learning）。
- [Tesauro (1995)。时序差分学习（Temporal Difference Learning）与 TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) —— 开启该领域的奠基之作。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) —— 结合自定义奖励函数（Reward Function）应用 GRPO 的生产级参考文档。
- [Qwen 团队 (2024)。Qwen2.5-Math — GRPO 复现](https://github.com/QwenLM/Qwen2.5-Math) —— 在多种模型规模上对 R1 训练流程的开源复现。
- [Sutton & Barto (2018)。第 17 章 — 强化学习的前沿](http://incompleteideas.net/book/RLbook2020.pdf) —— 为自我对弈、搜索与“人工设计奖励”（Designed Reward）提供了教科书级的理论框架，R1 正是在大语言模型规模上对这些概念的具体实现。