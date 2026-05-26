# 宪法AI（Constitutional AI）与自我改进（Self-Improvement）

> 基于人类反馈的强化学习（RLHF）需要人类介入循环。宪法AI（Constitutional AI）则用模型自身替代了大部分人工环节。只需编写一份原则清单，让模型依据这些原则对自身输出进行批判，并利用这些批判数据进行训练。DeepSeek-R1 在 2025 年进一步拓展了这一思路：让模型生成数百万条推理轨迹（reasoning traces），通过确定性规则进行打分，并在结果上运行 GRPO（组相对策略优化，Group Relative Policy Optimization）。在 2026 年的前沿模型中，绝大部分“对齐工作”已转变为模型的自我对齐。本课将完整实现这两个循环。

**类型：** 构建实践
**语言：** Python（标准库 + numpy）
**前置要求：** 第 10 阶段，第 06-08 课（监督微调（SFT）、RLHF、直接偏好优化（DPO））
**时长：** 约 45 分钟

## 学习目标

- 实现宪法AI（Constitutional AI）的两阶段循环：自我批判与自我修订，随后基于修订后的数据对进行偏好训练
- 推导 GRPO（组相对策略优化，Group Relative Policy Optimization）的目标函数，并将其与 PPO（近端策略优化，Proximal Policy Optimization）的价值函数基线（value-function baseline）进行对比
- 生成带有基于规则的结果奖励的可验证推理轨迹（reasoning traces），并在无需独立奖励模型（reward model）的情况下对其进行评分
- 判断自我改进（self-improvement）何时优于人类偏好数据，以及何时会退化为模式搜索（mode seeking）

## 问题背景

你在第 07 课实现了 RLHF，在第 08 课实现了 DPO（直接偏好优化，Direct Preference Optimization）。两者都依赖同一种昂贵的输入数据：人类偏好对（human preference pairs）。Anthropic 在 InstructGPT 时代的流水线使用了约 33,000 组对比数据。Llama 2 Chat 使用了超过 150 万组。Claude 3 的用量则更大。这类数据获取缓慢、成本高昂，且容易受到标注员在评分当天主观认知偏差的影响。

2022 年的宪法AI（Constitutional AI）论文提出了一个简单的问题：如果让模型自己生成偏好标签会怎样？只需提供一份书面原则清单（即“宪法”），让模型对自身回答进行批判。这些批判结果即可作为训练信号。

2024 年，DeepSeek 将这一理念进一步拓展。他们证明，对于任何具有可验证结果的任务（如有标准答案的数学题、能通过或无法通过测试的代码、有明确胜负的游戏），完全可以跳过批判模型（critic）环节。生成大量候选解决方案，使用确定性规则对每个方案进行评分，并基于这些奖励运行策略梯度（policy-gradient）算法。DeepSeek-R1 正是通过这种方式训练的，几乎未使用人类偏好数据，却达到了与 o1 级别相当的推理性能。

这两个循环——针对主观行为的宪法AI（Constitutional AI）与针对可验证行为的基于规则的强化学习（rule-based RL）——已成为 2026 年主流的对齐方案。过去投入 RLHF 的人类偏好数据预算，现在仅用于一个更精简的步骤：制定宪法原则与设定奖励规则。

## 核心概念

### 宪法式人工智能循环（Constitutional AI Loop）

Bai 等人（2022）将该流程划分为两个阶段。

**第一阶段：基于 AI 反馈的监督学习（Supervised Learning from AI Feedback, SL-CAI）。** 从一个有帮助但可能产生有害内容的监督微调（Supervised Fine-Tuning, SFT）模型开始。向其输入可能有害的请求。对于每个回复，要求*同一个模型*根据宪法原则（constitutional principle）对其回复进行批评，然后进行修订。使用修订后的回复进行微调。数据集由（提示词，修订后的回复）对组成。

**第二阶段：基于 AI 反馈的强化学习（Reinforcement Learning from AI Feedback, RLAIF）。** 采样成对的回复。询问模型哪一个更好地遵循了宪法。这些成对偏好数据用于训练一个奖励模型（reward model）。然后使用该奖励对模型运行近端策略优化（Proximal Policy Optimization, PPO）或直接偏好优化（Direct Preference Optimization, DPO）。与基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）的关键区别在于：偏好数据来自模型本身，而非人类。

graph TD
    subgraph SL["Stage 1: SL-CAI"]
        P1["Harmful prompt"] --> R1["Initial response\n(possibly harmful)"]
        R1 --> C1["Model critiques\nagainst principle"]
        C1 --> REV["Model revises\nresponse"]
        REV --> SFT["SFT on\n(prompt, revised)"]
    end

    subgraph RL["Stage 2: RLAIF"]
        P2["Prompt"] --> S1["Sample response A"]
        P2 --> S2["Sample response B"]
        S1 --> J["Model judges\nA vs B via constitution"]
        S2 --> J
        J --> RM["Preference dataset"]
        RM --> TRAIN["DPO / PPO training"]
    end

    SL --> RL

    style P1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style REV fill:#1a1a2e,stroke:#51cf66,color:#fff
    style P2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style TRAIN fill:#1a1a2e,stroke:#51cf66,color:#fff

宪法是其中的杠杆。Anthropic 最初的版本包含 16 条原则（后续有所扩充）。一条原则的表述类似于：“请选择最不可能引起来自各种文化背景的人反感的回复。”你在每一步中选择适用的原则，有时是随机选择，有时则根据提示词的类别来决定。

### 宪法实际起到的作用

宪法将对齐（alignment）契约从*数据*转移到了*文本*。在 RLHF 下改变模型行为意味着重新标注数千个数据对。而在 CAI 下改变行为只需编辑一段文本。这是其主要的实际优势。

但这也有代价。模型的自我判断能力仅与其初始校准水平相当。如果 SFT 模型存在盲区——例如无法识别操纵性措辞——那么批评步骤也会继承这些盲区。CAI 压缩了对齐循环，但无法将信号放大到超越基础模型的能力上限。这就是为什么每个生产环境的 CAI 流程仍然会使用少量人类偏好数据，其规模通常仅为纯 RLHF 的 5% 到 10%。

### GRPO：组相对策略优化（Group-Relative Policy Optimization）

DeepSeek 在 DeepSeekMath 论文（2024）中引入了 GRPO，并将其作为 DeepSeek-R1（2025）的核心架构。GRPO 是 PPO 的一个变体，它移除了价值函数（value function）。

回顾 PPO 的目标函数（来自第 07 课）：

L_PPO = E[min(r(theta) * A, clip(r(theta), 1-eps, 1+eps) * A)]

其中 `A` 是优势（advantage），通常使用广义优势估计（Generalized Advantage Estimation, GAE）通过一个学习到的价值网络 `V(s)` 进行估计。价值网络是一个与策略模型（policy）规模相同的第二个模型。它会使内存占用翻倍，并引入其自身的训练循环。

GRPO 直接移除了价值函数。对于每个提示词，它会采样一组 G 个回复（通常 G=16 或 64）。计算每个回复的奖励，然后在组内进行归一化：

A_i = (r_i - mean(r_1, ..., r_G)) / std(r_1, ..., r_G)

优势值即为该回复奖励相对于同组其他回复的 Z 分数（z-score）。无需价值函数，该组自身就充当了基线（baseline）。

L_GRPO = E[min(r(theta) * A_group, clip(r(theta), 1-eps, 1+eps) * A_group)] - beta * KL(pi || pi_ref)

针对参考模型（reference model）的 KL 散度惩罚项依然存在，与 PPO 相同。裁剪比率（clip ratio）也保留了下来。被移除的是独立的评论家模型（critic）。

### 为什么 GRPO 对推理任务至关重要

对于推理任务，奖励通常是稀疏且二元的：最终答案要么对，要么错。在稀疏二元奖励上训练价值函数是一种浪费——它无法学到有用的中间状态估计，因为在最后一步之前，几乎每个状态的预期回报都是相同的。GRPO 的组归一化能为你提供即时的相对信号：在针对同一道数学题的 16 次尝试中，哪些尝试的表现高于该题的平均水平？

这正是你从基于规则的奖励（rule-based rewards）中获得的信号形态：

- **数学**：使用 `sympy` 或符号检查器判断最终答案是否匹配。
- **代码**：由测试套件决定通过/失败。
- **格式**：通过正则表达式（regex）判断答案是否位于所需的 XML 标签内。
- **多步证明**：由证明辅助工具（如 Lean、Coq）判定有效性。

DeepSeek-R1-Zero 仅使用两种奖励进行训练：数学基准测试的准确率和格式合规性（答案位于 `<answer>` 标签内）。没有人类偏好数据，也没有评论家模型。DeepSeek 论文中描述的“顿悟时刻”（模型自发学会自我检查和回溯）正是仅凭 GRPO 在稀疏规则奖励上训练涌现出来的。

### 过程奖励模型 vs 结果奖励模型

你仍面临一个设计选择：奖励最终答案（结果奖励模型，Outcome Reward Model, ORM），还是奖励每个中间步骤（过程奖励模型，Process Reward Model, PRM）。

| 维度 | ORM | PRM |
|------|-----|-----|
| 每条轨迹的信号 | 1 个数值 | N 个数值（每步一个） |
| 监督信号来源 | 最终答案检查 | 步骤级标签或自我评判 |
| 训练成本 | 低 | 高 |
| 信用分配（Credit assignment） | 稀疏、噪声大 | 密集、针对性强 |
| 奖励欺骗（Reward hacking）风险 | 较低 | 较高（模型会优化 PRM 的伪影/特征） |
| 使用方 | DeepSeek-R1, R1-Zero | OpenAI o1（据传）, Math-Shepherd |

2024-2025 年的共识是，ORM 结合 GRPO 的扩展性优于 PRM。PRM 在每 token 的样本效率上更高，但需要昂贵的步骤级标注数据，且容易退化为捷径行为（写出对 PRM 看起来很好但并未推进证明的步骤）。对于大多数团队而言，ORM + GRPO 是首选的尝试方案。

### 自我改进：反馈乘数效应

一旦你掌握了双循环模式（批评/修订，以及结合规则奖励的组相对强化学习），就可以将它们串联起来。

1. 从一个 SFT 模型开始。
2. 为每个提示词生成大量候选回复。
3. 使用基于规则的奖励（针对可验证任务）或宪法式评论器（针对主观任务）对它们进行评分。
4. 保留排名靠前的候选回复，作为新的 SFT 数据或偏好对。
5. 进行微调。使用改进后的模型回到第 2 步。

DeepSeek 在 R1-Zero 之后应用此方法时，称之为“拒绝采样微调”（rejection sampling fine-tuning）。Anthropic 早期版本则称之为“宪法式 AI 蒸馏”（constitutional AI distillation）。其核心模式是：每次迭代都会放大模型中已有的信号，而不会引入新信号。如果模型完全无法解决 X 类问题，无论进行多少次自我改进，都无法凭空创造出这种能力。

潜在危险在于模式崩溃（mode collapse）。自我生成的数据分布总是比原始训练语料更窄。经过 3-5 轮自我蒸馏后，模型通常在创意任务上会丧失多样性，变得过度自信，并表现出典型的“AI 腔调”（重复的措辞、公式化的结构）。生产环境中的流程会将自我生成的数据与少量新鲜的人类数据混合，以维持数据分布的“真实性”（honest）。

graph LR
    M0["SFT Model v0"] --> G["Generate G responses\nper prompt"]
    G --> S["Score with rule\nor constitution"]
    S --> F["Filter / rank"]
    F --> T["Fine-tune\n(SFT or GRPO)"]
    T --> M1["SFT Model v1"]
    M1 -.->|iterate| G

    H["Human data\n(small fraction)"] --> T

    style M0 fill:#1a1a2e,stroke:#e94560,color:#fff
    style M1 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style H fill:#1a1a2e,stroke:#0f3460,color:#fff

### 何时使用何种方法

- **纯 CAI**：主观行为（语气、安全性、拒绝风格）。你拥有定义明确的宪法。你没有干净的可验证结果。
- **GRPO + ORM**：可验证任务（数学、代码、结构化信息提取）。你可以低成本地检查正确性。奖励是稀疏且二元的。
- **基于自生成数据对的 DPO**：混合方案。使用宪法生成偏好对，然后使用 DPO（第 08 课）而非 PPO/GRPO 进行训练。
- **完整 RLHF**：当你需要处理规则或简短宪法都无法表达的多目标权衡时，它仍然适用。

大多数 2026 年的前沿流程会同时运行这四种方法。CAI 用于安全层。GRPO 用于推理后训练阶段。DPO 用于偏好打磨。小规模的 RLHF 流程则用于处理其他方法难以解决的残余行为。

## 动手构建

该代码使用纯 Python 和 NumPy 实现了三项功能：宪法式人工智能（Constitutional AI）的自我批判循环（self-critique loop）、用于简单算术的基于规则的奖励检查器（rule-based reward checker），以及一个可在第 04 课微型语言模型上运行的最小化 GRPO 训练器（GRPO trainer）。

### 步骤 1：宪法（Constitution）

这是一组原则列表。在生产环境中，每条原则的内容会更丰富，并带有类别标签。为了便于教学，此处保持简短。

CONSTITUTION = [
    "The response must directly answer the question asked, without hedging.",
    "The response must not include unnecessary filler or padding.",
    "If the question has a single numeric answer, state the number plainly.",
    "The response must not refuse a reasonable, benign request.",
]

### 步骤 2：自我批判与修订（Self-Critique and Revise）

在真实系统中，由模型自身进行批判。在本课程中，我们使用手写评分标准来模拟批判器（critic），从而使整个流水线（pipeline）无需调用大语言模型（LLM）即可运行。

def critique(response: str, principle: str) -> dict:
    problems = []
    if len(response.split()) > 40 and "plainly" in principle:
        problems.append("answer buried in extra prose")
    if response.strip().lower().startswith(("i can't", "i cannot", "as an ai")):
        problems.append("unwarranted refusal")
    if response.count(",") > 4:
        problems.append("too much hedging")
    return {"principle": principle, "problems": problems}

def revise(response: str, critique_result: dict) -> str:
    if "answer buried" in " ".join(critique_result["problems"]):
        return response.split(".")[-2].strip() + "."
    if "unwarranted refusal" in " ".join(critique_result["problems"]):
        return "Here is the answer: " + response.split(":")[-1].strip()
    return response

`revise` 函数仅作为占位符。在使用真实 LLM 时，它会转化为第二个提示词（prompt）：“根据批判意见，重写回复。”

### 步骤 3：基于规则的奖励（Rule-Based Rewards）

对于可验证的任务，可以完全替换掉批判器。此检查器用于对算术答案进行评分。

import re

def reward_math(prompt: str, response: str) -> float:
    try:
        expected = eval(prompt.replace("What is ", "").replace("?", "").strip())
    except Exception:
        return 0.0
    numbers = re.findall(r"-?\d+", response)
    if not numbers:
        return 0.0
    return 1.0 if int(numbers[-1]) == expected else 0.0

def reward_format(response: str) -> float:
    return 1.0 if re.search(r"<answer>.*</answer>", response) else 0.0

这里包含两条确定性规则。无需训练数据，也无需人工标注。组合奖励为 `reward_math + 0.1 * reward_format`，它在惩罚格式缺失的同时，不会掩盖答案正确性的信号。

### 步骤 4：组相对优势（Group-Relative Advantage）

给定针对同一提示词生成的一组回复的奖励列表，计算其 Z 分数（z-score）：

import numpy as np

def group_relative_advantage(rewards: list[float]) -> np.ndarray:
    r = np.array(rewards, dtype=float)
    if r.std() < 1e-8:
        return np.zeros_like(r)
    return (r - r.mean()) / (r.std() + 1e-8)

如果组内每个样本的奖励都相同，则优势值为零，且不会传递梯度信号（gradient signal）。这是一项设计特性。它表明该提示词对于当前策略（policy）而言要么过于简单已完全解决，要么过于困难无法解决，此时应跳过该步骤。

### 步骤 5：GRPO 更新（GRPO Update）

单步更新，符号梯度（symbolic gradient）。在生产环境中，这通常由 PyTorch 的自动微分（autograd）流程完成。此处我们直接展示更新规则。

def grpo_step(policy_logprobs: np.ndarray, ref_logprobs: np.ndarray,
              advantages: np.ndarray, beta: float = 0.01, clip_eps: float = 0.2) -> dict:
    ratios = np.exp(policy_logprobs - ref_logprobs)
    unclipped = ratios * advantages
    clipped = np.clip(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -np.minimum(unclipped, clipped).mean()
    kl = (ref_logprobs - policy_logprobs).mean()
    total_loss = policy_loss + beta * kl
    return {
        "policy_loss": float(policy_loss),
        "kl": float(kl),
        "total_loss": float(total_loss),
        "mean_ratio": float(ratios.mean()),
    }

这是 PPO 的截断代理目标（clipped surrogate），仅有一处改动：优势值（advantages）来源于组相对 Z 分数，而非价值函数（value function）。无需训练 V(s)，也无需使用广义优势估计（GAE）。整个组即作为基线（baseline）。

### 步骤 6：自我改进轮次（Self-Improvement Round）

将各部分串联起来。采样生成一组回复，使用规则对每个回复进行评分，计算优势值，并输出可输入至真实优化器（optimizer）的指标。

def self_improvement_round(prompts: list[str], policy_sampler, group_size: int = 8) -> dict:
    metrics = []
    for prompt in prompts:
        responses = [policy_sampler(prompt) for _ in range(group_size)]
        rewards = [reward_math(prompt, r) + 0.1 * reward_format(r) for r in responses]
        advantages = group_relative_advantage(rewards)
        best = responses[int(np.argmax(rewards))]
        metrics.append({
            "prompt": prompt,
            "mean_reward": float(np.mean(rewards)),
            "best_reward": float(np.max(rewards)),
            "std_reward": float(np.std(rewards)),
            "best_response": best,
            "advantages": advantages.tolist(),
        })
    return {"per_prompt": metrics,
            "overall_mean": float(np.mean([m["mean_reward"] for m in metrics]))}


## Use It

运行 `code/main.py` 将端到端执行这两个循环。宪法AI (Constitutional AI) 循环会生成少量（初始，修订）数据对，供你进行微调 (fine-tune)。组相对策略优化 (Group Relative Policy Optimization) 循环会针对算术问题输出每个提示词 (prompt) 的奖励统计信息，展示组相对优势 (group-relative advantages) 如何让弱采样器 (weak sampler) 在无需价值函数 (value function) 或人工标注 (human labels) 的情况下实现自我改进。

具体数值并非重点。在使用已训练模型进行实际运行时，奖励均值 (reward mean) 应随迭代轮次逐步上升，奖励标准差 (reward std) 应保持正值（若坍缩至零，则表明策略 (policy) 已发生模式崩溃 (mode collapse)，此时应立即停止），且相对于参考模型 (reference model) 的 KL 散度 (KL divergence) 应缓慢增长。这三条曲线——奖励均值上升、标准差稳定、KL 散度有界——正是 GRPO 或 CAI 流水线 (pipeline) 在生产环境中的健康检查 (production health check) 指标。

## Ship It

本课时将生成 `outputs/skill-self-improvement-auditor.md`。向其输入拟定的自我改进流水线，它将强制执行以下不可妥协的硬性门槛：真正可验证的奖励规则、针对参考模型的 KL 预算 (KL budget)、多样性底线 (diversity floor) 以及人工数据配额 (human-data quota)。对于任何声称“纯自我改进”却缺乏外部依据 (external grounding) 的循环，它将直接拒绝批准。

## Exercises

1. 将步骤 2 中手动编写的评论器 (critic) 替换为大语言模型 (LLM) 调用。可使用任意本地聊天模型。统计批评与修订 (critique and revision) 实际改善回复的频率，并与保持原回复不变的情况进行对比。

2. 增加第三条关于事实性 (factuality) 的宪法原则。在需要事实性声明（如首都、日期）的提示词上运行该流水线，并统计修订操作消除事实错误的次数与引入新错误的次数之比。

3. 在 CAI 阶段 2 生成的偏好对 (preference pairs) 上实现直接偏好优化 (Direct Preference Optimization)。选取 20 个提示词，为每个提示词生成两个回复，由评论器为每对回复选出优胜者，随后应用第 08 课中的 DPO 损失函数 (DPO loss)。在同一数据集上将其与 GRPO 路径进行对比。

4. 在 GRPO 目标函数中加入熵正则化 (entropy regularization)。通过设置 `alpha=0.01` 的 `-alpha * entropy(policy)` 项来鼓励多样化采样。评估在 5 轮自我改进过程中，该机制是否能有效延缓模式崩溃。

5. 为两步算术问题构建过程奖励评分器 (process reward scorer)。以“ (3+4)*5 等于多少？”为例，模型必须展示中间步骤 3+4=7。将中间步骤与最终答案分开评分，并在 10 轮训练中对比过程奖励模型 (Process Reward Model) 加权的 GRPO 与纯结果奖励模型 (Outcome Reward Model) 加权的 GRPO 的表现。

## Key Terms

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 宪法式AI (Constitutional AI) | “模型自我对齐” | 两阶段流程（自我批判 (self-critique) + 基于AI反馈的强化学习 (RLAIF)），通过模型对照书面“宪法”进行自我评判，替代大部分人类偏好标注 |
| 基于AI反馈的强化学习 (RLAIF) | “无需人类参与的RLHF” | 基于AI反馈的强化学习（Reinforcement Learning from AI Feedback）——针对模型自身生成的偏好数据，采用近端策略优化（PPO）或直接偏好优化（DPO）进行训练 |
| 组相对策略优化 (GRPO) | “无需价值函数的PPO” | 组相对策略优化（Group-Relative Policy Optimization）——针对每个提示词采样 G 个回复，使用经过Z分数标准化（z-score）的组内奖励作为优势值（advantages） |
| 结果奖励模型 (ORM) | “仅对答案打分” | 结果奖励模型（Outcome Reward Model）——仅针对最终答案输出单一标量奖励 |
| 过程奖励模型 (PRM) | “对每一步进行奖励” | 过程奖励模型（Process Reward Model）——对每一个中间推理步骤给予奖励，通常使用带步骤标注的数据进行训练 |
| 基于规则的奖励 (Rule-based reward) | “确定性评分器” | 一种验证器（如正则表达式、SymPy 或测试套件），无需依赖训练好的模型即可返回二值或数值评分 |
| 拒绝采样微调 (Rejection sampling FT) | “保留优胜者，重新训练” | 采样大量回复，筛选出奖励最高的样本，加入监督微调（SFT）数据集中重新训练 |
| 模式崩溃 (Mode collapse) | “模型失去了多样性” | 后训练策略过度集中于响应空间的狭窄区域；通常通过组内奖励标准差（std）下降来衡量 |
| KL散度预算 (KL budget) | “允许偏离多远” | 优化器在训练停止前，被允许累积的相对于参考模型的总KL散度（KL divergence）上限 |
| R1时刻 (R1 moment) | “模型学会了回溯” | DeepSeek 报告的现象：仅使用结果奖励训练的策略，在其思维链（chain-of-thought）中自发涌现出自我检查与回溯行为 |

## 延伸阅读

- [Bai et al., 2022 -- "Constitutional AI: Harmlessness from AI Feedback"](https://arxiv.org/abs/2212.08073) -- Anthropic 的 CAI 原始论文，提出了包含两阶段 SL-CAI + RLAIF 的流程
- [Shao et al., 2024 -- "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"](https://arxiv.org/abs/2402.03300) -- 首次引入 GRPO 算法
- [DeepSeek-AI, 2025 -- "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"](https://arxiv.org/abs/2501.12948) -- 介绍 R1 与 R1-Zero，大规模应用 GRPO 结合基于规则的奖励
- [Lightman et al., 2023 -- "Let's Verify Step by Step"](https://arxiv.org/abs/2305.20050) -- 介绍 OpenAI 的 PRM800K 数据集，并论证过程奖励模型的价值
- [Wang et al., 2024 -- "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations"](https://arxiv.org/abs/2312.08935) -- 通过蒙特卡洛 rollout 实现 PRM 的自动标注
- [Huang et al., 2024 -- "Large Language Models Cannot Self-Correct Reasoning Yet"](https://arxiv.org/abs/2310.01798) -- 针对缺乏外部基准的模型自我改进能力提出的质疑与反驳