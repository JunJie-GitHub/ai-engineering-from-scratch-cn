# 奖励建模与基于人类反馈的强化学习 (Reward Modeling & RLHF)

> 人类无法为“优质的助手回复”编写奖励函数 (reward function)，但他们可以比较两个回复并选出更好的一个。将奖励模型 (reward model) 拟合到这些比较结果上，然后针对该模型对语言模型 (language model) 进行强化学习 (RL)。Christiano 2017。InstructGPT 2022。正是这套方案将 GPT-3 变成了 ChatGPT。到了 2026 年，它大多已被直接偏好优化 (DPO) 取代——但其核心思维模型依然适用。

**类型:** 构建
**语言:** Python
**前置要求:** 第 5 阶段 · 05 (情感分析), 第 9 阶段 · 08 (PPO)
**耗时:** 约 45 分钟

## 问题

你基于下一个词元预测 (next-token-prediction) 目标训练了一个语言模型。它能写出语法正确的英文。但它也会撒谎、啰嗦，并且该拒绝时不拒绝。你无法通过更多的预训练来解决这个问题——网络文本本身就是问题所在，而非解药。

你需要一个*标量奖励 (scalar reward)*，它能表达“对于指令 X，回复 A 优于回复 B”。手动编写这样的奖励函数是不可能的。“有用性”并不是基于词元 (token) 的闭式表达式 (closed-form expression)。但人类可以比较两个输出并标记偏好。这种数据可以低成本地大规模收集。

RLHF（Christiano 等人 2017；Ouyang 等人 2022）将偏好转化为奖励模型，然后通过近端策略优化 (PPO) 针对该奖励优化语言模型。分为三步：监督微调 (SFT) → 奖励建模 (RM) → PPO。正是这套方案在 2023–2025 年间催生了 ChatGPT、Claude、Gemini 以及所有其他对齐的大语言模型 (aligned-LLM)。

到了 2026 年，PPO 步骤大多已被 DPO（第 10 阶段 · 08）取代，因为它成本更低，且在对齐调优方面效果几乎相当。但*奖励模型*这一核心组件仍然是所有 N 选优采样器 (Best-of-N sampler)、所有基于可验证奖励的强化学习 (RL-from-verifiable-rewards) 流水线，以及所有使用过程奖励模型 (process reward model) 的推理模型的基石。理解了 RLHF，你就理解了整个对齐技术栈 (alignment stack)。

## 核心概念

![三阶段 RLHF：SFT、基于成对偏好的奖励模型训练、带 KL 惩罚的 PPO](../assets/rlhf.svg)

**阶段 1：监督微调（Supervised Fine-Tuning, SFT）。**从预训练基础模型开始，在人类编写的目标行为示范数据（如遵循指令的回复、有帮助的回答等）上进行微调。最终得到一个模型 `π_SFT`，该模型*倾向于生成良好行为*，但其动作空间（action space）仍然是无界的。

**阶段 2：奖励模型（Reward Model, RM）训练。**

- 收集针对提示词 `x` 的回复对 `(y_+, y_-)`，由人类标注为“`y_+` 优于 `y_-`”。
- 训练一个奖励模型 `R_φ(x, y)`，使其为 `y_+` 分配更高的分数。
- 损失函数：采用 **Bradley-Terry 成对逻辑损失（Bradley-Terry pairwise logistic）**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  其中 σ 为 Sigmoid 函数。奖励值的差异对应着偏好对数几率（log-odds）。自 1952 年（Bradley-Terry）提出以来，该方法一直是标准做法，也是现代 RLHF 中的主流选择。

- `R_φ` 通常以 SFT 模型为基础进行初始化，并在顶部添加一个标量输出头（scalar head）。两者共享相同的 Transformer 骨干网络，仅通过一个线性层输出奖励值。

**阶段 3：结合 KL 惩罚的 PPO 对抗奖励模型。**

- 从 `π_SFT` 初始化可训练的策略模型 `π_θ`。同时保留一个冻结的*参考模型* `π_ref = π_SFT`。
- 在回复 `y` 生成结束时的奖励计算如下：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚项用于防止 `π_θ` 过度偏离 `π_SFT`——它是一种*正则化器（regularizer）*，而非硬性信任域（hard trust region）。`β` 的取值通常在 `0.01` 到 `0.05` 之间。
- 使用该奖励运行 PPO（第 08 课）。优势函数（Advantages）基于词元级（token-level）轨迹进行计算，但奖励模型仅对完整回复进行打分。

**为什么需要 KL 惩罚？**如果没有它，PPO 会轻易找到奖励黑客策略（reward-hacking strategies）——因为奖励模型仅在分布内（in-distribution）续写数据上训练过。分布外（out-of-distribution）的回复得分可能会超过任何人类编写的回复。KL 惩罚能将 `π_θ` 约束在奖励模型训练所覆盖的流形（manifold）附近。它是 RLHF 中最重要的关键调节参数（knob）。

**2026 年现状：**

- **DPO**（Rafailov, 2023）：通过闭式代数推导（closed-form algebra），将第 2 和第 3 阶段合并为基于偏好数据的单一监督损失。无需奖励模型，也无需 PPO。在对齐基准测试中能达到同等质量，但计算成本仅为原来的一小部分。详见第 10 阶段 · 第 08 课。
- **GRPO**（DeepSeek, 2024–2025）：使用组相对基线（group-relative baseline）替代价值网络（critic），奖励来自*验证器*（verifier）（如代码能否运行/数学答案是否匹配）而非人类训练的奖励模型。在推理模型中占据主导地位。详见第 9 阶段 · 第 12 课。
- **过程奖励模型（Process Reward Models, PRMs）：**对部分解答（即每个推理步骤）进行打分，广泛应用于面向推理任务的 RLHF 和 GRPO 变体中。
- **宪法 AI（Constitutional AI）/ RLAIF：**使用已对齐的大语言模型替代人类来生成偏好数据，从而大幅扩展偏好数据预算（preference budget）。

## 动手实践

本课程使用以字符串形式表示的微型合成“提示词（prompts）”和“回复（responses）”。奖励模型（Reward Model, RM）是一个基于词袋（bag-of-tokens）表示的线性打分器。此处不涉及真实的大语言模型（Large Language Model, LLM）——关键在于流水线的*结构（shape）*，而非规模。请参阅 `code/main.py`。

### 步骤 1：合成偏好数据

PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)

在真实的基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）中，这一步会由人工标注员替代。其数据形式（shape）—— `(prompt, preferred_response, rejected_response)` ——是完全一致的。

### 步骤 2：Bradley-Terry 奖励模型

线性打分公式：`R(x, y) = w · bag(y)`。通过训练最小化 Bradley-Terry（BT）成对对数损失（pairwise log-loss）：

def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt

经过数百次更新后，权重向量 `w` 会对优质词汇的词元（tokens）赋予正权重，对劣质词汇赋予负权重。

### 步骤 3：基于奖励模型的类 PPO 策略

我们的示例策略（toy policy）会从词表中生成单个词元。我们在奖励模型下对该词元进行打分，计算 `log π_θ(token | prompt)`，加入相对于参考模型（reference model）的 KL 散度（Kullback-Leibler divergence, KL）惩罚项，并应用截断的近端策略优化（Proximal Policy Optimization, PPO）代理目标函数（surrogate）。

def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...

### 步骤 4：监控 KL 散度

在每次更新时跟踪平均 `KL(π_θ || π_ref)`。如果该值缓慢超过 `~5-10`，说明策略已严重偏离监督微调（Supervised Fine-Tuning, SFT）模型 `π_SFT`——此时可能是 `β` 值设置过低导致 KL 散度上升，或是出现了奖励欺骗（reward hacking）现象。这是真实 RLHF 流程中最核心的诊断指标。

### 步骤 5：使用 TRL 的生产级方案

在理解了简易流水线之后，以下是实际库用户编写相同训练循环的方式。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——第二阶段使用 `RewardTrainer`，第三阶段使用 `PPOTrainer`（内置了相对于参考模型的 KL 惩罚）。

# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()

# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics

该库为你自动处理了三件事。`adap_kl_ctrl=True` 实现了自适应 `β` 调度策略：如果观测到的 KL 散度超过 `target_kl`，`β` 值翻倍；如果低于一半，`β` 值减半。按照惯例，参考模型是冻结的——你绝不能意外地与 `policy` 共享参数。此外，价值头（value head）与策略共享同一个主干网络（backbone）（`AutoModelForCausalLMWithValueHead` 会附加一个标量多层感知机（Multilayer Perceptron, MLP）头），这也是为什么 TRL 会分别报告 `policy/kl` 和 `value/loss`。

## 常见陷阱

- **过度优化 / 奖励黑客攻击（Reward Hacking）。** 奖励模型（Reward Model, RM）并不完美；策略模型 `π_θ` 会找到得分很高但实际质量很差的对抗性续写。表现：奖励值无限攀升，而人工评估分数却停滞或下降。解决方法：提前停止训练、提高 `β` 值、扩充奖励模型的训练数据。
- **长度刷分（Length Hacking）。** 在有用回复上训练的奖励模型往往会隐式地偏好长文本。策略模型会学会给回复注水。应对方案：使用长度归一化奖励，或采用具备长度感知能力的奖励模型进行基于AI反馈的强化学习（RLAIF）。
- **奖励模型过小。** 奖励模型的参数量至少需要与策略模型相当。过小的奖励模型无法准确评估策略模型的输出。
- **KL散度调优（KL Tuning）。** `β` 过低会导致策略漂移和奖励黑客攻击；`β` 过高则策略几乎不会更新。标准技巧是使用*自适应* `β`，以每一步固定的KL散度为目标进行调节。
- **偏好数据噪声（Preference-data Noise）。** 约30%的人工标注存在噪声或模棱两可。可通过在一致性过滤后的数据上训练奖励模型进行校准，或在Bradley-Terry（BT）模型中引入温度参数。
- **异策略问题（Off-policy Problems）。** 在第一个训练周期（epoch）之后，近端策略优化（PPO）的数据会轻微偏离当前策略。请参照第08课监控裁剪比例（Clip Fraction）。

## 实际应用

2026年的基于人类反馈的强化学习（RLHF）架构呈现分层化：

| 层级 | 目标 | 方法 |
|-------|--------|--------|
| 指令遵循、有用性、无害性 | 对齐（Alignment） | 优先使用直接偏好优化（DPO）（第10阶段 · 08课），而非RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力（Capability） | 结合验证器奖励的组相对策略优化（GRPO）（第9阶段 · 12课）。 |
| 长程多步任务 | 智能体（Agentic） | 结合逐步过程奖励模型的PPO / GRPO。 |
| 安全性 / 拒绝行为 | 安全（Safety） | 使用独立安全奖励模型的RLHF-PPO，或宪法AI（Constitutional AI）。 |
| 推理时最佳N选一（Best-of-N） | 快速对齐 | 在解码阶段使用奖励模型；无需训练策略。 |
| 奖励蒸馏（Reward Distillation） | 推理计算优化 | 在冻结的语言模型（LM）顶部训练一个小型“奖励头（reward head）”。 |

RLHF曾是2022–2024年的*主流*方法。到了2026年，生产环境的对齐流水线已转向“DPO优先”，仅在奖励模型密集型或安全关键步骤中使用PPO。

## 交付与部署

保存为 `outputs/skill-rlhf-architect.md`：

---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.

## 练习

1. **简单。** 在 `code/main.py` 中，使用 500 对合成偏好对（synthetic preference pairs）训练 Bradley-Terry 奖励模型（Bradley-Terry reward model）。在预留的 100 对数据上评估成对准确率（pairwise accuracy）。结果应超过 90%。
2. **中等。** 设置 `β ∈ {0.0, 0.1, 1.0}` 并运行简易 PPO-RLHF 循环（toy PPO-RLHF loop）。针对每个 β 值，绘制训练更新过程中奖励模型得分（RM score）与参考模型 KL 散度（KL-to-reference）的变化曲线。观察哪些实验出现了奖励欺骗（reward hacking）现象？
3. **困难。** 在相同的偏好数据上实现 DPO（直接偏好优化，采用闭式偏好似然损失 closed-form preference-likelihood loss），并从计算开销和最终奖励模型得分两个维度，将其与 RLHF-PPO 流程进行对比。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| RLHF（基于人类反馈的强化学习） | “对齐强化学习” | 包含 SFT（监督微调）+ RM（奖励模型）+ PPO 的三阶段流程（Christiano 2017, Ouyang 2022）。 |
| 奖励模型（Reward Model, RM） | “打分网络” | 通过 Bradley-Terry 模型拟合成对偏好数据而学习到的标量函数。 |
| Bradley-Terry 模型 | “成对逻辑损失” | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；奖励模型的标准优化目标。 |
| KL 惩罚（KL penalty） | “靠近参考模型” | 奖励函数中的 `β · KL(π_θ || π_ref)` 项；用于防止奖励欺骗的正则化项。 |
| 奖励欺骗（Reward hacking） | “古德哈特定律” | 策略利用奖励模型的缺陷；典型症状：奖励分数上升，但人类评估结果停滞。 |
| RLAIF（基于 AI 反馈的强化学习） | “AI 标注偏好” | 偏好标签由另一个大语言模型（LM）而非人类提供的 RLHF 变体。 |
| PRM（过程奖励模型） | “过程奖励模型” | 对部分推理步骤进行打分；常用于推理流水线中。 |
| 宪法 AI（Constitutional AI） | “Anthropic 的方法” | 由明确规则引导、由 AI 生成偏好数据的方法。 |

## 延伸阅读

- [Christiano 等人（2017）。Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — 开启基于人类反馈的强化学习（Reinforcement Learning from Human Feedback）研究的开山之作。
- [Ouyang 等人（2022）。InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT 背后的核心训练方案。
- [Stiennon 等人（2020）。Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 早期将 RLHF 应用于文本摘要的研究。
- [Rafailov 等人（2023）。Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — 直接偏好优化（Direct Preference Optimization）；2026 年后 RLHF 时代的默认标准。
- [Bai 等人（2022）。Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — 基于 AI 反馈的强化学习（Reinforcement Learning from AI Feedback）与自我批判循环（Self-Critique Loop）。
- [Anthropic RLHF 论文（Bai 等人，2022）。Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — 即著名的有益且无害（Helpful and Harmless）论文。
- [Hugging Face TRL 库](https://huggingface.co/docs/trl) — 提供生产环境可用的 `RewardTrainer` 和 `PPOTrainer`。建议阅读训练器源码以了解自适应 KL 散度（Adaptive KL）与价值头（Value Head）的实现细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf)（作者：Lambert, Castricato, von Werra, Havrilla）— 配有图解的三阶段流水线（Three-Stage Pipeline）权威指南。
- [von Werra 等人（2020）。TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — 该开源库；`examples/` 目录提供了针对 Llama、Mistral 和 Qwen 模型的端到端 RLHF 脚本。
- [Sutton & Barto（2018）。第 17.4 章 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — 阐述奖励假设（Reward Hypothesis）观点；是理解奖励欺骗（Reward Hacking）问题的必备前置知识。