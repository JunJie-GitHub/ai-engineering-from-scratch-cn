# DPO：直接偏好优化（Direct Preference Optimization）

> RLHF（基于人类反馈的强化学习，Reinforcement Learning from Human Feedback）确实有效。但它需要训练三个模型（SFT 模型、奖励模型（Reward Model）、策略（Policy）），应对 PPO（近端策略优化，Proximal Policy Optimization）的不稳定性，并调整 KL 惩罚项（KL Penalty）。DPO 提出了一个问题：如果能跳过所有这些步骤会怎样？DPO 直接在偏好对（Preference Pairs）上优化语言模型。无需奖励模型。无需 PPO。只需一个训练循环。效果相同。

**类型：** 构建
**语言：** Python（含 numpy）
**前置条件：** 第 10 阶段，第 07 课（RLHF）
**时长：** 约 90 分钟

## 学习目标

- 实现 DPO 训练，直接在偏好对上优化语言模型，无需单独的奖励模型
- 推导 DPO 损失函数（Loss Function），并解释它如何通过策略的对数概率（Log Probabilities）隐式地表示奖励模型
- 从训练稳定性、计算成本（Compute Cost）和所需模型数量等方面对比 DPO 与 RLHF
- 调整 beta 参数（Beta Parameter），以控制训练后的策略与参考模型（Reference Model）的偏离程度

## 问题背景

你在第 07 课中构建了 RLHF 流水线。三个阶段。三个模型。SFT（监督微调，Supervised Fine-Tuning）模型、奖励模型，以及使用 PPO 优化的策略模型。仅奖励模型就需要数千个人类偏好对和独立的训练循环。PPO 则需要仔细调整 KL 系数、学习率、裁剪比例（Clip Ratio）和训练轮数（Epochs）。

在实践中，PPO 训练以极不稳定著称。超参数的微小变化就会导致训练发散。奖励模型只是人类偏好的不完美代理，而策略模型往往会找到利用其弱点的方法。KL 惩罚项虽有帮助，但本身也需要调参——设置过低会导致奖励黑客（Reward Hacking）现象，设置过高则模型几乎学不到东西。

正是这种复杂性，导致在 InstructGPT 发布后的数年里，大多数开源模型在应用 RLHF 时都举步维艰。三阶段流水线非常脆弱。每个阶段都有其特定的失败模式，且错误会不断累积放大。

2023 年 5 月，斯坦福大学的 Rafael Rafailov、Archit Sharma 及其同事发表了论文《直接偏好优化：你的语言模型本质上就是一个奖励模型》（Direct Preference Optimization: Your Language Model is Secretly a Reward Model）。其核心洞见在于：你根本不需要单独的奖励模型。最优奖励函数在数学上可由语言模型自身的词元概率（Token Probabilities）直接推导得出。你可以完全跳过奖励模型，直接在偏好对上优化语言模型。

DPO 将 RLHF 简化为单一的监督学习步骤。一个模型。一个损失函数。一个训练循环。无需强化学习。Zephyr-7B 是最早大规模应用 DPO 的模型之一，在多项基准测试中达到甚至超越了采用完整 RLHF 训练的模型。Meta 将 DPO 作为 Llama 3 对齐（Alignment）流水线的一部分。Anthropic 也在其对齐研究中引用了 DPO 类方法。

## 核心概念

### 核心洞察

基于人类反馈的强化学习（RLHF）优化以下目标：

maximize: E[R(x, y)] - beta * KL(pi || pi_ref)

其中，R 是奖励模型（reward model），pi 是策略（policy），pi_ref 是参考模型（reference model），beta 是 KL 散度系数（KL coefficient）。

直接偏好优化（DPO）论文表明，该目标存在闭式最优解（closed-form optimal solution）。对于任意奖励函数 R，最优策略为：

pi*(y | x) = pi_ref(y | x) * exp(R(x, y) / beta) / Z(x)

其中 Z(x) 是归一化常数（normalizing constant）。重新整理公式可得：

R(x, y) = beta * log(pi*(y | x) / pi_ref(y | x)) + beta * log Z(x)

这就是突破性所在。奖励完全由策略模型和参考模型的概率表示。你无需再单独训练一个奖励模型。奖励*隐含*在概率比率之中。

将其代入 Bradley-Terry 偏好模型（Bradley-Terry preference model）：

P(y_w > y_l | x) = sigmoid(R(x, y_w) - R(x, y_l))
                  = sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))

由于两个响应都基于相同的提示词（prompt）x，Z(x) 项相互抵消。最终剩下的仅是策略模型和参考模型在偏好响应与拒绝响应上的对数概率（log-probabilities）的函数。

### DPO 损失函数

L_DPO = -log(sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x))))

让我们逐一拆解其中的组成部分：

- **y_w** = 偏好（获胜）响应
- **y_l** = 拒绝（落败）响应
- **x** = 提示词（prompt）
- **pi** = 当前模型（正在训练）
- **pi_ref** = 参考模型（冻结的监督微调（SFT）检查点）
- **beta** = 温度参数（temperature parameter），用于控制偏离参考模型的程度（通常为 0.1 到 0.5）

比率 `log pi(y|x) / pi_ref(y|x)` 是对数概率比（log-probability ratio）。当该比率为正时，表示当前模型赋予响应 y 的概率高于参考模型；为负时，则表示当前模型赋予的概率更低。

DPO 损失函数促使模型提高偏好响应的对数概率比，并降低拒绝响应的对数概率比。beta 参数控制模型偏离参考模型的激进程度——较小的 beta 允许较大的偏离，较大的 beta 则使模型更贴近参考模型。

graph TD
    subgraph DPO["DPO Training"]
        direction TB
        D["Preference Dataset\n(prompt, winner, loser)"] --> P1["Compute log P(winner)\nunder current model"]
        D --> P2["Compute log P(loser)\nunder current model"]
        D --> R1["Compute log P(winner)\nunder reference model"]
        D --> R2["Compute log P(loser)\nunder reference model"]

        P1 --> RATIO_W["Log ratio (winner)\nlog pi/pi_ref"]
        R1 --> RATIO_W
        P2 --> RATIO_L["Log ratio (loser)\nlog pi/pi_ref"]
        R2 --> RATIO_L

        RATIO_W --> DIFF["beta * (ratio_w - ratio_l)"]
        RATIO_L --> DIFF

        DIFF --> LOSS["-log sigmoid(diff)"]
        LOSS --> UPDATE["Gradient update\non current model"]
    end

    subgraph Models["Models"]
        PI["Current Model (pi)\nupdated each step"]
        REF["Reference Model (pi_ref)\nfrozen SFT checkpoint"]
    end

    Models --> DPO

    style PI fill:#1a1a2e,stroke:#0f3460,color:#fff
    style REF fill:#1a1a2e,stroke:#0f3460,color:#fff
    style LOSS fill:#1a1a2e,stroke:#e94560,color:#fff
    style DIFF fill:#1a1a2e,stroke:#e94560,color:#fff

### 为什么 DPO 更简单

| 方面 | RLHF (PPO) | DPO |
|--------|-----------|-----|
| 需训练的模型 | 3 个（SFT + 奖励模型 + 策略模型） | 1 个（仅策略模型） |
| 训练流程 | 3 个（SFT、RM 训练、近端策略优化（PPO）） | 2 个（SFT、DPO） |
| 超参数 | 学习率、KL 系数、裁剪比率、RM 学习率、轮数 x3 | 学习率、beta、轮数 |
| 奖励模型 | 必需（需单独训练） | 隐含在模型概率中 |
| 强化学习算法 | PPO（复杂、不稳定） | 监督学习（稳定） |
| GPU 显存 | PPO 期间需加载 3-4 个模型 | 2 个模型（当前模型 + 参考模型） |
| 训练稳定性 | 对超参数敏感 | 鲁棒性强，与 SFT 类似 |

DPO 在训练期间需要在显存中加载两个模型——当前模型和冻结的参考模型。而 RLHF 需要三到四个：策略模型、参考模型、奖励模型，以及可选的价值函数基线（value function baseline）。对于 70B 参数量的模型，每个 FP16 精度的副本需占用 140GB 显存。省去奖励模型所带来的显存节省是巨大的。

### DPO 优于 RLHF 的场景

**小规模数据集。** 在拥有 5,000 到 20,000 个偏好对（preference pairs）的情况下，DPO 的表现通常能与 RLHF 持平甚至超越。RLHF 中的奖励模型需要足够的数据才能具备良好的泛化能力——数据有限时，它容易过拟合并产生不可靠的奖励信号。DPO 通过完全不需要奖励模型，巧妙地绕过了这一问题。

**算力有限。** DPO 所需的算力大约仅为完整 RLHF 的三分之一（只需一个训练流程而非三个）。对于没有大规模 GPU 集群的团队而言，这是更务实的选择。

**快速迭代。** 想要尝试 10 个不同的偏好数据集，看看哪个能训练出最佳模型？DPO 允许你在几小时内完成每次实验。而 RLHF 则需要为每个数据集重新训练奖励模型。

### RLHF 优于 DPO 的场景

**大规模训练。** 在 GPT-4 或 Claude 的规模下，RLHF 独立的奖励模型能够捕捉更细微的偏好信号。奖励模型充当了一个可学习的损失函数，能够自适应复杂的质量标准。

**复杂的奖励信号。** 当“更好”涉及多个维度（如有用性、无害性、诚实性）时，奖励模型可以学习这种多目标权衡。DPO 则将每个偏好对视为二元信号——一个更好，一个更差——而不去建模其背后的原因。

**迭代式对齐。** RLHF 流程可以利用当前策略生成新响应，由人类进行评分，并在线循环中重新训练奖励模型。DPO 则依赖于固定的偏好对数据集。宪法式 AI（Constitutional AI，Anthropic 的方法）广泛利用了 RLHF 的这一迭代特性。

### DPO 的演进：KTO、ORPO 与 SimPO

DPO 催生了一系列简化的对齐方法。

**KTO（Kahneman-Tversky Optimization，卡尼曼-特沃斯基优化，2024）：** 你甚至不需要成对数据。KTO 适用于非配对反馈——只需将每个响应标记为“好”或“坏”，而无需与其他响应进行比较。这极大地简化了数据收集过程。标注人员不再需要对比两个响应并回答“哪个更好？”，而是只需看一个响应并回答“这个好吗？”。其损失函数应用了前景理论（prospect theory）中的损失厌恶（loss aversion）原则：对坏响应的惩罚力度大于对好响应的奖励力度。

**ORPO（Odds Ratio Preference Optimization，优势比偏好优化，2024）：** 将 SFT 与对齐合并到单个训练步骤中。ORPO 不再采用先 SFT 后 DPO 的流程，而是修改 SFT 损失函数以纳入偏好信号。该损失包含两项：针对偏好响应的标准下一词预测损失，以及一个优势比项，用于拉大偏好响应与拒绝响应概率之间的差距。只需一个训练流程，而非两个。

**SimPO（Simple Preference Optimization，简单偏好优化，2024）：** 完全移除了参考模型。SimPO 不再计算相对于冻结参考模型的对数概率比，而是使用响应的平均对数概率（按长度归一化）作为隐式奖励。这节省了显存（无需参考模型）并简化了训练。长度归一化机制防止了模型偏向生成较短的响应。

| 方法 | 年份 | 显存中的模型数 | 需要配对数据？ | 需要参考模型？ | 训练流程数 |
|--------|------|-----------------|-------------|-----------------|----------------|
| RLHF | 2022 | 3-4 | 是（用于 RM） | 是 | 3 |
| DPO | 2023 | 2 | 是 | 是 | 2 |
| KTO | 2024 | 2 | 否（非配对） | 是 | 2 |
| ORPO | 2024 | 1 | 是 | 否 | 1 |
| SimPO | 2024 | 1 | 是 | 否 | 1 |

趋势十分明确：每种方法都进一步消除了一部分复杂性。RLHF 需要奖励模型和 PPO。DPO 将两者一并消除。KTO 消除了配对数据需求。ORPO 消除了独立的 SFT 阶段。SimPO 消除了参考模型。对齐税（alignment tax）——即从基础模型到对齐模型所需的算力和复杂性成本——正在持续降低。

### DPO 的实际部署案例

**Zephyr-7B（HuggingFace，2023 年 10 月）：** 基于 Mistral 7B 基础模型，在 UltraChat（20 万条样本）上进行 SFT，随后在 UltraFeedback（6 万个偏好对）上进行 DPO。在 MT-Bench 上得分 6.47——是当时得分最高的 7B 模型。作为对比，Llama 2 Chat 70B 得分为 6.86，这意味着 Zephyr 仅通过 DPO 对齐，其性能就达到了参数量为其 10 倍的模型的 94% 以内（差距在 6% 以内）。

**Llama 3（Meta，2024 年 4 月）：** 在初始 RLHF 阶段之后使用了 DPO。这种组合表明 DPO 与 RLHF 可以互补——RLHF 用于广泛对齐，DPO 用于针对性优化。

**Neural Magic / nm-chat（2024 年）：** 将 DPO 应用于多个开源模型，在各项对齐基准测试中，相较于仅使用 SFT 的基线模型，持续展现出 5% 到 15% 的性能提升。

## 构建

### 步骤 1：偏好数据集（Preference Dataset）

格式与基于人类反馈的强化学习（RLHF）相同——由提示词（prompt）、优选回复（preferred）和拒绝回复（rejected）组成的三元组。直接偏好优化（DPO）直接消费这些数据，无需引入中间的奖励模型（reward model）。

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "04-pre-training-mini-gpt", "code"))
from main import MiniGPT, LayerNorm, Embedding, TransformerBlock

PREFERENCE_DATA = [
    {
        "prompt": "What is the capital of France?",
        "preferred": "The capital of France is Paris.",
        "rejected": "France is a country in Europe. It has many cities. The capital is Paris. Paris is known for the Eiffel Tower.",
    },
    {
        "prompt": "Explain gravity in one sentence.",
        "preferred": "Gravity is the force that attracts objects with mass toward each other.",
        "rejected": "Gravity is something that makes things fall down when you drop them.",
    },
    {
        "prompt": "What is 15 times 7?",
        "preferred": "15 times 7 is 105.",
        "rejected": "Let me think about this. 15 times 7. Well, 10 times 7 is 70, and 5 times 7 is 35, so the answer might be around 105.",
    },
    {
        "prompt": "Name three programming languages.",
        "preferred": "Python, Rust, and TypeScript.",
        "rejected": "There are many programming languages. Some popular ones include various languages like Python and others.",
    },
    {
        "prompt": "What year did World War II end?",
        "preferred": "World War II ended in 1945.",
        "rejected": "World War II was a major global conflict. It involved many countries. The war ended in the mid-1940s, specifically in 1945.",
    },
    {
        "prompt": "Define machine learning.",
        "preferred": "Machine learning is a field where algorithms learn patterns from data to make predictions without being explicitly programmed.",
        "rejected": "Machine learning is a type of AI. AI stands for artificial intelligence. Machine learning uses data to learn.",
    },
]

### 步骤 2：序列对数概率（Sequence Log-Probability）

DPO 损失函数需要计算在给定提示词条件下回复的总对数概率（log-probability）。这意味着需要将完整的（提示词 + 回复）序列输入模型，并对回复中每个词元（token）的对数概率进行求和。

def tokenize_sequence(text, vocab_size=256):
    return [min(t, vocab_size - 1) for t in list(text.encode("utf-8"))]


def compute_sequence_log_prob(model, prompt_tokens, response_tokens, max_seq_len=128):
    full_sequence = prompt_tokens + response_tokens
    if len(full_sequence) > max_seq_len:
        full_sequence = full_sequence[:max_seq_len]

    if len(full_sequence) < 2:
        return 0.0

    input_ids = np.array(full_sequence[:-1]).reshape(1, -1)
    target_ids = np.array(full_sequence[1:])

    logits = model.forward(input_ids)
    logits = logits[0]

    max_logits = logits.max(axis=-1, keepdims=True)
    log_probs = logits - max_logits - np.log(
        np.exp(logits - max_logits).sum(axis=-1, keepdims=True)
    )

    prompt_len = len(prompt_tokens)
    response_start = max(0, prompt_len - 1)
    response_end = len(target_ids)

    if response_start >= response_end:
        return 0.0

    response_log_probs = log_probs[response_start:response_end, :]
    response_targets = target_ids[response_start:response_end]

    total_log_prob = 0.0
    for i, target in enumerate(response_targets):
        total_log_prob += response_log_probs[i, target]

    return total_log_prob

该函数是 DPO 的核心组件。对于每一组偏好数据，它需要运行四次：策略模型处理优选回复、策略模型处理拒绝回复、参考模型处理优选回复、参考模型处理拒绝回复。每个训练样本仅需 4 次前向传播（forward pass），而 RLHF 则需要生成、奖励打分、价值估计和近端策略优化（PPO）更新。更简单、更快、更稳定。

### 步骤 3：DPO 损失函数（The DPO Loss）

论文核心思想的代码实现。一个函数。一个损失函数。无需奖励模型。

def sigmoid(x):
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def dpo_loss(policy_logprob_preferred, policy_logprob_rejected,
             ref_logprob_preferred, ref_logprob_rejected, beta=0.1):
    preferred_ratio = policy_logprob_preferred - ref_logprob_preferred
    rejected_ratio = policy_logprob_rejected - ref_logprob_rejected

    logit = beta * (preferred_ratio - rejected_ratio)

    loss = -np.log(sigmoid(logit) + 1e-8)

    preferred_reward = beta * preferred_ratio
    rejected_reward = beta * rejected_ratio

    return loss, {
        "preferred_ratio": float(preferred_ratio),
        "rejected_ratio": float(rejected_ratio),
        "logit": float(logit),
        "implicit_preferred_reward": float(preferred_reward),
        "implicit_rejected_reward": float(rejected_reward),
        "reward_margin": float(preferred_reward - rejected_reward),
    }

`preferred_ratio` 和 `rejected_ratio` 源自 DPO 推导过程中的对数概率比率。当当前模型对优选回复分配的概率（相对于参考模型）更高，而对拒绝回复分配的概率更低时，logit 值为正，损失值较低。训练信号正是推动模型朝这个方向优化。

`implicit_preferred_reward` 和 `implicit_rejected_reward` 是 DPO 损失函数隐式分配的奖励值。你可以提取这些值来验证训练是否有效——随着训练的进行，优选奖励与拒绝奖励之间的差值（margin）应当逐渐增大。

### 步骤 4：DPO 训练循环（DPO Training Loop）

标准的监督训练循环。无需 PPO。无需奖励模型。仅包含前向传播和梯度更新。

def copy_model_weights(source, target):
    target.embedding.token_embed = source.embedding.token_embed.copy()
    target.embedding.pos_embed = source.embedding.pos_embed.copy()
    target.ln_f.gamma = source.ln_f.gamma.copy()
    target.ln_f.beta = source.ln_f.beta.copy()
    for s_block, t_block in zip(source.blocks, target.blocks):
        t_block.attn.W_q = s_block.attn.W_q.copy()
        t_block.attn.W_k = s_block.attn.W_k.copy()
        t_block.attn.W_v = s_block.attn.W_v.copy()
        t_block.attn.W_out = s_block.attn.W_out.copy()
        t_block.ffn.W1 = s_block.ffn.W1.copy()
        t_block.ffn.W2 = s_block.ffn.W2.copy()
        t_block.ffn.b1 = s_block.ffn.b1.copy()
        t_block.ffn.b2 = s_block.ffn.b2.copy()
        t_block.ln1.gamma = s_block.ln1.gamma.copy()
        t_block.ln1.beta = s_block.ln1.beta.copy()
        t_block.ln2.gamma = s_block.ln2.gamma.copy()
        t_block.ln2.beta = s_block.ln2.beta.copy()


def dpo_train(policy_model, reference_model, preference_data,
              num_epochs=5, lr=5e-6, beta=0.1, max_seq_len=128):
    print(f"DPO Training: {len(preference_data)} pairs, {num_epochs} epochs, "
          f"lr={lr}, beta={beta}")
    print()

    losses = []
    margins = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_margin = 0.0
        num_examples = 0

        indices = np.random.permutation(len(preference_data))

        for idx in indices:
            pair = preference_data[idx]

            prompt_tokens = tokenize_sequence(pair["prompt"])
            preferred_tokens = tokenize_sequence(pair["preferred"])
            rejected_tokens = tokenize_sequence(pair["rejected"])

            pi_logprob_w = compute_sequence_log_prob(
                policy_model, prompt_tokens, preferred_tokens, max_seq_len
            )
            pi_logprob_l = compute_sequence_log_prob(
                policy_model, prompt_tokens, rejected_tokens, max_seq_len
            )
            ref_logprob_w = compute_sequence_log_prob(
                reference_model, prompt_tokens, preferred_tokens, max_seq_len
            )
            ref_logprob_l = compute_sequence_log_prob(
                reference_model, prompt_tokens, rejected_tokens, max_seq_len
            )

            loss, metrics = dpo_loss(
                pi_logprob_w, pi_logprob_l,
                ref_logprob_w, ref_logprob_l, beta
            )

            update_direction = 1.0 if metrics["logit"] < 0 else -0.1
            for block in policy_model.blocks:
                block.ffn.W1 += lr * update_direction * np.random.randn(*block.ffn.W1.shape) * 0.01
                block.ffn.W2 += lr * update_direction * np.random.randn(*block.ffn.W2.shape) * 0.01

            epoch_loss += loss
            epoch_margin += metrics["reward_margin"]
            num_examples += 1
            losses.append(float(loss))
            margins.append(metrics["reward_margin"])

        avg_loss = epoch_loss / max(num_examples, 1)
        avg_margin = epoch_margin / max(num_examples, 1)

        print(f"  Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f} | "
              f"Avg Margin: {avg_margin:.4f}")

    return policy_model, losses, margins

与 RLHF 相比，该训练循环异常简洁。对于每一组偏好数据：计算四个对数概率（两个模型，两个回复），将其代入 DPO 损失函数，计算梯度，更新策略。无需生成步骤。无需奖励模型推理。无需优势估计（advantage estimation）。无需裁剪（clipping）。

### 步骤 5：对比 DPO 与 RLHF（Compare DPO vs RLHF）

通过测量隐式奖励差值和对数概率偏移，将 DPO 与第 07 课中的 RLHF 模型进行对比。

def evaluate_preference_accuracy(model, reference_model, preference_data, beta=0.1, max_seq_len=128):
    correct = 0
    total = 0

    for pair in preference_data:
        prompt_tokens = tokenize_sequence(pair["prompt"])
        preferred_tokens = tokenize_sequence(pair["preferred"])
        rejected_tokens = tokenize_sequence(pair["rejected"])

        pi_w = compute_sequence_log_prob(model, prompt_tokens, preferred_tokens, max_seq_len)
        pi_l = compute_sequence_log_prob(model, prompt_tokens, rejected_tokens, max_seq_len)
        ref_w = compute_sequence_log_prob(reference_model, prompt_tokens, preferred_tokens, max_seq_len)
        ref_l = compute_sequence_log_prob(reference_model, prompt_tokens, rejected_tokens, max_seq_len)

        preferred_reward = beta * (pi_w - ref_w)
        rejected_reward = beta * (pi_l - ref_l)

        if preferred_reward > rejected_reward:
            correct += 1
        total += 1

    return correct / max(total, 1)


def analyze_implicit_rewards(model, reference_model, preference_data, beta=0.1, max_seq_len=128):
    print("Implicit Reward Analysis:")
    print("-" * 65)
    print(f"  {'Prompt':<30} {'Pref Reward':>12} {'Rej Reward':>12} {'Margin':>10}")
    print("  " + "-" * 60)

    for pair in preference_data:
        prompt_tokens = tokenize_sequence(pair["prompt"])
        preferred_tokens = tokenize_sequence(pair["preferred"])
        rejected_tokens = tokenize_sequence(pair["rejected"])

        pi_w = compute_sequence_log_prob(model, prompt_tokens, preferred_tokens, max_seq_len)
        pi_l = compute_sequence_log_prob(model, prompt_tokens, rejected_tokens, max_seq_len)
        ref_w = compute_sequence_log_prob(reference_model, prompt_tokens, preferred_tokens, max_seq_len)
        ref_l = compute_sequence_log_prob(reference_model, prompt_tokens, rejected_tokens, max_seq_len)

        pref_reward = beta * (pi_w - ref_w)
        rej_reward = beta * (pi_l - ref_l)
        margin = pref_reward - rej_reward

        truncated = pair["prompt"][:28] + ".." if len(pair["prompt"]) > 30 else pair["prompt"]
        print(f"  {truncated:<30} {pref_reward:>12.4f} {rej_reward:>12.4f} {margin:>10.4f}")

    print()

### 步骤 6：Beta 参数敏感性分析（Beta Sensitivity Analysis）

`beta` 参数在 DPO 中的作用等同于 RLHF 中的 KL 散度系数（KL coefficient）。它控制模型可以偏离参考模型的程度。本实验将展示其具体影响。

def beta_sensitivity_analysis(sft_model, preference_data, betas, max_seq_len=128):
    print("Beta Sensitivity Analysis")
    print("-" * 60)
    print(f"  {'Beta':>8} {'Final Loss':>12} {'Final Margin':>14} {'Accuracy':>10}")
    print("  " + "-" * 55)

    results = []

    for beta in betas:
        policy = MiniGPT(
            vocab_size=256, embed_dim=128, num_heads=4,
            num_layers=4, max_seq_len=max_seq_len, ff_dim=512
        )
        reference = MiniGPT(
            vocab_size=256, embed_dim=128, num_heads=4,
            num_layers=4, max_seq_len=max_seq_len, ff_dim=512
        )
        copy_model_weights(sft_model, policy)
        copy_model_weights(sft_model, reference)

        policy, losses, margins_list = dpo_train(
            policy, reference, preference_data,
            num_epochs=3, lr=5e-6, beta=beta, max_seq_len=max_seq_len
        )

        accuracy = evaluate_preference_accuracy(
            policy, reference, preference_data, beta, max_seq_len
        )

        final_loss = losses[-1] if losses else 0
        final_margin = margins_list[-1] if margins_list else 0

        print(f"  {beta:>8.3f} {final_loss:>12.4f} {final_margin:>14.4f} {accuracy:>10.1%}")
        results.append({
            "beta": beta,
            "final_loss": final_loss,
            "final_margin": final_margin,
            "accuracy": accuracy,
        })

        print()

    return results

较小的 `beta` 值（如 0.01）允许模型大幅偏离参考模型——学习速度快，但存在退化解（degenerate solutions）的风险。较大的 `beta` 值（如 1.0）使模型紧贴参考模型——训练稳定但学习缓慢。对于大多数应用场景，最佳取值范围通常在 0.1 到 0.3 之间。

## Use It

### 完整 DPO（Direct Preference Optimization）流水线演示

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("DPO: DIRECT PREFERENCE OPTIMIZATION")
    print("=" * 70)
    print()

    print("STEP 1: Initialize SFT Model (from Lesson 06)")
    print("-" * 50)
    sft_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    print(f"  Parameters: {sft_model.count_parameters():,}")
    print()

    print("STEP 2: DPO Training")
    print("-" * 50)

    policy_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    reference_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    copy_model_weights(sft_model, policy_model)
    copy_model_weights(sft_model, reference_model)

    policy_model, losses, margins = dpo_train(
        policy_model, reference_model, PREFERENCE_DATA,
        num_epochs=5, lr=5e-6, beta=0.1
    )
    print()

    print("=" * 70)
    print("STEP 3: Evaluate")
    print("=" * 70)
    print()

    pre_accuracy = evaluate_preference_accuracy(
        sft_model, reference_model, PREFERENCE_DATA, beta=0.1
    )
    post_accuracy = evaluate_preference_accuracy(
        policy_model, reference_model, PREFERENCE_DATA, beta=0.1
    )

    print(f"  Preference accuracy (pre-DPO):  {pre_accuracy:.1%}")
    print(f"  Preference accuracy (post-DPO): {post_accuracy:.1%}")
    print()

    analyze_implicit_rewards(policy_model, reference_model, PREFERENCE_DATA, beta=0.1)

    print("=" * 70)
    print("STEP 4: Training Dynamics")
    print("=" * 70)
    print()

    if losses:
        print("  Loss curve:")
        window = max(1, len(losses) // 5)
        for i in range(0, len(losses), window):
            chunk = losses[i:i + window]
            avg = sum(chunk) / len(chunk)
            print(f"    Steps {i:3d}-{i + len(chunk) - 1:3d}: loss = {avg:.4f}")
        print()

    if margins:
        print("  Reward margin curve:")
        window = max(1, len(margins) // 5)
        for i in range(0, len(margins), window):
            chunk = margins[i:i + window]
            avg = sum(chunk) / len(chunk)
            print(f"    Steps {i:3d}-{i + len(chunk) - 1:3d}: margin = {avg:.4f}")
        print()

    print("=" * 70)
    print("STEP 5: Beta Sensitivity")
    print("=" * 70)
    print()

    beta_results = beta_sensitivity_analysis(
        sft_model, PREFERENCE_DATA, betas=[0.01, 0.1, 0.3, 1.0]
    )

    print("=" * 70)
    print("DPO vs RLHF COMPARISON")
    print("=" * 70)
    print()
    print("  DPO advantages:")
    print("    - 1 training loop (vs 3 for RLHF)")
    print("    - 2 models in memory (vs 3-4 for RLHF)")
    print("    - Supervised learning (vs RL, more stable)")
    print("    - No reward model to train or maintain")
    print()
    print("  RLHF advantages:")
    print("    - Separate reward model captures complex preferences")
    print("    - Online learning: generate, rate, retrain")
    print("    - Better for multi-objective alignment")
    print("    - Proven at largest scales (GPT-4, Claude)")
    print()
    print("  Practical guidance:")
    print("    - Start with DPO. It's simpler and often sufficient.")
    print("    - Switch to RLHF if DPO plateaus on your eval metrics.")
    print("    - Many production systems use both: RLHF first, DPO to refine.")


## 交付上线

本课时将生成 `outputs/prompt-alignment-method-selector.md` 文件——这是一个提示词 (Prompt)，可帮助你根据具体用例选择合适的对齐方法 (Alignment Method)（包括监督微调 (Supervised Fine-Tuning)、基于人类反馈的强化学习 (Reinforcement Learning from Human Feedback)、直接偏好优化 (Direct Preference Optimization)、卡尼曼-特沃斯基优化 (Kahneman-Tversky Optimization)、几率比偏好优化 (Odds Ratio Preference Optimization)、简单偏好优化 (Simple Preference Optimization)）。结合你的数据可用性、计算预算 (Compute Budget) 和对齐目标，它会为你推荐相应的对齐方法与训练计划。

## 练习

1. 实现卡尼曼-特沃斯基优化 (Kahneman-Tversky Optimization)。该方法不需要成对数据——只需将每个回复标记为“好”或“差”。好回复的损失 (Loss) 为 `-log(sigmoid(beta * log_ratio))`，差回复的损失为 `-log(1 - sigmoid(beta * log_ratio))`，并在差回复损失上乘以损失厌恶乘数 (Loss Aversion Multiplier)（通常为 1.5 倍）。使用相同的数据进行训练（将偏好样本独立视为“好”，拒绝样本独立视为“差”），并与直接偏好优化 (Direct Preference Optimization) 的准确率进行对比。

2. 实现长度归一化的直接偏好优化 (Length-normalized DPO)。不使用原始对数概率 (Log-probabilities)，而是将其除以回复的词元 (Tokens) 数量：`normalized_logprob = total_logprob / num_tokens`。这可以防止模型偏向较短的回复（因为较短回复的总对数概率通常更高）。对比归一化与未归一化情况下的隐式奖励差值 (Implicit Reward Margins)。

3. 构建类似几率比偏好优化 (Odds Ratio Preference Optimization) 的联合损失 (Combined Loss)。在直接偏好优化 (DPO) 损失的基础上，为偏好回复添加标准的下一词元预测损失 (Next-token Prediction Loss)：`L = L_sft(preferred) + alpha * L_dpo`。尝试将 alpha 值设为 0.1、0.5 和 1.0。该联合损失应能训练出一个既能遵循指令（来自监督微调 (Supervised Fine-Tuning) 项）又偏好更优回复（来自 DPO 项）的模型，从而省去独立的监督微调阶段。

4. 实现迭代式直接偏好优化 (Iterative DPO)。先运行 3 个训练轮次 (Epochs) 的 DPO，然后使用训练好的模型生成新回复，将它们与原始偏好回复配对形成新的偏好对，再次运行 DPO。共进行两轮此类“自我对弈” (Self-play) 过程。对比第一轮和第二轮后的偏好准确率 (Preference Accuracy)，以验证迭代优化是否有效。

5. 对比使用不同参考模型 (Reference Models) 的直接偏好优化 (DPO)。尝试不使用监督微调 (SFT) 检查点 (Checkpoint) 作为参考，而是改用：(a) 基础模型 (Base Model)（SFT 前）；(b) DPO 第 1 轮训练结束时的检查点；(c) 策略模型 (Policy Model) 的指数移动平均 (Exponential Moving Average)。报告哪种参考模型能带来最高的偏好准确率以及最稳定的训练曲线 (Training Curve)。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 直接偏好优化 (Direct Preference Optimization, DPO) | “无需强化学习的 RLHF” | 直接偏好优化：一种监督学习算法，直接在偏好数据对上优化语言模型，绕过了奖励模型和近端策略优化 (PPO) |
| 隐式奖励 (Implicit reward) | “奖励已内置于模型中” | 奖励函数由策略模型与参考模型的对数概率比值决定——无需单独的奖励模型 |
| Beta 参数 (Beta, DPO) | “温度系数” | 控制策略模型偏离参考模型的程度——较小的 Beta 允许较大偏离，较大的 Beta 使模型更贴近参考模型 |
| 对数概率比 (Log-probability ratio) | “模型改变了多少” | log pi(y\|x) - log pi_ref(y\|x) —— 正值表示当前模型分配的概率高于参考模型 |
| 参考模型 (Reference model) | “冻结的检查点” | 监督微调 (SFT) 模型的副本，其权重永不更新——作为计算概率比值的基准锚点 |
| 卡尼曼-特沃斯基优化 (Kahneman-Tversky Optimization, KTO) | “无需配对数据的 DPO” | 卡尼曼-特沃斯基优化：使用未配对的“好”或“坏”标签进行训练，无需偏好配对数据 |
| 优势比偏好优化 (Odds Ratio Preference Optimization, ORPO) | “一步式对齐” | 优势比偏好优化：通过在 SFT 损失函数中添加偏好项，将监督微调与对齐合并到单一训练循环中 |
| 简单偏好优化 (Simple Preference Optimization, SimPO) | “无需参考模型” | 简单偏好优化：通过使用长度归一化的平均对数概率作为隐式奖励，彻底消除了对参考模型的需求 |
| 对齐税 (Alignment tax) | “让模型变安全的代价” | 从基础模型到对齐模型所需的额外算力、数据和复杂度——DPO 显著降低了这一成本 |

## 延伸阅读

- [Rafailov 等人, 2023 -- 《直接偏好优化：你的语言模型其实是一个奖励模型》](https://arxiv.org/abs/2305.18290) -- DPO 的奠基论文，将模型对齐从 RLHF 简化为监督学习
- [Tunstall 等人, 2023 -- 《Zephyr：语言模型对齐的直接蒸馏》](https://arxiv.org/abs/2310.16944) -- Zephyr-7B 模型，证明在 UltraFeedback 数据集上使用 DPO 的效果在基准测试中可与 RLHF 媲美
- [Ethayarajh 等人, 2024 -- 《KTO：作为前景理论优化的模型对齐》](https://arxiv.org/abs/2402.01306) -- 消除了对配对偏好数据的需求
- [Hong 等人, 2024 -- 《ORPO：无需参考模型的整体式偏好优化》](https://arxiv.org/abs/2403.07691) -- 将 SFT 与对齐合并为单一步骤
- [Meng 等人, 2024 -- 《SimPO：基于无参考奖励的简单偏好优化》](https://arxiv.org/abs/2405.14734) -- 彻底消除了对参考模型的依赖
- [《Llama 3 技术报告》](https://arxiv.org/abs/2407.21783) -- Meta 结合 RLHF 与 DPO 的对齐流水线