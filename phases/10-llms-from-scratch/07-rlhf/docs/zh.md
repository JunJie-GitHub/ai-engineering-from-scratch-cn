# 基于人类反馈的强化学习（RLHF）：奖励模型（Reward Model）+ 近端策略优化（PPO）

> 监督微调（Supervised Fine-Tuning, SFT）教会模型遵循指令，但它无法告诉模型哪个回复“更好”。两个语法正确、事实准确的答案在实用性上可能存在巨大差异。RLHF 正是将人类判断编码到模型行为中的方法。它正是让 Claude 变得乐于助人、让 GPT 保持礼貌的关键。

**类型：** 构建
**语言：** Python（含 numpy）
**前置要求：** 第 10 阶段，第 06 课（指令微调 / SFT）
**时长：** 约 90 分钟

## 学习目标

- 构建一个奖励模型（Reward Model），该模型能够根据人类偏好对（优选 vs 拒绝）对回复质量进行评分
- 实现近端策略优化（Proximal Policy Optimization, PPO）训练循环，在奖励模型的指导下优化语言模型策略，并引入 KL 散度惩罚（KL Penalty）
- 解释为何 RLHF 需要三个模型（SFT 模型、奖励模型、策略模型），以及 KL 约束如何防止奖励黑客攻击（Reward Hacking）
- 通过对比偏好优化前后的回复质量，评估 RLHF 的效果

## 核心问题

向模型提问“解释量子计算”，它可能会生成：

**回复 A：**“量子计算使用量子比特（qubits），量子比特可以处于叠加态，这意味着它们可以同时是 0、1 或两者的组合。这使得量子计算机在处理某些计算时，速度比经典计算机呈指数级提升。关键算法包括用于大数分解的 Shor 算法和用于无序数据库搜索的 Grover 算法。”

**回复 B：**“量子计算是一种利用量子力学现象的计算方式。它最早于 20 世纪 80 年代被提出。Richard Feynman 曾建议用量子计算机来模拟量子系统。自那时起，该领域取得了显著发展。如今许多公司都在研发量子计算机。IBM、Google 等机构已取得进展。Google 于 2019 年宣称实现了量子优越性。”

两个回复在事实上都是正确的，语法也都无误，且都遵循了指令。但回复 A 显然更胜一筹。它更简洁、信息量更大、结构也更合理。人类每次都会选择 A。

SFT 无法捕捉这种差异。它仅基于“正确”的回复来训练模型，但缺乏一种机制来表明“这个回复比那个更好”。它将每个训练样本视为同等优质。如果 A 和 B 同时出现在 SFT 数据集中，模型会对两者进行同等程度的学习。

RLHF 正是为了解决这一问题而生。它训练一个奖励模型来预测人类更偏好哪个回复，随后利用该奖励信号引导语言模型生成更高质量的输出。InstructGPT（ChatGPT 的前身）利用 RLHF 大幅提升了 GPT-3 的实用性、真实性和安全性。尽管 InstructGPT 的参数量仅为 GPT-3 的 1/135（13 亿 vs 1750 亿），但 OpenAI 内部评估人员在 85% 的情况下更倾向于选择 InstructGPT 的输出。

## 核心概念

### 三个阶段

基于人类反馈的强化学习 (RLHF) 并非单次训练过程，而是一个包含三个连续阶段的流水线，每个阶段都建立在前一个阶段的基础之上。

**阶段 1：监督微调 (SFT)。** 在指令-回复对 (Instruction-Response Pairs) 上训练基础模型 (Base Model)（参见第 06 课）。这将得到一个能够遵循指令的模型，但它尚不具备判断不同回复优劣的能力。

**阶段 2：奖励模型 (Reward Model)。** 收集人类偏好数据：向标注人员展示针对同一提示词 (Prompt) 的两个回复 (Response)，并询问“哪个更好？”训练一个模型来预测这些偏好。奖励模型以（提示词，回复）作为输入，并输出一个标量分数。

**阶段 3：近端策略优化 (PPO)。** 利用奖励模型为语言模型生成训练信号。语言模型生成回复，奖励模型对其进行打分，PPO 则更新语言模型以生成更高分的回复。KL 散度 (KL Divergence) 惩罚项用于防止语言模型过度偏离 SFT 检查点 (Checkpoint)。

graph TD
    subgraph Stage1["Stage 1: SFT"]
        B["Base Model"] --> S["SFT Model"]
        D["Instruction Data\n(27K examples)"] --> S
    end

    subgraph Stage2["Stage 2: Reward Model"]
        S --> |"Generate responses"| P["Preference Pairs\n(prompt, winner, loser)"]
        H["Human Annotators"] --> P
        P --> R["Reward Model\nR(prompt, response) → score"]
    end

    subgraph Stage3["Stage 3: PPO"]
        S --> |"Initialize policy"| PI["Policy Model\n(being optimized)"]
        S --> |"Freeze as reference"| REF["Reference Model\n(frozen SFT)"]
        PI --> |"Generate"| RESP["Response"]
        RESP --> R
        R --> |"Reward signal"| PPO["PPO Update"]
        REF --> |"KL penalty"| PPO
        PPO --> |"Update"| PI
    end

    style S fill:#1a1a2e,stroke:#51cf66,color:#fff
    style R fill:#1a1a2e,stroke:#e94560,color:#fff
    style PI fill:#1a1a2e,stroke:#0f3460,color:#fff
    style REF fill:#1a1a2e,stroke:#0f3460,color:#fff
    style PPO fill:#1a1a2e,stroke:#e94560,color:#fff

### 奖励模型

奖励模型本质上是一个被重新用作评分器的语言模型。以 SFT 模型为基础，将其语言建模头 (Language Modeling Head，输出词汇表上的概率分布) 替换为标量头 (Scalar Head，输出单个数值)。在最后一层之前，两者的架构完全相同。

输入：拼接在一起的提示词与回复。输出：单个标量奖励分数。

训练数据为人类偏好对。对于每个提示词，标注人员会看到两个回复并选出更优的一个。由此生成训练三元组：（提示词，优选回复，淘汰回复）。

损失函数采用用于成对偏好的 Bradley-Terry 模型：

loss = -log(sigmoid(reward(preferred) - reward(rejected)))

这是核心公式。`sigmoid(reward(A) - reward(B))` 表示回复 A 优于回复 B 的概率。该损失函数会促使奖励模型为优选回复分配更高的分数。

为什么采用成对比较而非绝对打分？因为人类极不擅长给出绝对质量分数（“这个回复是 10 分制里的 7.3 分还是 7.5 分？”），但非常擅长进行相对比较（"A 是否比 B 更好？”）。Bradley-Terry 模型能够将相对比较转化为一致的绝对评分体系。

**InstructGPT 数据：** OpenAI 从 40 名外包人员处收集了 33,000 个比较对。每次比较耗时约 5 分钟。这意味着奖励模型的训练数据共耗费了 2,750 小时的人工标注时间。

### PPO：近端策略优化

PPO 是一种强化学习 (Reinforcement Learning) 算法。在 RLHF 中，“环境 (Environment)”是奖励模型，“智能体 (Agent)”是语言模型，而“动作 (Action)”则是生成一个词元 (Token)。

优化目标：

maximize: E[R(prompt, response)] - beta * KL(policy || reference)

第一项促使模型生成高奖励的回复。第二项（KL 散度惩罚项）防止模型过度偏离 SFT 检查点。

为什么需要 KL 惩罚项？如果没有它，模型会找到退化解 (Degenerate Solutions)。奖励模型是在有限的人类偏好数据集上训练的，存在盲区。语言模型会利用这些盲区——找出在奖励模型上得分很高但实际上毫无意义的输出。经典案例如下：

- 重复“我非常乐于助人且无害！”会在有用性/无害性奖励模型上获得高分
- 生成冗长、听起来正式但内容空洞的回复，以模式匹配“高质量”特征
- 利用训练数据中恰好与高奖励相关的特定短语

KL 惩罚项的作用在于：允许模型改进，但不能使其变成一个完全不同的模型。必须保持与原本就已合理的 SFT 版本相近。偏离过远时，KL 代价将主导奖励信号。

**InstructGPT 数据：** PPO 训练使用学习率 (lr)=1.5e-5，KL 系数 beta=0.02，256K 个回合 (Episodes，即提示词-回复对)，每批次进行 4 个 PPO 轮次 (Epochs)。整个 RLHF 流水线在 GPU 集群上运行了数天。

graph LR
    subgraph PPO["PPO Training Loop"]
        direction TB
        PROMPT["Sample prompt\nfrom dataset"] --> GEN["Policy generates\nresponse"]
        GEN --> SCORE["Reward model\nscores response"]
        GEN --> KL["Compute KL divergence\nvs reference model"]
        SCORE --> OBJ["Objective:\nreward - beta * KL"]
        KL --> OBJ
        OBJ --> UPDATE["PPO gradient update\n(clipped surrogate loss)"]
        UPDATE --> |"repeat"| PROMPT
    end

    style PROMPT fill:#1a1a2e,stroke:#0f3460,color:#fff
    style SCORE fill:#1a1a2e,stroke:#51cf66,color:#fff
    style KL fill:#1a1a2e,stroke:#e94560,color:#fff
    style OBJ fill:#1a1a2e,stroke:#e94560,color:#fff

### PPO 优化目标详解

PPO 使用“截断代理目标函数 (Clipped Surrogate Objective)”来防止更新幅度过大。新旧策略概率的比值会被截断至 `[1 - epsilon, 1 + epsilon]` 范围内，其中 epsilon 通常设为 0.2。

ratio = pi_new(action | state) / pi_old(action | state)
clipped_ratio = clip(ratio, 1 - epsilon, 1 + epsilon)
loss = -min(ratio * advantage, clipped_ratio * advantage)

优势函数 (Advantage Function) 用于估计当前回复相较于预期质量的优劣程度。在 RLHF 中：

advantage = reward(prompt, response) - baseline

基线 (Baseline) 通常是近期回复的平均奖励。正优势表示该回复优于平均水平；负优势则表示劣于平均水平。PPO 会提高优于平均水平的回复的生成概率，并降低劣于平均水平的回复的概率。

截断操作可防止灾难性更新。如果单个回复获得异常高的奖励，未截断的比值可能会非常大，导致模型剧烈地向该回复偏移。截断限制了更新幅度，从而维持了训练的稳定性。

### 奖励黑客 (Reward Hacking)

这是 RLHF 的阴暗面。语言模型针对奖励模型进行优化，而奖励模型只是人类偏好的一个不完美代理 (Proxy)。随着语言模型越来越擅长最大化奖励，它开始利用奖励模型的弱点。

常见失效模式：

| 失效模式 | 具体表现 | 原因 |
|---------|-------------|-----|
| 冗长 (Verbosity) | 模型生成的回复越来越长 | 人类标注人员通常偏好更长、更详细的回复，因此奖励模型会给长度分配更高的分数 |
| 阿谀奉承 (Sycophancy) | 模型对用户的所有说法都表示赞同 | 标注人员更倾向于赞同问题前提的回复 |
| 模棱两可 (Hedging) | 模型拒绝给出明确答案 | 模棱两可的回复（“这是一个复杂的话题，存在多种观点……”）很少被标记为错误 |
| 格式刷分 (Format gaming) | 模型过度使用项目符号和标题 | 格式化的回复在标注人员眼中显得更“精致” |

缓解策略：加大 KL 惩罚项（防止模型偏离到足以利用弱点的程度）、在对抗样本上训练奖励模型（修补已知失效模式），以及使用多个不同架构的奖励模型（同时攻破所有模型的难度更大）。

### 真实的 RLHF 流水线

| 模型 | 比较对数量 | 标注人员 | 奖励模型参数量 | PPO 步数 | KL 系数 |
|-------|-----------------|------------|---------|-----------|----------|
| InstructGPT | 33K | 40 | 6B | 256K | 0.02 |
| Llama 2 Chat | ~1M | 未公开 | 70B | 未公开 | 0.01 |
| Claude | 未公开 | 未公开 | 未公开 | 未公开 | 未公开 |
| Anthropic RLHF 论文 | 22K | 20 | 52B | 50K | 0.001 |

Anthropic 2022 年的论文在 22,000 个比较对上训练了一个 52B 参数的奖励模型。更大的奖励模型能产生更可靠的信号，从而使 PPO 训练更加稳定。使用小型奖励模型来训练大型语言模型存在风险——因为奖励模型的容量不足以捕捉优质回复与劣质回复之间的细微差别。

## 构建项目

### 步骤 1：合成偏好数据 (Synthetic Preference Data)

在实际生产环境中，偏好数据通常由人工标注员创建。我们将构建合成数据对，其中“优选”回复在客观上更优（更简洁、更准确、更有帮助）。

import numpy as np

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

优选回复简洁直接。而被拒绝的回复则表现出常见的失败模式：不必要的填充、含糊其辞、冗余解释以及表述不精确。这正是监督微调 (Supervised Fine-Tuning, SFT) 无法捕捉，但基于人类反馈的强化学习 (Reinforcement Learning from Human Feedback, RLHF) 能够处理的差异。

### 步骤 2：奖励模型架构 (Reward Model Architecture)

奖励模型 (Reward Model) 复用了迷你 GPT (mini GPT) 中的 Transformer 架构，但将原本词汇表大小的输出头替换为单个标量投影。

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "04-pre-training-mini-gpt", "code"))
from main import MiniGPT, LayerNorm, Embedding, TransformerBlock


class RewardModel:
    def __init__(self, vocab_size=256, embed_dim=128, num_heads=4,
                 num_layers=4, max_seq_len=128, ff_dim=512):
        self.embedding = Embedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(embed_dim)
        self.reward_head = np.random.randn(embed_dim) * 0.02

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)

        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x, mask)
        x = self.ln_f.forward(x)

        last_hidden = x[:, -1, :]
        reward = last_hidden @ self.reward_head

        return reward

奖励模型提取*最后一个*词元 (token) 位置的隐藏状态，并将其投影为一个标量。为什么选择最后一个词元？因为因果注意力掩码 (causal attention mask) 意味着最后一个位置已经关注了之前的所有词元。因此，它拥有整个（提示词，回复）序列最完整的表征。

### 步骤 3：Bradley-Terry 损失函数 (Bradley-Terry Loss)

使用 Bradley-Terry 成对损失函数在偏好数据对上训练奖励模型。

def tokenize_for_reward(prompt, response, vocab_size=256):
    prompt_tokens = [min(t, vocab_size - 1) for t in list(prompt.encode("utf-8"))]
    response_tokens = [min(t, vocab_size - 1) for t in list(response.encode("utf-8"))]
    return prompt_tokens + [0] + response_tokens


def sigmoid(x):
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def bradley_terry_loss(reward_preferred, reward_rejected):
    diff = reward_preferred - reward_rejected
    loss = -np.log(sigmoid(diff) + 1e-8)
    return loss


def train_reward_model(rm, preference_data, num_epochs=10, lr=1e-4, max_seq_len=128):
    print(f"Training Reward Model: {len(preference_data)} preference pairs, {num_epochs} epochs")
    print()

    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        num_pairs = 0

        indices = np.random.permutation(len(preference_data))

        for idx in indices:
            pair = preference_data[idx]

            preferred_tokens = tokenize_for_reward(pair["prompt"], pair["preferred"])
            rejected_tokens = tokenize_for_reward(pair["prompt"], pair["rejected"])

            preferred_tokens = preferred_tokens[:max_seq_len]
            rejected_tokens = rejected_tokens[:max_seq_len]

            preferred_ids = np.array(preferred_tokens).reshape(1, -1)
            rejected_ids = np.array(rejected_tokens).reshape(1, -1)

            r_preferred = rm.forward(preferred_ids)[0]
            r_rejected = rm.forward(rejected_ids)[0]

            loss = bradley_terry_loss(r_preferred, r_rejected)

            if r_preferred > r_rejected:
                epoch_correct += 1

            diff = r_preferred - r_rejected
            grad = sigmoid(diff) - 1.0

            rm.reward_head -= lr * grad * rm.ln_f.forward(
                rm.embedding.forward(preferred_ids)
            )[:, -1, :].flatten()

            epoch_loss += loss
            num_pairs += 1

        avg_loss = epoch_loss / max(num_pairs, 1)
        accuracy = epoch_correct / max(num_pairs, 1)
        losses.append(avg_loss)
        accuracies.append(accuracy)

        if epoch % 2 == 0:
            print(f"  Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1%}")

    return rm, losses, accuracies

准确率指标非常直观：奖励模型能正确排序的偏好数据对占比是多少？随机模型的得分应为 50%。在干净数据上训练良好的奖励模型应超过 70%。InstructGPT 的奖励模型在预留测试集上的准确率约为 72%，这听起来不高，但实际上已经相当不错——因为许多偏好数据对即使对人类来说也存在歧义（标注员间的一致性约为 73%）。

### 步骤 4：简化版 PPO 循环 (PPO Loop)

完整的近端策略优化 (Proximal Policy Optimization, PPO) 算法较为复杂。本实现捕捉了其核心机制：生成回复、进行评分、计算优势值 (advantage)，并使用 KL 散度惩罚 (KL penalty) 更新策略。

def compute_kl_divergence(policy_logits, reference_logits):
    policy_probs = np.exp(policy_logits - policy_logits.max(axis=-1, keepdims=True))
    policy_probs = policy_probs / policy_probs.sum(axis=-1, keepdims=True)
    policy_probs = np.clip(policy_probs, 1e-10, 1.0)

    ref_probs = np.exp(reference_logits - reference_logits.max(axis=-1, keepdims=True))
    ref_probs = ref_probs / ref_probs.sum(axis=-1, keepdims=True)
    ref_probs = np.clip(ref_probs, 1e-10, 1.0)

    kl = np.sum(policy_probs * np.log(policy_probs / ref_probs), axis=-1)
    return kl.mean()


def generate_response(model, prompt_tokens, max_new_tokens=30, temperature=0.8, max_seq_len=128):
    tokens = list(prompt_tokens)

    for _ in range(max_new_tokens):
        context = np.array(tokens[-max_seq_len:]).reshape(1, -1)
        logits = model.forward(context)
        next_logits = logits[0, -1, :]

        next_logits = next_logits / max(temperature, 1e-8)
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / probs.sum()

        next_token = np.random.choice(len(probs), p=probs)
        tokens.append(int(next_token))

    return tokens


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


def ppo_training(policy_model, reference_model, reward_model, prompts,
                 num_episodes=20, lr=1.5e-5, kl_coeff=0.02, max_seq_len=128):
    print(f"PPO Training: {num_episodes} episodes, lr={lr}, KL coeff={kl_coeff}")
    print()

    rewards_history = []
    kl_history = []

    for episode in range(num_episodes):
        prompt_text = prompts[episode % len(prompts)]
        prompt_tokens = [min(t, 252) for t in list(prompt_text.encode("utf-8"))]

        response_tokens = generate_response(
            policy_model, prompt_tokens,
            max_new_tokens=20, temperature=0.8, max_seq_len=max_seq_len
        )

        response_ids = np.array(response_tokens[:max_seq_len]).reshape(1, -1)
        reward = reward_model.forward(response_ids)[0]

        policy_logits = policy_model.forward(response_ids)
        ref_logits = reference_model.forward(response_ids)
        kl = compute_kl_divergence(policy_logits, ref_logits)

        total_reward = reward - kl_coeff * kl

        rewards_history.append(float(reward))
        kl_history.append(float(kl))

        for block in policy_model.blocks:
            update_scale = lr * total_reward
            block.ffn.W1 += update_scale * np.random.randn(*block.ffn.W1.shape) * 0.01
            block.ffn.W2 += update_scale * np.random.randn(*block.ffn.W2.shape) * 0.01

        if episode % 5 == 0:
            avg_reward = np.mean(rewards_history[-5:]) if rewards_history else 0
            avg_kl = np.mean(kl_history[-5:]) if kl_history else 0
            print(f"  Episode {episode:3d} | Reward: {reward:.4f} | KL: {kl:.4f} | "
                  f"Avg Reward: {avg_reward:.4f}")

    return policy_model, rewards_history, kl_history

核心循环流程如下：(1) 采样提示词，(2) 生成回复，(3) 使用奖励模型评分，(4) 计算与冻结参考模型之间的 KL 散度 (KL divergence)，(5) 计算调整后的奖励（原始奖励减去 KL 惩罚），(6) 更新策略。随着策略偏离参考模型，KL 惩罚会相应增大，从而自动防止奖励刷分 (reward hacking)。

### 步骤 5：奖励分数对比 (Reward Score Comparison)

经过 RLHF 训练后，策略模型 (policy model) 生成的回复在奖励模型上的得分应高于原始 SFT 模型的回复。

def compare_models(sft_model, rlhf_model, reward_model, prompts, max_seq_len=128):
    print("Model Comparison (reward scores)")
    print("-" * 60)
    print(f"  {'Prompt':<35} {'SFT':>10} {'RLHF':>10}")
    print("  " + "-" * 55)

    sft_total = 0.0
    rlhf_total = 0.0

    for prompt in prompts:
        prompt_tokens = [min(t, 252) for t in list(prompt.encode("utf-8"))]

        sft_response = generate_response(
            sft_model, prompt_tokens,
            max_new_tokens=20, temperature=0.6, max_seq_len=max_seq_len
        )
        rlhf_response = generate_response(
            rlhf_model, prompt_tokens,
            max_new_tokens=20, temperature=0.6, max_seq_len=max_seq_len
        )

        sft_ids = np.array(sft_response[:max_seq_len]).reshape(1, -1)
        rlhf_ids = np.array(rlhf_response[:max_seq_len]).reshape(1, -1)

        sft_reward = reward_model.forward(sft_ids)[0]
        rlhf_reward = reward_model.forward(rlhf_ids)[0]

        sft_total += sft_reward
        rlhf_total += rlhf_reward

        truncated_prompt = prompt[:33] + ".." if len(prompt) > 35 else prompt
        print(f"  {truncated_prompt:<35} {sft_reward:>10.4f} {rlhf_reward:>10.4f}")

    n = len(prompts)
    print("  " + "-" * 55)
    print(f"  {'Average':<35} {sft_total/n:>10.4f} {rlhf_total/n:>10.4f}")

    return sft_total / n, rlhf_total / n


## 上手实践

### 完整 RLHF（基于人类反馈的强化学习）流水线演示

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("RLHF PIPELINE: REWARD MODEL + PPO")
    print("=" * 70)
    print()

    print("STAGE 1: SFT Model (from Lesson 06)")
    print("-" * 40)
    sft_model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    print(f"  Parameters: {sft_model.count_parameters():,}")
    print()

    print("STAGE 2: Train Reward Model")
    print("-" * 40)
    rm = RewardModel(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )

    rm, rm_losses, rm_accuracies = train_reward_model(rm, PREFERENCE_DATA, num_epochs=10, lr=1e-4)
    print()

    print("Reward Model Evaluation:")
    print("-" * 40)
    correct = 0
    for pair in PREFERENCE_DATA:
        pref_tokens = tokenize_for_reward(pair["prompt"], pair["preferred"])[:128]
        rej_tokens = tokenize_for_reward(pair["prompt"], pair["rejected"])[:128]

        r_pref = rm.forward(np.array(pref_tokens).reshape(1, -1))[0]
        r_rej = rm.forward(np.array(rej_tokens).reshape(1, -1))[0]

        if r_pref > r_rej:
            correct += 1
        print(f"  Preferred: {r_pref:+.4f} | Rejected: {r_rej:+.4f} | {'Correct' if r_pref > r_rej else 'Wrong'}")

    print(f"\n  Accuracy: {correct}/{len(PREFERENCE_DATA)} = {correct/len(PREFERENCE_DATA):.1%}")
    print()

    print("STAGE 3: PPO Training")
    print("-" * 40)

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

    train_prompts = [pair["prompt"] for pair in PREFERENCE_DATA]

    policy_model, rewards, kls = ppo_training(
        policy_model, reference_model, rm,
        train_prompts, num_episodes=20, lr=1.5e-5, kl_coeff=0.02
    )
    print()

    print("=" * 70)
    print("COMPARISON: SFT vs RLHF")
    print("=" * 70)
    print()

    eval_prompts = [
        "What is the capital of France?",
        "Explain gravity.",
        "Name three programming languages.",
    ]

    sft_avg, rlhf_avg = compare_models(sft_model, policy_model, rm, eval_prompts)
    print()

    print("=" * 70)
    print("KL DIVERGENCE ANALYSIS")
    print("=" * 70)
    print()

    if kls:
        print(f"  Initial KL: {kls[0]:.4f}")
        print(f"  Final KL:   {kls[-1]:.4f}")
        print(f"  Max KL:     {max(kls):.4f}")
        kl_threshold = 0.1
        print(f"  KL > {kl_threshold}: {'Yes (model drifted significantly)' if max(kls) > kl_threshold else 'No (model stayed close to reference)'}")

## 部署上线

本课程将生成 `outputs/prompt-reward-model-designer.md` —— 一个用于设计奖励模型（Reward Model）训练流程的提示词（Prompt）。给定目标行为（如有用性、编码能力、安全性），它将生成数据收集协议、标注员指南以及奖励模型评估标准。

## 练习

1. 修改奖励模型，使其使用所有隐藏状态（Hidden States）的均值，而非仅使用最后一个位置的状态。比较两者的准确率。均值池化（Mean Pooling）方法为每个词元（Token）赋予相同的权重，而末位位置方法则依赖因果注意力（Causal Attention）来聚合信息。在 6 个偏好对（Preference Pairs）上进行测试，并报告哪种方法的准确率更高。

2. 实现奖励模型校准（Reward Model Calibration）。训练完成后，将所有偏好对输入奖励模型并计算：(a) 偏好回复的平均奖励值，(b) 拒绝回复的平均奖励值，(c) 差值（偏好回复奖励减去拒绝回复奖励）。校准良好的模型应具有明显的奖励差值。随后添加 4 个新的偏好对，检查该差值在未见数据（Unseen Data）上是否依然成立。

3. 模拟奖励欺骗（Reward Hacking）。创建一个对长回复给予高分的奖励模型（reward = len(response) / 100）。使用此有缺陷的奖励模型运行近端策略优化（Proximal Policy Optimization, PPO），观察策略模型（Policy Model）生成越来越长且重复的输出。随后添加 0.1 的 KL 惩罚（KL Penalty），并证明其能够防止这种退化行为。

4. 实现多目标奖励（Multi-objective Reward）。训练两个奖励模型——一个针对有用性，另一个针对简洁性。将它们组合为 R = 0.7 * R_helpful + 0.3 * R_concise。证明该组合目标能够生成既有用又简洁的回复，从而避免单一有用性奖励导致的冗长陷阱。

5. 比较不同的 KL 系数（KL Coefficients）。分别使用 beta=0.001（过低，导致奖励欺骗）、beta=0.02（标准值）和 beta=0.5（过高，无法学习）运行 PPO。绘制每种情况下的奖励曲线和 KL 曲线。beta=0.02 的运行结果应显示奖励稳步提升，且 KL 散度保持在有界范围内。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 基于人类反馈的强化学习 (RLHF) | “使用人类反馈进行训练” | 基于人类反馈的强化学习：一个包含三个阶段（监督微调 SFT、奖励模型、PPO）的流程，利用人类偏好信号来优化语言模型的输出 |
| 奖励模型 (Reward Model) | “给回复打分的模型” | 带有标量输出头的 Transformer 模型，使用 Bradley-Terry 损失函数在成对人类偏好数据上进行训练 |
| Bradley-Terry 模型 | “比较模型” | 一种概率模型，其中 P(A > B) = sigmoid(score(A) - score(B))，用于将成对偏好转化为一致的评分函数 |
| 近端策略优化 (PPO) | “强化学习算法” | 近端策略优化：通过裁剪更新幅度来防止训练不稳定，同时更新策略以最大化奖励 |
| KL 散度 (KL Divergence) | “两个分布之间的差异程度” | 用于衡量策略模型与参考模型的词元分布差异的指标——作为惩罚项使用，以防止奖励刷分 (Reward Hacking) |
| KL 惩罚 (KL Penalty) | “约束模型的缰绳” | 从奖励信号中减去 Beta * KL(policy \|\| reference)——防止策略模型偏离监督微调 (SFT) 检查点过远 |
| 奖励刷分 (Reward Hacking) | “钻奖励机制的空子” | 策略模型利用奖励模型的缺陷来寻找退化的（无意义但）高奖励输出，而非真正提升生成质量 |
| 偏好对 (Preference Pair) | “A 和 B 哪个更好？” | 由（提示词，优选回复，拒绝回复）组成的训练样本——RLHF 训练数据的基本单元 |
| 参考模型 (Reference Model) | “冻结的 SFT 检查点” | SFT 模型的副本，其权重在训练过程中保持不变——用作计算 KL 散度的基准锚点 |

## 延伸阅读

- [Ouyang 等人, 2022 -- “使用人类反馈训练语言模型遵循指令” (InstructGPT)](https://arxiv.org/abs/2203.02155) -- 使 RLHF 在大语言模型上具备实用性的开创性论文
- [Schulman 等人, 2017 -- “近端策略优化算法”](https://arxiv.org/abs/1707.06347) -- OpenAI 提出的原始 PPO 论文
- [Bai 等人, 2022 -- “使用基于人类反馈的强化学习训练有益且无害的助手”](https://arxiv.org/abs/2204.05862) -- Anthropic 的 RLHF 论文，详细分析了奖励刷分与 KL 惩罚机制
- [Stiennon 等人, 2020 -- “通过人类反馈学习文本摘要”](https://arxiv.org/abs/2009.01325) -- 将 RLHF 应用于摘要任务，证明奖励模型能够捕捉细微的质量评判
- [Christiano 等人, 2017 -- “基于人类偏好的深度强化学习”](https://arxiv.org/abs/1706.03741) -- 从人类比较中学习奖励函数的奠基性工作