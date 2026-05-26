# 指令微调（Instruction Tuning / Supervised Fine-Tuning, SFT）

> 基础模型（Base Model）仅用于预测下一个词元（Token）。仅此而已。它不会遵循指令、回答问题，也不会拒绝有害请求。SFT 是连接词元预测器与实用助手之间的桥梁。你曾交互过的每一个模型——Claude、GPT、Llama Chat——都经历了这一步。

**类型：** 构建
**语言：** Python（含 numpy）
**前置条件：** 第 10 阶段，第 04 课（预训练迷你 GPT）
**时长：** 约 90 分钟

## 学习目标

- 实现监督微调（Supervised Fine-Tuning, SFT），将基础语言模型转化为能够遵循指令的助手
- 使用包含系统（system）、用户（user）和助手（assistant）角色的对话模板（Chat Template）格式化训练数据，并对非助手词元进行损失掩码（Mask Loss）处理
- 阐明为何需要 SFT：基础模型倾向于续写文本，而非回答问题
- 通过在预留指令集（Held-out Instruction Set）上对比基础模型与微调模型（Fine-tuned Model）的回复，评估 SFT 的质量

## 问题背景

你在第 04 课中训练了一个模型。给定一段序列，它能够预测下一个词元。输入“The transformer architecture”，它可能会续写为“has revolutionized natural language processing.”。对于一个词元预测器而言，这已经相当出色。

现在试试这个：输入“What is the capital of France?”。基础模型不会回答“Paris”。它只会延续模式。它可能会生成“What is the capital of Germany? What is the capital of Spain?”，因为它从包含问题列表的文档中学习到了这种模式。或者它可能会生成“is a question that many people ask”，因为这在词元续写上看起来合理。该模型根本没有*回答*的概念，它只知道*续写*。

这就是 GPT-3（基础模型，2020 年 6 月发布）与 ChatGPT（指令微调模型，2022 年 11 月发布）之间的差距。架构相同，预训练（Pre-training）相同。区别在于那 2 万到 10 万条精心构建的（指令，回复）数据对，它们教会了模型遵循对话模式。

斯坦福 Alpaca 项目证明了你并不需要数百万条样本。2023 年 3 月，他们仅使用 GPT-3.5 生成的 52,000 条指令-回复数据对（Instruction-Response Pairs），就对 Llama 7B 进行了微调（Fine-tuning）。总成本：600 美元。结果是一个能够遵循指令、回答问题并进行对话的聊天机器人。虽然不及 ChatGPT，但仅花费 600 美元和几小时的训练就能达到如此接近的效果，令人震惊。

Meta 的 Llama 2 Chat 在初始 SFT 阶段仅使用了约 27,000 条高质量样本。核心洞见在于：质量重于数量。由熟练标注人员撰写的 27,000 条样本，其效果远超从互联网上抓取的 100 万条噪声样本（Noisy Examples）。

## 核心概念

### SFT 的实际作用

监督微调 (Supervised Fine-Tuning, SFT) 延续了预训练 (Pre-training) 中的相同训练循环——前向传播 (Forward Pass)、计算损失 (Compute Loss)、反向传播 (Backward Pass)、更新权重 (Update Weights)——但使用的是不同类型的数据。你不再使用原始文本进行训练，而是使用结构化的对话数据：

{
  "system": "You are a helpful assistant.",
  "user": "What is the capital of France?",
  "assistant": "The capital of France is Paris."
}

模型其实已经知道巴黎是法国的首都。这是在预训练阶段通过维基百科、教科书和网页学习到的。SFT 并不会教给模型新的事实，而是教给它一种新的*行为*：看到问题时给出答案，看到指令时生成补全内容，看到有害请求时进行拒绝。

可以这样理解：预训练赋予模型知识，而 SFT 赋予模型“礼仪”。

### 数据格式

业界主要流行三种格式。它们使用不同的分隔符来编码相同的信息——即谁说了什么。

**Alpaca 格式**（斯坦福大学，2023 年 3 月）：

{
  "instruction": "Summarize the following article in 3 sentences.",
  "input": "The European Central Bank raised interest rates...",
  "output": "The ECB increased rates by 25 basis points..."
}

该格式简单且被广泛使用。`input` 字段是可选的——许多指令并不需要额外的上下文。斯坦福大学以该格式发布了 52,000 条示例，这些数据由 GPT-3.5 生成，成本仅为 600 美元。此举直接推动了开源指令微调 (Instruction Tuning) 运动的发展。

**ShareGPT 格式**（社区，2023 年）：

{
  "conversations": [
    {"from": "system", "value": "You are a helpful assistant."},
    {"from": "human", "value": "What causes tides?"},
    {"from": "gpt", "value": "Tides are caused by the gravitational pull of the Moon..."},
    {"from": "human", "value": "How often do they occur?"},
    {"from": "gpt", "value": "Most coastal areas experience two high tides and two low tides per day..."}
  ]
}

支持多轮对话。按照惯例，`"from"` 字段使用 `"human"` 和 `"gpt"`，与实际使用的模型无关。Vicuna 模型就是基于 70,000 条从用户分享的 ChatGPT 对话记录中抓取的 ShareGPT 数据训练而成的。

**ChatML 格式**（OpenAI，被众多开源模型采用）：

<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>

使用特殊标记 (Special Tokens)（`<|im_start|>`、`<|im_end|>`）来划分角色。在微调过程中，这些标记会被添加到分词器 (Tokenizer) 的词表中。Qwen、Yi 以及许多其他模型都采用了 ChatML 格式。

这三种格式的目的完全一致：告诉模型“这是指令，这是回复，请学习这种模式。”

### 为什么它有效

模型在预训练阶段已经掌握了语言知识。它已经见过数十亿条“问题后接答案”、“指令后接补全”以及“人类对话”的示例。这些模式早已编码在模型的权重中。

SFT 的作用是将这种潜在能力集中激发出来。模型不再需要根据上下文去猜测自己应该回答问题还是续写文档，SFT 会显式地针对对话模式进行训练。经过几千个示例的训练后，模型就会学会：当看到助手角色标记时，生成有帮助的回复。

这就是为什么 27,000 条示例就足够了。你并不是在教模型英语，也不是在教它世界知识。你只是在教它一个简单的行为：响应指令。相关的知识早已存在于模型之中。

### 掩码损失 (Masked Loss)

这是 SFT 中最重要的技术细节，但大多数教程都会忽略它。

在预训练阶段，你需要对每个词元 (Token) 计算损失。模型学习预测序列中的每一个下一个词元。而在 SFT 阶段，你只对*回复*部分的词元计算损失。指令部分的词元仅用于提供上下文，即使模型“错误预测”了它们，也不会受到惩罚。

为什么？因为你并不希望模型学会*生成*指令，而是希望它学会*响应*指令。如果对指令词元计算损失，就相当于在训练模型去预测“法国的首都是哪里？”，仿佛它才是提问者。这会浪费梯度信号，并可能导致模型对其角色产生混淆。

在实际操作中，你会创建一个损失掩码 (Loss Mask)：回复词元标记为 1，指令词元标记为 0。在求平均之前，将每个词元的损失与该掩码相乘。

Tokens:    [SYS] You are helpful [USER] What is the capital? [ASST] Paris is the capital [EOS]
Loss mask:   0    0    0     0      0     0   0  0     0       1     1    1   1     1      1

只有 `[ASST]` 之后的词元才会对损失产生贡献。在前向传播过程中，模型会看到完整的对话（它需要指令来生成正确的回复），但仅会根据其预测回复的准确程度来更新权重。

### 训练超参数 (Training Hyperparameters)

SFT 使用的超参数与预训练截然不同。你不是在从头开始训练，而是在调整一个已经具备良好基础的模型。

| 参数 (Parameter) | 预训练 (Llama 2 7B) | SFT (Llama 2 Chat) |
|-----------|---------------------------|---------------------|
| 学习率 (Learning rate) | 3e-4 (峰值) | 2e-5 |
| 训练轮数 (Epochs) | 1（数据单次遍历） | 2 |
| 批次大小 (Batch size) | 400 万词元 | 64 个示例 |
| 预热步数 (Warmup steps) | 2,000 | 0-100 |
| 权重衰减 (Weight decay) | 0.1 | 0.0-0.1 |
| 数据规模 (Data size) | 2 万亿词元 | 27,000 个示例 |

SFT 的学习率比预训练低 15 倍。这一点至关重要。在微调阶段使用较高的学习率会破坏预训练获得的知识。模型会“遗忘”已学内容，并对小规模微调数据集产生过拟合 (Overfitting)。这就是灾难性遗忘 (Catastrophic Forgetting)。

2 个 Epoch 意味着模型会看到每个训练示例两次。在小数据集上超过 3 个 Epoch 会导致死记硬背——模型开始逐字复现训练示例，而不是进行泛化。

### 灾难性遗忘 (Catastrophic Forgetting)

微调可能会破坏模型的通用能力。如果在指令遵循数据上训练时间过长，模型会丧失编写代码、进行数学运算或生成创意文本的能力。它会变得非常擅长处理训练数据的特定格式，但在其他所有任务上表现糟糕。

三种缓解策略：

1. **较低的学习率。** 1e-5 到 5e-5。较小的更新幅度意味着对预训练特征的破坏更少。
2. **较短的训练时间。** 1-3 个 Epoch。在模型过拟合之前停止训练。
3. **混合预训练数据。** Llama 2 Chat 将一小部分（2-5%）原始预训练数据混合到了 SFT 数据集中。这能在模型学习新指令遵循行为的同时，“提醒”它保持原有的通用能力。

### 实际数据参考

在单张 NVIDIA A100 80GB GPU 上，使用 10,000 条高质量指令对微调一个 7B 模型大约需要 1 小时。具体计算如下：

- 10,000 个示例 × 平均 512 个词元 = 512 万词元
- 2 个 Epoch = 总计 1024 万词元
- A100 微调 7B 模型的吞吐量：约 3,000 词元/秒
- 1024 万 / 3,000 ≈ 3,400 秒 ≈ 57 分钟

对于我们的小型 GPT（4 层，128 维）而言，训练几乎是瞬间完成的。这里的重点在于理解其运行机制，而非具体的规模。

graph TD
    subgraph SFT["Supervised Fine-Tuning Pipeline"]
        direction TB
        D["Instruction Dataset\n(10K-100K examples)"] --> F["Format into\n(instruction, response) pairs"]
        F --> T["Tokenize with\nchat template"]
        T --> M["Create loss mask\n(1 for response, 0 for instruction)"]
        M --> FW["Forward pass\n(full sequence)"]
        FW --> L["Compute masked loss\n(response tokens only)"]
        L --> BW["Backward pass"]
        BW --> U["Update weights\n(lr=2e-5, 1-3 epochs)"]
    end

    subgraph Base["Base Model\n(pre-trained)"]
        B1["Knows language"]
        B2["Knows facts"]
        B3["No conversation pattern"]
    end

    subgraph Chat["Chat Model\n(after SFT)"]
        C1["Knows language"]
        C2["Knows facts"]
        C3["Follows instructions"]
    end

    Base --> SFT --> Chat

    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style L fill:#1a1a2e,stroke:#e94560,color:#fff
    style B3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style C3 fill:#1a1a2e,stroke:#51cf66,color:#fff


## 构建

### 步骤 1：指令数据集 (Instruction Dataset)

创建一个合成指令数据集。在实际生产环境中，Scale AI 和 Anthropic 等公司会雇佣人工标注员来编写这些数据。为了演示格式，我们将通过编程方式生成它们。

import numpy as np

INSTRUCTION_DATA = [
    {
        "instruction": "What is the capital of France?",
        "response": "The capital of France is Paris."
    },
    {
        "instruction": "Explain gravity in one sentence.",
        "response": "Gravity is the force that attracts objects with mass toward each other."
    },
    {
        "instruction": "Write a haiku about the ocean.",
        "response": "Waves crash on the shore, salt and foam beneath the sun, endless blue expanse."
    },
    {
        "instruction": "What is 15 multiplied by 7?",
        "response": "15 multiplied by 7 is 105."
    },
    {
        "instruction": "Name three programming languages.",
        "response": "Three programming languages are Python, Rust, and TypeScript."
    },
    {
        "instruction": "Summarize photosynthesis.",
        "response": "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen."
    },
    {
        "instruction": "What year did World War II end?",
        "response": "World War II ended in 1945."
    },
    {
        "instruction": "Define machine learning.",
        "response": "Machine learning is a field where algorithms learn patterns from data to make predictions."
    },
]

8 个样本非常少。斯坦福的 Alpaca 模型使用了 52,000 个样本。但无论你使用 8 个还是 52,000 个，其底层机制是完全相同的：分词 (Tokenize)、掩码 (Mask)，并且仅对回复部分计算损失 (Loss)。

### 步骤 2：使用对话模板 (Chat Template) 进行分词

将指令-回复对转换为带有特殊角色标记的 token 序列。这些标记用于告知模型指令的结束位置和回复的起始位置。

SPECIAL_TOKENS = {
    "INST_START": 253,
    "INST_END": 254,
    "RESP_START": 255,
}


def tokenize_instruction_pair(instruction, response, vocab_size=256):
    inst_tokens = list(instruction.encode("utf-8"))
    resp_tokens = list(response.encode("utf-8"))

    inst_tokens = [min(t, vocab_size - 4) for t in inst_tokens]
    resp_tokens = [min(t, vocab_size - 4) for t in resp_tokens]

    tokens = (
        [SPECIAL_TOKENS["INST_START"]]
        + inst_tokens
        + [SPECIAL_TOKENS["INST_END"]]
        + [SPECIAL_TOKENS["RESP_START"]]
        + resp_tokens
    )

    return tokens


def create_loss_mask(tokens):
    mask = np.zeros(len(tokens), dtype=np.float32)
    in_response = False

    for i, token in enumerate(tokens):
        if token == SPECIAL_TOKENS["RESP_START"]:
            in_response = True
            continue
        if in_response:
            mask[i] = 1.0

    return mask

损失掩码 (Loss Mask) 对指令 token 全为 0，对回复 token 全为 1。`RESP_START` token 本身的掩码值为 0，因为它只是一个分隔符，不属于回复内容的一部分。

### 步骤 3：掩码交叉熵损失 (Masked Cross-Entropy Loss)

标准的交叉熵损失 (Cross-Entropy Loss)，但会乘以损失掩码。只有回复 token 会对梯度 (Gradient) 产生贡献。

def masked_cross_entropy_loss(logits, targets, loss_mask):
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    mask_flat = loss_mask.reshape(-1)

    max_logits = logits_flat.max(axis=-1, keepdims=True)
    log_softmax = logits_flat - max_logits - np.log(
        np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
    )

    per_token_loss = -log_softmax[np.arange(len(targets_flat)), targets_flat]

    masked_loss = per_token_loss * mask_flat
    num_response_tokens = mask_flat.sum()
    if num_response_tokens == 0:
        return 0.0
    loss = masked_loss.sum() / num_response_tokens

    return loss

分母是 `num_response_tokens`（回复 token 数量），而不是 `seq_len`（序列长度 (Sequence Length)）。如果除以总序列长度，较长的指令会稀释梯度信号。除以回复 token 数量可以确保无论指令长短，每个回复 token 都具有相同的权重。

### 步骤 4：SFT 训练循环 (SFT Training Loop)

复用第 04 课中的 MiniGPT。该训练循环与预训练 (Pre-training) 几乎完全相同，但增加了指令格式化和掩码损失。

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "04-pre-training-mini-gpt", "code"))
from main import MiniGPT, LayerNorm, FeedForward, MultiHeadAttention, TransformerBlock, Embedding


def sft_train(model, dataset, num_epochs=2, lr=2e-5, seq_len=64):
    formatted_data = []
    for example in dataset:
        tokens = tokenize_instruction_pair(example["instruction"], example["response"])
        mask = create_loss_mask(tokens)
        formatted_data.append((tokens, mask))

    print(f"SFT Training: {len(formatted_data)} examples, {num_epochs} epochs, lr={lr}")
    print(f"Total tokens: {sum(len(t) for t, _ in formatted_data):,}")
    print()

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        indices = np.random.permutation(len(formatted_data))

        for idx in indices:
            tokens, mask = formatted_data[idx]

            if len(tokens) < 3:
                continue
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
                mask = mask[:seq_len]

            input_ids = np.array(tokens[:-1]).reshape(1, -1)
            target_ids = np.array(tokens[1:]).reshape(1, -1)
            loss_mask = np.array(mask[1:]).reshape(1, -1)

            logits = model.forward(input_ids)
            loss = masked_cross_entropy_loss(logits, target_ids, loss_mask)

            batch_size, s_len, v_size = logits.shape
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            dlogits = probs.copy()
            dlogits[np.arange(batch_size)[:, None], np.arange(s_len), target_ids] -= 1.0

            mask_expanded = loss_mask[:, :, np.newaxis]
            num_resp = loss_mask.sum()
            if num_resp > 0:
                dlogits = dlogits * mask_expanded / num_resp

            for block in model.blocks:
                block.ffn.W1 -= lr * np.random.randn(*block.ffn.W1.shape) * 0.01
                block.ffn.W2 -= lr * np.random.randn(*block.ffn.W2.shape) * 0.01
                block.ffn.b1 -= lr * np.random.randn(*block.ffn.b1.shape) * 0.01
                block.ffn.b2 -= lr * np.random.randn(*block.ffn.b2.shape) * 0.01

            epoch_loss += loss
            num_batches += 1
            losses.append(loss)

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

    return model, losses

学习率 (Learning Rate) 设置为 2e-5，与 Llama 2 Chat 保持一致。相比之下，预训练使用的学习率为 3e-4，缩小了 15 倍。梯度经过掩码处理：指令 token 产生的梯度为零。只有回复 token 会推动权重更新。

### 步骤 5：对比基座模型 (Base Model) 与 SFT 模型

监督微调 (Supervised Fine-Tuning) 的核心目的在于改变模型的行为。我们可以通过对比模型对指令格式输入与原始文本续写的响应方式来评估这一变化。

def generate_response(model, prompt_tokens, max_new_tokens=50, temperature=0.8):
    tokens = list(prompt_tokens)
    seq_len = model.embedding.pos_embed.shape[0]

    for _ in range(max_new_tokens):
        context = np.array(tokens[-seq_len:]).reshape(1, -1)
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


def evaluate_instruction_following(model, instructions):
    print("Evaluating instruction following:")
    print("-" * 50)

    for instruction in instructions:
        tokens = (
            [SPECIAL_TOKENS["INST_START"]]
            + [min(t, 252) for t in list(instruction.encode("utf-8"))]
            + [SPECIAL_TOKENS["INST_END"]]
            + [SPECIAL_TOKENS["RESP_START"]]
        )

        output = generate_response(model, tokens, max_new_tokens=30, temperature=0.6)
        response_start = len(tokens)
        response_tokens = output[response_start:]
        response_bytes = bytes([t for t in response_tokens if t < 128])
        response_text = response_bytes.decode("utf-8", errors="replace")

        print(f"  Q: {instruction}")
        print(f"  A: {response_text[:80]}")
        print()

在一个仅使用 8 个样本训练的微型模型上，生成的回复可能没有实际意义。这是符合预期的。关键在于*结构*：模型学会了在回复标记之后生成输出，而不是继续生成更多的指令。

### 步骤 6：评估灾难性遗忘 (Catastrophic Forgetting)

对比 SFT 前后模型的下一个 token 预测能力。如果 SFT 损害了模型的通用能力，原始文本上的损失将会上升。

def measure_forgetting(model, test_text, seq_len=64):
    tokens = np.array(list(test_text.encode("utf-8")[:512]))

    total_loss = 0.0
    num_windows = 0

    for start in range(0, len(tokens) - seq_len - 1, seq_len):
        input_ids = tokens[start:start + seq_len].reshape(1, -1)
        target_ids = tokens[start + 1:start + seq_len + 1].reshape(1, -1)

        logits = model.forward(input_ids)

        batch, s_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_ids.reshape(-1)

        max_logits = logits_flat.max(axis=-1, keepdims=True)
        log_softmax = logits_flat - max_logits - np.log(
            np.exp(logits_flat - max_logits).sum(axis=-1, keepdims=True)
        )

        loss = -log_softmax[np.arange(len(targets_flat)), targets_flat].mean()
        total_loss += loss
        num_windows += 1

    return total_loss / max(num_windows, 1)

在实际的微调 (Fine-tuning) 过程中，你需要在整个训练期间持续跟踪这一指标。如果原始文本损失上升超过 10-15%，说明你的 SFT 策略过于激进。此时应降低学习率或减少训练轮数 (Epochs)。

## 实践应用

### 完整监督微调（Supervised Fine-Tuning, SFT）流水线演示

if __name__ == "__main__":
    np.random.seed(42)

    test_text = """The transformer architecture processes sequences through self-attention.
Each layer applies multi-head attention followed by a feedforward network.
Residual connections and layer normalization stabilize deep networks.
The model learns to predict the next token given all previous tokens."""

    print("=" * 70)
    print("INSTRUCTION TUNING (SFT) DEMO")
    print("=" * 70)
    print()

    model = MiniGPT(
        vocab_size=256, embed_dim=128, num_heads=4,
        num_layers=4, max_seq_len=128, ff_dim=512
    )
    print(f"Model: {model.count_parameters():,} parameters")
    print(f"Config: 4 layers, 4 heads, 128 dims (mini GPT from Lesson 04)")
    print()

    print("PRE-SFT: Measuring base model loss on raw text")
    base_loss = measure_forgetting(model, test_text)
    print(f"  Base model loss: {base_loss:.4f}")
    print()

    print("=" * 70)
    print("SFT TRAINING")
    print("=" * 70)

    model, losses = sft_train(
        model, INSTRUCTION_DATA, num_epochs=3, lr=2e-5, seq_len=128
    )

    print()
    print("POST-SFT: Measuring fine-tuned model loss on raw text")
    sft_loss = measure_forgetting(model, test_text)
    print(f"  SFT model loss: {sft_loss:.4f}")
    print(f"  Change: {((sft_loss - base_loss) / base_loss * 100):+.1f}%")
    if abs(sft_loss - base_loss) / base_loss < 0.15:
        print("  Minimal forgetting (< 15% change)")
    else:
        print("  Significant forgetting detected")
    print()

    print("=" * 70)
    print("INSTRUCTION FOLLOWING EVALUATION")
    print("=" * 70)
    print()

    test_instructions = [
        "What is the capital of France?",
        "Name a programming language.",
        "Define gravity.",
    ]
    evaluate_instruction_following(model, test_instructions)

    print("=" * 70)
    print("DATA FORMAT EXAMPLES")
    print("=" * 70)
    print()

    for i, example in enumerate(INSTRUCTION_DATA[:3]):
        tokens = tokenize_instruction_pair(example["instruction"], example["response"])
        mask = create_loss_mask(tokens)
        resp_count = int(mask.sum())
        total_count = len(tokens)
        print(f"  Example {i + 1}: {total_count} tokens, {resp_count} response tokens ({resp_count/total_count:.0%} of sequence)")
        print(f"    Instruction: {example['instruction']}")
        print(f"    Response: {example['response']}")
        print()

    print("=" * 70)
    print("TRAINING LOSS CURVE")
    print("=" * 70)
    print()

    if losses:
        window = max(1, len(losses) // 5)
        for i in range(0, len(losses), window):
            chunk = losses[i:i + window]
            avg = sum(chunk) / len(chunk)
            print(f"  Steps {i:3d}-{i + len(chunk) - 1:3d}: avg loss = {avg:.4f}")

## 发布上线

本课程将生成 `outputs/prompt-sft-data-curator.md` —— 一个用于辅助你设计和整理监督微调（Supervised Fine-Tuning, SFT）指令数据集的提示词（Prompt）。针对目标能力（如代码生成、数学推理、对话交互），它会生成一份包含格式规范、质量标准和多样性要求的数据收集计划。

## 练习

1. 添加系统提示词（System Prompt）支持。修改 `tokenize_instruction_pair` 以接收系统消息，并将其拼接在指令之前。创建 5 个包含不同系统提示词（如“你是一位诗人”、“你是一位数学导师”）的示例，并验证模型在训练过程中能否正确读取不同的系统提示词。

2. 实现数据混合（Data Mixing）。创建一个函数，接收 SFT 数据集和原始文本语料库，生成训练批次（Training Batches），其中 5% 的样本为原始文本（不应用掩码），95% 为指令对（应用掩码）。运行 3 个训练轮次（Epochs），并将遗忘指标（Forgetting Metrics）与纯 SFT 训练的结果进行对比。

3. 构建数据质量评分器（Data Quality Scorer）。针对每个指令-回复对（Instruction-Response Pair），计算以下指标：(a) 回复的 Token 长度，(b) 指令与回复的长度比例，(c) 词汇多样性（唯一 Token 数 / 总 Token 数）。过滤掉回复长度小于 10 个 Token 或多样性低于 0.3 的样本。展示数据过滤对最终损失（Loss）的影响。

4. 实现多轮对话训练（Multi-turn Conversation Training）。扩展分词（Tokenization）逻辑以处理 3 轮对话（用户-助手-用户-助手-用户-助手）。损失掩码（Loss Mask）应覆盖所有三个助手回复轮次。通过打印单个示例的 Token 与掩码对齐情况，验证掩码是否正确。

5. 对比学习率（Learning Rate）。使用 lr=1e-4、lr=2e-5 和 lr=1e-6 分别对同一模型进行三次训练。绘制损失曲线（Loss Curve）。lr=1e-4 的训练应显示初期快速下降但最终损失较高（过拟合/Overfitting）。lr=1e-6 的训练应几乎无变化。lr=2e-5 的训练应达到最佳平衡点。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 监督微调 (Supervised Fine-Tuning, SFT) | “在对话数据上进行微调” | 监督微调：在（指令，回复）数据对上继续训练，且损失 (loss) 仅针对回复部分的词元 (token) 进行计算 |
| 指令微调 (Instruction Tuning) | “教模型遵循指令” | 在明确的指令-回复数据对上进行训练，使基础模型学习对话模式，而非获取新知识 |
| 损失掩码 (Loss Masking) | “忽略提示词 (prompt)” | 将指令部分词元的损失设为零，使梯度 (gradient) 仅从回复词元的预测中反向传播 |
| ChatML | “聊天标记语言” | 一种使用 `<\|im_start\|>` 和 `<\|im_end\|>` 分隔符的词元格式，用于在对话数据中标记说话人角色 |
| Alpaca 格式 | “斯坦福的格式” | 一种包含 `instruction`/`input`/`output` 字段的 JSON 格式，用于斯坦福大学以 600 美元成本生成的 52K 条 GPT-3.5 示例数据 |
| 灾难性遗忘 (Catastrophic Forgetting) | “模型变笨了” | 微调会破坏预训练能力，因为梯度更新会用特定任务的模式覆盖模型的通用知识 |
| 权重绑定 (Weight Tying) | “共享嵌入层 (embeddings)” | 在输入词元嵌入层和输出预测头使用相同的权重矩阵，以节省参数量并提升生成连贯性 |
| 聊天模板 (Chat Template) | “提示词的格式化方式” | 为模型构建对话结构的特定词元序列（包含角色标记、分隔符等） |

## 延伸阅读

- [Ouyang 等人, 2022 -- "Training language models to follow instructions with human feedback" (InstructGPT)](https://arxiv.org/abs/2203.02155) -- OpenAI 引入指令微调与基于人类反馈的强化学习 (RLHF) 的开创性论文
- [Taori 等人, 2023 -- "Stanford Alpaca: An Instruction-following LLaMA Model"](https://github.com/tatsu-lab/stanford_alpaca) -- 仅花费 600 美元生成 52K 条指令示例，证明了监督微调在小数据集上同样有效
- [Touvron 等人, 2023 -- "Llama 2: Open Foundation and Fine-Tuned Chat Models"](https://arxiv.org/abs/2307.09288) -- Meta 采用 27K 条高质量示例构建的 SFT + RLHF 训练流水线
- [Chiang 等人, 2023 -- "Vicuna: An Open-Source Chatbot Impressing GPT-4"](https://lmsys.org/blog/2023-03-30-vicuna/) -- 基于 70K 条 ShareGPT 对话数据进行训练
- [Zhou 等人, 2023 -- "LIMA: Less Is More for Alignment"](https://arxiv.org/abs/2305.11206) -- 证明仅需 1,000 条精心筛选的示例，即可达到与大规模数据集监督微调相当的效果