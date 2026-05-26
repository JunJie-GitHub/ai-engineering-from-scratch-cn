# GPT —— 因果语言建模 (Causal Language Modeling)

> BERT 能够看到上下文的两端，而 GPT 只能看到过去。三角形掩码 (triangle mask) 是现代人工智能中最具影响力的一行代码。

**类型：** 构建
**语言：** Python
**前置知识：** 第 7 阶段 · 02（自注意力机制 (Self-Attention)）、第 7 阶段 · 05（完整 Transformer (Full Transformer)）、第 7 阶段 · 06（BERT）
**预计耗时：** 约 75 分钟

## 核心问题

语言模型只需回答一个问题：给定前 `t-1` 个词元 (token)，第 `t` 个词元的概率分布是什么？基于这一信号——即下一个词元预测 (next-token prediction)——进行训练，你就能得到一个可以逐词元生成任意文本的模型。

若要在整个序列上进行端到端的并行训练，必须确保每个位置的预测仅依赖于其之前的位置。否则，模型会直接“偷看”答案，从而轻易作弊。

因果掩码 (causal mask) 正是为此而生。它是一个由 `-inf` 值构成的上三角矩阵，在 softmax 操作前被加到注意力分数 (attention scores) 上。经过 softmax 处理后，这些位置的概率值会变为 0。每个位置只能关注自身及其之前的位置。由于只需对整个序列应用一次该掩码，你就能在一次前向传播 (forward pass) 中并行获得 N 个下一个词元的预测结果。

GPT-1 (2018)、GPT-2 (2019)、GPT-3 (2020)、GPT-4 (2023)、GPT-5 (2024)、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi —— 它们本质上都是仅解码器因果 Transformer (decoder-only causal Transformer)，共享相同的核心循环。区别仅在于模型规模更大、训练数据更优质，以及基于人类反馈的强化学习 (RLHF) 效果更好。

## 核心概念

![因果掩码（Causal Mask）创建三角形注意力矩阵（Attention Matrix）](../assets/causal-attention.svg)

### 掩码（Mask）

对于长度为 `N` 的序列，构建一个 `N × N` 的矩阵：

M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i

在 Softmax 操作之前，将 `M` 加到原始注意力分数上。由于 `exp(-inf) = 0`，被掩码的位置权重为零。注意力矩阵的每一行仅表示对先前位置的概率分布。

实现成本：仅需调用一次 `torch.tril()`。计算耗时：纳秒级。对领域的影响：重塑了一切。

### 并行训练，串行推理

训练：将整个 `(N, d_model)` 序列进行一次前向传播（Forward Pass），计算 N 个交叉熵损失（Cross-Entropy Loss）（每个位置一个），求和后进行反向传播（Backpropagation）。沿序列维度并行计算。这正是 GPT 训练能够高效扩展的原因——你可以在一次 GPU 前向传递中处理包含 100 万个 Token 的批次。

推理：逐 Token 生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV 缓存（KV Cache，第 12 课）会保存 `t1…tn` 的隐藏状态（Hidden States），避免每一步都重新计算。但在推理时，串行深度等于输出长度。这就是自回归（Autoregressive）带来的额外开销，也是解码（Decoding）成为所有大语言模型（LLM）延迟瓶颈的原因。

### 损失函数——错位一位（Shift-by-One）

给定 Token 序列 `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对于每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])` 并求和。这就是整个序列的交叉熵损失。

你所熟知的每一个 Transformer 语言模型（Transformer LM）都基于此损失函数进行训练。预训练（Pre-training）、微调（Fine-tuning）、监督微调（SFT）——损失函数相同，仅数据不同。

### 解码策略（Decoding Strategies）

训练完成后，采样策略的选择往往比人们想象的更为重要。

| 方法 | 作用机制 | 适用场景 |
|--------|--------------|-------------|
| 贪婪解码（Greedy） | 每一步取 Argmax | 确定性任务、代码补全 |
| 温度采样（Temperature） | 将 Logits 除以 T 后采样 | 创意类任务，T 值越高多样性越强 |
| Top-k 采样 | 仅从概率最高的 k 个 Token 中采样 | 过滤低概率尾部 |
| Top-p 采样（核采样，Nucleus） | 从累积概率 ≥ p 的最小 Token 集合中采样 | 2020 年后的默认选项；自适应分布形状 |
| Min-p 采样 | 保留满足 `p > min_p * max_p` 的 Token | 2024 年起流行；比 Top-p 更能有效过滤长尾分布 |
| 投机解码（Speculative Decoding） | 草稿模型（Draft Model）生成 N 个 Token，大模型进行验证 | 在保持同等质量的前提下，降低 2–3 倍延迟 |

到 2026 年，对于开源权重模型（Open-weights Models），Min-p 配合 0.7 的温度值是一个合理的默认配置。而投机解码已成为任何生产级推理栈（Production Inference Stack）的必备基础。

### “GPT 配方”为何奏效

1. **仅解码器架构（Decoder-only）。** 无编码器（Encoder）开销。每层仅需一次注意力机制（Attention）与前馈神经网络（FFN）的前向传递。
2. **规模扩展（Scaling）。** 1.24 亿 → 15 亿 → 1750 亿 → 万亿级。Chinchilla 缩放定律（Scaling Laws，第 13 课）指导你如何高效分配算力。
3. **上下文学习（In-context Learning）。** 在 60 亿至 130 亿参数规模时涌现。模型无需微调即可遵循少样本（Few-shot）示例。
4. **基于人类反馈的强化学习（RLHF）。** 基于人类偏好的后期训练（Post-training）将原始预训练文本转化为聊天助手。
5. **前置归一化（Pre-norm）+ 旋转位置编码（RoPE）+ SwiGLU 激活函数。** 保障大规模训练的稳定性。

自 GPT-2 以来，核心架构并未发生太大变化。真正带来突破的进展均集中在数据、模型规模以及后期训练上。

## 构建它

### 步骤 1：因果掩码 (Causal Mask)

参见 `code/main.py`。只需一行代码：

def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]

在 Softmax 之前将其加到注意力分数 (Attention Scores) 上。这就是整个机制。

### 步骤 2：一个类似 GPT 的双层模型

堆叠两个解码器块 (Decoder Blocks)（包含掩码自注意力 (Masked Self-Attention) 和前馈神经网络 (Feed-Forward Network, FFN)，无交叉注意力 (Cross-Attention)）。添加词元嵌入 (Token Embedding)、位置编码 (Positional Encoding) 以及反嵌入层 (Unembedding)（与词元嵌入矩阵权重共享——这是自 GPT-2 以来的标准技巧）。

### 步骤 3：端到端的下一个词元预测 (Next-Token Prediction)

在一个包含 20 个词元的玩具词表 (Toy Vocabulary) 上，在每个位置生成逻辑值 (Logits)。针对偏移一位的目标序列计算交叉熵损失 (Cross-Entropy Loss)。无需计算梯度——这仅是一次前向传播 (Forward Pass) 的健全性检查 (Sanity Check)。

### 步骤 4：采样 (Sampling)

实现贪婪采样 (Greedy Sampling)、温度采样 (Temperature Sampling)、Top-k、Top-p 和 Min-p 策略。在固定的提示词 (Prompt) 上分别运行并比较输出结果。一个采样函数只需 10 行代码。

## 使用它

PyTorch 2026 年惯用写法：

from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))

在底层，`generate()` 会执行前向传播，提取最后一个位置的逻辑值，采样下一个词元，将其追加到序列中，并重复此过程。每个生产级大语言模型推理栈 (LLM Inference Stack)（如 vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都实现了相同的循环，并进行了深度优化——包括批量预填充 (Batched Prefill)、连续批处理 (Continuous Batching)、键值缓存分页 (KV Cache Paging) 和投机解码 (Speculative Decoding)。

**GPT 与 BERT 的一句话对比：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。损失函数 (Loss Function) 的设计决定了模型是否具备生成能力。

## 交付它

参见 `outputs/skill-sampling-tuner.md`。该技能 (Skill) 会为新的生成任务挑选采样参数，并在需要确定性解码 (Deterministic Decoding) 时发出标记。

## 练习

1. **简单。** 运行 `code/main.py`，验证经过 Softmax 后的因果注意力矩阵 (Causal Attention Matrix) 是否为下三角矩阵。抽查验证：第 3 行应仅在 0–3 列有权重。
2. **中等。** 实现宽度为 4 的束搜索 (Beam Search)。在 10 个短提示词上对比宽度为 4 的束搜索与贪婪采样的困惑度 (Perplexity)。束搜索是否总是更优？（提示：通常在翻译任务中表现更好，而非开放式对话。）
3. **困难。** 实现投机解码 (Speculative Decoding)：使用一个微型双层模型作为草稿模型 (Draft Model)，一个六层模型作为验证模型 (Verifier)。在 100 个长度为 64 的生成任务上测量实际运行时间加速比 (Wall-clock Speedup)。确认输出结果与验证模型的贪婪采样结果一致。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 因果掩码 (Causal mask) | “三角形” | 添加到注意力分数中的上三角 `-inf` 矩阵，确保位置 `i` 仅能关注位置 `≤ i` 的信息。 |
| 下一词元预测 (Next-token prediction) | “损失函数” | 模型在每个位置上的预测分布与真实下一词元之间的交叉熵。 |
| 自回归 (Autoregressive) | “逐个生成” | 将输出反馈作为输入；仅在训练阶段支持并行，生成阶段不可并行。 |
| 逻辑值 (Logits) | “Softmax 前的分数” | 语言模型头 (LM head) 在 Softmax 之前的原始输出；采样操作基于这些值进行。 |
| 温度系数 (Temperature) | “创造力旋钮” | 将 logits 除以 T；T→0 时为贪婪采样，T→∞ 时趋近均匀分布。 |
| Top-p 采样 (Top-p) | “核采样” | 将概率分布截断至累积概率 ≥p 的最小词元集合；从截断后的分布中进行采样。 |
| Min-p 采样 (Min-p) | “优于 Top-p” | 保留满足 `p ≥ min_p × max_p` 的词元；根据分布的尖锐程度自适应调整截断阈值。 |
| 投机解码 (Speculative decoding) | “草稿 + 验证” | 轻量级模型生成 N 个候选词元；大型模型进行并行验证。 |
| 教师强制 (Teacher forcing) | “训练技巧” | 训练期间，输入真实的上一词元而非模型自身的预测。这是所有序列到序列语言模型的标准做法。 |

## 延伸阅读

- [Radford 等人 (2018)。通过生成式预训练提升语言理解能力](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford 等人 (2019)。语言模型是无监督的多任务学习者](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。
- [Brown 等人 (2020)。语言模型是少样本学习者](https://arxiv.org/abs/2005.14165) — GPT-3 与上下文学习 (In-context learning)。
- [Leviathan, Kalman, Matias (2023)。通过投机解码实现 Transformer 的快速推理](https://arxiv.org/abs/2211.17192) — 投机解码相关论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 标准的因果语言模型 (Causal-LM) 参考代码。