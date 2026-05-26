# T5、BART —— 编码器-解码器模型（Encoder-Decoder Models）

> 编码器负责理解，解码器负责生成。将两者结合，你就能得到一个专为输入 → 输出任务构建的模型：翻译、摘要、重写、转录。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 第 7 阶段 · 05（完整 Transformer）、第 7 阶段 · 06（BERT）、第 7 阶段 · 07（GPT）
**Time:** 约 45 分钟

## 核心问题

纯解码器（Decoder-only）架构的 GPT 与纯编码器（Encoder-only）架构的 BERT 各自出于不同目的，对 2017 年的原始架构进行了精简。然而，许多任务天然属于输入到输出的映射：

- 翻译：英文 → 法文。
- 摘要生成：5,000 个词元（token）的文章 → 200 个词元的摘要。
- 语音识别：音频词元 → 文本词元。
- 结构化信息抽取：自然语言文本 → JSON。

针对这类任务，编码器-解码器（Encoder-Decoder）架构是最契合的选择。编码器负责生成源输入的稠密表示（dense representation）。解码器则负责生成输出，并在每一步通过交叉注意力（cross-attention）机制与该表示进行交互。训练时，输出侧采用逐位偏移（shift-by-one）策略。其损失函数与 GPT 一致，仅额外以编码器的输出作为条件。

两篇论文奠定了现代的实践范式：

1. **T5**（Raffel 等人，2019）。《Text-to-Text Transfer Transformer》。将每项自然语言处理（NLP）任务重新定义为“文本输入，文本输出”。采用单一架构、单一词表、单一损失函数。通过掩码片段预测（masked span prediction）进行预训练（破坏输入中的文本片段，并在输出中将其解码还原）。
2. **BART**（Lewis 等人，2019）。《Bidirectional and Auto-Regressive Transformer》。去噪自编码器（denoising autoencoder）：通过多种方式破坏输入（打乱、掩码、删除、旋转），要求解码器重建原始文本。

到了 2026 年，在输入结构至关重要的场景中，编码器-解码器格式依然活跃：

- Whisper（语音 → 文本）。
- Google 的翻译技术栈。
- 部分具有明确上下文与编辑结构区分的代码补全/修复模型。
- 用于结构化推理任务的 Flan-T5 及其变体。

尽管纯解码器架构占据了聚光灯下的主流地位，但编码器-解码器架构从未真正退出舞台。

## 核心概念

![带交叉注意力 (cross-attention) 的编码器-解码器 (encoder-decoder)](../assets/encoder-decoder.svg)

### 前向传播循环 (Forward Loop)

source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits

关键在于，编码器 (encoder) 对每个输入仅执行一次。解码器 (decoder) 采用自回归 (autoregressive) 方式运行，但在每一步都会对*同一份*编码器输出进行交叉注意力计算。缓存编码器输出能为长输入带来零成本的加速效果。

### T5 预训练 —— 片段破坏 (Span Corruption)

随机选取输入文本中的片段（平均长度为 3 个词元 (token)，占总长度的 15%）。将每个片段替换为唯一的哨兵标记 (sentinel)：`<extra_id_0>`、`<extra_id_1>` 等。解码器仅输出带有对应哨兵前缀的待补全片段：

source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy

相比预测完整序列，该训练信号的计算成本更低。在 T5 原论文的消融实验 (ablation study) 中，其表现与掩码语言模型 (Masked Language Model, MLM，如 BERT) 和前缀语言模型 (Prefix-LM，如 UniLM) 具有竞争力。

### BART 预训练 —— 多噪声去噪 (Multi-Noise Denoising)

BART 尝试了五种噪声注入函数：

1. 词元掩码 (Token masking)。
2. 词元删除 (Token deletion)。
3. 文本填充 (Text infilling)（掩码一个片段，解码器插入正确长度的内容）。
4. 句子重排 (Sentence permutation)。
5. 文档轮转 (Document rotation)。

将文本填充与句子重排相结合，能在下游任务中取得最佳指标。解码器始终负责重建原始完整文本。由于 BART 的输出是完整序列而非仅补全被破坏的片段，其预训练计算开销高于 T5。

### 推理 (Inference)

采用与 GPT 相同的自回归生成机制。支持贪婪解码 (greedy decoding) / 束搜索 (beam search) / top-p 采样 (top-p sampling)。在翻译和摘要任务中，束搜索（宽度通常为 4–5）是标准做法，因为这类任务的输出分布比开放域对话更为集中。

### 2026 年如何选择各架构变体

| 任务 | 是否使用编码器-解码器？ | 原因 |
|------|------------------|-----|
| 机器翻译 | 是，通常如此 | 源序列清晰；输出分布固定；束搜索效果佳 |
| 语音转文本 | 是（如 Whisper） | 输入模态与输出不同；编码器负责提取音频特征 |
| 对话 / 推理 | 否，仅解码器 (decoder-only) | 没有固定的“输入”——对话本身即为序列 |
| 代码补全 | 通常否 | 长上下文仅解码器架构占优；如 Qwen 2.5 Coder 等代码模型均为仅解码器架构 |
| 文本摘要 | 两者皆可 | BART、PEGASUS 曾优于早期仅解码器基线；现代仅解码器大语言模型已与之持平 |
| 结构化信息抽取 | 两者皆可 | T5 架构更简洁，因为“文本 → 文本”范式可兼容任意输出格式 |

自 2022 年以来的趋势表明，仅解码器架构正逐步接管原本由编码器-解码器架构主导的任务，原因在于：(a) 经过指令微调 (instruction-tuned) 的仅解码器大语言模型可通过提示词 (prompting) 泛化至各类任务；(b) 单一架构的扩展性优于双架构；(c) 基于人类反馈的强化学习 (Reinforcement Learning from Human Feedback, RLHF) 默认基于解码器架构。编码器-解码器架构则在输入模态与输出不同（如语音、图像）或对束搜索生成质量要求较高的场景中依然占据一席之地。

## 动手构建

请参阅 `code/main.py`。我们为一个示例语料库实现了 T5 风格的跨度破坏（span corruption）——这是本课程中最核心的内容，因为自此之后，它已成为所有编码器-解码器（encoder-decoder）预训练方案（pretraining recipe）的标准配置。

### 步骤 1：跨度破坏（span corruption）

def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...

目标格式遵循 T5 惯例：`<sent0> span0 <sent1> span1 ...`。经过破坏的输入会在跨度位置将原始词元（token）与哨兵词元（sentinel token）交错拼接。

### 步骤 2：验证往返一致性（round-trip）

给定破坏后的输入与目标，尝试重建原始句子。若破坏过程可逆，则前向传播（forward pass）的定义便是明确的。这属于一项健全性检查（sanity check）——实际训练时并不会执行此操作，但该测试开销极小，能有效捕获跨度管理中的差一错误（off-by-one bug）。

### 步骤 3：BART 噪声注入（BART noising）

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## 实际应用

HuggingFace 参考示例：

from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))

T5 的核心技巧：将任务名称直接作为输入文本的一部分。同一模型之所以能处理数十种任务，是因为所有任务均遵循“文本输入、文本输出”的范式。到 2026 年，这一模式已被指令微调（instruction-tuned）的纯解码器（decoder-only）模型广泛泛化，但 T5 是首个将其确立为标准的模型。

## 交付上线

请参阅 `outputs/skill-seq2seq-picker.md`。该技能会根据输入输出结构、延迟和质量目标，为新任务在编码器-解码器与纯解码器架构之间做出选择。

## 练习

1. **简单。** 运行 `code/main.py`，对一个包含 30 个词元的句子应用跨度破坏，验证将非哨兵源词元与解码后的目标跨度拼接后是否能还原原始句子。
2. **中等。** 实现 BART 的 `text_infill` 噪声：用单个 `<mask>` 词元替换随机跨度，解码器必须推断出正确的跨度长度及内容。展示一个示例。
3. **困难。** 在小型英语 → 猪拉丁语（pig-Latin）语料库（200 对）上微调（fine-tune）`flan-t5-small`。在预留的 50 对测试集上测量 BLEU 分数（BLEU）。与在相同数据和相同算力下微调 `Llama-3.2-1B` 的结果进行对比。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 编码器-解码器 (Encoder-Decoder) | “序列到序列 Transformer (Seq2Seq Transformer)” | 两个堆叠结构：用于处理输入的双向编码器，以及通过交叉注意力机制生成输出的因果解码器。 |
| 交叉注意力 (Cross-Attention) | “源序列与目标序列的交互点” | 解码器的查询向量（Q）与编码器的键/值向量（K/V）相乘。这是编码器信息传入解码器的唯一通道。 |
| 片段破坏 (Span Corruption) | “T5 的预训练技巧” | 用哨兵标记替换随机文本片段；解码器负责输出这些被替换的片段。 |
| 去噪目标 (Denoising Objective) | “BART 的核心任务” | 对输入施加噪声函数，训练解码器以重建原始干净序列。 |
| 哨兵标记 (Sentinel Token) | “`<extra_id_N>` 占位符” | 用于在源文本中标记被破坏片段，并在目标文本中重新标记的特殊标记。 |
| Flan | “指令微调版 T5” | 在超过 1,800 个任务上微调的 T5；使编码器-解码器架构在指令遵循任务中具备竞争力。 |
| 束搜索 (Beam Search) | “解码策略” | 在每一步保留前 k 个部分序列；是机器翻译和文本摘要的标准解码方法。 |
| 教师强制 (Teacher Forcing) | “训练阶段的输入方式” | 训练期间，将真实的上一时刻输出标记输入解码器，而非模型采样生成的标记。 |

## 进一步阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5 模型。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART 模型。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5 模型。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年标准的编码器-解码器架构。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现代码。