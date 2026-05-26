# InternVL3：原生多模态预训练（Native Multimodal Pretraining）

> 在 InternVL3 之前，所有开源视觉语言模型（Vision-Language Model, VLM）都遵循相同的三步法：取一个在数万亿文本词元（text token）上预训练好的纯文本大语言模型（Large Language Model, LLM），拼接一个视觉编码器（vision encoder），然后对连接处进行微调。这种方法虽然有效，但会产生“对齐债务（alignment debt）”——纯文本 LLM 已将全部预训练预算用于纯文本，并不原生理解视觉词元（visual token）。事后（post-hoc）添加视觉模块后，LLM 必须在不遗忘文本能力的前提下，重新学习如何将视觉输入与其文本推理关联起来。InternVL3（Zhu 等人，2025 年 4 月）摒弃了这种事后拼接的方案：仅进行一次预训练，从第一步开始就将文本与多模态数据交错混合（interleaved）。最终，该模型在 780 亿参数开源规模下，于 MMMU-Pro 基准测试中达到了与 Gemini 2.5 Pro 相当的水平。本课时将深入探讨原生预训练（native pretraining）的理论依据，以及实施该方案所带来的改变。

**Type:** 学习
**Languages:** Python（标准库、训练语料混合器）
**Prerequisites:** Phase 12 · 05，Phase 12 · 07（训练配方）
**Time:** 约 120 分钟

## 学习目标

- 解释为何事后（post-hoc）VLM 训练会累积对齐债务（alignment debt），并列举三个可量化的症状：灾难性遗忘（catastrophic forgetting）、答案漂移（answer drift）、视觉-文本不一致（visual-text inconsistency）。
- 描述 InternVL3 的原生预训练语料混合（corpus mix）策略，并解释文本：交错数据：图像描述（text : interleaved : caption）比例的重要性。
- 对比可变视觉位置编码（Variable Visual Position Encoding, V2PE）与 Qwen2-VL 的 M-RoPE。
- 列举视觉分辨率路由器（Visual Resolution Router, ViR）与解耦视觉-语言（Decoupled Vision-Language, DvD）部署优化技术。

## 问题背景

事后（post-hoc）VLM 训练是目前的默认范式。LLaVA、BLIP-2、Qwen-VL、Idefics 等模型均基于已预训练好的 LLM（如 Llama、Vicuna、Qwen、Mistral）并添加视觉模块。其训练阶段通常如下：

1. 冻结 LLM + 冻结视觉编码器 + 可训练投影层（projector），使用图像-描述配对数据进行训练以对齐嵌入表示（embedding）。
2. 解冻 LLM，使用指令微调数据（如 LLaVA-Instruct、ShareGPT4V）进行训练。
3. 可选的特定任务微调。

对齐债务（alignment debt）会表现出以下三种症状：

- 灾难性遗忘（Catastrophic forgetting）。事后训练的 VLM 会遗忘纯文本技能。GSM8K 得分下降 5-10 分，HellaSwag 得分下降，纯文本智能体（agent）能力出现倒退。
- 答案漂移（Answer drift）。对同一视觉问题的微小措辞变化会导致不同的回答。视觉编码器与 LLM 之间的连接绑定强度弱于 LLM 自身词元（token）之间的绑定。
- 视觉-文本不一致（Visual-text inconsistency）。VLM 能够正确描述图像，但随后回答的问题却与自身描述相矛盾。视觉词元（visual token）无法像文本词元那样参与 LLM 的内部一致性检查。

这些症状已有充分记录。MM1.5 的第 4 节对其进行了量化分析，LLaVA-OneVision 的消融实验（ablation study）也暗示了这些问题。原生预训练（native pretraining）正是解决之道。

## 核心概念

### 原生多模态预训练 (Native Multimodal Pretraining)

InternVL3 从一开始就在原生多模态语料库上进行从零训练。数据混合比例如下：

- 40% 纯文本数据（FineWeb、Proof-Pile-2 等）
- 35% 图文交错数据（Interleaved Image-Text Data）（OBELICS、MMC4 风格）
- 20% 图文配对数据（Paired Image-Caption Data）
- 5% 视频文本数据（Video-Text Data）

视觉词元 (Vision Tokens)、文本词元 (Text Tokens) 以及跨模态交互 (Cross-Modal Interactions) 从第一个梯度步开始就共同参与同一个损失函数计算。无需对齐预训练 (Alignment Pretraining)，无需投影器冻结 (Projector Freezing) 阶段，也无需从灾难性遗忘 (Catastrophic Forgetting) 中恢复。

基座模型 (Base Model) 的训练为单阶段。随后进行指令微调 (Instruction Tuning)，但基座模型本身已将视觉词元视为一等公民进行理解。

### V2PE（可变视觉位置编码，Variable Visual Position Encoding）

Qwen2-VL 使用具有固定轴分配的 M-RoPE。InternVL3 引入了 V2PE：位置编码会根据模态类型（文本、图像、视频）的不同而变化，并采用可学习的缩放比例。具体而言：

- 文本词元获取一维位置（文本索引）。
- 图像块 (Image Patches) 获取二维位置（行、列）。
- 视频帧获取三维位置（时间、行、列）。

三者共享相同的旋转位置编码 (RoPE) 频率基，但每个频段的隐藏维度 (Hidden Dimension) 分配是一个可学习参数，而非固定划分。这赋予了模型在预训练期间自由权衡时间频率分辨率与空间频率分辨率的能力。

V2PE 的消融实验 (Ablation Study) 表明：在相同算力下，其在视频基准测试上的得分比 M-RoPE 高出 1-2 分。虽非革命性突破，但设计更为简洁优雅。

### 视觉分辨率路由器 (Visual Resolution Router, ViR)

部署优化方案。并非所有图像都需要全分辨率编码。对于细节较少、仅包含单一物体的照片，若以原生 1280px 分辨率进行编码会浪费大量词元 (Tokens)。ViR 是一个小型分类器，在编码前即可预测回答问题所需的最低分辨率。

路由分为三个层级：低分辨率（256 个词元）、中分辨率（576 个词元）、高分辨率（2048+ 个词元）。在生产流量中，60% 的查询仅需低或中分辨率即可满足。最终效果：在保持同等质量的前提下，吞吐量提升 2-3 倍。

### 视觉-语言解耦部署 (Decoupled Vision-Language Deployment, DvD)

在部署大型视觉语言模型 (Vision-Language Model, VLM) 时，视觉编码器 (Vision Encoder) 对每张图像仅运行一次，而大语言模型 (Large Language Model, LLM) 则需为每个输出词元自回归 (Autoregressive) 运行。这两个组件的瓶颈不同（视觉部分 = 卷积与注意力机制的 GPU 显存带宽；LLM 部分 = KV 缓存 (KV Cache)）。DvD 将它们拆分到独立的 GPU 上，并通过流式传输进行通信。

对于 8B 参数模型搭配 400M 参数编码器的架构，与同机部署相比，DvD 可使单节点吞吐量大致翻倍。

### 单阶段与多阶段训练的质量对比

InternVL3 的核心基准测试主张：在 78B 参数量下，性能匹敌 Gemini 2.5 Pro 的 MMMU-Pro；在 38B 参数量下，匹敌 GPT-4o；在 8B 参数量下，领跑开源 8B 模型榜单。这一切均基于“单阶段预训练 + 指令微调”的训练范式。

“对齐债务假说 (Alignment-Debt Hypothesis)”在此得到了量化验证：在视觉基准测试得分每提升一个单位时，InternVL3-8B 在文本基准测试（MMLU、GSM8K）上的分数损失少于 Qwen2.5-VL-7B。由于训练过程是统一的整体而非割裂的两步，该模型展现出了更强的通用性。

### InternVL3.5 与 InternVL-U

InternVL3.5（2025 年 8 月）对该训练范式进行了扩展。沿用相同的原生预训练方法，但增加了数据量与参数量。在 MMMU 上的提升属于渐进式改进。

InternVL-U（2026 年）引入了统一生成能力——在相同骨干网络 (Backbone) 之上通过 MMDiT 头输出图像。“U”代表“理解 + 生成 (Understanding + Generation)”，旨在追求类似 Transfusion 架构的统一模型（参见第 12.13 课）。同一原生预训练骨干网络可同时支持理解头与生成头。

### 原生预训练的权衡

原生预训练并非没有代价：

- 算力成本。从零训练一个新的 VLM 与训练一个文本 LLM 的成本相当——需要数百万 GPU 小时。事后适配 (Post-hoc Adaptation) 可复用现有 LLM 权重，从而节省大部分成本。
- 数据稀缺。大规模图文交错语料库十分罕见。OBELICS 包含 1.41 亿份文档，MMC4 为 5.71 亿份。而纯文本数据规模已达 15T 词元。多模态预训练数据的稀缺性是一个硬性约束。
- 基座 LLM 复用性。原生预训练放弃了后续直接替换新 LLM 的灵活性。而事后适配方案允许仅通过重新训练适配器 (Adapter)，即可将 Llama-3.1 替换为 Llama-4。

InternVL3 的核心赌注是：对齐债务带来的负面影响大于放弃复用性所造成的损失。基准测试结果支持了这一主张。高昂的制造成本也构筑了壁垒，使未来实验室难以低成本复制。事后适配型 VLM 仍将继续存在，因为对于大多数项目而言，它们依然更具成本效益。

## 使用指南

`code/main.py` 是一个训练语料混合器（training-corpus mixer）与 ViR 路由模拟器（ViR router simulator）。其功能包括：

- 接收目标语料混合比例（文本 %、交错数据 %、图像描述 %、视频 %），并计算每种模态的预期训练步数。
- 在一批查询请求上模拟 ViR 路由（分布：50% 低细节、30% 中等、20% 高细节），并报告平均 Token 数量。
- 根据视觉编码器与大语言模型（LLM）的浮点运算次数（FLOPs），报告 DvD 吞吐量估算值。
- 并排对比事后训练（post-hoc training）与原生预训练（native pretraining）在参数量、算力、数据需求以及预期的对齐债务（alignment debt）表现方面的差异。

## 交付使用

本课时将生成 `outputs/skill-native-vs-posthoc-auditor.md` 文件。针对拟定的视觉语言模型（VLM）训练计划，该脚本会评估应采用原生预训练还是事后训练，标记对齐债务风险，并推荐语料混合比例。当你正在规划一个新的开源 VLM 项目并需要选择训练策略时，请使用此工具。

## 练习

1. 估算 InternVL3-8B（原生预训练）与 LLaVA-OneVision-7B（事后训练）之间的算力差异。GPU 小时数的比例大约是多少？造成这一差距的原因是什么？

2. InternVL3 报告的语料比例为 40% 文本 / 35% 交错数据 / 20% 图像描述 / 5% 视频。如果你的目标任务以视频为主，请提出一个新的比例，并论证为什么基础模型仍然需要大量的文本和图像描述数据。

3. 阅读 MM1.5 第 4 节关于遗忘（forgetting）的内容。指出事后训练出现最大性能回退（regression）的具体基准测试名称。该回退造成的代价是多少？

4. ViR 将 60% 的流量路由至低分辨率编码。它会错误路由哪些类型的查询（即需要高分辨率时却发送到了低分辨率）？请提出三种路由器故障模式。

5. DvD 将视觉编码器与 LLM 拆分到不同的 GPU 上。在何种流量模式下，DvD 反而会降低吞吐量而非提升？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 原生多模态预训练 (Native multimodal pretraining) | “从零开始联合训练” | 文本、图像和视频 Token 从第一步起就参与损失函数计算，而非后期拼接附加 |
| 对齐债务 (Alignment debt) | “事后惩罚” | 将视觉模块附加到冻结的 LLM 上后，在文本能力和回答一致性方面出现的可量化性能回退 |
| V2PE (Variable visual position encoding) | “可变视觉位置编码” | 按模态分配的可学习位置编码机制；InternVL3 中 M-RoPE 的继任方案 |
| ViR (Vision-Resolution router) | “分辨率路由器” | 一个小型分类器，在编码前为每个查询选择所需的最低分辨率，从而节省推理 Token |
| DvD (Decoupled deployment) | “解耦部署” | 视觉编码器与 LLM 分别部署在不同 GPU 上，通过流式交接数据；可使大型 VLM 的吞吐量翻倍 |
| InternVL-U | “统一理解与生成” | 2026 年的后续版本，在原生预训练主干网络上增加了图像生成头 |
| 交错语料 (Interleaved corpus) | “OBELICS / MMC4” | 文本与图像按自然阅读顺序交织排列的文档；原生多模态预训练的原始素材 |

## 延伸阅读

- [Chen 等 — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu 等 — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang 等 — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)