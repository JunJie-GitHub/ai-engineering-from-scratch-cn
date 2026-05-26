# LLaVA 与视觉指令微调 (Visual Instruction Tuning)

> LLaVA（2023 年 4 月）是当今被复制最多的多模态架构 (Multimodal Architecture)。它用 2 层多层感知机 (MLP) 替换了 BLIP-2 的 Q-Former，用朴素的 Token 拼接 (Naive Token Concatenation) 替换了 Flamingo 的门控交叉注意力 (Gated Cross-Attention)，并使用 GPT-4 仅基于纯文本描述生成的 15.8 万条视觉指令对话数据进行训练。在 2023 年至 2026 年间构建视觉语言模型 (VLM) 的从业者，几乎都基于 LLaVA 的某种变体进行开发。LLaVA-1.5 引入了 AnyRes。LLaVA-NeXT 提升了分辨率。LLaVA-OneVision 将单图、多图和视频统一到了同一套训练方案中。本课程将深入解析该方案，实现投影层 (Projector)，并阐明为何“越简单越成功”。

**类型：** 构建实践
**语言：** Python（标准库，投影层 + 指令模板构建器）
**前置要求：** 第 12 阶段 · 02 (CLIP)，第 11 阶段 (大语言模型工程 — 指令微调)
**时长：** 约 180 分钟

## 学习目标

- 构建一个 2 层 MLP 投影层，将视觉 Transformer (ViT) 的图像块嵌入 (Patch Embeddings，维度 1024) 映射到大语言模型 (LLM) 的嵌入维度 (维度 4096)。
- 逐步实践 LLaVA 的两阶段训练方案：(1) 在 55.8 万对图文描述数据上进行投影层对齐，(2) 在 15.8 万条 GPT-4 生成的对话数据上进行视觉指令微调。
- 构建符合 LLaVA 格式的提示词 (Prompt)，包含图像 Token 占位符、系统提示词以及用户/助手对话轮次。
- 解释为何尽管 Q-Former 在 Token 预算控制上占优，社区仍转向使用 MLP。

## 问题背景

BLIP-2 的 Q-Former（第 12.03 课）可将图像压缩为 32 个 Token。设计简洁、效率高，在基准测试中表现优异。但它存在两个问题。

首先，Q-Former 是可训练的，但其损失函数并非最终任务目标。第一阶段训练图像-文本对比 (ITC)、图像-文本匹配 (ITM) 和图像-文本生成 (ITG)。第二阶段训练语言模型 (LM) 损失。查询向量 (Queries) 学习到的是一种中间表示，随后还需由 LLM 进行解码。信息在这一瓶颈处发生了丢失。

其次，Q-Former 包含 1.88 亿参数，在 LLaVA 2023 年的规模下，你必须将其与目标 LLM 进行联合设计。更换 LLM，就得重新训练 Q-Former。更换视觉编码器 (Vision Encoder)，也得重新训练。每一种组合都相当于一个独立的研发项目。

LLaVA 的解决方案简单得令人惊叹：直接提取 ViT 的 576 个图像块 Token，让每个 Token 通过一个 2 层 MLP（`1024 → 4096 → 4096`），然后将这 576 个 Token 全部直接输入 LLM 的序列中。没有瓶颈。无需在第一阶段使用间接的预训练目标。只需使用直接的语言模型损失来训练 MLP 即可。

数据从何而来？LLaVA 的第二个核心洞察：利用 GPT-4（纯文本模式）生成指令数据。将图像的 COCO 描述和边界框 (Bounding Box) 数据输入 GPT-4，要求它生成对话、详细描述以及复杂的推理问题。免费获得 15.8 万条指令-回复对话轮次。无需人工标注。

最终成果：一个仅需 8 张 A100 显卡运行一天即可训练完成的 VLM，在 MMMU 基准测试中击败了 Flamingo，并发布了可供社区扩展的开源模型检查点 (Checkpoint)。到 2023 年底，它已衍生出 50 多个分支版本。

## 核心概念

### 架构（Architecture）

LLaVA-1.5（13B 版本）：
- 视觉编码器（Vision Encoder）：CLIP ViT-L/14 @ 336（第一阶段冻结，第二阶段可选择解冻）。
- 投影器（Projector）：带有 GELU 激活函数的 2 层多层感知机（MLP），维度为 `1024 → 4096 → 4096`。
- 大语言模型（LLM）：Vicuna-13B（后续版本为 Llama-3.1-8B）。

图像与文本提示词的前向传播（Forward Pass）流程：

img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response

图像会占用大语言模型上下文（Context）中的 576 个词元（Token）。在 2048 长度的上下文中，这将为文本留下 1472 个词元的空间；而在 32k 长度的上下文中，这种占用几乎可以忽略不计（仅为舍入误差级别）。

### 第一阶段：投影器对齐（Projector Alignment）

冻结 ViT。冻结 LLM。仅训练 2 层 MLP。数据集：55.8 万组图像-描述对（LAION-CC-SBU）。损失函数（Loss）：基于投影后的图像词元作为条件，对图像描述进行语言建模（Language Modeling）。

在批次大小（Batch Size）为 128 的情况下，仅需一个训练轮次（Epoch）即可在几小时内完成。投影器在此过程中学习将 ViT 特征空间映射到 LLM 特征空间。此阶段无需特定任务的监督信号。

### 第二阶段：视觉指令微调（Visual Instruction Tuning）

解冻投影器（保持可训练状态）。解冻 LLM（通常全量解冻，有时使用 LoRA）。使用 15.8 万轮视觉指令对话数据进行训练。

指令数据是其中的关键技巧。Liu 等人通过以下步骤生成该数据：
1. 选取一张 COCO 数据集中的图像。
2. 提取文本描述（5 条人工标注的图像描述 + 边界框列表）。
3. 使用三种提示词模板（Prompt Templates）发送给 GPT-4：
   - 对话生成：“生成一段用户与助手围绕该图像进行的来回对话。”
   - 详细描述：“提供对该图像丰富且详细的描述。”
   - 复杂推理：“提出一个需要基于图像进行推理才能回答的问题，并给出答案。”
4. 将 GPT-4 的输出解析为（指令，回复）对。

整个过程并未直接处理图像本身，仅依赖文本描述。GPT-4 会“幻觉”出合理的图像内容。尽管存在一定噪声，但该方法行之有效：15.8 万轮对话数据足以解锁模型的对话能力。

### 社区广泛采用此架构的原因

- 无需为第一阶段设计特定的损失函数。全程统一使用语言模型损失（LM Loss）。
- 投影器训练仅需数小时，而非数天。
- 大语言模型可灵活替换（如 LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需重新训练投影器即可。
- 视觉指令数据流水线依赖 GPT-4，针对新领域重新生成数据的成本极低。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）新增：
- 将学术任务数据（VQA、OKVQA、RefCOCO）混合加入指令微调中。
- 优化后的系统提示词（System Prompt）。
- 上下文长度从 2048 扩展至 32k。

LLaVA-NeXT（2024 年 1 月）新增：
- AnyRes 技术：将高分辨率图像分割为 2x2 或 1x3 网格的 336x336 裁剪块，外加一张全局低分辨率缩略图。每个裁剪块生成 576 个词元；每张图像总计约 2880 个视觉词元。OCR 与图表理解任务的性能大幅提升。
- 引入 ShareGPT4V（高质量 GPT-4V 描述）优化指令数据混合比例。
- 采用更强大的基座大语言模型（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

第 12.08 课将深入讲解 OneVision。简而言之：采用相同的投影器，但通过课程学习（Curriculum Learning）策略进行训练，使单一模型在共享视觉词元预算的前提下，同时支持单图、多图和视频理解。

### 与 Q-Former 的对比

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| 每张图像的视觉词元数 | 32 | 576（基础版）或 2880（AnyRes） |
| 可训练参数量 | 1.88 亿 + 语言模型 | 4000 万 + 语言模型 |
| 第一阶段损失函数 | ITC+ITM+ITG | 仅语言模型损失（LM） |
| LLM 即插即用性 | 需要重新训练 | 仅需极少量重新训练即可替换 |
| 多图处理 | 较为繁琐 | 自然支持（直接拼接） |
| 视频处理 | 较为繁琐 | 自然支持（逐帧拼接） |
| 词元预算 | 较小 | 较大 |

MLP 在架构简洁性和词元灵活性上胜出。Q-Former 则在词元预算控制上占优。但到了 2023 年底，词元预算已不再是核心瓶颈（大语言模型上下文长度已扩展至 32k-128k 以上），简洁性成为主导因素。

### 提示词格式（Prompt Format）

A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...

`<image>` 是一个占位符词元（Placeholder Token）。在分词（Tokenization）之前，它会被替换为 576 个视觉词元（使用 AnyRes 时为 2880 个）。分词器（Tokenizer）看到的序列长度略长于其训练时的长度，但由于第一阶段已教会模型如何处理此类输入，大语言模型能够顺利应对这种新颖的输入形式。

### 参数经济性（Parameter Economy）

LLaVA-1.5-7B 参数构成：
- CLIP ViT-L/14 @ 336：3.03 亿参数（第一阶段冻结，第二阶段通常解冻）。
- 投影器（2 层线性层）：约 2200 万可训练参数。
- Llama-7B：70 亿参数。
- 总计：73 亿参数。第二阶段可训练参数：完整的 70 亿 + 2200 万投影器参数。

第二阶段训练成本：在 8 张 A100 GPU 上约需 20 小时。这是关键数据——仅需一天、单台服务器节点、且结果可复现。这正是 LLaVA 得以广泛传播的原因。

## 使用方法

`code/main.py` 实现了：

1. 纯 Python 实现的双层多层感知机投影器（MLP Projector）（在玩具规模下，维度为 16 → 32 → 32）。
2. 提示词构建流水线（Prompt-building Pipeline）：系统提示词（System Prompt）+ 将 `<image>` 替换为 N 个投影后的词元（Token）+ 用户输入轮次 + 助手生成占位符。
3. 一个可视化工具，用于展示包含 576 个词元的视觉块（Visual Block）在大语言模型上下文（LLM Context）中的实际占比（即消耗 2k / 32k / 128k 上下文窗口的百分比）。

## 交付成果

本课时将生成 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA 系列模型检查点（Checkpoint），该脚本会运行一套包含 10 个提示词的直观评估（Vibes-eval）测试套件（3 个图像描述、3 个视觉问答、2 个逻辑推理、2 个安全拒绝），并输出一份人类可读的评分卡。这并非严格的基准测试（Benchmark），而是一次冒烟测试（Smoke Test），旨在验证投影器与大语言模型之间的协同连接是否良好。

## 练习

1. 计算维度为 `1024 → 4096 → 4096` 的双层 MLP 投影器的可训练参数量。在包含 GELU 激活函数和偏置项（Bias）的情况下，它占 LLaVA-13B 模型总参数量的比例是多少？

2. 为“拒绝回答”场景构建一个 LLaVA 提示词——图像中包含私人个体。写出预期的助手回复。为什么 LLaVA 应该在零样本（Zero-shot）设置下拒绝该请求？需要什么样的训练数据来强化这种拒绝行为？

3. 阅读 LLaVA-NeXT 博客中关于 AnyRes 的部分。计算在 AnyRes 机制下，一张 1344x672 分辨率图像产生的视觉词元（Visual Token）数量。并与 336x336 分辨率下的基础 576 个词元进行对比。

4. LLaVA 第一阶段（Stage 1）的投影器是使用语言模型损失（LM Loss）在图像描述数据上进行训练的。如果跳过第一阶段直接进入第二阶段（视觉指令微调，Visual Instruction Tuning）会发生什么？请引用 Prismatic VLMs 消融实验（Ablation）（arXiv:2402.07865）中的结论作答。

5. LLaVA-Instruct-150k 数据集利用 GPT-4 结合 COCO 图像描述来生成指令数据。针对一个新领域（如医学 X 光片、卫星图像），请描述生成该领域指令数据的四步数据流水线（Data Pipeline）。每个步骤中可能出现什么问题？

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Projector（投影器） | “MLP 桥接层” | 带有 GELU 激活函数的双层 MLP，用于将视觉 Transformer（ViT）维度映射到大语言模型（LLM）维度 |
| Image token（图像词元） | “`<image>` 占位符” | 提示词标记，在推理前会被替换为 N 个投影后的视觉词元 |
| Visual instruction tuning（视觉指令微调） | “LLaVA 第二阶段” | 使用 GPT-4 生成的（图像、指令、回复）三元组数据进行训练 |
| Stage 1 alignment（第一阶段对齐） | “投影器预训练” | 冻结 ViT 和 LLM，仅使用图像描述数据配合语言模型损失训练投影器 |
| AnyRes | “多裁剪平铺” | 将高分辨率图像分割为网格图块，并拼接每个图块的视觉词元 |
| LLaVA-Instruct | “GPT-4 生成数据” | 基于 COCO 图像描述与 GPT-4 合成的 15.8 万条指令-回复对 |
| Vision encoder freeze（视觉编码器冻结） | “主干网络锁定” | 在第一阶段不更新 CLIP 权重，有时在第二阶段也不更新 |
| ShareGPT4V | “更优质的描述” | 由 GPT-4V 生成的 100 万条密集描述，用于实现更高质量的对齐 |
| VQA（视觉问答） | “视觉问题回答” | 针对图像回答开放式问题的任务 |
| Prismatic VLMs | “设计空间论文” | Karamcheti 等人（2024）的消融研究，系统性地测试了投影器架构与数据选择方案 |

## 延伸阅读

- [Liu 等人 — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — 视觉指令微调（Visual Instruction Tuning）的 LLaVA 论文。
- [Liu 等人 — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen 等人 — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — 密集描述（Dense Captions）数据集。
- [Karamcheti 等人 — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 视觉语言模型（Vision-Language Models, VLMs）的设计空间消融（Design-Space Ablations）实验。
- [Li 等人 — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一支持单图像、多图像与视频处理。