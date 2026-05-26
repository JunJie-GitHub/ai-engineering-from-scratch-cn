# Show-o 与离散扩散统一模型

> Transfusion 混合了连续表示与离散表示（discrete representations）。而 Show-o（Xie 等人，2024 年 8 月）则反其道而行：文本词元采用因果性下一词元预测（causal next-token prediction），图像词元则借鉴 MaskGIT 的思想，采用掩码离散扩散（masked discrete diffusion）。两者共存于同一个 Transformer 中，并配备混合注意力掩码（hybrid attention mask）。该架构在单一主干网络、每种模态一个分词器（tokenizer）、单一损失函数形式（将下一词元预测扩展为掩码预测）的基础上，统一了视觉问答（VQA）、文本生成图像（text-to-image）、图像修复（inpainting）以及混合模态生成任务。本课程将深入解析 Show-o 的设计——阐明为何掩码离散扩散是一种并行、少步数的图像生成器——并将其与 Transfusion 和 Emu3 进行对比。

**类型：** 学习
**语言：** Python（标准库，掩码离散扩散采样器）
**前置知识：** 第 12 · 13 阶段（Transfusion）
**时长：** 约 120 分钟

## 学习目标

- 解释掩码离散扩散（masked discrete diffusion）：一种按均匀策略掩码词元，随后要求 Transformer 恢复它们的调度策略（schedule）。
- 在速度与质量方面，对比并行图像解码（parallel image decoding）（如 Show-o、MaskGIT）与自回归图像解码（autoregressive image decoding）（如 Chameleon、Emu3）。
- 列举 Show-o 在单一检查点（checkpoint）中处理的三项任务：文本生成图像（T2I）、视觉问答（VQA）与图像修复。
- 选择一种掩码调度策略（masking schedule）（余弦、线性或截断），并分析其对生成样本质量的影响。

## 问题背景

Transfusion 的双损失训练虽然有效，但优化动态更为棘手——连续扩散损失与离散下一词元预测（next-token prediction, NTP）损失处于不同的数值量级。平衡损失权重需要进行超参数搜索（hyperparameter search）。该架构虽然有效，但较为复杂。

Show-o 的解决方案是：保持两种模态均为离散形式（类似 Chameleon），但通过掩码离散扩散并行生成图像，而非串行生成。其训练目标简化为单一的掩码词元预测（masked-token-prediction），该目标可自然泛化下一词元预测。

## 核心概念

### 掩码离散扩散（Masked Discrete Diffusion）

Chang 等人（2022）提出的原始 MaskGIT 技巧非常巧妙。从完全掩码的图像开始（每个词元（token）均为特殊的 `<MASK>` ID）。在每一步中，并行预测所有被掩码的词元，随后保留置信度最高的 Top-K 个预测结果，并将其余部分重新掩码。经过约 8-16 次迭代后，所有词元均被填充完毕。每一步解除掩码的词元数量调度策略（schedule）经过精心调优——余弦调度（cosine schedule）效果最佳。

训练过程很简单：从 [0, 1] 区间均匀采样掩码比例（masking ratio），将其应用于图像的矢量量化词元（VQ tokens），然后训练 Transformer 以恢复被掩码的词元。这正是 BERT 在文本领域所做的事情，只是将其扩展到了图像生成任务。

### Show-o：单一 Transformer 与混合掩码

Show-o 将 MaskGIT 集成到了因果语言模型（causal-language-model）Transformer 中。其注意力掩码（attention mask）设计如下：

- 文本词元：因果掩码（标准大语言模型 LLM 模式）。
- 图像词元：在图像块内采用全双向注意力（因此在预测时，被掩码的词元可以“看到”所有其他图像词元）。
- 文本到图像（Text-to-Image）：文本关注先前的图像，图像关注先前的文本。

训练过程在以下任务间交替进行：
1. 对文本序列进行标准的下一个词元预测（Next Token Prediction, NTP）。
2. 文本到图像（Text-to-Image, T2I）样本：文本 → 图像，使用被掩码的图像词元，并计算掩码词元预测损失。
3. 视觉问答（Visual Question Answering, VQA）样本：图像 → 文本，使用被掩码的文本词元（本质上仍是 NTP）。

统一的损失函数是针对 `<MASK>` 词元的交叉熵（cross-entropy），它同时涵盖了文本 NTP（仅最后一个词元被“掩码”）和图像掩码扩散（随机子集被掩码）。

### 并行采样（Parallel Sampling）

Show-o 仅需约 16 步即可生成一张图像，而非逐词元自回归（autoregressive）所需的约 1000 步，或扩散模型（diffusion）所需的约 20 步。在每一步中，并行预测所有被掩码的词元；提交置信度最高的 Top-K 个结果；重复此过程。

对比如下：
- Chameleon / Emu3（基于词元的自回归）：需要 N_tokens 次前向传播（forward passes），通常每张图像需 1024-4096 次。
- Transfusion（连续扩散）：约 20 步，每步均为一次完整的 Transformer 前向传播。
- Show-o（掩码离散扩散）：约 16 步，每步均为一次完整的 Transformer 前向传播。

在同等规模的模型中，Show-o 的速度快于 Chameleon；其步数与 Transfusion 大致相当，但单步计算成本更低（离散词表逻辑值（logits）对比连续均方误差（MSE）损失）。

### 单一检查点（Checkpoint）支持多任务

Show-o 在推理阶段支持四种任务，通过提示词（prompt）格式进行选择：

- 文本生成：标准的自回归文本输出。
- 视觉问答（VQA）：输入图像，输出文本。
- 文本到图像（T2I）：输入文本，通过掩码离散扩散输出图像。
- 图像修复（Inpainting）：输入部分词元被掩码的图像，进行填充修复。

图像修复能力是掩码预测训练天然具备的附加功能。只需掩码 VQ 词元网格中的某个区域，输入其余部分及文本提示词，即可预测被掩码的词元。

### 掩码调度策略（Masking Schedule）

每一步解除掩码的词元数量调度策略直接影响生成质量。Show-o 推荐使用余弦调度：

mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T

在第 0 步，所有词元均被掩码（比例为 1.0）。在第 T 步，无词元被掩码。余弦调度将概率质量集中在中等比例区间，此时预测提供的信息量最大。线性调度（linear schedule）同样有效，但性能提升更快进入平台期。

### Show-o2

Show-o2（2025 年后续工作，arXiv 2506.15564）对 Show-o 进行了扩展：采用更大的 LLM 基座、更优的分词器（tokenizer）以及改进的掩码调度策略。整体架构模式保持不变。

### Show-o 的定位

在 2026 年的模型分类体系（taxonomy）中：

- 离散词元 + NTP：Chameleon、Emu3。架构简单但推理速度较慢。
- 离散词元 + 掩码扩散：Show-o、MaskGIT、LlamaGen、Muse。支持并行采样，但受分词器限制仍存在信息损失。
- 连续表示 + 扩散模型：Transfusion、MMDiT、DiT。质量最高，但训练更为复杂。
- 视觉语言模型（Vision-Language Model, VLM）中的连续表示 + 流匹配（flow matching）：JanusFlow、InternVL-U。最新架构。

根据任务需求进行选择：若希望在一个开源模型中以合理速度同时实现 T2I、图像修复和 VQA，请选择 Show-o；若将生成质量置于首位，且能够承担双重损失函数架构（two-loss plumbing）带来的工程复杂度，则选择 Transfusion。

## 使用方法

`code/main.py` 模拟了 Show-o 的采样过程：

- 一个包含 16 个 VQ 词元（VQ tokens）的示例网格。
- 一个模拟的 Transformer，它根据提示词（prompt）和当前未掩码的词元来预测对数几率（logits）。
- 采用余弦调度（cosine schedule）在 8 个步骤中进行并行掩码采样（masked sampling）。
- 打印中间状态（掩码模式的演变过程）以及最终的词元。

运行该脚本，观察掩码如何逐步消解。

## 实际部署

本课时将生成 `outputs/skill-unified-gen-model-picker.md` 文件。针对一款既需具备理解能力（如视觉问答 VQA、图像描述 captioning）又需具备生成能力（如文生图 T2I、图像修复 inpainting），且受限于必须采用开放权重（open-weights）的产品，该文档将在 Show-o 系列、Transfusion/MMDiT 系列以及 Emu3 / Chameleon 系列之间进行选型，并提供具体的权衡分析。

## 练习题

1. 掩码离散扩散（masked discrete diffusion）的采样通常需要约 16 步。为什么不能只用 1 步？如果在第 0 步就取消所有掩码，会发生什么问题？
2. 掩码扩散天然支持图像修复（inpainting）。请提出一个产品用例（真实或假设均可），说明 Show-o 的修复能力如何优于专用模型。
3. 余弦调度（cosine schedule）与线性调度（linear schedule）对比：追踪 T=8 时每一步未掩码词元的数量变化。哪种调度方式更为均衡？
4. 一张 512x512 的 Show-o 图像对应 1024 个词元。在词表大小 K=16384 的情况下，模型输出的数据量为 1024 * log2(16384) = 14,336 比特（约 1.75 KiB）。而 Stable Diffusion 输出的原始像素数据为 512*512*24 比特 = 6,291,456 比特（约 768 KiB）。请计算压缩比，并分析这种压缩带来了怎样的质量收益？
5. 阅读 LlamaGen 论文（arXiv:2406.06525）。LlamaGen 的类别条件自回归（class-conditional autoregressive）图像模型与 Show-o 的掩码方法有何不同？

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 掩码离散扩散（Masked discrete diffusion） | “MaskGIT 风格” | 训练目标是预测被掩码的词元；在推理阶段，迭代式地取消置信度最高的预测词的掩码 |
| 余弦调度（Cosine schedule） | “去掩码调度” | 掩码比例随推理步骤衰减的规律；使置信度的增长集中在中间阶段 |
| 并行解码（Parallel decoding） | “一次性处理所有词元” | 每一步通过一次前向传播预测全部被掩码的词元序列，随后提交 Top-K 结果 |
| 混合注意力（Hybrid attention） | “因果 + 双向” | 一种注意力掩码机制：对文本词元采用因果（causal）掩码，对图像块内部采用双向（bidirectional）掩码 |
| 图像修复（Inpainting） | “填充式生成” | 以部分词元被掩码的图像为条件，预测缺失的词元；该能力由训练目标自然衍生，无需额外设计 |
| 提交率（Commitment rate） | “每步 Top-K” | 每次迭代中确认为“已完成”的词元数量；用于控制推理速度与生成质量之间的权衡 |

## 延伸阅读

- [Xie 等人 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang 等人 — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun 等人 — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang 等人 — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)