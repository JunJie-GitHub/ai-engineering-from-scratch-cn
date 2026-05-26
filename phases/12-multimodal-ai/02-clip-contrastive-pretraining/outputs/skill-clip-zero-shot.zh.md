---
name: clip-zero-shot
description: 使用 CLIP / SigLIP 检查点（checkpoint）运行零样本（zero-shot）图像分类，生成带有相似度分数的排序预测结果。
version: 1.0.0
phase: 12
lesson: 02
tags: [clip, siglip, zero-shot, vision-language]
---

给定一组图像（文件路径或 URL）和一组候选类别名称，使用指定的 CLIP 或 SigLIP 检查点生成排序的零样本分类结果。该技能仅用于纯预测，不涉及训练或微调（finetune）。

输出内容：

1. 提示词（prompt）构建。为每个类别生成 N 个文本模板（默认：`a photo of a {class}`、`a picture of a {class}`、`an image of a {class}`）。使用文本编码器对每个提示词进行嵌入（embedding），并取平均值以形成类别原型（class prototype）。
2. 图像嵌入。使用指定的视觉编码器（vision encoder）对每张输入图像进行嵌入。将两侧的向量归一化为单位长度。
3. 排序预测。计算每个图像嵌入与每个类别原型之间的余弦相似度（cosine similarity）。返回得分最高的 top-1 和 top-5 结果及其分数。
4. 检查点元数据。注明所使用的确切 Hugging Face 检查点名称（例如 `openai/clip-vit-large-patch14` 或 `google/siglip2-so400m-patch14-384`）及其期望的输入分辨率。
5. 诚实性声明。明确指出对预训练分布（pretraining distribution）之外的类别进行零样本分类是不可靠的；将 top-1 分数作为置信度代理指标进行展示，并在分数低于 0.2 时发出警告。

硬性拒绝条件：
- 任何将输出结果作为调用者提供列表之外的类别的确定性标签的用法。
- 声称不同检查点之间的分数具有可比性；SigLIP 和 CLIP 的分数基于不同的量纲。
- 在已知包含人物的图像上运行，且缺乏下游的知情同意策略。

拒绝规则：
- 如果调用者要求对医疗、法律或安全关键类别（如诊断、身份识别、受保护属性）进行分类，则予以拒绝，并引导其使用带有审计追踪（audit trails）的监督模型（supervised models）。
- 如果调用者仅提供单个类别名称（无备选方案的单向分类），则予以拒绝——零样本分类至少需要两个候选类别才有意义。
- 如果未指定检查点，则予以拒绝，并要求明确选择（CLIP、OpenCLIP、SigLIP、SigLIP 2）中的哪一种及其具体规模（scale）。

输出：每张图像的 top-5 排序预测列表（附带余弦相似度分数）、检查点名称、使用的提示词模板以及置信度标志。最后附加一个“下一步阅读”段落，指向第 12.06 课以了解 NaFlex（处理可变宽高比），或推荐 SigLIP 2 论文以供深入研读。