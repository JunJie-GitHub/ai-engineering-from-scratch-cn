# 开放权重视觉语言模型（VLM）训练配方：什么才是真正重要的

> 2024-2026 年间的开放权重视觉语言模型（VLM）文献犹如一片消融实验表格（ablation tables）的森林。Apple 的 MM1 测试了图像编码器（image encoder）、连接器（connector）与数据混合（data mix）的 13 种组合。Allen AI 的 Molmo 证明了详尽的人工描述（human captions）优于 GPT-4V 的蒸馏数据。Cambrian-1 进行了 20 余次编码器对比。Idefics2 形式化了五轴设计空间（five-axis design space）。Prismatic VLMs 在受控基准测试上对比了 27 种训练配方（training recipes）。在纷繁的数据中，少数结论在多篇论文中高度一致：图像编码器的重要性高于连接器架构，数据混合的重要性又凌驾于二者之上，且详尽的人工描述优于蒸馏合成数据（distilled synthetic data）。本节将为你解读这些表格，免去你亲自梳理的繁琐。

**Type:** 学习 + 实验
**Languages:** Python（标准库，消融表格解析器 + 配方选择器）
**Prerequisites:** 第 12 阶段 · 05（LLaVA 基线）
**Time:** 约 180 分钟

## 学习目标

- 说出五轴 VLM 设计空间的构成：图像编码器、连接器、大语言模型（LLM）、数据混合与分辨率调度（resolution schedule）。
- 阅读 MM1 / Idefics2 / Cambrian-1 的消融实验表格，并预测调整哪个参数（knob）会对特定基准测试产生影响。
- 在给定算力预算和任务组合的情况下，为新 VLM 挑选合适的训练配方（编码器、连接器、数据、分辨率）。
- 解释为何在相同 token 数量下，详尽的人工描述优于 GPT-4V 蒸馏数据。

## 问题背景

目前已有数百个开放权重 VLM。从“优秀”到“最先进水平（state-of-the-art）”的差距，主要不在于模型架构，而在于数据、分辨率调度与编码器选择。当模型表现不佳时，知道优先调整哪个参数，能帮你避免浪费 500 万 GPU 小时的试错成本。

2023 年的第一波模型（LLaVA-1.5、InstructBLIP、MiniGPT-4）主要依赖图文对预训练（caption-pair pretraining）加上 LLaVA-Instruct-150k 指令微调。这是一个不错的基线，但在 MMMU 基准上的表现止步于 35% 左右。

2024 年的第二波模型（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）则进行了详尽的消融实验。其结果既出人意料，又极具实践指导意义。

## 核心概念

### 五轴设计空间

Idefics2（Laurençon 等人，2024）将这些轴命名为：

1. 图像编码器（Image encoder）。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。不同编码器在图像块大小（Patch size）、分辨率和预训练目标（Pretraining objective）上存在差异。
2. 连接器（Connector）。MLP（2-4 层）、Q-Former（32 个查询向量 + 交叉注意力 Cross-attention）、Perceiver Resampler（64 个查询向量）、C-Abstractor（卷积 + 双线性池化 Bilinear pooling）。
3. 语言模型（Language model）。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。大语言模型（LLM）的规模是参数成本的主要决定因素。
4. 训练数据（Training data）。图文对（CC3M、LAION）、交错数据（Interleaved，如 OBELICS、MMC4）、指令数据（Instruction，如 LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. 分辨率调度策略（Resolution schedule）。固定 224/336/448、AnyRes、原生动态分辨率。在训练过程中逐步提升（Ramped）或保持恒定。

每个投入生产的视觉语言模型（Vision-Language Model, VLM）都会在这五个轴上做出选择。MMMU 得分的大部分差异可由轴 1、4 和 5 解释，而非取决于你选择了哪种连接器。

### 轴 1：编码器 > 连接器

MM1 第 3.2 节表明：将编码器从 CLIP ViT-L/14 替换为 SigLIP SO400m/14 可使 MMMU 得分提升 3 分以上。而将连接器从 MLP 替换为 Perceiver Resampler 带来的提升不足 1 分。Idefics2 复现了这一结论：在相同 Token 数量下，SigLIP 优于 CLIP，且 Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的“Cambrian Vision Encoders Match-Up”（Tong 等人，2024）在以视觉为中心的基准测试（CV-Bench）上评估了 20 多种编码器。排行榜前列是 DINOv2 和 SigLIP 的混合；CLIP 处于中游；ImageBind 和 ViT-MAE 排名靠后。在 CV-Bench 上，从 CLIP ViT-L 到 DINOv2 ViT-g/14 的差距约为 5-7 分。

2026 年开源 VLM 的默认编码器是 SigLIP 2 SO400m/14，用于提取语义特征与密集特征（Dense features），有时会与 DINOv2 ViT-g/14 的特征进行拼接（Concatenated）（Cambrian 的“空间视觉聚合器 Spatial Vision Aggregator”正是采用此方法）。

### 轴 2：连接器设计差异不大

MM1、Idefics2、Prismatic 和 MM-Interleaved 均得出了相同结论：在视觉 Token 数量固定的情况下，连接器架构的影响微乎其微。在相同的 Token 预算下，对平均池化（Mean-pooled）后的图像块使用 2 层 MLP，其性能与使用 32 个查询向量的 Q-Former 相差不到 1 分。

真正关键的是 Token 数量。更多的视觉 Token 意味着更多的 LLM 计算量，性能会随之提升，但达到一定阈值后收益递减。每张图像 64 个 Token 对于光学字符识别（Optical Character Recognition, OCR）来说太少了。576-1024 个 Token 是大多数开源 VLM 的最佳平衡点。2048+ 个 Token 仅对文档和图表处理有帮助。

Q-Former 与 MLP 的选择是成本问题，而非质量问题：无论图像分辨率如何，Q-Former 都会将 Token 数量限制在 32-64 个；而 MLP 会输出所有图像块 Token。对于高分辨率输入，Q-Former 能节省 LLM 的上下文窗口；对于低分辨率输入，两者的差异可忽略不计。

### 轴 3：LLM 规模决定性能上限

在所有 VLM 相关论文中，将 LLM 规模从 7B 翻倍至 13B，均能稳定地在 MMMU 上带来 2-4 分的提升。达到 70B 时，模型在大多数基准测试上已趋于饱和。VLM 的多模态推理上限取决于 LLM 的文本推理上限——视觉编码器只能为其提供输入，无法替代其进行推理。

这正是 Qwen2.5-VL-72B 和 Claude Opus 4.7 在 MMMU-Pro 和 ScreenSpot-Pro 上表现碾压的原因：其“语言大脑”规模庞大。7B 的 VLM 无法仅凭巧妙的连接器设计来替代 70B 的 VLM。

### 轴 4：数据——详细的人工描述优于知识蒸馏

Molmo + PixMo（Deitke 等人，2024）是 2024 年每个人都应该阅读的重要成果。Allen AI 让人类标注员通过 1-3 分钟密集的语音转文本流程来描述图像，最终生成了 71.2 万张带有密集描述（Densely-captioned）的图像。训练数据中完全没有使用 GPT-4V 进行知识蒸馏（Distillation）。

Molmo-72B 在全部 11 项基准测试中均击败了 Llama-3.2-90B-Vision。这一差距并非源于架构，而是描述质量。详细的人工描述每张图像包含的信息量是简短网络描述的 5-10 倍，并且在 GPT-4V 蒸馏容易产生幻觉（Hallucination）的地方，人工描述依然能保持事实准确性。

ShareGPT4V（Chen 等人，2023）和 Cauldron（Idefics2）也采用了相同的策略，混合使用人工与 GPT-4V 生成的描述。趋势已十分明确：面向 2026 年的前沿模型，描述密度 > 描述数量 > 蒸馏便利性。

### 轴 5：分辨率及其调度策略

Idefics2 的消融实验（Ablation）表明：分辨率从 384 提升至 448 可增加 1-2 分。结合图像切分（Image splitting，如 AnyRes）将分辨率从 448 提升至 980，在 OCR 基准测试上可再提升 3-5 分。固定分辨率训练会在中等精度处遇到瓶颈；而分辨率逐步提升策略（从 224 开始，最终达到 448 或原生分辨率）训练速度更快，且最终精度更高。

Cambrian-1 进行了分辨率与 Token 数量的权衡实验：在固定算力下，你可以选择低分辨率下的更多 Token，或高分辨率下的更少 Token。对于 OCR 任务，高分辨率更胜一筹；对于通用场景理解，低分辨率配合更多 Token 效果更好。

2026 年的生产级训练配方：第一阶段（Stage 1）使用固定的 384 分辨率进行训练；第二阶段（Stage 2）采用动态分辨率（最高 1280），以应对重度依赖 OCR 的任务。

### Prismatic 的受控对比实验

Prismatic VLMs（Karamcheti 等人，2024）是一篇严格控制了所有变量的论文。该研究使用相同的 13B LLM、相同的指令数据和相同的评估标准——每次仅改变一个轴。结果如下：

- 每张图像的视觉 Token 数量解释了约 60% 的性能差异。
- 编码器选择解释了约 20%。
- 连接器架构解释了约 5%。
- 其他所有因素（数据混合比例、调度器、学习率 LR）占剩余的约 15%。

这是一个粗略的分解，但它是现有文献中对“我应该优先对哪个模块进行消融实验”这一问题最清晰明确的回答。

### 2026 年选型指南

基于上述证据，2026 年启动新项目的默认开源 VLM 配方如下：

- 编码器：使用支持 NaFlex 的原生分辨率 SigLIP 2 SO400m/14；若需分割或视觉定位（Grounding）任务，可拼接 DINOv2 ViT-g/14 以获取密集特征。
- 连接器：基于图像块 Token 的 2 层 MLP。除非受限于 Token 数量，否则无需使用 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2。追求成本效益选 7B，追求质量选 70B，具体根据目标延迟（Latency）决定。
- 数据：PixMo + ShareGPT4V + Cauldron，并补充特定任务的指令数据。
- 分辨率：动态调整（长边最小 256 像素，最大 1280 像素）。
- 训练调度：第一阶段对齐（仅训练投影层 Projector），第二阶段全量微调（Full fine-tune），第三阶段特定任务微调。

上述每一项默认配置，均可追溯至本章节末尾所引用论文中经过实测的消融实验结果。

## 使用方法

`code/main.py` 是一个消融表（ablation table）解析器与配方选择器（recipe picker）。它编码了 MM1 和 Idefics2 的消融表（精简版），并允许你进行如下查询：

- “在预算 X 和任务 Y 的条件下，哪种配方表现最佳？”
- “如果在 7B Llama 上将 SigLIP 替换为 CLIP，预期的 MMMU 指标变化量（delta）是多少？”
- “为了获得 80% 置信度的答案，我应该首先对哪个设计维度（axis）进行消融？”

输出结果是一个排序后的配方列表，包含预期的基准测试指标变化量，以及“优先消融”建议。

## 交付上线

本节将生成 `outputs/skill-vlm-recipe-picker.md` 文件。在给定目标任务组合、计算预算和延迟目标的情况下，它会输出一份完整的配方（包含视觉编码器、连接器、大语言模型、数据混合比例、分辨率调度策略），并附带支持各项选择的消融实验引用依据。避免工程师在每次启动新的视觉语言模型（VLM）项目时，都重新摸索 Idefics2 的消融表。

## 练习

1. 阅读 MM1 论文第 3.2 节。在固定使用 2B 大语言模型且预算为 5000 万张图像的情况下，哪种视觉编码器胜出？如果换成 13B 大语言模型，结论是否会反转？为什么？

2. Cambrian-1 发现，将 DINOv2 与 SigLIP 拼接使用在以视觉为中心的基准测试中优于单独使用任一模型，但在 MMMU 上未带来额外增益。请预测哪些基准测试会有提升，哪些会保持平稳。

3. 你的目标是基于 2B 大语言模型构建一个移动端 UI 智能体。请选择视觉编码器、连接器、分辨率策略和数据混合方案，并使用具体的消融表为每项选择提供依据。

4. Molmo 发布了 4B 和 72B 两个版本的模型。其中 4B 模型的性能可与闭源 7B 视觉语言模型相媲美；72B 模型则在 11/11 项基准测试中击败了 Llama-3.2-90B-Vision。这对你理解“大语言模型规模收益递减假说（LLM-size plateau hypothesis）”有何启示？

5. 设计一个消融表，以在 7B 视觉语言模型上分离数据混合质量与视觉编码器质量的影响。最少需要多少次训练运行？请提出四个维度的具体设置。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 消融实验（Ablation） | “只调一个旋钮” | 训练多个仅在单一设计空间维度上存在差异的运行实例，同时保持其他所有条件不变 |
| 连接器（Connector） | “桥梁”/“投影器” | 可训练模块，用于将视觉编码器的输出映射到大语言模型的词元空间（如 MLP、Q-Former、Perceiver） |
| 详细人工描述（Detailed human caption） | “密集描述” | 由人工撰写的多句描述（通常为 80-300 个词元），比网页替代文本（alt text）包含更丰富的信息 |
| 知识蒸馏（Distillation） | “GPT-4V 生成的描述” | 由更强大的闭源视觉语言模型生成的训练数据；虽然便捷，但容易继承原始模型的幻觉问题 |
| AnyRes / 动态分辨率（AnyRes / dynamic res） | “高分辨率路径” | 通过图像分块（tiling）或 M-RoPE 技术，将超出编码器原生分辨率的图像输入模型的策略 |
| 分辨率爬坡（Resolution ramp） | “课程学习” | 从低分辨率开始并逐步提升的训练调度策略，可加速模态对齐学习 |
| 视觉中心型基准测试（Vision-centric bench） | “CV-Bench / BLINK” | 侧重于细粒度视觉感知评估，而非重度依赖语言推理的测试 |
| PixMo | “Molmo 的数据集” | Allen AI 发布的包含 71.2 万张密集描述图像的数据集；将人类语音转录为密集描述文本 |

## 延伸阅读

- [McKinzie 等 — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon 等 — Idefics2 / 构建视觉语言模型（Vision-Language Models, VLMs）的关键要素 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke 等 — Molmo 与 PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong 等 — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti 等 — Prismatic 视觉语言模型（VLMs）(arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)