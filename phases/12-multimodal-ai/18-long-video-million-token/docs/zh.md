# 百万级 Token 上下文下的长视频理解

> 一段 1 小时、24 FPS 的 4K 视频，经过分块（patch）与嵌入（embedding）处理后，大约会产生 6000 万个 token。转录一段 2 小时的播客节目约为 3 万个 token。一部完整的蓝光电影，即使采用激进的池化（pooling）进行压缩，也会达到数十万个 token。Google 的 Gemini 1.5（2024 年 3 月）以 1000 万 token 的上下文长度开启了这一时代，能够在长达一小时的视频中实现可靠的“大海捞针”（needle-in-a-haystack）召回。LWM（Liu 等人，2024 年 2 月）展示了环形注意力（ring attention）的扩展路径。LongVILA 和 Video-XL 进一步扩展了数据摄入（ingestion）能力。VideoAgent 则用智能体检索（agentic retrieval）替代了原始上下文。每种方法在计算开销、召回率和工程复杂度之间都有不同的权衡。本节课程将对它们进行横向对比。

**Type:** 构建
**Languages:** Python（标准库、大海捞针模拟器 + 智能体检索路由）
**Prerequisites:** 第 12 阶段 · 第 17 阶段（视频时序 Token）
**Time:** 约 180 分钟

## 学习目标

- 计算不同帧率（FPS）和池化策略下长视频的视觉 Token 总数。
- 解释三种扩展路径：直接堆叠上下文（brute context，如 Gemini 1.5）、环形注意力（ring attention，如 LWM）以及 Token 压缩（token compression，如 LongVILA / Video-XL）。
- 对比原始上下文视频视觉语言模型（VLM）与基于智能体检索的视频 VLM（VideoAgent）在准确率和延迟上的表现。
- 为一段 30 分钟的视频设计“大海捞针”测试，并测量特定时间点的召回率。

## 核心问题

在 384 原生分辨率下，采用 Qwen2.5-VL 尺寸的分块处理单帧图像，大约需要 729 个 token。若采用 3x3 池化，则每帧降至 81 个 token。一段 30 分钟、1 FPS 的视频包含 1800 帧，即 145,800 个 token。2025 年的开源视觉语言模型尚能勉强处理，但已十分吃紧。若提升至 2 FPS，则达到 291,600 个 token——此时仅有支持超大上下文的模型能够容纳。

一部 2 小时、1 FPS 的电影将产生 58.3 万个 token。这超出了大多数 2026 年开源模型的能力范围；需要依赖 Gemini 2.5 Pro 或采用更激进的池化策略。

由此衍生出三种扩展路径。

## 核心概念

### 路径 1：暴力上下文（brute context）（Gemini 1.5, Claude Opus）

通过堆砌硬件来解决该问题。将上下文窗口扩展至数百万个词元（token），并在一次前向传播（forward pass）中处理所有内容。

Gemini 1.5 Pro 发布时支持 100 万词元；Gemini 1.5 Ultra 扩展至 1000 万；2026 年的 Gemini 2.5 Pro 已能可靠处理数小时的视频。相关论文（arXiv:2403.05530）记录了在约 950 万词元范围内，“大海捞针（needle-in-a-haystack）”召回率达到 99.7%。

工程实现：采用自定义的注意力机制（attention）实现，结合内存层级结构（局部 + 全局 + 稀疏），并辅以混合专家模型（Mixture of Experts, MoE）的路由机制以提升长上下文处理效率。未公开完整技术细节。非开源。

### 路径 2：环形注意力（ring attention）（LWM, LongVILA）

环形注意力将长序列分布在呈“环形”拓扑的设备上，每个设备持有一个数据块。全序列的注意力计算通过每个设备按环形模式将自身数据块发送给下一个设备、计算局部注意力并聚合结果来实现。

LWM（Liu 等人，2024）以此方式训练了支持 100 万词元上下文的模型。训练计算量随上下文长度呈线性扩展，而非二次方增长——注意力机制的二次方计算开销被分摊到了环形网络中的各个设备上。

LongVILA（arXiv:2408.10188）将该模式适配至视觉语言模型（Vision-Language Model, VLM）。1400 帧视频，每帧 192 个词元 = 26.8 万上下文长度，通过 8 路并行结合环形注意力进行训练。

### 路径 3：词元压缩（token compression）（Video-XL, LongVA）

成本低于暴力上下文：在大语言模型（Large Language Model, LLM）处理序列前进行激进压缩。

Video-XL（arXiv:2409.14485）使用视觉摘要词元：每段包含 N 帧的视频片段会生成一个“摘要”词元，该词元会对这 N 帧进行注意力计算。在推理阶段，LLM 每个片段仅看到一个摘要词元，从而大幅缩减上下文长度。

LongVA 通过“长上下文迁移”技术，将 LLM 的上下文从 20 万扩展至 200 万。先在长上下文文本上进行训练，再通过共享表示迁移至长上下文视频。

词元压缩以牺牲特定时间戳的召回率为代价，换取可扩展性。模型能大致了解发生了什么，但有时会遗漏精确的帧。

### 路径 4：智能体检索（agentic retrieval）（VideoAgent）

不将完整视频输入 LLM。而是将视频视为数据库，并使用 LLM 对其进行查询。

VideoAgent（arXiv:2403.10517）：

1. LLM 读取问题。
2. LLM 向检索工具请求相关片段（“显示包含猫的片段”）。
3. 工具返回匹配片段的时间戳。
4. LLM 通过 VLM 读取这些片段。
5. LLM 组织答案或提出后续查询。

这是将“LLM 即智能体”模式应用于长视频场景。推理成本更低（仅编码相关片段），但工程难度更高（检索质量成为瓶颈）。

### 大海捞针（needle-in-a-haystack）基准测试

标准的长上下文测试：在视频的随机位置插入一个独特的视觉或文本标记，然后提出需要回忆该标记的查询。

评估指标：跨视频长度和标记位置的 Recall@k。

Gemini 2.5 Pro 在长达 90 分钟的视频中召回率超过 99%。开源 72B 模型（Qwen2.5-VL-72B、InternVL3-78B）在 30 分钟视频上得分约为 85-90%，超过 60 分钟后性能下降。

VideoAgent 在 2 小时以上的视频中能够匹敌甚至超越原始上下文模型，因为只要检索工具足够优秀，就能精准命中目标。

### 路径选择建议

对于 15 分钟片段且要求顶尖精度：开源 72B 模型 + 原生上下文通常即可满足。推荐选择 Qwen2.5-VL-72B。

对于 30 分钟至 1 小时的内容：开源方案选 LongVILA 或 Video-XL；闭源方案选 Gemini 2.5 Pro。质量门槛至关重要——追求前沿性能通常需转向闭源模型。

对于 2 小时以上的内容：采用 VideoAgent 或类似的检索模式。或者，将视频摘要为更小的块，并输入分层摘要。

### 2026 年生产环境模式

在实际应用中，生产环境的长视频处理流水线通常采用混合架构：

1. 对整个视频运行动态帧率（dynamic-FPS）采样与激进池化（pooling）（获取 10 万词元的全局表示）。
2. 输入至 72B VLM 生成全局摘要。
3. 若用户提出细节问题，则以该摘要为索引运行智能体检索。

该方案结合了暴力上下文用于全局理解，以及检索机制用于局部细节。

## 实际使用

`code/main.py`：

- 计算不同帧率（FPS）与池化（pooling）策略下，1分钟至3小时视频的 Token 预算（token budget）。
- 模拟“大海捞针”（needle-in-a-haystack）测试：在随机时间戳注入标记，提出问题，并评估召回率（recall）。
- 包含一个智能体检索路由（agentic-retrieval router）模拟器，用于挑选特定视频片段输入至下游视觉语言模型（VLM）。

运行预算表，直观感受规模差距。

## 交付成果

本课时将生成 `outputs/skill-long-video-strategy-planner.md` 文件。给定视频时长与查询复杂度，该脚本会在全量上下文（brute-context）、压缩（compression）与智能体检索（agentic retrieval）之间进行策略选择，并计算预期的延迟（latency）与质量（quality）。

## 练习

1. 一段45分钟的讲座视频，帧率为1 FPS，每帧81个 Token。总 Token 数是多少？能适配哪些模型的上下文（context）？
2. 设计一个“大海捞针”（needle-in-a-haystack）测试：你将在第几分钟注入标记？具体的查询格式是什么？
3. 在1小时视频上，对比采用全量上下文（brute-context）策略的 Qwen2.5-VL-72B（80k 上下文）与 VideoAgent（Claude 3.5 + 检索）。谁的召回率（recall）更高？谁的延迟（latency）更低？
4. 环形注意力（Ring attention）的内存开销随序列长度和设备数量呈线性增长。请解释其原因，并说明如果省略环形轮转（ring-rotation）阶段会导致什么问题。
5. 阅读 Gemini 1.5 论文第5节关于“大海捞针”（needle-in-a-haystack）测试的内容。该论文在100万（1M）与1000万（10M）Token 边界处的召回率方面得出了什么结论？

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 全量上下文（Brute context） | “就是堆更多 Token” | 将大语言模型（LLM）上下文扩展至数百万 Token；单次前向传播处理全部内容 |
| 环形注意力（Ring attention） | “LWM 风格的并行” | 一种分布式注意力模式，每个设备持有一个数据块并进行轮转 |
| Token 压缩（Token compression） | “摘要 Token” | 在输入 LLM 前，通过训练好的压缩器减少每个视频片段的 Token 数量 |
| 大海捞针（Needle-in-haystack） | “NIH 测试” | 在随机位置插入唯一标记，在测试时要求模型回忆该标记 |
| 智能体检索（Agentic retrieval） | “LLM 充当查询规划器” | LLM 调用检索工具获取相关片段，通过 VLM 阅读后组织答案 |
| VideoAgent | “视频检索范式” | 标准的智能体检索架构：提问 -> 调用工具 -> 获取片段 -> 生成答案 |

## 延伸阅读

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)