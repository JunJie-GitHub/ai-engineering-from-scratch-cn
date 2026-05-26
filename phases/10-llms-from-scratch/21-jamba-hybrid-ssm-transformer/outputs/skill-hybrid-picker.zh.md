---
name: 混合架构选择器
description: 针对给定工作负载，在纯 Transformer 架构（Transformer）、Jamba 风格混合架构（Jamba-style hybrid）和纯状态空间模型（State Space Model, SSM）之间进行选择。
version: 1.0.0
phase: 10
lesson: 21
tags: [jamba, mamba, ssm, 混合架构, 长上下文, 内存预算, 架构]
---

给定工作负载规格（上下文长度分布（context length profile）p50/p99、任务混合比例、单 GPU 内存预算、目标吞吐量、质量与速度优先级），请在纯 Transformer 架构（Transformer，搭配混合专家模型（Mixture of Experts, MoE）与多头潜在注意力（Multi-Head Latent Attention, MLA））、Jamba 风格混合架构（Jamba-style hybrid）和纯 Mamba 模型之间给出推荐。

输出内容：

1. 上下文长度区间（Context-length bucket）。短（16k 以下）、中（16k-64k）、长（64k-256k）或超长（256k 以上）。用于驱动初步决策。
2. 架构推荐。从纯 Transformer、1:7 混合架构、1:3 混合架构、1:15 混合架构或纯 Mamba 中选择其一。结合上下文区间及任务的上下文内召回（in-context-recall）需求进行论证。
3. 内存预算检查。计算目标上下文下的 KV 缓存（KV cache）与 SSM 状态（SSM state）占用。在计入模型权重和激活内存（通常在权重和 KV 缓存基础上额外占用 10-20 GB）后，确认其可适配目标加速器。
4. 质量权衡说明。记录所选稀疏度（sparsity level）带来的质量损耗。比例低于 1:7 的混合架构在上下文内检索（in-context retrieval）任务上会出现可衡量的性能下降；纯 Mamba 模型在某些状态跟踪（state-tracking）任务上会失效。
5. 推理栈兼容性。确认所选架构是否受目标推理栈（inference stack）（vLLM, TensorRT-LLM, SGLang, llama.cpp）支持。混合架构的工具链覆盖范围通常比纯 Transformer 更窄。

硬性拒绝条件：
- 上下文长度低于 16k 时使用 Jamba 风格混合架构。其架构开销无法得到合理补偿。
- 在重度推理或多文档交叉引用任务中使用纯 Mamba 模型。其状态跟踪能力的局限性将产生显著影响。
- 混合比例低于 1:15。低于此阈值时，上下文内召回的可靠性无法保证。
- 任何超出指定加速器计算内存预算的推荐方案。

拒绝规则：
- 若工作负载确实混合了短上下文与长上下文，则拒绝混合架构推荐，转而推荐纯 Transformer（若可能则搭配 MLA）——混合架构的优势主要体现在长上下文工作负载中。
- 若加速器为消费级（24GB 或更低显存），则拒绝混合尺寸模型，推荐蒸馏的小型混合架构或量化纯 Transformer。
- 若工作负载为对延迟敏感的单请求生成（batch-1 generation），且模型为全新模型（无现成部署路径），则予以拒绝，并推荐采用投机解码（speculative decoding）（第 10 阶段 · 第 15 课）且生态支持完善的纯 Transformer，作为更稳妥的替代方案。

输出要求：一份单页推荐报告，需列出上下文区间、架构选择、目标上下文下的 KV 缓存占用、质量权衡说明以及推理栈兼容性。最后需附加一段“监控指标（what to monitor）”说明，明确指出用于验证该推荐方案的首批 1 万次生产请求中应采用的具体长上下文评估基准（如 RULER、LongBench、大海捞针测试（needle-in-haystack））。