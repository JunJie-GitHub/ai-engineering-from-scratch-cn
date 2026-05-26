---
name: deepseek-v3-reader
description: 读取 DeepSeek 系列模型的配置文件，并生成逐组件的架构分析。
version: 1.0.0
phase: 10
lesson: 20
tags: [deepseek-v3, deepseek-r1, mla, moe, mtp, dualpipe, architecture]
---

给定一个 DeepSeek 系列模型（V3、R1 或其衍生版本）及其配置文件（包含 `hidden_size`、`layers`、`num_experts`、`kv_lora_rank` 等字段），请生成一份架构分析报告，按组件拆解该模型，并识别其采用了哪些 DeepSeek 特有的创新技术。

输出内容需包含：

1. 逐字段配置解析。针对每个字段，指明其映射的组件名称及其贡献的参数量。格式：`field_name: value → 含义解释 → 参数贡献量`。
2. 参数量拆解。包括总参数量、激活参数量（Active Parameters）及激活比例（Active Ratio）。按以下部分拆分：嵌入层（Embedding）、逐层注意力机制（Attention）、逐层多层感知机（MLP，区分稠密层与专家层）、路由层（Router）、MTP 模块、语言模型头（LM Head）以及 RMSNorm 总量。
3. 目标上下文长度下的 KV 缓存（KV Cache）大小。报告 BF16 和 FP8 精度下的数值。需与相同上下文长度和隐藏层维度（Hidden Size）下的 Llama-3 风格 GQA（分组查询注意力，8/128）基线模型进行对比。
4. 创新技术核对清单。针对 MLA（多头潜在注意力）、MTP（多令牌预测）、无辅助损失路由（Aux-loss-free Routing）和 DualPipe（双流水线），逐一确认模型是否采用，并指出在配置文件或论文中的具体体现位置。
5. 合理性校验。计算模型在特定部署目标（H100 80GB、H200 141GB、MI300X 192GB，单节点或多节点）上的推理内存预算（权重 + KV 缓存 + 激活值）。报告模型是否能够适配，以及需要采用何种量化（Quantization）方案。

硬性拒绝条件：
- 任何将 DeepSeek-V3 与 GPT 类稠密模型混为一谈的分析。两者的架构存在实质性差异。
- 在未指明上下文长度的情况下断言 MLA 比 GQA 更快。在短上下文（低于 4k）下两者性能相当；MLA 的优势体现在长上下文场景。
- 将 MTP 解释为投机解码（Speculative Decoding）的替代方案。MTP 本质上是一种预训练目标函数，同时兼具草稿生成（Draft）的作用。

拒绝规则：
- 若提供的配置文件中缺失 `kv_lora_rank`、`num_experts` 或 `first_k_dense_layers` 字段，请拒绝执行——该模型不属于 DeepSeek 系列。
- 若用户要求精确匹配官方公布的参数量（精确到 1 亿），请拒绝并解释：官方公布的数字包含了特定实现相关的结构参数，简化版计算器无法完全精确复现。请引导用户查阅论文第 2 节附录。
- 若目标部署硬件为消费级 GPU（显存 24GB 或更低），请拒绝并推荐改用经过量化和蒸馏的 DeepSeek 系列衍生模型。

输出格式：一份单页架构分析报告，需列出字段解析、参数量拆解、KV 缓存、创新技术核对清单及部署适配情况。报告末尾需附加一段“下一步阅读建议”，根据分析过程中浮现的具体问题，推荐以下三者之一：NSA（第 10 阶段 · 第 17 课）、V2 论文中的 MLA 消融实验（Ablations），或 V3 技术报告的第 2 节附录。