---
name: open-model-picker
description: 为给定的部署目标选择开源大语言模型（LLM）家族、量化方案及推理服务栈（serving stack）。
version: 1.0.0
phase: 10
lesson: 14
tags: [open-models, llama, deepseek, mixtral, qwen, gemma, moe, gqa, mla, quantization]
---

给定部署目标（GPU 类型、单卡显存（VRAM）、GPU 数量、目标上下文长度、目标 p50/p99 延迟（latency）、峰值并发请求数）和任务画像（对话、代码、推理、长上下文检索、工具调用），请推荐一个开源模型及配套推理服务栈（serving stack），并针对第 14 课中的六个架构调节项（architectural knobs）逐一给出明确的推理依据。

输出内容：

1. 模型候选短名单。提供三个候选模型，每个需包含：总参数量（total params）、激活参数量（active params，需考虑混合专家模型（MoE）特性）、架构标志位（architecture flags：归一化 / 激活函数 / 位置编码 / 注意力机制 / MoE / 上下文），以及入选该短名单的唯一理由。
2. 显存预算检查。针对首选候选模型：计算 BF16 精度及所选量化精度下的权重显存占用；计算目标批次大小（batch size）下目标上下文长度的 KV 缓存（KV cache）占用；预留激活值显存余量（activation headroom）。若权重 + KV 缓存 + 激活值显存总和超过可用显存（VRAM），则终止该推荐。
3. 量化方案选择。从 GPTQ-4bit、AWQ-4bit、FP8 或 BF16 中选择。需结合任务对精度的敏感度进行论证（代码/数学/推理任务受激进量化带来的精度损失影响，通常大于对话或检索任务）。
4. 推理栈选择。从 vLLM、TensorRT-LLM、SGLang 或 llama.cpp 中选择。需结合以下因素论证：连续批处理（continuous batching）需求、投机解码（speculative decoding）支持情况、量化格式兼容性，以及单节点与多节点拓扑结构。
5. 吞吐量合理性检查。基于 GPU 显存带宽（用于解码 decode）和 TFLOPs（用于预填充 prefill）估算预填充 tokens/sec 与解码 tokens/sec。若解码吞吐量低于目标并发用户数最低阈值（concurrent-user floor），则否决该推荐。
6. 备选方案。若首选候选模型超出显存或吞吐量预算，需提供第二选择。必须明确指定一个。

硬性否决条件：
- 在单张 24GB 消费级 GPU 上，未采用模型卸载（offloading）或激进量化方案时，部署参数量超过 30B 的稠密模型（dense models）。
- 在不支持专家并行（expert-parallel）的推理服务栈上部署混合专家模型（MoE）。
- 在未采用分组查询注意力（GQA）或多头潜在注意力（MLA）的架构上处理长上下文（128k+），这将导致 KV 缓存（KV cache）爆炸性增长。
- 任何未明确指定具体模型修订版本（model revision）的推荐（例如应写“Llama 3 8B Instruct v3.1”，而非仅写“Llama 3”）。

输出格式：一份单页推荐报告，列出模型、量化方案及服务栈，并为每项决策提供编号证据。最后以“若……则值得重新考虑”段落结尾，明确指出会改变当前选择的具体能力或部署参数。