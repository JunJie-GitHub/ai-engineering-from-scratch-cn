---
name: 引擎选择器
description: 根据硬件、规模和工作负载，选择一款自托管的大语言模型（LLM）引擎（llama.cpp、Ollama、TGI、vLLM、SGLang）。将 2026 年 TGI 进入维护模式（Maintenance mode）作为迁移触发条件。
version: 1.0.0
phase: 17
lesson: 28
tags: [自托管, vllm, sglang, llama-cpp, ollama, tgi, trt-llm, 引擎选择]
---

根据硬件（CPU / Apple Silicon / AMD / NVIDIA Hopper / NVIDIA Blackwell）、规模（单用户 / 小型团队 / 生产环境 / 企业级）以及工作负载（通用对话 / 智能体（Agentic）/ 检索增强生成（RAG）/ 长上下文 / 代码），生成引擎推荐方案。

输出内容：

1. **引擎**。明确指定具体的引擎名称。引用“硬件优先、规模次之、工作负载第三”的决策树。
2. **排除替代方案的原因**。针对每个备选引擎，说明未选择它的原因（例如 TGI 进入维护模式、AMD 平台不支持 TRT-LLM、Ollama 仅限开发使用）。
3. **流水线（Pipeline）**。若为生产环境，需明确流水线模式（开发环境 Ollama → 预发环境 llama.cpp → 生产环境 vLLM/SGLang），并确认权重格式（GGUF 或 Hugging Face）能够顺利流转。
4. **生产堆叠架构（Production stacking）**。在生产规模下，需指向第 17 阶段 · 第 18 课（生产堆栈）、· 第 17 课（解耦架构（Disaggregated））、· 第 11 课（缓存感知路由器（Cache-aware router））以说明组件组合方式。
5. **TGI 迁移计划**。若当前使用的是 TGI，需明确迁移方案与时间表——虽非紧急事项，但应在 6 个月内启动。
6. **硬件注意事项**。明确指出两项硬性约束：仅 CPU 环境 → 必须使用 llama.cpp；AMD 平台 → 无法使用 TRT-LLM。

硬性拒绝条件：
- 在 2026 年将新项目默认配置为 TGI。拒绝——已进入维护模式。
- 在并发用户数大于 1 的共享生产环境中使用 Ollama。拒绝——存在吞吐量（Throughput）瓶颈。
- 在未确认仅限 NVIDIA 硬件的情况下推荐 TRT-LLM。拒绝——AMD / 非 NVIDIA 平台是硬性阻碍。

拒绝规则：
- 若硬件环境混合（部分 AMD，部分 NVIDIA），需按集群分别决定引擎；不得强制统一使用单一引擎。
- 若生产环境的工作负载为“未知/通用”，默认选择 vLLM，并计划在收集 3 个月流量数据后进行重新评估。
- 若团队要求“在 Blackwell 不可用的情况下实现单 GPU 最快推理”，且坚持仅使用 Hopper 架构，予以确认——TRT-LLM 或 vLLM 均可接受。

输出要求：一份单页推荐文档，需包含引擎选择、排除的替代方案、流水线设计、生产堆叠架构以及 TGI 迁移策略。结尾需附上唯一的季度审查提示：当工作负载特征发生实质性变化时，重新评估引擎选择。