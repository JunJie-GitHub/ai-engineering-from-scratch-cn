# 自托管服务引擎选型 — llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026年，四大引擎主导了自托管推理（Self-Hosted Inference）领域。选型需基于硬件、规模与生态。**llama.cpp** 在 CPU 上速度最快——支持模型最广，可完全控制量化（Quantization）与线程调度（Threading）。**Ollama** 是开发者笔记本上的一键安装方案，速度比 llama.cpp 慢约 15-30%（受限于 Go + CGo + HTTP 序列化），在类生产负载下吞吐量（Throughput）差距可达 3 倍。**TGI 已于 2025 年 12 月 11 日进入维护模式（Maintenance Mode）**——仅修复漏洞，原始吞吐量比 vLLM 慢约 10%，但历史上在可观测性（Observability）与 Hugging Face 生态集成方面表现最佳。该维护状态使其长期风险较高——对于新项目，SGLang 或 vLLM 是更稳妥的默认选择。**vLLM** 是通用生产环境的默认选择——v0.15.1（2026 年 2 月）新增了对 PyTorch 2.10、RTX Blackwell SM120 及 H200 的优化。**SGLang** 是面向智能体（Agentic）多轮对话/前缀密集型场景的专家——已在生产环境中部署超 40 万张 GPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件限制：仅 CPU → 只能选 llama.cpp。AMD / 非 NVIDIA → 只能选 vLLM（TRT-LLM 仅限 NVIDIA）。2026 年典型流水线模式：开发环境 = Ollama，预发环境 = llama.cpp，生产环境 = vLLM 或 SGLang。全程使用相同的 GGUF/HF 权重（Weights）文件。

**Type:** 学习
**Languages:** Python（标准库，引擎决策树遍历器）
**Prerequisites:** 第 17 阶段所有涵盖引擎的课程（04、06、07、09、18）
**Time:** 约 45 分钟

## 学习目标

- 根据硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 用户 / 100 / 10,000）和工作负载（通用聊天 / 智能体 / 长上下文）选择合适的引擎。
- 说明 TGI 在 2026 年的维护模式状态（2025 年 12 月 11 日生效），以及为何该状态会使新项目更倾向于选择 vLLM 或 SGLang。
- 描述在开发/预发/生产流水线中全程复用相同 GGUF 或 HF 权重文件的实践。
- 解释为何“仅 CPU”环境强制使用 llama.cpp，以及“AMD”环境为何排除 TRT-LLM。

## 问题背景

你的团队启动了一个新的自托管大语言模型（LLM）项目。一位工程师推荐 Ollama，另一位推荐 vLLM，第三位则问：“TGI 不是开箱即用（Out of the Box）吗？”这三种说法在各自适用的场景下都是正确的，但没有一种能通吃所有场景。

在 2026 年，选型决策树（Decision Tree）至关重要：硬件优先，规模次之，工作负载第三。而 2025 年的一项特定事件——TGI 于 12 月 11 日进入维护模式——改变了新项目的默认选型策略。

## 核心概念

### 五大推理引擎 (Inference Engine)

| 引擎 (Engine) | 适用场景 | 备注 |
|--------|----------|-------|
| **llama.cpp** | CPU / 边缘计算 (Edge Computing) / 依赖极少 / 模型支持最广 | CPU 上速度最快，完全可控 |
| **Ollama** | 开发笔记本、单用户、一键安装 | 比 llama.cpp 慢 15-30%；生产环境吞吐量 (Throughput) 差距达 3 倍 |
| **TGI** | Hugging Face (HF) 生态、受监管行业 | **2025 年 12 月 11 日进入维护模式 (Maintenance Mode)** |
| **vLLM** | 通用生产环境、100+ 用户 | 广泛的生产环境默认选择；v0.15.1 于 2026 年 2 月发布 |
| **SGLang** | 智能体 (Agent) 多轮对话、前缀密集型负载 | 生产环境部署超 40 万张 GPU |

### 硬件优先决策原则

**仅使用 CPU** → llama.cpp。Ollama 也可用但速度较慢。在 CPU 上，其他引擎均无竞争力。

**AMD GPU** → vLLM（支持 AMD ROCm）。SGLang 同样可用。TRT-LLM 仅限 NVIDIA，故排除。

**NVIDIA Hopper (H100 / H200)** → vLLM、SGLang 或 TRT-LLM。三者均属顶级。

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM 吞吐量领先（第 17 阶段 · 07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon (M 系列)** → llama.cpp（支持 Metal 加速）。Ollama 底层封装了该引擎。

### 按规模决策原则

**1 名用户 / 本地开发** → Ollama。一条命令，首字元延迟 (Time to First Token, TTFT) 仅需数秒。

**10-100 名用户 / 小型团队** → vLLM 单 GPU 部署。

**100-1 万名用户 / 生产环境** → vLLM 生产栈 (Production Stack)（第 17 阶段 · 18）或 SGLang。

**1 万+ 用户 / 企业级** → vLLM 生产栈 + 解耦架构 (Disaggregated Architecture)（第 17 阶段 · 17）+ LMCache（第 17 阶段 · 18）。

### 按负载类型决策原则

**通用聊天 / 问答** → vLLM 凭借广泛的默认配置胜出。

**智能体多轮交互（工具调用、规划、记忆）** → SGLang 的 RadixAttention（基数注意力机制，第 17 阶段 · 06）占据主导。

**重度前缀复用的检索增强生成 (Retrieval-Augmented Generation, RAG)** → 选择 SGLang。

**代码生成** → vLLM 表现良好；SGLang 在缓存利用上略胜一筹。

**长上下文（128K+）** → vLLM + 分块预填充 (Chunked Prefill)；SGLang + 分层 KV 缓存 (Tiered KV Cache)。

### TGI 的维护模式陷阱

Hugging Face TGI 已于 2025 年 12 月 11 日进入维护模式——此后仅修复漏洞。历史表现：具备顶级的可观测性 (Observability)、业内最佳的 HF 生态集成（模型卡片、安全工具），但原始吞吐量略逊于 vLLM。

对于 2026 年的新项目：默认避开 TGI。现有的 TGI 部署可继续运行，但最终应进行迁移。SGLang 和 vLLM 是更稳妥的默认选择。

### 流水线模式

开发环境（Ollama）→ 预发环境（llama.cpp）→ 生产环境（vLLM）。全程使用相同的 GGUF 或 HF 权重文件。工程师在笔记本上快速迭代；预发环境镜像生产环境的量化 (Quantization) 配置；生产环境作为最终服务目标。

### Ollama 的注意事项

Ollama 非常适合开发环境。但不适用于共享生产环境：Go 语言的 HTTP 序列化 (Serialization) 会增加开销，并发管理 (Concurrency Management) 比 vLLM 更简单，OpenTelemetry 支持相对滞后。在 Ollama 擅长的场景（单用户、一键启动）中使用它，而在共享场景下切换至 vLLM。

### 自建与托管是独立的决策

第 17 阶段 · 01（托管超大规模云厂商）和 · 02（推理平台）已涵盖托管方案。本节假设您已决定自建。自建的理由包括：数据驻留 (Data Residency) 要求、自定义微调、规模化下的总体拥有成本 (Total Cost of Ownership, TCO) 优势，以及托管平台未提供特定领域模型。

### 关键数据备忘

- TGI 进入维护模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月发布；基于 PyTorch 2.10；支持 Blackwell SM120 架构。
- SGLang 生产环境部署规模：超 40 万张 GPU。
- Ollama 与 llama.cpp 的吞吐量差距：慢 15-30%；在生产负载下差距达 3 倍。

## 使用它

`code/main.py` 是一个决策树遍历器（decision-tree walker）：根据硬件配置、规模和工作负载，自动选择推理引擎并给出选择理由。

## 部署上线

本课时将生成 `outputs/skill-engine-picker.md` 文件。该脚本会根据给定的约束条件选择推理引擎，并撰写迁移方案。

## 练习

1. 使用你的硬件配置、规模和工作负载运行 `code/main.py`。输出结果是否符合你的预期？
2. 你的基础设施包含 12 张 H100 和 8 张 AMD MI300X。应该选择哪个引擎？为什么 TRT-LLM 不在考虑范围内？
3. 某团队计划在 2026 年继续使用 TGI，理由是“我们只熟悉这个”。请论证迁移的必要性。
4. 从 Ollama 开发环境迁移至 vLLM 生产环境：在量化（quantization）、配置和可观测性（observability）方面需要做哪些调整？
5. 某检索增强生成（RAG）产品，其 P99 前缀长度为 8K，且跨租户复用率较高。请选择一个引擎，并将其与第 17 阶段 · 11 + 18 的技术栈进行整合。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| llama.cpp | “跑在 CPU 上的那个” | 模型支持最广，CPU 推理速度最快 |
| Ollama | “笔记本上用的那个” | 一条命令安装，吞吐量适合开发环境 |
| TGI | “Hugging Face 的推理服务” | 自 2025 年 12 月起进入维护模式 |
| vLLM | “默认首选” | 2026 年广泛采用的生产环境基线 |
| SGLang | “面向智能体（Agent）的那个” | 侧重前缀缓存，采用基数注意力机制（RadixAttention） |
| TRT-LLM | “仅限 NVIDIA” | Blackwell 架构（Blackwell）吞吐量领先，仅支持 NVIDIA 硬件 |
| GGUF | “llama.cpp 的格式” | 内置多种 K 位量化（K-quant）变体 |
| Production-stack | “vLLM 的 K8s 部署” | 第 17 阶段 · 18 的参考部署方案 |
| Pipeline pattern | “开发→预发→生产” | 基于相同模型权重，从 Ollama → llama.cpp → vLLM 的流水线模式 |

## 延伸阅读

- [AI Made Tools — vLLM 对比 Ollama、llama.cpp 与 TGI（2026 版）](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp 对比 Ollama（2026 版）](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — 大语言模型推理引擎全面对比](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 2026 年十大 vLLM 替代方案](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI 维护公告](https://github.com/huggingface/text-generation-inference) — 版本说明。
- [vLLM v0.15.1 版本说明](https://github.com/vllm-project/vllm/releases)