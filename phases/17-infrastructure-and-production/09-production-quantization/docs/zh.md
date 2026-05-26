# 生产环境量化（Production Quantization）— AWQ、GPTQ、GGUF K-quants、FP8、MXFP4/NVFP4

> 量化（Quantization）格式并非通用选择——它取决于硬件、推理引擎（serving engine）和工作负载（workload）。GGUF Q4_K_M 或 Q5_K_M 主导了 CPU 和边缘计算场景，主要通过 llama.cpp 和 Ollama 交付。当需要在同一基座模型上运行多 LoRA（multi-LoRA）时，GPTQ 在 vLLM 中表现最佳。AWQ 配合 Marlin-AWQ 内核在 7B 级别模型上可实现约 741 tok/s 的吞吐量，并在 INT4 精度下取得最佳的 Pass@1 表现——这已成为 2026 年数据中心生产环境的默认选择。FP8 在 Hopper、Ada 和 Blackwell 架构上保持折中定位——近乎无损且获得广泛支持。NVFP4 和 MXFP4（Blackwell 微缩放（microscaling）格式）较为激进，需要逐块验证。团队常踩的两个陷阱是：校准数据集（calibration dataset）必须与部署领域匹配；以及 KV 缓存（KV cache）与权重量化是分离的——AWQ 的经验教训“我的模型现在只有 4 GB”往往忽略了在生产批次大小下 KV 缓存仍需 10-30 GB 的事实。

**Type:** 学习
**Languages:** Python（标准库，跨格式的简易内存与吞吐量对比）
**Prerequisites:** 第 10 阶段 · 13（量化基础），第 17 阶段 · 04（vLLM 推理服务内部原理）
**Time:** 约 75 分钟

## 学习目标

- 列举六种生产环境量化格式及其在 2026 年的优势区间。
- 根据硬件（CPU 与 GPU、Hopper 与 Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（常规对话、逻辑推理、多 LoRA）选择合适的格式。
- 计算所选格式节省的权重内存以及未受影响的 KV 缓存大小。
- 指出会导致量化模型在垂直领域流量上性能下降的校准数据集陷阱。

## 问题背景

量化能够降低内存占用和高带宽内存（HBM）带宽需求，而这正是解码（decode）阶段所急需的。一个 FP16 精度的 70B 模型权重约为 140 GB。将权重量化至 INT4（AWQ 或 GPTQ）后，模型体积降至 35 GB——足以装入单张 H100 显卡，并为 KV 缓存留出空间。这一点至关重要，因为在 128 个并发序列且上下文（context）长度为 2k 的情况下，仅 KV 缓存就需要 20-30 GB。

但量化并非没有代价。激进的量化会降低模型质量，尤其是在重度推理任务上。不同的格式适配不同的引擎，不同的硬件原生支持不同的精度。2026 年的量化格式生态确实繁杂，你无法直接照搬他人的选择——必须根据自身技术栈进行决策。

## 核心概念

### 六种格式

| 格式 | 位宽 | 最佳适用场景 | 推理引擎 |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘设备、笔记本电脑 | llama.cpp, Ollama |
| GPTQ | 4-8 | vLLM 上的多 LoRA 部署 | vLLM, TGI |
| AWQ | 4 | 数据中心 GPU 生产环境 | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell 多用户场景 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户场景 | TRT-LLM |

### GGUF —— CPU/边缘设备的默认选择

GGUF 是一种文件格式，而非严格意义上的量化方案 (quantization scheme) —— 它将多种 K-quant 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包在一个容器中。Q4_K_M 和 Q5_K_M 是生产环境的默认选择，能在 4-5 位宽下提供接近 BF16 的质量。由于 llama.cpp 是目前最快的 CPU 推理引擎 (inference engine)，因此它是 CPU 或边缘设备部署的最佳选择。

在 vLLM 中的吞吐量惩罚 (throughput penalty)：7B 模型约为 93 tok/s —— 该格式并未针对 GPU 内核进行优化。仅当部署目标为 CPU/边缘设备时才使用 GGUF，其他情况不建议使用。

### GPTQ —— vLLM 中的多 LoRA 支持

GPTQ 是一种包含校准步骤的训练后量化算法 (post-training quantization algorithm)。Marlin 内核使其在 GPU 上运行极快（相比非 Marlin 版 GPTQ 提速 2.6 倍）。7B 模型吞吐量约为 712 tok/s。

其独特优势在于：GPTQ-Int4 在 vLLM 中支持 LoRA 适配器。如果你需要部署一个基础模型外加 10-50 个微调变体（每个均为 LoRA），GPTQ 是你的必经之路。截至 2026 年初，NVFP4 尚不支持 LoRA。

### AWQ —— 数据中心 GPU 的默认选择

激活感知权重量化 (Activation-aware Weight Quantization)。在量化过程中保护约 1% 最显著的权重。Marlin-AWQ 内核：相比朴素实现提速 10.9 倍。7B 模型吞吐量约为 741 tok/s，在所有 INT4 格式中 Pass@1 指标最佳。

除非你需要多 LoRA 支持（选 GPTQ）或追求极致的 Blackwell FP4 性能（选 NVFP4），否则新的 GPU 部署应首选 AWQ。

### FP8 —— 可靠的折中之选

8 位浮点数 (8-bit floating point)。近乎无损。支持广泛。Hopper Tensor Core 原生加速 FP8，Blackwell 架构也继承了这一特性。当质量不可妥协时（如推理、医疗、代码生成），FP8 是 2026 年最稳妥的默认选择。其内存节省幅度仅为 INT4 的一半，但质量风险却低得多。

### MXFP4 / NVFP4 —— Blackwell 的激进之选

微缩放 FP4 (Microscaling FP4)。每个权重块拥有独立的缩放因子。方案激进，但在 Blackwell Tensor Core 上具备硬件加速支持。相比 FP8，每 token 字节数减半——这正是第 17 阶段 · 07 中提到的经济性优势。

注意事项：
- 截至 2026 年初尚不支持 LoRA。
- 在重度推理工作负载上会出现明显的质量下降。
- 务必针对每个模型在评估集上进行验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集 (calibration dataset) —— 通常为 C4 或 WikiText。对于垂直领域模型（代码、医疗、法律），若使用通用网络文本进行校准，算法可能会错误判断需要保护的权重。这会导致 HumanEval 的 Pass@1 指标下降数个百分点。

解决方法：使用领域内数据进行校准。通常数百条领域样本即可满足需求。在发布前务必在评估集上进行测试。

### KV Cache 陷阱

AWQ 将权重压缩至 4 位。但键值缓存 (KV cache) 是独立的，通常保持 FP16/FP8 精度。以 70B 模型搭配 AWQ 为例：

- 权重：约 35 GB（INT4 格式，原始为 140 GB）。
- KV Cache（128 并发 × 2k 上下文）：约 20 GB。
- 激活值 (activations)：约 5 GB。
- 总计：约 60 GB —— 可装入 H100 80GB 显存。

天真地认为“我把模型量化到了 4 GB”会忽略掉另外 30-50 GB 的开销。必须整体规划高带宽内存 (HBM) 预算。

另外，KV Cache 量化（FP8 KV 或 INT8 KV）是另一种独立的选择，各有其权衡取舍——它会直接影响注意力机制的精度，并非毫无代价的优化。

### AWQ INT4 对推理任务存在风险

思维链 (Chain-of-thought)、数学计算、长上下文代码生成等任务会因激进量化而受到明显影响。AWQ INT4 在 MATH 基准测试上会损失约 3-5 分。对于重度推理工作负载，请部署 FP8 或 BF16，并接受相应的内存开销。

### 2026 选型指南

- CPU/边缘设备部署：GGUF Q4_K_M。直接选用即可。
- GPU 部署，常规对话，无需 LoRA：AWQ。
- GPU 部署，多 LoRA：搭配 Marlin 的 GPTQ。
- 推理工作负载：FP8。
- Blackwell 数据中心，质量已验证：NVFP4 + FP8 KV。
- 难以抉择时：针对每种候选格式运行 1,000 条样本的评估测试。

## 使用它

`code/main.py` 计算了不同模型规模下六种格式的内存占用 (Memory Footprint)（权重 + KV 缓存 (KV Cache) + 激活值 (Activations)）及相对吞吐量 (Throughput)。该脚本展示了 KV 缓存何时成为主导因素、权重压缩 (Weight Compression) 何时能带来收益，以及何时选择 FP8 (8位浮点数) 是最稳妥的方案。

## 交付它

本节将生成 `outputs/skill-quantization-picker.md`。根据给定的硬件配置、模型规模、工作负载类型和质量容忍度，该脚本会推荐合适的量化格式，并生成校准 (Calibration) 与验证 (Validation) 计划。

## 练习

1. 运行 `code/main.py`。针对 70B 模型、128 并发和 2k 上下文长度，计算每种格式的总 HBM (高带宽内存) 占用。哪种格式能让你将其部署在单张 80GB 的 H100 显卡上？
2. 你有一个 7B 的代码生成模型。请选择一种量化格式并说明理由。如果实际的质量容忍度与预期不符，你的回退/恢复方案是什么？
3. 计算为医疗领域模型校准 AWQ (激活值感知量化) 所需的校准数据集 (Calibration Dataset) 规模。为什么数据量并非越多越好？
4. 阅读 Marlin-AWQ 内核 (Kernel) 的论文或发布说明。用三句话解释为什么 AWQ 在 7B 模型上能达到 741 tok/s (词元/秒)，而原始 GPTQ (生成式预训练量化) 仅能达到约 712 tok/s。
5. 在什么情况下，将 AWQ 权重与 FP8 KV 缓存结合使用比保持 KV 缓存为 BF16 (Bfloat16) 更合理？

## 关键术语

| 术语 | 常见叫法 | 实际含义 |
|------|----------------|------------------------|
| GGUF | “llama.cpp 格式” | 打包了 K-quant (K 量化) 变体的文件格式；CPU/边缘端默认选择 |
| Q4_K_M | “Q4 K M” | 4-bit K-quant medium（4 位 K 量化中等精度）；生产环境 GGUF 默认配置 |
| GPTQ | “gee pee tee q” | 基于校准的后训练 INT4 (4位整型) 量化；在 vLLM 中支持 LoRA (低秩自适应) |
| AWQ | “a w q” | 激活值感知 INT4 量化；采用 Marlin 内核 (Marlin Kernels)；在 INT4 下 Pass@1 (首次通过率) 表现最佳 |
| Marlin kernels | “快速 INT4 内核” | 专为 Hopper 架构 INT4 优化的自定义 CUDA 内核 (CUDA Kernels)；带来 10 倍加速 |
| FP8 | “八位浮点数” | Hopper/Ada/Blackwell (GPU 架构) 上稳妥的默认精度 |
| MXFP4 / NVFP4 | “微缩放四位” | Blackwell 架构的 4 位浮点数，采用逐块缩放因子 |
| Calibration dataset | “校准数据” | 用于确定量化参数的输入文本；必须与目标领域匹配 |
| KV cache quantization | “KV INT8” | 独立于权重的选择；直接影响注意力机制的精度 |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 对比基准测试。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 各格式的吞吐量数据。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 逐格式选型指南。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持的格式与配置参数 (Flags)。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — AWQ 原始论文。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — GPTQ 原始论文。