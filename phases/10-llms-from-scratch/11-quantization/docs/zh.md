# 量化（Quantization）：让模型适配硬件

> 16位浮点数（FP16）精度的 70B 模型需要 140GB 显存。仅加载权重就需要两张 A100。量化至 8位浮点数（FP8）：一张 80GB 显存的 GPU 即可。量化至 4位整数（INT4）：一台 MacBook 就能跑。

**类型：** 实战构建
**编程语言：** Python（使用 numpy）
**前置知识：** 第 10 阶段，第 01-10 课（从零构建大语言模型）
**预计耗时：** 约 120 分钟

## 学习目标

- 实现从 16位浮点数（FP16）到 8位整数（INT8）和 4位整数（INT4）的对称量化（Symmetric Quantization）与非对称量化（Asymmetric Quantization），包括逐张量（Per-tensor）和逐通道（Per-channel）缩放
- 计算量化带来的显存节省量，并确定哪种精度适配指定 GPU 的显存（VRAM）
- 解释训练后量化（Post-training Quantization, PTQ）与量化感知训练（Quantization-aware Training, QAT）的区别
- 应用 GPTQ 或 AWQ 对真实模型进行量化，并在基准测试（Benchmark）上评估精度与显存占用的权衡

## 问题背景

Llama 3 70B 拥有 700 亿个参数。每个参数都是一个 16位浮点数（FP16）。这意味着需要 1400 亿字节，即 140GB 的存储空间。而单张 A100 显卡仅有 80GB 显存（VRAM）。你甚至无法在单张 GPU 上加载权重，更不用说运行推理（Inference）了。为了部署这一个模型，你需要两张 A100，每张每小时租金 2 美元。

但每个参数使用 16 位是一种浪费。神经网络中的大多数权重都集中在零附近。16位浮点数（FP16）的完整动态范围（从 0.000000059 到 65,504）几乎完全没有被利用。如果你测量 Llama 3 70B 权重的实际分布，会发现 95% 的权重都落在 -0.1 到 +0.1 之间。你实际上是在用 16 位去表示那些仅需 4 位就能容纳的数值。

量化（Quantization）的核心思想是用低精度数值替换高精度数值。从 16位浮点数（FP16）降至 8位浮点数（FP8），显存占用减半；降至 4位整数（INT4），则缩减为原来的四分之一。那个 140GB 的模型将变成 35GB，足以装入单张消费级显卡。如果进一步采用 2位量化（激进、有损，但适用于部分任务），同样的模型甚至能在 16GB 显存的笔记本电脑上运行。

代价是精度损失。每减少一位都会破坏部分信息。关键在于你会损失多少精度，以及损失在哪些地方。一个经过良好量化的 4位整数（INT4）模型在大多数基准测试（Benchmark）上能保留原始模型 95% 到 99% 的性能。而粗暴的 4位整数（INT4）量化则可能彻底毁掉模型。两者的区别在于技术手段。

社区使用 GPTQ 将 Llama 3 量化至 4位整数（INT4）的实验表明，在 WikiText 数据集上的困惑度（Perplexity）仅上升约 1-2 个点。Mistral 发布的 Mixtral 8x22B 8位浮点数（FP8）模型检查点（Checkpoints）在 MMLU 测试中未出现可测量的质量损失。GGUF 格式驱动了 llama.cpp，使得搭载 M 系列芯片的 MacBook 能够运行 70B 模型。量化绝非一种临时取巧手段（Hack），而是所有参数量大于 7B 模型的标准部署路径。

## 核心概念

### 数值格式：每一位的作用

每个浮点数（floating-point number）都包含三个部分：符号位（sign）、指数位（exponent）和尾数位（mantissa，也称有效数字 significand）。符号位占 1 位。指数位决定数值的范围（即数字能有多大或多小）。尾数位决定精度（即你能获得多少位小数）。

FP32:  [1 sign] [8 exponent] [23 mantissa]  = 32 bits
FP16:  [1 sign] [5 exponent] [10 mantissa]  = 16 bits
BF16:  [1 sign] [8 exponent] [7  mantissa]  = 16 bits
FP8:   [1 sign] [4 exponent] [3  mantissa]  = 8  bits (E4M3)
FP8:   [1 sign] [5 exponent] [2  mantissa]  = 8  bits (E5M2)
INT8:  [1 sign] [7 value]                   = 8  bits (uniform steps)
INT4:  [1 sign] [3 value]                   = 4  bits (16 levels total)

**FP32** 代表全精度（full precision）。23 位尾数可提供约 7 位十进制有效数字。其范围约为 1.2 x 10^-38 至 3.4 x 10^38。早期的模型训练完全在 FP32 下进行。如今，它仍专门用于累加操作（accumulation，即矩阵乘法过程中的累加求和）。

**FP16** 将位数减半。10 位尾数提供约 3.3 位十进制精度。指数位缩减至 5 位，大幅缩小了数值范围（最大值约 65,504）。这对于权重（通常集中在零附近）来说没问题，但对于训练期间可能激增的激活值（activations）和梯度（gradients）则很危险。FP16 训练需要损失缩放（loss scaling）来防止下溢（underflow）。

**BF16**（Brain Float 16）保留了 FP32 的 8 位指数位，但将尾数缩减至 7 位。其范围与 FP32 相同，但精度低于 FP16。Google 专为深度学习设计了该格式。其核心直觉是：对神经网络而言，范围比精度更重要。在 FP16 中会下溢为零的 10^-20 梯度，在 BF16 中得以保留。在 BF16 中舍入为 0.0734 的 0.07342 权重也足够接近。如今所有的现代训练任务都使用 BF16 或 BF16/FP32 混合精度。

**FP8** 有两种变体。E4M3（4 位指数，3 位尾数）用于推理阶段的权重和激活值。E5M2（5 位指数，2 位尾数）用于训练阶段的梯度，此时范围比精度更重要。在 H100 GPU 上进行 FP8 推理相比 FP16 可实现 30-50% 的加速，且质量损失微乎其微。

**INT8** 是一种整数格式。没有指数位，也没有尾数位。仅有从 -128 到 127 的 256 个均匀间隔的值。你需要一个缩放因子（scale factor）将浮点权重映射到该范围内。其优势在于：整数运算比浮点运算更快且更节能。在 A100 上，INT8 矩阵乘法的算力可达 624 TOPS，而 FP16 为 312 TFLOPS。

**INT4** 则更进一步。仅有 16 个可能的取值。缩放因子承担了主要工作。模型质量完全取决于你如何选择缩放因子以及对哪些权重进行量化（quantize）。最先进的 INT4 方法（如 GPTQ、AWQ）能保留原始模型 95% 以上的质量。

graph LR
    subgraph Formats["Number Format Landscape"]
        direction TB
        FP32["FP32\n32 bits\n4 bytes/param\nTraining gold standard"]
        BF16["BF16\n16 bits\n2 bytes/param\nTraining default"]
        FP16["FP16\n16 bits\n2 bytes/param\nInference baseline"]
        FP8["FP8\n8 bits\n1 byte/param\n30-50% faster"]
        INT8["INT8\n8 bits\n1 byte/param\n2x throughput"]
        INT4["INT4\n4 bits\n0.5 bytes/param\n4x compression"]
    end

    FP32 -->|"training"| BF16
    BF16 -->|"inference"| FP16
    FP16 -->|"H100 native"| FP8
    FP16 -->|"server deploy"| INT8
    FP16 -->|"edge/laptop"| INT4

    style FP32 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style BF16 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style FP16 fill:#1a1a2e,stroke:#ffa500,color:#fff
    style FP8 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style INT8 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style INT4 fill:#1a1a2e,stroke:#e94560,color:#fff

### 量化（Quantization）的工作原理

核心操作很简单。获取一个浮点数值张量（tensor），计算缩放因子，进行乘法运算，四舍五入到最接近的整数，然后存储这些整数及缩放因子。

**量化（Quantize）：**
scale = max(abs(tensor)) / max_int_value
quantized = round(tensor / scale)

**反量化（Dequantize）：**
reconstructed = quantized * scale

对于对称范围（-127 到 127）的 INT8：
scale = max(abs(tensor)) / 127
quantized = clamp(round(tensor / scale), -128, 127)

误差即为舍入误差。每个值的偏差最多为 `scale / 2`。整个层的总误差取决于权重的数量以及模型对这些权重扰动的敏感程度。

**逐张量化（Per-tensor quantization）与逐通道量化（Per-channel quantization）**。逐张量化为整个权重矩阵使用一个缩放因子。实现简单但有损：如果某一列数值较大而另一列较小，较小数值会损失大部分精度。逐通道量化为每个输出通道（权重矩阵的每一行或列）使用一个缩放因子。开销更大（需存储 N 个缩放因子而非 1 个），但质量显著提升。所有生产环境的量化方法均采用逐通道或更细粒度的策略。

**非对称量化（Asymmetric quantization）** 增加了一个零点偏移（zero-point offset）：`quantized = round(tensor / scale) + zero_point`。这用于处理不以零为中心的分布。例如，ReLU 激活值始终为非负数。对称量化会将一半的整数范围浪费在永远不会出现的负值上。非对称量化则将实际范围 [min, max] 映射到完整的整数范围。

### 敏感度层级

模型中的各个组件对量化的容忍度并不相同。存在一个明确的敏感度层级。

**权重（最稳健）**。模型权重在训练期间变化缓慢，且大致遵循以零为中心的高斯分布（Gaussian distribution）。它们非常适合量化。采用逐通道缩放的 INT8 权重几乎能实现无损结果。INT4 需要更复杂的方法，但同样可行。

**激活值（中等敏感度）**。激活值是推理过程中流经网络的中间值。它们的动态范围比权重更广，且包含异常值（outliers）。单个注意力头（attention head）产生的激活值可能比平均值大 100 倍。这些异常值对模型质量至关重要。粗暴地量化它们会破坏信息。解决方案：将异常值通道保留在更高精度（如 LLM.int8()），或使用逐 token 或逐通道的激活值缩放。

**KV 缓存（高敏感度）**。键值缓存（key-value cache）存储所有先前 token 的注意力状态。在长上下文场景下，KV 缓存占据主导内存。对于 32K 上下文的 70B 模型，仅 KV 缓存在 FP16 下就达 40GB。将 KV 缓存量化为 FP8 或 INT8 可节省大量内存，但任何误差都会在后续所有注意力计算中累积。其对质量的影响随序列长度增加而放大。

**注意力 Logits（最敏感）**。注意力机制中的 softmax 对其输入的微小变化高度敏感。softmax 前 logits 中 0.01 的量化误差就足以显著改变注意力分布。大多数量化方案即使在其他部分都已量化的情况下，仍会将注意力计算保留在更高精度（FP16 或 BF16）。

graph TD
    subgraph Sensitivity["Quantization Sensitivity (Low to High)"]
        direction LR
        W["Weights\nGaussian, near zero\nINT4 works well"]
        A["Activations\nWider range, outliers\nINT8 with care"]
        KV["KV Cache\nErrors compound\nFP8 or INT8"]
        ATT["Attention Logits\nSoftmax amplifies error\nKeep in FP16"]
    end

    W -->|"safe"| A
    A -->|"careful"| KV
    KV -->|"dangerous"| ATT

    style W fill:#1a1a2e,stroke:#51cf66,color:#fff
    style A fill:#1a1a2e,stroke:#ffa500,color:#fff
    style KV fill:#1a1a2e,stroke:#e94560,color:#fff
    style ATT fill:#1a1a2e,stroke:#ff0000,color:#fff

### PTQ 与 QAT

**训练后量化（Post-Training Quantization, PTQ）** 对已训练好的模型进行量化。无需重新训练。你只需获取 FP16 权重，计算缩放因子，进行舍入，然后部署。速度快（几分钟到几小时）且成本低。对 INT8 和 FP8 效果良好。对于 INT4，朴素的 PTQ 往往因舍入误差累积而表现极差。先进的 PTQ 方法（如 GPTQ、AWQ）会使用校准数据（calibration data）来最小化量化误差。

**量化感知训练（Quantization-Aware Training, QAT）** 在训练的前向传播（forward pass）中插入伪量化操作。模型会学习将权重放置在舍入误差较小的位置。梯度通过伪量化操作流动，使用的是直通估计器（Straight-Through Estimator, STE）：即假设舍入操作的梯度为 1。QAT 生成的 INT4 和 INT2 模型优于 PTQ，但需要完整的训练流程。Google 在 Gemini 的高效服务中使用了 QAT。Meta 也在部分 Llama 的部署目标中采用了 QAT。

| 方面 | PTQ | QAT |
|--------|-----|-----|
| 成本 | 几分钟到几小时 | 完整训练流程 |
| INT8 质量 | 极佳（损失 < 0.1%） | 极佳 |
| INT4 质量 | 配合 GPTQ/AWQ 良好（损失 1-3%） | 更好（损失 < 1%） |
| INT2 质量 | 较差 | 部分任务可用 |
| 校准数据 | 128-1024 个样本 | 完整训练数据集 |
| 适用场景 | 部署、快速迭代 | 低比特宽度下的极致质量 |

### GPTQ、AWQ 与 GGUF

**GPTQ（GPT Quantization）** 是一种一次性 PTQ 方法。它逐层量化权重，使用小型校准数据集（通常为 128 个样本）来测量海森矩阵（Hessian，即输出对每个权重敏感度的二阶信息）。海森矩阵判定为重要的权重会被更精细地量化。GPTQ 是首个使 INT4 量化在大语言模型（Large Language Model, LLM）中具备实用价值的方法。Hugging Face 上的 TheBloke 通过发布数百个模型的量化版本，极大地推广了 GPTQ。

**AWQ（Activation-Aware Weight Quantization，激活感知权重量化）** 发现一小部分权重（约 1%）具有不成比例的重要性，因为它们会与较大的激活值相乘。AWQ 利用校准数据识别这些显著权重，并在量化前将其放大（随后将对应的激活值缩小）。这使得重要权重保持在 INT4 量化精度较高的范围内。AWQ 的质量通常与 GPTQ 持平或略优，且应用速度快 1.5-2 倍。

**GGUF（GPT-Generated Unified Format）** 是 llama.cpp 及其生态系统使用的文件格式。它支持混合量化：不同层采用不同的比特宽度。首层和末层（嵌入层和输出头）通常保留较高精度。中间层使用 INT4 或 INT3。GGUF 文件是自包含的：权重、分词器（tokenizer）、元数据全部集成在一个文件中。该格式专为 CPU 推理和 Apple Silicon 设计，其标准路径是将整个模型加载到内存中，并在 CPU 或 Metal GPU 上运行矩阵乘法。Q4_K_M 是最流行的 GGUF 量化变体，在质量和体积之间取得了良好平衡。

graph TD
    subgraph Methods["Quantization Methods"]
        direction TB
        GPTQ_["GPTQ\nHessian-guided\nPer-layer optimization\nPopular on HuggingFace"]
        AWQ_["AWQ\nActivation-aware\nSalient weight scaling\n1.5-2x faster than GPTQ"]
        GGUF_["GGUF\nMixed precision\nCPU + Metal optimized\nllama.cpp ecosystem"]
    end

    subgraph Use["Best For"]
        GPU["GPU inference\n(CUDA, ROCm)"]
        EDGE["Edge / Laptop\n(CPU, Metal)"]
    end

    GPTQ_ --> GPU
    AWQ_ --> GPU
    GGUF_ --> EDGE

    style GPTQ_ fill:#1a1a2e,stroke:#ffa500,color:#fff
    style AWQ_ fill:#1a1a2e,stroke:#51cf66,color:#fff
    style GGUF_ fill:#1a1a2e,stroke:#0f3460,color:#fff

### 质量评估

如何判断量化后的模型是否依然优秀？

**困惑度（Perplexity）**。最常用的指标。数值越低越好。在预留数据集（通常使用 WikiText-2）上分别计算原始模型和量化模型的困惑度。两者的差值（delta）告诉你量化破坏了多少信息。经验法则：差值 < 0.5 为极佳，0.5-1.0 为良好，1.0-2.0 对大多数任务可接受，> 2.0 则说明出了问题。

**特定任务基准测试**。在 MMLU、HumanEval、GSM8K 或自定义评估套件上运行量化模型。与原始模型进行对比。量化对不同能力的影响并不均匀。数学和代码任务对精度损失的敏感度高于通用知识任务。

**输出对比**。让两个模型对相同的提示词（prompts）生成回复并进行比较。大模型作为裁判（LLM-as-judge，见第 10 课）在此场景下效果很好。计算胜率：量化模型在多大比例的提示词上能够匹配或超越原始模型？

**延迟与吞吐量**。量化的目的是让模型更快、更便宜。测量每秒生成 token 数、首 token 延迟（time to first token）以及内存占用。如果量化后的模型比原始模型还慢，那就毫无意义。

| 模型 | 格式 | 大小 | 困惑度 (WikiText-2) | MMLU | 吞吐量 (A100, tokens/sec) |
|-------|--------|------|------------------------|------|-------------------|
| Llama 3 70B | FP16 | 140GB | 3.12 | 79.5% | 38 |
| Llama 3 70B | FP8 | 70GB | 3.14 | 79.3% | 55 |
| Llama 3 70B | GPTQ INT4 | 35GB | 4.32 | 77.8% | 72 |
| Llama 3 70B | AWQ INT4 | 35GB | 4.18 | 78.1% | 75 |
| Llama 3 70B | GGUF Q4_K_M | 40GB | 4.25 | 77.9% | 28 (CPU) |

规律如下：FP8 几乎零成本。INT4 会损失 1-2 个 MMLU 分数，但吞吐量翻倍，内存占用降至四分之一。对于几乎所有部署场景而言，这种权衡都是值得的。

### 实际数据

H100 上从 FP16 到 FP8：推理速度提升 30-50%，质量损失 < 0.1%。这是无需犹豫的量化选择。每个 H100 部署都应采用它。

从 FP16 到 INT8（LLM.int8()）：内存减半，质量损失 < 0.5%。这种混合精度方法将异常特征保留在 FP16，而将其余部分量化为 INT8。

从 FP16 到 INT4（GPTQ/AWQ）：内存降至四分之一，质量损失 1-3%（取决于模型和方法）。使得单张 48GB GPU 能够运行 70B 模型。

从 FP16 到 INT4（GGUF Q4_K_M）：内存减少至约 1/3.5，质量损失 1-2%。专为 CPU 推理优化。Q4_K_M 格式的 70B 模型大小约 40GB，在配备 64GB 内存的 M3 Max 上运行速度为 10-15 tokens/秒。

从 FP16 到 INT2：内存减少至 1/8，质量损失 5-15%。仅适用于可容忍性能下降的特定窄域任务。目前仍处于研究前沿，尚未达到通用生产就绪标准。

## 构建

### 步骤 1：数值格式表示

构建每种格式的位级表示（bit-level representation），以精确观察符号位（sign）、指数位（exponent）和尾数位（mantissa）的具体作用。

import numpy as np


def float_to_fp32_bits(value):
    bits = np.float32(value).view(np.uint32)
    sign = (bits >> 31) & 1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return {"sign": int(sign), "exponent": int(exponent), "mantissa": int(mantissa),
            "exponent_bits": format(int(exponent), '08b'),
            "mantissa_bits": format(int(mantissa), '023b'),
            "value": float(value),
            "actual_exponent": int(exponent) - 127}


def float_to_fp16_bits(value):
    fp16 = np.float16(value)
    bits = fp16.view(np.uint16)
    sign = (bits >> 15) & 1
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x3FF
    return {"sign": int(sign), "exponent": int(exponent), "mantissa": int(mantissa),
            "exponent_bits": format(int(exponent), '05b'),
            "mantissa_bits": format(int(mantissa), '010b'),
            "value": float(fp16),
            "actual_exponent": int(exponent) - 15}


def float_to_bf16_bits(value):
    fp32_bits = np.float32(value).view(np.uint32)
    bf16_bits = (fp32_bits >> 16).astype(np.uint16)
    sign = (bf16_bits >> 15) & 1
    exponent = (bf16_bits >> 7) & 0xFF
    mantissa = bf16_bits & 0x7F
    reconstructed = np.uint32(bf16_bits.astype(np.uint32) << 16).view(np.float32)
    return {"sign": int(sign), "exponent": int(exponent), "mantissa": int(mantissa),
            "exponent_bits": format(int(exponent), '08b'),
            "mantissa_bits": format(int(mantissa), '07b'),
            "value": float(reconstructed),
            "actual_exponent": int(exponent) - 127}


def simulate_fp8_e4m3(value):
    sign = 1 if value < 0 else 0
    abs_val = abs(value)
    max_val = 448.0
    abs_val = min(abs_val, max_val)
    if abs_val == 0:
        return {"sign": sign, "exponent": 0, "mantissa": 0, "value": 0.0,
                "exponent_bits": "0000", "mantissa_bits": "000"}
    exp = int(np.floor(np.log2(abs_val)))
    exp = max(-6, min(8, exp))
    mantissa_val = abs_val / (2.0 ** exp) - 1.0
    mantissa_quant = round(mantissa_val * 8) / 8
    mantissa_quant = max(0, min(0.875, mantissa_quant))
    reconstructed = (1.0 + mantissa_quant) * (2.0 ** exp)
    if sign:
        reconstructed = -reconstructed
    mantissa_int = int(round(mantissa_quant * 8))
    return {"sign": sign, "exponent": exp + 7, "mantissa": mantissa_int,
            "exponent_bits": format(exp + 7, '04b'),
            "mantissa_bits": format(mantissa_int, '03b'),
            "value": float(reconstructed),
            "actual_exponent": exp}


def display_format_comparison(value):
    fp32 = float_to_fp32_bits(value)
    fp16 = float_to_fp16_bits(value)
    bf16 = float_to_bf16_bits(value)
    fp8 = simulate_fp8_e4m3(value)

    print(f"\n  Value: {value}")
    print(f"  {'Format':<8} {'Stored Value':>14} {'Error':>12} {'Sign':>5} {'Exp Bits':>10} {'Man Bits':>25}")
    print(f"  {'-'*76}")
    print(f"  {'FP32':<8} {fp32['value']:>14.6f} {abs(fp32['value'] - value):>12.8f} {fp32['sign']:>5} {fp32['exponent_bits']:>10} {fp32['mantissa_bits']:>25}")
    print(f"  {'FP16':<8} {fp16['value']:>14.6f} {abs(fp16['value'] - value):>12.8f} {fp16['sign']:>5} {fp16['exponent_bits']:>10} {fp16['mantissa_bits']:>25}")
    print(f"  {'BF16':<8} {bf16['value']:>14.6f} {abs(bf16['value'] - value):>12.8f} {bf16['sign']:>5} {bf16['exponent_bits']:>10} {bf16['mantissa_bits']:>25}")
    print(f"  {'FP8e4m3':<8} {fp8['value']:>14.6f} {abs(fp8['value'] - value):>12.8f} {fp8['sign']:>5} {fp8['exponent_bits']:>10} {fp8['mantissa_bits']:>25}")

### 步骤 2：对称量化（逐张量与逐通道）

基础的量化操作。逐张量量化（per-tensor quantization）为整个矩阵使用单一缩放因子（scale）。逐通道量化（per-channel quantization）则为每一行或每一列分别分配一个缩放因子。

def quantize_symmetric(tensor, num_bits=8):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    abs_max = np.max(np.abs(tensor))
    if abs_max == 0:
        return np.zeros_like(tensor, dtype=np.int32), 1.0
    scale = abs_max / qmax
    quantized = np.clip(np.round(tensor / scale), qmin, qmax).astype(np.int32)
    return quantized, float(scale)


def dequantize_symmetric(quantized, scale):
    return quantized.astype(np.float64) * scale


def quantize_per_channel(tensor, num_bits=8, axis=0):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    if axis == 0:
        abs_max = np.max(np.abs(tensor), axis=1, keepdims=True)
    else:
        abs_max = np.max(np.abs(tensor), axis=0, keepdims=True)

    abs_max = np.where(abs_max == 0, 1.0, abs_max)
    scales = abs_max / qmax
    quantized = np.clip(np.round(tensor / scales), qmin, qmax).astype(np.int32)
    return quantized, scales.squeeze()


def dequantize_per_channel(quantized, scales, axis=0):
    if axis == 0:
        return quantized.astype(np.float64) * scales.reshape(-1, 1)
    else:
        return quantized.astype(np.float64) * scales.reshape(1, -1)


def quantize_asymmetric(tensor, num_bits=8):
    qmin = 0
    qmax = 2 ** num_bits - 1
    t_min = np.min(tensor)
    t_max = np.max(tensor)
    if t_max == t_min:
        return np.zeros_like(tensor, dtype=np.int32), 1.0, 0
    scale = (t_max - t_min) / (qmax - qmin)
    zero_point = int(np.round(qmin - t_min / scale))
    zero_point = max(qmin, min(qmax, zero_point))
    quantized = np.clip(np.round(tensor / scale + zero_point), qmin, qmax).astype(np.int32)
    return quantized, float(scale), int(zero_point)


def dequantize_asymmetric(quantized, scale, zero_point):
    return (quantized.astype(np.float64) - zero_point) * scale

### 步骤 3：质量评估

评估量化过程造成的信息损失程度。主要指标包括原始张量与重建张量之间的均方误差（Mean Squared Error, MSE）、信噪比（Signal-to-Noise Ratio, SNR）以及余弦相似度（Cosine Similarity）。

def quantization_error(original, reconstructed):
    diff = original - reconstructed
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    max_error = float(np.max(np.abs(diff)))
    signal_power = float(np.mean(original ** 2))
    snr_db = 10 * np.log10(signal_power / max(mse, 1e-20))

    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    norm_orig = np.linalg.norm(orig_flat)
    norm_recon = np.linalg.norm(recon_flat)
    if norm_orig == 0 or norm_recon == 0:
        cosine_sim = 0.0
    else:
        cosine_sim = float(np.dot(orig_flat, recon_flat) / (norm_orig * norm_recon))

    return {"mse": mse, "rmse": rmse, "max_error": max_error,
            "snr_db": float(snr_db), "cosine_similarity": cosine_sim}


def compare_quantization_methods(tensor, num_bits=8):
    q_pt, s_pt = quantize_symmetric(tensor, num_bits)
    recon_pt = dequantize_symmetric(q_pt, s_pt)
    err_pt = quantization_error(tensor, recon_pt)

    q_pc, s_pc = quantize_per_channel(tensor, num_bits, axis=0)
    recon_pc = dequantize_per_channel(q_pc, s_pc, axis=0)
    err_pc = quantization_error(tensor, recon_pc)

    q_asym, s_asym, zp = quantize_asymmetric(tensor, num_bits)
    recon_asym = dequantize_asymmetric(q_asym, s_asym, zp)
    err_asym = quantization_error(tensor, recon_asym)

    print(f"\n  Quantization Comparison ({num_bits}-bit, tensor shape {tensor.shape}):")
    print(f"  {'Method':<20} {'MSE':>12} {'SNR (dB)':>10} {'Cosine Sim':>12} {'Max Error':>12}")
    print(f"  {'-'*68}")
    print(f"  {'Per-tensor sym':<20} {err_pt['mse']:>12.8f} {err_pt['snr_db']:>10.2f} {err_pt['cosine_similarity']:>12.8f} {err_pt['max_error']:>12.8f}")
    print(f"  {'Per-channel sym':<20} {err_pc['mse']:>12.8f} {err_pc['snr_db']:>10.2f} {err_pc['cosine_similarity']:>12.8f} {err_pc['max_error']:>12.8f}")
    print(f"  {'Asymmetric':<20} {err_asym['mse']:>12.8f} {err_asym['snr_db']:>10.2f} {err_asym['cosine_similarity']:>12.8f} {err_asym['max_error']:>12.8f}")

    return {"per_tensor": err_pt, "per_channel": err_pc, "asymmetric": err_asym}

### 步骤 4：位宽扫描

使用不同的位宽（2、3、4、8、16）对同一张量进行量化，并测量每个位宽级别下的质量。这能精确揭示质量断崖（quality cliff）出现的位置。

def bit_width_sweep(tensor):
    print(f"\n  Bit-Width Sweep (tensor shape {tensor.shape}):")
    print(f"  {'Bits':>6} {'Levels':>8} {'MSE':>14} {'SNR (dB)':>10} {'Cosine Sim':>12} {'Compression':>12}")
    print(f"  {'-'*64}")

    results = []
    for bits in [2, 3, 4, 8, 16]:
        q, s = quantize_per_channel(tensor, bits, axis=0)
        recon = dequantize_per_channel(q, s, axis=0)
        err = quantization_error(tensor, recon)
        levels = 2 ** bits
        compression = 32.0 / bits

        print(f"  {bits:>6} {levels:>8} {err['mse']:>14.8f} {err['snr_db']:>10.2f} {err['cosine_similarity']:>12.8f} {compression:>11.1f}x")
        results.append({"bits": bits, "levels": levels, "error": err, "compression": compression})

    return results

### 步骤 5：敏感度实验

模拟对 Transformer 模型的不同部分进行量化，并测量哪些组件最为敏感。该实验展示了敏感度层级关系：权重（weights）< 激活值（activations）< KV 缓存（KV cache）< 注意力机制（attention）。

def simulate_transformer_layer(input_data, weights, kv_scale=1.0):
    hidden = input_data @ weights["qkv"]
    seq_len = hidden.shape[1]
    d_model = weights["qkv"].shape[1] // 3
    q, k, v = hidden[:, :, :d_model], hidden[:, :, d_model:2*d_model], hidden[:, :, 2*d_model:]

    attn_scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(d_model) * kv_scale
    attn_max = np.max(attn_scores, axis=-1, keepdims=True)
    attn_exp = np.exp(attn_scores - attn_max)
    attn_weights = attn_exp / np.sum(attn_exp, axis=-1, keepdims=True)

    attn_output = attn_weights @ v
    output = attn_output @ weights["out"]
    return output, {"q": q, "k": k, "v": v, "attn_scores": attn_scores,
                    "attn_weights": attn_weights, "attn_output": attn_output}


def sensitivity_experiment(batch_size=2, seq_len=16, d_model=64, num_bits=8):
    np.random.seed(42)
    input_data = np.random.randn(batch_size, seq_len, d_model) * 0.1

    weights = {
        "qkv": np.random.randn(d_model, 3 * d_model) * (2.0 / d_model) ** 0.5,
        "out": np.random.randn(d_model, d_model) * (2.0 / d_model) ** 0.5,
    }

    baseline_output, baseline_internals = simulate_transformer_layer(input_data, weights)

    experiments = {}

    q_qkv, s_qkv = quantize_per_channel(weights["qkv"], num_bits, axis=0)
    q_out, s_out = quantize_per_channel(weights["out"], num_bits, axis=0)
    quantized_weights = {
        "qkv": dequantize_per_channel(q_qkv, s_qkv, axis=0),
        "out": dequantize_per_channel(q_out, s_out, axis=0),
    }
    weight_quant_output, _ = simulate_transformer_layer(input_data, quantized_weights)
    experiments["Weights only"] = quantization_error(baseline_output, weight_quant_output)

    _, fresh_internals = simulate_transformer_layer(input_data, weights)
    q_act, s_act = quantize_per_channel(
        fresh_internals["attn_output"].reshape(-1, d_model), num_bits, axis=0
    )
    quant_attn_out = dequantize_per_channel(q_act, s_act, axis=0).reshape(batch_size, seq_len, d_model)
    act_quant_output = quant_attn_out @ weights["out"]
    experiments["Activations only"] = quantization_error(baseline_output, act_quant_output)

    q_k, s_k = quantize_per_channel(fresh_internals["k"].reshape(-1, d_model), num_bits, axis=0)
    q_v, s_v = quantize_per_channel(fresh_internals["v"].reshape(-1, d_model), num_bits, axis=0)
    quant_k = dequantize_per_channel(q_k, s_k, axis=0).reshape(batch_size, seq_len, d_model)
    quant_v = dequantize_per_channel(q_v, s_v, axis=0).reshape(batch_size, seq_len, d_model)
    attn_scores_kv = (fresh_internals["q"] @ quant_k.transpose(0, 2, 1)) / np.sqrt(d_model)
    attn_max_kv = np.max(attn_scores_kv, axis=-1, keepdims=True)
    attn_exp_kv = np.exp(attn_scores_kv - attn_max_kv)
    attn_weights_kv = attn_exp_kv / np.sum(attn_exp_kv, axis=-1, keepdims=True)
    kv_quant_output = (attn_weights_kv @ quant_v) @ weights["out"]
    experiments["KV cache only"] = quantization_error(baseline_output, kv_quant_output)

    noise_scale = np.std(fresh_internals["attn_scores"]) * 0.05
    noisy_scores = fresh_internals["attn_scores"] + np.random.randn(*fresh_internals["attn_scores"].shape) * noise_scale
    noisy_max = np.max(noisy_scores, axis=-1, keepdims=True)
    noisy_exp = np.exp(noisy_scores - noisy_max)
    noisy_weights = noisy_exp / np.sum(noisy_exp, axis=-1, keepdims=True)
    attn_quant_output = (noisy_weights @ fresh_internals["v"]) @ weights["out"]
    experiments["Attention logits (5% noise)"] = quantization_error(baseline_output, attn_quant_output)

    print(f"\n  Sensitivity Experiment ({num_bits}-bit quantization):")
    print(f"  {'Component':<30} {'MSE':>14} {'SNR (dB)':>10} {'Cosine Sim':>12}")
    print(f"  {'-'*68}")
    for name, err in sorted(experiments.items(), key=lambda x: x[1]["mse"]):
        print(f"  {name:<30} {err['mse']:>14.8f} {err['snr_db']:>10.2f} {err['cosine_similarity']:>12.8f}")

    return experiments

### 步骤 6：GPTQ 模拟

GPTQ 每次量化一列，并利用海森矩阵（Hessian matrix）来决定如何分配舍入误差。这是一个简化版本，但保留了核心思想：使用校准数据（calibration data）来衡量权重的重要性，随后对重要性较低的权重进行更激进的量化。

def simulated_gptq(weight_matrix, calibration_inputs, num_bits=4):
    n_in, n_out = weight_matrix.shape
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    H = np.zeros((n_in, n_in))
    for x in calibration_inputs:
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        for row in range(x.shape[0]):
            xi = x[row].reshape(-1, 1)
            H += xi @ xi.T
    H /= len(calibration_inputs)
    H += np.eye(n_in) * 1e-4

    weight_importance = np.diag(H)

    quantized = np.zeros_like(weight_matrix, dtype=np.int32)
    scales = np.zeros(n_out)
    errors = np.zeros(n_out)

    W = weight_matrix.copy()

    for col in range(n_out):
        w_col = W[:, col]
        abs_max = np.max(np.abs(w_col))
        if abs_max == 0:
            scales[col] = 1.0
            continue
        scale = abs_max / qmax
        scales[col] = scale

        q_col = np.clip(np.round(w_col / scale), qmin, qmax).astype(np.int32)
        quantized[:, col] = q_col

        quant_error = w_col - q_col * scale
        errors[col] = np.sqrt(np.mean(quant_error ** 2))

        if col < n_out - 1:
            importance_weights = weight_importance / (np.max(weight_importance) + 1e-10)
            for next_col in range(col + 1, min(col + 4, n_out)):
                compensation = quant_error * importance_weights * 0.1
                W[:, next_col] += compensation

    return quantized, scales, {"column_errors": errors,
                               "mean_error": float(np.mean(errors)),
                               "max_error": float(np.max(errors))}


def dequantize_gptq(quantized, scales):
    result = np.zeros_like(quantized, dtype=np.float64)
    for col in range(quantized.shape[1]):
        result[:, col] = quantized[:, col] * scales[col]
    return result

### 步骤 7：AWQ 模拟

AWQ 会识别显著权重（salient weights，即与较大激活值相乘的权重），并在量化前通过缩放操作对其进行保护。

def simulated_awq(weight_matrix, calibration_inputs, num_bits=4, salient_fraction=0.01):
    n_in, n_out = weight_matrix.shape
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    activation_magnitudes = np.zeros(n_in)
    for x in calibration_inputs:
        if x.ndim == 1:
            activation_magnitudes += np.abs(x)
        else:
            activation_magnitudes += np.mean(np.abs(x), axis=0)
    activation_magnitudes /= len(calibration_inputs)

    n_salient = max(1, int(n_in * salient_fraction))
    salient_indices = np.argsort(activation_magnitudes)[-n_salient:]

    scale_factors = np.ones(n_in)
    for idx in salient_indices:
        col_max = np.max(np.abs(weight_matrix[idx, :]))
        if col_max > 0:
            scale_factors[idx] = min(4.0, 1.0 / (col_max + 1e-8) * np.mean(np.abs(weight_matrix)))

    scaled_weights = weight_matrix * scale_factors.reshape(-1, 1)

    quantized, scales = quantize_per_channel(scaled_weights, num_bits, axis=0)
    dequantized = dequantize_per_channel(quantized, scales, axis=0)

    result = dequantized / scale_factors.reshape(-1, 1)

    err = quantization_error(weight_matrix, result)

    return result, {"salient_indices": salient_indices,
                    "scale_factors": scale_factors[salient_indices],
                    "error": err,
                    "n_salient": n_salient}

### 步骤 8：完整流水线

将所有模块整合在一起。在相同的权重矩阵上，对比朴素量化（naive quantization）、逐通道量化、GPTQ 和 AWQ 的效果。

def full_quantization_comparison(d_in=256, d_out=512, num_bits=4, n_calibration=32):
    np.random.seed(42)

    weight = np.random.randn(d_in, d_out) * 0.02
    outlier_rows = np.random.choice(d_in, size=5, replace=False)
    weight[outlier_rows] *= 10

    calibration = [np.random.randn(8, d_in) * 0.1 for _ in range(n_calibration)]

    q_naive, s_naive = quantize_symmetric(weight, num_bits)
    recon_naive = dequantize_symmetric(q_naive, s_naive)
    err_naive = quantization_error(weight, recon_naive)

    q_pc, s_pc = quantize_per_channel(weight, num_bits, axis=0)
    recon_pc = dequantize_per_channel(q_pc, s_pc, axis=0)
    err_pc = quantization_error(weight, recon_pc)

    q_gptq, s_gptq, gptq_info = simulated_gptq(weight, calibration, num_bits)
    recon_gptq = dequantize_gptq(q_gptq, s_gptq)
    err_gptq = quantization_error(weight, recon_gptq)

    recon_awq, awq_info = simulated_awq(weight, calibration, num_bits)
    err_awq = awq_info["error"]

    print(f"\n  Full Quantization Comparison ({num_bits}-bit, {d_in}x{d_out} matrix)")
    print(f"  Matrix has {len(outlier_rows)} outlier rows (10x scale)")
    print()
    print(f"  {'Method':<20} {'MSE':>14} {'SNR (dB)':>10} {'Cosine Sim':>12}")
    print(f"  {'-'*58}")
    print(f"  {'Naive per-tensor':<20} {err_naive['mse']:>14.8f} {err_naive['snr_db']:>10.2f} {err_naive['cosine_similarity']:>12.8f}")
    print(f"  {'Per-channel':<20} {err_pc['mse']:>14.8f} {err_pc['snr_db']:>10.2f} {err_pc['cosine_similarity']:>12.8f}")
    print(f"  {'Simulated GPTQ':<20} {err_gptq['mse']:>14.8f} {err_gptq['snr_db']:>10.2f} {err_gptq['cosine_similarity']:>12.8f}")
    print(f"  {'Simulated AWQ':<20} {err_awq['mse']:>14.8f} {err_awq['snr_db']:>10.2f} {err_awq['cosine_similarity']:>12.8f}")

    test_input = np.random.randn(4, d_in) * 0.1
    baseline = test_input @ weight
    output_naive = test_input @ recon_naive
    output_pc = test_input @ recon_pc
    output_gptq = test_input @ recon_gptq
    output_awq = test_input @ recon_awq

    print(f"\n  End-to-End Output Error (matmul with test input):")
    print(f"  {'Method':<20} {'Output MSE':>14} {'Output Cosine':>14}")
    print(f"  {'-'*50}")
    for name, output in [("Naive", output_naive), ("Per-channel", output_pc),
                          ("GPTQ", output_gptq), ("AWQ", output_awq)]:
        out_err = quantization_error(baseline, output)
        print(f"  {name:<20} {out_err['mse']:>14.8f} {out_err['cosine_similarity']:>14.8f}")

    return {"naive": err_naive, "per_channel": err_pc, "gptq": err_gptq, "awq": err_awq}


def memory_calculator(num_params_billions, bits_per_param):
    bytes_per_param = bits_per_param / 8
    total_bytes = num_params_billions * 1e9 * bytes_per_param
    total_gb = total_bytes / (1024 ** 3)
    return total_gb


def print_memory_table():
    print("\n  Memory Requirements by Model and Precision:")
    print(f"  {'Model':<15} {'FP32':>8} {'FP16':>8} {'FP8':>8} {'INT8':>8} {'INT4':>8} {'INT2':>8}")
    print(f"  {'-'*64}")
    for name, params in [("7B", 7), ("13B", 13), ("34B", 34), ("70B", 70), ("405B", 405)]:
        fp32 = memory_calculator(params, 32)
        fp16 = memory_calculator(params, 16)
        fp8 = memory_calculator(params, 8)
        int8 = memory_calculator(params, 8)
        int4 = memory_calculator(params, 4)
        int2 = memory_calculator(params, 2)
        print(f"  {name:<15} {fp32:>7.1f}G {fp16:>7.1f}G {fp8:>7.1f}G {int8:>7.1f}G {int4:>7.1f}G {int2:>7.1f}G")


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 70)
    print("QUANTIZATION: MAKING MODELS FIT")
    print("=" * 70)

    print("\nSTEP 1: Number Format Comparison")
    print("-" * 50)
    for val in [0.1, 3.14159, -0.00073, 42.5, 0.0000012]:
        display_format_comparison(val)

    print("\n\nSTEP 2: Memory Requirements")
    print("-" * 50)
    print_memory_table()

    print("\n\nSTEP 3: Quantization Methods Comparison")
    print("-" * 50)
    weight_matrix = np.random.randn(128, 256) * 0.02
    weight_matrix[0] *= 15
    weight_matrix[42] *= 8
    compare_quantization_methods(weight_matrix, num_bits=8)
    compare_quantization_methods(weight_matrix, num_bits=4)

    print("\n\nSTEP 4: Bit-Width Sweep")
    print("-" * 50)
    sweep_tensor = np.random.randn(64, 128) * 0.05
    bit_width_sweep(sweep_tensor)

    print("\n\nSTEP 5: Sensitivity Experiment")
    print("-" * 50)
    print("\n  INT8:")
    sensitivity_experiment(num_bits=8)
    print("\n  INT4:")
    sensitivity_experiment(num_bits=4)

    print("\n\nSTEP 6: GPTQ vs AWQ vs Naive (INT4)")
    print("-" * 50)
    full_quantization_comparison(d_in=256, d_out=512, num_bits=4)

    print("\n\nSTEP 7: Distribution Analysis")
    print("-" * 50)
    np.random.seed(0)
    simulated_weights = np.random.randn(1000) * 0.02
    abs_vals = np.abs(simulated_weights)
    pct_in_range = np.mean(abs_vals < 0.1) * 100
    print(f"\n  Simulated weight distribution (1000 params, std=0.02):")
    print(f"  Weights in [-0.1, 0.1]: {pct_in_range:.1f}%")
    print(f"  Weights in [-0.05, 0.05]: {np.mean(abs_vals < 0.05) * 100:.1f}%")
    print(f"  Weights in [-0.01, 0.01]: {np.mean(abs_vals < 0.01) * 100:.1f}%")
    print(f"  Max absolute value: {np.max(abs_vals):.6f}")
    print(f"  Mean absolute value: {np.mean(abs_vals):.6f}")

    histogram = np.histogram(simulated_weights, bins=20)
    print(f"\n  Weight histogram:")
    max_count = max(histogram[0])
    for i in range(len(histogram[0])):
        bar_len = int(histogram[0][i] / max_count * 40)
        lo = histogram[1][i]
        hi = histogram[1][i + 1]
        print(f"  [{lo:>7.4f}, {hi:>7.4f}] {'#' * bar_len} ({histogram[0][i]})")

    print("\n\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


## 实际应用

### 使用 AutoGPTQ 进行量化 (Quantization)

# pip install auto-gptq transformers
# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
# from transformers import AutoTokenizer
#
# model_id = "meta-llama/Llama-3.1-8B"
# quantize_config = BaseQuantizeConfig(
#     bits=4,
#     group_size=128,
#     desc_act=False,
# )
#
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
#
# calibration = [tokenizer(t, return_tensors="pt") for t in calibration_texts[:128]]
# model.quantize(calibration)
# model.save_quantized("llama-8b-gptq-int4")

### 使用 AutoAWQ 进行量化

# pip install autoawq
# from awq import AutoAWQForCausalLM
# from transformers import AutoTokenizer
#
# model_id = "meta-llama/Llama-3.1-8B"
# model = AutoAWQForCausalLM.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
#
# model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4})
# model.save_quantized("llama-8b-awq-int4")

### 转换为 GGUF 格式

# pip install llama-cpp-python
# python convert_hf_to_gguf.py meta-llama/Llama-3.1-8B --outtype q4_k_m --outfile llama-8b-q4km.gguf
# llama-server -m llama-8b-q4km.gguf -c 4096 -ngl 99

### 使用 vLLM 进行模型服务化 (Serving)

# pip install vllm
# vllm serve model-awq --quantization awq --dtype half --max-model-len 8192

vLLM 原生支持 AWQ 和 GPTQ 模型。它会在矩阵乘法过程中自动处理反量化 (Dequantization)，并为键值缓存 (KV Cache) 采用分页注意力机制 (Paged Attention)。若要在 H100 上使用 FP8 精度 (FP8 Precision)，请添加 `--dtype float8_e4m3fn` 参数。

## 部署上线

本课程将生成 `outputs/skill-quantization.md` 文件，这是一个用于选择合适量化策略的决策框架。根据你的模型规模、目标硬件和质量要求，它会指导你选择正确的格式、方法及验证步骤。内容涵盖内存预算计算、各组件精度建议，以及针对 vLLM、llama.cpp 和 TensorRT-LLM 的部署方案。

## 练习

1. 实现分组量化（Group Quantization）。与每个通道（Channel）仅使用一个缩放因子（Scale Factor）不同，改为在通道内每 128 个权重为一组分配一个缩放因子。这正是 GPTQ 和 AWQ 实际采用的方案。在相同的权重矩阵（Weight Matrix）上，对比分组大小为 32、64、128 和 256 的效果。较小的分组能带来更好的量化质量，但会增加缩放因子的存储开销（Storage Overhead）。

2. 构建混合精度量化器（Mixed-Precision Quantizer）。将多层网络（Multi-Layer Network）的首层和末层量化为 INT8，中间层量化为 INT4。将端到端（End-to-End）输出质量与统一使用 INT4 和统一使用 INT8 的方案进行对比。测量相较于全 INT8 方案所节省的内存（Memory Savings）。

3. 为量化感知训练（Quantization-Aware Training）实现直通估计器（Straight-Through Estimator, STE）。在用于回归任务（Regression Task）的简单双层网络的前向传播（Forward Pass）中，插入伪量化/反量化（Fake Quantize/Dequantize）操作。对比正常训练（随后进行训练后量化 PTQ 至 INT4）的模型与从一开始就使用 QAT 训练的模型的最终损失（Final Loss）。

4. 构建受 LLM.int8() 启发的异常值感知量化器（Outlier-Aware Quantizer）。检测激活幅值（Activation Magnitude）超过均值 6 倍的通道。将这些通道保留为 FP16 精度，其余部分量化为 INT8。在步骤 5 的 Transformer 层（Transformer Layer）上，使用不同的异常值阈值（Outlier Thresholds）（3 倍、6 倍、10 倍）测量端到端质量。

5. 实现量化质量看板（Quantization Quality Dashboard）。给定一个权重矩阵，计算并展示以下内容：权重分布直方图（Weight Distribution Histogram）、量化误差分布（Quantization Error Distribution）、逐通道缩放因子（Per-Channel Scale Factors）、量化效果最差的通道（即重建误差 Reconstruction Error 最高的通道），以及在 100 个随机输入下原始输出与量化输出之间的余弦相似度（Cosine Similarity）。识别出哪些通道应保留在更高精度（Higher Precision）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 半精度浮点数 (FP16) | “半精度” | 包含5位指数和10位尾数的16位浮点数，最大值为65,504，是标准的推理格式 |
| Brain浮点数 (BF16) | “Brain浮点” | 包含8位指数（与FP32范围相同）和7位尾数的16位浮点数，由Google设计用于模型训练 |
| 8位浮点数 (FP8) | “8位浮点” | 包含两种变体：E4M3（用于推理，精度更高）和E5M2（用于训练，动态范围更大），为H100原生支持 |
| 8位整数 (INT8) | “8位整数” | 从-128到127均匀分布的256个数值，需要缩放因子将其从浮点数映射过来 |
| 4位整数 (INT4) | “4位整数” | 总共16个量化级别，需要借助复杂算法（如GPTQ、AWQ）来保持模型质量 |
| 逐通道量化 (Per-channel quantization) | “每行一个缩放比例” | 为每个输出通道使用独立的缩放因子，而非对整个张量使用单一因子，可大幅降低量化误差 |
| 基于海森矩阵的量化方法 (GPTQ) | “海森矩阵法” | 一种训练后量化技术，利用二阶信息逐层最小化输出误差 |
| 激活感知量化 (AWQ) | “激活感知” | 在量化前对显著权重（即与较大激活值相乘的权重）进行缩放保护，以防止精度损失 |
| llama.cpp模型格式 (GGUF) | “llama.cpp格式” | 包含混合精度层的独立模型文件格式，针对CPU和Apple Silicon推理进行了优化 |
| 训练后量化 (PTQ) | “训练后量化” | 无需重新训练即可将已训练模型的权重转换为低精度格式，速度快但在极高压缩率下效果受限 |
| 量化感知训练 (QAT) | “训练中量化” | 在前向传播中插入伪量化操作，使模型学习适应舍入误差，在INT4/INT2等极低精度下表现更佳 |
| 校准数据 (Calibration data) | “那128个样本” | 输入模型的一小部分数据集，用于计算激活值统计信息，从而确定缩放因子 |
| 缩放因子 (Scale factor) | “乘数” | 用于在浮点数范围与整数范围之间进行转换：`float_val = int_val * scale` |
| 困惑度差值 (Perplexity delta) | “性能下降多少” | 原始模型与量化模型之间的困惑度差值，< 0.5表示效果极佳，> 2.0则说明存在问题 |

## 延伸阅读

- [Frantar 等人，2022 -- "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"](https://arxiv.org/abs/2210.17323) -- 该论文利用海森矩阵（Hessian）引导的权重舍入技术，使 INT4 量化（INT4 Quantization）在大语言模型（Large Language Model, LLM）中具备了实际可行性
- [Lin 等人，2023 -- "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"](https://arxiv.org/abs/2306.00978) -- 通过在量化前进行缩放来保护显著权重（Salient Weights），其性能与 GPTQ 相当甚至更优
- [Dettmers 等人，2022 -- "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"](https://arxiv.org/abs/2208.07339) -- 采用混合精度 INT8（Mixed-Precision INT8）方案，将异常值特征（Outlier Features）保留在 FP16 格式中，从而在无损质量的前提下实现 INT8 推理（INT8 Inference）
- [Xiao 等人，2023 -- "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"](https://arxiv.org/abs/2211.10438) -- 将量化难度从激活值（Activations）转移至权重（Weights），以支持 W8A8 部署
- [Micikevicius 等人，2022 -- "FP8 Formats for Deep Learning"](https://arxiv.org/abs/2209.05433) -- 由 NVIDIA、ARM 和 Intel 联合发表的论文，定义了 E4M3 与 E5M2 格式，这两种格式现已成为 H100 的原生支持格式