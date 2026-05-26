# 使用 LoRA (Low-Rank Adaptation) 与 QLoRA (Quantized Low-Rank Adaptation) 进行微调 (Fine-Tuning)

> 对 7B 模型进行全量微调 (Full Fine-Tuning) 需要 56GB 的显存 (VRAM)。你没有这么多，大多数公司也没有。LoRA 允许你仅通过训练不到 1% 的参数，在 6GB 显存内完成同一模型的微调。这并非妥协——它在大多数任务上的效果与全量微调相当。整个开源微调生态都建立在这一技巧之上。

**类型：** 构建 (Build)
**语言：** Python
**前置要求：** 第 10 阶段，第 06 课（指令微调 (Instruction Tuning) / 监督微调 (Supervised Fine-Tuning, SFT)）
**耗时：** 约 75 分钟
**相关：** 第 10 阶段从零开始讲解 SFT 与直接偏好优化 (Direct Preference Optimization, DPO) 的循环流程。本课将这些内容接入 2026 年的参数高效微调 (Parameter-Efficient Fine-Tuning, PEFT) 工具链（PEFT、TRL、Unsloth、Axolotl、LLaMA-Factory）。

## 学习目标

- 通过将低秩适配器矩阵（A 和 B）注入预训练模型的注意力层 (Attention Layers) 来实现 LoRA
- 计算 LoRA 相较于全量微调的参数节省量：在维度为 d_model 的情况下，秩 (Rank) 为 r 时仅需训练 2*r*d 个参数，而非 d^2 个
- 使用 QLoRA（4 比特量化 (Quantization) 基座模型 + LoRA 适配器）对模型进行微调，使其适配消费级 GPU 显存
- 将 LoRA 权重合并回基座模型以便部署，并对比使用与不使用适配器时的推理 (Inference) 速度

## 核心问题

你有一个基座模型 (Base Model)：Llama 3 8B。你希望它能以你公司的口吻回复客户支持工单。监督微调 (SFT) 是解决方案，但它存在成本问题。

全量微调会更新模型中的每一个参数。Llama 3 8B 拥有 80 亿个参数。在半精度浮点数 (fp16) 格式下，每个参数占用 2 字节。仅加载权重就需要 16GB。在训练过程中，你还需要存储梯度 (Gradients)、Adam 优化器状态（动量与方差共需 32GB）以及激活值 (Activations)。总计：单个 8B 模型大约需要 56GB 的显存。

单张 80GB 的 A100 显卡勉强能装下。在云服务商处，两张 A100 的租用成本约为每小时 3-4 美元。在 5 万条样本上训练 3 个轮次 (Epochs) 需要 6-10 小时。这意味着每次实验的成本为 30-40 美元。为了调准超参数 (Hyperparameters) 运行 10 次实验，你在部署任何内容之前就已经花费了 400 美元。

如果将规模扩展到 Llama 3 70B，数字会变得极其夸张。仅权重就需要 140GB。你必须使用计算集群。每次实验的成本超过 100 美元。

此外还有一个更深层的问题。全量微调会修改模型中的每一个权重。如果你使用客户支持数据进行微调，可能会削弱模型的通用能力。这被称为灾难性遗忘 (Catastrophic Forgetting)。模型在你特定任务上的表现变好了，但在其他所有任务上的表现却变差了。

你需要一种方法，能够训练更少的参数、占用更少的内存，并且不会破坏模型已有的知识。

## 核心概念

### LoRA：低秩自适应（Low-Rank Adaptation）

微软的 Edward Hu 及其同事于 2021 年 6 月发表了 LoRA 论文。该论文的核心洞察是：微调（Fine-tuning）过程中的权重更新具有较低的内禀秩（Intrinsic Rank）。你无需更新 4096x4096 权重矩阵中的全部 1670 万个参数。更新中的有效信息完全可以通过一个秩为 16 或 32 的矩阵来捕获。

其数学原理如下。一个标准的线性层（Linear Layer）计算过程为：

y = Wx

其中 W 是一个 d_out x d_in 的矩阵。对于 4096x4096 的注意力投影（Attention Projection），这相当于 16,777,216 个参数。

LoRA 会冻结 W，并添加一个低秩分解（Low-Rank Decomposition）：

y = Wx + BAx

其中 B 的维度为 (d_out x r)，A 的维度为 (r x d_in)。秩 r 远小于 d，通常取值为 8、16 或 32。

对于 4096x4096 的层，当 r=16 时：
- 原始参数量：4096 x 4096 = 16,777,216
- LoRA 参数量：(4096 x 16) + (16 x 4096) = 65,536 + 65,536 = 131,072
- 缩减比例：131,072 / 16,777,216 = 0.78%

你仅需训练 0.78% 的参数，即可获得 95% 到 100% 的模型质量。

graph LR
    X["Input x"] --> W["Frozen W (d x d)"]
    X --> A["A (r x d)"]
    A --> B["B (d x r)"]
    W --> Plus["+ (merge)"]
    B --> Plus
    Plus --> Y["Output y"]

    style W fill:#1a1a2e,stroke:#e94560,color:#fff
    style A fill:#0f3460,stroke:#16213e,color:#fff
    style B fill:#0f3460,stroke:#16213e,color:#fff

A 使用随机高斯分布初始化，B 初始化为零。这意味着 LoRA 的初始贡献为零——模型将从其原始行为开始训练，并逐步学习自适应调整。

### 缩放因子：Alpha

LoRA 引入了一个缩放因子 alpha，用于控制低秩更新对输出的影响程度：

y = Wx + (alpha / r) * BAx

当 alpha = r 时，缩放比例为 1 倍。当 alpha = 2r（常见默认值）时，缩放比例为 2 倍。该超参数（Hyperparameter）独立于基础学习率，专门控制 LoRA 路径的学习率。

实践建议：
- `alpha = 2 * rank` 是社区常用的惯例（原论文在大多数实验中使用了 `alpha = rank`）
- `alpha = rank` 提供 1 倍缩放，较为保守但更稳定
- 更高的 alpha 意味着每步更新幅度更大，可能加速收敛，也可能引发不稳定

### LoRA 的应用位置

Transformer 架构包含大量线性层。你无需为所有层都添加 LoRA。原论文测试了不同的组合：

| 目标层 | 可训练参数量（7B 模型） | 质量表现 |
|--------------|----------------------|---------|
| 仅 q_proj | 4.7M | 良好 |
| q_proj + v_proj | 9.4M | 更好 |
| q_proj + k_proj + v_proj + o_proj | 18.9M | 注意力机制最佳 |
| 所有线性层（注意力 + MLP） | 37.7M | 收益递减，参数量翻倍 |

对于大多数任务而言，最佳平衡点是 `q_proj + v_proj`。这针对的是自注意力（Self-Attention）中的查询（Query）和值（Value）投影，它们决定了模型关注什么以及提取何种信息。对于代码生成等复杂任务，添加多层感知机（MLP）层会有所帮助；但对于简单任务，这会使参数量翻倍，且收益递减。

### 秩（Rank）的选择

秩 r 控制着自适应调整的表达能力：

| 秩 | 每层可训练参数量 | 适用场景 |
|------|---------------------------|----------|
| 4 | 32,768 | 简单分类、情感分析 |
| 8 | 65,536 | 单领域问答、文本摘要 |
| 16 | 131,072 | 多领域任务、指令遵循 |
| 32 | 262,144 | 复杂推理、代码生成 |
| 64 | 524,288 | 对大多数任务收益递减 |
| 128 | 1,048,576 | 极少有使用必要 |

Hu 等人的研究表明，对于简单任务，r=4 已能捕获大部分自适应调整所需的信息。在实践中，r=8 和 r=16 是最常见的选择。超过 r=64 通常难以进一步提升质量，反而会开始丧失 LoRA 的显存优势。

### QLoRA：4 比特量化（4-Bit Quantization）+ LoRA

华盛顿大学的 Tim Dettmers 及其同事于 2023 年 5 月发表了 QLoRA 论文。其核心思路是：将冻结的基础模型量化至 4 比特精度，然后在其上方附加 fp16 精度的 LoRA 适配器（Adapter）。

这极大地改变了显存占用情况：

| 方法 | 权重显存（7B） | 训练显存（7B） | 所需 GPU |
|--------|-------------------|---------------------|-------------|
| 全量微调（fp16） | 14GB | ~56GB | 1x A100 80GB |
| LoRA（fp16 基础） | 14GB | ~18GB | 1x A100 40GB |
| QLoRA（4 比特基础） | 3.5GB | ~6GB | 1x RTX 3090 24GB |

QLoRA 做出了三项技术贡献：

**NF4（Normal Float 4-bit，正态浮点 4 比特）**：一种专为神经网络权重设计的新数据类型。神经网络权重大致遵循正态分布。NF4 将其 16 个量化级别设置在标准正态分布的分位数上。从信息论角度来看，这对正态分布数据是最优的。相比均匀 4 比特量化（INT4）或标准 Float4，它丢失的信息更少。

**双重量化（Double Quantization）**：量化常数本身也会占用显存。每 64 个权重块需要一个 fp32 缩放因子（4 字节）。对于 7B 模型，这会额外增加 0.4GB 显存。双重量化将这些常数进一步量化为 fp8，将开销降至 0.1GB。虽然单次节省不多，但累积效果显著。

**分页优化器（Paged Optimizers）**：在训练过程中，优化器状态（如 Adam 的动量和方差）在处理长序列时可能超出 GPU 显存。分页优化器利用 NVIDIA 的统一内存（Unified Memory）技术，在 GPU 显存耗尽时自动将优化器状态交换至 CPU 内存，并在需要时换回。这以牺牲部分吞吐量为代价，有效防止了显存溢出（OOM）崩溃。

### 质量影响问题

减少参数量或量化基础模型是否会损害质量？多篇论文的结果如下：

| 方法 | MMLU（5-shot） | MT-Bench | HumanEval |
|--------|--------------|----------|-----------|
| 全量微调（Llama 2 7B） | 48.3 | 6.72 | 14.6 |
| LoRA r=16 | 47.9 | 6.68 | 14.0 |
| QLoRA r=16（NF4） | 47.5 | 6.61 | 13.4 |
| QLoRA r=64（NF4） | 48.1 | 6.70 | 14.2 |

在大多数基准测试中，r=16 的 LoRA 与全量微调的性能差距在 1% 以内。r=16 的 QLoRA 仅再损失零点几个百分点。而 r=64 的 QLoRA 在显存占用减少 90% 的情况下，性能基本与全量微调持平。

### 实际成本

在 50,000 条样本上微调 Llama 3 8B（3 个 Epoch）：

| 方法 | GPU | 耗时 | 成本 |
|--------|-----|------|------|
| 全量微调 | 2x A100 80GB | 8 小时 | ~$32 |
| LoRA r=16 | 1x A100 40GB | 4 小时 | ~$8 |
| QLoRA r=16 | 1x RTX 4090 24GB | 6 小时 | ~$5 |
| QLoRA r=16（Unsloth） | 1x RTX 4090 24GB | 2.5 小时 | ~$2 |
| QLoRA r=16 | 1x T4 16GB | 12 小时 | ~$4 |

在单张消费级 GPU 上运行 QLoRA 的成本甚至不到一顿午餐钱。这正是 2023 年开源权重微调社区爆发式增长的原因，也是为何下文列出的所有训练框架在 2026 年都默认集成 QLoRA 的原因。

### 2026 年的 PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）技术栈

| 框架 | 简介 | 适用场景 |
|-----------|-----------|-----------|
| **Hugging Face PEFT** | 标准的 LoRA/QLoRA/DoRA/IA³ 库 | 你需要底层控制权，且训练循环已基于 `transformers.Trainer` |
| **TRL** | Hugging Face 的基于反馈的强化学习训练器（SFT、DPO、GRPO、PPO、ORPO） | 你在 SFT 后需要 DPO/GRPO；构建于 PEFT 之上 |
| **Unsloth** | 使用 Triton 内核重写的前向/反向传播 | 你希望在不损失精度的前提下获得 2-5 倍加速并节省一半显存；适用于 Llama/Mistral/Qwen 系列 |
| **Axolotl** | 基于 PEFT + TRL + DeepSpeed + Unsloth 的 YAML 配置封装 | 你需要可复现、支持版本控制的训练流程 |
| **LLaMA-Factory** | 基于 PEFT + TRL 的 GUI/CLI/API | 你希望零代码微调；支持 100+ 模型家族 |
| **torchtune** | 原生 PyTorch 配方，无 `transformers` 依赖 | 你希望依赖最小化，且团队已标准化使用 PyTorch |

经验法则：科研用途或一次性实验 → PEFT。可复现的生产流水线 → 启用 Unsloth 内核的 Axolotl。快速原型验证 → LLaMA-Factory。

### 合并适配器（Merging Adapters）

训练完成后，你将得到两部分：冻结的基础模型和一个小型 LoRA 适配器（通常为 10-100MB）。你可以选择：

1. **保持分离**：加载基础模型，再在其上加载适配器。针对不同任务切换适配器。这是如何从单一基础模型服务多个微调变体的标准做法。
2. **永久合并**：计算 `W' = W + (alpha/r) * BA` 并将结果保存为一个新的完整模型。合并后的模型大小与原始模型相同。无推理开销，也无需管理适配器。

若需服务多项任务（如客服适配器、代码适配器、翻译适配器），请保持分离。若需部署单一专用模型，则进行合并。

用于组合多个适配器的高级合并技术：

- **TIES-Merging**（Yadav 等人，2023）：修剪小幅度参数，解决符号冲突，然后进行合并。可减少适配器之间的干扰。
- **DARE**（Yu 等人，2023）：在合并前随机丢弃部分适配器参数，并对剩余参数进行重新缩放。在能力融合方面效果出奇地好。
- **任务算术（Task Arithmetic）**：直接对适配器权重进行加减运算。将“代码”适配器与“数学”适配器相加，通常能生成同时擅长这两项任务的模型。

### 何时不应进行微调

微调是第三选择，而非首选。

**首选：提示词工程（Prompt Engineering）**。编写更优的系统提示词。添加少样本（Few-shot）示例。使用思维链（Chain-of-Thought）。这零成本且只需几分钟。如果提示词能达到 80% 的效果，你很可能无需微调。

**次选：RAG（Retrieval-Augmented Generation，检索增强生成）**。如果模型需要了解你的特定数据（文档、知识库、产品目录），检索比将数据烘焙进权重中更便宜且更易维护。请参阅第 06 课。

**第三：微调**。当需要模型采用提示词无法实现的特定风格、格式或推理模式时使用。当你需要一致的结构化输出时。当你需要将大模型蒸馏（Distill）到小模型时。当延迟至关重要，且你无法承担少样本提示带来的额外 Token 开销时。

graph TD
    Start["Need better model behavior?"] --> PE["Try prompt engineering"]
    PE -->|"Works"| Done["Ship it"]
    PE -->|"Not enough"| RAG["Need external knowledge?"]
    RAG -->|"Yes"| RAGBuild["Build RAG pipeline"]
    RAG -->|"No, need style/format change"| FT["Fine-tune with LoRA/QLoRA"]
    RAGBuild -->|"Works"| Done
    RAGBuild -->|"Also need style change"| FT
    FT --> Done

    style Start fill:#1a1a2e,stroke:#e94560,color:#fff
    style Done fill:#0f3460,stroke:#16213e,color:#fff


## 开始构建

我们将使用纯 PyTorch 从零开始实现低秩自适应（LoRA）。不依赖任何第三方库，也没有黑魔法。你将亲手构建 LoRA 层，将其注入模型，进行训练，最后将权重合并回原模型。

### 步骤 1：构建 LoRA 层

import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.randn(in_features, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling

矩阵 A 使用缩放后的随机值进行初始化，矩阵 B 初始化为零。由于乘积 BA 的初始值为零，模型在训练开始时会保持其原始行为。

### 步骤 2：封装 LoRA 的线性层（Linear Layer）

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)

原始线性层的参数将被冻结。只有 LoRA 的参数（A 和 B）是可训练的。

### 步骤 3：将 LoRA 注入模型

def inject_lora(model, target_modules, rank=8, alpha=16):
    for param in model.parameters():
        param.requires_grad = False

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)
                lora_layers[name] = lora_linear
    return lora_layers

首先，冻结模型中的所有参数。然后遍历模型树，找到与目标名称匹配的线性层，并将其替换为封装了 LoRA 的版本。此时，LoRA 的 A 和 B 矩阵将成为整个模型中唯一可训练的参数。

### 步骤 4：统计参数量

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }

### 步骤 5：将权重合并回原模型

def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            with torch.no_grad():
                merged = (
                    module.lora.A @ module.lora.B
                ) * module.lora.scaling
                module.linear.weight.data += merged.T
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, module.linear)

合并完成后，LoRA 层将被移除。模型的大小与原始模型完全一致，而微调带来的适配效果已直接融入权重中。推理阶段不会产生任何额外开销。

### 步骤 6：模拟 QLoRA 量化（Quantization）

def quantize_to_nf4(tensor, block_size=64):
    blocks = tensor.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
    scales = torch.clamp(scales, min=1e-8)
    quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
    return quantized, scales

def dequantize_from_nf4(quantized, scales, original_shape):
    dequantized = quantized.float() * scales
    return dequantized.reshape(original_shape)

该函数通过将权重映射到大小为 64 的数据块中的 16 个离散级别，来模拟 4 比特量化。在生产环境中，QLoRA 会使用 `bitsandbytes` 库在 GPU 上实现真正的 NF4 量化。

### 步骤 7：训练循环

def train_lora(model, data, epochs=5, lr=1e-3, batch_size=4):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(data["inputs"]))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = data["inputs"][batch_idx]
            y = data["targets"][batch_idx]

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

    return losses

### 步骤 8：完整演示

def demo():
    torch.manual_seed(42)
    d_model = 256
    n_classes = 10

    model = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )

    n_samples = 500
    x = torch.randn(n_samples, d_model)
    y = torch.randint(0, n_classes, (n_samples,))
    y_onehot = torch.zeros(n_samples, n_classes).scatter_(1, y.unsqueeze(1), 1.0)

    data = {"inputs": x, "targets": y_onehot}

    params_before = count_parameters(model)

    lora_layers = inject_lora(
        model, target_modules=["0", "2"], rank=8, alpha=16
    )

    params_after = count_parameters(model)

    losses = train_lora(model, data, epochs=20, lr=1e-3)

    merge_lora_weights(model)
    params_merged = count_parameters(model)

    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_merged": params_merged,
        "losses": losses,
    }

该演示脚本会创建一个小型模型，向其中两个层注入 LoRA，执行训练，最后将权重合并。在 LoRA 训练期间，可训练参数量会从 100% 骤降至约 1%；合并完成后，模型将恢复为原始架构。

## 使用方法

借助 Hugging Face 生态系统，在真实模型上应用低秩自适应（LoRA）仅需约 20 行代码：

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

对于量化低秩自适应（QLoRA），只需添加 bitsandbytes 量化配置：

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

model = get_peft_model(model, lora_config)

就这么简单。训练循环（training loop）与数据流水线（data pipeline）均保持不变。基座模型（base model）现在以 4-bit 精度存储，低秩自适应适配器（LoRA adapters）以半精度浮点数（fp16）进行训练，整个方案仅需 6GB 显存即可运行。

若使用 Hugging Face Trainer 进行训练：

from transformers import TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./lora-adapter")

保存的适配器（adapter）大小仅为 10-100MB。基座模型保持原样不受影响。你可以直接在 Hugging Face Hub 上分享适配器，而无需重新分发完整的模型。

## 交付上线

本章节将生成以下文件：
- `outputs/prompt-lora-advisor.md` -- 一个提示词（prompt），用于帮助你针对特定任务确定低秩自适应的秩（rank）、目标模块（target modules）及超参数（hyperparameters）
- `outputs/skill-fine-tuning-guide.md` -- 一项技能指南，用于向智能体（agents）传授何时以及如何微调（fine-tune）的决策树

## 练习

1. **秩消融实验（Rank ablation study）。** 分别使用秩（rank）为 2、4、8、16、32 和 64 运行演示程序。绘制最终损失（loss）与秩的关系图。找出边际收益递减点，即秩翻倍不再使损失减半的临界点。对于基于 256 维特征的简单分类任务，该临界点通常在 r=8-16 左右。

2. **目标模块对比（Target module comparison）。** 修改 inject_lora 使其仅针对第 "0" 层、仅针对第 "2" 层、仅针对第 "4" 层，以及同时针对这三层。对每个变体训练 20 个轮次（epoch）。比较它们的收敛速度与最终损失。这模拟了实际应用中决定仅针对 q_proj、仅针对 v_proj 还是针对所有线性层（linear layers）的决策过程。

3. **量化误差分析（Quantization error analysis）。** 提取训练好的模型在执行 quantize_to_nf4 / dequantize_from_nf4 操作前后的权重矩阵。计算均方误差（mean squared error）、最大绝对误差（max absolute error），以及原始权重与重构权重之间的相关性。尝试将 block_size 设置为 32、64、128 和 256 进行实验。

4. **多适配器服务（Multi-adapter serving）。** 在数据的不同子集（偶数索引样本与奇数索引样本）上分别训练两个 LoRA 适配器（LoRA adapters）。保存这两个适配器。仅加载一次基座模型（base model），然后切换适配器，并验证它们在相同输入下是否产生不同的输出。这正是生产环境中如何基于单一基座模型服务多个微调模型（fine-tuned models）的典型做法。

5. **合并与未合并推理对比（Merge vs. unmerged inference）。** 在相同的 100 个输入上，比较执行 merge_lora_weights 操作前后 LoRA 模型的输出。验证两者输出是否一致（允许 1e-5 的浮点容差（floating-point tolerance））。随后对两种情况进行推理速度基准测试——合并后的模型应略快一些，因为它只需执行一次矩阵乘法（matrix multiply），而非两次。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 低秩自适应 (LoRA) | “高效微调” | 冻结基础权重，训练两个小型矩阵 A 和 B，其乘积用于近似完整的权重更新 |
| 量化低秩自适应 (QLoRA) | “在笔记本上微调” | 以 4-bit NF4 格式加载基础模型，并在其之上使用 fp16 精度训练 LoRA 适配器，从而实现在 6GB VRAM 下对 7B 参数模型进行微调 |
| 秩 (Rank, r) | “模型能学多少” | 矩阵 A 和 B 的内部维度；用于控制模型表达能力与参数量之间的平衡 |
| 缩放系数 (Alpha) | “LoRA 学习率” | 应用于 LoRA 输出的缩放因子；通过 alpha/r 的比例来调节自适应部分对最终输出的贡献程度 |
| 正态浮点 4 位 (NF4) | “4 位量化” | 一种 4 位数据类型，其量化级别基于正态分布的分位数设定，非常适合神经网络权重 |
| 适配器 (Adapter) | “训练过的小部分” | 将 LoRA 的 A 和 B 矩阵保存为独立文件（10-100MB），可加载到任意基础模型副本之上 |
| 目标模块 (Target modules) | “对哪些层应用 LoRA” | 注入 LoRA 适配器的特定线性层（q_proj, v_proj 等） |
| 权重合并 (Merging) | “固化到模型中” | 计算 W + (alpha/r) * BA 并替换原始权重，从而在推理阶段消除适配器带来的额外开销 |
| 分页优化器 (Paged optimizers) | “训练时不爆显存” | 当 GPU 显存耗尽时，将优化器状态（Adam 动量、方差）卸载至 CPU |
| 灾难性遗忘 (Catastrophic forgetting) | “微调把其他能力都搞坏了” | 当更新全部权重时，导致模型丧失先前已习得的能力 |

## 延伸阅读

- Hu 等人，《LoRA：大语言模型的低秩自适应（Low-Rank Adaptation）》（2021）—— 提出低秩分解（Low-Rank Decomposition）方法的原始论文，在 GPT-3 175B 上进行了验证，秩（Rank）最低可设为 4。
- Dettmers 等人，《QLoRA：量化语言模型的高效微调》（2023）—— 引入了 NF4、双重量化（Double Quantization）与分页优化器（Paged Optimizers），使得在单张 48GB 显存的 GPU 上微调 65B 模型成为可能。
- PEFT 库文档 (huggingface.co/docs/peft) —— Hugging Face 生态系统中用于 LoRA、QLoRA 及其他参数高效（Parameter-Efficient）方法的标准库。
- Yadav 等人，《TIES-Merging：解决模型合并时的干扰问题》（2023）—— 在不降低模型质量的前提下，合并多个 LoRA 适配器（Adapter）的技术。
- [Rafailov 等人，《直接偏好优化（Direct Preference Optimization）：你的语言模型本质上是一个奖励模型》（NeurIPS 2023）](https://arxiv.org/abs/2305.18290) —— DPO 的推导过程；位于监督微调（Supervised Fine-Tuning）之后的偏好调优阶段，无需额外训练奖励模型（Reward Model）。
- [TRL 文档](https://huggingface.co/docs/trl/) —— `SFTTrainer`、`DPOTrainer`、`KTOTrainer` 的官方参考指南，以及与 PEFT/bitsandbytes/Unsloth 的集成接口（Integration Surface）说明。
- [Unsloth 文档](https://docs.unsloth.ai/) —— 采用融合内核（Fused Kernels）使微调吞吐量翻倍并减半内存占用；作为 TRL 底层的性能加速层。
- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) —— 基于 YAML 配置的多 GPU SFT/DPO/QLoRA 训练器；以配置即代码（Config-as-Code）替代手写脚本的解决方案。