# 视觉语言模型（Vision-Language Models）—— ViT-MLP-LLM 模式

> 视觉编码器（Vision Encoder）将图像转换为词元（Tokens）。多层感知机（MLP）投影器将这些词元映射到大语言模型（Large Language Model, LLM）的嵌入空间（Embedding Space）中。语言模型（Language Model）则负责后续处理。这一模式——ViT-MLP-LLM——正是 2026 年所有生产级视觉语言模型（Vision-Language Model, VLM）的通用架构。

**类型：** 学习与应用
**编程语言：** Python
**前置知识：** 第 4 阶段第 14 课（ViT）、第 4 阶段第 18 课（CLIP）、第 7 阶段第 2 课（自注意力机制（Self-Attention））
**预计耗时：** 约 75 分钟

## 学习目标

- 阐述 ViT-MLP-LLM 架构，并解释三个组件各自的作用
- 从参数量、上下文长度和基准测试性能三个维度，对比 Qwen3-VL、InternVL3.5、LLaVA-Next 和 GLM-4.6V
- 解释 DeepStack 机制：为何多层级 ViT 特征比单一末层特征能更好地强化视觉-语言对齐（Vision-Language Alignment）
- 使用跨模态错误率（Cross-Modal Error Rate, CMER）在生产环境中评估 VLM 的幻觉（Hallucination）现象，并据此采取优化措施

## 问题背景

CLIP（第 4 阶段第 18 课）为图像和文本提供了一个共享的嵌入空间，这足以支持零样本分类（Zero-Shot Classification）和检索（Retrieval）任务。但它无法回答“这张图里有几辆红色汽车？”这类问题，因为 CLIP 并不具备文本生成能力——它仅能计算相似度得分。

视觉语言模型（VLMs）——如 Qwen3-VL、InternVL3.5、LLaVA-Next 和 GLM-4.6V——将 CLIP 家族的图像编码器与完整的大语言模型相结合。模型接收图像与问题作为输入，并生成相应的回答。到 2026 年，开源 VLM 在多模态基准测试（MMMU、MMBench、DocVQA、ChartQA、MathVista、OSWorld）上的表现已媲美甚至超越 GPT-5 和 Gemini-2.5-Pro。

由视觉变换器（Vision Transformer, ViT）、投影器（Projector）和大语言模型（LLM）组成的三件套已成为行业标准。不同模型之间的差异仅在于选用的 ViT、投影器、LLM 的具体版本，以及训练数据和对齐策略（Alignment Recipe）。一旦掌握了这一模式，替换其中任意组件都将变得像搭积木一样简单直接。

## 核心概念

### ViT-MLP-LLM 架构

flowchart LR
    IMG["Image<br/>(H x W x 3)"] --> ViT["Vision encoder<br/>(ViT, CLIP-L,<br/>SigLIP, DINOv3)"]
    ViT --> FEATS["Image tokens<br/>(N, d_vit)"]
    FEATS --> PROJ["Projector<br/>(2-4 layer MLP<br/>or Q-former)"]
    PROJ --> VTOK["Image tokens<br/>in LLM space<br/>(N, d_llm)"]
    TXT["Text prompt"] --> TOK["LLM tokenizer"]
    TOK --> TTOK["Text tokens<br/>(M, d_llm)"]
    VTOK --> CONCAT["Interleave<br/>or concat"]
    TTOK --> CONCAT
    CONCAT --> LLM["Decoder LLM<br/>(Qwen3, LLaMA, etc.)"]
    LLM --> OUT["Text answer"]

    style ViT fill:#dbeafe,stroke:#2563eb
    style PROJ fill:#fef3c7,stroke:#d97706
    style LLM fill:#dcfce7,stroke:#16a34a

1. **视觉编码器（Vision encoder）**——预训练的 ViT（CLIP-L/14、SigLIP、DINOv3 或其微调变体）。用于生成图像块令牌（patch tokens）。
2. **投影器（Projector）**——一个小型模块（2-4 层 MLP 或 Q-former），负责将视觉令牌映射到大语言模型（LLM）的嵌入维度。大部分微调工作都在此进行。
3. **大语言模型（LLM）**——仅解码器架构的语言模型（Qwen3、Llama、Mistral、GLM、InternLM）。按顺序读取视觉与文本令牌，并生成文本。

理论上这三个组件均可训练。但在实践中，视觉编码器和 LLM 通常保持冻结状态，仅训练投影器——以极低的成本处理数十亿参数的信号。

### DeepStack

传统投影方法仅使用 ViT 的最后一层。DeepStack（Qwen3-VL）则从 ViT 的多个深度采样特征并进行堆叠。深层网络承载高层语义信息，浅层网络则保留细粒度的空间与纹理信息。将两者同时输入 LLM，有效弥合了“图像包含什么”（语义）与“具体位置在哪”（空间定位）之间的差距。

### 三个训练阶段

现代视觉语言模型（VLM）通常分阶段训练：

1. **对齐（Alignment）**——冻结 ViT 和 LLM。仅在图像-描述对（image-caption pairs）上训练投影器。旨在教会投影器将视觉空间映射到语言空间。
2. **预训练（Pre-training）**——解冻所有参数。在大规模图文交错数据（5 亿对以上）上进行训练。用于构建模型的视觉知识库。
3. **指令微调（Instruction tuning）**——在精心整理的（图像、问题、答案）三元组上进行微调。用于教授对话行为与任务格式。这一步将“具备视觉感知能力的语言模型”转化为真正可用的助手。

大多数低秩自适应（LoRA）微调都针对第三阶段，并使用小规模标注数据集。

### 模型家族对比（2026 年初）

| 模型 | 参数量 | 视觉编码器 | 大语言模型（LLM） | 上下文窗口 | 优势 |
|-------|--------|----------------|-----|---------|-----------|
| Qwen3-VL-235B-A22B (MoE) | 235B（22B 激活） | 定制 ViT + DeepStack | Qwen3 | 256K | 通用业界最优（SOTA），图形用户界面智能体（GUI agent） |
| Qwen3-VL-30B-A3B (MoE) | 30B（3B 激活） | 定制 ViT + DeepStack | Qwen3 | 256K | 更轻量级的混合专家模型（MoE）替代方案 |
| Qwen3-VL-8B (dense) | 8B | 定制 ViT | Qwen3 | 128K | 生产环境默认稠密模型 |
| InternVL3.5-38B | 38B | InternViT-6B | Qwen3 + GPT-OSS | 128K | MMBench / MMVet 表现强劲 |
| InternVL3.5-241B-A28B | 241B（28B 激活） | InternViT-6B | Qwen3 | 128K | 与 GPT-4o 竞争力相当 |
| LLaVA-Next 72B | 72B | SigLIP | Llama-3 | 32K | 开源，易于微调 |
| GLM-4.6V | ~70B | 定制 | GLM | 64K | 开源，光学字符识别（OCR）能力强 |
| MiniCPM-V-2.6 | 8B | SigLIP | MiniCPM | 32K | 适合边缘设备部署 |

### 视觉智能体（Visual agents）

Qwen3-VL-235B 在 OSWorld 上达到了全球顶尖水平——这是一个针对操作图形用户界面（GUI，涵盖桌面、移动端和网页）的**视觉智能体（visual agents）**基准测试。该模型能够查看截图、理解 UI 界面，并输出操作指令（点击、输入、滚动）。结合外部工具，它能够完成常见桌面任务的闭环。这也是 2026 年大多数“AI PC”演示背后的核心技术。

### 智能体能力与旋转位置编码（RoPE）变体

视觉语言模型需要知道视频帧出现的**具体时间**。Qwen3-VL 从 T-RoPE（时间旋转位置编码，temporal rotary position embeddings）演进到了**基于文本的时间对齐**——将显式的时间戳文本令牌与视频帧交错排列。模型会看到“`<timestamp 00:32>` 帧，提示词”这样的输入，从而能够进行时间关系推理。

### 对齐问题

在爬取的数据集中，约有 12% 的图文对包含未完全基于图像内容的描述。在此类数据上训练的 VLM 会潜移默化地学会产生幻觉（hallucinate）——凭空捏造物体、误读数字或虚构关系。在生产环境中，这是最主要的故障模式。

Skywork.ai 引入了**跨模态错误率（Cross-Modal Error Rate, CMER）**来追踪该问题：

CMER = fraction of outputs where the text confidence is high but the image-text similarity (via a CLIP-family checker) is low

CMER 偏高意味着模型在自信地输出与图像内容不符的信息。在其实际部署中，通过监控 CMER 并将其作为生产环境的关键绩效指标（KPI），幻觉率降低了约 35%。关键技巧不在于“修复模型”，而在于“将高 CMER 的输出路由至人工审核”。

### 使用 LoRA / QLoRA 进行微调

对 70B 规模的 VLM 进行全量微调对大多数团队而言难以企及。在注意力层与投影器层应用低秩自适应（LoRA，秩 16-64），或使用 4 比特基础权重的量化低秩自适应（QLoRA），单张 A100 / H100 显卡即可运行。成本估算：5,000 至 50,000 条样本，计算成本 100 至 5,000 美元，训练时长 2 至 10 小时。

### 空间推理能力依然薄弱

当前 VLM 在空间推理基准测试（上下、左右、计数、距离）中的得分仅为 50%-60%。如果您的应用场景依赖于“哪个物体在另一个物体上方”这类判断，请务必进行严格验证——通用 VLM 的表现仍低于人类水平。针对纯空间任务，优于 VLM 的替代方案包括：专用的关键点/姿态估计器、深度估计模型，或结合边界框几何后处理的目标检测模型。

## 构建

### 步骤 1：投影层（Projector）

这是训练频率最高的组件。通常采用 2 到 4 层的多层感知机（Multilayer Perceptron, MLP），并搭配 GELU 激活函数。

import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, vit_dim=768, llm_dim=4096, hidden=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vit_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, llm_dim),
        )

    def forward(self, x):
        return self.net(x)

输入是一个形状为 `(N_patches, d_vit)` 的标记（Token）张量。输出形状为 `(N_patches, d_llm)`。大语言模型（Large Language Model, LLM）会将输出的每一行视为普通的文本标记。

### 步骤 2：端到端组装 ViT-MLP-LLM

这是一个极简视觉语言模型（Vision-Language Model, VLM）前向传播（Forward Pass）的代码骨架。实际开发中通常会使用 `transformers` 库，此处仅展示概念性结构。

class MinimalVLM(nn.Module):
    def __init__(self, vit, projector, llm, image_token_id):
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.llm = llm
        self.image_token_id = image_token_id  # placeholder token in text prompt

    def forward(self, image, input_ids, attention_mask):
        # 1. vision features
        vision_tokens = self.vit(image)                     # (B, N_patches, d_vit)
        vision_embeds = self.projector(vision_tokens)       # (B, N_patches, d_llm)

        # 2. text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, M, d_llm)

        # 3. replace image placeholder tokens with vision embeds
        merged = self._merge(text_embeds, vision_embeds, input_ids)

        # 4. run LLM
        return self.llm(inputs_embeds=merged, attention_mask=attention_mask)

    def _merge(self, text_embeds, vision_embeds, input_ids):
        out = text_embeds.clone()
        expected = vision_embeds.size(1)
        for b in range(input_ids.size(0)):
            positions = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(positions) != expected:
                raise ValueError(
                    f"batch item {b} has {len(positions)} image tokens but vision_embeds has {expected} patches."
                    " Every sample in the batch must be pre-padded to the same number of image placeholder tokens.")
            out[b, positions] = vision_embeds[b]
        return out

文本中的 `<image>` 占位符标记（Placeholder Token）会被替换为真实的图像嵌入（Image Embeddings）——这与 LLaVA、Qwen-VL 和 InternVL 采用的模式一致。

### 步骤 3：计算跨模态错误率（Cross-Modal Error Rate, CMER）

一种轻量级的运行时检查机制。

import torch.nn.functional as F


def cross_modal_error_rate(image_emb, text_emb, text_confidence, sim_threshold=0.25, conf_threshold=0.8):
    """
    image_emb, text_emb: embeddings of image and generated text (normalised internally)
    text_confidence:     mean per-token probability in [0, 1]
    Returns:             fraction of high-confidence outputs with low image-text alignment
    """
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    sim = (image_emb * text_emb).sum(dim=-1)        # cosine similarity
    high_conf_low_sim = (text_confidence > conf_threshold) & (sim < sim_threshold)
    return high_conf_low_sim.float().mean().item()

应将 CMER 视为生产环境的关键绩效指标（Key Performance Indicator, KPI）。建议按接口端点、提示词（Prompt）类型和客户维度进行监控。CMER 的上升表明模型在特定输入分布上开始出现幻觉（Hallucination）。

### 步骤 4：简易视觉语言模型分类器（Toy VLM Classifier，可运行）

用于演示投影层的训练流程。输入模拟的“ViT 特征”，由一个微型 LLM 风格的分类头预测类别。

class ToyVLM(nn.Module):
    def __init__(self, vit_dim=32, llm_dim=64, num_classes=5):
        super().__init__()
        self.projector = Projector(vit_dim, llm_dim, hidden=64)
        self.head = nn.Linear(llm_dim, num_classes)

    def forward(self, vision_tokens):
        projected = self.projector(vision_tokens)
        pooled = projected.mean(dim=1)
        return self.head(pooled)

只需不到 200 个训练步（Steps），即可在合成的（特征，类别）数据对上完成拟合——这足以验证投影层架构的有效性。

## 使用方式

2026年，生产团队使用视觉语言模型（Vision Language Models, VLMs）的三种方式：

- **托管API（Hosted API）** — OpenAI Vision、Anthropic Claude Vision、Google Gemini Vision。零基础设施成本，但存在供应商风险。
- **开源自托管（Open-source self-host）** — 通过 `transformers` 和 `vllm` 部署 Qwen3-VL 或 InternVL3.5。完全自主控制，但前期投入较高。
- **领域微调（Fine-tune on domain）** — 加载 Qwen2.5-VL-7B 或 LLaVA-1.6-7B，使用 5k-50k 条自定义样本进行 LoRA 微调，并通过 `vllm` 或 `TGI` 提供服务。

from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image

model_id = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": Image.open("plot.png")},
        {"type": "text", "text": "What does this chart show?"},
    ],
}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
generated = model.generate(**inputs, max_new_tokens=256)
answer = processor.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

`apply_chat_template` 隐藏了 `<image>` 占位符的标记化（tokenization）过程；模型会在内部处理特征融合。

## 交付部署

本章节将生成以下文件：

- `outputs/prompt-vlm-selector.md` — 根据准确率、延迟、上下文长度和预算，自动选择 Qwen3-VL / InternVL3.5 / LLaVA-Next 或 API。
- `outputs/skill-cmer-monitor.md` — 生成用于监控生产环境 VLM 端点的代码，包含跨模态错误率（Cross-Modal Error Rate, CMER）、各端点独立仪表盘及告警阈值。

## 练习

1. **（简单）** 选取五张图片，使用任意开源 VLM 运行三个提示词（“这是什么？”、“数一数物体数量”、“描述场景”）。人工将每个答案评分为正确 / 部分正确 / 幻觉。计算初步的类跨模态错误率（CMER-like rate）。
2. **（中等）** 使用带标题的 500 张目标领域图像，对 Qwen2.5-VL-3B 或 LLaVA-1.6-7B 进行 LoRA（秩为 16）微调。对比零样本（zero-shot）与微调后在 MMBench 风格（MMBench-style）测试上的准确率。
3. **（困难）** 将 VLM 的图像编码器替换为 DINOv3，而非默认的 SigLIP/CLIP。仅重新训练投影层（projector）（冻结大语言模型（Large Language Model, LLM）与 DINOv3）。评估密集预测任务（dense-prediction tasks）（如计数、空间推理）的性能是否提升。

## 关键术语

| 术语 | 业界俗称 | 实际含义 |
|------|----------------|----------------------|
| ViT-MLP-LLM | “VLM 模式” | 视觉编码器 (Vision Encoder) + 投影层 (Projector) + 语言模型 (Language Model)；2026 年所有视觉语言模型 (VLM) 的通用架构 |
| 投影层 (Projector) | “桥梁” | 2 至 4 层的多层感知机 (MLP)（或 Q-former），负责将视觉词元 (Vision Tokens) 映射至大语言模型 (LLM) 的嵌入空间 (Embedding Space) |
| DeepStack | “Qwen3-VL 的特征技巧” | 堆叠多层视觉变换器 (ViT) 特征，而非仅提取最后一层特征 |
| 图像词元 (Image Token) | “`<image>` 占位符” | 文本流中的特殊词元，在输入时会被替换为投影后的视觉嵌入向量 (Vision Embeddings) |
| CMER | “幻觉评估指标” | 跨模态错误率 (Cross-Modal Error Rate)；当文本生成置信度高但图文相似度低时，该指标值偏高 |
| 视觉智能体 (Visual Agent) | “会点击的 VLM” | 通过工具调用 (Tool Calls) 操作图形用户界面 (GUI)（涵盖 OSWorld、移动端及网页端）的视觉语言模型 |
| Q-former | “固定数量词元桥梁” | 采用 BLIP-2 架构的投影层，用于生成固定数量的视觉查询词元 (Visual Query Tokens) |
| 对齐 / 预训练 / 指令微调 (Alignment / Pre-training / Instruction Tuning) | “三个阶段” | 标准的视觉语言模型训练流水线 |

## 延伸阅读

- [Qwen3-VL 技术报告 (arXiv 2511.21631)](https://arxiv.org/abs/2511.21631)
- [InternVL3.5：推进开源多模态模型发展 (arXiv 2508.18265)](https://arxiv.org/html/2508.18265v1)
- [LLaVA-Next 系列](https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/)
- [BentoML：2026 年最佳开源 VLM 指南](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
- [MMMU：多学科多模态理解基准测试](https://mmmu-benchmark.github.io/)
- [制造业中的 VLM 应用（Robotics Tomorrow，2026 年 3 月）](https://www.roboticstomorrow.com/story/2026/03/when-machines-learn-to-see-like-experts-the-rise-of-vision-language-models-in-manufacturing/26335/)