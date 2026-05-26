# SAM 3 与开放词汇分割（Open-Vocabulary Segmentation）

> 向模型输入文本提示词（text prompt）和图像，即可获取所有匹配对象的掩码（mask）。SAM 3 将这一过程简化为单次前向传播（forward pass）。

**类型：** 使用 + 构建
**语言：** Python
**前置知识：** 第 4 阶段第 07 课（U-Net）、第 4 阶段第 08 课（Mask R-CNN）、第 4 阶段第 18 课（CLIP）
**时长：** 约 60 分钟

## 学习目标

- 区分 SAM（仅支持视觉提示词（visual prompts））、Grounded SAM / SAM 2（检测器（detector）+ SAM）以及 SAM 3（通过可提示概念分割（Promptable Concept Segmentation）原生支持文本提示词）
- 阐述 SAM 3 架构：共享主干网络（shared backbone）+ 图像检测器 + 基于记忆的视频跟踪器（memory-based video tracker）+ 存在性预测头（presence head）+ 解耦的检测器-跟踪器设计（decoupled detector-tracker design）
- 使用 Hugging Face `transformers` 库中的 SAM 3 集成接口，实现基于文本提示词的检测、分割与视频跟踪
- 根据延迟（latency）、概念复杂度及部署目标，在 SAM 3、Grounded SAM 2、YOLO-World 和 SAM-MI 之间进行选择

## 问题背景

2023 年发布的 SAM 是一款仅支持视觉提示词的模型：用户点击一个点或绘制一个边界框，模型便会返回对应的掩码。若要实现“找出这张照片里所有的橙子”，则需要先使用检测器（如 Grounding DINO）生成边界框，再交由 SAM 对每个框进行分割。Grounded SAM 将这一流程整合为流水线（pipeline），但它本质上是两个冻结模型（frozen models）的级联（cascade），不可避免地会导致误差累积（error accumulation）。

SAM 3（Meta 于 2025 年 11 月发布，ICLR 2026 收录）消除了级联步骤。它接受简短的名词短语或图像示例（image exemplar）作为提示词，并在单次前向传播中返回所有匹配的掩码与实例 ID（instance IDs）。这就是**可提示概念分割（Promptable Concept Segmentation, PCS）**。结合 2026 年 3 月推出的对象多路复用（Object Multiplex）更新（SAM 3.1），该模型能够高效地在视频中跟踪同一概念的多个实例。

本课将重点探讨这一架构所代表的结构性转变。二维分割（2D segmentation）、目标检测与视觉定位（text-image grounding）已融合至单一模型中。在生产环境中，核心问题不再是“我该串联哪条流水线”，而是“哪款可提示模型能够端到端（end-to-end）地处理我的业务场景”。

## 核心概念

### 三代模型

flowchart LR
    subgraph SAM1["SAM (2023)"]
        A1["Image + point/box prompt"] --> A2["ViT encoder"] --> A3["Mask decoder"]
        A3 --> A4["Mask for that prompt"]
    end
    subgraph GSAM2["Grounded SAM 2 (2024)"]
        B1["Text"] --> B2["Grounding DINO"] --> B3["Boxes"] --> B4["SAM 2"] --> B5["Masks + tracking"]
        B6["Image"] --> B2
        B6 --> B4
    end
    subgraph SAM3["SAM 3 (2025)"]
        C1["Text OR image exemplar"] --> C2["Shared backbone"]
        C3["Image"] --> C2
        C2 --> C4["Image detector + memory tracker<br/>+ presence head"]
        C4 --> C5["All matching masks<br/>+ instance IDs"]
    end

    style SAM1 fill:#e5e7eb,stroke:#6b7280
    style GSAM2 fill:#fef3c7,stroke:#d97706
    style SAM3 fill:#dcfce7,stroke:#16a34a

### 可提示概念分割（Promptable Concept Segmentation）

“概念提示（concept prompt）”是一个简短的名词短语（如 `"yellow school bus"`、`"striped red umbrella"`、`"hand holding a mug"`）或一张图像示例。模型会返回图像中所有匹配该概念的实例的分割掩码（segmentation masks），并为每个匹配项分配一个唯一的实例 ID（instance ID）。

这与经典的视觉提示 SAM（visual-prompt SAM）在三个方面有所不同：

1. 无需针对每个实例单独提示——单个文本提示即可返回所有匹配结果。
2. 开放词汇（open-vocabulary）——概念可以是任何能用自然语言描述的事物。
3. 一次性返回多个实例，而非每个提示仅输出一个掩码。

### 核心架构组件

- **共享主干网络（shared backbone）**——由单个视觉 Transformer（ViT）处理图像。检测头（detector head）和基于记忆的跟踪器（memory-based tracker）均从中读取特征。
- **存在性预测头（presence head）**——用于预测图像中是否包含该概念。将“是否存在”与“位于何处”解耦，从而降低对不存在概念的误报率。
- **解耦检测与跟踪模块（decoupled detector-tracker）**——图像级检测和视频级跟踪使用独立的预测头，避免相互干扰。
- **记忆库（memory bank）**——跨帧存储每个实例的特征以用于视频跟踪（采用与 SAM 2 相同的机制）。

### 大规模训练

SAM 3 在 **400 万个独立概念**上进行了训练，这些概念由一个数据引擎生成，该引擎通过 AI 与人工审核相结合的方式进行迭代标注与修正。全新的 **SA-CO 基准测试集（SA-CO benchmark）**包含 27 万个独立概念，规模是以往基准的 50 倍。在 SA-CO 上，SAM 3 达到了人类表现水平的 75%-80%，并在图像与视频可提示概念分割（PCS）任务上，性能达到现有系统的两倍。

### SAM 3.1 对象多路复用（Object Multiplex）

2026 年 3 月更新：**对象多路复用（Object Multiplex）**引入了一种共享内存机制，用于同时联合跟踪同一概念的多个实例。过去，跟踪 N 个实例需要 N 个独立的记忆库。多路复用技术将其整合为一个共享内存，并配合针对每个实例的查询。结果是：在不牺牲精度的前提下，大幅提升了多目标跟踪的速度。

### 2026 年 Grounded SAM 的适用场景

- 需要替换为特定的开放词汇检测器（如 DINO-X、Florence-2）时。
- SAM 3 的许可证（在 Hugging Face 上需申请访问权限）成为使用障碍时。
- 需要比 SAM 3 提供的接口更精细地控制检测器阈值时。
- 针对检测器组件进行研究或消融实验（ablation study）时。

模块化流水线（modular pipelines）依然有其用武之地。但对于大多数生产环境而言，SAM 3 是更简洁的解决方案。

### YOLO-World 与 SAM 3 对比

- **YOLO-World**——仅提供开放词汇检测功能（无掩码）。支持实时推理。适用于需要高帧率（fps）输出边界框的场景。
- **SAM 3**——提供完整的分割与跟踪功能。速度较慢，但输出信息更丰富。

生产环境选型建议：对于仅需快速检测的流水线（如机器人导航、实时仪表盘），选用 YOLO-World；对于任何需要掩码或跟踪功能的场景，选用 SAM 3。

### SAM-MI 的效率优化

SAM-MI（2025-2026）旨在解决 SAM 的解码器瓶颈（decoder bottleneck）问题。核心思路如下：

- **稀疏点提示（sparse point prompting）**——使用少量精心挑选的点替代密集提示，使解码器调用次数减少 96%。
- **浅层掩码聚合（shallow mask aggregation）**——将粗糙的掩码预测结果融合为更清晰的单一掩码。
- **解耦掩码注入（decoupled mask injection）**——解码器直接接收预计算的掩码特征，无需重新运行计算。

结果：在开放词汇基准测试中，速度较 Grounded-SAM 提升约 1.6 倍。

### 三款模型的输出格式

三者均返回相同的通用结构（边界框 + 标签 + 置信度分数 + 掩码 + ID），这一设计非常实用——下游流水线无需根据实际运行的模型进行分支处理。

## 构建

### 步骤 1：提示词构建 (Prompt Construction)

构建一个辅助函数，将用户输入的句子转换为 SAM 3 概念提示词 (Concept Prompts) 列表。这是“用户输入内容”与“模型实际处理内容”之间的边界。

def split_concepts(sentence):
    """
    Heuristic splitter for multi-concept prompts.
    Returns list of short noun phrases.
    """
    for sep in [",", ";", "and", "or", "&"]:
        if sep in sentence:
            parts = [p.strip() for p in sentence.replace("and ", ",").split(",")]
            return [p for p in parts if p]
    return [sentence.strip()]

print(split_concepts("cats, dogs and balloons"))

SAM 3 每次前向传播 (Forward Pass) 仅接受一个概念；对于多概念查询，需使用循环或批处理 (Batching) 方式。

### 步骤 2：后处理辅助函数

将 SAM 3 的原始输出转换为干净的检测列表，以符合我们“第四阶段第 16 课”流水线 (Pipeline) 的接口契约。

from dataclasses import dataclass
from typing import List

@dataclass
class ConceptDetection:
    concept: str
    instance_id: int
    box: tuple          # (x1, y1, x2, y2)
    score: float
    mask_rle: str       # run-length encoded


def rle_encode(binary_mask):
    flat = binary_mask.flatten().astype("uint8")
    runs = []
    prev, count = flat[0], 0
    for v in flat:
        if v == prev:
            count += 1
        else:
            runs.append((int(prev), count))
            prev, count = v, 1
    runs.append((int(prev), count))
    return ";".join(f"{v}x{c}" for v, c in runs)

游程编码 (Run-Length Encoding, RLE) 即使在处理大量高分辨率掩码 (Masks) 时，也能保持响应负载 (Payload) 较小。该格式在 SAM 2、SAM 3 和 Grounded SAM 2 中通用。

### 步骤 3：统一的开放词汇分割 (Open-Vocabulary Segmentation) 接口

使用单一方法封装你拥有的任意后端（如 SAM 3、Grounded SAM 2、YOLO-World + SAM 2）。当后端更换时，下游代码无需修改。

from abc import ABC, abstractmethod
import numpy as np

class OpenVocabSeg(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray, concept: str) -> List[ConceptDetection]:
        ...


class StubOpenVocabSeg(OpenVocabSeg):
    """
    Deterministic stub used for pipeline testing when real models are not loaded.
    """
    def detect(self, image, concept):
        h, w = image.shape[:2]
        return [
            ConceptDetection(
                concept=concept,
                instance_id=0,
                box=(w * 0.2, h * 0.3, w * 0.5, h * 0.8),
                score=0.89,
                mask_rle="0x100;1x50;0x200",
            ),
            ConceptDetection(
                concept=concept,
                instance_id=1,
                box=(w * 0.55, h * 0.25, w * 0.85, h * 0.75),
                score=0.74,
                mask_rle="0x80;1x40;0x220",
            ),
        ]

实际的 `SAM3OpenVocabSeg` 子类将封装 `transformers.Sam3Model` 和 `Sam3Processor`。

### 步骤 4：Hugging Face SAM 3 使用示例（参考）

对于实际模型，`transformers` 库的集成方式如下：

from transformers import Sam3Processor, Sam3Model
import torch

processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3").eval()

inputs = processor(images=pil_image, return_tensors="pt")
inputs = processor.set_text_prompt(inputs, "yellow school bus")

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_masks(
    outputs.masks, inputs.original_sizes, inputs.reshaped_input_sizes
)
boxes = outputs.boxes
scores = outputs.scores

单次提示词输入，单次调用即可返回所有匹配结果。

### 步骤 5：评估 Grounded SAM 2 原生提供的优势

进行一次客观的基准测试 (Benchmark)：在实际流水线中，用 SAM 3 替换 Grounded SAM 2 会发生什么？

- 延迟 (Latency)：SAM 3 节省了一次前向传播（无需独立的检测器），但模型本身更庞大；通常总体延迟持平或略有提升。
- 准确率 (Accuracy)：SAM 3 在罕见或组合概念（如“条纹红雨伞”）上表现显著更优。在常见的单字概念上两者表现相近。
- 灵活性 (Flexibility)：Grounded SAM 2 允许你更换检测器（如 DINO-X、Florence-2、Grounding DINO 1.5）；而 SAM 3 是单体架构 (Monolithic)。

结论：SAM 3 将成为 2026 年开放词汇分割的默认选择。当你需要检测器灵活性或不同的许可证条款 (License Terms) 时，Grounded SAM 2 仍是更合适的选择。

## 使用方式

生产环境部署模式：

- **实时标注** — SAM 3 结合 CVAT 的“标签即文本提示”功能。标注员选择标签名称后，SAM 3 会自动预标注所有匹配的实例。随后进行人工复核与修正。
- **视频分析** — 使用 SAM 3.1 的对象多路复用（Object Multiplex）功能进行多目标跟踪；将视频帧输入基于记忆的跟踪器中。
- **机器人控制** — 利用 SAM 3 实现开放词汇（open-vocabulary）操作（例如“拿起红色的杯子”）；作为规划原语（planning primitive）运行。
- **医学影像** — 在医学概念上经过微调（fine-tuning）的 SAM 3 模型；需在 Hugging Face (HF) 上申请访问权限。

Ultralytics 在其 Python 包中封装了 SAM 3：

from ultralytics import SAM

model = SAM("sam3.pt")
results = model(image_path, prompts="yellow school bus")

其接口与 YOLO 和 SAM 2 保持一致。

## 交付内容

本课时将生成以下文件：

- `outputs/prompt-open-vocab-stack-picker.md` — 一个提示词（prompt），可根据延迟、概念复杂度和许可证要求，在 SAM 3 / Grounded SAM 2 / YOLO-World / SAM-MI 之间进行技术栈选型。
- `outputs/skill-concept-prompt-designer.md` — 一项技能（skill），用于将用户的自然语言表述转换为格式规范的 SAM 3 概念提示词（包含拆分、消歧和回退机制）。

## 练习

1. **（简单）** 使用自选的概念提示词在 10 张图像上运行 SAM 3。在相同图像上与 SAM 2 + Grounding DINO 1.5 进行对比。报告各模型未能识别出的概念。
2. **（中等）** 基于 SAM 3 构建一个“点击包含/点击排除”的用户界面（UI）：输入文本提示词返回候选实例；用户通过点击决定哪些实例计入正样本（positive）。最终将概念集以 JSON 格式输出。
3. **（困难）** 使用自定义概念集（例如 5 类电子元器件，每类 20 张标注图像）对 SAM 3 进行微调（fine-tuning）。在相同测试集上与零样本（zero-shot）SAM 3 进行对比；测量掩码交并比（mask IoU）的提升幅度。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 开放词汇分割（Open-vocabulary segmentation） | “通过文本分割” | 为自然语言描述的对象生成掩码，而非依赖固定的标签集 |
| PCS（Promptable Concept Segmentation） | “可提示概念分割” | SAM 3 的核心任务——给定名词短语或图像示例，分割所有匹配的实例 |
| 概念提示词（Concept prompt） | “文本输入” | 简短的名词短语或图像示例；并非完整句子 |
| 存在性检测头（Presence head） | “它在这里吗？” | SAM 3 的模块，用于在定位之前判断图像中是否存在该概念 |
| SA-CO | “SAM 3 基准测试” | 包含 27 万个概念的开放词汇分割基准数据集；规模是以往同类基准的 50 倍 |
| 对象多路复用（Object Multiplex） | “SAM 3.1 更新” | 基于共享内存的多目标跟踪；实现对大量实例的快速联合跟踪 |
| Grounded SAM 2 | “模块化流水线” | 检测器与 SAM 2 的级联架构；在需要灵活替换检测器时仍具实用价值 |
| SAM-MI | “高效 SAM 变体” | 采用掩码注入（Mask Injection）技术，速度较 Grounded-SAM 提升 1.6 倍 |

## 延伸阅读

- [SAM 3：基于概念的分割一切模型 (arXiv 2511.16719)](https://arxiv.org/abs/2511.16719)
- [SAM 3.1：对象复用 (Meta AI，2026年3月)](https://ai.meta.com/blog/segment-anything-model-3/)
- [Hugging Face 上的 SAM 3 模型页面](https://huggingface.co/facebook/sam3)
- [Grounded SAM 2 教程 (PyImageSearch)](https://pyimagesearch.com/2026/01/19/grounded-sam-2-from-open-set-detection-to-segmentation-and-tracking/)
- [Ultralytics SAM 3 文档](https://docs.ultralytics.com/models/sam-3/)
- [SAM3-I：指令感知型 SAM (arXiv 2512.04585)](https://arxiv.org/abs/2512.04585)