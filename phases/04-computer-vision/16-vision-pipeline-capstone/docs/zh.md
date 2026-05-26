# 构建完整的视觉流水线（Vision Pipeline） —— 综合实战（Capstone）

> 生产级视觉系统是由模型和规则通过数据契约（Data Contract）串联而成的链条。本阶段已涵盖所有组件，而本综合实战将把它们端到端地连接起来。

**类型：** 构建
**语言：** Python
**前置条件：** 第 4 阶段 第 01-15 课
**预计耗时：** 约 120 分钟

## 学习目标

- 设计一个生产级视觉流水线，能够检测目标、对其进行分类，并输出结构化的 JSON 数据——同时妥善处理所有异常路径
- 将目标检测器（Detector，如 Mask R-CNN 或 YOLO）、分类器（Classifier，如 ConvNeXt-Tiny）以及数据契约（基于 Pydantic）集成到单一服务中
- 对端到端流水线进行性能基准测试（Benchmark），并定位首个性能瓶颈（通常是预处理阶段，其次是检测器）
- 交付一个精简的 FastAPI 服务，该服务接收图像上传请求，运行流水线，并返回带有分类结果的目标检测数据

## 问题背景

单一的视觉模型固然有用，但视觉产品往往是多个模型的组合链条。例如，零售货架盘点系统由目标检测器、商品分类器和价格光学字符识别（OCR）流水线组成；自动驾驶系统融合了 2D 检测器、3D 检测器、图像分割器（Segmenter）、目标跟踪器（Tracker）以及路径规划器（Planner）；医疗初筛系统则包含分割器、区域分类器以及医生交互界面。

将这些链条正确连接，正是区分机器学习（ML）原型与成熟产品的关键所在。模型之间的每个接口都可能引入新的缺陷。每一次坐标变换、每一次归一化（Normalization）、每一次掩码（Mask）缩放，都可能导致静默失败（Silent Failure）。流水线的整体强度取决于其最薄弱的接口。

本综合实战将搭建一个最小可行流水线（Minimum Viable Pipeline）：检测 + 分类 + 结构化输出 + 服务层。第 4 阶段的其他所有内容均可嵌入此骨架中：例如将 Mask R-CNN 替换为 YOLOv8，添加 OCR 头（Head），增加分割分支，或接入跟踪器。该架构保持稳定，而各个组件均可插拔替换。

## 核心概念

### 流水线 (Pipeline)

flowchart LR
    REQ["HTTP request<br/>+ image bytes"] --> LOAD["Decode<br/>+ preprocess"]
    LOAD --> DET["Detector<br/>(YOLO / Mask R-CNN)"]
    DET --> CROP["Crop + resize<br/>each detection"]
    CROP --> CLS["Classifier<br/>(ConvNeXt-Tiny)"]
    CLS --> AGG["Aggregate<br/>detections + classes"]
    AGG --> SCHEMA["Pydantic<br/>validation"]
    SCHEMA --> RESP["JSON response"]

    REQ -.->|error| RESP

    style DET fill:#fef3c7,stroke:#d97706
    style CLS fill:#dbeafe,stroke:#2563eb
    style SCHEMA fill:#dcfce7,stroke:#16a34a

共包含七个阶段。其中两个模型阶段计算开销较大；而其余五个阶段则是缺陷的高发区。

### 基于 Pydantic 的数据契约 (Data Contracts)

每个模型边界都会转换为类型化对象。这能将静默失败 (Silent Failures) 转化为显式报错。

Detection(
    box: tuple[float, float, float, float],   # (x1, y1, x2, y2), absolute pixels
    score: float,                              # [0, 1]
    class_id: int,                             # from detector's label map
    mask: Optional[list[list[int]]],           # RLE-encoded if present
)

PipelineResult(
    image_id: str,
    detections: list[Detection],
    classifications: list[Classification],
    inference_ms: float,
)

当检测器返回的边界框格式为 `(cx, cy, w, h)` 而非 `(x1, y1, x2, y2)` 时，Pydantic 会在边界处触发验证失败，让你立即发现问题，而不是去调试下游裁剪操作静默返回空区域的隐蔽错误。

### 延迟分布 (Latency Distribution)

在几乎所有视觉流水线中，都遵循以下三条铁律：

1. **预处理通常是最大的单一耗时模块。** 解码 JPEG、转换色彩空间、调整图像尺寸等操作均受限于 CPU 性能，且极易被忽视。
2. **检测器占据绝大部分 GPU 时间。** 70% 到 90% 的 GPU 耗时都集中在检测模型的前向传播 (Forward Pass) 阶段。
3. **后处理（如非极大值抑制 (NMS)、游程编码 (RLE) 编解码）在 GPU 上开销极小，但在 CPU 上却十分昂贵。** 务必使用实际目标硬件进行性能剖析 (Profiling)。

只有清楚延迟的具体分布，才能将优化工作转化为一份优先级明确的清单。

### 故障模式 (Failure Modes)

- **空检测结果** —— 返回空列表，切勿引发崩溃。记录日志。
- **越界边界框** —— 在裁剪前将其坐标限制 (Clamp) 在图像尺寸范围内。
- **过小裁剪区域** —— 若边界框尺寸小于分类器的最小输入要求，则跳过分类步骤。
- **损坏的上传文件** —— 返回带有特定错误码的 400 响应，而非 500 内部服务器错误。
- **模型加载失败** —— 应在服务启动时直接报错退出，而非等到处理第一个请求时才失败。

生产环境的流水线应妥善处理上述每种情况，避免使用笼统的 `try/except` 掩盖故障。每种故障都应分配明确的错误码并返回对应的响应。

### 批处理 (Batching)

生产服务通常需要同时服务多个客户端。跨请求对检测和分类任务进行批处理能成倍提升吞吐量。其代价是：等待批次填满会引入额外的延迟。典型配置为：收集请求最长 20 毫秒，合并为一批次进行处理，随后分发响应。`torchserve` 和 `triton` 原生支持此功能；对于负载可预测的小型服务，通常会自行实现微批处理器 (Micro-batcher)。

## 动手构建

### 步骤 1：数据契约（Data Contracts）

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class Detection(BaseModel):
    box: Tuple[float, float, float, float]
    score: float = Field(ge=0, le=1)
    class_id: int = Field(ge=0)
    mask_rle: Optional[str] = None


class Classification(BaseModel):
    detection_index: int
    class_id: int
    class_name: str
    score: float = Field(ge=0, le=1)


class PipelineResult(BaseModel):
    image_id: str
    detections: List[Detection]
    classifications: List[Classification]
    inference_ms: float

在任何严谨的流水线（Pipeline）中，花五秒钟编写代码，就能节省一小时的调试时间。

### 步骤 2：一个极简的 Pipeline 类

import time
import numpy as np
import torch
from PIL import Image

class VisionPipeline:
    def __init__(self, detector, classifier, class_names,
                 device="cpu", min_crop=32):
        self.detector = detector.to(device).eval()
        self.classifier = classifier.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.min_crop = min_crop

    def preprocess(self, image):
        """
        image: PIL.Image or np.ndarray (H, W, 3) uint8
        returns: CHW float tensor on device
        """
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.device)

    @torch.no_grad()
    def detect(self, image_tensor):
        return self.detector([image_tensor])[0]

    @torch.no_grad()
    def classify(self, crops):
        if len(crops) == 0:
            return []
        batch = torch.stack(crops).to(self.device)
        logits = self.classifier(batch)
        probs = logits.softmax(-1)
        scores, cls = probs.max(-1)
        return list(zip(cls.tolist(), scores.tolist()))

    def run(self, image, image_id="anonymous"):
        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        det = self.detect(tensor)

        crops = []
        detections = []
        valid_indices = []
        for i, (box, score, cls) in enumerate(zip(det["boxes"], det["scores"], det["labels"])):
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                score=float(score),
                class_id=int(cls),
            ))
            if (x2 - x1) < self.min_crop or (y2 - y1) < self.min_crop:
                continue
            crop = tensor[:, y1:y2, x1:x2]
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )[0]
            crops.append(crop)
            valid_indices.append(i)

        class_preds = self.classify(crops)

        classifications = []
        for valid_idx, (cls_id, cls_score) in zip(valid_indices, class_preds):
            classifications.append(Classification(
                detection_index=valid_idx,
                class_id=int(cls_id),
                class_name=self.class_names[cls_id],
                score=float(cls_score),
            ))

        return PipelineResult(
            image_id=image_id,
            detections=detections,
            classifications=classifications,
            inference_ms=(time.perf_counter() - t0) * 1000,
        )

每个接口都具备类型注解。每个异常路径都有明确的处理策略。

### 步骤 3：连接检测器与分类器

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import convnext_tiny

# Use ImageNet-pretrained weights for a realistic pipeline without training
detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
classifier = convnext_tiny(weights="DEFAULT")
class_names = [f"imagenet_class_{i}" for i in range(1000)]

pipe = VisionPipeline(detector, classifier, class_names)

# Smoke test with a synthetic image
test_image = (np.random.rand(400, 600, 3) * 255).astype(np.uint8)
result = pipe.run(test_image, image_id="demo")
print(result.model_dump_json(indent=2)[:500])

### 步骤 4：FastAPI 服务

from fastapi import FastAPI, UploadFile, HTTPException
from io import BytesIO

app = FastAPI()
pipe = None  # initialised on startup

@app.on_event("startup")
def load():
    global pipe
    detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT").eval()
    classifier = convnext_tiny(weights="DEFAULT").eval()
    pipe = VisionPipeline(detector, classifier, class_names=[f"c{i}" for i in range(1000)])

@app.post("/detect")
async def detect_endpoint(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="unsupported image type")
    data = await file.read()
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="cannot decode image")
    result = pipe.run(img, image_id=file.filename or "upload")
    return result.model_dump()

使用 `uvicorn main:app --host 0.0.0.0 --port 8000` 运行。通过 `curl -F 'file=@dog.jpg' http://localhost:8000/detect` 进行测试。

### 步骤 5：对流水线进行基准测试（Benchmark）

import time

def benchmark(pipe, num_runs=20, image_size=(400, 600)):
    img = (np.random.rand(*image_size, 3) * 255).astype(np.uint8)
    pipe.run(img)  # warm up

    stages = {"preprocess": [], "detect": [], "classify": [], "total": []}
    for _ in range(num_runs):
        t0 = time.perf_counter()
        tensor = pipe.preprocess(img)
        t1 = time.perf_counter()
        det = pipe.detect(tensor)
        t2 = time.perf_counter()
        crops = []
        for box in det["boxes"]:
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            if (x2 - x1) >= pipe.min_crop and (y2 - y1) >= pipe.min_crop:
                crop = tensor[:, y1:y2, x1:x2]
                crop = torch.nn.functional.interpolate(
                    crop.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                )[0]
                crops.append(crop)
        pipe.classify(crops)
        t3 = time.perf_counter()
        stages["preprocess"].append((t1 - t0) * 1000)
        stages["detect"].append((t2 - t1) * 1000)
        stages["classify"].append((t3 - t2) * 1000)
        stages["total"].append((t3 - t0) * 1000)

    for stage, times in stages.items():
        times.sort()
        print(f"{stage:12s}  p50={times[len(times)//2]:7.1f} ms  p95={times[int(len(times)*0.95)]:7.1f} ms")

在 CPU 上的典型输出：预处理（preprocess）约 3 毫秒，检测（detect）300-500 毫秒，分类（classify）20-40 毫秒，总计 350-550 毫秒。在 GPU 上，检测耗时降至 20-40 毫秒，此时预处理与分类的耗时在相对占比中开始变得更为重要。

## 投入生产使用

生产环境模板通常收敛为相同的结构，并额外包含以下特性：

- **模型版本控制 (Model versioning)** — 始终在响应中记录模型名称和权重哈希值。
- **单次请求追踪 ID (Per-request trace IDs)** — 记录每个请求在各阶段的耗时，以便将响应延迟与具体阶段关联起来。
- **降级路径 (Fallback path)** — 若分类器超时，则返回未分类的检测结果，而非让整个请求失败。
- **安全过滤器 (Safety filters)** — 在响应离开服务前、分类完成后运行 NSFW（不适宜内容）/ PII（个人身份信息）过滤。
- **批量处理端点 (Batch endpoint)** — 提供一个 `/detect_batch` 接口，接收图像 URL 列表以进行批量处理。

对于生产环境服务部署，`torchserve`、`Triton Inference Server` 和 `BentoML` 开箱即用地支持批处理、版本控制、指标监控和健康检查。直接运行 `FastAPI` 则适用于原型验证和小规模产品。

## 交付成果

本章节将产出以下文件：

- `outputs/prompt-vision-service-shape-reviewer.md` — 一个提示词 (Prompt)，用于审查视觉服务代码中是否存在违反数据契约或响应结构的问题，并指出首个导致功能中断的缺陷。
- `outputs/skill-pipeline-budget-planner.md` — 一项技能 (Skill)，在给定目标延迟 (Latency) 和吞吐量 (Throughput) 的前提下，为流水线 (Pipeline) 的每个阶段分配时间预算，并标记出哪个阶段将最先超出预算。

## 练习

1. **（简单）** 使用任意公开数据集中的 10 张图像运行该流水线。报告每个阶段的平均耗时，以及每张图像的检测结果数量分布。
2. **（中等）** 在 `Detection` 中添加掩码 (Mask) 输出字段，并将其编码为 RLE（游程编码）。验证即使图像包含 10 个目标，生成的 JSON 大小仍保持在 1MB 以内。
3. **（困难）** 在分类器前添加微批处理器 (Micro-batcher)：最多收集 10 毫秒的裁剪区域 (Crops)，通过单次 GPU 调用完成全部分类，并按请求返回结果。测量在每秒 5 个并发请求下的吞吐量提升幅度以及新增的延迟。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 流水线 (Pipeline) | “整个系统” | 由预处理、推理和后处理步骤组成的有序链条，每两个步骤之间通过类型化接口连接 |
| 数据契约 (Data contract) | “数据模式” | 每个阶段输入和输出都必须遵循的 Pydantic / dataclass 定义；用于在边界处捕获集成缺陷 |
| 预处理 (Preprocessing) | “模型运行前” | 解码、色彩转换、缩放、归一化；通常是 CPU 耗时最大的环节 |
| 后处理 (Postprocessing) | “模型运行后” | 非极大值抑制 (NMS)、掩码缩放、阈值过滤、RLE 编码；在 GPU 上开销小，在 CPU 上开销大 |
| 微批处理器 (Microbatcher) | “收集后转发” | 聚合器，等待固定时间窗口内的多个请求，执行单次批量前向传播 |
| 追踪 ID (Trace ID) | “请求 ID” | 每个请求的唯一标识符，在各阶段均会记录，以便端到端追踪慢请求 |
| 失败代码 (Failure code) | “具名错误” | 针对每类失败情况的具体错误代码，而非通用的 500 错误；便于客户端实现重试逻辑 |
| 健康检查 (Health check) | “就绪探针” | 轻量级端点，用于报告服务是否具备响应能力；负载均衡器依赖此机制 |

## 延伸阅读

- [全栈深度学习 — 模型部署](https://fullstackdeeplearning.com/course/2022/lecture-5-deployment/) — 生产环境机器学习（Machine Learning, ML）部署的权威概述
- [BentoML 文档](https://docs.bentoml.com) — 支持批处理（Batching）、版本控制（Versioning）与指标监控（Metrics）的模型服务（Serving）框架
- [TorchServe 文档](https://pytorch.org/serve/) — PyTorch 官方模型服务库
- [NVIDIA Triton 推理服务器](https://developer.nvidia.com/triton-inference-server) — 支持批处理与多模型（Multi-model）的高吞吐量模型服务