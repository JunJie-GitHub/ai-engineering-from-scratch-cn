---
name: skill-pipeline-budget-planner
description: 给定目标延迟 (latency) 和吞吐量 (throughput)，为每个流水线阶段分配时间预算，并标记哪个阶段将最先超出预算
version: 1.0.0
phase: 4
lesson: 16
tags: [视觉, 流水线, 性能, 部署]
---

# 流水线预算规划器 (Pipeline Budget Planner)

将延迟/吞吐量目标转化为分阶段的预算，使每位团队成员都清楚自己需要优化的具体指标。

## 适用场景

- 在构建新的视觉服务之前，用于设定各阶段的预期目标。
- 在首次基准测试 (benchmark) 之后，用于查看哪个阶段偏离预算最远。
- 当服务等级协议 (SLA) 发生变更且需要重新协商预算时。

## 输入参数

- `p95_latency_target_ms`：单次请求的预算。
- `target_qps`：每个副本的吞吐量。
- `stages`：包含 `{ name: str, current_ms: float }` 的列表。

## 分配规则

若未提供当前测量数据，则按以下比例在七个标准阶段间进行默认分配：

| 阶段 | 占比 |
|-------|-------|
| 解码与预处理 (decode + preprocess) | 15% |
| 检测器前向传播 (detector forward) | 55% |
| 检测结果后处理（非极大值抑制 NMS、边界截断 clamp） | 5% |
| 分类器裁剪与缩放 (crop + resize for classifier) | 5% |
| 分类器前向传播 (classifier forward) | 15% |
| 模式验证 (schema validation) | <1% |
| 响应序列化 (response serialisation) | 4% |

在受 GPU 限制的流水线（云端）中，检测器的占比通常会升至 70%。而在 CPU 环境下，预处理和分类器批处理会占用更多时间。

## 报告输出

[budget plan]
  p95 target:  <ms>
  throughput:  <qps per replica>

| stage               | target_ms | current_ms | headroom | gate |
|---------------------|-----------|------------|----------|------|
| decode+preprocess   | ...       | ...        | ...      | ok|X |
| detector            | ...       | ...        | ...      | ok|X |
| ...                 | ...       | ...        | ...      |      |

[bottleneck]
  stage:  <name>
  miss:   <ms over budget>
  lever:  <specific action>

[levers]
  decode+preprocess:   Pillow-SIMD, libjpeg-turbo, decode on GPU via NVJPEG
  detector:            smaller backbone, lower input resolution, INT8, TensorRT
  postprocess:         GPU-side NMS (torchvision.ops), fused masks
  crop+resize:         GPU crop with grid_sample, batched interpolate
  classifier:          smaller backbone, INT8, warm cache, batch
  schema:              skip validation in hot path, validate at boundaries only
  response:            orjson, stream protobuf

## 规则

- 切勿建议在生产路径中移除模式验证；应建议将其移至系统边界处执行。
- 若预处理超出预算，在更换模型之前，应优先尝试使用 Pillow-SIMD 或 NVJPEG。
- 若检测器超出预算的幅度超过目标值的 30%，应直接更换模型，而非继续优化当前模型。
- 当 `current_ms > 1.1 * target_ms` 时，将门控标记为 `X`；若处于预算的 10% 容差范围内，则标记为 `ok`。