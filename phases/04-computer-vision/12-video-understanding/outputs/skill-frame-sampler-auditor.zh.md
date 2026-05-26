---
name: skill-frame-sampler-auditor
description: 审查视频流水线（video pipeline）的帧采样器（frame sampler），检查差一错误（off-by-one）、短片段处理（short-clip handling）和裁剪一致性（crop consistency）
version: 1.0.0
phase: 4
lesson: 12
tags: [计算机视觉, 视频, 采样, 调试]
---

# 帧采样器审查器（Frame Sampler Auditor）

帧采样（frame sampling）是视频流水线中最容易出问题的环节。此处的缺陷会传播到下游的每一个评估指标中。

## 适用场景

- 编写新的视频数据加载器（video data loader）时。
- 复现论文结果时，发现训练准确率低于报告值。
- 调试视频模型时，发现其评估准确率在不同运行间不稳定。

## 输入参数

- `sampler_code`：Python 函数，接收 `(num_frames_total, T)` 作为参数并返回 T 个索引。
- `T`：目标片段长度。
- 可选测试用例：用于测试的 `num_frames_total` 值（例如 `[3, T-1, T, T+1, 30, 300, 3000]`）。

## 检查项

### 1. 短片段处理（Short clip handling）
传入 `num_frames_total < T`。返回的每个索引必须位于 `[0, num_frames_total - 1]` 范围内。标准的填充策略（padding policy）是对剩余位置重复最后一帧。

### 2. 边界索引（Boundary indices）
传入 `num_frames_total == T`。返回的索引应严格为 `[0, 1, ..., T-1]`。

### 3. 均匀分布（Uniform distribution）
传入 `num_frames_total == 10 * T`。返回的索引应单调递增且大致均匀分布。

### 4. 密集窗口边界（Dense window bounds）
对于密集采样（dense sampling），传入 `num_frames_total == 3 * T`。返回的索引应构成一个连续窗口，且绝不能超出片段末尾。

### 5. 确定性（Determinism）
使用相同的输入（对于确定性采样器还需使用相同的随机数生成器 RNG）调用采样器两次。返回的索引应完全一致。

### 6. 裁剪一致性（Crop consistency）
如果流水线还会为每一帧返回空间裁剪（spatial crop），请使用相同的种子对同一片段运行采样器两次，并确认每一帧都使用相同的裁剪框（即相同的 `(x, y, w, h)`）。同一片段内各帧使用不同的裁剪框会破坏时间连贯性（temporal coherence），这是一种典型的隐蔽缺陷（silent bug）。可接受的变体：数据增强（augmentation）按*片段*应用，且在片段内保持一致。

## 报告格式

[sampler audit]
  name: <function name>
  T:    <int>

[short-clip handling]
  passed | failed (<details>)

[boundary]
  passed | failed

[uniform spacing]
  passed | failed (<stddev of gaps>)

[dense window]
  passed | failed (<details>)

[determinism]
  passed | failed

[crop consistency]
  passed | failed (<per-frame crop varies: yes/no>)

[verdict]
  ok | fix required

## 规则

- 如果短片段处理返回了越界索引，绝不可将采样器标记为“ok”。
- 密集采样器返回的窗口绝不可超出 `num_frames_total - 1`。
- 如果采样器是随机性的（stochastic），仅在显式指定种子的随机数生成器（RNG）下测试其确定性。
- 建议采用标准策略：用最后一帧填充、将窗口截断至末尾、对半开区间进行舍入；但切勿在后台静默修复。