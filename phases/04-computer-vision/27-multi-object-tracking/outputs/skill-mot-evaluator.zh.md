---
name: skill-mot-evaluator
description: 编写针对真实轨迹（ground-truth tracks）的完整评估框架（evaluation harness），用于计算 MOTA / IDF1 / HOTA 指标
version: 1.0.0
phase: 4
lesson: 27
tags: [mot, evaluation, tracking, metrics]
---

# MOT 评估器（MOT Evaluator）

将你的跟踪器（tracker）输出接入标准的 MOTA/IDF1/HOTA 评估流程，以便与现有文献进行公平对比。

## 适用场景

- 在 MOT17 / MOT20 / DanceTrack / SportsMOT 等数据集上对新跟踪器进行基准测试（benchmarking）。
- 在你自己的视频素材上对比 ByteTrack、BoT-SORT 与 SAM 2 的表现。
- 为学术论文或拉取请求（Pull Request）描述生成可复现的评估指标。

## 输入参数

- `predictions`：按帧组织的列表，包含 `(track_id, x, y, w, h, confidence)` 元组。
- `ground_truth`：按帧组织的列表，包含 `(gt_id, x, y, w, h)` 元组。
- `iou_threshold`：交并比阈值（IoU threshold），MOTA 通常设为 0.5；HOTA 则采用阈值扫描（sweep）。
- `evaluator`：`py-motmetrics`（用于 MOTA、IDF1）或 `TrackEval`（用于 HOTA）。

## 输出格式规范

`py-motmetrics` 和 `TrackEval` 均要求特定的磁盘文件格式：

# predictions.txt
<frame>,<track_id>,<x>,<y>,<w>,<h>,<confidence>,-1,-1,-1

# ground_truth.txt
<frame>,<gt_id>,<x>,<y>,<w>,<h>,1,-1,-1,-1

帧索引从 1 开始，边界框格式为 (x, y, w, h) 而非 (x1, y1, x2, y2)。格式转换是集成过程中最容易引发缺陷（integration bugs）的环节。

## 操作步骤

1. 将跟踪器的输出转换为 MOT 挑战赛（MOT Challenge）文本格式。
2. 对两个文件分别运行 `py-motmetrics.io.loadtxt`。
3. 使用 `mm.metrics.create().compute()` 计算 MOTA 与 IDF1。
4. 对于 HOTA，使用相同的文件并配置 `Metrics: HOTA` 调用 `TrackEval`。
5. 将结果保存为 JSON 格式，以便接入可视化仪表盘。

## 代码实现示例

import motmetrics as mm

def evaluate_mota_idf1(pred_path, gt_path):
    gt = mm.io.loadtxt(gt_path, fmt="mot15-2D")
    pred = mm.io.loadtxt(pred_path, fmt="mot15-2D")
    acc = mm.utils.compare_to_groundtruth(gt, pred, dist="iou", distth=0.5)
    metrics = mm.metrics.create().compute(
        acc, metrics=["num_frames", "mota", "motp", "idf1", "idp", "idr", "num_switches"]
    )
    return metrics


def write_mot_txt(predictions, path):
    with open(path, "w") as f:
        for frame_idx, detections in enumerate(predictions, start=1):
            for tid, x, y, w, h, conf in detections:
                f.write(f"{frame_idx},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.3f},-1,-1,-1\n")

## 评估报告格式

[mot evaluation]
  frames:     <int>
  gt tracks:  <int>
  pred tracks: <int>

[metrics]
  MOTA:       <float>
  MOTP:       <float>
  IDF1:       <float>
  IDP/IDR:    <float/float>
  ID switches: <int>
  HOTA:       <float>  (from TrackEval)

## 注意事项

- 输出文本文件中的帧索引必须从 1 开始；MOT 相关工具链均以此为标准。
- 写入文件前，务必将 (x1, y1, x2, y2) 格式转换为 (x, y, w, h)。
- 现代对比评估中不应仅报告 MOTA；需同时包含 IDF1 与 HOTA。
- 注意 MOT17 数据集中的私有检测（private detections）与公开检测（public detections）——两者需分开评估，混合使用会导致分数虚高。
- 记录每个视频序列（sequence）的独立得分；整体聚合分数会掩盖在单个困难序列上的失败情况。