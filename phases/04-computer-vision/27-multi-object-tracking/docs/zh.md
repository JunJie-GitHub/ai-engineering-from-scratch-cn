# 多目标跟踪与视频记忆

> 跟踪即检测加关联。逐帧检测。通过 ID 将当前帧的检测结果与上一帧的轨迹进行匹配。

**类型：** 实战构建
**语言：** Python
**前置条件：** 第 4 阶段第 06 课（YOLO 检测）、第 4 阶段第 08 课（Mask R-CNN）、第 4 阶段第 24 课（SAM 3）
**时长：** 约 60 分钟

## 学习目标

- 区分基于检测的跟踪（Tracking-by-Detection）与基于查询的跟踪（Query-Based Tracking），并列举相关算法家族（SORT、DeepSORT、ByteTrack、BoT-SORT、SAM 2 记忆跟踪器、SAM 3.1 对象复用）
- 从零实现交并比（IoU）结合匈牙利分配算法（Hungarian Assignment）的逻辑，用于经典的基于检测的跟踪
- 解释 SAM 2 的记忆库（Memory Bank）机制，以及为何它在处理遮挡（Occlusion）时优于基于 IoU 的关联方法
- 解读三大跟踪评估指标（MOTA、IDF1、HOTA），并针对具体应用场景选择合适的指标

## 问题背景

检测器（Detector）能告诉你单帧图像中物体的位置。而跟踪器（Tracker）则能告诉你第 `t` 帧中的某个检测结果与第 `t-1` 帧中的哪个检测结果属于同一个物体。缺乏这一能力，你将无法统计穿越警戒线的物体数量、在遮挡情况下持续追踪球的轨迹，也无法判断“4 号车已在车道内停留了 8 秒”。

跟踪技术是各类视频应用产品的核心基石：体育分析、安防监控、自动驾驶、医疗视频分析、野生动物监测以及文字商标计数等。其核心构建模块是通用的：逐帧检测器、运动模型（Motion Model，如卡尔曼滤波 Kalman Filter 或更复杂的模型）、关联步骤（Association Step，基于 IoU/余弦相似度/学习特征的匈牙利算法匹配），以及轨迹生命周期（Track Lifecycle，包含生成、更新与消亡）。

2026 年涌现出两种新范式：**基于 SAM 2 记忆的跟踪**（使用特征记忆替代运动模型关联）与 **SAM 3.1 对象复用（Object Multiplex）**（为同一概念的多个实例共享记忆）。本课将首先梳理经典技术栈，随后深入讲解基于记忆的方法。

## 核心概念

### 检测跟踪（Tracking-by-detection）

flowchart LR
    F1["Frame t"] --> DET["Detector"] --> D1["Detections at t"]
    PREV["Tracks up to t-1"] --> PREDICT["Motion predict<br/>(Kalman)"]
    PREDICT --> PRED["Predicted tracks at t"]
    D1 --> ASSOC["Hungarian assignment<br/>(IoU / cosine / motion)"]
    PRED --> ASSOC
    ASSOC --> UPDATE["Update matched tracks"]
    ASSOC --> NEW["Birth new tracks"]
    ASSOC --> DEAD["Age unmatched tracks; delete after N"]
    UPDATE --> NEXT["Tracks at t"]
    NEW --> NEXT
    DEAD --> NEXT

    style DET fill:#dbeafe,stroke:#2563eb
    style ASSOC fill:#fef3c7,stroke:#d97706
    style NEXT fill:#dcfce7,stroke:#16a34a

到 2026 年，你所接触到的所有跟踪器（Tracker）都是该循环的变体。主要区别在于：

- **SORT**（2016）：卡尔曼滤波（Kalman filter）+ 基于交并比（IoU）的匈牙利算法。结构简单、速度快，不包含外观模型。
- **DeepSORT**（2017）：在 SORT 基础上为每个轨迹引入基于卷积神经网络（CNN）的外观特征（重识别嵌入，ReID embedding）。能更好地处理目标交叉情况。
- **ByteTrack**（2021）：将低置信度检测框作为第二阶段进行关联；无需外观特征，但在 MOT17 数据集上表现最佳。
- **BoT-SORT**（2022）：ByteTrack + 相机运动补偿（Camera motion compensation）+ 重识别（ReID）。
- **StrongSORT / OC-SORT**：ByteTrack 的衍生算法，在运动建模和外观特征方面进行了优化。

### 一段话理解卡尔曼滤波（Kalman filter）

卡尔曼滤波为每个目标轨迹维护一个包含协方差（Covariance）的状态向量 `(x, y, w, h, dx, dy, dw, dh)`。在每一帧中，首先使用匀速模型（Constant-velocity model）**预测**状态，然后利用匹配到的检测框进行**更新**。当预测不确定性较高时，更新过程会赋予检测框更高的权重。这种方法能够生成平滑的运动轨迹，并具备在短暂遮挡（1-5 帧）期间维持轨迹跟踪的能力。

所有经典跟踪算法均在运动预测步骤中使用卡尔曼滤波。

### 匈牙利算法（Hungarian algorithm）

给定一个 `M x N` 的代价矩阵（Cost matrix）（轨迹 × 检测框），寻找使总代价最小的一对一分配方案。代价通常定义为 `1 - IoU(track_bbox, detection_bbox)` 或外观特征余弦相似度的负值。其时间复杂度为 O((M+N)^3)；当 M、N 达到约 1000 时，通过 Python 的 `scipy.optimize.linear_sum_assignment` 仍能保证足够的运行速度。

### ByteTrack 的核心思想

传统跟踪器通常会直接丢弃低置信度（< 0.5）的检测框。ByteTrack 则将其保留为**第二阶段候选框**：在将轨迹与高置信度检测框完成匹配后，未匹配的轨迹会尝试使用稍宽松的交并比阈值与低置信度检测框进行二次匹配。该方法能有效恢复短暂遮挡，并减少人群密集区域的 ID 切换（ID switch）问题。

### SAM 2 基于记忆的跟踪（Memory-based tracking）

SAM 2 通过维护一个包含每个实例时空特征的**记忆库**（Memory bank）来处理视频数据。当在某一帧提供提示（点击、边界框或文本）时，模型会将该实例编码并存入记忆库。在后续帧中，记忆库特征会与新帧特征进行交叉注意力（Cross-attention）计算，解码器随后为新帧中同一实例生成掩码（Mask）。

无需卡尔曼滤波，也无需匈牙利算法分配。目标关联过程隐式地包含在记忆注意力机制中。

优势：
- 对严重遮挡具有鲁棒性（记忆机制可在多帧间保持实例身份）。
- 结合 SAM 3 的文本提示后，支持开放词汇（Open-vocabulary）跟踪。
- 无需依赖独立的运动模型。

劣势：
- 在多目标跟踪场景下，速度慢于 ByteTrack。
- 记忆库会随时间增长，从而限制了上下文窗口（Context window）的大小。

### SAM 3.1 对象多路复用（Object Multiplex）

早期的 SAM 2 / SAM 3 跟踪方案为每个实例维护独立的记忆库。若有 50 个目标，则需 50 个记忆库。对象多路复用（Object Multiplex，2026 年 3 月发布）将它们合并为一个共享记忆库，并引入**实例专属查询令牌**（Per-instance query tokens）。其计算开销随实例数量呈亚线性增长。

多路复用已成为 2026 年人群跟踪的新默认方案：适用于演唱会人群、仓库作业人员及交通路口等场景。

### 三大核心评估指标

- **MOTA**（多目标跟踪准确率，Multi-Object Tracking Accuracy） — 计算公式为 1 - (FN + FP + ID switches) / GT。该指标按错误类型加权，是一个将检测失败与关联失败合并考量的单一综合指标。
- **IDF1**（ID F1 分数） — ID 精确率（Precision）与召回率（Recall）的调和平均数。专门衡量每个真实轨迹（Ground-truth track）在时间维度上保持 ID 一致性的能力。在对 ID 切换敏感的任务中，其表现优于 MOTA。
- **HOTA**（高阶跟踪准确率，Higher Order Tracking Accuracy） — 可分解为检测准确率（DetA）和关联准确率（AssA）。自 2020 年起成为学术界标准，评估最为全面。

在安防监控（身份识别）场景中，应报告 IDF1；在体育分析（如传球计数）场景中，使用 HOTA；在常规学术对比中，同样采用 HOTA。

## 构建

### 步骤 1：基于交并比（IoU）的代价矩阵

import numpy as np


def bbox_iou(a, b):
    """
    a, b: (N, 4) arrays of [x1, y1, x2, y2].
    Returns (N_a, N_b) IoU matrix.
    """
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)

### 步骤 2：极简 SORT 风格跟踪器

为简洁起见，此处省略了固定匀速卡尔曼滤波（Kalman Filter）部分——我们在此仅使用简单的 IoU 关联；在实际生产环境中，卡尔曼预测（Kalman Predict）步骤是必不可少的。完整的实现可参考 `sort` Python 包。

from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, tid, bbox, frame):
        self.id = tid
        self.bbox = bbox
        self.last_frame = frame
        self.hits = 1

    def update(self, bbox, frame):
        self.bbox = bbox
        self.last_frame = frame
        self.hits += 1


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=5):
        self.tracks = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def step(self, detections, frame):
        if not self.tracks:
            for d in detections:
                self.tracks.append(Track(self.next_id, d, frame))
                self.next_id += 1
            return [(t.id, t.bbox) for t in self.tracks]

        track_boxes = np.array([t.bbox for t in self.tracks])
        det_boxes = np.array(detections) if len(detections) else np.empty((0, 4))

        iou = bbox_iou(track_boxes, det_boxes) if len(det_boxes) else np.zeros((len(track_boxes), 0))
        cost = 1 - iou
        cost[iou < self.iou_threshold] = 1e6

        matched_track = set()
        matched_det = set()
        if cost.size > 0:
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if cost[r, c] < 1.0:
                    self.tracks[r].update(det_boxes[c], frame)
                    matched_track.add(r); matched_det.add(c)

        for i, d in enumerate(det_boxes):
            if i not in matched_det:
                self.tracks.append(Track(self.next_id, d, frame))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if frame - t.last_frame <= self.max_age]
        return [(t.id, t.bbox) for t in self.tracks]

仅 60 行代码。接收逐帧检测结果（detections），返回逐帧跟踪 ID（track IDs）。实际系统还会加入卡尔曼预测、ByteTrack 的第二阶段重匹配（re-match）以及外观特征（appearance features）。

### 步骤 3：合成轨迹测试

def synthetic_frames(num_frames=20, num_objects=3, H=240, W=320, seed=0):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(20, 200, size=(num_objects, 2))
    velocities = rng.uniform(-5, 5, size=(num_objects, 2))
    frames = []
    for f in range(num_frames):
        dets = []
        for i in range(num_objects):
            cx, cy = starts[i] + f * velocities[i]
            dets.append([cx - 10, cy - 10, cx + 10, cy + 10])
        frames.append(dets)
    return frames


tracker = SimpleTracker()
for f, dets in enumerate(synthetic_frames()):
    tracks = tracker.step(dets, f)

三个沿直线运动的物体应在全部 20 帧中保持其 ID 不变。

### 步骤 4：ID 切换（ID-switch）指标

def count_id_switches(tracks_per_frame, gt_per_frame):
    """
    tracks_per_frame:  list of list of (track_id, bbox)
    gt_per_frame:      list of list of (gt_id, bbox)
    Returns number of ID switches.
    """
    prev_assignment = {}
    switches = 0
    for tracks, gts in zip(tracks_per_frame, gt_per_frame):
        if not tracks or not gts:
            continue
        t_boxes = np.array([b for _, b in tracks])
        g_boxes = np.array([b for _, b in gts])
        iou = bbox_iou(g_boxes, t_boxes)
        for g_idx, (gt_id, _) in enumerate(gts):
            j = iou[g_idx].argmax()
            if iou[g_idx, j] > 0.5:
                t_id = tracks[j][0]
                if gt_id in prev_assignment and prev_assignment[gt_id] != t_id:
                    switches += 1
                prev_assignment[gt_id] = t_id
    return switches

这是一个简化的类 IDF1（ID F1 Score）指标：用于统计真实标注（ground-truth）对象更改其分配的预测跟踪 ID 的次数。完整的 MOTA（多目标跟踪准确率）/ IDF1 / HOTA（高阶跟踪准确率）评估工具可在 `py-motmetrics` 和 `TrackEval` 库中找到。

## 实际应用

2026 年生产环境中的目标跟踪器（Object Trackers）：

- `ultralytics` — 内置 YOLOv8 + ByteTrack / BoT-SORT。使用 `results = model.track(source, tracker="bytetrack.yaml")` 调用。此为默认选项。
- `supervision`（Roboflow）—— 提供 ByteTrack 封装器及标注工具。
- SAM 2 / SAM 3.1 —— 通过 `processor.track()` 实现基于记忆库（Memory Bank）的跟踪。
- 自定义技术栈（Custom Stack）：检测器（Detector）（YOLOv8 / RT-DETR） + `sort-tracker` / `OC-SORT` / `StrongSORT`。

选型建议：

- 行人/车辆/边界框且帧率 30+ fps：**搭配 `ultralytics` 的 ByteTrack**。
- 人群中同一类别的大量实例：**SAM 3.1 Object Multiplex（对象复用）**。
- 严重遮挡但外观特征可辨识：**DeepSORT / StrongSORT**（依赖重识别 ReID 特征）。
- 体育运动/复杂交互场景：**BoT-SORT** 或学习型跟踪器（Learned Trackers）（如 MOTRv3）。

## 交付物

本课时将产出：

- `outputs/prompt-tracker-picker.md` —— 根据场景类型、遮挡模式及延迟预算，自动推荐 SORT / ByteTrack / BoT-SORT / SAM 2 / SAM 3.1。
- `outputs/skill-mot-evaluator.md` —— 编写完整的评估框架（Evaluation Harness），用于针对真实轨迹（Ground-Truth Tracks）计算 MOTA / IDF1 / HOTA 指标。

## 练习

1. **（简单）** 使用上述合成跟踪器分别处理包含 3、10 和 30 个目标的场景。报告每种情况下的 ID 切换（ID-Switch）次数。指出仅依赖交并比（IoU）的简单关联策略在何处开始失效。
2. **（中等）** 在关联步骤前加入恒定速度的卡尔曼滤波（Kalman Filter）预测步骤。验证短时（2-3 帧）遮挡不再引发 ID 切换。
3. **（困难）** 集成 SAM 2 的基于记忆库的跟踪器（通过 `transformers` 库）作为替代跟踪后端。在一段 30 秒的人群视频片段上同时运行 SimpleTracker 和 SAM 2，对比两者的 ID 切换次数，并手动为 5 个显著人物标注真实 ID（Ground-Truth IDs）。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Tracking-by-detection（检测后跟踪） | “先检测，后关联” | 逐帧检测器 + 基于交并比（IoU）/外观的匈牙利算法（Hungarian Algorithm）分配 |
| Kalman filter（卡尔曼滤波） | “运动预测” | 线性动力学模型 + 协方差矩阵，用于平滑轨迹预测与处理遮挡 |
| Hungarian algorithm（匈牙利算法） | “最优分配” | 求解最小代价二分图匹配问题；对应 `scipy.optimize.linear_sum_assignment` |
| ByteTrack | “低置信度二次匹配” | 将未匹配的轨迹与低置信度检测结果重新匹配，以恢复短时遮挡目标 |
| DeepSORT | “SORT + 外观特征” | 引入重识别（ReID）特征进行跨帧匹配；更利于保持 ID 一致性 |
| Memory bank（记忆库） | “SAM 2 的秘诀” | 跨帧存储每个实例的时空特征；通过交叉注意力（Cross-Attention）机制替代显式关联 |
| Object Multiplex（对象复用） | “SAM 3.1 共享记忆” | 单一共享记忆配合每个实例的独立查询（Query），实现快速的多目标跟踪 |
| HOTA | “现代跟踪评估指标” | 将检测精度与关联精度解耦；已成为社区标准 |

## 延伸阅读

- [SORT (Bewley 等, 2016)](https://arxiv.org/abs/1602.00763) — 极简的基于检测的跟踪（tracking-by-detection）论文
- [DeepSORT (Wojke 等, 2017)](https://arxiv.org/abs/1703.07402) — 引入外观特征（appearance feature）
- [ByteTrack (Zhang 等, 2022)](https://arxiv.org/abs/2110.06864) — 低置信度二次关联（second pass）
- [BoT-SORT (Aharon 等, 2022)](https://arxiv.org/abs/2206.14651) — 相机运动补偿（camera motion compensation）
- [HOTA (Luiten 等, 2020)](https://arxiv.org/abs/2009.07736) — 解耦式跟踪评估指标（decomposed tracking metric）
- [SAM 2 视频分割 (Meta, 2024)](https://ai.meta.com/sam2/) — 基于记忆的跟踪器（memory-based tracker）
- [SAM 3.1 对象多路复用 (Meta, 2026年3月)](https://ai.meta.com/blog/segment-anything-model-3/)