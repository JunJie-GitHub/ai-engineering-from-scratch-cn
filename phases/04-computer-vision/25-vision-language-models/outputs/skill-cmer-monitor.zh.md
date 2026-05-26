---
name: skill-cmer-monitor
description: 为生产环境的视觉语言模型（VLM）端点集成跨模态错误率（Cross-Modal Error Rate）监控、仪表盘和告警
version: 1.0.0
phase: 4
lesson: 25
tags: [vlm, 生产环境, 监控, 幻觉]
---

# CMER 监控器

将跨模态对齐（cross-modal alignment）视为一等生产环境关键绩效指标（KPI）。

## 适用场景

- 部署任何基于图像生成文本的视觉语言模型（VLM）端点。
- 调查模型产生幻觉（hallucination）响应的报告。
- 追踪输入分布偏移（input distribution shift）是否削弱了模型的事实依据能力（grounding）。

## 输入参数

- `vlm_output`：生成的文本。
- `text_confidence`：经过 softmax 后的平均 token 概率，范围在 `[0, 1]`。计算方式为 `exp(mean(log_probs))`。请勿传入原始逻辑值（logits）；原始逻辑值无界，而 `conf_threshold` 假设输入为概率值。
- `image_embedding`：图像的 CLIP 系列嵌入向量（embedding）（如 DINOv3、SigLIP、CLIP）。
- `text_embedding`：生成文本的 CLIP 系列嵌入向量。
- 可选参数 `prompt_type`：用于分组的标签（如 vqa / ocr / captioning / agent）。

## 单次请求计算

import torch

def cmer_flag(image_emb, text_emb, text_conf, sim_thr=0.25, conf_thr=0.8):
    if image_emb.shape != text_emb.shape:
        raise ValueError(f"emb shape mismatch: {image_emb.shape} vs {text_emb.shape}")
    image_emb = image_emb / (image_emb.norm() + 1e-8)
    text_emb = text_emb / (text_emb.norm() + 1e-8)
    sim = float((image_emb * text_emb).sum())
    flagged = (text_conf > conf_thr) and (sim < sim_thr)
    return {"sim": sim, "flagged": flagged}

嵌入向量应为来自独立 CLIP 系列编码器的 1 维 PyTorch 张量（`torch.float32`）。若使用 NumPy 数组，请将 `.norm()` 替换为 `np.linalg.norm(...)` 并相应转换输出类型。

将 `sim`、`text_conf`、`flagged`、`prompt_type`、`timestamp`、`model_version`、`request_id` 存储至你的监控流水线（如 Prometheus、DataDog、OpenTelemetry）。

## 聚合指标

CMER = (flagged requests in window) / (total requests in window)

按端点、`prompt_type` 和模型版本分别上报。

## 告警阈值

- 基线 CMER：基于 7 天的正常流量数据建立。
- 警告：CMER 持续 1 小时达到基线的 1.5 倍及以上。
- 严重：CMER 持续 30 分钟达到基线的 2 倍及以上，或在任意时间窗口内绝对值超过 15%。

## 仪表盘面板

1. CMER 随时间变化趋势（5 分钟时间桶，7 天时间窗口）。
2. 按 `prompt_type` 划分的 CMER（堆叠柱状图）。
3. 每小时 `sim` 值分布（直方图）。
4. 高频幻觉输出（每日抽样 20 个被标记的响应供人工复核）。

## CMER 突增时的应对措施

1. 抽样检查被标记的请求。
2. 核实模型版本是否发生意外变更。
3. 检查输入数据分布（是否出现新文件格式？新图像来源？压缩方式不同？）。
4. 将受影响的流量路由至人工审核，直至突增现象消除。
5. 若突增持续存在，请对模型进行微调或替换；切勿直接屏蔽告警。

## 核心规则

- 切勿使用 VLM 自身的嵌入向量计算 CMER；必须使用独立的编码器（如 DINOv3、SigLIP 或 CLIP-L/14）。否则你测量的将是模型的自洽性（self-consistency），而非跨模态对齐程度。
- 始终记录原始 `sim` 值，而非仅记录 `flagged` 标志位；分布偏移通常会在标记率发生变化前，率先体现在下四分位数（lower quartile）中。
- 未集成 CMER 监控的 VLM 端点严禁上线；幻觉是生产环境中最主要的故障模式，若无此指标，故障将处于静默状态。
- 对于敏感领域（医疗、法律、金融），请将 `sim_threshold` 提高至 0.35 或更高；标记条件为 `sim < sim_threshold`，因此更高的阈值会将更多输出捕获为潜在缺乏事实依据（ungrounded）的结果——这是高风险应用场景的正确默认设置。