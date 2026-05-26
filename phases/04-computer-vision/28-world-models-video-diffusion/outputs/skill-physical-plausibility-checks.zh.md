---
name: 物理合理性检查技能
description: 在发布前对任何生成视频进行物体恒存性、重力和连续性的自动化检查
version: 1.0.0
phase: 4
lesson: 28
tags: [视频生成, 质量, 物理, 评估]
---

# 物理合理性检查 (Physical Plausibility Checks)

生成视频的生产级部署需要自动化护栏（automated guardrails）。人工审核难以规模化扩展；物理检查能有效捕捉典型的失败模式（failure modes）。

## 适用场景

- 任何通过文本或图像提示词（prompts）生成视频的产品。
- 对视频生成 API 端点进行自动化质量保证（QA）。
- 在微调（fine-tuning）或基础模型（base-model）更新后，监控视频模型的质量漂移（quality drift）。

## 输入

- `video`：形状为 `(T, H, W, 3)` 的张量（tensor）或 mp4 文件路径。
- 可选参考信息：预期物体数量、初始场景描述。

## 检查项

### 1. 物体恒存性 (Object Permanence)
使用 SAM 3.1 Object Multiplex 跨帧跟踪每一个检测目标。当稳定轨迹消失不超过 3 帧后重新出现时进行标记——这表明模型暂时丢失了该物体。若物体在画面中心区域（非边缘）消失，则判定为硬性失败（hard fail）；若在边缘消失，则判定为软性失败（soft fail）。

### 2. 运动平滑度 (Motion Smoothness)
连续帧之间的光流（optical flow）应保持基本连续。像素级光流的突然激增通常意味着物体发生了瞬移。使用 RAFT 计算光流；当某帧的第 99 百分位光流幅值超过中位数的 10 倍以上时，进行标记。

### 3. 重力/支撑 (Gravity / Support)
对于被检测为固体的物体（如食物、球体、工具），在没有抬起动作的情况下，检查其垂直位置是否非递增。除非在物体附近检测到“抓取的手”，否则对向上漂移进行标记。

### 4. 身份一致性 (Identity Consistency)
对于人物或角色，跨帧使用人脸识别嵌入向量（face-recognition embedding）。在 5 帧的滑动窗口内，持续身份的余弦相似度（cosine similarity）应保持在 0.8 以上。低于该阈值意味着角色发生了形变。

### 5. 手部与肢体 (Hands and Limbs)
运行姿态估计器（pose estimator，见第 21 课）。标记以下帧：手部可见手指数大于 5 或小于 4；手臂长度在帧间翻倍；肢体穿过身体表面发生穿插。

### 6. 文本渲染（若提示词要求生成文本）
如果用户提示词中包含引号内的字符串，则对生成帧进行光学字符识别（OCR），并计算与目标字符串的字符错误率（CER）。若 CER 超过 20%，则进行标记。

## 报告

[plausibility]
  video frames:           <T>
  permanence violations:  <N>
  smoothness violations:  <N>
  gravity violations:     <N>
  identity drift:         <N of 5-frame windows>
  limb anomalies:         <N>
  OCR CER vs requested:   <float>

[verdict]
  ship | hold | reject

[samples for review]
  frame ranges where each failure occurred

## 规则

- 不要因单一检查项而硬性拦截；汇总各项得分，当总异常数超过阈值时，将视频挂起以待人工审核。
- 赋予身份漂移和恒存性违规最高的权重——用户通常最先注意到这些问题。
- 记录各检查项随时间推移的失败率；上升趋势通常意味着基础模型已更新或提示词分布发生了偏移。
- 切勿删除被标记的视频；保留它们用于模型调试和事后复盘（post-mortems）。
- 对于敏感内容（人物、儿童、公众人物），无论评分如何，均要求对每段视频进行人工审核。