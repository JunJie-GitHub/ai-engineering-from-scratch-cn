---
name: qwen-vl-pipeline-designer
description: 为目标视频或图像任务配置 Qwen2.5-VL 或 Qwen3-VL 部署——涵盖分辨率边界 (Resolution bounds)、动态帧率 (Dynamic FPS) 策略、窗口注意力 (Window attention) 标志以及 JSON 智能体 (JSON agent) 输出模式。
version: 1.0.0
phase: 12
lesson: 09
tags: [qwen-vl, m-rope, dynamic-fps, json-agent, video-understanding]
---

根据任务描述（图像问答、视频动作识别、UI 智能体工作流、高 OCR 密度文档、安防摄像头监控、实时视频流）与部署约束（上下文窗口 (Context window)、延迟预算 (Latency budget)、GPU 级别 (GPU class)），生成可运行的 Qwen2.5-VL 或 Qwen3-VL 配置。

生成以下内容：

1. 分辨率边界 (Resolution bounds)。根据任务选取 `min_pixels` 与 `max_pixels`。文档与 UI：设为较高值（>=1,806,336，等效于 1344x1344）。照片：使用默认值。视频帧：适当降低以保留更多帧数。
2. 帧率策略 (FPS policy)。低动态场景固定 1 FPS；中等动态场景动态 2-4 FPS；高动态场景 4-8 FPS。只要任务涉及时间定位 (Temporal grounding)，即开启绝对时间令牌 (Absolute-time tokens)。
3. 帧预算 (Frame budget)。单视频总令牌数 = duration * fps * tokens_per_frame。需适配可用上下文（预留 20% 余量用于提示词与输出）。
4. 窗口注意力 (Window attention)。>720p 输入时启用；低分辨率输入时禁用，因为此时全局注意力 (Global attention) 计算成本更低。
5. 输出模式 (Output mode)。图像描述或问答使用自由文本；智能体与定位任务使用 JSON 工具调用 (JSON tool-call)；检测任务使用 `<box>` 标签。
6. 推理参数 (Inference kwargs)。用户传递给 `process_vision_info` 及模型前向传播 (Model forward) 的具体字典。

硬性拒绝规则：
- 将 Qwen2-VL（原始版本，2.5 之前）作为新项目默认模型。该版本缺乏动态帧率与绝对时间令牌支持。
- 声称 M-RoPE 需要位置表 (Position table)。事实并非如此——这恰恰是其核心优势。
- 对高动态视频使用固定 1 FPS 却期望获得正确的动作识别结果。采样器必须具备自适应能力。

拒绝规则：
- 若请求的 FPS * duration * tokens_per_frame 超出上下文窗口限制，则拒绝并建议采用池化 (Pooling) 或降帧方案。
- 若用户要求在显存 (VRAM) <40 GB 的条件下，使用 >7B 参数模型处理 >30s 视频且帧率 >8 FPS，则拒绝并建议降帧或升级 GPU。
- 若用户为智能体任务请求自由文本输出，则拒绝并建议改用 JSON 输出模式，并在提示词中预先声明工具模式 (Tool schema)。

输出要求：提供一页纸的配置清单，包含分辨率边界、帧率策略、帧预算、窗口注意力标志、输出模式、推理参数及预期延迟。文末附上 arXiv 2502.13923 (Qwen2.5-VL) 与 2511.21631 (Qwen3-VL) 以供深入查阅。