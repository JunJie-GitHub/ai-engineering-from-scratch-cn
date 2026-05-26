---
name: video-vlm-frame-planner
description: 为视频语言模型（Video-Language Model）部署规划帧采样（Frame Sampling）、逐帧池化（Per-frame Pooling）、输出格式及基准测试目标。
version: 1.0.0
phase: 12
lesson: 17
tags: [video-vlm, temporal-grounding, tmrope, dynamic-fps, benchmarks]
---

给定视频任务（动作识别（Action Recognition）、时序定位（Temporal Grounding）、摘要生成（Summarization）、监控（Monitoring）、智能体工作流回放（Agent-Workflow Replay））与部署约束（模型上下文长度（Model Context）、延迟预算（Latency Budget）、吞吐量（Throughput）），生成帧采样与输出方案。

输出内容：

1. 帧采样器选择。静态内容采用均匀采样（Uniform Sampling），混合运动场景采用动态帧率（Dynamic-FPS），动作密集型场景采用事件驱动（Event-Driven），影视级内容采用关键帧+上下文（Keyframe+Context）。
2. 逐帧池化（Per-frame Pooling）。高细节场景采用 2x2，默认采用 3x3，智能体工作流（Agent Workflows）中覆盖范围比内容密度更重要时采用 4x4 或 6x6。
3. 时序编码（Temporal Encoding）。Qwen2.5-VL 系列模型使用 TMRoPE；较小模型使用可学习时序嵌入（Learned Temporal Embedding）；单片段任务无需编码。
4. 输出格式。定位任务使用包含 `{event, start, end, confidence}` 的 JSON；摘要任务使用自由文本；混合流程使用 Token 分隔格式（Token-delimited）。
5. 基准测试计划。通用任务使用 VideoMME，定位任务使用 TempCompass，长程任务使用 EgoSchema。需明确预期准确率等级。
6. 上下文/延迟预算。总 Token 数 = 时长 * 帧率 * 每帧 Token 数。若超过上下文长度的 40% 则发出警告。

硬性拒绝条件：
- 为动作密集型视频推荐均匀采样。会丢失峰值事件。
- 声称 Token 分隔输出在下游解析中能达到与 JSON 相同的准确率。JSON 的鲁棒性更强。
- 为 2026 年启动的任何项目推荐 Video-LLaMA。旧架构已不再具备竞争力。

拒绝规则：
- 若视频时长 > 10 分钟且上下文长度 < 32k，则拒绝并推荐分层摘要（Hierarchical Summarization）或智能体检索（Agentic Retrieval）（第 12.18 课）。
- 若目标准确率为前沿水平（在 VideoMME 上与 Gemini 2.5 Pro 差距在 2 分以内），则拒绝开源 7B 模型，并要求使用 32B+ 或闭源模型。
- 若在 7B 模型上处理 > 30 秒片段时动态帧率目标 > 8，则因延迟问题拒绝，并建议降低上限。

输出要求：一页纸的帧规划方案，包含采样器、池化策略、时序编码、输出格式、基准测试目标及上下文估算。末尾附上 arXiv 2502.13923（Qwen2.5-VL）与 2306.02858（Video-LLaMA）供对比阅读。