---
name: video-qa
description: 构建一个包含场景分割、多向量索引、时间定位和时间戳引用的视频理解流水线。
version: 1.0.0
phase: 19
lesson: 12
tags: [综合项目, 视频, 多模态, gemini, qwen-vl, molmo, transnet, qdrant]
---

给定 100 小时的视频，构建一个数据摄入流水线（ingestion pipeline）与查询系统，使其能够回答自然语言问题，并附带（起始时间，结束时间）时间戳及关键帧预览。

构建计划：

1. 摄入视频（YouTube 链接或 MP4 文件）；必要时降采样至 720p。
2. 使用 TransNetV2 或 PySceneDetect 进行场景分割（scene segmentation）；输出 `[{scene_id, start_ms, end_ms, keyframe_path}]`。
3. 使用 Whisper-v3-turbo（faster-whisper）进行自动语音识别（ASR），生成词级时间戳；按场景进行切片。
4. 使用 Gemini 2.5 Pro、Qwen3-VL-Max 或 Molmo 2 进行视觉语言模型（VLM）图像描述生成；输出描述文本与帧嵌入向量（frame embedding）。
5. 在 Qdrant 中构建多向量索引（multi-vector index），每个场景包含三个命名向量（caption_emb、frame_emb、transcript_emb）及负载数据 {video_id, scene_id, start_ms, end_ms, keyframe_url}。
6. 查询：执行三个并行稠密查询（dense queries）；使用倒数排名融合（reciprocal rank fusion）进行合并；返回 top-k=5 个场景。
7. 时间定位（temporal grounding）（使用 TimeLens 适配器或 VideoITG）在排名最高的场景中精调（起始时间，结束时间）。
8. 视觉语言模型合成（VLM synthesis）（使用 Gemini 2.5 Pro），输入为查询语句 + 前 3 个场景片段 + 转录文本；要求提供 `(video_id, start_ms, end_ms)` 引用。
9. 在 ActivityNet-QA、NeXT-GQA 以及包含 100 个查询的人工标注自定义数据集上进行评估（evaluation）。报告整体准确率及按问题类别（描述性、计数、动作类型）划分的准确率。

评估标准：

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 时间定位交并比（IoU） | 在预留定位数据集上的 IoU |
| 20 | 问答准确率 | NeXT-GQA 与 100 查询自定义数据集 |
| 20 | 摄入吞吐量 | 每美元索引的视频小时数 |
| 20 | 用户界面与引用体验 | 时间戳链接、缩略图条、跳转至指定帧 |
| 15 | 幻觉率（hallucination rate） | 单独报告计数与动作类型的准确率 |

硬性否决条件：

- 每个场景仅池化（pool）单个向量的流水线。必须使用多向量以体现类别差异。
- 未提供（起始时间，结束时间）引用的答案。
- 仅报告整体准确率，未提供计数/动作子集的细分数据。
- 视觉语言模型合成未直接接收场景帧（仅文本输入会丢失视觉定位信息）。

拒绝服务规则：

- 拒绝提供版权来源不明的视频服务；要求每个 video_id 必须附带许可证标签。
- 当摄入速率超过实测吞吐量时，拒绝宣称“实时”响应。
- 拒绝将计数/动作幻觉数据隐藏在整体准确率数值中。

输出：一个代码仓库，需包含场景分割 + 自动语音识别（ASR） + 图像描述生成流水线、多向量 Qdrant 集合、时间定位适配器、支持时间戳深度链接（deep-links）的 Next.js 15 查看器、三项基准测试的评估结果（ActivityNet-QA、NeXT-GQA、自定义数据集），以及一份说明文档，需列出你观察到的三类计数或动作类型失败案例，并说明为降低每类失败率所采取的检索或合成策略调整。