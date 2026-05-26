# 综合项目 12 — 视频理解流水线（Video Understanding Pipeline）（场景、问答、搜索）

> Twelve Labs 已将 Marengo 与 Pegasus 产品化。VideoDB 推出了面向视频的 CRUD API。AI2 的 Molmo 2 发布了开源的视觉语言模型（Vision-Language Model, VLM）检查点。Gemini 的长上下文能力已原生支持处理数小时的视频。TimeLens-100K 确立了大规模时间定位（temporal grounding）的标准。2026 年的处理流水线已定型：场景分割（scene segmentation）、逐场景字幕生成与嵌入（embedding）、转录文本对齐（transcript alignment）、多向量索引（multi-vector index），以及能够返回（起始, 结束）时间戳并附带帧预览的查询机制。本综合项目的目标是摄入 100 小时视频数据，在公开基准测试中取得成绩，并量化模型在计数与动作类问题上的幻觉（hallucination）程度。

**Type:** 综合项目
**Languages:** Python（流水线）、TypeScript（用户界面）
**Prerequisites:** 第 4 阶段（计算机视觉，Computer Vision）、第 6 阶段（语音处理）、第 7 阶段（Transformer 架构）、第 11 阶段（大语言模型工程，LLM Engineering）、第 12 阶段（多模态，Multimodal）、第 17 阶段（基础设施）
**Phases exercised:** P4 · P6 · P7 · P11 · P12 · P17
**Time:** 30 小时

## 问题描述

在 2026 年的技术规模下，长视频问答（Video QA）是极其消耗带宽与计算资源的多模态任务。尽管 Gemini 2.5 Pro 已能原生解析长达 2 小时的视频，但要将 100 小时的视频数据转化为可查询语料库，仍需依赖场景级索引。生产级架构通常结合以下组件：场景分割（TransNetV2 或 PySceneDetect）、基于视觉语言模型（VLM）的逐场景字幕生成（如 Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、带词级时间戳的转录文本对齐（Whisper-v3-turbo），以及将字幕、帧嵌入（frame embedding）和转录文本并列存储的多向量索引。查询流水线最终返回（起始, 结束）时间戳及对应的帧预览。

评估基准采用公开数据集（ActivityNet-QA、NeXT-GQA）以及自定义的 100 条查询测试集。模型在计数类与动作类问题上极易产生幻觉，这是公认的难点；本项目将对此进行专项量化评估。

## 核心概念

在数据摄入阶段，三条流水线并行运行。**场景分割**负责将视频切分为独立场景。**VLM 字幕生成**为每个场景生成描述性字幕，并从关键帧中提取帧嵌入向量。**自动语音识别（Automatic Speech Recognition, ASR）对齐**生成词级时间戳。这三条数据流通过（场景 ID, 时间范围）进行关联。每个场景在多向量索引（Qdrant）中存储三种向量类型：字幕嵌入、关键帧嵌入和转录文本嵌入。

在查询阶段，自然语言问题会同时检索这三种向量；检索结果通过倒数排名融合（Reciprocal Rank Fusion, RRF）进行合并；随后，时间定位适配器（temporal-grounding adapter，TimeLens 风格）会在排名最高的场景中进一步精细化（起始, 结束）时间窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）接收查询语句、排名靠前的场景数据及裁剪后的帧图像，最终输出附带引用时间戳和帧预览的答案。

幻觉量化评估至关重要。计数类（如“有多少人进入房间？”）与动作时序类（如“厨师是先倒水还是先搅拌？”）问题的回答历来可靠性较低。需将此类问题的准确率与描述性问题的准确率分开报告。

## 架构设计

video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations

## 技术栈

- 场景分割（Scene Segmentation）：TransNetV2（2024-26 年最先进模型）或 PySceneDetect
- 自动语音识别（ASR）：通过 faster-whisper 运行 Whisper-v3-turbo，输出词级时间戳
- 视觉语言模型（VLM）描述生成与问答：Gemini 2.5 Pro、Qwen3-VL-Max 或 Molmo 2
- 时序定位（Temporal Grounding）：基于 TimeLens-100K 训练的适配器或 VideoITG
- 索引（Index）：支持多向量（Multi-vector）的 Qdrant（描述向量 / 帧向量 / 转录文本向量）
- 用户界面（UI）：Next.js 15，集成 HTML5 视频播放器与场景缩略图
- 评估（Evaluation）：ActivityNet-QA、NeXT-GQA 以及包含 100 道题的自定义人工标注数据集
- 幻觉基准测试（Hallucination Benchmark）：包含人工标注的计数类与动作类子集

## 构建步骤

1. **数据摄取脚本（Ingest Walker）。** 接受 YouTube URL 或本地 MP4 文件。必要时将视频降采样至 720p。持久化存储 `{video_id, file_path}`。

2. **场景分割（Scene Segmentation）。** 运行 TransNetV2 或 PySceneDetect 生成 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标处理 100 小时视频：约产生 6k-8k 个场景。

3. **语音识别处理（ASR Pass）。** 对音频运行 Whisper-v3-turbo；导出词级时间戳；按场景切分转录文本片段。

4. **VLM 图像描述生成（VLM Captioning）。** 针对每个场景，将关键帧与简短描述模板输入 Gemini 2.5 Pro（或 Qwen3-VL-Max）。生成描述文本与帧嵌入向量。

5. **多向量索引（Multi-vector Index）。** 创建包含三个命名向量的 Qdrant 集合。负载数据（Payload）：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询（Query）。** 自然语言问题触发三个密集向量查询（Dense Queries）；使用倒数排名融合（Reciprocal Rank Fusion, RRF）进行合并；返回前 k=5 个场景。

7. **时序定位（Temporal Grounding）。** 在排名最高的场景上运行 TimeLens 风格的适配器，以细化该场景内的 (start, end) 时间窗口。

8. **VLM 答案合成（VLM Synthesis）。** 将查询问题、排名前 3 的场景片段（以图像或短视频形式）及转录文本输入 Gemini 2.5 Pro。要求输出包含 `(video_id, start_ms, end_ms)` 引用来源。

9. **评估（Evaluation）。** 运行 ActivityNet-QA 与 NeXT-GQA 测试集。构建包含 100 个查询的自定义数据集。报告整体准确率及各类别细分结果（计数类、动作类、描述类）。

## 使用方法

$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]

## 交付上线

`outputs/skill-video-qa.md` 是最终交付物。给定一个 YouTube URL 或上传的视频，该流水线 (pipeline) 会对场景进行索引，并附带带时间戳的引用 (timestamped citations) 来回答问题。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 时间定位交并比 (Temporal grounding IoU) | 在预留定位数据集 (held-out grounding set) 上计算交并比 (Intersection-over-union) |
| 20 | 问答准确率 (QA accuracy) | NeXT-GQA 数据集与自定义的 100 条查询测试集 |
| 20 | 数据摄入吞吐量 (Ingest throughput) | 每美元成本可处理的视频时长（小时） |
| 20 | 界面与引用体验 (UI and citation UX) | 时间戳链接、缩略图条带、跳转至指定帧 |
| 15 | 幻觉率 (Hallucination rate) | 分别统计计数类与动作类问题的准确率 |
| **100** | | |

## 练习

1. 在字幕生成 (captioning) 阶段，将 Gemini 2.5 Pro 替换为 Qwen3-VL-Max。在人工评分的 50 个场景样本上报告字幕质量差异 (delta)。
2. 将每个场景的帧嵌入 (frame embedding) 从多向量缩减为单个池化向量 (pooled vector)。评估检索性能的回退 (regression) 幅度。
3. 构建“严格计数”模式：合成器 (synthesizer) 提取每个计数实例及其时间戳，供用户点击验证。评估用户验证是否能降低幻觉 (hallucination)。
4. 基准测试数据摄入成本：对比三种视觉语言模型 (VLM) 选项的“每美元视频处理时长”。选择最佳平衡点 (sweet spot)。
5. 添加说话人分离 (speaker diarization) 转录文本：在音频上运行 `pyannote` 说话人分离，并对每位说话人的转录文本进行嵌入 (embed)。演示“关于 X，Alice 说了什么？”这类查询。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 场景分割 (Scene segmentation) | “镜头检测” | 在镜头边界处将视频切分为场景 |
| 多向量索引 (Multi-vector index) | “字幕 + 帧 + 转录文本” | Qdrant 集合，为每种表示形式分配命名向量 |
| 时间定位 (Temporal grounding) | “具体发生在什么时候” | 为查询答案细化（起始，结束）时间窗口 |
| 帧嵌入 (Frame embedding) | “视觉表示” | 关键帧的向量嵌入；用于场景视觉相似度匹配 |
| RRF 融合 (RRF fusion) | “倒数排名融合” | 跨多个排序列表的合并策略；一种经典的混合检索 (hybrid-retrieval) 技巧 |
| 计数幻觉 (Counting hallucination) | “数错” | 视觉语言模型 (VLM) 在回答“有多少个 X”问题时的已知失效模式 |
| ActivityNet-QA | “视频问答基准” | 长视频问答准确率基准测试 |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开源视觉语言模型（VLM）检查点
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — 大规模时序定位（Temporal Grounding）
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — 云端托管参考实现
- [VideoDB](https://videodb.io) — 视频增删改查（CRUD）API 参考文档
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商业化参考方案
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割（Scene Segmentation）模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典开源替代方案
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 参考评估基准（Evaluation Benchmark）