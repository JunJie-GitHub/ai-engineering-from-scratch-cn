---
name: ai-tutor
description: 交付一款针对特定学科的自适应多模态个人辅导系统，集成贝叶斯知识追踪（Bayesian Knowledge Tracing）、课程图谱（Curriculum Graph）、安全过滤器，并开展为期两周的量化效能研究。
version: 1.0.0
phase: 19
lesson: 17
tags: [综合项目, 辅导系统, 自适应, bkt, fsrs, livekit, 多模态, coppa]
---

给定一门学科（K-12 代数或 Python 入门），构建一款支持文本、语音及拍照解题输入的个人辅导系统。该系统需集成贝叶斯知识追踪（Bayesian Knowledge Tracing, BKT）学习者模型、基于课程图谱（Curriculum Graph）驱动的概念选择机制、符合《儿童在线隐私保护法》（COPPA）的数据留存策略以及安全过滤器。随后，招募 10 名学习者开展为期两周的效能研究。

构建计划：

1. 基于 Neo4j 构建课程图谱：包含 50-150 个概念节点，节点间通过先修关系边连接，并关联开放教育资源（Open Educational Resources, OER）内容（如 OpenStax、Open Textbook）。
2. 学习者模型：采用贝叶斯知识追踪算法，为每个概念设置猜测（guess）、失误（slip）和学习率（learn-rate）的先验概率；为每位学习者持久化保存状态。
3. 辅导策略（基于 LangGraph 与 Claude Sonnet 4.7，结合提示词缓存）：read_signal -> select_concept（图谱遍历） -> scaffold（苏格拉底式引导） -> update_mastery。
4. 记忆模块：采用类似 agentmemory 的持久化情景记忆与语义记忆存储；符合 COPPA 要求，数据满 1 年自动删除；提供家长可操作的删除接口。
5. 语音交互：使用 LiveKit Agents 工作节点，集成 Whisper-v3-turbo 自动语音识别（Automatic Speech Recognition, ASR）与 Cartesia Sonic-2 文本转语音（Text-to-Speech, TTS）；复用综合项目 03 的流水线。
6. 拍照解题：采用 dots.ocr 或 PaliGemma 2 进行公式识别；将结构化输入传递给辅导系统。
7. 安全防护：输入与输出均接入 Llama Guard 4；部署适龄过滤器，拦截自残、成人内容及暴力信息；实现基于学习者作用域的记忆隔离。
8. 为每位学习者生成每周 PDF 进度报告。
9. 效能研究：招募 10 名学习者，进行前测（标准化 30 题基线测试），开展为期 2 周的辅导会话（每周 3 次），随后进行后测；与非自适应线性对照组（Non-adaptive Linear Cohort）进行对比。

评估量规：

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 学习增益差值（Learning Gain Delta） | 10 名学习者为期两周研究的前后测差值 |
| 20 | 苏格拉底式引导保真度（Socratic Fidelity） | 对话转录样本的评分表得分 |
| 20 | 多模态用户体验（Multimodal UX） | 语音、拍照与文本交互的端到端连贯性 |
| 20 | 安全与隐私合规状态（Safety + Privacy Posture） | Llama Guard 4 通过率 + 符合 COPPA 的数据留存策略 + 跨学习者隔离机制 |
| 15 | 课程广度与图谱质量（Curriculum Breadth and Graph Quality） | 概念覆盖率 + 先修图谱一致性 |

硬性否决项：

- 辅导策略直接“倾倒答案”而非提出下一个引导性问题。苏格拉底式引导为硬性要求。
- 学习者模型未随每次交互进行更新。贝叶斯知识追踪（BKT）为最低标准。
- 记忆模块未遵循符合 COPPA 的数据留存策略。对于 K-12 受众而言不可接受。
- 效能声明未设置非自适应基线对照组。

拒绝执行规则：

- 若输入与输出未同时接入 Llama Guard 4，则拒绝部署。
- 若未提供家长可访问的数据删除界面，则拒绝持久化存储学习者数据。
- 若未同步运行非自适应基线对照组，则拒绝宣称系统具备“自适应”能力。

交付物：一个代码仓库，需包含课程图谱、BKT 学习者模型、LangGraph 辅导策略、多模态输入处理器、LiveKit 语音流水线、安全流水线、家长控制面板、效能研究运行器、前后测测试框架，以及一份详细记录学习增益差值（含置信区间）并与线性基线进行对比的说明文档。