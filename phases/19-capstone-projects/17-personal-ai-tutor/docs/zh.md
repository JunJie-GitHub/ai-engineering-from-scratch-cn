# 综合项目 17 — 个人 AI 导师（自适应、多模态、带记忆）

> Khanmigo（可汗学院）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 以及 Synthesis Tutor 均在 2026 年大规模推出了自适应多模态辅导产品。它们的共同架构包括：苏格拉底式策略（Socratic policy，绝不直接给出答案）、每次交互后更新的学习者模型（learner model，采用贝叶斯知识追踪 Bayesian knowledge tracing 风格）、语音+文本+拍照解题输入、课程图谱检索（curriculum graph retrieval）、间隔重复调度（spaced-repetition scheduling），以及针对适龄内容的严格安全过滤器（safety filters）。本综合项目的目标是交付一个特定学科的辅导系统（K-12 代数或 Python 入门），与 10 名学习者开展为期两周的有效性研究（efficacy study），并通过内容安全审计。

**类型：** 综合项目
**语言：** Python（后端、学习者模型）、TypeScript（Web 应用）、SQL（通过 Postgres + Neo4j 构建课程图谱）
**前置要求：** 第 5 阶段（自然语言处理 NLP）、第 6 阶段（语音）、第 11 阶段（大语言模型工程 LLM engineering）、第 12 阶段（多模态）、第 14 阶段（智能体 agents）、第 17 阶段（基础设施）、第 18 阶段（安全）
**涉及阶段：** P5 · P6 · P11 · P12 · P14 · P17 · P18
**耗时：** 30 小时

## 问题背景

自适应辅导（adaptive tutoring）曾是教育科技领域的研究细分方向，但到 2026 年已演变为成熟的消费级产品。Khanmigo 已部署于美国大多数学区。Duolingo Max 的月活跃用户（MAU）突破数千万。Google 的 LearnLM / Gemini for Education 为 Google Classroom 中的辅导功能提供底层支持。Quizlet Q-Chat 与闪卡功能并列使用。Synthesis Tutor 则凭借“面向好奇儿童的导师”定位迅速走红。它们的共同要素包括：多模态输入（multimodal input，支持打字、语音、拍摄数学公式）、苏格拉底式教学法（Socratic pedagogy，先提问后讲解）、每次交互后实时更新的学习者模型，以及严格的适龄安全机制。

你将为特定学习群体构建其中一款产品。衡量标准是一项真实的有效性研究：在两周内对 10 名学习者进行前测与后测评分对比。语音交互循环（voice loop）必须自然流畅（参考综合项目 03 子技术栈）。记忆模块必须严格遵循隐私保护原则。安全过滤器必须通过面向 K-12 场景、符合《儿童在线隐私保护法》（COPPA）规范的红队测试（red-team testing）。

## 核心概念

包含四个核心组件。**导师策略（Tutor Policy）**采用苏格拉底式循环（Socratic Loop）：当学习者直接索要答案时，该策略会提出引导性问题；当学习者回答正确时，则推进至下一个概念；当学习者卡壳时，则提供脚手架式提示（Scaffolded Hint）。**学习者模型（Learner Model）**基于贝叶斯知识追踪（Bayesian Knowledge Tracing, BKT）或其简化变体，在每次交互后更新每个课程节点（Curriculum Node）的掌握概率（Mastery Probability）。**课程图谱（Curriculum Graph）**是一个基于 Neo4j 构建的概念网络，包含前置依赖边（Prerequisite Edges）；策略模块通过遍历该图谱来选择下一个学习概念。**记忆模块（Memory）**采用类 AgentMemory 架构的情景与语义混合存储（Episodic + Semantic Store），用于保存历史交互记录、常见错误及用户偏好。

用户体验（User Experience, UX）为多模态（Multimodal）设计。支持文本输入用于键入答案；通过 LiveKit + Whisper 实现语音输入（复用综合项目 03 的架构）；通过 dots.ocr 或 PaliGemma 2 实现数学题拍照输入；通过 Cartesia Sonic-2 实现语音输出。安全机制采用 Llama Guard 4 结合适龄过滤器（Age-Appropriate Filter，屏蔽成人内容、暴力及自残信息），并实施符合《儿童在线隐私保护法》（COPPA）的记忆留存策略（Memory Retention Policy）。

有效性研究（Efficacy Study）是核心交付物。招募 10 名学习者，进行为期两周的前测（Pre-test）与后测（Post-test）。报告需包含学习增益差值（Learning Gain Delta）及置信区间（Confidence Interval）。同时需与非自适应基线（Non-Adaptive Baseline，即相同内容以线性方式交付，不启用导师策略）进行对比。

## 架构

learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope

## 技术栈

- 学科选择：K-12 代数或 Python 入门（择一深入）
- 导师策略：基于 Claude Sonnet 4.7 的 LangGraph（启用提示词缓存 Prompt Caching）
- 学习者模型：经典贝叶斯知识追踪（BKT）或用于间隔重复调度（Spaced-Repetition Scheduler）的 FSRS
- 课程图谱：基于 Neo4j 的概念图谱，包含前置依赖边与开放教育资源（Open Educational Resources, OER）内容
- 记忆模块：类 AgentMemory 架构的持久化向量存储，结合情景与语义存储
- 语音交互：LiveKit Agents 1.0 + Cartesia Sonic-2（复用综合项目 03 子技术栈）
- 拍照解题：使用 dots.ocr 或 PaliGemma 2 进行公式识别
- 安全机制：Llama Guard 4 + 自定义适龄过滤器
- 评估体系：基于布鲁姆分类法（Bloom's Taxonomy）的问题生成、前/后测测试框架（Test Harness）及有效性研究工具链

## 开始构建

1. **课程图谱（Curriculum Graph）。** 构建一个包含 50-150 个概念节点的 Neo4j 图数据库（例如，K-12 代数从“数轴”到“求根公式”），并设置先修关系边。为每个节点附加开放教育资源（Open Educational Resources, OER）内容（如 Open Textbook、OpenStax）。

2. **学习者模型（Learner Model）。** 使用先验参数初始化贝叶斯知识追踪（Bayesian Knowledge Tracing, BKT）：猜测概率（guess）、失误概率（slip）和学习率（learn-rate）。在每次交互后更新各概念的掌握度，并按学习者进行持久化存储。

3. **辅导策略（Tutor Policy）。** 基于 LangGraph 构建，包含以下节点：`read_signal`（判断学习者的回答是正确/部分正确/卡壳？）、`select_concept`（遍历课程图谱并选择优先级最高的概念）、`scaffold`（苏格拉底式提示/Socratic Prompt）以及 `update_mastery`。

4. **记忆模块（Memory）。** 每次交互均写入情景记忆存储（Episodic Store）。错误记录与偏好设置将晋升至语义记忆（Semantic Memory）。采用符合《儿童在线隐私保护法》（Children's Online Privacy Protection Act, COPPA）的数据留存策略：数据 1 年后自动删除，且支持家长访问。

5. **语音交互路径（Voice Path）。** 将 LiveKit Agents worker 接入辅导策略。通过 Whisper-v3-turbo 实现自动语音识别（Automatic Speech Recognition, ASR），通过 Cartesia Sonic-2 实现文本转语音（Text-to-Speech, TTS）。支持打断功能（Barge-in，复用 capstone 03 的交互机制）。

6. **拍照解题路径（Photo-Math Path）。** 上传或拍摄图片；运行 dots.ocr 或 PaliGemma 2 识别数学公式；将其作为结构化输入传递给辅导系统。

7. **安全机制（Safety）。** 所有模型输出均需通过 Llama Guard 4 及适龄内容过滤器（拦截自残、成人内容及暴力内容）。记忆访问权限按学习者 ID 隔离；提供家长端数据删除入口。

8. **有效性研究（Efficacy Study）。** 招募 10 名学习者，进行前测（标准化 30 题基线测试），开展为期两周的辅导交互（每周 3 次），随后进行后测。与使用相同内容但采用非自适应基线策略的 10 人对照组进行效果对比。

9. **每周进度报告（Weekly Progress Reports）。** 为每位学习者自动生成 PDF 摘要，涵盖已探索主题、掌握度变化轨迹及推荐的后续学习步骤。

## 使用示例

learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"

## 交付物

`outputs/skill-ai-tutor.md` 为最终交付物。这是一个具备多模态输入、学习者模型、记忆模块、安全机制及可量化有效性的学科专属自适应辅导系统。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 学习增益差值（Learning Gain Delta） | 10 人两周研究中的前后测分数差值 |
| 20 | 苏格拉底式教学保真度（Socratic Fidelity） | 对话转录样本的评分量表得分 |
| 20 | 多模态用户体验（Multimodal UX） | 端到端的语音、图像与文本交互连贯性 |
| 20 | 安全与隐私合规性（Safety & Privacy Posture） | Llama Guard 4 通过率 + 符合 COPPA 的数据留存策略 |
| 15 | 课程广度与图谱质量（Curriculum Breadth & Graph Quality） | 概念覆盖率 + 先修关系图谱一致性 |
| **100** | | |

## 练习

1. 分别在使用和不使用自适应学习者模型（adaptive learner model，采用随机概念顺序）的情况下开展有效性研究（efficacy study）。报告性能差异（delta）。预期自适应模型表现更优，但具体的提升幅度才是值得关注的关键数据。

2. 增加多模态探针（multimodal probe）：将同一概念问题分别以文本、语音和图片形式呈现。测量学习者在使用其偏好的模态时，知识掌握收敛（converge）速度是否更快。

3. 构建家长仪表盘（parent dashboard）：展示已练习主题、掌握轨迹（mastery trajectories）、即将学习的概念以及安全事件（任何安全护栏触发（guardrail hits）记录）。设计需符合《儿童在线隐私保护法》（COPPA）标准。

4. 增加语言切换模式（language-switch mode）：辅导系统支持接收西班牙语输入并使用西班牙语进行教学。测量 X-Guard 的防护覆盖率（coverage）。

5. 对记忆隐私（memory privacy）进行压力测试：验证学习者 A 无法通过语音片段重新注入攻击（voice-clip re-ingest attack）获取学习者 B 的数据。记录所有尝试访问的行为并触发安全告警。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 苏格拉底式策略（Socratic policy） | “提问，而非灌输” | 辅导系统提出引导性问题，而非直接给出答案 |
| 贝叶斯知识追踪（Bayesian knowledge tracing） | “BKT” | 用于计算每个概念掌握概率的经典学习者模型方程 |
| FSRS | “自由间隔重复调度器” | 2024 年推出的间隔重复调度算法，性能优于 SM-2 |
| 课程图谱（Curriculum graph） | “概念有向无环图（DAG）” | 使用 Neo4j 构建的概念图，包含前置依赖关系边 |
| 情景记忆（Episodic memory） | “单次交互日志” | 存储每次交互记录，供后续检索使用 |
| 语义记忆（Semantic memory） | “学习模式存储库” | 从情景记忆中提炼并压缩的错误记录与偏好数据 |
| COPPA | “儿童隐私法” | 美国限制收集 13 岁以下儿童数据的法律 |

## 延伸阅读

- [Khanmigo (可汗学院)](https://www.khanmigo.ai) — 面向消费者的 K-12 辅导系统参考案例
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 语言学习辅导系统参考案例
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — 托管参考模型
- [Quizlet Q-Chat](https://quizlet.com) — 替代参考案例
- [Synthesis Tutor](https://www.synthesis.com) — 初创公司参考案例
- [FSRS 算法](https://github.com/open-spaced-repetition/fsrs4anki) — 间隔重复调度器
- [贝叶斯知识追踪](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — 学习者模型经典算法
- [LiveKit Agents](https://github.com/livekit/agents) — 语音技术栈