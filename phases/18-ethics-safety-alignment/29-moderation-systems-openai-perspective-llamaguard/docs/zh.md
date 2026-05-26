# 内容审核系统（Moderation Systems）— OpenAI, Perspective, Llama Guard

> 生产环境中的内容审核系统（Moderation Systems）将第12-16课中定义的安全策略工程化落地（Operationalize）。OpenAI 内容审核 API（Moderation API）：基于 GPT-4o 构建的 `omni-moderation-latest`（2024）可在单次调用中同时分类文本与图像；在多语言测试集上的表现较上一版本提升 42%；其响应模式（Response Schema）返回 13 个类别的布尔值——骚扰（harassment）、骚扰/威胁（harassment/threatening）、仇恨（hate）、仇恨/威胁（hate/threatening）、非法（illicit）、非法/暴力（illicit/violent）、自残（self-harm）、自残/意图（self-harm/intent）、自残/指导（self-harm/instructions）、色情（sexual）、色情/未成年人（sexual/minors）、暴力（violence）、暴力/血腥（violence/graphic）；对大多数开发者免费。分层模式（Layered Patterns）：输入审核（Input Moderation，生成前）、输出审核（Output Moderation，生成后）、自定义审核（Custom Moderation，领域规则）。异步并行调用（Async Parallel Calls）可隐藏延迟；触发标记时返回占位响应（Placeholder Responses）。Llama Guard 3/4（第16课）：涵盖 14 项 MLCommons 危害类别、代码解释器滥用（Code Interpreter Abuse），支持 8 种语言（v3）及多图像输入（v4）。Perspective API（Google Jigsaw）：早于“大语言模型作为审核员（LLM-as-Moderator）”浪潮的毒性评分（Toxicity Scoring）工具；主要提供单一维度的毒性评分，并包含严重毒性（severe-toxicity）、侮辱（insult）、亵渎（profanity）等变体；是内容审核研究领域的基线（Baseline）工具。弃用说明：Azure Content Moderator 已于 2024 年 2 月弃用，将于 2027 年 2 月正式停用，由 Azure AI Content Safety 取代。

**Type:** 构建（Build）
**Languages:** Python（标准库，三层审核框架）
**Prerequisites:** 第18阶段 · 第16课（Llama Guard / Garak / PyRIT）
**Time:** 约 60 分钟

## 学习目标

- 描述 OpenAI 内容审核 API 的类别分类体系，并说明其与 Llama Guard 3 的 MLCommons 类别集有何不同。
- 描述三层审核模式（输入、输出、自定义），并列举每种模式的一种失效模式（Failure Mode）。
- 说明 Perspective API 作为大语言模型（LLM）时代之前基线工具的地位，以及它为何仍在研究中被广泛使用。
- 陈述 Azure 相关服务的弃用时间线。

## 问题背景

第12-16课介绍了攻击手段与防御工具。第29课则聚焦于已部署的内容审核系统，这些系统在用户交互的产品表层将防御策略落地执行。三层审核模式是 2026 年的默认配置。

## 核心概念

### OpenAI 审核 API (OpenAI Moderation API)

`omni-moderation-latest`（2024 年）。基于 GPT-4o 构建。支持单次调用同时分类文本与图像。对大多数开发者免费。

分类类别（响应模式 (response schema) 中包含 13 个布尔值）：
- harassment, harassment/threatening
- hate, hate/threatening
- self-harm, self-harm/intent, self-harm/instructions
- sexual, sexual/minors
- violence, violence/graphic
- illicit, illicit/violent

多模态 (multimodal) 支持适用于 `violence`、`self-harm` 和 `sexual`，但不包括 `sexual/minors`；其余类别仅支持文本。

出于教学简化的目的，在 `code/main.py` 的代码脚手架 (code harness) 中，我们将 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 等子类别合并至其顶层父类别。生产环境代码应使用完整的 13 类别响应模式。

在多语言测试集上的表现比上一代审核端点提升 42%。提供各类别独立评分；应用程序需自行设定阈值。

### Llama Guard 3/4

详见第 16 课。包含 14 个 MLCommons 危害类别（分类方式与 OpenAI 的 13 个响应模式布尔值不同）。支持 8 种语言（v3 版本）。Llama Guard 4（2025 年 4 月发布）原生支持多模态，参数量为 12B。

OpenAI 与 Llama Guard 的分类体系 (taxonomy) 既有重叠又存在差异。OpenAI 将“非法内容（illicit）”作为一个宽泛类别；而 Llama Guard 则将“暴力犯罪”与“非暴力犯罪”分开。实际部署时，需根据策略与分类体系的匹配度进行选择。

### Perspective API (Google Jigsaw)

早于“大语言模型作为审核器 (LLM-as-moderator)”浪潮的毒性评分系统 (toxicity scoring system)（2020 年之前）。类别包括：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。提供单一维度的主评分（TOXICITY）及多个子维度变体。

由于该 API 稳定、文档完善且拥有多年的校准数据，被广泛用作内容审核研究的基线 (baseline)。对于现代大语言模型相关用例，通常 Llama Guard 或 OpenAI Moderation 是更合适的选择。

### 三层架构模式 (Three-layer pattern)

1. **输入审核 (Input moderation)。** 在生成前对用户提示词进行分类。若被标记则拒绝。延迟：一次分类器调用。
2. **输出审核 (Output moderation)。** 在交付前对模型输出进行分类。若被标记则替换为拒绝响应。延迟：生成后的一次分类器调用。
3. **自定义审核 (Custom moderation)。** 领域特定规则（正则表达式、白名单、业务策略）。可在输入或输出阶段运行。

这三层在设计上是顺序执行的：输入审核必须在生成前完成，输出审核则在生成后运行。并行性适用于同一层内——对同一段文本并发运行多个分类器（例如 OpenAI Moderation + Llama Guard + Perspective）可掩盖单个分类器的延迟。作为一种可选优化，在输入审核完成且首个 token 流式传输延迟期间，可显示占位符响应（“请稍候，正在检查……”）。标记行为可配置：拒绝、净化或升级至人工审核。

### 故障模式 (Failure modes)

- **仅输入审核。** 无法捕获输出幻觉 (hallucinations)（第 12-14 课中的编码攻击 (encoding attacks) 可绕过输入分类器）。
- **仅输出审核。** 允许任何输入直达模型；增加成本；并向攻击者暴露内部推理过程。
- **仅自定义审核。** 跨类别鲁棒性不足；正则表达式较为脆弱。

默认采用分层架构。双重保险（Belt-and-suspenders）。

### Azure 弃用公告 (Azure deprecation)

Azure Content Moderator：2024 年 2 月弃用，2027 年 2 月正式停用。由基于大语言模型且与 Azure OpenAI 集成的 Azure AI Content Safety 取代。对于 Azure 部署而言，迁移工作是一项贯穿 2024 至 2027 年的现场级项目。

### 在第 18 阶段中的定位 (Where this fits in Phase 18)

第 16 课在红队测试 (red-team) 背景下介绍审核工具。第 29 课涵盖已部署的审核系统。第 30 课以当前双重用途能力 (dual-use capability) 的证据作为收尾。

## Use It

`code/main.py` 构建了一个三层内容审核（Content Moderation）框架：输入审核器（Input Moderator，结合关键词与类别评分）、输出审核器（Output Moderator，对模型输出应用相同的分类器）以及自定义审核器（Custom Moderator，基于领域规则）。你可以将输入传入该框架，观察具体是哪一层拦截了哪些内容。

## Ship It

本课时将生成 `outputs/skill-moderation-stack.md` 文件。针对特定的部署场景，它会推荐一套审核栈（Moderation Stack）配置方案：包括输入端使用何种分类器（Classifier）、输出端使用何种分类器、采用哪些自定义规则，以及针对边界情况（Edge Cases）使用何种裁判模型（Judge）。

## Exercises

1. 运行 `code/main.py`。将一条良性（Benign）、临界（Borderline）和有害（Harmful）的输入依次通过全部三层审核。报告每种输入分别触发了哪一层。
2. 在框架中针对特定类别扩展类似 Perspective API 的毒性评分（Toxicity Scoring）功能。将其阈值表现与类别评分进行对比。
3. 阅读 OpenAI Moderation API 文档和 Llama Guard 3 的类别列表。将每个 OpenAI 类别映射到最接近的 Llama Guard 类别。找出三个无法清晰映射的类别。
4. 为代码助手部署场景（例如 GitHub Copilot）设计一套审核栈。识别出最相关和最不相关的类别，并提出自定义规则建议。
5. Azure Content Moderator 将于 2027 年 2 月正式停用。请规划向 Azure AI Content Safety 的迁移方案。指出迁移过程中风险最高的环节。

## Key Terms

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| OpenAI Moderation | "omni-moderation-latest" | 基于 GPT-4o 的 13 类别（文本）分类器，具备部分多模态（Multimodal）支持能力 |
| Perspective API | "Google Jigsaw toxicity" | 大语言模型（LLM）时代之前的毒性评分基准 |
| Llama Guard | "MLCommons 14-category" | Meta 推出的风险分类器（v3：8B 文本模型，支持 8 种语言；v4：12B 多模态模型） |
| 输入审核（Input Moderation） | "pre-generation filter" | 在调用模型前对用户提示词（Prompt）进行分类审核 |
| 输出审核（Output Moderation） | "post-generation filter" | 在交付结果前对模型输出进行分类审核 |
| 自定义审核（Custom Moderation） | "domain rules" | 针对特定部署场景的规则（正则表达式 Regex、白名单 Allowlist、策略 Policy 等） |
| 分层审核（Layered Moderation） | "all three layers" | 标准的生产环境部署模式 |

## Further Reading

- [OpenAI Moderation API 文档](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation 端点
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard 代码仓库
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — 毒性评分工具
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure 替代方案