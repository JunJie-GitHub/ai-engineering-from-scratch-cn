# 浏览器代理（Browser Agents）与长程网页任务（Long-Horizon Web Tasks）

> ChatGPT 代理（2025 年 7 月）将 Operator 与深度研究（deep research）功能合并为单一的浏览器/终端代理，并在 BrowseComp 基准测试中创下 68.9% 的业界最优水平（SOTA）。OpenAI 于 2025 年 8 月 31 日关闭了 Operator——这标志着产品层面的整合。Anthropic 收购 Vercept 后，Claude Sonnet 在 OSWorld 上的表现从不足 15% 跃升至 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始 WebArena 中 11.3 个百分点的假阴性率（false-negative rate），并发布了包含 258 个任务的 Hard 子集。这些数据是真实的。攻击面（attack surface）同样真实存在：OpenAI 安全准备负责人公开表示，针对浏览器代理的间接提示词注入（indirect prompt injection）“并非一个可以完全修补的漏洞”。已记录的 2025–2026 年攻击事件包括：Tainted Memories（Atlas CSRF 攻击）、HashJack（Cato Networks 披露）以及 Perplexity Comet 中的一键劫持漏洞。

**类型:** 学习
**语言:** Python（标准库，间接提示词注入攻击面模型）
**前置要求:** 第 15 阶段 · 10（权限模式），第 15 阶段 · 01（长程代理）
**预计耗时:** 约 45 分钟

## 问题所在

浏览器代理是一种长程代理（long-horizon agent），它会读取不可信内容并执行具有实际影响的操作。代理访问的每一个页面都是用户未曾编写过的输入。每个页面上的每一个表单都可能成为潜在的控制通道。2025–2026 年的攻击语料库表明，这绝非假设：Tainted Memories 攻击允许攻击者通过精心构造的页面将恶意指令绑定到代理的记忆中；HashJack 攻击将指令隐藏在代理访问的 URL 片段中；而 Perplexity Comet 劫持仅需一次点击即可触发。

当前的防御形势不容乐观。OpenAI 安全准备负责人直言不讳地指出：间接提示词注入“并非一个可以完全修补的漏洞”。这是因为该攻击存在于代理的读取与执行边界（reading-vs-acting boundary）处，而这一边界在架构上是模糊的——原则上，模型读取的每一个词元（token）都有可能被解读为指令。

本节将明确界定攻击面，梳理基准测试格局（benchmark landscape）（BrowseComp、OSWorld、WebArena-Verified），并构建一个最小化的间接提示词注入场景模型，以便你在第 14 节和第 18 节中推演实际的防御策略。

## 核心概念

### 2026年技术格局：各系统简述

**ChatGPT 智能体（OpenAI）**。于2025年7月发布。整合了 Operator（网页浏览）与 Deep Research（多小时深度研究）功能。独立版 Operator 已于2025年8月31日下线。在 BrowseComp 基准测试中达到 68.9% 的当前最优水平 (State-of-the-Art)；在 OSWorld 和 WebArena-Verified 上表现强劲。

**Claude Sonnet + Vercept（Anthropic）**。Anthropic 收购 Vercept 旨在强化计算机操作 (Computer-Use) 能力。该举措使 Claude Sonnet 在 OSWorld 上的得分从不足 15% 跃升至 72.5%。Claude Computer Use 以工具 API (Tool API) 的形式发布。

**Gemini 3 Pro with Browser Use（DeepMind）**。Browser Use 集成提供了计算机操作控制功能；FSF v3（2026年4月，第20课）专门追踪机器学习研发 (ML R&D) 领域的自主性水平。

**WebArena-Verified（ServiceNow, ICLR 2026）**。修复了一个已有充分记录的问题：原始 WebArena 存在约 11.3% 的假阴性率 (False-Negative Rate，即实际已解决却被标记为失败的任务)。Verified 版本采用人工审核的成功标准重新评分，并新增了一个包含 258 个任务的高难度 (Hard) 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp 对比 OSWorld 对比 WebArena

| 基准测试 (Benchmark) | 评估内容 | 时间跨度 |
|---|---|---|
| BrowseComp | 在时间压力下于开放网络中查找特定事实 | 分钟级 |
| OSWorld | 智能体操作完整桌面环境（鼠标、键盘、命令行） | 数十分钟级 |
| WebArena-Verified | 模拟网站中的事务性网页任务 | 分钟级 |
| Hard 子集 | 涉及多页面状态转换的 WebArena-Verified 任务 | 数十分钟级 |

评估维度各不相同。BrowseComp 得分高仅说明智能体擅长查找事实，并不代表它能预订机票。OSWorld 得分更接近“它能否在我的桌面上正常运行”。WebArena-Verified 则更接近“它能否走完一个完整流程”。任何生产环境决策都需要选择与任务分布相匹配的基准测试。

### 攻击面剖析

1. **间接提示词注入 (Indirect Prompt Injection)**。不可信页面内容中隐藏了指令。智能体读取并执行了这些指令。公开案例：2024年 Kai Greshake 等人研究、2025年《Tainted Memories》论文、2026年 HashJack（Cato Networks）。
2. **URL 片段/查询参数注入 (URL Fragment / Query Injection)**。被抓取 URL 的 `#fragment` 或查询字符串中包含命令。这些内容不会在页面上可见渲染，但仍会进入智能体的上下文 (Context) 中。
3. **记忆绑定攻击 (Memory-Binding Attacks)**。页面指示智能体写入持久化记忆（第12课涵盖持久状态）。在后续会话中，该记忆会在无可见触发条件的情况下激活恶意载荷 (Payload)。
4. **针对已认证会话的类 CSRF 攻击 (CSRF-Shaped Attacks)**。属于《Tainted Memories》论文所述类别：智能体已在某处登录；攻击者页面发出状态变更请求，智能体携带用户 Cookie 执行了这些请求。
5. **一键劫持 (One-Click Hijack)**。一个视觉上无害的按钮承载了智能体会跟随执行的恶意载荷。属于 Comet 类别。
6. **智能体宿主面的内容安全策略 (Content-Security-Policy) 漏洞**。渲染层和工具层本身即可成为攻击向量；“浏览器内嵌浏览器智能体”的技术栈攻击面较广。

### 为何“无法完全修补”

攻击形态与智能体的能力同构。智能体必须读取不可信内容才能完成任务。智能体读取的任何内容都可能包含指令，而其遵循的任何指令都可能与用户的真实意图发生偏离。防御措施（信任边界、分类器、工具白名单、关键操作的人工介入 (Human-In-The-Loop)）能够提高攻击成本并缩小影响范围，但无法彻底消除此类攻击。

这与洛布定理 (Löb's Theorem，第8课) 的推理模式一致：智能体无法证明下一个词元 (Token) 是安全的；它只能构建一个系统，使不安全的词元更容易被检测出来。

### 可实际落地的防御姿态

- **读/写边界 (Read/Write Boundary)**。读取操作本身不产生实质性影响。写入操作（提交表单、发布内容、调用具有副作用的工具）若由信任边界外的内容触发，则必须经过人工重新审批。
- **按任务配置工具白名单 (Tool Allowlist)**。智能体可以浏览网页，但除非该工具已针对当前任务明确启用，否则无法发起电汇等操作。第13课涵盖预算控制相关内容。
- **会话隔离 (Session Isolation)**。浏览器智能体会话仅使用受限凭证运行。不接入生产环境认证，不关联个人邮箱。保留所有 HTTP 请求日志以供审计。
- **内容清洗器 (Content Sanitizer)**。在将获取的 HTML 拼接至模型上下文之前，会剥离已知的恶意模式。（可降低简单攻击的成功率，但无法阻止复杂的恶意载荷。）
- **关键操作的人工介入 (HITL)**。采用“先提议后执行” (Propose-then-Commit) 模式（第15课）。
- **记忆金丝雀令牌 (Canary Tokens)**。若某条记忆条目被触发，用户将收到提示（第14课）。

## 使用示例

`code/main.py` 模拟了一个小型浏览器代理（browser-agent）在三个合成页面上的运行过程。其中一个页面是良性的，一个在可见文本中嵌入了直接的提示词注入（prompt-injection）载荷，另一个则包含 URL 片段注入（URL-fragment injection）（不可见，但存在于代理的上下文中）。该脚本展示了：(a) 朴素代理（naïve agent）会采取的操作；(b) 读写边界（read/write boundary）能够拦截的内容；(c) 内容清理器（sanitizer）能够拦截的内容；(d) 两者均无法拦截的内容。

## 部署上线

`outputs/skill-browser-agent-trust-boundary.md` 界定了拟议的浏览器代理部署范围：它将触及哪些信任域（trust zones），被授权写入哪些内容，以及在首次运行前必须就位哪些防御措施。

## 练习

1. 运行 `code/main.py`。识别出内容清理器能拦截但读写边界无法拦截的攻击，以及仅读写边界能拦截的攻击。

2. 扩展内容清理器，使其能够检测一类 HashJack 风格的 URL 片段注入。测量其在带有合法片段的良性 URL 上的误报率（false-positive rate）。

3. 选取一个你熟悉的真实浏览器代理工作流（例如“预订航班”）。列出所有的读取与写入操作。标记哪些写入操作需要人工介入（HITL）并说明原因。

4. 阅读 WebArena-Verified ICLR 2026 论文。找出原始 WebArena 评分不可靠的一类任务，并解释 Verified 子集如何解决该问题。

5. 为浏览器代理环境设计一个内存金丝雀（memory canary）。你会存储什么数据？存放在何处？何种条件会触发警报？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| 间接提示词注入（Indirect prompt injection） | “不良页面文本” | 代理读取的页面中包含的不可信内容，其中含有代理会执行的指令 |
| 污染记忆（Tainted Memories） | “记忆攻击” | 代理将攻击者提供的指令写入持久化记忆；在下次会话时触发 |
| HashJack | “URL 片段攻击” | 隐藏在 URL 片段/查询字符串中的载荷存在于代理上下文中，但不会在页面上可见渲染 |
| 一键劫持（One-click hijack） | “恶意按钮” | 可见的交互元素携带后续载荷，代理会执行该载荷 |
| BrowseComp | “网页搜索基准测试” | 在开放网络中查找特定事实；时间跨度为分钟级 |
| OSWorld | “桌面基准测试” | 完整操作系统控制；多步骤图形界面（GUI）任务 |
| WebArena-Verified | “修复版网页任务基准测试” | ServiceNow 重新评分的 WebArena，包含 Hard（困难）子集 |
| 读写边界（Read/write boundary） | “副作用闸门” | 读取操作绝不产生实质性影响；若内容超出信任范围，写入操作需重新获得批准 |

## 延伸阅读

- [OpenAI — 推出 ChatGPT 智能体 (Agent)](https://openai.com/index/introducing-chatgpt-agent/) — 融合了 Operator 与深度研究 (Deep Research) 功能；在 BrowseComp 基准测试中达到最先进水平 (SOTA)。
- [OpenAI — 计算机操作智能体 (Computer-Using Agent)](https://openai.com/index/computer-using-agent/) — 继承自 Operator 的技术路线，以及最终演变为 ChatGPT 智能体的底层架构。
- [Zhou 等人 — WebArena](https://webarena.dev/) — 该领域的原始基准测试 (Benchmark)。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 会议关于固定子集 (Fixed-Subset) 的论文。
- [Anthropic — 实践中的智能体自主性评估](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含针对计算机操作智能体的攻击面 (Attack Surface) 讨论。