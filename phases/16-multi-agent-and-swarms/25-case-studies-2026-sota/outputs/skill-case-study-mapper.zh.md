---
name: 案例研究映射器
description: 将拟议的多智能体系统 (multi-agent system) 设计映射至最接近的 2026 年生产环境参考案例（Anthropic Research、MetaGPT/ChatDev 或 OpenClaw/Moltbook）。揭示已知的权衡取舍、推荐框架，以及已在生产环境中验证的具体设计决策。
version: 1.0.0
phase: 16
lesson: 25
tags: [多智能体, 案例研究, 生产环境, 框架选择, 参考架构]
---

给定一个拟议的多智能体系统 (multi-agent system) 设计，请挑选最接近的 2026 年标准案例研究 (case study) 并进行适配。

产出内容：

1. **设计指纹 (Design fingerprint)。** 任务类型（研究 / 工程 / 群体 / 自动化）、智能体数量、验证要求 (verification requirement)、运行时长、角色区分度、面向用户的网络暴露面。
2. **最接近的案例研究 (case study)。**
   - **Anthropic Research** 适用条件：研究或知识检索任务、强制要求验证、运行时长达数小时、智能体主要通过上下文和作用域进行区分（采用全新上下文的子智能体 (fresh-context subagents) 更具优势）。
   - **MetaGPT / ChatDev** 适用条件：工程或结构化工作流、角色清晰可辨（规划者 / 编码者 / 审查者 / 测试者）、交接产物 (handoff artifacts) 具备明确的类型定义。
   - **OpenClaw / Moltbook** 适用条件：群体规模、面向用户的智能体网络、提示词注入 (prompt-injection) 构成实质性威胁、涌现经济 (emergent economy) 具有重要影响。
3. **可复用的模式 (Patterns to copy)。** 所选案例研究中适用的具体设计决策：全新上下文子智能体 (fresh-context subagents)、彩虹部署 (rainbow deploy)、通信式去幻觉 (communicative dehallucination)、有向无环图路由 (DAG routing)、不可写验证器 (unwritable verifier)、底层安全 (substrate-level security)。
4. **框架推荐 (Framework recommendation)。** LangGraph、CrewAI、AG2、Microsoft Agent Framework、OpenAI Agents SDK、Google ADK、Anthropic Claude Agent SDK 或自定义框架。默认采用该案例研究的典型框架；若存在更契合当前特定设计的框架，需予以注明。
5. **案例中的反模式 (Anti-patterns)。** 参考案例中已证实无效的做法。在新设计中应避免。
6. **成本预估 (Cost projection)。** 预期 Token 消耗乘数 (token multiplier)（Anthropic Research：约 15 倍；MetaGPT：约 5 倍；OpenClaw：取决于网络效应）。预期的实际运行时间 (wall-clock) 与美元成本区间。
7. **评估方法 (Evaluation approach)。** 适用的基准测试（MARBLE、SWE-bench Pro 或内部基准）；设定相对于案例研究基线的合理改进目标 (delta)。

硬性拒绝条件 (Hard rejects)：

- 在任务具有正确性要求时却忽略验证的设计。每个案例研究都需承担验证开销 (verification tax)。
- 声称采用新底层架构 (substrate) 却未将提示词注入 (prompt-injection) 视为攻击面的设计。OpenClaw/Moltbook 案例表明这是生产环境中的实际关切，而非理论假设。
- 无法映射至任何案例研究的“革命性”主张。多智能体系统自 2024 年起已投入生产环境；新颖主张需提供明确的对比分析。
- 无正当理由跳过 MCP 或 A2A 协议采用的设计。协议支持是基本门槛。

拒绝规则 (Refusal rules)：

- 若设计缺乏明确的任务类型，建议在挑选案例研究前先界定任务范围。“适用于一切的多智能体”并非有效设计。
- 若设计声称具备生产就绪状态 (production readiness) 但缺乏故障模式审计，建议在参考映射前执行 MAST 风格审计 (MAST-style audit)（第 25 课）。
- 若设计纯属实验/研究性质，需注明在采用任何案例研究的生产模式前，哪些方面需要进行加固。

输出要求：一份两页的简报。以一句话总结开头（“最接近的案例研究：MetaGPT / ChatDev。采用角色标准作业程序分解 (role-SOP decomposition)、通信式去幻觉 (communicative dehallucination) 与结构化交接产物 (structured handoff artifacts)；使用 CrewAI 或自定义框架。”），随后展开上述七个部分。结尾附上 90 天适配计划：从参考案例中复制什么、定制什么，以及针对基准测试验证什么。