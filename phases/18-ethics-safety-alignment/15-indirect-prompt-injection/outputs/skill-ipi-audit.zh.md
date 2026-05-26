---
name: ipi-audit
description: 审计智能体部署（Agentic Deployment）中的间接提示注入（Indirect Prompt Injection, IPI）暴露风险及信息流控制（Information-Flow Control, IFC）覆盖范围。
version: 1.0.0
phase: 18
lesson: 15
tags: [ipi, indirect-prompt-injection, ifc, agent-security, owasp-llm01]
---

给定智能体部署（Agentic Deployment）描述，审计该部署的间接提示注入（Indirect Prompt Injection, IPI）暴露风险。

输出内容：

1. 不可信内容清单。列出智能体可能读取的所有内容来源：检索增强生成（Retrieval-Augmented Generation, RAG）文档、收件箱、日历、工具输出、工单、产品评论、第三方 API。每一项均为潜在的 IPI 攻击向量。
2. 信任标签（Trust Labelling）。部署是否将可信内容（用户提示）与不可信内容（检索内容）进行了隔离？若内容在未添加标签的情况下被拼接至同一提示中，则信息流控制（Information-Flow Control, IFC）未生效。
3. 动作门控（Action Gating）。哪些工具可被调用？针对每个工具，其调用是否仅由可信提示进行门控，还是不可信内容也能影响调用决策？
4. 自适应攻击评估（Adaptive-Attack Evaluation）。该部署是否已按照 Nasr 等人（2025）的研究，使用自适应攻击（梯度攻击、强化学习攻击、人工红队测试）进行过测试？仅依赖静态攻击的评估是不充分的。
5. 越权边界（Scope-Violation Boundaries）。识别所有跨信任边界（例如：收件箱 -> 发送邮件，文档 -> 调用外部 API）。针对每个边界，验证该动作在不可信内容影响下是否被禁止，或是否已获得可信提示的明确批准。

硬性拒绝条件：
- 任何未在检索内容上设置明确信任标签的智能体部署。
- 任何仅基于静态攻击的防御声明。
- 任何声称“我们的智能体可免疫提示注入”却未指明具体信息流控制（IFC）机制的声明。

拒绝规则：
- 若用户询问过滤机制是否足够，应予以拒绝，并解释 Nasr（2025）的研究结果：自适应攻击可突破超过 90% 的基于过滤的防御（Filter-Based Defenses）。
- 若用户寻求“一劳永逸”的防御方案，应予以拒绝——IPI 防御需要结合信息流控制（IFC）、分层响应审核（Layered Response Moderation）以及针对高风险动作的人工审计（Human Audit）。

输出要求：一份单页审计报告，需完整填写上述五个部分，标出最危险的“不可信至可信”边界，并指出最急需添加的一项控制措施。需分别引用一次 MDPI Information 17(1):54 (2026) 和 Nasr 等人（2025年10月）的文献。