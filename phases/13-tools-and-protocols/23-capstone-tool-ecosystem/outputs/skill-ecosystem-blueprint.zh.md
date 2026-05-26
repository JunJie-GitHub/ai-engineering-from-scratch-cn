---
name: 生态系统蓝图
description: 根据产品需求生成完整的第 13 阶段生态系统架构；明确原语、安全态势、遥测和打包方式。
version: 1.0.0
phase: 13
lesson: 22
tags: [模型上下文协议, 综合实践, 生态系统, 架构, 智能体间通信, 开放遥测]
---

给定产品需求（研究、摘要生成、自动化或任何智能体驱动的工作流），生成完整的架构。

输出内容：

1. 模型上下文协议 (Model Context Protocol, MCP) 原语。需要哪些工具、资源、提示词和任务？是否包含 `ui://` 应用？是否有异步任务？
2. 安全态势 (Security Posture)。OAuth 2.1 作用域集合、网关基于角色的访问控制 (Role-Based Access Control, RBAC) 矩阵、哈希锁定清单 (Pinned Hash Manifest)、双重规则 (Rule of Two) 审计。
3. 智能体间通信 (Agent-to-Agent, A2A) 协作。识别任何子智能体调用。定义它们的智能体卡片 (Agent Cards)。
4. 遥测 (Telemetry)。开放遥测 (OpenTelemetry, OTel) 生成式人工智能 (Generative AI, GenAI) 跨度层级。导出器与后端选择。
5. 打包/部署封装。AGENTS.md、SKILL.md 以及部署载体（Docker Compose、K8s）。
6. 映射至第 13 阶段课程。每项设计决策分别对应哪一节课程。

硬性拒绝条件：
- 任何在单轮交互中混合不可信输入、敏感数据和关键性操作 (Consequential Action) 的架构（违反双重规则）。
- 任何在 MCP 和 A2A 跳转间缺乏链路追踪传播 (Trace Propagation) 的架构。
- 任何在大语言模型 (Large Language Model, LLM) 层未配置至少一个备用提供商 (Fallback Provider) 的架构。

拒绝规则：
- 若产品需求更适合通过直接调用 LLM 解决，则拒绝搭建完整生态系统。
- 若团队缺乏针对网关的站点可靠性工程 (Site Reliability Engineering, SRE) 支持，建议采用托管网关（如 Cloudflare MCP Portals、Portkey）。
- 若架构涉及支付功能，需将 AP2 标记为存在行为偏移风险 (Drift Risk) 的 A2A 扩展，并建议单独审批确认 (Signoff)。

输出：一份单页蓝图，包含原语、安全态势、A2A 跳转、遥测计划、打包方案及课程映射。结尾用一句话指出该部署面临的最严峻的单一运维风险。