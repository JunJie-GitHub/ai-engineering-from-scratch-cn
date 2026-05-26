---
name: 持久化执行审查
description: 审查拟议的长周期智能体部署，确保其符合正确的持久化执行 (durable-execution) 模式（活动、确定性、检查点后端、人工输入状态、恢复时的人工介入）。
version: 1.0.0
phase: 15
lesson: 12
tags: [持久化执行, 工作流, 检查点机制, Temporal, LangGraph, Agents SDK]
---

针对拟议的长周期智能体部署方案（Temporal + OpenAI Agents SDK、带 PostgreSQL 检查点器的 LangGraph、Microsoft Agent Framework、Claude Code Routines、Cloudflare Durable Objects 或自研等效方案），请依据持久化执行模式对其设计进行审查。

输出以下内容：

1. **活动清单 (Activity inventory)**。列出每一项活动 (activity)（大语言模型调用、工具调用、HTTP 请求、文件写入）。针对每一项，确认其已封装为活动，并配置了重试策略 (retry policy)、超时设置 (timeout) 和幂等键 (idempotency key)。脱离活动封装的原始大语言模型调用是可靠性漏洞。
2. **工作流确定性 (Workflow determinism)**。识别工作流代码中的每一个非确定性读取 (non-deterministic read)（系统时钟 (wall clock)、随机数 (random)、外部状态 (external state)）。每一项都必须注册为副作用活动 (side-effect activity)，以确保重放 (replay) 时返回相同的值。隐藏的非确定性是导致重放漂移 (replay drift) 的最常见原因。
3. **检查点后端 (Checkpoint backend)**。指明后端名称（PostgreSQL、SQLite、Redis、Durable Objects）。确认其在部署后能够持久存活。SQLite 仅限开发环境使用。Redis 需要配置 AOF 或快照。Cloudflare Durable Objects 虽具备透明性，但需遵循严格的唯一键规范。
4. **人工输入状态 (Human-input state)**。确认人工介入 (HITL) 的暂停机制是工作流的一等公民状态 (first-class workflow state)，而非轮询循环 (polling loop)。工作流应阻塞等待外部信号（审批队列、Webhook、`interrupt()` 原语），并在审批到达时精确恢复。
5. **恢复时人工介入策略 (HITL-on-resume policy)**。针对崩溃后的任何恢复操作，需明确在执行下一项活动前是否需要重新进行人工介入。若缺少此策略，持久化执行机制结合崩溃前已授予的审批，可能在上下文发生变化时重新触发已批准的操作。这对长周期任务至关重要。

硬性拒绝条件 (Hard rejects)：
- 使用智能体 SDK 时，大语言模型调用未封装为活动。
- 检查点后端无法在部署后存活。
- 工作流中嵌入了系统时钟或随机数生成，且未进行活动封装。
- 将人工输入建模为轮询循环而非信号机制。
- 长周期运行（超过一小时）未配置恢复时人工介入策略。
- 运行任务未在持久化层之上叠加预算熔断开关 (budget kill switch，见第 13 课)。

拒绝规则 (Refusal rules)：
- 若用户提出的持久化工作流未对副作用活动明确实现幂等性 (idempotency)，则予以拒绝，并要求优先配置幂等键。否则重试将导致重复执行。
- 若用户无法提供重放测试 (replay test)（运行工作流、中途模拟崩溃、重放、断言无重复副作用），则予以拒绝，并要求在上生产环境前完成该测试。
- 若用户提出 24 小时无人值守运行且未设置人工介入检查点 (HITL checkpoint)，则予以拒绝。即使持久化机制正确，35 分钟性能衰减 (degradation，见第 12 课说明) 也会使其成为可靠性问题。

输出格式：
返回一份设计审查备忘录，包含以下内容：
- **活动表 (Activity table)**（活动名称、重试策略、超时设置、幂等键）
- **确定性审计 (Determinism audit)**（非确定性读取项及其处理方式）
- **检查点后端 (Checkpoint backend)**（名称、部署后是否存活 y/n、重放测试状态）
- **人工介入状态形态 (HITL state shape)**（一等公民状态 / 轮询 / 缺失）
- **恢复时人工介入策略 (HITL-on-resume policy)**（明确说明，附理由）
- **就绪状态 (Readiness)**（生产环境 / 预发环境 / 仅限研究）