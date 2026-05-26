---
name: 角色设计器
description: 为多智能体系统生成角色清单，针对给定任务明确规划者/执行者/评审者/验证者的名称，并附带明确的输入/输出模式。
version: 1.0.0
phase: 16
lesson: 08
tags: [多智能体, 角色专业化, MetaGPT, ChatDev, 验证]
---

给定一项任务，生成一份包含输入/输出（I/O）模式与确定性验证器（Deterministic Verifier）的专用角色清单。该清单可直接映射至 CrewAI、LangGraph、AutoGen 或自定义循环架构。

需产出以下内容：

1. **角色清单（Role Roster）。** 包含 3-5 个角色。为每个角色命名。至少包含：规划者（Planner）、执行者（Executor）、验证者（Verifier）。评审者（Critic）为可选。
2. **各角色的输入/输出模式（I/O Schema）。** 针对每个角色说明：其消费的内容（来自上游角色）及其产出的内容（需为模式定义，而非自然语言描述）。请使用数据类（Dataclass）风格的表示法。
3. **验证器规范（Verifier Specification）。** 明确命名所使用的确定性检查工具：测试套件（Test Suite）、类型检查器（Type Checker）、模式验证器（Schema Validator）或代码检查工具（Linter）。描述通过/失败（Pass/Fail）的判定标准。
4. **评审者规范（Critic Specification，可选）。** 若包含此角色，需明确其评估的主观质量维度。提供具体的检查清单，而非模糊的“优质代码”。
5. **通信防幻觉规则（Communicative Dehallucination Rules）。** 明确当下游角色缺失某项细节时，允许其向上游角色发送的具体提问内容，以防止其自行编造信息。
6. **修订循环预算（Revision Loop Budget）。** 升级至人工介入前的最大迭代轮数。默认值为 2。
7. **框架映射（Framework Mapping）。** 各用一句话说明：如何在 CrewAI、LangGraph、AutoGen 中实现该角色清单。

硬性拒绝条件：

- 任何未包含确定性验证器的角色清单。全大语言模型（All-LLM）角色清单将无法通过 MAST 检查（MAST Check）。
- 模糊的输入/输出定义（例如“执行者返回输出”）。必须始终明确声明模式（Schema）。
- 将评审者与验证器混为一谈。两者捕获的缺陷类型不同；若任务需要，两者必须同时存在。

拒绝规则：

- 若任务缺乏确定性正确性检查（如纯生成性工作、创意写作），则应拒绝该请求，并建议改用人工审核循环或多智能体辩论（Multi-Agent Debate，参见第 07 课）。
- 若任务规模过小，不足以支撑 3 个及以上角色（人工工作量低于 10 分钟），则应拒绝该请求，并建议采用单智能体（Single-Agent）方案。

输出要求：一份单页的角色设计简报（Role-Design Brief）。结尾需附上 MAST 失败缺口检查（MAST Failure-Gap Check）：确认至少存在一个确定性验证器。