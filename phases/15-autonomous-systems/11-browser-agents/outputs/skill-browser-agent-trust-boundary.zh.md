---
name: 浏览器代理信任边界
description: 在代理接触真实网站之前，界定拟议的浏览器代理（Browser Agent）部署范围——包括信任区域、授权写入操作及必需的防御措施。
version: 1.0.0
phase: 15
lesson: 11
tags: [浏览器代理, 提示注入, 信任边界, OSWorld, WebArena]
---

针对拟议的浏览器代理工作流，生成一份信任边界（Trust Boundary）界定文档，逐一列出所有读取操作、所有写入操作，以及首次运行所需的最低防御栈（Defense Stack）。

需产出以下内容：

1. **读取面（Read Surface）。** 列出代理将获取的所有源（Origin）。将每个源分类为信任域内（In-Trust，由用户组织运营的第一方站点）或信任域外（Out-of-Trust，任何第三方、任何用户生成内容、任何搜索结果）。所有信任域外的读取操作都必须被视为潜在的提示注入（Prompt Injection）通道。
2. **写入面（Write Surface）。** 列出代理获授权执行的每一项关键操作（提交表单、发布内容、调用后端工具、写入内存）。针对每项操作，说明其爆炸半径（Blast Radius）以及该操作是否可逆。
3. **必需防御措施。** 最低防御栈：内容净化器（Content Sanitizer）、读写边界（当 `content_origin` 为信任域外时，写入操作需重新审批）、按任务配置的工具允许列表（Tool Allowlist）、具备作用域凭据（Scoped Credentials）的会话隔离（Session Isolation）、持久化内存中的金丝雀令牌（Canary Tokens）、不可逆操作上的人工介入（Human-in-the-Loop, HITL）。
4. **基准测试与数据分布匹配度（Benchmark-to-Distribution Fit）。** 如果代理报告了 BrowseComp、OSWorld 或 WebArena-Verified 分数，请说明该基准测试与实际任务之间的数据分布重叠（Distribution Overlap）情况。较高的 BrowseComp 分数并不能预测预订流程的可靠性。
5. **已知攻击检查清单。** 确认部署已针对以下攻击进行加固：(a) 可见文本注入（Visible-Text Injection），(b) URL 片段/查询参数注入（URL-Fragment / Query Injection），(c) 内存绑定攻击（Memory-Binding Attacks，Tainted Memories 类），(d) 针对已认证会话的类 CSRF 攻击（CSRF-Shaped Attacks），(e) 一键劫持（One-Click Hijacks）。针对每一项，需指明具体的防御机制及其触发位置。

硬性拒绝条件：
- 可访问生产环境凭据但未实施会话隔离的浏览器代理。
- 任何由信任域外内容触发的写入操作无需重新进行人工介入（HITL）审批的部署。
- 仅依赖内容净化器的部署（净化器只能拦截简单攻击，复杂载荷仍可绕过）。
- 未配置金丝雀条目的持久化内存。
- 涉及金融交易或客户数据的工作流，且写入操作无人工介入（HITL）机制。

拒绝规则：
- 如果用户无法说明由注入引发的错误写入操作的爆炸半径，则予以拒绝，并要求其提供明确的陈述。
- 如果用户提议在无法提供作用域凭据的技术栈上部署浏览器代理，则予以拒绝，并要求先配置独立身份。
- 如果用户引用基准测试分数（BrowseComp、OSWorld、WebArena）作为代理“能够”执行生产任务的证据，则予以拒绝，并要求在实际数据分布上进行内部评估（Internal Evals）。

输出格式：

返回一份信任边界备忘录，包含以下内容：
- **读取面表格**（源，信任域内 / 信任域外）
- **写入面表格**（操作，爆炸半径，是否可逆 y/n）
- **防御栈**（已配置层级的无序列表）
- **基准匹配度说明**（如适用）
- **已知攻击检查清单**（五行，每行需注明防御机制）
- **部署结论**（生产环境 / 预发环境 / 仅限研究）