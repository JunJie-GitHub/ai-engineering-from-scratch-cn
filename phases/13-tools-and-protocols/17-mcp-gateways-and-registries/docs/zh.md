# MCP 网关（MCP Gateway）与注册表（Registry）—— 企业控制平面（Enterprise Control Plane）

> 企业不能让每位开发者随意安装未经审核的 MCP 服务器。网关将身份验证（Authentication）、基于角色的访问控制（Role-Based Access Control, RBAC）、审计（Audit）、速率限制（Rate Limiting）、缓存（Caching）以及工具投毒检测（Tool-Poisoning Detection）集中管理，随后将整合后的工具集以单一 MCP 端点（Endpoint）的形式对外暴露。官方 MCP 注册表（Official MCP Registry，由 Anthropic、GitHub、PulseMCP 和 Microsoft 联合维护，经过命名空间验证）是权威的上游源（Canonical Upstream）。本课程将阐明网关的架构定位，演示一个最小化实现方案，并盘点 2026 年的供应商生态格局。

**类型：** 学习
**语言：** Python（标准库，最小化网关）
**前置知识：** Phase 13 · 15（工具投毒），Phase 13 · 16（OAuth 2.1）
**时长：** 约 45 分钟

## 学习目标

- 阐明 MCP 网关的架构位置（位于 MCP 客户端与多个后端 MCP 服务器之间）。
- 实现网关的五大核心职责：身份验证、基于角色的访问控制、审计、速率限制与策略管理（Policy）。
- 在网关层强制实施固定工具哈希清单（Pinned-Tool-Hash Manifest）。
- 区分官方 MCP 注册表与元注册表（Metaregistry，如 Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题背景

一家财富 500 强企业拥有 30 个已获批的 MCP 服务器、5000 名开发者，同时面临合规与审计要求，其安全团队也要求实施集中式策略管控。允许每位开发者在集成开发环境（Integrated Development Environment, IDE）中随意安装任意服务器是绝对不可行的。

网关模式（Gateway Pattern）如下：

1. 网关作为单一的流式 HTTP 端点（Streamable HTTP Endpoint）运行，供开发者连接。
2. 网关集中保管每个后端 MCP 服务器的访问凭证（Credentials）。
3. 所有开发者请求均通过网关自身的 OAuth 协议进行身份验证与作用域（Scope）管控。
4. 网关将调用路由至后端服务器，并在此过程中应用既定策略。
5. 所有调用记录均被留存以供审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway 等产品均在 2025 至 2026 年间推出了网关或网关相关功能。

与此同时，官方 MCP 注册表作为标准上游源正式上线：它提供经过人工筛选、命名空间验证及反向 DNS 命名的服务器，供网关拉取使用。而元注册表则负责聚合来自多个来源的服务器。

## 核心概念

### 网关的五大职责

1. **身份验证（Auth）。** 使用 OAuth 2.1 识别开发者身份，并映射到相应的用户角色。
2. **基于角色的访问控制（RBAC）。** 针对每个用户的策略：允许访问哪些服务器、哪些工具以及哪些权限范围（scopes）。
3. **审计（Audit）。** 记录每一次调用的详细信息，包括调用者、操作内容、时间及结果。
4. **速率限制（Rate limit）。** 针对用户、工具或服务器设置调用上限，以防止滥用。
5. **策略（Policy）。** 拒绝投毒描述（poisoned descriptions），强制执行“双重规则（Rule of Two）”，并脱敏处理个人身份信息（PII）。

### 网关作为单一端点

对开发者而言，网关表现为一个单一的 MCP（Model Context Protocol）服务器。在内部，它负责将请求路由至 N 个后端。会话 ID（Session IDs，参见第 13 阶段 · 09）会在边界处进行重写。

### 凭据保险库（Credential vaulting）

开发者永远无法直接看到后端令牌。网关负责保管这些令牌（或代理至持有令牌的身份提供商）。如果开发者在网关上拥有 `notes:read` 权限，则可以通过网关自身的后端凭据，以传递方式访问 notes MCP 服务器——但这必须受限于绑定该传递访问权限的策略约束。

### 网关处的工具哈希锁定（Tool-hash pinning）

网关维护着一份已批准工具描述清单（包含 SHA256 哈希值）。在服务发现阶段，网关会获取每个后端的 `tools/list`，将哈希值与清单进行比对，并移除任何描述发生变异的工具。这是将第 13 阶段 · 15 中提到的防抽地毯攻击（rug-pull defense）机制集中化应用。

### 策略即代码（Policy-as-code）

高级网关使用 OPA/Rego、Kyverno 或 Styra 等工具以代码形式表达策略。例如“用户 `alice` 仅可在 `acme` 组织的仓库中调用 `github.open_pr`”这类规则会被以声明式（declaratively）编码。简易网关则采用手写 Python 代码实现。两种架构模式均有效。

### 会话感知路由（Session-aware routing）

当用户的会话涉及多个服务器时，网关会进行多路复用（multiplexes）：开发者的单个 MCP 会话将维持 N 个后端会话（每个服务器对应一个）。来自任意后端的通知都会经由网关路由至开发者的会话中。

### 命名空间合并（Namespace merging）

网关会合并来自所有后端的工具命名空间，通常在发生冲突时采用添加前缀的方式处理。例如 `github.open_pr`、`notes.search`。这使得路由指向明确无歧义。

### 注册表（Registries）

- **官方 MCP 注册表（`registry.modelcontextprotocol.io`）。** 由 Anthropic、GitHub、PulseMCP 和 Microsoft 共同主导推出。采用命名空间验证机制（反向 DNS 格式：`io.github.user/server`）。已进行基础质量预过滤。
- **Glama。** 以搜索为中心的元注册表（metaregistry），聚合了多个来源的数据。
- **MCPMarket。** 偏向商业化的目录，提供供应商列表。
- **MCP.so。** 社区目录，支持开放提交。
- **Smithery。** 采用类似包管理器（package-manager）的安装流程。
- **LobeHub。** 集成在其 LobeChat 应用中的 UI 注册表。

企业级网关默认从官方注册表拉取数据，允许管理员从元注册表中精选添加内容，并拒绝任何未固定版本（unpinned）的组件。

### 反向 DNS 命名（Reverse-DNS naming）

官方注册表强制要求公共服务器使用反向 DNS 命名格式：`io.github.alice/notes`。命名空间机制可有效防止域名抢注（squatting），并使信任委派（trust delegation）更加清晰。

### 供应商调研（2026 年 4 月）

| 供应商 | 优势 |
|--------|----------|
| Cloudflare MCP Portals | 边缘托管（Edge-hosted）；集成 OAuth；提供免费层级 |
| Kong AI Gateway | 原生支持 Kubernetes（K8s）；细粒度策略（fine-grained policy）；日志输出至 OpenTelemetry |
| IBM ContextForge | 企业级身份与访问管理（IAM）；合规性（compliance）支持；审计导出（audit export） |
| TrueFoundry | 偏向 DevOps；指标优先（metrics-first） |
| MintMCP | 面向开发者平台 |
| Envoy AI Gateway | 开源；支持自定义过滤器（customizable filters） |

第 17 阶段（生产基础设施）将深入探讨网关的运维操作。

## 使用它

`code/main.py` 提供了一个约 150 行代码的极简网关 (Gateway) 实现：它通过伪造的 Bearer 令牌 (Bearer Token) 进行用户身份验证，维护基于用户的基于角色的访问控制 (RBAC) 策略，将请求路由至两个后端模型上下文协议 (MCP) 服务器，将每次调用写入审计日志 (Audit Log)，实施速率限制 (Rate Limit)，并拒绝任何描述哈希值与固定清单 (Pinned Manifest) 不匹配的后端工具。

重点关注以下内容：

- 以 `user_id` 为键的 `RBAC` 字典，包含允许的 `server_tool` 条目。
- `AUDIT_LOG` 是一个仅追加 (append-only) 的事件列表。
- 速率限制采用基于用户的令牌桶 (Token Bucket) 算法。
- 固定清单是一个 `server::tool -> hash` 的字典映射。

## 交付上线

本课时将生成 `outputs/skill-gateway-bootstrap.md` 文件。在提供企业级 MCP 规划（包含用户、后端及合规要求）的前提下，该技能 (Skill) 会输出一份网关配置规范。

## 练习

1. 运行 `code/main.py`。分别以授权用户、未授权用户身份发起调用，随后模拟超出速率限制的突发请求。验证这三种场景的完整流程。

2. 添加一项策略，在将结果返回给客户端前对结果中的个人身份信息 (PII) 进行脱敏。可使用简单的正则表达式 (Regex) 匹配社会安全号码 (SSN) 格式的字符串；请注意其中的遗漏项（如电子邮件、电话号码）。

3. 扩展审计日志以输出 OpenTelemetry GenAI 跨度 (Spans)。第 13 阶段（共 20 阶段）详细说明了具体的属性字段。

4. 为一个拥有 50 名开发者的团队设计 RBAC 策略，该团队需接入五个后端服务（notes、github、postgres、jira、slack）。明确每个后端谁拥有只读权限？谁拥有写入权限？

5. 通读 Cloudflare 关于企业级 MCP 的博文。找出一个 Cloudflare 已提供但本标准库网关尚未实现的功能。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 网关 (Gateway) | “MCP 代理” | 位于客户端与后端之间的中心化服务器 |
| 凭据保险库 (Credential Vaulting) | “后端令牌保留在服务器端” | 开发人员永远无法接触到上游令牌 |
| 会话感知路由 (Session-Aware Routing) | “多后端会话” | 网关为每个开发者会话复用 N 个后端会话 |
| 工具哈希锁定 (Tool-Hash Pinning) | “已批准清单” | 记录每个已批准工具描述的 SHA256 哈希值；从中心层面阻断恶意替换 (rug-pulls) |
| 基于角色的访问控制 (RBAC) | “基于用户的策略” | 针对工具和服务器的基于角色的访问控制 |
| 策略即代码 (Policy-as-Code) | “声明式规则” | 在网关层强制执行的 OPA/Rego、Kyverno、Styra 策略 |
| 审计日志 (Audit Log) | “谁、做了什么、何时” | 用于合规审查的仅追加事件日志 |
| 速率限制 (Rate Limit) | “基于用户的令牌桶” | 按分钟设置上限以防止滥用 |
| 官方 MCP 注册表 (Official MCP Registry) | “规范上游源” | `registry.modelcontextprotocol.io`，经过命名空间验证 |
| 反向 DNS 命名 (Reverse-DNS Naming) | “注册表命名空间” | 遵循 `io.github.user/server` 命名规范 |

## 扩展阅读

- [官方 MCP 注册表](https://registry.modelcontextprotocol.io/) — 权威上游源 (Canonical Upstream)，命名空间已验证 (Namespace-Verified)
- [Cloudflare — 企业级 MCP](https://blog.cloudflare.com/enterprise-mcp/) — 集成 OAuth 与策略的网关模式 (Gateway Pattern)
- [agentic-community — MCP 网关注册表](https://github.com/agentic-community/mcp-gateway-registry) — 开源参考网关 (Open-Source Reference Gateway)
- [TrueFoundry — 什么是 MCP 网关？](https://www.truefoundry.com/blog/what-is-mcp-gateway) — 功能对比文章 (Feature Comparison Article)
- [IBM — MCP 上下文锻造器](https://github.com/IBM/mcp-context-forge) — IBM 推出的企业级网关 (Enterprise Gateway)