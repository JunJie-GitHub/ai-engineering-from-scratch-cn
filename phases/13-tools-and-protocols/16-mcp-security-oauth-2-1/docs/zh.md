# MCP 安全 II — OAuth 2.1、资源指示器（Resource Indicators）与增量作用域（Incremental Scopes）

> 远程 MCP 服务器不仅需要身份验证（Authentication），还需要授权（Authorization）。2025-11-25 规范与 OAuth 2.1 + PKCE + 资源指示器（RFC 8707）+ 受保护资源元数据（Protected-Resource Metadata，RFC 9728）保持一致。SEP-835 增加了增量作用域同意机制，并在收到 403 WWW-Authenticate 响应时触发升级授权（Step-up Authorization）。本课将升级授权流程实现为状态机（State Machine），以便你清晰观察每一步跳转。

**类型：** 构建实践
**语言：** Python（标准库、OAuth 状态机模拟器）
**前置知识：** 第 13 阶段 · 09（传输层），第 13 阶段 · 15（安全 I）
**预计耗时：** 约 75 分钟

## 学习目标

- 区分资源服务器（Resource Server）与授权服务器（Authorization Server）的职责。
- 逐步演练受 PKCE 保护的 OAuth 2.1 授权码流程（Authorization Code Flow）。
- 使用 `resource` 参数（RFC 8707）与受保护资源元数据来防止混淆代理攻击（Confused Deputy Attacks）。
- 实现升级授权：服务器返回 403 状态码及 WWW-Authenticate 头以请求更高权限的作用域（Scope）；客户端重新提示用户授权并重试请求。

## 问题背景

早期的 MCP（2025 年之前）在部署远程服务器时，通常使用临时性的 API 密钥，甚至完全不提供身份验证机制。2025-11-25 规范通过引入完整的 OAuth 2.1 配置文件（Profile）填补了这一安全空白。

实际应用中存在三大核心需求：

- **普通远程服务器。** 用户安装的远程 MCP 服务器需要访问其 Notion / GitHub / Gmail 等第三方服务。采用带 PKCE 的 OAuth 2.1 是最合适的架构。
- **作用域升级（Scope Escalation）。** 一个已获 `notes:read` 权限的笔记服务器，在执行特定操作时可能需要 `notes:write` 权限。无需重新走完整套授权流程，升级授权机制（SEP-835）可直接请求新增的作用域。
- **防止混淆代理攻击。** 客户端持有的令牌（Token）原本仅面向服务器 A。若服务器 A 存在恶意行为，试图将该令牌出示给服务器 B，资源指示器（RFC 8707）可将令牌严格绑定至其预期的受众，从而阻断此类攻击。

OAuth 2.1 本身并非新技术。MCP 的创新之处在于其专属的配置文件规范：明确规定了必须使用的流程（仅限授权码流程 + PKCE；默认禁用隐式授权（Implicit Grant）与客户端凭证模式（Client Credentials Grant））、强制要求在每次令牌请求中携带资源指示器，以及发布受保护资源元数据，以便客户端准确定位授权端点。

## 核心概念

### 角色 (Roles)

- **客户端 (Client)**。MCP 客户端（如 Claude Desktop、Cursor 等）。
- **资源服务器 (Resource Server)**。MCP 服务器（如笔记服务、GitHub、Postgres 等）。
- **授权服务器 (Authorization Server)**。负责颁发令牌 (Token)。它可以与资源服务器是同一服务，也可以是独立的身份提供商 (IdP)（如 Auth0、Keycloak、Cognito）。

在 MCP 的配置规范中，资源服务器和授权服务器**可以**是同一主机，但**应当**通过不同的 URL 进行区分。

### 授权码模式 + PKCE (Proof Key for Code Exchange)

流程如下：

1. 客户端生成随机的 `code_verifier` 和对应的 `code_challenge`（SHA256 哈希值）。
2. 客户端将用户重定向至 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户授权同意。授权服务器将用户重定向至 `redirect_uri?code=...`。
4. 客户端向 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...` 发起 POST 请求。
5. 授权服务器验证 `code_verifier` 的哈希值是否与之前存储的 `code_challenge` 匹配，匹配成功后颁发访问令牌 (Access Token)。
6. 客户端在向资源服务器发起的每次请求中携带该令牌：`Authorization: Bearer ...`。

PKCE 机制可防止授权码拦截攻击。资源指示符 (Resource Indicators) 则确保令牌无法在其他服务中被滥用。

### 受保护资源元数据 (Protected-Resource Metadata, RFC 9728)

资源服务器会发布一个 `.well-known/oauth-protected-resource` 文档：

{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}

客户端可通过资源服务器自动发现授权服务器。这大幅简化了配置——客户端仅需知道资源 URL 即可。

### 资源指示符 (Resource Indicators, RFC 8707)

令牌请求中的 `resource` 参数用于固定该令牌的预期受众 (Audience)。颁发的令牌将包含 `aud: "https://notes.example.com"`。若其他 MCP 服务器收到此令牌，会校验 `aud` 字段并拒绝该请求。

### 权限范围模型 (Scope Model)

权限范围 (Scope) 是以空格分隔的字符串。MCP 的常见约定如下：

- `notes:read`, `notes:write`, `notes:delete`
- `admin:*` 用于管理员权限（应谨慎使用）
- `profile:read` 用于身份标识

权限范围的选择应遵循最小权限原则：仅申请当前所需权限，在需要更多权限时再进行权限升级 (Step-up)。

### 权限升级授权 (Step-up Authorization, SEP-835)

用户最初仅授予了 `notes:read` 权限。随后，用户要求智能体 (Agent) 删除一条笔记。此时服务器将返回：

HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"

客户端捕获到 `insufficient_scope` 错误后，会弹出授权对话框请求用户同意新增的权限范围，随后执行一次轻量级的 OAuth 流程获取新令牌，并使用新令牌重试请求。

### 令牌受众校验 (Token Audience Validation)

每次请求时，服务器都会校验 `token.aud == self.resource_url`。若不匹配则返回 401 状态码。此举可有效防止令牌在不同服务器间被跨域复用。

### 短期令牌与轮换机制 (Short-lived Tokens and Rotation)

访问令牌**应当**设置较短的有效期（默认 1 小时）。刷新令牌 (Refresh Token) 在每次刷新时都会轮换。客户端会在后台静默处理令牌刷新流程。

### 禁止令牌透传 (No Token Passthrough)

采样服务器（Phase 13 · 11）**严禁**将客户端的令牌透传至其他服务。采样请求即为安全边界。

### 防止混淆代理攻击 (Confused Deputy Prevention)

令牌绑定至 `aud`，客户端绑定至 `client_id`。每次请求均需同时校验这两项。该规范明确禁止了 MCP 之前远程工具生态中常见的“令牌透传”旧模式。

### 客户端 ID 发现机制 (Client ID Discovery)

每个 MCP 客户端都会在固定 URL 发布其元数据。授权服务器可获取该客户端的元数据文档，以自动发现重定向 URI 和联系信息。这免去了手动注册客户端的繁琐步骤。

### 网关与 OAuth (Gateways and OAuth)

Phase 13 · 17 展示了企业网关如何处理 OAuth：网关持有上游服务器的凭证，向客户端颁发的令牌由网关签发，且上游服务器的令牌永远不会离开网关。这彻底改变了信任模型——用户只需向网关进行一次身份认证，后续由网关代为处理 N 个服务器的授权。

## 使用方法

`code/main.py` 将完整的 OAuth 2.1 逐步授权（step-up authorization）流程模拟为状态机（state machine）。它实现了：

- PKCE（Proof Key for Code Exchange）代码验证器（code-verifier）与挑战（challenge）生成。
- 带资源指示器（resource indicator）的授权码流程（authorization code flow）。
- 受保护资源元数据（protected-resource metadata）端点。
- 包含受众（audience）检查的令牌验证。
- 在遇到 `insufficient_scope` 时触发逐步授权。

本课程不包含 HTTP 服务器；状态机在内存中运行，以便你追踪每一步跳转。第 13 阶段（共 17 阶段）的网关课程会将其连接到实际的传输层。

## 交付物

本课程将生成 `outputs/skill-oauth-scope-planner.md`。给定一个带有工具的远程 MCP（Model Context Protocol）服务器，该技能（skill）将设计作用域集合、绑定规则以及逐步授权策略。

## 练习

1. 运行 `code/main.py`。追踪双作用域的逐步授权流程。注意在逐步授权时哪些跳转步骤会重复。
2. 添加刷新令牌轮换（refresh-token rotation）：每次刷新都会签发新的刷新令牌并使旧令牌失效。模拟在轮换后使用被盗的刷新令牌，并确认其会失败。
3. 使用标准库 `http.server` 将受保护资源元数据端点实现为真实的 HTTP 响应。镜像第 09 课中的 `/mcp` 端点。
4. 为 GitHub MCP 服务器设计作用域层级（scope hierarchy）：读取仓库、编写 PR、批准 PR、合并 PR、管理员。在每个层级之间使用逐步授权。
5. 阅读 RFC 8707 和 RFC 9728。找出 RFC 9728 中 MCP 使用方式与 RFC 示例不同的那个字段。（提示：与 `scopes_supported` 有关。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| OAuth 2.1 | “现代 OAuth” | 整合后的 RFC，强制要求使用 PKCE 并禁止隐式授权流程 |
| PKCE | “所有权证明” | 代码验证器 + 挑战，用于防御授权码拦截攻击 |
| 资源指示器 | “令牌受众” | RFC 8707 中的 `resource` 参数，用于将令牌绑定至单一服务器 |
| 受保护资源元数据 | “发现文档” | RFC 9728 定义的 `.well-known/oauth-protected-resource` 端点 |
| 逐步授权 | “增量同意” | SEP-835 流程，用于按需添加作用域 |
| `insufficient_scope` | “带 WWW-Authenticate 的 403 错误” | 服务器发出的信号，要求重新授权以获取更大作用域 |
| 混淆代理攻击 | “跨服务令牌重用” | 攻击者利用受信任持有者不当转发令牌的安全漏洞 |
| 短期令牌 | “访问令牌生存时间” | 快速过期的持有者令牌；由刷新令牌进行续期 |
| 作用域层级 | “最小权限栈” | 分级作用域集合，层级之间通过逐步授权过渡 |
| 客户端 ID 元数据 | “客户端发现文档” | 客户端发布自身 OAuth 元数据的 URL |

## 进一步阅读

- [MCP — 授权规范](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 标准 MCP OAuth 配置文件 (OAuth Profile)
- [den.dev — MCP 11月授权规范](https://den.dev/blog/mcp-november-authorization-spec/) — 2025年11月25日变更内容的逐步解读
- [RFC 8707 — OAuth 2.0 资源指示器](https://datatracker.ietf.org/doc/html/rfc8707) — 定义受众绑定 (Audience Pinning) 的 RFC 文档
- [RFC 9728 — OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) — 定义发现文档 (Discovery Document) 的 RFC 文档
- [Aembit — MCP OAuth 2.1、PKCE 与 AI 授权的未来](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 升级验证流程 (Step-up Flow) 的实操指南