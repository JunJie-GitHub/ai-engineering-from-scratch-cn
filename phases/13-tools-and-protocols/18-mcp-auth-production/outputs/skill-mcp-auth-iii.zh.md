---
name: mcp-auth-iii-wiring
description: 将生产环境 MCP 授权（RFC 8414、7591、8707、7636 PKCE、9728）接入 `iii` 原语——使用 `registerTrigger` 处理 HTTP/cron，使用 `registerFunction` 进行验证，使用 `state::*` 缓存 JWKS。
version: 1.0.0
phase: 13
lesson: 18
tags: [mcp, oauth, dcr, jwks, iii, rfc8414, rfc7591, rfc8707, rfc7636, rfc9728]
---

给定 MCP 服务器配置与身份提供商（Identity Provider, IdP）能力集，输出构成生产环境授权面（Authorization Surface）的 `iii` 原语与拒绝规则。

输入：

- `mcp_resource_url` — 规范资源 URL（不含路径），用作 `aud`（受众）以及受保护资源元数据（Protected-Resource Metadata）中的 `resource` 值。
- `idp_metadata_url` — IdP 的 `/.well-known/oauth-authorization-server` URL。
- `idp_capabilities` — 观测到的 `code_challenge_methods_supported`、`grant_types_supported`、`registration_endpoint`、`response_types_supported` 的值。
- `tools` — MCP 工具列表及其各自所需的权限范围（Scope）。

输出：

1. **拒绝门控（Refusal Gate）**。若以下四个条件中任一不满足，则拒绝接入并终止：
   - `code_challenge_methods_supported` 中缺少 `S256`。
   - `grant_types_supported` 中缺少 `authorization_code`。
   - 缺少 `registration_endpoint`（不支持 RFC 7591 动态客户端注册 DCR）。
   - `response_types_supported` 的值不是精确的 `["code"]`。

2. **受保护资源元数据文档（Protected-Resource Metadata Document）**（RFC 9728），供 MCP 服务器发布在 `/.well-known/oauth-protected-resource`。包含 `resource`、`authorization_servers`（颁发者允许列表）、`scopes_supported`、`bearer_methods_supported: ["header"]`。

3. **`iii` 触发器注册（Trigger Registration）**。逐字输出以下调用：
   - `iii.registerTrigger("http", {"path": "/.well-known/oauth-protected-resource", "method": "GET"}, "auth::serve-protected-resource")`
   - `iii.registerTrigger("http", {"path": "/mcp", "method": "POST"}, "mcp::dispatch")` — 调度器（Dispatcher）在运行任何工具前调用 `iii.trigger("auth::validate-jwt", ...)`。
   - `iii.registerTrigger("cron", {"schedule": "<rotation_schedule>"}, "auth::rotate-jwks")` — 默认调度计划为 `0 */6 * * *`；对于高频轮换的 IdP，可收紧至 `*/15 * * * *`。

4. **`iii` 函数注册（Function Registration）**。逐字输出以下调用：
   - `iii.registerFunction("auth::validate-jwt", handler)` — 校验 `iss`（颁发者）允许列表、基于缓存的 JWKS 验证签名、`aud == mcp_resource_url`、`exp`（过期时间）以及所需权限范围。
   - `iii.registerFunction("auth::rotate-jwks", handler)` — 获取 `jwks_uri`，并写入 `state::set("auth/jwks/<iss>", {keys, fetched_at})`。
   - `iii.registerFunction("auth::serve-protected-resource", handler)` — 返回第 (2) 步中的文档。
   - `iii.registerFunction("auth::issue-step-up", handler)` — 仅当工具列表包含需要用户初始未授予的权限范围才能执行的操作时注册。

5. **状态键规划（State Key Plan）**。每个被接受的颁发者对应一个键：`auth/jwks/<issuer>`，存储 `{keys, fetched_at}`。记录读取模式：验证器从 `state::get` 读取，若发生 `kid`（密钥 ID）未命中，则回退至同步调用 `iii.trigger("auth::rotate-jwks", ...)`。

6. **权限范围映射（Scope Mapping）**。将每个工具映射到其所需的权限范围。输出表格：
   `| tool | required_scope | rationale |`。将破坏性工具（Destructive Tools）归入独立的权限范围；切勿将读取权限范围复用于写入工具。

7. **运行时拒绝规则（Runtime Refusal Rules）**（验证器必须实现这些规则——在处理器主体中输出）：
   - 当 `aud != mcp_resource_url` 时拒绝。
   - 当 `iss` 不在 `authorization_servers` 中时拒绝。
   - 在单次轮换回退后，若 `kid` 仍不在缓存的 JWKS 中，则拒绝。
   - 当缺少所需权限范围时拒绝 → 返回 403 `Bearer error="insufficient_scope", scope="<required>", resource="<mcp_resource_url>"`。
   - 拒绝任何缺少 `code_verifier` 或 `resource` 参数的令牌请求。

硬性拒绝（Hard Reject）（绝不接入以下情况——直接拒绝请求并记录原因）：

- 禁止在 `iii` 状态存储 (State Store) 中以明文形式保存 `client_secret`。公共客户端 (Public Clients) 使用 `token_endpoint_auth_method: none`；机密客户端 (Confidential Clients) 使用 `private_key_jwt`。`state::*` 或注册响应日志中不得出现明文共享密钥。
- 验证器 (Validator) 不得跳过 `aud` 校验。防范混淆代理问题 (Confused Deputy) 正是引入 RFC 8707 与 RFC 9728 的根本原因。
- 不得允许无 PKCE (Proof Key for Code Exchange) 的授权码请求 (Authorization Code Requests)。OAuth 2.1 已明确禁止该做法；验证器必须拒绝任何其存储的授权码记录中缺失 `code_challenge` 的 `/token` 交换请求。
- 禁止在未配置刷新任务 (Refresh Job) 的情况下缓存 JWKS (JSON Web Key Set)。必须随版本交付定时触发器 (Cron Trigger)，否则认证服务面 (Auth Surface) 将不予部署。
- 禁止在未配置允许列表 (Allow-list) 的情况下盲目信任 `iss` 声明。任何接受任意 `iss` 来源令牌的验证器，都会导致攻击者能够自行搭建身份提供商 (IdP) 并伪造令牌。
- 禁止以明文形式存储 `registration_access_token`。必须采用静态哈希 (Hash-at-rest) 机制；每次更新操作时需提供明文。

输出要求：一份单页集成连线方案 (Wiring Plan)，内容需涵盖受保护资源文档 (Protected-Resource Document)、三次 `registerTrigger` 调用、四次 `registerFunction` 调用、状态键规划 (State Key Plan)、作用域映射表 (Scope Mapping Table) 以及编码运行时拒绝规则 (Encoded Runtime Refusal Rules)。文末需明确指出针对所选身份提供商 (IdP) 最可能暴露的单一阻碍部署问题 (Deployment-Blocking Gap)——通常为企业单点登录 (Enterprise SSO) 场景下的动态客户端注册 (DCR) 可用性。