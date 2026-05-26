# 生产环境中的 MCP 身份验证 (MCP Auth) — 基于 iii 原语 (iii Primitives) 的动态客户端注册 (DCR)、JWKS 密钥轮换 (JWKS Rotation) 与受众绑定令牌 (Audience-Pinned Tokens)

> 第 16 课在内存中构建了 OAuth 2.1 状态机 (OAuth 2.1 State Machine)。到 2026 年，你交付给真实企业的每个 MCP 服务器都将部署在生产级身份验证之后：动态客户端注册 (Dynamic Client Registration, RFC 7591)、授权服务器元数据发现 (Authorization Server Metadata Discovery, RFC 8414)、不会导致凌晨 3 点令牌验证中断的 JWKS 密钥轮换，以及拒绝混淆代理 (Confused Deputy) 重用的受众绑定令牌。本课将通过 iii 原语将所有这些功能串联起来——使用 `iii.registerTrigger` 处理 HTTP 和 cron，使用 `iii.registerFunction` 处理身份验证逻辑，使用 `state::set/get` 管理缓存密钥——从而使身份验证层具备可观测性、可重启性和可重放性，与引擎中的其他工作负载保持一致。

**类型：** 构建实践
**语言：** Python（标准库，iii 原语已针对本课环境进行模拟）
**前置条件：** 第 13 阶段 · 第 16 课（OAuth 2.1 状态机）、第 13 阶段 · 第 17 课（网关）
**预计耗时：** 约 90 分钟

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器并验证其接口契约。
- 实现 RFC 7591 动态客户端注册，使 MCP 客户端无需管理员干预即可完成注册。
- 使用 cron 触发器缓存并轮换 JWKS 密钥，确保密钥切换期间签名验证不受影响。
- 使用 RFC 8707 资源指示符将令牌绑定至单一 MCP 资源，防止混淆代理重用。
- 将每个端点和后台作业封装为 iii 原语——包括 HTTP 触发器、cron 触发器、命名函数以及 `state::*` 读取操作——确保单次重启即可重建整个身份验证层。
- 解读身份提供商 (Identity Provider, IdP) 能力矩阵，当 IdP 无法满足 MCP 的身份验证配置要求时拒绝部署。

## 问题背景

第 16 课的模拟器在内存中运行 OAuth 2.1。实际生产环境存在三个运维缺口（operational gaps），这是纯内存模拟器无法察觉的。

第一个缺口是注册（enrollment）。真实组织会运行数百个 MCP（Model Context Protocol）服务器和数千个 MCP 客户端。运维人员不会手动将每个 Cursor 用户注册为 OAuth 客户端。RFC 7591 动态客户端注册（dynamic client registration）允许客户端向授权服务器（authorization server）发送 `POST /register` 请求，并即时获取 `client_id`（以及可选的 `client_secret`）。服务器会在其 RFC 8414 元数据中发布 `registration_endpoint`；客户端无需带外配置（out-of-band configuration）即可自动发现该端点。

第二个缺口是密钥轮换（key rotation）。JWT（JSON Web Token）验证依赖于授权服务器发布的签名密钥，这些密钥以 JSON Web 密钥集（JSON Web Key Set, JWKS）的形式公开。授权服务器会按计划轮换这些密钥（通常每小时一次，在应急响应期间可能更快）。如果 MCP 服务器仅在启动时获取一次 JWKS，那么在轮换窗口期之前验证一切正常——但轮换发生后，所有请求都会失败，直到服务器重启。生产环境会将 JWKS 作为缓存值，并配置一个刷新任务，在旧密钥过期前覆盖缓存；同时，当收到由比缓存更新的密钥签名的令牌时，还会在缓存未命中（cache miss）时触发回退获取机制。

第三个缺口是受众绑定（audience binding）。第 16 课介绍了 RFC 8707 资源指示符（resource indicators）。在生产环境中，该指示符会成为对每个请求的硬性声明检查（hard claim check）。MCP 服务器会将 `token.aud` 与其自身的规范资源 URL 进行比对，若不一致则返回 HTTP 401 拒绝请求。这是防止上游 MCP 服务器（或持有本应发给某台服务器令牌的恶意客户端）在同一信任网格（trust mesh）中将该令牌重放（replaying）到其他服务器的唯一防御手段。

本课将上述每一个缺口都视为 iii 原语（iii primitive）。元数据文档是一个 HTTP 触发器（HTTP trigger），用于返回函数的输出。JWKS 轮换是一个定时触发器（cron trigger），它会调用 `auth::rotate-jwks`，并将结果写入 `state::set("auth/jwks/<issuer>", ...)`。JWT 验证是一个函数，其他组件可通过 `iii.trigger("auth::validate-jwt", token)` 来调用它。MCP 服务器本身也只是另一个 HTTP 触发器，在分发请求前会先调用验证逻辑。重启引擎后：触发器注册表（trigger registry）会重建；状态数据得以保留；认证服务面（auth surface）无需人工干预即可恢复运行。

## 核心概念

### RFC 8414 — OAuth 授权服务器元数据 (OAuth Authorization Server Metadata)

位于 `/.well-known/oauth-authorization-server` 的文档描述了客户端所需的一切信息：

{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}

当客户端获得一个 MCP 资源 URL 时，会进行链式发现：RFC 9728 中的 `oauth-protected-resource`（资源服务器文档）指明了颁发者 (issuer)，随后 `oauth-authorization-server`（本 RFC）列出了所有端点。客户端绝不会硬编码授权 URL。

在信任某个身份提供商 (Identity Provider, IdP) 用于 MCP 之前，你需要验证的契约如下：

- `code_challenge_methods_supported` 包含 `S256`（符合 RFC 7636 的 PKCE 规范）。
- `grant_types_supported` 包含 `authorization_code`，并拒绝 `password` 和 `implicit`。
- 存在 `registration_endpoint`（支持 RFC 7591）。
- 对于 OAuth 2.1，`response_types_supported` 必须严格为 `["code"]`。

如果缺少其中任何一项，MCP 服务器将拒绝针对该 IdP 进行部署。这是部署清单 (deployment manifest) 配置错误，而非代码问题。

### RFC 9728（回顾）— 受保护资源元数据 (Protected Resource Metadata)

第 16 课已涵盖 RFC 9728。在生产环境中的关键差异在于：该文档是客户端查找*此* MCP 服务器所信任的授权服务器的唯一位置。单个 MCP 服务器可能接受来自多个 IdP 的令牌（例如一个用于内部员工，一个用于合作伙伴）。RFC 9728 声明了该集合；RFC 8414 则记录了每个 IdP 支持的具体功能。

{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}

### RFC 7591 — 动态客户端注册 (Dynamic Client Registration)

如果没有动态客户端注册 (Dynamic Client Registration, DCR)，每个 MCP 客户端（如 Cursor、Claude Desktop 或自定义代理）都需要与 IdP 管理员进行带外 (out-of-band) 交换。启用 DCR 后，客户端只需发送 POST 请求：

POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}

服务器将返回 `client_id` 以及用于后续更新的 `registration_access_token`：

{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}

对于运行在用户设备上的 MCP 客户端，`token_endpoint_auth_method: none` 是正确的默认配置。它们仅获取 `client_id`——没有可被窃取 (exfiltrate) 的 `client_secret`。PKCE 提供了公共客户端所需的持有证明 (proof-of-possession)。

生产环境中的三个常见陷阱：

- 注册端点必须按源 IP 进行速率限制 (rate-limit)。否则，恶意攻击者可通过脚本发起数百万次虚假注册，耗尽 `client_id` 命名空间。`iii` 框架使这变得非常简单：注册 HTTP 触发器在分发给注册器之前，会先调用 `auth::rate-limit` 函数。
- 某些企业级 IdP 要求提供 `software_statement`（用于担保客户端身份的已签名 JWT）。课程中的模拟环境跳过了此步骤；但在生产环境中，必须接入验证环节，拒绝除 localhost 重定向 URI 之外的所有未签名注册请求。
- `registration_access_token` 必须以哈希形式存储，而非明文。该令牌一旦失窃，攻击者即可篡改客户端的重定向 URI。

### RFC 8707（回顾）— 资源指示符 (Resource Indicators)

第 16 课已确立了其基本结构。生产环境规则：每个令牌请求都必须包含 `resource=<canonical-mcp-url>`，且 MCP 服务器在每次调用时都会验证 `token.aud` 是否与其自身的资源 URL 匹配。如果 MCP 服务器的访问地址为 `https://notes.example.com/mcp`，则规范 URL (canonical URL) 应为 `https://notes.example.com`——排除路径组件是为了让单个服务器能在同一受众 (audience) 下托管多个路径。

### RFC 7636（回顾）— PKCE

PKCE 在 OAuth 2.1 中是强制要求的。课程中的授权码流程始终携带 `code_challenge` 和 `code_verifier`。服务器会拒绝任何缺少验证器 (verifier) 或验证器哈希值与存储的挑战值 (challenge) 不匹配的令牌请求。

### MCP 规范 2025-11-25 身份验证配置文件 (Auth Profile)

MCP 规范（2025-11-25）明确规定了 MCP 服务器授权层必须执行的操作：

- 发布 `/.well-known/oauth-protected-resource`（RFC 9728）。
- 仅通过 `Authorization: Bearer ...` 接受令牌。
- 针对每个请求验证 `aud`、`iss`、`exp` 以及所需的权限范围 (scopes)。
- 对于所有 401 和 403 响应，返回携带 `Bearer error=...` 的 `WWW-Authenticate` 头，并在适用时包含 `scope=` 和 `resource=` 参数。
- 拒绝 `aud` 与规范资源不匹配的令牌。
- 拒绝 `iss` 不在受保护资源元数据的 `authorization_servers` 列表中的令牌。

OAuth 2.1 草案是底层基础；RFC 8414/7591/8707/9728 与 RFC 7636 构成了表层协议；而 MCP 规范则是具体的应用配置文件 (profile)。

### IdP 能力矩阵

并非所有 IdP 都支持完整的 MCP 配置文件。下表记录了截至 2025-11-25 规范的实际能力声明。它是一个*部署门槛 (deployment gate)*，而非推荐建议。

| IdP 类别 | RFC 8414 元数据 | RFC 7591 DCR | RFC 8707 资源 | RFC 7636 S256 PKCE | 备注 |
|---|---|---|---|---|---|
| 自托管 (Self-hosted, Keycloak) | 是 | 是 | 是（自 24.x 起） | 是 | 本课程 MCP 配置文件的参考 IdP；端到端支持所有 RFC。 |
| 企业级 SSO (Enterprise SSO, Microsoft Entra ID) | 是 | 是（高级版层级） | 是 | 是 | DCR 的可用性因租户层级而异；部署前请在目标租户中验证。 |
| 企业级 SSO (Enterprise SSO, Okta) | 是 | 是（Okta CIC / Auth0） | 是 | 是 | DCR 在 Auth0（现为 Okta CIC）上可用；经典 Okta 组织需要管理员预先注册。 |
| 社交登录 IdP (Social login IdPs, 通用) | 视情况而定 | 极少 | 极少 | 是 | 大多数社交 IdP 将客户端视为静态合作伙伴；不要依赖 DCR。仅将其用作身份源，并在其之上构建你自己的 MCP 感知授权服务器。 |
| 自定义/自研 (Custom / homegrown) | 视情况而定 | 视情况而定 | 视情况而定 | 视情况而定 | 如果你自行开发，请提供完整的配置文件。跳过上述四个 RFC 中的任何一个都会破坏 MCP 身份验证契约。 |

部署清单的拒绝规则：如果所选 IdP 未返回 `registration_endpoint` 且未在 `code_challenge_methods_supported` 中列出 `S256`，MCP 服务器将拒绝启动。不存在降级运行模式。

### 结合 iii 的 JWKS 轮换模式

生产环境中的典型故障模式是 JSON Web 密钥集 (JSON Web Key Set, JWKS) 缓存过期。可通过定时触发器 (cron trigger) 和 `state::*` 缓存来解决：

iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)

每六小时，定时触发器会调用 `auth::rotate-jwks`，该函数获取 `<issuer>/.well-known/jwks.json` 并将其写入 `state::set("auth/jwks/<issuer>", {keys, fetched_at})`。验证器则通过 `state::get` 读取数据。如果缓存中缺少某个令牌的 `kid`，将作为后备机制同步触发 `auth::rotate-jwks` 调用。这同时处理了两种情况：计划轮换（定时任务）和密钥重叠窗口（同步后备）。

状态数据结构如下：

{
  "auth/jwks/https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}

同时保留两个密钥是稳态。授权服务器在停用旧密钥（`k_2026_03`）之前会先引入新密钥（`k_2026_04`）进行轮换，因此使用旧密钥签发的令牌在过期前依然有效。缓存保存的是两者的并集；验证器根据 `kid` 进行匹配。

### iii 原语接线（本课的实际核心内容）

五个原语 (primitives) 构成了身份验证层：

# 1. RFC 8414 metadata document
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591 dynamic client registration
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT validation as a callable function (the resource server triggers it)
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. Step-up issuance for incremental scope (SEP-835 from L16)
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. Cron-driven JWKS rotation
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)

MCP 服务器本身从不直接调用验证逻辑。它的做法是：

result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}

这种间接调用正是 `iii` 框架的设计赌注。明天你可以将验证器替换为并行查询两个 IdP 的扇出 (fanout) 逻辑，或者添加跨度发射器 (span emitter)，亦或缓存成功的验证结果。而 MCP 服务器本身无需任何改动。

### 结合受众绑定的混淆代理 (Confused Deputy) 攻击演练

服务器 A（`notes.example.com`）和服务器 B（`tasks.example.com`）均在同一个授权服务器上注册。服务器 A 遭到入侵。攻击者获取了用户的笔记令牌，并将其重放 (replay) 给服务器 B。

服务器 B 的验证器执行以下操作：

1. 解码 JWT，根据 `kid` 获取 JWKS，验证签名。
2. 对照其受保护资源元数据中的 `authorization_servers` 检查 `iss`。（通过——同一 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败——令牌的 `aud` 为 `https://notes.example.com`。）
4. 返回 401 状态码，并附带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch"`。

在协议层，受众声明 (audience claim) 是防御此类攻击的唯一手段。为了性能而跳过它是生产环境中最常见的错误；验证器必须在每次请求时运行，而不仅仅是在会话开始时。

### 故障模式

- **JWKS 缓存过期。** 密钥轮换后，验证器会拒绝有效的令牌。修复方法是采用上述的定时任务+后备模式。切勿在没有刷新任务的情况下缓存 JWKS。
- **缺少 `aud` 声明。** 某些 IdP 默认会省略 `aud`，除非令牌请求中包含 `resource`。验证器必须拒绝缺少 `aud` 的令牌，绝不能将缺失视为通配符。
- **权限升级竞态条件。** 针对同一用户的两个并发逐步升级 (step-up) 流程可能同时成功，并生成两个具有不同权限范围的访问令牌。验证器必须使用请求中提供的令牌，而不是去查询“用户当前的权限范围”——这会引发时间差攻击 (Time-of-Check to Time-of-Use, TOCTOU) 窗口。
- **注册令牌失窃。** 泄露的 `registration_access_token` 允许攻击者重写重定向 URI。必须在静态存储时对其进行哈希处理；要求客户端在每次更新时提供明文；一旦怀疑泄露立即轮换。
- **未固定 `iss`。** 接受任意 `iss` 的验证器会让攻击者能够搭建自己的授权服务器，为目标受众注册客户端并签发令牌。受保护资源元数据中的 `authorization_servers` 列表就是白名单；必须严格执行。

## 使用它

`code/main.py` 使用 Python 标准库 (stdlib) 和一个小型 `iii_mock` 注册表 (registry) 逐步演示了完整的生产环境流程，该注册表模拟了 `iii.registerFunction`、`iii.registerTrigger`、`iii.trigger` 以及 `state::set/get`。流程如下：

1. 授权服务器 (authorization server) 在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据 (metadata)。
2. MCP 客户端 (MCP client) 调用元数据端点，发现注册端点 (registration endpoint)。
3. MCP 客户端向 `/register` 发送 POST 请求（遵循 RFC 7591），并接收一个 `client_id`。
4. MCP 客户端运行受 PKCE 保护的授权码流程 (PKCE-protected authorization code flow)（遵循 RFC 7636），并附带资源指示器 (resource indicator)（遵循 RFC 8707）。
5. MCP 客户端使用 `Authorization: Bearer ...` 调用 MCP 服务器上的工具。
6. MCP 服务器触发 `auth::validate-jwt`，该函数从 `state::get` 读取 JWKS。
7. 定时触发器 (cron trigger) 触发 `auth::rotate-jwks`，替换状态中的 JWKS。
8. 下一次调用将针对新密钥进行验证，无需重启服务。
9. 针对其他 MCP 资源的混淆代理攻击尝试 (confused-deputy attempt) 将因受众不匹配 (audience mismatch) 返回 401 状态码。

此处的模拟 JWT (mock JWT) 采用 HS256 算法与共享密钥 (shared secret)（因此本教程仅依赖标准库即可运行）。生产环境使用 RS256 或 EdDSA 算法配合上述 JWKS 模式；除此之外，验证逻辑完全相同。

## 发布上线

本教程将生成 `outputs/skill-mcp-auth-iii.md`。给定 MCP 服务器配置和身份提供商 (Identity Provider, IdP) 能力集，该技能将输出用于注册的 `iii` 原语 (iii primitives)、JWKS 轮换计划 (JWKS rotation schedule)、作用域映射 (scope mapping)，以及当 IdP 不支持完整 RFC 规范配置文件 (RFC profile) 时需应用的拒绝规则 (refusal rules)。

## 练习

1. 运行 `code/main.py`。跟踪上述 9 步流程。注意观察 `state::get` 在 `auth::rotate-jwks` 覆盖它之前立即返回旧数据的位置，以及下一次请求如何针对新密钥进行验证。

2. 在受保护资源元数据 (protected-resource metadata) 的 `authorization_servers` 列表中添加一个新的 IdP。签发一个由新 IdP 签名的令牌，并确认验证器接受它。签发一个由未列出的 IdP 签名的令牌，并确认验证器拒绝该请求，返回 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`。

3. 将 `auth::rate-limit` 实现为一个 `iii` 函数，并在注册 HTTP 触发器运行注册逻辑之前从内部调用它。使用存储在 `state::set("auth/ratelimit/<ip>", ...)` 中的按源 IP 划分的令牌桶 (token bucket)。

4. 阅读 RFC 7591，找出本教程 `/register` 处理器未验证的两个字段。添加相应的验证逻辑。（提示：`software_statement` 和 `redirect_uris` 的 URI 方案。）

5. 阅读 MCP 规范 2025-11-25 的授权部分。找出本教程验证器当前未发出的关于 `WWW-Authenticate` 标头的一项规范性要求 (normative requirement)。将其添加进去。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| ASM (Authorization Server Metadata) | “OAuth 元数据文档” | RFC 8414 定义的 `/.well-known/oauth-authorization-server` 端点返回的 JSON 数据 |
| DCR (Dynamic Client Registration) | “自助式客户端注册” | RFC 7591 规定的 `POST /register` 注册流程 |
| JWKS (JSON Web Key Set) | “用于 JWT 验证的公钥” | 从 `jwks_uri` 获取的 JSON Web 密钥集，通过 `kid` 进行索引 |
| 资源指示器 (Resource Indicator) | “受众参数” | RFC 8707 定义的 `resource` 参数，用于将令牌绑定至特定服务器 |
| `aud` 声明 (Audience Claim) | “受众” | JWT 中的声明字段，验证器会将其与规范化的资源 URL 进行比对 |
| 混淆代理攻击 (Confused Deputy) | “令牌重放” | 一种攻击手法：将原本为服务器 A 签发的令牌提交给服务器 B 使用 |
| `iss` 允许列表 (Issuer Allow-list) | “受信任的授权服务器” | 受保护资源元数据中 `authorization_servers` 字段所列出的服务器集合 |
| 密钥轮换 (Key Rotation) | “滚动更新 JWKS” | 定期替换签名密钥，并设置重叠过渡期 |
| 公共客户端 (Public Client) | “原生应用或浏览器客户端” | 不持有 `client_secret` 的 OAuth 客户端；通过 PKCE 机制进行防护 |
| `WWW-Authenticate` | “401/403 响应头” | 携带 `Bearer error=...` 指令，用于引导客户端执行恢复流程 |

## 进一步阅读

- [MCP — 授权规范 (2025-11-25)](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 本课程所实现的 MCP 授权配置文件
- [RFC 8414 — OAuth 2.0 授权服务器元数据](https://datatracker.ietf.org/doc/html/rfc8414) — 服务发现契约
- [RFC 7591 — OAuth 2.0 动态客户端注册协议](https://datatracker.ietf.org/doc/html/rfc7591) — DCR
- [RFC 7636 — 代码交换证明密钥 (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端的持有者证明机制
- [RFC 8707 — OAuth 2.0 资源指示器](https://datatracker.ietf.org/doc/html/rfc8707) — 受众绑定
- [RFC 9728 — OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现规范
- [OAuth 2.1 草案](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 整合后的 OAuth 基础协议