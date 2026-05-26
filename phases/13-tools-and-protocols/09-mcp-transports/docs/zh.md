# MCP（Model Context Protocol）传输协议 — 标准输入输出（stdio）对比 可流式 HTTP（Streamable HTTP）与 服务器推送事件（SSE）迁移

> 标准输入输出（stdio）仅适用于本地环境。可流式 HTTP（Streamable HTTP）（2025-03-26 规范）是远程通信的标准。旧的 HTTP+SSE 传输协议已被弃用，并将于 2026 年中期被移除。选错传输协议将导致额外的迁移成本；选对协议则能获得支持远程托管、具备会话连续性且能防御 DNS 重绑定（DNS-rebinding）的 MCP 服务器。

**Type:** 学习
**Languages:** Python（标准库，Streamable HTTP 端点骨架）
**Prerequisites:** 第 13 阶段 · 07, 08（MCP 服务器与客户端）
**Time:** 约 45 分钟

## 学习目标

- 根据部署形态（本地与远程、单进程与集群）在 stdio 与 Streamable HTTP 之间做出选择。
- 实现 Streamable HTTP 单端点模式：使用 POST 处理请求，使用 GET 建立会话流。
- 强制执行 `Origin` 验证与会话 ID 语义，以防御 DNS 重绑定攻击。
- 在 2026 年中期的废弃移除截止日期前，将传统的 HTTP+SSE 服务器迁移至 Streamable HTTP。

## 问题背景

首个 MCP 远程传输协议（2024-11 版）采用的是 HTTP+SSE 方案：包含两个端点，一个用于接收客户端的 POST 请求，另一个作为服务器推送事件通道用于向客户端推送数据流。该方案虽然可行，但架构较为笨重：每个会话需占用两个端点，部分内容分发网络（CDN）前的缓存机制容易失效，且严重依赖长连接的 SSE 通道，而某些 Web 应用防火墙（WAF）会主动且激进地切断此类连接。

2025-03-26 规范已将其替换为 Streamable HTTP：仅需一个端点，POST 用于客户端请求，GET 用于建立会话流，两者共享 `Mcp-Session-Id` 请求头。自该规范发布以来，所有新建或迁移的服务器均采用 Streamable HTTP。旧的 SSE 模式正逐步被废弃——Atlassian Rovo 已于 2026 年 6 月 30 日移除该功能；Keboola 于 2026 年 4 月 1 日移除；其余大多数企业级服务器也将在 2026 年底前完成移除。

此外，stdio 对于本地服务器依然至关重要。Claude Desktop、VS Code 以及各类集成开发环境（IDE）形态的客户端均通过 stdio 启动服务器。正确的认知模型是：stdio 用于“本机通信”，Streamable HTTP 用于“网络通信”。两者互不交叉。

## 核心概念

### stdio

- 子进程传输（Child-process transport）。客户端生成服务器进程，通过标准输入/标准输出（stdin/stdout）进行通信。
- 每行一个 JSON 对象。以换行符分隔。
- 无会话 ID（Session ID）；进程身份即代表会话。
- 无需身份验证（子进程继承父进程的信任边界）。
- 切勿用于远程服务器——若需使用 SSH 或 socat 进行隧道转发，此时应直接使用 Streamable HTTP。

### Streamable HTTP

单一端点 `/mcp`（或任意路径）。支持三种 HTTP 方法：

- **POST /mcp。** 客户端发送 JSON-RPC 消息。服务器回复单个 JSON 响应，或包含一个或多个响应的 SSE（Server-Sent Events）流（适用于批处理响应及与该请求相关的通知）。
- **GET /mcp。** 客户端建立长连接 SSE 通道。服务器用于向客户端发起请求（采样、通知、引导/elicitation）。
- **DELETE /mcp。** 客户端显式终止会话。

会话通过服务器在首次响应中设置的 `Mcp-Session-Id` 头部进行标识，客户端在后续每次请求中需回显该头部。会话 ID 必须具备密码学随机性（128 位以上）；出于安全考虑，服务器将拒绝客户端自行指定的 ID。

### 单端点与双端点

旧版规范中的双端点模式在 2026 年仍可调用——规范将其声明为“向后兼容（legacy compatible）”。但所有新服务器均应采用单端点模式。官方 SDK 默认输出单端点配置；仅在与尚未迁移的远程服务器通信时才使用旧版模式。

### `Origin` 验证与 DNS 重绑定（DNS-rebinding）

浏览器目前并非 MCP 客户端，但攻击者可构造恶意网页，诱使浏览器向 `localhost:1234/mcp`（用户本地 MCP 服务器的监听地址）发送 POST 请求。若服务器未校验 `Origin` 头部，浏览器的同源策略（Same-Origin Policy）将无法提供保护，因为 `Origin: http://evil.com` 属于合法的跨域请求。

2025-11-25 版规范要求服务器拒绝 `Origin` 不在白名单中的请求。该白名单通常包含 MCP 客户端主机（如 `https://claude.ai`、`vscode-webview://*`）以及用于本地 UI 的 localhost 变体。

### 会话 ID 生命周期

1. 客户端发送首个请求，不携带 `Mcp-Session-Id`。
2. 服务器分配随机 ID，并在响应头中设置 `Mcp-Session-Id`。
3. 客户端在后续所有请求及用于建立流的 `GET /mcp` 请求中回显该头部。
4. 服务器可撤销会话；客户端在后续请求中将收到 404 状态码，并必须重新初始化。
5. 客户端可显式发送 DELETE 请求以优雅关闭会话。

### 保活与重连

SSE 连接可能会断开。客户端通过携带相同的 `Mcp-Session-Id` 重新发起 GET 请求来重建连接。服务器必须将中断期间遗漏的事件加入队列（在合理的时间窗口内），并通过客户端回显的 `last-event-id` 头部进行重放。

Phase 13 · 13 介绍了任务（Tasks）机制，该机制允许长时间运行的工作即使在完整会话重连后也能继续存活。

### 向后兼容性探测

若客户端希望同时兼容新旧版服务器：

1. 向 `/mcp` 发送 POST 请求。
2. 若响应为 `200 OK` 且返回 JSON 或 SSE，则为 Streamable HTTP。
3. 若响应为 `200 OK`，包含 `Content-Type: text/event-stream` 且带有指向次级端点的 `Location` 头部，则为旧版 HTTP+SSE 模式；客户端应跟随 `Location` 指示。

### Cloudflare、ngrok 与托管部署

2026 年的生产级远程 MCP 服务器通常运行于 Cloudflare Workers（配合其 MCP Agents SDK）、Vercel Functions 或容器化的 Node/Python 环境中。关键点：托管平台必须支持用于 SSE GET 请求的长连接 HTTP。Vercel 免费版限制为 10 秒，不适用此场景。Cloudflare Workers 支持无限期流式传输。

### 网关组合

当使用网关代理多个 MCP 服务器时（Phase 13 · 17），网关表现为单一的 Streamable HTTP 端点，负责重写会话 ID 并对上游进行多路复用。工具在网关层进行合并；客户端仅看到一个逻辑服务器。

### 传输层故障模式

- **stdio SIGPIPE。** 子进程在写入中途崩溃会触发 SIGPIPE 信号；服务器应优雅退出。客户端应检测 EOF（文件结束符）并将会话标记为已失效。
- **HTTP 502 / 504。** Cloudflare、nginx 及其他代理在上游故障时会返回这些状态码。Streamable HTTP 客户端应在短暂退避（backoff）后重试一次。
- **SSE 连接断开。** TCP RST、代理超时或客户端网络切换会导致流关闭。客户端需携带 `Mcp-Session-Id` 及可选的 `last-event-id` 重新连接以恢复状态。
- **会话撤销。** 服务器使会话 ID 失效；客户端在下一次请求时将收到 404。客户端必须重新握手。
- **时钟偏差（Clock skew）。** 客户端的资源 TTL（Time-To-Live）计算与服务器产生偏差。客户端应以服务器时间戳为准。

### 何时绕过 Streamable HTTP

部分企业在内部网络中将 MCP 服务器部署于 gRPC 或消息队列传输层之后。这属于非标准做法——MCP 规范并未正式定义此类传输方式。网关可向 MCP 客户端暴露 Streamable HTTP 接口，同时在内部使用 gRPC。请确保对外接口符合规范；协议转换由网关负责。

## 使用方法

`code/main.py` 使用 `http.server`（标准库 (stdlib)）实现了一个极简的可流式传输 HTTP (Streamable HTTP) 端点。它在 `/mcp` 路径上处理 POST、GET 和 DELETE 请求，在首次响应时设置 `Mcp-Session-Id`，验证 `Origin`（来源 (Origin)）请求头，并拒绝来自未加入白名单来源的请求。该处理程序复用了第 07 课笔记服务器的分发逻辑。

需要关注的要点：

- POST 处理程序读取 JSON-RPC 请求体，进行路由分发，并写入 JSON 响应（此为单响应变体；服务器发送事件 (SSE) 变体在结构上与之类似）。
- `Origin` 检查会拒绝默认的 `http://evil.example` 探测请求，但接受 `http://localhost`。
- 会话 ID (Session ID) 为随机的 128 位十六进制字符串；服务器在内存中维护每个会话的状态。

## 交付与部署

本课将生成 `outputs/skill-mcp-transport-migrator.md`。针对基于 HTTP+SSE（旧版）的 MCP 服务器，该技能将生成一份迁移至可流式传输 HTTP 的方案，涵盖会话 ID 连续性、`Origin` 检查以及向后兼容的探测支持。

## 练习

1. 运行 `code/main.py`。使用 `curl` 发送 `initialize` 的 POST 请求，并观察响应头中的 `Mcp-Session-Id`。发送第二个 POST 请求并回显该请求头，以验证会话连续性。

2. 添加一个 GET 处理程序以开启 SSE 流。每五秒发送一次 `notifications/progress` 事件。使用相同的会话 ID 重新发起 GET 请求进行重连，并确认服务器接受该连接。

3. 实现 `last-event-id` 重放逻辑。在重连时，重放自该 ID 之后生成的所有事件。

4. 扩展 `Origin` 验证逻辑以支持通配符模式（`https://*.example.com`），并确认其接受 `https://app.example.com` 但拒绝 `https://evil.example.com.attacker.net`。

5. 从官方注册表中选取一个旧版 HTTP+SSE 服务器（此类服务器有多个），并草拟迁移方案：说明在端点处理、会话 ID 生成以及请求头语义方面需要做出哪些变更。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 标准输入输出传输 (stdio transport) | “本地子进程” | 基于标准输入/输出 (stdin/stdout) 的 JSON-RPC，以换行符分隔 |
| 可流式传输 HTTP (Streamable HTTP) | “远程传输” | 单端点 POST + GET + 可选 SSE，遵循 2025-03-26 规范 |
| HTTP+SSE | “旧版” | 双端点模型，将于 2026 年中移除 |
| `Mcp-Session-Id` | “会话请求头” | 服务器分配的随机 ID，在后续每次请求中回显 |
| `Origin` 白名单 (`Origin` allowlist) | “DNS 重绑定防御” | 拒绝来源 (Origin) 未经批准的请求 |
| 单端点 (Single endpoint) | “单一 URL” | `/mcp` 统一处理所有会话操作的 POST / GET / DELETE |
| `last-event-id` | “SSE 重放” | 用于恢复中断的流且确保不丢失事件的请求头 |
| 向后兼容探测 (Backwards-compat probe) | “新旧版本检测” | 客户端通过检查响应结构自动选择传输方式 |
| 长连接 HTTP (Long-lived HTTP) | “SSE 流式传输” | 服务器在单个 TCP 连接上持续推送事件数分钟或数小时 |
| 会话撤销 (Session revocation) | “强制重新初始化” | 服务器使会话 ID 失效；客户端必须重新握手 |

## 进一步阅读

- [模型上下文协议（MCP）— 基础传输规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — 标准输入/输出（stdio）与可流式传输 HTTP（Streamable HTTP）的权威参考
- [MCP — 基础传输规范 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的修订版本
- [Cloudflare — MCP 传输](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — 基于 Workers（Cloudflare Workers）托管的 Streamable HTTP 模式
- [AWS — MCP 传输机制](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 跨不同部署形态（deployment shapes）的对比
- [Atlassian — HTTP+SSE（服务器发送事件）弃用通知](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体的迁移截止日期示例