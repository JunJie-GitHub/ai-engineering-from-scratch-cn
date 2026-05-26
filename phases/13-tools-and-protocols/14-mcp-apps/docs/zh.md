# MCP 应用（MCP Apps）— 基于 `ui://` 的交互式 UI 资源

> 纯文本的工具输出限制了智能体（Agent）的展示能力。MCP 应用（MCP Apps，SEP-1724，于 2026 年 1 月 26 日正式发布）允许工具返回沙盒化交互式 HTML（sandboxed interactive HTML），并在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中以内联方式渲染。仪表盘、表单、地图、3D 场景，均可通过单一扩展实现。本课程将深入讲解 `ui://` 资源方案（resource scheme）、`text/html;profile=mcp-app` MIME 类型（MIME type）、iframe 沙盒 postMessage 协议（iframe-sandbox postMessage protocol），以及允许服务器渲染 HTML 所带来的安全面（security surface）。

**类型：** 构建（Build）
**语言：** Python（标准库、UI 资源生成器）、HTML（示例应用）
**前置知识：** 第 13 阶段 · 07（MCP 服务器）、第 13 阶段 · 10（资源）
**时长：** 约 75 分钟

## 学习目标

- 从工具调用中返回 `ui://` 资源，并设置正确的 MIME 类型与元数据。
- 使用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具关联的 UI。
- 实现用于 UI 到宿主（host）通信的 iframe 沙盒 postMessage JSON-RPC。
- 应用内容安全策略（Content Security Policy, CSP）和权限策略（permissions-policy）默认值，以防御源自 UI 的攻击。

## 问题背景

在 2025 年，一个 `visualize_timeline` 工具可能只会返回“以下是按时间顺序排列的 14 条笔记：……”。这仅仅是一段文本。而用户真正需要的是交互式时间轴。在 MCP 应用出现之前，可选方案只有两种：依赖特定客户端的组件 API（如 Claude Artifacts、OpenAI Custom GPT HTML），或者完全不提供 UI。

MCP 应用（SEP-1724，于 2026 年 1 月 26 日发布）对此进行了标准化约定。工具返回的结果中包含一个 `resource`，其 URI 为 `ui://...`，MIME 类型为 `text/html;profile=mcp-app`。宿主应用会在受限的 CSP 沙盒 iframe 中渲染该资源，除非明确授权，否则禁止网络访问。iframe 内部的 UI 通过一种精简的 postMessage JSON-RPC 协议变体（dialect）向宿主发送消息。

所有兼容的客户端（Claude Desktop、ChatGPT、Goose、VS Code）都会以相同的方式渲染同一个 `ui://` 资源。一套服务器，一个 HTML 打包文件，通用 UI。

## 核心概念

### `ui://` 资源协议（Resource Scheme）

工具返回：

{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}

随后，宿主（Host）在 `ui://notes/timeline` URI 上调用 `resources/read` 方法，并收到：

{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}

### Iframe 沙箱（Iframe Sandbox）

宿主在沙箱化的 `<iframe>` 中渲染 HTML，并配置以下属性：

- `sandbox="allow-scripts allow-same-origin"`（或根据服务器声明设置更严格的策略）
- 通过响应头应用服务器声明的内容安全策略（Content Security Policy, CSP）。
- 不携带宿主源（Origin）的 Cookie 或 localStorage。
- 网络访问仅限于 CSP 中 `connectSrc` 指定的范围。

### postMessage 协议（postMessage Protocol）

Iframe 通过 `window.postMessage` 与宿主进行通信。这是一种精简版的 JSON-RPC 2.0 方言：

务必将 `targetOrigin` 严格限定为对端的确切源（Origin），接收方在处理任何负载前，需对照白名单验证 `event.origin`。该通道的两端均严禁使用 `"*"`，因为消息体中承载了工具调用和资源读取请求。

// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});

UI 可调用的宿主端方法包括：

- `host.callTool(name, arguments)` — 调用服务器工具。
- `host.readResource(uri)` — 读取 MCP（Model Context Protocol）资源。
- `host.getPrompt(name, arguments)` — 获取提示词模板（Prompt Template）。
- `host.close()` — 关闭/隐藏 UI。

所有调用仍遵循 MCP 协议，并继承服务器的权限设置。

### 权限（Permissions）

`_meta.ui.permissions` 列表用于请求额外能力：

- `camera` — 访问用户摄像头（用于文档扫描类 UI）。
- `microphone` — 语音输入。
- `geolocation` — 地理位置。
- `network:*` — 比仅靠 `connectSrc` 更广泛的网络访问权限。

每项权限都会在 UI 渲染前向用户弹出授权提示。

### 安全风险（Security Risks）

Iframe 中的 HTML 依然是 HTML，这引入了新的攻击面：

- **通过 UI 进行提示词注入（Prompt Injection）。** 恶意服务器 UI 可能显示伪装成系统消息的文本以欺骗用户。宿主渲染时应清晰区分服务器 UI 与宿主 UI。
- **通过 `connectSrc` 窃取数据。** 若 CSP 允许 `connect-src: *`，UI 可将数据发送至任意地址。默认策略应保持严格。
- **点击劫持（Clickjacking）。** UI 可能覆盖宿主界面（Chrome）。宿主必须防止 z-index 篡改并强制执行透明度规则。
- **劫持焦点。** UI 可能抢占键盘焦点并捕获下一条消息。宿主必须进行拦截。

第 13 阶段（共 15 阶段）将作为 MCP 安全的一部分深入探讨这些内容；本节仅作初步介绍。

### `ui/initialize` 握手（Handshake）

Iframe 加载完成后，会通过 postMessage 发送 `ui/initialize` 请求：

{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}

宿主将返回支持的能力列表及会话令牌（Session Token）。UI 在后续每次调用宿主时均需携带该会话令牌。

### AppRenderer / AppFrame SDK 基础原语（Primitives）

ext-apps SDK 提供了两个便捷的基础原语：

- `AppRenderer`（服务器端）—— 封装 React / Vue / Solid 组件，并生成带有正确 MIME 类型和元数据的 `ui://` 资源。
- `AppFrame`（客户端）—— 接收资源、挂载 iframe，并代理 postMessage 通信。

你可以直接使用这些组件，也可以手动编写 HTML 和 JSON-RPC 逻辑。

### 生态现状（Ecosystem Status）

MCP Apps 于 2026 年 1 月 26 日正式发布。截至 2026 年 4 月的客户端支持情况如下：

- **Claude Desktop。** 自 2026 年 1 月起提供完整支持。
- **ChatGPT。** 通过 Apps SDK 提供完整支持（底层采用相同的 MCP Apps 协议）。
- **Cursor。** 测试版（Beta）；需在设置中手动启用。
- **VS Code。** 仅限 Insider 内部版本。
- **Goose。** 完整支持。
- **Zed、Windsurf。** 已列入开发路线图（Roadmapped）。

已投入生产的服务器应用场景包括：仪表盘、地图可视化、数据表格、图表构建器以及沙箱 IDE 预览。

## Use It

`code/main.py` 为笔记服务器扩展了一个 `visualize_timeline` 工具，该工具会返回一个 `ui://notes/timeline` 资源。此外，它还提供了一个针对该 URI 的 `resources/read` 处理器，用于返回一个包含 SVG 时间轴的小型但完整的 HTML 资源包。该 HTML 采用标准库（stdlib）模板生成——无需构建系统。由于标准库无法直接驱动浏览器，`postMessage` 的调用逻辑仅以 JS 注释的形式进行了示意。

重点关注：

- 工具响应中的 `_meta.ui` 字段携带了 `resourceUri`、内容安全策略（CSP）和权限信息。
- 该 HTML 无需网络访问即可渲染；所有数据均已内联。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（在此标准库演示中已记录但处于非激活状态）。

## Ship It

本课时将生成 `outputs/skill-mcp-apps-spec.md` 文件。针对适合添加交互式用户界面（UI）的工具，该技能（skill）会生成完整的 MCP Apps 契约：包括 `ui://` URI、CSP、权限、`postMessage` 入口点以及安全检查清单。

## Exercises

1. 运行 `code/main.py` 并检查生成的 HTML。直接在浏览器中打开该 HTML 文件；验证 SVG 是否正常渲染。随后，草拟 UI 用于调用 `host.callTool("notes_update", ...)` 的 `postMessage` 契约。

2. 收紧 CSP：移除 `'unsafe-inline'` 并改用基于随机数（nonce）的脚本策略。HTML 生成代码需要做哪些相应调整？

3. 添加第二个 UI 资源 `ui://notes/editor`，其中包含一个用于就地编辑笔记的表单。当用户提交时，iframe 将调用 `host.callTool("notes_update", ...)`。

4. 审查该 UI 的攻击面（attack surface）。恶意服务器可能在何处注入内容？iframe 沙箱（sandbox）能防御哪些威胁，又无法防御哪些？

5. 阅读 SEP-1724 规范，并找出 MCP Apps SDK 中的一项本玩具实现（toy implementation）未使用的功能。（提示：组件级状态同步（component-level state sync）。）

## Key Terms

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| MCP Apps | “交互式 UI 资源” | 于 2026-01-26 发布的 SEP-1724 扩展 |
| `ui://` | “应用 URI 方案” | 用于 UI 资源包的资源协议 |
| `text/html;profile=mcp-app` | “该 MIME 类型” | MCP App HTML 的内容类型（Content-Type） |
| Iframe sandbox | “渲染容器” | 结合 CSP 和权限对 UI 进行浏览器沙箱隔离 |
| postMessage JSON-RPC | “UI 到主机的通信线路” | 用于主机调用的轻量级 JSON-RPC-over-postMessage 方言 |
| `_meta.ui` | “工具-UI 绑定” | 将工具结果与 UI 资源关联的元数据 |
| CSP | “内容安全策略” | 声明脚本、网络和样式允许的加载源 |
| AppRenderer | “服务器 SDK 原语” | 将框架组件转换为 `ui://` 资源 |
| AppFrame | “客户端 SDK 原语” | 负责挂载 iframe 并协调 `postMessage` 的辅助工具 |
| `ui/initialize` | “握手协议” | UI 向主机发送的首条 `postMessage` |

## Further Reading

- [MCP 扩展应用 (ext-apps) — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 参考实现 (reference implementation) 与软件开发工具包 (SDK)
- [MCP 应用规范 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 正式规范文档 (formal specification document)
- [MCP — 应用扩展概述](https://modelcontextprotocol.io/extensions/apps/overview) — 高层级文档 (high-level documentation)
- [MCP 博客 — MCP 应用发布](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 年 1 月发布博文
- [MCP 应用 API 参考](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc 风格的 SDK 参考文档