---
name: mcp-apps-spec
description: 为需要交互式 UI（用户界面）资源的工具生成完整的 MCP Apps 合约。
version: 1.0.0
phase: 13
lesson: 14
tags: [mcp, apps, ui-resources, csp, iframe-sandbox]
---

针对适合采用交互式 UI（时间轴、表单、仪表盘、地图、图表）的工具，生成 MCP Apps 合约。

需生成以下内容：

1. `ui://` URI（统一资源标识符）。为 UI 资源指定一个规范名称（例如 `ui://notes/timeline`）。
2. 工具结果结构。包含 `text` 前言和 `ui_resource` 块的 `content[]` 数组；需填充 `_meta.ui` 字段。
3. CSP（内容安全策略，Content Security Policy）。为 `default-src`、`script-src`、`connect-src`、`img-src`、`style-src` 配置最小允许列表（allowlist）。除非必要，否则避免使用 `'unsafe-inline'`。
4. 权限列表。按需申请摄像头、麦克风、地理位置或网络权限；若无需则留空。
5. `postMessage` 入口点。明确 UI 将调用哪些 `host.*` 方法及其返回值。
6. 安全检查清单。需与宿主环境（host）区分、防止点击劫持（clickjacking）、严格限制 `connect-src`，若渲染任何用户内容则必须进行 HTML 净化（sanitization）。

硬性拒绝条件：
- CSP 配置为 `default-src *`。存在极大的安全风险。
- 申请超出 UI 实际使用范围的任何 `permissions`。遵循最小权限原则。
- 任何加载外部脚本的 `ui://` 资源。必须打包（bundle）或直接拒绝。
- 任何未经净化即渲染用户可控 HTML 的 UI。存在 XSS（跨站脚本攻击）向量。

拒绝规则：
- 若 UI 仅为静态结果展示，则拒绝搭建（scaffold）应用；直接返回文本内容。
- 若该工具更适合使用宿主原生组件（如进度条、确认对话框），则建议优先采用这些组件。
- 若宿主环境尚未支持 MCP Apps（截至 2026 年 4 月的 VS Code 稳定版、Zed、Windsurf），需标记降级至文本（fallback-to-text）的路径。

输出要求：一份单页合约，包含 `ui://` URI、工具结果 JSON、CSP、权限列表、`postMessage` 入口点及安全检查清单。最后用一句话说明能够渲染此 UI 的最低版本宿主环境。