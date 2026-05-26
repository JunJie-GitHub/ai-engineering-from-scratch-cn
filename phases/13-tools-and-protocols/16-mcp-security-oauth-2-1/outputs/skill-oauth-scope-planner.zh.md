---
name: oauth-scope-planner
description: 为远程 MCP（Model Context Protocol）服务器设计 OAuth 2.1 作用域（Scope）集合、固定规则（Pinning Rules）及逐步升级策略（Step-up Policy）。
version: 1.0.0
phase: 13
lesson: 16
tags: [oauth, pkce, resource-indicators, step-up, sep-835]
---

给定一个包含工具列表的远程 MCP 服务器，请设计其授权模型。

输出内容：

1. 作用域（Scope）层级结构。分级作用域集合（例如 `read` -> `write` -> `delete` -> `admin`）。每个操作类别对应一个作用域；避免作用域集合过度膨胀。
2. 作用域到工具的映射关系。为每个工具标注其所需的作用域。标记出任何需要多个作用域的工具。
3. 逐步升级策略（Step-up Policy）。明确哪些操作需要逐步升级授权而非仅依赖初始同意。典型情况：破坏性操作需要逐步升级。
4. 资源指示符（Resource Indicator）值。用于 `resource` 参数的规范 URL。确保该 URL 与 `.well-known/oauth-protected-resource` 中的 `resource` 字段相匹配。
5. 受保护资源元数据（Protected-resource Metadata）。起草 `.well-known/oauth-protected-resource` JSON 文件，需包含 `authorization_servers`、`scopes_supported` 和 `resource` 字段。

硬性拒绝条件：
- 任何需要 `admin` 作用域但在调用时未显示明确确认对话框的工具。必须采用逐步升级授权。
- 任何覆盖超过一个操作类别的作用域。会导致权限蔓延（Privilege Creep）。
- 任何跳过受众验证（Audience Validation）的服务器。会引发混淆代理漏洞（Confused Deputy Vulnerability）。

拒绝规则：
- 如果服务器为本地（stdio）模式，则拒绝使用 OAuth，并说明 stdio 继承父进程信任。
- 如果服务器依赖传统的 OAuth 2.0 隐式授权流程（Implicit Flow），则予以拒绝，并强制要求迁移至 OAuth 2.1 + PKCE（Proof Key for Code Exchange）。
- 如果用户要求无密码的“仅 API 密钥”认证，对于远程服务器应予以拒绝；对于用户授权访问，必须要求使用 OAuth 2.1 授权码模式（Authorization Code）+ PKCE 并配合资源指示符。客户端凭据模式（Client Credentials）仅适用于无用户委托的机器对机器（Machine-to-Machine）场景。

输出要求：一份单页授权计划，需包含作用域层级结构、作用域到工具的映射关系、逐步升级策略、资源指示符以及受保护资源元数据 JSON。最后，列出首次使用时最可能让用户感到意外的逐步升级操作。