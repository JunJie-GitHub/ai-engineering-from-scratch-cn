---
name: mcp-server-scaffolder
description: 搭建特定领域的 MCP（Model Context Protocol）服务器，合理划分工具（tools）/资源（resources）/提示词（prompts），并提供 SDK 进阶路径（graduation path）。
version: 1.0.0
phase: 13
lesson: 07
tags: [mcp, server, fastmcp, scaffold]
---

给定一个领域（如笔记、工单、文件、数据库等），生成一份 MCP 服务器规划：明确哪些能力应作为工具暴露，哪些作为资源，哪些作为提示词，并规划向 Python 或 TypeScript SDK 的进阶路径。

需产出以下内容：

1. 工具列表。用户明确要求执行的原子操作。需包含名称、描述（适用场景模式）、输入结构（input schema）及注解提示。
2. 资源列表。用户希望读取的数据。需包含 URI 方案（URI scheme）、MIME 类型（MIME type），以及是否启用 `resources/subscribe`。
3. 提示词列表。宿主（host）应作为斜杠命令（slash-commands）暴露的可复用模板。需包含参数列表。
4. 能力声明。服务器在 `initialize` 阶段返回的确切 `capabilities` 对象。
5. 进阶说明。针对上述各项，提供 FastMCP（Python）或 TypeScript SDK 的等效实现。指出一个 SDK 特性（例如 `lifespan`、`context`），用于替代脚手架中手动实现的标准库（stdlib）模式。

硬性拒绝条件：
- 任何仅作为工具暴露而未作为资源暴露的“数据库查询”。正确的划分方式是：将 `/list` 和 `/read` 作为资源，将带参数的 `/query` 作为工具。
- 任何在同一命名空间（namespace）下混合用户输入工具与特权工具，且未添加注解的服务器。
- 任何声称具备 `resources/subscribe` 能力，但缺乏持久化通知机制的服务器脚手架。

拒绝规则：
- 若该领域不存在只读接口（read-only surface），则拒绝搭建资源；建议仅使用工具的服务器。
- 若该领域没有天然的斜杠命令模板，则拒绝搭建提示词。
- 若用户要求配置认证方案（auth scheme），则予以拒绝，并引导至第 13 阶段 · 第 16 课（OAuth 2.1）。

输出要求：一份单页服务器规划，包含上述三个基础列表、能力对象，以及一段约 10 行、采用 `@app.tool()` 装饰器（decorator）风格的进阶示例代码。最后附上服务器应设置的最关键的一个注解标志（annotation flag）。