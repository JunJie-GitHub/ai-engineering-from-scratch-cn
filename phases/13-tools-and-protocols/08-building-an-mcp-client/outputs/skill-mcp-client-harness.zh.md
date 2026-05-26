---
name: mcp-client-harness
description: 给定 MCP 服务器（Model Context Protocol 服务器）的声明式列表（名称、命令、参数），搭建一个支持握手（handshake）、命名空间合并（namespace merge）与路由（routing）的多服务器客户端。
version: 1.0.0
phase: 13
lesson: 08
tags: [mcp, 客户端, 多服务器, 路由, 命名空间]
---

根据待运行的 MCP 服务器配置，生成一个客户端框架（client harness）。该框架负责启动各个服务器、完成握手流程、将其工具列表合并至单一命名空间，并将每次调用路由至对应的所属服务器。

生成以下内容：

1. 服务器配置解析器。建立 `name -> {command, args, env}` 的映射。验证命令是否存在于系统路径中。
2. 进程启动方案。使用 `subprocess.Popen` 配合 stdin/stdout/stderr 管道，设置 `bufsize=1` 并启用文本模式。为每个服务器分配一个后台读取线程。
3. 握手流水线。针对每个会话（session）：发送 `initialize`，等待响应，持久化能力（capabilities），发送 `notifications/initialized`。
4. 命名空间合并。选择冲突策略（collision policy）：`prefix-on-collision`（默认）、`reject-on-collision` 或 `silent-overwrite`（禁止）。在启动时打印合并后的工具列表。
5. 路由函数。`client.call(canonical_name, arguments)` 查找所属会话并写入 `tools/call` 消息。通过待处理请求表（pending-request table）中的 future 对象等待匹配 ID 的响应。

硬性拒绝条件：
- 任何未为每个服务器独立生成进程的框架。进程内多路复用（in-process multiplexing）会破坏隔离模型（isolation model）。
- 任何将 `silent-overwrite` 设为默认冲突策略的框架。存在安全风险。
- 任何在读取 stdout 时阻塞主线程的框架。这将导致通知停滞。

拒绝规则：
- 若服务器命令不受信任（未列入固定允许列表 allowlist），则拒绝启动，并路由至第 13 阶段 · 15 进行安全检查。
- 若用户无理由配置超过 10 个服务器，则发出警告并建议改用网关（gateway）（第 13 阶段 · 17）。
- 若要求在此处处理 OAuth，则拒绝并路由至第 13 阶段 · 16。

输出：一个完整的客户端框架 Python 文件（约 150 行），包含 Session（会话）、合并逻辑、路由功能，以及一个用于测试每个已配置服务器的主循环。结尾需附一行总结，注明所采用的冲突策略及合并后的工具数量。