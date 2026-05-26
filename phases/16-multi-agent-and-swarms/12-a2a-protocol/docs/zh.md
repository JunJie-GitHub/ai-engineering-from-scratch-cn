# A2A — 智能体间协议 (Agent-to-Agent Protocol)

> Google 于 2025 年 4 月发布了 A2A；截至 2026 年 4 月，其规范已发布在 https://a2a-protocol.org/latest/specification/，并获得了 150 多家组织的支持。A2A 是 MCP（第 13 课）的水平补充 (horizontal complement)：MCP 采用垂直架构 (vertical)，而 A2A 采用点对点架构 (peer-to-peer)。它定义了智能体卡片 (Agent Cards，用于服务发现 discovery)、包含工件 (artifacts，如文本、结构化数据、视频) 的任务、不透明的任务生命周期 (opaque task lifecycles) 以及身份验证 (auth)。在生产系统中，MCP 与 A2A 的协同使用正日益普遍。Google Cloud 在 2025 至 2026 年间已将 A2A 支持集成至 Vertex AI Agent Builder 中。

**类型：** 学习与构建
**编程语言：** Python（标准库，`http.server`，`json`）
**前置条件：** 第 16 阶段 · 04（基础模型）
**预计耗时：** 约 75 分钟

## 问题

当你的智能体 (Agent) 需要调用部署在其他系统上的另一个智能体时，该如何实现？你可以暴露一个 HTTP 端点，定义一套专用的 JSON 模式 (JSON schema)，并期望对方能够兼容。但这会导致每对智能体之间都变成一次定制化集成。

A2A 正是为此类调用设计的通用通信协议 (wire protocol)。它提供了标准化的服务发现 (discovery)、任务模型、传输机制与工件 (artifacts) 格式。其理念类似于 HTTP+REST，但将智能体作为一等公民 (first-class citizens) 进行支持。

## 概念

### 四大核心要素

**智能体卡片（Agent Card）**。位于 `/.well-known/agent.json` 的 JSON 文档，用于描述智能体：名称、技能、端点、支持的模态（modalities）以及认证要求。通过读取该卡片即可完成发现（Discovery）过程。

GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }

**任务（Task）**。工作单元。一个具有生命周期的异步、有状态对象：`submitted → working → completed / failed / canceled`。客户端发送任务后，通过轮询或订阅来获取更新。

**产出物（Artifact）**。任务产生的结果类型。包括文本、结构化 JSON、图像、视频、音频等。产出物具有明确的类型定义，使得不同模态均作为一等公民（first-class）处理。

**不透明生命周期（Opaque lifecycle）**。A2A 不规定远程智能体*如何*解决任务。客户端仅能看到状态转换和产出物；具体实现可自由使用任何框架。

### MCP 与 A2A 的分工

- **MCP**（第 13 课）：智能体 ↔ 工具。智能体通过 JSON-RPC 向工具服务器进行读写操作。默认无状态。
- **A2A**：智能体 ↔ 智能体。对等协议（Peer protocol）；通信双方均为具备独立推理能力的智能体。

生产环境中的多智能体系统（Multi-agent systems）通常同时使用两者。A2A 对等节点会调用其本地的 MCP 工具。这种分工使两种关注点保持清晰独立。

### 发现流程

Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──

或者采用流式传输：通过订阅 `/tasks/{id}/events` 的服务器发送事件（SSE）获取推送更新。

### 认证机制

A2A 支持三种常见模式：

- **承载令牌（Bearer token）** — OAuth2 或不透明令牌（opaque token）。
- **双向 TLS（mTLS）** — 相互 TLS 认证；组织之间互相验证身份。
- **签名请求（Signed requests）** — 对请求负载（payload）进行 HMAC 签名。

认证方式在智能体卡片中声明；客户端负责发现并遵循。

### 截至 2026 年 4 月已有 150 多家组织采用

企业级采用推动了 A2A 的规模化发展。核心要点在于：A2A 已成为企业智能体系统跨越信任边界的标准方式。Google Cloud 已在其 Vertex AI Agent Builder 中提供 A2A 支持；Microsoft Agent Framework 同样支持该协议；主流框架（如 LangGraph、CrewAI、AutoGen）均内置了 A2A 适配器。

### A2A 的优势场景

- **跨组织调用**。公司 A 的智能体调用公司 B 的智能体。若无 A2A，每对调用都需要定制专属契约。
- **异构框架集成**。LangGraph 智能体调用 CrewAI 智能体，再调用自定义 Python 智能体。A2A 实现了标准化统一。
- **类型化产出物**。视频结果、结构化 JSON、音频等——均作为一等公民处理。
- **长耗时任务**。不透明生命周期结合轮询机制，使长达数小时的任务处理变得简单直接。

### A2A 的局限性

- **对延迟敏感的微调用**。A2A 的生命周期是异步的。亚毫秒级的智能体间通信并不适用；此类场景应使用直接 RPC。
- **紧耦合的进程内智能体**。若两个智能体运行在同一 Python 进程中，A2A 的 HTTP 往返通信显得过于繁重。
- **小型团队**。规范带来的开销是客观存在的；仅限内部使用的智能体可能无需如此正式的流程。

### A2A 与 ACP、ANP、NLIP 的对比

2024 至 2026 年间涌现了多项相关规范：

- **ACP**（IBM/Linux 基金会）—— A2A 的前身，适用范围较窄。
- **ANP**（智能体网络协议，Agent Network Protocol）—— 侧重对等发现，优先采用去中心化架构。
- **NLIP**（Ecma 自然语言交互协议，2025 年 12 月标准化）—— 专注于自然语言内容类型。

截至 2026 年 4 月，A2A 已成为采用率最高的对等协议。如需详细对比，请参阅 arXiv:2505.02279（Liu 等人，《智能体互操作性协议综述》）。

## 构建实现

`code/main.py` 使用 `http.server` 和 JSON 实现了一个极简的 A2A (Agent-to-Agent) 服务器与客户端。服务器端：

- 暴露 `/.well-known/agent.json` 端点，
- 接受 `POST /tasks` 请求，
- 管理任务 (Task) 状态，
- 在 `GET /tasks/{id}` 请求中返回产出物 (Artifact)。

客户端：

- 获取智能体卡片 (Agent Card)，
- 提交任务，
- 轮询直至任务完成，
- 读取产出物。

运行：

python3 code/main.py

该脚本会在后台线程中启动服务器，随后运行客户端与其交互。你将看到完整的流程：发现 (Discovery)、提交、轮询、获取产出物。

## 实际应用

`outputs/skill-a2a-integrator.md` 规划了 A2A 集成方案：涵盖智能体卡片内容、任务模式 (Schema)、认证方式选择，以及流式传输与轮询的对比。

## 部署上线

检查清单：

- **锁定规范版本。** A2A 仍在演进中；智能体卡片应明确声明所使用的协议版本。
- **任务创建需具备幂等性 (Idempotent)。** 重复提交（如网络重试）应仅生成一个任务。
- **产出物模式 (Schema)。** 明确声明智能体返回的数据结构；消费方应进行校验。
- **速率限制与认证。** A2A 面向公开网络；需应用标准的 Web 安全措施。
- **失败任务进入死信队列 (Dead-letter)。** 长期观察失败模式，以识别反复出现的故障类型。

## 练习

1. 运行 `code/main.py`。确认客户端能够发现服务器并接收到正确的产出物。
2. 为服务器添加第二个技能（例如 "summarize"）。更新智能体卡片。编写一个客户端，使其能根据任务类型自动选择对应技能。
3. 实现一个服务器发送事件 (SSE) 流式端点：`/tasks/{id}/events`，用于推送状态变更。客户端需要做哪些相应调整？
4. 阅读 A2A 规范（https://a2a-protocol.org/latest/specification/）。找出规范中强制要求但本演示未实现的三个功能。
5. 对比 A2A（基于智能体卡片的发现机制）与模型上下文协议 (MCP)（通过 `listTools` 在服务端列出能力）。自我描述型智能体与能力探测机制之间的权衡是什么？

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| A2A | “智能体对智能体” | 跨系统智能体间调用的对等协议。由 Google 于 2025 年提出。 |
| Agent Card (智能体卡片) | “智能体的名片” | 位于 `/.well-known/agent.json` 的 JSON 文件，用于描述技能、端点和认证方式。 |
| Task (任务) | “工作单元” | 具有生命周期的异步有状态对象；完成后生成产出物。 |
| Artifact (产出物) | “结果” | 带类型的输出：文本、结构化 JSON、图像、视频、音频。作为一等公民的媒体类型。 |
| Opaque lifecycle (不透明生命周期) | “具体怎么解决是智能体自己的事” | 客户端仅能看到状态流转；服务端可自由选择框架或工具。 |
| Discovery (发现) | “找到智能体” | `GET /.well-known/agent.json` 返回智能体卡片。 |
| MCP vs A2A | “工具 vs 对等节点” | MCP：垂直方向的智能体 ↔ 工具。A2A：水平方向的智能体 ↔ 智能体。 |
| ACP / ANP / NLIP | “同族协议” | 相邻/相关规范；A2A 是 2026 年采用率最高的协议。 |

## 延伸阅读

- [A2A 规范](https://a2a-protocol.org/latest/specification/) — 权威标准规范
- [Google 开发者博客 — A2A 公告](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月发布博文
- [A2A GitHub 仓库](https://github.com/a2aproject/A2A) — 参考实现与软件开发工具包 (SDK)
- [Liu 等人 — 智能体互操作性协议 (Agent Interoperability Protocols) 综述](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A 与 ANP 对比