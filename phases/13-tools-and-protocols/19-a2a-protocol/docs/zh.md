# A2A — 智能体间协议（Agent-to-Agent Protocol）

> MCP（模型上下文协议）是智能体到工具（agent-to-tool）的协议。A2A（智能体间协议）则是智能体到智能体（agent-to-agent）的协议——一种开放协议，旨在让基于不同框架构建的黑盒智能体（opaque agents）进行协作。该协议由 Google 于 2025 年 4 月发布，同年 6 月捐赠给 Linux 基金会，并于 2026 年 4 月达到 v1.0 版本，获得了包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow 在内的 150 多家机构的支持。它整合了 IBM 的 ACP 协议，并新增了 AP2 支付扩展。本课程将逐步讲解智能体卡片（Agent Card）、任务（Task）生命周期以及两种传输绑定（transport bindings）方式。

**类型：** 构建（Build）
**语言：** Python（标准库、智能体卡片与任务测试框架）
**前置条件：** 第 13 阶段 · 06（MCP 基础），第 13 阶段 · 08（MCP 客户端）
**时长：** 约 75 分钟

## 学习目标

- 区分智能体到工具（MCP）与智能体到智能体（A2A）的使用场景。
- 在 `/.well-known/agent.json` 路径发布智能体卡片（Agent Card），包含技能与端点元数据。
- 梳理任务（Task）生命周期（已提交 → 处理中 → 需要输入 → 已完成 / 失败 / 已取消 / 已拒绝）。
- 使用包含部件（Parts，如文本、文件、数据）的消息（Messages）以及产出物（Artifacts）作为输出。

## 问题背景

在 A2A 出现之前，若客服智能体需要将撰写报告的任务委派给专业的写作智能体，通常有以下几种方案：

- 自定义 REST API。虽然可行，但每次对接都需要单独开发。
- 共享代码库。要求两个智能体必须运行在相同的框架上。
- MCP。并不适用：MCP 专用于调用工具，而非让两个智能体在保留各自黑盒内部推理过程的前提下进行协作。

A2A 填补了这一空白。它将交互建模为一个智能体向另一个智能体发送任务（Task），并附带生命周期、消息和产出物。被调用智能体的内部状态保持黑盒化——调用方只能看到任务状态的流转以及最终的输出结果。

A2A 是“让跨框架智能体相互通信”的协议。它并非要取代 MCP，两者是互补关系。

## 核心概念

### 智能体卡片 (Agent Card)

每个符合 A2A (Agent-to-Agent) 规范的智能体都会在 `/.well-known/agent.json` 路径下发布一张卡片：

{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}

发现机制基于 URL：获取卡片后，即可获知 A2A 端点的 URL 并枚举其技能。

### 签名智能体卡片 (Signed Agent Cards / AP2)

AP2 (Agent Payments) 扩展规范（2025 年 9 月）为智能体卡片引入了加密签名。发布者使用 JWT (JSON Web Token) 对自身卡片进行签名，消费者负责验证。此举可有效防止身份冒充。

### 任务生命周期 (Task Lifecycle)

submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)

客户端通过 `tasks/send` 发起请求。被调用的智能体会在不同状态间流转；客户端可通过 SSE (Server-Sent Events) 订阅状态更新，或采用轮询方式获取。

### 消息与部件 (Messages and Parts)

一条消息可承载一个或多个部件 (Part)：

- `text` — 纯文本内容。
- `file` — 带有 MIME 类型的 Base64 编码数据块。
- `data` — 类型化的 JSON 负载（为被调用智能体提供的结构化输入）。

示例：

{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}

### 产出物 (Artifacts)

输出结果为产出物 (Artifact)，而非原始字符串。产出物是一种具有名称和类型的输出：

{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}

产出物支持以分块 (chunk) 形式流式传输。调用方负责累积接收。

### 两种传输绑定 (Transport Bindings)

1. **基于 HTTP 的 JSON-RPC。** `/a2a` 端点，通过 POST 发送请求，可选 SSE 用于流式传输。此为默认绑定方式。
2. **gRPC。** 适用于原生支持 gRPC 的企业级环境。

两种绑定方式承载的逻辑消息结构完全一致。

### 状态不透明性保持 (Opacity Preservation)

一项核心设计原则：被调用智能体的内部状态对外不透明。调用方仅能看到任务状态和产出物。被调用智能体的思维链 (Chain-of-Thought)、工具调用以及子智能体委派过程——全部不可见。这与 MCP (Model Context Protocol) 不同，后者的工具调用是透明的。

设计初衷：A2A 使得竞争对手能够在不暴露内部实现的前提下进行协作。通过 A2A，调用方只需“调用该客服智能体”，而无需了解该智能体具体如何实现服务。

### 时间线 (Timeline)

- **2025-04-09。** Google 宣布推出 A2A。
- **2025-06-23。** 捐赠至 Linux 基金会。
- **2025-08。** 整合 IBM 的 ACP (Agent Communication Protocol)。
- **2025-09。** AP2 扩展规范（智能体支付）正式发布。
- **2026-04。** v1.0 版本发布，获得 150 多家机构支持。

### 与 MCP 的关系 (Relationship to MCP)

| 维度 | MCP | A2A |
|-----------|-----|-----|
| 应用场景 | 智能体到工具 (Agent-to-tool) | 智能体到智能体 (Agent-to-agent) |
| 不透明性 | 工具调用透明 | 内部推理不透明 |
| 典型调用方 | 智能体运行时 (Agent runtime) | 其他智能体 |
| 状态 | 工具调用结果 | 具有生命周期的任务 |
| 授权机制 | OAuth 2.1 (Phase 13 · 16) | 经 JWT 签名的智能体卡片 (AP2) |
| 传输协议 | Stdio / 可流式传输的 HTTP (Streamable HTTP) | 基于 HTTP 的 JSON-RPC / gRPC |

当需要调用特定工具时，请使用 MCP；当需要将完整任务委派给另一个智能体时，请使用 A2A。许多生产系统会同时采用两者：智能体利用 MCP 处理工具层，利用 A2A 处理协作层。

## 使用它

`code/main.py` 实现了一个极简的智能体间通信（A2A）框架：一个研究智能体发布其智能体卡片（Agent Card），一个写作智能体接收包含 PDF 和文本指令等部件（parts）的 `tasks/send` 请求，状态依次经历 working → input_required → working → completed，并最终返回一个文本制品（artifact）。全部使用标准库；采用内存传输（in-memory transport）以专注于消息结构。

重点关注：

- 智能体卡片（Agent Card）的 JSON 结构。
- 任务（Task）ID 分配与状态流转。
- 包含混合类型部件（parts）的消息。
- 任务执行中途的需补充输入（input_required）分支。
- 任务完成时的制品（artifact）返回。

## 发布它

本课时将生成 `outputs/skill-a2a-agent-spec.md`。针对一个需要被其他智能体调用的新智能体，该技能（skill）会生成智能体卡片（Agent Card）JSON、技能模式（skills schema）以及端点蓝图（endpoint blueprint）。

## 练习

1. 运行 `code/main.py`。追踪完整的任务（Task）生命周期，包括被调用智能体请求澄清时触发的需补充输入（input_required）暂停状态。

2. 添加带签名的智能体卡片（Agent Card）。使用 HMAC 对卡片的规范化 JSON（canonical JSON）进行签名。编写一个验证器，并确认其在卡片被篡改后会验证失败。

3. 实现任务流式传输（task streaming）：写作智能体通过服务器发送事件（SSE）发送三个增量制品（artifact）数据块，调用方负责累积这些数据。

4. 设计一个封装模型上下文协议（MCP）服务器的智能体间通信（A2A）智能体。将每个 MCP 工具（tool）映射为一个 A2A 技能（skill）。注意其中的权衡——会损失哪些不透明性（opacity）？

5. 阅读 A2A v1.0 发布公告，找出截至 2026 年 4 月尚未被任何框架实现的一项功能。（提示：该功能与多跳任务委派（multi-hop task delegation）有关。）

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 智能体间通信（A2A） | “智能体间通信协议” | 用于智能体间黑盒协作的开放协议 |
| 智能体卡片（Agent Card） | “`.well-known/agent.json`” | 描述智能体技能与端点的已发布元数据 |
| 技能（Skill） | “可调用单元” | 智能体支持的命名操作（类似于 MCP 工具） |
| 任务（Task） | “委派单元” | 具有生命周期和最终制品的工作项 |
| 消息（Message） | “任务输入” | 承载部件（Parts，含文本、文件、数据）的载体 |
| 部件（Part） | “类型化数据块” | 消息中的 `text` / `file` / `data` 元素 |
| 制品（Artifact） | “任务输出” | 任务完成时返回的具名、具类型输出 |
| AP2 | “智能体支付协议” | 用于建立信任与支付的签名智能体卡片（Agent Card）扩展 |
| 不透明性（Opacity） | “黑盒协作” | 被调用智能体的内部实现对调用方隐藏 |
| 需补充输入（Input-required） | “任务暂停” | 智能体需要更多信息时的生命周期状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A（Agent-to-Agent）官方规范
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 参考实现与软件开发工具包（SDK）
- [Linux 基金会 — A2A 启动新闻稿](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月治理权移交
- [Google Cloud — A2A 协议升级](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 发展路线图与合作伙伴生态进展
- [Google Dev — A2A 1.0 里程碑](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 版本说明与向后兼容（backward compatibility）指南