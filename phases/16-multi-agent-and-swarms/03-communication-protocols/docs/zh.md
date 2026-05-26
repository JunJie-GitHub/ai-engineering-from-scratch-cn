# 通信协议 (Communication Protocols)

> 无法使用同一种语言交流的代理 (Agent) 算不上一个团队，它们只是对着虚空呐喊的陌生人。

**类型：** 构建实践
**编程语言：** TypeScript
**前置条件：** 第 14 阶段（代理工程 (Agent Engineering)），第 16.01 课（为何需要多代理 (Why Multi-Agent)）
**预计耗时：** 约 120 分钟

## 学习目标

- 实现模型上下文协议 (MCP) 的工具发现与调用，使代理能够使用外部服务器暴露的工具
- 构建代理间通信协议 (A2A) 的代理卡片 (Agent Card) 与任务端点 (Task Endpoint)，允许一个代理通过 HTTP 将工作委派给另一个代理
- 对比 MCP（工具访问）、A2A（代理间协作）、ACP（企业审计）和 ANP（去中心化信任），并阐明各协议分别解决何种问题
- 在单一系统中串联多种协议，实现代理通过 MCP 发现工具并通过 A2A 委派任务

## 问题背景

你将系统拆分为多个代理：一个负责研究，一个负责编码，一个负责审查。它们在各自的任务上表现出色，但现在你需要让它们真正相互协作。

你的初次尝试往往很直接：在代理之间传递字符串。研究代理返回一大段文本，编码代理尽其所能去解析。这种方法起初看似可行，直到编码代理误解了研究摘要，或者两个代理因相互等待而陷入死锁，又或者你需要让不同团队开发的代理进行协作。此时，“仅仅传递字符串”的方案便彻底崩溃了。

这就是通信协议问题。如果没有一套共享的契约来规范代理间的信息交换方式，多代理系统 (Multi-Agent Systems) 将变得脆弱、无法审计，且一旦规模超出你亲手编写的少数几个代理，便再也无法扩展。

AI 生态为此推出了四种协议，各自解决该问题的不同切面：

- **MCP**：用于工具访问
- **A2A**：用于代理间协作
- **ACP**：用于企业级审计
- **ANP**：用于去中心化身份与信任

本课程将深入探讨这些协议。你将阅读各规范中的真实线路格式 (wire formats)，构建可运行的实现，并将这四种协议整合到一个统一的系统中。

## 核心概念

### 协议全景

可以将这四种协议视为不同的层级，每一层都致力于解决一个特定的问题：

block-beta
  columns 1
  block:ANP["ANP — How do agents trust strangers?\nDecentralized identity (DID), E2EE, meta-protocol"]
  end
  block:A2A["A2A — How do agents collaborate on goals?\nAgent Cards, task lifecycle, streaming, negotiation"]
  end
  block:ACP["ACP — How do agents talk in auditable systems?\nRuns, trajectory metadata, session continuity"]
  end
  block:MCP["MCP — How does an agent use a tool?\nTool discovery, execution, context sharing"]
  end

  style ANP fill:#f3e8ff,stroke:#7c3aed
  style A2A fill:#dbeafe,stroke:#2563eb
  style ACP fill:#fef3c7,stroke:#d97706
  style MCP fill:#d1fae5,stroke:#059669

它们并非竞争关系，而是在不同层级上解决不同的问题。

### MCP（回顾）

MCP 在第 13 阶段（Phase 13）中有详细讲解。快速回顾：MCP 标准化了大语言模型（LLM）连接外部工具和数据源的方式。它是一种**客户端-服务器（client-server）**协议，智能体（客户端）在此协议下发现并调用服务器暴露的工具。

sequenceDiagram
    participant Agent as Agent (client)
    participant MCP1 as MCP Server<br/>(database, API, files)

    Agent->>MCP1: list tools
    MCP1-->>Agent: tool definitions
    Agent->>MCP1: call tool X
    MCP1-->>Agent: result

MCP 专注于**智能体到工具（agent-to-tool）**的通信。它并不用于智能体之间的相互通信。

### A2A（Agent2Agent 协议）

**创建者：** Google（现归属于 Linux 基金会，标识为 `lf.a2a.v1`）
**规范版本：** 1.0.0
**解决的问题：** 自主智能体（autonomous agents）如何相互协作、协商并委派任务？

A2A 是用于**点对点智能体协作（peer-to-peer agent collaboration）**的协议。如果说 MCP 负责将智能体连接到工具，那么 A2A 则负责将智能体连接到其他智能体。每个智能体都会在标准路径（well-known URL）上发布一张**智能体卡片（Agent Card）**，其他智能体通过该卡片发现它、与之协商并向其委派任务。

#### A2A 的工作原理

sequenceDiagram
    participant Client as Client Agent
    participant Remote as Remote Agent

    Client->>Remote: GET /.well-known/agent-card.json
    Remote-->>Client: Agent Card (skills, modes, security)

    Client->>Remote: POST /message:send
    Remote-->>Client: Task (submitted/working)

    alt Polling
        Client->>Remote: GET /tasks/{id}
        Remote-->>Client: Task status + artifacts
    else Streaming
        Client->>Remote: POST /message:stream
        Remote-->>Client: SSE: statusUpdate
        Remote-->>Client: SSE: artifactUpdate
        Remote-->>Client: SSE: completed
    end

#### 真实的智能体卡片

以下是实际环境中 A2A 智能体卡片的真实模样。通过 `GET /.well-known/agent-card.json` 提供：

{
  "name": "Research Agent",
  "description": "Searches documentation and summarizes findings",
  "version": "1.0.0",
  "supportedInterfaces": [
    {
      "url": "https://research-agent.example.com/a2a/v1",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    },
    {
      "url": "https://research-agent.example.com/a2a/rest",
      "protocolBinding": "HTTP+JSON",
      "protocolVersion": "1.0"
    }
  ],
  "provider": {
    "organization": "Your Company",
    "url": "https://example.com"
  },
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "defaultInputModes": ["text/plain", "application/json"],
  "defaultOutputModes": ["text/plain", "application/json"],
  "skills": [
    {
      "id": "web-research",
      "name": "Web Research",
      "description": "Searches the web and synthesizes findings",
      "tags": ["research", "search", "summarization"],
      "examples": ["Research the latest changes in React 19"]
    },
    {
      "id": "doc-analysis",
      "name": "Documentation Analysis",
      "description": "Reads and analyzes technical documentation",
      "tags": ["docs", "analysis"],
      "inputModes": ["text/plain", "application/pdf"],
      "outputModes": ["application/json"]
    }
  ],
  "securitySchemes": {
    "bearer": {
      "httpAuthSecurityScheme": {
        "scheme": "Bearer",
        "bearerFormat": "JWT"
      }
    }
  },
  "security": [{ "bearer": [] }]
}

需要注意的关键点：
- **技能（Skills）**定义了智能体能够执行的操作。每项技能都包含 ID、标签以及支持的输入/输出 MIME 类型。客户端智能体正是通过这些信息来判断远程智能体是否能处理其请求。
- **supportedInterfaces** 列出了多种协议绑定。单个智能体可以同时支持 JSON-RPC、REST 和 gRPC。
- **安全性（Security）**已内置于卡片中。客户端在发起任何请求之前，就能明确所需的认证方式。

#### 任务生命周期

任务是 A2A 中的核心工作单元。它们会经历一系列预定义的状态：

stateDiagram-v2
    [*] --> submitted
    submitted --> working
    working --> input_required: needs more info
    input_required --> working: client sends data
    working --> completed: success
    working --> failed: error
    working --> canceled: client cancels
    submitted --> rejected: agent declines

    completed --> [*]
    failed --> [*]
    canceled --> [*]
    rejected --> [*]

    note right of completed: Terminal states are immutable.\nFollow-ups create new tasks\nwithin the same contextId.

全部 8 种状态（规范中还定义了 `UNSPECIFIED` 作为哨兵值，此处省略）：

| 状态 | 终态？ | 含义 |
|---|---|---|
| `TASK_STATE_SUBMITTED` | 否 | 已确认接收，尚未开始处理 |
| `TASK_STATE_WORKING` | 否 | 正在积极处理中 |
| `TASK_STATE_INPUT_REQUIRED` | 否 | 智能体需要客户端提供更多信息 |
| `TASK_STATE_AUTH_REQUIRED` | 否 | 需要进行身份验证 |
| `TASK_STATE_COMPLETED` | 是 | 已成功完成 |
| `TASK_STATE_FAILED` | 是 | 处理出错并终止 |
| `TASK_STATE_CANCELED` | 是 | 在完成前被取消 |
| `TASK_STATE_REJECTED` | 是 | 智能体拒绝了该任务 |

一旦任务进入终态，其状态将不可变，且不再接收后续消息。后续的跟进操作会在相同的 `contextId` 下创建一个新任务。

#### 网络报文格式（Wire Format）

A2A 使用 JSON-RPC 2.0。以下是真实消息交换的示例：

**客户端发送任务：**
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "msg-001",
      "role": "ROLE_USER",
      "parts": [{ "text": "Research React 19 compiler features" }]
    },
    "configuration": {
      "acceptedOutputModes": ["text/plain", "application/json"],
      "historyLength": 10
    }
  }
}

**智能体响应任务：**
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "task": {
      "id": "task-abc-123",
      "contextId": "ctx-xyz-789",
      "status": {
        "state": "TASK_STATE_COMPLETED",
        "timestamp": "2026-03-27T10:30:00Z"
      },
      "artifacts": [
        {
          "artifactId": "art-001",
          "name": "research-results",
          "parts": [{
            "data": {
              "findings": [
                "React 19 compiler auto-memoizes components",
                "No more manual useMemo/useCallback needed",
                "Compiler runs at build time, not runtime"
              ]
            },
            "mediaType": "application/json"
          }]
        }
      ]
    }
  }
}

**通过 SSE 进行流式传输：**
POST /message:stream HTTP/1.1
Content-Type: application/json
A2A-Version: 1.0

data: {"task":{"id":"task-123","status":{"state":"TASK_STATE_WORKING"}}}

data: {"statusUpdate":{"taskId":"task-123","status":{"state":"TASK_STATE_WORKING","message":{"role":"ROLE_AGENT","parts":[{"text":"Searching documentation..."}]}}}}

data: {"artifactUpdate":{"taskId":"task-123","artifact":{"artifactId":"art-1","parts":[{"text":"partial findings..."}]},"append":true,"lastChunk":false}}

data: {"statusUpdate":{"taskId":"task-123","status":{"state":"TASK_STATE_COMPLETED"}}}

### ACP（Agent Communication Protocol）

**创建者：** IBM / BeeAI
**规范版本：** 0.2.0（OpenAPI 3.1.1）
**状态：** 正在 Linux 基金会框架下并入 A2A
**解决的问题：** 智能体如何在具备完整可审计性、会话连续性和轨迹追踪能力的前提下进行通信？

ACP 是面向**企业级（enterprise）**的协议。与许多摘要所声称的不同，ACP **并未**使用 JSON-LD。它是一个通过 OpenAPI 定义的简洁 REST/JSON API。其独特之处在于**轨迹元数据（TrajectoryMetadata）**：智能体的每次响应都可以附带生成该响应所经历的推理步骤和工具调用的详细日志。

sequenceDiagram
    participant Client
    participant ACP as ACP Agent
    participant Audit as Audit Log

    Client->>ACP: POST /runs (mode: sync)
    ACP->>ACP: Process request...
    ACP->>Audit: Log trajectory:<br/>reasoning + tool calls
    ACP-->>Client: Response + TrajectoryMetadata
    Note over Audit: Every step recorded:<br/>tool_name, tool_input,<br/>tool_output, reasoning

#### ACP 中的智能体发现

ACP 定义了四种发现方法：

graph LR
    A[Agent Discovery] --> B["Runtime<br/>GET /agents"]
    A --> C["Open<br/>.well-known/agent.yml"]
    A --> D["Registry<br/>Centralized catalog"]
    A --> E["Embedded<br/>Container labels"]

    style B fill:#dbeafe,stroke:#2563eb
    style C fill:#d1fae5,stroke:#059669
    style D fill:#fef3c7,stroke:#d97706
    style E fill:#f3e8ff,stroke:#7c3aed

**AgentManifest** 比 A2A 的 Agent Card 更为简洁：

{
  "name": "summarizer",
  "description": "Summarizes documents with source citations",
  "input_content_types": ["text/plain", "application/pdf"],
  "output_content_types": ["text/plain", "application/json"],
  "metadata": {
    "tags": ["summarization", "RAG"],
    "framework": "BeeAI",
    "capabilities": [
      {
        "name": "Document Summarization",
        "description": "Condenses long documents into key points"
      }
    ],
    "recommended_models": ["llama3.3:70b-instruct-fp16"],
    "license": "Apache-2.0",
    "programming_language": "Python"
  }
}

#### 运行生命周期

ACP 使用“运行（Runs）”而非“任务（Tasks）”。一次运行代表智能体的一次执行过程，包含三种模式：

| 模式 | 行为 |
|---|---|
| `sync` | 阻塞式。响应中包含完整结果。 |
| `async` | 立即返回 202 状态码。通过轮询 `GET /runs/{id}` 获取状态。 |
| `stream` | SSE 流式传输。智能体处理过程中实时触发事件。 |

stateDiagram-v2
    [*] --> created
    created --> in_progress
    in_progress --> completed: success
    in_progress --> failed: error
    in_progress --> awaiting: needs input
    awaiting --> in_progress: client resumes
    in_progress --> cancelling: cancel request
    cancelling --> cancelled

    completed --> [*]
    failed --> [*]
    cancelled --> [*]

#### TrajectoryMetadata（审计追踪）

这是 ACP 的核心差异化特性。消息的每个部分都可以包含元数据，精确展示智能体执行的操作：

{
  "role": "agent/researcher",
  "parts": [
    {
      "content_type": "text/plain",
      "content": "The weather in San Francisco is 72F and sunny.",
      "metadata": {
        "kind": "trajectory",
        "message": "I need to check the weather for this location",
        "tool_name": "weather_api",
        "tool_input": { "location": "San Francisco, CA" },
        "tool_output": { "temperature": 72, "condition": "sunny" }
      }
    }
  ]
}

对于受监管行业而言，这极具价值。每个答案都附带可验证的推理链条：调用了哪些工具、使用了什么输入、收到了何种输出。彻底告别黑盒操作。

ACP 还支持用于来源归属的**引用元数据（CitationMetadata）**：

{
  "kind": "citation",
  "start_index": 0,
  "end_index": 47,
  "url": "https://weather.gov/sf",
  "title": "NWS San Francisco Forecast"
}

### ANP（Agent Network Protocol）

**创建者：** 开源社区（由 GaoWei Chang 发起）
**仓库：** [github.com/agent-network-protocol/AgentNetworkProtocol](https://github.com/agent-network-protocol/AgentNetworkProtocol)
**解决的问题：** 在没有中心化权威机构的情况下，不同组织的智能体如何建立互信？

ANP 是一种**去中心化身份协议（decentralized identity protocol）**。它利用 W3C 去中心化标识符（DIDs）和端到端加密（end-to-end encryption）来构建信任。与 A2A 通过已知端点发现智能体不同，ANP 允许智能体通过密码学手段证明自身身份。

ANP 包含三个层级：

graph TB
    subgraph Layer3["Layer 3: Application Protocol"]
        AD[Agent Description Documents]
        DISC[Discovery endpoints]
    end
    subgraph Layer2["Layer 2: Meta-Protocol"]
        NEG[AI-powered protocol negotiation]
        CODE[Dynamic code generation]
    end
    subgraph Layer1["Layer 1: Identity & Secure Communication"]
        DID["did:wba (W3C DID)"]
        HPKE[HPKE E2EE - RFC 9180]
        SIG[Signature verification]
    end

    Layer3 --> Layer2
    Layer2 --> Layer1

    style Layer1 fill:#d1fae5,stroke:#059669
    style Layer2 fill:#dbeafe,stroke:#2563eb
    style Layer3 fill:#f3e8ff,stroke:#7c3aed

#### DID 文档（真实结构）

ANP 使用一种名为 `did:wba`（Web-Based Agent）的自定义 DID 方法。DID `did:wba:example.com:user:alice` 会解析至 `https://example.com/user/alice/did.json`：

{
  "@context": [
    "https://www.w3.org/ns/did/v1",
    "https://w3id.org/security/suites/jws-2020/v1",
    "https://w3id.org/security/suites/secp256k1-2019/v1"
  ],
  "id": "did:wba:example.com:user:alice",
  "verificationMethod": [
    {
      "id": "did:wba:example.com:user:alice#key-1",
      "type": "EcdsaSecp256k1VerificationKey2019",
      "controller": "did:wba:example.com:user:alice",
      "publicKeyJwk": {
        "crv": "secp256k1",
        "x": "NtngWpJUr-rlNNbs0u-Aa8e16OwSJu6UiFf0Rdo1oJ4",
        "y": "qN1jKupJlFsPFc1UkWinqljv4YE0mq_Ickwnjgasvmo",
        "kty": "EC"
      }
    },
    {
      "id": "did:wba:example.com:user:alice#key-x25519-1",
      "type": "X25519KeyAgreementKey2019",
      "controller": "did:wba:example.com:user:alice",
      "publicKeyMultibase": "z9hFgmPVfmBZwRvFEyniQDBkz9LmV7gDEqytWyGZLmDXE"
    }
  ],
  "authentication": [
    "did:wba:example.com:user:alice#key-1"
  ],
  "keyAgreement": [
    "did:wba:example.com:user:alice#key-x25519-1"
  ],
  "humanAuthorization": [
    "did:wba:example.com:user:alice#key-1"
  ],
  "service": [
    {
      "id": "did:wba:example.com:user:alice#agent-description",
      "type": "AgentDescription",
      "serviceEndpoint": "https://example.com/agents/alice/ad.json"
    }
  ]
}

需要注意的关键点：
- 强制执行**密钥分离（Key separation）**。签名密钥（secp256k1）与加密密钥（X25519）相互独立。
- **`humanAuthorization`** 是 ANP 独有的特性。使用此类密钥前必须经过明确的人工授权（如生物识别、密码或硬件安全模块 HSM）。资金转账等高风险操作均通过此路径进行。
- **`keyAgreement`** 密钥用于 HPKE 端到端加密（RFC 9180）。
- **service** 部分链接至智能体描述文档。

#### ANP 中的信任机制

ANP **并未**采用信任网络（web-of-trust）或背书图谱。信任是双向的，且在每次交互时进行验证：

sequenceDiagram
    participant A as Agent A
    participant Domain as Agent A's Domain
    participant B as Agent B

    A->>B: HTTP request + DID + signature
    B->>Domain: Fetch DID document (HTTPS)
    Domain-->>B: DID document + public key
    B->>B: Verify signature with public key
    B-->>A: Issue access token
    A->>B: Subsequent requests use token
    Note over A,B: Trust = TLS domain verification<br/>+ DID signature verification<br/>+ Principle of least trust

信任来源于三个方面：
1. **域名级 TLS** 验证 DID 文档的主机
2. **DID 密码学签名** 验证智能体身份
3. **最小信任原则（Principle of least trust）** 仅授予最低必要权限

这里不存在基于 gossip 的信任传播或 PageRank 评分。你直接通过每个智能体的 DID 进行验证。

#### 元协议协商（Meta-Protocol Negotiation）

这是 ANP 最具创新性的特性。当来自不同生态系统的两个智能体相遇时，它们无需预先约定数据格式，而是通过自然语言进行协商：

{
  "action": "protocolNegotiation",
  "sequenceId": 0,
  "candidateProtocols": "I can communicate using:\n1. JSON-RPC with hotel booking schema\n2. REST with OpenAPI 3.1 spec\n3. Natural language over HTTP",
  "modificationSummary": "Initial proposal",
  "status": "negotiating"
}

sequenceDiagram
    participant A as Agent A
    participant B as Agent B

    A->>B: protocolNegotiation (candidateProtocols)
    B->>A: protocolNegotiation (counter-proposal)
    A->>B: protocolNegotiation (accepted)
    Note over A,B: Agents dynamically generate code<br/>to handle the agreed format.<br/>Max 10 rounds, then timeout.

智能体会进行多轮交互（最多 10 轮），直到就格式达成一致，随后动态生成代码来处理该格式。状态值包括：`negotiating`、`rejected`、`accepted`、`timeout`。

这意味着两个从未交互过的智能体，无需任何人预先定义共享模式（schema），就能自行摸索出通信方式。

### 对比（已修正）

| | MCP | A2A | ACP | ANP |
|---|---|---|---|---|
| **创建者** | Anthropic | Google / Linux 基金会 | IBM / BeeAI | 社区 |
| **规范格式** | JSON-RPC | JSON-RPC / REST / gRPC | OpenAPI 3.1 (REST) | JSON-RPC |
| **主要用途** | 智能体到工具 | 智能体到智能体 | 智能体到智能体 | 智能体到智能体 |
| **发现机制** | 工具列表 | `/.well-known/agent-card.json` | `GET /agents`, `/.well-known/agent.yml` | `/.well-known/agent-descriptions`, DID 服务端点 |
| **身份验证** | 隐式（本地） | 安全方案（OAuth, mTLS） | 服务器级 | W3C DID (`did:wba`) 配合端到端加密（E2EE） |
| **审计追踪** | 不适用 | 基础（任务历史） | TrajectoryMetadata（工具调用、推理过程） | 未正式规范 |
| **状态机** | 不适用 | 9 种任务状态 | 7 种运行状态 | 不适用 |
| **流式传输** | 不适用 | SSE | SSE | 传输层无关（Transport-agnostic） |
| **独特特性** | 工具模式（schemas） | Agent Cards + Skills | 轨迹审计追踪 | 元协议协商 |
| **最佳适用场景** | 工具与数据 | 动态协作 | 受监管行业 | 跨组织信任 |
| **当前状态** | 稳定 | 稳定（v1.0） | 正在并入 A2A | 积极开发中 |

### 它们如何协同工作

这些协议并非互斥。一个现实的企业级系统通常会组合使用多种协议：

graph TB
    subgraph org["Your Organization"]
        RA[Research Agent] <-->|A2A| CA[Coding Agent]
        RA -->|MCP| SS[Search Server]
        CA -->|MCP| GS[GitHub Server]
        AUDIT["All agent responses carry<br/>ACP TrajectoryMetadata"]
    end

    subgraph ext["External (DID verified via ANP)"]
        EA[External Agent]
        PA[Partner Agent]
    end

    RA <-->|ANP + A2A| EA
    CA <-->|ANP + A2A| PA

    style org fill:#f8fafc,stroke:#334155
    style ext fill:#fef2f2,stroke:#991b1b
    style AUDIT fill:#fef3c7,stroke:#d97706

- **MCP** 将每个智能体连接到其工具
- **A2A** 处理智能体之间的协作（内部与外部）
- **ACP** 将响应包裹在轨迹元数据中以实现可审计性
- **ANP** 为你无法直接控制的智能体提供身份验证

## 构建

### 步骤 1：核心消息类型

每个多智能体系统（Multi-Agent System）都始于一种消息格式。我们定义的类型将映射到实际协议所使用的结构：

import crypto from "node:crypto";

type MessageRole = "user" | "agent";

type MessagePart =
  | { kind: "text"; text: string }
  | { kind: "data"; data: unknown; mediaType: string }
  | { kind: "file"; name: string; url: string; mediaType: string };

type TrajectoryEntry = {
  reasoning: string;
  toolName?: string;
  toolInput?: unknown;
  toolOutput?: unknown;
  timestamp: number;
};

type AgentMessage = {
  id: string;
  role: MessageRole;
  parts: MessagePart[];
  trajectory?: TrajectoryEntry[];
  replyTo?: string;
  timestamp: number;
};

function createMessage(
  role: MessageRole,
  parts: MessagePart[],
  replyTo?: string
): AgentMessage {
  return {
    id: crypto.randomUUID(),
    role,
    parts,
    replyTo,
    timestamp: Date.now(),
  };
}

function textMessage(role: MessageRole, text: string): AgentMessage {
  return createMessage(role, [{ kind: "text", text }]);
}

注意：`MessagePart` 支持多模态（Multimodal）（文本、结构化数据、文件），这与实际的 A2A（Agent-to-Agent）和 ACP（Agent Communication Protocol）规范一致。`TrajectoryEntry` 用于捕获推理链（Reasoning Chain），与 ACP 的 TrajectoryMetadata 相匹配。

### 步骤 2：A2A 智能体卡片与注册表

构建符合实际 A2A 规范的智能体发现（Agent Discovery）机制：

type Skill = {
  id: string;
  name: string;
  description: string;
  tags: string[];
  inputModes: string[];
  outputModes: string[];
};

type AgentCard = {
  name: string;
  description: string;
  version: string;
  url: string;
  capabilities: {
    streaming: boolean;
    pushNotifications: boolean;
  };
  defaultInputModes: string[];
  defaultOutputModes: string[];
  skills: Skill[];
};

class AgentRegistry {
  private cards: Map<string, AgentCard> = new Map();

  register(card: AgentCard) {
    this.cards.set(card.name, card);
  }

  discoverBySkillTag(tag: string): AgentCard[] {
    return [...this.cards.values()].filter((card) =>
      card.skills.some((skill) => skill.tags.includes(tag))
    );
  }

  discoverByInputMode(mimeType: string): AgentCard[] {
    return [...this.cards.values()].filter(
      (card) =>
        card.defaultInputModes.includes(mimeType) ||
        card.skills.some((skill) => skill.inputModes.includes(mimeType))
    );
  }

  resolve(name: string): AgentCard | undefined {
    return this.cards.get(name);
  }

  listAll(): AgentCard[] {
    return [...this.cards.values()];
  }
}

这比简单的“名称到能力”映射要丰富得多。你可以像实际 A2A 规范支持的那样，通过技能标签、输入 MIME 类型或名称来发现智能体。

### 步骤 3：A2A 任务生命周期

构建完整的任务状态机（Task State Machine）：

type TaskState =
  | "submitted"
  | "working"
  | "input-required"
  | "auth-required"
  | "completed"
  | "failed"
  | "canceled"
  | "rejected";

const TERMINAL_STATES: TaskState[] = [
  "completed",
  "failed",
  "canceled",
  "rejected",
];

type TaskStatus = {
  state: TaskState;
  message?: AgentMessage;
  timestamp: number;
};

type Artifact = {
  id: string;
  name: string;
  parts: MessagePart[];
};

type Task = {
  id: string;
  contextId: string;
  status: TaskStatus;
  artifacts: Artifact[];
  history: AgentMessage[];
};

type TaskEvent =
  | { kind: "statusUpdate"; taskId: string; status: TaskStatus }
  | {
      kind: "artifactUpdate";
      taskId: string;
      artifact: Artifact;
      append: boolean;
      lastChunk: boolean;
    };

type TaskHandler = (
  task: Task,
  message: AgentMessage
) => AsyncGenerator<TaskEvent>;

class TaskManager {
  private tasks: Map<string, Task> = new Map();
  private handlers: Map<string, TaskHandler> = new Map();
  private listeners: Map<string, ((event: TaskEvent) => void)[]> = new Map();

  registerHandler(agentName: string, handler: TaskHandler) {
    this.handlers.set(agentName, handler);
  }

  subscribe(taskId: string, listener: (event: TaskEvent) => void) {
    const existing = this.listeners.get(taskId) ?? [];
    existing.push(listener);
    this.listeners.set(taskId, existing);
  }

  async sendMessage(
    agentName: string,
    message: AgentMessage,
    contextId?: string
  ): Promise<Task> {
    const handler = this.handlers.get(agentName);
    if (!handler) {
      const task = this.createTask(contextId);
      task.status = {
        state: "rejected",
        timestamp: Date.now(),
        message: textMessage("agent", `No handler for ${agentName}`),
      };
      return task;
    }

    const task = this.createTask(contextId);
    task.history.push(message);
    task.status = { state: "submitted", timestamp: Date.now() };

    this.processTask(task, handler, message).catch((err) => {
      task.status = {
        state: "failed",
        timestamp: Date.now(),
        message: textMessage("agent", String(err)),
      };
    });
    return task;
  }

  getTask(taskId: string): Task | undefined {
    return this.tasks.get(taskId);
  }

  cancelTask(taskId: string): boolean {
    const task = this.tasks.get(taskId);
    if (!task || TERMINAL_STATES.includes(task.status.state)) return false;
    task.status = { state: "canceled", timestamp: Date.now() };
    this.emit(taskId, {
      kind: "statusUpdate",
      taskId,
      status: task.status,
    });
    return true;
  }

  private createTask(contextId?: string): Task {
    const task: Task = {
      id: crypto.randomUUID(),
      contextId: contextId ?? crypto.randomUUID(),
      status: { state: "submitted", timestamp: Date.now() },
      artifacts: [],
      history: [],
    };
    this.tasks.set(task.id, task);
    return task;
  }

  private async processTask(
    task: Task,
    handler: TaskHandler,
    message: AgentMessage
  ) {
    task.status = { state: "working", timestamp: Date.now() };
    this.emit(task.id, {
      kind: "statusUpdate",
      taskId: task.id,
      status: task.status,
    });

    try {
      for await (const event of handler(task, message)) {
        if (TERMINAL_STATES.includes(task.status.state)) break;

        if (event.kind === "statusUpdate") {
          task.status = event.status;
        }
        if (event.kind === "artifactUpdate") {
          const existing = task.artifacts.find(
            (a) => a.id === event.artifact.id
          );
          if (existing && event.append) {
            existing.parts.push(...event.artifact.parts);
          } else {
            task.artifacts.push(event.artifact);
          }
        }
        this.emit(task.id, event);
      }
    } catch (err) {
      task.status = {
        state: "failed",
        timestamp: Date.now(),
        message: textMessage("agent", String(err)),
      };
      this.emit(task.id, {
        kind: "statusUpdate",
        taskId: task.id,
        status: task.status,
      });
    }
  }

  private emit(taskId: string, event: TaskEvent) {
    for (const listener of this.listeners.get(taskId) ?? []) {
      listener(event);
    }
  }
}

这实现了真实的 A2A 任务生命周期（Task Lifecycle）：已提交、处理中、需要输入、终止状态。处理器（Handler）是异步生成器，会产出事件（状态更新和工件数据块），这与 SSE（Server-Sent Events）流式传输模型相匹配。

### 步骤 4：ACP 风格审计轨迹

使用轨迹跟踪（Trajectory Tracking）封装通信过程：

type AuditEntry = {
  runId: string;
  agentName: string;
  input: AgentMessage[];
  output: AgentMessage[];
  trajectory: TrajectoryEntry[];
  status: "created" | "in-progress" | "completed" | "failed" | "awaiting";
  startedAt: number;
  completedAt?: number;
  sessionId?: string;
};

class AuditableRunner {
  private log: AuditEntry[] = [];
  private handlers: Map<
    string,
    (input: AgentMessage[]) => Promise<{
      output: AgentMessage[];
      trajectory: TrajectoryEntry[];
    }>
  > = new Map();

  registerAgent(
    name: string,
    handler: (input: AgentMessage[]) => Promise<{
      output: AgentMessage[];
      trajectory: TrajectoryEntry[];
    }>
  ) {
    this.handlers.set(name, handler);
  }

  async run(
    agentName: string,
    input: AgentMessage[],
    sessionId?: string
  ): Promise<AuditEntry> {
    const entry: AuditEntry = {
      runId: crypto.randomUUID(),
      agentName,
      input: structuredClone(input),
      output: [],
      trajectory: [],
      status: "created",
      startedAt: Date.now(),
      sessionId,
    };
    this.log.push(entry);

    const handler = this.handlers.get(agentName);
    if (!handler) {
      entry.status = "failed";
      return entry;
    }

    entry.status = "in-progress";
    try {
      const result = await handler(input);
      entry.output = structuredClone(result.output);
      entry.trajectory = structuredClone(result.trajectory);
      entry.status = "completed";
      entry.completedAt = Date.now();
    } catch (err) {
      entry.status = "failed";
      entry.trajectory.push({
        reasoning: `Error: ${String(err)}`,
        timestamp: Date.now(),
      });
      entry.completedAt = Date.now();
    }
    return entry;
  }

  getFullAuditLog(): AuditEntry[] {
    return structuredClone(this.log);
  }

  getAuditLogForAgent(agentName: string): AuditEntry[] {
    return structuredClone(
      this.log.filter((e) => e.agentName === agentName)
    );
  }

  getAuditLogForSession(sessionId: string): AuditEntry[] {
    return structuredClone(
      this.log.filter((e) => e.sessionId === sessionId)
    );
  }

  getTrajectoryForRun(runId: string): TrajectoryEntry[] {
    const entry = this.log.find((e) => e.runId === runId);
    return entry ? structuredClone(entry.trajectory) : [];
  }
}

每次智能体执行都会生成完整的审计条目（Audit Entry）：包含输入内容、输出内容，以及中间完整的工具调用和推理步骤轨迹。你可以按智能体、会话或单次运行进行查询。

### 步骤 5：ANP 风格身份验证

构建基于 DID（Decentralized Identifier，去中心化标识符）的身份与验证机制：

type VerificationMethod = {
  id: string;
  type: string;
  controller: string;
  publicKeyDer: string;
};

type DIDDocument = {
  id: string;
  verificationMethod: VerificationMethod[];
  authentication: string[];
  keyAgreement: string[];
  humanAuthorization: string[];
  service: { id: string; type: string; serviceEndpoint: string }[];
};

type AgentIdentity = {
  did: string;
  document: DIDDocument;
  privateKey: crypto.KeyObject;
  publicKey: crypto.KeyObject;
};

class IdentityRegistry {
  private documents: Map<string, DIDDocument> = new Map();

  publish(doc: DIDDocument) {
    this.documents.set(doc.id, doc);
  }

  resolve(did: string): DIDDocument | undefined {
    return this.documents.get(did);
  }

  verify(did: string, signature: string, payload: string): boolean {
    const doc = this.documents.get(did);
    if (!doc) return false;

    const authKeyIds = doc.authentication;
    const authKeys = doc.verificationMethod.filter((vm) =>
      authKeyIds.includes(vm.id)
    );

    for (const key of authKeys) {
      const publicKey = crypto.createPublicKey({
        key: Buffer.from(key.publicKeyDer, "base64"),
        format: "der",
        type: "spki",
      });
      const isValid = crypto.verify(
        null,
        Buffer.from(payload),
        publicKey,
        Buffer.from(signature, "hex")
      );
      if (isValid) return true;
    }
    return false;
  }

  requiresHumanAuth(did: string, operationKeyId: string): boolean {
    const doc = this.documents.get(did);
    if (!doc) return false;
    return doc.humanAuthorization.includes(operationKeyId);
  }
}

function createIdentity(domain: string, agentName: string): AgentIdentity {
  const did = `did:wba:${domain}:agent:${agentName}`;
  const { publicKey, privateKey } = crypto.generateKeyPairSync("ed25519");

  const publicKeyDer = publicKey
    .export({ format: "der", type: "spki" })
    .toString("base64");

  const keyId = `${did}#key-1`;
  const encKeyId = `${did}#key-x25519-1`;

  const document: DIDDocument = {
    id: did,
    verificationMethod: [
      {
        id: keyId,
        type: "Ed25519VerificationKey2020",
        controller: did,
        publicKeyDer,
      },
      {
        id: encKeyId,
        type: "X25519KeyAgreementKey2019",
        controller: did,
        publicKeyDer,
      },
    ],
    authentication: [keyId],
    keyAgreement: [encKeyId],
    humanAuthorization: [],
    service: [
      {
        id: `${did}#agent-description`,
        type: "AgentDescription",
        serviceEndpoint: `https://${domain}/agents/${agentName}/ad.json`,
      },
    ],
  };

  return { did, document, privateKey, publicKey };
}

function signPayload(identity: AgentIdentity, payload: string): string {
  return crypto
    .sign(null, Buffer.from(payload), identity.privateKey)
    .toString("hex");
}

这镜像了真实的 ANP（Agent Network Protocol）身份模型：智能体拥有 DID 文档，其中包含独立的身份验证、密钥协商和人工授权密钥。`IdentityRegistry` 模拟了 DID 解析（DID Resolution）过程（在生产环境中，这将是向智能体域名发起的 HTTP 请求）。

### 步骤 6：协议网关

将四种协议连接到一个统一的系统中：

graph LR
    REQ[Incoming Request] --> ANP_V{ANP: Verify DID}
    ANP_V -->|Valid| A2A_D{A2A: Discover Agent}
    ANP_V -->|Invalid| REJECT[Reject]
    A2A_D -->|Found| ACP_A[ACP: Audit Run]
    A2A_D -->|Not Found| REJECT
    ACP_A --> A2A_T[A2A: Create Task]
    A2A_T --> RESULT[Task + Audit Entry]

    style ANP_V fill:#d1fae5,stroke:#059669
    style A2A_D fill:#dbeafe,stroke:#2563eb
    style ACP_A fill:#fef3c7,stroke:#d97706
    style A2A_T fill:#dbeafe,stroke:#2563eb

class ProtocolGateway {
  private registry: AgentRegistry;
  private taskManager: TaskManager;
  private auditRunner: AuditableRunner;
  private identityRegistry: IdentityRegistry;

  constructor(
    registry: AgentRegistry,
    taskManager: TaskManager,
    auditRunner: AuditableRunner,
    identityRegistry: IdentityRegistry
  ) {
    this.registry = registry;
    this.taskManager = taskManager;
    this.auditRunner = auditRunner;
    this.identityRegistry = identityRegistry;
  }

  async delegateTask(
    fromDid: string,
    signature: string,
    targetAgent: string,
    message: AgentMessage,
    sessionId?: string
  ): Promise<{ task: Task; audit: AuditEntry } | { error: string }> {
    if (!this.identityRegistry.verify(fromDid, signature, message.id)) {
      return { error: "Identity verification failed" };
    }

    const card = this.registry.resolve(targetAgent);
    if (!card) {
      return { error: `Agent ${targetAgent} not found in registry` };
    }

    const audit = await this.auditRunner.run(
      targetAgent,
      [message],
      sessionId
    );
    const task = await this.taskManager.sendMessage(targetAgent, message);

    return { task, audit };
  }

  discoverAndDelegate(
    fromDid: string,
    signature: string,
    skillTag: string,
    message: AgentMessage
  ): Promise<{ task: Task; audit: AuditEntry } | { error: string }> {
    const candidates = this.registry.discoverBySkillTag(skillTag);
    if (candidates.length === 0) {
      return Promise.resolve({
        error: `No agents found with skill tag: ${skillTag}`,
      });
    }
    return this.delegateTask(
      fromDid,
      signature,
      candidates[0].name,
      message
    );
  }
}

网关在一次调用中完成四项操作：
1. **ANP**：通过 DID 签名验证调用者身份
2. **A2A**：发现目标智能体并检查其能力
3. **ACP**：将执行过程封装在带有轨迹的审计日志中
4. **A2A**：创建具备完整生命周期跟踪的任务

### 步骤 7：整合所有组件

async function protocolDemo() {
  const registry = new AgentRegistry();
  registry.register({
    name: "researcher",
    description: "Searches and summarizes findings",
    version: "1.0.0",
    url: "https://researcher.local/a2a/v1",
    capabilities: { streaming: true, pushNotifications: false },
    defaultInputModes: ["text/plain"],
    defaultOutputModes: ["text/plain", "application/json"],
    skills: [
      {
        id: "web-research",
        name: "Web Research",
        description: "Searches the web",
        tags: ["research", "search", "summarization"],
        inputModes: ["text/plain"],
        outputModes: ["application/json"],
      },
    ],
  });
  registry.register({
    name: "coder",
    description: "Writes code from specs",
    version: "1.0.0",
    url: "https://coder.local/a2a/v1",
    capabilities: { streaming: false, pushNotifications: false },
    defaultInputModes: ["text/plain", "application/json"],
    defaultOutputModes: ["text/plain"],
    skills: [
      {
        id: "code-gen",
        name: "Code Generation",
        description: "Generates code",
        tags: ["coding", "generation"],
        inputModes: ["text/plain", "application/json"],
        outputModes: ["text/plain"],
      },
    ],
  });

  const taskManager = new TaskManager();
  const auditRunner = new AuditableRunner();

  const researchTrajectory: TrajectoryEntry[] = [];

  taskManager.registerHandler(
    "researcher",
    async function* (task, message) {
      yield {
        kind: "statusUpdate" as const,
        taskId: task.id,
        status: { state: "working" as const, timestamp: Date.now() },
      };

      researchTrajectory.push({
        reasoning: "Searching for React 19 documentation",
        toolName: "web_search",
        toolInput: { query: "React 19 compiler features" },
        toolOutput: {
          results: ["react.dev/blog/react-19", "github.com/react/react"],
        },
        timestamp: Date.now(),
      });

      researchTrajectory.push({
        reasoning: "Extracting key findings from search results",
        toolName: "doc_analysis",
        toolInput: { url: "react.dev/blog/react-19" },
        toolOutput: {
          summary:
            "React 19 compiler auto-memoizes, no manual useMemo needed",
        },
        timestamp: Date.now(),
      });

      yield {
        kind: "artifactUpdate" as const,
        taskId: task.id,
        artifact: {
          id: crypto.randomUUID(),
          name: "research-results",
          parts: [
            {
              kind: "data" as const,
              data: {
                findings: [
                  "React 19 compiler auto-memoizes components",
                  "No more manual useMemo/useCallback needed",
                  "Compiler runs at build time, not runtime",
                ],
                sources: ["react.dev/blog/react-19"],
              },
              mediaType: "application/json",
            },
          ],
        },
        append: false,
        lastChunk: true,
      };

      yield {
        kind: "statusUpdate" as const,
        taskId: task.id,
        status: { state: "completed" as const, timestamp: Date.now() },
      };
    }
  );

  auditRunner.registerAgent("researcher", async () => ({
    output: [
      textMessage("agent", "React 19 compiler auto-memoizes components"),
    ],
    trajectory: researchTrajectory,
  }));

  const identityRegistry = new IdentityRegistry();

  const coderIdentity = createIdentity("coder.local", "coder");
  const researcherIdentity = createIdentity("researcher.local", "researcher");

  identityRegistry.publish(coderIdentity.document);
  identityRegistry.publish(researcherIdentity.document);

  const gateway = new ProtocolGateway(
    registry,
    taskManager,
    auditRunner,
    identityRegistry
  );

  console.log("=== Protocol Demo ===\n");

  console.log("1. Agent Discovery (A2A)");
  const researchAgents = registry.discoverBySkillTag("research");
  console.log(
    `   Found ${researchAgents.length} agent(s):`,
    researchAgents.map((a) => a.name)
  );

  console.log("\n2. Identity Verification (ANP)");
  const message = textMessage("user", "Research React 19 compiler features");
  const signature = signPayload(coderIdentity, message.id);
  const verified = identityRegistry.verify(
    coderIdentity.did,
    signature,
    message.id
  );
  console.log(`   Coder DID: ${coderIdentity.did}`);
  console.log(`   Signature verified: ${verified}`);

  console.log("\n3. Task Delegation (A2A + ACP + ANP)");
  const result = await gateway.delegateTask(
    coderIdentity.did,
    signature,
    "researcher",
    message,
    "session-001"
  );

  if ("error" in result) {
    console.log(`   Error: ${result.error}`);
    return;
  }

  console.log(`   Task ID: ${result.task.id}`);
  console.log(`   Task state: ${result.task.status.state}`);
  console.log(`   Artifacts: ${result.task.artifacts.length}`);

  console.log("\n4. Audit Trail (ACP)");
  console.log(`   Run ID: ${result.audit.runId}`);
  console.log(`   Status: ${result.audit.status}`);
  console.log(`   Trajectory steps: ${result.audit.trajectory.length}`);
  for (const step of result.audit.trajectory) {
    console.log(`     - ${step.reasoning}`);
    if (step.toolName) {
      console.log(`       Tool: ${step.toolName}`);
    }
  }

  console.log("\n5. Full Audit Log");
  const fullLog = auditRunner.getFullAuditLog();
  console.log(`   Total runs: ${fullLog.length}`);
  for (const entry of fullLog) {
    const duration = entry.completedAt
      ? `${entry.completedAt - entry.startedAt}ms`
      : "in-progress";
    console.log(`   ${entry.agentName}: ${entry.status} (${duration})`);
  }
}

protocolDemo().catch((err) => {
  console.error("Protocol demo failed:", err);
  process.exitCode = 1;
});


## 实际运行中的问题

协议通常只解决理想路径（Happy Path）。以下是生产环境中容易出问题的地方：

**模式漂移（Schema Drift）。** 智能体 A 发布了一张智能体卡片（Agent Card），声明其输出格式为 `application/json`。但不同版本间的 JSON 模式（JSON Schema）发生了变化。智能体 B 仍按旧格式解析，结果得到无效数据。解决方法：对技能和输出模式进行版本控制。A2A 规范（A2A Spec）正是出于此原因，在智能体卡片中支持了 `version` 字段。

**状态机违规（State Machine Violations）。** 智能体处理器触发了一个 `completed` 事件后，又试图产出更多产物（Artifacts）。此时任务状态已不可变。你的代码要么静默丢弃这些更新，要么直接抛出异常。解决方法：在产出事件前检查是否已处于终止状态（Terminal State）。上文中的 `TaskManager` 通过在终止状态后执行 `break` 来强制执行此规则。

**信任解析失败（Trust Resolution Failures）。** 智能体 A 尝试验证智能体 B 的去中心化标识符（DID），但智能体 B 的域名已宕机，导致无法获取 DID 文档。此时你是选择故障开放（Fail Open，即接受未验证的智能体）还是故障关闭（Fail Closed，即拒绝所有请求）？ANP 建议遵循最小信任原则（Principle of Least Trust），采用故障关闭策略。

**轨迹膨胀（Trajectory Bloat）。** ACP 轨迹日志（ACP Trajectory Logging）功能强大但开销高昂。一个复杂的智能体每次运行可能发起 200 次工具调用（Tool Calls），从而生成海量的审计记录。解决方法：按可配置的详细程度级别记录轨迹。为满足合规要求，记录工具名称和输入输出（I/O）；对于非受监管的工作负载，则跳过记录推理步骤。

**发现惊群效应（Discovery Thundering Herd）。** 启动时，50 个智能体同时查询 `GET /agents`。解决方法：为智能体卡片设置带生存时间（TTL）的缓存、错开发现间隔，或使用基于推送的注册机制替代轮询。

## 开始使用

### 实际实现

**智能体到智能体协议（Agent-to-Agent, A2A）** 是目前最成熟的。Google 的[官方规范](https://github.com/google/A2A)已在 Linux 基金会下开源，并提供 Python 和 TypeScript 的 SDK。如果你的智能体需要进行动态发现（Dynamic Discovery）与协作，建议从这里开始。

**智能体通信协议（Agent Communication Protocol, ACP）** 正在并入 A2A。IBM 的 [BeeAI 项目](https://github.com/i-am-bee/acp) 最初将 ACP 打造为一种以 REST 为核心的替代方案，但其轨迹元数据（Trajectory Metadata）概念正被吸收进 A2A 生态系统中。即使你使用 A2A 作为传输层，也建议采用 ACP 的设计模式（如轨迹日志记录、运行生命周期管理）。

**智能体网络协议（Agent Network Protocol, ANP）** 目前最具实验性。其[社区仓库](https://github.com/agent-network-protocol/AgentNetworkProtocol)提供了一个 Python SDK（AgentConnect）。其中的元协议协商（Meta-Protocol Negotiation）概念确实颇具新意。对于跨组织的智能体部署场景，值得持续关注。

**模型上下文协议（Model Context Protocol, MCP）** 已在第 13 阶段中涵盖。如果你希望智能体能够调用工具，MCP 就是行业标准。

### 选择合适的协议

graph TD
    START{Do agents need<br/>to use tools?}
    START -->|Yes| MCP_R[Use MCP]
    START -->|No| TALK{Do agents need to<br/>talk to each other?}
    TALK -->|No| NONE[You don't need<br/>a protocol]
    TALK -->|Yes| AUDIT{Need audit trails<br/>for compliance?}
    AUDIT -->|Yes| ACP_R[A2A + ACP<br/>trajectory patterns]
    AUDIT -->|No| ORG{All agents<br/>within your org?}
    ORG -->|Yes| A2A_R[A2A<br/>Agent Cards + Tasks]
    ORG -->|No| INFRA{Shared<br/>infrastructure?}
    INFRA -->|Yes| BROKER[A2A + message broker]
    INFRA -->|No| ANP_R[ANP + A2A<br/>DID verification]

    style MCP_R fill:#d1fae5,stroke:#059669
    style A2A_R fill:#dbeafe,stroke:#2563eb
    style ACP_R fill:#fef3c7,stroke:#d97706
    style ANP_R fill:#f3e8ff,stroke:#7c3aed
    style BROKER fill:#e0e7ff,stroke:#4338ca

## 交付上线

本章节将产出：
- `code/main.ts` -- 四种协议模式的完整实现代码
- `outputs/prompt-protocol-selector.md` -- 用于辅助你为系统选择协议的提示词（Prompt）

## 练习

1. **多跳任务委派（Multi-hop task delegation）。** 扩展 `TaskManager`，使智能体处理器（agent handler）能够将子任务委派给其他智能体。研究智能体接收任务后，将“搜索”和“总结”子任务分别委派给两个专家智能体，等待两者执行完毕，再将结果合并至自身的产出物（artifacts）中。

2. **流式审计追踪（Streaming audit trail）。** 修改 `AuditableRunner` 以支持流式模式。无需等待完整结果，而是在新增轨迹条目（trajectory entries）时，实时产出 `AuditEntry` 更新。请使用能够生成审计快照的异步生成器（async generator）。

3. **DID 轮换（DID rotation）。** 为 `IdentityRegistry` 添加密钥轮换功能。智能体应能够发布包含更新密钥的新 DID 文档，同时维持对 `previousDid` 的引用。在宽限期（grace period）内，验证者应同时接受当前密钥与历史密钥的签名。

4. **协议协商（Protocol negotiation）。** 实现 ANP 的元协议（meta-protocol）概念。两个智能体交换包含候选格式的 `protocolNegotiation` 消息（例如，“我支持 JSON-RPC”对比“我更倾向于 REST”）。最多经过 3 轮交互后，双方将确定最终格式或触发超时。协商一致的格式将决定它们调用哪个 `TaskManager` 或 `AuditableRunner`。

5. **限流发现（Rate-limited discovery）。** 添加 `RateLimitedRegistry` 包装器，用于缓存智能体卡片（Agent Card）的查询结果（支持可配置的生存时间 TTL），并限制每个智能体每秒的发现查询次数。模拟 100 个智能体在启动时相互发现所引发的惊群效应（thundering herd），并对比测量性能差异。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| MCP | “AI 工具协议” | 一种客户端-服务器协议，用于智能体（Agent）发现和使用工具。属于智能体到工具（Agent-to-tool）的交互，而非智能体到智能体（Agent-to-agent）。 |
| A2A | “Google 的智能体协议” | Linux 基金会旗下的对等网络（Peer-to-Peer）协议，用于智能体协作。通过智能体卡片（Agent Card）进行发现，包含 9 个状态的任务生命周期，并通过服务器发送事件（Server-Sent Events, SSE）实现流式传输。支持 JSON-RPC、REST 和 gRPC 绑定。 |
| ACP | “企业级智能体消息传递” | IBM/BeeAI 提供的用于智能体运行的 REST API，包含 `TrajectoryMetadata`：每个响应都携带完整的推理链和工具调用记录。目前正在并入 A2A。 |
| ANP | “去中心化智能体身份” | 一种社区协议，使用 `did:wba`（去中心化标识符，DID）进行加密身份验证，采用 HPKE 实现端到端加密（End-to-End Encryption, E2EE），并通过 AI 驱动的元协议（Meta-protocol）协商机制，使从未交互过的智能体能够建立通信。 |
| Agent Card | “智能体的名片” | 位于 `/.well-known/agent-card.json` 的 JSON 文档，用于描述智能体的技能、支持的 MIME 类型、安全方案及协议绑定。 |
| DID | “去中心化身份” | W3C 标准，用于在智能体自有域名上托管可密码学验证的身份。ANP 使用 `did:wba` 方法。 |
| TrajectoryMetadata | “审计凭证” | ACP 的一种机制，用于将推理步骤、工具调用及其输入/输出附加到每个智能体响应中。 |
| Meta-protocol | “智能体协商通信方式” | ANP 采用的一种方法，智能体使用自然语言动态协商数据格式，随后生成代码来处理这些格式。 |
| Task | “工作单元” | A2A 中的有状态对象，用于跟踪工作从提交到完成的全过程。一旦进入终止状态即不可变。 |

## 延伸阅读

- [Google A2A 规范](https://github.com/google/A2A) -- 官方规范与 SDK（v1.0.0，Linux 基金会）
- [IBM/BeeAI ACP 规范](https://github.com/i-am-bee/acp) -- 用于智能体运行和轨迹元数据的 OpenAPI 3.1 规范
- [Agent Network Protocol](https://github.com/agent-network-protocol/AgentNetworkProtocol) -- 基于 DID 的身份验证、端到端加密（E2EE）与元协议协商
- [Model Context Protocol 文档](https://modelcontextprotocol.io/) -- Anthropic 的 MCP 规范（已在第 13 阶段涵盖）
- [W3C 去中心化标识符](https://www.w3.org/TR/did-core/) -- 支撑 ANP 的身份标准
- [RFC 9180 (HPKE)](https://www.rfc-editor.org/rfc/rfc9180) -- ANP 用于端到端加密（E2EE）的加密方案
- [FIPA 智能体通信语言](http://www.fipa.org/specs/fipa00061/SC00061G.html) -- 现代智能体协议的学术先驱