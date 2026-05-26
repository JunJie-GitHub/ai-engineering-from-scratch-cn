# FIPA-ACL 与言语行为（Speech Acts）的传承

> 在模型上下文协议（MCP）与智能体间通信协议（A2A）问世之前，FIPA-ACL 早已存在。2000 年，IEEE 智能物理代理基金会（FIPA）批准了一种智能体通信语言，包含二十种施为型（performatives）、两种内容语言以及一套交互协议——合同网协议（contract net）、订阅/通知（subscribe/notify）与条件请求（request-when）。该标准之所以在工业界逐渐淡出，是因为其本体论（ontology）开销对当时的 Web 而言过于沉重；但随着大语言模型（LLM）推动多智能体系统（multi-agent systems）的复兴，业界正在悄然重新实现这些理念，只是剥离了形式化语义：JSON 契约替代了施为型，自然语言替代了本体论。本节将深入研读 FIPA-ACL，助你厘清 2026 年的协议决策中哪些是重复造轮子，哪些是真正的新颖设计，以及当前的技术浪潮将在何处重新遭遇那些 2000 年代早已解决的问题。

**Type:** 学习
**Languages:** Python（标准库）
**Prerequisites:** 第 16 阶段 · 01（为何使用多智能体）
**Time:** 约 60 分钟

## 问题背景

2026 年的智能体协议（agent-protocol）生态可谓百花齐放：面向工具的 MCP、面向智能体的 A2A、用于企业审计的 ACP、用于去中心化信任的 ANP、用于自然语言内容的 NLIP，再加上 CA-MCP 以及二十余项研究提案。每项规范都宣称自己是基石。

客观来看，其中大多数规范实际上是在重新探索一棵二十年前就已非常明确的决策树。奥斯汀（Austin, 1962）与塞尔（Searle, 1969）提出的言语行为理论（Speech-act theory）为我们奠定了“话语即行为”的理念。知识查询与操作语言（KQML，1993）将其转化为网络传输协议。FIPA-ACL（2000 年获批）则确立了参考标准：二十种施为型、SL0/SL1 内容语言，以及用于合同网与订阅/通知的交互协议。JADE 和 JACK 曾是 Java 领域的参考平台。该方向在 2010 年左右逐渐式微，原因在于本体论开销过于沉重，且 Web 架构最终胜出。

当你审视 MCP 的 `tools/call`、A2A 的任务生命周期（task lifecycle），或是 CA-MCP 的共享上下文存储（shared context store）时，你看到的其实是 FIPA 决策的一种更轻量、原生支持 JSON 的重新演绎。了解这段传承能告诉你两件事：哪些所谓的“创新”实则是重复造轮子，以及新规范将会重新踩中哪些旧有的失败模式（failure modes）。

## 核心概念

### 一句话概括言语行为（Speech Acts）

奥斯汀（Austin）注意到，有些句子并非在描述世界，而是在改变世界。“我承诺。”“我请求。”“我宣布。”他将这类话语称为施为句（performative utterances）。塞尔（Searle）将其形式化为五大类别：断言类（assertive）、指令类（directive）、承诺类（commissive）、表达类（expressive）和宣告类（declarative）。KQML（Finin 等人，1993）使这一理论在软件智能体（software agents）中得以落地：一条消息由一个施为类型（performative，即动作）和内容（content，即动作的对象）组成。FIPA-ACL（FIPA 智能体通信语言）弥补了 KQML 的缺陷，并标准化了约二十种施为类型。

### FIPA 的二十种施为类型（部分列表）

| Performative | Intent |
|---|---|
| `inform` | “我告知你 P 为真” |
| `request` | “我请求你执行 X” |
| `query-if` | “P 是否为真？” |
| `query-ref` | “X 的值是什么？” |
| `propose` | “我建议我们执行 X” |
| `accept-proposal` | “我接受该提议” |
| `reject-proposal` | “我拒绝该提议” |
| `agree` | “我同意执行 X” |
| `refuse` | “我拒绝执行 X” |
| `confirm` | “我确认 P 为真” |
| `disconfirm` | “我否认 P” |
| `not-understood` | “无法解析你的消息” |
| `cfp` | “就 X 征集提案” |
| `subscribe` | “当 X 发生变化时通知我” |
| `cancel` | “取消正在进行的 X” |
| `failure` | “我尝试执行 X 但失败了” |

完整列表见 `fipa00037.pdf`（FIPA ACL 消息结构）。重点不在于死记硬背，而在于这些类型中的每一个，最终都会作为基础原语（primitive）被大语言模型（LLM）协议重新引入。

### 标准的 FIPA-ACL 消息格式

(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)

其中七个字段承载协议信封（protocol envelope），一个字段（`content`）承载有效载荷（payload）。其余字段正是你每次在 JSON 协议上硬塞重试机制、线程管理和本体（ontology）时，不得不重新发明的那些东西。

### 两大遗留平台

**JADE**（Java 智能体开发框架，1999–2020 年代）是使用最广泛的符合 FIPA 标准的运行时环境。智能体通过继承基类、交换 ACL 消息、在容器内运行，并使用“行为（behaviors）”进行协调。其内置的交互协议库包含了合同网（contract-net）、订阅/通知（subscribe-notify）、条件请求（request-when）以及提议/接受（propose-accept）等模式。

**JACK**（Agent Oriented Software，商业软件）在 FIPA 消息之上强调 BDI（信念-愿望-意图，Belief-Desire-Intention）推理。它更为形式化，但采用率较低。

随着 Web 技术栈吞噬了多智能体（multi-agent）用例，两者均走向衰落。MCP 和 A2A 正是 2026 年的运行时“容器”。

### FIPA 为何衰落

- **本体（Ontology）开销过大。** FIPA 要求共享本体才能解析 `content`。就本体达成一致是一个耗时数年的标准化过程。而 Web 技术栈直接采用了 HTTP + JSON。
- **无人使用的形式化语义（formal semantics）。** SL（语义语言，Semantic Language）提供了严格的真值条件，但大多数生产系统使用自由格式的内容，完全忽略了这些形式化规范。
- **工具链锁定。** JADE 仅限 Java；JACK 是商业软件。多语言技术栈团队纷纷绕开它们另寻他路。
- **互联网技术栈胜出。** REST，随后是 JSON-RPC，再到 gRPC，逐步取代了 ACL 的传输层。

### 大语言模型的复兴是“轻量版 FIPA（FIPA-lite）”

将 FIPA 的 `request` 与 MCP 的 `tools/call` 进行对比：

(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }

相同的信封结构，不同的语法。两者都承载了：发送方、接收方、意图、有效载荷和相关 ID（correlation id）。两者并非谁颠覆了谁，而是同一设计理念下的不同权衡取舍。

Liu 等人于 2025 年发表的综述论文（《智能体互操作协议综述：MCP、ACP、A2A、ANP》，arXiv:2505.02279）明确指出了这一传承脉络：MCP 对应工具使用类言语行为，A2A 对应智能体对等类言语行为，ACP 对应审计追踪类言语行为，ANP 对应去中心化身份扩展。这些新规范本质上是采用 JSON 语法且语义更宽松的 ACL 后代。

### 直白地说，这就是权衡

**FIPA 提供而现代规范舍弃的：**

- 形式化语义——你可以证明 `inform` 蕴含发送方相信该内容为真。
- 标准化的施为类型目录——你无需再反复争论“我们是否需要一个 `cancel`？”。
- 数十年的交互协议模式——合同网、订阅/通知、提议/接受——均具备已验证的正确性属性。

**现代规范提供而 FIPA 不具备的：**

- 原生 JSON 有效载荷，兼容所有现代工具。
- 自然语言内容，大语言模型无需人工编码的本体即可理解。
- Web 技术栈传输协议（HTTP、SSE、WebSocket）。
- 通过自描述文档进行能力发现（MCP `listTools`，A2A Agent Card）。

用更宽松的意图语义换取更易实现。这正是核心权衡所在。

### 值得移植的交互协议

FIPA 内置了约 15 种交互协议。其中三种值得引入大语言模型多智能体系统中：

1. **合同网协议（Contract Net Protocol, CNP）。** 管理者发布 `cfp`（征集提案）；竞标者回复 `propose`；管理者接受或拒绝。这是标准的任务市场模式（Phase 16 · 16 Negotiation）。
2. **订阅/通知（Subscribe/Notify）。** 订阅者发送 `subscribe`；每当主题发生变化时，发布者发送 `inform`。这就是 2026 年所有事件总线（event-bus）的底层逻辑。
3. **条件请求（Request-When）。** “当条件 Y 满足时执行 X。” 带有前置条件的延迟执行。其在 2026 年的对应物是持久化工作流引擎（durable workflow engines）中的延迟任务（Phase 16 · 22 Production Scaling）。

它们都能清晰地映射到现代消息队列、HTTP + 轮询或 SSE 流式传输上。

### 舍弃本体后会出什么问题

缺乏共享本体时，智能体只能从自然语言内容中推断含义。2026 年已记录的典型故障模式是**语义漂移（semantic drift）**：两个智能体对同一个词（如 `"customer"`）赋予了细微不同的概念，接收方智能体基于错误的理解采取行动，且没有任何模式验证器能捕获此问题。而 FIPA 的本体要求会在解析阶段直接拒绝该消息。

在不引入完整本体的情况下的缓解措施：

- 对 `content` 应用 JSON Schema——在网络传输层直接拒绝结构错误。
- 类型化构件（typed artifacts，A2A）——拒绝错误的模态。
- 在信封中显式声明施为类型——即使内容是自然语言，也能确保意图明确无误。

### 2026 年规范与言语行为传承的映射

| Modern spec | FIPA analog | What it keeps | What it drops |
|---|---|---|---|
| MCP `tools/call` | `request` | 显式意图、相关 ID | 形式化语义、本体 |
| MCP `resources/read` | `query-ref` | 显式意图、相关 ID | 形式化语义 |
| A2A Task lifecycle | contract-net + request-when | 异步生命周期、状态转换 | 形式化完备性保证 |
| A2A streaming events | subscribe/notify | 异步推送 | 类型化谓词订阅 |
| CA-MCP shared context | blackboard (Hayes-Roth 1985) | 多写入者共享内存 | 逻辑一致性模型 |
| NLIP | natural-language content | 原生适配大语言模型 | 模式/架构 |

从上到下阅读该表，其规律显而易见：保留结构原语，舍弃形式化规范，交由大语言模型来消解歧义。

## 构建它

`code/main.py` 实现了一个仅依赖标准库的 FIPA-ACL（智能物理代理基金会代理通信语言）转换器。它对标准的 ACL 信封（ACL envelope）进行编码与解码，并展示了所有 MCP（模型上下文协议）与 A2A（代理间通信）消息结构如何统一归约为相同的七个字段。该演示：

- 将五条 MCP 风格与 A2A 风格的消息编码为 FIPA-ACL 格式。
- 将 FIPA-ACL 解码回现代等效格式。
- 使用 `cfp`、`propose`、`accept-proposal` 和 `reject-proposal` 指令，在一个管理者与三个竞标者之间运行一个简易的合同网（Contract Net）协商演示。

运行：

python3 code/main.py

输出结果是一个并排追踪日志，展示了每条现代消息的 2026 年 JSON 格式与其 FIPA-ACL 格式，随后是合同网竞标的往返（round-trip）过程。相同的协议原语（protocol primitives）在往返过程中得以保留，仅语法有所不同。

## 使用它

`outputs/skill-fipa-mapper.md` 是一个技能（skill），用于读取任意代理协议规范（agent-protocol spec）并生成对应的 FIPA-ACL 映射。在采用新协议前使用它，以回答以下问题：“这真的是全新协议，还是仅仅换了 JSON 语法的 `inform`（告知）指令？”

## 发布它

不要将 FIPA-ACL 重新引入。但请带回它的检查清单：

- 每条消息的意图原语（performative）是什么？
- 是否包含用于请求-响应与取消操作的相关性 ID（correlation id）？
- 是否指定了明确的内容语言（content language）（如 JSON-RPC、纯文本或结构化类型化产物）？
- 交互协议（interaction protocols）是否作为一等公民支持，还是你正在从零开始重新实现合同网协议？
- 当两个代理对内容含义产生分歧（语义漂移，semantic drift）时会发生什么？

在将任何新协议投入生产环境之前，请先将上述五个问题记录在案。

## 练习

1. 运行 `code/main.py`。观察往返编码过程。找出与 `tools/call`、`resources/read` 以及 A2A 任务创建相对应的 FIPA 意图原语。
2. 为合同网演示扩展一个 `cancel` 意图原语，使管理者能够在竞标中途撤回任务。`cancel` 解决了哪些仅靠重试无法解决的故障场景？
3. 阅读 FIPA ACL 消息结构（http://www.fipa.org/specs/fipa00037/）第 4.1–4.3 节。挑选一个本课程未涵盖的意图原语，并描述其对应的现代 JSON-RPC 等效实现。
4. 阅读 Liu 等人的论文（arXiv:2505.02279）。针对 MCP、A2A、ACP 和 ANP，分别列出它们保留和舍弃的 FIPA 意图原语族（performative families）。
5. 为你自己系统中的 `request` 意图原语的 `content` 字段设计一个最简 JSON-Schema。该 Schema 能提供哪些纯自然语言无法提供的能力，又会带来什么代价？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 言语行为 (Speech Act) | “能执行操作的话语” | Austin/Searle 理论：话语即行为。智能体通信语言 (Agent Communication Language, ACL) 的理论源头。 |
| 国际智能物理代理基金会 (Foundation for Intelligent Physical Agents, FIPA) | “那个老旧的 XML 玩意儿” | IEEE 下属机构。于 2000 年对 ACL 进行了标准化。 |
| 智能体通信语言 (Agent Communication Language, ACL) | “智能体通信语言” | FIPA 的消息信封格式：包含施为型 (Performative) + 内容 + 元数据。 |
| 施为型 (Performative) | “动词” | 消息的意图类别：`inform`、`request`、`propose`、`cfp` 等。 |
| 知识查询与操作语言 (Knowledge Query and Manipulation Language, KQML) | “FIPA 的前身” | 知识查询与操作语言（1993 年）。更简单、适用范围更窄。 |
| 本体 (Ontology) | “共享词汇表” | 对内容语言所讨论概念的正式定义。 |
| SL0 / SL1 (语义语言 0/1 级) | “FIPA 内容语言” | 语义语言 (Semantic Language) 0 级和 1 级——正式的内容语言家族。 |
| 合同网协议 (Contract Net) | “任务市场” | 管理者发布 `cfp`；竞标者提交提案；管理者接受。经典的交互协议。 |
| 交互协议 (Interaction Protocol) | “消息模式” | 具有已知正确性的施为型序列：request-when、subscribe-notify 等。 |

## 延伸阅读

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) —— 2025 年权威综述，将现代规范与 FIPA 传统相连接
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) —— 2000 年正式批准的消息信封格式规范
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) —— 完整的施为型 (Performative) 目录
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) —— 现代工具调用规范，相当于 `request`/`query-ref` 的当代实现
- [A2A specification](https://a2a-protocol.org/latest/specification/) —— 现代智能体对等通信规范，相当于合同网协议 (Contract Net) 与 subscribe-notify 的当代实现