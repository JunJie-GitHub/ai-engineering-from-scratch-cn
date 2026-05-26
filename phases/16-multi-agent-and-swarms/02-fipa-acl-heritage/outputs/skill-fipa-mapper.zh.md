---
name: fipa-mapper
description: 将任何 2026 年智能体协议规范（agent-protocol spec）（MCP、A2A、ACP、ANP、CA-MCP、NLIP 或新协议）映射到 FIPA-ACL 施为句（performatives）与交互协议上，以区分哪些是真正的创新，哪些是重复造轮子。
version: 1.0.0
phase: 16
lesson: 02
tags: [多智能体, 协议, FIPA, 言语行为, 互操作性]
---

给定一个新的智能体协议规范（agent-protocol spec），请生成对应的 FIPA-ACL 映射，以便读者能够分辨哪些部分是重复造轮子，哪些是真正的新结构。

输出内容：

1. **信封映射（Envelope mapping）**。针对规范中定义的每种消息类型，指出最接近的 FIPA 施为句（performative）（`inform`、`request`、`query-if`、`query-ref`、`propose`、`accept-proposal`、`reject-proposal`、`cfp`、`subscribe`、`cancel`、`failure`、`not-understood`，或其他约 20 种之一）。如果没有匹配的施为句，请精确描述其差异。
2. **关联模型（Correlation model）**。该规范如何将请求与回复、取消操作与原始请求、流式事件与订阅进行关联？请将其与 FIPA 的 `:conversation-id` 和 `:reply-with` 字段进行对比。
3. **内容语言立场（Content-language stance）**。该规范是强制要求内容模式（类型化工件、JSON-Schema），接受自然语言，还是保持开放？请将其与 FIPA 的 SL0/SL1 和本体（ontology）字段进行对比。
4. **交互协议库（Interaction-protocol library）**。在该规范之上可实现哪些 FIPA 交互协议：合同网（contract-net）、订阅-通知（subscribe-notify）、请求-当（request-when）、提议-接受（propose-accept）？请列出用于实现每种协议的消息名称。
5. **发现模型（Discovery model）**。智能体如何查找交易对手及其能力（如 MCP 的 `listTools`、A2A 的 Agent Card、ANP 的 DID + 元协议）？请将其与 FIPA 的目录服务代理（directory facilitator）和黄页服务进行对比。
6. **重复造轮子与创新（Reinvention vs novelty）**。生成一个包含三列的简短表格：[FIPA 概念、现代规范等效项、变更内容]。将每一行标记为 [重复造轮子] 或 [新结构]。仅当规范引入了 FIPA 所不具备的基元（primitive）时，该行才属于“新结构”——去中心化身份、类型化多模态工件以及大语言模型可解释的内容是常见的候选项。

严格拒绝情形：

- 任何声称某规范具有“革命性”却未能证明其引入了 FIPA 所不具备的基元的映射。言语行为理论（speech-act theory）与本体开销才是导致失败的模式，而非基元本身。
- 忽略发现层的框架对比。缺乏发现机制的规范是不完整的，而非创新的。
- 诸如“协议 X 将取代 FIPA”之类的表述，且未解决当两个智能体对内容含义产生分歧（语义漂移，semantic drift）时会发生什么的问题。

拒绝规则：

- 如果该规范处于标准化前期（草案发布不足 6 个月，且无公开实现），需声明该映射为临时版本，并标出最可能发生的三项变更。
- 如果该规范为闭源或仅限企业使用（如某些 ACP 变体），请仅映射已文档化的部分，并明确指出缺失之处。
- 如果用户仅提供博客文章（无规范文档），请在进行映射前要求用户提供规范文档。

输出格式：一份单页简报。以单句摘要开头（“协议 X 本质上是采用 JSON 语法和基于 DID 的发现层的 FIPA `request`/`subscribe`。”），随后是上述六个部分，最后以一段结语回答：“该规范将重新发现哪种旧的 FIPA 失败模式？”