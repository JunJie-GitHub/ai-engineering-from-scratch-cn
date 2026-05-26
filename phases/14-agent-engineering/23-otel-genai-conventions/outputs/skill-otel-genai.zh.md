---
name: otel-genai
description: 使用 OpenTelemetry GenAI 语义规范（OpenTelemetry GenAI Semantic Conventions）对智能体进行插桩——为 `invoke_agent`、`chat`、`tool_call` 跨度（span）配置正确的属性，并支持按需内容捕获。
version: 1.0.0
phase: 14
lesson: 23
tags: [opentelemetry, 生成式AI, 可观测性, 链路追踪, 语义规范]
---

给定智能体运行时（agent runtime），集成 OTel GenAI 语义规范。

产出要求：

1. 每次智能体运行生成一个 `invoke_agent` 跨度。针对远程智能体服务，跨度类型（Kind）设为 CLIENT；针对进程内调用，设为 INTERNAL。名称格式：`invoke_agent {gen_ai.agent.name}`。
2. 每次大语言模型（LLM）调用生成一个 `chat` 跨度，需包含 `gen_ai.operation.name=chat`、`gen_ai.provider.name`、`gen_ai.request.model` 和 `gen_ai.response.model` 属性。
3. 每次工具调用生成一个 `tool_call` 跨度，需包含 `gen_ai.tool.name` 属性，并在适用时包含 `gen_ai.data_source.id`（用于检索增强生成（RAG）语料库或记忆存储）。
4. 按需内容捕获（Opt-in content capture）：默认关闭；开启时，将输入/输出存储至外部系统，并在跨度上记录 `*.reference_id`。
5. 上下文传播（Context propagation）：使用 W3C 追踪上下文标头（W3C trace context headers），确保多进程运行（如 Claude Agent SDK CLI 子进程）能够拼接为同一条追踪链路。

硬性拒绝条件：

- 默认以内联方式捕获完整的提示词（prompt）与输出。存在个人身份信息（PII）及密钥泄露风险，且违反规范。
- 缺失 `gen_ai.provider.name` 属性。会导致多提供商仪表盘（dashboard）功能异常。
- 出现孤立工具跨度。必须始终通过活跃上下文（active context）设置父子关系。

拒绝规则：

- 若运行时无法跨进程边界传播上下文，则拒绝实现。Claude Agent SDK + CLI 用户必须进行多进程追踪拼接。
- 若产品受监管合规要求限制（如 HIPAA、GDPR），则拒绝内联内容捕获。仅允许使用具备访问控制的外部存储。
- 若后端未设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，则发出警告：在收集器（collector）升级时，属性名称可能会发生变更。

输出文件：`tracer.py`、`attributes.py`、`content_store.py`、`README.md`。其中 `README.md` 需说明跨度结构、稳定性按需启用（stability opt-in）机制以及内容捕获策略。文档末尾需包含“下一步阅读”部分，指向第 24 课（后端：Langfuse、Phoenix、Opik）或第 17 课（Claude Agent SDK 追踪上下文传播）。