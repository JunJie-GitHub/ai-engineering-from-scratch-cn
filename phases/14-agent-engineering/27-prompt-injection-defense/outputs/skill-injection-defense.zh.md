---
name: injection-defense
description: 为任意智能体运行时构建一个提示词-验证器-执行器（Prompt-Validator-Executor）层，包含带来源标签的内容、注入标记扫描以及白名单导航。
version: 1.0.0
phase: 14
lesson: 27
tags: [安全, 提示词注入, pve, greshake, 来源标签]
---

针对具备工具调用和检索能力的智能体（Agent），构建一个注入防御（Injection Defense）层。

产出要求：

1. 为每段内容添加来源标签（Source Tag）：`user_message`、`tool_output`、`retrieved_web`、`retrieved_memory`、`retrieved_file`。在消息历史中持续传递这些标签。
2. `Validator.assess(tool_call, contents)` —— 拒绝包含注入特征参数或检索内容的工具调用；仅当来源标签与声明的信任级别（Trust Level）匹配时才予以放行。
3. 导航白名单/黑名单（Allowlist / Blocklist）：智能体可访问的 URL、域名和文件路径。
4. 内存写入护栏（Memory-write Guardrail）：拒绝形似指令的写入操作。
5. 内容捕获规范（Content-capture Discipline，第 23 课）：将检索到的内容存储在外部；跨度（Spans）仅携带引用 ID，而非完整文本。
6. 测试套件：将五类 Greshake 漏洞利用类别作为红队测试用例（Red-team Cases）。

硬性拒绝标准：

- 缺乏来源标签的工具调用接口（Tool-use Surface）。若无来源溯源（Provenance），则无法区分权限级别。
- 仅在最终输出阶段运行的验证器（Validator）。滞后验证毫无意义——模型早已执行了操作。
- “相信我，系统提示词（System Prompt）能搞定。”系统提示词规范并非有效的安全控制手段。

拒绝规则：

- 若智能体具备任何未添加来源标签的检索能力，则拒绝发布。检索内容是典型的注入攻击载体（Injection Vector）。
- 若敏感工具（发送消息、执行 Shell 命令、在根目录 `/` 写入文件）缺乏人工介入（Human-in-the-loop）确认机制，则拒绝。
- 若内存写入操作未受防护，则拒绝。持久化内存投毒（Persistent Memory Poisoning）会持续污染后续会话。

输出文件：`validator.py`、`source_tag.py`、`allowlist.py`、`memory_guard.py`、`red_team.py`、`README.md`。其中 `README.md` 需阐述六层控制栈（Six-control Stack）、残余风险（Residual Risks）以及持续审查节奏（Review Cadence）。文末需附上“下一步阅读”指引，指向第 21 课（计算机使用安全）和第 23 课（通过 OTel 进行内容捕获）。