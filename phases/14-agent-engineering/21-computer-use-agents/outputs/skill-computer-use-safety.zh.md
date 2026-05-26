---
name: 计算机使用安全
description: 为计算机使用代理构建逐步安全分类器（per-step safety classifier）与确认门控（confirmation gate），支持白名单（allowlist）导航与注入标记（injection-marker）过滤。
version: 1.0.0
phase: 14
lesson: 21
tags: [计算机使用, 安全, claude, openai-cua, gemini]
---

给定一个计算机使用代理（computer-use agent）和目标应用列表，构建一个安全层（safety layer），在执行前对每个操作进行分类。

产出内容：

1. `SafetyClassifier.assess(action, screen) -> SafetyVerdict`，包含字段 `allow`、`reason`、`needs_confirmation`。
2. 代理可点击的元素标签白名单；否则拒绝。
3. 代理可导航的 URL 白名单；若重定向至列表外则拒绝。
4. 针对 DOM 文本、检索内容及输入文本的注入标记过滤器。任何匹配项都将阻断该操作。
5. 针对敏感操作（登录、购买、删除、发布）的确认门控。提供人在回路（human-in-the-loop）回调接口。
6. 追踪发射器（trace emitter）：记录每项决策，包含（操作、裁决结果、原因）。

硬性拒绝条件（hard rejects）：

- 仅在首个操作运行的安全分类器。必须对每个操作进行分类。
- 形式为 `*` 的白名单。允许一切的白名单不叫白名单。
- 因模型“看似自信”而跳过确认。自信不等于安全。

拒绝规则（refusal rules）：

- 若代理具备计算机使用权限但缺乏逐步安全机制，则拒绝发布。
- 若代理可导航至任意 URL，则拒绝。必须要求白名单或黑名单（blocklist）。
- 若敏感操作在任何模式下绕过确认门控，则拒绝。

输出文件：`classifier.py`、`allowlist.py`、`confirmation.py`、`trace.py`、`README.md`（需说明门控策略、注入标记及白名单维护流程）。结尾附上“下一步阅读”指引，指向第 27 课（提示词注入 prompt injection）与第 23 课（用于安全决策的 OTel 跨度归因 OTel span attribution）。