---
name: 交接生成器（handoff-generator）
description: 从工作台工件（workbench artifacts）生成会话结束交接数据包，产出人类可读的 Markdown 和机器可读的 JSON，并以七个规范字段为键。
version: 1.0.0
phase: 14
lesson: 40
tags: [交接, 生成器, 会话结束, 数据包, 下一步行动]
---

给定工作台（workbench，包含状态、裁决结论、评审记录、反馈日志、差异对比），构建一个接入代理运行时（agent runtime）的会话结束交接生成器。

产出内容：

1. `tools/generate_handoff.py`，对外暴露 `generate_handoff(snapshot) -> (markdown, payload)` 接口。
2. `outputs/handoff/<session_id>/handoff.md` 与 `handoff.json`。
3. `handoff.schema.json`，涵盖七个必填字段及反馈尾部（feedback tail）格式。
4. 会话结束钩子脚本（session-end hook script），用于运行生成器，并在缺失任何字段时拒绝关闭会话。
5. `docs/handoff.md`，列出七个字段、其数据来源以及裁剪策略。

硬性拒绝条件：

- 缺少 `next_action` 的交接。伪装成交接的状态报告会污染下一个会话。
- 手动编写摘要的生成器。代理（agent）的职责是让工作台处于可生成状态。
- 与 JSON 不一致的 Markdown 数据包。JSON 是数据源；Markdown 仅是 JSON 的渲染结果。
- 超过 30 条记录的反馈尾部。完整日志已存入版本控制；数据包必须保持精简。

拒绝规则：

- 若缺少验证报告（verification report），则拒绝生成数据包。没有裁决结论的交接只是空想。
- 若缺少评审报告且预期需要人工评审，则拒绝生成，并要求先通过评审。
- 若差异对比摘要（diff summary）为空但会话运行时间超过 5 分钟，则在生成前暴露该异常；应怀疑会话卡死（wedged session），而非真正的无操作（no-op）。

输出结构：

<repo>/
├── outputs/handoff/<session_id>/
│   ├── handoff.md
│   └── handoff.json
├── tools/generate_handoff.py
├── handoff.schema.json
└── docs/handoff.md

结尾附上“下一步阅读”指引，指向：

- 第 41 课：在真实风格的示例应用上进行端到端（end-to-end）练习。
- 第 42 课：将生成器打包至核心工作台套件（capstone workbench pack）。
- 第 29 课（生产运行时）：将会话结束逻辑接入队列、事件和定时任务（cron）触发器。