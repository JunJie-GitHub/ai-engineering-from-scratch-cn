---
name: agent-bundle
description: 为工作流生成可移植的 SKILL.md + AGENTS.md + MCP 服务器蓝图，可在 Claude Code、Cursor、Codex 及兼容的代理（agent）中加载。
version: 1.0.0
phase: 13
lesson: 21
tags: [技能, agents-md, apps-sdk, 跨代理, 可移植性]
---

根据工作流描述，生成一个代理包（agent bundle）。

生成以下内容：

1. `SKILL.md`。包含 `name` 和 `description` 的 YAML frontmatter，以及带有编号步骤的 Markdown 正文。如果正文较长，需包含渐进式披露（progressive disclosure）子资源引用。
2. `AGENTS.md` 条目。在仓库的 `AGENTS.md` 中添加几行内容，反映该技能所依赖的任何规范（如代码检查命令、测试命令）。
3. MCP 服务器蓝图。说明该技能通过 MCP 调用哪些工具；包括名称、描述（使用场景模式/Use-when pattern）以及输入模式（input schema）。
4. 跨代理转换说明。以 SkillKit 风格的注释说明此 `SKILL.md` 如何映射到 Cursor 规则、Codex `.codex.md` 以及 Windsurf 规则。
5. 加载路径。代理发现此包的位置：`~/.anthropic/skills/`、`./skills/`、`~/.claude/skills/`。

硬性拒绝条件：
- 任何 `name` 不符合 `kebab-case`（短横线命名法）的 `SKILL.md`。会导致发现机制失效。
- 任何 frontmatter 中缺少 `description` 的 `SKILL.md`。代理运行时会跳过它。
- 任何 MCP 工具未按照“第 13 阶段 · 05 规则”命名的包。

拒绝规则：
- 如果工作流仅为单次提示词（one-shot prompt），则拒绝生成技能；建议采用内联提示词工程（inline prompt-engineering）。
- 如果工作流需要 OAuth 认证（例如发布 Slack 消息），需标记说明 MCP 服务器的首次运行引导（first-run elicitation）必须处理该流程。
- 如果目标代理不支持 `SKILL.md`（部分 IDE），建议通过 SkillKit 或类似工具进行转换。

输出：一份单页文档，包含上述三个文件的草图、跨代理转换说明以及加载路径。最后注明应优先用于测试该包的单个代理。