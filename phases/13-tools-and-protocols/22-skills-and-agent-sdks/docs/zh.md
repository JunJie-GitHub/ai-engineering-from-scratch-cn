# 技能与智能体 SDK（Skills and Agent SDKs）—— Anthropic 智能体技能（Anthropic Agent Skills）、AGENTS.md、OpenAI Apps SDK

> 模型上下文协议（Model Context Protocol, MCP）说明“存在哪些工具”。技能（Skills）说明“如何完成任务”。2026 年的技术栈将两者分层结合。Anthropic 的智能体技能（Agent Skills，2025 年 12 月发布的开放标准）以 SKILL.md 文件形式交付，并采用渐进式披露（progressive disclosure）机制。OpenAI 的 Apps SDK 本质上是 MCP 加上组件元数据（widget metadata）。AGENTS.md（目前已应用于 60,000+ 个代码仓库）位于仓库根目录，作为项目级别的智能体上下文。本课将阐明各部分涵盖的内容，并构建一个可在不同智能体间通用的最小化 SKILL.md + AGENTS.md 组合包。

**Type:** 学习（Learn）
**Languages:** Python（标准库 stdlib、SKILL.md 解析器与加载器）
**Prerequisites:** 第 13 阶段 · 07（MCP 服务器）
**Time:** 约 45 分钟

## 学习目标

- 区分三个层级：AGENTS.md（项目上下文）、SKILL.md（可复用的专业知识）、MCP（工具）。
- 编写包含 YAML 前置元数据（YAML frontmatter）和渐进式披露机制的 SKILL.md。
- 以文件系统风格将技能加载到智能体运行时（agent runtime）中。
- 将技能与 MCP 服务器及 AGENTS.md 组合，使单个软件包能在 Claude Code、Cursor 和 Codex 中通用。

## 问题背景

一位工程师将编写发布说明的工作流提炼为一个多步骤提示词（prompt）：“阅读最新合并的拉取请求（Pull Request, PR）。按模块分组。逐一总结。遵循团队风格编写变更日志条目。发布至 Slack 草稿。”他们将其放入 Notion 文档供团队使用。

现在，他们希望从 Claude Code、Cursor 和 Codex CLI 中使用该工作流。每个智能体加载指令的方式各不相同：Claude Code 使用斜杠命令（slash-commands），Cursor 使用规则（rules），Codex 使用 `.codex.md`。工程师不得不将工作流复制三份并维护三个版本。

AGENTS.md 与 SKILL.md 的结合解决了这一问题：

- **AGENTS.md** 位于仓库根目录。所有兼容的智能体在会话启动时都会读取它。“本项目如何运作？有哪些规范？哪些命令用于运行测试？”
- **SKILL.md** 是一个可移植的打包文件：包含 YAML 前置元数据（YAML frontmatter，如名称、描述）+ Markdown 正文 + 可选资源。支持技能的智能体会按需按名称加载它们。
- **MCP**（第 13 阶段 · 06-14）负责处理技能需要调用的工具。

三个层级，一个可移植的交付物。

## 核心概念

### AGENTS.md (agents.md)

于 2025 年底推出，截至 2026 年 4 月已被 60,000+ 个代码仓库采用。位于仓库根目录的单一文件。格式如下：

# Project: my-service

## Conventions

- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run


- 使用 `pnpm dev` 启动本地开发服务器。
- 使用 `pnpm build` 构建生产环境包。

智能体（Agents）在会话启动时读取此文件，并据此校准其在当前项目中的行为。2026 年的所有编程智能体（Coding Agents）均支持 `AGENTS.md`：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf 和 Zed。

### SKILL.md 格式

Anthropic 的智能体技能（Agent Skills，于 2025 年 12 月作为开放标准发布）：

---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes


- 绝不包含没有关联 PR（Pull Request）的提交。
- 在公开变更日志（changelog）中跳过“chore”（日常维护）类条目。

前置元数据（Frontmatter）用于声明技能（skill）的身份标识。正文（body）则是技能加载时展示给模型（model）的提示词（prompt）。

### 渐进式披露（Progressive disclosure）
技能可以引用子资源（sub-resources），智能体（agent）仅在需要时才会获取这些资源。示例：

skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh

`SKILL.md` 中注明“请参阅 `style-guide.md` 获取样式规范”。智能体仅在技能处于活跃运行状态时才会拉取 `style-guide.md`。这避免了在提示词中堆砌模型可能不需要的细节，从而防止提示词过度膨胀。

### 文件系统发现（Filesystem discovery）
智能体运行时（Agent runtimes）会扫描以下已知目录以查找 `SKILL.md` 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目目录 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

加载机制基于文件夹名称以及 frontmatter 中的 `name` 字段。Claude Code、Anthropic Claude Agent SDK 以及 SkillKit（跨智能体）均遵循此模式。

### Anthropic Claude Agent SDK
`@anthropic-ai/claude-agent-sdk`（TypeScript）与 `claude-agent-sdk`（Python）会在会话（session）开始时加载技能，并在运行时内部将其暴露为可调用的“智能体”。当用户调用某项技能时，智能体循环（agent loop）会将其分派（dispatch）至对应技能执行。

### OpenAI Apps SDK
于 2025 年 10 月发布；直接构建于模型上下文协议（MCP）之上。它将 OpenAI 此前的连接器（Connectors）与自定义 GPT 操作（Custom GPT Actions）统一至单一开发者界面。一个 Apps SDK 应用包含：

- 一个 MCP 服务器（提供工具、资源、提示词）。
- 附加用于 ChatGPT 用户界面的组件元数据（widget metadata）。
- 附加可选的 MCP Apps `ui://` 资源，用于构建交互式界面。

协议一致，交互体验（UX）更丰富。

### 基于 SkillKit 的跨智能体可移植性（Cross-agent portability）
SkillKit 等工具及类似的跨智能体分发层（cross-agent distribution layers）能够将单个 `SKILL.md` 转换为 32 多种 AI 智能体的原生格式（如 Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）。单一事实来源（source of truth），多方消费调用。

### 三层架构（The three-layer stack）

| 层级 | 文件 | 加载时机 | 用途 |
|-------|------|-------------|---------|
| AGENTS.md | 仓库根目录 | 会话启动时 | 项目级规范 |
| SKILL.md | 技能目录 | 技能被调用时 | 可复用工作流 |
| MCP 服务器 | 外部进程 | 需要工具时 | 可调用操作 |

三者协同工作：智能体在会话启动时读取 `AGENTS.md`，用户调用某项技能，该技能的指令中包含 MCP 工具调用，智能体随后通过 MCP 客户端进行分派。

## 使用指南（Use It）

`code/main.py` 内置了一个基于标准库（stdlib）的 `SKILL.md` 解析器与加载器。它会扫描 `./skills/` 目录下的技能，解析 YAML frontmatter 与 Markdown 正文，并生成一个以技能名称为键的字典（dict）。随后，它会模拟一个智能体循环，按名称调用 `release-notes-writer`。

重点关注：

- 使用极简的标准库解析器处理 YAML frontmatter（无需依赖 `pyyaml`）。
- 技能正文原样存储；智能体在调用时会将其前置拼接到系统提示词（system prompt）中。
- 通过 `read_subresource` 函数演示渐进式披露，该函数按需拉取被引用的文件。

## 发布上线（Ship It）

本课时将生成 `outputs/skill-agent-bundle.md` 文件。在给定工作流（Workflow）的情况下，该技能（Skill）会生成包含 SKILL.md、AGENTS.md 和 MCP-server-blueprint 的组合捆绑包，该捆绑包支持跨智能体（Agent）移植。

## 练习

1. 运行 `code/main.py`。在 `skills/` 目录下添加第二个技能（Skill），并确认加载器（Loader）能够成功识别并加载它。

2. 为本课程代码仓库编写一份 AGENTS.md 文件。内容需包含测试命令、代码风格规范以及第 13 阶段的心智模型（Mental Model）。

3. 将团队内部文档中的一个多步骤工作流（Workflow）移植到 SKILL.md 中。验证其能否在 Claude Code 中正常加载。

4. 手动将该技能转换为 Cursor 和 Codex 的原生规则格式。统计各格式之间的差异——这正是 SkillKit 自动化工具所覆盖的格式转换工作量。

5. 阅读 Anthropic 关于 Agent Skills 的博客文章。找出 Claude Agent SDK 中本课时加载器未涵盖的一项功能。（提示：智能体子调用（Agent Sub-invocation）。）

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| SKILL.md | “技能文件” | YAML 前置元数据（YAML frontmatter）与 Markdown 正文的组合，由智能体运行时（Agent runtime）加载 |
| AGENTS.md | “仓库根目录智能体上下文” | 项目级规范文件，在会话启动时读取 |
| 渐进式披露（Progressive disclosure） | “延迟加载子资源” | 技能正文中引用的文件，仅在需要时才会被拉取 |
| 前置元数据（Frontmatter） | “顶部的 YAML 代码块” | 位于 `---` 分隔符内的元数据（如名称、描述） |
| Claude Agent SDK | “Anthropic 的技能运行时” | `@anthropic-ai/claude-agent-sdk`，负责加载技能并处理路由 |
| OpenAI Apps SDK | “MCP + 组件元数据” | OpenAI 基于模型上下文协议（MCP）构建的开发平台，并集成 ChatGPT UI 钩子 |
| 技能发现（Skill discovery） | “文件系统扫描” | 遍历已知目录以查找 SKILL.md，并以名称作为索引键 |
| 跨智能体可移植性（Cross-agent portability） | “一个技能适配多个智能体” | 借助 SkillKit 类工具，将单个 SKILL.md 转换为 32 种以上智能体的格式 |
| 智能体技能（Agent Skill） | “可移植的专业知识” | 独立于 MCP 工具概念的可复用任务模板 |
| Apps SDK | “MCP 加 ChatGPT UI” | 将连接器与自定义 GPT 统一集成在 MCP 之上 |

## 扩展阅读

- [Anthropic — Agent Skills 公告](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — Agent Skills 文档](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 面向 ChatGPT 的基于 MCP 的开发者平台
- [agents.md](https://agents.md/) — AGENTS.md 格式及采用情况列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方技能示例