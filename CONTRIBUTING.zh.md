# 贡献指南

课程 (Lesson)、翻译、修复、产出物 (Output)——我们全部欢迎。每个 Pull Request (PR) 仅包含一项贡献，这能加快审查速度，并确保贡献者统计和署名准确无误。

## 重要提示：README 和 ROADMAP 用于生成网站内容

`site/build.js` 会解析 `README.md`、`ROADMAP.md` 和 `glossary/terms.md` 以生成 `site/data.js`。任何修改这些文件的 Pull Request 都必须保留以下两种格式：

- 阶段 (Phase) 标题必须采用 `### Phase N: Name \`X lessons\`` 或 `<details><summary><b>Phase N — Name</b> ... <code>X lessons</code> ... <em>Description</em></summary>` 格式。
- 课程表格的列结构必须为 `| # | Lesson | Type | Lang |`（综合项目 (Capstone) 表格则为 `| # | Project | Combines | Lang |`）。`Lang` 列接受纯文本（如 `Python, TypeScript`）或传统的 Emoji 旗帜（`🐍 🟦 🦀 🟣 ⚛️`）；两者在解析器 (Parser) 中等效。
- 阶段标题和课程行中的 ROADMAP 状态符号（`✅`、`🚧`、`⬚`）。请勿将其替换为文本——解析器依赖这些精确字符进行匹配。

编辑这些文件后，请运行 `node site/build.js`；如果你的修改未破坏结构，`git diff site/data.js` 应仅显示时间戳的变更。

## 贡献方式

### 1. 添加新课程

每个课程位于 `phases/XX-phase-name/NN-lesson-name/` 目录下，结构如下：

NN-lesson-name/
├── code/           At least one runnable implementation
├── notebook/       Jupyter notebook for experimentation (optional)
├── docs/
│   └── en.md       Lesson documentation (required)
└── outputs/        Prompts, skills, or agents this lesson produces (if applicable)

**课程文档格式** (`en.md`)：

# Lesson Title

> One-line motto — the core idea in one sentence.

## The Problem

Why does this matter? What can't you do without this?

## The Concept

Explain with diagrams, visuals, and intuition. Code comes later.

## Build It

Step-by-step implementation from scratch.

## Use It

Now use a real framework or library to do the same thing.

## Ship It

The prompt, skill, agent, or tool this lesson produces.

## Exercises

1. Exercise one
2. Exercise two
3. Challenge exercise

### 2. 添加翻译

在任意课程的 `docs/` 文件夹中创建新文件：

docs/
├── en.md    (English — always required)
├── zh.md    (Chinese)
├── ja.md    (Japanese)
├── es.md    (Spanish)
├── hi.md    (Hindi)
└── ...

保持与英文版本相同的结构。仅翻译内容，不翻译代码。

### 3. 添加产出物

如果某个课程需要生成可复用的提示词 (Prompt)、技能 (Skill)、智能体 (Agent) 或 MCP 服务器 (MCP Server)：

1. 将其创建在课程的 `outputs/` 文件夹中
2. 在顶层 `outputs/` 索引中添加引用

**提示词格式：**

---
name: prompt-name
description: What this prompt does
phase: 14
lesson: 01
---

[System prompt or template here]

**技能格式：**

---
name: skill-name
description: What this skill teaches
version: 1.0.0
phase: 14
lesson: 01
tags: [agents, loops]
---

[Skill content here]

### 4. 修复错误或改进现有课程

- 修复无法运行的代码
- 优化解释说明
- 补充更清晰的图表
- 更新过时信息

### 5. 添加练习或项目

我们非常欢迎更多的练习和项目，尤其是那些能够串联多个阶段的内容。

## 贡献规范

- **代码必须可运行。** 每个代码文件在列出依赖项的情况下都应能无错误执行。
- **代码中不添加注释。** 代码应具备自解释性。详细说明请放在文档中。
- **选用最适合的语言。** 如果 TypeScript 或 Rust 是更好的选择，不要强行使用 Python。
- **优先从零构建。** 在展示框架版本之前，务必先基于第一性原理 (First Principles) 实现核心概念。
- **注重实践。** 理论应为实践服务，而非本末倒置。
- **拒绝 AI 生成废话 (AI Slop)。** 像人类一样写作。直截了当。剔除冗余内容。

## Pull Request 流程

1. 派生（Fork）仓库
2. 创建功能分支（`git checkout -b add-lesson-phase3-gradient-descent`）
3. 进行修改
4. 确保所有代码均可正常运行
5. 提交附带清晰说明的拉取请求（Pull Request）

## 行为准则（Code of Conduct）

请参阅 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。请保持友善、乐于助人、秉持建设性态度。

## 写作风格

- 行文直接。剔除冗余。保持技术手册的语调，而非营销文案风格。
- 标题中不使用装饰性表情符号。语言列（Lang Column）的表情符号旗帜是唯一的例外，这仅是因为解析器（Parser）需要对其进行映射。
- 代码需能直接运行，且依赖项（Dependencies）需与课程中列出的保持一致。
- 优先从零开始构建，其次再使用框架。