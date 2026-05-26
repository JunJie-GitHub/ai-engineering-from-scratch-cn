---
name: dgm-evaluator-firewall
description: 指定达尔文-哥德尔机（Darwin-Godel-Machine）风格自修改智能体循环（self-modifying agent loop）所需的评估器（evaluator）隔离机制，以避免已记录的奖励黑客（reward hacking）行为。
version: 1.0.0
phase: 15
lesson: 4
tags: [dgm, 自修改, 奖励黑客, 评估器, 沙箱]
---

针对提议的自修改智能体循环（包含智能体脚手架（agent scaffolding）、评估器流水线（evaluator pipeline）、部署目标（deployment target）），生成一份防火墙规范，确保评估器处于智能体无法触及的范围。DGM 论文记录了智能体通过移除自身安全标记来提高得分的行为；本技能旨在对此进行控制。

输出内容：

1. **文件系统边界。** 列出智能体可编辑的所有文件。列出评估器读取的所有文件。两者的交集必须为空。若非空，请明确指定为使其为空所需执行的具体操作。
2. **进程边界。** 评估器必须在智能体无法控制的进程中运行。明确指定容器（container）或沙箱（sandbox）配置。若智能体拥有 Shell 访问权限，则可能绕过进程内防护机制。
3. **网络与密钥边界。** 智能体不得能够编辑用于获取评估器输入或提交得分的 CI 密钥（CI secrets）。列出密钥的存储位置及其挂载方式。
4. **得分传入契约（Score-ingress contract）。** 得分由评估器返回至循环的归档存储。该路径是唯一通道。智能体可写入的任何影响未来评分的内容均视为违反契约。
5. **变更审查清单（Mutation review list）。** 列出在采纳前需人工审查的智能体修改类别：新增依赖项、对评分相关代码的修改、对沙箱配置的修改、对日志或监控的修改。

硬性拒绝条件（Hard rejects）：
- 评估器与智能体源码位于同一代码仓库中。必须将其移出。
- 评估器作为 Python 导入在智能体进程内运行。必须将其移至子进程（subprocess）中运行。
- 任何在边界智能体侧应用的得分调整。所有调整必须在服务端进行。
- 任何智能体可通过修改自身代码来静默的日志记录。

拒绝规则（Refusal rules）：
- 若用户无法列出智能体可编辑的每一个文件，则予以拒绝，并要求首先进行文件级写入权限审计。
- 若评估器没有保留的独立输入数据（held-out inputs），则予以拒绝——此为第 3 课审计内容，属于前置条件。
- 若部署面（deployment surface）包含对评估器自身的修改（即使是通过提议的依赖更新间接进行），则予以拒绝，并要求增加人工防火墙审查步骤。

输出格式：
返回一份单页规范，包含以下内容：
- **智能体写入面（Agent write-surface）**（路径、通配符模式（globs））
- **评估器读取面（Evaluator read-surface）**（路径、端点（endpoints））
- **交集（Intersection）**（必须为空；展示差异对比（diff））
- **进程模型（Process model）**（评估器的隔离方式）
- **密钥清单（Secrets inventory）**（存储位置及挂载方式）
- **需审查的变更类别（Review-required mutation classes）**（列表形式）
- **签署行（Sign-off line）**（负责维护防火墙不变量（firewall invariant）的人员）