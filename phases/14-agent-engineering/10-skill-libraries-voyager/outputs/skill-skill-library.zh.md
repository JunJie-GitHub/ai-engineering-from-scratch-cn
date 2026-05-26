---
name: skill-library
description: 生成一个符合 Voyager 架构的技能库，支持注册、基于相似度的检索、组合式执行以及由失败驱动的优化。
version: 1.0.0
phase: 14
lesson: 10
tags: [voyager, 技能, 库, 组合, 优化]
---

给定目标运行时 (runtime) 和特定领域，生成一个支持 Voyager 三大组件的技能库：课程钩子 (curriculum hook)、可检索技能存储 (retrievable skill store)、迭代优化 (iterative refinement)。

产出内容：

1. `Skill` 类型，包含 `name`、`description`、`code`、`version`、`tags`、`depends_on`、`history` 字段。每次写入均需记录先前的代码。
2. `SkillLibrary` 类，包含 `register(skill, dedup=True)`（新增或版本升级）、`search(query, top_k, tag_filter)`、`get(name)`、`topo_order(name)`（依赖解析）、`execute(name, context)`（拓扑执行）。
3. 检索 (Retrieval) 必须使用嵌入相似度 (embedding similarity) 或 BM25，禁止对整个库进行大语言模型 (LLM) 打分。允许在 top-k 候选列表上使用 LLM 重排序 (re-rank)。
4. 执行 (Execution) 必须按技能捕获异常，并将其暴露到追踪 (trace) 中，作为优化循环 (refinement loop) 可消费的反馈。
5. 优化钩子 (refinement hook)：在 `execute` 失败后，运行时收集 `(task, skill_name, error, env_state)`，将其传递给模型，并对重写后的技能调用 `register`。版本号递增；历史记录保留旧代码。

硬性拒绝条件：

- 技能库中的技能必须是可执行代码，而非纯文本描述。技能需具备可执行性，说明性文本应置于 `description` 中。
- 缺乏拓扑排序 (topological sort) 的组合逻辑。在未进行环检测 (cycle detection) 的情况下使用深度优先遍历，会在技能有向无环图 (DAG) 上导致崩溃。
- 静默覆盖版本。每次优化必须递增 `version`，并将旧代码推入 `history` 以供审计。

拒绝规则：

- 若目标运行时缺乏用于技能执行的沙箱 (sandbox)，则针对涉及生产系统的领域场景应予以拒绝。在发布前必须要求提供沙箱环境（遵循第 09 课原则）。
- 若用户要求“每次失败均自动重试且不进行优化”，则应拒绝。缺乏优化的重试只会放大缺陷，无法修复问题。
- 若技能库规模超过约 200 个技能且仅采用扁平化检索，则拒绝将其标记为“生产就绪 (production-ready)”。需优先引入标签过滤与分层命名空间。

输出文件：`skill.py`、`library.py`、`execute.py`、`refine.py`，以及一份 `README.md`，用于说明去重规则、检索后端、优化提示词 (prompt) 及版本策略。文末需附上“下一步阅读”指引，指向第 17 课（Claude Agent SDK 集成）、第 16 课（OpenAI Agents SDK 工具转换）或第 30 课（技能库质量评估）。