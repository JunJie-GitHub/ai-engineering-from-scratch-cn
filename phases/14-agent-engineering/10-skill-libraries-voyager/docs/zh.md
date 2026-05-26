# 技能库（Skill Libraries）与终身学习（Voyager）

> Voyager（Wang 等人，TMLR 2024）将可执行代码视为一种技能（Skill）。技能具有命名、可检索、可组合的特性，并能通过环境反馈进行优化。这是 Claude Agent SDK 技能、skillkit 以及 2026 年技能库模式的参考架构。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 07（MemGPT），第 14 阶段 · 08（Letta Blocks）
**耗时：** 约 75 分钟

## 学习目标

- 指出 Voyager 的三个组成部分——自动课程（Automatic Curriculum）、技能库和迭代提示（Iterative Prompting）——并说明各自的作用。
- 解释为何 Voyager 将动作空间（Action Space）设定为代码，而非原始指令。
- 使用标准库实现一个技能库，涵盖注册、检索、组合以及由失败驱动的优化机制。
- 将 Voyager 的模式映射到 2026 年的 Claude Agent SDK 技能体系与 skillkit 生态系统中。

## 问题背景

在每次会话中都从零开始重建所有能力的智能体（Agent）会犯三个错误：

1. **浪费 Token。** 每个任务都会重复触发相同的推理过程。
2. **丢失进度。** 在会话 A 中学到的修正无法迁移到会话 B。
3. **长程组合失败。** 复杂任务需要能力层级结构；单次提示（One-Shot Prompt）无法表达这些结构。

Voyager 的解决方案是：将每个可复用的能力视为库中一段已命名的代码块，通过相似度进行检索，与其他技能组合，并根据执行反馈进行优化。

## 核心概念

### 三大核心组件

Voyager（arXiv:2305.16291）围绕以下三个部分构建智能体（Agent）：

1. **自动课程生成（Automatic Curriculum）**。由好奇心驱动的提议模块会根据智能体当前的技能集与环境状态来规划下一项任务。探索过程采用自下而上的策略。
2. **技能库（Skill Library）**。每项技能均为可执行代码。当任务成功完成后，新技能会被存入库中。系统通过计算查询语句与技能描述之间的相似度来检索技能。
3. **迭代提示机制（Iterative Prompting Mechanism）**。若执行失败，智能体会接收执行报错、环境反馈及自我验证结果，并据此对技能进行迭代优化。

在《我的世界》（Minecraft）评估中（Wang et al., 2024）：相比基线模型（Baselines），获取的独特物品数量提升 3.3 倍，制作石制工具速度加快 8.5 倍，铁制工具加快 6.4 倍，地图探索距离延长 2.3 倍。这些数据虽针对特定游戏场景，但其底层架构模式具有通用迁移性。

### 动作空间即代码（Action Space = Code）

大多数智能体输出的是基础指令，而 Voyager 输出的是 JavaScript 函数。一项技能的结构如下：

async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}

该技能由多个子技能组合而成。系统以描述文本和嵌入向量（Embedding）作为键进行存储。检索时直接返回程序代码，而非提示词（Prompt）。

这正是 2026 版 Claude Agent SDK 中“技能”的核心定义：一段具名且可检索的代码片段，附带供智能体按需加载的指令说明。

### 技能检索（Skill Retrieval）

面对新任务“制作钻石镐”，智能体将执行以下步骤：

1. 对任务描述进行嵌入（Embedding）处理。
2. 在技能库中查询相似度最高的 Top-K 个技能。
3. 检索出 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等技能。
4. 结合检索到的基础技能与新逻辑，组合生成新技能。

这正是 MCP 资源（第 13 阶段）与 Agent SDK 技能所实现的架构模式：在知识或代码库表层进行检索，并将范围严格限定于当前任务上下文。

### 迭代优化（Iterative Refinement）

Voyager 的反馈循环流程如下：

1. 智能体编写一项技能。
2. 该技能在环境中执行。
3. 返回三种信号之一：`success`（成功）、`error`（附带堆栈跟踪的报错）或 `self-verification failure`（自我验证失败）。
4. 智能体将该信号作为上下文，重写技能代码。
5. 循环执行，直至成功或达到最大迭代轮数。

该机制是将 Self-Refine（第 05 课）应用于代码生成任务，并结合基于环境的验证（Environment-Grounded Verification）。CRITIC（第 05 课）遵循相同模式，区别在于其使用外部工具充当验证器。

### 课程规划与探索（Curriculum and Exploration）

Voyager 的课程模块会基于智能体当前持有的资源与尚未尝试的行为，动态生成诸如“在湖畔搭建庇护所”等任务。提议模块综合环境状态与技能库存，精准挑选难度略高于当前能力的任务，从而锁定探索的“最佳区间”（Sweet Spot）。

在面向生产环境的智能体中，该逻辑转化为一个“缺失项分析”算子：基于现有技能库与目标领域，识别尚未覆盖的技能缺口。在实际工程落地中，团队通常以人工课程评审（Curriculum Review）的方式来实现这一环节。

### 该模式的常见陷阱

- **技能库腐化（Skill Library Rot）**。同一技能因描述文本存在细微差异而被重复录入多次。应在写入阶段引入去重逻辑，确保检索时仅返回单一实例。
- **组合技能漂移（Composed-Skill Drift）**。父技能所依赖的子技能在后续迭代中被优化。必须对技能实施版本控制；锁定在 v1 版本的父技能不会自动同步 v3 版本的更新。
- **检索质量衰减**。当技能库规模突破数百项后，基于技能描述的向量检索（Vector Retrieval）准确率会显著下降。需引入标签过滤与硬性约束条件（如“仅匹配 `category=tooling` 的技能”）进行补充。

## 构建它

`code/main.py` 实现了一个标准库（Standard Library）技能库（Skill Library）：

- `Skill`（技能）—— 包含名称、描述、代码（字符串形式）、版本、标签和依赖项。
- `SkillLibrary`（技能库）—— 支持注册、搜索（基于词元重叠/Token Overlap）、组合（依赖项的拓扑排序/Topological Sort）以及优化（更新时触发版本升级/Version Bump）。
- 一个脚本化智能体（Scripted Agent），用于注册三个基础技能，组合出第四个技能，触发一次失败，并进行迭代优化。

运行方式：

python3 code/main.py

执行轨迹（Trace）展示了库写入、检索、组合、执行失败以及 v2 版本的优化过程——完整呈现了 Voyager 的端到端循环。

## 使用它

- **Claude Agent SDK skills**（Anthropic）—— 2026 年的参考实现：每个技能包含描述、代码和指令；在智能体会话期间按需加载。
- **skillkit**（npm: skillkit）—— 面向 32 款以上 AI 编程智能体的跨智能体技能管理工具。
- **自定义技能库（Custom Skill Libraries）**—— 面向特定领域（例如为数据智能体提供 SQL 技能，为基础设施智能体提供 Terraform 技能）。Voyager 模式可向下适配。
- **OpenAI Agents SDK `tools`**—— 轻量级方案；每个工具（tool）本质上就是一个轻量级技能。

## 交付它

`outputs/skill-skill-library.md` 会生成一个符合 Voyager 架构的技能库，其中已内置注册、检索、版本管理和优化机制，可适配任意目标运行时环境。

## 练习

1. 在 `compose()` 中添加依赖环检测器。当技能 A 依赖技能 B，而技能 B 又依赖技能 A 时会发生什么？应抛出错误还是仅发出警告？
2. 实现按技能版本锁定（Version Pinning）。当父技能组合子技能 `crafting@1` 时，若 `crafting` 被优化至 `@2` 版本，父技能不应被静默升级。
3. 将基于词元重叠的检索替换为 `sentence-transformers` 嵌入向量（Embeddings）（或标准库实现的 BM25 算法）。在一个包含 50 个技能的示例库上测试检索@5（Retrieval@5）指标。
4. 添加一个“课程智能体”（Curriculum Agent）：根据当前技能库和领域描述，推荐 5 个缺失的技能。设定为每周调用一次。
5. 阅读 Anthropic 的 Claude Agent SDK 技能文档。将该示例库移植到 SDK 的技能模式（Schema）中。技能的可发现性（Discoverability）会发生什么变化？

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 技能（Skill） | “可复用的能力” | 带有名称的代码块 + 描述，可通过相似度检索 |
| 技能库（Skill Library） | “智能体的操作记忆” | 技能的持久化存储，支持搜索与组合 |
| 课程学习（Curriculum） | “任务提议者” | 由当前能力缺口驱动的自下而上目标生成器 |
| 组合（Composition） | “技能有向无环图（DAG）” | 技能相互调用；执行时按拓扑顺序排序 |
| 迭代优化（Iterative Refinement） | “自校正循环” | 环境反馈 + 错误信息 + 自我验证结果反馈至下一版本 |
| 代码化动作空间（Action-space-as-code） | “程序化动作” | 输出函数而非基础命令，以实现时序扩展行为（Temporally Extended Behavior） |
| 写入时去重（Dedup on Write） | “技能合并” | 将高度相似的描述合并为单一的标准技能 |

## 延伸阅读

- [Wang 等人，Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — 技能库（skill library）的原始论文
- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview) — 技能（skills）作为 2026 年的产品化（productization）方向
- [Anthropic，使用 Claude Agent SDK 构建智能体（agents）](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 技能与子智能体（subagents）的实践应用
- [Madaan 等人，Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — Voyager 底层的精炼循环（refinement loop）