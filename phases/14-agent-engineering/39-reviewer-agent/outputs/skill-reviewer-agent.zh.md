---
name: 评审智能体（reviewer-agent）
description: 建立一个具备五维评分量表（five-dimension rubric）的评审智能体（reviewer agent）角色，用于读取构建智能体（builder agent）的产出物，生成结构化的评审报告，并让人类评审从已有文档开始，而非从零起步。
version: 1.0.0
phase: 14
lesson: 39
tags: [评审者, 评分量表, 角色分离, 第二循环, 评审报告]
---

在构建智能体（builder agent）已能生成工作台产出物的前提下，部署一个评审智能体来读取这些产出物并撰写结构化报告。

产出物：

1. `agents/reviewer.md`：包含评审系统提示词（system prompt），要求仅具备只读权限、采用五维评分量表，且每个评分必须引用对应的产出物路径。
2. `tools/reviewer.py`：从工作台加载 `ReviewerInputs`，并针对每个维度运行大语言模型（LLM）评分器。
3. `outputs/review/<task_id>.json`：作为标准评审报告的存储路径。
4. `docs/reviewer-rubric.md`：列出五个维度、每个维度所回答的问题，以及 0-1-2 分的锚点描述（anchor descriptions）。
5. 持续集成（CI）步骤：每当构建任务关闭时，自动将评审报告作为拉取请求（PR）评论发布。

硬性拒绝条件：

- 评审智能体拥有对代码差异（diff）的写入权限。构建者与评审者之间的隔离是核心信号所在，打破这种隔离将彻底破坏可靠性。
- 评分量表缺乏每个分数对应的锚点描述。没有锚点的“0到2分”评分会沦为主观臆断（vibes）。
- 评审报告缺少引用。每个评分都必须指向具体的文件或追踪记录（trace entry）。
- 共享构建智能体的系统提示词。使用相同模型可以接受，但使用相同提示词则不行。

拒绝执行规则：

- 如果构建智能体未生成验证报告（verification report），则拒绝运行评审。在请求评判之前，必须先满足验收条件。
- 如果项目已关闭的任务少于三个，则不得声称评分量表已完成校准。应将首批报告保存为校准集（calibration set）。
- 如果要求评审智能体在低于最低置信度（confidence）的情况下进行评分，则应拒绝执行，并将不确定的维度交由人类处理。

输出结构：

<repo>/
├── agents/reviewer.md
├── tools/reviewer.py
├── outputs/review/
│   └── <task_id>.json
├── docs/reviewer-rubric.md
└── .github/workflows/review.yml

结尾附上“下一步阅读”指引，指向：

- 第 40 课：了解结合验证与评审的交接包（handoff packet）。
- 第 41 课：通过实战风格任务，端到端演练构建/评审分离流程。
- 第 05 课（自我优化与 CRITIC）：了解本课所改进的单智能体自我评审基线（baseline）。