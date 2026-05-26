---
name: eval-suite
description: 构建包含评估器-优化器循环（evaluator-optimizer loop）和持续集成门禁（CI gates）的三层评估套件（静态基准测试、自定义离线、在线生产环境）。
version: 1.0.0
phase: 14
lesson: 30
tags: [评估, 持续集成, 回归, 基准测试, 大语言模型裁判]
---

针对智能体（Agent）产品，构建一套接入持续集成（CI）流水线的三层评估套件（eval suite）。

产出物：

1. **静态基准测试层（Static benchmark layer）** — 至少包含一项相关基准测试（benchmark）（代码类使用 SWE-bench Verified，工具调用类使用 BFCL V4，网页交互类使用 WebArena，桌面操作类使用 OSWorld，通用类使用 GAIA）。必须同时报告经过人工抽检审计的得分（+-audited score）。
2. **自定义离线层（Custom offline layer）** — 至少包含一项基于大语言模型裁判（LLM-judge）的评分量规（rubric），针对领域特定维度（事实准确性、语气、范围、拒绝质量）进行打分。至少包含一项基于执行的测试用例（execution-based case），用于探测智能体运行后的实际状态。至少包含一项基于轨迹的测试用例（trajectory-based case），并附带标准路径（gold path）。
3. **在线评估层（Online eval layer）** — 会话回放（session replays）、由护栏（guardrail）触发的告警，以及通过 OTel GenAI spans（第 23 课）实现的单步成本/延迟追踪。
4. **评估器-优化器运行器（Evaluator-optimizer runner）** — 将智能体封装在“提议/评判/优化”（propose / judge / refine）循环中，并设置最大轮次上限。
5. **持续集成门禁（CI gate）** — 当性能较基线（baseline）出现 >=5% 的回归（regression）时，构建失败。需随时间推移持续追踪基线表现。
6. **用例映射（Case mapping）** — 第 14 阶段课程中的每一项护栏和每一条习得规则（learned rule）都至少对应一个测试用例。

硬性拒绝条件：

- 评估套件缺少基线。没有参照物就无法检测性能回归。
- 在事实性任务中使用大语言模型裁判却缺乏外部事实锚定（external grounding）。必须采用 CRITIC 模式（CRITIC pattern）（第 05 课）。
- 存在不稳定的测试用例（flaky cases），且未固定随机种子（pinned seeds）或快照状态（snapshot state）。误报会严重削弱团队对评估体系的信任。

拒绝规则：

- 如果用户要求“仅覆盖正常路径（happy path）”，请拒绝。每一种失败模式（failure mode）（第 26 课）都应配备对应的测试用例。
- 如果用户要求“不设置持续集成门禁”，对于已面向付费用户的产品请拒绝。否则评估漂移（eval drift）将无法被察觉。
- 如果用户要求“全部使用大语言模型裁判”，在事实性与合规性任务上请拒绝。此类任务必须采用基于执行或程序化的评估器（programmatic evaluators）。

输出：`cases/benchmarks/`、`cases/custom/`、`cases/online/`、`runner.py`、`ci_gate.py`、`README.md`（需详细说明评分量规、基线数据以及第 14 阶段映射表）。结尾需附上“下一步阅读建议”，指向第 24 课（可观测性）、第 26 课（失败模式）或第 23 课（OTel）以获取底层支撑知识。