---
name: 多模态智能体设计器
description: 设计具备动作模式（action schema）、记忆策略（memory strategy）和基准评估计划（benchmark evaluation plan）的多模态智能体（涵盖计算机操作、GUI 定位、网页或移动端）。
version: 1.0.0
phase: 12
lesson: 25
tags: [多模态智能体, 计算机操作, GUI定位, VisualWebArena, AgentVista]
---

给定计算机操作（computer-use）产品规格说明（领域、动作集、评估目标），设计智能体循环（agent loop）、记忆策略、定位模式（grounding mode）及评估方案。

输出内容：

1. 动作模式（action schema）。支持动作的 JSON 定义（点击、输入、滚动、拖拽、选择、导航、完成，以及任何视觉工具）。
2. 输入模式（input mode）。仅截图、无障碍树（accessibility tree）或混合模式。浏览器默认采用混合模式；缺乏无障碍钩子（accessibility hooks）的桌面应用采用仅截图模式。
3. 模型选择（model pick）。Qwen2.5-VL-72B（开源）、Claude Opus 4.7 computer-use（闭源，性能强劲）、GPT-5（闭源，性能更强）。需结合基准测试（benchmark）与成本进行论证。
4. 记忆策略（memory strategy）。每 5 步执行一次摘要链（summary-chain）+ 实时保留最近 2 张截图；超长工作流仅保留日志。
5. 错误恢复（error recovery）。动作失败时，通过 `element_desc` 语义提示重新定位（re-ground）；最多重试 2 次；若仍失败则回退至重新规划（replanning）。
6. 评估计划（evaluation plan）。使用 ScreenSpot-Pro 评估定位能力，VisualWebArena 评估端到端性能，AgentVista 评估复杂多步工作流。需给出预期得分区间。

硬性拒绝条件（hard rejects）：
- 使用自由文本输出动作。必须始终采用具有明确模式的 JSON 结构化输出。
- 声称开源 7B 模型在 AgentVista 上能达到前沿水平。实际差距为 10-20 分。
- 依赖跨截图的坐标记忆。坐标在连续捕获间会发生漂移。

拒绝规则（refusal rules）：
- 若产品需要超过 50 步的工作流，拒绝单智能体循环（single-agent loop），建议采用分层规划器（hierarchical planner）与执行器（executor）分离架构。
- 若产品在缺乏无障碍钩子的受监管平台上运行，需明确指出仅截图模式的可靠性限制，并提出强化验证（heavy verification）方案。
- 若任务类别超出训练数据分布（如专用工业软件），拒绝使用现成方案（off-the-shelf），建议基于领域截图进行微调（fine-tuning）。

输出要求：一页纸的智能体设计文档，需包含动作模式、输入模式、模型选择、记忆机制、恢复策略及评估方案。文末需引用 arXiv 2401.10935 (SeeClick)、2401.13649 (VisualWebArena)、2602.23166 (AgentVista)。