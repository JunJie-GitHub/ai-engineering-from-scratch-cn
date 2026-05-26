# 多模态智能体（Multimodal Agent）与计算机操控（Computer-Use）（综合项目 Capstone）

> 2026 年的前沿产品是一种多模态智能体（Multimodal Agent），它能够读取屏幕截图、点击按钮、导航网页用户界面（Web UI）、填写表单，并端到端地完成工作流。SeeClick 和 CogAgent（2024）验证了图形用户界面定位（GUI-grounding）这一基础原语。Ferret-UI 将其扩展至移动端。ChartAgent 引入了针对图表的视觉工具使用（Visual Tool-Use）能力。VisualWebArena 和 AgentVista（2026）是前沿技术竞相追逐的基准测试（Benchmark）——即便是 Gemini 3 Pro 和 Claude Opus 4.7，在 AgentVista 的高难度任务上也仅能获得约 30% 的得分。本综合项目（Capstone）将第 12 阶段的所有核心线索串联起来：感知（高分辨率视觉语言模型 Vision-Language Model, VLM）、推理（具备工具使用能力的大语言模型 Large Language Model, LLM）、定位（坐标输出）、长程记忆（Long-horizon Memory）以及评估（Evaluation）。

**类型：** 综合项目（Capstone）
**编程语言：** Python（标准库，动作模式 Action Schema + 智能体循环骨架 Agent Loop Skeleton）
**前置要求：** 第 12 阶段 · 05 (LLaVA)、第 12 阶段 · 09 (Qwen-VL JSON)、第 14 阶段（智能体工程 Agent Engineering）
**预计耗时：** 约 240 分钟

## 学习目标

- 设计多模态智能体循环（Multimodal Agent Loop）：感知 → 推理 → 执行 → 观察 → 重复。
- 构建图形用户界面定位（GUI-grounding）输出模式（Output Schema）（包含点击坐标、输入文本、滚动、拖拽），使视觉语言模型（VLM）能够以 JSON 格式输出。
- 对比纯截图智能体、基于无障碍树（Accessibility Tree）的智能体与混合架构智能体。
- 在 VisualWebArena 的子集上搭建多模态智能体基准测试（Benchmark）评估流程。

## 问题背景

一个机票预订网站的工作流：“帮我找一张 4 月 15 日飞往东京的机票，靠过道座位，价格低于 800 美元，并完成预订。”

多模态智能体需要执行以下步骤：

1. 截取浏览器屏幕截图。
2. 将屏幕截图、URL 与任务目标解析为执行计划。
3. 输出结构化动作：点击（坐标 x,y）、输入“Tokyo”（在元素 E 处）、向下滚动、选择（单选按钮）。
4. 将动作应用于浏览器。
5. 观察新状态（下一张屏幕截图）。
6. 重复上述过程，直至任务完成。

每一步都需要调用多模态视觉语言模型（VLM）。VLM 的输出必须是可解析的 JSON 格式。由于错误会在各步骤间累积放大，因此错误恢复机制至关重要。

## 核心概念

### GUI 定位（GUI grounding）—— 基础原语

GUI 定位（GUI grounding）是指：给定一张截图和一条自然语言指令，输出需要点击（或执行其他操作）的 (x, y) 坐标。

SeeClick (arXiv:2401.10935) 是首个大规模开源成果：在合成与真实 GUI 数据上微调视觉语言模型（Vision-Language Model, VLM），将坐标作为纯文本 Token 输出。效果可行。

CogAgent (arXiv:2312.08914) 引入了 1120x1120 的高分辨率编码以应对密集的 UI 界面。得分：在网页导航任务上达到约 84%。

Ferret-UI (arXiv:2404.05719) 专注于移动端 UI，并集成了 iOS 无障碍（Accessibility）数据。

输出格式通常为 JSON：

{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}

`element_desc` 字段有助于错误恢复：如果截图间的坐标发生偏移，该语义提示能让系统重新进行定位（re-ground）。

### 动作模式（Action schemas）

典型的动作模式包含 6 到 10 种动作类型：

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

智能体（Agent）每一步输出一个动作。浏览器封装层（Browser wrapper）负责执行该动作并返回新的状态。

### 仅截图模式 vs 无障碍树模式

输入模式：

- 仅截图（Screenshot-only）：完整图像，无结构信息。通用性最强；适用于任何应用。
- 无障碍树（Accessibility tree）：结构化的 DOM / iOS 无障碍信息。定位更可靠；仅在可获取该树结构时有效。
- 混合模式（Hybrid）：结合两者，以树结构作为原子动作的可靠定位依据，以截图提供语义上下文。

生产环境中的智能体在条件允许时均采用混合模式。浏览器自动化（Selenium + 无障碍接口）始终具备树结构；桌面应用则有时具备。

### 长程记忆（Long-horizon memory）

一个 20 步的工作流会生成 20 张截图。VLM 的上下文窗口会迅速被填满。三种压缩策略如下：

- 摘要链（Summary-chain）：每执行 5 步后，对已发生的内容进行总结，并丢弃旧截图。
- 跳帧（Skip-frame）：仅保留第一张、最后一张以及每隔 3 张的截图。
- 工具记录日志（Tool-recorded log）：执行动作并保留文本操作日志；不再回溯查看旧截图。

Claude 的 computer-use API 采用了日志模式。该方式更简单、更可靠。

### 视觉工具调用（Visual tool use）

ChartAgent (arXiv:2510.04514) 引入了用于图表理解的视觉工具调用：裁剪、缩放、OCR（光学字符识别）以及调用外部检测模型。智能体可以输出类似“裁剪至区域 (100, 200, 300, 400) 然后调用 OCR”的指令作为工具调用。工具返回文本结果后，VLM 继续进行推理。

该模式具有通用性：标记集提示（Set-of-mark prompting）、区域标注以及外部检测工具均符合同一“输出工具调用指令，接收结构化响应”的模式。

### 2026 年基准测试（Benchmarks）

- ScreenSpot-Pro：在约 1000 张网页截图上进行 GUI 定位。开源 SOTA（State-of-the-Art）模型 Qwen2.5-VL-72B 得分约 85%。前沿（Frontier）模型约 90%。
- VisualWebArena：端到端网页任务（购物、论坛、分类信息）。开源 SOTA 约 20%。Gemini 3 Pro 约 27%。
- AgentVista (arXiv:2602.23166)：2026 年最具挑战性的基准测试。涵盖 12 个领域的真实工作流。前沿模型得分 27-40%；开源模型 10-20%。
- WebArena / WebShop：较早期的基准测试；性能已被前沿模型逼近饱和。

### 为何依然困难

智能体性能瓶颈：

1. 细粒度视觉定位。在移动端分辨率下，“点击小 X 号”等指令经常失败。
2. 长程规划。执行 10 个动作后，智能体容易偏离初始目标。
3. 错误恢复。当点击失败（点错按钮）时，检测错误并恢复的操作极少出现在训练数据中。
4. 跨页面上下文。在标签页或长表单之间跳转时容易丢失状态。

研究方向：记忆架构、显式重规划、多模态验证（通过截图比对确认动作是否成功）。

### 综合实战项目（Capstone build-it）

综合实战任务：构建一个计算机操作智能体，要求能够：

1. 读取预订网站模拟页面的 HTML 代码与截图。
2. 规划多步操作序列：搜索 → 选择 → 填写表单 → 提交。
3. 输出符合动作模式的 JSON 动作指令。
4. 在固定的 10 个任务子集上进行评估。

本课程提供了脚手架代码，可轻松扩展至真实浏览器环境中运行。

## 实践应用

`code/main.py` 是核心项目脚手架：

- 动作模式（Action Schema）的 JSON 定义（包含 10 个动作）。
- 以字典（dict）形式模拟浏览器状态。
- 智能体循环（Agent Loop）骨架：接收状态、输出动作、执行应用、循环迭代。
- 包含 10 个任务的微型基准测试（使用合成页面），用于衡量端到端成功率。
- 错误恢复钩子（Error-recovery Hook），用于处理动作执行失败的情况。

## 交付成果

本课时将生成 `outputs/skill-multimodal-agent-designer.md` 文件。给定一个计算机操作（Computer-Use）产品（包含领域、动作集、评估目标），该文件将设计完整的智能体循环、记忆策略、定位模式（Grounding Mode）以及预期的基准测试得分。

## 练习

1. 在动作模式中添加 `screenshot_region` 工具（裁剪 + 缩放）。哪些任务会从中受益？

2. 阅读 AgentVista (arXiv:2602.23166)。描述最具挑战性的任务类别，并解释为何前沿模型（Frontier Models）仍然会失败。

3. 长程记忆压缩（Long-horizon Memory Compression）：设计一个摘要链（Summary-chain），保持最多 4 张截图处于活跃状态，其余任意数量的截图仅做存档记录。

4. 构建错误恢复钩子：当动作执行失败（例如未找到按钮）时，智能体下一步该做什么？

5. 在 10 个网页任务上，对比仅使用截图的 Claude 4.7 与采用“截图 + 无障碍树（Accessibility Tree）”混合模式的 Qwen2.5-VL。它们在哪些任务上各自胜出？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| GUI 定位（GUI Grounding） | “点击坐标” | 模型在截图上输出指令目标对应的 (x,y) 坐标 |
| 动作模式（Action Schema） | “工具定义” | 有效动作（点击、输入、滚动、拖拽）的 JSON 描述 |
| 无障碍树（Accessibility Tree） | “结构化 DOM” | 来自浏览器/iOS API 的机器可读 UI 层级结构 |
| 混合智能体（Hybrid Agent） | “截图 + 树” | 同时使用图像与结构化信息；比单一模态更可靠 |
| 视觉工具调用（Visual Tool Use） | “缩放/裁剪/检测” | 智能体在规划过程中调用外部视觉工具（OCR、目标检测） |
| 摘要链（Summary-chain） | “记忆压缩” | 定期生成的文本摘要替代冗长的截图历史记录 |
| VisualWebArena | “端到端网页基准” | 2024 年发布的端到端网页任务基准测试 |
| AgentVista | “2026 高难度基准” | 涵盖 12 个领域的真实工作流；即便是 Gemini 3 Pro 得分也仅约 30% |

## 延伸阅读

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)