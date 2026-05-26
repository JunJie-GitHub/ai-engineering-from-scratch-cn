# 基准测试：WebArena 与 OSWorld

> WebArena 在四个自托管应用（self-hosted apps）上测试网页智能体（web-agent）的能力。OSWorld 在 Ubuntu、Windows 和 macOS 上测试桌面智能体（desktop-agent）的能力。在发布时（2023–2024 年），两者均显示出顶尖智能体与人类之间存在巨大差距。尽管该差距正在缩小，但失败模式（failure modes）并未改变。

**类型：** 学习
**语言：** Python (stdlib)
**前置要求：** Phase 14 · 19（SWE-bench、GAIA）
**耗时：** 约 60 分钟

## 学习目标

- 描述 WebArena 的四个自托管应用，并解释为何基于执行的评估（execution-based evaluation）至关重要。
- 解释为何 OSWorld 使用真实的操作系统截图，而非辅助功能 API（accessibility APIs）。
- 指出 OSWorld 的两种主要失败模式：GUI 定位（GUI grounding）与操作知识（operational knowledge）。
- 总结 OSWorld-G 和 OSWorld-Human 在基础基准测试之上增加了哪些内容。

## 问题背景

通用智能体（generalist agents）能够调用工具。但它们能否通过约 20 次点击操作浏览器以完成购物结账？能否仅凭键盘和鼠标配置 Linux 系统？这正是 WebArena 和 OSWorld 旨在回答的问题。

## 核心概念

### WebArena（Zhou 等人，ICLR 2024）

- 涵盖四个自托管（self-hosted）Web 应用的 812 个长程（long-horizon）任务：一个购物网站、一个论坛、一个类 GitLab 的开发工具以及一个企业级内容管理系统（CMS）。
- 附带实用工具：地图、计算器和草稿板。
- 评估基于执行结果，通过 Gym API 进行——订单是否已提交、问题是否已关闭、CMS 页面是否已更新？
- 发布时：表现最佳的 GPT-4 智能体（agent）成功率为 14.41%，而人类为 78.24%。

自托管（self-hosted）的架构设计至关重要——由于目标应用版本已固定且可复现，该基准测试（benchmark）的结果非常稳定，不会出现波动。

### 扩展

- **VisualWebArena** —— 视觉定位（visually grounded）任务，其成功与否取决于对图像的解析能力（将截图作为一等观测输入）。
- **TheAgentCompany**（2024 年 12 月）—— 增加了终端操作与编程任务；更贴近真实的远程办公环境。

### OSWorld（Xie 等人，NeurIPS 2024）

- 涵盖 Ubuntu、Windows 和 macOS 的 369 个真实计算机操作任务。
- 对真实应用程序进行自由形式的键盘与鼠标控制。
- 以 1920×1080 分辨率的截图作为观测输入。
- 发布时：表现最佳的模型成功率为 12.24%，而人类为 72.36%。

### 主要失败模式

1. **GUI 定位（GUI grounding）**。像素到元素的映射。模型在 1920×1080 分辨率下难以可靠地定位 UI 元素。
2. **操作知识（Operational knowledge）**。某个设置位于哪个菜单、使用哪个快捷键、打开哪个偏好设置面板。这是人类经过多年积累形成的长尾知识。

### 后续研究

- **OSWorld-G** —— 包含 564 个样本的定位（grounding）测试套件及 Jedi 训练集。将定位与规划（planning）解耦，以便分别进行测量。
- **OSWorld-Human** —— 人工标注的黄金动作轨迹（gold action trajectories）。研究表明，顶尖智能体使用的步骤比必要步骤多出 1.4 至 2.7 倍（即轨迹效率差距）。

### 为何重要

Claude 计算机使用功能、OpenAI CUA 以及 Gemini 2.5 Computer Use（第 21 课）均基于受 WebArena 和 OSWorld 塑造的工作负载进行训练。基准测试是靶心，而生产级模型则是交付的答案。

### 基准测试的常见误区

- **仅依赖截图的评估**。OSWorld 以截图为驱动；若在 OSWorld 上评估使用文档对象模型（DOM）或无障碍 API（accessibility APIs）的智能体，则会掩盖其面临的定位挑战。
- **忽略轨迹长度**。仅以成功率评分会遗漏 OSWorld-Human 所揭示的 1.4 至 2.7 倍的步骤低效问题。
- **过时的自托管应用**。WebArena 的应用固定了特定版本；若在未重新整理（re-curation）的情况下直接更新，将破坏结果的可比性。

## 动手构建

`code/main.py` 实现了一个简易的 Web 智能体测试框架（harness）：

- 一个极简的“购物应用”状态机：`list_items`、`add_to_cart`、`checkout`。
- 3 个任务的黄金轨迹。
- 一个尝试执行各项任务的脚本化智能体。
- 基于执行的评估器（状态检查）与轨迹效率指标（实际步骤数与黄金步骤数对比）。

运行方式：

python3 code/main.py

输出：各任务的成功率与轨迹效率，其方法论与 OSWorld-Human 保持一致。

## 实际应用

- **WebArena Verified**（WebArena 验证版）在内部集群上自托管，用于持续评估。
- **OSWorld**（OSWorld 桌面基准）部署在虚拟机集群（VM Fleet）中，用于桌面智能体（Desktop Agents）。
- **Computer-use agents**（计算机操作智能体）（第 21 课）—— Claude、OpenAI CUA、Gemini —— 均基于此类工作负载进行训练。
- **Your own product flows**（自有产品流程）—— 为排名前 20 的任务捕获黄金轨迹（Gold Trajectories）；每周让智能体在这些轨迹上运行测试。

## 交付上线

`outputs/skill-web-desktop-harness.md` 构建了一个 Web/桌面智能体测试框架（Agent Harness），包含基于执行的评估（Execution-based Eval）和轨迹效率指标（Trajectory Efficiency Metric）。

## 练习

1. 在简易测试框架（Toy Harness）中添加第二个应用（一个论坛）。编写 3 个任务及其对应的黄金轨迹。
2. 为每个任务添加轨迹效率报告。在您的简易环境中，智能体的步数是黄金轨迹的 1 倍、2 倍还是 3 倍？
3. 实现一个“干扰项”工具（Distractor Tool）——即黄金轨迹中从未使用过的工具。脚本化智能体（Scripted Agent）会被它吸引吗？
4. 阅读 OSWorld-G 相关资料。在您自己的评估中，如何将定位失败（Grounding Failures）与规划失败（Planning Failures）区分开来？
5. 阅读 WebArena 的应用 README 文档。当您升级某个锁定版本的应用时，哪些功能会失效？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| WebArena | “Web 智能体基准测试” | 涵盖 4 个自托管应用的 812 个任务；采用 Gym 风格的评估方式 |
| VisualWebArena | “视觉版 WebArena” | 基于视觉定位的 WebArena；使用屏幕截图作为观测输入 |
| OSWorld | “桌面智能体基准测试” | 在真实 Ubuntu/Windows/macOS 系统上运行的 369 个任务 |
| GUI grounding | “像素到元素的映射” | 模型在 1920x1080 分辨率下定位 UI 元素的能力 |
| Operational knowledge | “操作系统操作知识” | 知道使用哪个菜单、哪个快捷键、哪个偏好设置面板 |
| OSWorld-G | “定位测试套件” | 564 个纯定位样本 + 训练集 |
| OSWorld-Human | “黄金轨迹” | 用于衡量效率的手动专家操作序列 |
| Trajectory efficiency | “相对于黄金轨迹的步数比” | 智能体步数除以人类最少步数 |

## 延伸阅读

- [Zhou 等人，WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) —— 四应用 Web 基准测试
- [Xie 等人，OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) —— 跨操作系统桌面基准测试
- [Anthropic，Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) —— Claude 针对基准测试优化的能力
- [OpenAI，Computer-Using Agent](https://openai.com/index/computer-using-agent/) —— OSWorld 与 WebArena 的性能数据