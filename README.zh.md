<p align="center">
  <img src="assets/banner.svg" alt="AI Engineering from Scratch — reference manual banner" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-1a1a1a?style=flat-square&labelColor=fafaf5" alt="MIT License"></a>
  <a href="ROADMAP.md"><img src="https://img.shields.io/badge/lessons-435-3553ff?style=flat-square&labelColor=fafaf5" alt="435 lessons"></a>
  <a href="#contents"><img src="https://img.shields.io/badge/phases-20-3553ff?style=flat-square&labelColor=fafaf5" alt="20 phases"></a>
  <a href="https://github.com/rohitg00/ai-engineering-from-scratch/stargazers"><img src="https://img.shields.io/github/stars/rohitg00/ai-engineering-from-scratch?style=flat-square&labelColor=fafaf5&color=3553ff" alt="GitHub stars"></a>
  <a href="https://aiengineeringfromscratch.com"><img src="https://img.shields.io/badge/web-aiengineeringfromscratch.com-3553ff?style=flat-square&labelColor=fafaf5" alt="Website"></a>
</p>

░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒

> **84% 的学生已经在使用人工智能（AI）工具，但仅有 18% 的人认为自己已具备专业使用它们的能力。** 本课程旨在弥合这一差距。
>
> 435 节课。20 个阶段。约 320 小时。涵盖 Python、TypeScript、Rust、Julia。每节课都会交付一个可复用的成果物（artifact）：提示词（prompt）、技能（skill）、智能体（agent）或 MCP 服务器（MCP server）。免费、开源、采用 MIT 许可证。
>
> 你不仅是在学习 AI，更是在亲手构建它。端到端，从零开始。

## 课程运作方式

大多数 AI 学习资料都是零散拼凑的。这里一篇论文，那里一篇微调（fine-tuning）教程，别处还有一个炫酷的智能体演示。这些碎片往往难以串联。你可能部署了一个聊天机器人，却无法解释其损失曲线（loss curve）；你可能为智能体接入了一个函数，却说不出调用它的模型内部注意力机制（attention）究竟在做什么。

本课程就是贯穿始终的骨架。20 个阶段，435 节课，四种语言：Python、TypeScript、Rust、Julia。一端是线性代数（linear algebra），另一端是自主集群（autonomous swarms）。每个算法都将从底层数学原理开始构建。反向传播（backprop）。分词器（tokenizer）。注意力机制（attention）。智能体循环（agent loop）。等到 PyTorch 登场时，你早已清楚它在底层究竟是如何运作的。

每节课都遵循相同的闭环：阅读问题、推导数学公式、编写代码、运行测试、保留成果物。没有五分钟速成视频，没有复制粘贴式的部署，更没有保姆式教学。免费、开源，专为在你的个人笔记本电脑上运行而设计。

░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒

## 课程架构

二十个阶段层层递进。数学是地基，智能体与生产环境部署是屋顶。如果你已经掌握底层知识，可以跳过前面的内容；但切勿盲目跳跃，否则当顶层模块出现问题时，你将无从排查。

%%{init: {'theme':'base','themeVariables':{'primaryColor':'#fafaf5','primaryTextColor':'#1a1a1a','primaryBorderColor':'#3553ff','lineColor':'#3553ff','fontFamily':'JetBrains Mono','fontSize':'12px'}}}%%
flowchart TB
  P0["Phase 0 — Setup &amp; Tooling"] --> P1["Phase 1 — Math Foundations"]
  P1 --> P2["Phase 2 — ML Fundamentals"]
  P2 --> P3["Phase 3 — Deep Learning Core"]
  P3 --> P4["Phase 4 — Vision"]
  P3 --> P5["Phase 5 — NLP"]
  P3 --> P6["Phase 6 — Speech &amp; Audio"]
  P3 --> P9["Phase 9 — RL"]
  P5 --> P7["Phase 7 — Transformers"]
  P7 --> P8["Phase 8 — GenAI"]
  P7 --> P10["Phase 10 — LLMs from Scratch"]
  P10 --> P11["Phase 11 — LLM Engineering"]
  P10 --> P12["Phase 12 — Multimodal"]
  P11 --> P13["Phase 13 — Tools &amp; Protocols"]
  P13 --> P14["Phase 14 — Agent Engineering"]
  P14 --> P15["Phase 15 — Autonomous Systems"]
  P15 --> P16["Phase 16 — Multi-Agent &amp; Swarms"]
  P14 --> P17["Phase 17 — Infrastructure &amp; Production"]
  P15 --> P18["Phase 18 — Ethics &amp; Alignment"]
  P16 --> P19["Phase 19 — Capstone Projects"]
  P17 --> P19
  P18 --> P19

░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒

## 单课结构

每个课程都位于独立的文件夹中，整个课程体系采用统一的结构：

phases/<NN>-<phase-name>/<NN>-<lesson-name>/
├── code/      runnable implementations (Python, TypeScript, Rust, Julia)
├── docs/
│   └── en.md  lesson narrative
└── outputs/   prompts, skills, agents, or MCP servers this lesson produces

每个课程都遵循六个核心环节（beats）。*从零构建（Build It）/ 实际应用（Use It）* 的划分是课程的主轴——你首先会从零开始实现算法，然后再通过生产级库（production library）运行相同的逻辑。因为你亲手编写过简化版，所以你能真正理解框架底层在做什么。

%%{init: {'theme':'base','themeVariables':{'primaryColor':'#fafaf5','primaryTextColor':'#1a1a1a','primaryBorderColor':'#3553ff','lineColor':'#3553ff','fontFamily':'JetBrains Mono','fontSize':'13px'}}}%%
flowchart LR
  M["MOTTO<br/><sub>one-line core idea</sub>"] --> Pr["PROBLEM<br/><sub>concrete pain</sub>"]
  Pr --> C["CONCEPT<br/><sub>diagrams &amp; intuition</sub>"]
  C --> B["BUILD IT<br/><sub>raw math, no frameworks</sub>"]
  B --> U["USE IT<br/><sub>same thing in PyTorch / sklearn</sub>"]
  U --> S["SHIP IT<br/><sub>prompt · skill · agent · MCP</sub>"]

## 快速开始

提供三种入门方式，任选其一。

**选项 A — 直接阅读。** 在 [aiengineeringfromscratch.com](https://aiengineeringfromscratch.com) 打开任意已完成的课程，或在 [目录](#contents) 下展开某个阶段（phase）。无需配置环境，也无需克隆仓库。

**选项 B — 克隆并运行。**

git clone https://github.com/rohitg00/ai-engineering-from-scratch.git
cd ai-engineering-from-scratch
python phases/01-math-foundations/01-linear-algebra-intuition/code/vectors.py

**选项 C — 评估你的水平 *（推荐）*。** 智能跳过已掌握的内容。在 Claude、Cursor、Codex、OpenClaw、Hermes 或任何安装了 SkillKit 的智能体（agent）中运行：

/find-your-level

共十道题目。系统会将你的知识水平映射到起始阶段，并生成带有预估学习时长的个性化路径。完成每个阶段后：

/check-understanding 3        # quiz yourself on phase 3
ls phases/03-deep-learning-core/05-loss-functions/outputs/
# ├── prompt-loss-function-selector.md
# └── prompt-loss-debugger.md

### 前置要求

- 具备编程能力（任何语言均可，掌握 Python 更佳）。
- 希望深入理解 AI **底层的工作原理**，而不仅仅是调用 API。

### 内置智能体技能（SkillKit / Claude, Cursor, Codex, OpenClaw, Hermes）

| 技能（Skill） | 功能说明 |
|---|---|
| [`/find-your-level`](.claude/skills/find-your-level/SKILL.md) | 十道定位测试题。将你的知识水平映射至起始阶段，并生成附带预估学习时长的个性化路径。 |
| [`/check-understanding <phase>`](.claude/skills/check-understanding/SKILL.md) | 针对每个阶段的测验，共八道题，提供反馈并指出需要复习的具体课程。 |

░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒

## 每个课程都会交付实用工具

其他课程体系通常以*“恭喜，你学会了 X。”*作为结尾。而这里的每个课程最终都会交付一个**可复用的工具**，你可以直接安装或将其集成到日常开发工作流中。

<table>
<tr>
<th align="left" width="25%"><img src="site/assets/figures/001-a-prompts.svg" width="96" height="96" alt="FIG_001.A 提示词 (Prompts)"/><br/><sub>FIG_001 · A</sub><br/><b>提示词 (Prompts)</b></th>
<th align="left" width="25%"><img src="site/assets/figures/001-b-skills.svg" width="96" height="96" alt="FIG_001.B 技能 (Skills)"/><br/><sub>FIG_001 · B</sub><br/><b>技能 (Skills)</b></th>
<th align="left" width="25%"><img src="site/assets/figures/001-c-agents.svg" width="96" height="96" alt="FIG_001.C 智能体 (Agents)"/><br/><sub>FIG_001 · C</sub><br/><b>智能体 (Agents)</b></th>
<th align="left" width="25%"><img src="site/assets/figures/001-d-mcp-servers.svg" width="96" height="96" alt="FIG_001.D MCP 服务器 (MCP Servers)"/><br/><sub>FIG_001 · D</sub><br/><b>MCP 服务器 (MCP Servers)</b></th>
</tr>
<tr>
<td valign="top">粘贴至任意 AI 助手，即可在特定任务上获得专家级辅助。</td>
<td valign="top">放入 Claude、Cursor、Codex、OpenClaw、Hermes 或任何支持读取 <code>SKILL.md</code> 的智能体中。</td>
<td valign="top">部署为自主工作节点——你在第 14 阶段已亲手编写了该循环。</td>
<td valign="top">接入任意兼容 MCP 的客户端。在第 13 阶段已完成端到端构建。</td>
</tr>
</table>

> 使用 [SkillKit](https://github.com/rohitg00/skillkit) 一键安装全部工具。这是实战利器，而非课后作业。完成本系列课程后，你将拥有一份包含 435 个作品的作品集，且因为你亲手构建了它们，你将对其原理了如指掌。

### FIG_002 · 完整示例

第 14 阶段，第 1 课：智能体循环 (Agent Loop)。约 120 行纯 Python 代码，零依赖。

<table>
<tr>
<td valign="top" width="50%">

**`code/agent_loop.py`** &nbsp; <sub><i>构建它</i></sub>

def run(query, tools):
    history = [user(query)]
    for step in range(MAX_STEPS):
        msg = llm(history)
        if msg.tool_calls:
            for call in msg.tool_calls:
                result = tools[call.name](**call.args)
                history.append(tool_result(call.id, result))
            continue
        return msg.content
    raise StepLimitExceeded

</td>
<td valign="top" width="50%">

**`outputs/skill-agent-loop.md`** &nbsp; <sub><i>发布它</i></sub>

---
name: agent-loop
description: 适用于任意工具列表的 ReAct 风格 (ReAct-style) 循环
phase: 14
lesson: 01
---

Implement a minimal agent loop that...

**`outputs/prompt-debug-agent.md`**

You are an agent debugger. Given the trace
of an agent run, identify the step where
the agent went wrong and explain why...

</td>
</tr>
</table>

░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒

<a id="contents"></a>

## 目录

共二十个阶段。点击任意阶段即可展开其课程列表。

<a id="phase-0"></a>
### 第 0 阶段：环境设置与工具链 `12 节课`
> 为后续所有内容准备好你的开发环境。

| # | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [开发环境](phases/00-setup-and-tooling/01-dev-environment/) | 构建 | Python, TypeScript, Rust |
| 02 | [Git 与协作](phases/00-setup-and-tooling/02-git-and-collaboration/) | 学习 | — |
| 03 | [GPU 设置与云服务](phases/00-setup-and-tooling/03-gpu-setup-and-cloud/) | 构建 | Python |
| 04 | [API 与密钥](phases/00-setup-and-tooling/04-apis-and-keys/) | 构建 | Python, TypeScript |
| 05 | [Jupyter Notebook](phases/00-setup-and-tooling/05-jupyter-notebooks/) | 构建 | Python |
| 06 | [Python 环境](phases/00-setup-and-tooling/06-python-environments/) | 构建 | Python |
| 07 | [面向 AI 的 Docker](phases/00-setup-and-tooling/07-docker-for-ai/) | 构建 | Python |
| 08 | [编辑器配置](phases/00-setup-and-tooling/08-editor-setup/) | 构建 | — |
| 09 | [数据管理](phases/00-setup-and-tooling/09-data-management/) | 构建 | Python |
| 10 | [终端与 Shell](phases/00-setup-and-tooling/10-terminal-and-shell/) | 学习 | — |
| 11 | [面向 AI 的 Linux](phases/00-setup-and-tooling/11-linux-for-ai/) | 学习 | — |
| 12 | [调试与性能分析](phases/00-setup-and-tooling/12-debugging-and-profiling/) | 构建 | Python |

<details id="phase-1">
<summary><b>第一阶段 — 数学基础 (Math Foundations)</b> &nbsp;<code>22 节课</code>&nbsp; <em>通过代码深入理解每个 AI 算法背后的直觉。</em></summary>
<br/>

| 序号 | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [线性代数直觉](phases/01-math-foundations/01-linear-algebra-intuition/) | 学习 | Python, Julia |
| 02 | [向量、矩阵与运算](phases/01-math-foundations/02-vectors-matrices-operations/) | 实践 | Python, Julia |
| 03 | [矩阵变换与特征值](phases/01-math-foundations/03-matrix-transformations/) | 实践 | Python, Julia |
| 04 | [机器学习微积分：导数与梯度](phases/01-math-foundations/04-calculus-for-ml/) | 学习 | Python |
| 05 | [链式法则与自动微分](phases/01-math-foundations/05-chain-rule-and-autodiff/) | 实践 | Python |
| 06 | [概率与分布](phases/01-math-foundations/06-probability-and-distributions/) | 学习 | Python |
| 07 | [贝叶斯定理与统计思维](phases/01-math-foundations/07-bayes-theorem/) | 实践 | Python |
| 08 | [优化：梯度下降族](phases/01-math-foundations/08-optimization/) | 实践 | Python |
| 09 | [信息论：熵与 KL 散度](phases/01-math-foundations/09-information-theory/) | 学习 | Python |
| 10 | [降维：PCA、t-SNE 与 UMAP](phases/01-math-foundations/10-dimensionality-reduction/) | 实践 | Python |
| 11 | [奇异值分解](phases/01-math-foundations/11-singular-value-decomposition/) | 实践 | Python, Julia |
| 12 | [张量运算](phases/01-math-foundations/12-tensor-operations/) | 实践 | Python |
| 13 | [数值稳定性](phases/01-math-foundations/13-numerical-stability/) | 实践 | Python |
| 14 | [范数与距离](phases/01-math-foundations/14-norms-and-distances/) | 实践 | Python |
| 15 | [机器学习统计学](phases/01-math-foundations/15-statistics-for-ml/) | 实践 | Python |
| 16 | [采样方法](phases/01-math-foundations/16-sampling-methods/) | 实践 | Python |
| 17 | [线性系统](phases/01-math-foundations/17-linear-systems/) | 实践 | Python |
| 18 | [凸优化](phases/01-math-foundations/18-convex-optimization/) | 实践 | Python |
| 19 | [AI 中的复数](phases/01-math-foundations/19-complex-numbers/) | 学习 | Python |
| 20 | [傅里叶变换](phases/01-math-foundations/20-fourier-transform/) | 实践 | Python |
| 21 | [机器学习图论](phases/01-math-foundations/21-graph-theory/) | 实践 | Python |
| 22 | [随机过程](phases/01-math-foundations/22-stochastic-processes/) | 学习 | Python |

</details>

<details id="phase-2">
<summary><b>第二阶段 — 机器学习基础 (ML Fundamentals)</b> &nbsp;<code>18 节课</code>&nbsp; <em>经典机器学习——仍是大多数生产级 AI 的基石。</em></summary>
<br/>

| # | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [什么是机器学习 (Machine Learning)](phases/02-ml-fundamentals/01-what-is-machine-learning/) | 学习 | Python |
| 02 | [从零实现线性回归 (Linear Regression)](phases/02-ml-fundamentals/02-linear-regression/) | 实战构建 | Python |
| 03 | [逻辑回归 (Logistic Regression) 与分类 (Classification)](phases/02-ml-fundamentals/03-logistic-regression/) | 实战构建 | Python |
| 04 | [决策树 (Decision Trees) 与随机森林 (Random Forests)](phases/02-ml-fundamentals/04-decision-trees/) | 实战构建 | Python |
| 05 | [支持向量机 (Support Vector Machines)](phases/02-ml-fundamentals/05-support-vector-machines/) | 实战构建 | Python |
| 06 | [K近邻 (KNN) 与距离度量 (Distance Metrics)](phases/02-ml-fundamentals/06-knn-and-distances/) | 实战构建 | Python |
| 07 | [无监督学习 (Unsupervised Learning)：K-Means 与 DBSCAN](phases/02-ml-fundamentals/07-unsupervised-learning/) | 实战构建 | Python |
| 08 | [特征工程 (Feature Engineering) 与特征选择](phases/02-ml-fundamentals/08-feature-engineering/) | 实战构建 | Python |
| 09 | [模型评估 (Model Evaluation)：评估指标 (Metrics) 与交叉验证 (Cross-Validation)](phases/02-ml-fundamentals/09-model-evaluation/) | 实战构建 | Python |
| 10 | [偏差 (Bias)、方差 (Variance) 与学习曲线 (Learning Curve)](phases/02-ml-fundamentals/10-bias-variance/) | 学习 | Python |
| 11 | [集成方法 (Ensemble Methods)：Boosting、Bagging 与 Stacking](phases/02-ml-fundamentals/11-ensemble-methods/) | 实战构建 | Python |
| 12 | [超参数调优 (Hyperparameter Tuning)](phases/02-ml-fundamentals/12-hyperparameter-tuning/) | 实战构建 | Python |
| 13 | [机器学习流水线 (ML Pipelines) 与实验追踪 (Experiment Tracking)](phases/02-ml-fundamentals/13-ml-pipelines/) | 实战构建 | Python |
| 14 | [朴素贝叶斯 (Naive Bayes)](phases/02-ml-fundamentals/14-naive-bayes/) | 实战构建 | Python |
| 15 | [时间序列基础 (Time Series Fundamentals)](phases/02-ml-fundamentals/15-time-series/) | 实战构建 | Python |
| 16 | [异常检测 (Anomaly Detection)](phases/02-ml-fundamentals/16-anomaly-detection/) | 实战构建 | Python |
| 17 | [处理不平衡数据 (Handling Imbalanced Data)](phases/02-ml-fundamentals/17-imbalanced-data/) | 实战构建 | Python |
| 18 | [特征选择](phases/02-ml-fundamentals/18-feature-selection/) | 实战构建 | Python |

</details>

<details id="phase-3">
<summary><b>阶段 3 — 深度学习核心 (Deep Learning Core)</b> &nbsp;<code>13 节课</code>&nbsp; <em>从第一性原理构建神经网络。在亲手实现之前，不使用任何框架。</em></summary>
<br/>

| # | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [感知机 (Perceptron)：一切的起点](phases/03-deep-learning-core/01-the-perceptron/) | 实战构建 | Python |
| 02 | [多层网络 (Multi-Layer Networks) 与前向传播 (Forward Pass)](phases/03-deep-learning-core/02-multi-layer-networks/) | 实战构建 | Python |
| 03 | [从零实现反向传播 (Backpropagation)](phases/03-deep-learning-core/03-backpropagation/) | 实战构建 | Python |
| 04 | [激活函数 (Activation Functions)：ReLU、Sigmoid、GELU 及其原理](phases/03-deep-learning-core/04-activation-functions/) | 实战构建 | Python |
| 05 | [损失函数 (Loss Functions)：MSE、交叉熵 (Cross-Entropy) 与对比损失 (Contrastive)](phases/03-deep-learning-core/05-loss-functions/) | 实战构建 | Python |
| 06 | [优化器 (Optimizers)：SGD、动量法 (Momentum)、Adam 与 AdamW](phases/03-deep-learning-core/06-optimizers/) | 实战构建 | Python |
| 07 | [正则化 (Regularization)：Dropout、权重衰减 (Weight Decay) 与批归一化 (BatchNorm)](phases/03-deep-learning-core/07-regularization/) | 实战构建 | Python |
| 08 | [权重初始化 (Weight Initialization) 与训练稳定性 (Training Stability)](phases/03-deep-learning-core/08-weight-initialization/) | 实战构建 | Python |
| 09 | [学习率调度 (Learning Rate Schedules) 与预热 (Warmup)](phases/03-deep-learning-core/09-learning-rate-schedules/) | 实战构建 | Python |
| 10 | [构建你的迷你框架 (Mini Framework)](phases/03-deep-learning-core/10-mini-framework/) | 实战构建 | Python |
| 11 | [PyTorch 入门](phases/03-deep-learning-core/11-intro-to-pytorch/) | 实战构建 | Python |
| 12 | [JAX 入门](phases/03-deep-learning-core/12-intro-to-jax/) | 实战构建 | Python |
| 13 | [神经网络调试 (Debugging Neural Networks)](phases/03-deep-learning-core/13-debugging-neural-networks/) | 实战构建 | Python |

</details>

<details id="phase-4">
<summary><b>阶段 4 — 计算机视觉 (Computer Vision)</b> &nbsp;<code>28 节课</code>&nbsp; <em>从像素到理解——涵盖图像、视频、3D、视觉语言模型 (VLMs) 与世界模型 (World Models)。</em></summary>
<br/>

| 序号 | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [图像基础：像素 (Pixel)、通道 (Channel) 与色彩空间 (Color Space)](phases/04-computer-vision/01-image-fundamentals/) | 学习 | Python |
| 02 | [从零实现卷积 (Convolution)](phases/04-computer-vision/02-convolutions-from-scratch/) | 实战 | Python |
| 03 | [卷积神经网络 (Convolutional Neural Network, CNN)：从 LeNet 到 ResNet](phases/04-computer-vision/03-cnns-lenet-to-resnet/) | 实战 | Python |
| 04 | [图像分类 (Image Classification)](phases/04-computer-vision/04-image-classification/) | 实战 | Python |
| 05 | [迁移学习 (Transfer Learning) 与微调 (Fine-Tuning)](phases/04-computer-vision/05-transfer-learning/) | 实战 | Python |
| 06 | [目标检测 (Object Detection) — 从零实现 YOLO](phases/04-computer-vision/06-object-detection-yolo/) | 实战 | Python |
| 07 | [语义分割 (Semantic Segmentation) — U-Net](phases/04-computer-vision/07-semantic-segmentation-unet/) | 实战 | Python |
| 08 | [实例分割 (Instance Segmentation) — Mask R-CNN](phases/04-computer-vision/08-instance-segmentation-mask-rcnn/) | 实战 | Python |
| 09 | [图像生成 (Image Generation) — 生成对抗网络 (Generative Adversarial Network, GAN)](phases/04-computer-vision/09-image-generation-gans/) | 实战 | Python |
| 10 | [图像生成 — 扩散模型 (Diffusion Model)](phases/04-computer-vision/10-image-generation-diffusion/) | 实战 | Python |
| 11 | [Stable Diffusion — 架构与微调](phases/04-computer-vision/11-stable-diffusion/) | 实战 | Python |
| 12 | [视频理解 (Video Understanding) — 时序建模 (Temporal Modeling)](phases/04-computer-vision/12-video-understanding/) | 实战 | Python |
| 13 | [3D 视觉 (3D Vision)：点云 (Point Cloud) 与神经辐射场 (Neural Radiance Field, NeRF)](phases/04-computer-vision/13-3d-vision-nerf/) | 实战 | Python |
| 14 | [视觉 Transformer (Vision Transformer, ViT)](phases/04-computer-vision/14-vision-transformers/) | 实战 | Python |
| 15 | [实时视觉：边缘部署 (Edge Deployment)](phases/04-computer-vision/15-real-time-edge/) | 实战 | Python, Rust |
| 16 | [构建完整视觉流水线 (Vision Pipeline)](phases/04-computer-vision/16-vision-pipeline-capstone/) | 实战 | Python |
| 17 | [自监督视觉 (Self-Supervised Vision) — SimCLR、DINO 与 MAE](phases/04-computer-vision/17-self-supervised-vision/) | 实战 | Python |
| 18 | [开放词汇视觉 (Open-Vocabulary Vision) — CLIP](phases/04-computer-vision/18-open-vocab-clip/) | 实战 | Python |
| 19 | [光学字符识别 (Optical Character Recognition, OCR) 与文档理解 (Document Understanding)](phases/04-computer-vision/19-ocr-document-understanding/) | 实战 | Python |
| 20 | [图像检索 (Image Retrieval) 与度量学习 (Metric Learning)](phases/04-computer-vision/20-image-retrieval-metric/) | 实战 | Python |
| 21 | [关键点检测 (Keypoint Detection) 与姿态估计 (Pose Estimation)](phases/04-computer-vision/21-keypoint-pose/) | 实战 | Python |
| 22 | [从零实现 3D 高斯溅射 (3D Gaussian Splatting)](phases/04-computer-vision/22-3d-gaussian-splatting/) | 实战 | Python |
| 23 | [扩散 Transformer (Diffusion Transformer) 与整流流 (Rectified Flow)](phases/04-computer-vision/23-diffusion-transformers-rectified-flow/) | 实战 | Python |
| 24 | [SAM 3 与开放词汇分割 (Open-Vocabulary Segmentation)](phases/04-computer-vision/24-sam3-open-vocab-segmentation/) | 实战 | Python |
| 25 | [视觉语言模型 (Vision-Language Model, VLM)](phases/04-computer-vision/25-vision-language-models/) | 实战 | Python |
| 26 | [单目深度 (Monocular Depth) 与几何估计 (Geometry Estimation)](phases/04-computer-vision/26-monocular-depth/) | 实战 | Python |
| 27 | [多目标跟踪 (Multi-Object Tracking) 与视频记忆 (Video Memory)](phases/04-computer-vision/27-multi-object-tracking/) | 实战 | Python |
| 28 | [世界模型 (World Model) 与视频扩散 (Video Diffusion)](phases/04-computer-vision/28-world-models-video-diffusion/) | 实战 | Python |

</details>

<details id="phase-5">
<summary><b>第五阶段 — 自然语言处理 (Natural Language Processing, NLP)：从基础到进阶</b> &nbsp;<code>29 课时</code>&nbsp; <em>语言是通往智能的接口。</em></summary>
<br/>

| # | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [文本处理：分词（Tokenization）、词干提取（Stemming）、词形还原（Lemmatization）](phases/05-nlp-foundations-to-advanced/01-text-processing/) | 实战 | Python |
| 02 | [词袋模型（Bag of Words）、TF-IDF 与文本表示（Text Representation）](phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf/) | 实战 | Python |
| 03 | [词嵌入（Word Embeddings）：从零实现 Word2Vec](phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec/) | 实战 | Python |
| 04 | [GloVe、FastText 与子词嵌入（Subword Embeddings）](phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword/) | 实战 | Python |
| 05 | [情感分析（Sentiment Analysis）](phases/05-nlp-foundations-to-advanced/05-sentiment-analysis/) | 实战 | Python |
| 06 | [命名实体识别（Named Entity Recognition, NER）](phases/05-nlp-foundations-to-advanced/06-named-entity-recognition/) | 实战 | Python |
| 07 | [词性标注（POS Tagging）与句法分析（Syntactic Parsing）](phases/05-nlp-foundations-to-advanced/07-pos-tagging-parsing/) | 实战 | Python |
| 08 | [文本分类（Text Classification）— 用于文本的卷积神经网络（CNNs）与循环神经网络（RNNs）](phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text/) | 实战 | Python |
| 09 | [序列到序列模型（Sequence-to-Sequence Models）](phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence/) | 实战 | Python |
| 10 | [注意力机制（Attention Mechanism）— 突破性进展](phases/05-nlp-foundations-to-advanced/10-attention-mechanism/) | 实战 | Python |
| 11 | [机器翻译（Machine Translation）](phases/05-nlp-foundations-to-advanced/11-machine-translation/) | 实战 | Python |
| 12 | [文本摘要（Text Summarization）](phases/05-nlp-foundations-to-advanced/12-text-summarization/) | 实战 | Python |
| 13 | [问答系统（Question Answering Systems）](phases/05-nlp-foundations-to-advanced/13-question-answering/) | 实战 | Python |
| 14 | [信息检索（Information Retrieval）与搜索](phases/05-nlp-foundations-to-advanced/14-information-retrieval-search/) | 实战 | Python |
| 15 | [主题建模（Topic Modeling）：LDA、BERTopic](phases/05-nlp-foundations-to-advanced/15-topic-modeling/) | 实战 | Python |
| 16 | [文本生成（Text Generation）](phases/05-nlp-foundations-to-advanced/16-text-generation-pre-transformer/) | 实战 | Python |
| 17 | [聊天机器人（Chatbots）：从基于规则到神经网络](phases/05-nlp-foundations-to-advanced/17-chatbots-rule-to-neural/) | 实战 | Python |
| 18 | [多语言自然语言处理（Multilingual NLP）](phases/05-nlp-foundations-to-advanced/18-multilingual-nlp/) | 实战 | Python |
| 19 | [子词分词（Subword Tokenization）：BPE、WordPiece、Unigram、SentencePiece](phases/05-nlp-foundations-to-advanced/19-subword-tokenization/) | 学习 | Python |
| 20 | [结构化输出（Structured Outputs）与受限解码（Constrained Decoding）](phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding/) | 实战 | Python |
| 21 | [自然语言推理（Natural Language Inference, NLI）与文本蕴含（Textual Entailment）](phases/05-nlp-foundations-to-advanced/21-nli-textual-entailment/) | 学习 | Python |
| 22 | [嵌入模型（Embedding Models）深度解析](phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive/) | 学习 | Python |
| 23 | [检索增强生成（Retrieval-Augmented Generation, RAG）的分块策略（Chunking Strategies）](phases/05-nlp-foundations-to-advanced/23-chunking-strategies-rag/) | 实战 | Python |
| 24 | [共指消解（Coreference Resolution）](phases/05-nlp-foundations-to-advanced/24-coreference-resolution/) | 学习 | Python |
| 25 | [实体链接（Entity Linking）与消歧（Disambiguation）](phases/05-nlp-foundations-to-advanced/25-entity-linking/) | 实战 | Python |
| 26 | [关系抽取（Relation Extraction）与知识图谱构建（Knowledge Graph Construction）](phases/05-nlp-foundations-to-advanced/26-relation-extraction-kg/) | 实战 | Python |
| 27 | [大语言模型评估（LLM Evaluation）：RAGAS、DeepEval、G-Eval](phases/05-nlp-foundations-to-advanced/27-llm-evaluation-frameworks/) | 实战 | Python |
| 28 | [长上下文评估（Long-Context Evaluation）：NIAH、RULER、LongBench、MRCR](phases/05-nlp-foundations-to-advanced/28-long-context-evaluation/) | 学习 | Python |
| 29 | [对话状态跟踪（Dialogue State Tracking）](phases/05-nlp-foundations-to-advanced/29-dialogue-state-tracking/) | 实战 | Python |

</details>

<details id="phase-6">
<summary><b>第六阶段 — 语音与音频</b> &nbsp;<code>17 节课</code>&nbsp; <em>聆听、理解、表达。</em></summary>
<br/>

| # | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [音频基础：波形、采样与快速傅里叶变换 (FFT)](phases/06-speech-and-audio/01-audio-fundamentals) | 学习 | Python |
| 02 | [语谱图、梅尔刻度 (Mel Scale) 与音频特征](phases/06-speech-and-audio/02-spectrograms-mel-features) | 构建 | Python |
| 03 | [音频分类](phases/06-speech-and-audio/03-audio-classification) | 构建 | Python |
| 04 | [语音识别 (Automatic Speech Recognition, ASR)](phases/06-speech-and-audio/04-speech-recognition-asr) | 构建 | Python |
| 05 | [Whisper：架构与微调 (Fine-Tuning)](phases/06-speech-and-audio/05-whisper-architecture-finetuning) | 构建 | Python |
| 06 | [说话人识别与验证](phases/06-speech-and-audio/06-speaker-recognition-verification) | 构建 | Python |
| 07 | [文本转语音 (Text-to-Speech, TTS)](phases/06-speech-and-audio/07-text-to-speech) | 构建 | Python |
| 08 | [声音克隆与语音转换](phases/06-speech-and-audio/08-voice-cloning-conversion) | 构建 | Python |
| 09 | [音乐生成](phases/06-speech-and-audio/09-music-generation) | 构建 | Python |
| 10 | [音频-语言模型 (Audio-Language Models)](phases/06-speech-and-audio/10-audio-language-models) | 构建 | Python |
| 11 | [实时音频处理](phases/06-speech-and-audio/11-real-time-audio-processing) | 构建 | Python, Rust |
| 12 | [构建语音助手流水线 (Pipeline)](phases/06-speech-and-audio/12-voice-assistant-pipeline) | 构建 | Python |
| 13 | [神经音频编解码器 (Neural Audio Codecs) — EnCodec, SNAC, Mimi, DAC](phases/06-speech-and-audio/13-neural-audio-codecs) | 学习 | Python |
| 14 | [语音活动检测 (Voice Activity Detection, VAD) 与话轮转换](phases/06-speech-and-audio/14-voice-activity-detection-turn-taking) | 构建 | Python |
| 15 | [流式语音到语音 (Streaming Speech-to-Speech) — Moshi, Hibiki](phases/06-speech-and-audio/15-streaming-speech-to-speech-moshi-hibiki) | 学习 | Python |
| 16 | [语音反欺骗 (Voice Anti-Spoofing) 与音频水印](phases/06-speech-and-audio/16-anti-spoofing-audio-watermarking) | 构建 | Python |
| 17 | [音频评估 — 词错误率 (Word Error Rate, WER)、平均意见得分 (Mean Opinion Score, MOS)、MMAU 与排行榜](phases/06-speech-and-audio/17-audio-evaluation-metrics) | 学习 | Python |

</details>

<details id="phase-7">
<summary><b>阶段 7 — Transformer 架构深入解析 (Transformers Deep Dive)</b> &nbsp;<code>14 lessons</code>&nbsp; <em>改变一切的架构。</em></summary>
<br/>

| # | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [为什么使用 Transformer：RNN（循环神经网络）的局限性](phases/07-transformers-deep-dive/01-why-transformers/) | 学习 | Python |
| 02 | [从零实现自注意力机制（Self-Attention）](phases/07-transformers-deep-dive/02-self-attention-from-scratch/) | 构建 | Python |
| 03 | [多头注意力机制（Multi-Head Attention）](phases/07-transformers-deep-dive/03-multi-head-attention/) | 构建 | Python |
| 04 | [位置编码（Positional Encoding）：正弦编码、RoPE 与 ALiBi](phases/07-transformers-deep-dive/04-positional-encoding/) | 构建 | Python |
| 05 | [完整 Transformer 架构：编码器（Encoder）与解码器（Decoder）](phases/07-transformers-deep-dive/05-full-transformer/) | 构建 | Python |
| 06 | [BERT — 掩码语言建模（Masked Language Modeling）](phases/07-transformers-deep-dive/06-bert-masked-language-modeling/) | 构建 | Python |
| 07 | [GPT — 因果语言建模（Causal Language Modeling）](phases/07-transformers-deep-dive/07-gpt-causal-language-modeling/) | 构建 | Python |
| 08 | [T5 与 BART — 编码器-解码器模型（Encoder-Decoder Models）](phases/07-transformers-deep-dive/08-t5-bart-encoder-decoder/) | 学习 | Python |
| 09 | [视觉 Transformer（Vision Transformers, ViT）](phases/07-transformers-deep-dive/09-vision-transformers/) | 构建 | Python |
| 10 | [音频 Transformer — Whisper 架构](phases/07-transformers-deep-dive/10-audio-transformers-whisper/) | 学习 | Python |
| 11 | [混合专家模型（Mixture of Experts, MoE）](phases/07-transformers-deep-dive/11-mixture-of-experts/) | 构建 | Python |
| 12 | [KV 缓存（KV Cache）、Flash Attention 与推理优化（Inference Optimization）](phases/07-transformers-deep-dive/12-kv-cache-flash-attention/) | 构建 | Python |
| 13 | [缩放定律（Scaling Laws）](phases/07-transformers-deep-dive/13-scaling-laws/) | 学习 | Python |
| 14 | [从零构建 Transformer](phases/07-transformers-deep-dive/14-build-a-transformer-capstone/) | 构建 | Python |

</details>

<details id="phase-8">
<summary><b>阶段 8 — 生成式 AI（Generative AI）</b> &nbsp;<code>14 节课</code>&nbsp; <em>创建图像、视频、音频、3D 模型等。</em></summary>
<br/>

| # | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [生成模型（Generative Models）：分类与历史](phases/08-generative-ai/01-generative-models-taxonomy-history/) | 学习 | Python |
| 02 | [自编码器（Autoencoders）与变分自编码器（VAE）](phases/08-generative-ai/02-autoencoders-vae/) | 构建 | Python |
| 03 | [生成对抗网络（GANs）：生成器（Generator）与判别器（Discriminator）](phases/08-generative-ai/03-gans-generator-discriminator/) | 构建 | Python |
| 04 | [条件生成对抗网络（Conditional GANs）与 Pix2Pix](phases/08-generative-ai/04-conditional-gans-pix2pix/) | 构建 | Python |
| 05 | [StyleGAN](phases/08-generative-ai/05-stylegan/) | 构建 | Python |
| 06 | [扩散模型（Diffusion Models）— 从零实现 DDPM](phases/08-generative-ai/06-diffusion-ddpm-from-scratch/) | 构建 | Python |
| 07 | [潜在扩散模型（Latent Diffusion）与 Stable Diffusion](phases/08-generative-ai/07-latent-diffusion-stable-diffusion/) | 构建 | Python |
| 08 | [ControlNet、LoRA 与条件控制（Conditioning）](phases/08-generative-ai/08-controlnet-lora-conditioning/) | 构建 | Python |
| 09 | [图像修复（Inpainting）、外扩（Outpainting）与编辑](phases/08-generative-ai/09-inpainting-outpainting-editing/) | 构建 | Python |
| 10 | [视频生成（Video Generation）](phases/08-generative-ai/10-video-generation/) | 构建 | Python |
| 11 | [音频生成（Audio Generation）](phases/08-generative-ai/11-audio-generation/) | 构建 | Python |
| 12 | [3D 生成（3D Generation）](phases/08-generative-ai/12-3d-generation/) | 构建 | Python |
| 13 | [流匹配（Flow Matching）与整流流（Rectified Flows）](phases/08-generative-ai/13-flow-matching-rectified-flows/) | 构建 | Python |
| 14 | [模型评估：FID 与 CLIP 分数（CLIP Score）](phases/08-generative-ai/14-evaluation-fid-clip-score/) | 构建 | Python |

</details>

<details id="phase-9">
<summary><b>阶段 9 — 强化学习（Reinforcement Learning）</b> &nbsp;<code>12 节课</code>&nbsp; <em>RLHF（基于人类反馈的强化学习）与游戏 AI 的基石。</em></summary>
<br/>

| 序号 | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [马尔可夫决策过程（Markov Decision Processes, MDPs）、状态、动作与奖励](phases/09-reinforcement-learning/01-mdps-states-actions-rewards/) | 学习 | Python |
| 02 | [动态规划（Dynamic Programming）](phases/09-reinforcement-learning/02-dynamic-programming/) | 实践 | Python |
| 03 | [蒙特卡洛方法（Monte Carlo Methods）](phases/09-reinforcement-learning/03-monte-carlo-methods/) | 实践 | Python |
| 04 | [Q学习（Q-Learning）与SARSA](phases/09-reinforcement-learning/04-q-learning-sarsa/) | 实践 | Python |
| 05 | [深度Q网络（Deep Q-Networks, DQN）](phases/09-reinforcement-learning/05-dqn/) | 实践 | Python |
| 06 | [策略梯度（Policy Gradients）— REINFORCE](phases/09-reinforcement-learning/06-policy-gradients-reinforce/) | 实践 | Python |
| 07 | [演员-评论家（Actor-Critic）— A2C、A3C](phases/09-reinforcement-learning/07-actor-critic-a2c-a3c/) | 实践 | Python |
| 08 | [近端策略优化（Proximal Policy Optimization, PPO）](phases/09-reinforcement-learning/08-ppo/) | 实践 | Python |
| 09 | [奖励建模（Reward Modeling）与基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）](phases/09-reinforcement-learning/09-reward-modeling-rlhf/) | 实践 | Python |
| 10 | [多智能体强化学习（Multi-Agent Reinforcement Learning）](phases/09-reinforcement-learning/10-multi-agent-rl/) | 实践 | Python |
| 11 | [仿真到现实迁移（Sim-to-Real Transfer）](phases/09-reinforcement-learning/11-sim-to-real-transfer/) | 实践 | Python |
| 12 | [游戏强化学习（Reinforcement Learning for Games）](phases/09-reinforcement-learning/12-rl-for-games/) | 实践 | Python |

</details>

<details id="phase-10">
<summary><b>第10阶段 — 从零构建大语言模型（Large Language Models, LLMs）</b> &nbsp;<code>22节课</code>&nbsp; <em>构建、训练并深入理解大语言模型。</em></summary>
<br/>

| # | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [分词器 (Tokenizers)：BPE、WordPiece 与 SentencePiece](phases/10-llms-from-scratch/01-tokenizers/) | 构建 | Python |
| 02 | [从零构建分词器](phases/10-llms-from-scratch/02-building-a-tokenizer/) | 构建 | Python |
| 03 | [预训练 (Pre-Training) 数据流水线](phases/10-llms-from-scratch/03-data-pipelines/) | 构建 | Python |
| 04 | [预训练迷你 GPT 模型（1.24 亿参数）](phases/10-llms-from-scratch/04-pre-training-mini-gpt/) | 构建 | Python |
| 05 | [分布式训练 (Distributed Training)、FSDP 与 DeepSpeed](phases/10-llms-from-scratch/05-scaling-distributed/) | 构建 | Python |
| 06 | [指令微调 (Instruction Tuning) —— SFT](phases/10-llms-from-scratch/06-instruction-tuning-sft/) | 构建 | Python |
| 07 | [基于人类反馈的强化学习 (RLHF) —— 奖励模型与 PPO](phases/10-llms-from-scratch/07-rlhf/) | 构建 | Python |
| 08 | [直接偏好优化 (DPO)](phases/10-llms-from-scratch/08-dpo/) | 构建 | Python |
| 09 | [宪法 AI (Constitutional AI) 与自我改进](phases/10-llms-from-scratch/09-constitutional-ai-self-improvement/) | 构建 | Python |
| 10 | [模型评估 (Evaluation) —— 基准测试与评测](phases/10-llms-from-scratch/10-evaluation/) | 构建 | Python |
| 11 | [模型量化 (Quantization)：INT8、GPTQ、AWQ 与 GGUF](phases/10-llms-from-scratch/11-quantization/) | 构建 | Python, Rust |
| 12 | [推理 (Inference) 优化](phases/10-llms-from-scratch/12-inference-optimization/) | 构建 | Python |
| 13 | [构建完整的大语言模型 (LLM) 流水线](phases/10-llms-from-scratch/13-building-complete-llm-pipeline/) | 构建 | Python |
| 14 | [开源模型 (Open Models)：架构详解](phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs/) | 学习 | Python |
| 15 | [投机解码 (Speculative Decoding) 与 EAGLE-3](phases/10-llms-from-scratch/15-speculative-decoding-eagle3/) | 构建 | Python |
| 16 | [差分注意力机制 (Differential Attention)（V2）](phases/10-llms-from-scratch/16-differential-attention-v2/) | 构建 | Python |
| 17 | [原生稀疏注意力机制 (Native Sparse Attention)（DeepSeek NSA）](phases/10-llms-from-scratch/17-native-sparse-attention/) | 构建 | Python |
| 18 | [多 Token 预测 (Multi-Token Prediction)（MTP）](phases/10-llms-from-scratch/18-multi-token-prediction/) | 构建 | Python |
| 19 | [DualPipe 并行策略](phases/10-llms-from-scratch/19-dualpipe-parallelism/) | 学习 | Python |
| 20 | [DeepSeek-V3 架构详解](phases/10-llms-from-scratch/20-deepseek-v3-walkthrough/) | 学习 | Python |
| 21 | [Jamba —— 混合 SSM-Transformer 架构](phases/10-llms-from-scratch/21-jamba-hybrid-ssm-transformer/) | 学习 | Python |
| 22 | [异步与 Hogwild! 推理](phases/10-llms-from-scratch/22-async-hogwild-inference/) | 构建 | Python |

</details>

<details id="phase-11">
<summary><b>阶段 11 —— 大语言模型工程 (LLM Engineering)</b> &nbsp;<code>15 节课</code>&nbsp; <em>将大语言模型投入生产环境实战。</em></summary>
<br/>

| 序号 | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [提示词工程（Prompt Engineering）：技术与模式](phases/11-llm-engineering/01-prompt-engineering/) | Build | Python |
| 02 | [少样本学习（Few-Shot）、思维链（CoT）与思维树（Tree-of-Thought）](phases/11-llm-engineering/02-few-shot-cot/) | Build | Python |
| 03 | [结构化输出（Structured Outputs）](phases/11-llm-engineering/03-structured-outputs/) | Build | Python, TypeScript |
| 04 | [嵌入（Embeddings）与向量表示（Vector Representations）](phases/11-llm-engineering/04-embeddings/) | Build | Python |
| 05 | [上下文工程（Context Engineering）](phases/11-llm-engineering/05-context-engineering/) | Build | Python, TypeScript |
| 06 | [检索增强生成（Retrieval-Augmented Generation, RAG）](phases/11-llm-engineering/06-rag/) | Build | Python, TypeScript |
| 07 | [高级 RAG：分块（Chunking）与重排序（Reranking）](phases/11-llm-engineering/07-advanced-rag/) | Build | Python |
| 08 | [使用 LoRA 与 QLoRA 进行微调（Fine-Tuning）](phases/11-llm-engineering/08-fine-tuning-lora/) | Build | Python |
| 09 | [函数调用（Function Calling）与工具使用（Tool Use）](phases/11-llm-engineering/09-function-calling/) | Build | Python |
| 10 | [评估（Evaluation）与测试（Testing）](phases/11-llm-engineering/10-evaluation/) | Build | Python |
| 11 | [缓存（Caching）、速率限制（Rate Limiting）与成本（Cost）](phases/11-llm-engineering/11-caching-cost/) | Build | Python |
| 12 | [安全护栏（Guardrails）与安全机制（Safety）](phases/11-llm-engineering/12-guardrails/) | Build | Python |
| 13 | [构建生产级大语言模型（LLM）应用](phases/11-llm-engineering/13-production-app/) | Build | Python |
| 14 | [模型上下文协议（Model Context Protocol, MCP）](phases/11-llm-engineering/14-model-context-protocol/) | Build | Python |
| 15 | [提示词缓存（Prompt Caching）与上下文缓存（Context Caching）](phases/11-llm-engineering/15-prompt-caching/) | Build | Python |

</details>

<details id="phase-12">
<summary><b>第 12 阶段 — 多模态 AI（Multimodal AI）</b> &nbsp;<code>25 节课</code>&nbsp; <em>实现跨模态的视觉、听觉、阅读与推理——从视觉变换器（ViT）图块到计算机操作智能体。</em></summary>
<br/>

| # | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [视觉 Transformer (Vision Transformer) 与 Patch-Token 基元](phases/12-multimodal-ai/01-vision-transformer-patch-tokens/) | 学习 | Python |
| 02 | [CLIP 与对比视觉-语言预训练](phases/12-multimodal-ai/02-clip-contrastive-pretraining/) | 实战 | Python |
| 03 | [BLIP-2 Q-Former 作为模态桥梁](phases/12-multimodal-ai/03-blip2-qformer-bridge/) | 实战 | Python |
| 04 | [Flamingo 与门控交叉注意力机制](phases/12-multimodal-ai/04-flamingo-gated-cross-attention/) | 学习 | Python |
| 05 | [LLaVA 与视觉指令微调](phases/12-multimodal-ai/05-llava-visual-instruction-tuning/) | 实战 | Python |
| 06 | [任意分辨率视觉 — Patch-n'-Pack 与 NaFlex](phases/12-multimodal-ai/06-any-resolution-patch-n-pack/) | 实战 | Python |
| 07 | [开放权重视觉语言模型 (VLM) 实践指南：核心要素解析](phases/12-multimodal-ai/07-open-weight-vlm-recipes/) | 学习 | Python |
| 08 | [LLaVA-OneVision：单图、多图与视频](phases/12-multimodal-ai/08-llava-onevision-single-multi-video/) | 实战 | Python |
| 09 | [Qwen-VL 系列与动态帧率视频](phases/12-multimodal-ai/09-qwen-vl-family-dynamic-fps/) | 学习 | Python |
| 10 | [InternVL3 原生多模态预训练](phases/12-multimodal-ai/10-internvl3-native-multimodal/) | 学习 | Python |
| 11 | [Chameleon 早期融合纯 Token 架构](phases/12-multimodal-ai/11-chameleon-early-fusion-tokens/) | 实战 | Python |
| 12 | [Emu3 基于下一 Token 预测的生成](phases/12-multimodal-ai/12-emu3-next-token-for-generation/) | 学习 | Python |
| 13 | [Transfusion 自回归 + 扩散模型](phases/12-multimodal-ai/13-transfusion-autoregressive-diffusion/) | 实战 | Python |
| 14 | [Show-o 离散扩散统一架构](phases/12-multimodal-ai/14-show-o-discrete-diffusion-unified/) | 学习 | Python |
| 15 | [Janus-Pro 解耦编码器](phases/12-multimodal-ai/15-janus-pro-decoupled-encoders/) | 实战 | Python |
| 16 | [MIO 任意到任意流式处理](phases/12-multimodal-ai/16-mio-any-to-any-streaming/) | 学习 | Python |
| 17 | [视频-语言时序定位](phases/12-multimodal-ai/17-video-language-temporal-grounding/) | 实战 | Python |
| 18 | [百万 Token 上下文长视频处理](phases/12-multimodal-ai/18-long-video-million-token/) | 实战 | Python |
| 19 | [音频-语言模型：从 Whisper 到 AF3](phases/12-multimodal-ai/19-audio-language-whisper-to-af3/) | 实战 | Python |
| 20 | [全模态模型：思考者-讲述者流式架构](phases/12-multimodal-ai/20-omni-models-thinker-talker/) | 实战 | Python |
| 21 | [具身视觉-语言-动作模型 (VLA)：RT-2、OpenVLA、π0、GR00T](phases/12-multimodal-ai/21-embodied-vlas-openvla-pi0-groot/) | 学习 | Python |
| 22 | [文档与图表理解](phases/12-multimodal-ai/22-document-diagram-understanding/) | 实战 | Python |
| 23 | [ColPali 视觉原生文档检索增强生成 (RAG)](phases/12-multimodal-ai/23-colpali-vision-native-rag/) | 实战 | Python |
| 24 | [多模态 RAG 与跨模态检索](phases/12-multimodal-ai/24-multimodal-rag-cross-modal/) | 实战 | Python |
| 25 | [多模态智能体与计算机操作（综合项目）](phases/12-multimodal-ai/25-multimodal-agents-computer-use/) | 实战 | Python |

</details>

<details id="phase-13">
<summary><b>第 13 阶段 — 工具与协议</b> &nbsp;<code>23 节课</code>&nbsp; <em>人工智能与现实世界之间的接口。</em></summary>
<br/>

| # | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [工具接口 (Tool Interface)](phases/13-tools-and-protocols/01-the-tool-interface/) | 学习 | Python |
| 02 | [函数调用深入解析 (Function Calling Deep Dive)](phases/13-tools-and-protocols/02-function-calling-deep-dive/) | 构建 | Python |
| 03 | [并行与流式工具调用 (Parallel and Streaming Tool Calls)](phases/13-tools-and-protocols/03-parallel-and-streaming-tool-calls/) | 构建 | Python |
| 04 | [结构化输出 (Structured Output)](phases/13-tools-and-protocols/04-structured-output/) | 构建 | Python |
| 05 | [工具模式设计 (Tool Schema Design)](phases/13-tools-and-protocols/05-tool-schema-design/) | 学习 | Python |
| 06 | [MCP 基础 (MCP Fundamentals)](phases/13-tools-and-protocols/06-mcp-fundamentals/) | 学习 | Python |
| 07 | [构建 MCP 服务器 (Building an MCP Server)](phases/13-tools-and-protocols/07-building-an-mcp-server/) | 构建 | Python |
| 08 | [构建 MCP 客户端 (Building an MCP Client)](phases/13-tools-and-protocols/08-building-an-mcp-client/) | 构建 | Python |
| 09 | [MCP 传输层 (MCP Transports)](phases/13-tools-and-protocols/09-mcp-transports/) | 学习 | Python |
| 10 | [MCP 资源与提示词 (MCP Resources and Prompts)](phases/13-tools-and-protocols/10-mcp-resources-and-prompts/) | 构建 | Python |
| 11 | [MCP 采样 (MCP Sampling)](phases/13-tools-and-protocols/11-mcp-sampling/) | 构建 | Python |
| 12 | [MCP 根目录与引导 (MCP Roots and Elicitation)](phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/) | 构建 | Python |
| 13 | [MCP 异步任务 (MCP Async Tasks)](phases/13-tools-and-protocols/13-mcp-async-tasks/) | 构建 | Python |
| 14 | [MCP 应用 (MCP Apps)](phases/13-tools-and-protocols/14-mcp-apps/) | 构建 | Python |
| 15 | [MCP 安全 I — 工具投毒 (MCP Security I — Tool Poisoning)](phases/13-tools-and-protocols/15-mcp-security-tool-poisoning/) | 学习 | Python |
| 16 | [MCP 安全 II — OAuth 2.1 (MCP Security II — OAuth 2.1)](phases/13-tools-and-protocols/16-mcp-security-oauth-2-1/) | 构建 | Python |
| 17 | [MCP 网关与注册中心 (MCP Gateways and Registries)](phases/13-tools-and-protocols/17-mcp-gateways-and-registries/) | 学习 | Python |
| 18 | [生产环境 MCP 认证 — iii 上的 DCR + JWKS (MCP Auth in Production — DCR + JWKS on iii)](phases/13-tools-and-protocols/18-mcp-auth-production/) | 构建 | Python |
| 19 | [A2A 协议 (A2A Protocol)](phases/13-tools-and-protocols/19-a2a-protocol/) | 构建 | Python |
| 20 | [OpenTelemetry GenAI (OpenTelemetry GenAI)](phases/13-tools-and-protocols/20-opentelemetry-genai/) | 构建 | Python |
| 21 | [LLM 路由层 (LLM Routing Layer)](phases/13-tools-and-protocols/21-llm-routing-layer/) | 学习 | Python |
| 22 | [技能与智能体 SDK (Skills and Agent SDKs)](phases/13-tools-and-protocols/22-skills-and-agent-sdks/) | 学习 | Python |
| 23 | [综合实战 — 工具生态 (Capstone — Tool Ecosystem)](phases/13-tools-and-protocols/23-capstone-tool-ecosystem/) | 构建 | Python |

</details>

<details id="phase-14">
<summary><b>第 14 阶段 — 智能体工程 (Agent Engineering)</b> &nbsp;<code>42 lessons</code>&nbsp; <em>从第一性原理构建智能体 —— 循环、记忆、规划、框架、基准测试、生产部署、工作台。</em></summary>
<br/>

| # | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [智能体循环 (Agent Loop)](phases/14-agent-engineering/01-the-agent-loop/) | 实战 | Python |
| 02 | [ReWOO 与计划执行模式 (Plan-and-Execute)](phases/14-agent-engineering/02-rewoo-plan-and-execute/) | 实战 | Python |
| 03 | [反思机制 (Reflexion) 与语言强化学习 (Verbal Reinforcement Learning)](phases/14-agent-engineering/03-reflexion-verbal-rl/) | 实战 | Python |
| 04 | [思维树 (Tree of Thoughts) 与 LATS](phases/14-agent-engineering/04-tree-of-thoughts-lats/) | 实战 | Python |
| 05 | [自我优化 (Self-Refine) 与 CRITIC](phases/14-agent-engineering/05-self-refine-and-critic/) | 实战 | Python |
| 06 | [工具使用 (Tool Use) 与函数调用 (Function Calling)](phases/14-agent-engineering/06-tool-use-and-function-calling/) | 实战 | Python |
| 07 | [记忆机制 (Memory) — 虚拟上下文 (Virtual Context) 与 MemGPT](phases/14-agent-engineering/07-memory-virtual-context-memgpt/) | 实战 | Python |
| 08 | [记忆块 (Memory Blocks) 与休眠期计算 (Sleep-Time Compute)](phases/14-agent-engineering/08-memory-blocks-sleep-time-compute/) | 实战 | Python |
| 09 | [混合记忆 (Hybrid Memory) — Mem0 向量 (Vector) + 图 (Graph) + 键值存储 (KV)](phases/14-agent-engineering/09-hybrid-memory-mem0/) | 实战 | Python |
| 10 | [技能库 (Skill Libraries) 与终身学习 (Lifelong Learning) — Voyager](phases/14-agent-engineering/10-skill-libraries-voyager/) | 实战 | Python |
| 11 | [基于 HTN 与进化搜索 (Evolutionary Search) 的规划 (Planning)](phases/14-agent-engineering/11-planning-htn-and-evolutionary/) | 实战 | Python |
| 12 | [Anthropic 的工作流模式 (Workflow Patterns)](phases/14-agent-engineering/12-anthropic-workflow-patterns/) | 实战 | Python |
| 13 | [LangGraph — 有状态图 (Stateful Graphs) 与持久化执行 (Durable Execution)](phases/14-agent-engineering/13-langgraph-stateful-graphs/) | 实战 | Python |
| 14 | [AutoGen v0.4 — 参与者模型 (Actor Model)](phases/14-agent-engineering/14-autogen-actor-model/) | 实战 | Python |
| 15 | [CrewAI — 基于角色的团队 (Role-Based Crews) 与流程 (Flows)](phases/14-agent-engineering/15-crewai-role-based-crews/) | 实战 | Python |
| 16 | [OpenAI Agents SDK — 任务交接 (Handoffs)、安全护栏 (Guardrails) 与链路追踪 (Tracing)](phases/14-agent-engineering/16-openai-agents-sdk/) | 实战 | Python |
| 17 | [Claude Agent SDK — 子智能体 (Subagents) 与会话存储 (Session Store)](phases/14-agent-engineering/17-claude-agent-sdk/) | 实战 | Python |
| 18 | [Agno 与 Mastra — 生产级运行时 (Production Runtimes)](phases/14-agent-engineering/18-agno-and-mastra-runtimes/) | 学习 | Python, TypeScript |
| 19 | [基准测试 (Benchmarks) — SWE-bench、GAIA、AgentBench](phases/14-agent-engineering/19-benchmarks-swebench-gaia/) | 学习 | Python |
| 20 | [基准测试 — WebArena 与 OSWorld](phases/14-agent-engineering/20-benchmarks-webarena-osworld/) | 学习 | Python |
| 21 | [计算机操作 (Computer Use) — Claude、OpenAI CUA、Gemini](phases/14-agent-engineering/21-computer-use-agents/) | 实战 | Python |
| 22 | [语音智能体 (Voice Agents) — Pipecat 与 LiveKit](phases/14-agent-engineering/22-voice-agents-pipecat-livekit/) | 实战 | Python |
| 23 | [OpenTelemetry GenAI 语义规范 (Semantic Conventions)](phases/14-agent-engineering/23-otel-genai-conventions/) | 实战 | Python |
| 24 | [智能体可观测性 (Agent Observability) — Langfuse、Phoenix、Opik](phases/14-agent-engineering/24-agent-observability-platforms/) | 学习 | Python |
| 25 | [多智能体辩论 (Multi-Agent Debate) 与协作 (Collaboration)](phases/14-agent-engineering/25-multi-agent-debate/) | 实战 | Python |
| 26 | [故障模式 (Failure Modes) — 智能体为何失效](phases/14-agent-engineering/26-failure-modes-agentic/) | 实战 | Python |
| 27 | [提示词注入 (Prompt Injection) 与 PVE 防御](phases/14-agent-engineering/27-prompt-injection-defense/) | 实战 | Python |
| 28 | [编排模式 (Orchestration Patterns) — 监督者 (Supervisor)、蜂群 (Swarm)、层级架构 (Hierarchical)](phases/14-agent-engineering/28-orchestration-patterns/) | 实战 | Python |
| 29 | [生产级运行时 — 队列 (Queue)、事件 (Event)、定时任务 (Cron)](phases/14-agent-engineering/29-production-runtimes/) | 学习 | Python |
| 30 | [评估驱动的智能体开发 (Eval-Driven Agent Development)](phases/14-agent-engineering/30-eval-driven-agent-development/) | 实战 | Python |
| 31 | [智能体工作台 (Agent Workbench)：为何强大模型仍会失败](phases/14-agent-engineering/31-agent-workbench-why-models-fail/) | 学习 | Python |
| 32 | [极简智能体工作台](phases/14-agent-engineering/32-minimal-agent-workbench/) | 实战 | Python |
| 33 | [将智能体指令转化为可执行约束 (Executable Constraints)](phases/14-agent-engineering/33-instructions-as-executable-constraints/) | 实战 | Python |
| 34 | [仓库记忆 (Repo Memory) 与持久化状态 (Durable State)](phases/14-agent-engineering/34-repo-memory-and-state/) | 实战 | Python |
| 35 | [智能体初始化脚本 (Initialization Scripts)](phases/14-agent-engineering/35-initialization-scripts/) | 实战 | Python |
| 36 | [范围契约 (Scope Contracts) 与任务边界 (Task Boundaries)](phases/14-agent-engineering/36-scope-contracts/) | 实战 | Python |
| 37 | [运行时反馈循环 (Runtime Feedback Loops)](phases/14-agent-engineering/37-runtime-feedback-loops/) | 实战 | Python |
| 38 | [验证关卡 (Verification Gates)](phases/14-agent-engineering/38-verification-gates/) | 实战 | Python |
| 39 | [审查智能体 (Reviewer Agent)：分离构建者 (Builder) 与标记者 (Marker)](phases/14-agent-engineering/39-reviewer-agent/) | 实战 | Python |
| 40 | [多会话交接 (Multi-Session Handoff)](phases/14-agent-engineering/40-multi-session-handoff/) | 实战 | Python |
| 41 | [在真实代码库上运行工作台](phases/14-agent-engineering/41-workbench-for-real-repos/) | 实战 | Python |
| 42 | [综合项目 (Capstone)：交付可复用的智能体工作台工具包](phases/14-agent-engineering/42-agent-workbench-capstone/) | 实战 | Python |

第14阶段（Phase 14）的每个工作台课程（31-42）均会附带一份 `mission.md` 文件，用于在智能体（agent）打开完整课程文档前为其提供任务简报。

</details>

<details id="phase-15">
<summary><b>第15阶段 — 自主系统（Autonomous Systems）</b> &nbsp;<code>22 lessons</code>&nbsp; <em>长程智能体（Long-horizon agents）、自我改进（self-improvement）以及2026年安全栈（safety stack）。</em></summary>
<br/>

| # | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [从聊天机器人到长程智能体（METR）](phases/15-autonomous-systems/01-long-horizon-agents/) | 学习 | Python |
| 02 | [STaR、V-STaR、Quiet-STaR：自学推理（Self-Taught Reasoning）](phases/15-autonomous-systems/02-star-family-reasoning/) | 学习 | Python |
| 03 | [AlphaEvolve：进化式编程智能体（Evolutionary Coding Agents）](phases/15-autonomous-systems/03-alphaevolve-evolutionary-coding/) | 学习 | Python |
| 04 | [达尔文哥德尔机（Darwin Gödel Machine）：自修改智能体（Self-Modifying Agents）](phases/15-autonomous-systems/04-darwin-godel-machine/) | 学习 | Python |
| 05 | [AI Scientist v2：研讨会级研究（Workshop-Level Research）](phases/15-autonomous-systems/05-ai-scientist-v2/) | 学习 | Python |
| 06 | [自动化对齐研究（Automated Alignment Research，Anthropic AAR）](phases/15-autonomous-systems/06-automated-alignment-research/) | 学习 | Python |
| 07 | [递归式自我改进（Recursive Self-Improvement）：能力与对齐（Capability vs Alignment）](phases/15-autonomous-systems/07-recursive-self-improvement/) | 学习 | Python |
| 08 | [有界自我改进设计（Bounded Self-Improvement Designs）](phases/15-autonomous-systems/08-bounded-self-improvement/) | 学习 | Python |
| 09 | [自主编程智能体生态（Autonomous Coding Agent Landscape，SWE-bench、CodeAct）](phases/15-autonomous-systems/09-coding-agent-landscape/) | 学习 | Python |
| 10 | [Claude Code 权限模式与自动模式（Permission Modes and Auto Mode）](phases/15-autonomous-systems/10-claude-code-permission-modes/) | 学习 | Python |
| 11 | [浏览器智能体（Browser Agents）与间接提示注入（Indirect Prompt Injection）](phases/15-autonomous-systems/11-browser-agents/) | 学习 | Python |
| 12 | [长运行智能体的持久化执行（Durable Execution）](phases/15-autonomous-systems/12-durable-execution/) | 学习 | Python |
| 13 | [动作预算（Action Budgets）、迭代上限（Iteration Caps）与成本调控器（Cost Governors）](phases/15-autonomous-systems/13-cost-governors/) | 学习 | Python |
| 14 | [终止开关（Kill Switches）、断路器（Circuit Breakers）与金丝雀令牌（Canary Tokens）](phases/15-autonomous-systems/14-kill-switches-canaries/) | 学习 | Python |
| 15 | [人在回路（HITL）：先提议后提交（Propose-Then-Commit）](phases/15-autonomous-systems/15-propose-then-commit/) | 学习 | Python |
| 16 | [检查点（Checkpoints）与回滚（Rollback）](phases/15-autonomous-systems/16-checkpoints-rollback/) | 学习 | Python |
| 17 | [宪法式人工智能（Constitutional AI）与规则覆盖（Rule Overrides）](phases/15-autonomous-systems/17-constitutional-ai/) | 学习 | Python |
| 18 | [Llama Guard 与输入/输出分类（Input/Output Classification）](phases/15-autonomous-systems/18-llama-guard/) | 学习 | Python |
| 19 | [Anthropic 负责任扩展策略 v3.0（Responsible Scaling Policy）](phases/15-autonomous-systems/19-anthropic-rsp/) | 学习 | Python |
| 20 | [OpenAI 准备度框架（Preparedness Framework）与 DeepMind FSF](phases/15-autonomous-systems/20-openai-preparedness-deepmind-fsf/) | 学习 | Python |
| 21 | [METR 时间跨度（Time Horizons）与外部评估（External Evaluation）](phases/15-autonomous-systems/21-metr-external-evaluation/) | 学习 | Python |
| 22 | [CAIS、CAISI 与社会级风险（Societal-Scale Risk）](phases/15-autonomous-systems/22-cais-caisi-societal-risk/) | 学习 | Python |

</details>

<details id="phase-16">
<summary><b>第16阶段 — 多智能体与群体（Multi-Agent & Swarms）</b> &nbsp;<code>25 lessons</code>&nbsp; <em>协调（Coordination）、涌现（emergence）与群体智能（collective intelligence）。</em></summary>
<br/>

| # | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [为什么需要多智能体 (Multi-Agent)](phases/16-multi-agent-and-swarms/01-why-multi-agent/) | 学习 | TypeScript |
| 02 | [FIPA-ACL 的传承与言语行为 (Speech Acts)](phases/16-multi-agent-and-swarms/02-fipa-acl-heritage/) | 学习 | Python |
| 03 | [通信协议 (Communication Protocols)](phases/16-multi-agent-and-swarms/03-communication-protocols/) | 构建 | TypeScript |
| 04 | [多智能体基础原语模型 (Primitive Model)](phases/16-multi-agent-and-swarms/04-primitive-model/) | 学习 | Python |
| 05 | [监督者/编排者-工作者模式 (Supervisor / Orchestrator-Worker Pattern)](phases/16-multi-agent-and-swarms/05-supervisor-orchestrator-pattern/) | 构建 | Python |
| 06 | [分层架构与分解漂移 (Decomposition Drift)](phases/16-multi-agent-and-swarms/06-hierarchical-architecture/) | 学习 | Python |
| 07 | [心智社会 (Society of Mind) 与多智能体辩论](phases/16-multi-agent-and-swarms/07-society-of-mind-debate/) | 构建 | Python |
| 08 | [角色专业化 (Role Specialization) — 规划者/批评者/执行者/验证者](phases/16-multi-agent-and-swarms/08-role-specialization/) | 构建 | Python |
| 09 | [并行群体与网络化架构 (Networked Architectures)](phases/16-multi-agent-and-swarms/09-parallel-swarm-networks/) | 构建 | Python |
| 10 | [群聊与发言者选择 (Speaker Selection)](phases/16-multi-agent-and-swarms/10-group-chat-speaker-selection/) | 构建 | Python |
| 11 | [任务交接与例程 (Routines)（无状态编排 (Stateless Orchestration)）](phases/16-multi-agent-and-swarms/11-handoffs-and-routines/) | 构建 | Python |
| 12 | [A2A — 智能体间协议 (Agent-to-Agent Protocol)](phases/16-multi-agent-and-swarms/12-a2a-protocol/) | 构建 | Python |
| 13 | [共享内存与黑板模式 (Blackboard Patterns)](phases/16-multi-agent-and-swarms/13-shared-memory-blackboard/) | 构建 | Python |
| 14 | [共识机制与拜占庭容错 (Byzantine Fault Tolerance)](phases/16-multi-agent-and-swarms/14-consensus-and-bft/) | 构建 | Python |
| 15 | [投票、自一致性 (Self-Consistency) 与辩论拓扑 (Debate Topology)](phases/16-multi-agent-and-swarms/15-voting-debate-topology/) | 构建 | Python |
| 16 | [协商与博弈 (Negotiation and Bargaining)](phases/16-multi-agent-and-swarms/16-negotiation-bargaining/) | 构建 | Python |
| 17 | [生成式智能体 (Generative Agents) 与涌现仿真 (Emergent Simulation)](phases/16-multi-agent-and-swarms/17-generative-agents-simulation/) | 构建 | Python |
| 18 | [心智理论 (Theory of Mind) 与涌现式协调 (Emergent Coordination)](phases/16-multi-agent-and-swarms/18-theory-of-mind-coordination/) | 构建 | Python |
| 19 | [群体优化 (Swarm Optimization)（PSO、ACO）](phases/16-multi-agent-and-swarms/19-swarm-optimization-pso-aco/) | 构建 | Python |
| 20 | [多智能体强化学习 (MARL) — MADDPG、QMIX、MAPPO](phases/16-multi-agent-and-swarms/20-marl-maddpg-qmix-mappo/) | 学习 | Python |
| 21 | [智能体经济 (Agent Economies)、代币激励 (Token Incentives) 与声誉机制 (Reputation)](phases/16-multi-agent-and-swarms/21-agent-economies/) | 学习 | Python |
| 22 | [生产级扩展 (Production Scaling) — 队列 (Queues)、检查点 (Checkpoints) 与持久性 (Durability)](phases/16-multi-agent-and-swarms/22-production-scaling-queues-checkpoints/) | 构建 | Python |
| 23 | [故障模式 (Failure Modes) — MAST、群体思维 (Groupthink) 与单一文化 (Monoculture)](phases/16-multi-agent-and-swarms/23-failure-modes-mast-groupthink/) | 学习 | Python |
| 24 | [评估与协调基准测试 (Coordination Benchmarks)](phases/16-multi-agent-and-swarms/24-evaluation-coordination-benchmarks/) | 学习 | Python |
| 25 | [案例研究与 2026 年技术前沿 (State of the Art)](phases/16-multi-agent-and-swarms/25-case-studies-2026-sota/) | 学习 | Python |

</details>

<details id="phase-17">
<summary><b>第 17 阶段 — 基础设施与生产环境</b> &nbsp;<code>28 节课</code>&nbsp; <em>将 AI 部署至真实世界。</em></summary>
<br/>

| # | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | 托管大语言模型平台 (Managed LLM Platforms) — Bedrock, Azure OpenAI, Vertex AI | 学习 | Python |
| 02 | 推理平台经济学 (Inference Platform Economics) — Fireworks, Together, Baseten, Modal | 学习 | Python |
| 03 | Kubernetes 上的 GPU 自动扩缩容 (GPU Autoscaling) — Karpenter, KAI Scheduler | 学习 | Python |
| 04 | vLLM 服务内部机制 (vLLM Serving Internals) — 分页注意力机制 (PagedAttention)、连续批处理 (Continuous Batching)、分块预填充 (Chunked Prefill) | 学习 | Python |
| 05 | 生产环境中的 EAGLE-3 投机解码 (Speculative Decoding) | 学习 | Python |
| 06 | 面向前缀密集型负载 (Prefix-Heavy Workloads) 的 SGLang 与 RadixAttention | 学习 | Python |
| 07 | Blackwell 架构上的 TensorRT-LLM 与 FP8/NVFP4 | 学习 | Python |
| 08 | 推理指标 (Inference Metrics) — 首词延迟 (TTFT)、词元间延迟 (TPOT)、迭代间延迟 (ITL)、有效吞吐量 (Goodput)、P99 延迟 | 学习 | Python |
| 09 | 生产环境量化 (Production Quantization) — AWQ, GPTQ, GGUF, FP8, NVFP4 | 学习 | Python |
| 10 | 无服务器大语言模型 (Serverless LLMs) 的冷启动缓解 (Cold Start Mitigation) | 学习 | Python |
| 11 | 多区域大语言模型服务与 KV 缓存局部性 (KV Cache Locality) | 学习 | Python |
| 12 | 边缘推理 (Edge Inference) — ANE, Hexagon, WebGPU, Jetson | 学习 | Python |
| 13 | 大语言模型可观测性技术栈选型 (LLM Observability Stack Selection) | 学习 | Python |
| 14 | 提示词缓存 (Prompt Caching) 与语义缓存经济学 (Semantic Caching Economics) | 学习 | Python |
| 15 | 批量 API (Batch APIs) — 50% 折扣作为行业标准 | 学习 | Python |
| 16 | 模型路由 (Model Routing) 作为成本优化原语 (Primitive) | 学习 | Python |
| 17 | 预填充/解码分离架构 (Disaggregated Prefill/Decode) — NVIDIA Dynamo 与 llm-d | 学习 | Python |
| 18 | 结合 LMCache KV 卸载 (KV Offloading) 的 vLLM 生产栈 | 学习 | Python |
| 19 | AI 网关 (AI Gateways) — LiteLLM, Portkey, Kong, Bifrost | 学习 | Python |
| 20 | 影子发布、金丝雀发布与渐进式部署 (Progressive Deployment) | 学习 | Python |
| 21 | 大语言模型功能 A/B 测试 (A/B Testing) — GrowthBook 与 Statsig | 学习 | Python |
| 22 | 大语言模型 API 负载测试 (Load Testing) — k6, LLMPerf, GenAI-Perf | 实战 | Python |
| 23 | AI 领域的站点可靠性工程 (SRE) — 多智能体事件响应 (Multi-Agent Incident Response) | 学习 | Python |
| 24 | 大语言模型生产环境混沌工程 (Chaos Engineering) | 学习 | Python |
| 25 | 安全 — 密钥管理 (Secrets)、个人身份信息脱敏 (PII Scrubbing)、审计日志 (Audit Logs) | 学习 | Python |
| 26 | 合规 — SOC 2, HIPAA, GDPR、欧盟人工智能法案 (EU AI Act)、ISO 42001 | 学习 | Python |
| 27 | 大语言模型 FinOps (FinOps for LLMs) — 单元经济学 (Unit Economics) 与多租户成本归因 (Multi-Tenant Attribution) | 学习 | Python |
| 28 | 自托管服务选型 (Self-Hosted Serving Selection) — llama.cpp, Ollama, TGI, vLLM, SGLang | 学习 | Python |

</details>

<details id="phase-18">
<summary><b>阶段 18 — 伦理、安全与对齐 (Ethics, Safety & Alignment)</b> &nbsp;<code>30 节课</code>&nbsp; <em>构建造福人类的 AI。这并非可选项。</em></summary>
<br/>

| # | 课程 | 类型 | 语言 |
|:---:|--------|:----:|------|
| 01 | [指令遵循作为对齐信号 (Instruction-Following as Alignment Signal)](phases/18-ethics-safety-alignment/01-instruction-following-alignment-signal/) | 学习 | Python |
| 02 | [奖励破解与古德哈特定律 (Reward Hacking & Goodhart's Law)](phases/18-ethics-safety-alignment/02-reward-hacking-goodhart/) | 学习 | Python |
| 03 | [直接偏好优化系列 (Direct Preference Optimization Family)](phases/18-ethics-safety-alignment/03-direct-preference-optimization-family/) | 学习 | Python |
| 04 | [谄媚行为作为 RLHF 的放大效应 (Sycophancy as RLHF Amplification)](phases/18-ethics-safety-alignment/04-sycophancy-rlhf-amplification/) | 学习 | Python |
| 05 | [宪法式 AI 与基于 AI 反馈的强化学习 (Constitutional AI & RLAIF)](phases/18-ethics-safety-alignment/05-constitutional-ai-rlaif/) | 学习 | Python |
| 06 | [内部优化与欺骗性对齐 (Mesa-Optimization & Deceptive Alignment)](phases/18-ethics-safety-alignment/06-mesa-optimization-deceptive-alignment/) | 学习 | Python |
| 07 | [潜伏代理 — 持续性欺骗 (Sleeper Agents — Persistent Deception)](phases/18-ethics-safety-alignment/07-sleeper-agents-persistent-deception/) | 学习 | Python |
| 08 | [前沿模型中的上下文内谋划 (In-Context Scheming in Frontier Models)](phases/18-ethics-safety-alignment/08-in-context-scheming-frontier-models/) | 学习 | Python |
| 09 | [对齐伪装 (Alignment Faking)](phases/18-ethics-safety-alignment/09-alignment-faking/) | 学习 | Python |
| 10 | [AI 控制 — 颠覆下的安全性保障 (AI Control — Safety Despite Subversion)](phases/18-ethics-safety-alignment/10-ai-control-subversion/) | 学习 | Python |
| 11 | [可扩展监督与弱到强泛化 (Scalable Oversight & Weak-to-Strong)](phases/18-ethics-safety-alignment/11-scalable-oversight-weak-to-strong/) | 学习 | Python |
| 12 | [红队测试：PAIR 与自动化攻击 (Red-Teaming: PAIR & Automated Attacks)](phases/18-ethics-safety-alignment/12-red-teaming-pair-automated-attacks/) | 实践 | Python |
| 13 | [多样本越狱 (Many-Shot Jailbreaking)](phases/18-ethics-safety-alignment/13-many-shot-jailbreaking/) | 学习 | Python |
| 14 | [ASCII 艺术与视觉越狱 (ASCII Art & Visual Jailbreaks)](phases/18-ethics-safety-alignment/14-ascii-art-visual-jailbreaks/) | 实践 | Python |
| 15 | [间接提示词注入 (Indirect Prompt Injection)](phases/18-ethics-safety-alignment/15-indirect-prompt-injection/) | 实践 | Python |
| 16 | [红队工具：Garak、Llama Guard 与 PyRIT (Red-Team Tooling: Garak, Llama Guard, PyRIT)](phases/18-ethics-safety-alignment/16-red-team-tooling-garak-llamaguard-pyrit/) | 实践 | Python |
| 17 | [WMDP 与双重用途能力评估 (WMDP & Dual-Use Capability Evaluation)](phases/18-ethics-safety-alignment/17-wmdp-dual-use-evaluation/) | 学习 | Python |
| 18 | [前沿安全框架 — RSP、PF 与 FSF (Frontier Safety Frameworks — RSP, PF, FSF)](phases/18-ethics-safety-alignment/18-frontier-safety-frameworks-rsp-pf-fsf/) | 学习 | — |
| 19 | [模型福利研究 (Model Welfare Research)](phases/18-ethics-safety-alignment/19-model-welfare-research/) | 学习 | Python |
| 20 | [偏见与表征伤害 (Bias & Representational Harm)](phases/18-ethics-safety-alignment/20-bias-representational-harm/) | 实践 | Python |
| 21 | [公平性准则：群体、个体与反事实 (Fairness Criteria: Group, Individual, Counterfactual)](phases/18-ethics-safety-alignment/21-fairness-criteria-group-individual-counterfactual/) | 学习 | Python |
| 22 | [大语言模型的差分隐私 (Differential Privacy for LLMs)](phases/18-ethics-safety-alignment/22-differential-privacy-for-llms/) | 实践 | Python |
| 23 | [水印技术：SynthID、Stable Signature 与 C2PA (Watermarking: SynthID, Stable Signature, C2PA)](phases/18-ethics-safety-alignment/23-watermarking-synthid-stable-signature-c2pa/) | 实践 | Python |
| 24 | [监管框架：欧盟、美国、英国与韩国 (Regulatory Frameworks: EU, US, UK, Korea)](phases/18-ethics-safety-alignment/24-regulatory-frameworks-eu-us-uk-korea/) | 学习 | — |
| 25 | [EchoLeak 与 AI 漏洞通用编号 (EchoLeak & CVEs for AI)](phases/18-ethics-safety-alignment/25-echoleak-cves-for-ai/) | 学习 | Python |
| 26 | [模型、系统与数据集卡片 (Model, System & Dataset Cards)](phases/18-ethics-safety-alignment/26-model-system-dataset-cards/) | 实践 | Python |
| 27 | [数据溯源与训练数据治理 (Data Provenance & Training-Data Governance)](phases/18-ethics-safety-alignment/27-data-provenance-training-governance/) | 学习 | Python |
| 28 | [对齐研究生态：MATS、Redwood、Apollo 与 METR (Alignment Research Ecosystem: MATS, Redwood, Apollo, METR)](phases/18-ethics-safety-alignment/28-alignment-research-ecosystem/) | 学习 | — |
| 29 | [内容审核系统：OpenAI、Perspective 与 Llama Guard (Moderation Systems: OpenAI, Perspective, Llama Guard)](phases/18-ethics-safety-alignment/29-moderation-systems-openai-perspective-llamaguard/) | 实践 | Python |
| 30 | [双重用途风险：网络、生物、化学与核领域 (Dual-Use Risk: Cyber, Bio, Chem, Nuclear)](phases/18-ethics-safety-alignment/30-dual-use-risk-cyber-bio-chem-nuclear/) | 学习 | — |

</details>

<details id="phase-19">
<summary><b>第19阶段 — 综合实战项目 (Capstone Projects)</b> &nbsp;<code>17个项目</code>&nbsp; <em>2026年端到端可交付产品，每个项目耗时20-40小时。</em></summary>
<br/>

| # | 项目 | 涉及阶段 | 语言 |
|:---:|---------|----------|------|
| 01 | [终端原生编程智能体 (Terminal-Native Coding Agent)](phases/19-capstone-projects/01-terminal-native-coding-agent/) | P0 P5 P7 P10 P11 P13 P14 P15 P17 P18 | TypeScript, Python |
| 02 | [代码库检索增强生成（跨仓库语义搜索）(RAG over Codebase)](phases/19-capstone-projects/02-rag-over-codebase/) | P5 P7 P11 P13 P17 | Python, TypeScript |
| 03 | [实时语音助手（ASR → LLM → TTS）](phases/19-capstone-projects/03-realtime-voice-assistant/) | P6 P7 P11 P13 P14 P17 | Python, TypeScript |
| 04 | [多模态文档问答（视觉优先）(Multimodal Document QA)](phases/19-capstone-projects/04-multimodal-document-qa/) | P4 P5 P7 P11 P12 P17 | Python, TypeScript |
| 05 | [自主研究智能体（AI科学家类）(Autonomous Research Agent)](phases/19-capstone-projects/05-autonomous-research-agent/) | P0 P2 P3 P7 P10 P14 P15 P16 P18 | Python |
| 06 | [Kubernetes DevOps 故障排查智能体](phases/19-capstone-projects/06-devops-troubleshooting-agent/) | P11 P13 P14 P15 P17 P18 | Python, TypeScript |
| 07 | [端到端微调流水线 (End-to-End Fine-Tuning Pipeline)](phases/19-capstone-projects/07-end-to-end-fine-tuning-pipeline/) | P2 P3 P7 P10 P11 P17 P18 | Python |
| 08 | [生产级 RAG 聊天机器人（受监管垂直领域）](phases/19-capstone-projects/08-production-rag-chatbot/) | P5 P7 P11 P12 P17 P18 | Python, TypeScript |
| 09 | [代码迁移智能体（仓库级升级）](phases/19-capstone-projects/09-code-migration-agent/) | P5 P7 P11 P13 P14 P15 P17 | Python, TypeScript |
| 10 | [多智能体软件工程团队](phases/19-capstone-projects/10-multi-agent-software-team/) | P11 P13 P14 P15 P16 P17 | Python, TypeScript |
| 11 | [大语言模型可观测性与评估仪表盘 (LLM Observability & Eval Dashboard)](phases/19-capstone-projects/11-llm-observability-dashboard/) | P11 P13 P17 P18 | TypeScript, Python |
| 12 | [视频理解流水线（场景 → 问答）](phases/19-capstone-projects/12-video-understanding-pipeline/) | P4 P6 P7 P11 P12 P17 | Python, TypeScript |
| 13 | [带注册表与治理的 MCP 服务器](phases/19-capstone-projects/13-mcp-server-with-registry/) | P11 P13 P14 P17 P18 | Python, TypeScript |
| 14 | [投机解码推理服务器 (Speculative-Decoding Inference Server)](phases/19-capstone-projects/14-speculative-decoding-server/) | P3 P7 P10 P17 | Python |
| 15 | [宪法安全护栏 + 红队演练场 (Constitutional Safety Harness + Red-Team Range)](phases/19-capstone-projects/15-constitutional-safety-harness/) | P10 P11 P13 P14 P18 | Python |
| 16 | [GitHub Issue 转 PR 自主智能体](phases/19-capstone-projects/16-github-issue-to-pr-agent/) | P11 P13 P14 P15 P17 | Python, TypeScript |
| 17 | [个人 AI 导师（自适应、多模态）](phases/19-capstone-projects/17-personal-ai-tutor/) | P5 P6 P11 P12 P14 P17 P18 | Python, TypeScript |

</details>

░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒

## 工具包 (Toolkit)

每节课都会产出可复用的工件 (artifact)。课程结束时，你将获得：

outputs/
├── prompts/      prompt templates for every AI task
└── skills/       SKILL.md files for AI coding agents

使用 [SkillKit](https://github.com/rohitg00/skillkit) 进行安装。将它们接入 Claude、Cursor、Codex、OpenClaw、Hermes 或任何兼容 MCP 的智能体 (agent)。这是真正的生产工具，而非课后作业。

### 将课程中的所有技能 (skill) 安装到你的智能体中

该仓库在 `phases/**/outputs/` 目录下提供了 378 个技能 (skill) 和 99 个提示词 (prompt)。
`scripts/install_skills.py` 会遍历所有工件，解析 YAML frontmatter，并将匹配的文件复制到目标目录中，目录结构符合你的智能体预期。

python3 scripts/install_skills.py ~/.claude/skills                 # every skill, SkillKit layout
python3 scripts/install_skills.py ./out --type all                 # skills + prompts + agents
python3 scripts/install_skills.py ./out --phase 14                 # one phase only
python3 scripts/install_skills.py ./out --tag rag                  # filter by tag
python3 scripts/install_skills.py ./out --layout flat              # flat files instead of SkillKit
python3 scripts/install_skills.py ./out --dry-run                  # preview without writing
python3 scripts/install_skills.py ./out --force                    # overwrite existing files

默认情况下，该脚本会拒绝覆盖已存在的目标路径，并在列出所有冲突路径后以退出码 1 终止。请使用 `--dry-run` 预览冲突，或使用 `--force` 强制覆盖。每次非 `--dry-run` 的运行都会在目标目录中生成一个 `manifest.json` 文件，其中包含按类型和阶段分组的完整清单。请选择你的智能体（Agent）能够读取的布局格式：

| `--layout`  | 写入路径 |
|---|---|
| `skillkit`  | `<target>/<name>/SKILL.md`（适用于 Claude / Cursor / SkillKit） |
| `by-phase`  | `<target>/phase-NN/<name>.md` |
| `flat`      | `<target>/<name>.md` |

### 将智能体工作台（Agent Workbench）集成到你自己的仓库中

第 14 阶段的综合实践项目（Capstone）提供了一个可复用的智能体工作台工具包（包含 `AGENTS.md`、数据模式（Schema）、初始化/验证/交接脚本）。可通过以下命令将其脚手架化（Scaffold）到任意仓库中：

python3 scripts/scaffold_workbench.py path/to/your-repo            # full pack + seeds
python3 scripts/scaffold_workbench.py path/to/your-repo --minimal  # skip docs/
python3 scripts/scaffold_workbench.py path/to/your-repo --dry-run  # preview only
python3 scripts/scaffold_workbench.py path/to/your-repo --force    # overwrite

执行后，你将获得已配置好的七个工作台面板（Workbench Surfaces）、一个初始的 `task_board.json` 文件，以及一个版本为 `schema_version: 1` 的全新 `agent_state.json` 文件。接下来：编辑任务、修改 `AGENTS.md`、运行 `scripts/init_agent.py`，然后将交互契约（Contract）交付给你的智能体。该工具包的源码位于 `phases/14-agent-engineering/42-agent-workbench-capstone/outputs/agent-workbench-pack/`。

### 以 JSON 格式浏览完整课程

`scripts/build_catalog.py` 会遍历磁盘上的每个阶段、每节课以及所有产出物（Artifact），并在仓库根目录生成 `catalog.json`。单文件即可完整呈现课程数据。

python3 scripts/build_catalog.py               # writes <repo>/catalog.json
python3 scripts/build_catalog.py --stdout      # to stdout, do not touch repo
python3 scripts/build_catalog.py --out path/to/file.json

该目录（Catalog）直接派生自文件系统，而非 README 文件，因此统计数量始终与磁盘上的实际内容保持一致。你可以将其用于网站构建、下游工具链，或验证 README 中的统计数据是否发生偏移。其数据结构（Schema）已在脚本顶部进行了文档说明。

一个 GitHub Action（`.github/workflows/curriculum.yml`）会在每次拉取请求（PR）时重新构建 `catalog.json`，如果已提交的文件过期，构建将会失败。在编辑任何课程后，请运行 `python3 scripts/build_catalog.py` 并提交结果，否则持续集成（CI）将拒绝该 PR。同一工作流还会以仅警告模式（Warn-only Mode）运行 `audit_lessons.py`（因此现有的数据偏移不会阻塞贡献者）。

### 对每节课的 Python 代码进行冒烟测试（Smoke Test）

`scripts/lesson_run.py` 会对每节课 `code/` 目录下的所有 `.py` 文件进行字节码编译。默认模式仅进行语法检查（Syntax Check）——不执行代码、无需 API 密钥、也不依赖重型机器学习（ML）库。它能捕获贡献者最常引入的回归问题（Regression）（如缩进错误、损坏的 f-string、误编辑等）。

python3 scripts/lesson_run.py                  # syntax-check the whole curriculum
python3 scripts/lesson_run.py --phase 14       # one phase only
python3 scripts/lesson_run.py --json           # JSON report on stdout
python3 scripts/lesson_run.py --strict         # exit 1 if any lesson fails
python3 scripts/lesson_run.py --execute        # actually run, 10s timeout per lesson


`--execute` 参数会以 10 秒超时限制运行每个课程的 `code/main.py`（或首个 `.py` 文件）。若入口文件以 `# requires: pkg1, pkg2` 注释开头并列出了非标准库 (Stdlib) 依赖项，则该课程将被跳过，并标记原因为 `needs <deps>`。该脚本为可选启用 (Opt-in)，未集成至持续集成 (CI) 流程中。

仅使用标准库，需 Python 3.10 及以上版本。可通过设置 `LINK_CHECK_SKIP=domain1,domain2` 环境变量来覆盖默认跳过列表（`twitter.com`、`x.com`、`linkedin.com`、`instagram.com`、`medium.com` —— 这些域名会严格拦截自动化的 HEAD/GET 请求）。

## 从哪里开始

| 背景 | 起点 | 预计耗时 |
|---|---|---|
| 编程与人工智能 (AI) 零基础 | Phase 0 — 环境配置 | ~306 小时 |
| 熟悉 Python，机器学习 (ML) 零基础 | Phase 1 — 数学基础 | ~270 小时 |
| 熟悉机器学习，深度学习 (Deep Learning) 零基础 | Phase 3 — 深度学习核心 | ~200 小时 |
| 熟悉深度学习，想学习大语言模型 (LLMs) 与智能体 (Agents) | Phase 10 — 从零构建大语言模型 | ~100 小时 |
| 资深工程师，仅关注智能体工程 | Phase 14 — 智能体工程 | ~60 小时 |

░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒

## 为何当下至关重要

<table>
<tr>
<th align="left" width="50%"><sub>FIG_003 · A</sub><br/><b>行业风向</b></th>
<th align="left" width="50%"><sub>FIG_003 · B</sub><br/><b>涵盖的基础论文</b></th>
</tr>
<tr>
<td valign="top">

> *“当下最热门的新编程语言是英语。”*<br/>
> — **Andrej Karpathy** ([推文](https://x.com/karpathy/status/1617979122625712128))

> *“软件工程正在我们眼前被重塑。”*<br/>
> — **Boris Cherny**，Claude Code 创始人

> *“模型会持续进化。真正能产生复利效应的技能是**知道该构建什么**。”*<br/>
> — 2026 年行业共识

</td>
<td valign="top">

- *Attention Is All You Need* — Vaswani 等人，2017 → [Phase 7](#phase-7)
- *Language Models are Few-Shot Learners* (GPT-3) → [Phase 10](#phase-10)
- *Denoising Diffusion Probabilistic Models* → [Phase 8](#phase-8)
- *InstructGPT / RLHF* → [Phase 10](#phase-10)
- *Direct Preference Optimization* → [Phase 10](#phase-10)
- *Chain-of-Thought Prompting* → [Phase 11](#phase-11)
- *ReAct: Reasoning + Acting in LLMs* → [Phase 14](#phase-14)
- *Model Context Protocol* — Anthropic → [Phase 13](#phase-13)

</td>
</tr>
</table>

░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒

## 贡献指南

| 目标 | 阅读文档 |
|---|---|
| 贡献课程或修复问题 | [CONTRIBUTING.md](CONTRIBUTING.md) |
| 为团队或学校创建分支 (Fork) | [FORKING.md](FORKING.md) |
| 课程模板 | [LESSON_TEMPLATE.md](LESSON_TEMPLATE.md) |
| 跟踪进度 | [ROADMAP.md](ROADMAP.md) |
| 术语表 | [glossary/terms.md](glossary/terms.md) |
| 行为准则 | [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) |

提交课程前，请运行不变量检查 (Invariant Check)：

python3 scripts/audit_lessons.py           # full curriculum
python3 scripts/audit_lessons.py --phase 14  # single phase
python3 scripts/audit_lessons.py --json    # CI-friendly output

若任何规则校验失败，退出码将为非零值。规则（L001–L010）用于验证目录结构、`docs/en.md` 文件是否存在及是否包含一级标题 (H1)、`code/` 目录非空、`quiz.json` 的数据结构 (Schema)（会拒绝导致 issue #102 的旧版 `q/choices/answer` 键），以及课程文档内的相对链接。

░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒

## 赞助本项目

免费开源，采用 MIT 许可证，共 435 节课程。本课程的维护完全依赖赞助。仅接受资金赞助。

**覆盖范围（数据更新于 2026-05-14）：** 月访问量 55,593 · 页面浏览量 90,709 · 7.5K Stars · Twitter/X 为首要获客渠道。

| 赞助等级 | 美元/月 | 包含权益 |
|------|------|---|
| 支持者 | $25 | 名字列入 `BACKERS.md` |
| 青铜 | $250 | 在 README 赞助区块中显示纯文本行 + 项目启动日推文鸣谢 |
| 白银 | $750 | 在 README 中显示小型 Logo + 在 API 课程中列为支持的服务提供商之一 |
| 黄金 | $2,000 | 在 README 中显示中型 Logo + 专属赞助页面 + 每季度在 X / LinkedIn 联合展示 |
| 铂金 | $5,000 | 首屏显著位置展示大型 Logo + 一节专属集成教程，限 1 家合作伙伴 |

完整价目表、硬性规则、定价基准及触达数据请参阅：[SPONSORS.md](SPONSORS.md)。
请通过 [GitHub Sponsors](https://github.com/sponsors/rohitg00) 注册赞助。

░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒░░░▒▒▒

## Star 历史

<a href="https://star-history.com/#rohitg00/ai-engineering-from-scratch&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=rohitg00/ai-engineering-from-scratch&type=Date&theme=dark">
    <img alt="Star 历史" src="https://api.star-history.com/svg?repos=rohitg00/ai-engineering-from-scratch&type=Date" width="100%">
  </picture>
</a>

如果本手册对您有所帮助，请为仓库点亮 Star。您的支持是项目持续发展的动力。

## 许可证

采用 MIT 许可证。您可以随意使用——随意 Fork、用于教学、商业销售或集成发布。建议保留署名，但非强制要求。

由 [Rohit Ghumare](https://github.com/rohitg00) 与社区共同维护。

<sub>
  <a href="https://x.com/ghumare64">@ghumare64</a> &nbsp;·&nbsp;
  <a href="https://aiengineeringfromscratch.com">aiengineeringfromscratch.com</a> &nbsp;·&nbsp;
  <a href="https://github.com/rohitg00/ai-engineering-from-scratch/issues/new/choose">反馈 / 建议</a>
</sub>