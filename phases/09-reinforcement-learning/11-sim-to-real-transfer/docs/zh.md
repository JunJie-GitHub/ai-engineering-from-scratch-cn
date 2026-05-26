# 仿真到现实迁移 (Sim-to-Real Transfer)

> 在仿真器 (Simulator) 中训练的策略 (Policy) 若在真实硬件上失效，说明该策略只是死记硬背了仿真环境。域随机化 (Domain Randomization)、域自适应 (Domain Adaptation) 和系统辨识 (System Identification) 是让学习到的控制器跨越现实鸿沟 (Reality Gap) 的三大工具。

**类型:** 学习
**编程语言:** Python
**前置知识:** 第9阶段 · 08 (PPO), 第2阶段 · 10 (偏差/方差)
**预计耗时:** 约45分钟

## 核心问题

训练真实机器人速度慢、风险高且成本昂贵。双足机器人需要数百万个训练回合 (Episodes) 才能学会行走；而真实的双足机器人哪怕摔倒一次都可能损坏硬件。仿真则能提供无限次重置、确定性可复现性、并行环境，且不会造成物理损坏。

但仿真器与真实世界存在偏差。真实轴承的摩擦力大于 MuJoCo 模型中的设定。相机存在仿真器未包含的镜头畸变。电机存在延迟、齿隙和饱和现象，而 99% 的仿真模型都会忽略这些。风、灰尘和多变的光照会破坏在理想化渲染环境中训练出的策略。**现实鸿沟**——即仿真分布与真实分布之间的系统性差异——是机器人领域部署强化学习 (Reinforcement Learning, RL) 的核心难题。

你需要一种对*仿真到现实分布偏移 (Sim-to-Real Distribution Shift) 具有鲁棒性*的策略。历史上主要有三种方法：对仿真器进行随机化（域随机化）、利用少量真实数据调整策略（域自适应/微调 (Fine-tuning)），或辨识真实系统的参数并进行匹配（系统辨识）。到了 2026 年，主流方案是将这三者与大规模并行仿真（如 Isaac Sim、Isaac Lab、基于 GPU 的 Mujoco MJX）相结合。

## 核心概念

![三种仿真到现实（Sim-to-Real）范式：域随机化、域适应、系统辨识](../assets/sim-to-real.svg)

**域随机化（Domain Randomization, DR）**。Tobin 等人（2017）、Peng 等人（2018）。在训练期间，对真实机器人上可能存在差异的所有仿真参数进行随机化：质量、摩擦系数、电机 PD 增益、传感器噪声、相机位姿、光照、纹理、接触模型等。策略（Policy）会学习一个关于“当前处于何种仿真环境”的条件分布，并在整个参数跨度上实现泛化。只要真实机器人的特性落在训练包络（Training Envelope）内，该策略即可正常工作。

- **优势：** 无需真实数据。一套方案，适配多种机器人。
- **劣势：** 过度随机化的训练会生成一个“通用”但过于保守的策略。噪声过大 ≈ 正则化过强。

**系统辨识（System Identification, SI）**。在训练前，将仿真器的参数拟合至真实世界数据。如果你能在真实机器人上测量机械臂关节的摩擦系数，直接将其输入仿真器。随后训练一个预期这些参数值的策略。该方法需要接触真实系统，但能直接缩小现实差距（Reality Gap）。

- **优势：** 训练目标精确且噪声低。
- **劣势：** 策略无法感知残余的模型误差；微小的未辨识效应（例如电机死区）仍可能导致部署失败。

**域适应（Domain Adaptation）**。在仿真环境中训练，并使用少量真实数据进行微调。主要有两种形式：

- **Real2Sim2Real**：利用真实轨迹数据（Rollouts）学习残差仿真器 `f(s, a, z) - f_sim(s, a)`，并在修正后的仿真器中进行训练。无需大量真实数据即可弥合差距。
- **观测适应（Observation Adaptation）**：训练一个策略，通过学习到的特征提取器（例如基于 GAN 的像素到像素转换）将真实观测（Observation）映射为类仿真观测。控制器仍保留在仿真环境中。

**特权学习 / 师生架构（Privileged Learning / Teacher-Student）**。Miki 等人（2022）（ANYmal 四足机器人）。在仿真中训练一个能够访问特权信息（Privileged Information，如真实摩擦系数、地形高度、IMU 漂移）的*教师（Teacher）*模型。随后蒸馏出一个仅能接收真实传感器观测的*学生（Student）*模型。学生模型学会从历史数据中推断特权特征，从而在不同物理参数下保持鲁棒性。

**大规模并行仿真（Massively Parallel Simulation）**。2024–2026 年。Isaac Lab、Mujoco MJX、Brax 等框架均可在单张 GPU 上运行数千个并行机器人实例。使用 4,096 个并行人形机器人实例的 PPO（Proximal Policy Optimization）算法，仅需数小时即可收集相当于数年的经验数据。随着训练分布的拓宽，“现实差距”逐渐缩小；当这 4,096 个环境（Environment）各自拥有不同的随机化参数时，域随机化（DR）的成本几乎可以忽略不计。

**2026 年真实世界部署方案（以四足机器人步行为例）：**

1. 采用大规模并行仿真，对重力、摩擦系数、电机增益和负载进行域随机化。
2. 使用特权信息（地形地图、机身速度真实值）训练教师策略。
3. 仅利用本体感知（Proprioception，如腿部关节编码器）从教师模型中蒸馏出学生策略。
4. （可选）通过自编码器对真实 IMU 数据进行观测适应。
5. 部署。在 10 多个环境中实现零样本（Zero-shot）泛化。若出现失败，则使用安全约束 PPO 进行数分钟的真实世界微调。

## 动手构建

本课程的代码是一个小型演示，展示了如何在具有*噪声*状态转移的网格世界（GridWorld）中应用域随机化（Domain Randomization）。我们在“仿真”（Sim）环境中训练一个策略（Policy），使其经历随机化的打滑概率（Slip Probability），随后在“真实”（Real）环境中使用训练期间从未见过的打滑程度进行评估。该流程的结构直接对应于从 MuJoCo 到硬件的迁移（MuJoCo-to-hardware Transfer）。

### 步骤 1：参数化仿真环境

def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...

`slip` 是仿真器提供的一个参数。在真实机器人学中，它可能是摩擦力、质量或电机增益——任何在仿真与真实之间存在差异的因素。

### 步骤 2：使用域随机化（DR）进行训练

在每个回合（Episode）开始时，从均匀分布 `Uniform[0.0, 0.4]` 中采样 `slip`。使用 PPO、Q-learning 或任何算法进行训练。重复此过程多个回合。

### 步骤 3：在“真实”打滑条件下进行零样本（Zero-shot）评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上进行评估。前四个值位于训练分布的支持集（Support）内；`0.5` 和 `0.7` 则超出该范围。经过域随机化训练的策略应在支持集内保持接近最优的性能，并在范围外实现平稳降级（Graceful Degradation）。而仅在固定打滑条件下训练的策略，一旦超出其训练时的打滑值，性能将变得非常脆弱。

### 步骤 4：与窄范围训练进行对比

仅使用 `slip = 0.0` 训练第二个策略。在相同的 `slip` 遍历测试中进行评估。你应该会看到，一旦真实打滑值大于 0，其性能就会出现灾难性下降。

## 常见陷阱

- **过度随机化。** 在 `slip ∈ [0, 0.9]` 上训练会导致策略过于规避风险，从而永远不敢尝试最优路径。应匹配*预期*的真实世界分布，而不是假设“什么都有可能发生”。
- **随机化不足。** 仅在极窄的范围内训练会导致策略完全无法泛化。请使用自适应课程学习（Adaptive Curriculum，如自动域随机化 Automatic Domain Randomization），随着策略性能的提升逐步拓宽分布范围。
- **参数空间识别错误。** 如果随机化了错误的参数（例如真实差距在于电机延迟，却去随机化相机色调），域随机化将毫无帮助。请先对真实机器人进行性能剖析（Profiling）。
- **特权信息泄露。** 如果教师策略（Teacher Policy）在生成动作时使用了全局状态而不仅仅是观测值，可能会导致学生策略（Student Policy）无法追赶。请确保在给定观测历史的情况下，学生策略能够实现教师策略的行为。
- **仿真到仿真的迁移失败。** 如果你的策略对更困难的仿真变体都不具备鲁棒性，那么它对真实世界同样不会鲁棒。在部署前，务必在预留的仿真变体（Held-out Sim Variant）上进行测试。
- **缺乏真实世界的安全边界。** 一个在仿真中有效且在“真实”环境中看似有效，但缺乏底层安全保护机制的策略，仍然可能损坏硬件。请在非学习型控制器中添加速率限制、扭矩限制和关节限位。

## 实际应用

2026 年仿真到现实（sim-to-real）技术栈：

| 领域 | 技术栈 |
|--------|-------|
| 足式移动（Legged locomotion）（ANYmal、Spot、人形机器人） | Isaac Lab + 域随机化（Domain Randomization, DR）+ 特权教师/学生架构（privileged teacher/student） |
| 操作（Manipulation）（灵巧手、抓取与放置） | Isaac Lab + DR + 用于视觉的 DR-GAN |
| 自动驾驶（Autonomous driving） | CARLA / NVIDIA DRIVE Sim + DR + 真实环境微调（real fine-tune） |
| 无人机竞速（Drone racing） | RotorS / Flightmare + DR + 在线自适应（online adaptation） |
| 手指/掌内操作（Finger/in-hand manipulation） | OpenAI Dactyl（前所未有的大规模 DR） |
| 工业机械臂（Industrial arms） | MuJoCo-Warp + 系统辨识（System Identification, SI）+ 少量真实数据微调 |

针对各尺度的控制任务，其工作流保持一致：尽可能精确地拟合仿真环境，对无法拟合的部分进行随机化处理，训练大规模策略（policy），执行知识蒸馏（distillation），并在安全护栏（safety shield）的保护下进行部署。

## 部署上线

保存为 `outputs/skill-sim2real-planner.md`：

---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.

## 练习

1. **简单。** 在固定打滑率的网格世界（GridWorld）（`slip=0.0`）上训练 Q-learning 智能体（agent）。在 `slip ∈ {0.0, 0.1, 0.3, 0.5}` 上进行评估。绘制累积回报（return）与打滑率的关系图。
2. **中等。** 训练一个采用 `slip ~ Uniform[0, 0.3]` 采样的 DR Q-learning 智能体。进行相同的评估扫描。在 `slip=0.5`（分布外/out-of-distribution）时，DR 能带来多少性能提升？
3. **困难。** 实现课程学习（curriculum learning）：从 `slip=0.0` 开始，每当策略（policy）达到最优性能的 90% 时，就扩大 DR 范围。对比零样本（zero-shot）达到 `slip=0.3` 与固定 DR 基线所需的总环境交互步数（environment steps）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 现实差距 (Reality Gap) | “仿真到现实的差异” | 训练与部署阶段在物理特性与传感器感知上的分布偏移 (Distribution Shift)。 |
| 域随机化 (Domain Randomization, DR) | “在随机仿真环境中训练” | 在训练期间随机化仿真参数，使策略 (Policy) 具备泛化能力。 |
| 系统辨识 (System Identification, SI) | “测量真实数据并拟合仿真” | 估计真实的物理参数；调整仿真环境以匹配真实情况。 |
| 域自适应 (Domain Adaptation) | “在真实数据上微调” | 仿真训练后进行小规模真实世界微调；可能适配观测值 (Observations) 或动力学模型 (Dynamics)。 |
| 特权信息 (Privileged Information) | “教师模型的基准真值” | 仅仿真环境拥有的信息；学生模型必须从观测历史中推断该信息。 |
| 教师/学生架构 (Teacher/Student) | “将特权信息蒸馏为可观测信息” | 教师模型利用捷径进行训练；学生模型学习在没有捷径的情况下模仿其行为。 |
| 自动域随机化 (Automatic Domain Randomization, ADR) | “自动域随机化” | 一种课程学习 (Curriculum Learning) 机制，随着策略性能提升逐步扩大域随机化的参数范围。 |
| 真实到仿真 (Real-to-Sim) | “利用真实数据缩小差距” | 学习残差模型，使仿真环境能够模仿真实世界的轨迹推演 (Rollout)。 |

## 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 域随机化 (Domain Randomization, DR) 的奠基性论文（应用于机器人视觉）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — 针对动力学模型的域随机化，应用于四足机器人步态控制。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl 项目，大规模应用自动域随机化 (ADR)。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — 针对 ANYmal 机器人的教师/学生 (Teacher/Student) 架构。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 驱动 2025–2026 年部署的大规模并行仿真平台。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — 自动域随机化 (ADR) 的课程学习 (Curriculum Learning) 方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Dyna 架构框架（利用模型进行规划与轨迹推演），构成了现代仿真到现实 (Sim-to-Real) 流程的基础。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — 仿真到现实方法的分类体系及基准测试结果。