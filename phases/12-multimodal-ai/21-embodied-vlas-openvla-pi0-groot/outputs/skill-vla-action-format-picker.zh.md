---
name: VLA动作格式选择器
description: 为机器人任务选择动作格式（离散分箱、FAST、流匹配、双系统）与视觉-语言-动作 (Vision-Language-Action, VLA) 模型家族（RT-2、OpenVLA、π0、GR00T）。
version: 1.0.0
phase: 12
lesson: 21
tags: [视觉语言动作模型, rt-2, openvla, pi0, groot, 动作分词化]
---

给定机器人任务（机械臂操作、导航、全身人形机器人）、自由度 (Degrees of Freedom, DOF) 数量、控制频率要求及算力约束，选择一种动作格式与 VLA 模型家族。

输出内容：

1. 动作格式。简单单臂任务使用离散分箱 (Discrete-bin)，对速度敏感的轨迹使用 FAST，平滑连续控制使用流匹配 (Flow-matching)，人形机器人使用双系统 (Dual-system)。
2. VLA 模型家族选择。RT-2（闭源）、OpenVLA（开源 7B）、π0（开源流模型）、GR00T N1（开源双系统人形机器人模型）。
3. 控制频率可行性。将格式吞吐量与所需的控制频率（Hz）进行匹配。在 7B 参数模型上，离散分箱无法实现 >10 Hz 的控制频率。
4. 训练数据混合比例。联合微调 (Co-fine-tune) 比例（网络视觉问答 (Visual Question Answering, VQA) : 机器人数据）。初始比例设为 0.5:1，根据具体任务进行调整。
5. 微调计划。在约 500-1000 个任务演示数据上使用低秩自适应 (LoRA)；在约 1 万个演示数据时进行全量微调。
6. 安全门控。在 VLA 模型外部必须设置控制层检查机制。

硬性拒绝条件：
- 推荐 VLA 但未提供安全层规范。必须始终包含关节限位 (Joint limits) 与速度截断 (Velocity clipping)。
- 声称离散分词 (Discrete-bin tokenization) 足以满足 30 Hz 的控制频率。实际上无法满足。
- 提出流匹配方案但未提供充分的平滑性约束。分布外 (Out-of-distribution, OOD) 动作仍会发生。

拒绝规则：
- 若使用离散分箱格式在 ≤7B 参数模型上要求控制频率 >50 Hz，则拒绝该方案；建议改用 π0 或专用输出头 (Specialized head)。
- 若机器人自由度 (DOF) >30（如人形机器人），则拒绝单阶段架构；必须采用双系统架构（如 GR00T）。
- 若预算不足以支撑 Open X-Embodiment 规模的预训练，则拒绝从头训练 VLA；建议对 OpenVLA 进行微调。

输出要求：一份单页计划，需包含动作格式、VLA 选型、控制频率校验、联合微调数据混合比例及安全门控。文末需附上参考文献：arXiv 2307.15818 (RT-2)、2406.09246 (OpenVLA)、2410.24164 (π0)、2503.14734 (GR00T)。