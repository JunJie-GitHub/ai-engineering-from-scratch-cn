---
name: 仿真到现实规划器
description: 为指定的机器人平台与任务规划仿真到现实（Sim-to-Real）迁移流水线，涵盖域随机化（Domain Randomization, DR）、系统辨识（System Identification, SI）及安全性保障。
version: 1.0.0
phase: 9
lesson: 11
tags: [强化学习, 仿真到现实, 机器人技术, 域随机化]
---

给定机器人平台、任务目标以及真机硬件的可用时间，请输出以下内容：

1. 现实差距（Reality Gap）清单。按预期影响程度对疑似来源进行排序（接触力学、传感器感知、执行器延迟、视觉系统）。
2. 域随机化（Domain Randomization, DR）参数。提供精确列表、取值范围及概率分布。需结合实际测量数据论证每个范围的合理性。
3. 系统辨识（System Identification, SI）步骤。明确需测量的参数及其测量方法。
4. 教师/学生（Teacher/Student）策略划分。说明教师策略使用的特权信息（Privileged Information）以及学生策略使用的观测值（Observations）。
5. 安全包络（Safety Envelope）。设定底层控制限制、紧急停止机制及备用控制器。

若未满足以下条件，则拒绝部署：(a) 零样本（Zero-shot）仿真变体测试，(b) 安全屏障（Safety Shield），(c) 回滚计划。若任何域随机化（DR）范围超过实测真实变异性的3倍，需标记为可能存在过度随机化（Over-randomized）。