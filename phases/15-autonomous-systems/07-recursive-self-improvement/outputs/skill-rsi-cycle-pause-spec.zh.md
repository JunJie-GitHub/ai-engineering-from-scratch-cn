---
name: rsi-cycle-pause-spec
description: 指定 RSI（递归自我改进）流水线在下一周期开始前必须暂停并等待人工审查的条件。
version: 1.0.0
phase: 15
lesson: 7
tags: [rsi, 自我改进, 对齐, 暂停阈值, rsp]
---

针对拟定的递归自我改进（Recursive Self-Improvement, RSI）流水线，需生成一份暂停规范（Pause Specification）：明确列出在下一周期开始前，必须中止循环以供人工检查的具体条件。缺乏暂停规范的流水线不具备运行条件。

需产出以下内容：

1. **周期级阈值（Cycle-level thresholds）。** 针对每个可量化维度（能力评分、对齐评分、预算、轨迹长度、资源使用量），定义一个数值阈值，一旦跨越该阈值即暂停循环。阈值必须在循环启动前设定并记录在案。
2. **周期间变化量（Cycle-over-cycle deltas）。** 设定单个周期内任意维度允许的最大变动幅度。若单周期内能力跃升超过 30%，几乎总是评估器操纵（evaluator gaming）的信号；此时必须暂停并进行审计。
3. **错位差距（Misalignment gap）。** 在每个周期结束后计算“能力值减去对齐值”的差值。若该差距扩大超过 X（由操作员设定），则暂停循环。此指标正是 `code/main.py` 中模拟器所测试的核心度量。
4. **回退监控（Regression watch）。** 若任意维度在单周期内下降超过 Y，则暂停循环。能力回退往往紧随能力激增之后；及时捕捉可防止虚假进展的加速。
5. **人工恢复协议（Human resumption contract）。** 循环暂停后恢复运行前，必须指定具体负责人审查暂停触发条件，酌情重新设定阈值，并将该决策记录至流水线外部的审计轨迹中。

硬性拒绝条件（Hard rejects）：
- 任何暂停后无需人工干预即可自动恢复的流水线。
- 任何依赖于循环内部评估器（internal evaluator）的阈值（智能体可对其进行操纵）。
- 任何阈值集合可被智能体自行修改的流水线。

拒绝规则（Refusal rules）：
- 若用户无法提前明确列出阈值，则予以拒绝。事后补设的阈值并非真正的阈值，而是事后合理化解释。
- 若流水线缺乏外部（循环外）评估器，则予以拒绝——回退与激增检测必须依赖外部视角。
- 若提议的恢复协议为“通知团队并在 24 小时后继续”，则予以拒绝。恢复运行必须是一项明确的主动操作。

输出格式：
返回一份单页规范，包含以下内容：
- **维度与阈值（Axes and thresholds）**（表格）
- **周期间变化量限制（Cycle-delta limits）**（表格）
- **错位差距公式与阈值（Misalignment gap formula and threshold）**
- **回退限制（Regression limits）**
- **外部评估器（External evaluator）**（定义及其运行时机）
- **恢复协议（Resumption contract）**（指定负责人、检查清单、日志存储位置）
- **签署行（Sign-off line）**（暂停不变量（pause invariant）的责任归属人）