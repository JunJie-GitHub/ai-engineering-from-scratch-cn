---
name: aar-deployment-review
description: 自动化对齐研究（Automated-Alignment-Research）流水线的部署前审查，包括沙箱隔离（Sandbox isolation）与日志完整性（Log integrity）。
version: 1.0.0
phase: 15
lesson: 6
tags: [aar, alignment-research, sandbox, log-integrity, rsp]
---

针对拟议的自动化对齐研究（Automated-Alignment-Research, AAR）部署方案（包含模型、沙箱、任务队列与论坛），请生成一份前沿实验室（frontier-lab）安全团队可签署认可的部署前审查报告。

请提供以下内容：

1. **沙箱隔离（Sandbox isolation）。** 验证每个 AAR 实例的文件系统、网络及进程限制。确认任一 AAR 均无法读取或修改其他 AAR 的沙箱。确认任一 AAR 均无法直接访问论坛存储的写入端点（所有帖子均需经由独立服务中转）。
2. **论坛日志完整性（Forum log integrity）。** 论坛日志必须为仅追加（append-only）且具备防篡改（tamper-evident）特性。请明确指定存储方案（例如启用对象锁的 S3、不可变账本或仅追加预写日志/WAL）。规定验证周期以及检测到篡改时的响应机制。
3. **任务分配策略（Task allocation policy）。** 阐明任务分配方式：由谁编写任务、由谁审批、AAR 是否可自行提议后续任务及其适用条件。自由分解（Free decomposition）在性能上占优；规定工作流（Prescribed workflow）在可审计性上占优。需明确记录此权衡取舍。
4. **人工审查关卡（Human review gate）。** 在任何 AAR 输出影响实际研究方向、训练数据或已发表成果之前，必须由指定的人工审查员签字确认。明确审查清单：实验可复现性、日志完整性、主张与证据的映射关系。
5. **阈值监控（Threshold monitoring）。** 追踪表明 AAR 系统正逼近 RSP v3.0 AI R&D-4 范畴的信号：任务复杂度增量、自主子任务生成、跨任务信息传递。定义触发升级审查的阈值。

硬性否决项（Hard rejects）：
- 任何缺乏防篡改且位于沙箱外日志的 AAR 部署。
- 任何 AAR 输出未经指定人工审查即直接输入训练数据或生产策略的流水线。
- 任何单个 AAR 拥有足以影响多个下游系统凭证的流水线。

拒绝规则（Refusal rules）：
- 若沙箱隔离未明确说明或仅依赖单一防护层（仅使用 Docker，未配置 seccomp / gVisor），则予以拒绝并要求实施纵深防御（defense-in-depth）。
- 若日志存储允许任何人（包括运维人员）编辑，则予以拒绝并要求使用一次写入介质（write-once media）。
- 若部署目标旨在自动化能力流水线（capability pipeline）的某个环节——而不仅限于对齐研究——则予以拒绝并升级至 RSP 审查。

输出格式：

返回一份审查备忘录，包含以下内容：
- **流水线摘要（Pipeline summary）**（一段话）
- **隔离评分（Isolation score）**（按维度：文件系统/fs、网络/net、进程/proc、对等节点/peer）
- **日志完整性评分（Log integrity score）**（附验证计划）
- **任务分配决策（Task allocation decision）**（固定/fixed / 自由/free / 混合/hybrid，附理由）
- **人工审查关卡（Human review gate）**（审查员姓名、清单）
- **阈值监控器（Threshold monitors）**（信号列表、阈值、响应机制）
- **部署裁决（Deployment verdict）**（通过/go / 暂缓/hold / 否决/no-go）