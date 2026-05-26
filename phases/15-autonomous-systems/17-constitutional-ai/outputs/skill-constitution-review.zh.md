---
name: constitution-review
description: 审计部署的宪法层（Constitutional Layer）——涵盖硬编码禁令（Hardcoded Prohibitions）、软编码默认值（Soft-coded Defaults）、操作员可调边界（Operator-Adjustable Bounds）以及四层级解析（Four-Tier Hierarchy Resolution）。
version: 1.0.0
phase: 15
lesson: 17
tags: [宪法AI, 规则覆盖, 层级结构, CAI, RLAIF, 硬编码禁令]
---

给定部署的宪法层（Constitutional Layer，含系统提示词、操作员配置、声明原则），请对照 Claude 宪法（Claude Constitution）参考文档进行审计，并标记缺失的硬编码禁令、模糊的原则或排序错误的层级。

输出内容：

1. **硬编码禁令清单（Hardcoded Prohibition Inventory）。** 列出所有无论操作员或用户如何指令都绝不可妥协的禁令。最低底线：生物武器/化学、生物、放射性和核（CBRN）技术提升、儿童性虐待材料（CSAM）、关键基础设施攻击规划、被要求时伪造身份。附加项因部署场景而异（例如，金融服务会添加特定的反欺诈禁令）。
2. **软编码默认值（Soft-coded Defaults）。** 列出操作员可调整的每一项行为。针对每一项，需声明其设定边界。没有声明边界的“可调”设置等同于后门覆盖（Back-Door Override）。
3. **层级排序（Tier Ordering）。** 确认解析顺序为：安全 > 伦理 > 指南 > 有益性。如果在已实现的解析器（Resolver）中，“有益性”曾凌驾于“伦理”之上，则标记为部署缺陷。
4. **原则歧义标记（Principle Ambiguity Flags）。** 识别任何文本留有实质性不同解释空间的原则。歧义会在训练周期中不断累积（原则漂移，Principle Drift）。
5. **层级完整性（Layer Completeness）。** 确认除宪法层外，运行时层控制（Runtime-Layer Controls，见第 10、13、14 课）是否已就位。仅靠宪法层不足以保证安全；仅靠运行时层同样不足。

**硬性拒绝条件（Hard Rejects）：**
- 缺乏任何硬编码禁令层的部署。
- 操作员配置声称可覆盖硬编码禁令（即使通过重命名实现）。
- 将“有益性”置于“伦理”之上的层级排序。
- 原则文本过于宽泛以致无法评估（例如“做个好人”）。
- 将宪法AI（Constitutional AI）视为运行时控制的替代品。

**拒绝规则（Refusal Rules）：**
- 如果用户指出了某项硬编码禁令，但无法指出其对应的运行时层后备机制，则将该部署标记为单层架构并拒绝投入生产。
- 如果操作员配置包含未声明边界的可调“安全”设置，则予以拒绝。
- 如果用户将 2023 年参与式宪法（Participatory Constitution）的研究结果视为当前部署的可执行依据，请核查：2026 年宪法并未纳入这些内容，因此“民主继承”是该部署无法证实的主张。

**输出格式：**

返回一份宪法审计报告，包含以下内容：
- **硬编码底线（Hardcoded Floor）**（禁令、执行层：权重 / 推理 / 两者兼有）
- **软编码默认值（Soft-coded Defaults）**（设置项、操作员边界、用户可见：是/否）
- **层级顺序（Tier Order）**（已列出；确认安全 > 伦理 > 指南 > 有益性）
- **歧义标记（Ambiguity Flags）**（原则、具体歧义、建议收紧方案）
- **层级完整性（Layer Completeness）**（宪法层：是/否，运行时控制：是/否，两者均为必需）
- **就绪状态（Readiness）**（生产环境 / 预发布环境 / 仅限研究）