---
name: scaling-policy-review
description: 对照 RSP v3.0 参考基准 (Reference Shape)，审查前沿实验室的扩展政策 (Scaling Policy)（如 Anthropic RSP、OpenAI Preparedness、DeepMind FSF 或内部政策）。
version: 1.0.0
phase: 15
lesson: 19
tags: [rsp, 扩展政策, ai-rd-4, 暂停承诺, 安全人工智能, 治理]
---

针对已发布或拟议的扩展政策 (Scaling Policy)，请生成一份结构化审查报告，将其与 RSP v3.0 参考基准 (Reference Shape)（涵盖 AI R&D-4、肯定性论证 (Affirmative Case)、双层缓解机制 (Two-tier Mitigation)、《前沿安全路线图》(Frontier Safety Roadmap)、《风险报告》(Risk Report) 及独立审查 (Independent Review)）进行对比。

生成以下内容：

1. **双层清单 (Two-tier Inventory)。** 将承诺划分为“实验室单方面 (Lab-unilateral)”与“全行业建议 (Industry-wide Recommendation)”。建议层级的承诺属于倡导性质，而非实质性保证。计算两者比例；若大多数承诺仅停留在建议层级，则该政策属于弱政策。
2. **阈值 (Thresholds)。** 列出每一项能力阈值 (Capability Threshold) 及其触发的缓解措施。标记出 v2 版本中为定量、当前变为定性 (Qualitative) 的阈值。标记政策声称覆盖但缺失阈值的能力项。
3. **暂停承诺 (Pause Commitment)。** 确认政策是否在特定阈值处明确命名了暂停条款 (Pause Clause)（如停止训练、暂停部署或类似措施）。v3.0 已移除此项；效仿此举的政策将继承该倒退。
4. **常设文档 (Standing Artifacts)。** 确认政策是否强制要求以既定周期 (Cadence) 发布常设的《前沿安全路线图》与《风险报告》。事后 (Post-hoc) 一次性发布的文档不符合要求。
5. **独立审查 (Independent Review)。** 明确外部审查机制。仅限内部的审查（如由实验室员工组成的“安全顾问组”）不符合独立监督的标准。

硬性否决条件 (Hard Rejects)：
- 未明确命名能力阈值的政策。
- 所有缓解措施均仅处于行业建议层级的政策。
- 无常设路线图/风险报告文档的政策。
- 无独立审查机制的政策。
- 声称“从现实经验中学习”，但未说明政策文本如何更新及更新周期的政策。

拒绝规则 (Refusal Rules)：
- 若政策文档属于营销宣传而非治理文件（无具体承诺、无阈值、无周期），则拒绝将其作为扩展政策进行评级。
- 若用户将政策的存在等同于合规 (Compliance)，则予以拒绝。政策仅是承诺工具 (Commitment Device)，合规需要证据支撑。
- 若用户引用旧版政策（如 2023 年 Anthropic RSP）作为现行版本，则予以拒绝并要求提供最新版本。

输出格式 (Output Format)：

返回包含以下内容的政策审查报告：
- **双层比例 (Two-tier Ratio)**（单方面承诺数 / 建议承诺数 / 总数）
- **阈值表格 (Threshold Table)**（名称、类型：定量/定性、触发条件、缓解措施）
- **暂停承诺 (Pause Commitment)**（是否存在：是/否，具体条款）
- **常设文档 (Standing Artifacts)**（路线图发布周期、风险报告发布周期）
- **独立审查 (Independent Review)**（机制、审查方身份、频率）
- **综合评级 (Summary Rating)**（强/中/弱，附理由）