---
name: 跨政策差异对比
description: 以 OpenAI Preparedness Framework v2（OpenAI 准备度框架 v2）、Anthropic RSP v3.0（Anthropic 负责任扩展政策 v3.0）和 DeepMind FSF v3（DeepMind 前沿安全框架 v3）为参考，针对特定能力生成跨政策对比。
version: 1.0.0
phase: 15
lesson: 20
tags: [准备度框架, fsf, rsp, 跨政策, 扩展政策]
---

给定一项特定的前沿能力（frontier capability）（例如“长程自主性”、“自主复制与适应”或“研发自动化”），生成一份跨政策差异对比（cross-policy diff），展示三大框架如何对该能力进行分类，以及会触发哪些缓解措施（mitigations）。

生成以下内容：

1. **OpenAI PF v2 分类。** 归类为“追踪类”（Tracked）还是“研究类”（Research）。若为追踪类，需指明《能力与安全保障报告》（Capabilities + Safeguards Report）的触发条件。若为研究类，需注明政策措辞为“潜在”缓解措施。
2. **Anthropic RSP v3.0 分类。** 属于哪个阈值（ASL-3、AI R&D-4 或硬性禁止（hardcoded prohibition））？适用哪种缓解措施（肯定性论证（affirmative case）、安全与部署（security + deployment））？确认该承诺属于 Anthropic 单边执行层级（Anthropic-unilateral tier）还是行业建议层级（industry-recommendation tier）。
3. **DeepMind FSF v3 分类。** 属于哪个领域（网络、生物、机器学习研发或化学/生物/放射/核武器（CBRN））？对应哪个关键能力等级（CCL）或追踪能力等级（Tracked Capability Level）？是否启用了欺骗性对齐（deceptive alignment）监控？
4. **趋同总结。** 三项政策对该能力严重程度的评估是否一致，还是存在实质性分歧？哪种分类最为严谨，哪种最宽松？
5. **测量依赖性。** 每项分类均依赖于能力测量（capability measurement）。说明该能力的测量方式，以及由哪家评估提供方（eval provider）（如 METR、Apollo、内部团队或第三方）负责该测量。

硬性拒绝条件：
- 仅基于公告措辞相似性而缺乏文档级证据的跨政策一致性主张。
- 无法指向源文档中具体条款的任何分类。
- 将“研究类”（Research Category，OpenAI）等同于“追踪类”（Tracked Category）——两者具有不同的操作后果。

拒绝规则：
- 若用户无法提供每项分类对应的源文档段落，则予以拒绝，并要求先提供引用。
- 若用户将政策的存在视为缓解措施已实际执行的证据，则予以拒绝，并要求提供具体缓解措施已触发的证据。
- 若声称某能力被某框架“覆盖”，但文档中并未出现该词，则予以拒绝，并要求提供具体的条款引用。

输出格式：

返回一份差异对比文档，包含：
- **能力定义**（一句话）
- **OpenAI PF v2 行**（分类、触发条件、源条款）
- **Anthropic RSP v3.0 行**（分类、触发条件、单边执行与行业建议对比）
- **DeepMind FSF v3 行**（领域、CCL / TCL、欺骗性对齐涉及情况）
- **趋同总结**（一致性与实质性分歧）
- **测量归属**（评估提供方、评估频率（eval cadence））
- **读者建议**（最严谨、最宽松及理由）