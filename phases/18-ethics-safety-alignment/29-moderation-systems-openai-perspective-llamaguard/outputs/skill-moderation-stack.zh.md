---
name: 内容审核技术栈
description: 为生产环境部署（Production Deployment）推荐内容审核技术栈（Moderation Stack）配置。
version: 1.0.0
phase: 18
lesson: 29
tags: [OpenAI审核, Perspective, Llama Guard, 分层审核, Azure内容安全]
---

针对生产环境部署，推荐跨越三个层级的内容审核技术栈配置。

输出内容：

1. 输入分类器（Input Classifier）。选择 OpenAI Moderation、Llama Guard 3/4 或 Perspective API。需与策略分类体系（Policy Taxonomy）相匹配。对于多模态部署（Multimodal Deployment），选用 Llama Guard 4 或 OpenAI 全模态审核（Omni-moderation）。
2. 输出分类器（Output Classifier）。可与输入分类器相同或不同。需将阈值（Thresholds）与下游风险模型（Downstream Risk Model）相匹配。
3. 自定义领域规则（Custom Domain Rules）。列举通用分类器无法捕获的特定领域规则：金融建议免责声明、医疗建议拒绝话术、法律免责声明模式。
4. 边缘案例（Edge Cases）裁决。明确人工升级路径（Human-escalation Path）。硬性拒绝（Hard Refusals）为最终决定；模糊案例需在服务等级协议（SLA）规定时间内转交人工审核。
5. 迁移计划（Migration Plan）。若技术栈中包含 Azure Content Moderator，需规划在 2027 年 2 月服务停用前迁移至 Azure AI Content Safety。

硬性拒绝条件：
- 任何未配置输出审核的部署（仅靠输入审核并不充分）。
- 任何在受监管业务场景（Regulated Surfaces，如金融、医疗、法律）中未配置自定义领域规则的部署。
- 任何在现代聊天应用中仅依赖大语言模型（LLM）时代前分类器（Pre-LLM-era Classifiers，如 Perspective）的部署。

拒绝响应规则：
- 若用户询问“唯一最佳”的分类器，应予以拒绝——分类器的选择取决于具体的策略分类体系。
- 若用户询问具体阈值，应拒绝提供单一数值——阈值取决于风险容忍度（Risk Tolerance）及下游影响（Downstream Effect）。

输出要求：一份单页推荐文档，需完整涵盖上述五个部分，明确各层级所选用的分类器名称，并标注迁移义务。需分别引用一次 OpenAI Moderation 官方文档和 Llama Guard 3/4 参考资料。