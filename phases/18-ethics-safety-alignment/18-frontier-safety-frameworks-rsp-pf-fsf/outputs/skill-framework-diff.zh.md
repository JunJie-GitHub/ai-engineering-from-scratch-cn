---
name: framework-diff
description: 将新的安全框架（Safety Framework）或发布说明与 RSP v3.0、PF v2、FSF v3.0 进行对比。
version: 1.0.0
phase: 18
lesson: 18
tags: [rsp, pf, fsf, 前沿安全, 安全论证]
---

给定一个新的安全框架、政策或发布说明，请沿五个结构维度将其与 Anthropic RSP v3.0、OpenAI PF v2 和 DeepMind FSF v3.0 进行对比。

输出内容：

1. 层级结构（Tier Structure）。该框架是否定义了离散的能力阈值（Capability Thresholds）？这些阈值是按领域划分（FSF 风格）还是全局统一（RSP 风格）？
2. 生化放核（CBRN）阈值。需要何种 CBRN 评估？是否引用了 WMDP（第 17 课）或等效基准？是否包含诱导研究（Elicitation Study）？
3. AI 研发（AI R&D）阈值。是否存在模型自主研究（Model-Autonomous Research）阈值？其标准是“初级研究员”（Anthropic AI R&D-2）还是“显著加速扩展”（Anthropic AI R&D-4）？
4. 竞争对手调整（Competitor-Adjustment）。如果竞争对手在未部署同等安全措施的情况下发布产品，该框架是否允许降低要求？请根据情况将其表述为竞赛动态（Race-Dynamic）或激励相容（Incentive-Compatibility）。
5. 安全论证（Safety Case）结构。是否需要书面安全论证？其目标是针对监控（Monitoring）、不可解释性（Illegibility）还是能力不足（Incapability）？证据标准（Evidence Bar）是什么？

硬性拒绝条件：
- 任何未定义分层能力阈值的安全框架。
- 任何缺少外部治理交叉引用（External Governance Cross-Reference，如英国 AISI、美国 CAISI、欧盟 AI 办公室）的框架。
- 任何声称“我们与所有已发布框架保持一致”但未提供具体阈值数字的框架。

拒绝规则：
- 如果用户询问哪个框架“最好”，请拒绝排名，并引导其关注结构对齐情况。
- 如果用户要求提供具体的数值阈值建议，请予以拒绝——阈值具有实验室特异性，且取决于其测量基础设施（Measurement Infrastructure）。

输出要求：一份单页的三框架并列对比表、已标记的差距分析，以及一项建议补充的具体阈值。需分别引用 RSP v3.0、PF v2 和 FSF v3.0 各一次。