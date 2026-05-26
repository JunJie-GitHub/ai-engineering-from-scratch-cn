---
name: 推理平台选择器
description: 根据工作负载画像 (workload profile)、服务等级协议 (SLA)、预算及运维约束条件，选择推理平台（Fireworks、Together、Baseten、Modal、Replicate、Anyscale 或定制芯片 (custom silicon)）。将按 Token、按分钟和按预测的定价进行标准化。
version: 1.0.0
phase: 17
lesson: 02
tags: [推理, fireworks, together, baseten, modal, replicate, anyscale, 经济学]
---

给定工作负载画像（模型、每日 Token 数、持续利用率、首字延迟 (TTFT) 服务等级协议 (SLA)、突发系数、合规要求、纯 Python 与混合技术栈），生成平台推荐方案。

输出内容：

1. 首选平台。指明平台名称及具体的定价层级（无服务器 (Serverless) vs 专用实例 (Dedicated) vs 批处理 (Batch)）。结合匹配的工作负载特征进行论证——例如：“选择 Fireworks 无服务器方案，因为 SLA 要求 TTFT < 500 毫秒，且流量具有突发性。”
2. 有效成本。将所选定价模型标准化为每百万输出 Token 的美元成本（$/M output tokens）。与至少两种替代方案进行对比。明确指出在何种情况下按分钟计费优于按 Token 计费（持续利用率高于约 30% 时），反之亦然。
3. 冷启动 (Cold-start) 应对方案。若选择无服务器方案（Fireworks、Modal、Replicate），需说明预期的冷启动延迟及缓解措施（预热 (pre-warming)、`min_workers=1`、实时迁移 (live-migration)）。若选择专用实例方案（Baseten、Anyscale），可跳过此部分，但需注明其权衡取舍。
4. 备选平台。指明第二顺位平台，并明确触发切换的具体条件（例如：“若达成需要 HIPAA 合规 + 专用 GPU 的企业级交易，则迁移至 Baseten”）。
5. 网关层 (Gateway layer)。建议是否在平台前端部署 AI 网关（LiteLLM、Portkey、Kong AI Gateway），以隔离产品与供应商变更带来的风险。默认建议：是，除非请求规模低于 500 RPS（每秒请求数）。

硬性否决项：
- 未进行标准化就直接对比按 Token 计费与按分钟计费。必须拒绝，并坚持使用有效的每百万 Token 美元成本（$/M tokens）进行对比。
- 仅因 Fireworks “最快”就选择它，而未对照已发布的基准测试验证 TTFT SLA。
- 为非延迟敏感型 (latency-bound) 工作负载推荐定制芯片（Groq、Cerebras、SambaNova）。此类芯片定价溢价较高，仅在对交互性 SLA 有严格要求时才具备合理性。

拒绝规则：
- 若工作负载需要受监管的合规框架（SOC 2 Type II、HIPAA），且客户选择了 Modal 或 Replicate，则必须拒绝——这两者在企业级覆盖面上均不及 Baseten 或 Anyscale。建议改用 Baseten。
- 若预期流量低于每日 10 万 Token，则拒绝推荐按分钟计费方案（Baseten、Modal、Anyscale）。此类场景下经济性不成立——默认应转向模型市场（OpenRouter、DeepInfra）或托管型超大规模云服务商 (hyperscaler)。
- 若客户仅追求“最便宜”，则必须拒绝——需明确指出多维成本函数（Token 费率 + 冷启动成本 + 归因分析 + 网关开销 + 开发者体验 (DX)）。

输出要求：一份单页推荐报告，需明确首选平台、有效成本、冷启动应对方案、备选平台及网关部署策略。结尾需指出唯一能暴露选型失误的关键指标（冷启动 P99 延迟、按 Token 费率或利用率漂移 (utilization drift)）。