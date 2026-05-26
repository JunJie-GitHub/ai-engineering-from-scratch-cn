---
name: gateway-picker
description: 根据业务规模、延迟预算、合规要求、运维现状及定价容忍度，选择一款 AI 网关（AI Gateway）（LiteLLM、Portkey、Kong AI、Cloudflare/Vercel）。
version: 1.0.0
phase: 17
lesson: 19
tags: [AI 网关, litellm, portkey, kong, cloudflare, vercel, bifrost, 故障回退, 速率限制, 安全护栏]
---

根据当前及未来 12 个月预测的每秒请求数（RPS）、延迟预算、合规要求（是否需要本地自托管？）、安全护栏（Guardrails）需求（个人身份信息（PII）脱敏、越狱检测、审计）以及定价容忍度，输出网关推荐方案。

输出内容：

1. 主网关（Primary Gateway）。明确工具名称。结合 RPS 上限、系统开销及功能匹配度进行论证。
2. 降级链（Fallback Chain）。按顺序列出三家提供商；OpenAI → Anthropic → 自托管为经典范式。计算预期可用性。
3. 速率限制策略（Rate-Limit Policy）。RPS > 500 时推荐滑动窗口（Sliding Window）算法；其他情况可使用令牌桶（Token Bucket）算法。支持按租户分级。
4. 安全护栏（Guardrails）。若需 PII 脱敏/越狱检测则选 Portkey；若需高扩展性+护栏则选 Kong；若仅限开发环境则选 LiteLLM。
5. 可观测性（Observability）交接。指向第 17 阶段 · 第 13 课的选择；确认 OpenTelemetry（OTel）生成式 AI（GenAI）规范的数据流已贯通。
6. 迁移方案。若从应用层集成迁移至网关，需采用分阶段发布（Staged Rollout）（网关侧先进行 1% 金丝雀发布（Canary），验证成功后逐步扩大流量）。

硬性否决条件：
- LiteLLM 用于 >2000 RPS 场景。予以否决——Kong 基准测试表明存在级联故障风险；需先完成迁移。
- Portkey 用于首字延迟（TTFT）第 99 百分位延迟（P99）< 100 ms 的服务等级协议（SLA）场景。予以否决——其 30 ms 的额外开销会过度占用延迟预算。
- 为受监管的本地部署（On-Prem）客户使用 Cloudflare AI Gateway。予以否决——仅提供托管服务，不支持自托管。

拒绝规则：
- 若规模预期存在较大不确定性（当前 100 RPS，计划 6 个月内增至 2000+ RPS），在确定采用 LiteLLM 前，必须要求提供迁移方案。
- 若合规要求 SOC 2 Type II 认证，且所选网关仅为开源软件（OSS）且无托管 SLA 保障，则需客户提供其自身的 SOC 2 认证声明。
- 若团队未使用 Kubernetes 却选择 Kong 自托管版本，予以否决——建议改用托管版 Kong 或托管版 Portkey。

输出要求：一份单页决策文档，需包含网关选型、降级链、速率限制策略、安全护栏配置、可观测性数据流及迁移计划。文末需附带一项核心指标：过去一小时的网关延迟 P99 值；若超出阈值则触发告警。