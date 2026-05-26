---
name: 托管平台选择器
description: 根据工作负载、服务等级协议（Service Level Agreement, SLA）和合规性要求，选择一款托管大语言模型（Large Language Model, LLM）平台（Bedrock、Azure OpenAI、Vertex AI）及一个备用平台以实现冗余，随后制定财务运营（Financial Operations, FinOps）可观测性计划。
version: 1.0.0
phase: 17
lesson: 01
tags: [bedrock, azure-openai, vertex-ai, ptu, finops, 托管平台]
---

给定工作负载画像（Workload Profile）（所需模型、月度 Token 消耗量、P50/P99 首字延迟（Time To First Token, TTFT）的服务等级协议（Service Level Agreement, SLA）、合规性约束、现有云基础设施覆盖范围），输出平台推荐方案。

输出内容：

1. **主平台**。明确平台名称、涵盖的具体模型，并根据利用率评估采用按需（On-Demand）模式还是预置吞吐量单元（Provisioned Throughput Units, PTU）/预置吞吐量（Provisioned Throughput）更为合适。引用盈亏平衡计算依据（PTU 通常在持续利用率达到 40-60% 时实现盈亏平衡）。
2. **备用平台**。明确至少包含两家供应商的降级方案。论证配对合理性——冗余设计必须覆盖模型重叠（例如 Bedrock 上的 Claude + Azure OpenAI 上的 GPT 是常见组合）与区域重叠。
3. **FinOps 可观测性配置**。明确首日需启用的功能：Bedrock 应用推理配置文件（Application Inference Profiles）、将 Azure 作用域（Scopes）与 PTU 预留作为成本对象、Vertex AI 的“每团队一项目”架构 + BigQuery 账单导出（Billing Export）。明确成本分摊维度——按用户、按任务、按租户。
4. **SLA 校验**。将目标 TTFT P99 与官方公布的基准数据进行对比（Azure OpenAI PTU P50 延迟约 50 毫秒；Bedrock 按需模式 P50 延迟约 75 毫秒）。若 SLA 要求严于按需模式所能提供的上限，则必须采用 PTU。
5. **合规性校验**。按需验证业务伙伴协议（Business Associate Agreement, BAA）、SOC 2 Type II、HIPAA 及欧盟数据驻留要求。需注意三者均满足基线标准，但在数据保留策略及滥用监控退出机制上存在差异。
6. **迁移路径**。明确团队本周可执行的一项可逆步骤（例如：通过抽象底层供应商的 AI 网关进行部署；配置成本分摊请求头）以及一项长期步骤（PTU 容量承诺；跨区域故障转移）。

**硬性拒绝条件**：
- 仅推荐单一平台而未指定明确的备用方案。必须拒绝并坚持至少采用两家供应商。
- 在未提供利用率预估的情况下选择 PTU。必须拒绝并要求提供持续利用率数据。
- 当成本分摊被列为需求时，忽略 Bedrock Application Inference Profiles。它们是最清晰的原生接口。

**拒绝规则**：
- 若工作负载要求 Claude、Gemini 和 GPT 均为 P0（Priority 0）模型，需明确指出三平台并存的现实（通过网关统一接入 Bedrock + Vertex AI + Azure OpenAI），而非虚构单一平台可同时满足三者。
- 若 SLA 要求 TTFT P99 < 100 毫秒，且预期预算无法支撑 PTU，则拒绝承诺该 SLA——需说明按需模式的延迟波动上限。
- 若客户要求“使用最便宜的供应商”，必须拒绝——定价是多维度的（Token 单价 + 专用容量 + 成本分摊开销 + 供应商锁定成本）。

**输出要求**：一份单页决策文档，包含主平台、备用平台、PTU 与按需模式对比、可观测性清单、SLA/合规性验证结果及两项迁移步骤。文末需附上用于监测计划偏离情况的单一核心指标（持续利用率、PTU 资源浪费率（PTU Waste）或成本分摊覆盖率（Attribution Coverage））。