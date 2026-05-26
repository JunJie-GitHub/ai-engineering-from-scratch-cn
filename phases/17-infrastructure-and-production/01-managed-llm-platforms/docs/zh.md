# 托管大语言模型平台（Managed LLM Platforms） — Bedrock、Vertex AI、Azure OpenAI

> 三大超大规模云厂商（Hyperscalers），三种截然不同的战略。AWS Bedrock 是一个模型市场（Model Marketplace）—— 通过单一 API 提供 Claude、Llama、Titan、Stability 和 Cohere 等模型。Azure OpenAI 则是与 OpenAI 的独家合作，并提供预配置吞吐量单元（Provisioned Throughput Units, PTUs）以实现专用容量（Dedicated Capacity）。Vertex AI 采取 Gemini 优先策略，在长上下文（Long-context）和多模态（Multimodal）方面表现最佳。2026 年，Artificial Analysis 的测试数据显示，在等效于 Llama 3.1 405B 的模型上，Azure OpenAI 的中位延迟（Median Latency）约为 50 毫秒，而 Bedrock 约为 75 毫秒——PTU 解释了这一差距，因为专用容量优于共享的按需（On-demand）资源。决策规则不是“哪个最快”，而是“哪个模型目录（Model Catalog）和 FinOps（财务运营）管理界面最契合我的产品”。本课程将教你基于明确的权衡分析（Tradeoffs）进行选择，而非凭直觉。

**类型：** 学习
**语言：** Python（标准库，简易成本与延迟对比工具）
**前置要求：** 第 11 阶段（大语言模型工程），第 13 阶段（工具与协议）
**时长：** 约 60 分钟

## 学习目标

- 说出三种平台战略（市场模式 vs 独家合作 vs Gemini 优先），并将每种战略与具体的产品用例相匹配。
- 解释 Azure OpenAI 中的预配置吞吐量单元（PTUs）能为你带来什么，以及为何在 405B 规模下，按需模式的 Bedrock 延迟通常高出约 25 毫秒。
- 绘制各平台的 FinOps 成本归因架构图（Bedrock 应用推理配置文件 vs Vertex 每团队独立项目 vs Azure 作用域 + PTU 预留）。
- 制定“至少使用两家供应商”的策略，并解释为何在 2026 年，单一供应商锁定（Vendor Lock-in）是一项代价高昂的错误。

## 问题背景

你为产品选定了 Claude 3.7 Sonnet 模型，现在需要将其部署上线。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者借助 API 网关（Gateway）进行路由。直接调用 API 最为简单；Bedrock 则额外提供了业务关联协议（BAAs）、虚拟私有云（VPC）端点、身份与访问管理（IAM）以及 CloudWatch 成本归因功能。而网关方案则能实现跨供应商的故障转移（Failover）、统一计费和速率限制（Rate Limits）。

更深层次的问题在于模型目录。如果你的同一款产品需要同时使用 Claude、Llama 和 Gemini，除非你同时接入 Bedrock、Vertex 和 Azure OpenAI，否则无法从单一渠道获取所有模型。这些超大规模云厂商并非可随意互换——它们各自对“谁将主导模型层”做出了不同的战略押注。

本课程将梳理这三大战略押注、延迟差异、FinOps 管理差异以及供应商锁定风险。

## 核心概念

### 三大策略

**AWS Bedrock** —— 市场聚合模式。集成 Claude（Anthropic）、Llama（Meta）、Titan（AWS 自研）、Stability（图像生成）、Cohere（嵌入模型 Embeddings）、Mistral，以及图像和嵌入子目录。统一的 API（应用程序接口）、统一的 IAM（身份与访问管理）控制面、统一的 CloudWatch 日志导出。Bedrock 的核心策略是：客户对模型选择灵活性的需求，远大于对单一模型的依赖。

**Azure OpenAI** —— 独家合作模式。提供 GPT-4 / 4o / 5 / o 系列、DALL·E、Whisper，以及 Azure 数据中心内的 OpenAI 模型微调（Fine-tuning）。“Azure OpenAI Service”目录中不包含非 OpenAI 模型——这些模型被归入 Azure AI Foundry（独立产品）。Azure 的押注在于：OpenAI 仍代表技术前沿，且客户希望针对这一特定合作关系获得企业级管控能力。

**Vertex AI** —— Gemini 优先，其余为辅。提供 Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro 版本，以及 Model Garden（第三方模型）。Vertex 的押注在于多模态长上下文（Multimodal Long-Context）—— 100 万 Token 的 Gemini 上下文窗口是其核心差异化优势。

### 规模化下的延迟差距

Artificial Analysis 持续运行基准测试（Benchmarks）。在同等规模的 Llama 3.1 405B 部署（共享按需实例）中，Azure OpenAI 的首 Token 延迟（First-Token Latency）中位数约为 50 毫秒；Bedrock 约为 75 毫秒。这一差距并非 AWS 的技术缺陷，而是容量模型（Capacity Model）的差异。Azure 销售预置吞吐量单元（Provisioned Throughput Units），可为您的租户预留 GPU 算力。Bedrock 的对应服务预置吞吐量（Provisioned Throughput）虽然存在，但单价起步约为 21 美元/小时/单元，且大多数客户仍停留在共享按需模式。

共享按需容量会与其他所有客户的流量产生资源竞争，而专用容量则不会。如果您的产品 SLA（服务等级协议）要求 P99 延迟下的 TTFT（首 Token 时间 Time To First Token）低于 100 毫秒，您要么在 Azure 购买 PTU，要么购买 Bedrock 预置吞吐量，要么只能接受默认的延迟波动。

### 预置吞吐量的经济学

Azure PTU：预留的推理（Inference）算力块。对于可预测的工作负载，相比按需模式最高可节省约 70% 的成本。费用按小时固定收取，与流量无关——即使闲置也需为预留资源付费。盈亏平衡点通常出现在持续利用率达到 40%-60% 时。

Bedrock 预置吞吐量：根据模型和区域不同，单价为 21-50 美元/小时。计算逻辑类似——盈亏平衡点约为峰值利用率的一半。需要按月承诺用量。

Vertex 的预置容量按 Gemini SKU（库存量单位）销售；价格因模型和区域而异，且公开披露较少。

### FinOps（云财务运营）管控面 —— 真正的差异化所在

**Bedrock 应用推理配置文件（Application Inference Profiles）** 提供了市场上最清晰的费用归因（Attribution）机制。为配置文件打上 `team`、`product`、`feature` 标签；将所有模型调用路由至该配置文件；CloudWatch 即可直接按配置文件拆分成本，无需后处理。该功能于 2025 年推出，目前仍是超大规模云厂商中原生支持最细粒度的方案。

**Vertex** 的费用归因采用“一团队一项目”加“全资源打标签”的模式。您将每个团队建模为一个 GCP 项目，为所有资源添加标签，并使用 BigQuery 账单导出（Billing Export）结合 DataStudio 进行数据汇总。虽然配置工作量更大，但 BigQuery 允许您对成本数据执行任意 SQL 查询。

**Azure** 依赖订阅/资源组作用域（Scopes）加标签，并将 PTU 预留作为一等成本对象。标签继承自资源组而非请求本身，因此要实现单次请求级别的费用归因，必须依赖 Application Insights 自定义指标，或通过网关在请求头中注入标签。

总结规律：Bedrock 的原生方案最清晰，Vertex 通过 BigQuery 最灵活，而 Azure 若不进行额外插桩（Instrumentation），则透明度最低。

### 供应商锁定是 2026 年的核心风险

当单一模型占据主导地位时，绑定单一超大规模云厂商尚可接受。但到了 2026 年，技术前沿每月都在更迭——本季度是 Claude 3.7，下季度是 Gemini 2.5，再下季度是 GPT-5。锁定单一平台，意味着您将错失三分之二的前沿模型能力。

实战团队普遍采用的模式：任何对产品至关重要的 LLM（大语言模型）调用，至少采用双供应商架构。Bedrock 搭配 Azure OpenAI 是常见组合——一个提供 Claude，另一个提供 GPT，通过同一网关实现故障转移（Failover）。由于网关会自动路由至最优路径，成本增幅微乎其微；而在发生中断时（如 2025 年 1 月 Azure OpenAI 故障、AWS us-east-1 区域宕机），可用性的提升则是决定性的。

### 数据驻留、BAA（商业伙伴协议）与受监管行业

Bedrock：大多数区域支持 BAA；提供 VPC（虚拟私有云）端点；内置安全护栏（Guardrails）。金融科技行业的常见默认选择。
Azure OpenAI：符合 HIPAA、SOC 2、ISO 27001 标准；支持欧盟数据驻留；企业级合规场景的默认选择。
Vertex：符合 HIPAA、GDPR 标准；按区域提供数据驻留；依托 Google Cloud 的合规技术栈。

三者均能满足基础合规要求。差异主要体现在数据保留策略、日志处理方式，以及滥用监控是否会读取您的流量数据（大多数默认开启，企业客户可选择退出）。

### 关键数据备忘

- Azure OpenAI 在等效 Llama 3.1 405B 部署上的 TTFT 中位数：约 50 毫秒（使用 PTU 时）。
- Bedrock 按需模式的 TTFT 中位数：约 75 毫秒。
- Bedrock 预置吞吐量：21-50 美元/小时/单元。
- Azure PTU 盈亏平衡点：持续利用率约 40%-60%。
- 高利用率下 PTU 相比按需模式的成本节省：最高可达 70%。

## 实践使用

`code/main.py` 在合成工作负载（synthetic workload）上对比了这三个平台——它对按需付费（on-demand）与预置吞吐量单元（PTU）的经济性、首字延迟（TTFT）方差以及成本归因精确度（cost attribution fidelity）进行了建模。运行该脚本，即可查看 PTU 在何种场景下更具性价比，以及模型市场（marketplace）的广度优势何时能弥补 TTFT 的差距。

## 交付产出

本课时将生成 `outputs/skill-managed-platform-picker.md`。基于给定的工作负载画像（workload profile，包含所需模型、首字延迟服务等级协议（TTFT SLA）、日均调用量及合规要求），该脚本会推荐主用平台、备用平台，并制定一套 FinOps 可观测性方案（FinOps instrumentation plan）。

## 练习

1. 运行 `code/main.py`。对于 70B 级别的模型，Azure PTU 在持续利用率达到多少时能优于按需付费模式？计算盈亏平衡点，并与官方宣传的 40-60% 区间进行对比。
2. 你的产品需要同时使用 Claude 3.7 Sonnet 和 GPT-4o。设计一个双供应商部署方案——分别将哪个模型部署到哪个超大规模云服务商（hyperscaler）？前端使用什么网关？故障转移（failover）策略是什么？
3. 某受监管的医疗客户需要签署业务伙伴协议（BAA）、数据驻留于美国东部区域，且 P99 TTFT 低于 100ms。选择一个平台，并用三个具体特性说明理由。
4. 你发现本月 Bedrock 账单暴涨 4 倍，但流量并无变化。在没有应用推理配置文件（Application Inference Profiles）的情况下，如何定位问题根源？如果使用了配置文件，需要多长时间？
5. 查阅 Azure OpenAI 和 Bedrock 的定价页面。对于每月 1 亿 token 的 Claude 工作负载，哪种方案更便宜——直接调用 Anthropic API、Bedrock 按需付费，还是 Bedrock 预置吞吐量（Provisioned Throughput）？

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Bedrock | “AWS 的大语言模型服务” | 涵盖 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | “Azure 版的 ChatGPT” | 部署于 Azure 数据中心、具备企业级管控能力的 OpenAI 专属模型服务 |
| Vertex AI | “Google 的大语言模型” | 以 Gemini 为核心的平台，并通过 Model Garden（模型花园）提供第三方模型 |
| PTU | “专用容量” | 预置吞吐量单元（Provisioned Throughput Unit）——预留的推理 GPU，按小时计费 |
| Application Inference Profile | “Bedrock 的标签功能” | 基于标签的按产品成本/用量画像，原生集成 CloudWatch |
| Model Garden | “Vertex 的模型目录” | Vertex AI 中独立于 Gemini 的第三方模型专区 |
| Two-provider minimum | “大语言模型冗余” | 要求所有关键大语言模型调用路径必须在至少 2 家超大规模云服务商上运行的策略 |
| BAA | “HIPAA 合规文件” | 业务伙伴协议（Business Associate Agreement）；处理受保护健康信息（PHI）的必备文件；三家云厂商均提供 |
| Abuse monitoring | “日志监控器” | 云服务商侧对提示词和输出内容的安全扫描；企业版支持选择退出 |

## 延伸阅读

- [AWS Bedrock 定价](https://aws.amazon.com/bedrock/pricing/) — 权威费率表及预置吞吐量（Provisioned Throughput）定价。
- [Azure OpenAI Service 定价](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — 预置吞吐量单元（PTU）成本结构与费率表。
- [Vertex AI 生成式 AI 定价](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini 定价阶梯与模型花园（Model Garden）附加费用。
- [Artificial Analysis 大语言模型（LLM）排行榜](https://artificialanalysis.ai/) — 跨云服务商的持续延迟与吞吐量基准测试。
- [The AI Journal — AWS Bedrock 与 Azure OpenAI 首席技术官（CTO）指南 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 企业级决策框架。
- [Finout — Bedrock、Vertex 与 Azure 财务运营（FinOps）对比](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 费用归因机制横向对比。