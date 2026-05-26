# 推理（Inference）平台经济学 — Fireworks, Together, Baseten, Modal, Replicate, Anyscale

> 2026年的推理市场已不再是单纯的GPU算力租赁。它已分化为三大阵营：定制芯片（Custom Silicon，如Groq、Cerebras、SambaNova）、GPU平台（GPU Platforms，如Baseten、Together、Fireworks、Modal）以及API优先（API-first）市场（如Replicate、DeepInfra）。Fireworks于2026年5月1日将每GPU价格上调至1美元/小时，而其10万亿+ Token/天的吞吐量支撑起40亿美元的估值，这充分证明了以量取胜的商业模式行之有效。Baseten于2026年1月以50亿美元估值完成了3亿美元的E轮融资。竞争定位的规则很简单：Fireworks优化延迟（Latency），Together优化模型目录广度（Catalog Breadth），Baseten优化企业级打磨（Enterprise Polish），Modal优化Python原生开发者体验（Python-native DX），Replicate优化多模态（Multimodal）覆盖范围，Anyscale优化分布式Python（Distributed Python）能力。本课程将为你提供一份决策矩阵，你可以直接将其交给创始人参考。

**Type:** 学习
**Languages:** Python（标准库，单次调用经济学对比演示程序）
**Prerequisites:** 第17阶段 · 01（托管大语言模型平台），第17阶段 · 04（vLLM服务引擎内部原理）
**Time:** 约60分钟

## 学习目标

- 说出三大市场细分领域（定制芯片、GPU平台、API优先），并将各供应商映射到对应领域。
- 解释为何“按Token计费（Per-token）”的API定价模型会向推理服务引擎（Serving Engine）的成本曲线（Cost Curve）靠拢，而非硬件成本曲线。
- 计算至少三家供应商的单次请求有效成本，并解释在何种场景下按分钟计费（Baseten、Modal）优于按Token计费。
- 针对特定工作负载（Workload，如无服务器突发流量、稳定高吞吐量、微调变体、多模态），识别应作为默认选择的平台。

## 问题

你已经评估过托管型超大规模云厂商（Hyperscaler）平台。你决定需要一个更专注、响应更快的供应商——例如为了低延迟选择Fireworks，为了模型广度选择Together，或者为了微调的定制模型选择Baseten。现在你面临六个实际选择，但它们的定价页面根本无法直接对齐。Fireworks显示的是美元/百万Token；Baseten显示的是美元/分钟；Modal显示的是美元/秒；Replicate显示的是美元/次预测。如果不对工作负载进行建模，你根本无法将它们进行横向对比。

更糟糕的是，每个定价页面背后的商业模式截然不同。Fireworks在共享GPU上运行其自研的定制引擎（FireAttention）；其按Token费率反映了他们的GPU利用率曲线（Utilization Curve）。Baseten提供Truss框架加独占GPU（Dedicated GPUs）；按分钟计费反映了资源的排他性。Modal是真正的Python无服务器（Serverless）架构——按秒计费，且冷启动（Cold Starts）时间低于一秒。同样的输出（一个大语言模型响应），却对应着三种完全不同的成本函数（Cost Functions）。

本课程将对这六家平台进行建模分析，并告诉你各自在何时最具优势。

## 概念

### 三大细分板块

**定制芯片（Custom silicon）** — Groq (LPU)、Cerebras (WSE)、SambaNova (RDU)。在相同模型下，其解码（decode）速度通常比基于 GPU 的集群快 5-10 倍。每 token 价格较高（2025 年底 Groq 在 Llama-70B 上约为 0.99 美元/百万 token），但在延迟敏感型（latency-sensitive）用例中无可匹敌。Groq 是语音智能体（voice agents）和实时翻译的生产环境首选。

**GPU 平台（GPU platforms）** — Baseten、Together、Fireworks、Modal、Anyscale。运行于 NVIDIA（2026 年包括 H100、H200、B200）或偶尔使用 AMD 硬件。定位介于“裸 GPU 租赁”（RunPod、Lambda）与“超大规模云厂商托管服务”（Bedrock）之间的经济型层级。

**API 优先型市场（API-first marketplaces）** — Replicate、DeepInfra、OpenRouter、Fal。模型目录广泛，采用按预测次数或按秒计费，强调首次调用响应时间（time-to-first-call）。

### Fireworks —— 延迟优化型 GPU 平台

- FireAttention 引擎（自研）；官方宣称在同等配置下延迟比 vLLM 低 4 倍。
- 针对非交互式工作负载提供批量处理层级（Batch tier），价格约为无服务器（serverless）费率的 50%。
- 微调模型的推理费率与基础模型相同——这与那些对 LoRA 微调模型收取溢价的服务商形成了鲜明差异。
- 2026 年中：自 2026 年 5 月 1 日起，按需 GPU 租赁价格上调至 1 美元/小时。大规模用量可协商批量定价。
- 财务指标：估值 40 亿美元，日处理量超 10 万亿（10T+）token。

### Together —— 广度优化型平台

- 提供 200 多个模型，包括上游开源发布后数天内即可上线的模型。
- 在同等大语言模型（LLM）上比 Replicate 便宜 50%-70%——其“AI 原生云（AI Native Cloud）”定位主打规模与目录广度。
- 通过单一 API 集成推理、微调与训练功能。

### Baseten —— 企业级打磨优化型平台

- Truss 框架：将模型依赖、密钥（secrets）与服务配置统一打包至单个清单文件（manifest）中。
- GPU 型号覆盖从 T4 到 B200。采用按分钟计费，并具备合理的冷启动（cold-start）缓解机制。
- 符合 SOC 2 Type II 标准，支持 HIPAA 合规。是金融科技与医疗健康领域的常见选择。
- 估值 50 亿美元，2026 年 1 月完成 E 轮融资（3 亿美元，由 CapitalG、IVP 和 NVIDIA 参投）。

### Modal —— Python 原生优化型平台

- 采用纯 Python 实现基础设施即代码（Infrastructure-as-code）。只需使用 `@modal.function(gpu="A100")` 装饰函数，即可通过一条命令完成部署。
- 按秒计费。配合预热机制冷启动时间为 2-4 秒；小型模型可低于 1 秒。
- 2025 年完成 8700 万美元 B 轮融资，估值达 11 亿美元。在独立调研中开发者体验（Developer Experience）评分最高。

### Replicate —— 多模态广度型平台

- 按预测次数计费。图像、视频与音频模型的默认首选平台。
- 丰富的集成生态（Zapier、Vercel、CMS 插件等）。
- 在 LLM 的每 token 费率上竞争力较弱，但在多模态模型多样性上占据优势。

### Anyscale —— Ray 原生型平台

- 基于 Ray 构建；RayTurbo 是 Anyscale 自研的推理引擎（与 vLLM 竞争）。
- 最适合分布式 Python 工作负载，尤其适用于推理步骤仅为更大计算图（computation graph）中一个节点的场景。
- 提供托管型 Ray 集群；与 Ray AIR 和 Ray Serve 深度集成。

### 按 Token 计费 vs 按分钟计费 —— 各自适用场景

当工作负载对延迟不敏感且呈突发性时，按 token 计费（per-token）更为合理——用多少付多少。当利用率高且可预测时，按分钟计费（per-minute）更划算——一旦 GPU 达到饱和，其成本就会低于按 token 计费。

经验法则：当专用 GPU 的持续利用率超过约 30% 时，按分钟计费（如 Baseten、Modal）开始优于按 token 计费（如 Fireworks、Together）。低于该阈值时，按 token 计费更胜一筹，因为可以避免为空闲资源付费。

### 自研引擎才是真正的护城河

除 vLLM 和 SGLang 外，上述平台均宣称拥有自研引擎，如 FireAttention、RayTurbo 以及 Baseten 的推理栈。所谓“自研引擎”多少带有营销色彩——客观而言，vLLM 与 SGLang 占据了生产环境中约 80% 的开源推理份额，而平台层面的真正差异点在于开发者体验（DX）、计费归因（attribution）以及服务等级协议（SLAs）。

### 关键数据备忘

- Fireworks GPU 租赁：自 2026 年 5 月 1 日起上调至 1 美元/小时。
- Fireworks 宣称：同等配置下延迟比 vLLM 低 4 倍。
- Together：在 LLM 上比 Replicate 便宜 50%-70%。
- Baseten 估值：50 亿美元（2026 年 1 月 E 轮，融资 3 亿美元）。
- Modal 估值：11 亿美元（2025 年 B 轮）。
- 持续利用率超过约 30% 时，按分钟计费优于按 token 计费。

## 实践应用

`code/main.py` 在合成工作负载（synthetic workload）下对比了六家供应商的不同定价模型。脚本会输出每日成本（$/day）和有效百万 Token 成本（effective $/M tokens）。运行该脚本可找出按 Token 计费（per-token）与按分钟计费（per-minute）模式之间的盈亏平衡点。

## 交付成果

本课时将生成 `outputs/skill-inference-platform-picker.md`。根据工作负载特征（workload profile）、服务等级协议（SLA）和预算，该脚本会选定主推理平台（inference platform）并给出备选方案。

## 练习题

1. 运行 `code/main.py`。在单张 H100 上运行 70B 模型时，Baseten（按分钟计费）的持续利用率（sustained utilization）达到多少时会优于 Fireworks（按 Token 计费）？请自行推导交叉点（crossover），并与经验法则（rule of thumb）进行对比。
2. 你的产品同时提供图像生成、聊天和语音转文本服务。请为每种模态（modality）选择合适的平台，并指出用于统一它们的网关模式（gateway pattern）。
3. Fireworks 将你主用模型的价格上调了 1 美元/小时。如果 40% 的流量转移到批次层级（batch tier，享受 50% 折扣），请建模分析其对混合成本（blended cost）的影响。
4. 某受监管客户需要 SOC 2 Type II + HIPAA 合规以及专用 GPU（dedicated GPUs）。哪三家平台符合条件？其中哪一家在 FinOps（财务运营）方面最具优势？
5. 对比 Llama 3.1 70B 在 Fireworks 无服务器（serverless）、Together 按需（on-demand）、Baseten 专用（dedicated）和 Replicate API 上的每 1,000 次预测（predictions）成本。在每天 10 次预测时哪家最便宜？每天 10,000 次时呢？

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 定制芯片（Custom silicon） | “非 GPU 芯片” | Groq LPU、Cerebras WSE、SambaNova RDU —— 针对解码（decode）阶段优化 |
| FireAttention | “Fireworks 引擎” | 自定义注意力机制内核（attention kernel）；宣传称延迟比 vLLM 低 4 倍 |
| Truss | “Baseten 的格式” | 模型打包清单（manifest）；包含依赖项 + 密钥 + 服务配置 |
| 按 Token 计费（Per-token） | “API 定价” | 按消耗的 Token 数量收费；空闲时不产生费用 |
| 按分钟计费（Per-minute） | “专用定价” | 按实际 GPU 运行时间（wall-clock GPU time）收费；在高利用率下更具优势 |
| 按预测次数计费（Per-prediction） | “Replicate 定价” | 按模型调用次数收费；常见于图像/视频生成 |
| RayTurbo | “Anyscale 引擎” | 基于 Ray 的专有推理引擎；在 Ray 集群上与 vLLM 竞争 |
| 批次层级（Batch tier） | “五折优惠” | 以较低费率运行的非交互式队列；常见于 Fireworks 和 OpenAI |
| 按基础模型费率计费（Fine-tuned at base rate） | “Fireworks LoRA” | 对使用 LoRA 服务的请求按基础模型费率收费（差异化优势） |

## 扩展阅读

- [Fireworks 定价](https://fireworks.ai/pricing) — 按 Token 计费费率（Per-token Rates）、批量处理层级（Batch Tier）、GPU 租赁（GPU Rental）。
- [Baseten 定价](https://www.baseten.co/pricing/) — 按分钟计费费率（Per-minute Rates）、承诺容量（Committed Capacity）、企业级套餐（Enterprise Tiers）。
- [Modal 定价](https://modal.com/pricing) — 按秒计费的 GPU 费率（Per-second GPU Rates）及免费套餐（Free Tier）。
- [Together AI 定价](https://www.together.ai/pricing) — 模型目录（Model Catalog）及按 Token 计费费率。
- [Anyscale 定价](https://www.anyscale.com/pricing) — RayTurbo 与托管 Ray（Managed Ray）定价。
- [Northflank — Fireworks AI 替代方案](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 对比评估（Comparative Assessment）。
- [Infrabase — 2026 年 AI 推理 API 提供商](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商全景图（Vendor Landscape）。