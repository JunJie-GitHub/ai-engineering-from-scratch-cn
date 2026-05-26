# AI 网关（AI Gateways）— LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关（Gateway）位于你的应用程序与模型提供商之间。核心功能包括提供商路由（Provider Routing）、故障转移（Fallback）、重试（Retries）、速率限制（Rate Limiting）、密钥引用（Secret References）、可观测性（Observability）以及安全护栏（Guardrails）。2026 年的市场格局如下：**LiteLLM** 采用 MIT 开源协议（MIT Open Source Software），支持 100 多家提供商，兼容 OpenAI 接口，但在约 2000 RPS（每秒请求数）时会出现性能瓶颈（8 GB 内存下，已发布的基准测试中会出现级联故障）；最适合 Python 环境、<500 RPS 的场景以及开发/原型设计。**Portkey** 定位为控制平面（Control Plane）（提供安全护栏、个人身份信息脱敏（PII Redaction）、越狱检测（Jailbreak Detection）、审计追踪（Audit Trails）等功能），于 2026 年 3 月转为 Apache 2.0 开源，延迟开销（Latency Overhead）为 20-40 毫秒，生产环境套餐价格为 49 美元/月。**Kong AI Gateway** 基于 Kong Gateway 构建——Kong 官方在相同 12 核 CPU 上的基准测试显示：其速度比 Portkey 快 228%，比 LiteLLM 快 859%；定价为 100 美元/模型/月（Plus 套餐最多支持 5 个模型）；如果你已在使用 Kong 生态，则非常适合企业级部署。**Bifrost**（Maxim AI 出品）——支持可配置退避策略（Backoff）的自动重试，当 OpenAI 返回 429 错误时可自动回退至 Anthropic。**Cloudflare / Vercel AI Gateways**——全托管（Managed）、零运维，提供基础重试功能。数据驻留（Data Residency）要求通常是决定自托管（Self-Hosted）的关键因素；Portkey 和 Kong 则处于中间地带，提供开源版本及可选的托管服务。

**Type:** 学习
**Languages:** Python（标准库，简易网关路由模拟器）
**Prerequisites:** 第 17 阶段 · 01（托管 LLM 平台），第 17 阶段 · 16（模型路由）
**Time:** 约 60 分钟

## 学习目标

- 列举网关的六大核心功能（路由、故障转移、重试、速率限制、密钥管理、可观测性、安全护栏）。
- 将 2026 年的四款网关（LiteLLM、Portkey、Kong AI、Bifrost）与其扩展上限及适用场景进行对应映射。
- 引用 Kong 的基准测试数据（比 Portkey 快 228%，比 LiteLLM 快 859%），并解释该数据对 >500 RPS 场景的意义。
- 结合数据驻留要求与运维预算，在自托管与全托管方案之间做出选择。

## 问题背景

你的产品需要调用 OpenAI、Anthropic 以及自托管的 Llama 模型。每个提供商的 SDK、错误处理模型、速率限制和身份验证方案各不相同。你希望实现故障转移（例如当 OpenAI 返回 429 错误时自动尝试 Anthropic）、统一的凭证存储、集中的可观测性监控，以及按租户划分的速率限制。

如果在应用层重新实现这些逻辑，会导致每个服务都与所有提供商强耦合。引入网关层可以将这些逻辑整合到一个进程中，对外提供统一的 API（通常兼容 OpenAI 接口），并在后端将请求分发至各个提供商。

## 核心概念

### 六大核心特性

1. **提供商路由 (Provider routing)** — 通过单一 API 统一接入 OpenAI、Anthropic、Gemini 及自托管模型等。
2. **故障回退 (Fallback)** — 当遇到 429（限流）、5xx（服务器错误）或生成质量不达标时，自动切换至其他提供商重试。
3. **重试机制 (Retries)** — 支持指数退避 (Exponential backoff) 与最大重试次数限制。
4. **速率限制 (Rate limits)** — 支持按租户、按 API 密钥、按模型进行独立限流。
5. **密钥引用 (Secret references)** — 运行时从密钥库 (Vault) 动态拉取凭证，绝不硬编码在应用代码中。
6. **可观测性 (Observability)** — 集成 OpenTelemetry (OTel) 与生成式 AI 属性 (GenAI attributes)（Phase 17 · 13）+ 成本分摊 (Cost attribution)。
7. **安全护栏 (Guardrails)** — 支持个人身份信息 (PII) 脱敏、越狱攻击检测 (Jailbreak detection) 及允许主题过滤。

### LiteLLM — MIT 开源协议，Python 实现

- 支持 100+ 提供商，兼容 OpenAI 接口，提供路由配置、故障回退及基础可观测性。
- 在 Kong 的基准测试中，约 2000 RPS（每秒请求数）时性能崩溃；内存占用约 8 GB，在持续高负载下易引发级联故障 (Cascading failures)。
- 最佳适用场景：Python 应用、<500 RPS、开发/预发环境网关、实验性路由。
- 成本：开源版免费；提供云端免费套餐。

### Portkey — 控制平面 (Control plane) 定位

- 自 2026 年 3 月起采用 Apache 2.0 开源协议。内置安全护栏、PII 脱敏、越狱检测及审计追踪 (Audit trails)。
- 单次请求延迟开销为 20-40 毫秒。
- 生产版定价 49 美元/月，包含数据留存与服务等级协议 (SLA)。
- 最佳适用场景：需要安全护栏与可观测性一体化方案的受监管行业。

### Kong AI Gateway — 面向大规模场景

- 基于 Kong Gateway 构建（成熟的 API 网关产品，采用 Lua + OpenResty 技术栈）。
- Kong 官方在等效 12 核 CPU 环境下的基准测试显示：性能比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：100 美元/模型/月，Plus 套餐最多支持 5 个模型。
- 最佳适用场景：已在使用 Kong 生态、>1000 RPS、愿意采购商业授权的企业。

### Bifrost (Maxim AI)

- 支持自动重试，退避策略可配置。
- 当 OpenAI 触发 429 限流时自动回退至 Anthropic 是其经典配置方案。
- 市场新入局者；商业闭源产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 全托管、零运维。提供基础重试与可观测性。
- 最佳适用场景：部署在 Cloudflare/Vercel 边缘网络上的 JavaScript 应用。
- 在安全护栏与速率限制方面，功能较 Kong/Portkey 有所局限。

### 自托管 vs 全托管

数据驻留 (Data residency) 是核心驱动因素。医疗与金融行业默认选择自托管（LiteLLM、Portkey 开源版或 Kong）。消费级产品默认选择全托管（Cloudflare AI Gateway）或中间层托管（Portkey 托管版）。混合架构：受监管租户采用自托管，其他租户采用托管服务。

### 延迟预算

- LiteLLM：典型延迟开销为 5-15 毫秒。
- Portkey：延迟开销为 20-40 毫秒。
- Kong：延迟开销为 3-8 毫秒。
- Cloudflare/Vercel：延迟开销为 1-3 毫秒（具备边缘计算优势）。

网关延迟会直接叠加至首字生成时间 (Time to First Token, TTFT)。若要求 TTFT P99 < 100 毫秒的服务等级协议 (SLA)，应选择 Kong 或 Cloudflare。若要求 P99 < 500 毫秒，上述方案均可满足。

### 速率限制语义至关重要

简单的令牌桶算法 (Token-bucket) 适用于中等规模场景。多租户架构需采用滑动窗口 (Sliding-window) + 突发流量配额 (Burst allowance) + 按租户分级策略。LiteLLM 默认提供令牌桶；Kong 提供滑动窗口；Portkey 提供分级限流。

### 网关 + 可观测性 + 路由的组合架构

Phase 17 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产环境中属于同一架构层。建议选择一款能同时覆盖这三项功能的工具，或进行精细的组件集成：2026 年的主流部署方案通常将 Helicone（可观测性）或 Portkey（安全护栏）与 Kong（高并发扩展）组合使用，以实现职责分离。

### 关键数据速记

- LiteLLM：约 2000 RPS 时性能崩溃，内存占用 8 GB。
- Portkey：20-40 毫秒延迟开销；自 2026 年 3 月起采用 Apache 2.0 协议。
- Kong：性能比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：100 美元/模型/月，Plus 套餐上限 5 个模型。
- Cloudflare/Vercel：边缘节点延迟开销仅 1-3 毫秒。

## 实践应用
`code/main.py` 模拟了在注入 429/5xx 错误时，跨 3 个服务提供商（Provider）的网关（Gateway）路由与回退（Fallback）机制。该脚本会报告延迟（Latency）、重试率（Retry Rate）和回退命中率（Fallback Hit Rate）。

## 交付产出
本课时将生成 `outputs/skill-gateway-picker.md` 文件。根据业务规模、运维态势（Ops Posture）、合规要求及延迟预算（Latency Budget），该工具可帮助选择合适的网关。

## 练习题
1. 运行 `code/main.py`。配置从 OpenAI→Anthropic→自托管（Self-hosted）的回退链路。在 5% 的提供商错误率下，预期的回退命中率是多少？
2. 你的服务等级协议（Service Level Agreement, SLA）要求首字延迟（Time To First Token, TTFT）的 P99 值在 300 ms 基线基础上低于 200 ms。哪些网关能保持在预算范围内？
3. 某医疗行业客户要求采用自托管架构 + 个人身份信息（Personally Identifiable Information, PII）脱敏 + 审计日志。请选择 Portkey OSS 或 Kong。
4. 对比 LiteLLM 与 Kong：团队应在达到何种每秒请求数（Requests Per Second, RPS）上限时进行迁移？
5. 为多租户 SaaS 设计限流（Rate-limit）策略：免费层、试用层、付费层。应选择令牌桶（Token-bucket）还是滑动窗口（Sliding-window）算法？

## 核心术语
| 术语 | 通俗叫法 | 实际含义 |
|------|----------|----------|
| 网关（Gateway） | "API 代理" | 位于应用程序与提供商之间的处理进程 |
| LiteLLM | "MIT 协议那个" | Python 开源项目，支持 100+ 提供商，在 2K RPS 时会出现性能瓶颈 |
| Portkey | "护栏网关" | 控制平面 + 可观测性，Apache 2.0 协议 |
| Kong AI Gateway | "高扩展那个" | 基于 Kong Gateway 构建，基准测试领先者 |
| Bifrost | "Maxim 的网关" | 重试机制 + Anthropic 回退方案 |
| Cloudflare AI Gateway | "边缘托管" | 部署在边缘的托管网关，零运维 |
| PII 脱敏（PII Redaction） | "数据清洗" | 在发送至模型前，使用正则表达式 + 命名实体识别（Named Entity Recognition, NER）进行掩码处理 |
| 越狱检测（Jailbreak Detection） | "提示词注入防护" | 针对用户输入的分类器模型 |
| 审计追踪（Audit Trail） | "合规日志" | 每次大语言模型（Large Language Model, LLM）调用的不可变记录 |
| 令牌桶（Token-bucket） | "简单限流" | 基于令牌补充机制的限流器 |
| 滑动窗口（Sliding-window） | "精确限流" | 基于时间窗口的限流器；公平性更佳 |

## 延伸阅读
- [Kong AI Gateway 基准测试](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — 2026 年 AI 网关对比](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — 2026 年顶级 LLM 网关工具](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway 文档](https://docs.konghq.com/gateway/latest/ai-gateway/)