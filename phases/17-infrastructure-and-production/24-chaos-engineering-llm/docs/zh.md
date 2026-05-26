# 面向大语言模型生产环境的混沌工程

> 到 2026 年，面向大语言模型（LLM）的混沌工程（Chaos Engineering）已发展成为一门独立的学科。在生产环境运行实验前的先决条件包括：明确定义的服务等级指标/目标（SLI/SLO）、涵盖追踪（Trace）+指标（Metric）+日志（Log）的可观测性（Observability）、自动化回滚（Automated Rollback）、运行手册（Runbooks）以及值班机制（On-call）。架构包含四个平面：控制平面（Control Plane，实验调度器）、目标平面（Target Plane，服务、基础设施、数据存储）、安全平面（Safety Plane，防护机制 + 中止机制 + 流量过滤器）、可观测性平面（Observability Plane，指标 + 追踪 + 日志），以及反馈回路（Feedback Loop，用于 SLO 调整）。必须设置防护护栏：若每日错误预算（Error Budget）消耗速率超过预期的 2 倍，则触发消耗速率告警（Burn-rate Alerts）并暂停实验；通过抑制窗口（Suppression Windows）与追踪 ID（Trace-ID）关联来消除告警噪声。执行节奏：每周进行小规模金丝雀发布（Canary）与 SLO 评审；每月开展演练日（Game Day）与事后复盘（Postmortem）；每季度进行跨团队韧性审计（Resilience Audit）与依赖关系映射（Dependency Mapping）。针对 LLM 的专项实验包括：内存过载、网络故障、服务提供商中断、格式错误的提示词（Prompt）、键值缓存（KV Cache）驱逐风暴。工具链：Harness Chaos Engineering（基于 LLM 的推荐、爆炸半径降级、MCP 工具集成）；LitmusChaos（CNCF 项目）；Chaos Mesh（CNCF 原生 Kubernetes 项目）。

**Type:** 学习
**Languages:** Python（标准库，简易混沌实验运行器）
**Prerequisites:** 第 17 阶段 · 23（面向 AI 的 SRE），第 17 阶段 · 13（可观测性）
**Time:** 约 60 分钟

## 学习目标

- 列出混沌工程的五项先决条件（SLI/SLO、可观测性、回滚机制、运行手册、值班机制），并解释为何跳过其中任何一项都会破坏该实践。
- 绘制四个架构平面（控制、目标、安全、可观测性）及其指向 SLO 的反馈回路图。
- 列举五项针对 LLM 的专项实验（内存过载、网络故障、提供商中断、格式错误的提示词、KV 缓存驱逐风暴）。
- 根据现有技术栈，从 Harness、LitmusChaos 或 Chaos Mesh 中选择合适的工具。

## 问题背景

传统技术栈中的混沌测试已相对成熟，但 LLM 技术栈引入了全新的故障模式。例如，包含恶意字符的 4K Token 提示词会导致分词器（Tokenizer）卡顿 12 秒；上游服务提供商返回 429 状态码（请求过多）；你的网关触发重试机制，导致重试放大的并发请求使服务发生内存溢出（OOM）；在突发负载下，KV 缓存驱逐风暴会引发重新预填充（Re-prefill）级联反应，从而耗尽计算资源。

这些故障均无法在单元测试中暴露。混沌工程正是帮助你在用户遭遇之前主动发现这些问题的关键手段。

## 核心概念

### 前置条件

在缺乏以下条件的情况下，切勿在生产环境中进行混沌工程（Chaos Engineering）测试：

1. **SLI/SLO**（服务级别指标/服务级别目标）—— 明确定义的服务级别指标与目标。
2. **可观测性**（Observability）—— 追踪（traces）、指标（metrics）、日志（logs）已接入仪表盘。
3. **自动回滚**（Automated rollback）—— 第 17 阶段 · 20 策略标志回滚。
4. **运行手册**（Runbooks）—— 结构化文档，第 17 阶段 · 23。
5. **值班响应**（On-call）—— 有专人负责响应。

缺少任何一项，混沌测试都可能演变为真实的生产事故。

### 四大平面 + 反馈机制

**控制平面**（Control plane）—— 实验调度器（如 Litmus 工作流、Chaos Mesh 调度、Harness UI）。

**目标平面**（Target plane）—— 服务、Pod、节点、负载均衡器、数据存储。

**安全平面**（Safety plane）—— 紧急开关（kill switch）、静默窗口、爆炸半径（blast-radius）限制、错误预算（error-budget）门控。

**可观测性平面**（Observability plane）—— 常规指标 + Trace-ID 关联，用于区分混沌测试引发的故障与自然故障。

**反馈循环**（Feedback loop）—— 将测试发现反馈至 SLO 调整、运行手册更新及代码修复中。

### 安全护栏是强制要求

- **消耗率告警**（Burn-rate alert）：若每日错误预算消耗超过预期值的 2 倍，则暂停实验。
- **静默窗口**（Suppression windows）：在实验期间，屏蔽爆炸半径内非实验相关的告警。
- **Trace-ID 关联**（Trace-ID correlation）：所有由实验引发的错误均携带特定标签，以便值班人员进行去重处理。

### 五项针对大语言模型（LLM）的专属实验

1. **内存过载**（Memory overload）—— 通过高并发发送长上下文请求，强制触发 KV 缓存（KV cache）抢占风暴。观察：服务是优雅降级（gracefully shed）还是直接崩溃？

2. **网络故障**（Network failure）—— 切断推理网关与模型提供商之间的连接。观察：故障转移（fallback）是否在 SLA 规定时间内生效？（第 17 阶段 · 19）

3. **提供商宕机模拟**（Provider outage simulation）—— 模拟 OpenAI 返回 100% 的 429 状态码。观察：路由是否故障转移至 Anthropic？（第 17 阶段 · 16, 19）

4. **畸形提示词**（Malformed prompt）—— 注入导致分词器（tokenizer）停滞的负载（例如深度嵌套的 Unicode 字符、超大 UTF-8 码点）。观察：单个请求是否会卡死工作进程（worker）？

5. **KV 驱逐风暴**（KV eviction storm）—— 通过耗尽 vLLM 块预算强制触发缓存驱逐。观察：LMCache 能否恢复，还是服务性能下降？

### 执行频率

- **每周**—— 在预发环境进行小规模金丝雀（Canary）实验，生产环境流量比例可设为 5%。
- **每月**—— 针对特定场景安排计划内的演练日（Game day）；跨团队参与；进行事后复盘（Postmortem）。
- **每季度**—— 跨团队韧性审计（Resilience audit）；更新依赖关系图（Dependency map）。

### 工具选型

- **Harness Chaos Engineering**—— 商业软件；提供 AI 驱动的实验推荐；支持爆炸半径动态缩减；集成 MCP 工具。
- **LitmusChaos**—— CNCF 毕业项目；基于 Kubernetes 工作流。
- **Chaos Mesh**—— CNCF 沙箱项目；采用 Kubernetes 原生 CRD 风格。
- **Gremlin**—— 商业软件；支持范围广泛。
- **AWS FIS** / **Azure Chaos Studio**—— 云厂商托管服务。

### 从小处着手

首次实验：在稳定流量下，对单个解码（decode）副本执行 Pod 终止（Pod-kill）操作。观察流量重路由与恢复情况。若实验顺利且表现安全，可逐步过渡到网络混沌测试。

首次 LLM 专属实验：模拟单一提供商返回 429 状态码，持续 5 分钟。观察故障转移机制。大多数团队会在此过程中发现其故障转移方案并未经过充分测试。

### 关键数据备忘

- 四大平面：控制、目标、安全、可观测性。
- 消耗率暂停阈值：每日预算消耗达到预期值的 2 倍。
- 执行频率：每周金丝雀实验、每月演练日、每季度审计。
- 五项 LLM 实验：内存过载、网络故障、提供商宕机、畸形提示词、KV 驱逐风暴。

## 动手实践
`code/main.py` 模拟了三个带有安全平面门控（safety plane gates）的混沌实验（chaos experiments）。报告会指出哪些实验会触发燃烧率中止（burn-rate abort）。

## 交付产出
本课时将生成 `outputs/skill-chaos-plan.md`。根据技术栈（tech stack）与系统成熟度（maturity），该脚本会筛选出前三个实验及对应的工具链。

## 练习题
1. 运行 `code/main.py`。哪个实验会触发燃烧率门控（burn-rate gate），原因是什么？
2. 为基于 vLLM 的检索增强生成（RAG）服务设计前五个混沌实验。需包含成功标准（success criteria）。
3. 你的燃烧率告警（burn-rate alert）暂停了一个实验。如何确定根本原因（root cause）——是混沌实验导致还是自然发生？
4. 论证混沌实验应在生产环境（production）运行还是仅在预发环境（staging）运行。何时选择生产环境才是正确的？
5. 列举三种通用网络混沌（generic network-chaos）无法复现的大语言模型（LLM）特有故障模式（failure modes）。

## 关键术语
| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| SLI / SLO（服务等级指标/服务等级目标） | “服务目标” | 指标与目标；必需的前置条件 |
| Blast radius（爆炸半径） | “影响范围” | 受实验影响的服务/用户集合 |
| Burn-rate alert（燃烧率告警） | “预算门控” | 当错误预算（error budget）消耗率超过预期 2 倍时触发 |
| Game day（演练日） | “月度演习” | 计划内的跨团队混沌演练 |
| LitmusChaos | “CNCF 工作流” | 已毕业的 CNCF Kubernetes 混沌工具 |
| Chaos Mesh | “CNCF CRD” | CNCF 沙箱阶段的 Kubernetes 原生混沌工具 |
| Harness CE | “商业 AI 辅助工具” | 集成 AI 推荐功能的 Harness 混沌工程平台 |
| Malformed prompt（畸形提示词） | “分词器炸弹” | 导致分词（tokenization）停滞的输入 |
| KV eviction storm（键值驱逐风暴） | “抢占级联” | 大规模键值缓存（KV cache）驱逐触发重新预填充（re-prefills） |

## 延伸阅读
- [DevSecOps School — 2026 混沌工程指南](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — 大语言模型可观测性（书籍）](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness 混沌工程](https://www.harness.io/products/chaos-engineering)
- [AWS FIS（故障注入模拟器）](https://aws.amazon.com/fis/)