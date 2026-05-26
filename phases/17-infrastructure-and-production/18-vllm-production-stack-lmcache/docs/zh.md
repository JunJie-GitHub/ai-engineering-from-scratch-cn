# 集成 LMCache KV 卸载的 vLLM 生产栈

> vLLM 的生产栈（production-stack）是参考级 Kubernetes 部署方案，将路由器（router）、推理引擎（engines）和可观测性（observability）组件集成在一起。LMCache 是 KV 卸载（KV-offloading）层，负责将 KV 缓存（KV cache）从 GPU 内存中提取出来，并在不同查询和引擎之间复用（优先使用 CPU 内存（CPU DRAM），其次为磁盘/Ceph）。vLLM 0.11.0 的 KV 卸载连接器（KV Offloading Connector，2026 年 1 月发布）通过连接器 API（Connector API，v0.9.0+）实现了该过程的异步化与插件化。卸载延迟对用户不可见。即使没有共享前缀（shared prefixes），LMCache 依然具有重要价值：当 GPU 的 KV 槽位（KV slots）耗尽时，被抢占的请求（preempted requests）可以直接从 CPU 恢复，而无需重新计算预填充（prefill）。在 4 台 a3-highgpu-4g 实例（共 16 张 H100 GPU，每张 80GB 高带宽内存（HBM））上发布的基准测试表明：当 KV 缓存超出 HBM 容量时，原生 CPU 卸载和 LMCache 均能显著提升吞吐量；在 KV 占用较低时，所有配置的性能均与基线持平，仅带来极小开销。

**类型：** 学习
**语言：** Python（标准库，简易 KV 溢出模拟器）
**前置要求：** 第 17 阶段 · 04（vLLM 服务内部机制），第 17 阶段 · 06（SGLang/RadixAttention）
**预计耗时：** 约 60 分钟

## 学习目标

- 绘制 vLLM 生产栈的架构图层：路由器、推理引擎、KV 卸载与可观测性组件。
- 解释 KV 卸载连接器 API（v0.9.0+）的工作原理，以及 0.11.0 版本中的异步路径如何隐藏卸载延迟。
- 量化评估 LMCache 使用 CPU 内存的适用场景（KV 缓存 > HBM 容量）与引入额外开销的场景（KV 缓存较小，可完全装入 HBM）。
- 根据实际部署约束，在原生 vLLM CPU 卸载与 LMCache 连接器之间做出选择。

## 核心问题

当并发量上升时，你的 vLLM 服务会出现 GPU 高带宽内存（HBM）占用率达到 100% 并触发请求抢占（preemption）的情况。请求被逐出、重新排队，导致同一 2K 词元（token）的提示词（prompt）在一分钟内被重复预填充四次。GPU 算力被浪费在冗余的预填充计算上，有效吞吐量（goodput）远低于原始吞吐量。

增加 GPU 会带来线性成本增长，而增加 HBM 容量在硬件上不可行。但 CPU 内存（CPU DRAM）成本较低——单路 CPU 可配备 512 GB 以上内存，虽然其延迟比 HBM 高出几个数量级，但足以胜任“临时温热”状态 KV 缓存的存储。

LMCache 将 KV 缓存提取至 CPU 内存，使被抢占的请求能够快速恢复，同时跨引擎的重复前缀可共享缓存，避免每个引擎重复执行预填充。

## 核心概念

### vLLM 生产栈 (Production Stack)
`github.com/vllm-project/production-stack` 是标准的 Kubernetes (Kubernetes) 部署参考实现：
- **路由器 (Router)** — 具备缓存感知能力（第 17 阶段 · 11）。负责消费 KV 事件。
- **引擎 (Engines)** — vLLM 工作进程。每个 GPU 或每个张量并行/流水线并行 (TP/PP) 组部署一个。
- **KV 缓存卸载 (KV Cache Offload)** — 采用 LMCache 部署或原生连接器。
- **可观测性 (Observability)** — Prometheus 指标抓取、Grafana 仪表盘、OpenTelemetry (OTel) 链路追踪。
- **控制平面 (Control Plane)** — 服务发现、配置管理与滚动更新。
以 Helm Chart 与 Operator 形式交付。

### KV 卸载连接器 API (KV Offloading Connector API) (v0.9.0+)
vLLM 0.9.0 引入了连接器 API (Connector API)，用于支持可插拔的 KV 缓存后端。引擎将缓存块卸载至连接器，由连接器负责存储（内存、磁盘、对象存储或 LMCache）。当请求需要某个缓存块时，连接器会将其重新加载回引擎。
vLLM 0.11.0（2026 年 1 月）新增了异步卸载路径 (Asynchronous Offload Path)——卸载操作可在后台执行，因此在常规情况下引擎不会因此阻塞。端到端延迟 (End-to-End Latency) 和吞吐量仍取决于工作负载特征、KV 缓存命中率以及系统压力；vLLM 官方文档明确指出，在命中率较低时，自定义内核 (Custom-Kernel) 卸载可能会降低吞吐量，且异步调度与投机解码 (Speculative Decoding) 之间存在已知的交互问题。

### 原生 CPU 卸载与 LMCache 对比
**原生 vLLM CPU 卸载 (Native vLLM CPU Offload)**：引擎本地化。将 KV 块存储在主机内存中。实现迅速，无网络跳转开销。无法跨引擎共享。
**LMCache 连接器 (LMCache Connector)**：集群级规模。将缓存块存储在共享的 LMCache 服务器中（CPU DRAM + Ceph/S3 存储层）。任何引擎均可访问这些缓存块。已发布基于 16 张 H100 的基准测试数据。
当单个引擎面临高带宽内存 (HBM) 压力时，选择原生方案。当多个引擎需要共享前缀时（例如使用通用系统提示词的检索增强生成 (RAG) 场景，或共享模板的多租户场景），选择 LMCache。

### 基准测试表现
在 4 台 `a3-highgpu-4g` 实例上分布部署 16 张 H100（80 GB HBM）的测试结果如下：
- KV 占用较低（短提示词、低并发）：所有配置均与基线持平，LMCache 会引入约 3-5% 的额外开销。
- KV 占用中等：LMCache 开始通过跨引擎的前缀复用带来性能提升。
- KV 占用超出 HBM 容量：原生 CPU 卸载与 LMCache 均能显著提升吞吐量；由于支持跨引擎共享，LMCache 的收益更大。

### LMCache 发挥关键作用的场景
- 多租户服务场景：不同租户共享相同的系统提示词。
- 检索增强生成 (RAG) 场景：不同查询重复访问相同的文档分块。
- 基于同一基座模型的微调变体（如 LoRA）场景：复用基座模型的 KV 缓存可大幅减少冗余计算。
- 抢占频繁的工作负载：从 CPU 恢复缓存的成本远低于重新执行预填充 (Pre-fill)。

### 不建议启用的场景
- HBM 压力较小：此时启用只会带来额外开销，无法获得收益。
- 上下文较短（<1K 个 Token）：数据传输耗时将超过重新预填充的时间。
- 单租户单提示词工作负载：不存在可复用的缓存内容。

### 与分离式服务架构的集成
第 17 阶段 · 17 的分离式服务 (Disaggregated Serving) 与 LMCache 结合可产生叠加效应：从预填充池传输至解码池的 KV 数据若未被立即使用，将暂存于 LMCache 中；后续查询可直接从 LMCache 拉取。第 17 阶段 · 11 的缓存感知路由器可将请求路由至本地缓存或 LMCache 共享缓存中已包含所需数据的引擎。

### 关键数据备忘
- vLLM 0.9.0：正式提供连接器 API。
- vLLM 0.11.0（2026 年 1 月）：引入异步卸载路径；端到端延迟的影响取决于工作负载、KV 命中率及系统压力（并非绝对保证）。
- 16x H100 基准测试：当 KV 占用超出 HBM 容量时，LMCache 能有效提升性能。
- HBM 压力较小时：会引入 3-5% 的额外开销且无实际收益。

## 使用它
`code/main.py` 模拟了在使用与未使用 LMCache 时的高抢占（preemption-heavy）工作负载。脚本会报告避免的重新预填充（re-prefill）次数、吞吐量（throughput）提升幅度，以及达到盈亏平衡点的高带宽内存（HBM）利用率。

## 部署上线
本课时将生成 `outputs/skill-vllm-stack-decider.md`。该文件会根据工作负载形态（workload shape）与 vLLM 部署情况，给出采用原生方案（native）、LMCache 或均不采用的决策建议。

## 练习
1. 运行 `code/main.py`。LMCache 在何种 HBM 利用率下开始产生收益？
2. 某租户在每小时 200 次查询中共享一个 6K token 的系统提示词（system prompt）。计算每个租户预期的 LMCache 节省量。
3. LMCache 服务器存在单点故障（single point of failure）风险。请设计高可用（HA）策略（副本、回退至原生方案）。
4. LMCache 将数据存储在使用机械硬盘（spinning disk）的 Ceph 上。对于 70B FP8 格式的 4K token KV 缓存（500 MB），其读取时间与重新预填充（re-prefill）时间相比如何？
5. 论证 vLLM 0.11.0 的异步路径（asynchronous path）是否真正“零开销”——其隐藏开销（overhead）位于何处？

## 关键术语
| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 生产栈（Production-stack） | “参考部署方案” | vLLM 的 Kubernetes Helm Chart + Operator |
| 连接器 API（Connector API） | “KV 后端接口” | vLLM 0.9.0+ 可插拔的 KV 存储接口 |
| 原生 CPU 卸载（Native CPU offload） | “引擎本地溢出” | 将 KV 缓存存储在同一引擎的主机内存（RAM）中 |
| LMCache | “集群 KV 缓存” | 基于 CPU DRAM + 磁盘的跨引擎 KV 缓存服务器 |
| 0.11.0 异步路径（0.11.0 async） | “非阻塞卸载” | 隐藏在引擎流（engine stream）背后的卸载操作 |
| 抢占（Preemption） | “驱逐以腾出空间” | HBM 满载时的 KV 缓存置换（shuffle） |
| 前缀复用（Prefix reuse） | “相同的系统提示词” | 多个查询共享开头部分；触发缓存命中 |
| Ceph 层级（Ceph tier） | “磁盘层级” | 缓存层级中位于 DRAM 之下的持久化存储 |

## 延伸阅读
- [vLLM 博客 — KV 卸载连接器（2026年1月）](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm Chart + Operator。
- [面向企业级大语言模型推理的 LMCache（arXiv:2510.09665）](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — 连接器实现代码。
- [vLLM 0.11.0 发布说明](https://github.com/vllm-project/vllm/releases) — 异步路径详情。