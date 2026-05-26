---
name: vllm-stack-decider
description: 根据工作负载 (workload) 与节点集群 (fleet) 规模，制定 vLLM 部署架构方案——涵盖生产栈 (production-stack) Helm Chart、KV 缓存卸载 (KV offload，原生 CPU 或 LMCache) 以及路由器/可观测性集成。
version: 1.0.0
phase: 17
lesson: 18
tags: [vllm, production-stack, lmcache, kv-offload, connector-api]
---

根据给定的工作负载 (workload)（提示词特征 (prompt shape)、并发量 (concurrency)、前缀复用模式 (prefix reuse pattern)）、节点集群 (fleet)（推理实例 (engines) 数量、GPU 类型）以及运维环境（Kubernetes 原生 (Kubernetes-native)、多租户 (multi-tenant)、预算），制定一份 vLLM 技术栈部署方案。

输出内容：

1. **技术栈 (Stack)**。使用 vLLM 生产栈 (production-stack) Helm Chart（推荐用于新部署）或自行构建。说明适用的控制器/自定义资源定义 (Operators/CRDs)。
2. **KV 缓存卸载 (KV offload)**。选择以下方案：
   - 无（提示词较短、并发量低——开销大于收益）。
   - vLLM 原生 CPU 卸载 (Native vLLM CPU offload)（单实例高带宽内存 (HBM) 压力较大，配置简单）。
   - LMCache 连接器 (LMCache connector)（多实例前缀复用、请求抢占 (preemption) 频繁，或多租户共享提示词场景）。
3. **高带宽内存利用率监控 (HBM utilization monitoring)**。设置 `--gpu-memory-utilization` 参数并预留余量；当持续利用率达到 92% 以上时触发告警，作为请求抢占的前置信号。
4. **路由器集成 (Router integration)**。接入缓存感知路由器 (Cache-aware router)（第 17 阶段 · 第 11 课）。确认 KV 事件通道 (KV-event channel) 已配置。
5. **可观测性 (Observability)**。为每个推理实例配置 Prometheus 数据抓取 (Prometheus scrape)，集成 OpenTelemetry 生成式 AI 属性 (OTel GenAI attributes)（第 17 阶段 · 第 13 课），并使用生产栈提供的 Grafana 仪表板模板。
6. **预期影响 (Expected impact)**。量化相较于当前方案的预期吞吐量 (throughput) 提升——参考 16 倍 H100 基准测试配置（当 KV 缓存占用超过 HBM 容量时，LMCache 将发挥显著作用）。

硬性拒绝条件：
- 在未共享前缀或无请求抢占场景下部署 LMCache。拒绝——徒增开销，毫无收益。
- 在未监控 HBM 压力的情况下运行 vLLM。拒绝——首次请求抢占将导致意外故障。
- 在 Helm Chart 已覆盖用例的情况下仍手动构建生产栈。拒绝——避免重复造轮子的成本。

拒绝规则：
- 若集群推理实例数量少于 2 个，拒绝使用 LMCache——其核心价值在于跨实例复用；单实例场景请使用原生方案。
- 若工作负载的提示词长度小于 1K token 且并发量低于 100，拒绝任何形式的卸载——HBM 预留余量已足够。
- 若团队不具备 Kubernetes (K8s) 运维能力，拒绝使用生产栈——应从单实例 vLLM 加简单代理 (proxy) 起步。

输出要求：一份单页方案，明确技术栈选型、KV 卸载策略、HBM 监控配置、路由器集成方式、可观测性方案及预期影响。结尾需附上唯一验收标准：过去 24 小时 HBM 利用率的第 99 百分位数 (P99)。