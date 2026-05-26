# 多区域大语言模型（LLM）服务与键值（KV）缓存局部性

> 轮询负载均衡（Round-Robin Load Balancing）对带缓存的大语言模型推理具有实质性的危害。如果请求未能命中持有其前缀（Prefix）的节点，就必须承担完整的预填充（Prefill）成本——在长提示词下，P50 延迟约为 800 毫秒，而缓存命中（Cache Hit）时仅需约 80 毫秒。到 2026 年，生产环境的典型模式是采用缓存感知路由器（Cache-Aware Router，如 Rust 编写的 vLLM Router 或 llm-d router），它通过消费 KV 缓存事件（KV-Cache Events）并基于前缀哈希匹配（Prefix-Hash Match）进行路由。近期研究（GORGO）已将跨区域网络延迟（Cross-Region Network Latency）作为显式项纳入路由目标（Routing Objective）中。商业化的“跨区域推理”服务（如 Bedrock cross-region inference、GKE multi-cluster gateways）将推理视为黑盒——它们主要处理可用性，而非首字延迟（Time To First Token, TTFT）。摩根大通和梅奥诊所于 2024 年 11 月进行了 us-east-1 区域的故障转移（Failover），耗时约 22 分钟。灾难恢复（Disaster Recovery, DR）的现实情况是：32% 的大语言模型灾难恢复失败，是因为团队备份了模型权重（Model Weights），却遗漏了分词器文件（Tokenizer Files）或量化配置（Quantization Configs）。

**Type:** 学习
**Languages:** Python（标准库，玩具级前缀缓存感知路由器模拟器）
**Prerequisites:** 第 17 阶段 · 04（vLLM Serving），第 17 阶段 · 06（SGLang RadixAttention）
**Time:** 约 60 分钟

## 学习目标

- 解释为何轮询负载均衡会破坏缓存推理，并量化其对首字延迟（TTFT）的惩罚。
- 绘制缓存感知路由器的架构图：输入（KV 缓存事件）、算法（前缀哈希匹配）、决胜条件（GPU 利用率）。
- 指出导致 32% 大语言模型灾难恢复失败的核心原因（缺失分词器文件/量化配置），并制定一份包含三个关键文件的灾难恢复检查清单。
- 区分商业跨区域服务（Bedrock CRI、GKE Multi-Cluster Gateway）与 KV 感知路由。

## 问题背景

您的服务部署在 us-east-1、us-west-2 和 eu-west-1 区域。您在服务前端部署了采用轮询策略的应用负载均衡器（Application Load Balancer, ALB）。生产环境中的前缀缓存命中率骤降至 8%。TTFT P50 延迟增加了两倍。您的 vLLM 日志显示，每个请求都在承担完整的预填充成本。

轮询策略对于无状态服务（Stateless Services）是最优的。但大语言模型推理在设计上是有状态的（Stateful）——KV 缓存编码了模型处理过的所有上下文。盲目路由等同于将请求错误地导向未缓存相关上下文的节点。

此外，您的团队制定了灾难恢复计划。您将模型权重跨区域备份至 S3。当某个区域发生中断时，您尝试进行故障转移，但副本节点拒绝启动。您忘记了 `tokenizer.json`、量化配置以及 RoPE 缩放配置位于一个您未同步的独立存储桶中。

多区域大语言模型服务本质上是一个缓存问题、一个路由问题以及一个灾难恢复基础维护（DR Hygiene）问题，而非单纯的负载均衡器问题。

## 核心概念

### 缓存感知路由 (Cache-aware routing)

请求携带提示词（prompt）到达。路由器对前缀进行哈希计算（例如前 512 个 token），并向每个副本（replica）查询：“你是否缓存了该前缀？”。副本在分配和淘汰（evict）内存块时，会通过发布/订阅（pub/sub）通道发布 KV 缓存（KV-cache）事件。路由器将请求路由至匹配的副本；若均无匹配，则回退至基于 GPU 利用率（GPU utilization）的调度策略。

**vLLM Router**（Rust，2026 生产栈）：订阅 `kv.cache.block_added` 事件，维护前缀哈希（prefix-hash）到副本索引的映射，以 O(1) 查找复杂度进行路由。无匹配时回退至队列深度最小（least-queue-depth）策略。

**llm-d router**：采用相同模式，原生支持 Kubernetes。通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（Phase 17 · 06）是副本内（intra-replica）的等效实现。跨副本（cross-replica）路由则严格由上游组件处理。

### 性能数据

在 2K token 提示词、Llama 3.3 70B FP8 模型及 H100 硬件下的首字延迟（Time To First Token, TTFT）P50 值：
- 缓存命中（同一副本，前缀驻留）：约 80 毫秒。
- 缓存未命中（冷预填充）：约 800 毫秒。

两者相差约 10 倍。若你的路由器在跨副本场景下的前缀缓存命中率达到 60-80%，即可在 N 副本容量下逼近单副本的性能表现。若命中率仅为 10%，则性能表现将接近简单的线性扩展（naive scaling）。

### 跨区域场景的新约束——网络延迟

跨区域往返时间（Round-Trip Time, RTT）：
- us-east-1 ↔ us-west-2：约 65 毫秒。
- us-east-1 ↔ eu-west-1：约 75 毫秒。
- us-east-1 ↔ ap-southeast-1：约 220 毫秒。

若将来自 us-east-1 的请求路由至 ap-southeast-1 的热前缀，所节省的预填充（prefill）时间（800 → 80 毫秒）将被 440 毫秒的往返延迟完全抵消。GORGO（2026 年研究）明确指出了这一点——应联合最小化 `prefill_time + network_latency`，而非仅优化预填充时间。通常的解决方案是保持区域内路由，除非遇到预填充时间占主导地位的超大型（多 MB）前缀。

### 商业化的“跨区域推理”在此场景下无济于事

AWS Bedrock 的跨区域推理功能会在容量压力期间自动将请求路由至其他区域。它优化的是可用性（availability），而非 TTFT，且将推理过程视为黑盒。GKE Multi-Cluster Gateway 同理——仅提供服务级故障转移（failover），不具备 KV 缓存感知能力。

即使使用这些服务，你仍然需要在应用层部署缓存感知路由器。它们负责处理“us-east-1 区域宕机”等极端故障场景，而缓存感知路由则专门用于优化 TTFT。

### 灾难恢复（Disaster Recovery, DR）规范——32% 的文件缺失问题

2026 年广泛引用的一项统计数据表明：32% 的大语言模型（LLM）灾难恢复失败，是因为团队仅备份了模型权重（weights），却遗漏了以下文件：

- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`、AWQ 缩放因子、GPTQ 零点值）
- 模型专属配置（RoPE 缩放、注意力掩码、对话模板）
- 推理引擎配置（`vllm_config.yaml`、采样默认参数、LoRA 适配器清单）

解决方案是遵循最低限度的三文件灾难恢复清单：

1. Hugging Face 模型仓库下的所有文件（权重 + 配置 + 分词器）。
2. 推理引擎专属的服务配置。
3. 部署清单（K8s YAML、Dockerfile、依赖锁定文件）。

此外：每季度执行一次灾难恢复演练。摩根大通（JPMorgan）在 2024 年 11 月的 us-east-1 区域演练中之所以能在 22 分钟内完成恢复，完全得益于对应急预案（playbook）的反复排练。

### 数据驻留（Data Residency）要求是独立维度

欧盟客户的受保护健康信息（Protected Health Information, PHI）严禁出境。若你的缓存感知路由器为了匹配前缀，将源自巴黎的请求发送至 us-east-1，无论 TTFT 提升多少，都已违反《通用数据保护条例》（GDPR）。在优化缓存命中率之前，必须先按数据驻留边界对路由器进行分区。

### 需牢记的关键数据

- 缓存命中与未命中的 TTFT 差距：约 10 倍（2K 提示词下为 80 毫秒 vs 800 毫秒）。
- 美欧跨区域往返时间（RTT）：约 75 毫秒。
- 灾难恢复失败原因：32% 遗漏分词器/量化配置。
- 摩根大通 us-east-1 故障转移（2024 年 11 月）：22 分钟（SLA 要求为 30 分钟）。

## Use It

`code/main.py` simulates three routing strategies (round-robin, cache-aware regional, cache-aware global) on a multi-region workload. Reports cache hit rate, TTFT P50/P99, and cross-region bill.

## Ship It

This lesson produces `outputs/skill-multi-region-router.md`. Given regions, residency constraints, and SLA, designs a routing plan.

## Exercises

1. Run `code/main.py`. At what prompt length does cross-region routing beat local-only routing, given 75 ms RTT?
2. Your cache hit rate drops from 70% to 12%. Diagnose three possible causes and the observables that would confirm each.
3. Design a DR manifest for a 70B AWQ-quantized model served in vLLM with 5 LoRA adapters. List every file and config.
4. Argue whether Bedrock cross-region inference is "enough" for a fintech with strict TTFT SLOs. Cite specific behaviors.
5. A Paris-origin request matches a prefix in us-east-1. Do you route it? Write the policy.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Cache-aware routing | "smart LB" | Route on prefix-hash match to KV-cache-holding replica |
| KV-cache events | "cache pub-sub" | Replicas publish block add/evict; router indexes |
| Prefix hash | "cache key" | Hash of first N tokens used as router lookup |
| GORGO | "cross-region routing research" | arXiv 2602.11688; network latency as explicit term |
| Cross-region inference | "Bedrock CRI" | AWS product; availability failover, not TTFT awareness |
| DR manifest | "the backup list" | Every file needed to restore — not just weights |
| Data residency | "GDPR boundary" | Legal constraint on which region sees user data |
| RTT | "round-trip time" | Network latency; 75 ms US-EU, 220 ms US-APAC |
| LLM-aware LB | "cache-hit LB" | Cache-aware router as a product category |

## Further Reading

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — cross-region KV-cache reuse with network latency term.
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — availability failover documentation.
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — cache-aware router source.