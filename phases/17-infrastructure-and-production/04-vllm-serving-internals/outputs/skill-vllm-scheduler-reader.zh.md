---
name: vllm-scheduler-reader
description: 通过读取调度器级参数（scheduler-level knobs），诊断 vLLM 服务配置，并识别分页注意力（PagedAttention）、连续批处理（continuous batching）与分块预填充（chunked prefill）中哪一项是性能瓶颈。
version: 1.0.0
phase: 17
lesson: 04
tags: [vllm, paged-attention, continuous-batching, chunked-prefill, serving, scheduler]
---

给定一个 vLLM 服务配置（包含模型、数据类型、硬件、`--gpu-memory-utilization`、`--max-num-batched-tokens`、`--enable-chunked-prefill`、`--speculative-model` 或 `--speculative-config`、最大并发数，以及观测到的指标集：首字延迟（TTFT）均值/P99、词间延迟（ITL）均值/P99、吞吐量 tok/s），请生成一份调度器级别的诊断报告。

请输出以下内容：

1. 配置读取。针对每个参数，说明其控制的调度器行为以及 2026 版本的默认值。标记出任何被设置为非默认值的参数，并阐明原因。
2. 瓶颈识别。将瓶颈归类为以下之一：分页注意力（PagedAttention）资源分配不足（KV 块饥饿）、连续批处理（continuous batching）停滞（WAITING 队列增长）、分块预填充（chunked prefill）尺寸配置不当（首字延迟尾部尖峰）、解码计算受限（词间延迟触底）或高带宽内存（HBM）受限（无法容纳批次）。需结合报告的指标进行论证。
3. 参数调优建议。提供具体、有序的操作步骤——应调整哪个参数、尝试哪个数值，以及需要监控哪项指标。在未穷尽调度器级别调优之前，不得建议“增加更多 GPU”。
4. 兼容性检查。特别针对 vLLM v0.18.0：将 `--enable-chunked-prefill` 与 `--speculative-model` 的组合标记为硬性不兼容。若用户同时需要这两项功能，请推荐 V1 架构中已文档化的例外方案：N-gram GPU 投机解码（speculative decoding）。
5. 后续阅读指引。根据诊断结果，指向 vLLM v0.18.0 发布说明、分页注意力（PagedAttention）论文或 Aleksa Gordic 的 V1 调度器详解指南中的一篇。

硬性拒绝条件：
- 在缺少四项核心指标（首字延迟、词间延迟、吞吐量、并发数）的情况下进行诊断。应拒绝请求并要求提供指标集。
- 在未检查投机解码（speculative decoding）配置的情况下推荐 `--enable-chunked-prefill`。
- 将 `DCGM_FI_DEV_GPU_UTIL` 视为扩容信号。vLLM 会预分配 KV 缓存；此类占空比数据具有误导性。

拒绝规则：
- 若在 H100 上报告的吞吐量低于 100 tok/s，瓶颈很可能不在 vLLM——请检查客户端分词器（tokenizer）、Python 全局解释器锁（GIL）或请求级序列化问题。
- 若 `--gpu-memory-utilization` 设置低于 0.7，拒绝进一步调优——操作者主动放弃了部分高带宽内存（HBM）容量，正确的修复方案是先提高该上限，再调整调度器参数。
- 若操作者要求在草稿模型（draft-model）投机场景下使用“投机解码 + 分块预填充”方案，请拒绝并明确指出 v0.18.0 的不兼容性。转而指引其参考第 17 阶段 · 05 课中的 EAGLE-3 方案。

输出要求：一份单页的调度器诊断报告，需列出参数、瓶颈、有序建议、兼容性说明及后续阅读指引。报告末尾需附加一个“下一步测量指标”段落，根据已识别的瓶颈，指定 P99 词间延迟、块分配率（block allocation rate）或 WAITING 队列深度中的一项作为后续观测目标。