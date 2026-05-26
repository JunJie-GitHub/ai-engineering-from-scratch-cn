---
name: 序列架构选择器
description: 根据序列长度、吞吐量及训练预算，选择序列架构（RNN、Transformer、SSM 或混合架构）。
version: 1.0.0
phase: 7
lesson: 1
tags: [Transformer, 架构, RNN, SSM]
---

给定一个序列问题（最大长度 Max Length、批次形状 Batch Shape、训练 Token 预算 Training Token Budget、推理延迟目标 Inference Latency Target、设备类别 Device Class），输出：

1. 主要架构（Primary Architecture）。选项包括：Transformer（Transformer）、状态空间模型（State-Space Model, SSM）（如 Mamba/RWKV）、SSM 与注意力混合架构（Hybrid SSM+Attention）、循环神经网络（Recurrent Neural Network, RNN）。提供一句理由，需与主导约束条件（Dominant Constraint）直接相关。
2. 上下文长度策略（Context Length Strategy）。若为 Transformer：全注意力截断长度（Full Attention Cutoff）、滑动窗口大小（Sliding Window Size）、RoPE 缩放因子（RoPE Scaling Factor）。若为 SSM：扫描块大小（Scan Chunk Size）。若为 RNN：隐藏层宽度（Hidden Width）。
3. 训练 FLOP 概况（Training FLOP Profile）。根据架构与上下文长度估算每 Token 的浮点运算次数（FLOPs）；注明该规格是否符合计算预算（Compute Budget）。
4. 推理内存概况（Inference Memory Profile）。Transformer 需关注键值缓存（Key-Value Cache, KV Cache），SSM 需关注状态大小（State Size），RNN 需关注每 Token 内存占用（Per-Token Memory）。若目标设备（Target Device）无法容纳批次大小为 1（Batch Size of 1）的推理，请予以标记。
5. 风险提示（Risk Note）。指出该选择在此规格规模下已知的一种具体故障模式（Failure Mode）（例如：在未启用 Flash Attention 的情况下，24GB GPU 处理 64K 上下文时 Transformer 发生显存溢出 Out Of Memory, OOM）。

若训练 Token 数量超过 10 亿（1B），除非明确说明梯度流（Gradient Flow）与并行化（Parallelism）方面的性能损耗，否则拒绝推荐纯 RNN。若上下文长度超过 64K，除非明确说明 `O(N^2)` 的内存开销（Memory Cost），否则拒绝推荐全注意力 Transformer（Full-Attention Transformer）。若架构为全新发布（发表不足 12 个月），除非提供明确的备选方案（Fallback），否则拒绝推荐用于生产环境（Production）。