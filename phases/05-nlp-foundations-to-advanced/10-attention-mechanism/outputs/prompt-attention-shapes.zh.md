---
name: attention-shapes
description: 调试注意力机制（Attention Mechanism）实现中的形状错误。
phase: 5
lesson: 10
---

给定一个存在缺陷的注意力机制实现，你需要找出形状不匹配（Shape Mismatch）的问题。输出以下内容：

1. 哪个矩阵的形状有误。指出该张量（Tensor）的名称。
2. 它应有的形状，需根据 `(d_s, d_h, d_attn, T_enc, T_dec, batch_size)` 推导得出。
3. 一行代码的修复方案。使用转置（Transpose）、重塑（Reshape）或投影（Project）。
4. 用于捕获回归缺陷（Regression）的测试。通常需断言 `output.shape == (batch, T_dec, d_h)` 且 `weights.shape == (batch, T_dec, T_enc)`，并且 `weights.sum(dim=-1)` 的值接近 1。

拒绝推荐会触发静默广播（Silent Broadcast）的修复方案。掩盖广播机制的缺陷会在后期表现为模型精度的无声下降。

针对 Bahdanau 机制的常见混淆，需明确解码器（Decoder）的输入应为 `s_{t-1}`（步骤前状态）。而对于 Luong 机制，则为 `s_t`（步骤后状态）。在点积注意力（Dot-Product Attention）中，初学者最常犯的错误是查询（Query）/键（Key）维度不匹配——请务必明确标出此问题。