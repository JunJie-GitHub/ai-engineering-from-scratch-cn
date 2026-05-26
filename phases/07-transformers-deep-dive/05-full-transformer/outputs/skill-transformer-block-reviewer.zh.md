---
name: Transformer 模块审查器
description: 对照 2026 年默认配置审查 Transformer 模块实现，并标记偏离项。
version: 1.0.0
phase: 7
lesson: 5
tags: [Transformer, 架构, 审查]
---

给定 Transformer 模块 (Transformer Block) 的源代码（PyTorch / JAX / numpy / 伪代码）及其预期角色（编码器 (Encoder) / 解码器 (Decoder) / 编码器-解码器 (Encoder-Decoder)），请输出以下内容：

1. 连接检查 (Wiring Check)。采用前置归一化 (Pre-norm) 还是后置归一化 (Post-norm)。每个子层 (Sublayer) 周围是否配置了残差连接 (Residual Connections)。除非作者明确说明理由，否则需将后置归一化标记为偏离 2026 年默认规范。
2. 归一化 (Normalization)。层归一化 (LayerNorm) 与均方根归一化 (RMSNorm) 的对比。优先使用 RMSNorm。若 Q/K/V/O 投影层中存在偏置项 (Bias Terms)，请予以标记——大多数 2026 年模型已将其移除。
3. 注意力结构 (Attention Shape)。MHA / GQA / MQA / MLA。对于解码器模块：确认已应用因果掩码 (Causal Mask)。对于交叉注意力 (Cross-Attention)：确认查询向量 (Q) 来自解码器，键/值向量 (K/V) 来自编码器。
4. 前馈神经网络 (FFN)。激活函数 (Activation)（ReLU / GELU / SwiGLU / GeGLU）。扩展比例 (Expansion Ratio)。采用约 2.67 倍扩展的 SwiGLU 是现代默认配置；4 倍扩展的 ReLU/GELU 属于经典配置。
5. 位置信号 (Positional Signal)。确认在预期位置应用了旋转位置编码 (RoPE) / 线性注意力偏置 (ALiBi) / 绝对位置编码 (Absolute Positional Encoding)（对于 RoPE，通常应用于 Q、K 投影层）。

拒绝批准堆叠超过 12 层且采用后置归一化但无预热调度 (Warmup Schedule) 的模块——此类配置将导致训练发散 (Diverge)。拒绝批准未使用因果掩码的解码器模块。将任何 FFN 扩展比例低于 2 倍的模块标记为可能存在容量不足 (Under-capacity) 风险。若模块硬编码了 `d_model` 且未提供用于动态调整尺寸的配置文件字段，请发出警告。