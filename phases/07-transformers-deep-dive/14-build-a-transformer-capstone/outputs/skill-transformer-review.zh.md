---
name: transformer 审查
description: 对照第 7 阶段的 13 节课程，审查从零实现 Transformer 的代码。
version: 1.0.0
phase: 7
lesson: 14
tags: [transformers, 审查, 综合项目]
---

给定一个从零实现 Transformer（Transformer-from-scratch）的代码库（PyTorch / JAX），请对照 2026 年默认标准进行审查，并标记缺失或错误的部分：

1. 注意力机制（Attention）。需包含因果掩码（Causal mask）。按 `sqrt(d_head)` 进行缩放。多头拆分（Multi-head split）需正常工作。若环境支持则使用 Flash Attention（Flash Attention）。若 `d_model` ≥ 1024，需提及分组查询注意力（Grouped-Query Attention, GQA）。
2. 位置编码（Positional encoding）。首选旋转位置编码（Rotary Position Embedding, RoPE，2026 年推荐）或可学习绝对位置编码（Learned absolute，适用于小型模型）。将正弦位置编码（Sinusoidal）标记为历史遗留方案。
3. 模块连接（Block wiring）。采用前置归一化（Pre-norm，而非后置归一化 Post-norm）。使用 RMSNorm（而非层归一化 LayerNorm）。前馈网络（FFN）采用 SwiGLU（而非 ReLU/GELU）。每个子层周围均需添加残差连接（Residuals）。线性层中移除偏置项（Biases dropped，现代默认配置）。
4. 训练（Training）。使用 AdamW 优化器（AdamW，或 2026 年及以后版本的 Muon 优化器 Muon），配合线性预热（Linear warmup）的余弦学习率调度（Cosine LR schedule），梯度裁剪（Gradient clipping）阈值设为 1.0，启用 bf16 自动混合精度（bf16 autocast）。词元嵌入（Token embedding）与 `lm_head` 之间进行权重共享（Weight tying）。
5. 损失函数（Loss）。在每个位置计算偏移一位的交叉熵（Shift-by-one cross-entropy）。若存在填充（Padding）则进行掩码处理。按固定间隔记录训练损失与验证损失（Train and val loss）。

若代码库存在以下任一情况，则拒绝通过审查：无明确理由使用后置归一化；2026 年生产环境代码中无合理依据使用层归一化；解码器自注意力（Decoder self-attention）中缺失因果掩码；小型语言模型（Language Model, LM）中未共享嵌入权重（Untied embeddings）。需标记警告的情况：未划分验证集（Validation split）、未进行梯度裁剪、学习率（Learning Rate, LR）> 1e-3 且无预热、或 `block_size` 超出位置编码范围且无回退机制（Fallback）。建议完整运行 `python code/main.py`，并检查在 nano 配置下 tinyshakespeare 数据集上的最终验证损失是否低于 2.5。