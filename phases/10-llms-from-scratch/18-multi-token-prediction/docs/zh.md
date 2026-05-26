# 多词元预测（Multi-Token Prediction, MTP）

> 从 GPT-2 到 Llama 3，每一个自回归大语言模型（Autoregressive LLM）在每个位置上都仅使用一个损失函数（loss）进行训练：预测下一个词元（token）。DeepSeek-V3 在每个位置上增加了第二个损失函数：预测再下一个词元。额外的 140 亿参数（基于 6710 亿参数模型）通过梯度流（gradient flow）蒸馏回主模型中，训练好的 MTP 预测头（MTP heads）在推理（inference）阶段被重新用作投机解码（speculative decoding）的草稿生成器（drafters），接受率（acceptance）超过 80%。1.8 倍的生成吞吐量（generation throughput）提升是免费附赠的。本课程将基于 DeepSeek 技术报告构建序列式 MTP 模块，计算损失函数与共享头（shared head）的参数布局，并解释为何 MTP 能够保持因果链（causal chain），而 Gloeckle 等人最初的并行 MTP 设计却破坏了它。

**Type:** 构建
**Languages:** Python（标准库）
**Prerequisites:** 第 10 阶段 · 04（预训练迷你 GPT 模型），第 10 阶段 · 15（投机解码）
**Time:** 约 60 分钟

## 学习目标

- 阐述 MTP 的训练目标，并推导跨预测深度的联合损失函数（joint loss）。
- 解释 Gloeckle 等人（2024）的并行 MTP 预测头与 DeepSeek-V3 的序列式 MTP 模块之间的区别，以及为何序列式设计能够保持因果链。
- 计算在预训练（pre-training）过程中添加 MTP 模块所带来的参数与内存开销（overhead）。
- 从零实现一个 MTP 模块：包括共享嵌入层（shared embedding）、按深度划分的 Transformer 块（Transformer block）、投影层（projection）以及共享输出头（shared output head）。

## 问题描述

下一个词元预测（Next-token prediction）是大型语言模型（Large Language Model, LLM）的标准训练目标。每个隐藏状态（hidden state）的监督目标都仅限于预测唯一的内容：紧随其后的词元（token）。这是一个出奇微弱的监督信号。序列中的大部分信息都超越了单个词元的范畴，例如结构、连贯性、事实准确性以及算术逻辑流。模型必须通过在数万亿词元的训练过程中不断累积这些单步预测信号，才能逐步掌握上述能力。

多词元预测（Multi-Token Prediction, MTP）提出了一个设想：如果让每个隐藏状态同时监督预测多个未来词元，效果会如何？Gloeckle 等人（Meta，2024）的研究证实了这一思路的有效性。他们的实现方式是在主干网络（backbone）之上堆叠多个独立的输出头（output heads），每个头负责预测不同偏移量（offset）的未来词元。该方案并行且简单，但各个输出头仅看到相同的隐藏状态，缺乏层次化的特征细化；同时，各步预测之间未形成因果链，因此无法用于推测解码（speculative decoding）。

DeepSeek-V3（2024年12月）对 MTP 进行了重新设计，将其构建为顺序模块，从而在每个预测深度（prediction depth）上保持因果链。模型首先基于 `h_i^(0)` 预测 `t+1`，随后结合 `h_i^(0)` 与 `E(t+1)` 的嵌入表示（embedding）生成新的隐藏状态 `h_i^(1)`，并据此预测 `t+2`，依此类推。每个预测深度都对应一个独立的小型 Transformer 模块（Transformer block）。共享的词元嵌入层与共享的输出头有效控制了额外的参数开销。以 DeepSeek-V3 的模型规模为例，在 671B 主模型权重的基础上，MTP 模块仅引入了 14B 的额外参数。这 2% 的参数开销不仅换来了更密集的训练信号，还在推理阶段直接提供了现成的推测解码草稿。

本教程将带你从零开始构建单个 MTP 模块以及 D 深度损失（D-depth loss）。其背后的数学推导十分简洁，核心代码实现仅需 150 行。

## 核心概念

### 顺序式多词元预测（MTP）方案

DeepSeek-V3 在主模型之上添加了 `D` 个 MTP 模块。每个模块 `k`（`k = 1..D`）负责预测深度 `k` 处的词元（token）——即给定截至位置 `i` 的前缀，预测 `t_{i+k}`。

模块 `k` 包含以下组件：

- 一个独立的 Transformer 块 `T_k`，包含其专属的注意力机制（attention）和多层感知机（MLP）。
- 一个投影矩阵 `M_k`，用于将上一深度的隐藏状态（hidden state）与下一深度真实词元（ground-truth token）的嵌入（embedding）进行拼接组合。
- 共享嵌入层 `E`（与主模型相同）。
- 共享输出头 `Out`（与主模型相同）。

在训练阶段，对于截至位置 `i` 的前缀，各深度的隐藏状态计算如下：

h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1

各深度的预测结果为：

logits_{i+k} = Out(h_i^(k-1))   for k = 1..D

各深度的损失函数为针对真实词元 `t_{i+k}` 的交叉熵（cross-entropy）：

L_k = CE(logits_{i+k}, t_{i+k})

跨深度的联合损失为：

L_MTP = (lambda / D) * sum_{k=1..D} L_k

`lambda` 是一个较小的权重系数——DeepSeek-V3 在训练的前 10% 阶段使用 0.3，之后调整为 0.1。总训练损失为 `L_main + L_MTP`。

### 为何采用顺序式而非并行式

Gloeckle 最初提出的并行 MTP 方案包含 `D` 个输出头，每个输出头直接作用于 `h_i^(0)`。每个头都基于相同的主干隐藏状态来预测 `t_{i+k}`。这种设计训练起来没有问题，但各预测结果之间缺乏条件依赖。你无法利用 `head_1` 的输出来辅助 `head_2`——因为所有输出头是并行触发的。

DeepSeek-V3 的顺序式设计通过 `h_i^(k-1)` 加上真实的下一个词元嵌入 `E(t_{i+k})` 来构建 `h_i^(k)`。这保留了因果链（causal chain）：为了预测 `t_{i+k+1}`，深度 `k+1` 处的模块能够“看到” `t_{i+k}` 处的信息。这种结构在本质上与自回归解码器（autoregressive decoder）消费自身输出的方式完全一致——这使得 MTP 模块可以直接用作投机解码（speculative decoding）的草稿生成器（drafter）。

在推理阶段：将 `h_i^(k-1)` 和已生成的草稿词元 `t_{i+k}` 输入模块 `k+1`，即可得到 `t_{i+k+1}` 的预测结果。重复此过程。这完全符合 EAGLE 风格的草稿生成机制，即直接使用训练好的 MTP 模块作为草稿网络。DeepSeek-V3 报告称，首个 MTP 模块的接受率（acceptance rate）超过 80%，并带来约 1.8 倍的加速效果。

### 参数量核算

对于隐藏层维度为 `h`、词表大小为 `V` 的模型：

- 主模型：包含数十亿参数，外加一个尺寸为 `V * h` 的输出头。
- 共享输出头：复用主模型的输出头。无额外参数。
- 共享嵌入层：复用主模型的嵌入层。无额外参数。
- 每个 MTP 模块：
  - 投影矩阵 `M_k`：`(2h) * h = 2h^2`。
  - Transformer 块 `T_k`：注意力机制（多头注意力 MHA 为 `4h^2`）加上 MLP（采用 8/3 扩展比的 SwiGLU 通常为 `8h^2`）。每个块约 `12h^2`。

每个模块的额外参数量总计：约 `14h^2`。以 DeepSeek-V3 的 `h = 7168`、`D = 1` 个模块为例：理论参数量约为 `14 * 7168^2 ≈ 7.2 亿`。但 DeepSeek-V3 官方报告为 140 亿（14B）——差异主要在于 MTP 模块中的专家层（expert layers）同样采用了混合专家模型（Mixture of Experts, MoE）架构。

### 投机解码的收益

在预训练阶段，MTP 模块会使训练速度降低约 10%（增加了前向计算量和额外损失）。但其收益是双重的：

1. 更密集的训练信号。每个隐藏状态会接收 `D+1` 个监督目标。在 MMLU、GSM8K、MATH 和 HumanEval 上的实测效果表明：DeepSeek-V3 的消融实验（ablation）中均观察到稳定的几个百分点的性能提升。
2. 推理阶段免费获得投机解码草稿。MTP 模块在训练时已具备预测后续多个词元的能力。将其复用为草稿网络后，可实现 80% 以上的接受率。在此水平下，采用 `N=3` 或 `N=5` 的投机解码可带来 1.8 倍的吞吐量提升。这 10% 的训练时间成本，在首次运行推理时即可收回。

### 与 EAGLE 的关系

EAGLE 在预训练结束后**单独**训练一个小型草稿模型。而 MTP 则将草稿生成能力直接融入预训练过程中。两种方法最终能达到相似的接受率，但实现路径不同：

| 维度 | EAGLE-3 | MTP（DeepSeek-V3） |
|-----------|---------|------------------|
| 训练时机 | 预训练之后 | 预训练期间 |
| 是否向后兼容现有权重 | 是 | 否（需重新训练） |
| 草稿模型参数量 | 1-2 个 Transformer 层 | 1 个 Transformer 块 + 投影矩阵 |
| 接受率 | 0.88-0.92 | 深度 1 处 >0.80 |
| 除加速外的额外收益 | 仅支持投机解码 | 更密集的训练信号 + 推理加速 |

## 构建

`code/main.py` 端到端构建了一个完整的 MTP（Multi-Token Prediction，多词元预测）模块：包含共享嵌入表（shared embedding table）、投影层（projection）、Transformer 块（transformer block）以及共享输出头（shared output head）。随后，该脚本会在一段简短的合成序列上计算各深度的交叉熵损失（cross-entropy loss），并按组件打印参数量。为保持数值清晰易读，示例采用了一个仅含 32 个词元的微型词表（toy vocabulary）。

### 步骤 1：共享嵌入表

主模型以及每个深度的每一个 MTP 模块都共用同一张 `vocab_size x hidden` 的嵌入表。这不是第二份副本，而是字面意义上完全相同的张量（tensor）。

### 步骤 2：逐深度组合

def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected

真实的 DeepSeek-V3 会将两个经过均方根归一化（RMSNorm）处理的向量拼接为 `[2h]` 维度，并使用一个 `h x 2h` 的矩阵进行投影。本示例为了保持标准库代码的简洁性，采用了向量加法作为替代方案。

### 步骤 3：深度 k 处的 Transformer 块

包含自注意力机制（self-attention）与多层感知机（MLP）。在示例中，为了在不依赖 `numpy` 的情况下保持结构清晰，采用了一个单层的线性注意力块（linear attention block）和一个 SwiGLU MLP。

### 步骤 4：共享输出头

复用主模型的输出投影层。计算词表上的逻辑值（logits）。

### 步骤 5：逐深度损失

计算 `softmax(logits)` 与偏移量 `k` 处的真实词元（ground-truth token）之间的交叉熵。使用 `lambda / D` 缩放因子对各深度的损失进行聚合。

### 步骤 6：参数统计

打印总参数量、共享部分（嵌入层、输出头）的参数量，以及每个模块的额外参数量。展示 MTP 额外参数与主模型规模的比值。

## 使用

MTP 已集成至 DeepSeek-V3（2024 年 12 月发布）及 DeepSeek-R1 系列模型中。在推理（inference）阶段：

- DeepSeek 自有的服务栈（serving stack）开箱即用地将 MTP 模块作为投机解码器（speculative decoder）进行调用。
- 截至 2026 年 4 月，vLLM 和 SGLang 已提供针对 DeepSeek-V3 MTP 的集成路径。
- AMD 的 ROCm SGLang 教程展示了一种特定的 MTP 投机解码配置，在 V3 检查点（checkpoint）上实测实现了 1.8 倍的加速。

在新一轮预训练（pre-training）中何时使用 MTP：

- 你掌控完整的预训练流水线，并希望获取更密集的训练信号（training signal）。
- 你明确模型未来将大规模部署，并希望零成本获得投机解码能力。
- 你的隐藏层维度（hidden size）至少为 4096。在 1B（十亿）参数规模下，额外开销带来的负面影响会超过收益。

何时不应使用：

- 对现有的预训练稠密模型（dense model）进行微调（fine-tuning）。MTP 模块本身并未参与训练。
- 需要干净基线（baseline）进行对比的研究模型。MTP 会改变模型架构。

## 交付

本教程将生成 `outputs/skill-mtp-planner.md` 文件。给定预训练运行规格（模型规模、数据、算力），它会返回一份 MTP 集成方案：包含深度数量 D、`lambda` 调度策略、内存开销，以及推理阶段的投机解码调用链路配置。

## 练习

1. 运行 `code/main.py`。展示随着合成信号（synthetic signal）增强，各深度损失（per-depth loss）如何单调递减。修改合成数据以采用固定模式，并验证深度 1（depth-1）与深度 2（depth-2）的损失均能收敛。

2. 计算配备 D=1 MTP 模块的稠密 70B 模型（隐藏维度 8192，80 层）的参数开销（parameter overhead）。将其与 DeepSeek-V3 报告的 14B 开销进行对比。解释 DeepSeek 数值更高的原因：MTP Transformer 块继承了相同的 MoE（Mixture of Experts，混合专家）结构，导致单个模块的参数数量大幅增加。

3. 在玩具模型中实现 D=2：添加第二个 MTP 模块，该模块接收 h^(1) 并预测 `t_{i+2}`。验证联合损失（joint loss）与参数核算（parameter accounting）是否与 DeepSeek 论文中的公式 19-21 相符。

4. 将玩具模型切换为并行 MTP（Parallel MTP，Gloeckle 风格）：在主隐藏状态之上添加 D 个输出头，每个头负责预测不同的偏移量。测量在相同合成信号下，各深度损失与串行版本（Sequential MTP）的对比情况。对于 k > 1，串行版本应产生更低的深度 k 损失，因为它以中间预测结果作为条件输入。

5. 将训练好的 MTP 模块用作 EAGLE 风格草稿（EAGLE-style draft）：在推理阶段调用模块 k 以提议 `t_{i+k}`。在预留序列（held-out sequence）上，测量这些草稿令牌（draft tokens）相对于主模型预测的接受率（acceptance rate）。若在玩具模型上该指标超过 50%，即表明你已成功复现 MTP 作为草稿的经验特性。

## Key Terms

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MTP 模块 | “额外的损失块” | 一个小型 Transformer 块加上投影层，用于预测主模型前方 `k` 个位置的令牌 |
| 预测深度（Prediction depth） | “哪个偏移量” | 整数 `k`，表示模块 `k` 根据截至位置 `i` 的前缀来预测 `t_{i+k}` |
| 并行 MTP（Parallel MTP） | “Gloeckle 风格” | 在同一骨干隐藏状态上附加 D 个独立头，无条件链 |
| 串行 MTP（Sequential MTP） | “DeepSeek-V3 风格” | 每个模块以前一深度的隐藏状态及下一个令牌的嵌入为条件；保留因果链 |
| 共享输出头（Shared output head） | “复用主头” | MTP 模块直接调用主模型的语言模型头（LM head），而非独立的输出投影层 |
| 共享嵌入（Shared embedding） | “复用主表” | 全局使用相同的词汇嵌入表；无重复参数 |
| 投影矩阵 M_k（Projection matrix M_k） | “结合隐藏状态 + 下一令牌” | 一个 `h x 2h` 的线性层，将前一隐藏状态与目标令牌嵌入折叠为下一深度的输入 |
| 联合损失 L_MTP（Joint loss L_MTP） | “平均额外损失” | 各深度交叉熵损失的算术平均值，并按 `lambda` 进行缩放 |
| 深度 1 接受率（Acceptance rate at depth 1） | “MTP 草稿的准确率” | D=1 MTP 模块的 top-1 预测与主模型 top-1 预测一致的比率；在 DeepSeek-V3 上超过 80% |
| Lambda 权重（Lambda weighting） | “额外损失的重要性” | 各深度的缩放因子；训练初期为 0.3，DeepSeek-V3 后期降至 0.1 |

## Further Reading

- [DeepSeek-AI — DeepSeek-V3 技术报告 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整的顺序多词元预测 (Sequential Multi-token Prediction) 描述（第 2.2 节），包含联合损失方程 (Joint-loss Equations) 以及推理 (Inference) 阶段 1.8 倍的加速效果
- [Gloeckle 等人 — 通过多词元预测实现更优更快的大语言模型 (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) — DeepSeek 设计所改进的并行多词元预测基线 (Parallel Multi-token Prediction Baseline)
- [Hugging Face 上的 DeepSeek-V3 模型卡片 (Model Card)](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 总计 685B 参数（主模型 671B + MTP 模块 14B），附部署说明 (Deployment Notes)
- [Leviathan 等人 — 通过投机解码加速 Transformer 推理 (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — MTP 所适用的投机解码 (Speculative Decoding) 框架
- [Li 等人 — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — EAGLE 的 2025 年初步架构，MTP 所对标的竞争方案