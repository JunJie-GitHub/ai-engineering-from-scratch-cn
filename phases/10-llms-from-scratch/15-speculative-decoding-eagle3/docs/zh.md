# 投机解码（Speculative Decoding）与 EAGLE-3

> 第 7 阶段 · 第 16 课已经证明了其数学原理：Leviathan 拒绝规则（Leviathan rejection rule）能够精确保持验证器（verifier）的分布。本课将从训练栈（training stack）的视角，剖析 2026 年生产环境中的投机解码技术。EAGLE-3 将草稿模型（draft model）从一种廉价的近似方案，转变为专门针对验证器自身隐藏状态（hidden states）进行训练的微型网络，并引入了训练时测试循环（training-time test loop）以对齐其训练与推理分布。最终成果：端到端加速比（end-to-end speedup）达到 3 倍至 6.5 倍，在对话场景下单 token 接受率（accepted per-token rate）超过 0.9，且无任何分布权衡（distributional tradeoff）。2026 年的所有生产级推理栈（inference stack）均已默认集成该技术。

**Type:** 构建
**Languages:** Python (stdlib)
**Prerequisites:** 第 7 阶段 · 16（投机解码数学原理），第 10 阶段 · 12（推理优化）
**Time:** 约 75 分钟

## 学习目标

- 用一句话陈述 Leviathan 定理，并证明投机循环（speculative loop）生成的样本与验证器直接采样的分布完全一致。
- 梳理从基础投机解码（vanilla spec-decoding，Leviathan 2023）到 EAGLE、EAGLE-2 及 EAGLE-3 的两年演进历程，并明确指出每一步所解决的具体局限性。
- 根据接受率（acceptance rate）`α` 和草稿模型与验证器的成本比（cost ratio）`c` 计算预期加速比，并为不同场景选择最优的草稿长度（draft length）`N`。
- 从零实现完整的投机循环：生成草稿、验证、从残差分布（residual distribution）中拒绝采样、在拒绝时回滚 KV 缓存（KV cache），以及在完全接受时输出奖励 token（bonus token）。

## 问题背景

在 H100 上运行 70B 参数模型的自回归解码（autoregressive decoding）时，速度大约仅为每秒 35 个 token。此时 GPU 远未达到饱和状态。真正的瓶颈在于内存带宽（memory bandwidth）：每个 token 都需要从 HBM 加载 70B 的权重，执行一步算术运算，然后仅输出一个浮点数。计算单元大部分时间处于空闲状态。

投机解码将这一困境转化为一个可实际解决的吞吐量问题。一个轻量级的草稿模型通过 `N` 次小型前向传播（forward pass）一次性提出 `N` 个 token。验证器随后对前缀及所有 `N` 个草稿 token 执行一次前向传播。如果验证器在位置 `i` 的分布与草稿一致（我们将在后文给出精确的统计学定义），则予以接受；否则拒绝，并从残差分布中采样进行修正。这样，单次大模型前向传播最多可产出 `N+1` 个被接受的 token，而非仅仅一个。

核心定理来自 Leviathan、Kalman 和 Matias（ICML 2023）：其输出分布与直接从验证器采样得到的分布完全相同。不是近似，而是严格一致。这正是投机解码能够被生产环境接纳的根本原因——它是一种纯粹的延迟优化，没有任何质量上的妥协。

第 7 阶段 · 第 16 课为你奠定了数学基础，而本课将为你揭示其训练栈的实现。一个优质的草稿模型所能带来的加速比，是廉价草稿模型的两倍以上。EAGLE、EAGLE-2 和 EAGLE-3（Li 等人，2024–2025）将“草稿模型 = 同架构的缩小版模型”这一概念，转化为一门精确的工程学科。2026 年的生产级推理服务器已默认采用 EAGLE-3。

## 核心概念

### 不变性：Leviathan 拒绝采样

设给定前缀时，草稿模型 (draft model) 对下一个 token 的分布为 `p(t)`，验证器 (verifier) 的分布为 `q(t)`。采样一个草稿 token `d ~ p`。以概率 `min(1, q(d) / p(d))` 接受该 token。若被拒绝，则从残差分布 (residual distribution) `(q - p)_+ / ||(q - p)_+||_1` 中进行采样。最终得到的样本将严格遵循分布 `q`。无论 `p` 的质量有多差，这一结论都成立——`p` 越差，拒绝的频率就越高，但输出结果始终保持精确。

将 `N` 次此类调用串联起来，仅需对 `prefix + d_1 + ... + d_N` 执行一次验证器前向传播 (forward pass)。验证器会同时返回 `q_1, q_2, ..., q_{N+1}`。从左至右依次遍历。若在位置 `j` 首次发生拒绝，则从 `residual(q_j, p_j)` 中采样并停止。若全部接受，则从 `q_{N+1}` 中额外采样一个奖励 token (bonus token)。

### 决定加速比的因素

设 `α` 为每个草稿 token 的预期接受率。设 `c = cost(draft) / cost(verifier)` 为成本比率。每次验证器前向传播预期接受的 token 数量为：

E[accepted] = (1 - α^(N+1)) / (1 - α)

每个被接受 token 的预期总实际耗时 (wall time) 为 `(N * c + 1) / E[accepted]`。针对 `N` 最小化该值，即可找到最佳平衡点。当 `α = 0.8, c = 0.05` 时：最优 `N` 约为 5–7，加速比可达 3.2×。当 `α = 0.95, c = 0.02` 时：最优 `N` 约为 8–10，加速比可逼近 5×。

影响最大的单一变量是 `α`。在固定 `N = 5` 的情况下，将 `α` 从 0.6（基础草稿模型 (vanilla draft)）提升至 0.9（EAGLE-3），每次验证器前向传播的预期接受 token 数将从 2.2 跃升至 4.1。在相同的验证器下，吞吐量几乎翻倍。

### 两年来的技术演进

**基础投机解码 (Vanilla speculative, Leviathan, 2023)**。草稿模型为同系列中独立训练的较小规模大语言模型 (LLM)。易于集成，`α ≈ 0.6`，最佳加速比约为 2×。

**EAGLE-1 (Li et al., 2024)**。草稿模型是一个微型 Transformer——通常仅有一到两层——它以验证器最后一层的隐藏状态 (hidden state) 作为输入，并直接预测下一个 token。由于草稿模型能够感知验证器的特征表示，其分布与验证器更为接近。`α` 提升至 0.7–0.8。

**EAGLE-2 (Li et al., 2024)**。引入了动态草稿树 (dynamic draft tree)：不再仅提议单一的 `N` 个 token 序列，而是生成一个小型候选树，通过一次前向传播（树注意力机制 (tree attention)）由验证器对每个节点进行打分，并沿最高概率路径遍历。草稿长度在每一步变为自适应。被接受路径上每个 token 的 `α` 突破 0.85。

**EAGLE-3 (Li et al., 2025, NeurIPS)**。进行了两项关键改进。首先，完全摒弃特征预测损失 (feature-prediction loss)——EAGLE-1/2 训练草稿模型以匹配验证器的隐藏状态，这限制了数据规模带来的收益上限。EAGLE-3 直接针对 token 预测进行训练。其次，训练时测试 (Training-Time Test, TTT)：在草稿模型训练期间，将其自身前几步的预测结果作为输入反馈回去，模拟推理时的实际运行方式。这使训练与测试分布对齐，并阻止了误差累积。实测加速比：在对话任务中最高达 6.5×，在 H100 上使用 SGLang 且 batch size 为 64 时，吞吐量提升 38%。

### KV 缓存回滚

验证过程会在一次前向传播中将验证器的 KV 缓存 (KV cache) 扩展 `N` 个条目。若在位置 `j` 发生拒绝，则位置 `j-1` 之后的缓存内容即变为无效。两种常见实现方式：写入临时缓冲区并在接受时提交（如 vLLM、TensorRT-LLM），或保留物理 KV 缓存与逻辑长度，并在拒绝时进行截断。无论哪种方式，回滚的开销仅为每层每个注意力头 (attention head) 的字节数，与前向传播的计算成本相比可忽略不计。

对于 EAGLE-2 的树搜索，验证器会使用符合树拓扑结构的非因果掩码 (non-causal mask) 运行注意力机制。工程实现较为繁琐，但计算层面仅是调用标准的 FlashAttention 并传入自定义掩码。

### 2026 年的草稿模型架构

| 策略 | 草稿类型 | `α` | 加速比 | 训练成本 |
|----------|-----------|-----|---------|---------------|
| 基础版 (Vanilla) | 独立的小型 LLM | 0.55-0.70 | 1.8-2.3× | 无（复用现有小型模型） |
| Medusa | 验证器上附加的额外语言模型头 (LM heads) | 0.65-0.75 | 2-3× | 约 10 亿 SFT token |
| EAGLE-1 | 基于隐藏状态的单层 Transformer | 0.70-0.80 | 2.5-3× | 约 600 亿 token |
| EAGLE-2 | EAGLE-1 + 动态草稿树 | 0.80-0.88 | 3-4× | 约 600 亿 token |
| EAGLE-3 | 多层特征融合 + TTT | 0.88-0.92 | 3.5-6.5× | 约 600-2000 亿 token |
| Lookahead | 无草稿模型（雅可比迭代 (Jacobi iteration)） | N/A | 1.3-1.6× | 无 |

在 2026 年的生产环境中：vLLM 和 SGLang 在可用时默认采用 EAGLE-3，否则回退至 EAGLE-2。TensorRT-LLM 针对 Meta 和 NVIDIA 的公开模型提供了最快的 Medusa 路径。llama.cpp 则为 CPU 部署提供了基础版草稿模型。

## 构建实现

参见 `code/main.py`。这是完整的 Leviathan 投机循环（speculative loop），包含所有核心组件：N 词元草稿（draft-of-N）、验证器并行计算（verifier parallel pass）、逐位置拒绝（per-position rejection）、残差采样（residual sampling）、奖励词元（bonus token）、KV 缓存回滚（KV rollback），以及经验验证（empirical verification）输出分布是否与直接从 `q` 采样相匹配。

### 步骤 1：拒绝规则

def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)

### 步骤 2：残差分布

def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]

### 步骤 3：完整的投机步骤

`spec_step` 函数从分布 `p` 中生成 `N` 个草稿词元，随后通过一次并行的 `q` 评估对它们全部进行验证。对于每个草稿词元，它会应用拒绝规则（rejection rule）；在首次遇到拒绝时，则从残差分布（residual distribution）中采样修正词元。如果全部接受，则从 `q_{N+1}` 中输出一个奖励词元。

### 步骤 4：KV 缓存回滚记录

模拟器为每个工作进程（worker）跟踪一个逻辑长度 `kv_length`。当接受 `k` 个草稿词元时，执行 `kv_length += k`。若在位置 `j` 发生拒绝，尽管缓存已写入超过 `j` 的位置，但逻辑长度会被设置为 `prefix_length + j + 1`（即修正词元之后的下一个位置）。后续的读取操作将截断至该逻辑长度。

### 步骤 5：Leviathan 验证

运行 50,000 次投机步骤（speculative step）。统计被接受词元的经验分布（empirical distribution）。将其与从 `q` 直接采样的 50,000 个样本进行对比。卡方统计量（chi-square statistic）应远低于临界值。该定理在实践中得以验证。

### 步骤 6：加速比与 α 的关系

通过以不同幅度扰动 `p` 使其偏离 `q`，来扫描草稿质量（draft quality）。测量 `α`，然后绘制每次验证器调用预期生成的词元数随 `α` 和 `N` 变化的曲线。代码会打印一张表格，展示 EAGLE-3 级别的草稿质量（`α ≈ 0.9`）如何实现每次验证器调用生成 4–5 个词元。

## 投入使用

使用 EAGLE-3 的生产级 `vllm serve` 命令：

vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'

根据 EAGLE-3 论文，在 H100 上以批次大小（batch size）64 运行 SGLang 配合 EAGLE-3 时，其吞吐量比批次大小 64 的原始解码（vanilla decoding）高出约 1.38 倍。

何时采用投机解码（speculative decoding）：

- 任何对 p50 延迟（p50 latency）的关注度高于峰值吞吐量（peak throughput）的交互式聊天工作负载。
- 代码生成与结构化输出（JSON、SQL）。由于目标分布具有高度可预测性，此时 `α` 通常高于 0.9。
- 长文本生成（数千个词元）。摊销加速比（amortized speedup）能持续带来收益。

何时不建议使用：

- 极小模型（< 3B）。草稿模型的计算成本并未比验证器低多少。
- 极小规模的 batch-1 CPU 部署。草稿模型带来的内存开销可能得不偿失。
- 极高温度的创造性采样场景，此时 `α` 会急剧下降。

## 部署上线

本课时将生成 `outputs/skill-eagle3-tuner.md`。给定推理工作负载（模型、批量大小、目标延迟、任务配置），它会推荐一种投机解码 (speculative decoding) 策略及调优参数（草稿模型族 (draft family)、`N`、树深度 (tree depth)、温度感知切换 (temperature-aware switching)）。

## 练习

1. 运行 `code/main.py`。确认在 50,000 个样本上，Leviathan 分布检验的卡方统计量 (chi-square statistic) 保持在 95% 临界值以下。

2. 在 `α` 固定为 0.9、`c` 固定为 0.04 的条件下，将 `N` 从 1 扫描至 10。绘制每次验证器调用 (verifier call) 的预期生成 token 数与每个 token 的墙上时间 (wall time)。找出使墙上时间最小化的 `N` 值，并解释该曲线的形态。

3. 修改代码以模拟 EAGLE-2 树搜索 (tree search)：在每一步中，草稿模型提出一个形状为 `[2, 2, 2]` 的树（共八条候选路径）。验证器仅运行一次，接受概率最高的路径胜出。计算每个叶子节点的 `α` 以及每次验证器调用的总 token 数。在同等计算量下，将其与线性链式投机解码 (linear-chain spec-decoding) 进行对比。

4. 实现一个针对两个并发序列的批量 KV 回滚 (KV rollback) 模拟器。序列 A 的所有草稿均被接受；序列 B 在位置 2 处被拒绝。证明每个序列的 `kv_length` 均得到了正确更新，且没有产生无效计算。

5. 阅读 EAGLE-3 论文的第 4 节（训练期测试）。请用两句话解释：为何在不使用 TTT 的情况下进行朴素的草稿模型训练会遭受暴露偏差 (exposure bias) 的影响，以及在训练过程中让草稿模型输入自身预测为何能解决该问题。请将此机制与序列到序列 (seq2seq) 模型中的计划采样 (scheduled sampling) 相关文献建立联系。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Leviathan 规则 (Leviathan rule) | "min(1, q over p)" | 以概率 `min(1, q(d)/p(d))` 进行伯努利接受/拒绝 (Bernoulli accept/reject)；当在拒绝时从残差分布中采样时，能精确保持验证器 (verifier) 的分布 |
| 残差分布 (Residual distribution) | "(q minus p) plus, normalized" | 将 `(q - p)_+` 截断至零并重新归一化 —— 拒绝时正确的采样分布 |
| 接受率 α (Acceptance rate α) | "草稿猜对的频率" | 在拒绝规则下每个 token 的伯努利成功期望概率；决定所有加速计算的核心参数 |
| EAGLE-1 | "基于隐藏状态的草稿" | 以验证器最后一层隐藏状态为条件的小型 Transformer 草稿模型 (Li et al., 2024) |
| EAGLE-2 | "动态草稿树" | EAGLE-1 的升级版，通过一次验证器前向传播，利用树注意力 (tree attention) 对候选续写树进行打分 |
| EAGLE-3 | "训练期测试" | 摒弃特征预测损失，改为直接进行 token 预测训练，并在训练期间让草稿模型使用自身的输出作为输入 |
| 训练期测试 (Training-time test, TTT) | "解决暴露偏差" | 在训练期间以自回归方式运行草稿模型，使训练和测试的输入分布保持一致 —— 计划采样 (scheduled sampling) 的直接类比 |
| KV 回滚 (KV rollback) | "撤销被拒草稿" | 记录机制，在发生拒绝后将验证器的 KV 缓存 (KV cache) 重置为已接受前缀的长度 |
| 奖励 Token (Bonus token) | "白嫖的一个" | 当全部 `N` 个草稿均被接受时，无需额外验证器计算开销，即可从 `q_{N+1}` 中额外采样一个 token |
| 树注意力 (Tree attention) | "一次性验证多个候选" | 采用非因果掩码 (non-causal mask) 的注意力机制，该掩码遵循草稿树的拓扑结构；通过一次前向传播即可计算树中每个节点的 `q_i` |

## 延伸阅读

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) —— 奠基性论文与等价性定理
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) —— 同期独立提出，附带严谨证明
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) —— EAGLE-1，基于隐藏状态条件的草稿模型
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) —— 动态树搜索
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) —— 2026 年生产环境默认方案
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) —— 另一种无需草稿模型的替代方案
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) —— 权威生产环境参考文档，已集成所有策略