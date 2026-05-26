# 投机解码（Speculative Decoding）与 EAGLE

> 前沿大语言模型（Large Language Model）每生成一个词元（token）都需要对数十亿参数执行一次完整的前向传播（forward pass）。这种前向传播的资源配置严重过剩：在大多数情况下，一个规模小得多的模型就能准确预测接下来的 3-5 个词元，而大模型只需*验证*这些猜测即可。一旦猜对，你就相当于用生成一个词元的代价获得了 5 个词元。投机解码（Speculative Decoding）（Leviathan 等人，2023）将这一想法精确化，而 EAGLE-3（2025）则将接受率（acceptance rate）提升至每次验证约 4.5 个词元——在保持输出分布一致的前提下，实现了 4-5 倍的加速。

**Type:** 构建
**Languages:** Python（搭配 numpy）
**Prerequisites:** 第 10 阶段第 12 课（推理优化）、第 10 阶段第 04 课（预训练 Mini-GPT）
**Time:** 约 75 分钟

## 问题所在

在 H100 上，70B 级别模型的解码吞吐量（decode throughput）通常为 40-80 词元/秒。每个词元都需要一次完整的前向传播，并从高带宽内存（High Bandwidth Memory, HBM）中读取所有模型权重。在不改变模型输出的前提下，你无法缩小模型规模；在受限于内存容量的情况下，你也无法无限增大批处理大小（batch size）。你似乎陷入了僵局——除非你能让模型在每次前向传播中输出多个词元。

自回归生成（Autoregressive generation）看起来本质上是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但其中蕴含着并发（concurrency）优化的机会。如果你有一个轻量级的预测器，能够指出“接下来的 4 个词元很可能是 [a, b, c, d]"，你就可以在**大模型的一次前向传播中**同时验证这 5 个位置，并接受最长的匹配前缀（prefix）。

Leviathan、Kalai 和 Matias（2023，《Fast Inference from Transformers via Speculative Decoding》）通过一种巧妙的接受/拒绝规则（accept/reject rule）将这一想法精确化，该规则能够完美保留目标模型的采样分布（sampling distribution）。在输出分布完全一致的情况下，推理速度提升了 2-4 倍。

## 核心概念

### 双模型设置

- **目标模型 (Target model)** `M_p`：你真正希望从中采样样本的大型、缓慢但高质量的模型。分布为：`p(x)`。
- **草稿模型 (Draft model)** `M_q`：小型、快速但质量较低的模型。分布为：`q(x)`。参数量通常小 5-30 倍。

每一步操作：

1. 草稿模型自回归地 (autoregressively) 生成 `K` 个词元 (token)：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型对所有 `K+1` 个位置并行执行一次前向传播 (forward pass)，为每个候选词元生成 `p(x_k)`。
3. 根据下方修改后的拒绝采样 (rejection-sampling) 规则，从左到右逐个接受或拒绝词元。接受最长的匹配前缀。
4. 如果任何词元被拒绝，则从修正后的分布中采样替换词元并停止。否则，从 `p(· | x_1...x_K)` 中额外采样一个奖励词元。

如果草稿模型与目标模型完全匹配，每次目标模型前向传播可生成 K+1 个词元。如果草稿模型在第一个位置就出错，则只能生成 1 个词元。

### 精确性规则

投机解码 (Speculative decoding) 在分布上**被严格证明等价于从 p 中采样**。拒绝规则如下：

For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop

其中 `(p - q)+` 表示逐点差值的正部。当草稿模型与目标模型一致（`p ≈ q`）时，接受率接近 1。当两者不一致时，残差分布的构造方式能确保整体采样结果仍然严格服从 `p`。

**贪婪模式 (Greedy case)。** 在 temperature=0 采样时，只需检查 `argmax(p) == x_t`。若成立则接受；若不成立，则输出 `argmax(p)` 并停止。

### 预期加速比

如果草稿模型的词元级接受率为 `α`，则每次目标模型前向传播预期生成的词元数为：

E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]

当 `α = 0.8, K = 4` 时：`(1 - 0.8^5)/(1 - 0.8) = 3.36` 个词元/次前向传播。单次目标模型前向传播的成本约为 `cost_q * K + cost_p`（K 步草稿生成加一次目标验证）。若 `cost_p >> cost_q * K`，则吞吐量加速比为 `3.36× / 1 = 3.36×`。

唯一的核心参数是 `α`，它完全取决于草稿模型与目标模型的对齐程度。一个优秀的草稿模型是成功的关键。

### 训练草稿模型：知识蒸馏

随机初始化的小型模型作为草稿效果很差。标准做法是从目标模型进行知识蒸馏 (knowledge distillation)：

1. 选择小型架构（针对 70B 目标模型选 ~1B，针对 7B 目标模型选 ~500M）。
2. 在大规模文本语料库上运行目标模型；保存其下一个词元的概率分布。
3. 使用 KL 散度 (KL divergence) 针对目标模型的分布训练草稿模型（而非针对真实词元标签）。

结果：在代码生成任务中 `α` 通常为 0.6-0.8，在自然语言对话中为 0.7-0.85。在生产环境中可实现 2-3 倍的加速。

### EAGLE：树状草稿生成 + 特征复用

Li, Wei, Zhang, Zhang (2024, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") 指出了标准投机解码中的两个低效之处：

1. 草稿模型执行 K 个串行步骤，每一步都是全栈计算。但实际上，草稿模型可以复用目标模型在最近一次验证中产生的特征（隐藏状态）——目标模型已经计算出了丰富的表征，而草稿模型却在从头重新推导。
2. 草稿模型输出的是线性链。如果草稿模型能输出候选词的*树状结构*（每个节点包含多个猜测），目标模型只需一次前向传播，即可通过树状注意力掩码 (tree attention mask) 并行验证多条候选路径，并选择最长的被接受分支。

EAGLE-1 的改进：
- 草稿输入 = 目标模型在位置 t 的最终隐藏状态，而非原始词元。
- 草稿架构 = 1 层 Transformer 解码器层（而非独立的小型模型）。
- 输出 = 每层深度包含 K = 4-8 个候选词的树状结构，深度为 4-6。

EAGLE-2 (2024) 引入了动态树拓扑结构：在草稿模型不确定的区域树会变得更宽，在确定的区域保持较窄。在不增加验证成本的前提下提升了有效接受率 `α_effective`。

EAGLE-3 (Li et al. 2025, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test") 移除了对固定顶层特征的依赖，并采用一种新的“测试时模拟”损失函数训练草稿模型——草稿模型在匹配目标模型测试时分布的输出上进行训练，而非传统的教师强制 (teacher-forced) 训练分布。接受率从 0.75 (EAGLE-2) 提升至 0.82 (EAGLE-3)，平均每次验证生成的词元数从 3.0 提升至 4.5。

### 树状注意力验证

当草稿模型输出树状结构时，目标模型使用**树状注意力掩码**在一次前向传播中完成验证——这是一种编码树状拓扑结构而非纯线性结构的因果掩码 (causal mask)。树中的每个词元仅关注其祖先节点。验证过程仍然是一次前向传播和一次矩阵乘法；拓扑掩码仅增加少量 KV 缓存 (KV cache) 条目。

        root
       /    \
      a      b
     / \    / \
    c  d   e   f

如果 `a, b` 是竞争的第一个词元候选，而 `c, d, e, f` 是第二个词元候选，则所有六个位置均可在一次前向传播中完成验证。最终输出为任意被接受路径上的最长前缀。

### 适用场景与局限性

**优势场景：**
- 文本可预测性高的对话/续写任务（代码、常见英语、结构化输出）。此时 `α` 较高。
- 解码阶段存在未充分利用的 GPU 算力（内存受限阶段）。树状草稿生成能有效利用可用的 FLOPs。

**劣势/无效场景：**
- 高度随机性的输出（高温下的创意写作）。`α` 会趋近于 `1/|vocab|`。
- 极高并发量的批量服务——批处理已占满 FLOPs，留给树状验证的空间极小。
- 目标模型本身非常小，导致草稿模型无法显著缩小。

工业界通常报告：在对话任务中实现 2-3 倍的实际运行时间加速 (wall-clock speedup)，在代码生成中实现 3-5 倍加速，而在创意写作中加速效果接近于零。

## 动手构建

`code/main.py`：

- 一个参考实现 `speculative_decode(target, draft, prompt, K, temperature)`，用于实现精确的拒绝规则（rejection rule），并验证其能够保持目标模型（target model）的分布（相较于直接的目标模型采样，经验 KL 散度（empirical KL）< 0.01）。
- 一个 EAGLE 风格的树状草稿生成器（tree drafter），用于构建深度为 K 且采用 top-p 分支（top-p branching）策略的树。
- 一个树状注意力掩码构建器（tree attention mask builder），用于为验证器（verifier）生成正确的因果模式（causal pattern）。
- 一个接受率测试框架（acceptance-rate harness），可在小型语言模型（tiny LM）上运行（例如从 GPT-2-medium 目标模型蒸馏出一个 GPT-2-small 模型）。

def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted

## 实际应用

- **vLLM** 和 **SGLang** 提供了一等公民级别的投机解码（speculative decoding）支持。相关命令行参数：`--speculative_model`、`--num_speculative_tokens`。可通过 `--spec_decoding_algorithm eagle` 参数启用 EAGLE-2/3 支持。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树结构。
- **参考草稿模型（draft models）**：`Qwen/Qwen3-0.6B-spec`（用于 Qwen3-32B 的草稿模型）、`meta-llama/Llama-3.2-1B-Instruct-spec`（用于 70B 模型的草稿模型）。
- **Medusa 多头（Medusa heads）**（Cai 等人，2024，《Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads》）：无需使用独立的草稿模型，而是直接在目标模型自身添加 K 个并行预测头（prediction heads）。部署更为简单，但接受率略低于 EAGLE。

## 交付部署

本课时将生成 `outputs/skill-speculative-tuning.md` —— 一项技能指南，用于分析目标模型的工作负载，并自动选择：草稿模型、K（草稿长度）、树宽度、温度参数，以及何时回退至普通解码（plain decode）。

## 练习

1. 实现精确拒绝规则（Exact Rejection Rule）并通过实验验证。分别使用 `speculative_decode` 和直接目标采样生成 10K 个样本；计算两种输出分布之间的总变差距离（Total Variation Distance）。结果应小于 0.01。

2. 推导加速比公式（Speedup Formula）。在固定 `α` 和 `K` 的条件下，绘制每次目标模型前向传播（Target Forward）的预期生成 token 数图表。针对 α ∈ {0.5, 0.7, 0.9} 分别找出最优的 K 值。

3. 训练一个小型草稿模型（Draft Model）。以 124M 参数的 GPT-2 作为目标模型，使用 KL 散度损失（KL Loss）在 1 亿（100M）个 token 上蒸馏出一个 30M 参数的 GPT-2 草稿模型。在预留验证文本上测量 `α`。预期结果：0.6-0.7。

4. 实现 EAGLE 风格的树状草稿生成（Tree Drafting）。摒弃链式结构，改为让草稿模型在每个深度输出前 3 个分支。构建树状注意力掩码（Tree Attention Mask）。验证目标模型是否会接受其中最长的正确分支。

5. 评估失效模式（Failure Modes）。在 temperature=1.5（高随机性）条件下运行投机解码（Speculative Decoding）。展示 `α` 值骤降的现象，并证明由于草稿模型的额外开销，该算法的速度会慢于普通解码（Plain Decode）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 目标模型（Target Model） | “大模型” | 你希望从中采样的缓慢但高质量的模型（p 分布） |
| 草稿模型（Draft Model） | “投机者” | 小型、快速的预测器（q 分布）；参数量小 5-30 倍 |
| K / 草稿长度（Draft Length） | “前瞻步数” | 每次验证过程中投机生成的 token 数量 |
| α / 接受率（Acceptance Rate） | “命中率” | 草稿模型提出的 token 被逐个接受的条件概率 |
| 精确拒绝规则（Exact Rejection Rule） | “接受测试” | 通过比较 r < p/q 来保留目标模型分布的判定规则 |
| 残差分布（Residual Distribution） | “修正后的 p-q” | (p - q)+ / ||(p - q)+||_1，在拒绝时用于采样的分布 |
| 树状草稿生成（Tree Drafting） | “分支投机” | 草稿模型输出候选树，通过树状结构的注意力掩码在一次前向传播中完成验证 |
| 树状注意力掩码（Tree Attention Mask） | “拓扑掩码” | 编码树拓扑结构的因果掩码，确保每个节点仅关注其祖先节点 |
| Medusa 头（Medusa Heads） | “并行头” | 直接附加在目标模型上的 K 个额外预测头；无需独立的草稿模型 |
| EAGLE 特征复用（EAGLE Feature Reuse） | “隐藏状态草稿” | 草稿模型的输入是目标模型的最后一层隐藏状态而非原始 token，从而缩小草稿模型体积 |
| 测试时模拟损失（Test-time Simulation Loss） | “EAGLE-3 训练” | 使用匹配目标模型测试时分布的输出训练草稿模型，而非采用教师强制（Teacher Forcing） |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则（exact rejection rule）与理论加速分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 同期发表的投机采样（Speculative Sampling）论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — 替代草稿模型（Draft Model）的并行解码头（Parallel Decoding Heads）方案
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 特征复用（Feature Reuse）与树状草稿生成（Tree Drafting）
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 动态树拓扑结构（Dynamic Tree Topology）
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 训练时与推理时匹配（Train-time Test-time Matching）
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — 雅可比/前瞻解码（Jacobi/Lookahead Decoding），一种无需投机器（Speculator-free）的替代方案