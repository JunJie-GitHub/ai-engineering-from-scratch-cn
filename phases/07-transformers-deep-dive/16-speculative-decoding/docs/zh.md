# 投机解码（Speculative Decoding）—— 草稿、验证、重复

> 自回归解码（Autoregressive Decoding）是串行的。每个词元（Token）都必须等待前一个词元生成完毕。投机解码打破了这一链条：由一个轻量级模型预先起草 N 个词元，再由一个大型模型通过一次前向传播（Forward Pass）完成对这 N 个词元的验证。只要草稿正确，你就只需为 N 次生成支付一次大型前向传播的计算代价。

**类型：** 构建
**语言：** Python
**前置要求：** 第 7 阶段 · 07（GPT 因果语言模型），第 7 阶段 · 12（KV 缓存与 Flash Attention）
**耗时：** 约 60 分钟

## 核心问题

在 H100 上，一个 70B 的大语言模型（LLM）采样单个词元大约需要 30 毫秒。而一个 3B 的草稿模型（Draft Model）仅需约 3 毫秒。如果我们让 3B 模型提前起草 5 个词元，然后让 70B 模型运行*一次*来验证这全部 5 个词元，那么对于最多 5 个被接受的词元，总耗时仅为 `5×3 + 30 = 45 ms`——相比之下，传统串行生成需要 `5×30 = 150 ms`。这就是投机解码的核心逻辑：用少量额外的 GPU 显存（用于加载草稿模型）换取 2 到 4 倍的解码延迟（Decode Latency）降低。

该技巧必须保持概率分布不变。由 Leviathan 等人（2023）与 Chen 等人同期提出的投机采样（Speculative Sampling）算法，能够保证输出序列与大模型独立生成时的分布**同分布（Identically Distributed）**。无需在质量上做出妥协，唯一的变化就是速度更快。

在 2026 年的推理（Inference）场景中，主流的草稿-验证器（Draft-Verifier）配对方案主要分为以下四类：

1. **基础投机解码（Vanilla Speculative，Leviathan 2023）。** 独立的草稿模型（例如 Llama 3 1B）+ 验证器（例如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 在验证器上挂载多个解码头（Decoding Heads），并行预测 `t+1..t+k` 位置的词元。无需独立的草稿模型。
3. **EAGLE 系列（Li 2024, 2025）。** 轻量级草稿模型，可复用验证器的隐藏状态（Hidden States）；接受率（Acceptance Rate）高于基础方案；通常可实现 3–4 倍加速。
4. **前瞻解码（Lookahead Decoding，Fu 2024）。** 基于雅可比迭代（Jacobi Iteration）；完全不需要草稿模型。属于自投机（Self-Speculation）机制。虽属小众方案，但无需额外依赖。

2026 年，所有面向生产环境的推理栈（Inference Stack）均已默认集成投机解码功能。vLLM、TensorRT-LLM、SGLang 以及 llama.cpp 等框架至少都支持基础投机解码与 EAGLE-2 方案。

## 核心概念

### 核心算法

给定一个验证器（verifier）`M_q` 和一个成本更低的草稿模型（draft model）`M_p`：

1. 设 `x_1..x_k` 为已解码的前缀。
2. **草稿生成（Draft）**：使用 `M_p` 自回归地（autoregressively）生成候选序列 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，并得到对应的草稿概率 `p_1..p_N`。
3. **并行验证（Verify in parallel）**：将 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 输入 `M_q` 进行一次前向传播，获取位置 `k+1..k+N+1` 对应的验证器概率 `q_1..q_{N+1}`。
4. **从左到右逐个接受/拒绝草稿词元（token）**：对于每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受该词元。
5. 若在位置 `j` 首次被拒绝：从归一化后的“残差”分布（residual distribution）`(q_j - p_j)_+` 中采样得到 `t_j`。位置 `j` 之后的所有草稿词元均被丢弃。
6. 若全部 `N` 个词元均被接受：从 `q_{N+1}` 中额外采样一个词元 `t_{N+1}`（即免费奖励词元）。

残差分布技巧是其中的核心数学洞察，它确保了最终输出的分布与直接让 `M_q` 从头开始采样完全一致。

### 决定加速比的因素

设 `α` 为每个草稿词元的期望接受率（acceptance rate）。设 `c` 为草稿模型与验证器的成本比。每一步中：

- 朴素生成（naive generation）每个词元需要调用一次大模型。
- 投机解码（speculative decoding）在 `α` 较高时，每生成 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个词元仅需调用一次大模型。

经验法则：当 `α = 0.75` 且 `N = 5` 时，大模型调用次数减少至原来的 1/3。草稿模型的成本仅为大模型的 1/5。总体实际运行时间（wall-clock time）缩短约 2.5 倍。

**`α` 取决于：**

- 草稿模型对验证器的逼近程度。同系列模型或使用相同训练数据可显著提升 `α`。
- 解码策略。草稿与验证器均采用贪婪解码（greedy decoding）时：`α` 较高。若使用温度采样（temperature sampling）：匹配难度增加，接受率下降。
- 任务类型。代码和结构化输出接受率更高（可预测性强）；自由形式的创意写作接受率较低。

### Medusa —— 无需独立草稿模型的方案

Medusa 通过在验证器上添加额外的输出头（output heads）来替代独立的草稿模型。在位置 `t`：

shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4

每个头输出各自的 logits。在推理阶段，你从每个头中采样以获取候选序列，然后使用树注意力机制（tree-attention scheme）进行一次前向传播进行验证，该机制可同时考虑所有候选续写路径。

优点：无需第二个模型。缺点：增加了可训练参数；需要监督微调（supervised fine-tuning）阶段（约 10 亿词元）；接受率略低于使用优质草稿模型的传统投机解码。

### EAGLE —— 通过复用隐藏状态提升草稿质量

EAGLE-1/2/3（Li 等人，2024–2025）将草稿模型设计为一个微型 Transformer（通常为 1 层），用于接收验证器最后一层的隐藏状态（hidden states）。由于草稿模型能够直接观察验证器的特征表示，其预测结果与验证器的输出分布高度相关。接受率从传统方案的 ~0.6 提升至 0.85 以上。

EAGLE-3（2025）在候选续写路径上引入了树搜索（tree search）。vLLM 和 SGLang 已将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认投机解码路径。

### KV 缓存（KV Cache）的协同处理

验证阶段通过一次前向传播将 `N` 个草稿词元输入验证器。这会使验证器的 KV 缓存（KV cache）扩展 `N` 个条目。若部分草稿词元被拒绝，则必须将缓存回滚至已接受前缀的长度。

生产环境实现（如 vLLM 的 `--speculative-model` 参数、TensorRT-LLM 的 `LookaheadDecoder`）通过临时 KV 缓冲区（scratch KV buffers）来处理此问题。先写入，接受后再提交。这在概念上并不复杂，但实现细节较为繁琐。

## 构建实现

请参阅 `code/main.py`。我们使用以下组件实现了核心的**投机采样**（speculative sampling）算法（包含拒绝步骤与残差分布）：

- 一个“大模型”（big model），它对人工设定的分布执行确定性 softmax 操作（以便我们能通过解析方法验证接受率的数学原理）。
- 一个“草稿模型”（draft model），作为大模型的扰动变体。
- 一个接受/拒绝循环，该循环生成的边缘分布（marginal distribution）与直接采样完全一致。

### 步骤 1：拒绝步骤

def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)

`u` 为均匀分布随机数。`q_prob` 表示验证器（verifier）对草稿 token 的预测概率，`p_prob` 为草稿模型的预测概率。Leviathan 定理表明，通过这一伯努利决策（Bernoulli decision），并在拒绝时从残差分布（residual distribution）中采样，能够精确保持验证器的原始分布。

### 步骤 2：残差分布

def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]

将 `q` 与 `p` 逐元素相减，将负值截断为零，随后进行重新归一化。每当发生拒绝时，均从此分布中采样。

### 步骤 3：单次投机步骤

def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix

5 个草稿 token 被接受 → 额外生成 1 个 token → 仅需一次验证器前向传播即可产出 6 个 token。

### 步骤 4：测量接受率

在不同草稿质量水平下运行 10,000 次投机步骤。绘制接受率与草稿分布及验证器分布之间 KL 散度（KL divergence）的关系曲线。你将观察到清晰的单调关系。

### 步骤 5：验证分布等价性

经验验证：投机循环生成的 token 直方图应与直接从验证器采样所得的直方图完全吻合。这正是 Leviathan 定理在实际中的体现。通过卡方检验（chi-square test）可在采样误差范围内证实该结论。

## 实际应用

生产环境：

# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5

截至 2026 年中，TensorRT-LLM 提供了最快的 Medusa 路径。`faster-whisper` 为 Whisper-large 封装了结合小型草稿模型（draft model）的投机解码（speculative decoding）功能。

**选择草稿模型：**

| 策略 | 适用场景 | 加速比 |
|----------|--------------|---------|
| 原生草稿模型（vanilla draft，1B/3B Llama 系列） | 快速原型验证，无需训练 | 1.8–2.3× |
| Medusa 头（Medusa heads） | 可对验证器（verifier）进行微调 | 2–3× |
| EAGLE-2 / 3 | 生产环境，追求极致速度 | 3–4× |
| 前瞻解码（lookahead） | 无需草稿模型、无需训练、无额外参数 | 1.3–1.6× |

**何时不应使用投机解码：**

- 生成 1–5 个 token 的单序列任务。此时系统开销将占据主导。
- 高度创造性 / 高温采样（high-temperature sampling）任务（接受率 α 会下降）。
- 内存受限的部署环境（草稿模型会增加显存 VRAM 占用）。

## 交付上线

请参阅 `outputs/skill-spec-decode-picker.md`。该技能模块会为新的推理工作负载选择一种投机解码策略（原生 / Medusa / EAGLE / 前瞻）及调优参数（N、草稿模型温度）。

## 练习

1. **简单。** 运行 `code/main.py`。在 50,000 个 token 的样本上，验证投机生成的 token 分布与验证器直接采样的分布是否一致（卡方检验 p > 0.05）。
2. **中等。** 针对 `α = 0.5, 0.7, 0.85`，绘制加速比（大模型单次前向传播生成的 token 数）随 `N` 变化的曲线。找出每个 α 对应的最优 `N`。（提示：单次验证调用的期望 token 数 = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现一个微型 Medusa：使用第 14 课的综合项目 GPT 模型，额外添加 3 个语言模型头（LM heads），分别用于预测 t+2、t+3、t+4 位置的 token。在 tinyshakespeare 数据集上使用联合多头损失（joint multi-head loss）进行训练。将其接受率与通过截断同一模型得到的原生草稿模型进行对比。
4. **困难。** 实现回滚（rollback）机制：从一个包含 10 个 token 前缀的 KV 缓存（KV cache）开始，输入 5 个草稿 token，模拟在第 3 个位置发生拒绝。验证在下一次迭代中，你的缓存读取是否正确匹配“前缀 + 前 2 个已接受的草稿 token”。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 草稿模型 (Draft model) | “便宜的那个” | 一个较小的模型，用于提出候选词元 (token)；计算成本通常仅为验证器 (Verifier) 的 1/10 到 1/50。 |
| 验证器 (Verifier) | “大的那个” | 我们需要保留其输出分布的目标模型；在每次投机步骤中运行一次。 |
| 接受率 (Acceptance rate, α) | “草稿猜对的频率” | 验证器接受草稿输出的逐词元概率。典型值在 0.7–0.9 之间。 |
| 残差分布 (Residual distribution) | “拒绝时的回退方案” | 对 `(q - p)_+` 进行归一化；在拒绝时从此分布中采样，可确保保留验证器的原始分布。 |
| 奖励词元 (Bonus token) | “白送的那个” | 当所有 N 个草稿词元均被接受时，从验证器的下一步分布中额外采样一个词元。 |
| Medusa | “无草稿的投机解码” | 在验证器上附加多个语言模型头 (LM heads)，并行预测 t+1..t+k 位置的词元。 |
| EAGLE | “基于隐藏状态的草稿” | 一个微型 Transformer 草稿模型，以验证器最后一层的隐藏状态 (hidden states) 为条件进行生成。 |
| 前瞻解码 (Lookahead decoding) | “雅可比迭代” | 使用不动点迭代 (fixed-point iteration) 进行自我投机；无需草稿模型。 |
| 树状注意力 (Tree attention) | “一次性验证多个候选” | 分支验证机制，可同时考虑多个草稿续写路径。 |
| KV 缓存回滚 (KV rollback) | “撤销被拒的草稿” | 使用临时 KV 缓冲区；接受时提交，拒绝时丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心算法与等价性定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 同期提出的工作；提供了简洁的伯努利拒绝 (Bernoulli-rejection) 证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 论文；采用树状注意力 (tree-attention) 进行验证。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；基于隐藏状态条件 (hidden-state-conditioned) 的草稿模型。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；动态树深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — 前瞻解码 (lookahead decoding)，无草稿模型方案。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 权威的生产环境参考文档，已集成全部四种策略。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 的参考代码实现。