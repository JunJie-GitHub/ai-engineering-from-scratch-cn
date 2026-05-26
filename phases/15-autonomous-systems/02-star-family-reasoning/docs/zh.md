# STaR、V-STaR 与 Quiet-STaR —— 自学推理 (Self-Taught Reasoning)

> 最小规模的自我改进循环 (self-improvement loop) 就内嵌于推理依据 (rationale) 之中。模型会生成思维链 (Chain of Thought)，保留那些能得出正确答案的路径，并以此进行微调 (fine-tuning)。这便是 STaR。V-STaR 引入了验证器 (verifier)，以提升推理时选择 (inference-time selection) 的质量。Quiet-STaR 则将推理依据细化至每一个词元 (token)。三者均切实有效。它们并非魔法——该循环会保留任何恰好能得出正确答案的捷径。

**Type:** 学习
**Languages:** Python（标准库、自举循环模拟器）
**Prerequisites:** 第 13 阶段 · 01-03（推理与思维链），第 15 阶段 · 01（长程任务框架）
**Time:** 约 60 分钟

## 问题

教模型进行推理最直接的方法是收集人类编写的推理轨迹 (reasoning traces)。这种方法成本高昂、速度缓慢，且受限于人类愿意编写的高质量思维链 (chain-of-thought) 的数量。

STaR (Self-Taught Reasoner, Zelikman et al., 2022) 提出了一个问题：如果让模型自己生成推理依据 (rationales)，并根据已知答案进行自我评分会怎样？其循环流程如下：

1. 采样一条推理轨迹及对应答案。
2. 如果最终答案正确，则保留该轨迹。
3. 在保留的轨迹上进行微调 (fine-tune)。
4. 重复上述步骤。

该方法切实有效。在无需新增人工标注的情况下，GSM8K 和 CommonsenseQA 的性能均得到了提升。但该循环存在一个固有偏差 (built-in bias)：只要最终答案正确，无论推理过程本身是否严谨，对应的推理依据都会被保留。V-STaR (Hosseini et al., 2024) 通过引入一个训练得到的验证器 (verifier) 修复了这一问题；Quiet-STaR (Zelikman et al., 2024) 则将这一思路泛化至逐词内部推理依据 (per-token internal rationales)。

## 概念

### STaR: bootstrap on what worked

Start from a base model with some weak reasoning ability. On each training problem, sample a rationale plus answer. If the answer matches the label, keep the (problem, rationale, answer) triple. Fine-tune the model on the kept set. Repeat.

One twist matters. If the model can never get a problem right, the loop cannot learn on it. STaR adds **rationalization**: for problems the model fails, inject the correct answer as a hint and re-prompt the model to produce a rationale that leads to it. Rationalized rationales are added to the training set.

Result in the original paper (Zelikman et al., 2022): a GPT-J base model improved on GSM8K from 5.8% to 10.7% through repeated STaR rounds with rationalization — about 5 percentage points absolute. On CommonsenseQA, STaR-trained GPT-J 6B reached 72.5%, comparable to a fine-tuned GPT-3 175B (~73%) — a roughly 30x larger model trained on hand-annotated rationales.

### V-STaR: train a verifier with DPO

STaR throws away incorrect rationales. Hosseini et al. (2024) observed those are also data: every pair of (rationale, "is this correct") can train a verifier. They use Direct Preference Optimization over both correct and incorrect solutions to build a ranker. At inference time, sample N rationales and pick the verifier's top choice.

Reported delta: +4 to +17 percentage points over prior self-improvement baselines on GSM8K and MATH, with most of the gain coming from using the verifier for inference-time selection rather than for additional generator fine-tuning.

### Quiet-STaR: per-token internal rationales

Zelikman et al. (2024) asked: what if the model learns to generate a short internal rationale at every token position, not just between problem and answer? Quiet-STaR trains a model to emit a hidden "thought" before each predicted token, then mixes the thought-aware prediction with the baseline prediction via a learned weight.

Result: Mistral 7B gained absolute zero-shot improvements on GSM8K from 5.9% to 10.9% and CommonsenseQA from 36.3% to 47.2% without task-specific fine-tuning. The model learned "when to think" — hard tokens get longer internal rationales; easy ones get almost none.

### Why all three share a safety concern

All three methods use the final answer as the gradient signal. A rationale that reaches the right answer via flawed reasoning — exploiting a shortcut, guessing, or using a non-generalizing pattern — gets positively reinforced. On in-distribution problems the shortcut works. On out-of-distribution problems it breaks silently.

V-STaR's verifier mitigates by learning to rank rationales, but the verifier is trained on the same label set. It can learn to prefer well-formatted wrong reasoning over honest uncertainty. The safer design is to combine STaR-style data with (a) process-supervised reward models (rewarding intermediate steps, not just answers) and (b) held-out OOD evaluation that breaks simple shortcuts.

### Comparison

| Method | Training signal | Inference cost | Data waste | Known failure mode |
|---|---|---|---|---|
| STaR | keep (rationale, answer) if correct | 1x | discards all incorrect rationales | shortcut rationales |
| STaR + rationalization | above + correct-answer hinted retries | 1x | less | rationalized rationales may be implausible |
| V-STaR | STaR + DPO verifier from both classes | Nx (best-of-N) | minimal | verifier can reinforce confident wrongness |
| Quiet-STaR | per-token rationale + mixing weight | 1.5-3x | minimal | still answer-conditioned gradient |

### Where this sits in the 2026 stack

STaR is old. But the pattern reappears everywhere in 2025-2026. RL on verifiable math problems (DeepSeek-R1, Kimi-k1.5, o1) is STaR's answer-conditioned gradient signal, scaled up. Process reward models (Lightman et al., 2023; OpenAI's "Let's verify step by step") are the process-supervised alternative. AlphaEvolve (Lesson 3) is STaR for code, with a program evaluator instead of a label. Darwin Godel Machine (Lesson 4) is STaR for the agent scaffolding itself.

Understanding STaR makes all of these click. It is the minimum-viable self-improvement loop.

## 使用方法

`code/main.py` 在一个简易算术任务上运行了模拟的 STaR 循环 (STaR loop)。你可以观察：

- 准确率如何在自举轮次 (bootstrap rounds) 中逐步攀升。
- 捷径 (shortcuts) 是如何悄然混入的：模拟器包含一个“惰性”推理依据类 (rationale class)，它有 40% 的概率能得出正确答案，但泛化能力很差。请观察 STaR 是否会保留它们。
- 验证器 (verifier)（V-STaR 风格）如何在推理 (inference) 阶段发挥作用，却无法完全剔除训练 (training) 阶段引入的捷径。

## 发布上线

`outputs/skill-star-loop-reviewer.md` 可帮助你在开始训练前，对拟定的自学推理流水线（self-taught-reasoning pipeline）进行审查。

## 练习

1. 运行模拟器 (simulator)。将捷径频率 (shortcut frequency) 设为 0，再设为 0.4。尽管两次运行在训练分布 (training distribution) 上的准确率均超过 90%，但最终准确率 (final accuracy) 的差异有多大？

2. 在模拟器中添加一个预留的分布外测试 (held-out OOD test)。从不同分布中抽取问题，并在分布内 (in-distribution) 和分布外 (OOD) 数据集上评估自举模型 (bootstrapped model)。量化两者之间的性能差距。

3. 阅读 Quiet-STaR 论文 (arXiv:2403.09629) 的第 3 节。请分别用三句话解释“思维结束”标记 ("end-of-thought" token) 与混合权重头 (mixing-weight head)。

4. 将 STaR 的“正确则保留”过滤器 (keep-if-correct filter) 与一种过程监督替代方案 (process-supervised alternative) 进行对比，该方案会对每个推理步骤 (rationale step) 独立给予奖励。请指出两者在标注成本 (labelling cost) 上的差异，以及可能产生的质量差异。

5. 设计一项评估方案，用于检测部署模型 (deployed model) 中的捷径推理 (shortcut rationales)。该方案无需完美，但必须能够打破 STaR 循环 (STaR loop) 所会强化的那些最基础的捷径。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| 自学习推理器 (Self-Taught Reasoner) | “自学推理器” | 在模型生成的、能得出正确答案的推理过程上进行微调；循环迭代 |
| 事后合理化 (Rationalization) | “提示重试” | 针对基础模型未能解决的问题，注入正确答案并重新提示模型生成推理过程 |
| 验证器 STaR (Verifier STaR) | “验证器 STaR” | 使用正确与错误的推理过程对验证器进行 DPO（直接偏好优化）训练，并在推理阶段用于结果筛选 |
| 静默 STaR (Quiet-STaR) | “逐 Token 推理” | 在每个 Token 位置生成隐藏思维；与基线预测结果相混合 |
| 答案条件梯度 (Answer-conditioned gradient) | “基于结果的信号” | 训练循环仅对最终答案给予奖励，而非推理步骤 |
| 过程奖励模型 (Process reward model) | “步骤级验证器” | 基于每一步的正确性而非最终结果进行训练的奖励模型——与 STaR 形成对比 |
| 捷径推理 (Shortcut rationale) | “答案正确，推理错误” | 通过无法泛化的模式得出标签的推理过程；STaR 会保留此类推理 |

## 延伸阅读

- [Zelikman 等人 (2022)。STaR：通过推理实现自举推理](https://arxiv.org/abs/2203.14465) — 原始论文。
- [Hosseini 等人 (2024)。V-STaR：为自学推理器训练验证器](https://arxiv.org/abs/2402.06457) — 引入了用于推理时选择的直接偏好优化 (DPO) 验证器。
- [Zelikman 等人 (2024)。Quiet-STaR：语言模型可自学在输出前思考](https://arxiv.org/abs/2403.09629) — 逐词元 (token) 的内部推理依据。
- [Lightman 等人 (2023)。Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — 过程奖励模型 (Process Reward Models)，提供替代性梯度信号。
- [DeepSeek-R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 在可验证任务上应用强化学习 (Reinforcement Learning)，并将 STaR 扩展至前沿模型训练。