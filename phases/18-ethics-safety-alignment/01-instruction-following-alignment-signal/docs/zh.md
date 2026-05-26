# 指令遵循（Instruction-Following）作为对齐信号

> 后续对基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）的每一项批评都直指该流程。在研究优化压力如何扭曲代理目标（Proxy）之前，你必须先看清这个代理目标本身。InstructGPT（Ouyang 等人，2022）确立了参考架构：在指令-回复对上进行监督微调（Supervised Fine-Tuning, SFT），在成对偏好排序上训练奖励模型（Reward Model），以及针对奖励模型使用近端策略优化（Proximal Policy Optimization, PPO）进行训练，并对 SFT 策略施加 KL 散度惩罚（KL Penalty）。在人类偏好评估中，1.3B 参数的 InstructGPT 优于 175B 参数的 GPT-3。正是这一项结果，使得 2026 年的所有前沿实验室仍在交付采用 RLHF 范式的后训练（Post-Training）流程。

**类型：** 学习
**语言：** Python（标准库，玩具级三阶段流程）
**前置知识：** 第 10 阶段 · 06（SFT）、第 10 阶段 · 07（RLHF）、第 10 阶段 · 08（DPO）
**耗时：** 约 45 分钟

## 学习目标

- 指出 InstructGPT 流程的三个阶段及其各自使用的损失函数（Loss）。
- 解释为何经过指令微调（Instruction-Tuning）的 1.3B 模型在人类偏好评估中击败了原始的 175B GPT-3。
- 说明第三阶段中的 KL 惩罚旨在防范什么，以及为何移除它会导致模型退化为模式搜索（Mode-Seeking）行为。
- 描述对齐税（Alignment Tax）以及 Ouyang 等人用于缓解该问题的 PPO-ptx 方法。

## 问题背景

预训练语言模型（Pre-trained Language Models）的任务是续写文本，而非回答问题。如果你向 GPT-3 提问“写一个反转列表的 Python 函数”，它通常会返回另一个提示词，因为其训练数据分布（Training Distribution）绝大多数是网页文本，而网页文本的后续内容依然是网页文本。模型确实在执行它的任务——只是这个任务本身设定错了。

各主流实验室用来解决此问题的代理目标（Proxy）是人类偏好。将两个生成结果交给标注员；标注员选出更优的一个；奖励模型（Reward Model）则学习标注员的偏好。随后，强化学习（Reinforcement Learning, RL）循环会推动策略（Policy）向奖励模型打分更高的输出方向偏移。这就是 InstructGPT 核心思想的三句话概括。论文的其余部分全是工程实现。

## 核心概念

### 阶段 1：监督微调（Supervised Fine-Tuning, SFT）

收集提示-回复对（prompt-response pairs），其中的回复应符合人类善意意图。Ouyang 等人使用了来自标注员和 OpenAI API 的 1.3 万条提示。使用标准交叉熵损失（cross-entropy loss）在此数据上对基座模型（base model）进行微调。

SFT 带来的效果：模型现在会回答问题，而非单纯续写文本。SFT 无法提供的：当存在多个合理答案时，关于评估者更偏好哪一个的任何信号。

### 阶段 2：奖励模型（Reward Model, RM）

针对每条提示，从 SFT 模型中采样 K 个生成结果。由标注员对这些结果进行排序。训练一个奖励模型，使其能够为任意提示-回复对打分，并确保在 `y_w` 优于 `y_l` 的样本对上满足以下条件：

L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))

这就是 Bradley-Terry 成对偏好损失（Bradley-Terry pairwise preference loss）。RM 通常以 SFT 模型为初始化起点，将其语言模型头（LM head）替换为标量头（scalar head）。

奖励模型通常规模较小：对于 175B 的 InstructGPT，6B 的奖励模型就已足够。它们也很脆弱——论文第 5 节主要讨论了在小规模实验中就已显现的奖励欺骗行为（reward hacking）。

### 阶段 3：带 KL 惩罚的近端策略优化（Proximal Policy Optimization, PPO）

定义优化目标：

J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))

使用 PPO 最大化该目标。KL 散度项（KL term）用于防止策略 `pi` 过度偏离 SFT 策略。若没有该项，优化器会找到对抗样本（adversarial examples）——这些字符串在 RM 下得分很高，仅仅是因为 RM 从未见过它们，而非人类真正偏好它们。

KL 系数 `beta` 是基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）中最重要的单一超参数。设置过低会导致奖励欺骗；设置过高则无法带来超越 SFT 的性能提升。

### 对齐税（Alignment Tax）

经过 RLHF 后，模型更受人类青睐，但在标准基准测试（如 SQuAD、HellaSwag、DROP）上的性能会出现退化。Ouyang 等人将此现象称为“对齐税”（alignment tax），并通过 PPO-ptx 加以解决：将预训练梯度混合到强化学习目标中，使模型不会遗忘那些从未获得过奖励的下游任务能力。

J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]

PPO-ptx 现已成为行业标准。Anthropic、DeepMind 和 Meta 等机构均采用其某种变体。

### 最终结果

标注员在约 70% 的情况下更偏好 1.3B 的 InstructGPT（SFT + RM + PPO-ptx），而非 175B 的基座 GPT-3。在来自生产环境流量的隐藏测试提示上，这一差距会进一步拉大。从该数据中可以得出两点结论：

1. 对齐（alignment）与能力（capability）是两个不同的维度。175B 模型具备更强的能力；1.3B 模型具备更高的对齐度；而标注员更偏好对齐度更高的模型。
2. 能力下限由基座模型决定。你无法通过 RLHF 让基座模型掌握它从未接触过的知识。

### 为何将其作为第 18 阶段的参考基准

后续课程中的每一项批评——奖励欺骗（Lesson 2）、直接偏好优化（Direct Preference Optimization, DPO）（Lesson 3）、讨好行为（sycophancy）（Lesson 4）、宪法式 AI（Constitutional AI, CAI）（Lesson 5）、潜伏代理（sleeper agents）（Lesson 7）、对齐伪装（alignment faking）（Lesson 9）——都针对该流程的某个环节提出了质疑。奖励欺骗攻击的是阶段 2；DPO 将阶段 2 和阶段 3 合并；CAI 用规则替代了人类标注员；讨好行为表明标注员本身是一个带有偏见的信号；对齐伪装则证明策略可以完全绕过阶段 3。如果不先在脑海中建立起该流程的框架，你将无法理解这些批评中的任何一项。

## 使用它

`code/main.py` 在示例偏好数据（toy preference data）上模拟了这三个阶段。基础“策略（policy）”相当于在动作 {A, B, C} 上投掷一枚有偏硬币（biased coin）。第一阶段监督微调（Supervised Fine-Tuning, SFT）在 200 个提示（prompt）上模仿标注员（labeler）的行为。第二阶段基于 500 组成对排序（pairwise ranking）拟合 Bradley-Terry 奖励模型（reward model）。第三阶段执行简化的近端策略优化（Proximal Policy Optimization, PPO）更新，并针对 SFT 策略施加 KL 惩罚（KL penalty）。你可以观察奖励（reward）的攀升、KL 散度（KL divergence）的增长以及策略漂移（policy drift）现象；你也可以关闭 KL 项，观察在 50 次更新步数内出现的奖励黑客行为（reward hacking）。

观察重点：

- `beta = 0.1` 与 `beta = 0.0` 时的奖励轨迹（reward trajectory）对比。
- 训练步数过程中的 KL(pi || pi_SFT) 变化。
- 最终动作分布（action distribution）与标注员偏好的对比。

## 交付它

本课时将生成 `outputs/skill-instructgpt-explainer.md`。给定一段基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）流水线描述或论文摘要，该工具能够识别出三个阶段中哪一个被修改、每个阶段使用了何种损失函数（loss），以及是否存在 KL 惩罚或等效的正则化项（regularizer）。

## 练习

1. 运行 `code/main.py`。将 `beta = 0.0`，并报告 200 步 PPO 后的动作分布。用一段话解释其模式搜索行为（mode-seeking behaviour）。

2. 修改奖励模型，使其对动作 B 产生 +0.5 的偏差（模拟奖励缺陷）。使用 `beta = 0.1` 运行 PPO。KL 惩罚能否阻止策略利用该偏差？在什么 `beta` 值下，利用行为（exploitation）开始显现？

3. 阅读 Ouyang 等人（arXiv:2203.02155）的图 1。分别运行 PPO 1、5、20、100 步，并测量相对于 SFT 模型的偏好，以复现标注员偏好曲线。

4. 论文第 4.3 节指出，1.3B 参数的 InstructGPT 在约 70% 的情况下优于 175B 参数的 GPT-3。为什么在隐藏的生产环境提示（production prompts）上，该胜率会高于标注员自己的提示？

5. 在相同的偏好数据上，将 PPO 损失替换为直接偏好优化（Direct Preference Optimization, DPO）（Phase 10 · 08）。对比最终的策略漂移（相对于 SFT 的 KL 散度）和最终奖励。在奖励匹配的情况下，哪种方法的漂移程度更大？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| SFT（监督微调） | “指令微调” | 第一阶段：在提示-回复对上进行交叉熵微调 |
| 奖励模型（Reward Model） | “RM” | 基于成对标签使用 Bradley-Terry 训练的标量回归器，输入为（提示，回复） |
| Bradley-Terry 模型 | “成对偏好损失” | `-log sigmoid(r_w - r_l)`；将成对排序转化为二分类问题 |
| KL 惩罚（KL Penalty） | “正则化项” | `beta * KL(pi || pi_SFT)` —— 使强化学习策略保持在 SFT 锚点附近 |
| PPO-ptx | “混合预训练的 PPO” | 在 PPO 目标函数中加入一部分预训练对数似然（log-likelihood），以抵消对齐税（alignment tax） |
| 对齐税（Alignment Tax） | “RLHF 性能回退” | RLHF 未针对的标准基准测试在 RLHF 后出现的性能下降 |
| 标注员偏好（Labeler Preference） | “真实基准” | 人类排序的样本；奖励模型仅是该样本的统计代理（statistical proxy），而非“人类价值观”的代理 |

## 扩展阅读

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — InstructGPT 论文，为后续所有基于人类反馈的强化学习（RLHF）流程（pipeline）奠定了基础
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) — 将 RLHF 应用于文本摘要任务的先驱之作
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) — 基于偏好的强化学习（RL）原始理论框架
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) — Anthropic 在 InstructGPT 流程基础上扩展出的“有益且无害”（Helpful and Harmless, HH）模型