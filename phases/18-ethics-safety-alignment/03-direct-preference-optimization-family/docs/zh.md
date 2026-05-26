# 直接偏好优化（Direct Preference Optimization, DPO）家族

> Rafailov 等人（2023）证明，基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）的最优解可通过偏好数据表示为闭式解（closed form），因此可跳过显式奖励模型（explicit reward model）直接优化策略（policy）。这一洞见催生了一个算法家族——IPO、KTO、SimPO、ORPO、BPO，它们各自修复了 DPO 的一种失效模式（failure mode）。到 2026 年，直接对齐算法（Direct Alignment Algorithms, DAAs）在前沿后训练（post-training）任务中的部署量已超过近端策略优化（Proximal Policy Optimization, PPO）。但第 2 课中的过度优化（over-optimization）曲线依然适用：DAAs 并未摆脱古德哈特定律（Goodhart's Law），它们只是改变了定律生效的切入点。

**Type:** 学习
**Languages:** Python（标准库，六种变体偏好损失比较器）
**Prerequisites:** 第 18 阶段 · 01（InstructGPT）、第 18 阶段 · 02（奖励欺骗/Reward hacking）、第 10 阶段 · 08（DPO 基础）
**Time:** 约 75 分钟

## 学习目标

- 从带 KL 散度约束的 RLHF 最优解推导 DPO 的闭式解。
- 说明 IPO、KTO、SimPO、ORPO、BPO 各自修复了 DPO 中的哪种失效模式。
- 区分“隐式奖励差距（implicit reward gap）”与“偏好强度（preference strength）”，并解释 IPO 的恒等映射（identity mapping）为何重要。
- 解释为何 Rafailov 等人（NeurIPS 2024）证明，即使没有显式奖励模型（RM），DAAs 仍会出现过度优化。

## 问题背景

RLHF 的目标函数（第 1 课）：

max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)

具有已知的最优解：

pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)

因此，奖励（reward）可由最优策略与参考策略的比值隐式定义：

r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)

将其代入 Bradley-Terry 偏好似然（Bradley-Terry preference likelihood）后，配分函数（partition function）`Z(x)` 会因仅依赖于 `x` 而被消去。最终剩下的仅是关于策略参数的损失函数——无需奖励模型。这就是 DPO。

难点在于：该推导假设最优解可达、偏好数据处于分布内（in-distribution），且参考策略是真实的模式锚点（mode anchor）。这些假设在现实中均无法严格成立。该算法家族的每个成员都针对其中某一项被违背的假设进行了修复。

## 核心概念

### DPO（Rafailov 等，2023）

L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)

潜在问题：

- 隐式奖励差距（implicit reward gap）`beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)`是无界的。微小的偏好差异也可能产生任意大的差距。
- 损失函数会驱使被选中（chosen）和被拒绝（rejected）响应的对数概率（log-probs）向相反方向变化。只要被拒绝响应的对数概率下降得更快，它就可能将被选中响应的绝对对数概率拉低。这就是“被选中响应退化（Degraded Chosen Response）”现象。
- 分布外（out-of-distribution）偏好（例如罕见样本对与罕见样本对的比较）会产生任意的隐式奖励。

### IPO（Azar 等，2024）

恒等偏好优化（Identity Preference Optimization, IPO）用偏好概率上的恒等映射替换了对数 Sigmoid 函数。其损失函数变为针对有界目标的平方误差：

L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2

其边界（margin）被限制在 `1/(2 beta)` 以内。偏好强度与隐式奖励差距成正比，不会出现数值爆炸（blow-up）。

### KTO（Ethayarajh 等，2024）

卡尼曼-特沃斯基优化（Kahneman-Tversky Optimization, KTO）完全摒弃了成对比较结构。给定单个带标签的输出以及二元的“期望（desirable）”或“不期望（undesirable）”信号，它会将其映射为前景理论（prospect-theory）效用：

v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)

并对收益和损失赋予不同的权重（损失厌恶，loss aversion）。优势在于：你可以使用非成对数据（unpaired data），这类数据的获取要丰富得多。

### SimPO（Meng 等，2024）

简单偏好优化（Simple Preference Optimization, SimPO）使训练信号与生成过程保持一致。它完全移除了参考策略（reference policy），并按长度对对数似然（log-likelihood）进行归一化：

L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)

并引入边界参数 `gamma` 以稳定训练。长度归一化消除了利用 DPO 长度偏差失效模式（length-bias failure mode）的动机（在 DPO 中，较长的 `y_w` 会因构造方式天然产生更大的对数概率差距）。

### ORPO（Hong 等，2024）

几率比偏好优化（Odds-Ratio Preference Optimization, ORPO）在标准的监督微调（Supervised Fine-Tuning, SFT）负对数似然损失中增加了一个偏好项：

L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))

无需参考策略——SFT 项本身充当正则化器。可直接从基座模型（base model）单阶段训练至对齐模型（aligned model），无需单独的 SFT 检查点（checkpoint）。

### BPO（ICLR 2026 投稿，OpenReview id=b97EwMUWu7）

该算法指出了“被选中响应退化（Degraded Chosen Responses）”问题：DPO 虽然保持了 `y_w > y_l` 的排序，但 `y_w` 的绝对对数概率可能会下降。BPO 增加了一行修正代码，对被选中响应的概率下降进行惩罚。据报道，在 Llama-3.1-8B-Instruct 的数学推理任务上，其准确率较 DPO 提升了 10.1%。

### 普遍结论：直接对齐算法（DAAs）仍存在过度优化问题

Rafailov 等人在《直接对齐算法中奖励模型过度优化的缩放定律》（NeurIPS 2024）一文中，在多个数据集和不同的 KL 散度预算（KL budgets）下，使用 DPO、IPO 和 SLiC 训练了策略。其“真实奖励 vs KL 散度”曲线呈现出与 Gao 等人研究中相同的“先升后降（peak-and-collapse）”形态。隐式奖励在训练过程中会查询分布外样本；KL 正则化无法稳定这一现象。

直接对齐算法（Direct Alignment Algorithms, DAAs）并未摆脱古德哈特定律（Goodhart's Law）。它们只是将问题爆发的表面从“奖励模型过度优化”转移到了“参考策略比率过度优化”。通用的修复方案——使用更优质的数据、模型集成（ensembles）和早停（early stopping）——对两者均适用。

### 算法选择指南（2026）

- 若拥有大量成对偏好数据：使用保守 `beta` 值的 DPO；若长度偏差明显，则选用 SimPO。
- 若拥有非成对的二元反馈数据：选用 KTO。
- 若希望从基座模型直接进行单阶段训练流水线：选用 ORPO。
- 若在 DPO 日志中发现被选中响应的对数概率退化：选用 BPO。
- 若偏好强度差异较大且 DPO 出现饱和现象：选用 IPO。

目前各实验室通常会在基准测试集（battery）上并行运行这五种算法，并针对具体任务择优选用。没有理由认为数学推理和安全对齐任务的最优算法会是同一个。

## 实践应用

`code/main.py` 在一个示例偏好数据集上对比了六种损失函数：直接偏好优化 (Direct Preference Optimization, DPO)、身份偏好优化 (Identity Preference Optimization, IPO)、卡尼曼-特沃斯基优化 (Kahneman-Tversky Optimization, KTO)、简单偏好优化 (Simple Preference Optimization, SimPO)、优势比率偏好优化 (Odds Ratio Preference Optimization, ORPO) 和边界偏好优化 (Boundary Preference Optimization, BPO)。该数据集中真实的偏好强度因样本对而异。每种损失函数均在相同的 500 对样本上，配合一个小型 softmax 策略 (softmax policy) 进行优化。代码会绘制出每种方法的最终胜率、选中响应对数概率漂移 (chosen-log-prob drift) 以及隐式奖励分布范围 (implicit-reward spread)。

## 交付成果

本节将生成 `outputs/skill-preference-loss-selector.md`。根据数据集的统计特征（成对与非成对数据、偏好强度可变与均匀、长度分布）以及训练目标（单阶段训练或先监督微调后偏好优化 (SFT-then-preference)），推荐一种合适的偏好损失函数，并说明该函数能够防范的失效模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 最终的选中响应对数概率下降值。BPO 应保留更高的选中响应绝对概率——请验证这一点。
2. 修改偏好数据，使所有样本对的偏好强度相等。六种方法中哪一种最稳健？哪一种性能下降？解释 IPO 在此处的优势。
3. 使拒绝响应的平均长度为选中响应的 2 倍。在不更改其他设置的情况下，用数值展示 DPO 的长度利用 (length exploitation) 现象以及 SimPO 的修复方案。
4. Rafailov 等人（NeurIPS 2024）指出直接对齐算法 (Direct Alignment Algorithms, DAAs) 存在过度优化问题。复现一个单点版本：绘制“选中响应减去拒绝响应”的 KL 散度 (KL divergence) 图，并观察 DPO 在较大 beta 值下的过度优化现象。
5. 阅读 BPO 论文的摘要（OpenReview b97EwMUWu7）。写下 BPO 为 DPO 添加的那一行修正项。对照 `code/main.py` 中的实现进行验证。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| DPO | “无需奖励模型的 RLHF” | 基于 RLHF 闭式最优解推导的损失函数；仅更新策略参数 |
| 隐式奖励 (Implicit reward) | “对数比率” | `beta * log(pi(y|x) / pi_ref(y|x))` —— DPO 隐含的奖励值 |
| IPO | “有界 DPO” | 用恒等函数替换对数 Sigmoid；隐式奖励差值上限为 `1/(2 beta)` |
| KTO | “非成对 DPO” | 基于前景理论 (Prospect Theory) 的单标签效用函数，包含损失厌恶机制 |
| SimPO | “无参考 DPO” | 长度归一化的对数似然 + 间隔 (margin)；无需参考策略 |
| ORPO | “单阶段 DPO” | 负对数似然 (Negative Log-Likelihood, NLL) + 优势比率偏好项；从基座模型一次性完成训练 |
| BPO | “保留选中响应的 DPO” | DPO 加上对选中响应绝对对数概率下降的惩罚项 |
| 选中响应退化 (Degraded Chosen) | “选中响应概率下降” | 只要拒绝响应的概率下降得更快，DPO 就会降低选中响应的对数概率 |
| DAA | “直接对齐算法” | 任何跳过显式奖励模型 (Reward Model, RM) 的偏好损失方法 |

## 延伸阅读

- [Rafailov 等人 — 直接偏好优化 (Direct Preference Optimization) (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar 等人 — 理解人类偏好学习的通用理论范式 (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO (Identity Preference Optimization)
- [Ethayarajh 等人 — KTO (Kahneman-Tversky Optimization)：作为前景理论优化的模型对齐 (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (Simple Preference Optimization) (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (Odds Ratio Preference Optimization) (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — 行为保持优化 (Behavior Preservation Optimization) (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov 等人 — 直接对齐算法 (Direct Alignment Algorithms) 中奖励模型 (Reward Model) 过度优化的缩放定律 (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)