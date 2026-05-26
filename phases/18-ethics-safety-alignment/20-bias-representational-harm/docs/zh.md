# 大语言模型 (Large Language Models) 中的偏见与表征性危害 (Representational Harm)

> Gallegos, Rossi, Barrow, Tanjim, Kim, Dernoncourt, Yu, Zhang, Ahmed（《计算语言学》2024，arXiv:2309.00770）。2024 年奠基性综述，将表征性危害（Representational Harms，如刻板印象、群体抹除）与分配性危害（Allocational Harms，如资源分配不均）区分开来，并将评估指标分为基于嵌入 (Embedding-based)、基于概率 (Probability-based) 或基于生成文本 (Generated-text-based) 三类。2024-2025 年实证研究：An 等人（《PNAS Nexus》，2025 年 3 月）在针对 20 个入门级职位的自动化简历评估中，测量了 GPT-3.5 Turbo、GPT-4o、Gemini 1.5 Flash、Claude 3.5 Sonnet 和 Llama 3-70B 在性别与种族交叉 (Intersectional) 维度上的偏见。WinoIdentity（COLM 2025，arXiv:2508.07111）引入了针对交叉身份的基于不确定性的公平性评估 (Uncertainty-based Fairness Evaluation) 方法。Yu & Ananiadou 2025 在多层感知机 (Multilayer Perceptron, MLP) 层中识别出性别神经元；Ahsan & Wallace 2025 使用稀疏自编码器 (Sparse Autoencoders, SAEs) 揭示临床场景中的种族偏见；Zhou 等人 2024（UniBias）通过操纵注意力头 (Attention Heads) 进行去偏。元批判研究（arXiv:2508.11067）指出：过去十年的文献过度集中于二元性别偏见。

**类型：** 构建
**语言：** Python（标准库，基于嵌入的偏见探测玩具示例）
**前置要求：** 阶段 05（词嵌入 (Word Embeddings)），阶段 18 · 01（指令遵循 (Instruction Following)）
**时间：** 约 60 分钟

## 学习目标

- 定义表征性危害与分配性危害，并各举一个在大语言模型部署中的实例。
- 列出 Gallegos 等人（2024）提出的三类评估指标，并分别描述其中一种指标。
- 阐述交叉性 (Intersectionality) 概念，并说明 WinoIdentity 基于不确定性的公平性测量如何弥补单一维度偏见评估的不足。
- 描述两种用于分析偏见的机制可解释性 (Mechanistic Interpretability) 方法（如性别神经元、稀疏自编码器特征、注意力头操纵）。

## 问题阐述

此前的课程已涵盖故意危害（如越狱攻击 (Jailbreaks)、策略性密谋 (Scheming)）与安全治理。偏见则是一种非故意产生的危害——它源于训练数据分布、提示词框架设计或长期积累的设计选择。测量和降低偏见是一项独立于对抗鲁棒性 (Adversarial Robustness) 的方法论挑战。

## 核心概念

### 表征性危害与分配性危害（Representational vs. Allocational）

- **表征性危害（Representational harm）。** 包括刻板印象、群体抹除和贬低性描绘。若大型语言模型（LLM）将护士仅描绘为女性，即产生了表征性危害。
- **分配性危害（Allocational harm）。** 指不平等的实质性结果。若 LLM 系统性地给黑人求职者的简历打低分，即产生了分配性危害。

两者并不相同。一个模型可能在“表征上无偏见”（生成多样化的描绘），但在“分配上存在偏见”（给出不平等的推荐）。评估工作需要对两者进行衡量。

### 三类评估指标（Gallegos 等人，2024）

- **基于嵌入的方法（Embedding-based）。** 在预 RLHF（基于人类反馈的强化学习）嵌入向量上进行 WEAT（词嵌入关联测试）风格的测试。衡量身份词与属性词之间的统计关联。局限性：仅衡量表征，而非实际行为。
- **基于概率的方法（Probability-based）。** 计算符合刻板印象与违背刻板印象的续写结果的对数似然（Log-likelihood）。属于解码器端测量。能够捕捉部分行为偏见。
- **基于生成文本的方法（Generated-text-based）。** 在生成文本上进行下游任务测量。例如简历评分、推荐信撰写、对话交互。生态效度（Ecological validity）最高，但也最难复现。

### 交叉性（Intersectionality）

仅针对“性别”的偏见评估会遗漏那些仅在（性别，种族）组合下才会触发的偏见。An 等人（2025）发现，GPT-4o 在简历评分中对黑人女性的惩罚程度，高于单独对黑人男性或白人女性的惩罚。单轴评估无法捕捉此类现象。

WinoIdentity（COLM 2025）引入了基于不确定性的交叉公平性（Intersectional fairness）评估。它衡量模型在不同交叉身份元组上的结果不确定性是否存在差异，而不仅仅是关注点预测（Point prediction）。这种方法能够捕捉到模型在不同群体上犯错程度相同，但对某些群体的不确定性更高的情况，而这会导致不同的下游分配行为。

### 机制化方法（Mechanistic approaches）

2024-2025 年的可解释性（Interpretability）研究为偏见干预提供了机制化途径：

- **性别神经元（Gender neurons，Yu & Ananiadou 2025）。** 特定的多层感知机（MLP）神经元与性别特定行为相关。消融（Ablating）这些神经元可在能力损失有限的情况下降低性别差距指标。
- **基于稀疏自编码器的临床种族偏见分析（Clinical racial bias via SAEs，Ahsan & Wallace 2025）。** 稀疏自编码器（Sparse Autoencoder, SAE）特征将内部表征分解为可解释的维度；可识别并抑制与种族相关的特征。
- **UniBias（Zhou 等人，2024）。** 通过注意力头（Attention-head）操作实现零样本去偏见（Zero-shot debiasing）。特定的注意力头会放大对身份类别的敏感度；将这些头置零或重新加权，无需微调即可降低偏见。

### 元批判（Meta-critique）

一项涵盖十年文献的综述（arXiv:2508.11067，2025）指出，该领域过度集中于二元性别偏见。而其他维度——如残障、宗教、移民身份、多语言身份——受到的关注远少得多。该元批判认为，狭隘的关注焦点会因忽视而伤害边缘化群体：一个在二元性别上已充分去偏见的模型，可能在无人核查的维度上存在严重偏见。

### 在第 18 阶段中的定位

第 20-21 课正式讲解偏见与公平性。第 22 课涵盖隐私。第 23 课涵盖水印技术。这些内容构成了用户危害层，是对早期欺骗/安全层的补充。

## 使用它

`code/main.py` 构建了一个基于嵌入（Embedding）的简易偏见探测工具（Bias Probe）：在简单的共现嵌入（Co-occurrence Embedding）中，测量身份词（Identity Terms）与属性词（Attribute Terms）之间的类词嵌入关联测试（Word Embedding Association Test, WEAT）距离。你可以注入偏见并观察该指标如何触发；随后应用简单的去偏见（Debiasing）操作，并观察指标的部分回落。

## 交付它

本课时将生成 `outputs/skill-bias-eval.md`。给定模型卡片（Model Card）或公平性声明，它将审计涵盖三个指标类别（嵌入、概率、生成文本）的评估、交叉性（Intersectionality）覆盖范围，以及任何去偏见干预的机制。

## 练习

1. 运行 `code/main.py`。报告去偏见步骤前后的类 WEAT 偏见分数。解释为何该指标不会降至零。

2. 使用交叉性测试扩展该探测工具：（性别、种族）x（职业、家庭）。报告跨轴偏见分数。

3. 阅读 An 等人 2025 年的论文（PNAS Nexus）。找出他们报告的两个单轴性别评估会遗漏的交叉性效应。

4. Yu & Ananiadou 2025 识别出了性别神经元（Gender Neurons）。设计一个证伪实验，以区分“这些神经元导致性别偏见”与“这些神经元与性别偏见相关”。

5. 元批判指出该领域对二元性别的关注过于狭隘。选择一个研究不足的维度，并描述针对它的表征危害（Representational Harm）测量协议。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 表征危害（Representational Harm） | “刻板印象 / 抹除” | 对某一群体的偏见性刻画 |
| 分配危害（Allocational Harm） | “不平等的决策” | 对某一群体的偏见性物质结果 |
| WEAT | “嵌入测试” | 词嵌入关联测试；基于共现的偏见探测工具 |
| 交叉性（Intersectionality） | “复合身份效应” | 在多个身份维度交汇处产生的偏见 |
| 性别神经元（Gender Neurons） | “MLP 偏见神经元” | 其激活状态与特定性别行为相关的特定神经元 |
| SAE 特征（SAE Feature） | “可解释维度” | 由稀疏自编码器（Sparse Autoencoder）识别的特征；适用于机制性偏见分析 |
| UniBias | “注意力头去偏见” | 通过重新加权注意力头（Attention Heads）实现的零样本（Zero-shot）去偏见 |

## 延伸阅读

- [Gallegos 等人 — 大语言模型中的偏见与公平性：综述 (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — 权威综述
- [An 等人 — 交叉性简历评估偏见 (PNAS Nexus, 2025年3月)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343) — 五模型交叉性研究
- [WinoIdentity — 基于不确定性的交叉性公平性 (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111) — 新基准测试
- [UniBias — 注意力头操控 (Zhou 等人 2024, ACL)](https://arxiv.org/abs/2405.20612) — 零样本去偏见