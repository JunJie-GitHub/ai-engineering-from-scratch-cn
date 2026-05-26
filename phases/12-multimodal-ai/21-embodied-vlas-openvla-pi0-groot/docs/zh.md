# 具身视觉-语言-动作（Vision-Language-Action, VLA）模型：RT-2、OpenVLA、π0、GR00T

> 模型首次从网站读取食谱并在厨房机器人中执行，是由 RT-2（Google DeepMind，2023 年 7 月）实现的。RT-2 将动作离散化为文本词元（text tokens），在网页数据与机器人动作数据上联合微调（co-fine-tuning）视觉语言模型（Vision-Language Model, VLM），证明了网络规模的视觉-语言知识能够迁移至机器人控制领域。OpenVLA（2024 年 6 月）发布了开源的 7B 参考模型。Physical Intelligence 的 π0 系列（2024-2025 年）引入了流匹配（flow-matching）动作专家模块。NVIDIA 的 GR00T N1（2025 年 3 月）为大规模人形机器人提供了双系统（dual-system）控制能力。VLA 基础原语（primitive）——即一个能看、能读、能执行的单一模型——是连接本阶段理解模型与第 15 阶段自主系统（Autonomous Systems）的桥梁。

**Type:** 学习
**Languages:** Python（标准库，动作分词器 + VLA 推理骨架）
**Prerequisites:** 第 12 阶段 · 05（LLaVA），第 15 阶段（自主系统，参考）
**Time:** 约 180 分钟

## 学习目标

- 阐述动作分词（action tokenization）技术：离散区间编码（discrete bin encoding，RT-2）、FAST 高效动作词元、连续流匹配动作（π0）。
- 解释为何在网页数据与机器人数据上进行联合微调能够保留通用知识向新任务的迁移能力。
- 在同一机器人任务上对比 OpenVLA（开源 7B Llama+VLM）、π0（流匹配）与 GR00T N1（双系统）的表现。
- 指出 Open X-Embodiment 数据集及其作为 RT-X 训练语料库的作用。

## 问题背景

自 20 世纪 70 年代以来，能够根据自然语言指令完成家务的机器人一直是研究目标。2020 年代的答案是：视觉-语言-动作（VLA）模型。它采用与视觉问答（Visual Question Answering, VQA）相同的视觉语言模型架构，但输出的是动作（关节力矩（joint torques）、末端执行器位姿（end-effector poses）、离散指令（discrete commands））而非文本。

VLA 面临的特有挑战：

1. 动作空间是连续（关节角度、力）且高维的（7 自由度（Degree of Freedom, DOF）机械臂 + 3 自由度夹爪 = 在 30 Hz 频率下为 10 维）。
2. 机器人专用训练数据稀缺。Open X-Embodiment 数据集仅包含约 100 万条轨迹（trajectories），而网络图文数据则超过 50 亿。
3. 控制频率至关重要。30 Hz 的控制循环（control loop）意味着每个动作仅有 33 毫秒的计算预算。
4. 安全性。错误的动作可能损坏硬件、伤及人员或破坏财产。

## 核心概念

### 动作分词（Action Tokenization）(RT-2)

RT-2 的巧妙之处在于：将每个关节目标表示为量化后的文本词元（Token）。将归一化后的 [-1, 1] 范围离散化为 256 个区间（Bins），并将每个区间映射到一个词汇表 ID。一个 10 自由度（Degree of Freedom, DOF）的动作在每个控制步长中会转化为 10 个词元。

在混合数据集上对 PaLM-X 视觉语言模型（Vision-Language Model, VLM）进行联合微调（Co-fine-tuning）：

- 网络图文对（用于图像描述生成（Captioning）、视觉问答（Visual Question Answering, VQA））。
- 机器人演示数据，将动作表示为词元。

模型的输入流程为：“拿起红色方块”（语言）→ 图像（视觉）→ 10 个词元的动作序列（离散化的关节目标）。网络预训练保留了通用知识的迁移能力：即使训练数据中从未出现过“快速移动”一词，RT-2 也能理解并执行“朝快速移动的物体靠近”的指令。

根据 RT-2 论文，其推理频率为 3-5 Hz，主要受限于 VLM 的自回归解码（Autoregressive Decoding）过程。

### OpenVLA —— 开源 7B 参考模型

OpenVLA（Kim 等人，2024 年 6 月）是 RT-2 的开源权重等效版本。它采用 7B 参数的 Llama 作为主干网络，结合 DINOv2 与 SigLIP 双视觉编码器，并在 256 个区间上进行动作分词。

该模型在 Open X-Embodiment 数据集（涵盖 22 种机器人的 97 万条轨迹）上进行训练。内置低秩自适应（Low-Rank Adaptation, LoRA）微调支持，便于适配新型机器人。

推理性能：在 A100 上结合量化技术可达 4-5 Hz。该速度足以应对慢速操作任务，但无法满足高频控制需求。

### FAST 分词器 —— 更快的动作解码

Pertsch 等人（2024 年）指出，离散区间分词效率较低——大多数动作往往集中在分桶空间的一个小区域内。FAST（频域动作序列分词器，Frequency-domain Action Sequence Tokenizer）通过离散余弦变换（Discrete Cosine Transform, DCT）压缩动作序列，并对变换系数进行量化。

一条 30 步的动作轨迹仅需约 10 个 FAST 词元，而非 300 个离散区间词元。推理速度提升 3-5 倍，且不会造成性能损失。

### π0 与流匹配（Flow-Matching）动作

Physical Intelligence 推出的 π0（Black 等人，2024 年 10 月）使用流匹配动作专家模块替代了离散的动作词元：

- 一个小型动作 Transformer 读取 VLM 的隐藏状态，并通过整流流（Rectified Flow）输出连续的 50 步动作序列。
- 动作头（Action Head）使用流匹配损失进行训练；VLM 的预训练过程保持不变。
- 推理阶段：仅需约 5 个去噪步骤即可生成完整动作序列，等效于 50 Hz 的控制频率。

π0 的宣称优势：在广泛的操控任务基准上优于 OpenVLA 和 Octo。连续动作的建模方式保留了被离散化所破坏的动作平滑性。

π0.5 和 π0-FAST 属于渐进式升级版本。其中 π0-FAST 将 FAST 分词技术与流匹配相结合。

### GR00T N1 —— 面向人形机器人的双系统架构

NVIDIA 的 GR00T N1（2025 年 3 月）专为全身自由度超过 30 的人形机器人设计：

- 系统 2（慢思考）：大型 VLM 负责解析场景与指令，以约 1 Hz 的频率生成高层子目标。
- 系统 1（快思考）：小型动作头 Transformer 根据子目标生成 50-100 Hz 的低层关节控制指令。

该架构划分对应了卡尼曼的“快思考与慢思考”理论：系统 2 负责规划，系统 1 负责执行。其优势在于：大参数 VLM 的慢速规划不会阻塞快速控制回路；系统 1 保持轻量级以确保低延迟。

GR00T N1.7（2025 年末）进一步优化了数据扩展能力。GR00T 系列利用来自 Omniverse 的仿真到现实（Sim-to-Real）数据进行微调。

### Open X-Embodiment 数据集

核心训练数据。RT-X（2023 年 10 月）整合了 22 个数据集，涵盖 22 种机器人的 100 万条轨迹。Open X-Embodiment 已成为业界通用的标准语料库：

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table。
- 每个样本包含：（机器人状态、相机视角、指令、动作序列）。
- 数据清洗与标准化：统一动作空间、归一化关节范围、调整相机分辨率。

OpenVLA 和 π0 均基于 Open X-Embodiment 进行训练。针对特定机器人的领域差异（Domain Gap），可通过在 100-1000 条特定任务演示数据上进行 LoRA 微调来弥合。

### 联合微调与纯机器人数据微调的对比

联合微调将网络 VQA 数据与机器人轨迹数据混合使用。数据比例至关重要：VQA 数据过多会导致模型遗忘动作控制能力；机器人数据过多则会使模型丧失通用知识。

RT-2 的比例约为 1:1。OpenVLA 的网络数据与机器人数据比例约为 0.5:1。π0 的比例与之类似。具体比例需根据数据集规模作为超参数进行调优。

仅使用机器人数据训练会生成任务专用模型，在面对分布外（Out-of-Distribution, OOD）指令时容易失效。联合微调的关键价值在于：它能让模型从“拿起演示中的红色方块”泛化到“拿起从左数第三大的物体（全新表述）”。

### 安全性与动作限制

每个投入实际生产的 VLA 模型均配备以下安全机制：

- 硬性关节限位（扭矩不可超出规格上限）。
- 速度限制（采用软截断处理）。
- 工作空间边界约束（末端执行器不可移出桌面范围）。
- 针对新任务引入人在回路（Human-in-the-Loop）审批机制。

这些机制作为控制层检查模块独立于 VLA 之外运行。VLA 的输出仅作为建议，而非直接执行的指令。

## 实践应用

`code/main.py`:

- 实现了 256 个离散区间的动作词元化（Action Tokenization）与逆词元化（De-tokenization）。
- 勾勒了基于离散余弦变换（DCT）与量化（Quantization）的 FAST 词元化器（FAST Tokenizer）架构。
- 对比了不同方案（离散区间、FAST、连续流）在每个动作步骤下的词元数量。
- 打印了 RT-2 → OpenVLA → π0 → GR00T 的技术演进脉络摘要。

## 交付说明

本课时将生成 `outputs/skill-vla-action-format-picker.md` 文件。针对给定的机器人任务（如机械臂操作、导航、人形机器人全身控制），该文件会在离散区间 + RT-2、FAST + OpenVLA、流匹配（Flow-Matching）+ π0 或双系统（Dual-System）+ GR00T 之间进行方案选型。

## 练习题

1. 一个 10 自由度（DOF）机械臂，控制频率为 30 Hz。采用 256 个离散区间的词元化方案，每秒会生成多少个词元（Token）？7B 参数的视觉语言模型（VLM）能否跟上该速度？

2. FAST 词元化方案可将 30 步的轨迹（Trajectory）压缩至约 10 个词元。如果轨迹包含高频运动（例如敲击鼓面），用户会损失哪些信息？

3. π0 的流匹配头（Flow-Matching Head）约需 5 步即可完成去噪。请将其吞吐量与 OpenVLA 在 4-5 Hz 频率下的自回归解码（Autoregressive Decode）进行对比。

4. GR00T 的系统 1 / 系统 2（System 1 / System 2）划分借鉴了卡尼曼（Kahneman）的理论。请提出一种不同的划分方式（例如系统 3？），以期为双足行走（Bipedal Walking）任务提供帮助。

5. 阅读 Open X-Embodiment 论文第 4 节关于数据集策展（Dataset Curation）的内容。列出防止领域数据泄露（Domain Leakage）的三条策展规则。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| VLA（视觉-语言-动作模型） | “视觉-语言-动作” | 接收图像与指令输入，并输出动作控制指令的模型 |
| 动作词元化（Action Tokenization） | “离散区间” | 将连续的关节目标值按维度量化为 256 个区间，每个区间对应一个词表 ID |
| FAST 词元化器（FAST Tokenizer） | “频域动作词元” | 结合离散余弦变换（DCT）与量化技术，将 30 步轨迹压缩至约 10 个词元 |
| 联合微调（Co-fine-tune） | “混合网络数据与机器人数据” | 在机器人演示数据之外，同时使用网络视觉问答（VQA）数据进行训练，以保留模型的通用知识 |
| 流匹配动作头（Flow-Matching Action Head） | “π0 连续输出” | 一个小型 Transformer，通过整流流（Rectified Flow）生成 50 步的动作序列 |
| 系统 1 / 系统 2（System 1 / System 2） | “双系统控制” | 大型 VLM 负责慢速规划，小型动作头负责快速执行；GR00T 的典型架构模式 |
| Open X-Embodiment（开放跨具身数据集） | “RT-X 数据集” | 包含 100 万条轨迹的跨机器人数据集；作为核心训练语料库 |

## 延伸阅读

- [Brohan 等人 — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim 等人 — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black 等人 — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment 协作组 — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)