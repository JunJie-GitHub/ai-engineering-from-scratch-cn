---
name: nsa-integrator
description: 在长上下文预训练（Long-context Pre-training）运行中集成原生稀疏注意力（Native Sparse Attention, NSA）的计划。
version: 1.0.0
phase: 10
lesson: 17
tags: [nsa, 稀疏注意力, 长上下文, 预训练, 内核对齐, deepseek]
---

给定长上下文预训练运行规范（目标上下文长度、基础架构、可用训练 token 数、GPU 拓扑结构、部署目标），生成一份 NSA 集成计划。

生成以下内容：

1. 压缩块大小（Compression Block Size）`l`。选择 32、64 或 128。根据目标上下文长度进行论证：16k-32k 上下文对应 `l = 32`，64k-128k 对应 `l = 64`，256k 及以上对应 `l = 128`。较大的 `l` 意味着压缩后的键（Keys）更少，但路由信号（Routing Signal）会更粗糙。
2. Top-k 选择数量（Top-k Selection Count）。在 8 到 32 之间选择。论文默认值为 16。根据目标任务组合进行论证：推理密集型任务（数学、代码）受益于较高的 `k`，因为选择精度更为关键。检索密集型任务在较低的 `k` 下即可良好运行。
3. 滑动窗口（Sliding Window）`W`。选择 256、512 或 1024。默认值为 512。对于高度结构化的内容（如代码），局部上下文已足够，因此窗口应较短；对于散文类文本，窗口应较长。
4. 门控多层感知机（Gate MLP）。指定宽度与初始化方式。默认配置：从 `hidden` 维度到 3 的线性层，使用 `sigmoid` 或 `softplus` 激活函数。若门控权重发生坍缩并偏向某一分支，需发出警告——这表明 `l`、`k` 或 `W` 参数调优不当。
5. 内核（Kernel）选择。确认目标加速器是否支持 Triton 或 CUDA 内核。拒绝在推理阶段回退到密集注意力（Dense Attention）（NSA 的核心目的正是为了节省解码计算量）。若仅有前向内核而无反向后向内核，则拒绝进行预训练，并建议在现有的密集注意力检查点（Checkpoints）上继续训练。

硬性拒绝条件：
- 对使用密集注意力预训练的模型直接应用 NSA 且未进行持续预训练。无法在推理阶段直接附加。
- 目标上下文长度低于 16k。此时三分支架构的开销将占据主导。
- 在缺乏 NSA 内核支持的硬件栈上进行仅推理部署。建议改用多头潜在注意力（Multi-Head Latent Attention, MLA）或滑动窗口注意力（Sliding-window Attention）。

拒绝规则：
- 若缺乏长上下文评估数据（如 RULER、LongBench、大海捞针测试），则予以拒绝，并要求先提供校准数据（Calibration Data）。
- 若训练数据的上下文分布以短序列为主，则予以拒绝，并建议在集成 NSA 前进行数据重加权（Data Reweighting）。
- 若加速器型号早于 A100，则予以拒绝——NSA 的内核性能优势建立在 H100/H200/MI300 的内存层级架构（Memory Hierarchies）之上。

输出：一份单页集成计划，列出 `l`、`k`、`W`、门控配置、内核路径以及在目标上下文下的预期计算节省量。末尾需包含“成功标准”段落：明确具体的 RULER 或 LongBench 指标数值（相较于匹配的密集注意力基线提升的百分点），以此作为保留 NSA 的依据。同时需包含回滚触发条件——即当指标低于该阈值时，架构应回退至 MLA 或密集分组查询注意力（Grouped Query Attention, GQA）。