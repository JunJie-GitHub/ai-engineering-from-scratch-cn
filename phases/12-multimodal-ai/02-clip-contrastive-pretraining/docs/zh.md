# CLIP 与对比视觉-语言预训练 (Contrastive Vision-Language Pretraining)

> OpenAI 的 CLIP (2021) 证明了一个足以驱动未来五年的核心思想：仅使用带有噪声的网络图文对 (image-caption pairs) 和对比损失 (contrastive loss)，将图像编码器 (image encoder) 与文本编码器 (text encoder) 对齐至同一向量空间 (vector space)。无需监督标签。4 亿对数据。由此生成的嵌入空间 (embedding space) 可实现零样本分类 (zero-shot classification) 与图文检索 (image-text retrieval)，并作为视觉主干 (vision tower) 接入所有 2026 年的视觉语言模型 (VLM)。SigLIP 2 (2025) 使用 Sigmoid 替换了 Softmax，以更低的成本实现了超越 CLIP 的规模扩展。本节课程将逐步推导从 InfoNCE 到 Sigmoid 成对损失 (sigmoid pairwise loss) 的数学原理，并使用 Python 标准库 (stdlib) 构建训练步骤。

**Type:** 构建实践
**Languages:** Python（标准库，InfoNCE 与 Sigmoid 损失实现）
**Prerequisites:** 第 12 阶段 · 01（ViT 图像块），第 7 阶段（Transformer）
**Time:** 约 180 分钟

## 学习目标

- 从互信息 (mutual information) 推导 InfoNCE 损失，并实现数值稳定的向量化版本。
- 解释为何 Sigmoid 成对损失 (SigLIP) 能够扩展至 32768+ 的批次大小，而无需 Softmax 所要求的全收集 (all-gather) 通信开销。
- 通过构建文本模板（`a photo of a {class}`）并在余弦相似度 (cosine similarity) 上取 argmax，运行零样本 ImageNet 分类。
- 列举 CLIP / SigLIP 预训练提供的四个关键调节杠杆：批次大小 (batch size)、温度系数 (temperature)、提示模板 (prompt template) 与数据质量 (data quality)。

## 问题背景

在 CLIP 出现之前，视觉模型主要依赖监督学习 (supervised learning)。收集标注数据集（如 ImageNet：120 万张图像，1000 个类别），训练卷积神经网络 (CNN)，然后部署。标注成本高昂，且标注结果往往受限于标注人员的主观共识；若不进行微调 (finetuning)，这些标注也无法迁移至新任务。

互联网上存在超过十亿对免费且弱标注的图文对 (loosely-labeled pairs)。一张带有替代文本“公园里我的狗 Max”的金毛寻回犬照片本身就携带了监督信号 (supervisory signal)——文本描述了图像内容。核心问题在于：能否将这些数据转化为有效的训练资源？

CLIP 的解决方案是：将图文对视为匹配任务。给定包含 N 张图像和 N 段文本的批次，模型需学习将每张图像与其对应的文本进行匹配，同时排除其余 N-1 个干扰项 (distractors)。其监督信号本质上是“这两者属于同一组；而那 N-1 个不是”。无需类别标签，无需人工标注，仅依靠对比损失即可。

由此生成的嵌入空间所能完成的任务远超 CLIP 最初的训练目标。ImageNet 零样本分类之所以有效，是因为“一张猫的照片”这一文本的嵌入向量，会与那些从未被明确标注为“猫”的猫咪图片向量在空间中彼此靠近。正是这一技术赌注，催生了 2026 年所有的视觉语言模型。

## 核心概念

### 双编码器（Dual Encoder）

CLIP 包含两个编码器塔（Towers）：

- 图像编码器（Image Encoder）`f`：采用 ViT 或 ResNet，为每张图像输出一个 D 维向量。
- 文本编码器（Text Encoder）`g`：采用小型 Transformer，为每个文本描述（Caption）输出一个 D 维向量。

两个塔均将输出归一化（Normalize）为单位长度。由于两者均为单位范数（Unit-Norm），相似度计算为 `cos(f(x), g(y)) = f(x)^T g(y)`。

对于包含 N 个（图像，文本描述）对的批次（Batch），构建形状为 `(N, N)` 的相似度矩阵（Similarity Matrix）`S`：

S[i, j] = cos(f(x_i), g(y_j)) / tau

其中 `tau` 是一个可学习的温度参数（Temperature）（CLIP 初始化为 0.07，在对数空间中进行学习）。

### InfoNCE 损失（InfoNCE Loss）

CLIP 在行和列上使用对称交叉熵（Symmetric Cross-Entropy）：

loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2

这就是 InfoNCE。交叉熵中的 Softmax 会强制每张图像与其对应文本描述的匹配度高于批次内的其他所有文本描述。这里的“负样本（Negatives）”即批次中的其他所有样本。批次越大 = 负样本越多 = 信号越强。CLIP 的训练批次大小为 32k；规模至关重要。

### 温度参数（Temperature）

`tau` 控制 Softmax 分布的尖锐程度。较低的 `tau` → 分布更尖锐，具有困难负样本挖掘（Hard Negative Mining）的效果。较高的 `tau` → 分布更平缓，所有样本均参与贡献。CLIP 学习 `log(1/tau)`，并进行截断以防止模型崩溃（Collapse）。SigLIP 2 则固定初始 `tau`，转而使用可学习的偏置（Bias）。

### 为什么 Sigmoid 扩展性更好（SigLIP）

Softmax 需要同步整个相似度矩阵。在分布式训练（Distributed Training）中，必须将所有嵌入向量（Embeddings）通过 All-Gather 操作同步到每个副本（Replica），然后再执行 Softmax。这会导致通信开销随世界规模（World Size）呈二次方增长。

SigLIP 使用逐元素 Sigmoid 替换了 Softmax：对于每一对 `(i, j)`，损失函数将其视为一个二分类问题：“它们是否是匹配对？”正样本标签位于对角线上，其余均为负样本。损失函数为：

L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]

当 `i == j` 时 `y_ij = 1`，否则为 0。每一对的损失计算相互独立，无需 All-Gather 操作。每个 GPU 只需计算其本地块（Local Block）并求和即可。SigLIP 2 能够以极低的成本将批次规模扩展至 32k-512k，而 CLIP 若达到同等规模则需要成比例增加通信开销。

### 零样本分类（Zero-Shot Classification）

给定 N 个类别名称，为每个类别构建一个文本模板（Text Template）：

"a photo of a {class}"

使用文本编码器对每个模板进行嵌入（Embed）。使用图像编码器对输入图像进行嵌入。取余弦相似度（Cosine Similarity）的 Argmax 值即为预测类别。无需在目标类别上进行训练。

提示模板（Prompt Templates）的设计至关重要。CLIP 原始论文为每个类别使用了 80 个模板（如普通、艺术、照片、绘画等），并对嵌入向量取平均。这在 ImageNet 上带来了 +3 个点的提升。现代用法通常仅选择一两个模板。

### 线性探测与微调（Linear Probes and Finetuning）

零样本分类仅作为基线（Baseline）。线性探测（Linear Probe，即在冻结的 CLIP 特征之上为目标类别训练一个线性层）在域内任务（In-Domain Tasks）上优于零样本分类。全量微调（Full Finetuning）在域内任务上优于线性探测，但可能会损害零样本迁移（Zero-Shot Transfer）能力。这三种模式各有其权衡（Trade-offs）。

### SigLIP 2：NaFlex 与密集特征（Dense Features）

SigLIP 2（2025）新增了以下特性：
- NaFlex：单一模型即可处理可变的宽高比（Aspect Ratios）和分辨率。
- 更优的密集特征（Dense Features），适用于分割（Segmentation）和深度估计（Depth Estimation），旨在作为视觉语言模型（VLMs）中的冻结骨干网络（Frozen Backbone）。
- 多语言支持：在 100 多种语言上进行训练，而 CLIP 仅支持英语。
- 参数量达到 10 亿（1B）级别，而 CLIP 的上限为 4 亿（400M）。

在 2026 年的开源视觉语言模型中，SigLIP 2 SO400m/14 已成为默认的视觉塔（Vision Tower）。对于纯图文检索任务，若你的查询模式与 LAION-2B 的特定训练分布相匹配，CLIP 仍是默认选择。

### ALIGN、BASIC、OpenCLIP 与 EVA-CLIP

ALIGN（Google，2021）：与 CLIP 理念相同，数据规模达 18 亿对，其中 90% 为噪声数据。证明了噪声数据同样具备扩展性。OpenCLIP（LAION）：在 LAION-400M / 2B 数据集上对 CLIP 的开源复现，提供多种规模，是首选的开源检查点（Checkpoint）。EVA-CLIP：从掩码图像建模（Masked Image Modeling）初始化；是 VLMs 的强大骨干网络。BASIC：Google 将 CLIP 与 ALIGN 结合的混合模型。它们同属一个技术家族，仅在数据和调优策略上有所不同。

### 零样本性能天花板（Zero-Shot Ceiling）

CLIP 类模型在 ImageNet 零样本分类上的性能上限约为 76%（如 CLIP-G、OpenCLIP-G）。若要突破此上限，要么需要规模大得多的数据（SigLIP 2 可达 80% 以上），要么需要架构上的改进（如引入监督头、增加参数量）。该基准测试已趋于饱和；其真正的价值在于为下游 VLMs 提供可消费的嵌入空间（Embedding Space）。

## 使用它
`code/main.py` 实现了：
1. 一个玩具级双编码器（Dual Encoder）（基于哈希的图像特征、文本字符特征），以便在不依赖 `numpy` 的情况下直观观察 InfoNCE 损失（InfoNCE Loss）的形态。
2. 纯 Python 实现的 InfoNCE 损失（通过 log-sum-exp 保证数值稳定性）。
3. 用于对比的 Sigmoid 成对损失（Sigmoid Pairwise Loss）。
4. 零样本分类（Zero-Shot Classification）流程：计算与一组文本提示（Text Prompts）的余弦相似度，并通过 `argmax` 获取预测结果。
运行代码并观察损失曲线。其中的绝对数值仅为示例，但其变化趋势与真实 CLIP 训练器输出的结果一致。

## 部署发布
本课时将生成 `outputs/skill-clip-zero-shot.md` 文件。给定一组图像（通过路径指定）和目标类别列表，它会使用 CLIP 模板构建文本提示，利用指定的模型检查点（Checkpoint）（例如 `openai/clip-vit-large-patch14`）对图像和文本两侧进行嵌入（Embedding），并返回带有相似度分数的 Top-1 / Top-5 预测结果。该功能模块仅对提示列表中的类别进行预测，拒绝识别列表之外的类别。

## 练习
1. 手动实现包含 4 个样本对的 InfoNCE 计算。构建 4x4 相似度矩阵，执行 Softmax 操作，提取对角线元素，并计算交叉熵（Cross-Entropy）。将你的 Python 实现结果与手动计算结果进行对比验证。
2. SigLIP 除了温度参数（Temperature）外，还引入了偏置参数（Bias Parameter）`b`：`S'[i,j] = S[i,j]/tau + b`。当批次数据存在严重的类别不平衡（Class Imbalance）（即每行中负样本远多于正样本）时，`b` 起到了什么作用？请阅读 SigLIP 论文第 3 节（arXiv:2303.15343）。
3. 构建一个用于区分猫和狗的零样本分类器。尝试使用两种提示模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上评估准确率。模板集成（Ensemble）的效果是否优于单一模板？
4. 计算在 512 张 GPU、批次大小为 32k 的运行环境下，Softmax InfoNCE 与 Sigmoid 成对损失的通信开销（Communication Cost）。哪种方法的复杂度为 O(N)，哪种为 O(N^2)？请引用 SigLIP 论文第 4 节。
5. 阅读 OpenCLIP 的缩放定律（Scaling Laws）论文（arXiv:2212.07143, Cherti 等人）。根据论文中的图表复现其关于数据缩放的结论：在模型规模固定的情况下，ImageNet 零样本准确率与训练数据规模之间呈现怎样的对数线性（Log-Linear）关系？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 信息噪声对比估计 (InfoNCE) | “对比损失” (Contrastive loss) | 基于批次相似度矩阵的交叉熵；每个样本的正样本是其配对样本，负样本为批次内其余所有样本 |
| Sigmoid 损失 (Sigmoid loss) | “SigLIP 损失” (SigLIP loss) | 逐对二元交叉熵；无需 softmax 和 all-gather 操作，在分布式训练中扩展成本低廉 |
| 温度参数 (Temperature) | “tau” (tau) | 在 softmax/sigmoid 前用于缩放 logits 的标量；控制概率分布的尖锐程度 |
| 零样本 (Zero-shot) | “免微调分类” (no-finetune classification) | 利用文本提示构建类别嵌入 (embeddings)，并通过余弦相似度 (cosine similarity) 进行分类；无需在目标类别上进行训练 |
| 提示模板 (Prompt template) | “一张……的照片” (a photo of a ...) | 围绕类别名称的文本框架；会使零样本准确率产生 1~5 个百分点的波动 |
| 双编码器 (Dual encoder) | “双塔” (Two-tower) | 包含一个图像编码器和一个文本编码器，将输出映射至共享的 D 维空间 |
| 困难负样本 (Hard negative) | “强力干扰项” (Tough distractor) | 与正样本高度相似的负样本，迫使模型付出更多努力才能将其区分开 |
| 线性探针 (Linear probe) | “冻结特征 + 单层分类器” (Frozen + one layer) | 仅在冻结的特征之上训练线性分类器；用于衡量特征质量 |
| 原生灵活分辨率 (NaFlex) | “原生灵活分辨率” (Native flexible resolution) | SigLIP 2 具备的能力，可直接输入任意宽高比和分辨率的图像，无需进行缩放调整 |
| 温度缩放 (Temperature scaling) | “对数参数化 tau” (log-parametrized tau) | CLIP 对 `log(1/tau)` 进行参数化以保证梯度行为稳定；并进行截断以防止 tau 坍缩至接近零 |

## 延伸阅读

- [Radford 等人 — 从自然语言监督中学习可迁移的视觉模型 (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 的原始论文。
- [Zhai 等人 — 用于语言图像预训练的 Sigmoid 损失 (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP 论文。
- [Tschannen 等人 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 支持多语言与 NaFlex。
- [Jia 等人 — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 利用含噪网络数据进行规模化训练。
- [Cherti 等人 — 对比语言图像学习的可复现缩放定律 (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP 的缩放定律。