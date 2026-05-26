# 评估 — FID、CLIP 分数与人类偏好

> 每个生成模型（generative model）排行榜都会引用 FID、CLIP 分数以及来自人类偏好竞技场（human-preference arena）的胜率。每个数值都存在特定的失效模式（failure mode），有决心的研究人员完全可以针对这些模式进行刷分（game）。如果你不了解这些失效模式，就无法区分真正的性能提升与指标博弈（gaming run）。

**类型：** 构建
**语言：** Python
**前置条件：** 第 8 阶段 · 01（分类体系），第 2 阶段 · 04（评估指标）
**耗时：** 约 45 分钟

## 核心问题

生成模型的评估主要基于*样本质量（sample quality）*与*条件遵循度（conditioning adherence）*。这两者均缺乏精确的闭式度量标准（closed-form measure）。你的模型需要生成 10,000 张图像；必须有某种机制为它们分配数值；你必须能够信任这些分数在不同模型家族、不同分辨率和不同架构之间的可比性。历经 2014 至 2026 年的重重考验，仅有三项指标存活下来：

- **FID（Fréchet Inception Distance）。** 衡量真实数据与生成数据在 Inception 网络特征空间中的分布距离。数值越低越好。
- **CLIP 分数（CLIP score）。** 计算生成图像的 CLIP 图像嵌入（CLIP-image embedding）与提示词的 CLIP 文本嵌入（CLIP-text embedding）之间的余弦相似度（cosine similarity）。数值越高越好。用于衡量提示词遵循度（prompt adherence）。
- **人类偏好（Human preference）。** 让两个模型在相同提示词下进行对决，由人类（或 GPT-4 级别的模型）选出更优者，最终汇总为 Elo 评分（Elo score）。

你还会看到以下指标：IS（Inception Score，已基本淘汰）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每一项指标都是为了修正前一项指标的某种缺陷而提出的。

## 核心概念

![FID、CLIP 与偏好：三个维度，不同的失效模式](../assets/evaluation.svg)

### FID（Fréchet Inception Distance）—— 样本质量（sample quality）
Heusel 等人（2017）。步骤：

1. 提取 N 张真实图像和 N 张生成图像的 Inception-v3 特征（2048 维）。
2. 为每个特征池拟合高斯分布（Gaussian）：计算均值 `μ_r, μ_g` 和协方差（covariance）`Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解读：特征空间（feature space）中两个多元高斯分布（multivariate Gaussians）之间的 Fréchet 距离（Fréchet distance）。数值越低表示分布越相似。

失效模式（Failure modes）：
- **小样本偏差（Biased on small N）。** FID 的计算依赖于特征分布的统计量——样本量 N 过小会低估协方差，导致 FID 虚假偏低。务必使用 N ≥ 10,000。
- **依赖 Inception 模型（Inception-dependent）。** Inception-v3 是在 ImageNet 上训练的。与 ImageNet 差异较大的领域（如人脸、艺术画作、文本图像）会产生无意义的 FID 值。请使用特定领域的特征提取器。
- **指标刷分（Gaming）。** 针对 Inception 先验进行过拟合可以在不提升视觉质量的情况下获得较低的 FID。可通过下文提到的 CMMD 来应对。

### CLIP 分数（CLIP score）—— 提示词遵循度（prompt adherence）
Radford 等人（2021）。针对生成的图像与提示词：

clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )

对 3 万张生成图像取平均值 → 得到一个可在不同模型间比较的标量。

失效模式：
- **CLIP 自身的盲区。** CLIP 的组合推理能力（compositional reasoning）较弱（例如“蓝色球体上的红色立方体”经常判断失败）。模型可能在 CLIP 分数上排名靠前，但实际上并未真正遵循复杂的提示词。
- **短提示词偏差（Short prompt bias）。** 在开放数据集中，短提示词更容易与 CLIP 图像特征匹配。从机制上讲，长提示词的 CLIP 分数天然偏低。
- **提示词刷分（Prompt gaming）。** 在提示词中加入“high quality, 4k, masterpiece”等词汇会人为抬高 CLIP 分数，但并不会改善图文对齐（image-text binding）效果。

CMMD（Jayasumana 等人，2024）修复了部分问题：它使用 CLIP 特征替代 Inception，并使用最大均值差异（Maximum Mean Discrepancy）替代 Fréchet 距离。在检测细微质量差异方面表现更佳。

### 人类偏好（Human preference）—— 真实基准（ground truth）
选取一组提示词池。分别使用模型 A 和模型 B 生成图像。将成对结果展示给人类（或强大的大语言模型裁判（LLM judge））。将胜负结果汇总为 Elo 或 Bradley-Terry 评分。常用基准：

- **PartiPrompts（Google）**：1,600 个多样化提示词，涵盖 12 个类别。
- **HPSv2**：10.7 万条人类标注数据，广泛用作自动化代理指标。
- **ImageReward**：13.7 万对提示词-图像偏好数据，采用 MIT 许可证。
- **PickScore**：基于 Pick-a-Pic 的 260 万条偏好数据训练而成。
- **类 Chatbot Arena 的图像竞技场**：https://imagearena.ai/ 及其他平台。

失效模式：
- **裁判方差（Judge variance）。** 非专家与专家的偏好存在差异。建议两者结合使用。
- **提示词分布偏差（Prompt distribution）。** 精心挑选的提示词会偏向某一类模型。务必做好文档记录。
- **大语言模型裁判的奖励机制漏洞利用（LLM-judge reward hacking）。** GPT-4 等裁判容易被“好看但错误”的输出欺骗。需与人类评估进行交叉验证。

## 综合使用

生产环境评估报告应包含：

1. 在 1 万至 3 万个样本上计算 FID（Fréchet Inception Distance），并与预留的真实数据分布进行对比（样本质量）。
2. 在相同样本及其对应提示词上计算 CLIP 分数（CLIP Score）/ CMMD（Cross-Maximum Mean Discrepancy）（提示词遵循度）。
3. 在盲测竞技场中与上一代模型对比的胜率（整体偏好）。
4. 失败模式分析：随机抽取 50 个输出结果，并针对已知问题（如手部结构、文字渲染、物体数量一致性）进行标记。

单一指标往往具有欺骗性。只有三个相互印证的指标加上定性审查，才能构成可靠的结论。

## 构建实现

`code/main.py` 在合成的“特征向量”上实现了 FID、类 CLIP 分数以及 Elo 评分（Elo Rating）聚合计算（我们使用 4 维向量作为 Inception 特征的替代）。你将看到：

- 在小样本量（N）与大样本量下计算 FID 的差异——即偏差（Bias）。
- 将“CLIP 分数”表示为特征池之间的余弦相似度（Cosine Similarity）。
- 基于合成偏好数据流的 Elo 更新规则。

### 步骤 1：四行代码实现 FID

def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term

### 步骤 2：类 CLIP 的余弦相似度

def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)

### 步骤 3：Elo 聚合计算

def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new

## 常见陷阱

- **N=1000 时的 FID。** 当样本量低于 1 万时，该启发式指标并不可靠。报告低样本量 FID 的论文往往存在指标操纵嫌疑。
- **跨分辨率比较 FID。** Inception 模型会将图像统一缩放至 299×299，这会改变特征分布。仅在分辨率匹配的情况下进行比较。
- **仅报告单一随机种子。** 至少运行 3 个不同的随机种子，并报告标准差（Standard Deviation）。
- **通过负面提示词虚增 CLIP 分数。** 部分流水线会通过过度拟合提示词来人为拉高 CLIP 分数。需检查图像是否出现视觉饱和现象。
- **提示词重叠导致的 Elo 偏差。** 如果两个模型在训练阶段都见过基准测试中的提示词，Elo 评分将失去意义。请使用预留的提示词集合。
- **付费众包人工评估的样本偏差。** Prolific 和 MTurk 的标注者群体通常偏年轻且更熟悉科技产品。建议混合招募艺术/设计领域的专家参与评估。

## 实际应用

2026年生产环境评估协议：

| 评估维度 | 最低要求 | 推荐标准 |
|--------|---------|-------------|
| 样本质量 | 1万样本与保留真实数据的 FID (Fréchet Inception Distance) | + 5千样本的 CMMD (CLIP-feature Maximum Mean Discrepancy) + 按类别划分的子集 FID |
| 提示词遵循度 | 3万样本的 CLIP 得分 (CLIP Score) | + HPSv2 + ImageReward + 视觉问答 (VQA) 式问答 |
| 偏好评估 | 与基线对比的 200 对盲测样本 | + 2000 对人工配对评估 + 大语言模型裁判 (LLM-judge) + Chatbot Arena |
| 失败分析 | 50 个手动标记样本 | 500 个手动标记样本 + 自动化安全分类器 |

一份报告涵盖全部四个维度 = 严谨声明。仅凭单一维度 = 营销话术。

## 发布上线

保存 `outputs/skill-eval-report.md`。Skill 工具接收新模型检查点 (checkpoint) 与基线模型，并输出完整的评估计划：样本量、评估指标、失败模式探测 (failure-mode probes) 以及验收标准 (sign-off criteria)。

## 练习

1. **简单。** 运行 `code/main.py`。在相同的合成数据分布上，对比 N=100 与 N=1000 时的 FID 值。报告偏差幅度。
2. **中等。** 基于合成的类 CLIP 特征实现 CMMD（公式参见 Jayasumana et al., 2024）。对比其与 FID 在质量差异敏感度上的表现。
3. **困难。** 复现 HPSv2 设置：从 Pick-a-Pic 数据集中抽取 1000 个图像-提示词对，基于偏好数据微调一个小型的基于 CLIP 的评分模型，并测量其与保留测试集 (held-out set) 的一致性。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| FID | “Fréchet Inception Distance” | 对真实与生成 Inception 特征进行高斯拟合后的 Fréchet 距离。 |
| CLIP 得分 | “图文相似度” | CLIP 图像嵌入与文本嵌入之间的余弦相似度。 |
| CMMD | “FID 的替代品” | 基于 CLIP 特征的最大均值差异 (Maximum Mean Discrepancy)；偏差更小，无需高斯分布假设。 |
| IS | “Inception Score” | Exp KL(p(y|x) || p(y))；在现代模型上相关性较差，已淘汰。 |
| HPSv2 / ImageReward / PickScore | “学习到的偏好代理” | 基于人类偏好数据训练的小型模型；用作自动化裁判。 |
| Elo | “国际象棋等级分” | 基于 Bradley-Terry 模型对两两胜负结果进行聚合的评分。 |
| PartiPrompts | “基准提示词集” | Google 策划的涵盖 12 个类别的 1,600 条提示词。 |
| FD-DINO | “自监督替代方案” | 使用 DINOv2 特征计算的 Fréchet 距离 (FD)；在 ImageNet 之外的领域表现更佳。 |

## 生产环境备注：评估同样属于推理负载 (inference workload)

在 1 万个样本上运行弗雷歇起始距离 (FID) 意味着需要生成 1 万张图像。对于在单张 L4 GPU 上以 1024² 分辨率运行 50 步的 SDXL 基础模型而言，这相当于约 11 小时的单请求推理时间。评估预算是切实存在的，而该场景完全契合离线推理 (offline inference) 的设定（即最大化吞吐量，忽略首图延迟 (TTFT)）：

- **尽可能使用大批量，忽略延迟。** 离线评估 = 采用显存允许的最大尺寸进行静态批处理 (static batching)。在 80GB H100 上，使用 `num_images_per_prompt=8` 调用 `pipe(...).images` 的实际耗时 (wall-clock time) 比单请求快 4-6 倍。
- **缓存真实特征。** 针对真实参考集进行的 Inception (FID) 或 CLIP (CLIP-score, CMMD) 特征提取只需运行*一次*，并保存为 `.npz` 文件。切勿在每次评估时重复计算。

针对持续集成 / 回归门禁 (CI / regression gates)：每个拉取请求 (PR) 在 500 个样本的子集上运行 FID + CLIP 分数评估（约 30 分钟）；每晚运行完整的 1 万样本 FID + HPSv2 + Elo 评估。

## 延伸阅读

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 原始论文。
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD 指标。
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP 模型。
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2 评分。
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward 模型。
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts 数据集。
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 评估指标失效模式综述。