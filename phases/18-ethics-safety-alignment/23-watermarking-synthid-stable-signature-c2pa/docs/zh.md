# 水印技术（Watermarking）—— SynthID、Stable Signature 与 C2PA

> 三项技术构成了 2026 年 AI 生成内容溯源（provenance）的框架。SynthID（Google DeepMind）—— 图像水印于 2023 年 8 月推出，文本与视频水印于 2024 年 5 月推出（Gemini + Veo），文本水印于 2024 年 10 月通过 Responsible GenAI Toolkit 开源，统一多媒体检测器于 2025 年 11 月随 Gemini 3 Pro 一同发布。文本水印通过微调下一个词元（token）的采样概率来实现，且对感知不可察觉；图像/视频水印能够抵御压缩、裁剪、滤镜和帧率变化。Stable Signature（Fernandez 等人，ICCV 2023，arXiv:2303.15435）—— 对潜在扩散解码器（latent diffusion decoder）进行微调，使每次生成的输出都包含固定信息；即使生成图像被裁剪（仅保留 10% 内容），在假阳性率（False Positive Rate, FPR）低于 1e-6 的条件下，检测率仍超过 90%。后续研究《Stable Signature is Unstable》（arXiv:2405.07145，2024 年 5 月）指出，通过微调可在保持生成质量的同时移除该水印。C2PA —— 一种采用加密签名且具备防篡改特性的元数据（metadata）标准（C2PA 2.2 Explainer 2025）。水印技术与 C2PA 互为补充：元数据可能被剥离，但能承载更丰富的溯源信息；水印在转码过程中依然留存，但携带的信息量较少。

**类型：** 构建实践
**编程语言：** Python（标准库，token-watermark 嵌入与检测）
**前置知识：** 第 10 阶段 · 04（采样），第 01 阶段 · 09（信息论）
**预计耗时：** 约 75 分钟

## 学习目标

- 描述词元级（token-level）水印技术（SynthID-text 风格）及其可被检测的机制。
- 阐述 Stable Signature 技术以及 2024 年攻破该技术的移除攻击（removal attack）。
- 说明 C2PA 的作用及其与水印技术互补的原因。
- 描述其主要局限性：模型特异性信号、在文本改写（paraphrase）下的鲁棒性（robustness）问题，以及语义保持攻击（meaning-preserving attacks）（arXiv:2508.20228）。

## 问题背景

2023 至 2024 年间，深度伪造（deepfakes）与 AI 生成内容大规模进入政治与消费领域。水印技术被提出作为技术溯源信号：在内容生成时进行标记，以便后续检测。2025 年的实践表明：没有任何水印具备无条件鲁棒性，但若与 C2PA 元数据结合使用，则能提供一套切实可行的溯源方案。

## 核心概念

### 文本水印（SynthID-text 风格）

Kirchenbauer 等人（2023）提出的机制，已由 Google 实现产品化：

1. 在每一步解码（decoding）过程中，对前 K 个词元（token）进行哈希（hash）处理，将词表（vocabulary）伪随机划分为“绿色”和“红色”集合。
2. 通过向绿色词元的对数几率（logits）添加 δ，使采样（sampling）偏向绿色集合。
3. 生成的文本中包含的绿色词元数量将高于随机概率下的预期值。

检测（detection）：对每个前缀（prefix）重新进行哈希运算，统计生成内容中的绿色词元数量，并计算 Z 分数（z-score）。带水印文本的 Z 分数大于 0，而人类文本的 Z 分数接近 0。

特性：
- 对读者不可察觉（δ 值足够小，质量损失微乎其微）。
- 在获取词表划分函数（vocabulary partition function）的情况下可被检测。
- 对改写（paraphrase）不具备鲁棒性（robustness）——重写文本会破坏水印信号。

SynthID-text 已于 2024 年 10 月通过 Google 的负责任生成式 AI 工具包（Responsible GenAI Toolkit）开源。

### 稳定签名（Stable Signature，图像）

Fernandez 等人（ICCV 2023）。通过微调（fine-tune）潜在扩散解码器（latent diffusion decoder），使每张生成的图像都在潜在表示（latent representation）中嵌入固定的二进制消息。检测时通过神经解码器（neural decoder）从潜在空间中解码该消息。即使图像被裁剪至原内容的 10%，在假阳性率（FPR）低于 1e-6 的条件下，检测率仍超过 90%。

2024 年 5 月论文《Stable Signature is Unstable》（arXiv:2405.07145）指出：对解码器进行微调即可在保持图像质量的同时移除水印。生成后对抗性微调（adversarial post-generation fine-tuning）成本极低，表明该水印的对抗鲁棒性（adversarial robustness）有限。

### SynthID 统一检测器（2025 年 11 月）

随 Gemini 3 Pro 一同发布：一款多媒体检测器，可通过单一 API 读取文本、图像、音频和视频中的 SynthID 信号。统一了 Google 的溯源技术栈（provenance stack）。

### C2PA

内容溯源与真实性联盟（Coalition for Content Provenance and Authenticity）。一种采用密码学签名（cryptographically signed）的防篡改元数据（tamper-evident metadata）标准。C2PA 2.2 说明文档（2025）。C2PA 清单（manifest）记录了溯源声明（provenance claims，包括创建者、创建时间及经历的变换），并由创建者的密钥进行签名。

与水印技术互补：
- 元数据可被剥离，而水印（难以）轻易移除。
- 元数据信息丰富（包含完整的溯源链），而水印仅承载少量比特信息。
- C2PA 依赖平台方的采纳，而水印可自动嵌入。

Google 已在搜索、广告及“关于此图片”功能中同时集成这两项技术。

### 局限性

- **模型特定性（Model-specific）**。SynthID 仅对启用了 SynthID 的模型生成内容进行水印标记。未启用 SynthID 的模型生成内容不会带水印，因此“未检测到 SynthID 信号”并不能作为内容真实性的证明。
- **改写（Paraphrase）**。文本水印无法在保持语义的改写中存活。
- **变换攻击（Transformation attacks）**。arXiv:2508.20228（2025）表明，保持语义的攻击手段可同时破坏文本水印及多数图像水印。
- **微调移除（Fine-tune removal）**。如《Stable Signature is Unstable》所述，生成后的微调操作可移除已嵌入的水印。

### 《欧盟人工智能法案》第 50 条

关于 AI 生成内容标注的透明度准则（Transparency Code）（初稿于 2025 年 12 月发布，二稿于 2026 年 3 月发布，根据[欧盟委员会状态页面](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)预计最终版将于 2026 年 6 月定稿）。截至 2026 年 4 月，该准则仍处于草案阶段，时间表可能调整。该法规层要求底层技术予以配合。深度伪造（Deepfakes）内容必须进行标注。

### 在第 18 阶段中的定位

第 22-23 课探讨模型输出的内容（私有数据、溯源信号）。第 27 课涵盖训练数据治理（training-data governance）。第 24 课则是要求实施这些技术措施的监管框架（regulatory framework）。

## 使用它

`code/main.py` 构建了一个简易的文本水印（text watermark）示例。词元（Token）为 0 到 N-1 的整数；带水印的采样过程会偏向由哈希定义的“绿色”词元集合（green set）。检测器会计算绿色词元的 Z 分数（Z-score）。你可以观察 1000 词元生成文本的检测效果，查看文本改写（paraphrase）如何破坏水印信号，并测量该水印在人类撰写文本上的误报率（false-positive rate）。

## 交付它

本课时将生成 `outputs/skill-provenance-audit.md`。针对带有溯源声明（provenance claim）的内容部署，该脚本会审计以下内容：水印机制（watermark mechanism，如有）、C2PA 签名链（C2PA signing chain，如有）、各项机制的对抗鲁棒性（adversarial robustness），以及各模态的覆盖范围（per-modality coverage）。

## 练习

1. 运行 `code/main.py`。报告带水印的 1000 词元生成文本与人类撰写文本的 Z 分数（Z-score）。确定在 95% 置信度阈值下的误报率（false-positive rate）。

2. 实现一种改写攻击（paraphrase attack），将 30% 的词元替换为同义词。重新测量 Z 分数。

3. 阅读 Kirchenbauer 等人 2023 年论文第 6 节关于鲁棒性的内容。为什么文本水印在改写攻击下会失效，而图像水印却能抵抗裁剪（cropping）？

4. 设计一个使用 SynthID-text + C2PA 元数据的部署方案。描述终端消费者看到的溯源链（provenance chain）。指出每个组件的一种故障模式（failure mode）。

5. 2024 年《Stable Signature is Unstable》的研究结果表明，微调（fine-tuning）会移除图像水印。设计一种部署控制措施来限制此类攻击——例如，要求对微调后的模型检查点（checkpoints）进行签名发布。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| SynthID | “Google 的水印” | 跨模态溯源信号（cross-modal provenance signal）；涵盖文本、图像、音频、视频 |
| Token watermark（词元水印） | “Kirchenbauer 风格” | 基于偏置采样的文本水印，可通过绿色词元 Z 分数（green-token z-score）检测 |
| Stable Signature | “图像水印” | 基于微调解码器的水印（fine-tuned-decoder watermark）；ICCV 2023 |
| C2PA | “元数据标准” | 经密码学签名、具备防篡改特性的溯源元数据（tamper-evident provenance metadata） |
| Paraphrase robustness（改写鲁棒性） | “改写会破坏它吗” | 文本水印特性；目前能力有限 |
| Fine-tune removal（微调移除） | “对抗性去水印” | 通过微调解码器移除图像水印的攻击 |
| Cross-modal detector（跨模态检测器） | “统一版 SynthID” | 2025 年 11 月推出的跨模态统一 API |

## 延伸阅读

- [Kirchenbauer 等人 —— 大语言模型水印（ICML 2023, arXiv:2301.10226）](https://arxiv.org/abs/2301.10226) —— 词元水印机制
- [Fernandez 等人 —— Stable Signature（ICCV 2023, arXiv:2303.15435）](https://arxiv.org/abs/2303.15435) —— 图像水印论文
- [《Stable Signature is Unstable》（arXiv:2405.07145）](https://arxiv.org/abs/2405.07145) —— 水印移除攻击
- [Google DeepMind —— SynthID](https://deepmind.google/models/synthid/) —— 跨模态水印
- [C2PA 2.2 说明文档（2025）](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) —— 元数据标准