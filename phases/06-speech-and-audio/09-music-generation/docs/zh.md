# 音乐生成（Music Generation）—— MusicGen、Stable Audio、Suno 与版权地震

> 2026 年音乐生成领域：Suno v5 和 Udio v4 主导商业市场；MusicGen、Stable Audio Open 和 ACE-Step 引领开源生态。技术难题已基本攻克。而法律纠纷（华纳音乐 5 亿美元和解案、环球音乐集团和解案）在 2025-2026 年间重塑了整个行业格局。

**类型：** 项目构建
**语言：** Python
**前置知识：** 第 6 阶段 · 02（声谱图 Spectrograms）、第 4 阶段 · 10（扩散模型 Diffusion Models）
**耗时：** 约 75 分钟

## 核心问题

文本 → 生成一段 30 秒至 4 分钟的音乐片段，需包含歌词、人声与曲式结构。可拆分为三个子问题：

1. **纯音乐生成（Instrumental generation）。** 输入如“带有温暖键盘音色的低保真嘻哈鼓点”等文本 → 输出音频。代表模型：MusicGen、Stable Audio、AudioLDM。
2. **完整歌曲生成（Song generation）。** 输入如“关于德克萨斯雨夜的乡村歌曲” → 输出完整歌曲。代表模型：Suno、Udio、YuE、ACE-Step。
3. **条件控制与可编辑性（Conditional / controllable）。** 扩展现有片段、重新生成桥段、切换曲风、音轨分离（stem separation）或局部重绘（inpaint）。Udio 的局部重绘与音轨分离功能是 2026 年业界竞相追赶的标杆特性。

## 核心概念

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### 基于神经编解码器 Token 的语言模型（Token LM over neural-codec tokens）

Meta 的 **MusicGen**（2023 年，MIT 协议）及其众多衍生模型：以文本或旋律嵌入（embeddings）为条件，自回归（autoregressively）预测 EnCodec Token（32 kHz，4 个码本），最后通过 EnCodec 解码。参数量在 3 亿至 33 亿之间。作为强基线模型，其在生成超过 30 秒的音频时表现欠佳。

**ACE-Step**（开源，2026 年 4 月发布 4B XL 版本）在此基础上扩展，实现了以歌词为条件的完整歌曲生成。它是开源社区中最接近 Suno 的模型。

### 基于梅尔频谱或潜空间的扩散模型（Diffusion over mels or latents）

**Stable Audio（2023）** 与 **Stable Audio Open（2024）**：在压缩音频上进行潜空间扩散（latent diffusion）。擅长生成循环乐段、音效设计与氛围纹理，但在生成结构完整的歌曲方面表现一般。

**AudioLDM / AudioLDM2**：采用类似文本到图像（T2I）的潜空间扩散架构实现文本到音频的生成，已泛化至音乐、音效与语音领域。

### 混合架构（Hybrid，工业级应用）—— Suno、Udio、Lyria

闭源权重。推测采用自回归编解码语言模型（AR codec LM）结合基于扩散的声码器（diffusion-based vocoder），并配备专用的人声/鼓点/旋律输出头（heads）。Suno v5（2026）以 ELO 1293 的评分领跑质量榜。Udio v4 新增了局部重绘与音轨分离功能（贝斯、鼓点、人声可分别下载）。

### 评估指标

- **弗雷歇音频距离（Fréchet Audio Distance, FAD）。** 利用 VGGish 或 PANNs 特征，计算生成音频与真实音频分布在嵌入层（embedding-level）的距离。数值越低越好。MusicGen small 在 MusicCaps 数据集上的 FAD 为 4.5；当前最先进水平（State of the Art, SOTA）约为 3.0。
- **音乐性（Musicality，主观评估）。** 基于人类偏好测试。Suno v5 以 ELO 1293 的评分领先。
- **文本-音频对齐度（Text-audio alignment）。** 提示词（prompt）与输出音频之间的 CLAP 评分。
- **音乐性瑕疵（Musicality artifacts）。** 节拍错位过渡、人声乐句漂移、超过 30 秒后曲式结构丢失等问题。

## 2026 年模型图谱

| 模型 | 参数量 | 时长 | 人声 | 许可证 |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 秒 | 无 | MIT |
| Stable Audio Open | 1.2B | 47 秒 | 无 | Stability 非商业许可 |
| ACE-Step XL (2026年4月) | 4B | &gt; 2 分钟 | 支持 | Apache-2.0 |
| YuE | 7B | &gt; 2 分钟 | 支持，多语言 | Apache-2.0 |
| Suno v5 (闭源) | ? | 4 分钟 | 支持，ELO 评分 1293 | 商业 |
| Udio v4 (闭源) | ? | 4 分钟 | 支持 + 分轨（stems） | 商业 |
| Google Lyria 3 (闭源) | ? | 实时 | 支持 | 商业 |
| MiniMax Music 2.5 | ? | 4 分钟 | 支持 | 商业 API |

## 法律环境（2025-2026）

- **华纳音乐与 Suno 达成和解。** 赔偿金额 5 亿美元。华纳音乐集团（WMG）现对 Suno 平台上的 AI 相似性（AI-likeness）、音乐版权及用户生成曲目拥有监督权。环球音乐集团（UMG）也与 Udio 达成了类似和解。
- **《欧盟人工智能法案》（EU AI Act）** + **《加州 SB 942 法案》**：必须披露 AI 生成的音乐。
- 基于 MIT 许可证的 **Riffusion / MusicGen** 没有合规负担，但也不提供商用级人声。

可安全交付的模式：

1. 仅生成纯器乐（使用 MusicGen、Stable Audio Open 或 MIT/CC0 许可证的输出）。
2. 使用商业 API（Suno、Udio、ElevenLabs Music），并按次生成获取许可。
3. 使用自有或已获授权的曲库进行训练（大多数企业最终会选择此路径）。
4. 为生成内容添加水印与元数据标签。

## 构建指南

### 步骤 1：使用 MusicGen 生成

from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)

提供三种尺寸：`small`（3 亿参数，速度快）、`medium`（15 亿参数）和 `large`（33 亿参数）。`small` 版本足以用于“验证核心构思能否成立”。

### 步骤 2：旋律条件控制（melody conditioning）

melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)

`MusicGen-melody` 接收音高色度图（chromagram），在保留原曲调的同时替换音色。适用于“将这段旋律改编为弦乐四重奏”等需求。

### 步骤 3：FAD 评估（Fréchet Audio Distance）

from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")

计算 VGGish 嵌入（VGGish-embedding）距离。适用于流派级别的回归测试（regression tests），但不能替代人工听审。

### 步骤 4：集成至大语言模型-音乐工作流（LLM-music workflow）

结合第 7-8 课的思路：

prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)

## 实际应用

| 目标 | 技术栈 |
|------|-------|
| 器乐音效设计 | Stable Audio Open |
| 游戏/自适应音乐 | Google Lyria RealTime（闭源） |
| 带人声的完整歌曲（商用） | Suno v5 或 Udio v4（需明确授权） |
| 带人声的完整歌曲（开源） | ACE-Step XL 或 YuE |
| 短广告配乐 | MusicGen（以哼唱参考为旋律条件） |
| 音乐视频背景 | MusicGen + Stable Video Diffusion |

## 2026 年部署时仍需注意的陷阱

- **版权规避提示词（Copyright-laundering prompts）。** “模仿泰勒·斯威夫特风格的歌曲”——目前商用版 Suno/Udio 已对此类提示词进行过滤，但开源模型尚未跟进。建议自行添加过滤词表。
- **30 秒后的重复/漂移。** 自回归模型（AR models）容易出现循环。可对多次生成的结果进行交叉淡入淡出（crossfade）处理，或使用 ACE-Step 以保持结构连贯性。
- **速度漂移。** 模型生成的节奏容易偏离设定的节拍（BPM）。可在提示词中加入 BPM 标签，并使用 librosa 的 `beat_track` 进行后处理过滤。
- **人声清晰度。** Suno 表现优异；开源模型在咬字上往往含糊不清。若歌词内容至关重要，建议使用商用 API 或进行微调（fine-tune）。
- **单声道输出。** 开源模型通常生成单声道（mono）或伪立体声（fake-stereo）。可通过专业的立体声重建（stereo reconstruction）工具进行升级（如 ezst、Cartesia 的立体声扩散模型）。

## 部署上线

保存为 `outputs/skill-music-designer.md`。为音乐生成（music-gen）部署任务选择模型、授权策略、时长/结构规划，并填写披露元数据。

## 练习

1. **简单。** 运行 `code/main.py`。它会以 ASCII 符号输出“生成式”和弦进行 + 鼓点模式——相当于音乐生成的简易示意图。如需试听，可通过任意 MIDI 渲染器播放。
2. **中等。** 安装 `audiocraft`，使用 MusicGen-small 针对 4 种不同风格提示词生成 10 秒音频片段，并对照参考风格集计算音频 Fréchet 距离（FAD）。
3. **困难。** 使用 ACE-Step（或 MusicGen-melody），通过不同的音色提示词生成同一旋律的三个变体。计算其与提示词的对比式语言-音频预训练（CLAP）相似度，以验证对齐效果。

## 核心术语

| 术语 | 常见叫法 | 实际含义 |
|------|-----------------|-----------------------|
| FAD | 音频 FID | 真实音频与生成音频的嵌入分布之间的 Fréchet 距离。 |
| Chromagram | 旋律音高表示 | 每帧 12 维向量；用于旋律条件控制（melody conditioning）的输入。 |
| Stems | 乐器分轨 | 分离出的贝斯/鼓/人声/旋律等独立 WAV 音轨。 |
| Inpainting | 局部重生成 | 对特定时间窗口添加掩码（mask），模型仅针对该区域重新生成。 |
| CLAP | 文本-音频版 CLIP | 对比式音频-文本嵌入模型；用于评估文本提示与生成音频的对齐度。 |
| EnCodec | 音乐编解码器 | Meta 研发的神经编解码器，为 MusicGen 底层组件；采样率 32 kHz，包含 4 个码本（codebooks）。 |

## 延伸阅读

- [Copet 等人 (2023). MusicGen](https://arxiv.org/abs/2306.05284) —— 开源自回归（Autoregressive）基准（Benchmark）模型。
- [Evans 等人 (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) —— 声音设计（Sound Design）领域的默认方案。
- [ACE-Step](https://github.com/ace-step/ACE-Step) —— 开源 4B 参数全曲生成模型，发布于 2026 年 4 月。
- [Suno v5 平台文档](https://suno.com) —— 商业音质领域的领先者。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) —— 用于音乐与音效生成的潜在扩散（Latent Diffusion）模型。
- [WMG 与 Suno 和解案报道](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) —— 2025 年 11 月的法律先例。