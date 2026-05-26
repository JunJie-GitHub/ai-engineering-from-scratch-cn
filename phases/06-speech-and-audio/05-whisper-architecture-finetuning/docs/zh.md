# Whisper — 架构与微调

> Whisper 是一个采用 30 秒窗口的 Transformer 编码器-解码器 (Transformer Encoder-Decoder) 模型，基于 68 万小时的多语言弱监督 (Weakly-Supervised) 音频-文本对训练而成。一套架构，多项任务，在 99 种语言中均具备出色的鲁棒性。它是 2026 年自动语音识别 (Automatic Speech Recognition, ASR) 的参考基准。

**类型：** 构建
**语言：** Python
**前置要求：** 第 6 阶段 · 04（ASR）、第 5 阶段 · 10（注意力机制 (Attention)）、第 7 阶段 · 05（完整 Transformer (Full Transformer)）
**时长：** 约 75 分钟

## 核心问题

Whisper 由 OpenAI 于 2022 年 9 月发布，是首款以“标准化产品”形式交付的自动语音识别 (ASR) 模型：输入音频，输出文本，支持 99 种语言，抗噪能力强，且可在笔记本电脑上运行。到 2024 年，OpenAI 已推出 Large-v3 和 Turbo 变体；至 2026 年，Whisper 已成为从播客转录、语音助手到 YouTube 字幕等各类应用的默认基线模型。

但 Whisper 并非一个可以永远当作黑盒 (Black Box) 使用的处理流水线。领域偏移 (Domain Shift) 会严重削弱其性能——例如专业术语、说话人口音、专有名词、短音频片段以及静音段。你需要了解：

1. 其内部的实际架构与原理。
2. 如何正确地向其输入分块 (Chunked)、流式 (Streaming) 或长格式 (Long-Form) 音频。
3. 何时需要进行微调 (Fine-Tuning) 以及具体操作方法。

## 核心概念

![Whisper 编码器-解码器、任务、分块推理、微调](../assets/whisper.svg)

**架构 (Architecture)。** 标准的 Transformer 编码器-解码器 (Transformer encoder-decoder) 结构。

- 输入：30 秒的对数梅尔频谱图 (log-mel spectrogram)，80 个梅尔频带 (mels)，10 毫秒帧移 (hop) → 3000 帧。较短的音频片段会进行零填充 (zero-padded)，较长的片段则进行分块处理 (chunked)。
- 编码器：卷积下采样 (conv-downsample，步长为 2) + `N` 个 Transformer 模块。以 Large-v3 为例：32 层，1280 维，20 个注意力头 (attention heads)。
- 解码器：包含 `N` 个 Transformer 模块，采用因果自注意力 (causal self-attention) 以及与编码器输出的交叉注意力 (cross-attention)。规模与编码器相同。
- 输出：基于 51,865 词表大小的字节对编码词元 (BPE tokens)。

Large-v3 拥有 15.5 亿参数。Turbo 版本将解码器层数从 32 层缩减至 4 层，在词错误率 (Word Error Rate, WER) 损失不到 1% 的情况下，将延迟降低了 8 倍。

**提示词格式 (Prompt format)。** Whisper 是一个多任务模型，其任务类型由解码器提示词中的特殊词元控制：

<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>

- `<|en|>` — 语言标签；用于强制指定翻译或转录行为。
- `<|transcribe|>` 或 `<|translate|>` — 将任意语言输入翻译为英文输出，或进行逐字转录。
- `<|notimestamps|>` — 跳过词级时间戳（推理速度更快）。

正是这种提示词机制使得单一模型能够执行多种任务。只需将 `<|en|>` 替换为 `<|fr|>`，模型即可转录法语。

**30 秒窗口 (30-second window)。** 所有处理均固定为 30 秒。较长的音频需要进行分块处理，较短的音频则需填充。该模型原生不支持流式处理窗口——这正是 WhisperX、Whisper-Streaming 和 faster-whisper 等衍生工具存在的原因。

**对数梅尔归一化 (Log-mel normalization)。** 计算公式为 `(log_mel - mean) / std`，其中的统计均值与标准差源自 Whisper 自身的训练语料库。你*必须*使用 Whisper 自带的预处理函数 (`whisper.audio.log_mel_spectrogram`)，而非 `librosa.feature.melspectrogram`。

### 2026 年版本变体

| 变体 | 参数量 | 延迟 (A100) | 词错误率 (WER) (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× 实时 | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | 流式 | 2.0% |

### 微调 (Fine-tuning)

2026 年的标准工作流如下：

1. 收集 10 至 100 小时的目标领域音频及其对齐的转录文本。
2. 使用 `generate_with_loss` 回调函数运行 `transformers.Seq2SeqTrainer`。
3. 参数高效微调：在注意力层的 `q_proj`、`k_proj`、`v_proj` 上应用低秩自适应 (LoRA)，可在词错误率损失小于 0.3% 的情况下，将 GPU 显存占用降低 4 倍。
4. 若数据量不足 10 小时，请冻结编码器，仅微调解码器。
5. 务必使用 Whisper 原生的分词器 (tokenizer) 和提示词格式，切勿替换其他分词器。

社区实践结果：在 20 小时医疗听写数据上微调 Medium 模型，可使医疗词汇的词错误率从 12% 降至 4.5%。在 4 小时冰岛语数据上微调 Turbo 模型，词错误率从 18% 降至 6%。

## 动手实践 (Build It)

### 步骤 1：开箱即用运行 Whisper

import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")

建议始终手动覆盖的关键默认参数：`temperature=0.0`（采样温度默认回退链为 0.0 → 0.2 → 0.4 …）、`condition_on_previous_text=False`（防止级联幻觉问题 (cascading hallucination problem)），以及 `no_speech_threshold=0.6`（静音检测阈值）。

### 步骤 2：长音频分块处理

# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)

WhisperX 增加了以下功能：(1) Silero 语音活动检测 (Voice Activity Detection, VAD) 门控，(2) 基于 wav2vec 2.0 的词级对齐 (word-level alignment)，(3) 通过 `pyannote.audio` 实现的说话人分离 (diarization)。它是 2026 年生产环境语音转写的主力工具。

### 步骤 3：使用 LoRA 进行微调

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total

随后接入标准的 Trainer 训练循环。每 1000 步保存一次检查点 (checkpoint)，并在预留数据集 (held-out set) 上使用词错误率 (Word Error Rate, WER) 进行评估。

### 步骤 4：检查每一层的学习内容

# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len

使用热力图 (heatmap) 进行可视化——当解码器步骤扫描编码器帧时，你会看到对角线对齐模式。这条对角线正是 Whisper 生成词级时间戳的底层逻辑。

## 实际应用指南

2026 年技术栈选型：

| 场景 | 推荐方案 |
|-----------|------|
| 通用英语，离线环境 | 通过 `whisperx` 调用 Large-v3-turbo |
| 移动端 / 边缘设备 | 量化版 (int8) Whisper-Tiny 或 Moonshine |
| 多语言长音频 | 通过 `whisperx` 调用 Large-v3 + 说话人分离 |
| 低资源语言 | 使用 LoRA 微调 Medium 或 Turbo 模型 |
| 流式处理（2 秒延迟） | Whisper-Streaming 或 Parakeet-TDT |
| 词级时间戳 | WhisperX（基于 wav2vec 2.0 的强制对齐 (forced alignment)） |

`faster-whisper`（基于 CTranslate2 后端）是 2026 年最快的 CPU+GPU 推理运行时 (inference runtime)——在输出结果完全一致的前提下，速度比原版快 4 倍。

## 2026 年依然存在的常见陷阱

- **静音幻觉（Hallucination on silence）。** 基于字幕数据训练的 Whisper 模型会输出“感谢观看！”、“订阅！”或歌词等内容。在调用模型前务必使用语音活动检测（Voice Activity Detection, VAD）进行前置过滤。
- **`condition_on_previous_text` 级联效应。** 单次幻觉会污染后续的音频窗口。除非你需要跨音频块保持语句连贯性，否则请将其设置为 `False`。
- **短音频填充（Short-clip padding）。** 将 2 秒的音频填充至 30 秒时，模型可能在尾部的静音段产生幻觉。请使用 `pad=False` 或进行 VAD 门控过滤。
- **错误的梅尔频谱统计（Wrong mel stats）。** 使用 `librosa` 生成的梅尔频谱（Mel spectrogram）而非 Whisper 原生的频谱会导致输出近乎随机。请使用 `whisper.audio.log_mel_spectrogram`。

## 部署上线（Ship It）

保存为 `outputs/skill-whisper-tuner.md`。为特定领域设计一个 Whisper 微调（fine-tune）或推理（inference）流水线。

## 练习

1. **简单。** 运行 `code/main.py`。该脚本会对 Whisper 风格的提示词进行分词（tokenize），计算解码形状预算（decoded shape budgets），并打印 10 分钟音频的分块调度计划（chunk schedule）。
2. **中等。** 安装 `faster-whisper`，转录一段 10 分钟的播客，并与人工转录文本对比词错误率（Word Error Rate, WER）。尝试对比 `language="auto"` 与强制指定 `language="en"` 的效果。
3. **困难。** 使用 Hugging Face `datasets` 库，挑选一个 Whisper 表现不佳的语言（如乌尔都语），在 2 小时音频数据上使用低秩自适应（Low-Rank Adaptation, LoRA）对 Medium 模型微调 2 个训练轮次（epoch），并报告词错误率（WER）的变化差值。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 30 秒窗口（30-sec window） | Whisper 的限制 | 硬性输入上限；需将更长音频分块处理。 |
| SOT | 转录起始标记（Start-of-transcript） | `<|startoftranscript|>` 用于启动解码器提示词。 |
| 时间戳标记（Timestamps token） | 时间对齐 | 每 0.02 秒的偏移量在 5.1 万词表中都对应一个特殊标记。 |
| Turbo | 快速变体 | 仅含 4 层解码器，速度提升 8 倍，词错误率（WER）回退幅度小于 1%。 |
| WhisperX | 长音频封装工具 | 结合语音活动检测（VAD）+ Whisper + wav2vec 对齐 + 说话人分离（diarization）。 |
| LoRA 微调（LoRA fine-tune） | 高效微调 | 在注意力机制中添加低秩适配器；仅训练约 0.3% 的参数。 |
| 幻觉（Hallucination） | 静默失败 | Whisper 会从噪声或静音中生成流畅的英文文本。 |

## 延伸阅读

- [Radford 等人 (2022). Whisper 论文](https://arxiv.org/abs/2212.04356) — 原始架构与训练方案。
- [OpenAI (2024). Whisper Large-v3-turbo 发布](https://github.com/openai/whisper/discussions/2363) — 4 层解码器，8 倍加速。
- [Bain 等人 (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 长音频处理、词级对齐、说话人分离。
- [Systran — faster-whisper 仓库](https://github.com/SYSTRAN/faster-whisper) — 基于 CTranslate2 实现，速度提升 4 倍。
- [HuggingFace — Whisper 微调教程](https://huggingface.co/blog/fine-tune-whisper) — 标准的 LoRA / 全量微调（full-FT）实战指南。