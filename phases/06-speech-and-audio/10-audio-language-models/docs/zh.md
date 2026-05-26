# 音频-语言模型（Audio-Language Models）—— Qwen2.5-Omni、Audio Flamingo 与 GPT-4o Audio

> 2026 年的音频-语言模型能够对语音、环境音和音乐进行联合推理。Qwen2.5-Omni-7B 在 MMAU-Pro 基准测试上的表现与 GPT-4o Audio 持平。Audio Flamingo Next 在 LongAudioBench 上超越了 Gemini 2.5 Pro。开源与闭源模型之间的差距已基本抹平——但在多音频任务上，所有模型的表现仍接近随机猜测。

**类型：** 学习
**编程语言：** Python
**前置知识：** 第 6 阶段 · 04（自动语音识别，ASR）、第 12 阶段 · 03（视觉-语言模型，Vision-Language Models）、第 7 阶段 · 10（音频 Transformer，Audio Transformers）
**预计耗时：** 约 45 分钟

## 问题

假设你有一段 5 秒的音频：先是狗叫声，接着有人大喊“停下！”，随后是寂静。针对这段音频，有价值的提问可以涵盖多个维度：

- **语音转写（Transcription）。**“刚才说了什么？”——这属于自动语音识别（ASR）的范畴。
- **语义推理（Semantic reasoning）。**“这个人有危险吗？”——需要模型联合理解狗叫、喊叫和寂静之间的上下文关系。
- **音乐推理（Music reasoning）。**“演奏这段旋律的是哪些乐器？”
- **长音频检索（Long-audio retrieval）。**“在这 90 分钟的讲座中，讲师是在哪个时间点讲解梯度下降的？”

能够仅凭一个提示词（prompt）就回答上述所有问题的单一模型，即为**音频-语言模型**（Audio-Language Model，LALM / ALM）。它与纯 ASR 模型的区别在于：LALM 能够生成自由形式的自然语言回答，而不仅仅是输出转写文本。

## 核心概念

![音频-语言模型：音频编码器 + 投影层 + 大语言模型解码器](../assets/alm-architecture.svg)

### 三组件架构模板

2026 年的所有大型音频-语言模型 (Large Audio-Language Model, LALM) 都遵循相同的骨架结构：

1. **音频编码器 (Audio Encoder)。** Whisper 编码器 · BEATs · CLAP · WavLM · 或各模型自定义的编码器。
2. **投影层 (Projector)。** 采用线性层或多层感知机 (MLP)，将音频编码器的特征映射到大语言模型 (Large Language Model, LLM) 的词元 (Token) 嵌入空间。
3. **大语言模型 (LLM)。** 基于 Llama / Qwen / Gemma 的解码器。接收交错的文本与音频词元，并生成文本。

训练流程：

- **第一阶段 (Stage 1)。** 冻结编码器与大语言模型；仅使用自动语音识别 (ASR) / 音频描述数据训练投影层。
- **第二阶段 (Stage 2)。** 在遵循指令的音频任务（问答、推理、音乐理解）上进行全量微调或低秩自适应 (LoRA) 微调。
- **第三阶段（可选）。** 语音输入/语音输出模式需额外添加语音解码器。Qwen2.5-Omni 和 AF3-Chat 采用了此设计。

### 2026 年模型图谱

| 模型 | 基座模型 | 音频编码器 | 输出模态 | 访问方式 |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自定义 + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自定义 | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商业许可 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商业许可 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro（闭源） | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio（闭源） | GPT-4o | 专有 | 文本 + 语音 | API |

### 基准测试现状评估（2026）

**MMAU-Pro。** 包含 1800 组问答对，涵盖语音 / 环境音 / 音乐 / 混合音频。内含多音频子集。

| 模型 | 综合得分 | 语音 | 环境音 | 音乐 | 多音频 |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench 上的业界最优 (State-of-the-Art, SOTA) | — | — | — | — |

**多音频列的数据对所有人来说都十分惨淡。** 四选一多项选择题的随机猜测概率为 25%；大多数模型的得分仅徘徊在此水平。大型音频-语言模型在对比两个音频片段时依然面临困难。

### 2026 年大型音频-语言模型的适用场景

- **呼叫中心录音合规审计。** “客服是否提及了必要的免责声明？”
- **无障碍辅助。** 为听障用户描述声音事件（而非仅仅提供语音转写）。
- **内容审核。** 检测暴力语言 + 威胁性语气 + 背景上下文。
- **播客 / 会议章节划分。** 生成语义摘要，而非仅按说话人轮次切分。
- **音乐曲库分析。** “找出所有在 B 段发生转调的曲目。”

### 目前尚不适用的场景

- 细粒度乐理分析（和弦级别以下）。
- 长对话中的说话人归属推理（超过 10 分钟后性能显著下降）。
- 多音频对比（22%-26% 的准确率仅略高于随机猜测）。
- 实时流式推理（大多数模型仍依赖离线批量推理）。

## 构建

### 步骤 1：查询 Qwen2.5-Omni

from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))

### 步骤 2：投影器模式 (Projector Pattern)

import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))

就这么简单。投影器 (Projector) 通常由 1 到 3 个线性层 (Linear Layer) 构成。在自动语音识别 (Automatic Speech Recognition, ASR) 配对数据（音频 → 转录文本）上对其进行训练，属于第一阶段的代理任务 (Pretext Task)。

### 步骤 3：基准测试 (Benchmarking) MMAU / LongAudioBench

from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")

请分别报告每个类别（语音 / 环境音 / 音乐 / 多音频）的结果。汇总的总分往往会掩盖模型在特定场景下的失败之处。

## 使用

| 任务 | 2026 年推荐方案 |
|------|-----------|
| 自由格式音频问答（开源） | Qwen2.5-Omni-7B |
| 长音频最佳开源模型 | Audio Flamingo Next |
| 最佳闭源模型 | Gemini 2.5 Pro |
| 语音输入/输出智能体 (Agent) | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 (Music Reasoning) | Audio Flamingo 3 或 2（音乐专用 AF-CLAP） |
| 呼叫中心质检 | 通过 API 调用 Gemini 2.5 Pro，并结合检索增强生成 (Retrieval-Augmented Generation, RAG) 处理您的政策文档 |

## 避坑指南

- **过度信任多音频处理能力。** 如果您的任务需要判断“哪个片段包含 X”，模型的实际表现可能真的只相当于随机猜测水平。
- **长音频性能衰减。** 超过 10 分钟后，大多数模型的说话人归属 (Speaker Attribution) 能力会失效。请先进行说话人分离 (Speaker Diarization)（见第 6 课），然后再进行摘要。
- **静音幻觉 (Hallucination)。** 使用 Whisper 编码器 (Encoder) 的大型音频语言模型 (Large Audio-Language Models, LALMs) 会继承同样的 Whisper 风格问题。请使用语音活动检测 (Voice Activity Detection, VAD) 进行门控过滤。
- **基准测试数据挑拣 (Cherry-picking)。** 厂商的博客文章通常会突出表现最好的类别。请自行运行 MMAU-Pro 的多音频子集进行测试。

## 交付

将文件保存为 `outputs/skill-alm-picker.md`。针对特定的音频理解任务，选择合适的大型音频语言模型 (LALM)、基准测试子集以及输出模态 (Output Modality)（文本或语音）。

## 练习

1. **简单。** 运行 `code/main.py` 即可查看一个简易的投影器（Projector）模式，以及模拟的大型音频语言模型（LALM）路由流程：将音频嵌入向量（audio-embedding）与文本词元（text-tokens）转换为输出词元（output tokens）。
2. **中等。** 在 100 个 MMAU-Pro 语音测试项上评估 Qwen2.5-Omni-7B，并将结果与论文报告的数值进行对比。
3. **困难。** 构建一个最简的音频描述（audio-captioning）基线模型：BEATs 编码器（encoder） + 2 层投影器 + 冻结参数的 Llama-3.2-1B。仅在 AudioCaps 数据集上微调投影器，并在 Clotho-AQA 数据集上与 SALMONN 模型进行对比。

## 关键术语

| 术语 | 通俗叫法 | 技术实质 |
|------|----------|----------|
| LALM | 音频版 ChatGPT | 音频编码器（Audio encoder） + 投影器 + 大语言模型（LLM）解码器（decoder）。 |
| Projector | 适配器（Adapter） | 将音频特征映射至大语言模型嵌入空间的小型多层感知机（MLP）。 |
| MMAU | 该基准测试（benchmark） | 涵盖语音、环境音与音乐的 1 万组音频问答对。 |
| MMAU-Pro | 进阶版 MMAU | 包含 1800 道多音频/强推理类问题。 |
| LongAudioBench | 长音频评估 | 针对数分钟长音频片段进行语义查询的测试集。 |
| Voice-in / voice-out | 原生语音交互 | 模型直接接收语音输入并生成语音输出，无需经过文本中转。 |

## 延伸阅读

- [Chu 等人 (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 参考架构。
- [阿里巴巴 (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 语音输入/语音输出。
- [英伟达 (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开源长音频领域的领先模型。
- [英伟达 (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench 基准测试的最优模型（SOTA）。
- [Tang 等人 (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 双编码器（dual-encoder）架构的先驱。
- [MMAU-Pro 排行榜](https://mmaubenchmark.github.io/) — 2026 年实时排名。