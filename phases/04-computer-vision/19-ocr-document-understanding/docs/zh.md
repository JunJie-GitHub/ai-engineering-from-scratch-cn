# 光学字符识别（OCR）与文档理解

> 光学字符识别（OCR）是一个三阶段流水线（pipeline）——检测文本框、识别字符，然后进行版面还原。现代 OCR 系统通常会调整这些阶段的顺序或将它们合并。

**类型：** 学习与应用
**编程语言：** Python
**前置知识：** 第4阶段第06课（检测），第7阶段第02课（自注意力机制（Self-Attention））
**预计耗时：** 约45分钟

## 学习目标

- 梳理经典的光学字符识别（OCR）流水线（检测 -> 识别 -> 版面还原）与现代端到端（end-to-end）替代方案（如 Donut、Qwen-VL-OCR）
- 实现连接时序分类（Connectionist Temporal Classification, CTC）损失函数，用于序列到序列（sequence-to-sequence）的 OCR 训练
- 使用 PaddleOCR 或 EasyOCR 进行免训练的生产级文档解析
- 区分光学字符识别（OCR）、版面解析（layout parsing）与文档理解（document understanding），并根据具体任务选择合适的工具

## 问题背景

充满文本的图像无处不在：收据、发票、身份证件、扫描书籍、表格、白板、标牌、截图等。从中提取结构化数据——不仅仅是识别字符，还要理解“这是总金额”——是应用视觉（applied vision）领域价值最高的问题之一。

该领域可划分为三个技能层级：

1. **核心 OCR**：将像素转换为文本。
2. **版面解析（layout parsing）**：将 OCR 输出结果按区域分组（如标题、正文、表格、页眉）。
3. **文档理解（document understanding）**：从版面中提取结构化字段（如 "invoice_total = $42.50"）。

每个层级都有经典与现代两种实现路径。从“我想从图片中提取文字”到“我需要从这张收据中获取总金额”，这两者之间的技术鸿沟比大多数团队意识到的要大得多。

## 核心概念

### 经典流水线

flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a

- **文本检测 (Text Detection)** 生成按行或按词的四边形边界框。
- **识别 (Recognition)** 将每个区域裁剪为固定高度，运行 CNN + BiLSTM + CTC 以生成字符序列。
- **版面分析 (Layout)** 重建阅读顺序（拉丁语系为从上到下、从左到右；阿拉伯语、日语等则不同）。

### 一段话理解 CTC

光学字符识别 (Optical Character Recognition, OCR) 模块从固定长度的特征图中生成可变长度的序列。连接时序分类 (Connectionist Temporal Classification, CTC)（Graves 等人，2006）使得模型无需字符级对齐即可进行训练。模型在每个时间步输出一个覆盖（词表 + 空白符）的概率分布；CTC 损失函数会对所有可能的对齐路径进行边缘化计算，这些路径在合并重复字符并去除空白符后，即可还原为目标文本。

raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"

CTC 正是卷积循环神经网络 (Convolutional Recurrent Neural Network, CRNN) 在 2015 年取得成功的原因，并且至今仍在 2026 年训练着大多数生产环境的 OCR 模型。

### 现代端到端模型

- **Donut**（Kim 等人，2022）—— 采用视觉 Transformer (Vision Transformer, ViT) 编码器 + 文本解码器；直接读取图像并输出 JSON。无需文本检测器，也无需版面分析模块。
- **TrOCR** —— ViT + Transformer 解码器，用于行级 OCR。
- **Qwen-VL-OCR / InternVL** —— 针对 OCR 任务微调的完整视觉语言模型 (Vision-Language Models)；在 2026 年处理复杂文档时准确率最高。
- **PaddleOCR** —— 基于经典 DB + CRNN 流水线的成熟生产级工具包；至今仍是开源领域的主力。

端到端模型需要更多的数据和算力，但避免了多阶段流水线中的误差累积问题。

### 版面解析

对于结构化文档，可运行版面检测器（如 LayoutLMv3、DocLayNet）对每个区域进行标注：标题、段落、图片、表格、脚注。此时的阅读顺序即为“按版面顺序遍历各区域并拼接”。

对于表单类文档，可使用**键值提取 (Key-Value Extraction)** 模型（Donut 适用于视觉丰富文档，LayoutLMv3 适用于普通扫描件）。它们接收图像 + 检测到的文本 + 位置信息，并预测结构化的键值对。

### 评估指标

- **字符错误率 (Character Error Rate, CER)** —— 莱文斯坦距离 (Levenshtein Distance) / 参考文本长度。数值越低越好。生产环境目标：在清晰扫描件上 < 2%。
- **词错误率 (Word Error Rate, WER)** —— 原理相同，但计算粒度为词级。
- **结构化字段 F1 分数** —— 适用于键值提取任务；用于衡量如 `{invoice_total: 42.50}` 等字段是否被正确提取。
- **JSON 编辑距离** —— 适用于端到端文档解析；Donut 论文中引入了归一化树编辑距离 (Normalised Tree Edit Distance)。

## 动手构建

### 步骤 1：CTC 损失 (CTC Loss) + 贪婪解码器 (Greedy Decoder)

import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out

在可用时，`F.ctc_loss` 会使用高效的 CuDNN 实现。贪婪解码器比束搜索 (Beam Search) 更简单，且其字符错误率 (Character Error Rate, CER) 通常与束搜索相差在 1% 以内。

### 步骤 2：微型 CRNN 识别器 (Tiny CRNN Recogniser)

用于行级光学字符识别 (Optical Character Recognition, OCR) 的极简卷积神经网络 (Convolutional Neural Network, CNN) + 双向长短期记忆网络 (Bidirectional Long Short-Term Memory, BiLSTM)。

class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)

固定高度输入（CNN 通过最大池化 (Max Pooling) 将高度压缩至 1）。宽度作为 CTC 的时间维度。

### 步骤 3：合成 OCR 数据

生成白底黑字的数字字符串，用于端到端 (End-to-End) 的冒烟测试 (Smoke Test)。

import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")

真实的 OCR 数据集会额外包含字体、噪声、旋转、模糊和色彩变化。上述处理流水线 (Pipeline) 完全一致。

### 步骤 4：训练流程示例

model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()

在此简单的合成数据上，经过 200 个训练步 (Steps) 后，损失值 (Loss) 应从约 3 下降至约 0.2。

## 使用方法

三条生产级路径：

- **PaddleOCR** —— 成熟、快速、支持多语言。单行调用示例：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR** —— 原生 Python 实现、支持多语言、基于 PyTorch 构建。
- **Tesseract** —— 经典方案；当现代模型表现不佳时，仍适用于处理老旧扫描文档。

对于端到端文档解析，可使用 Donut 或视觉语言模型（Vision-Language Model, VLM）：

from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

对于具有重复结构的收据、发票和表单，建议对 Donut 进行微调（Fine-tuning）。对于任意格式的文档或需要推理能力的 OCR 任务，目前默认推荐使用 Qwen-VL-OCR 等视觉语言模型。

## 交付产出

本课时将生成以下文件：

- `outputs/prompt-ocr-stack-picker.md` —— 一个提示词（Prompt），可根据文档类型、语言和结构自动选择 Tesseract / PaddleOCR / Donut / VLM-OCR。
- `outputs/skill-ctc-decoder.md` —— 一份技能指南，指导从零编写贪婪解码（Greedy Decoding）和束搜索（Beam Search）的 CTC 解码器，包含长度归一化（Length Normalisation）实现。

## 练习

1. **（简单）** 在 5 位随机数字字符串上训练 TinyCRNN 模型 500 步。在预留测试集（Held-out Set）上报告字符错误率（Character Error Rate, CER）。
2. **（中等）** 将贪婪解码替换为束搜索（`beam_width=5`）。报告 CER 差值（Delta）。在哪些输入样本上束搜索表现更优？
3. **（困难）** 使用 PaddleOCR 处理 20 张收据，提取明细行项目，并针对 `{item_name, price}` 键值对，与人工标注的真实标签（Ground Truth）计算 F1 分数。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| OCR（光学字符识别） | “从像素中提取文本” | 将图像区域转换为字符序列 |
| CTC（连接时序分类） | “免对齐损失函数” | 无需逐时间步标签即可训练序列模型的损失函数；通过对所有对齐路径进行边缘化（Marginalisation）计算 |
| CRNN（卷积循环神经网络） | “经典 OCR 模型” | 卷积特征提取器 + 双向 LSTM（BiLSTM）+ CTC；2015 年提出的基线模型，目前仍广泛用于生产环境 |
| Donut | “端到端 OCR” | ViT 编码器 + 文本解码器；直接从图像生成 JSON 输出 |
| Layout parsing（版面解析） | “定位区域” | 检测并标注文档中的标题、表格、图片、段落等区域 |
| Reading order（阅读顺序） | “文本序列” | 将识别出的区域按逻辑顺序排列成句；拉丁排版较简单，混合排版则较复杂 |
| CER / WER（字符/词错误率） | “错误率” | 字符或词粒度下的莱文斯坦距离（Levenshtein Distance）与参考文本长度的比值 |
| VLM-OCR（视觉语言模型 OCR） | “能阅读的 LLM” | 专为 OCR 任务训练或提示的视觉语言模型；目前在复杂文档处理上代表最先进水平（State-of-the-Art, SOTA） |

## 延伸阅读

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) — 首创的卷积神经网络 (CNN) + 循环神经网络 (RNN) + 连接时序分类 (CTC) 架构
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — 连接时序分类 (CTC) 的奠基性论文；密集阐述了核心算法思想
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664) — 免光学字符识别 (OCR) 的文档理解 Transformer 模型
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — 开源的生产级光学字符识别 (OCR) 技术栈