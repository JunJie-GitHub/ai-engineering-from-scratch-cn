---
name: ctc解码器
description: 从零开始编写贪心（Greedy）和束搜索（Beam Search）CTC解码器，包含长度归一化（Length Normalisation）
version: 1.0.0
phase: 4
lesson: 19
tags: [光学字符识别, 连接时序分类, 解码, 序列模型]
---

# CTC解码器（CTC Decoder）

为CTC输出生成两种解码例程：贪心解码（Greedy Decoding，速度快）和束搜索解码（Beam Search Decoding，在噪声输入下表现更优）。

## 适用场景

- 在自定义CRNN（卷积循环神经网络）输出上运行OCR（光学字符识别）推理。
- 使用不同解码器对预训练OCR模型进行基准测试。
- 在不引入 `ctcdecode` 的情况下实现简单的束搜索。

## 输入参数

- `log_probs`：词汇表上的对数Softmax（Log-Softmax）输出，形状为 (T, N, C)（按惯例索引 0 表示空白符）。
- `vocab`：包含 C 个字符的列表。
- `beam_width`（仅束搜索使用）：通常设置为 5-10。

## 贪心解码器（Greedy Decoder）

def greedy_ctc_decode(log_probs, vocab, blank=0):
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(vocab[idx])
            prev = idx
        out.append("".join(decoded))
    return out

## 束搜索解码器（Beam Search Decoder）

import heapq
import math

def beam_ctc_decode(log_probs, vocab, beam_width=5, blank=0):
    T, N, C = log_probs.shape
    lp = log_probs.cpu()
    results = []
    for n in range(N):
        beams = {("",): (0.0, -math.inf)}  # (prefix_tuple) -> (p_blank, p_nonblank)
        for t in range(T):
            logits_t = lp[t, n]
            new_beams = {}
            for prefix, (p_b, p_nb) in beams.items():
                for c in range(C):
                    p = logits_t[c].item()
                    if c == blank:
                        nb = p_b + p
                        nnb = p_nb + p
                        upd = new_beams.get(prefix, (-math.inf, -math.inf))
                        new_beams[prefix] = (
                            _logsumexp(upd[0], _logsumexp(nb, nnb)),
                            upd[1],
                        )
                    else:
                        last = prefix[-1] if prefix else ""
                        char = vocab[c]
                        if char == last:
                            # Case 1: stay on same prefix (collapse from p_nb)
                            upd = new_beams.get(prefix, (-math.inf, -math.inf))
                            new_beams[prefix] = (upd[0], _logsumexp(upd[1], p_nb + p))
                            # Case 2: extend prefix via blank-separated repeat ("a_a" -> "aa")
                            new_prefix = prefix + (char,)
                            upd = new_beams.get(new_prefix, (-math.inf, -math.inf))
                            new_beams[new_prefix] = (upd[0], _logsumexp(upd[1], p_b + p))
                        else:
                            new_prefix = prefix + (char,)
                            upd = new_beams.get(new_prefix, (-math.inf, -math.inf))
                            nb = _logsumexp(p_b, p_nb) + p
                            new_beams[new_prefix] = (upd[0], _logsumexp(upd[1], nb))
            beams = dict(heapq.nlargest(
                beam_width,
                new_beams.items(),
                key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
            ))
        best = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
        results.append("".join(best))
    return results


def _logsumexp(a, b):
    if a == -math.inf: return b
    if b == -math.inf: return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))

## 规则

- 在 PyTorch 的 `nn.CTCLoss` 中，按照惯例，CTC (Connectionist Temporal Classification) 的空白索引 (blank index) 默认为 0。
- 束搜索 (Beam Search) 可提升低置信度输入的准确率；在干净输入上，其带来的字符错误率 (Character Error Rate) 改善幅度不足 1%。
- 切勿将束宽 (beam width) 缩减至 5 以下；低于该值时，准确率与延迟的权衡 (accuracy-latency trade) 收益将趋于平缓。
- 若在严格的延迟预算下运行束搜索，建议降级为贪婪解码 (greedy decoding)；在大多数生产环境的 OCR 数据上，此举带来的质量损失微乎其微。
- 针对大规模词表 (vocabulary)（如包含 3000 余个字符的中日韩 (CJK) 字符集），建议切换至 `ctcdecode` (C++) 实现，而非使用上述纯 Python 版本；否则纯 Python 实现的束搜索将迅速成为性能瓶颈。