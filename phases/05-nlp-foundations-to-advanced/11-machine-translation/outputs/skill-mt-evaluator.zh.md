---
name: 机器翻译评估器
description: 评估待发布的机器翻译（machine translation）输出。
version: 1.0.0
phase: 5
lesson: 11
tags: [自然语言处理, 翻译, 评估]
---

给定源文本和候选翻译，输出以下内容：

1. 自动评分预估。说明预期的 BLEU 和 chrF 分数范围。声明是否提供参考译文（reference）。
2. 五点人工可验证清单：内容保真（content preservation，无幻觉 hallucinations）、目标语言正确、语域（register）/正式程度匹配、若提供术语表则保持术语一致性、无截断（truncation）或长度爆炸（length explosion）。
3. 一个需探查的领域特定（domain-specific）问题。法律领域：命名实体（named entities）、法规引用。医疗领域：药物名称、剂量。用户界面（UI）：如 `{name}` 的占位符变量。
4. 置信度标志。标记为“发布（Ship）”/“审核后发布（Ship with review）”/“不发布（Do not ship）”。该标志需与发现问题的严重程度挂钩。

若未对输出进行语言识别（language-ID）检查，则拒绝发布。除非用户明确选择无参考评分（reference-free scoring，如 COMET-QE、BLEURT-QE），否则在无参考译文的情况下拒绝评估。将超过 1000 个词元（token）的内容标记为可能需要分块翻译（chunked translation）。