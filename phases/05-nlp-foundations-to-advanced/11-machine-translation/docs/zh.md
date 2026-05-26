# 机器翻译（Machine Translation）

> 翻译任务在过去三十年里一直是自然语言处理（Natural Language Processing）研究的主要资金来源，并且至今仍在持续推动该领域的发展。

**类型：** 构建实践
**编程语言：** Python
**前置知识：** 第5阶段 · 10（注意力机制（Attention Mechanism）），第5阶段 · 04（GloVe、FastText、子词（Subword））
**预计耗时：** 约75分钟

## 问题背景

模型读取一种语言的句子，并生成另一种语言的句子。句子长度各不相同，词序也千差万别。部分源语言词汇会对应多个目标语言词汇，反之亦然。习语更是无法进行一对一映射。例如，法语中的“我想你”是“tu me manques”，字面意思是“你对我来说是缺失的”。在这种情况下，任何词级对齐（Word-level Alignment）都会失效。

机器翻译正是推动自然语言处理领域发明编码器-解码器（Encoder-Decoder）架构、注意力机制、Transformer 模型，并最终催生整个大语言模型（Large Language Model）范式的核心任务。每一次技术突破都源于翻译质量是可量化的，且人机之间的性能差距始终难以弥合。

本课程将跳过历史回顾，直接讲解 2026 年实际可用的工作流水线（Pipeline）：预训练多语言编码器-解码器模型（如 NLLB-200 或 mBART）、子词分词（Subword Tokenization）、束搜索（Beam Search）、BLEU 与 chrF 评估指标，以及那些仍会悄然流入生产环境的少数典型失败模式（Failure Modes）。

## 核心概念

![机器翻译流水线：分词 → 编码 → 带注意力的解码 → 逆分词](../assets/mt-pipeline.svg)

现代机器翻译（Machine Translation）是基于平行语料（Parallel Text）训练的 Transformer 编码器-解码器架构。编码器以源语言的分词形式读取输入。解码器则通过交叉注意力机制（Cross-Attention，见第10课）利用编码器的输出，每次生成一个子词。解码过程采用束搜索以避免陷入贪婪解码（Greedy Decoding）的局部最优陷阱。最终输出会经过逆分词（Detokenization）、恢复大小写（Detruecasing）处理，并与参考译文进行评分对比。

在实际应用中，有三个关键操作决策直接决定了机器翻译的质量。

- **分词器（Tokenizer）。** 基于混合语言语料库训练的 SentencePiece BPE（Byte Pair Encoding）。跨语言共享词表正是 NLLB 能够实现零样本（Zero-shot）翻译对的核心所在。
- **模型规模（Model Size）。** NLLB-200 蒸馏版 600M 参数模型可在笔记本电脑上运行。NLLB-200 3.3B 是官方推荐的生产环境默认配置。54.5B 则是当前的研究上限。
- **解码策略（Decoding）。** 通用内容通常设置束宽（Beam Width）为 4-5。引入长度惩罚（Length Penalty）以防止输出过短。在需要保证术语一致性时，则采用约束解码（Constrained Decoding）。

## 动手实践

### 步骤 1：调用预训练机器翻译 (Machine Translation, MT) 模型

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])

Les chats courent.

这里有三点需要注意。`src_lang` 用于告知分词器 (Tokenizer) 应用何种书写系统与分词规则。`forced_bos_token_id` 用于指示解码器 (Decoder) 生成目标语言。这两项均为 NLLB 模型特有的技巧；mBART 和 M2M-100 模型遵循各自的约定，彼此不可互换。

### 步骤 2：BLEU 与 chrF 指标

BLEU (Bilingual Evaluation Understudy) 用于衡量模型输出与参考译文之间的 n元语法 (n-gram) 重叠度。它计算 1 到 4 阶 n-gram 的精确率几何平均值，并对过短的输出施加简短惩罚 (Brevity Penalty)。得分范围为 [0, 100]。该指标应用广泛，但解读起来往往令人头疼：30 分代表“可用”，40 分代表“良好”，50 分代表“优异”；1 分以内的差异通常视为噪声。

chrF 衡量的是字符级别的 F分数 (F-score)。对于形态丰富的语言，BLEU 往往会低估匹配度，而 chrF 对此更为敏感。通常与 BLEU 结合报告。

import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")

务必使用 `sacrebleu` 库。它对分词进行了标准化处理，确保不同论文间的得分具有可比性。自行实现 BLEU 计算往往是导致基准测试 (Benchmark) 结果失真的根源。

### 三层评估体系（2026 版）

现代机器翻译评估采用三类互补的指标体系。实际部署时至少应包含其中两种。

- **启发式指标 (Heuristic)**（如 BLEU、chrF）：计算速度快、依赖参考译文、可解释性强，但对同义改写不敏感。适用于历史版本对比与回归检测。
- **学习型指标 (Learned)**（如 COMET、BLEURT、BERTScore）：基于人类评分数据训练的神经网络模型，用于比较译文与原文及参考译文的语义相似度。自 2023 年以来，COMET 与机器翻译研究的相关性最高，在 2026 年已成为对质量要求较高的生产环境默认指标。
- **大语言模型裁判 (LLM-as-judge)**（无参考）：通过提示词 (Prompt) 让大模型从流畅度、忠实度、语气及文化适宜性等维度对译文打分。当评分标准设计合理时，GPT-4 裁判与人类评分的一致性可达约 80%。适用于缺乏参考译文的开放式内容评估。

2026 年实用技术栈：使用 `sacrebleu` 计算 BLEU 和 chrF，使用 `unbabel-comet` 计算 COMET，并使用提示词驱动的大语言模型提供最终面向用户的评估信号。在将任何指标应用于生产数据之前，务必使用 50-100 条人工标注样本对其进行校准。

无参考指标（如 COMET-QE、BLEURT-QE、LLM-as-judge）允许你在没有参考译文的情况下评估翻译质量，这对于缺乏参考译文的长尾语言对 (Long-tail Language Pairs) 尤为重要。

### 步骤 3：生产环境中的常见故障

上述可运行的流水线在 80% 的情况下能生成流畅的译文，但在剩余 20% 的情况下会静默失败。已知的故障模式包括：

- **幻觉 (Hallucination)**：模型凭空生成原文中不存在的内容。常见于不熟悉的领域词汇。症状：输出流畅，但陈述了原文未提及的事实。缓解措施：对领域术语使用受限解码 (Constrained Decoding)，对受监管内容进行人工审核，监控输出长度显著超过输入的情况。
- **偏离目标语言 (Off-target Generation)**：模型翻译成了错误的语言。NLLB 在罕见语言对上极易出现此问题。缓解措施：仔细核对 `forced_bos_token_id`，并在解码后始终使用语言识别 (Language-ID) 模型对输出进行校验。
- **术语漂移 (Terminology Drift)**：例如“Sign up”在文档 1 中译为“s'inscrire”，在文档 2 中却译为“créer un compte”。对于界面文本和面向用户的字符串，一致性比绝对质量更重要。缓解措施：使用术语表约束解码或后处理词典。
- **语体不匹配 (Formality Mismatch)**：例如法语中的“tu”（你）与“vous”（您），或日语的敬语等级。模型通常会选择训练数据中更常见的形式。对于面向客户的内容，这通常是错误的。缓解措施：如果模型支持，可在提示词前缀添加语体控制标记 (Formality Token)，或仅在正式语料库上微调一个小模型。
- **短输入长度爆炸 (Length Explosion on Short Input)**：极短的输入句子经常生成过长的译文，因为当源文本 token 数低于约 5 个时，长度惩罚 (Length Penalty) 会急剧失效。缓解措施：设置与源文本长度成比例的硬性最大长度上限。

### 步骤 4：针对特定领域进行微调

预训练模型属于通用型。在法律、医疗或游戏对话翻译中，使用领域平行语料 (Parallel Data) 进行微调能带来显著的质量提升。具体流程并不复杂：

from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()

几千条高质量的平行语料样本，远胜于几十万条充满噪声的网络抓取数据。训练数据的质量是生产环境中影响效果最关键的杠杆。

## 使用指南

2026 年机器翻译（Machine Translation, MT）的生产级技术栈如下：

| 应用场景 | 推荐起点 |
|---------|---------------------------|
| 任意语言互译，支持 200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本电脑）或 `nllb-200-3.3B`（生产环境） |
| 以英语为中心，高质量，支持 50 种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短文本运行，低成本推理，英法/英德/英西互译 | Helsinki-NLP / Marian 模型 |
| 对延迟敏感的浏览器端应用 | ONNX 量化（ONNX-quantized）版 Marian（约 50 MB） |
| 追求最高质量，预算充足 | 配合翻译提示词（translation prompts）使用的 GPT-4 / Claude / Gemini |

截至 2026 年，大语言模型（Large Language Models, LLMs）在多个语言对上已超越专用机器翻译模型，尤其在习语（idiomatic content）和长上下文（long context）处理方面表现更佳。其代价在于单词元（token）成本和延迟。当上下文长度、风格一致性或通过提示词进行领域适配（domain adaptation）的重要性超过吞吐量（throughput）时，应优先选择 LLM。

## 部署上线

保存为 `outputs/skill-mt-evaluator.md`：

---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.

## 练习

1. **简单。** 使用 `nllb-200-distilled-600M` 将一段包含 5 个句子的英文段落翻译为法文，再回译成英文。评估回译结果与原文的接近程度。你应该会观察到语义保留（semantic preservation）但选词出现漂移（word-choice drift）。
2. **中等。** 使用 `fasttext lid.176` 或 `langdetect` 对翻译输出实现语言识别（Language-ID）检查。将其集成到机器翻译调用流程中，以便在返回结果前拦截非目标生成（off-target generations）。
3. **困难。** 在你自选的包含 5000 个句对的领域语料库（domain corpus）上微调 `nllb-200-distilled-600M`。在微调前后，分别使用保留集（held-out set）测量 BLEU 分数。报告哪些类型的句子质量得到提升，哪些出现退化（regressed）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| BLEU | 翻译得分 | 带简短惩罚的 N 元语法（N-gram）精确度。取值范围为 [0, 100]。 |
| chrF | 字符级 F 分数 | 字符级别的 F 分数。对形态丰富的语言更为敏感。 |
| 神经机器翻译（Neural Machine Translation） | 神经机器翻译 | 在平行语料（Parallel Text）上训练的 Transformer 编码器-解码器（Encoder-Decoder）架构。2017 年以来的默认方案。 |
| 不让任何语言掉队（No Language Left Behind） | 不让任何语言掉队 | Meta 推出的支持 200 种语言的机器翻译模型系列。 |
| 约束解码（Constrained Decoding） | 受控输出 | 强制特定词元（Token）或 N 元语法在输出中出现或不出现。 |
| 幻觉（Hallucination） | 虚构内容 | 模型生成的、缺乏源文本依据的输出内容。 |

## 扩展阅读

- [Costa-jussà 等人（2022）。No Language Left Behind：扩展以人为本的机器翻译](https://arxiv.org/abs/2207.04672) — NLLB 的原始论文。
- [Post（2018）。呼吁清晰报告 BLEU 分数](https://aclanthology.org/W18-6319/) — 解释为何 `sacrebleu` 是报告 BLEU 分数的唯一正确方式。
- [Popović（2015）。chrF：用于自动机器翻译评估的字符级 N 元语法 F 分数](https://aclanthology.org/W15-3049/) — chrF 的原始论文。
- [Hugging Face 机器翻译指南](https://huggingface.co/docs/transformers/tasks/translation) — 实用的微调（Fine-tuning）操作指南。