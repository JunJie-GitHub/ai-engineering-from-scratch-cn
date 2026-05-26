# 情感分析 (Sentiment Analysis)

> 自然语言处理 (NLP) 领域的标杆任务。关于传统文本分类 (Text Classification) 所需掌握的核心知识，大多在此体现。

**类型：** 构建实践
**语言：** Python
**前置知识：** 第 5 阶段 · 02（词袋模型 (BoW) + 词频-逆文档频率 (TF-IDF)），第 2 阶段 · 14（朴素贝叶斯 (Naive Bayes)）
**预计耗时：** 约 75 分钟

## 问题描述

“这食物不怎么样。”是正面还是负面？

情感分析听起来很简单。评论者表达了喜欢或不喜欢，只需给句子打上标签即可。它之所以成为 NLP 的标杆任务，是因为每一个看似简单的案例背后都隐藏着复杂的陷阱。否定词会反转语义，反讽会彻底颠覆原意。“一点也不差”尽管包含两个负面词汇，实际却是正面评价。表情符号所携带的信号往往比周围文本更强。领域词汇至关重要（例如音乐评论中的 `tight` 与时尚评论中的 `tight` 含义截然不同）。

情感分析是传统 NLP 的“试验田”。如果你能理解为何每个简单的基线模型 (Baseline) 都有特定的失效模式，你就能明白为何要发明更复杂的模型。本课程将从零开始构建一个朴素贝叶斯基线，引入逻辑回归 (Logistic Regression)，并剖析那些让生产环境中的情感分析成为合规级难题的陷阱。

## 核心概念

传统情感分析通常遵循两步流程。

1. **特征表示 (Represent)。** 将文本转换为特征向量。可使用词袋模型、词频-逆文档频率或 n-gram (n-grams)。
2. **分类 (Classify)。** 在带标签的样本上拟合线性模型（如朴素贝叶斯、逻辑回归或支持向量机 (SVM)）。

朴素贝叶斯是“最笨但有效”的模型。它假设在给定标签的条件下，所有特征相互独立。通过统计频次来估计 `P(word | positive)` 和 `P(word | negative)`。在推理阶段，将这些概率相乘即可。这种“朴素”的独立性假设虽然荒谬，但效果却出奇地好。原因在于：面对稀疏的文本特征和中等规模的数据集，分类器更关注每个词倾向于哪一类，而非其具体权重有多大。

逻辑回归修正了独立性假设。它会为每个特征学习一个权重，包括负权重。例如，将 `not good` 作为二元语法 (bigram) 特征时，会获得一个负权重。而对于从未在训练集中出现过的二元语法，朴素贝叶斯则无法做到这一点。

## 动手实践

### 步骤 1：构建一个真实的微型数据集

POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]

数据量小是刻意为之。实际工作中通常会使用数万条样本（如 IMDb、SST-2、Yelp Polarity 数据集）。但背后的数学原理是完全相同的。

### 步骤 2：从零实现多项式朴素贝叶斯（Multinomial Naive Bayes）

import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)

加性平滑（Additive Smoothing，此处 `alpha=1.0`）即拉普拉斯平滑（Laplace Smoothing）。如果不使用它，模型中未在某个类别出现过的词概率将为零，导致对数运算结果趋向负无穷（数值爆炸）。在实际应用中，`alpha=0.01` 更为常见，而 `alpha=1.0` 是教学场景下的默认值。

### 步骤 3：从零实现逻辑回归（Logistic Regression）

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)

这里的 L2 正则化（L2 Regularization）至关重要。文本特征通常非常稀疏；若不引入 L2 正则化，模型极易死记硬背训练样本（过拟合）。建议从 `0.01` 开始尝试并进行调优。

### 步骤 4：处理否定词（常见失效模式）

以“not good”（不好）和“not bad”（不错）为例。词袋模型（Bag-of-Words, BoW）分类器只会看到 `{not, good}` 和 `{not, bad}`，并根据训练集中哪个组合出现得更频繁来进行学习。而二元语法（Bigram）分类器则会将 `not_good` 和 `not_bad` 视为独立的特征进行学习。这通常已经足够解决问题。

如果你没有使用二元语法，还有一个更简单粗暴但有效的修复方法：**否定词作用域（Negation Scoping）**。在否定词之后、下一个标点符号之前的所有词元（Token）前加上 `NOT_` 前缀。

NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out

>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']

现在，`good` 和 `NOT_good` 变成了两个不同的特征，分类器可以为它们分配相反的权重。仅需三行预处理代码，就能在情感分析基准测试中带来可衡量的准确率提升。

### 步骤 5：关键评估指标

如果类别不平衡，仅看准确率（Accuracy）会产生误导。真实的情感语料库通常包含 70%~80% 的正向样本或 70%~80% 的负向样本；此时一个始终预测多数类的分类器也能达到 80% 的准确率，但毫无价值。请务必报告以下所有指标：

- **各类别的精确率（Precision）与召回率（Recall）**。每个类别对应一对数值。对它们进行宏平均（Macro-average），即可得到一个兼顾类别平衡的单一指标。
- **宏平均 F1 分数（Macro-F1，不平衡数据的首选指标）**。各类别 F1 分数的算术平均值，权重相等。当类别不平衡时，应使用它替代准确率。
- **加权 F1 分数（Weighted-F1，备选指标）**。计算方式与宏平均相同，但按类别频率进行加权。当类别不平衡本身具有业务意义时，建议与宏平均 F1 分数一同报告。
- **混淆矩阵（Confusion Matrix）**。原始计数。在信任任何标量指标前务必先检查它，它能直观揭示模型容易混淆哪两个类别。
- **各类别的错误样本**。从每个类别中抽取 5 个预测错误的样本。仔细阅读它们。没有任何方法能替代直接查看实际错误。

对于极度不平衡的数据（比例超过 95:5），应报告 **AUROC**（受试者工作特征曲线下面积）和 **AUPRC**（精确率-召回率曲线下面积）来替代准确率。AUPRC 对少数类更为敏感，而这通常正是你真正关心的场景（如垃圾邮件、欺诈检测、罕见情感倾向）。

**常见避坑指南**。在不平衡数据上报告微平均 F1（Micro-F1）而非宏平均 F1 时，得到的数值会虚高，因为它被多数类主导了。宏平均 F1 能强制你关注少数类的表现。

def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


## 使用它

使用 Scikit-learn (scikit-learn) 只需六行代码即可正确实现。

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))

需要注意三点。`stop_words=None` 会保留否定词 (negations)。`ngram_range=(1, 2)` 会添加二元语法 (bigrams)，使 `not_good` 成为一个特征。`sublinear_tf=True` 会削弱重复词的影响。在 SST-2 数据集上，这三个参数正是 75% 准确率基线 (baseline) 与 85% 准确率基线之间的关键差异。

### 何时使用 Transformer

- 讽刺检测 (sarcasm detection)。传统模型在此完全失效。毫无疑问。
- 情感在文档中途发生转变的长评论。
- 基于方面的情感分析 (aspect-based sentiment analysis)。“相机很棒，但电池糟透了。”你需要将情感归因到具体方面。仅能使用 Transformer 或结构化输出模型 (structured output models)。
- 非英语的低资源语言 (low-resource languages)。多语言 BERT (Multilingual BERT) 可免费提供零样本 (zero-shot) 基线。

如果你需要处理上述任何情况，请直接跳至第 7 阶段（Transformer 深度解析）。否则，基于 TF-IDF (词频-逆文档频率) 加上二元语法和否定词处理的朴素贝叶斯 (Naive Bayes) 或逻辑回归 (logistic regression)，就是你 2026 年的生产环境基线。

### 可复现性陷阱 (reproducibility trap)（再次强调）

重新训练情感分析模型是常规操作，但重新评估它们却并非如此。论文中报告的准确率数字使用的是特定的数据划分 (splits)、特定的预处理 (preprocessing) 流程和特定的分词器 (tokenizers)。如果你在不使用完全相同流水线 (pipeline) 的情况下将新模型与基线进行比较，将会得到具有误导性的差异值 (deltas)。务必在你的流水线上重新生成基线结果，而不是直接引用论文中的数字。

## 交付上线

保存为 `outputs/prompt-sentiment-baseline.md`：

---
name: 情感分析基线
description: 为新数据集设计情感分析基线。
phase: 5
lesson: 05
---

给定数据集描述（领域、语言、规模、标签粒度、延迟预算），你需要输出：

1. 特征提取方案。指定分词器 (tokenizer)、n-gram 范围、停用词策略 (stopword policy)（通常保留）、否定词处理 (negation handling)（限定前缀 (scoped prefix) 或二元语法）。
2. 分类器。基线使用朴素贝叶斯，生产环境使用逻辑回归，仅当领域需要处理讽刺/方面/跨语言时才使用 Transformer。
3. 评估计划。报告精确率 (precision)、召回率 (recall)、F1 分数、混淆矩阵 (confusion matrix) 以及各类别错误样本 (per-class error samples)（而不仅仅是标量指标）。
4. 部署后需监控的一种失败模式 (failure mode)。领域漂移 (domain drift) 和讽刺是排名前两位的。

拒绝为情感分析任务推荐去除停用词。当类别不平衡时（例如 90% 为正类），拒绝仅报告准确率作为唯一指标。标记富含子词的语言 (subword-rich languages) 为需要 FastText 或 Transformer 嵌入 (transformer embeddings)，而非基于词级别的 TF-IDF (word-level TF-IDF)。

## 练习

1. **简单。** 将 `apply_negation` 作为预处理步骤（preprocessing step）添加到 scikit-learn 管道（pipeline）中，并在一个小型情感分析数据集上测量 F1 分数变化量（F1 delta）。
2. **中等。** 实现类别加权逻辑回归（class-weighted logistic regression）（向 scikit-learn 传入 `class_weight="balanced"` 参数，或自行推导梯度）。在合成的 90-10 类别不平衡（class imbalance）数据集上评估其效果。
3. **困难。** 通过在情感分析模型的残差（residuals）上训练第二个分类器来构建讽刺检测器（sarcasm detector）。详细记录实验设置。当模型准确率低于随机基准（chance-level）时，务必向读者发出警告（二分类讽刺检测的随机基准约为 50%，大多数初次尝试的结果都会停留在此处）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 极性（Polarity） | 正面或负面 | 二元标签；有时扩展为中性或细粒度（如 5 星评分）。 |
| 基于方面的情感分析（Aspect-based sentiment） | 每个方面的极性 | 将情感倾向归因于文本中提及的特定实体或属性。 |
| 否定作用域（Negation scoping） | 反转附近词元 | 在 "not" 之后的词元前添加 `NOT_` 前缀，直到遇到标点符号为止。 |
| 拉普拉斯平滑（Laplace smoothing） | 给计数加 1 | 防止朴素贝叶斯（Naive Bayes）中出现零概率特征。 |
| L2 正则化（L2 regularization） | 缩小权重 | 在损失函数中添加 `lambda * sum(w^2)` 项。对稀疏文本特征至关重要。 |

## 延伸阅读

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) —— 奠基性综述。篇幅较长，但前四节涵盖了所有经典方法。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) —— 该论文证明了在短文本任务中，二元语法（bigrams）结合朴素贝叶斯（Naive Bayes）的基线模型极难被超越。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) —— `CountVectorizer`、`TfidfVectorizer` 以及你将调整的所有可调参数（knob）的参考文档。