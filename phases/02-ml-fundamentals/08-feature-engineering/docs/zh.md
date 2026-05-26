# 特征工程与特征选择

> 一个好的特征胜过千条数据。

**类型：** 构建
**语言：** Python
**前置要求：** 第一阶段（机器学习统计学、线性代数），第二阶段第 1-7 课
**预计时长：** 约 90 分钟

## 学习目标

- 实现数值变换（标准化 (Standardization)、最小-最大缩放 (Min-Max Scaling)、对数变换 (Log Transform)、分箱 (Binning)），并解释各自适用的场景
- 为类别特征构建独热编码 (One-Hot Encoding)、标签编码 (Label Encoding) 和目标编码 (Target Encoding)，并识别目标编码中的数据泄露 (Data Leakage) 风险
- 从零构建词频-逆文档频率 (TF-IDF) 向量化器，并解释为何它在文本分类任务中优于原始词频统计
- 应用基于过滤器的特征选择方法（方差阈值 (Variance Threshold)、相关性 (Correlation)、互信息 (Mutual Information)）来降低维度

## 问题背景

你拿到一个数据集，挑选了一个算法，进行训练，结果却平平无奇。你换了一个更复杂的算法，依然如此。你花了一周时间调优超参数 (Hyperparameters)，提升却微乎其微。

随后，有人将原始数据转换成了更优质的特征，结果一个简单的逻辑回归 (Logistic Regression) 模型就击败了你精心调优的梯度提升集成模型 (Gradient-Boosted Ensemble)。

这种情况屡见不鲜。在传统机器学习 (Classical ML) 中，数据的表示方式比算法的选择更为重要。无论学习器 (Learner) 多么复杂，使用“建筑面积”和“卧室数量”作为特征的房价预测模型，永远会胜过使用“原始字符串地址”的模型。算法只能基于你提供的数据进行工作。

特征工程 (Feature Engineering) 是将原始数据转换为更易于模型发现潜在模式的表示形式的过程。特征选择 (Feature Selection) 则是剔除那些只增加噪声而不提供有效信号的特征的过程。两者结合，是传统机器学习中投入产出比最高的核心工作。

## 核心概念

### 特征流水线 (Feature Pipeline)

flowchart LR
    A[Raw Data] --> B[Handle Missing Values]
    B --> C[Numerical Transforms]
    B --> D[Categorical Encoding]
    B --> E[Text Features]
    C --> F[Feature Interactions]
    D --> F
    E --> F
    F --> G[Feature Selection]
    G --> H[Model-Ready Data]

### 数值特征 (Numerical Features)

原始数值通常无法直接用于模型训练。常见的转换方法包括：

**缩放 (Scaling)：** 将特征调整到相同的数值范围，使基于距离的算法（如 K-Means、KNN、SVM）能够平等对待所有特征。最小-最大缩放（Min-max scaling）将数据映射到 [0, 1] 区间。标准化（Standardization，又称 z-score）将数据映射为均值（mean）为 0、标准差（std）为 1。

**对数变换 (Log transform)：** 压缩右偏分布（如收入、人口、词频）。将乘法关系转化为加法关系。

**分箱 (Binning)：** 将连续值转换为离散类别。当特征与目标变量之间的关系呈非线性但具有阶梯状特征时（例如年龄段划分）非常有用。

**多项式特征 (Polynomial features)：** 生成 x^2、x^3、x1*x2 等项。使线性模型能够捕捉非线性关系，但代价是特征数量增加。

### 类别特征 (Categorical Features)

模型需要数值输入，因此类别数据必须进行编码。

**独热编码 (One-hot encoding)：** 为每个类别创建一个二进制列。例如，“color = red/blue/green”会变为三列：is_red、is_blue、is_green。适用于基数（cardinality）较低的特征，但当类别过多时会导致维度爆炸。

**标签编码 (Label encoding)：** 将每个类别映射为一个整数：red=0, blue=1, green=2。这会引入虚假的顺序关系（模型可能会误认为 green > blue > red）。仅适用于基于树（tree-based）的模型，因为这类模型是基于单个值进行分裂的。

**目标编码 (Target encoding)：** 用该类别对应的目标变量均值替换类别本身。效果强大但存在风险：极易导致数据泄露（data leakage）。必须仅基于训练数据计算，并应用于测试数据。

### 文本特征 (Text Features)

**词频向量化 (Count vectorizer)：** 统计每个词在文档中出现的次数。例如，“the cat sat on the mat”会转换为 {the: 2, cat: 1, sat: 1, on: 1, mat: 1}。

**TF-IDF (Term Frequency-Inverse Document Frequency，词频-逆文档频率)：** 根据词语在文档集合中的独特性进行加权。像“the”这样的常见词权重较低，而罕见且具有区分度的词权重较高。

TF(word, doc) = count(word in doc) / total words in doc
IDF(word) = log(total docs / docs containing word)
TF-IDF = TF * IDF

### 缺失值 (Missing Values)

真实数据往往存在缺失。处理策略包括：

- **删除行 (Drop rows)：** 仅当缺失数据极少且为随机缺失时使用
- **均值/中位数插补 (Mean/median imputation)：** 方法简单，能保留分布形态（中位数对异常值更具鲁棒性）
- **众数插补 (Mode imputation)：** 适用于类别特征
- **指示列 (Indicator column)：** 在插补前添加一个二进制列（如 "was_this_missing"）。数据缺失这一事实本身可能包含重要信息
- **前向/后向填充 (Forward/backward fill)：** 适用于时间序列数据

### 特征交互 (Feature Interaction)

有时关键信息隐藏在特征的组合中。单独的“身高”和“体重”预测能力往往不如“BMI = 体重 / 身高^2”。特征交互会成倍增加特征空间，因此需要结合领域知识来选择合适的交互项。

### 特征选择 (Feature Selection)

特征并非越多越好。无关特征会引入噪声、增加训练时间，并可能导致过拟合（overfitting）。

**过滤法 (Filter methods，模型训练前)：**
- 相关性分析：移除彼此高度相关的特征（冗余特征）
- 互信息 (Mutual information)：衡量了解某个特征能在多大程度上降低对目标变量的不确定性
- 方差阈值 (Variance threshold)：移除变化极小的特征

**包装法 (Wrapper methods，基于模型)：**
- L1 正则化 (L1 regularization / Lasso)：将无关特征的权重压缩至精确为零
- 递归特征消除 (Recursive feature elimination)：训练模型，移除最不重要的特征，重复此过程

**为何特征选择至关重要：** 仅包含 10 个优质特征的模型，其表现通常优于包含 10 个优质特征加 90 个噪声特征的模型。噪声特征会让模型有机会在训练数据中过拟合那些无法泛化的模式。

## 构建

### 步骤 1：从零实现数值变换 (Numerical Transforms)

import math


def min_max_scale(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]


def standardize(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance) if variance > 0 else 1.0
    return [(v - mean) / std for v in values]


def log_transform(values):
    return [math.log(v + 1) for v in values]


def bin_values(values, n_bins=5):
    min_val = min(values)
    max_val = max(values)
    bin_width = (max_val - min_val) / n_bins
    if bin_width == 0:
        return [0] * len(values)
    result = []
    for v in values:
        bin_idx = int((v - min_val) / bin_width)
        bin_idx = min(bin_idx, n_bins - 1)
        result.append(bin_idx)
    return result


def polynomial_features(row, degree=2):
    n = len(row)
    result = list(row)
    if degree >= 2:
        for i in range(n):
            result.append(row[i] ** 2)
        for i in range(n):
            for j in range(i + 1, n):
                result.append(row[i] * row[j])
    return result

### 步骤 2：从零实现类别编码 (Categorical Encoding)

def one_hot_encode(values):
    categories = sorted(set(values))
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    n_cats = len(categories)

    encoded = []
    for v in values:
        row = [0] * n_cats
        row[cat_to_idx[v]] = 1
        encoded.append(row)

    return encoded, categories


def label_encode(values):
    categories = sorted(set(values))
    cat_to_int = {cat: i for i, cat in enumerate(categories)}
    return [cat_to_int[v] for v in values], cat_to_int


def target_encode(feature_values, target_values, smoothing=10):
    global_mean = sum(target_values) / len(target_values)

    category_stats = {}
    for feat, target in zip(feature_values, target_values):
        if feat not in category_stats:
            category_stats[feat] = {"sum": 0.0, "count": 0}
        category_stats[feat]["sum"] += target
        category_stats[feat]["count"] += 1

    encoding = {}
    for cat, stats in category_stats.items():
        cat_mean = stats["sum"] / stats["count"]
        weight = stats["count"] / (stats["count"] + smoothing)
        encoding[cat] = weight * cat_mean + (1 - weight) * global_mean

    return [encoding[v] for v in feature_values], encoding

### 步骤 3：从零实现文本特征 (Text Features)

def count_vectorize(documents):
    vocab = {}
    idx = 0
    for doc in documents:
        for word in doc.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    vectors = []
    for doc in documents:
        vec = [0] * len(vocab)
        for word in doc.lower().split():
            vec[vocab[word]] += 1
        vectors.append(vec)

    return vectors, vocab


def tfidf(documents):
    n_docs = len(documents)

    vocab = {}
    idx = 0
    for doc in documents:
        for word in doc.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    doc_freq = {}
    for doc in documents:
        seen = set()
        for word in doc.lower().split():
            if word not in seen:
                doc_freq[word] = doc_freq.get(word, 0) + 1
                seen.add(word)

    vectors = []
    for doc in documents:
        words = doc.lower().split()
        word_count = len(words)
        tf_map = {}
        for word in words:
            tf_map[word] = tf_map.get(word, 0) + 1

        vec = [0.0] * len(vocab)
        for word, count in tf_map.items():
            tf = count / word_count
            idf = math.log(n_docs / doc_freq[word])
            vec[vocab[word]] = tf * idf
        vectors.append(vec)

    return vectors, vocab

### 步骤 4：从零实现缺失值插补 (Missing Value Imputation)

def impute_mean(values):
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values), 0.0
    mean = sum(present) / len(present)
    return [v if v is not None else mean for v in values], mean


def impute_median(values):
    present = sorted(v for v in values if v is not None)
    if not present:
        return [0.0] * len(values), 0.0
    n = len(present)
    if n % 2 == 0:
        median = (present[n // 2 - 1] + present[n // 2]) / 2
    else:
        median = present[n // 2]
    return [v if v is not None else median for v in values], median


def impute_mode(values):
    present = [v for v in values if v is not None]
    if not present:
        return values, None
    counts = {}
    for v in present:
        counts[v] = counts.get(v, 0) + 1
    mode = max(counts, key=counts.get)
    return [v if v is not None else mode for v in values], mode


def add_missing_indicator(values):
    return [0 if v is not None else 1 for v in values]

### 步骤 5：从零实现特征选择 (Feature Selection)

def correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def mutual_information(feature, target, n_bins=10):
    feat_min = min(feature)
    feat_max = max(feature)
    bin_width = (feat_max - feat_min) / n_bins if feat_max != feat_min else 1.0
    feat_binned = [
        min(int((f - feat_min) / bin_width), n_bins - 1) for f in feature
    ]

    n = len(feature)
    target_classes = sorted(set(target))

    feat_bins = sorted(set(feat_binned))
    p_feat = {}
    for b in feat_bins:
        p_feat[b] = feat_binned.count(b) / n

    p_target = {}
    for t in target_classes:
        p_target[t] = target.count(t) / n

    mi = 0.0
    for b in feat_bins:
        for t in target_classes:
            joint_count = sum(
                1 for fb, tv in zip(feat_binned, target) if fb == b and tv == t
            )
            p_joint = joint_count / n
            if p_joint > 0:
                mi += p_joint * math.log(p_joint / (p_feat[b] * p_target[t]))

    return mi


def variance_threshold(features, threshold=0.01):
    n_features = len(features[0])
    n_samples = len(features)
    selected = []

    for j in range(n_features):
        col = [features[i][j] for i in range(n_samples)]
        mean = sum(col) / n_samples
        var = sum((v - mean) ** 2 for v in col) / n_samples
        if var >= threshold:
            selected.append(j)

    return selected


def remove_correlated(features, threshold=0.9):
    n_features = len(features[0])
    n_samples = len(features)

    to_remove = set()
    for i in range(n_features):
        if i in to_remove:
            continue
        col_i = [features[r][i] for r in range(n_samples)]
        for j in range(i + 1, n_features):
            if j in to_remove:
                continue
            col_j = [features[r][j] for r in range(n_samples)]
            corr = abs(correlation(col_i, col_j))
            if corr >= threshold:
                to_remove.add(j)

    return [i for i in range(n_features) if i not in to_remove]

### 步骤 6：完整流水线与演示 (Full Pipeline and Demo)

import random


def make_housing_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        sqft = random.uniform(500, 5000)
        bedrooms = random.choice([1, 2, 3, 4, 5])
        age = random.uniform(0, 50)
        neighborhood = random.choice(["downtown", "suburbs", "rural"])
        has_pool = random.choice([True, False])

        sqft_with_missing = sqft if random.random() > 0.05 else None
        age_with_missing = age if random.random() > 0.08 else None

        price = (
            50 * sqft
            + 20000 * bedrooms
            - 1000 * age
            + (50000 if neighborhood == "downtown" else 10000 if neighborhood == "suburbs" else 0)
            + (15000 if has_pool else 0)
            + random.gauss(0, 20000)
        )

        data.append({
            "sqft": sqft_with_missing,
            "bedrooms": bedrooms,
            "age": age_with_missing,
            "neighborhood": neighborhood,
            "has_pool": has_pool,
            "price": price,
        })
    return data


if __name__ == "__main__":
    data = make_housing_data(200)

    print("=== Raw Data Sample ===")
    for row in data[:3]:
        print(f"  {row}")

    sqft_raw = [d["sqft"] for d in data]
    age_raw = [d["age"] for d in data]
    prices = [d["price"] for d in data]

    print("\n=== Missing Value Handling ===")
    sqft_missing = sum(1 for v in sqft_raw if v is None)
    age_missing = sum(1 for v in age_raw if v is None)
    print(f"  sqft missing: {sqft_missing}/{len(sqft_raw)}")
    print(f"  age missing: {age_missing}/{len(age_raw)}")

    sqft_indicator = add_missing_indicator(sqft_raw)
    age_indicator = add_missing_indicator(age_raw)
    sqft_imputed, sqft_fill = impute_median(sqft_raw)
    age_imputed, age_fill = impute_mean(age_raw)
    print(f"  sqft filled with median: {sqft_fill:.0f}")
    print(f"  age filled with mean: {age_fill:.1f}")

    print("\n=== Numerical Transforms ===")
    sqft_scaled = standardize(sqft_imputed)
    age_scaled = min_max_scale(age_imputed)
    sqft_log = log_transform(sqft_imputed)
    age_binned = bin_values(age_imputed, n_bins=5)
    print(f"  sqft standardized: mean={sum(sqft_scaled)/len(sqft_scaled):.4f}, std={math.sqrt(sum(v**2 for v in sqft_scaled)/len(sqft_scaled)):.4f}")
    print(f"  age min-max: [{min(age_scaled):.2f}, {max(age_scaled):.2f}]")
    print(f"  age bins: {sorted(set(age_binned))}")

    print("\n=== Categorical Encoding ===")
    neighborhoods = [d["neighborhood"] for d in data]

    ohe, ohe_cats = one_hot_encode(neighborhoods)
    print(f"  One-hot categories: {ohe_cats}")
    print(f"  Sample encoding: {neighborhoods[0]} -> {ohe[0]}")

    le, le_map = label_encode(neighborhoods)
    print(f"  Label encoding map: {le_map}")

    te, te_map = target_encode(neighborhoods, prices, smoothing=10)
    print(f"  Target encoding: {({k: round(v) for k, v in te_map.items()})}")

    print("\n=== Text Features ===")
    descriptions = [
        "large modern house with pool",
        "small cozy cottage near downtown",
        "spacious family home with large yard",
        "modern apartment downtown with view",
        "rustic cabin in rural area",
    ]
    cv, cv_vocab = count_vectorize(descriptions)
    print(f"  Vocabulary size: {len(cv_vocab)}")
    print(f"  Doc 0 non-zero features: {sum(1 for v in cv[0] if v > 0)}")

    tf, tf_vocab = tfidf(descriptions)
    print(f"  TF-IDF vocabulary size: {len(tf_vocab)}")
    top_words = sorted(tf_vocab.keys(), key=lambda w: tf[0][tf_vocab[w]], reverse=True)[:3]
    print(f"  Doc 0 top TF-IDF words: {top_words}")

    print("\n=== Polynomial Features ===")
    sample_row = [sqft_scaled[0], age_scaled[0]]
    poly = polynomial_features(sample_row, degree=2)
    print(f"  Input: {[round(v, 4) for v in sample_row]}")
    print(f"  Polynomial: {[round(v, 4) for v in poly]}")
    print(f"  Features: [x1, x2, x1^2, x2^2, x1*x2]")

    print("\n=== Feature Selection ===")
    feature_matrix = [
        [sqft_scaled[i], age_scaled[i], float(sqft_indicator[i]), float(age_indicator[i])]
        + ohe[i]
        for i in range(len(data))
    ]

    print(f"  Total features: {len(feature_matrix[0])}")

    surviving_var = variance_threshold(feature_matrix, threshold=0.01)
    print(f"  After variance threshold (0.01): {len(surviving_var)} features kept")

    surviving_corr = remove_correlated(feature_matrix, threshold=0.9)
    print(f"  After correlation filter (0.9): {len(surviving_corr)} features kept")

    binary_prices = [1 if p > sum(prices) / len(prices) else 0 for p in prices]
    print("\n  Mutual information with target:")
    feature_names = ["sqft", "age", "sqft_missing", "age_missing"] + [f"neigh_{c}" for c in ohe_cats]
    for j in range(len(feature_matrix[0])):
        col = [feature_matrix[i][j] for i in range(len(feature_matrix))]
        mi = mutual_information(col, binary_prices, n_bins=10)
        print(f"    {feature_names[j]}: MI={mi:.4f}")

    print("\n  Correlation with price:")
    for j in range(len(feature_matrix[0])):
        col = [feature_matrix[i][j] for i in range(len(feature_matrix))]
        corr = correlation(col, prices)
        print(f"    {feature_names[j]}: r={corr:.4f}")


## 实际应用

在 scikit-learn 中，这些转换 (transforms) 可以组合成流水线 (pipelines)：

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("encoder", OneHotEncoder(sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["sqft", "age"]),
    ("cat", categorical_pipe, ["neighborhood"]),
])

从零实现 (from-scratch) 的版本清晰地展示了每个转换内部的具体运作过程。而库提供的版本则增加了边界情况处理 (edge-case handling)、稀疏矩阵 (sparse matrix) 支持以及流水线组合 (pipeline composition) 功能，但底层数学原理是相同的。

## 交付成果

本课程的产出包括：
- `outputs/prompt-feature-engineer.md` - 用于从原始数据中系统化进行特征工程 (feature engineering) 的提示词 (prompt)

## 练习

1. 为数值型转换添加稳健缩放 (robust scaling，使用中位数和四分位距 (interquartile range) 代替均值和标准差)。在包含极端异常值 (outliers) 的数据上，将其与标准缩放 (standard scaling) 进行对比。
2. 实现留一法目标编码 (leave-one-out target encoding)：针对每一行数据，在计算目标均值时排除该行自身的目标值。展示与朴素目标编码 (naive target encoding) 相比，该方法如何降低过拟合 (overfitting) 风险。
3. 构建一个自动化特征选择 (feature selection) 流水线，结合方差阈值 (variance threshold)、相关性过滤 (correlation filtering) 和互信息排序 (mutual information ranking)。将其应用于房屋数据集，并对比使用全部特征与仅使用筛选后特征时的模型性能（使用简单的线性回归 (linear regression)）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 特征工程 (Feature Engineering) | “创建新列” | 将原始数据转换为能够向模型揭示潜在模式的表示形式 |
| 标准化 (Standardization) | “让它变正态” | 减去均值并除以标准差，使特征的均值为 0、标准差为 1 |
| 独热编码 (One-hot Encoding) | “生成虚拟变量” | 为每个类别创建一个二进制列，使得每一行中恰好有一列的值为 1 |
| 目标编码 (Target Encoding) | “用答案来编码” | 用该类别对应的目标变量平均值替换每个类别，并加入平滑处理以防止过拟合 |
| 词频-逆文档频率 (TF-IDF) | “高级词频统计” | 词频乘以逆文档频率：根据词语在整个语料库中的区分度对其进行加权 |
| 缺失值填充 (Imputation) | “填补空白” | 用估算值（均值、中位数、众数或模型预测值）替换缺失值 |
| 特征选择 (Feature Selection) | “剔除坏列” | 移除引入噪声或冗余的特征，仅保留对目标变量具有预测信号的特征 |
| 互信息 (Mutual Information) | “一件事能透露多少关于另一件事的信息” | 衡量通过观测变量 X 所能获得的关于变量 Y 的不确定性减少程度 |
| 数据泄露 (Data Leakage) | “意外作弊” | 在训练过程中使用了预测时无法获取的信息，从而导致结果虚假乐观 |

## 扩展阅读

- [《特征工程与选择》（Max Kuhn & Kjell Johnson）](http://www.feat.engineering/) - 免费在线书籍，全面涵盖特征工程的各个领域
- [scikit-learn 预处理指南](https://scikit-learn.org/stable/modules/preprocessing.html) - 所有标准数据转换方法的实用参考
- [《正确实施目标编码》（Micci-Barreca, 2001）](https://dl.acm.org/doi/10.1145/507533.507538) - 提出带平滑处理的目标编码方法的原始论文