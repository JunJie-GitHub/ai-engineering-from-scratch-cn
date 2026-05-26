# 机器学习管道 (ML Pipelines)

> 模型本身不是产品，管道才是。管道涵盖了从原始数据到部署预测的完整流程，且其中的每一步都必须具备可复现性。

**类型：** 构建
**语言：** Python
**前置条件：** 第二阶段，第 12 课（超参数调优 (Hyperparameter Tuning)）
**时长：** 约 120 分钟

## 学习目标

- 从零开始构建一个机器学习管道 (ML Pipeline)，将缺失值填充 (Imputation)、特征缩放 (Scaling)、特征编码 (Encoding) 和模型训练串联为单一的可复现对象
- 识别数据泄露 (Data Leakage) 场景，并解释管道如何仅使用训练数据拟合转换器 (Transformer) 来防止此类问题
- 构建列转换器 (ColumnTransformer)，对数值型特征和类别型特征应用不同的预处理流程
- 实现管道序列化 (Pipeline Serialization)，并证明同一个已拟合的管道在训练和生产环境中能产生完全一致的结果

## 问题描述

你有一个 Notebook，它负责加载数据、用中位数填充缺失值、缩放特征、训练模型并输出准确率。代码能跑通，于是你将其部署上线。

一个月后，有人重新训练模型，却得到了不同的结果。原因在于：中位数是基于包含测试数据在内的完整数据集计算的（数据泄露 (Data Leakage)）；缩放参数未被保存，导致推理时使用了不同的统计量；训练和部署服务之间的特征工程代码是通过复制粘贴实现的，且两份代码逐渐产生了分歧；此外，生产环境中的某个类别型特征出现了编码器从未见过的新值。

这些并非假设，而是机器学习系统在生产环境中失败的最常见原因。管道通过将每一个转换步骤封装为单一、有序且可复现的对象，彻底解决了上述所有问题。

## 核心概念

### 什么是流水线（Pipeline）

流水线（Pipeline）是一系列按顺序执行的数据转换步骤，最后接一个模型。每个步骤都将前一步的输出作为输入。整个流水线仅在训练数据上拟合（fit）一次。在推理（inference）阶段，同一个已拟合的流水线会对新数据进行转换并生成预测结果。

flowchart LR
    A[Raw Data] --> B[Impute Missing Values]
    B --> C[Scale Numeric Features]
    C --> D[Encode Categoricals]
    D --> E[Train Model]
    E --> F[Prediction]

流水线能够保证：
- 转换步骤仅在训练数据上进行拟合（无数据泄露（Data Leakage））
- 推理时应用完全相同的转换逻辑
- 整个对象可被序列化并作为单一构件（Artifact）进行部署
- 交叉验证（Cross-validation）会在每个折（Fold）上独立应用流水线，防止隐蔽的数据泄露

### 数据泄露（Data Leakage）：沉默的杀手

当测试集或未来数据的信息污染了训练过程时，就会发生数据泄露。流水线能有效防止最常见的泄露形式。

**存在泄露（错误做法）：**
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test = X_scaled[:800], X_scaled[800:]
y_train, y_test = y[:800], y[800:]

缩放器（Scaler）“看到”了测试数据。计算出的均值和标准差包含了测试样本。这会虚高准确率评估结果。

**正确做法：**
X_train, X_test = X[:800], X[800:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

使用流水线后，你无需再操心这个问题。流水线会自动处理。

### scikit-learn 流水线（sklearn Pipeline）

scikit-learn 的 `Pipeline` 将转换器（Transformers）和估计器（Estimator）串联起来。它对外暴露了 `.fit()`、`.predict()` 和 `.score()` 方法，会按顺序执行所有步骤。

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)

当你调用 `pipe.fit(X_train, y_train)` 时：
1. 缩放器在 `X_train` 上调用 `fit_transform`
2. 模型在缩放后的 `X_train` 上调用 `fit`

当你调用 `pipe.predict(X_test)` 时：
1. 缩放器在 `X_test` 上调用 `transform`（而非 `fit_transform`）
2. 模型在缩放后的 `X_test` 上调用 `predict`

缩放器在拟合过程中永远不会接触到测试数据。这正是使用流水线的核心意义所在。

### ColumnTransformer：为不同列构建不同的流水线

真实数据集通常包含数值型和类别型列，它们需要不同的预处理方式。`ColumnTransformer` 专门用于处理这种情况。

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["age", "income", "score"]),
    ("cat", categorical_pipe, ["city", "gender", "plan"]),
])

full_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", GradientBoostingClassifier()),
])

`OneHotEncoder` 中的 `handle_unknown="ignore"` 参数对生产环境至关重要。当出现新类别（模型从未见过的城市）时，它会生成一个全零向量，而不是直接崩溃。

### 实验追踪（Experiment Tracking）

流水线让训练过程具备可复现性，但你还需要追踪各次实验的具体情况：使用了哪些超参数（Hyperparameters）、哪个版本的数据集、指标结果如何、运行的是哪版代码。

**MLflow** 是最常用的开源解决方案：

import mlflow

with mlflow.start_run():
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)

    pipe.fit(X_train, y_train)
    accuracy = pipe.score(X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(pipe, "model")

每次运行（Run）都会记录参数、指标、模型构件（Artifacts）以及完整模型。你可以对比不同运行记录、复现任意实验，并部署任意版本的模型。

**Weights & Biases (wandb)** 提供了相同的功能，并附带托管的可视化仪表盘（Dashboard）：

import wandb

wandb.init(project="my-pipeline")
wandb.config.update({"max_depth": 5, "n_estimators": 100})

pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)

wandb.log({"accuracy": accuracy})

### 模型版本控制（Model Versioning）

完成实验追踪后，你需要管理模型版本。哪个模型正在生产环境运行？哪个处于预发环境（Staging）？哪个是上周的版本？

MLflow 的模型注册表（Model Registry）提供以下功能：
- **版本追踪：** 每个保存的模型都会获得一个版本号
- **阶段流转：** 支持“预发（Staging）”、“生产（Production）”、“归档（Archived）”等状态切换
- **审批工作流：** 模型必须经过明确审批才能提升至生产环境
- **回滚（Rollback）：** 可瞬间切换回之前的版本

### 使用 DVC 进行数据版本控制

代码通常使用 git 进行版本控制。数据同样需要版本管理，但 git 无法处理大文件。DVC（Data Version Control）正是为了解决这一问题而生。

dvc init
dvc add data/training.csv
git add data/training.csv.dvc data/.gitignore
git commit -m "Track training data"
dvc push

DVC 将实际数据存储在远程存储（如 S3、GCS、Azure）中，并在 git 中保留一个记录哈希值的小型 `.dvc` 文件。当你检出（checkout）某个 git 提交时，执行 `dvc checkout` 即可恢复当时使用的确切数据。

这意味着每次 git 提交都能同时锁定代码和数据，从而实现完全的可复现性。

### 可复现的实验

一个可复现的实验需要满足四个条件：

1. **固定随机种子（Random Seeds）：** 为 numpy、random 以及所用框架（如 torch、sklearn）设置种子
2. **锁定依赖：** 使用 requirements.txt 或 poetry.lock 固定精确的版本号
3. **版本化数据：** 使用 DVC 或类似工具
4. **配置文件：** 将所有超参数写入配置文件，而非硬编码

import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

### 从 Notebook 到生产级流水线

flowchart TD
    A[Jupyter Notebook] --> B[Extract functions]
    B --> C[Build Pipeline object]
    C --> D[Add config file for hyperparameters]
    D --> E[Add experiment tracking]
    E --> F[Add data validation]
    F --> G[Add tests]
    G --> H[Package for deployment]

    style A fill:#fdd,stroke:#333
    style H fill:#dfd,stroke:#333

典型的演进路径如下：

1. **Notebook 探索：** 快速实验、可视化、构思特征
2. **提取函数：** 将预处理、特征工程、评估逻辑移至独立模块
3. **构建流水线：** 将转换步骤串联为 sklearn Pipeline 或自定义类
4. **配置管理：** 将所有超参数迁移至 YAML/JSON 配置文件
5. **实验追踪：** 接入 MLflow 或 wandb 日志记录
6. **数据验证：** 训练前检查数据模式（Schema）、分布及缺失值规律
7. **测试：** 为转换器编写单元测试，为完整流水线编写集成测试
8. **部署：** 序列化流水线，封装为 API（如 FastAPI、Flask），并进行容器化

### 常见的流水线错误

| 错误做法 | 为何有害 | 修复方案 |
|---------|-------------|-----|
| 在划分数据集前对全量数据进行拟合 | 导致数据泄露 | 结合 `cross_val_score` 使用 Pipeline |
| 在流水线外进行特征工程 | 训练与服务时的转换逻辑不一致 | 将所有转换步骤纳入 Pipeline |
| 未处理未知类别 | 遇到新值时生产环境崩溃 | `OneHotEncoder(handle_unknown="ignore")` |
| 硬编码列名 | 数据结构变更时程序中断 | 从配置文件中读取列名列表 |
| 缺乏数据验证 | 劣质数据会导致静默的错误预测 | 在预测前增加模式（Schema）检查 |
| 训练/服务偏差（Training/Serving Skew） | 模型在生产环境中看到的特征与训练时不同 | 训练与服务共用同一个 Pipeline 对象 |

## 动手构建

`code/pipeline.py` 中的代码从零开始构建了一个完整的机器学习 (Machine Learning, ML) 流水线 (Pipeline)：

### 步骤 1：自定义转换器 (Custom Transformer)

class CustomTransformer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.means) / self.stds

    def fit_transform(self, X):
        return self.fit(X).transform(X)

### 步骤 2：从零构建流水线 (Pipeline from Scratch)

class PipelineFromScratch:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        X_current = X.copy()
        for name, step in self.steps[:-1]:
            X_current = step.fit_transform(X_current)
        name, model = self.steps[-1]
        model.fit(X_current, y)
        return self

    def predict(self, X):
        X_current = X.copy()
        for name, step in self.steps[:-1]:
            X_current = step.transform(X_current)
        name, model = self.steps[-1]
        return model.predict(X_current)

### 步骤 3：结合流水线的交叉验证 (Cross-Validation)

该代码演示了结合流水线的交叉验证如何防止数据泄露 (Data Leakage)：缩放器 (Scaler) 会在每个折 (Fold) 的训练数据上独立进行拟合 (Fit)。

### 步骤 4：基于 sklearn 的完整生产级流水线

一个包含 `ColumnTransformer`、多条预处理路径和模型的完整流水线，采用规范的交叉验证和实验日志记录进行训练。

## 交付部署

本章节将生成：
- `outputs/prompt-ml-pipeline.md` -- 用于构建和调试机器学习流水线的技能文件
- `code/pipeline.py` -- 从零开始到基于 sklearn 的完整流水线代码

## 练习

1. 构建一个处理包含 3 个数值型列和 2 个类别型列的数据集的流水线。使用 `ColumnTransformer` 对数值型特征应用中位数插补 (Median Imputation) + 缩放 (Scaling)，对类别型特征应用众数插补 (Most-Frequent Imputation) + 独热编码 (One-Hot Encoding)。使用 5 折交叉验证 (5-Fold Cross-Validation) 进行训练。

2. 故意引入数据泄露：在划分数据集之前，使用完整数据集拟合缩放器。对比存在泄露的交叉验证得分与使用流水线的干净交叉验证得分。两者的差异有多大？

3. 使用 `joblib.dump` 序列化你的流水线。在另一个脚本中加载它并运行预测。验证两次预测结果是否完全一致。

4. 向流水线中添加一个自定义转换器，为两个最重要的数值型列创建多项式特征 (Polynomial Features，阶数为 2)。它应该放置在流水线的哪个位置？

5. 为流水线配置 MLflow 跟踪 (MLflow Tracking)。使用不同的超参数 (Hyperparameters) 运行 5 次实验。通过 MLflow 界面 (`mlflow ui`) 对比各次运行记录，并挑选出最佳模型。

## 关键术语

| 术语 | 人们的通俗说法 | 实际含义 |
|------|----------------|----------|
| 流水线 (Pipeline) | “转换器链 + 模型” | 按顺序排列的已拟合转换器与模型，作为一个整体单元应用，以防止数据泄露 |
| 数据泄露 (Data leakage) | “测试信息泄露到了训练中” | 在构建模型时使用了训练集之外的信息，导致性能评估结果虚高 |
| 列转换器 (ColumnTransformer) | “按列进行不同的预处理” | 对不同的列子集应用不同的流水线，并将结果合并 |
| 实验追踪 (Experiment tracking) | “记录你的运行日志” | 记录每次训练运行的参数、指标、模型工件 (artifacts) 及代码版本 |
| MLflow | “追踪并部署模型” | 用于实验追踪、模型注册表 (model registry) 和部署的开源平台 |
| DVC | “数据的 Git” | 针对大型数据文件的版本控制系统，将哈希值存储在 Git 中，而数据本身存储在远程存储中 |
| 模型注册表 (Model registry) | “模型版本目录” | 通过阶段标签（如预发布、生产、归档）来追踪模型版本的系统 |
| 训练/服务偏差 (Training/serving skew) | “在 Notebook 里明明能跑” | 训练阶段与推理阶段的数据处理方式存在差异，从而引发静默错误 |
| 可复现性 (Reproducibility) | “相同的代码，相同的结果” | 使用相同的代码、数据和配置能够获得完全一致结果的能力 |

## 延伸阅读

- [scikit-learn Pipeline 文档](https://scikit-learn.org/stable/modules/compose.html) -- 官方流水线参考指南
- [MLflow 文档](https://mlflow.org/docs/latest/index.html) -- 实验追踪与模型注册表
- [DVC 文档](https://dvc.org/doc) -- 数据版本控制
- [Sculley 等人，《机器学习系统中的隐藏技术债》（2015）](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) -- 探讨机器学习系统复杂性的奠基性论文
- [Google 机器学习最佳实践：机器学习规则](https://developers.google.com/machine-learning/guides/rules-of-ml) -- 面向生产环境的实用机器学习建议