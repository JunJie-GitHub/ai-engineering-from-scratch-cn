# 数据管理 (Data Management)

> 数据是燃料。你如何管理它，决定了你能跑多快。

**类型:** 构建 (Build)
**语言:** Python
**前置要求:** 第 0 阶段，第 01 课
**时长:** 约 45 分钟

## 学习目标

- 使用 Hugging Face 的 `datasets` 库加载、流式加载 (streaming) 和缓存 (caching) 数据集
- 在 CSV、JSON、Parquet 和 Arrow 格式之间进行转换，并解释各自的权衡取舍 (tradeoffs)
- 通过固定随机种子 (random seeds)，创建可复现的训练集/验证集/测试集划分 (train/validation/test splits)
- 使用 `.gitignore`、Git LFS 或 DVC 管理大型模型与数据集文件

## 问题

每个 AI 项目 (AI project) 都始于数据。你需要查找数据集 (datasets)，下载它们，进行格式转换，将其划分为训练集 (training set) 和评估集 (evaluation set)，并进行版本控制 (version control) 以确保实验可复现 (reproducible)。每次手动执行这些操作既缓慢又容易出错。你需要一套可重复的工作流 (repeatable workflow)。

## 概念

graph TD
    A["Hugging Face Hub"] --> B["datasets library"]
    B --> C["Load / Stream"]
    C --> D["Local Cache<br/>~/.cache/huggingface/"]
    B --> E["Format Conversion<br/>CSV, JSON, Parquet, Arrow"]
    E --> F["Data Splits<br/>train / val / test"]
    F --> G["Your Training Pipeline"]

Hugging Face 的 `datasets` 库 (datasets library) 是人工智能 (AI) 开发中加载数据的标准方式。它开箱即用地支持数据下载、缓存、格式转换与流式传输 (streaming)。

## 构建它

### 步骤 1：安装 datasets 库

pip install datasets huggingface_hub

### 步骤 2：加载数据集 (Dataset)

from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset)
print(dataset["train"][0])

这将下载 IMDB 电影评论数据集。首次下载后，后续加载将直接从 `~/.cache/huggingface/datasets/` 缓存中读取。

### 步骤 3：流式加载 (Streaming) 大型数据集

部分数据集体积过大，无法完整存入磁盘。流式加载会逐行读取数据，而无需下载完整文件。

dataset = load_dataset("wikimedia/wikipedia", "20220301.en", split="train", streaming=True)

for i, example in enumerate(dataset):
    print(example["title"])
    if i >= 4:
        break

流式加载会返回一个 `IterableDataset` 对象。你可以逐行处理到达的数据。无论数据集规模多大，内存占用都将保持恒定。

### 步骤 4：数据集格式 (Dataset Formats)

`datasets` 库底层使用 Apache Arrow 格式。你可以根据数据处理流水线 (Pipeline) 的需求，将其转换为其他格式。

dataset = load_dataset("imdb", split="train")

dataset.to_csv("imdb_train.csv")
dataset.to_json("imdb_train.json")
dataset.to_parquet("imdb_train.parquet")

格式对比：

| 格式 | 体积 | 读取速度 | 适用场景 |
|--------|------|-----------|----------|
| CSV | 大 | 慢 | 人类可读、电子表格 |
| JSON | 大 | 慢 | API 交互、嵌套数据 |
| Parquet | 小 | 快 | 数据分析、列式查询 |
| Arrow | 小 | 最快 | 内存处理（`datasets` 内部使用的格式） |

在 AI 开发中，Parquet 是最佳的存储格式，而 Arrow 是内存中处理数据的首选格式。CSV 和 JSON 则主要用于数据交换。

### 步骤 5：数据集划分 (Data Splits)

每个机器学习 (Machine Learning) 项目都需要将数据划分为三个部分：

- **训练集 (Train)**：模型从中学习（通常占 80%）
- **验证集 (Validation)**：用于在训练过程中监控进度（通常占 10%）
- **测试集 (Test)**：训练完成后进行最终评估（通常占 10%）

部分数据集已预先划分好。若未划分，你需要手动进行：

dataset = load_dataset("imdb", split="train")

split = dataset.train_test_split(test_size=0.2, seed=42)
train_val = split["train"].train_test_split(test_size=0.125, seed=42)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = split["test"]

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

务必设置随机种子 (Seed) 以确保结果可复现。相同的种子每次都会生成完全相同的划分结果。

### 步骤 6：下载与缓存模型

模型文件通常体积庞大。`huggingface_hub` 库负责处理模型的下载与缓存。

from huggingface_hub import hf_hub_download, snapshot_download

model_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="config.json"
)
print(f"Cached at: {model_path}")

model_dir = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
print(f"Full model at: {model_dir}")

模型会缓存至 `~/.cache/huggingface/hub/` 目录。下载完成后，后续运行即可瞬间加载。

### 步骤 7：处理大文件

模型权重 (Model Weights) 和大型数据集不应直接提交到 Git。有三种常用方案：

**方案 A：使用 `.gitignore`（最简单）**

*.bin
*.safetensors
*.pt
*.onnx
data/*.parquet
data/*.csv
models/

**方案 B：使用 Git LFS（在 Git 中追踪大文件）**

git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes

Git LFS 会在仓库中存储指针文件，而实际文件则存放在独立的服务器上。GitHub 提供 1 GB 的免费额度。

**方案 C：使用 DVC（数据版本控制 (Data Version Control)）**

pip install dvc
dvc init
dvc add data/training_set.parquet
git add data/training_set.parquet.dvc data/.gitignore
git commit -m "Track training data with DVC"

DVC 会生成小型的 `.dvc` 文件来指向你的数据。实际数据则存储在 S3、GCS 或其他远程存储后端中。

| 方案 | 复杂度 | 适用场景 |
|----------|-----------|----------|
| `.gitignore` | 低 | 个人项目、可随时重新下载的数据 |
| Git LFS | 中 | 团队通过 Git 共享模型权重 |
| DVC | 高 | 可复现实验、大型数据集、团队协作 |

对于本课程，使用 `.gitignore` 已足够。当你需要在不同机器上精确复现实验时，再考虑使用 DVC。

### 步骤 8：存储模式 (Storage Patterns)

**本地存储 (Local Storage)** 适用于体积小于约 10 GB 的数据集。Hugging Face (HF) 缓存机制会自动处理此类情况。

**云存储 (Cloud Storage)** 适用于更大规模的数据或需要在多台机器间共享的场景：

import os

local_path = os.path.expanduser("~/.cache/huggingface/datasets/")

# s3_path = "s3://my-bucket/datasets/"
# gcs_path = "gs://my-bucket/datasets/"

DVC 可直接与 S3 和 GCS 集成：

dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push

对于本课程，本地存储已完全足够。当你需要在远程 GPU 实例上进行模型微调 (Fine-tuning) 时，云存储才会变得必要。

## 本课程使用的数据集

| 数据集 (Dataset) | 课程 (Lessons) | 大小 (Size) | 教学内容 (What It Teaches) |
|---------|---------|------|----------------|
| IMDB | 分词 (Tokenization)、分类 (Classification) | 84 MB | 文本分类基础 |
| WikiText | 语言建模 (Language modeling) | 181 MB | 下一词元预测 (Next-token prediction) |
| SQuAD | 问答系统 (QA systems) | 35 MB | 问答 (Question answering)、答案跨度 (Spans) |
| Common Crawl（子集） | 嵌入 (Embeddings) | 不固定 (Varies) | 大规模文本处理 (Large-scale text processing) |
| MNIST | 视觉基础 (Vision basics) | 21 MB | 图像分类基础 (Image classification fundamentals) |
| COCO（子集） | 多模态 (Multimodal) | 不固定 (Varies) | 图文对 (Image-text pairs) |

您现在无需下载所有这些数据集。每节课程都会明确说明所需的具体内容。

## 使用方法

运行工具脚本以验证一切是否正常工作：

python code/data_utils.py

该脚本会下载一个小型数据集 (dataset)，对其进行转换与划分，并打印摘要信息。

## 发布上线

本节课程将生成：
- `code/data_utils.py` - 可复用的数据加载 (data loading) 与缓存 (caching) 工具
- `outputs/prompt-data-helper.md` - 用于为特定任务寻找合适数据集 (dataset) 的提示词 (prompt)

## 练习

1. 加载 `glue` 数据集 (dataset) 的 `mrpc` 配置 (config)，并查看前 5 个样本 (examples)
2. 流式加载 (stream) `c4` 数据集，并统计在 10 秒内可处理的样本数量
3. 将数据集转换为 Parquet 格式，并将其文件大小与 CSV 格式进行对比
4. 使用固定随机种子 (fixed seed) 创建 70/15/15 的训练集/验证集/测试集划分 (train/val/test split)，并验证各部分的数据量

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 数据集划分 (Dataset Split) | “训练数据” | 在机器学习生命周期不同阶段使用的已命名子集（如 train/val/test） |
| 流式处理 (Streaming) | “惰性加载” | 无需下载完整数据集，直接从远程源逐行处理数据 |
| Parquet 格式 (Parquet) | “压缩版 CSV” | 一种针对分析查询和存储效率进行优化的列式文件格式 |
| Apache Arrow (Arrow) | “高性能 DataFrame” | 一种内存列式格式，datasets 库内部使用它来实现零拷贝读取 |
| Git LFS (Large File Storage) | “大文件版 Git” | 一种扩展工具，将大文件存储在 Git 仓库外部，同时在版本控制系统中保留指针引用 |
| DVC (Data Version Control) | “数据版 Git” | 一种用于数据集和模型的版本控制系统，支持与云存储集成 |
| 缓存 (Cache) | “已下载” | 之前获取数据的本地副本，默认存储在 ~/.cache/huggingface/ 路径下 |