# Jupyter 笔记本 (Jupyter Notebook)

> Notebook 是 AI 工程（AI Engineering）的实验台。你在此进行原型开发（Prototyping），随后将验证可行的部分部署至生产环境（Production）。

**类型：** 构建
**语言：** Python
**前置条件：** 第 0 阶段，第 01 课
**时长：** 约 30 分钟

## 学习目标

- 安装并启动 JupyterLab、Jupyter Notebook 或带有 Jupyter 扩展的 VS Code
- 使用魔术命令 (magic commands)（`%timeit`、`%%time`、`%matplotlib inline`）进行基准测试 (benchmark) 与内联可视化 (inline visualization)
- 区分何时使用 Notebook 与脚本 (scripts)，并应用“在 Notebook 中探索，在脚本中交付”的工作流 (workflow)
- 识别并避免常见的 Notebook 陷阱：乱序执行 (out-of-order execution)、隐藏状态 (hidden state) 以及内存泄漏 (memory leaks)

## 问题

每一篇 AI 论文、教程和 Kaggle 竞赛都会使用 Jupyter 笔记本 (Jupyter Notebooks)。它们允许你分块运行代码、直接查看内联输出、将代码与解释混合，并快速迭代。如果你试图在不使用笔记本的情况下学习 AI，那就像做数学作业不用草稿纸一样。

但笔记本确实存在一些陷阱。人们把它们用于各种场景，包括它们根本不擅长的领域。了解何时该使用笔记本，何时该使用脚本 (Script)，将能帮你避免日后陷入调试 (Debugging) 噩梦。

## 概念

笔记本（Notebook）由一系列单元格（Cell）组成。每个单元格要么是代码，要么是文本。

graph TD
    A["**Markdown Cell**\n# My Experiment\nTesting learning rate 0.01"] --> B["**Code Cell** ► Run\nmodel.fit(X, y, lr=0.01)\n---\nOutput: loss = 0.342"]
    B --> C["**Code Cell** ► Run\nplt.plot(losses)\n---\nOutput: inline plot"]

内核（Kernel）是一个在后台运行的 Python 进程。当你运行某个单元格时，系统会将代码发送给内核，内核执行完毕后会将结果返回。由于所有单元格共享同一个内核，因此变量状态会在不同单元格之间持续保留。

graph LR
    A[Notebook UI] <--> B[Kernel\nPython process]
    B --> C[Keeps variables in memory]
    B --> D[Runs cells in whatever order you click]
    B --> E[Dies when you restart it]

这种“按任意点击顺序执行”的特性既是它的强大优势，也是容易“误伤自己”的陷阱。

## 构建

### 步骤 1：选择你的开发界面

三种选择，一种格式：

| 界面 | 安装 | 最佳适用场景 |
|-----------|---------|----------|
| JupyterLab | `pip install jupyterlab` 然后 `jupyter lab` | 完整的集成开发环境 (IDE) 体验，支持多标签页、文件浏览器和终端 |
| Jupyter Notebook | `pip install notebook` 然后 `jupyter notebook` | 简单轻量，适合逐个处理笔记本 |
| VS Code | 安装 "Jupyter" 扩展 | 无缝集成到现有编辑器，支持 Git 集成与调试 |

这三者都读写相同的 `.ipynb` 文件。选择你喜欢的即可。在 AI 开发中，JupyterLab 最为常用。

pip install jupyterlab
jupyter lab

### 步骤 2：关键快捷键

操作分为两种模式。按 `Escape` 进入命令模式 (Command mode)（左侧显示蓝色条），按 `Enter` 进入编辑模式 (Edit mode)（左侧显示绿色条）。

**命令模式（最常用）：**

| 按键 | 操作 |
|-----|--------|
| `Shift+Enter` | 运行当前单元格，并跳转到下一个 |
| `A` | 在上方插入单元格 |
| `B` | 在下方插入单元格 |
| `DD` | 删除单元格 |
| `M` | 转换为 Markdown 格式 |
| `Y` | 转换为代码格式 |
| `Z` | 撤销单元格操作 |
| `Ctrl+Shift+H` | 显示所有快捷键 |

**编辑模式：**

| 按键 | 操作 |
|-----|--------|
| `Tab` | 自动补全 |
| `Shift+Tab` | 显示函数签名/参数提示 |
| `Ctrl+/` | 切换注释 |

`Shift+Enter` 是你每天会用到上千次的快捷键，请优先掌握。

### 步骤 3：单元格类型

**代码单元格 (Code cells)** 用于运行 Python 代码并显示输出：

import numpy as np
data = np.random.randn(1000)
data.mean(), data.std()

输出：`(0.0032, 0.9987)`

**Markdown 单元格 (Markdown cells)** 用于渲染格式化文本。可用它们记录你的操作步骤及原因。支持标题、粗体、斜体、LaTeX 数学公式（`$E = mc^2$`）、表格和图片。

### 步骤 4：魔法命令 (Magic commands)

这些并非 Python 语法，而是 Jupyter 特有的命令，以 `%`（行魔法命令 (Line magic)）或 `%%`（单元格魔法命令 (Cell magic)）开头。

**代码计时：**

%timeit np.random.randn(10000)

输出：`45.2 us +/- 1.3 us per loop`

%%time
model.fit(X_train, y_train, epochs=10)

输出：`Wall time: 2.34 s`

`%timeit` 会多次运行代码并计算平均值。`%%time` 仅运行一次。建议使用 `%timeit` 进行微基准测试 (Microbenchmarks)，使用 `%%time` 监控模型训练过程。

**启用内联绘图 (Inline plots)：**

%matplotlib inline

现在，所有的 `plt.plot()` 或 `plt.show()` 都会直接在笔记本中渲染显示。

**无需离开笔记本即可安装依赖包：**

!pip install scikit-learn

使用 `!` 前缀可以执行任意 Shell 命令。

**查看环境变量：**

%env CUDA_VISIBLE_DEVICES

### 步骤 5：内联显示富文本输出 (Rich output)

笔记本会自动显示单元格中的最后一个表达式。但你也可以手动控制输出：

import pandas as pd

df = pd.DataFrame({
    "model": ["Linear", "Random Forest", "Neural Net"],
    "accuracy": [0.72, 0.89, 0.94],
    "training_time": [0.1, 2.3, 45.6]
})
df

这会渲染为格式化的 HTML 表格，而非纯文本输出。绘图同理：

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title("Inline Plot")
plt.show()

图表会直接显示在单元格下方。这正是笔记本在 AI 开发中占据主导地位的原因：你可以将数据、图表和代码集中查看。

显示图片：

from IPython.display import Image, display
display(Image(filename="architecture.png"))

### 步骤 6：Google Colab

Colab 是一款免费的云端 Jupyter 笔记本服务。它提供 GPU 算力、预装常用库，并深度集成 Google Drive。无需任何本地配置即可使用。

1. 访问 [colab.research.google.com](https://colab.research.google.com)
2. 上传本课程中的任意 `.ipynb` 文件
3. Runtime > Change runtime type > T4 GPU（免费）

Colab 与本地 Jupyter 的主要区别：
- 会话结束后文件不会保留（需保存至 Drive 或手动下载）
- 预装库：numpy, pandas, matplotlib, torch, tensorflow, sklearn
- `from google.colab import files` 上传/下载文件
- `from google.colab import drive; drive.mount('/content/drive')` 挂载持久化存储
- 免费版会话在无操作 90 分钟后会自动超时断开

## 使用方法

### 交互式笔记本（Notebooks）与脚本（Scripts）：何时使用哪种

| 适用于笔记本（Notebooks）的场景 | 适用于脚本（Scripts）的场景 |
|-------------------|-----------------|
| 探索数据集 | 训练流水线（Training pipelines） |
| 原型设计模型 | 可复用工具（Reusable utilities） |
| 可视化结果 | 包含 `if __name__` 的代码 |
| 展示你的工作成果 | 定时运行的代码 |
| 快速实验 | 生产环境代码 |
| 课程练习 | 软件包与库（Packages and libraries） |

核心原则：**在笔记本中探索，在脚本中交付**。

AI 开发中的常见工作流：
1. 在笔记本中探索数据
2. 在笔记本中构建模型原型
3. 验证可行后，将代码迁移至 `.py` 文件
4. 将 `.py` 文件重新导入笔记本，以便进行后续实验

### 常见陷阱

**乱序执行（Out-of-order execution）。** 你先运行了第 5 个单元格，接着是第 2 个，然后是第 7 个。笔记本在你自己的机器上运行正常，但别人从上到下顺序执行时就会报错。解决方法：分享前点击 Kernel > Restart & Run All。

**隐藏状态（Hidden state）。** 你删除了某个单元格，但它创建的变量仍残留在内存中。笔记本表面看起来很干净，但实际上依赖于一个“幽灵单元格”。解决方法：定期重启内核。

**内存泄漏（Memory leaks）。** 加载一个 4GB 的数据集，训练模型，再加载另一个数据集。内存始终得不到释放。解决方法：使用 `del variable_name` 和 `gc.collect()`，或直接重启内核。

## 发布

本课时将生成：
- `outputs/prompt-notebook-helper.md`，用于调试笔记本（Notebook）问题

## 练习

1. 打开 JupyterLab，创建一个笔记本，并使用 `%timeit` 比较列表推导式 (list comprehension) 与 NumPy 在创建包含 100,000 个随机数的数组时的性能。
2. 创建一个同时包含 Markdown 和代码单元格的笔记本，用于加载 CSV 文件、显示数据框 (DataFrame) 并绘制图表。随后运行 Kernel > Restart & Run All，以验证其能否从上至下完整执行。
3. 将 `code/notebook_tips.py` 中的代码复制并粘贴到 Colab 笔记本中，并使用免费的图形处理器 (GPU) 运行。

## 关键术语

| 术语 | 通常的说法 | 实际含义 |
|------|----------------|----------------------|
| 内核 (Kernel) | "运行我代码的东西" | 一个独立的 Python 进程，负责执行单元格并将变量保留在内存中 |
| 单元格 (Cell) | "代码块" | 笔记本中可独立运行的单元，可以是代码或 Markdown 格式 |
| 魔法命令 (Magic command) | "Jupyter 小技巧" | 以 `%` 或 `%%` 开头的特殊命令，用于控制笔记本环境 |
| `.ipynb` | "笔记本文件" | 包含单元格、输出和元数据的 JSON 文件。全称为 IPython Notebook |

## 延伸阅读

- [JupyterLab 文档](https://jupyterlab.readthedocs.io/) 以了解完整功能集
- [Google Colab 常见问题解答](https://research.google.com/colaboratory/faq.html) 以了解 Colab 特有的限制与功能
- [28 个 Jupyter Notebook 技巧](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/) 以掌握高级用户快捷方式