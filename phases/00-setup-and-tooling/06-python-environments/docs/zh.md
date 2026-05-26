# Python 环境 (Python Environments)

> 依赖地狱 (Dependency Hell) 确实存在。虚拟环境 (Virtual Environments) 正是解药。

**类型：** 构建
**语言：** Python
**先决条件：** 第 0 阶段，第 01 课
**时长：** 约 30 分钟

## 学习目标

- 使用 `uv`、`venv` 或 `conda` 创建隔离的虚拟环境 (Virtual Environments)
- 编写包含可选依赖组 (Dependency Groups) 的 `pyproject.toml` 文件，并生成锁定文件 (Lockfiles) 以确保可复现性
- 诊断并修复常见陷阱：全局安装 (Global Installs)、pip/conda 混用、CUDA 版本不匹配 (CUDA Version Mismatches)
- 针对存在依赖冲突的项目，实施分阶段环境策略 (Per-phase Environment Strategy)

## 问题

你为某个微调（fine-tuning）项目安装了 PyTorch 2.4。下周，另一个项目因其 CUDA 构建（CUDA build）版本被锁定（pinned），需要 PyTorch 2.1。你进行了全局升级，结果第一个项目无法运行。你降级后，第二个项目又无法运行。

这就是依赖地狱（dependency hell）。在 AI/ML 开发中这种情况屡见不鲜，原因在于：

- PyTorch、JAX 和 TensorFlow 各自附带独立的 CUDA 绑定（CUDA bindings）
- 模型库会锁定特定的框架版本
- 全局 `pip install` 会覆盖之前安装的任何内容
- CUDA 11.8 构建无法与 CUDA 12.x 驱动配合使用（反之亦然）

解决方案：为每个项目创建独立的隔离环境（isolated environment），并配置专属的依赖包。

## 概念

graph TD
    subgraph without["Without virtual environments"]
        SP[System Python] --> T24["torch 2.4.0 (CUDA 12.4)\nProject A needs this"]
        SP --> T21["torch 2.1.0 (CUDA 11.8)\nProject B needs this"]
        SP --> CONFLICT["CONFLICT: only one\ntorch version can exist"]
    end

    subgraph with["With virtual environments"]
        PA["Project A (.venv/)"] --> PA1["torch 2.4.0 (CUDA 12.4)"]
        PA --> PA2["transformers 4.44"]
        PB["Project B (.venv/)"] --> PB1["torch 2.1.0 (CUDA 11.8)"]
        PB --> PB2["diffusers 0.28"]
    end


## 动手构建

### 选项 1：uv venv（推荐）

`uv` 是目前最快的 Python 包管理器（Python Package Manager）（速度比 pip 快 10-100 倍）。它将虚拟环境（Virtual Environment）、Python 版本管理与依赖解析（Dependency Resolution）集成于单一工具中。

curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12

cd your-project
uv venv
source .venv/bin/activate

安装软件包：

uv pip install torch numpy

一步创建包含 `pyproject.toml` 的项目：

uv init my-ai-project
cd my-ai-project
uv add torch numpy matplotlib

### 选项 2：venv（内置）

如果无法安装 `uv`，Python 自带了 `venv`：

python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

pip install torch numpy

速度比 `uv` 慢，但在任何安装了 Python 的地方都能使用。

### 选项 3：conda（按需使用）

Conda 用于管理非 Python 依赖项（Non-Python Dependencies），例如 CUDA 工具包（CUDA Toolkit）、cuDNN 和 C 语言库（C Library）。在以下情况下建议使用：

- 你需要特定版本的 CUDA 工具包，且不想在系统范围内安装
- 你在使用共享集群，无法安装系统级软件包
- 某个库的安装说明明确要求“使用 conda”

# Install miniconda (not the full Anaconda)
curl -LsSf https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
bash miniconda.sh -b

conda create -n myproject python=3.12
conda activate myproject

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

一条重要原则：如果在一个环境中使用 conda，则该环境中的所有软件包都应通过 conda 安装。在 conda 环境中混用 `pip install` 会导致依赖冲突，调试起来非常痛苦。

### 本课程建议：分阶段策略

你可以为整个课程只创建一个环境，但强烈不建议这样做。不同阶段需要不同的（有时甚至是相互冲突的）依赖项。

策略如下：

ai-engineering-from-scratch/
├── .venv/                    <-- shared lightweight env for phases 0-3
├── phases/
│   ├── 04-neural-networks/
│   │   └── .venv/            <-- PyTorch env
│   ├── 05-cnns/
│   │   └── .venv/            <-- same PyTorch env (symlink or shared)
│   ├── 08-transformers/
│   │   └── .venv/            <-- might need different transformer versions
│   └── 11-llm-apis/
│       └── .venv/            <-- API SDKs, no torch needed

`code/env_setup.sh` 中的脚本将为本课程创建基础环境。

## pyproject.toml 基础

每个 Python 项目都应包含一个 `pyproject.toml` 文件。它将 `setup.py`、`setup.cfg` 和 `requirements.txt` 的功能统一整合至单个文件中。

[project]
name = "ai-engineering-from-scratch"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "matplotlib>=3.8",
    "jupyter>=1.0",
    "scikit-learn>=1.4",
]

[project.optional-dependencies]
torch = ["torch>=2.3", "torchvision>=0.18"]
llm = ["anthropic>=0.39", "openai>=1.50"]

然后执行安装：

uv pip install -e ".[torch]"    # base + PyTorch
uv pip install -e ".[llm]"     # base + LLM SDKs
uv pip install -e ".[torch,llm]" # everything


## 锁定文件 (Lockfiles)

锁定文件 (lockfile) 会将所有依赖 (dependency)（包括传递依赖 (transitive dependencies)）固定到精确版本。这保证了可复现性 (reproducibility)：任何基于该锁定文件进行安装的用户，都将获得完全相同的软件包。

# uv generates uv.lock automatically when using uv add
uv add numpy

# pip-tools approach
uv pip compile pyproject.toml -o requirements.lock
uv pip install -r requirements.lock

将锁定文件提交至 Git。当他人克隆仓库 (repository) 时，他们通过锁定文件安装，即可获得完全一致的版本。

## 常见错误

### 1. 全局安装

pip install torch  # BAD: installs to system Python

source .venv/bin/activate
pip install torch  # GOOD: installs to virtual environment

检查软件包的安装路径：

which python       # should show .venv/bin/python, not /usr/bin/python
which pip           # should show .venv/bin/pip

### 2. 混用 pip 和 conda

conda create -n myenv python=3.12
conda activate myenv
conda install pytorch -c pytorch
pip install some-other-package   # BAD: can break conda's dependency tracking
conda install some-other-package # GOOD: let conda manage everything

如果必须在 conda 环境中使用 pip（某些软件包仅提供 pip 版本），请先安装所有 conda 软件包，最后再安装 pip 软件包。

### 3. 忘记激活环境

python train.py           # uses system Python, missing packages
source .venv/bin/activate
python train.py           # uses project Python, packages found

你的 Shell 提示符 (Shell Prompt) 应显示环境名称：

(.venv) $ python train.py

### 4. 将 .venv 提交至 Git

echo ".venv/" >> .gitignore

虚拟环境 (Virtual Environment) 的大小通常在 200MB 到 2GB 之间。它们仅存在于本地，无法在不同机器间直接移植。建议改为提交 `pyproject.toml` 和锁定文件 (Lockfile)。

### 5. CUDA 版本不匹配

nvidia-smi                # shows driver CUDA version (e.g., 12.4)
python -c "import torch; print(torch.version.cuda)"  # shows PyTorch CUDA version

# These must be compatible.
# PyTorch CUDA version must be <= driver CUDA version.


## 使用方法

运行设置脚本（setup script）以创建课程环境（course environment）：

bash phases/00-setup-and-tooling/06-python-environments/code/env_setup.sh

这将在仓库根目录（repo root）创建一个 `.venv`，并完成核心依赖（core dependencies）的安装与验证。

## 练习

1. 运行 `env_setup.sh` 并验证所有检查均通过
2. 创建第二个虚拟环境 (virtual environment)，在其中安装不同版本的 numpy，并确认这两个环境相互隔离
3. 为需要同时使用 PyTorch 和 Anthropic SDK 的项目编写 `pyproject.toml`
4. 故意在未激活 venv 的情况下全局安装一个软件包，观察其安装位置，然后将其卸载

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 虚拟环境 (Virtual environment) | "一个 venv" | 一个包含 Python 解释器和依赖包的独立目录，与系统 Python 环境完全隔离 |
| 锁定文件 (Lockfile) | "固定版本的依赖" | 一个列出所有软件包及其精确版本的文件，确保在不同机器上安装的结果完全一致 |
| pyproject.toml | "新版 setup.py" | Python 项目的标准配置文件，用于替代 setup.py/setup.cfg/requirements.txt |
| 传递依赖 (Transitive dependency) | "依赖的依赖" | 软件包 B 依赖于 C；如果你安装依赖于 B 的 A，那么 C 就是 A 的传递依赖 |
| CUDA 版本不匹配 (CUDA mismatch) | "我的 GPU 没法用" | PyTorch 编译时使用的 CUDA 版本与你的 GPU 驱动程序所支持的版本不一致 |