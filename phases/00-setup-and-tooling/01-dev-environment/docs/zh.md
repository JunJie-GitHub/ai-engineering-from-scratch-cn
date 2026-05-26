# 开发环境 (Development Environment)

> 工具塑造思维。一次性配置到位，确保准确无误。

**类型：** 构建 (Build)
**语言：** Python、Node.js、Rust
**前置要求：** 无
**时长：** 约 45 分钟

## 学习目标

- 从零开始搭建 Python 3.11+、Node.js 20+ 和 Rust 工具链 (toolchains)
- 配置虚拟环境 (virtual environments) 与包管理器 (package managers)，以实现可复现构建 (reproducible builds)
- 通过 CUDA/MPS 验证 GPU 访问权限，并运行测试张量运算 (tensor operation)
- 理解四层技术栈 (four-layer stack)：系统层 (system)、软件包层 (packages)、运行时层 (runtimes) 与 AI 库层 (AI libraries)

## 问题

你将通过 200 多节课程学习 AI 工程（AI Engineering），使用 Python、TypeScript、Rust 和 Julia 进行实践。如果你的开发环境（Development Environment）配置不当，每一节课都会变成与工具链（Tooling）的搏斗，而非专注于学习本身。

大多数人会跳过环境配置（Environment Setup）。随后他们不得不花费数小时去调试导入错误（Import Errors）、版本冲突（Version Conflicts）以及缺失的 CUDA 驱动程序（CUDA Drivers）。我们将一次性把这件事彻底做好。

## 概念

AI 工程环境（AI Engineering Environment）包含四个层级：

graph TD
    A["4. AI/ML Libraries\nPyTorch, JAX, transformers, etc."] --> B["3. Language Runtimes\nPython 3.11+, Node 20+, Rust, Julia"]
    B --> C["2. Package Managers\nuv, pnpm, cargo, juliaup"]
    C --> D["1. System Foundation\nOS, shell, git, editor, GPU drivers"]

我们采用自底向上的顺序进行安装。每一层都依赖于其下方的层级。

## 构建

### 步骤 1：系统基础

检查您的系统并安装基础开发工具。

# macOS
xcode-select --install
brew install git curl wget

# Ubuntu/Debian
sudo apt update && sudo apt install -y build-essential git curl wget

# Windows (use WSL2)
wsl --install -d Ubuntu-24.04

### 步骤 2：使用 uv 配置 Python

我们使用 `uv` —— 它的运行速度比 pip 快 10 到 100 倍，且能自动管理虚拟环境 (Virtual Environments)。

curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12

uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

uv pip install numpy matplotlib jupyter

验证安装：

import sys
print(f"Python {sys.version}")

import numpy as np
print(f"NumPy {np.__version__}")
a = np.array([1, 2, 3])
print(f"Vector: {a}, dot product with itself: {np.dot(a, a)}")

### 步骤 3：使用 pnpm 配置 Node.js

用于 TypeScript 相关课程（智能体 (Agents)、MCP 服务器 (MCP Servers)、Web 应用）。

curl -fsSL https://fnm.vercel.app/install | bash
fnm install 22
fnm use 22

npm install -g pnpm

node -e "console.log('Node', process.version)"

### 步骤 4：Rust

用于对性能要求较高的课程（模型推理 (Inference)、系统编程 (Systems Programming)）。

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

rustc --version
cargo --version

### 步骤 5：Julia（可选）

用于数学计算密集型的课程，这也是 Julia 的强项所在。

curl -fsSL https://install.julialang.org | sh

julia -e 'println("Julia ", VERSION)'

### 步骤 6：GPU 配置（如果您有显卡）

# NVIDIA
nvidia-smi

# Install PyTorch with CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

没有 GPU (Graphics Processing Unit)？没问题。大多数课程在 CPU 上即可运行。对于训练密集型课程，建议使用 Google Colab 或云端 GPU。

### 步骤 7：全面验证

运行验证脚本：

python phases/00-setup-and-tooling/01-dev-environment/code/verify.py


## 使用方法

您的环境现已为本课程的每一节课准备就绪。以下是各项技术的具体应用场景：

| 语言 | 适用阶段 | 包管理器 |
|----------|---------|-----------------|
| Python | 第 1-12 阶段（机器学习 ML、深度学习 DL、自然语言处理 NLP、计算机视觉 Vision、音频处理 Audio、大语言模型 LLMs） | uv |
| TypeScript | 第 13-17 阶段（工具 Tools、智能体 Agents、多智能体集群 Swarms、基础设施 Infra） | pnpm |
| Rust | 第 12、15-17 阶段（性能关键型系统 Performance-critical systems） | cargo |
| Julia | 第 1 阶段（数学基础 Math foundations） | Pkg |

## 发布上线

本节课程将生成一个验证脚本 (verification script)，任何人都可以运行它来检查其环境配置 (setup)。

请参阅 `outputs/prompt-env-check.md`，其中包含一个提示词 (prompt)，可帮助 AI 助手 (AI assistants) 诊断环境问题 (environment issues)。

## 练习

1. 运行验证脚本 (verification script) 并修复任何失败项
2. 为本课程创建 Python 虚拟环境 (Python virtual environment) 并安装 PyTorch
3. 使用全部四种语言编写 "hello world" 程序并逐一运行