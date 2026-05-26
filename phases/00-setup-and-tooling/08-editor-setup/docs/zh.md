# 编辑器设置

> 编辑器是你的编程副驾驶。只需配置一次，让它不再干扰你的工作流，而是真正为你分担任务。

**类型：** 构建
**语言：** --
**前置要求：** 阶段 0，第 01 课
**时长：** 约 20 分钟

## 学习目标

- 安装 VS Code 并配置 Python、Jupyter、代码检查 (Linting) 与远程 SSH (Remote SSH) 等核心扩展
- 为 AI 工作流 (AI Workflows) 配置保存时格式化 (Format-on-Save)、类型检查 (Type Checking) 以及 Notebook 输出滚动 (Notebook Output Scrolling)
- 设置远程 SSH (Remote SSH)，以便像在本地一样在远程 GPU 机器上编辑和调试代码
- 评估编辑器替代方案（如 Cursor、Windsurf、Neovim）及其在 AI 开发中的权衡 (Tradeoffs)

## 问题

你将在编辑器 (Editor) 中花费数千小时编写 Python 代码、运行 Notebook、调试训练循环 (Training Loop)，以及通过 SSH 登录 GPU 服务器 (GPU Server)。配置不当的编辑器会让每次工作都充满阻力：没有自动补全 (Autocomplete)、没有类型提示 (Type Hints)、没有内联错误提示 (Inline Error)、需要手动格式化，以及笨拙的终端工作流 (Terminal Workflow)。

正确的配置只需 20 分钟。跳过它，你每天将为此浪费 20 分钟。

## 概念

搭建 AI 工程（AI Engineering）编辑器环境需要以下五个要素：

graph TD
    L5["5. Remote Development<br/>SSH into GPU boxes, cloud VMs"] --> L4
    L4["4. Terminal Integration<br/>Run scripts, debug, monitor GPU"] --> L3
    L3["3. AI-Specific Settings<br/>Auto-format, type checking, rulers"] --> L2
    L2["2. Extensions<br/>Python, Jupyter, Pylance, GitLens"] --> L1
    L1["1. Base Editor<br/>VS Code — free, extensible, universal"]


## 构建

### Step 1: Install VS Code

VS Code is the recommended editor. It is free, runs on every OS, has first-class Jupyter notebook support, and the extension ecosystem covers everything you need for AI work.

Download it from [code.visualstudio.com](https://code.visualstudio.com/).

Verify from the terminal:

```bash
code --version
```

If `code` is not found on macOS, open VS Code, press `Cmd+Shift+P`, type "Shell Command", and select "Install 'code' command in PATH".

### Step 2: Install Essential Extensions

Open the integrated terminal in VS Code (`Ctrl+`` ` or `` Cmd+` ``) and install the extensions that matter for AI work:

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension eamodio.gitlens
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-python.debugpy
code --install-extension ms-python.black-formatter
code --install-extension charliermarsh.ruff
```

What each one does:

| Extension | Why |
|-----------|-----|
| Python | Language support, virtual env detection, run/debug |
| Pylance | Fast type checking, autocomplete, import resolution |
| Jupyter | Run notebooks inside VS Code, variable explorer |
| GitLens | See who changed what, inline git blame |
| Remote SSH | Open a folder on a remote GPU box as if it were local |
| Debugpy | Step-through debugging for Python |
| Black Formatter | Auto-format on save, consistent style |
| Ruff | Fast linting, catches common mistakes |

The file `code/.vscode/extensions.json` in this lesson contains the full recommendations list. When you open the project folder, VS Code will prompt you to install them.

### Step 3: Configure Settings

Copy the settings from `code/.vscode/settings.json` in this lesson, or apply them manually through `Settings > Open Settings (JSON)`.

The key settings for AI work:

```jsonc
{
    "python.analysis.typeCheckingMode": "basic",
    "editor.formatOnSave": true,
    "editor.rulers": [88, 120],
    "notebook.output.scrolling": true,
    "files.autoSave": "afterDelay"
}
```

Why these matter:

- **Type checking on basic**: Catches wrong argument types before you run. Saves debugging time on tensor shape mismatches and wrong API parameters.
- **Format on save**: Never think about formatting again. Black handles it.
- **Rulers at 88 and 120**: Black wraps at 88. The 120 marker shows when docstrings and comments are getting too long.
- **Notebook output scrolling**: Training loops print thousands of lines. Without scrolling, the output panel explodes.
- **Auto-save**: You will forget to save. Your training script will run stale code. Auto-save prevents that.

### Step 4: Terminal Integration

VS Code's integrated terminal is where you run training scripts, monitor GPUs, and manage environments.

Set it up properly:

```jsonc
{
    "terminal.integrated.defaultProfile.osx": "zsh",
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.fontSize": 13,
    "terminal.integrated.scrollback": 10000
}
```

Useful shortcuts:

| Action | macOS | Linux/Windows |
|--------|-------|---------------|
| Toggle terminal | `` Ctrl+` `` | `` Ctrl+` `` |
| New terminal | `Ctrl+Shift+`` ` | `Ctrl+Shift+`` ` |
| Split terminal | `Cmd+\` | `Ctrl+\` |

Split terminals are useful: one for running your script, one for monitoring GPU with `nvidia-smi -l 1` or `watch -n 1 nvidia-smi`.

### Step 5: Remote Development (SSH into GPU Boxes)

This is the most important extension for AI work. You will run training on remote machines (cloud VMs, lab servers, Lambda, Vast.ai). Remote SSH lets you open the remote filesystem, edit files, run terminals, and debug as if everything were local.

Setup:

1. Install the Remote SSH extension (done in Step 2).
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P`), type "Remote-SSH: Connect to Host".
3. Enter `user@your-gpu-box-ip`.
4. VS Code installs its server component on the remote machine automatically.

For passwordless access, set up SSH keys:

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
ssh-copy-id user@your-gpu-box-ip
```

Add the host to `~/.ssh/config` for convenience:

```
Host gpu-box
    HostName 203.0.113.50
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
```

Now `Remote-SSH: Connect to Host > gpu-box` connects instantly.

## 替代方案

### Cursor

[cursor.com](https://cursor.com) 是一款内置 AI 代码生成（AI code generation）功能的 VS Code 分支（fork）。它沿用相同的扩展生态系统和配置文件格式。如果你使用 Cursor，本课程的所有内容依然完全适用。只需导入相同的 `settings.json` 和 `extensions.json` 即可。

### Windsurf

[windsurf.com](https://windsurf.com) 是另一款主打 AI 优先（AI-first）的 VS Code 分支。情况如出一辙：兼容相同的扩展、采用相同的设置格式，并提供相同的远程 SSH（Remote SSH）支持。

### Vim/Neovim

如果你已经在使用 Vim 或 Neovim 且能保持高效工作，请继续使用。进行 AI Python 开发的最低配置要求如下：

- 使用 **pyright** 或 **pylsp** 进行类型检查（type checking）（通过 Mason 或手动安装）
- 使用 **nvim-lspconfig** 实现语言服务器（language server）集成
- 使用 **jupyter-vim** 或 **molten-nvim** 实现类似 Notebook 的代码执行
- 使用 **telescope.nvim** 进行文件/符号搜索
- 使用 **none-ls.nvim** 配合 black 和 ruff 进行代码格式化（formatting）与静态检查（linting）

如果你尚未使用过 Vim，建议现在不要从零开始。它的学习曲线会分散你学习 AI 工程（AI engineering）的精力。请直接使用 VS Code。

## 使用方法

完成此配置后，您的日常工作流程如下：

1. 在 VS Code 中打开项目文件夹（或通过 Remote SSH 连接到 GPU 主机 (GPU box)）。
2. 在编辑器中编写 Python 代码，支持自动补全 (autocomplete)、类型提示 (type hints) 和内联错误 (inline errors)。
3. 借助 Jupyter 扩展 (Jupyter extension) 在编辑器内直接运行 Jupyter Notebook。
4. 使用集成终端 (integrated terminal) 运行训练脚本 (training scripts)、执行 `uv pip install` 以及进行 GPU 监控 (GPU monitoring)。
5. 在提交 (commit) 前，使用 GitLens 审查更改。

## 练习

1. 安装 Visual Studio Code (VS Code) 以及第 2 步中列出的所有扩展 (Extensions)
2. 将本课的 `settings.json` 复制到你的 VS Code 配置目录中
3. 打开一个 Python 文件，验证 Pylance 是否显示类型提示 (Type Hints)，以及 Black 是否在保存时自动格式化代码
4. 如果你能访问远程计算机 (Remote Machine)，请配置 Remote SSH 并在该机器上打开一个文件夹

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 语言服务器协议 (LSP) | “自动补全引擎” | 语言服务器协议 (Language Server Protocol)：一种标准协议，允许编辑器从特定语言的服务器获取类型信息、代码补全建议和诊断数据 |
| Pylance | “Python 插件” | 微软开发的 Python 语言服务器，底层采用 Pyright 进行类型检查与智能感知 (IntelliSense) |
| 远程 SSH (Remote SSH) | “在服务器上工作” | VS Code 扩展插件，可在远程主机上运行轻量级服务端，并将图形界面实时同步至本地编辑器 |
| 保存时格式化 (Format on save) | “自动美化” | 每次保存文件时，编辑器会自动调用格式化工具（如 Black、Ruff），确保代码风格始终保持一致 |