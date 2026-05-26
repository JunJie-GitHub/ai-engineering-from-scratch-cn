# 终端 (Terminal) 与 Shell

> 终端是 AI 工程师的日常阵地。请在这里熟悉并掌握它。

**类型：** 学习
**语言：** --
**前置要求：** 阶段 0，课程 01
**时长：** 约 35 分钟

## 学习目标

- 使用管道（piping）、重定向（redirects）和 `grep` 从命令行过滤并处理训练日志
- 创建包含多个窗格的持久化 tmux 会话（persistent tmux sessions），用于并发训练（concurrent training）与 GPU 监控（GPU monitoring）
- 使用 `htop`、`nvtop` 和 `nvidia-smi` 监控系统与 GPU 资源
- 使用 SSH、`scp` 和 `rsync` 在本地与远程主机之间传输文件

## 问题

你在终端 (Terminal) 上花费的时间将超过任何编辑器 (Editor)。模型训练 (Training runs)、GPU 监控 (GPU monitoring)、日志追踪 (log tailing)、远程 SSH 会话 (remote SSH sessions) 以及环境管理 (environment management)。每一个 AI 工作流 (AI workflow) 都会与 Shell 打交道。如果你在这里操作缓慢，那么在其他任何环节也会如此。

本课程将涵盖对 AI 工作至关重要的终端技能。不讲 Unix 历史，也不深入探讨 Bash 脚本 (Bash scripting)。只讲你真正需要的内容。

## 概念

graph TD
    subgraph tmux["tmux session: training"]
        subgraph top["Top row"]
            P1["Pane 1: Training run<br/>python train.py<br/>Epoch 12/100 ..."]
            P2["Pane 2: GPU monitor<br/>watch -n1 nvidia-smi<br/>GPU: 78% | Mem: 14/24G"]
        end
        P3["Pane 3: Logs + experiments<br/>tail -f logs/train.log | grep loss"]
    end

三个任务同时运行。只需一个终端。你可以分离 (detach) 会话，回家后再通过 SSH 重新接入 (reattach)。训练任务会持续运行。

## 构建它

### 步骤 1：了解你的终端环境（Shell）

检查你当前运行的 Shell（Shell）：

echo $SHELL

大多数系统使用 `bash` 或 `zsh`。两者都能正常工作，本课程中的命令在两者中均可使用。

需要掌握的关键操作：

# Move around
cd ~/projects/ai-engineering-from-scratch
pwd
ls -la

# History search (most useful shortcut you'll learn)
# Ctrl+R then type part of a previous command
# Press Ctrl+R again to cycle through matches

# Clear terminal
clear   # or Ctrl+L

# Cancel a running command
# Ctrl+C

# Suspend a running command (resume with fg)
# Ctrl+Z

### 步骤 2：管道（Piping）与重定向（Redirects）

管道（Piping）用于将多个命令连接起来。这是处理日志、过滤输出以及串联工具的标准方式，在实际开发中会频繁使用。

# Count how many times "loss" appears in a log
cat train.log | grep "loss" | wc -l

# Extract just the loss values from training output
grep "loss:" train.log | awk '{print $NF}' > losses.txt

# Watch a log file update in real time, filtering for errors
tail -f train.log | grep --line-buffered "ERROR"

# Sort experiments by final accuracy
grep "final_accuracy" results/*.log | sort -t= -k2 -n -r

# Redirect stdout and stderr to separate files
python train.py > output.log 2> errors.log

# Redirect both to the same file
python train.py > train_full.log 2>&1

你需要掌握的重定向符号：

| 符号 | 作用 |
|--------|-------------|
| `>` | 将标准输出（stdout）写入文件（覆盖） |
| `>>` | 将标准输出追加到文件 |
| `2>` | 将标准错误（stderr）写入文件 |
| `2>&1` | 将标准错误重定向到与标准输出相同的位置 |
| `\|` | 将前一个命令的标准输出作为下一个命令的标准输入（stdin） |

### 步骤 3：后台进程（Background Processes）

模型训练通常需要数小时。你肯定不希望一直开着终端窗口等待。

# Run in background (output still goes to terminal)
python train.py &

# Run in background, immune to hangup (closing terminal won't kill it)
nohup python train.py > train.log 2>&1 &

# Check what's running in background
jobs
ps aux | grep train.py

# Bring a background job to foreground
fg %1

# Kill a background process
kill %1
# or find its PID and kill that
kill $(pgrep -f "train.py")

`&`、`nohup` 与 `screen`/`tmux` 的区别：

| 方法 | 关闭终端后是否继续运行？ | 能否重新连接？ |
|--------|-------------------------|---------------|
| `command &` | 否 | 否 |
| `nohup command &` | 是 | 否（需查看日志文件） |
| `screen` / `tmux` | 是 | 是 |

对于运行时间超过几分钟的任务，建议使用 `tmux`。

### 步骤 4：终端复用器（tmux）

`tmux` 允许你创建包含多个窗格（panes）的持久化终端会话。这是管理训练任务最实用的工具。

# Install
# macOS
brew install tmux
# Ubuntu
sudo apt install tmux

# Start a named session
tmux new -s training

# Split horizontally
# Ctrl+B then "

# Split vertically
# Ctrl+B then %

# Navigate between panes
# Ctrl+B then arrow keys

# Detach (session keeps running)
# Ctrl+B then d

# Reattach
tmux attach -t training

# List sessions
tmux ls

# Kill a session
tmux kill-session -t training

典型的 AI 工作流会话示例：

tmux new -s train

# Pane 1: start training
python train.py --epochs 100 --lr 1e-4

# Ctrl+B, " to split, then run GPU monitor
watch -n1 nvidia-smi

# Ctrl+B, % to split vertically, tail the logs
tail -f logs/experiment.log

# Now detach with Ctrl+B, d
# SSH out, go get coffee, come back
# tmux attach -t train

### 步骤 5：使用 htop 和 nvtop 进行监控

# System processes (better than top)
htop

# GPU processes (if you have NVIDIA GPU)
# Install: sudo apt install nvtop (Ubuntu) or brew install nvtop (macOS)
nvtop

# Quick GPU check without nvtop
nvidia-smi

# Watch GPU usage update every second
watch -n1 nvidia-smi

# See which processes are using the GPU
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv

你会用到的 `htop` 快捷键：
- `F6` 或 `>`：按列排序（例如按内存排序以查找内存泄漏）
- `F5`：切换树状视图（查看子进程）
- `F9`：终止进程
- `/`：搜索进程名称

### 步骤 6：通过 SSH 连接远程 GPU 服务器

当你租用云 GPU 服务器（如 Lambda、RunPod、Vast.ai）时，需要通过安全外壳协议（SSH）进行连接。

# Basic connection
ssh user@gpu-box-ip

# With a specific key
ssh -i ~/.ssh/my_gpu_key user@gpu-box-ip

# Copy files to remote
scp model.pt user@gpu-box-ip:~/models/

# Copy files from remote
scp user@gpu-box-ip:~/results/metrics.json ./

# Sync a whole directory (faster for many files)
rsync -avz ./data/ user@gpu-box-ip:~/data/

# Port forward (access remote Jupyter/TensorBoard locally)
ssh -L 8888:localhost:8888 user@gpu-box-ip
# Now open localhost:8888 in your browser

# SSH config for convenience
# Add to ~/.ssh/config:
# Host gpu
#     HostName 192.168.1.100
#     User ubuntu
#     IdentityFile ~/.ssh/gpu_key
#
# Then just:
# ssh gpu

### 步骤 7：AI 开发常用命令别名（Aliases）

将这些内容添加到你的 `~/.bashrc` 或 `~/.zshrc` 文件中：

source phases/00-setup-and-tooling/10-terminal-and-shell/code/shell_aliases.sh

或者只复制你需要的部分。核心别名如下：

# GPU status at a glance
alias gpu='nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'

# Kill all Python training processes
alias killtraining='pkill -f "python.*train"'

# Quick virtual environment activate
alias ae='source .venv/bin/activate'

# Watch training loss
alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'

完整配置请参见 `code/shell_aliases.sh`。

### 步骤 8：AI 开发常用终端模式

这些命令在实际工作中会反复出现：

# Run training, log everything, notify when done
python train.py 2>&1 | tee train.log; echo "DONE" | mail -s "Training complete" you@email.com

# Compare two experiment logs side by side
diff <(grep "accuracy" exp1.log) <(grep "accuracy" exp2.log)

# Find the largest model files (clean up disk space)
find . -name "*.pt" -o -name "*.safetensors" | xargs du -h | sort -rh | head -20

# Download a model from Hugging Face
wget https://huggingface.co/model/resolve/main/model.safetensors

# Untar a dataset
tar xzf dataset.tar.gz -C ./data/

# Count lines in all Python files (see how big your project is)
find . -name "*.py" | xargs wc -l | tail -1

# Check disk space (training data fills disks fast)
df -h
du -sh ./data/*

# Environment variable check before training
env | grep -i cuda
env | grep -i torch


## 使用方法

在本课程中，各工具的具体使用场景如下：

| 工具 | 使用场景 |
|------|----------------|
| tmux | 每次训练运行（第 3 阶段及以后） |
| `tail -f` + `grep` | 监控训练日志 |
| `nohup` / `&` | 快速执行后台任务 |
| `htop` / `nvtop` | 排查训练缓慢及 OOM（内存溢出）错误 |
| SSH + `rsync` | 在云端 GPU 上进行开发 |
| 管道 (Piping) + 重定向 (Redirects) | 处理实验结果 |
| 命令别名 (Aliases) | 节省重复命令的输入时间 |

## 练习

1. 安装 tmux，创建一个包含三个窗格（panes）的会话（session）。在其中一个窗格中运行 `htop`，另一个运行 `watch -n1 date`，第三个运行 Python 脚本。随后执行分离（detach）与重新连接（reattach）操作。
2. 将 `code/shell_aliases.sh` 中的别名（aliases）添加至你的 Shell 配置文件，并通过 `source ~/.zshrc`（或 `~/.bashrc`）重新加载。
3. 使用 `for i in $(seq 1 100); do echo "epoch $i loss: $(echo "scale=4; 1/$i" | bc)"; sleep 0.1; done > fake_train.log` 生成一个模拟训练日志（training log），随后使用 `grep`、`tail` 和 `awk` 仅提取其中的损失值（loss values）。
4. 为你有访问权限的服务器配置一个 SSH 配置条目（SSH config entry）（或使用 `localhost` 练习相关语法）。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|----------------|----------------------|
| 命令行解释器 (Shell) | “终端” | 解释并执行你输入的命令的程序（如 bash、zsh、fish） |
| 终端复用工具 (tmux) | “终端复用器” | 允许在单个窗口中运行多个终端会话，并支持分离与重新连接的程序 |
| 管道 (Pipe) | “那个竖线符号” | 将前一个命令的输出作为后一个命令输入的 `\|` 操作符 |
| 进程标识符 (PID) | “进程 ID” | 分配给每个运行中进程的唯一编号，用于监控或终止该进程 |
| 忽略挂断命令 (nohup) | “不挂断” | 使命令不受挂断信号影响，从而在关闭终端后仍能继续运行 |
| 安全外壳协议 (SSH) | “连服务器” | 安全外壳协议，一种用于在远程机器上执行命令的加密通信协议 |