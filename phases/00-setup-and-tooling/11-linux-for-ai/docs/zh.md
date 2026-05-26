# 面向人工智能 (AI) 的 Linux

> 大多数 AI 系统都运行在 Linux 上。你需要掌握足够的知识，以免在操作中受阻。

**Type:** 学习
**Languages:** --
**Prerequisites:** Phase 0, Lesson 01
**Time:** 约 30 分钟

## 学习目标

- 浏览 Linux 文件系统（Linux file system）并通过命令行（command line）执行核心文件操作
- 使用 `chmod` 和 `chown` 管理文件权限（file permissions），以解决权限拒绝（Permission denied）报错
- 使用 `apt` 安装系统软件包（system packages），并为 AI 开发（AI work）配置全新的 GPU 服务器（GPU box）
- 识别 macOS 与 Linux 之间的常见差异（macOS-to-Linux differences），避免在远程服务器（remote machines）上开发时踩坑

## 问题

你平时可能在 macOS 或 Windows 上进行开发。但当你通过 SSH (Secure Shell) 连接到云端 GPU (Graphics Processing Unit) 服务器、租用 Lambda 实例，或是启动一台 EC2 (Elastic Compute Cloud) 实例时，你面对的将是 Ubuntu 系统。终端 (Terminal) 是你唯一的操作界面。这里没有 Finder，没有资源管理器 (Explorer)，也没有图形用户界面 (GUI)。如果你无法通过命令行 (Command Line) 导航文件系统、安装软件包或管理进程，你就只能一边在谷歌上搜索“如何在 Linux 中解压文件”，一边白白烧钱支付闲置的 GPU 机时。

这是一份生存指南。它精准涵盖了在远程 Linux 机器上进行人工智能 (AI) 工作所需的一切操作技能。仅此而已，绝无冗余。

## 文件系统布局 (File System Layout)

Linux 将所有内容组织在单个根目录（Root Directory）`/` 下。系统中不存在 `C:\` 或 `/Volumes`。你实际会接触到的目录如下：

graph TD
    root["/"] --> home["home/your-username/<br/>Your files — clone repos, run training"]
    root --> tmp["tmp/<br/>Temporary files, cleared on reboot"]
    root --> usr["usr/<br/>System programs and libraries"]
    root --> etc["etc/<br/>Config files"]
    root --> varlog["var/log/<br/>Logs — check when something breaks"]
    root --> mnt["mnt/ or /media/<br/>External drives and volumes"]
    root --> proc["proc/ and /sys/<br/>Virtual files — kernel and hardware info"]

你的主目录（Home Directory）是 `~` 或 `/home/your-username`。你几乎所有的操作都在这里进行。

## 核心命令

这 15 个命令涵盖了你在远程 GPU 服务器 (Remote GPU Server) 上 95% 的日常操作。

### 目录导航 (Directory Navigation)

pwd                         # Where am I?
ls                          # What's here?
ls -la                      # What's here, including hidden files with details?
cd /path/to/dir             # Go there
cd ~                        # Go home
cd ..                       # Go up one level

### 文件与目录 (Files and Directories)

mkdir my-project            # Create a directory
mkdir -p a/b/c              # Create nested directories in one shot

cp file.txt backup.txt      # Copy a file
cp -r src/ src-backup/      # Copy a directory (recursive)

mv old.txt new.txt          # Rename a file
mv file.txt /tmp/           # Move a file

rm file.txt                 # Delete a file (no trash, it's gone)
rm -rf my-dir/              # Delete a directory and everything inside

`rm -rf` 操作是永久性的，无法撤销。在按下回车键 (Enter) 前，请务必仔细核对路径 (Path)。

### 查看文件 (Reading Files)

cat file.txt                # Print entire file
head -20 file.txt           # First 20 lines
tail -20 file.txt           # Last 20 lines
tail -f log.txt             # Follow a log file in real time (Ctrl+C to stop)
less file.txt               # Scroll through a file (q to quit)

### 搜索 (Searching)

grep "error" training.log           # Find lines containing "error"
grep -r "learning_rate" .           # Search all files in current directory
grep -i "cuda" config.yaml          # Case-insensitive search

find . -name "*.py"                 # Find all Python files under current dir
find . -name "*.ckpt" -size +1G     # Find checkpoint files larger than 1GB


## 权限 (Permissions)

Linux 中的每个文件都有一个所有者（owner）和权限位（permission bits）。当脚本无法执行或无法向目录写入内容时，你通常会遇到这个问题。

ls -l train.py
# -rwxr-xr-- 1 user group 2048 Mar 19 10:00 train.py
#  ^^^             owner permissions: read, write, execute
#     ^^^          group permissions: read, execute
#        ^^        everyone else: read only

常见解决方法：

chmod +x train.sh           # Make a script executable
chmod 755 deploy.sh         # Owner: full, others: read+execute
chmod 644 config.yaml       # Owner: read+write, others: read only

chown user:group file.txt   # Change who owns a file (needs sudo)

当系统提示“权限被拒绝（Permission denied）”时，几乎总是权限问题。使用 `chmod +x` 或 `sudo` 通常就能解决大多数情况。

## 软件包管理 (apt)

Ubuntu 使用 `apt`。这是安装系统级软件（system-level software）的方法。

sudo apt update             # Refresh the package list (always do this first)
sudo apt install -y htop    # Install a package (-y skips confirmation)
sudo apt install -y build-essential  # C compiler, make, etc. Needed by many Python packages
sudo apt install -y tmux    # Terminal multiplexer (keep sessions alive after disconnect)

apt list --installed        # What's installed?
sudo apt remove htop        # Uninstall

在全新的 GPU 服务器（GPU box）上，你通常需要安装以下常用软件包：

sudo apt update && sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    tmux \
    htop \
    unzip \
    python3-venv


## 用户与 sudo

您通常以普通用户 (regular user) 身份登录。某些操作需要 root（管理员）权限 (root/admin access)。

whoami                      # What user am I?
sudo command                # Run a single command as root
sudo su                     # Become root (exit to go back, use sparingly)

在云 GPU 实例 (cloud GPU instances) 上，您通常是唯一用户，且已具备 sudo 权限。请勿以 root 身份运行所有命令。仅在必要时使用 sudo。

## 进程与 systemd

当训练 (training) 卡死，或你需要检查正在运行的进程 (process) 时：

htop                        # Interactive process viewer (q to quit)
ps aux | grep python        # Find running Python processes
kill 12345                  # Gracefully stop process with PID 12345
kill -9 12345               # Force kill (use when graceful doesn't work)
nvidia-smi                  # GPU processes and memory usage

systemd 负责管理服务 (services)（后台守护进程 (background daemons)）。若需运行推理服务器 (inference servers)，你将用到它：

sudo systemctl start nginx          # Start a service
sudo systemctl stop nginx           # Stop it
sudo systemctl restart nginx        # Restart it
sudo systemctl status nginx         # Check if it's running
sudo systemctl enable nginx         # Start automatically on boot


## 磁盘空间

GPU 主机 (GPU machines) 的磁盘空间 (disk space) 通常有限。模型 (models) 和数据集 (datasets) 会迅速将其占满。

df -h                       # Disk usage for all mounted drives
df -h /home                 # Disk usage for /home specifically

du -sh *                    # Size of each item in current directory
du -sh ~/.cache             # Size of your cache (pip, huggingface models land here)
du -sh /data/checkpoints/   # Check how big your checkpoints are

# Find the biggest space hogs
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20

常用的节省空间方法：

# Clear pip cache
pip cache purge

# Clear apt cache
sudo apt clean

# Remove old checkpoints you don't need
rm -rf checkpoints/epoch_01/ checkpoints/epoch_02/


## 网络 (Networking)

你将在命令行（command line）中下载模型、传输文件并调用 API。

# Download files
wget https://example.com/model.bin                   # Download a file
curl -O https://example.com/data.tar.gz              # Same thing with curl
curl -s https://api.example.com/health | python3 -m json.tool  # Hit an API, pretty-print JSON

# Transfer files between machines
scp model.bin user@remote:/data/                     # Copy file to remote machine
scp user@remote:/data/results.csv .                  # Copy file from remote to local
scp -r user@remote:/data/checkpoints/ ./local-dir/   # Copy directory

# Sync directories (faster than scp for large transfers, resumes on failure)
rsync -avz --progress ./data/ user@remote:/data/
rsync -avz --progress user@remote:/results/ ./results/

对于大型传输任务，请优先使用 `rsync` 而非 `scp`。它仅传输发生变化的字节，并能妥善处理连接中断的情况。

## tmux：保持会话活跃

当你通过 SSH 连接到远程主机时，合上笔记本电脑会导致训练任务 (training run) 中断。tmux 可以避免这一问题。

tmux new -s train           # Start a new session named "train"
# ... start your training, then:
# Ctrl+B, then D            # Detach (training keeps running)

tmux ls                     # List sessions
tmux attach -t train        # Reattach to session

# Inside tmux:
# Ctrl+B, then %            # Split pane vertically
# Ctrl+B, then "            # Split pane horizontally
# Ctrl+B, then arrow keys   # Switch between panes

务必始终在 tmux 中运行长时间的训练任务。切记。

## 面向 Windows 用户的适用于 Linux 的 Windows 子系统 2 (WSL2)

如果你使用的是 Windows 系统，适用于 Linux 的 Windows 子系统 2 (WSL2) 可以在无需双系统启动 (dual-booting) 的情况下为你提供真实的 Linux 环境。

# In PowerShell (admin)
wsl --install -d Ubuntu-24.04

# After restart, open Ubuntu from Start menu
sudo apt update && sudo apt upgrade -y

WSL2 运行着真实的 Linux 内核 (Linux kernel)。本课程中的所有内容均可在其中正常运行。在 WSL 内部，你的 Windows 文件位于 `/mnt/c/Users/YourName/` 路径下。

GPU 直通 (GPU passthrough) 功能可通过在 Windows 端安装 NVIDIA 驱动程序 (NVIDIA drivers) 来实现。请安装 Windows 版本的 NVIDIA 驱动（而非 Linux 版本），这样 CUDA 即可在 WSL2 内部直接使用。

## 常见陷阱 (Gotchas)：从 macOS 到 Linux

如果你从 macOS 转过来，可能会让你踩坑的地方：

| macOS | Linux | 备注 |
|-------|-------|------|
| `brew install` | `sudo apt install` | 软件包名称有时不同。`brew install htop` 与 `sudo apt install htop` 效果相同，但 `brew install readline` 与 `sudo apt install libreadline-dev` 则不同。 |
| `open file.txt` | `xdg-open file.txt` | 但在远程主机上通常没有图形用户界面 (GUI)。请使用 `cat` 或 `less`。 |
| `pbcopy` / `pbpaste` | 不可用 | 通过 SSH 连接时，无法使用管道与剪贴板进行交互。 |
| `~/.zshrc` | `~/.bashrc` | macOS 默认使用 zsh。大多数 Linux 服务器使用 bash。 |
| `/opt/homebrew/` | `/usr/bin/`, `/usr/local/bin/` | 可执行文件 (Binaries) 的存放路径不同。 |
| `sed -i '' 's/a/b/' file` | `sed -i 's/a/b/' file` | macOS 的 `sed` 命令在 `-i` 后需要跟一个空字符串。Linux 则不需要。 |
| 不区分大小写的文件系统 (Case-insensitive filesystem) | 区分大小写的文件系统 (Case-sensitive filesystem) | 在 Linux 上，`Model.py` 和 `model.py` 是两个不同的文件。 |
| 换行符 (Line endings) `\n` | 换行符 `\n` | 两者相同。但 Windows 使用 `\r\n`，这会导致 bash 脚本出错。运行 `dos2unix` 即可修复。 |

## 快速参考卡

Navigation:     pwd, ls, cd, find
Files:          cp, mv, rm, mkdir, cat, head, tail, less
Search:         grep, find
Permissions:    chmod, chown, sudo
Packages:       apt update, apt install
Processes:      htop, ps, kill, nvidia-smi
Services:       systemctl start/stop/restart/status
Disk:           df -h, du -sh
Network:        curl, wget, scp, rsync
Sessions:       tmux new/attach/detach


## 练习

1. 通过 SSH 连接到任意 Linux 机器（或打开 WSL2），并进入主目录 (home directory)。创建一个项目文件夹，使用 `touch` 在其中创建三个空文件，然后使用 `ls -la` 列出它们。
2. 使用 `apt` 安装 `htop`，运行该程序，并找出占用内存 (memory) 最多的进程 (process)。
3. 启动一个 `tmux` 会话 (session)，在其中运行 `sleep 300`，执行分离 (detach) 操作，列出所有会话，最后重新连接 (reattach)。
4. 使用 `df -h` 检查可用磁盘空间 (disk space)，然后使用 `du -sh ~/.cache/*` 找出缓存 (cache) 中占用空间的内容。
5. 使用 `scp` 将文件从本地机器传输到远程机器，随后使用 `rsync` 执行相同的传输操作，并对比两者的使用体验。