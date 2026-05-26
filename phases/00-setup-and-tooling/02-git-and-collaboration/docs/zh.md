# Git 与协作

> 版本控制 (Version Control) 并非可选项。你在此处进行的每一次实验、构建的每一个模型、完成的每一节课都会被追踪记录。

**类型：** 学习
**语言：** --
**前置条件：** 第 0 阶段，第 01 课
**时长：** 约 30 分钟

## 学习目标

- 配置 Git 身份标识 (Git identity)，并使用包含 add、commit 和 push 的日常开发工作流
- 创建并合并分支 (branch) 以进行隔离实验，同时确保不破坏 main 分支
- 编写 `.gitignore` 文件，以排除模型检查点 (model checkpoints) 和大型二进制文件 (binary files)
- 使用 `git log` 浏览提交历史 (commit history)，以了解项目的演进过程

## 问题

在接下来的 20 个阶段中，你将编写数百个代码文件。如果没有版本控制 (Version Control)，你将面临丢失工作成果、造成无法回退的破坏，以及无法与他人协作的风险。

Git 是核心工具，GitHub 是代码托管平台。本课仅涵盖本课程所需的内容，不作多余展开。

## 概念

sequenceDiagram
    participant WD as Working Directory
    participant SA as Staging Area
    participant LR as Local Repo
    participant R as Remote (GitHub)
    WD->>SA: git add
    SA->>LR: git commit
    LR->>R: git push
    R->>LR: git fetch
    LR->>WD: git pull

牢记以下三点：
1. 频繁提交（`git commit`）
2. 推送至远程仓库（`git push`）
3. 为实验创建分支（`git checkout -b experiment`）

## 构建

### 步骤 1：配置 Git

git config --global user.name "Your Name"
git config --global user.email "you@example.com"

### 步骤 2：日常工作流 (workflow)

git status
git add file.py
git commit -m "Add perceptron implementation"
git push origin main

### 步骤 3：为实验创建分支 (branching)

git checkout -b experiment/new-optimizer

# ... make changes, commit ...

git checkout main
git merge experiment/new-optimizer

### 步骤 4：使用本课程代码仓库 (repository)

git clone https://github.com/rohitg00/ai-engineering-from-scratch.git
cd ai-engineering-from-scratch

git checkout -b my-progress
# work through lessons, commit your code
git push origin my-progress


## 使用方法

本课程只需掌握以下命令：

| 命令 | 使用场景 |
|---------|------|
| `git clone` | 获取课程代码仓库（repo） |
| `git add` + `git commit` | 保存工作进度 |
| `git push` | 将代码备份至 GitHub |
| `git checkout -b` | 在不影响主分支（main）的情况下尝试新功能 |
| `git log --oneline` | 查看提交历史 |

仅此而已。本课程无需使用变基（rebase）、拣选（cherry-pick）或子模块（submodules）。

## 练习

1. 克隆 (Clone) 此仓库 (Repository)，创建一个名为 `my-progress` 的分支 (Branch)，新建一个文件，提交 (Commit) 并推送 (Push)
2. 创建 `.gitignore` 文件以排除模型检查点文件 (Model Checkpoint Files)（`.pt`、`.pth`、`.safetensors`）
3. 使用 `git log --oneline` 查看此仓库的提交历史 (Commit History)，了解课程是如何添加的

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 提交 (Commit) | “保存” | 项目在某一时间点的完整快照 |
| 分支 (Branch) | “副本” | 指向某个提交的指针，会随着你的开发工作不断向前推进 |
| 合并 (Merge) | “合并代码” | 将一个分支的更改提取并应用到另一个分支上 |
| 远程仓库 (Remote) | “云端” | 托管在其他位置（如 GitHub、GitLab）的仓库副本 |