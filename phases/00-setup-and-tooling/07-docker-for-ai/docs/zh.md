# 面向 AI 的 Docker

> 容器 (Containers) 让“在我机器上能运行”成为历史。

**类型：** 构建
**语言：** Python
**前置条件：** 第 0 阶段，第 01 和 03 课
**时长：** 约 60 分钟

## 学习目标

- 基于 Dockerfile 构建支持 GPU 的 Docker 镜像 (Docker image)，集成 CUDA、PyTorch 及 AI 库 (AI libraries)
- 将主机目录 (host directories) 挂载为卷 (volumes)，以便在容器重建 (container rebuilds) 期间持久化模型 (models)、数据集 (datasets) 和代码
- 配置 NVIDIA Container Toolkit，以便在容器 (containers) 内部暴露 GPU
- 使用 Docker Compose 编排多服务 AI 应用 (multi-service AI applications)（推理服务器 (inference server) + 向量数据库 (vector database)）

## 问题

你在笔记本电脑上使用 PyTorch 2.3、CUDA 12.4 和 Python 3.12 训练了一个模型 (Model)。你的同事使用的是 PyTorch 2.1、CUDA 11.8 和 Python 3.10。你的模型在他们的机器上崩溃了，而你的 Dockerfile 在两台机器上都能正常运行。

AI 项目 (AI Projects) 往往是依赖地狱 (Dependency Nightmares)。典型的技术栈 (Tech Stack) 包括 Python、PyTorch、CUDA 驱动程序 (CUDA Drivers)、cuDNN、系统级 C 库 (System-level C Libraries)，以及像 flash-attn 这样需要精确匹配编译器版本 (Compiler Versions) 的专用软件包。Docker 将所有这些打包成一个单一的镜像 (Image)，确保在任何环境中都能完全一致地运行。

## 概念

Docker 将你的代码、运行时环境、依赖库和系统工具打包到一个称为容器（Container）的隔离单元中。你可以将其视为一台轻量级虚拟机（Virtual Machine），但它共享宿主机操作系统内核（Host OS Kernel）而非运行独立内核，因此启动仅需数秒而非数分钟。

graph TD
    subgraph without["Without Docker"]
        A1["Your machine<br/>Python 3.12<br/>CUDA 12.4<br/>PyTorch 2.3"] -->|crashes| X1["???"]
        A2["Their machine<br/>Python 3.10<br/>CUDA 11.8<br/>PyTorch 2.1"] -->|crashes| X2["???"]
        A3["Server<br/>Python 3.11<br/>CUDA 12.1<br/>PyTorch 2.2"] -->|crashes| X3["???"]
    end

    subgraph with_docker["With Docker — Same image everywhere"]
        B1["Your machine<br/>Python 3.12 | CUDA 12.4<br/>PyTorch 2.3 | Your code"]
        B2["Their machine<br/>Python 3.12 | CUDA 12.4<br/>PyTorch 2.3 | Your code"]
        B3["Server<br/>Python 3.12 | CUDA 12.4<br/>PyTorch 2.3 | Your code"]
    end

### 为什么 AI 项目比大多数项目更需要 Docker

1. **GPU 驱动程序（GPU Drivers）非常脆弱。** CUDA 12.4 的代码无法在 CUDA 11.8 上运行。Docker 将 CUDA 工具包（CUDA Toolkit）隔离在容器内部，同时通过 NVIDIA Container Toolkit 共享宿主机的 GPU 驱动程序。

2. **模型权重（Model Weights）体积庞大。** 一个 70 亿（7B）参数的模型在 fp16 精度下占用 14 GB 空间。你肯定不希望每次重新构建镜像时都重新下载它。Docker 卷（Docker Volumes）允许你从宿主机直接挂载模型目录。

3. **多服务架构（Multi-service Architectures）十分常见。** 一个真正的 AI 应用不仅仅是一个 Python 脚本。它通常包含推理服务器（Inference Server）、用于检索增强生成（RAG）的向量数据库（Vector Database），可能还有一个 Web 前端（Web Frontend）。Docker Compose 只需一条命令即可统一编排所有这些服务。

### 核心词汇

| 术语 | 含义 |
|------|---------------|
| 镜像（Image） | 只读模板。相当于你的配方。由 Dockerfile 构建而成。 |
| 容器（Container） | 镜像的运行实例。相当于你的厨房。 |
| Dockerfile | 构建镜像的指令集。逐层构建。 |
| 卷（Volume） | 持久化存储，在容器重启后数据依然保留。 |
| docker-compose | 用于在 YAML 中定义多容器应用的工具。 |

### AI 中常见的容器模式

Dev Container
  Full toolkit. Editor support. Jupyter. Debugging tools.
  Used during development and experimentation.

Training Container
  Minimal. Just the training script and dependencies.
  Runs on GPU clusters. No editor, no Jupyter.

Inference Container
  Optimized for serving. Small image. Fast cold start.
  Runs behind a load balancer in production.


## 构建

### 步骤 1：安装 Docker

# macOS
brew install --cask docker
open /Applications/Docker.app

# Ubuntu
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in for group change to take effect

验证：

docker --version
docker run hello-world

### 步骤 2：安装 NVIDIA Container Toolkit（配备 NVIDIA GPU 的 Linux 系统）

这使得 Docker 容器能够访问你的 GPU。macOS 和 Windows (WSL2) 用户可以跳过此步骤；在这些平台上，Docker Desktop 处理 GPU 直通 (GPU Passthrough) 的方式有所不同。

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

在容器内测试 GPU 访问权限：

docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

如果能看到你的 GPU 信息，说明工具包已正常工作。

### 步骤 3：了解基础镜像 (Base Images)

选择合适的基础镜像可以节省数小时的调试时间。

nvidia/cuda:12.4.1-devel-ubuntu22.04
  Full CUDA toolkit. Compilers included.
  Use for: building packages that need nvcc (flash-attn, bitsandbytes)
  Size: ~4 GB

nvidia/cuda:12.4.1-runtime-ubuntu22.04
  CUDA runtime only. No compilers.
  Use for: running pre-built code
  Size: ~1.5 GB

pytorch/pytorch:2.3.1-cuda12.4-cudnn9-runtime
  PyTorch pre-installed on top of CUDA.
  Use for: skipping the PyTorch install step
  Size: ~6 GB

python:3.12-slim
  No CUDA. CPU only.
  Use for: inference on CPU, lightweight tools
  Size: ~150 MB

### 步骤 4：为 AI 开发编写 Dockerfile

以下是位于 `code/Dockerfile` 的 Dockerfile。我们来逐步解析：

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    jupyter \
    transformers \
    datasets \
    accelerate \
    safetensors

WORKDIR /workspace

VOLUME ["/workspace", "/models"]

EXPOSE 8888

CMD ["python"]

构建镜像：

docker build -t ai-dev -f phases/00-setup-and-tooling/07-docker-for-ai/code/Dockerfile .

首次构建需要一些时间（下载 CUDA 基础镜像与 PyTorch）。后续构建将使用缓存层 (Cached Layers)。

运行容器：

docker run --rm -it --gpus all \
    -v $(pwd):/workspace \
    -v ~/models:/models \
    ai-dev python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

在容器内运行 Jupyter：

docker run --rm -it --gpus all \
    -v $(pwd):/workspace \
    -v ~/models:/models \
    -p 8888:8888 \
    ai-dev jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

### 步骤 5：为数据和模型配置数据卷挂载 (Volume Mounts)

数据卷挂载对于 AI 工作至关重要。如果没有它们，当容器停止时，你下载的 14 GB 模型将会消失。

# Mount your code
-v $(pwd):/workspace

# Mount a shared models directory
-v ~/models:/models

# Mount datasets
-v ~/datasets:/data

在你的训练脚本中，从挂载路径加载模型：

from transformers import AutoModel

model = AutoModel.from_pretrained("/models/llama-7b")

模型实际存储在你的主机文件系统上。你可以随意重建容器，而无需重新下载。

### 步骤 6：使用 Docker Compose 部署多服务 AI 应用

一个真实的检索增强生成 (Retrieval-Augmented Generation, RAG) 应用需要推理服务器 (Inference Server) 和向量数据库 (Vector Database)。Docker Compose 只需一条命令即可同时运行两者。

参见 `code/docker-compose.yml`：

services:
  ai-dev:
    build:
      context: .
      dockerfile: Dockerfile
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ../../../:/workspace
      - ~/models:/models
      - ~/datasets:/data
    ports:
      - "8888:8888"
    stdin_open: true
    tty: true
    command: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

  qdrant:
    image: qdrant/qdrant:v1.12.5
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:

启动所有服务：

cd phases/00-setup-and-tooling/07-docker-for-ai/code
docker compose up -d

现在，你的 AI 开发容器可以通过服务名称访问位于 `http://qdrant:6333` 的向量数据库。Docker Compose 会自动创建一个共享网络 (Shared Network)。

在 AI 容器内测试连接：

from qdrant_client import QdrantClient

client = QdrantClient(host="qdrant", port=6333)
print(client.get_collections())

停止所有服务：

docker compose down

添加 `-v` 参数以同时删除 qdrant 数据卷：

docker compose down -v

### 步骤 7：AI 开发常用的 Docker 命令

# List running containers
docker ps

# List all images and their sizes
docker images

# Remove unused images (reclaim disk space)
docker system prune -a

# Check GPU usage inside a running container
docker exec -it <container_id> nvidia-smi

# Copy a file from container to host
docker cp <container_id>:/workspace/results.csv ./results.csv

# View container logs
docker logs -f <container_id>


## 使用方法

你现在已经拥有了一个可复现的 AI 开发环境 (Reproducible AI Development Environment)。在本课程的后续部分中：

- 使用 `docker compose up` 同时启动你的开发环境和向量数据库 (Vector Database)
- 将你的代码、模型和数据挂载为数据卷 (Volumes)，以确保在重建容器时不会丢失任何内容
- 当课程需要新的 Python 包 (Python Package) 时，将其添加到 Dockerfile 中并重新构建
- 与团队成员共享你的 Dockerfile。他们将获得完全相同的环境。

### 没有 GPU？

移除 `--gpus all` 标志和 NVIDIA 部署块 (Deploy Block)。该容器仍然适用于基于 CPU 的课程。PyTorch 会检测到 CUDA 的缺失并自动回退到 CPU。

## 练习

1. 构建 Dockerfile，并在容器（container）内运行 `python -c "import torch; print(torch.__version__)"`
2. 启动 Docker Compose 服务栈（docker-compose stack），并验证 AI 容器（AI container）能否通过 `http://qdrant:6333/collections` 访问 Qdrant
3. 将 `flask` 添加至 Dockerfile，重新构建，并在 5000 端口（port）上运行一个简单的 API 服务器（API server）。使用 `-p 5000:5000` 映射端口
4. 使用 `docker images` 查看镜像大小（image size）。尝试将基础镜像（base image）从 `devel` 切换为 `runtime` 并比较大小

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 容器 (Container) | “轻量级虚拟机” | 使用宿主机内核的隔离进程，拥有独立的文件系统和网络 |
| 镜像层 (Image layer) | “缓存步骤” | `Dockerfile` 中的每条指令都会创建一个层。未更改的层会被缓存，因此重新构建速度很快。 |
| NVIDIA 容器工具包 (NVIDIA Container Toolkit) | “Docker 里的 GPU” | 一种运行时钩子，通过 `--gpus` 标志将宿主机的 GPU 暴露给容器 |
| 卷挂载 (Volume mount) | “共享文件夹” | 将宿主机上的目录映射到容器内。容器停止后，其中的更改仍会保留。 |
| 基础镜像 (Base image) | “起点” | `Dockerfile` 基于 `FROM` 指令指定的镜像进行构建。它决定了预安装了哪些内容。 |