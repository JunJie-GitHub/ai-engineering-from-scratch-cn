# 3D 生成 (3D Generation)

> 3D 是 2D 到 3D 技术迁移 (2D-to-3D leverage) 优势最显著的模态。2023 年的突破性进展是 3D 高斯溅射 (3D Gaussian Splatting)。2024 至 2026 年的生成式技术推进在此基础上叠加了多视角扩散模型 (multi-view diffusion) 与 3D 重建 (3D reconstruction)，从而能够仅凭单个提示词或照片生成物体与场景。

**类型：** 学习
**语言：** Python
**前置条件：** 第 4 阶段（视觉），第 8 阶段 · 07（潜在扩散模型 (Latent Diffusion)）
**耗时：** 约 45 分钟

## 核心问题

3D 内容的制作一直是个难题：

- **表示方法 (Representation)。** 网格 (Meshes)、点云 (point clouds)、体素网格 (voxel grids)、符号距离场 (SDFs)、神经辐射场 (NeRFs)、3D 高斯 (3D Gaussians)。每种方法都有其权衡取舍。
- **数据稀缺 (Data scarcity)。** ImageNet 拥有 1400 万张图像。目前最大的高质量 3D 数据集（Objaverse-XL，2023 年）仅包含约 1000 万个物体，且大多数质量较低。
- **内存消耗 (Memory)。** 一个 512³ 的体素网格包含 1.28 亿个体素；一个实用的场景 NeRF 每条光线需要 100 万个采样点。生成任务比重建任务更为困难。
- **监督信号 (Supervision)。** 对于 2D 图像，你拥有完整的像素信息。而对于 3D，你通常只有少量 2D 视角图像，必须将其“提升”（lift）至 3D 空间。

2026 年的技术栈将这两个问题进行了分离。首先，利用扩散模型 (diffusion model) 生成 *2D 多视角图像*。其次，将 *3D 表示 (3D representation)*（通常为高斯溅射 (Gaussian splatting)）拟合到这些图像上。

## 核心概念

![3D生成：多视角扩散 + 3D重建](../assets/3d-generation.svg)

### 表示方法：3D高斯溅射（3D Gaussian Splatting）(Kerbl et al., 2023)

将场景表示为约100万个3D高斯（3D Gaussians）的点云。每个高斯包含59个参数：位置（3个）、协方差（covariance，6个，或四元数（quaternion）4个 + 缩放（scale）3个）、不透明度（opacity，1个）、球谐函数（spherical harmonics）颜色（3阶时为48个，0阶时为3个）。

渲染过程 = 投影 + Alpha合成（Alpha-compositing）。速度极快（在RTX 4090上1080p分辨率可达约100 fps）。具备可微分（Differentiable）特性。通过梯度下降算法对照真实图像（ground-truth photos）进行拟合。在消费级GPU上，完成一个场景的拟合仅需5至30分钟。

在此基础上，2023至2024年间涌现出两项重要创新：
- **生成式高斯溅射（Generative Gaussian Splats）。** LGM、LRM、InstantMesh 等模型能够直接从单张或少数几张图像预测出高斯点云。
- **4D高斯溅射（4D Gaussian Splatting）。** 为动态场景中的高斯点添加逐帧偏移量。

### 多视角扩散（Multi-view Diffusion）

对预训练的图像扩散模型（Image Diffusion Model）进行微调，使其能够根据文本提示或单张图像生成同一物体的多个一致视角。代表性工作包括 Zero123 (Liu et al., 2023)、MVDream (Shi et al., 2023)、SV3D (Stability, 2024) 和 CAT3D (Google, 2024)。通常围绕物体输出4到16个视角，随后通过高斯溅射或神经辐射场（Neural Radiance Field, NeRF）将其提升至3D空间。

### 文本到3D生成流程（Text-to-3D Pipelines）

| 模型 | 输入 | 输出 | 耗时 |
|-------|-------|--------|------|
| DreamFusion (2022) | 文本 | 基于分数蒸馏采样（SDS）的NeRF | 每个资产约1小时 |
| Magic3D | 文本 | 网格 + 纹理 | 约40分钟 |
| Shap-E (OpenAI, 2023) | 文本 | 隐式3D表示 | 约1分钟 |
| SJC / ProlificDreamer | 文本 | NeRF / 网格 | 约30分钟 |
| LRM (Meta, 2023) | 图像 | 三平面（Triplane） | 约5秒 |
| InstantMesh (2024) | 图像 | 网格 | 约10秒 |
| SV3D (Stability, 2024) | 图像 | 新视角 | 约2分钟 |
| CAT3D (Google, 2024) | 1-64张图像 | 3D NeRF | 约1分钟 |
| TripoSR (2024) | 图像 | 网格 | 约1秒 |
| Meshy 4 (2025) | 文本 + 图像 | 基于物理的渲染（PBR）网格 | 约30秒 |
| Rodin Gen-1.5 (2025) | 文本 + 图像 | PBR网格 | 约60秒 |
| Tencent Hunyuan3D 2.0 (2025) | 图像 | 网格 | 约30秒 |

2025-2026年发展方向：直接生成适用于游戏引擎的、带有PBR材质的文本到网格（Text-to-Mesh）模型。对于通用物体而言，多视角扩散作为中间步骤仍然是目前效果最佳的方案。

### NeRF（背景补充）

神经辐射场（Neural Radiance Field）(Mildenhall et al., 2020)。一个小型多层感知机（Multi-Layer Perceptron, MLP）接收 `(x, y, z, view direction)` 作为输入，并输出 `(color, density)`。通过沿光线积分完成渲染。其新视角合成质量优于基于网格的方法，但渲染速度慢100至1000倍。在大多数实时应用场景中已被高斯溅射取代，但在学术研究中仍占据主导地位。

## 动手构建

`code/main.py` 实现了一个简易的二维“高斯溅射（Gaussian Splatting）”拟合示例：将合成的目标图像（平滑渐变）表示为多个二维高斯溅射的叠加。通过梯度下降（Gradient Descent）优化位置、颜色和协方差，以匹配目标图像。你将看到两个核心操作：前向渲染（Forward Rendering，包含溅射与 Alpha 合成（Alpha Compositing））以及基于梯度下降的拟合。

### 步骤 1：二维高斯溅射

def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))

### 步骤 2：通过叠加溅射进行渲染

def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img

真实的三维高斯溅射会按深度对高斯体进行排序，并按顺序进行 Alpha 合成。我们的二维示例仅做简单叠加。

### 步骤 3：基于梯度下降进行拟合

for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)

## 常见陷阱

- **视角不一致（View Inconsistency）。** 如果独立生成 4 个视角且它们对物体结构的理解存在冲突，三维拟合结果会变得模糊。解决方法：采用共享注意力（Shared Attention）机制的多视角扩散模型（Multi-view Diffusion）。
- **背面幻觉（Back-side Hallucination）。** 单张图像转三维必须“脑补”未见的背面，生成质量波动极大。
- **高斯溅射爆炸（Gaussian Splat Explosion）。** 无约束训练会导致溅射数量激增至千万级并引发过拟合。必须引入密集化与剪枝（Densification & Pruning）启发式策略（源自 3D-GS 原论文）。
- **拓扑问题（Topology Issues）。** 从隐式场（Implicit Fields，如符号距离场 SDF）提取的网格常存在破洞或自相交。在交付前需运行网格重拓扑（Remeshing）工具（如 Blender 的体素重网格化）。
- **训练数据许可（Training Data License）。** Objaverse 数据集包含多种混合许可证，各模型的商用权限各不相同。

## 应用选型

| 任务 | 2026 年推荐方案 |
|------|-----------|
| 基于照片的场景重建 | 高斯溅射（3DGS, Gsplat, Scaniverse） |
| 游戏用文本生成三维物体 | Meshy 4 或 Rodin Gen-1.5（输出 PBR 材质） |
| 图像转三维 | Hunyuan3D 2.0, TripoSR, InstantMesh |
| 基于少量图像的新视角合成 | CAT3D, SV3D |
| 动态场景重建 | 4D 高斯溅射（4D Gaussian Splatting） |
| 虚拟化身 / 着装人体 | Gaussian Avatar, HUGS |
| 学术研究 / 前沿技术（SOTA） | 上周刚发布的最新模型 |

若要在游戏或电商管线中交付生产级三维资产：Meshy 4 或 Rodin Gen-1.5 可直接输出基于物理的渲染（PBR）网格，无缝接入 Unity / Unreal 引擎。

## 交付与部署

保存至 `outputs/skill-3d-pipeline.md`。该技能接收一份三维需求简报（输入：文本 / 单张图像 / 少量图像；输出：网格 / 溅射 / 神经辐射场（NeRF）；用途：渲染 / 游戏 / VR），并输出以下内容：处理管线（多视角扩散 + 拟合，或直接网格生成模型）、基础模型、迭代预算、拓扑后处理方案以及所需的材质通道。

## 练习

1. **简单。** 分别使用 4、16、64 个高斯（Gaussians）运行 `code/main.py`。报告最终的均方误差（MSE）与目标值的对比结果。
2. **中等。** 扩展至支持彩色高斯（RGB）。验证重建结果是否与目标颜色模式一致。
3. **困难。** 使用 `gsplat` 或 `Nerfstudio`，基于 50 张照片的采集数据重建真实物体。报告拟合耗时以及在预留视图上的最终结构相似性（SSIM）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| 3D高斯溅射（3D Gaussian Splatting） | “3DGS” | 将场景表示为 3D 高斯点云；通过可微的 Alpha 合成（alpha-composite）进行渲染。 |
| 神经辐射场（NeRF） | “神经辐射场” | 多层感知机（MLP），用于输出 3D 空间某点的颜色与密度；通过光线积分（ray integration）进行渲染。 |
| 三平面（Triplane） | “三个二维平面” | 将 3D 特征分解为三个与坐标轴对齐的 2D 特征网格；计算成本低于体素表示。 |
| 分数蒸馏采样（SDS） | “分数蒸馏采样” | 利用 2D 扩散模型的分数（score）作为伪梯度（pseudo-gradient）来训练 3D 模型。 |
| 多视角扩散（Multi-view diffusion） | “一次性生成多个视角” | 能够输出一批视角一致的相机视图的扩散模型。 |
| 基于物理的渲染（PBR） | “基于物理的渲染” | 包含反照率（albedo）、粗糙度、金属度、法线通道的材质表示。 |
| 致密化（Densification） | “增加高斯数量” | 3DGS 训练启发式策略：在梯度较高的区域对高斯进行分裂（split）或克隆（clone）。 |

## 生产环境说明：3D 领域尚未形成统一的底层架构

与图像（潜在扩散（Latent Diffusion）+ DiT）和视频（时空 DiT（Spatiotemporal DiT））不同，3D 领域在 2026 年尚未出现单一主导的运行时（runtime）。生产环境的决策树会根据表示方法（representation）产生分支：

- **NeRF / 三平面（Triplane）。** 推理过程为光线步进（ray-marching）加上每个采样点的一次 MLP 前向传播。渲染一张 512² 的图像需要数百万次 MLP 前向计算。需对光线采样进行激进批处理；可应用 SDPA/xformers 进行加速。
- **多视角扩散 + LRM 重建。** 两阶段流水线。第一阶段（多视角 DiT）是一个扩散模型服务器，与第 07 课类似。第二阶段（LRM Transformer）是对所有视图进行一次性前向传播。整体延迟特征为“扩散生成 + 一次性推理”——需据此为各阶段选择合适的服务原语（serving primitives）。
- **SDS / DreamFusion。** 属于针对单个资产的优化过程，而非推理。应构建批处理作业（jobs），而非请求处理器（request handlers）。

对于大多数 2026 年的产品而言，正确的架构是“按需运行多视角扩散模型，异步重建为 3DGS，并提供 3DGS 用于实时查看”。这种设计将工作负载清晰地拆分至 GPU 推理服务器（快速）与离线优化器（慢速）之间。

## 延伸阅读

- [Mildenhall 等人（2020）。NeRF：将场景表示为神经辐射场（Neural Radiance Fields）](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl 等人（2023）。用于实时辐射场渲染的 3D 高斯溅射（3D Gaussian Splatting）](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole 等人（2022）。DreamFusion：基于二维扩散模型（2D Diffusion）的文本到三维（Text-to-3D）生成](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu 等人（2023）。Zero-1-to-3：零样本（Zero-shot）单图像到三维物体](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi 等人（2023）。MVDream](https://arxiv.org/abs/2308.16512) — 多视角扩散（Multi-view Diffusion）。
- [Hong 等人（2023）。LRM：面向单图像到三维的大型重建模型（Large Reconstruction Model）](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao 等人（2024）。CAT3D：利用多视角扩散模型（Multi-View Diffusion Models）创建任意三维内容](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI（2024）。Stable Video 3D（SV3D）](https://stability.ai/research/sv3d) — SV3D。