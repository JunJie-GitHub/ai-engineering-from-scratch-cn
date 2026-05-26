# 从零开始实现 3D 高斯溅射 (3D Gaussian Splatting)

> 场景本质上是由数百万个 3D 高斯函数 (3D Gaussians) 构成的点云。每个高斯函数都具备位置、朝向、缩放比例、不透明度，以及随观察视角变化的颜色。只需将它们进行光栅化 (Rasterization)，并在光栅化过程中执行反向传播 (Backpropagation)，即可大功告成。

**类型：** 构建实践
**编程语言：** Python
**前置知识：** 第四阶段第 13 课（3D 视觉与神经辐射场 (NeRF)）、第一阶段第 12 课（张量运算 (Tensor Operations)）、第四阶段第 10 课（扩散模型基础 (Diffusion Basics)，可选）
**预计耗时：** 约 90 分钟

## 学习目标

- 解释为何 3D 高斯溅射在 2026 年取代 NeRF，成为照片级真实感 3D 重建 (Photorealistic 3D Reconstruction) 的工业界默认方案
- 列出每个高斯函数的六个参数（位置、旋转四元数 (Rotation Quaternion)、缩放比例、不透明度、球谐函数颜色 (Spherical Harmonics Colour)、可选特征），并说明每个参数贡献的浮点数数量
- 使用 `alpha` 混合 (Alpha Compositing) 从零开始实现一个 2D 高斯溅射光栅化器，并展示 3D 情况如何投影到相同的循环逻辑中
- 使用 `nerfstudio`、`gsplat` 或 `SuperSplat` 工具，基于 20-50 张照片重建场景，并将其导出为 `KHR_gaussian_splatting` glTF 扩展 (glTF Extension) 格式或 OpenUSD 26.03 的 `UsdVolParticleField3DGaussianSplat` 模式 (Schema)

## 核心问题

神经辐射场 (NeRF) 将场景存储为多层感知机 (MLP) 的权重。渲染每个像素都需要沿光线进行数百次 MLP 查询。训练耗时数小时，渲染耗时数秒，且权重无法直接编辑——如果你想移动场景中的一把椅子，就必须重新训练模型。

3D 高斯溅射 (Kerbl, Kopanas, Leimkühler, Drettakis, SIGGRAPH 2023) 彻底改变了这一现状。场景被表示为一组显式的 3D 高斯函数。渲染过程直接在 GPU 上进行光栅化 (GPU Rasterization)，帧率可达 100+ fps。训练仅需几分钟。编辑也是直接的：平移一部分高斯函数，就等于移动了椅子。到 2026 年，Khronos Group 已正式批准了用于高斯溅射的 glTF 扩展标准，OpenUSD 26.03 也内置了高斯溅射模式，Zillow 和 Apartments.com 等房地产平台已采用该技术进行房源渲染，且大多数关于 3D 重建的最新研究论文都是基于核心 3DGS 思想的变体。

其思维模型 (Mental Model) 非常直观，但涉及的数学细节较多，因此大多数入门教程往往直接从光栅化讲起，而跳过投影和球谐函数部分。本课程将完整构建整个流程——先从 2D 版本入手，再扩展到 3D 实现。

## 核心概念

### 高斯椭球携带的参数

一个三维高斯椭球（3D Gaussian）是空间中的一个参数化团块，包含以下属性：

position         mu         (3,)    centre in world coordinates
rotation         q          (4,)    unit quaternion encoding orientation
scale            s          (3,)    log-scales per axis (exponentiated at render time)
opacity          alpha      (1,)    post-sigmoid opacity [0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   view-dependent colour

旋转与缩放共同构建了一个 3x3 的协方差矩阵（Covariance Matrix）：`Sigma = R S S^T R^T`。这决定了高斯椭球在三维空间中的形状。球谐函数（Spherical Harmonics, SH）使得颜色能够随观察方向变化——例如镜面高光、微妙的光泽以及视角相关的辉光——而无需为每个视角存储纹理。当球谐函数阶数（Degree）为 3 时，每个颜色通道有 16 个系数，仅颜色一项每个高斯椭球就需要 48 个浮点数。

一个场景通常包含 100 万到 500 万个高斯椭球。每个高斯椭球大约存储 60 个浮点数（3 + 4 + 3 + 1 + 48 + 其他）。对于一个包含 500 万个高斯椭球的场景，这仅占用约 240 MB 内存——远小于带有逐点纹理的等效点云，也比高分辨率下重新渲染的神经辐射场（Neural Radiance Fields, NeRF）的多层感知机（Multi-Layer Perceptron, MLP）权重小一个数量级。

### 光栅化，而非光线步进

flowchart LR
    SCENE["Millions of 3D Gaussians<br/>(position, rotation, scale,<br/>opacity, SH colour)"] --> PROJ["Project to 2D<br/>(camera extrinsics + intrinsics)"]
    PROJ --> TILES["Assign to tiles<br/>(16x16 screen-space)"]
    TILES --> SORT["Depth-sort<br/>per tile"]
    SORT --> ALPHA["Alpha-composite<br/>front-to-back"]
    ALPHA --> PIX["Pixel colour"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a

共五个步骤，均对 GPU 高度友好。无需对每个像素进行多层感知机（MLP）查询。单张 RTX 3080 Ti 显卡即可以 147 fps 的帧率渲染 600 万个高斯溅射（Splats）。

### 投影步骤

位于世界坐标 `mu` 且三维协方差为 `Sigma` 的三维高斯椭球，会投影为屏幕坐标 `mu'` 处的二维高斯椭球，其二维协方差为 `Sigma'`：

mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = viewing transform (rotation + translation of camera)
J = Jacobian of the perspective projection at mu'

该二维高斯椭球的覆盖范围是一个椭圆，其主轴方向由 `Sigma'` 的特征向量决定。椭圆内的每个像素都会接收到该高斯椭球的贡献，权重由 `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))` 计算得出。

### Alpha 合成规则

对于单个像素，覆盖它的高斯椭球会按从后到前的顺序排序（或等价地使用反转公式按从前到后排序）。颜色的合成采用自 20 世纪 80 年代以来所有半透明光栅化器（Rasteriser）通用的公式：

C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       transmittance up to i
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   local contribution
c_i = eval_SH(SH_i, view_direction)    view-dependent colour

这与神经辐射场（NeRF）的体积渲染（Volumetric Rendering）公式**完全相同**，只是积分对象从沿光线的密集采样点替换为了显式的稀疏高斯椭球集合。正是这种数学等价性使得渲染质量能够媲美 NeRF——两者本质上都在对相同的辐射场方程进行积分。

### 为何具备可微性

投影、图块分配（Tile Assignment）、Alpha 合成以及球谐函数求值等每一步，相对于高斯椭球的参数都是可微的。给定一张真实图像（Ground-truth Image），计算渲染像素的损失（Loss），通过光栅化器进行反向传播（Backpropagation），并利用梯度下降（Gradient Descent）更新所有 `(mu, q, s, alpha, c_lm)` 参数。经过约 30,000 次迭代，高斯椭球便能自动寻找到正确的位置、缩放比例和颜色。

### 致密化与剪枝

固定数量的高斯椭球无法充分覆盖复杂场景。训练过程包含两种自适应机制：

- **克隆（Clone）**：当某个高斯椭球的梯度幅值较大但缩放比例较小时，在其当前位置克隆一个新的高斯椭球——这表明该区域的重建需要更多细节。
- **分裂（Split）**：当某个大尺度高斯椭球的梯度较大时，将其分裂为两个较小的高斯椭球——单个大高斯椭球过于平滑，无法拟合该区域的细节。
- **剪枝（Prune）**：移除透明度低于阈值的高斯椭球——它们对最终渲染已无贡献。

致密化（Densification）操作每 N 次迭代执行一次。场景中的高斯椭球数量通常会从初始的约 10 万个（由运动恢复结构 Structure from Motion, SfM 点云初始化）增长至训练结束时的 100 万到 500 万个。

### 一段话理解球谐函数

视角相关颜色是定义在单位球面上的函数 `c(direction)`。球谐函数（Spherical Harmonics）本质上是球面上的傅里叶基（Fourier Basis）。将其截断至阶数 `L`，每个颜色通道即可得到 `(L+1)^2` 个基函数。计算新视角的颜色，只需将学习到的球谐系数与在该视角方向上求值的基函数进行点积。阶数 0 = 1 个系数 = 恒定颜色。阶数 3 = 16 个系数 = 足以捕捉朗伯着色（Lambertian Shading）、镜面高光及微弱反射。3D 高斯溅射（3D Gaussian Splatting）相关论文默认使用阶数 3。

### 2026 年生产级技术栈

1. Capture         smartphone / DJI drone / handheld scanner
2. SfM / MVS       COLMAP or GLOMAP derives camera poses + sparse points
3. Train 3DGS      nerfstudio / gsplat / inria official / PostShot (~10-30 min on RTX 4090)
4. Edit            SuperSplat / SplatForge (clean floaters, segment)
5. Export          .ply -> glTF KHR_gaussian_splatting or .usd (OpenUSD 26.03)
6. View            Cesium / Unreal / Babylon.js / Three.js / Vision Pro

### 4D 与生成式变体

- **4D 高斯溅射（4D Gaussian Splatting）**——高斯椭球是时间的函数；主要用于体积视频（Volumetric Video）制作（如《超人》2026 版、A$AP Rocky 的《Helicopter》MV）。
- **生成式高斯溅射（Generative Splats）**——文本生成高斯模型（如 World Labs 的 Marble），能够生成完整的三维场景。
- **3D 高斯无迹变换（3D Gaussian Unscented Transform）**——NVIDIA NuRec 的变体，专用于自动驾驶仿真。

## 构建

### 步骤 1：二维高斯分布 (2D Gaussian)

我们首先构建一个二维光栅化器 (2D rasteriser)。三维情况在投影后即可简化为该二维形式。

import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      centres
    covs:   (G, 2, 2)   covariance matrices
    points: (H, W, 2)   pixel coordinates
    returns: (G, H, W)  density at every pixel for every Gaussian
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)

`einsum` 为每个（高斯分布，像素）对计算二次型 `diff^T Sigma^-1 diff`。

### 步骤 2：二维溅射光栅化器 (2D splatting rasteriser)

采用从前到后的 Alpha 合成 (Alpha-compositing)。在二维中深度没有实际意义，因此我们使用每个高斯分布学习到的标量来进行排序。

def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     in [0, 1]
    depths:    (G,)     per-Gaussian scalar used for ordering
    image_size: (H, W)
    returns:   (H, W, 3) rendered image
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out

速度并不快——实际实现会使用基于图块的 CUDA 内核 (tile-based CUDA kernels)——但数学原理完全正确，且支持完全可微 (fully differentiable)。

### 步骤 3：可训练的二维溅射场景 (trainable 2D splat scene)

class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # single angle in 2D
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)

`log_scale`、`opacity_logit` 和 `colour_logits` 均为无约束参数，在渲染时通过相应的激活函数进行映射。这是所有 3D 高斯溅射 (3D Gaussian Splatting, 3DGS) 实现的标准模式。

### 步骤 4：将二维高斯分布拟合至目标图像

import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Red circle
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # Blue square
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")

经过 200 步迭代，这 64 个高斯分布将收敛并拟合出这两个形状。这就是核心思想——对显式几何基元 (explicit geometric primitives) 进行梯度下降 (gradient-descent)。

### 步骤 5：从二维到三维

三维扩展保留了相同的循环逻辑。新增内容包括：

1. 每个高斯分布的旋转使用四元数 (quaternion) 而非单一角度。
2. 协方差矩阵为 `R S S^T R^T`，其中 `R` 由四元数构建，`S = diag(exp(log_scale))`。
3. 投影 `(mu, Sigma) -> (mu', Sigma')` 使用相机外参 (camera extrinsics) 以及在 `mu` 处的透视投影雅可比矩阵 (Jacobian)。
4. 颜色变为球谐函数 (spherical-harmonics) 展开式；需沿观察方向进行求值。
5. 深度排序基于实际的相机空间 z 轴坐标，而非学习到的标量。

所有生产级实现（如 `gsplat`、`inria/gaussian-splatting`、`nerfstudio`）均在 GPU 上使用基于图块的 CUDA 内核精确执行此流程。

### 步骤 6：球谐函数求值 (Spherical harmonics evaluation)

最高到 3 阶的球谐基 (SH basis) 每个通道包含 16 个项。求值过程如下：

def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   last dim is RGB channels
    dirs:      (..., 3)       unit vectors
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # degree 3 terms omitted here for brevity; full 16-coefficient version in the code file
    return result

学习得到的 `sh_coeffs` 存储了该高斯分布“在各个方向上的颜色”。在渲染时，你只需针对当前观察方向进行求值，即可得到一个三维 RGB 向量。

## 如何使用

对于实际的 3DGS（3D Gaussian Splatting）工作，请使用 `gsplat`（Meta）或 `nerfstudio`：

pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data

`splatfacto` 是 `nerfstudio` 的 3DGS 训练器。在典型场景下，使用 RTX 4090 运行该过程通常需要 10 到 30 分钟。

2026 年值得关注的导出格式选项：

- `.ply` — 原始高斯点云（可移植性强，文件体积最大）。
- `.splat` — PlayCanvas / SuperSplat 的量化格式。
- glTF `KHR_gaussian_splatting` — Khronos 标准，可在各类查看器间通用（2026 年 2 月 RC 版）。
- OpenUSD `UsdVolParticleField3DGaussianSplat` — USD 原生格式，适用于 NVIDIA Omniverse 和 Vision Pro 工作流。

对于 4D / 动态场景，`4DGS` 和 `Deformable-3DGS` 通过引入随时间变化的均值（means）和不透明度（opacities），扩展了相同的底层机制。

## 交付成果

本课时将生成以下文件：

- `outputs/prompt-3dgs-capture-planner.md` — 用于为指定场景类型规划采集流程（照片数量、相机路径、光照设置）的提示词（prompt）。
- `outputs/skill-3dgs-export-router.md` — 一项技能（skill），可根据下游查看器或引擎自动选择最合适的导出格式（`.ply` / `.splat` / glTF / USD）。

## 练习

1. **（简单）** 使用上述 2D splat 训练器处理另一张合成图像。将 `num_splats` 参数在 `[16, 64, 256]` 之间调整，并分别绘制均方误差（MSE）随训练步数（step）变化的曲线。找出收益递减的临界点。
2. **（中等）** 扩展 2D 光栅化器（rasteriser），使其支持基于标量“视角”并通过二阶谐波（degree-2 harmonic）计算每个高斯（Gaussian）的 RGB 颜色。使用一对目标图像进行训练，并验证模型能否同时重建这两张图像。
3. **（困难）** 克隆 `nerfstudio` 仓库，并使用你手头的任意场景（如书桌、植物、人脸或房间）的 20 张照片数据集训练 `splatfacto`。将其导出为 glTF `KHR_gaussian_splatting` 格式，并在查看器中打开（如 Three.js `GaussianSplats3D`、SuperSplat 或 Babylon.js V9）。记录训练耗时、高斯数量以及渲染帧率（fps）。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 3DGS（3D Gaussian Splatting） | “高斯溅射” | 将场景显式表示为数百万个 3D 高斯，每个高斯包含独立的位置、旋转、缩放、不透明度和球谐函数（SH）颜色 |
| 协方差（Covariance） | “高斯的形状” | `Sigma = R S S^T R^T`；表示单个高斯的方向与各向异性缩放 |
| Alpha 合成（Alpha compositing） | “从后向前混合” | 与 NeRF 的体积渲染使用相同的公式，但现应用于显式的稀疏点集上 |
| 致密化（Densification） | “克隆与分裂” | 在重建欠拟合的区域自适应地添加新高斯 |
| 剪枝（Pruning） | “删除低不透明度” | 移除在训练过程中不透明度衰减至接近零的高斯 |
| 球谐函数（Spherical harmonics） | “视角相关颜色” | 球面上的傅里叶基；将颜色存储为观察方向的函数 |
| Splatfacto | “nerfstudio 的 3DGS” | 2026 年训练 3DGS 最便捷的途径 |
| `KHR_gaussian_splatting` | “glTF 标准” | Khronos 2026 年推出的扩展规范，使 3DGS 能够在不同查看器和引擎间无缝移植 |

## 延伸阅读

- [用于实时辐射场（Radiance Field）渲染的3D高斯溅射（3D Gaussian Splatting）(Kerbl 等人，SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — 原始论文
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) — 生产级 CUDA 光栅化器（CUDA Rasteriser）
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) — 参考训练方案（Training Recipe）
- [Khronos KHR_gaussian_splatting 扩展（Extension）](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) — 2026 年可移植格式（Portable Format）
- [OpenUSD 26.03 版本说明（Release Notes）](https://openusd.org/release/) — UsdVolParticleField3DGaussianSplat 模式（Schema）
- [THE FUTURE 3D：2026 年高斯溅射发展现状](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) — 行业概览（Industry Overview）