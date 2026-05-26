---
name: 3d-pipeline
description: 根据输入类型、输出格式和使用场景，选择 3D 生成或重建流水线 (Pipeline)。
version: 1.0.0
phase: 8
lesson: 12
tags: [3D, 高斯溅射 (Gaussian Splatting), 神经辐射场 (NeRF), 网格 (Mesh)]
---

根据输入（文本提示词 / 单张图像 / 少量图像 / 照片采集 / 视频）、目标输出（网格 (Mesh) / 高斯溅射 (Gaussian Splat) / 神经辐射场 (NeRF) / 点云 (Point Cloud)）以及使用场景（实时渲染 (Real-time Rendering)、游戏引擎、AR / VR、影视级），输出以下内容：

1. 流水线 (Pipeline)。(a) 多视角扩散模型 + 3D 拟合 (SV3D, CAT3D + 3DGS)，(b) 直接单次生成 (LRM, TripoSR, InstantMesh)，(c) 带基于物理的渲染 (PBR) 材质的文本生成网格 (Meshy 4, Rodin Gen-1.5, Hunyuan3D 2.0)，(d) 照片采集 + 3D 高斯溅射 (3DGS) (Gsplat, Postshot, Scaniverse)。
2. 基础模型与托管服务。指定模型名称 + 开源/托管状态。需包含与商业用途相关的许可证信息。
3. 迭代预算。首次输出预期时间、迭代成本及优化策略。
4. 拓扑结构与材质。是否需要重新网格化 (Remesh) 步骤？PBR 通道需求（反照率 (Albedo)、粗糙度 (Roughness)、金属度 (Metallic)、法线 (Normal)）？UV 布局是自动还是手动？
5. 评估指标。在预留视图上的结构相似性 (SSIM)、CLIP 评分 (CLIP Score)、网格水密性 (Mesh Watertightness)、多边形数量 (Poly Count)、纹理分辨率 (Texture Resolution)。
6. 目标平台。Unity / Unreal / Blender / Web (three.js / Babylon) / AR (USDZ / glb)。

拒绝在未进行网格转换步骤的情况下，直接将 3DGS 导入游戏引擎（大多数引擎不支持原生渲染溅射点）。拒绝为复杂的可关节绑定角色使用文本生成 3D 方案——应改用支持骨骼绑定 (Rigging) 的流水线。当下游工具无法渲染 NeRF 时（大多数数字内容创作 (DCC) 工具），需标记任何仅输出 NeRF 的方案。