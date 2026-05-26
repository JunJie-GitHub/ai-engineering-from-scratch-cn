---
name: skill-3dgs-export-router
description: 根据下游查看器或引擎选择合适的 3DGS (3D Gaussian Splatting) 导出格式（.ply / .splat / glTF KHR_gaussian_splatting / USD）
version: 1.0.0
phase: 4
lesson: 22
tags: [3D高斯溅射, 导出, glTF, OpenUSD, 流水线]
---

# 3DGS (3D Gaussian Splatting) 导出路由 (3DGS Export Router)

将下游目标映射至正确的文件格式。可节省数小时排查“无法加载”问题的调试时间。

## 适用场景

- 完成 3DGS 场景训练后，在将其移交至内容流水线之前。
- 在研究级格式（.ply）与生产级格式（glTF / USD）之间进行选择时。
- 流水线交接：采集团队 -> 3DGS 工程师 -> 游戏设计师 / 视觉特效艺术家 / Web 开发者。

## 输入参数

- `target_engine`: unreal | unity | omniverse | blender | vision_pro | three_js | babylon_js | cesium | playcanvas | supersplat
- `priority`: portability | file_size | quality_preservation
- `include_sh_degree`: 0 | 1 | 2 | 3

## 格式决策

| 目标平台 | 推荐格式 | 原因 |
|--------|--------------------|-----|
| Unreal Engine（虚拟制片） | Volinga 插件或 glTF KHR_gaussian_splatting | 原生 Unreal SDK 路径 |
| Unity（XR / 游戏） | 通过 Aras-P Unity-GaussianSplatting 插件使用 .ply | 社区标准的 Unity 流水线 |
| NVIDIA Omniverse、Pixar 工具 | OpenUSD (Open Universal Scene Description) 26.03 (UsdVolParticleField3DGaussianSplat) | 原生 USD 图元类型 |
| Apple Vision Pro | OpenUSD 26.03 | visionOS 2.x 原生支持 |
| Blender | .ply + KIRI Engine 插件 | 社区插件可读取原始溅射数据 |
| Three.js Web 查看器 | glTF KHR_gaussian_splatting 或 .splat | 浏览器标准，兼容 `GaussianSplats3D` |
| Babylon.js V9+ | glTF KHR_gaussian_splatting | V9 版本已添加原生支持 |
| Cesium (CesiumJS 1.139+, Cesium for Unreal 2.23+) | glTF KHR_gaussian_splatting | 已内置明确支持 |
| PlayCanvas | .splat | PlayCanvas 原生量化格式 |
| SuperSplat（编辑器） | .ply 或 .splat | 支持导入与导出 |

## 量化 (Quantisation) 权衡

- `.ply` 全精度：文件体积最大，无损，兼容任意查看器。
- `.splat`：体积缩小 4-8 倍，SH3 系数 (Spherical Harmonics degree 3 coefficients) 有轻微质量损失，PlayCanvas 生态标准格式。
- glTF KHR：可通过 `EXT_meshopt_compression` 配置；在保持最高兼容性的同时实现最小体积。
- USD：通过 USDZ 打包压缩；适用于 Apple 流水线的最小体积方案。

## 输出报告

[export plan]
  target:         <engine>
  format:         <name>
  sh degree:      <0|1|2|3>
  compression:    <none|meshopt|quantisation|usdz>
  expected size:  <MB>
  compatible with: <list of viewers>

[pipeline]
  1. source: <.ply from training>
  2. optional: SuperSplat cleanup pass
  3. convert: <tool + CLI or API call>
  4. package: <.gltf / .glb / .usd / .usdz / .splat / .ply>
  5. validate: <viewer sanity check>

## 规则

- 切勿静默移除 SH3 系数——这会显著改变镜面反射效果。
- 若 `priority == file_size`，推荐 `.splat` 或启用 meshopt 的 glTF；需提示质量损失风险。
- 面向 Apple 平台时，2026 年应优先选择 USD / USDZ 而非 glTF；USDZ 在 visionOS 中享有原生级支持。
- 若目标查看器的 3DGS 支持处于标准发布前阶段（2026 年 2 月前），推荐 `.ply` 及该查看器的自定义加载器；Khronos 标准的 glTF 届时可能尚未被识别。
- 在交付前，务必在至少一个查看器中验证导出文件；量化过程中可能发生静默数据损坏。