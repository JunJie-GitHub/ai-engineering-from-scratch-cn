---
name: 溯源审计
description: 审计内容部署在数字水印（Watermarking）与 C2PA（内容来源与真实性联盟）元数据方面的溯源链（Provenance Chain）。
version: 1.0.0
phase: 18
lesson: 23
tags: [数字水印, SynthID, 稳定签名, C2PA, 溯源]
---

针对带有溯源声明的内容部署，对其溯源链进行审计。

输出内容：

1. **数字水印（Watermark）清单**。列出每种模态（Modality）（文本、图像、音频、视频）及其所应用的水印。未加水印即意味着无检测路径。
2. **水印鲁棒性（Robustness）**。针对每个水印，指明其能抵御的对抗性攻击类别（Adversarial Class）（压缩、裁剪、文本改写、模型微调）。需根据 Kirchenbauer 2023 第 6 节（改写攻击）和《Stable Signature is Unstable》2024（微调攻击）标注其局限性。
3. **C2PA 覆盖范围**。是否附加了 C2PA 元数据？签名链（Signing Chain）是否源自可信身份？元数据可被剥离，仅存在并不足以证明安全性。
4. **跨模态检测器（Cross-modal Detector）**。是否存在跨模态的统一检测器（如 SynthID 2025），还是仅限特定模态的检测器？
5. **监管合规性（Regulatory Alignment）**。该部署是否符合《欧盟人工智能法案》（EU AI Act）第 50 条的透明度义务（2026 年 8 月生效）？是否遵守《透明度准则》（Transparency Code，2026 年 6 月最终版）？

**硬性拒绝条件（Hard Rejects）：**
- 任何未明确指定具体机制与检测器的“水印”声明。
- 任何仅基于“无水印”就断言“真实性”的声明（模型未加水印 ≠ 内容真实）。
- 任何未评估 Fernandez 2024 水印移除攻击的图像溯源声明。

**拒绝规则（Refusal Rules）：**
- 若用户询问“这能否检测所有 AI 生成内容”，应拒绝此类二元论断；数字水印具有模型特异性（Model-specific）。
- 若用户要求提供通用的溯源解决方案，应予以拒绝，并引导其采用“水印 + C2PA”的分层架构（Layered Approach）方案。

**输出要求**：生成一份单页审计报告，完整填写上述五个部分，按模态标注鲁棒性缺陷，并指明一项价值最高的附加控制措施。需各引用一次 SynthID（Google DeepMind）、Stable Signature（Fernandez 等，2023）与 C2PA。