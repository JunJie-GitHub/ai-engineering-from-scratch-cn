# 更新日志

课程新增内容。按时间倒序排列（最新内容在前）。

格式大致遵循 [Keep a Changelog](https://keepachangelog.com/) 规范。每条记录均标明阶段（Phase）、课程（Lesson）及变更内容，以便学习者直接定位到具体变更（Delta）。

## [未发布]

### 新增
- `scripts/scaffold-lesson.sh` — 脚手架脚本（Scaffolder），用于创建包含完整目录结构的 `phases/NN-phase/NN-lesson/` 文件夹，并根据 `LESSON_TEMPLATE.md` 预填充 `docs/en.md` 骨架文件。
- `.github/PULL_REQUEST_TEMPLATE.md` — 贡献者检查清单（代码可运行、无代码注释、优先从零构建、按课程原子化提交、ROADMAP 行使用 Markdown 链接）。
- `.github/ISSUE_TEMPLATE/bug_report.md` 和 `new_lesson_proposal.md` — 用于规范化收集错误报告（Bug Report）和课程提案的结构化模板。
- 本 `CHANGELOG.md` 文件。

## 2026-04 — 阶段 4：计算机视觉（Computer Vision）完成

### 新增
- 阶段 4 的全部 28 节课程，内容涵盖图像基础至多模态视觉（Multimodal Vision，包括视觉语言模型 VLMs、3D 视觉、视频处理与自监督学习 Self-supervised Learning）。
- `ROADMAP.md` 中阶段 4 的行已添加指向对应课程文件夹的 Markdown 链接，以便网站能够展示这些内容。

### 修复
- 对阶段 4 的 15 余节课程进行了准确性校对（Precision Pass）：
  - `phase-4/02`：形状计算器（Shape Calculator）明确了自适应池化（Adaptive Pool）、展平（Flatten）和线性层（Linear）的感受野（Receptive Field, RF）/步幅（Stride）处理规则。
  - `phase-4/03`：主干网络选择器（Backbone Selector）描述列出了所有涵盖的模型家族；新增了针对光学字符识别（Optical Character Recognition, OCR）、医疗影像和工业场景的头部网络（Head）指导。
  - `phase-4/04`：分类诊断（Classification Diagnostics）针对每种失败模式设定了量化阈值；对未定义的指标声明为 `n/a`；增加了类别数少于 3 时的防护逻辑。
  - `phase-4/06`：检测指标读取器（Detection Metric Reader）使用 `AP@0.5`（而非 `mAP@0.5`）；声明按类别召回率（Per-class Recall）为可选；锚框设计器（Anchor Designer）明确了步幅截断规则及每层级单锚框路径。
  - `phase-4/10`：采样器选择器（Sampler Picker）将 `unet_forward_ms` 声明为输入；ControlNet 防护逻辑提升为规则 0。
  - `phase-4/14`：视觉 Transformer（Vision Transformer, ViT）检查器与拒绝规则（Refusal Rule）对齐——移植尝试（Port Attempts）仅用于审计，不作背书。
  - `phase-4/24`：开放词汇栈选择器（Open-vocabulary Stack Picker）明确了规则优先级和许可证过滤语义；概念设计器（Concept Designer）解决了步骤 5 与规则 80 之间的冲突。
  - `phase-4/25`：VLM 文档中的 `_merge` 在占位符不匹配时会抛出描述性的 `ValueError`；CMER 在内部进行归一化处理。
  - `phase-4/27`：`synthetic_frames` 将真实标注框（Ground Truth Boxes）裁剪至帧的高/宽（H/W）范围内。
  - `phase-4/28`：`rope_3d` 增加了维度拆分验证；移除了 DiT 块示例中未使用的 `F` 导入。

## 2026年第一季度及更早

### 新增
- 阶段 0（环境配置与工具链）：全部 12 节课程。
- 阶段 1（数学基础）：全部 22 节课程。
- 阶段 2（机器学习基础）：全部 18 节课程。
- 阶段 3（深度学习核心）：涵盖感知机（Perceptron）、反向传播（Backpropagation）与优化器（Optimizers）的核心课程。
- 内置 Claude Code 技能：`find-your-level`（定级测验 Placement Quiz）和 `check-understanding`（分阶段测验 Per-phase Quiz）。
- 网站 `aiengineeringfromscratch.com`：包含课程目录、单课页面、路线图及包含 277 个术语的词汇表。
- 全部 20 个阶段的初始脚手架结构（`phases/00-*` 至 `phases/19-*`）。
- `LESSON_TEMPLATE.md`、`CONTRIBUTING.md`、`ROADMAP.md` 和 `README.md` 文件。

[未发布]: https://github.com/rohitg00/ai-engineering-from-scratch/compare/HEAD...HEAD