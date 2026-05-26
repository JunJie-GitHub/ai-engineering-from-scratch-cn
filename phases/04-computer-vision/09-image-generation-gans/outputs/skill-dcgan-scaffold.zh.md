---
name: skill-dcgan-scaffold
description: 根据 `z_dim`、`image_size` 和 `num_channels` 编写完整的深度卷积生成对抗网络（DCGAN）脚手架，包含训练循环与样本保存功能
version: 1.0.0
phase: 4
lesson: 9
tags: [计算机视觉, 生成对抗网络, 深度卷积生成对抗网络, 脚手架]
---

# 深度卷积生成对抗网络（DCGAN）脚手架

给定三个参数，生成一个可运行的 DCGAN 项目骨架，其架构尺寸已针对目标图像分辨率进行正确配置。

## 适用场景

- 在小型数据集上启动新的生成式实验。
- 使用可运行的最小示例教授 DCGAN 基础知识。
- 原型设计条件生成对抗网络（Conditional GAN）（标签注入在同一脚手架中完成）。

## 输入参数

- `image_size`：32、64 或 128 之一（必须为 2 的幂）。
- `num_channels`：1（灰度）或 3（RGB）。
- `z_dim`：通常为 64 或 128。
- `with_spectral_norm`：yes | no；默认为 yes。

## 架构尺寸配置

生成器（Generator）中的转置卷积（Transposed Convolution）块数量与判别器（Discriminator）中的步长卷积（Strided Convolution）块数量取决于 `image_size`：

| image_size | G blocks | D blocks |
|------------|----------|----------|
| 32         | 4        | 4        |
| 64         | 5        | 5        |
| 128        | 6        | 6        |

每增加一个块，空间维度在 G 中翻倍，在 D 中减半。特征通道数从 32 开始，并按 `feat_base * 2^block_index` 的比例缩放。

## 输出文件

- `model.py` — 生成器（Generator）与判别器（Discriminator）类
- `train.py` — 训练循环、损失函数与优化器配置
- `sample.py` — 样本网格保存器
- `config.json` — 超参数
- `README.md` — 10 行快速入门指南

## 报告输出

[scaffold]
  image_size:       <int>
  num_channels:     <int>
  z_dim:            <int>
  spectral_norm:    yes | no

[arch]
  G blocks:         <N>, channels: [list]
  D blocks:         <N>, channels: [list]
  G params (est):   <N>
  D params (est):   <N>

[training defaults]
  optimizer:   Adam(lr=2e-4, betas=(0.5, 0.999))
  batch_size:  64
  epochs:      50
  sample_every: 1 epoch

[files written]
  - model.py
  - train.py
  - sample.py
  - config.json
  - README.md

## 规则

- 始终在 G 的输出端使用 `nn.Tanh()`，并在训练期间将数据缩放至 [-1, 1]。
- 始终在 D 中使用 `LeakyReLU(0.2)`。
- 当 `with_spectral_norm == yes` 时，使用 `spectral_norm()`（谱归一化）包装 D 中的每个卷积层，并从 D 中移除 BatchNorm（批归一化）。G 中保留 BatchNorm。
- 切勿为 `image_size > 128` 生成脚手架——超过该尺寸后 DCGAN 会变得不稳定；请引导用户使用 StyleGAN 或扩散模型（Diffusion Model）。