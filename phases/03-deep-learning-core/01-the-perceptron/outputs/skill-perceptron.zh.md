---
name: 感知机
description: 理解感知机模式以及何时使用单层与多层架构
version: 1.0.0
phase: 3
lesson: 1
tags: [感知机, 神经网络, 分类, 深度学习]
---

# 感知机 (Perceptron) 模式

感知机计算输入的加权和并加上偏置 (Bias)，然后应用阶跃函数 (Step Function) 以产生二值输出。它是神经网络 (Neural Network) 的基本单元。

output = step(w1*x1 + w2*x2 + ... + wn*xn + bias)

## 何时单个感知机已足够

- 问题是线性可分 (Linearly Separable) 的：一条直线（或超平面 (Hyperplane)）即可将两类数据分开
- 逻辑门：与 (AND)、或 (OR)、非 (NOT)、与非 (NAND)
- 简单的阈值决策：“分数是否高于 X？”
- 针对聚类为两个不重叠区域的数据的二分类器 (Binary Classifier)

## 何时需要多层架构

- 问题是非线性可分的：无法用单一直线将类别分开
- 异或 (XOR) 与奇偶校验问题
- 任何需要“是此而非彼”推理的任务（条件的组合）
- 现实世界中的分类任务：图像、文本、音频——几乎总是非线性的

## 决策检查清单

1. 绘制或检查你的数据。能否在类别之间画出一条单一的直线边界？
   - 是：单个感知机即可胜任
   - 否：你至少需要两层网络
2. 该问题能否分解为更简单线性决策的“与/或”组合？
   - 这种分解方式能告诉你所需的最小网络结构
   - XOR = (A OR B) AND (NOT (A AND B)) = 2 层网络中的 3 个感知机
3. 对于包含两个以上类别的问题，每个类别需要一个输出节点

## 训练规则

error = expected - predicted
weight_new = weight_old + learning_rate * error * input
bias_new = bias_old + learning_rate * error

如果预测正确，则不作任何更改。如果预测错误，权重将发生偏移以减小误差。此规则仅适用于单层感知机。多层网络需要反向传播 (Backpropagation)。

## 常见误区

- 试图用单个感知机学习非线性模式（模型将永远无法收敛 (Convergence)）
- 学习率 (Learning Rate) 设置过高（权重震荡）或过低（训练耗时极长）
- 遗漏偏置项（没有它，决策边界必须穿过原点）
- 混淆感知机收敛性（对线性可分数据有保证）与通用神经网络收敛性（无法保证）