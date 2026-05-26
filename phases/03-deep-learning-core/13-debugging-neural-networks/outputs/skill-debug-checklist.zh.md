---
name: 技能调试清单
description: 用于调试神经网络训练失败的决策树清单
version: 1.0.0
phase: 3
lesson: 13
tags: [调试, 神经网络, 训练, 诊断, 深度学习]
---

# 神经网络调试清单

训练出现问题时的系统化调试流程。请按顺序逐一排查——大多数错误都能在前 3 步中被发现。

## 训练前（预防错误）

1. 打印模型架构与参数量。该规模是否与你的数据量相匹配？
2. 使用随机输入执行一次前向传播（forward pass）。输出形状（output shape）是否与目标形状一致？
3. 检查标签的数据类型（dtype）是否正确（`CrossEntropyLoss` 需要 `Long` 类型，`BCELoss` 需要 `Float` 类型）
4. 验证数据归一化（data normalization）：输入数据的均值应接近 0，标准差应接近 1
5. 打印 5 组随机的（输入，标签）对。标签是否符合你的预期？
6. 确认训练集/测试集划分中没有重复样本

## 单批次过拟合测试（60 秒，可捕获 80% 的错误）

1. 从训练集中抽取 8-32 个样本
2. 使用合理的学习率（learning rate）训练 200 步
3. 损失值（loss）应趋近于 0。训练准确率应达到 100%
4. 若失败：错误通常出在模型、损失函数（loss function）或训练循环（training loop）中——而非数据或超参数（hyperparameters）
5. 若通过：继续进行完整训练

## 损失值未下降

1. 检查学习率。尝试 3 个值：当前值/10、当前值、当前值*10
2. 打印各层的梯度范数（gradient norms）。全为零意味着网络已“死亡”或计算图（computation graph）已断开
3. 检查参数是否设置了 `requires_grad=True`。确认已调用 `loss.backward()`
4. 确认在调用 `loss.backward()` 之前已调用 `optimizer.zero_grad()`
5. 确认在调用 `loss.backward()` 之后已调用 `optimizer.step()`
6. 验证模型参数是否已正确传入优化器：`optimizer = Adam(model.parameters())`

## 损失值为 NaN 或 Inf

1. 将学习率降低 10 倍
2. 为所有 `log()` 调用添加极小值（epsilon）：`torch.log(x + 1e-7)`
3. 为所有除法运算添加极小值：`x / (y + 1e-8)`
4. 在计算 `BCE` 损失前对预测值进行截断（clamp）：`torch.clamp(pred, 1e-7, 1 - 1e-7)`
5. 使用 `torch.autograd.detect_anomaly()` 定位引发异常的具体操作
6. 检查输入数据中是否存在 `NaN`：`assert not torch.isnan(x).any()`

## 损失值震荡

1. 将学习率降低 3-10 倍
2. 增大批次大小（batch size）（可降低梯度噪声）
3. 添加梯度裁剪（gradient clipping）：`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
4. 从 `SGD` 切换至 `Adam`（为每个参数提供自适应学习率）
5. 在训练的前 5-10% 阶段添加学习率预热（learning rate warmup）

## 过拟合（训练准确率高，测试准确率低）

1. 添加 Dropout（从 `p=0.1` 开始，可逐步增加至 0.5）
2. 为优化器添加权重衰减（weight decay）：`Adam(params, weight_decay=1e-4)`
3. 缩减模型规模（减少层数或缩小层宽）
4. 添加数据增强（data augmentation）
5. 使用早停法（early stopping）：当验证集损失连续 5 个以上 epoch 上升时停止训练
6. 检查训练集与测试集之间是否存在数据泄露（data leakage）

## 欠拟合（训练和测试准确率均较低）

1. 提升模型容量（model capacity）（增加层数或加宽层）
2. 增加训练轮数（epochs）
3. （谨慎地）提高学习率
4. 暂时移除正则化（regularization），以验证模型是否具备学习能力
5. 检查模型对该任务的表达能力（expressiveness）是否足够

## ReLU 神经元死亡

1. 检查每一层中零激活值（zero activations）的比例。超过 50% 即为问题
2. 切换至 `LeakyReLU(0.01)` 或 `GELU` 激活函数
3. 对权重使用 Kaiming 初始化（Kaiming initialization）
4. 降低学习率（过大的参数更新可能将神经元推入死亡区）
5. 在激活函数前添加批归一化（batch normalization）

## 快速参考：学习率初始值

| 优化器 | 任务 | 初始学习率 |
|-----------|------|------------|
| Adam | 从头训练 | 1e-3 |
| Adam | 微调预训练模型 | 1e-5 |
| SGD + momentum | 从头训练 | 1e-1 |
| SGD + momentum | 微调预训练模型 | 1e-3 |
| AdamW | Transformer 训练 | 3e-4 |

## 快速参考：批次大小的影响

| 批次大小 (Batch Size) | 梯度噪声 (Gradient Noise) | 内存占用 (Memory) | 泛化能力 (Generalization) |
|-----------|---------------|--------|---------------|
| 8-16 | 高（噪声大） | 低 | 通常更好 |
| 32-64 | 中等 | 中等 | 推荐的默认值 |
| 128-256 | 低（平滑） | 高 | 可能需要预热 (Warmup) |
| 512+ | 极低 | 极高 | 需要学习率缩放 (LR Scaling) |

## 当所有方法均无效时

1. 将模型简化为仅含 1 个隐藏层 (Hidden Layer)。它能否正常学习？
2. 将数据简化为 100 个样本。它是否会出现过拟合 (Overfitting)？
3. 将损失函数 (Loss Function) 替换为均方误差 (MSE)。它能否收敛？
4. 将优化器 (Optimizer) 替换为 SGD(lr=0.01)。训练是否有进展？
5. 将数据替换为合成数据 (Synthetic Data)（例如 y = x[0] > 0）。它能否正常学习？
6. 若以上均无效：说明 Bug 存在于你尚未仔细排查的代码中（如数据加载 (Data Loading)、预处理 (Preprocessing)、张量形状 (Tensor Shapes) 等）。