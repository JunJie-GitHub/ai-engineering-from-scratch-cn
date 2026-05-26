# 调试神经网络

> 你的网络编译通过，成功运行，并输出了一个数值。该数值是错误的，但程序并未崩溃。欢迎来到最棘手的调试场景——那种没有任何错误提示的调试。

**类型：** 实践练习
**语言：** Python, PyTorch
**前置要求：** 第 03 阶段第 01-10 课（尤其是反向传播 (backpropagation)、损失函数 (loss functions)、优化器 (optimizers)）
**预计耗时：** 约 90 分钟

## 学习目标

- 运用系统化的调试策略，诊断常见的神经网络故障（如 NaN 损失 (NaN loss)、平坦的损失曲线 (flat loss curve)、过拟合 (overfitting)、震荡 (oscillation)）
- 应用“单批次过拟合 (overfit one batch)”技术，验证模型架构和训练循环 (training loop) 是否正确
- 检查梯度幅值 (gradient magnitudes)、激活值分布 (activation distributions) 和权重范数 (weight norms)，以识别梯度消失/爆炸 (vanishing/exploding gradient) 问题
- 构建一份调试检查清单，涵盖数据流水线 (data pipeline)、模型架构、损失函数、优化器以及学习率 (learning rate) 相关问题

## 问题描述

传统软件在出错时会直接崩溃。空指针会抛出异常。类型不匹配会在编译期报错。差一错误 (off-by-one error) 会产生明显错误的输出。

神经网络可不会给你这种“优待”。

一个存在缺陷的神经网络会完整运行到底，打印出损失值，并输出预测结果。损失值可能确实在下降。预测结果看起来也可能合情合理。但模型实际上已经“静默”出错——它可能在学习捷径、记忆噪声，或者收敛到一个毫无用处的局部极小值 (local minimum)。谷歌研究人员估计，60% 到 70% 的机器学习调试时间都花在了这类“静默”缺陷上，它们不会引发报错，却会严重降低模型质量。

一个能正常工作的模型与一个存在缺陷的模型之间，往往只差一行放错位置的代码：漏掉了 `zero_grad()`、维度转置错误，或者学习率相差了 10 倍。经典的《训练神经网络指南》(Recipe for Training Neural Networks, 2019) 开篇便指出：“神经网络最常见的错误，就是那些不会导致崩溃的缺陷。”

本课程将教你如何揪出这些缺陷。

## 核心概念

### 调试思维（Debugging Mindset）

摒弃“打印并祈祷”式调试（Print-and-Pray Debugging）。神经网络调试需要系统化的方法，因为其反馈循环较慢（每次训练运行需数分钟到数小时），且症状往往模棱两可（损失值不佳可能意味着 20 种不同的问题）。

黄金法则：**从简单开始，逐步增加复杂度，并独立验证每个组件。**

flowchart TD
    A["Loss not decreasing"] --> B{"Check learning rate"}
    B -->|"Too high"| C["Loss oscillates or explodes"]
    B -->|"Too low"| D["Loss barely moves"]
    B -->|"Reasonable"| E{"Check gradients"}
    E -->|"All zeros"| F["Dead ReLUs or vanishing gradients"]
    E -->|"NaN/Inf"| G["Exploding gradients"]
    E -->|"Normal"| H{"Check data pipeline"}
    H -->|"Labels shuffled"| I["Random-chance accuracy"]
    H -->|"Preprocessing bug"| J["Model learns noise"]
    H -->|"Data is fine"| K{"Check architecture"}
    K -->|"Too small"| L["Underfitting"]
    K -->|"Too deep"| M["Optimization difficulty"]

### 症状 1：损失值（Loss）不下降

这是最常见的抱怨。训练循环在运行，轮次（Epoch）在增加，但损失值却保持平稳或剧烈震荡。

**学习率（Learning Rate）设置错误。** 过高：损失值震荡或跳变为 `NaN`。过低：损失值下降极其缓慢，看起来像是停滞。对于 `Adam` 优化器，建议从 `1e-3` 开始；对于 `SGD`，建议从 `1e-1` 或 `1e-2` 开始。在断定其他环节有问题之前，务必尝试 3 个相差 10 倍的学习率（例如 `1e-2`、`1e-3`、`1e-4`）。

**ReLU 神经元死亡（Dead ReLUs）。** 如果 `ReLU` 神经元接收到较大的负输入，其输出为 0，梯度也为 0，此后将永远无法再次激活。如果大量神经元死亡，网络将无法学习。检查方法：打印每个 `ReLU` 层后激活值恰好为 0 的比例。如果死亡比例 >50%，请改用 `LeakyReLU` 或降低学习率。

**梯度消失（Vanishing Gradients）。** 在使用 `sigmoid` 或 `tanh` 激活函数的深层网络中，梯度在反向传播时会呈指数级衰减。当它们到达第一层时，已接近 0，导致浅层网络停止学习。解决方法：使用 `ReLU`/`GELU` 激活函数、添加残差连接（Residual Connections）或使用批归一化（Batch Normalization）。

**梯度爆炸（Exploding Gradients）。** 与上述相反的问题——梯度呈指数级增长。常见于循环神经网络（RNN）和极深的网络中。损失值会跳变为 `NaN`。解决方法：梯度裁剪（`torch.nn.utils.clip_grad_norm_`）、降低学习率或添加归一化层。

### 症状 2：损失值下降但模型表现不佳

损失值在下降，训练准确率达到了 99%，但测试准确率只有 55%。或者模型在真实数据上输出了毫无意义的结果。

**过拟合（Overfitting）。** 模型死记硬背训练数据，而非学习潜在规律。训练损失与验证损失之间的差距随时间不断扩大。解决方法：增加数据量、使用 `Dropout`、添加权重衰减（Weight Decay）、早停法（Early Stopping）或数据增强（Data Augmentation）。

**数据泄露（Data Leakage）。** 测试数据泄露到了训练集中。准确率高得可疑。常见原因：在划分数据集前进行打乱、使用完整数据集的统计信息进行预处理、不同划分间存在重复样本。解决方法：先划分数据集，再进行预处理，并检查重复样本。

**标签错误（Label Errors）。** 大多数真实数据集中有 5-10% 的标签是错误的（Northcutt 等人，2021 年——《测试集中普遍存在的标签错误》）。模型会学习到这些噪声。解决方法：使用置信学习（Confident Learning）查找并修正错误标签，或使用损失截断（Loss Truncation）忽略高损失样本。

### 症状 3：损失值出现 `NaN` 或 `Inf`

损失值变为 `nan` 或 `inf`。训练彻底停滞。

**学习率过高。** 梯度更新步长过大，导致权重爆炸。解决方法：将学习率降低 10 倍。

**`log(0)` 或 `log(负数)`。** 交叉熵损失（Cross-Entropy Loss）会计算 `log(p)`。如果模型输出恰好为 0 或负概率，对数运算将爆炸。解决方法：将预测值截断（clamp）到 `[eps, 1-eps]` 范围内，其中 `eps=1e-7`。

**除以零。** 批归一化会除以标准差。如果某个批次内的值完全相同，其标准差为 0。解决方法：在分母中添加 epsilon（PyTorch 默认已处理，但自定义实现可能未处理）。

**数值溢出（Numerical Overflow）。** 较大的激活值输入到 `exp()` 会产生 `Inf`。`Softmax` 函数尤其容易出现此问题。解决方法：在指数运算前减去最大值（即 log-sum-exp 技巧）。

### 技巧 1：梯度检查（Gradient Checking）

将解析梯度（来自反向传播）与数值梯度（来自有限差分法）进行对比。如果两者不一致，说明反向传播存在错误。

参数 `w` 的数值梯度：

grad_numerical = (loss(w + eps) - loss(w - eps)) / (2 * eps)

一致性指标（相对差异）：

rel_diff = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)

如果 `rel_diff < 1e-5`：正确。如果 `rel_diff > 1e-3`：几乎肯定存在错误。

flowchart LR
    A["Parameter w"] --> B["w + eps"]
    A --> C["w - eps"]
    B --> D["Forward pass"]
    C --> E["Forward pass"]
    D --> F["loss+"]
    E --> G["loss-"]
    F --> H["(loss+ - loss-) / 2eps"]
    G --> H
    H --> I["Compare to backprop gradient"]

### 技巧 2：激活值统计（Activation Statistics）

在训练过程中监控每一层后激活值的均值和标准差。健康的网络在经过归一化后，其激活值的均值应接近 0，标准差接近 1，或至少保持在有界范围内。

| 健康指标 | 均值 | 标准差 | 诊断结果 |
|-----------------|------|-----|-----------|
| 健康 | ~0 | ~1 | 网络正常学习 |
| 饱和 | >>0 或 <<0 | ~0 | 激活值卡在极值处 |
| 死亡 | 0 | 0 | 神经元死亡（全为零） |
| 爆炸 | >>10 | >>10 | 激活值无界增长 |

### 技巧 3：梯度流可视化（Gradient Flow Visualization）

绘制每一层的平均梯度幅值。在健康的网络中，各层的梯度幅值应大致相近。如果浅层的梯度比深层小 1000 倍，则说明出现了梯度消失。

graph LR
    subgraph "Healthy Gradient Flow"
        L1["Layer 1<br/>grad: 0.05"] --- L2["Layer 2<br/>grad: 0.04"] --- L3["Layer 3<br/>grad: 0.06"] --- L4["Layer 4<br/>grad: 0.05"]
    end

graph LR
    subgraph "Vanishing Gradient Flow"
        V1["Layer 1<br/>grad: 0.0001"] --- V2["Layer 2<br/>grad: 0.003"] --- V3["Layer 3<br/>grad: 0.02"] --- V4["Layer 4<br/>grad: 0.08"]
    end

### 技巧 4：单批次过拟合测试（Overfit-One-Batch Test）

这是深度学习中最重要的一项调试技巧。

选取一个小批次（8-32 个样本）。在其上训练 100 次以上迭代。损失值应降至接近零，训练准确率应达到 100%。如果未能达到，说明模型或训练循环存在根本性错误——请勿继续进行完整训练。

该测试可捕获以下问题：
- 损坏的损失函数
- 错误的反向传播
- 网络架构过小，无法拟合数据
- 优化器未正确绑定模型参数
- 数据与标签未对齐

运行此测试仅需 30 秒，却能节省数小时调试完整训练的时间。

### 技巧 5：学习率查找器（Learning Rate Finder）

Leslie Smith（2017 年）提出，在一个轮次内将学习率从极小值（`1e-7`）扫描至极大值（`10`），同时记录损失值。绘制损失值与学习率的关系图。最佳学习率通常比损失值下降最快时的学习率小约 10 倍。

graph TD
    subgraph "LR Finder Plot"
        direction LR
        A["1e-7: loss=2.3"] --> B["1e-5: loss=2.3"]
        B --> C["1e-3: loss=1.8"]
        C --> D["1e-2: loss=0.9 -- steepest"]
        D --> E["1e-1: loss=0.5"]
        E --> F["1.0: loss=NaN -- too high"]
    end

此示例中的最佳学习率：~`1e-3`（比最陡峭点低一个数量级）。

### 常见 PyTorch 错误

以下是 PyTorch 社区中耗费开发者时间最多的常见错误：

| 错误 | 症状 | 修复方法 |
|-----|---------|-----|
| 忘记调用 `optimizer.zero_grad()` | 梯度在批次间累积，损失值震荡 | 在 `loss.backward()` 前添加 `optimizer.zero_grad()` |
| 测试时忘记调用 `model.eval()` | `Dropout` 和批归一化行为异常，测试准确率在不同运行间波动 | 添加 `model.eval()` 和 `torch.no_grad()` |
| 张量形状错误 | 静默广播（Silent Broadcasting）导致结果错误，且不报错 | 调试时在每次操作后打印形状 |
| CPU/GPU 不匹配 | `RuntimeError: expected CUDA tensor` | 对模型和数据均使用 `.to(device)` |
| 未分离张量 | 计算图无限增长，导致内存溢出（OOM） | 使用 `.detach()` 或 `with torch.no_grad()` |
| 原地操作破坏自动求导 | `RuntimeError: modified by in-place operation` | 将 `x += 1` 替换为 `x = x + 1` |
| 数据未归一化 | 损失值卡在随机猜测水平 | 将输入归一化为均值=0，标准差=1 |
| 标签数据类型错误 | 交叉熵期望 `Long` 类型，实际传入 `Float` | 转换标签类型：`labels.long()` |

### 综合调试对照表

| 症状 | 可能原因 | 首要尝试操作 |
|---------|-------------|-------------------|
| 损失值卡在 `-log(1/num_classes)` | 模型输出均匀分布 | 检查数据流水线，验证标签与输入是否匹配 |
| 几步后损失值变为 `NaN` | 学习率过高 | 将学习率降低 10 倍 |
| 立即出现 `NaN` | `log(0)` 或除以零 | 在对数/除法运算中添加 epsilon |
| 损失值剧烈震荡 | 学习率过高或批次大小过小 | 降低学习率，增大批次大小 |
| 损失值下降后进入平台期 | 微调阶段学习率过高 | 添加学习率调度器（余弦衰减或阶梯衰减） |
| 训练准确率高，测试准确率低 | 过拟合 | 添加 `Dropout`、权重衰减或更多数据 |
| 训练准确率 = 测试准确率 = 随机水平 | 模型未学到任何内容 | 运行单批次过拟合测试 |
| 训练准确率 = 测试准确率，但两者均较低 | 欠拟合（Underfitting） | 使用更大模型、增加层数或特征 |
| 梯度全为零 | `ReLU` 神经元死亡或计算图被分离 | 改用 `LeakyReLU`，检查 `.requires_grad` |
| 训练期间内存溢出（OOM） | 批次过大或计算图未释放 | 减小批次大小，评估时使用 `torch.no_grad()` |

## 构建项目

一个用于监控激活值 (activations)、梯度 (gradients) 和损失曲线 (loss curves) 的诊断工具包。你将故意破坏一个神经网络，并使用该工具包来诊断每一个问题。

### 步骤 1：NetworkDebugger 类

通过钩子 (hooks) 机制接入 PyTorch 模型，以记录每一层的激活值与梯度统计信息。

import torch
import torch.nn as nn
import math


class NetworkDebugger:
    def __init__(self, model):
        self.model = model
        self.activation_stats = {}
        self.gradient_stats = {}
        self.loss_history = []
        self.lr_losses = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.LeakyReLU)):
                hook = module.register_forward_hook(self._make_activation_hook(name))
                self.hooks.append(hook)
                hook = module.register_full_backward_hook(self._make_gradient_hook(name))
                self.hooks.append(hook)

    def _make_activation_hook(self, name):
        def hook(module, input, output):
            with torch.no_grad():
                out = output.detach().float()
                self.activation_stats[name] = {
                    "mean": out.mean().item(),
                    "std": out.std().item(),
                    "fraction_zero": (out == 0).float().mean().item(),
                    "min": out.min().item(),
                    "max": out.max().item(),
                }
        return hook

    def _make_gradient_hook(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                with torch.no_grad():
                    grad = grad_output[0].detach().float()
                    self.gradient_stats[name] = {
                        "mean": grad.mean().item(),
                        "std": grad.std().item(),
                        "abs_mean": grad.abs().mean().item(),
                        "max": grad.abs().max().item(),
                    }
        return hook

    def record_loss(self, loss_value):
        self.loss_history.append(loss_value)

    def check_loss_health(self):
        if len(self.loss_history) < 2:
            return "NOT_ENOUGH_DATA"
        recent = self.loss_history[-10:]
        if any(math.isnan(v) or math.isinf(v) for v in recent):
            return "NAN_OR_INF"
        if len(self.loss_history) >= 20:
            first_half = sum(self.loss_history[:10]) / 10
            second_half = sum(self.loss_history[-10:]) / 10
            if second_half >= first_half * 0.99:
                return "NOT_DECREASING"
        if len(recent) >= 5:
            diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            if max(diffs) - min(diffs) > 2 * abs(sum(diffs) / len(diffs)):
                return "OSCILLATING"
        return "HEALTHY"

    def check_activations(self):
        issues = []
        for name, stats in self.activation_stats.items():
            if stats["fraction_zero"] > 0.5:
                issues.append(f"DEAD_NEURONS: {name} has {stats['fraction_zero']:.0%} zero activations")
            if abs(stats["mean"]) > 10:
                issues.append(f"EXPLODING_ACTIVATIONS: {name} mean={stats['mean']:.2f}")
            if stats["std"] < 1e-6:
                issues.append(f"COLLAPSED_ACTIVATIONS: {name} std={stats['std']:.2e}")
        return issues if issues else ["HEALTHY"]

    def check_gradients(self):
        issues = []
        grad_magnitudes = []
        for name, stats in self.gradient_stats.items():
            grad_magnitudes.append((name, stats["abs_mean"]))
            if stats["abs_mean"] < 1e-7:
                issues.append(f"VANISHING_GRADIENT: {name} abs_mean={stats['abs_mean']:.2e}")
            if stats["abs_mean"] > 100:
                issues.append(f"EXPLODING_GRADIENT: {name} abs_mean={stats['abs_mean']:.2e}")
        if len(grad_magnitudes) >= 2:
            first_mag = grad_magnitudes[0][1]
            last_mag = grad_magnitudes[-1][1]
            if last_mag > 0 and first_mag / last_mag > 100:
                issues.append(f"GRADIENT_RATIO: first/last = {first_mag/last_mag:.0f}x (vanishing)")
        return issues if issues else ["HEALTHY"]

    def print_report(self):
        print("\n=== NETWORK DEBUGGER REPORT ===")
        print(f"\nLoss health: {self.check_loss_health()}")
        if self.loss_history:
            print(f"  Last 5 losses: {[f'{v:.4f}' for v in self.loss_history[-5:]]}")
        print("\nActivation diagnostics:")
        for item in self.check_activations():
            print(f"  {item}")
        print("\nGradient diagnostics:")
        for item in self.check_gradients():
            print(f"  {item}")
        print("\nPer-layer activation stats:")
        for name, stats in self.activation_stats.items():
            print(f"  {name}: mean={stats['mean']:.4f} std={stats['std']:.4f} zero={stats['fraction_zero']:.1%}")
        print("\nPer-layer gradient stats:")
        for name, stats in self.gradient_stats.items():
            print(f"  {name}: abs_mean={stats['abs_mean']:.2e} max={stats['max']:.2e}")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

### 步骤 2：单批次过拟合测试 (Overfit-One-Batch Test)

def overfit_one_batch(model, x_batch, y_batch, criterion, lr=0.01, steps=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    print("\n=== OVERFIT ONE BATCH TEST ===")
    print(f"Batch size: {x_batch.shape[0]}, Steps: {steps}")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                preds = (output > 0).float() if output.shape[-1] == 1 else output.argmax(dim=1)
                targets = y_batch if y_batch.dim() == 1 else y_batch.squeeze()
                acc = (preds.squeeze() == targets).float().mean().item()
            print(f"  Step {step:3d} | Loss: {loss.item():.6f} | Accuracy: {acc:.1%}")

    final_loss = loss.item()
    if final_loss > 0.1:
        print(f"\n  FAIL: Loss did not converge ({final_loss:.4f}). Model or training loop is broken.")
        return False
    print(f"\n  PASS: Loss converged to {final_loss:.6f}")
    return True

### 步骤 3：学习率查找器 (Learning Rate Finder)

def find_learning_rate(model, x_data, y_data, criterion, start_lr=1e-7, end_lr=10, steps=100):
    import copy
    original_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1 / steps)

    model.train()
    results = []
    best_loss = float("inf")
    current_lr = start_lr

    print("\n=== LEARNING RATE FINDER ===")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_data)
        loss = criterion(output, y_data)

        if math.isnan(loss.item()) or loss.item() > best_loss * 10:
            break

        best_loss = min(best_loss, loss.item())
        results.append((current_lr, loss.item()))

        loss.backward()
        optimizer.step()

        current_lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

    model.load_state_dict(original_state)

    if len(results) < 10:
        print("  Could not complete LR sweep -- loss diverged too quickly")
        return results

    min_loss_idx = min(range(len(results)), key=lambda i: results[i][1])
    suggested_lr = results[max(0, min_loss_idx - 10)][0]

    print(f"  Swept {len(results)} steps from {start_lr:.0e} to {results[-1][0]:.0e}")
    print(f"  Minimum loss {results[min_loss_idx][1]:.4f} at lr={results[min_loss_idx][0]:.2e}")
    print(f"  Suggested learning rate: {suggested_lr:.2e}")

    return results

### 步骤 4：梯度检查器 (Gradient Checker)

def _flat_to_multi_index(flat_idx, shape):
    multi_idx = []
    remaining = flat_idx
    for dim in reversed(shape):
        multi_idx.insert(0, remaining % dim)
        remaining //= dim
    return tuple(multi_idx)


def gradient_check(model, x, y, criterion, eps=1e-4):
    model.train()
    x_double = x.double()
    y_double = y.double()
    model_double = model.double()

    print("\n=== GRADIENT CHECK ===")
    overall_max_diff = 0
    checked = 0

    for name, param in model_double.named_parameters():
        if not param.requires_grad:
            continue

        layer_max_diff = 0

        model_double.zero_grad()
        output = model_double(x_double)
        loss = criterion(output, y_double)
        loss.backward()
        analytical_grad = param.grad.clone()

        num_checks = min(5, param.numel())
        for i in range(num_checks):
            idx = _flat_to_multi_index(i, param.shape)
            original = param.data[idx].item()

            param.data[idx] = original + eps
            with torch.no_grad():
                loss_plus = criterion(model_double(x_double), y_double).item()

            param.data[idx] = original - eps
            with torch.no_grad():
                loss_minus = criterion(model_double(x_double), y_double).item()

            param.data[idx] = original

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytical = analytical_grad[idx].item()

            denom = max(abs(numerical), abs(analytical), 1e-8)
            rel_diff = abs(numerical - analytical) / denom

            layer_max_diff = max(layer_max_diff, rel_diff)
            checked += 1

        overall_max_diff = max(overall_max_diff, layer_max_diff)
        status = "OK" if layer_max_diff < 1e-5 else "MISMATCH"
        print(f"  {name}: max_rel_diff={layer_max_diff:.2e} [{status}]")

    model.float()

    print(f"\n  Checked {checked} parameters")
    if overall_max_diff < 1e-5:
        print("  PASS: Gradients match (rel_diff < 1e-5)")
    elif overall_max_diff < 1e-3:
        print("  WARN: Small differences (1e-5 < rel_diff < 1e-3)")
    else:
        print("  FAIL: Gradient mismatch detected (rel_diff > 1e-3)")
    return overall_max_diff

### 步骤 5：人为破坏的网络 (Deliberately Broken Networks)

现在将该工具包应用于存在缺陷的网络，并逐一进行诊断。

def demo_broken_networks():
    torch.manual_seed(42)
    x = torch.randn(64, 10)
    y = (x[:, 0] > 0).long()

    print("\n" + "=" * 60)
    print("BUG 1: Learning rate too high (lr=10)")
    print("=" * 60)
    model1 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger1 = NetworkDebugger(model1)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=10.0)
    criterion = nn.CrossEntropyLoss()
    for step in range(20):
        optimizer1.zero_grad()
        out = model1(x)
        loss = criterion(out, y)
        debugger1.record_loss(loss.item())
        loss.backward()
        optimizer1.step()
    debugger1.print_report()
    debugger1.remove_hooks()

    print("\n" + "=" * 60)
    print("BUG 2: Dead ReLUs from bad initialization")
    print("=" * 60)
    model2 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
    with torch.no_grad():
        for m in model2.modules():
            if isinstance(m, nn.Linear):
                m.weight.fill_(-1.0)
                m.bias.fill_(-5.0)
    debugger2 = NetworkDebugger(model2)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    for step in range(50):
        optimizer2.zero_grad()
        out = model2(x)
        loss = criterion(out, y)
        debugger2.record_loss(loss.item())
        loss.backward()
        optimizer2.step()
    debugger2.print_report()
    debugger2.remove_hooks()

    print("\n" + "=" * 60)
    print("BUG 3: Missing zero_grad (gradients accumulate)")
    print("=" * 60)
    model3 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger3 = NetworkDebugger(model3)
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.01)
    for step in range(50):
        out = model3(x)
        loss = criterion(out, y)
        debugger3.record_loss(loss.item())
        loss.backward()
        optimizer3.step()
    debugger3.print_report()
    debugger3.remove_hooks()

    print("\n" + "=" * 60)
    print("HEALTHY NETWORK: Correct setup for comparison")
    print("=" * 60)
    model_good = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger_good = NetworkDebugger(model_good)
    optimizer_good = torch.optim.Adam(model_good.parameters(), lr=1e-3)
    for step in range(50):
        optimizer_good.zero_grad()
        out = model_good(x)
        loss = criterion(out, y)
        debugger_good.record_loss(loss.item())
        loss.backward()
        optimizer_good.step()
    debugger_good.print_report()
    debugger_good.remove_hooks()

    print("\n" + "=" * 60)
    print("OVERFIT-ONE-BATCH TEST (healthy model)")
    print("=" * 60)
    model_test = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    overfit_one_batch(model_test, x[:8], y[:8], criterion)

    print("\n" + "=" * 60)
    print("LEARNING RATE FINDER")
    print("=" * 60)
    model_lr = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    find_learning_rate(model_lr, x, y, criterion)

    print("\n" + "=" * 60)
    print("GRADIENT CHECK")
    print("=" * 60)
    model_grad = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 2))
    gradient_check(model_grad, x[:4], y[:4], criterion)


## 使用方法

### PyTorch 内置工具

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

with torch.autograd.detect_anomaly():
    output = model(input_tensor)
    loss = criterion(output, target)
    loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_mean={param.grad.abs().mean():.2e}")

### Weights & Biases 集成

import wandb

wandb.init(project="debug-training")

for epoch in range(100):
    loss = train_one_epoch()
    wandb.log({
        "loss": loss,
        "lr": optimizer.param_groups[0]["lr"],
        "grad_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")),
    })

    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"grad/{name}": wandb.Histogram(param.grad.cpu().numpy())})

### TensorBoard

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/debug_experiment")

for epoch in range(100):
    loss = train_one_epoch()
    writer.add_scalar("Loss/train", loss, epoch)

    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param, epoch)
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, epoch)

### 调试检查清单（完整训练前）

1. 运行单批次过拟合（overfit-one-batch）测试。若失败则立即停止。
2. 打印模型摘要（model summary）——验证参数量是否合理。
3. 使用随机数据执行单次前向传播（forward pass）——检查输出形状（output shape）。
4. 训练 5 个轮次（epoch）——验证损失值（loss）是否下降。
5. 检查激活值统计信息——确保无死亡层（dead layers），无激活值爆炸。
6. 检查梯度流（gradient flow）——确保无梯度消失（vanishing gradients），无梯度爆炸（exploding gradients）。
7. 验证数据流水线（data pipeline）——打印 5 个带标签的随机样本。

## 部署上线

本章节将生成以下文件：
- `outputs/prompt-nn-debugger.md` —— 用于诊断神经网络训练故障的提示词（prompt）
- `outputs/skill-debug-checklist.md` —— 用于调试训练问题的决策树（decision-tree）检查清单

调试相关的关键部署模式：
- 在生产环境训练脚本中添加监控钩子（monitoring hooks）
- 每隔 N 个步骤将激活值和梯度统计信息记录至 W&B 或 TensorBoard
- 针对 NaN 损失（NaN loss）、死亡神经元（dead neurons，>80% 为零）或梯度爆炸（gradient explosion）实现自动告警
- 在更改模型架构（architecture）或数据流水线时，务必运行单批次过拟合测试

## 练习

1. **添加梯度爆炸检测器（Exploding Gradient Detector）。** 修改 `NetworkDebugger`，使其能够在梯度超过设定阈值时进行检测，并自动建议一个梯度裁剪（Gradient Clipping）值。在一个不含归一化层的 20 层网络上进行测试。

2. **构建死亡神经元复活器（Dead Neuron Resurrector）。** 编写一个函数，用于识别处于死亡状态的 ReLU 神经元（始终输出 0），并使用 Kaiming 初始化（Kaiming Initialization）重新初始化其输入权重。演示该方法如何恢复一个超过 70% 神经元已死亡的网络。

3. **实现带绘图功能的学习率查找器（Learning Rate Finder）。** 扩展 `find_learning_rate` 函数，将结果保存为 CSV 文件，并编写一个独立的脚本来读取该 CSV 文件，使用 matplotlib 绘制学习率（LR）与损失（Loss）的关系曲线。找出 ResNet-18 在 CIFAR-10 数据集上的最佳学习率。

4. **创建数据流水线验证器（Data Pipeline Validator）。** 编写一个函数，用于检查以下问题：训练集/测试集划分中的重复样本、标签分布不平衡（比例超过 10:1）、输入数据归一化情况（均值接近 0，标准差接近 1），以及数据中是否存在 NaN/Inf 值。在一个人为构造的损坏数据集上运行该函数。

5. **调试真实故障。** 使用第 10 课中的微型框架，引入一个隐蔽的 Bug（例如在 `backward` 中错误地转置了权重矩阵），并利用梯度检查（Gradient Checking）精确定位哪个参数的梯度计算有误。记录完整的调试过程。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 静默 Bug (Silent Bug) | “代码能跑但结果很差” | 一种不会抛出错误但会降低模型质量的 Bug——这是机器学习中最主要的故障模式 |
| 死亡 ReLU (Dead ReLU) | “神经元死了” | 输入始终为负的 ReLU 神经元，导致其输出恒为 0 且永久接收不到梯度 |
| 梯度消失 (Vanishing Gradients) | “浅层网络停止学习” | 梯度在逐层传递过程中呈指数级衰减，导致浅层权重实际上被冻结 |
| 梯度爆炸 (Exploding Gradients) | “损失值变成了 NaN” | 梯度在逐层传递过程中呈指数级增长，导致权重更新幅度过大而发生数值溢出 |
| 梯度检查 (Gradient Checking) | “验证反向传播是否正确” | 将反向传播计算出的解析梯度与通过有限差分法计算的数值梯度进行对比 |
| 单批次过拟合测试 (Overfit-One-Batch) | “最重要的调试测试” | 仅使用单个小批次数据进行训练，以验证模型是否具备学习能力——如果连这都做不到，说明底层逻辑存在根本性错误 |
| 学习率查找器 (LR Finder) | “扫描以寻找合适的学习率” | 在一个 Epoch 内指数级递增学习率，并选取损失值开始发散前的那个学习率 |
| 数据泄露 (Data Leakage) | “测试数据泄露到了训练集中” | 测试集的信息污染了训练过程，导致模型准确率虚高 |
| 激活统计量 (Activation Statistics) | “监控层健康状态” | 跟踪每一层输出的均值、标准差和零值比例，以检测死亡、饱和或爆炸的神经元 |
| 梯度裁剪 (Gradient Clipping) | “限制梯度幅值” | 当梯度范数超过阈值时按比例缩小梯度，防止梯度爆炸导致的更新异常 |

## 扩展阅读

- Smith，《用于训练神经网络的循环学习率》（2017）—— 引入了学习率范围测试（Learning Rate Range Test，LR finder）的论文
- Northcutt 等人，《测试集中普遍存在的标签错误会破坏机器学习基准的稳定性》（2021）—— 证明了 ImageNet、CIFAR-10 及其他主要基准（benchmarks）数据集中有 3% 至 6% 的标签存在错误
- Zhang 等人，《理解深度学习需要重新思考泛化》（2017）—— 该论文表明神经网络能够记忆随机标签，这也正是过拟合单批次测试（overfit-one-batch test）有效的原因
- PyTorch 关于 `torch.autograd.detect_anomaly` 和 `torch.autograd.set_detect_anomaly` 的文档，用于内置的 NaN/Inf 检测