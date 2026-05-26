# 数值稳定性 (Numerical Stability)

> 浮点数是一种“泄漏的抽象”（leaky abstraction）。它会在训练过程中突然发难，让你措手不及。

**类型：** 构建实践 (Build)
**语言：** Python
**前置要求：** 第一阶段，第 01-04 课
**预计耗时：** 约 120 分钟

## 学习目标

- 使用最大值相减法（max-subtraction trick）实现数值稳定的 softmax 与 log-sum-exp
- 识别浮点计算中的上溢（overflow）、下溢（underflow）与灾难性抵消（catastrophic cancellation）
- 使用中心差分法（centered finite differences）验证解析梯度（analytical gradients）与数值梯度（numerical gradients）
- 解释为何训练时优先使用 bfloat16 而非 float16，以及损失缩放（loss scaling）如何防止梯度下溢

## 问题所在

你的模型训练了三个小时，随后损失值（loss）变成了 `NaN`。你添加了一条打印语句。在第 9,000 步时，logits 还是正常的。到了第 9,001 步，它们变成了 `inf`。等到第 9,002 步，所有梯度都变成了 `nan`，训练彻底崩溃。

又或者：你的模型顺利训练完成，但准确率比论文声称的低了 2%。你检查了所有环节。架构一致。超参数一致。数据也一致。问题在于，论文使用的是 float32，而你直接用了 float16 却没有进行正确的缩放。32 位浮点运算中累积的舍入误差（rounding error）悄无声息地吞噬了你的准确率。

再或者：你从零开始实现交叉熵损失（cross-entropy loss）。它在处理较小的 logits 时运行正常。但当 logits 超过 100 时，它返回了 `inf`。这是因为 `exp(100)` 超出了 float32 的表示范围，导致 softmax 发生上溢。所有主流的机器学习框架都通过短短两行代码的技巧解决了这个问题，而你却不知道这个技巧的存在。

数值稳定性绝非纸上谈兵的理论问题。它决定了训练是成功还是悄无声息地失败。你未来调试的每一个严重的机器学习 bug，归根结底都会追溯到浮点数运算上。

## 核心概念

### IEEE 754：计算机如何存储实数

计算机遵循 IEEE 754 标准，将实数存储为浮点数（floating point）。一个浮点数包含三个部分：符号位（sign bit）、指数（exponent）和尾数（mantissa）。

Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa

尾数决定精度（有效数字的位数）。指数决定数值范围（数字能表示的最大或最小值）。

Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38

float32 提供大约 7 位十进制精度。这意味着它能区分 1.0000001 和 1.0000002，但无法区分 1.00000001 和 1.00000002。超过 7 位之后的数字全是舍入噪声（rounding noise）。

float16 仅提供约 3 位精度。它能表示的最大数值是 65,504。在机器学习（machine learning, ML）中，这个范围小得令人不安，因为对数几率（logits）、梯度（gradients）和激活值（activations）经常超出此限制。

bfloat16 是 Google 为解决 float16 范围问题而提出的方案。它拥有与 float32 相同的 8 位指数（范围相同，最高达 3.4e38），但只有 7 位尾数（精度低于 float16）。在训练神经网络（neural networks）时，数值范围比精度更重要，因此 bfloat16 通常更胜一筹。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 无法在二进制浮点数中精确表示。在二进制下，它是一个无限循环小数：

0.1 in binary = 0.0001100110011001100110011... (repeating forever)

Float32 将其截断为 23 位尾数。存储的值约为 0.100000001490116。类似地，0.2 存储为约 0.200000002980232。它们的和是 0.300000004470348，而不是 0.3。

In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False

这对机器学习至关重要，因为：

1. 类似 `if loss < threshold` 的损失比较可能会得出错误结果
2. 累加大量微小值（数千步的梯度更新）会导致结果偏离真实总和
3. 如果使用 `==` 比较浮点数，校验和与可复现性测试将会失败

解决方法：永远不要用 `==` 比较浮点数。请使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消（Catastrophic Cancellation）

当两个几乎相等的浮点数相减时，有效数字会相互抵消，剩下的舍入噪声会被提升为高位有效数字。

a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%

仅一次减法就产生了 19% 的相对误差。在机器学习中，以下情况会引发此问题：

- 计算均值较大的数据方差：当 `E[x]` 较大时，使用 `E[x^2] - E[x]^2`
- 相减几乎相等的对数概率（log-probabilities）
- 使用过小的 epsilon 计算有限差分梯度（finite-difference gradients）

解决方法：重新排列公式，避免大数相减。对于方差计算，使用 Welford 算法或先对数据进行中心化。对于对数概率，全程在对数空间（log-space）中进行运算。

### 上溢（Overflow）与下溢（Underflow）

当结果过大无法表示时发生上溢。当结果过小（比可表示的最小正数更接近零）时发生下溢。

Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0

在机器学习中，`exp()` 函数是上溢的主要来源：

exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)

而 `log()` 函数则会导致相反方向的问题：

log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)

在机器学习中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵（cross-entropy）、对数似然（log-likelihoods）和 KL 散度（KL divergence）中。如果没有正确的技巧，组合使用 `log(exp(x))` 极易踩坑。

### Log-Sum-Exp 技巧（Log-Sum-Exp Trick）

直接计算 `log(sum(exp(x_i)))` 在数值上非常危险。如果任意 `x_i` 较大，`exp(x_i)` 会上溢。如果所有 `x_i` 都非常小（负值），每个 `exp(x_i)` 都会下溢为零，导致 `log(0)` 返回 `-inf`。

技巧：在取指数前减去最大值。

log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))

原理：减去 `max(x)` 后，最大的指数项变为 `exp(0) = 1`，因此不可能发生上溢。求和项中至少有一项为 1，总和至少为 1，且 `log(1) = 0`，因此也不可能下溢至 `-inf`。

证明：

log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))

令 `c = max(x)` 即可消除上溢。

该技巧在机器学习中无处不在：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型（Mixture of Gaussians）
- 变分推断（Variational inference）

### 为什么 Softmax 需要最大值相减技巧

Softmax 将对数几率（logits）转换为概率：

softmax(x_i) = exp(x_i) / sum(exp(x_j))

如果不使用该技巧，logits 为 [100, 101, 102] 时将导致上溢：

exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.

使用该技巧后，减去 max(x) = 102：

exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]

计算出的概率完全相同，但计算过程是安全的。这不是性能优化，而是保证正确性的必要条件。

### NaN 与 Inf：检测与预防

`nan`（非数字，Not a Number）和 `inf`（无穷大，infinity）会在计算中像病毒一样传播。梯度更新中一旦出现一个 `nan`，权重就会变成 `nan`，进而导致后续所有输出均为 `nan`。训练会在一步之内彻底崩溃。

`inf` 的产生原因：
- 对较大的正数调用 `exp()`
- 除以零：`1.0 / 0.0`
- 累加过程中的 `float32` 上溢

`nan` 的产生原因：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数调用 `sqrt()`
- 对负数调用 `log()`
- 任何涉及已有 `nan` 的算术运算

检测方法：

import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf

预防策略：

1. 限制 `exp()` 的输入范围：`exp(clamp(x, -80, 80))`
2. 在分母添加 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 内部添加 epsilon：`log(x + 1e-8)`
4. 使用数值稳定的实现（如 log-sum-exp、stable softmax）
5. 使用梯度裁剪（gradient clipping）防止权重爆炸
6. 调试时，在每次前向传播（forward pass）后检查 `nan`/`inf`

### 数值梯度检查（Numerical Gradient Checking）

解析梯度（来自反向传播，backpropagation）可能存在 bug。数值梯度检查通过有限差分法（finite differences）计算梯度来进行验证。

中心差分公式：

df/dx ~= (f(x + h) - f(x - h)) / (2h)

该公式具有 O(h^2) 的精度，远优于仅具有 O(h) 精度的前向差分 `(f(x+h) - f(x)) / h`。

选择 h：过大则近似结果错误；过小则灾难性抵消会破坏结果。通常 `h = 1e-5` 到 `1e-7` 较为合适。

检查方法：计算解析梯度与数值梯度之间的相对误差。

relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)

经验法则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，大概率正确
- relative_error > 1e-3：存在问题
- relative_error > 1：梯度完全错误

在实现新层或损失函数时，务必检查梯度。PyTorch 提供了 `torch.autograd.gradcheck()` 用于此目的。

### 混合精度训练（Mixed Precision Training）

现代 GPU 配备了专用硬件（Tensor Cores），其 float16 矩阵乘法速度比 float32 快 2 到 8 倍。混合精度训练正是利用了这一点：

1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights

纯 float16 训练的问题：梯度通常非常小（1e-8 或更小）。Float16 会将低于约 6e-8 的值下溢为零。由于所有梯度更新都变为零，模型将停止学习。

解决方法是损失缩放（loss scaling）：

1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow

动态损失缩放会自动调整缩放因子。初始值设为较大数（如 65536）。如果梯度上溢为 `inf`，则将其减半。如果连续 N 步未发生上溢，则将其翻倍。

### bfloat16 与 float16：为什么 bfloat16 更适合训练

float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]

float16 精度更高（10 位尾数 vs 7 位），但范围有限（最大约 65,504）。bfloat16 精度较低，但拥有与 float32 相同的范围（最大约 3.4e38）。

在训练神经网络时：

- 训练过程中，激活值和 logits 经常因峰值超过 65,504。float16 会上溢，而 bfloat16 能妥善处理。
- float16 必须进行损失缩放，而 bfloat16 通常不需要，因为其范围足以覆盖梯度的量级分布。
- bfloat16 本质上是 float32 的简单截断：直接丢弃尾数的低 16 位。转换过程极其简单，且指数部分无损。

在推理阶段，由于数值有界且精度更重要，通常首选 float16。而在训练阶段，范围更重要，因此首选 bfloat16。这也是 TPU 和现代 NVIDIA GPU（A100、H100）原生支持 bfloat16 的原因。

### 梯度裁剪（Gradient Clipping）

梯度爆炸（exploding gradients）发生在梯度经过多层网络呈指数级增长时（常见于 RNN、深度网络和 Transformer）。单个巨大的梯度可能在一步之内破坏所有权重。

两种裁剪方式：

**按值裁剪（Clip by value）：** 独立限制每个梯度元素的值。

grad = clamp(grad, -max_val, max_val)

实现简单，但可能会改变梯度向量的方向。

**按范数裁剪（Clip by norm）：** 缩放整个梯度向量，使其范数不超过设定阈值。

if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)

保留梯度方向。`torch.nn.utils.clip_grad_norm_()` 正是采用此方法，也是业界的标准选择。

典型取值：Transformer 常用 `max_norm=1.0`，强化学习（RL）常用 `max_norm=0.5`，较简单的网络常用 `max_norm=5.0`。

梯度裁剪并非权宜之计，而是一种安全机制。没有它，单个异常批次（outlier batch）产生的巨大梯度足以毁掉数周的训练成果。

### 归一化层作为数值稳定器

批归一化（Batch normalization）、层归一化（Layer normalization）和 RMS 归一化（RMS normalization）通常被介绍为帮助训练收敛的正则化器（regularizers）。它们同时也是数值稳定器。

如果没有归一化，激活值可能会在网络层间呈指数级增长或衰减：

Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]

归一化会在每一层对激活值进行重新居中和缩放：

LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta

`epsilon`（通常为 1e-5）用于防止所有激活值相同时发生除以零的错误。可学习参数 `gamma` 和 `beta` 允许网络恢复所需的任意尺度。

这使得整个网络中的数值保持在安全范围内，既防止了前向传播中的上溢，也避免了反向传播中的梯度爆炸。

### 常见的机器学习数值 Bug

**Bug：训练几个 epoch 后 Loss 变为 NaN。**
原因：logits 过大导致 softmax 上溢。或学习率过高导致权重发散。
解决：使用稳定的 softmax（最大值相减）、降低学习率、添加梯度裁剪。

**Bug：Loss 卡在 log(num_classes)。**
原因：模型输出接近均匀概率。通常意味着梯度消失（vanishing gradients）或模型根本没有在学习。
解决：检查数据标签是否正确、验证损失函数、检查是否存在“死亡”的 ReLU 神经元（dead ReLUs）。

**Bug：验证集准确率比预期低 1-3%。**
原因：使用混合精度但未正确进行损失缩放。梯度下溢会静默地将微小更新清零。
解决：启用动态损失缩放，或切换至 bfloat16。

**Bug：某些层的梯度范数为 0.0。**
原因：ReLU 神经元死亡（所有输入为负），或 float16 下溢。
解决：改用 LeakyReLU 或 GELU、使用梯度缩放、检查权重初始化。

**Bug：模型在一块 GPU 上运行正常，在另一块上结果不同。**
原因：浮点数累加顺序的非确定性。GPU 并行归约在不同硬件上的求和顺序不同，而浮点加法不满足结合律。
解决：接受微小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受性能损耗。

**Bug：损失计算中 `exp()` 返回 `inf`。**
原因：未使用最大值相减技巧，直接将原始 logits 传入 `exp()`。
解决：使用 `torch.nn.functional.log_softmax()`，其内部已实现 log-sum-exp。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示低于 6e-8 的梯度量级或高于 65,504 的激活值。
解决：使用带损失缩放的混合精度（AMP），或改用 bfloat16。

## 动手构建

### 步骤 1：演示浮点数精度（Floating Point Precision）限制

print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")

### 步骤 2：实现朴素与稳定的 Softmax（Softmax）

import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]

### 步骤 3：实现稳定的对数-指数和（Log-Sum-Exp）

def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf

### 步骤 4：实现稳定的交叉熵（Cross-Entropy）

def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")

### 步骤 5：梯度检查（Gradient Checking）

def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)

## 实际应用

### 混合精度模拟 (Mixed Precision Simulation)

import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]

### 梯度裁剪 (Gradient Clipping)

def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")

### NaN/Inf 检测 (NaN/Inf Detection)

def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])

完整实现及所有边界情况演示请参见 `code/numerical.py`。

## 交付 (Ship It)

本章节将生成：
- 包含稳定版 Softmax (Softmax)、对数求和指数 (Log-Sum-Exp)、交叉熵 (Cross-Entropy)、梯度检查 (Gradient Checking) 以及混合精度模拟的 `code/numerical.py`
- 用于诊断训练过程中 NaN/Inf 及数值问题的 `outputs/prompt-numerical-debugger.md`

这些稳定的实现将在第 3 阶段构建训练循环 (Training Loop) 以及第 4 阶段实现注意力机制 (Attention Mechanism) 时再次出现。

## 练习

1. **灾难性抵消（Catastrophic cancellation）**。使用 `float32` 和朴素公式 `E[x^2] - E[x]^2` 计算 `[1000000.0, 1000001.0, 1000002.0]` 的方差。随后使用韦尔福德在线算法（Welford's online algorithm）重新计算。将计算误差与真实方差（0.6667）进行对比。

2. **精度探索（Precision hunt）**。在 Python 中寻找最小的正 `float32` 值 `x`，使得 `1.0 + x == 1.0` 成立。该值即为机器精度（machine epsilon）。请验证其结果是否与 `numpy.finfo(numpy.float32).eps` 一致。

3. **Log-sum-exp 边界情况（Log-sum-exp edge cases）**。使用以下场景测试你的 `logsumexp_stable` 函数：(a) 所有数值相等；(b) 某一数值远大于其他数值；(c) 所有数值均为极小的负数（-1000）。验证在朴素实现（naive version）失效的场景下，该函数仍能输出正确结果。

4. **神经网络层梯度检查（Gradient checking a neural network layer）**。实现一个单层线性层 `y = Wx + b` 及其解析反向传播（analytical backward pass）。使用 `numerical_gradient` 验证 3x2 权重矩阵下的计算正确性。

5. **损失缩放实验（Loss scaling experiment）**。模拟使用 `float16` 的训练过程：生成范围在 `[1e-9, 1e-3]` 的随机梯度，将其转换为 `float16`，并统计下溢为零的比例。随后应用损失缩放（乘以 1024），转换为 `float16`，再逆向缩放回原值，并再次统计变为零的比例。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| IEEE 754 | “浮点数标准” | 定义二进制浮点数格式、舍入规则及特殊值（inf、nan）的国际标准。所有现代 CPU 和 GPU 均遵循该标准实现。 |
| 机器精度 (Machine epsilon) | “精度极限” | 在给定浮点数格式下，满足 1.0 + e != 1.0 的最小值 e。对于 float32，该值约为 1.19e-7。 |
| 灾难性抵消 (Catastrophic cancellation) | “相减导致的精度丢失” | 当两个非常接近的浮点数相减时，有效数字相互抵消，舍入误差将主导最终结果。 |
| 上溢 (Overflow) | “数值过大” | 计算结果超出可表示的最大值，变为 inf。例如，exp(89) 会导致 float32 上溢。 |
| 下溢 (Underflow) | “数值过小” | 计算结果比可表示的最小正数更接近零，从而变为 0.0。例如，exp(-104) 会导致 float32 下溢。 |
| Log-sum-exp 技巧 (Log-sum-exp trick) | “先减去最大值” | 通过提取 exp(max(x)) 公因子来计算 log(sum(exp(x)))，以防止上溢和下溢。广泛应用于 softmax、交叉熵 (cross-entropy) 及对数概率计算中。 |
| 稳定版 Softmax (Stable softmax) | “不会数值爆炸的 Softmax” | 在进行指数运算前减去 max(logits)。数值结果完全一致，且彻底避免上溢。 |
| 梯度检查 (Gradient checking) | “验证反向传播” | 将反向传播 (backpropagation) 得到的解析梯度与有限差分法 (finite differences) 计算的数值梯度进行对比，以排查实现中的错误。 |
| 混合精度 (Mixed precision) | “前向传播用 Float16，反向传播用 Float32” | 在对速度要求高的操作中使用低精度浮点数，在对数值敏感的操作中使用高精度浮点数。通常可带来 2-3 倍的加速。 |
| 损失缩放 (Loss scaling) | “防止梯度下溢” | 在反向传播前将损失值乘以一个较大的常数，使梯度保持在 float16 的可表示范围内，随后在更新权重前除以相同的常数。 |
| bfloat16 | “Brain 浮点数” | Google 推出的 16 位浮点格式，包含 8 位指数位（动态范围与 float32 相同）和 7 位尾数位（精度低于 float16）。是模型训练的首选格式。 |
| 梯度裁剪 (Gradient clipping) | “限制梯度范数” | 对梯度向量进行缩放，使其范数不超过设定阈值。可防止梯度爆炸破坏模型权重。 |
| NaN | “非数字” | 由未定义运算（如 0/0、inf-inf、sqrt(-1)）产生的特殊浮点值。该值会在后续所有算术运算中持续传播。 |
| Inf | “无穷大” | 由上溢或除以零产生的特殊浮点值。参与特定运算时可能生成 NaN（如 inf - inf、inf * 0）。 |
| 数值梯度 (Numerical gradient) | “暴力求导” | 通过计算 f(x+h) 和 f(x-h) 并除以 2h 来近似导数。计算较慢，但用于验证时非常可靠。 |

## 进一步阅读

- [每位计算机科学家都应了解的浮点运算（Floating-Point Arithmetic）(Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考资料，内容密集但完整全面
- [混合精度训练（Mixed Precision Training）(Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- NVIDIA 发表的论文，首次为 float16 训练引入了损失缩放（Loss Scaling）技术
- [AMP：自动混合精度（Automatic Mixed Precision）(PyTorch 文档)](https://pytorch.org/docs/stable/amp.html) -- PyTorch 中混合精度训练的实用指南
- [bfloat16 格式 (Google Cloud TPU 文档)](https://cloud.google.com/tpu/docs/bfloat16) -- Google 为何在 TPU 中采用该格式
- [Kahan 求和算法（Kahan Summation）(维基百科)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 用于降低浮点数求和过程中舍入误差（Rounding Error）的算法