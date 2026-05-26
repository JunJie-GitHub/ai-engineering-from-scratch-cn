# JAX 简介

> PyTorch 会原地修改张量（tensor）。TensorFlow 构建计算图（computation graph）。JAX 编译纯函数（pure function）。最后这一点将彻底改变你对深度学习（deep learning）的思考方式。

**类型：** 构建
**语言：** Python
**前置要求：** 第 03 阶段第 01-10 课，NumPy 基础
**时长：** 约 90 分钟

## 学习目标

- 使用 JAX 的函数式 API（functional API）（`jax.numpy`、`jax.grad`、`jax.jit`、`jax.vmap`）编写纯函数风格的神经网络代码
- 解释 PyTorch 的即时原地修改（eager mutation）模式与 JAX 的函数式编译模型（functional compilation model）之间的核心设计差异
- 应用 JIT 编译（just-in-time compilation）和 vmap 向量化（vectorization）技术，相较于原生 Python 加速训练循环
- 在 JAX 中训练一个简单网络，并将其显式状态管理（explicit state management）与 PyTorch 的面向对象（object-oriented）方法进行对比

## 问题所在

你已经知道如何在 PyTorch 中构建神经网络。你定义一个 `nn.Module`，调用 `.backward()`，然后让优化器执行一步更新。这套流程行之有效，且已被数百万开发者广泛使用。

但 PyTorch 的底层架构中存在一个固有约束：它在 Python 中以即时执行模式逐条追踪操作。每一次 `tensor + tensor` 都会触发一次独立的内核（kernel）启动。每一个训练步骤都会重新解释相同的 Python 代码。在常规规模下这完全没问题，但当你需要在 2,048 个 TPU 上训练一个拥有 5400 亿参数的模型时，这种模式带来的开销就会成为致命瓶颈。

Google DeepMind 使用 JAX 训练 Gemini，Anthropic 也使用 JAX 训练了 Claude。这些绝非小规模实验，而是目前地球上规模最大的神经网络训练任务。他们选择 JAX 的原因在于：JAX 将你的训练循环视为一个可编译的程序，而非一系列 Python 函数调用。

JAX 本质上是拥有三大“超能力”的 NumPy：自动微分（automatic differentiation）、面向 XLA 的 JIT 编译，以及自动向量化（automatic vectorization）。你只需编写一个处理单个样本的函数，JAX 就能为你生成一个能够处理批量数据、计算梯度、编译为机器码并在多设备上运行的函数。而这一切，都无需修改你最初编写的函数。

## 核心概念

### JAX 的设计哲学 (JAX Philosophy)

JAX 是一个函数式框架 (Functional Framework)。没有类，没有可变状态，也没有 `.backward()` 方法。取而代之的是：

| PyTorch | JAX |
|---------|-----|
| 带有状态的 `nn.Module` 类 | 纯函数 (Pure Function)：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 动态图执行 (Eager Execution) | 通过 XLA 进行即时编译 (JIT Compilation) |
| `for x in batch:` 手动循环 | `jax.vmap(f)` 自动向量化 (Auto-vectorization) |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行化 (Auto-parallelism) |
| 可变的 `model.parameters()` | 不可变 (Immutable) 的数组树状结构 (Pytree) |

这并非单纯的编程风格偏好，而是编译器的硬性约束。即时编译 (JIT Compilation) 要求函数必须是纯函数——相同的输入永远产生相同的输出，且无任何副作用。正是这一限制，使得百倍级别的加速成为可能。

### jax.numpy：熟悉的表层接口

JAX 在硬件加速器上重新实现了 NumPy API：

import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)

相同的函数名，相同的广播规则 (Broadcasting Rules)，相同的切片语义。但数组实际驻留在 GPU/TPU 上，且每个操作都可被编译器追踪。

一个关键区别在于：JAX 数组是不可变的。不能使用 `a[0] = 5`，而必须写成 `a = a.at[0].set(5)`。起初可能会觉得别扭，但适应一周后就会豁然开朗——正是不可变性，使得 `grad`、`jit` 和 `vmap` 等变换能够自由组合。

### jax.grad：函数式自动微分 (Functional Autodiff)

PyTorch 将梯度附加在张量上（`.grad`），而 JAX 将梯度附加在函数上。

import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)

`jax.grad` 接收一个函数，并返回一个用于计算梯度的新函数。无需调用 `.backward()`，也不会在张量上存储计算图 (Computation Graph)。梯度本身只是一个可以调用、组合或进行即时编译的普通函数。

它可以任意组合：

d2f = jax.grad(jax.grad(f))
d2f(3.0)

二阶导数、三阶导数、雅可比矩阵 (Jacobians)、海森矩阵 (Hessians)。全部只需组合 `grad` 即可实现。PyTorch 也能做到这一点（如 `torch.autograd.functional.hessian`），但属于后期附加的功能。而在 JAX 中，这是底层基石。

约束条件：`grad` 仅适用于纯函数。内部不能有 `print` 语句（它们会在追踪 (Tracing) 阶段而非执行阶段运行）。不能修改外部状态。没有显式的密钥管理就不能生成随机数。

### jit：编译至 XLA

@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)

在首次调用时，JAX 会对函数进行追踪——记录发生了哪些操作，但不实际执行它们。随后，JAX 将该追踪结果交给 XLA（Accelerated Linear Algebra，加速线性代数编译器），这是 Google 为 TPU 和 GPU 开发的编译器。XLA 会融合操作、消除冗余的内存拷贝，并生成优化后的机器码。

后续调用将完全跳过 Python 层。编译后的代码将以 C++ 级别的速度在加速器上运行。

JIT 适用的场景：
- 训练步骤（相同的计算重复数千次）
- 推理阶段（相同的模型，不同的输入）
- 任何被多次调用且输入形状相似的函数

JIT 不适用的场景：
- 包含依赖具体值的 Python 控制流的函数（例如 `if x > 0`，其中 `x` 是被追踪的数组）
- 一次性计算（编译开销超过实际运行时间）
- 调试过程（追踪机制会掩盖实际执行流程）

控制流的限制是真实存在的。`jax.lax.cond` 替代了 `if/else`，`jax.lax.scan` 替代了 `for` 循环。这些并非可选项，而是为了获得编译加速所必须付出的代价。

### vmap：自动向量化 (Automatic Vectorization)

你编写一个处理单个样本的函数：

def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']

`vmap` 将其提升为处理整个批次 (Batch)：

batch_predict = jax.vmap(predict, in_axes=(None, 0))

`in_axes=(None, 0)` 的含义是：不对 `params` 进行批处理（共享参数），仅对 `x` 的第 0 轴进行批处理。无需手动编写 `for` 循环，无需重塑形状，也无需手动传递批次维度。JAX 会自动推断批次维度并对整个计算过程进行向量化。

这绝非语法糖。`vmap` 会生成融合后的向量化代码，其运行速度比 Python 循环快 10 到 100 倍。并且它能与 `jit` 和 `grad` 自由组合：

per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))

逐样本梯度 (Per-example Gradients)。仅需一行代码。在 PyTorch 中，若不借助一些非常规手段，这几乎是不可能实现的。

### pmap：跨设备数据并行 (Data Parallelism)

parallel_step = jax.pmap(train_step, axis_name='devices')

`pmap` 会在所有可用设备（GPU/TPU）上复制该函数，并自动切分批次数据。在函数内部，可通过 `jax.lax.pmean` 和 `jax.lax.psum` 实现跨设备的梯度同步。

Google 正是使用 `pmap`（及其后继者 `shard_map`）在数千块 TPU v5e 芯片上训练 Gemini 模型的。其编程范式非常简单：编写单设备版本的代码，用 `pmap` 包装，即可完成。

### Pytrees：通用数据结构

JAX 基于“Pytree”进行操作——即列表、元组、字典和数组的嵌套组合。你的模型参数就是一个 Pytree：

params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}

JAX 的每一项变换（`grad`、`jit`、`vmap`）都懂得如何遍历 Pytree。`jax.tree.map(f, tree)` 会将函数 `f` 应用到每一个叶子节点上。优化器正是通过这种方式一次性更新所有参数：

params = jax.tree.map(lambda p, g: p - lr * g, params, grads)

无需 `.parameters()` 方法，也无需注册参数。树状结构本身就是模型。

### 函数式 vs 面向对象

PyTorch 将状态存储在对象内部：

class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)

JAX 使用带有显式状态的纯函数：

def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']

参数通过显式传入。不存储任何状态，不修改任何数据。这使得每个函数都易于测试、组合和编译。这也意味着你需要自行管理参数——或者借助 Flax 或 Equinox 等库来代劳。

### JAX 生态系统

JAX 提供底层原语 (Primitives)，而第三方库则提供易用性封装 (Ergonomics)：

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | 神经网络层 | 带有显式状态的 `nn.Module` |
| **Equinox** (Patrick Kidger) | 神经网络层 | 基于 Pytree，符合 Python 风格 |
| **Optax** (DeepMind) | 优化器 + 学习率调度 | 可组合的梯度变换 |
| **Orbax** (Google) | 模型检查点 | 保存/恢复 Pytree |
| **CLU** (Google) | 指标计算 + 日志记录 | 训练循环工具集 |

Optax 是标准的优化器库。它将梯度变换（如 Adam、SGD、梯度裁剪）与参数更新分离开来，使得组合变得极其简单：

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)

### 何时选择 JAX 还是 PyTorch

| Factor | JAX | PyTorch |
|--------|-----|---------|
| TPU 支持 | 原生一等支持（Google 同时开发了两者） | 社区维护（torch_xla） |
| GPU 支持 | 良好（通过 XLA 调用 CUDA） | 业界最佳（原生 CUDA） |
| 调试难度 | 较难（追踪 + 编译机制） | 简单（动态图，逐行执行） |
| 生态系统 | 偏向学术研究（Flax, Equinox） | 极其庞大（HuggingFace, torchvision 等） |
| 人才招聘 | 小众（Google/DeepMind/Anthropic） | 主流（无处不在） |
| 大规模训练 | 卓越（XLA, pmap, mesh） | 良好（FSDP, DeepSpeed） |
| 原型开发速度 | 较慢（函数式范式有额外开销） | 较快（直接修改状态即可） |
| 生产环境推理 | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| 主要使用者 | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

坦诚的建议：除非你有特定理由，否则请优先使用 PyTorch。这些特定理由包括：需要访问 TPU、必须计算逐样本梯度、需要在超大规模下进行多设备训练，或者你正在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数生成

JAX 没有全局随机状态。每次随机操作都需要显式传入一个伪随机数生成器密钥 (PRNG Key)：

key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))

起初可能会觉得繁琐。但它保证了跨设备和跨编译过程的可复现性——这是 PyTorch 的 `torch.manual_seed` 在多 GPU 环境下无法保证的特性。

## 构建项目

### 步骤 1：环境设置与数据准备

我们将使用 JAX 和 Optax 在 MNIST 数据集上训练一个 3 层多层感知机 (MLP)。网络包含 784 个输入节点，两个隐藏层分别有 256 和 128 个神经元，以及 10 个输出类别。

import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test

### 步骤 2：初始化参数

不使用类。仅定义一个返回 PyTree（树状结构）的函数：

def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params

手动实现 He 初始化 (He-initialization)。通过一个随机种子拆分出三个伪随机数生成器 (PRNG) 密钥。每个权重都是嵌套字典中的不可变数组。

### 步骤 3：前向传播

def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))

采用纯函数设计。输入参数，输出预测结果。没有 `self`，也不维护任何状态。`loss_fn` 从零开始计算交叉熵 (Cross-Entropy) 损失——依次执行 softmax、取对数、计算负均值。

### 步骤 4：JIT 编译的训练步骤

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)

`jax.value_and_grad` 在一次计算中同时返回损失值和梯度。`@jax.jit` 装饰器会将这两个函数编译为 XLA 代码。首次调用后，后续的每个训练步骤都将直接运行编译后的代码，无需经过 Python 解释器。

### 步骤 5：训练循环

optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

共训练 10 个轮次 (Epoch)。测试集准确率约为 97%。第一个轮次较慢（因为需要进行即时编译 (JIT)），第 2 到第 10 个轮次则非常快。

请注意代码中省略了什么：没有 `.zero_grad()`，没有 `.backward()`，也没有 `.step()`。整个参数更新过程仅通过一次组合函数调用完成。梯度的计算、Adam 优化器的变换以及参数的更新——全部都在 `train_step` 内部完成。

## 使用指南

### Flax：Google 官方标准库

Flax 是最常用的 JAX 神经网络库。它重新引入了 `nn.Module`，但采用了显式状态管理（explicit state management）：

import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)

其结构与 PyTorch 类似，但 `params`（参数）与模型对象是分离的。`model.init()` 用于创建参数，`model.apply(params, x)` 执行前向传播（forward pass）。模型对象本身不持有任何状态。

### Equinox：更符合 Python 风格的替代方案

Equinox（由 Patrick Kidger 开发）将模型表示为树形数据结构（PyTree）：

import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)

模型本身就是一个 PyTree。无需调用 `.apply()`。参数仅仅是模型的叶子节点（leaves）。这种设计更贴近 JAX 的核心思想。

### Optax：可组合的优化器

Optax 将梯度变换（gradient transformation）与参数更新解耦：

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)

梯度裁剪（gradient clipping）、学习率预热（learning rate warmup）和权重衰减（weight decay）等功能都被组合成一个变换链。每个变换接收梯度、进行修改，然后传递给下一个变换。不再需要庞大臃肿的单一优化器类。

## 交付使用

**安装：**

pip install jax jaxlib optax flax

如需 GPU 支持：

pip install jax[cuda12]

如需 TPU（Google Cloud）支持：

pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

**性能注意事项：**

- 首次即时编译（JIT）调用较慢（因为需要编译）。在进行基准测试（benchmarking）前请先进行预热。
- 避免在 JIT 编译的代码中使用 Python 循环遍历 JAX 数组。请改用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 可以在 JIT 内部正常工作，而普通的 `print()` 则不行。
- 使用 `jax.profiler` 或 TensorBoard 进行性能分析（profiling）。XLA 编译可能会掩盖性能瓶颈。
- JAX 默认会预分配 75% 的 GPU 显存。可通过设置环境变量 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 来禁用此行为。

**检查点（Checkpointing）：**

import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')

**本课时产出：**
- `outputs/prompt-jax-optimizer.md` -- 用于选择合适 JAX 优化器配置的提示词（prompt）
- `outputs/skill-jax-patterns.md` -- 涵盖 JAX 函数式模式（functional patterns）的技能文档

## 练习

1. 为多层感知机 (MLP) 添加随机失活 (Dropout) 层。在 JAX 中，Dropout 需要一个伪随机数生成器 (PRNG) 密钥——请将该密钥贯穿前向传播 (forward pass) 过程，并在每个 Dropout 层处对其进行分割。对比使用与不使用 Dropout 时的测试准确率。

2. 使用 `jax.vmap` 计算包含 32 张 MNIST 图像批次的逐样本梯度 (per-example gradients)。计算每个样本的梯度范数 (gradient norm)。哪些样本的梯度最大？原因是什么？

3. 将手动编写的前向传播函数替换为通用的 `mlp_forward(params, x)`，使其能够适配任意数量的层。使用 `jax.tree.leaves` 自动确定网络深度 (depth)。

4. 对使用与不使用 `@jax.jit` 的训练步骤进行基准测试 (benchmark)。分别记录 100 步的耗时。在你的硬件上加速比 (speedup) 有多大？首次调用时的编译开销 (compilation overhead) 是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现梯度裁剪 (gradient clipping)。分别在使用和不使用裁剪的情况下进行训练。绘制训练过程中梯度范数的变化曲线，以观察其效果。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| XLA | “让 JAX 变快的东西” | 加速线性代数 (Accelerated Linear Algebra) —— 一种编译器，用于融合算子并从计算图生成针对 GPU/TPU 优化的内核 |
| JIT | “即时编译” | JAX 在首次调用时追踪函数，将其编译为 XLA 代码，随后在后续调用中直接运行编译后的版本 |
| 纯函数 (Pure function) | “无副作用” | 输出仅取决于输入的函数——不依赖全局状态、不产生突变、不使用显式密钥之外的随机性 |
| vmap | “自动批处理” | 将处理单个样本的函数转换为处理批次的函数，无需重写代码 |
| pmap | “自动并行化” | 在多个设备上复制函数并拆分输入批次 |
| Pytree | “嵌套的数组字典” | JAX 能够遍历和转换的任意嵌套结构（包含列表、元组、字典和数组） |
| 追踪 (Tracing) | “记录计算过程” | JAX 使用抽象值执行函数以构建计算图，而不计算实际结果 |
| 函数式自动微分 (Functional autodiff) | “函数的梯度” | 通过变换函数来计算导数，而非将梯度存储附加到张量上 |
| Optax | “JAX 的优化器库” | 一个可组合的梯度变换库——包含 Adam、SGD、梯度裁剪、学习率调度等，支持链式组合 |
| Flax | “JAX 的 nn.Module” | Google 为 JAX 开发的神经网络库，在保持状态显式化的同时提供了网络层抽象 |

## 扩展阅读

- JAX 文档：https://jax.readthedocs.io/ -- 官方文档，包含关于 grad、jit 和 vmap 的优质教程
- 《JAX：Python+NumPy 程序的可组合变换 (composable transformations)》（Bradbury 等，2018）-- 阐述其设计哲学的原始论文
- Flax 文档：https://flax.readthedocs.io/ -- Google 为 JAX 开发的神经网络库
- Patrick Kidger，《Equinox：通过可调用 PyTree 与过滤变换 (filtered transformations) 在 JAX 中构建神经网络》（2021）-- Flax 的 Pythonic（符合 Python 风格）替代方案
- DeepMind，《Optax：可组合的梯度变换 (gradient transformation) 与优化》-- 标准的优化器库
- 《你不知道的 JAX》（Colin Raffel，2020）-- 由 T5 作者之一撰写的实用指南，涵盖 JAX 的常见陷阱 (gotchas) 与惯用模式 (patterns)