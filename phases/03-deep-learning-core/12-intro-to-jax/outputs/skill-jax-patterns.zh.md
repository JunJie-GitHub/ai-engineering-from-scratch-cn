---
name: JAX 函数式编程模式
description: JAX 中的函数式编程模式 -- 何时及如何使用 grad、jit、vmap 和 pmap
version: 1.0.0
phase: 3
lesson: 12
tags: [JAX, 函数式编程, 自动微分, 编译, 向量化]
---

# JAX 函数式编程模式

JAX 的核心是对纯函数（pure functions）进行变换。以下所有模式都遵循一个原则：编写一个接收输入并返回输出、且无副作用（side effects）的函数，然后对其应用变换。

## 四大变换

### grad -- 对函数求导

grads = jax.grad(loss_fn)(params, x, y)
loss, grads = jax.value_and_grad(loss_fn)(params, x, y)

适用场景：优化过程中需要计算梯度（gradients）。
限制条件：函数必须返回标量（scalar）。若输出为非标量，请使用 `jax.jacobian`。

### jit -- 编译函数

fast_fn = jax.jit(f)

适用场景：函数将被多次调用，且输入形状（shape）相同。
限制条件：不能包含依赖于追踪值（traced values）的 Python 控制流。条件判断请使用 `jax.lax.cond`，循环请使用 `jax.lax.scan`。

### vmap -- 向量化函数

batch_fn = jax.vmap(f, in_axes=(None, 0))

适用场景：你为单个样本编写了函数，现在需要使其支持批量（batch）处理。
`in_axes` 指定对哪个参数轴进行批处理。`None` 表示不进行批处理（即广播，broadcast）。

### pmap -- 跨设备并行化

parallel_fn = jax.pmap(f, axis_name='devices')

适用场景：拥有多个 GPU/TPU 且希望实现数据并行（data parallelism）。
在函数内部，`jax.lax.pmean(x, 'devices')` 用于计算跨设备的平均值。

## 组合规则

变换可以组合使用，且顺序至关重要：

per_example_grads = jax.jit(jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0)))

从右向左阅读：先对 `loss_fn` 求导，再沿样本维度向量化，最后编译结果。

有效组合：
- `jit(grad(f))` -- 编译后的梯度计算
- `jit(vmap(f))` -- 编译后的批量计算
- `vmap(grad(f))` -- 逐样本梯度
- `pmap(jit(f))` -- 并行编译计算
- `grad(jit(f))` -- 编译函数的梯度（效果等同于 `jit(grad(f))`）

## 参数管理模式

JAX 中的参数以树状结构（pytrees，即嵌套的数组字典）形式组织：

params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 10)),  'b': jnp.zeros(10)},
}

一次性更新所有参数：
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)

统计参数量：
n_params = sum(p.size for p in jax.tree.leaves(params))

## PRNG 密钥管理

JAX 要求显式管理伪随机数生成器密钥（PRNG keys）：

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape)

若需执行多次随机操作，可一次性拆分：
keys = jax.random.split(key, n)

切勿重复使用同一密钥。使用前务必先进行拆分。

## 常见错误

1. **在 `jit` 中修改数组**：JAX 数组是不可变的（immutable）。请使用 `x.at[i].set(v)` 替代 `x[i] = v`。

2. **在 `jit` 中使用 Python `print`**：`print` 仅在追踪（tracing）阶段执行，而非实际运行时。请使用 `jax.debug.print("{}", x)`。

3. **在 `jit` 中对追踪值使用 Python `if/for`**：请改用 `jax.lax.cond`、`jax.lax.switch`、`jax.lax.scan` 或 `jax.lax.fori_loop`。

4. **忘记调用 `.block_until_ready()`**：JAX 采用异步分发（async dispatch）。进行性能基准测试（benchmarking）时，请调用 `.block_until_ready()` 以确保等待实际执行完成。

5. **重复使用 PRNG 密钥**：使用相同密钥的两次操作会生成完全相同的“随机”值。务必始终先拆分。

6. **在 `jit` 函数中使用全局状态**：全局变量在追踪阶段即被捕获，追踪后的修改将不可见。请将所有数据作为参数传入。

## 决策检查清单

1. 该函数是否会被多次调用？添加 `@jax.jit`。
2. 是否需要计算梯度（gradients）？使用 `jax.grad` 或 `jax.value_and_grad` 进行包装。
3. 函数是否仅处理单个样本，但你实际拥有批量数据？使用 `jax.vmap` 进行包装。
4. 是否使用多个设备？使用 `jax.pmap` 进行包装。
5. 是否涉及随机性？需显式传递伪随机数生成器（PRNG）密钥。
6. 是否包含基于数组值的 Python 控制流（control flow）？请替换为 `jax.lax` 原语（primitives）。

## 何时使用 JAX

在以下情况使用 JAX：
- 需要逐样本梯度（per-example gradients）（适用于差分隐私（differential privacy）、费雪信息（Fisher information）等场景）
- 在张量处理单元（TPU）上进行训练（JAX 是其原生框架）
- 需要高阶导数（higher-order derivatives）（如海森矩阵（Hessians）、雅可比矩阵（Jacobians））
- 希望将整个训练步骤编译为单个内核（kernel）
- 你的团队隶属于 Google DeepMind 或 Anthropic

在以下情况使用 PyTorch：
- 希望使用最庞大的生态系统（ecosystem）（如 HuggingFace、torchvision、Lightning）
- 优先考虑调试便捷性而非极致性能
- 使用 TorchServe/Triton 部署至 NVIDIA GPU
- 正在招聘（PyTorch 开发者数量更多）
- 希望快速迭代新架构