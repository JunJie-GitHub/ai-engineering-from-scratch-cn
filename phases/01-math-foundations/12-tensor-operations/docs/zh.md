# 张量操作 (Tensor Operations)

> 张量 (Tensor) 是数据与深度学习之间的通用语言。每一张图像、每一句话、每一个梯度 (gradient) 都经由它们流转。

**类型：** 构建
**语言：** Python
**前置要求：** 第一阶段，课程 01（线性代数直觉）、02（向量、矩阵与运算）
**时长：** 约 90 分钟

## 学习目标

- 从零开始实现一个张量类，包含形状 (shape)、步长 (strides)、重塑 (reshape)、转置 (transpose) 和逐元素运算 (element-wise operations)
- 应用广播规则 (broadcasting rules)，在不复制数据的情况下对不同形状的张量进行运算
- 编写爱因斯坦求和约定 (einsum) 表达式，用于点积 (dot product)、矩阵乘法 (matrix multiplication)、外积 (outer product) 及批处理运算 (batched operation)
- 追踪多头注意力机制 (multi-head attention) 每一步中确切的张量形状变化

## 问题背景

你构建了一个 Transformer 模型。前向传播 (forward pass) 的代码看起来清晰无误。运行后却报错：`RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x768)`。你盯着这些形状参数，尝试进行一次转置。结果又提示 `Expected 4D input (got 3D input)`。你添加了一个 `unsqueeze` 操作，紧接着其他部分又崩溃了。

形状错误是深度学习代码中最常见的 Bug。从概念上讲它们并不难理解——每个运算都有其形状契约 (shape contract)——但它们会迅速累积。一个 Transformer 模型通常串联着数十次重塑、转置和广播 (broadcast) 操作。只要一个轴 (axis) 选错，错误就会级联放大。更糟糕的是，某些形状错误根本不会抛出异常。它们会沿着错误的维度进行广播，或在错误的轴上求和，从而悄无声息地输出毫无意义的结果。

矩阵只能处理两组事物之间的成对关系。而真实数据往往无法塞进二维空间。例如，一批 32 张 224x224 的 RGB 图像就是一个 4D 张量：`(32, 3, 224, 224)`。带有 12 个头的自注意力机制 (self-attention) 同样是 4D 的：`(batch, heads, seq_len, head_dim)`。你需要一种能够泛化到任意维度的数据结构，并且其运算能在所有维度上干净利落地组合。这种结构就是张量。熟练掌握其运算后，形状错误将变得极易调试。

## 核心概念

### 张量（Tensor）是什么

张量（Tensor）是一个具有统一数据类型的多维数值数组。维度的数量称为**秩（Rank）**（或**阶（Order）**）。每个维度称为一个**轴（Axis）**。**形状（Shape）**是一个元组，列出了沿每个轴的大小。

graph LR
    S["Scalar<br/>rank 0<br/>shape: ()"] --> V["Vector<br/>rank 1<br/>shape: (3,)"]
    V --> M["Matrix<br/>rank 2<br/>shape: (2,3)"]
    M --> T3["3D Tensor<br/>rank 3<br/>shape: (2,2,2)"]
    T3 --> T4["4D Tensor<br/>rank 4<br/>shape: (B,C,H,W)"]

元素总数等于所有维度大小的乘积。例如，形状 `(2, 3, 4)` 包含 `2 * 3 * 4 = 24` 个元素。

### 深度学习中的张量形状

按照惯例，不同的数据类型会映射到特定的张量形状。

graph TD
    subgraph Vision
        V1["(B, C, H, W)<br/>32, 3, 224, 224"]
    end
    subgraph NLP
        N1["(B, T, D)<br/>16, 128, 768"]
    end
    subgraph Attention
        A1["(B, H, T, D)<br/>16, 12, 128, 64"]
    end
    subgraph Weights
        W1["Linear: (out, in)<br/>Conv2D: (out_c, in_c, kH, kW)<br/>Embedding: (vocab, dim)"]
    end

PyTorch 采用 NCHW（通道优先）格式，而 TensorFlow 默认使用 NHWC（通道最后）格式。布局不匹配会导致性能隐性下降或引发错误。

### 内存布局的工作原理

内存中的二维数组实际上是一维的字节序列。**步长（Strides）**指示了沿每个轴移动一步时需要跳过的元素数量。

graph LR
    subgraph "Row-major (C order)"
        R["a b c d e f<br/>strides: (3, 1)"]
    end
    subgraph "Column-major (F order)"
        C["a d b e c f<br/>strides: (1, 2)"]
    end

转置（Transpose）操作并不会移动数据。它只是交换了步长，从而使张量变为**非连续（Non-contiguous）**状态——即同一行的元素在内存中不再相邻。

### 广播机制（Broadcasting）规则

广播机制允许你在不复制数据的情况下对不同形状的张量进行运算。形状从右侧开始对齐。当两个维度大小相等或其中一个为 1 时，它们即为兼容。维度较少的张量会在左侧用 1 进行填充。

Tensor A:     (8, 1, 6, 1)
Tensor B:        (7, 1, 5)
Padded B:     (1, 7, 1, 5)
Result:       (8, 7, 6, 5)

### Einsum：通用的张量运算

爱因斯坦求和约定（Einstein summation）使用字母为每个轴进行标记。出现在输入中但未出现在输出中的轴会被求和。同时出现在输入和输出中的轴则会被保留。

graph LR
    subgraph "matmul: ik,kj -> ij"
        A["A(I,K)"] --> |"sum over k"| C["C(I,J)"]
        B["B(K,J)"] --> |"sum over k"| C
    end

常用模式：`i,i->`（点积）、`i,j->ij`（外积）、`ii->`（迹）、`ij->ji`（转置）、`bij,bjk->bik`（批量矩阵乘法）、`bhtd,bhsd->bhts`（注意力分数）。

## 动手实践

代码位于 `code/tensors.py` 中。每个步骤均引用该文件中的具体实现。

### 步骤 1：张量（Tensor）存储与步幅（Stride）

张量存储一个扁平的数字列表以及形状（Shape）元数据。步幅用于指导索引逻辑，将多维索引映射到一维扁平位置。

class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_nested(data)
        elif isinstance(data, np.ndarray):
            self._data = data.flatten().tolist()
            self._shape = tuple(data.shape)
        else:
            self._data = [data]
            self._shape = ()

        if shape is not None:
            total = reduce(lambda a, b: a * b, shape, 1)
            if total != len(self._data):
                raise ValueError(
                    f"Cannot reshape {len(self._data)} elements into shape {shape}"
                )
            self._shape = tuple(shape)

        self._strides = self._compute_strides(self._shape)

    @staticmethod
    def _compute_strides(shape):
        if len(shape) == 0:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

对于形状 `(3, 4)`，步幅为 `(4, 1)` —— 前进一行需跳过 4 个元素，前进一列需跳过 1 个元素。

### 步骤 2：重塑（Reshape）、压缩（Squeeze）与扩展（Unsqueeze）

重塑在不改变元素顺序的前提下修改张量形状。元素总数必须保持不变。可将某一维度设为 `-1` 以自动推断其大小。

t = Tensor(list(range(12)), shape=(2, 6))
r = t.reshape((3, 4))
r = t.reshape((-1, 3))

压缩用于移除大小为 1 的轴，而扩展则用于插入一个大小为 1 的轴。扩展操作对广播（Broadcasting）机制至关重要 —— 例如，将偏置向量 `(D,)` 加到批次数据 `(B, T, D)` 上时，需先将其扩展为 `(1, 1, D)`。

t = Tensor(list(range(6)), shape=(1, 3, 1, 2))
s = t.squeeze()
v = Tensor([1, 2, 3])
u = v.unsqueeze(0)

### 步骤 3：转置（Transpose）与重排（Permute）

转置用于交换两个轴，而重排则用于重新排列所有轴。这是实现 NCHW 与 NHWC 格式相互转换的方法。

mat = Tensor(list(range(6)), shape=(2, 3))
tr = mat.transpose(0, 1)

t4d = Tensor(list(range(24)), shape=(1, 2, 3, 4))
perm = t4d.permute((0, 2, 3, 1))

执行转置或重排后，张量在内存中将变为非连续（Non-contiguous）状态。在 PyTorch 中，对非连续张量调用 `view` 会失败 —— 此时应使用 `reshape`，或先调用 `.contiguous()` 方法。

### 步骤 4：逐元素操作（Element-wise Operations）与归约（Reductions）

逐元素操作（如加、乘、减）独立作用于每个元素并保持形状不变。归约操作（如求和、求均值、求最大值）则会折叠一个或多个轴。

a = Tensor([[1, 2], [3, 4]])
b = Tensor([[10, 20], [30, 40]])
c = a + b
d = a * 2
s = a.sum(axis=0)

卷积神经网络（CNN）中的全局平均池化：`(B, C, H, W).mean(axis=[2, 3])` 输出形状为 `(B, C)`。自然语言处理（NLP）中的序列平均池化：`(B, T, D).mean(axis=1)` 输出形状为 `(B, D)`。

### 步骤 5：使用 NumPy 进行广播

`tensors.py` 中的 `demo_broadcasting_numpy()` 函数展示了核心模式。

activations = np.random.randn(4, 3)
bias = np.array([0.1, 0.2, 0.3])
result = activations + bias

images = np.random.randn(2, 3, 4, 4)
scale = np.array([0.5, 1.0, 1.5]).reshape(1, 3, 1, 1)
result = images * scale

a = np.array([1, 2, 3]).reshape(-1, 1)
b = np.array([10, 20, 30, 40]).reshape(1, -1)
outer = a * b

通过广播计算成对距离：将 `(M, 2)` 重塑为 `(M, 1, 2)`，将 `(N, 2)` 重塑为 `(1, N, 2)`，相减、平方、沿最后一个轴求和，最后开平方。结果形状为 `(M, N)`。

### 步骤 6：爱因斯坦求和约定（Einsum）操作

`demo_einsum()` 和 `demo_einsum_gallery()` 函数逐步演示了所有常见模式。

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
dot = np.einsum("i,i->", a, b)

A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
B = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)
matmul = np.einsum("ik,kj->ij", A, B)

batch_A = np.random.randn(4, 3, 5)
batch_B = np.random.randn(4, 5, 2)
batch_mm = np.einsum("bij,bjk->bik", batch_A, batch_B)

张量缩并（Contraction）的计算成本等于所有索引维度大小（包含保留与求和的维度）的乘积。以 `bij,bjk->bik` 为例，当 B=32、I=128、J=64、K=128 时：`32 * 128 * 64 * 128 = 33,554,432` 次乘加运算（Multiply-Adds）。

### 步骤 7：通过 Einsum 实现注意力机制（Attention Mechanism）

`demo_attention_einsum()` 函数完整实现了多头注意力（Multi-Head Attention）机制。

B, H, T, D = 2, 4, 8, 16
E = H * D

X = np.random.randn(B, T, E)
W_q = np.random.randn(E, E) * 0.02

Q = np.einsum("bte,ek->btk", X, W_q)
Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3)

scores = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(D)
weights = softmax(scores, axis=-1)
attn_output = np.einsum("bhts,bhsd->bhtd", weights, V)

concat = attn_output.transpose(0, 2, 1, 3).reshape(B, T, E)
output = np.einsum("bte,ek->btk", concat, W_o)

每一步均为张量操作：投影（通过 einsum 实现矩阵乘法）、头部分割（重塑 + 转置）、注意力分数计算（通过 einsum 实现批次矩阵乘法）、加权求和（通过 einsum 实现批次矩阵乘法）、头部合并（转置 + 重塑）以及输出投影（通过 einsum 实现矩阵乘法）。

## 实践应用

### 手写实现 (Scratch) 与 NumPy 对比

| 操作 | 手写实现 (Tensor 类) | NumPy |
|---|---|---|
| 创建 | `Tensor([[1,2],[3,4]])` | `np.array([[1,2],[3,4]])` |
| 重塑形状 (Reshape) | `t.reshape((3,4))` | `a.reshape(3,4)` |
| 转置 (Transpose) | `t.transpose(0,1)` | `a.T` 或 `a.transpose(0,1)` |
| 压缩维度 (Squeeze) | `t.squeeze(0)` | `np.squeeze(a, 0)` |
| 求和 (Sum) | `t.sum(axis=0)` | `a.sum(axis=0)` |
| 爱因斯坦求和 (Einsum) | 不适用 (N/A) | `np.einsum("ij,jk->ik", a, b)` |

### 手写实现 (Scratch) 与 PyTorch 对比

import torch

t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
t.shape
t.stride()
t.is_contiguous()

t.reshape(3, 2)
t.unsqueeze(0)
t.transpose(0, 1)
t.transpose(0, 1).contiguous()

torch.einsum("ik,kj->ij", A, B)

PyTorch 增加了自动微分 (autograd)、GPU 支持以及优化的基础线性代数子程序内核 (BLAS kernels)。两者的形状语义 (shape semantics) 完全一致。如果你理解了手写实现版本，PyTorch 的形状报错 (shape errors) 将变得清晰易懂。

### 将每个神经网络层视为张量操作

| 操作 | 张量形式 | 爱因斯坦求和 (Einsum) |
|---|---|---|
| 线性层 (Linear layer) | `Y = X @ W.T + b` | `"bd,od->bo"` + 偏置 (bias) |
| 注意力查询-键-值 (Attention QKV) | `Q = X @ W_q` | `"btd,dh->bth"` |
| 注意力分数 (Attention scores) | `Q @ K.T / sqrt(d)` | `"bhtd,bhsd->bhts"` |
| 注意力输出 (Attention output) | `softmax(scores) @ V` | `"bhts,bhsd->bhtd"` |
| 批归一化 (Batch norm) | `(X - mu) / sigma * gamma` | 逐元素操作 (element-wise) + 广播 (broadcast) |
| Softmax | `exp(x) / sum(exp(x))` | 逐元素操作 (element-wise) + 归约 (reduction) |

## 交付成果

本节课程将生成两个可复用的提示词 (prompts)：

1. **`outputs/prompt-tensor-shapes.md`** -- 用于调试张量形状不匹配 (tensor shape mismatches) 的系统化提示词。包含针对每种常见操作（矩阵乘法 (matmul)、广播 (broadcast)、拼接 (cat)、线性层 (Linear)、二维卷积 (Conv2d)、批归一化 (BatchNorm)、Softmax）的决策表，以及修复方案速查表。

2. **`outputs/prompt-tensor-debugger.md`** -- 当形状错误阻碍开发时，可粘贴至任意 AI 助手的分步调试提示词。只需输入错误信息和你的张量形状，即可获取精确的修复方案。

## 练习

1. **简单 -- 重塑（Reshape）往返**。取一个形状为 `(2, 3, 4)` 的张量（Tensor）。将其重塑为 `(6, 4)`，再重塑为 `(24,)`，最后恢复为 `(2, 3, 4)`。通过打印展平（Flatten）后的数据，验证每一步的元素顺序是否保持不变。

2. **中等 -- 实现广播（Broadcasting）机制**。为 `Tensor` 类扩展一个 `broadcast_to(shape)` 方法，用于将大小为 1 的维度扩展以匹配目标形状。随后修改 `_elementwise_op`，使其在执行运算前自动进行广播。使用形状 `(3, 1)` 和 `(1, 4)` 进行测试，验证其能否正确生成 `(3, 4)` 的结果。

3. **困难 -- 从零构建爱因斯坦求和（Einsum）**。实现一个基础的 `einsum(subscripts, *tensors)` 函数，至少需支持：点积（`i,i->`）、矩阵乘法（`ij,jk->ik`）、外积（`i,j->ij`）以及转置（`ij->ji`）。解析下标字符串，识别收缩（Contraction）索引，并遍历所有索引组合。将你的计算结果与 `np.einsum` 进行对比。

4. **困难 -- 注意力（Attention）形状追踪器**。编写一个函数，接收 `batch_size`、`seq_len`、`embed_dim` 和 `num_heads` 作为输入，并打印多头注意力（Multi-head Attention）每一步的确切形状：输入、Q/K/V 投影、头部分割、注意力分数、Softmax 权重、加权求和、头部合并、输出投影。将结果与 `demo_attention_einsum()` 的输出进行验证对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| 张量（Tensor） | “就是维度更多的矩阵” | 具有统一数据类型、明确形状、步长及运算规则的多维数组 |
| 秩（Rank） | “维度的数量” | 轴的数量。矩阵的秩为 2，而非线性代数中的矩阵秩 |
| 形状（Shape） | “张量的大小” | 列出每个轴大小的元组。`(2, 3)` 表示 2 行 3 列 |
| 步长（Stride） | “内存的排布方式” | 沿每个轴前进一个位置所需跳过的元素数量 |
| 广播（Broadcasting） | “形状不同时它会自动适配” | 一套严格的规则：从右侧对齐，对应维度必须相等或其中之一为 1 |
| 连续（Contiguous） | “张量是标准的” | 元素在内存中按逻辑布局顺序连续存储，无间隙或重排 |
| 爱因斯坦求和（Einsum） | “写矩阵乘法的炫酷方式” | 一种通用记号，可用一行代码表达任意张量收缩、外积、迹或转置操作 |
| 视图（View） | “和重塑一样” | 共享同一内存缓冲区但具有不同形状/步长元数据的张量。在非连续数据上会失败 |
| 收缩（Contraction） | “对某个索引求和” | 一种通用操作：将张量间共享的索引相乘并求和，生成秩更低的结果 |
| NCHW / NHWC | “PyTorch 与 TensorFlow 的格式区别” | 图像张量的内存布局惯例。NCHW 将通道维度置于空间维度之前，NHWC 则置于其后 |

## 扩展阅读

- [NumPy 广播机制 (Broadcasting)](https://numpy.org/doc/stable/user/basics.broadcasting.html) -- 配有可视化示例的标准规则
- [PyTorch 张量视图 (Tensor Views)](https://pytorch.org/docs/stable/tensor_view.html) -- 视图的生效条件与数据复制时机
- [einops](https://github.com/arogozhnikov/einops) -- 一款让张量重塑 (Tensor Reshaping) 操作更易读且更安全的库
- [图解 Transformer](https://jalammar.github.io/illustrated-transformer/) -- 可视化流经注意力机制 (Attention) 的张量形状变化
- [NumPy 中的爱因斯坦求和 (Einstein Summation)](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) -- 完整的 einsum 文档及示例