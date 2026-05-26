# AI 中的复数 (Complex Numbers)

> -1 的平方根并非虚幻。它是理解旋转、频率以及信号处理半壁江山的关键。

**Type:** 学习
**Language:** Python
**Prerequisites:** 第一阶段，第 01-04 课（线性代数、微积分）
**Time:** 约 60 分钟

## 学习目标

- 在直角坐标形式 (rectangular form) 和极坐标形式 (polar form) 下执行复数运算（加、乘、除、共轭）
- 应用欧拉公式 (Euler's formula) 在复指数函数 (complex exponentials) 与三角函数 (trigonometric functions) 之间进行转换
- 使用复数单位根 (complex roots of unity) 实现离散傅里叶变换 (Discrete Fourier Transform)
- 解释复数旋转如何作为 Transformer 架构中旋转位置编码 (Rotary Position Embedding, RoPE) 和正弦位置编码 (sinusoidal positional encodings) 的底层原理

## 问题所在

当你打开一篇关于傅里叶变换 (Fourier Transform) 的论文时，会发现满篇都是 `i`。当你查看 Transformer 的位置编码时，会看到不同频率下的 `sin` 和 `cos`——这正是复指数函数的实部与虚部。当你阅读量子计算相关资料时，会发现所有内容都建立在复向量空间 (complex vector space) 中。

复数看起来似乎很抽象。一个建立在 -1 的平方根之上的数系，感觉就像是一种数学戏法。但它并非戏法，而是描述旋转与振荡的自然语言。每当涉及旋转、振动或周期性波动时，复数都是最合适的工具。

如果不理解复数，你就无法掌握离散傅里叶变换。你无法理解快速傅里叶变换 (Fast Fourier Transform, FFT)。你无法弄清旋转位置编码在现代大语言模型中的工作原理。你也无法明白原始 Transformer 论文中的正弦位置编码为何要采用特定的频率。

本课程将从零开始构建复数运算体系，将其与几何直观相联系，并为你清晰展示复数在机器学习中的具体应用场景。

## 核心概念

### 什么是复数 (Complex Number)？

复数 (Complex Number) 由两部分组成：实部 (Real Part) 和虚部 (Imaginary Part)。

z = a + bi

where:
  a is the real part
  b is the imaginary part
  i is the imaginary unit, defined by i^2 = -1

仅此而已。你将数轴扩展为一个平面。实数位于一条轴上，虚数位于另一条轴上。每一个复数都是该平面上的一个点。

### 复数运算 (Complex Arithmetic)

**加法 (Addition)**。将实部相加，虚部相加。

(a + bi) + (c + di) = (a + c) + (b + d)i

Example: (3 + 2i) + (1 + 4i) = 4 + 6i

**乘法 (Multiplication)**。运用分配律，并记住 i^2 = -1。

(a + bi)(c + di) = ac + adi + bci + bdi^2
                 = ac + adi + bci - bd
                 = (ac - bd) + (ad + bc)i

Example: (3 + 2i)(1 + 4i) = 3 + 12i + 2i + 8i^2
                            = 3 + 14i - 8
                            = -5 + 14i

**共轭 (Conjugate)**。将虚部的符号取反。

conjugate of (a + bi) = a - bi

一个复数与其共轭复数的乘积始终为实数：

(a + bi)(a - bi) = a^2 + b^2

**除法 (Division)**。将分子和分母同时乘以分母的共轭复数。

(a + bi) / (c + di) = (a + bi)(c - di) / (c^2 + d^2)

这样可以消除分母中的虚部，从而得到一个标准的复数形式。

### 复平面 (Complex Plane)

复平面将每个复数映射为一个二维点。水平轴为实轴，垂直轴为虚轴。

z = 3 + 2i  corresponds to the point (3, 2)
z = -1 + 0i corresponds to the point (-1, 0) on the real axis
z = 0 + 4i  corresponds to the point (0, 4) on the imaginary axis

复数既可以表示一个点，也可以表示从原点出发的向量。这种双重解释正是复数在几何学中如此实用的原因。

### 极坐标形式 (Polar Form)

平面上的任意一点都可以通过其到原点的距离以及与正实轴的夹角来描述。

z = r * (cos(theta) + i*sin(theta))

where:
  r = |z| = sqrt(a^2 + b^2)     (magnitude, or modulus)
  theta = atan2(b, a)             (phase, or argument)

直角坐标形式 (a + bi) 适用于加法运算，而极坐标形式 (r, theta) 则适用于乘法运算。

**极坐标形式下的乘法**。将模长相乘，将角度相加。

z1 = r1 * e^(i*theta1)
z2 = r2 * e^(i*theta2)

z1 * z2 = (r1 * r2) * e^(i*(theta1 + theta2))

这就是为什么复数非常适合表示旋转。乘以一个模长为 1 的复数，等价于执行一次纯旋转。

### 欧拉公式 (Euler's Formula)

连接复指数函数与三角函数的桥梁：

e^(i*theta) = cos(theta) + i*sin(theta)

这是本课程中最核心的公式。当 theta = pi 时：

e^(i*pi) = cos(pi) + i*sin(pi) = -1 + 0i = -1

Therefore: e^(i*pi) + 1 = 0

五个基本数学常数（e, i, pi, 1, 0）被统一在一个等式中。

### 为什么欧拉公式对机器学习 (Machine Learning, ML) 至关重要

欧拉公式表明，随着 theta 的变化，`e^(i*theta)` 会沿着单位圆运动。当 theta = 0 时，位于 (1, 0)；当 theta = pi/2 时，位于 (0, 1)；当 theta = pi 时，位于 (-1, 0)；当 theta = 3*pi/2 时，位于 (0, -1)。完整旋转一周对应 theta = 2*pi。

这意味着复指数函数本质上就是旋转操作。而旋转在信号处理和机器学习中无处不在。

### 与二维旋转的关联

将复数 (x + yi) 乘以 e^(i*theta)，相当于将点 (x, y) 绕原点旋转 theta 角度。

Rotation via complex multiplication:
  (x + yi) * (cos(theta) + i*sin(theta))
  = (x*cos(theta) - y*sin(theta)) + (x*sin(theta) + y*cos(theta))i

Rotation via matrix multiplication:
  [cos(theta)  -sin(theta)] [x]   [x*cos(theta) - y*sin(theta)]
  [sin(theta)   cos(theta)] [y] = [x*sin(theta) + y*cos(theta)]

两者产生的结果完全一致。复数乘法本质上就是二维旋转。旋转矩阵仅仅是用矩阵符号表示的复数乘法。

graph TD
    subgraph "Complex Multiplication = 2D Rotation"
        A["z = x + yi<br/>Point (x, y)"] -->|"multiply by e^(i*theta)"| B["z' = z * e^(i*theta)<br/>Point rotated by theta"]
    end
    subgraph "Equivalent Matrix Form"
        C["vector [x, y]"] -->|"multiply by rotation matrix"| D["[x cos theta - y sin theta,<br/> x sin theta + y cos theta]"]
    end
    B -.->|"same result"| D

### 相量 (Phasor) 与旋转信号

复指数函数 e^(i*omega*t) 表示一个以角频率 omega 绕单位圆旋转的点。随着 t 的增加，该点描绘出圆形轨迹。

该旋转点的实部为 cos(omega*t)，虚部为 sin(omega*t)。正弦信号本质上就是旋转复数在实轴上的投影。

e^(i*omega*t) = cos(omega*t) + i*sin(omega*t)

Real part:      cos(omega*t)    -- a cosine wave
Imaginary part: sin(omega*t)    -- a sine wave

这就是相量表示法。与其追踪波动的正弦曲线，不如追踪一个平滑旋转的箭头。相位偏移转化为角度偏移，振幅变化转化为模长变化，信号叠加则转化为向量加法。

### 单位根 (Roots of Unity)

N 次单位根是均匀分布在单位圆上的 N 个点：

w_k = e^(2*pi*i*k/N)    for k = 0, 1, 2, ..., N-1

当 N = 4 时，单位根为：1, i, -1, -i（对应四个基本方位）。
当 N = 8 时，则包含四个基本方位点以及四个对角线方向的点。

单位根是离散傅里叶变换 (Discrete Fourier Transform, DFT) 的基础。DFT 将信号分解为在这 N 个等间隔频率上的分量。

### 与离散傅里叶变换 (DFT) 的关联

信号 x[0], x[1], ..., x[N-1] 的离散傅里叶变换定义为：

X[k] = sum_{n=0}^{N-1} x[n] * e^(-2*pi*i*k*n/N)

每个 X[k] 衡量的是信号与第 k 个单位根（即频率为 k 的复正弦波）的相关程度。DFT 将信号拆解为 N 个旋转相量，并给出每个相量的振幅和相位。

### 为什么 i 并非“虚”数

“虚数 (Imaginary)”一词纯属历史偶然，笛卡尔最初使用它时带有贬义。但 i 的“虚幻”程度，并不亚于负数刚被提出时人们对其的排斥。负数回答了“3 减去 5 等于多少？”的问题，而虚数单位 i 回答了“什么数的平方等于 -1？”的问题。

更实用的理解是：i 是一个 90 度旋转算子。将实数乘以 i 一次，相当于向虚轴方向旋转 90 度；再乘以 i 一次（即 i^2），则再旋转 90 度——此时方向指向负实轴。这就是 i^2 = -1 的原因。它并不神秘，只是由两次四分之一旋转组合而成的半圈旋转。

这就是为什么复数在工程领域无处不在。任何涉及旋转的事物——电磁波、量子态、信号振荡、位置编码——都可以用复数自然地描述。

### 复指数函数与三角函数的对比

在欧拉公式出现之前，工程师通常将信号表示为 A*cos(omega*t + phi)（振幅 A，频率 omega，相位 phi）。这种方法虽然可行，但运算极其繁琐。将两个不同相位的余弦函数相加，需要依赖复杂的三角恒等式。

使用复指数函数后，同样的信号可表示为 A*e^(i*(omega*t + phi))。信号相加只需将两个复数相加；相乘（调制）只需将模长相乘、角度相加。相位偏移转化为角度相加，频率偏移则转化为与相量的乘法。

整个信号处理领域之所以全面转向复指数表示法，正是因为其数学推导更为简洁。所谓的“真实信号”始终只是复数表示的实部。虚部则作为辅助记录被保留下来，使得所有代数运算都能自然顺畅地进行。

### 与 Transformer 架构的关联

**正弦位置编码 (Sinusoidal Positional Encodings)**（原始 Transformer 论文）：

PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

正弦和余弦对实际上是不同频率下复指数函数的实部与虚部。每种频率为位置编码提供了不同的“分辨率”。低频变化缓慢（用于粗粒度位置），高频变化迅速（用于细粒度位置）。它们共同为每个位置赋予了独一无二的频率指纹。

**旋转位置编码 (Rotary Position Embedding, RoPE)** 在此基础上更进一步。它显式地将查询 (Query) 和键 (Key) 向量与复数旋转矩阵相乘。两个词元 (Token) 之间的相对位置被转化为旋转角度。注意力机制 (Attention) 基于这些旋转后的向量进行计算，从而使模型能够通过复数乘法对相对位置保持敏感。

| 运算 | 代数形式 | 几何意义 |
|-----------|---------------|-------------------|
| 加法 | (a+c) + (b+d)i | 平面内的向量加法 |
| 乘法 | (ac-bd) + (ad+bc)i | 旋转与缩放 |
| 共轭 | a - bi | 关于实轴的镜像反射 |
| 模长 | sqrt(a^2 + b^2) | 到原点的距离 |
| 相位 | atan2(b, a) | 与正实轴的夹角 |
| 除法 | multiply by conjugate | 逆向旋转与重新缩放 |
| 幂运算 | r^n * e^(i*n*theta) | 旋转 n 次，缩放 r^n 倍 |

graph LR
    subgraph "Unit Circle"
        direction TB
        U1["e^(i*0) = 1"] -.-> U2["e^(i*pi/2) = i"]
        U2 -.-> U3["e^(i*pi) = -1"]
        U3 -.-> U4["e^(i*3pi/2) = -i"]
        U4 -.-> U1
    end
    subgraph "Applications"
        A1["Euler's formula:<br/>e^(i*theta) = cos + i*sin"]
        A2["DFT uses roots of unity:<br/>e^(2*pi*i*k/N)"]
        A3["RoPE uses rotation:<br/>q * e^(i*m*theta)"]
    end
    U1 --> A1
    U1 --> A2
    U1 --> A3


## 构建

### 步骤 1：复数类 (Complex Class)

构建一个复数类 (Complex number class)，支持算术运算、模长 (magnitude)、相位 (phase)，以及直角坐标形式 (rectangular form) 与极坐标形式 (polar form) 之间的转换。

import math

class Complex:
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)

    def __truediv__(self, other):
        denom = other.real ** 2 + other.imag ** 2
        r = (self.real * other.real + self.imag * other.imag) / denom
        i = (self.imag * other.real - self.real * other.imag) / denom
        return Complex(r, i)

    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self):
        return math.atan2(self.imag, self.real)

    def conjugate(self):
        return Complex(self.real, -self.imag)

### 步骤 2：极坐标转换与欧拉公式 (Euler's Formula)

def to_polar(z):
    return z.magnitude(), z.phase()

def from_polar(r, theta):
    return Complex(r * math.cos(theta), r * math.sin(theta))

def euler(theta):
    return Complex(math.cos(theta), math.sin(theta))

验证：`euler(theta).magnitude()` 的结果应始终为 1.0。`euler(0)` 应返回 (1, 0)。`euler(pi)` 应返回 (-1, 0)。

### 步骤 3：旋转 (Rotation)

将点 (x, y) 绕原点旋转角度 theta，只需进行一次复数乘法：

point = Complex(3, 4)
rotated = point * euler(math.pi / 4)

旋转后模长保持不变，仅角度发生变化。

### 步骤 4：基于复数运算的离散傅里叶变换 (DFT)

def dft(signal):
    N = len(signal)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            total = total + Complex(signal[n], 0) * euler(angle)
        result.append(total)
    return result

这是时间复杂度为 O(N^2) 的离散傅里叶变换 (Discrete Fourier Transform, DFT)。每个输出 X[k] 均为信号采样值与单位根 (roots of unity) 相乘后的累加和。

### 步骤 5：逆离散傅里叶变换 (Inverse DFT)

逆离散傅里叶变换 (Inverse Discrete Fourier Transform, IDFT) 用于从频谱中重建原始信号。与正向 DFT 相比，仅需做两处修改：将指数部分的符号取反，并将结果除以 N。

def idft(spectrum):
    N = len(spectrum)
    result = []
    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            total = total + spectrum[k] * euler(angle)
        result.append(Complex(total.real / N, total.imag / N))
    return result

这可实现信号的完美重建。依次应用 DFT 和 IDFT 后，即可以机器精度 (machine precision) 还原原始信号，过程中不会丢失任何信息。

### 步骤 6：单位根 (Roots of Unity)

def roots_of_unity(N):
    return [euler(2 * math.pi * k / N) for k in range(N)]

验证以下两个性质：
- 每个单位根的模长均严格等于 1。
- 所有 N 个单位根之和为零（基于对称性相互抵消）。

正是这些性质保证了 DFT 的可逆性。单位根构成了频域的一组正交基 (orthogonal basis)。

## 使用方法

Python 内置了对复数（complex number）的支持。字面量 `j` 表示虚数单位（imaginary unit）。

z = 3 + 2j
w = 1 + 4j

print(z + w)
print(z * w)
print(abs(z))

import cmath
print(cmath.phase(z))
print(cmath.exp(1j * cmath.pi))

对于数组，NumPy 原生支持复数处理：

import numpy as np

z = np.array([1+2j, 3+4j, 5+6j])
print(np.abs(z))
print(np.angle(z))
print(np.conj(z))
print(np.real(z))
print(np.imag(z))

signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 128))
spectrum = np.fft.fft(signal)
freqs = np.fft.fftfreq(128, d=1/128)

## 运行输出

运行 `code/complex_numbers.py` 以生成 `outputs/skill-complex-arithmetic.md`。

## 练习题

1. **手动复数运算。** 计算 (2 + 3i) * (4 - i) 并使用代码验证结果。接着计算 (5 + 2i) / (1 - 3i)。在复平面（complex plane）上绘制这两个结果，并验证乘法操作是否对第一个数进行了旋转和缩放。

2. **旋转序列。** 从点 (1, 0) 开始，连续乘以 e^(i*pi/6) 十二次。验证经过 12 次乘法后是否回到 (1, 0)。打印每一步的坐标，并确认它们描绘出一个正十二边形（regular 12-gon）。

3. **已知信号的离散傅里叶变换（DFT）。** 创建一个信号，该信号为 sin(2*pi*3*t) 与 0.5*sin(2*pi*7*t) 之和，并在 32 个点上进行采样。运行你的 DFT。验证幅度谱（magnitude spectrum）在频率 3 和 7 处出现峰值，且频率 7 处的峰值高度为频率 3 处峰值的一半。

4. **单位根可视化。** 计算 8 次单位根（8th roots of unity）。验证它们的和为零。验证将任意根乘以本原根（primitive root） e^(2*pi*i/8) 是否得到下一个根。

5. **旋转矩阵等价性验证。** 针对 10 个随机角度和 10 个随机点，验证复数乘法的结果是否与使用 2x2 旋转矩阵（rotation matrix）进行矩阵-向量乘法（matrix-vector multiplication）的结果一致。打印最大数值差异。

## 关键术语

| 术语 | 含义 |
|------|------|
| 复数 (Complex number) | 形如 a + bi 的数，其中 a 为实部，b 为虚部，且 i^2 = -1 |
| 虚数单位 (Imaginary unit) | 由 i^2 = -1 定义的数 i。并非哲学意义上的“虚幻”，而是一个旋转算子 |
| 复平面 (Complex plane) | 二维平面，其中 x 轴为实轴，y 轴为虚轴。也称为阿甘德平面 (Argand plane) |
| 模/模长 (Magnitude/Modulus) | 到原点的距离：sqrt(a^2 + b^2)。记作 \|z\| |
| 相位/辐角 (Phase/Argument) | 与正实轴的夹角：atan2(b, a)。记作 arg(z) |
| 共轭复数 (Conjugate) | 关于实轴的镜像：a + bi 的共轭为 a - bi |
| 极坐标形式 (Polar form) | 将 z 表示为 r * e^(i*theta) 而非 a + bi。可使乘法运算更简便 |
| 欧拉公式 (Euler's formula) | e^(i*theta) = cos(theta) + i*sin(theta)。将指数函数与三角函数联系起来 |
| 相量 (Phasor) | 表示正弦信号的旋转复数 e^(i*omega*t) |
| 单位根 (Roots of unity) | 当 k = 0 到 N-1 时的 N 个复数 e^(2*pi*i*k/N)。单位圆上 N 个等间距的点 |
| 离散傅里叶变换 (Discrete Fourier Transform, DFT) | 利用单位根将信号分解为复正弦分量 |
| 旋转位置编码 (Rotary Position Embedding, RoPE) | 在 Transformer 注意力机制中，利用复数乘法对相对位置进行编码 |

## 延伸阅读

- [欧拉公式的可视化入门](https://betterexplained.com/articles/intuitive-understanding-of-eulers-formula/) - 无需繁杂的数学符号即可建立几何直觉
- [Su 等人：RoFormer (2021)](https://arxiv.org/abs/2104.09864) - 引入利用复数旋转实现旋转位置编码的论文
- [Vaswani 等人：Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - 提出 Transformer 架构及正弦位置编码的原始论文
- [3Blue1Brown：结合群论入门讲解欧拉公式](https://www.youtube.com/watch?v=mvmuCPvRoWQ) - 直观解释为何 e^(i*pi) = -1
- [Needham：《Visual Complex Analysis》](https://global.oup.com/academic/product/visual-complex-analysis-9780198534464) - 关于复数最出色的可视化著作，充满几何洞察力
- [Strang：《Introduction to Linear Algebra》第 10 章](https://math.mit.edu/~gs/linearalgebra/) - 在线性代数与特征值背景下讲解复数