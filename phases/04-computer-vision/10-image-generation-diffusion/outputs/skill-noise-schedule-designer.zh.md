---
name: skill-noise-schedule-designer
description: 给定步数 T 和目标噪声破坏程度，生成线性、余弦或 S 型（sigmoid）的 beta 调度（beta schedule），并附带信噪比（SNR）曲线图
version: 1.0.0
phase: 4
lesson: 10
tags: [计算机视觉, 扩散模型, 噪声调度, 训练]
---

# 噪声调度设计器（Noise Schedule Designer）

beta 调度（beta schedule）控制着在每个扩散步骤（diffusion step）中保留多少信号。糟糕的调度会限制训练效率，并在后续的每个决策环节中制约生成样本的质量。

## 适用场景

- 启动新的扩散模型训练任务，并选择步数 T 和 beta 值时。
- 调试扩散模型时，若生成样本模糊（调度过于激进）或无法学习结构特征（调度过于平缓）。
- 对比不同论文中采用不同调度方案的设计时。

## 输入参数

- `T`：时间步数（timesteps），通常为 100-1000。
- `type`：调度类型，可选 linear（线性）| cosine（余弦）| sigmoid（S 型）。
- `target_alpha_bar_final`：在 t=T 时保留的信号比例，默认值为 0.001（即 99.9% 被噪声破坏）。
- 可选参数 `image_resolution`（图像分辨率）—— 较大尺寸的图像更适合采用噪声破坏速度较慢的调度方案（如余弦调度或偏移调度）。

## 调度公式

### 线性调度（Linear）
beta_t = beta_start + (beta_end - beta_start) * (t - 1) / (T - 1)
默认值：beta_start=1e-4, beta_end=0.02（源自 DDPM 论文）。

### 余弦调度（Cosine, Nichol & Dhariwal, 2021）
alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi/2)
beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
s = 0.008。该方案能更长时间地保留信号，在较少步数下表现更佳。

### S 型调度（Sigmoid）
alpha_bar_t = 1 / (1 + exp(k * (t/T - 0.5)))
k 取值范围为 6 到 12。这是一种良好的折中方案，被部分 SDXL 变体所采用。

## 操作步骤

1. 根据公式计算 beta 值。
2. 预计算 `alphas`、`alphas_cumprod`、`sqrt_alphas_cumprod`、`sqrt_one_minus_alphas_cumprod`。
3. 计算 SNR_t = alpha_bar_t / (1 - alpha_bar_t)；生成随时间变化的信噪比（SNR）汇总报告。
4. 验证 `alphas_cumprod[T-1]` 是否处于 `target_alpha_bar_final` 的 10% 误差范围内；若超出，则调整 beta_end（线性）、s（余弦）或 k（S 型）并重新计算。
5. 报告三个关键检查点：
   - `t=T*0.25` —— 早期噪声破坏阶段
   - `t=T*0.5` —— 中期阶段
   - `t=T*0.75` —— 接近最终阶段

## 输出报告

[schedule]
  type:   <name>
  T:      <int>
  beta_start: <float>   beta_end: <float>

[signal retention]
  t=0.25T:  alpha_bar=<X>  SNR=<X>
  t=0.5T:   alpha_bar=<X>  SNR=<X>
  t=0.75T:  alpha_bar=<X>  SNR=<X>
  t=T:      alpha_bar=<X>  SNR=<X>

[warnings]
  - <if alpha_bar collapses before 0.75T>
  - <if beta_end produces NaN in log-SNR>

## 规则

- 严禁输出任何包含 `alpha_bar_t <= 0` 的调度；若数值低于 1e-5 则进行截断（clamp）并发出警告。
- 对于少步数采样（< 30 步），默认推荐使用余弦调度（Cosine）。
- 当 `quality_target == research` 时，默认使用线性调度（Linear）—— DDPM 基线模型均采用线性调度进行报告。
- 当 `image_resolution > 256` 时，建议采用偏移调度（shifted schedule, Chen, 2023），以便在高分辨率下保留更多信号。