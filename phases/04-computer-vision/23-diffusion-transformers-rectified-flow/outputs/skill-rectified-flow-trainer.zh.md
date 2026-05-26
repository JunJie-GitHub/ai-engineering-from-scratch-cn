---
name: skill-rectified-flow-trainer
description: 编写一个结合 AdaLN DiT 与欧拉采样（Euler sampling）的完整整流流（rectified flow）训练循环
version: 1.0.0
phase: 4
lesson: 23
tags: [扩散模型 (diffusion), 整流流 (rectified flow), DiT, 训练 (training)]
---

# 整流流（Rectified Flow）训练器

生成一个简洁、精简的训练循环，用于在任何图像张量数据集上成功训练基于整流流的小型 DiT（Diffusion Transformer）。

## 适用场景

- 在小规模上复现 SD3 / FLUX 的训练目标。
- 在相同数据上对比评估整流流（rectified flow）与 DDPM（去噪扩散概率模型）。
- 为非标准领域（如医疗、卫星图像）构建自定义的整流流模型。

## 输入参数

- `model`：一个 `nn.Module`，接收 `(x, t)` 并返回预测的速度（velocity）。
- `dataset`：模型领域内干净图像的可迭代对象。
- `optimizer`：AdamW 优化器，参数为 `lr=1e-4`、`weight_decay=0.01`、`betas=(0.9, 0.99)`。
- `scheduler`：带预热（warmup）的余弦调度器，默认预热步数为 1000。

## 训练步骤

def rectified_flow_train_step(model, x0, optimizer, device):
    model.train()
    x0 = x0.to(device)
    n = x0.size(0)
    t = torch.rand(n, device=device)                     # uniform in [0, 1]
    epsilon = torch.randn_like(x0)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * epsilon
    target_v = epsilon - x0                              # velocity target
    pred_v = model(x_t, t)
    loss = F.mse_loss(pred_v, target_v)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

## 采样（欧拉法 / Euler）

@torch.no_grad()
def sample(model, shape, steps=20, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    t = torch.ones(shape[0], device=device)
    for _ in range(steps):
        v = model(x, t)
        x = x - dt * v
        t = t - dt
    return x

## 技巧与建议

- 使用 `torch.rand` 生成均匀分布的 `t`；采用 logit-normal 分布或 SD3 风格的加权采样对 `t` 进行采样会有轻微帮助，但并非入门必需。
- 模型权重的指数移动平均（EMA）是标准做法；建议维护一个衰减率为 0.9999 的 `ema_model`。
- 条件模型的无分类器引导（Classifier-free guidance）：训练时以 10% 的概率将条件替换为空/零嵌入（empty/null embedding）；推理时混合 `v_uncond + w * (v_cond - v_uncond)`，其中 `w` 取值约为 3-5。
- 对于 LDM（潜在扩散模型）风格的训练（如 FLUX、SD3），整个循环在 VAE（变分自编码器）的潜在空间（latent space）中运行；上述的干净 `x0` 实际上是 `VAE.encode(image)`。
- 在 32x32 玩具数据集上的典型收敛步数：2000-5000 步。在实际的 SD3 潜在空间训练中：数十万步。

## 报告格式

[rectified flow training]
  steps:        <int>
  final loss:   <float>
  ema decay:    <float>
  vae?:         yes | no
  cfg dropout:  <fraction>

[sampling]
  default steps: 20
  schnell / turbo target: 4
  full quality reference: 50+ (for comparison only)

## 注意事项

- 切勿在 RGB `uint8` 数据上直接使用图像空间速度目标训练整流流；请先将其归一化为零均值、单位方差。
- 务必按时间步区间（timestep-bucket）记录训练损失；如果早期时间步（接近 0）的损失高于晚期时间步（接近 1），则速度参数化（velocity parameterisation）可能存在实现错误。
- 不要在同一训练循环中混合整流流速度目标与 DDPM 噪声目标；请仅选择其一。
- 在 Ampere 架构及以上的 GPU 上使用 bfloat16 进行训练；由于速度幅值较大，float16 有时会在整流流训练中产生 NaN 梯度。