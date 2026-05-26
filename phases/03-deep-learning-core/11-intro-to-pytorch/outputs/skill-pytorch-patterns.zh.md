---
name: pytorch模式技能
description: PyTorch 训练、评估与部署的参考模式
version: 1.0.0
phase: 03
lesson: 11
tags: [pytorch, 训练, 深度学习, gpu, 模式]
---

## 标准训练循环 (Canonical Training Loop)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

## 混合精度训练 (Mixed Precision Training)

from torch.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    with autocast(device_type="cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

适用场景：在支持 float16 的 GPU 硬件（如 V100、A100、H100、RTX 3090 及以上）上进行训练。预计可获得约 1.5 到 2 倍的速度提升，并减少约 50% 的显存占用。

## 梯度累积 (Gradient Accumulation)

accumulation_steps = 4
optimizer.zero_grad()
for i, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

适用场景：当所需的有效批次大小 (effective batch size) 超出 GPU 显存限制时。将损失值除以 `accumulation_steps` 可保持梯度缩放比例一致。

## 保存与加载 (Save and Load)

torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item(),
}, "checkpoint.pt")

checkpoint = torch.load("checkpoint.pt", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

若需恢复训练，请务必保存优化器状态 (optimizer state)。若仅用于推理 (inference)，则只需保存 `model.state_dict()`。

## 自定义数据集 (Custom Dataset)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = self._load_samples(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def _load_samples(self, data_dir):
        ...

## DataLoader 配置 (DataLoader Configuration)

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)

| 参数 | 作用 | 适用场景 |
|-----------|-------------|-------------|
| num_workers=4 | 并行数据加载 | 在多核机器上始终启用 |
| pin_memory=True | 锁定页 CPU 内存 (Page-locked CPU memory) | 在 GPU 上训练时 |
| drop_last=True | 丢弃最后一个不完整的批次 | 使用批归一化 (Batch Normalization) 时 |
| persistent_workers=True | 跨训练轮次保持工作进程存活 | 当 num_workers > 0 时 |

## 学习率调度策略 (Learning Rate Schedules)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=num_epochs * len(train_loader),
    pct_start=0.1,
)

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        ...
        optimizer.step()
        scheduler.step()

`OneCycleLR`：适用于大多数任务的最佳默认选择。学习率先预热至 `max_lr`，随后进行余弦衰减（cosine decay）。应在每个批次（batch）后调用 `scheduler.step()`，而非每个训练轮次（epoch）后。

## 权重初始化（Weight Initialization）

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

model.apply(init_weights)

## 推理模式（Inference Mode）

model.eval()

with torch.inference_mode():
    outputs = model(inputs)

`torch.inference_mode()` 的速度快于 `torch.no_grad()`，因为它会完全禁用自动求导（autograd），而不仅仅是抑制梯度计算。

## 常见错误检查清单（Common Mistakes Checklist）

1. 在 `CrossEntropyLoss` 之前应用 `softmax`（该损失函数内部已包含 `log_softmax`）
2. 在验证阶段忘记调用 `model.eval()`
3. 忘记将张量（tensor）移动到与模型相同的计算设备（device）上
4. 未调用 `optimizer.zero_grad()`（默认情况下梯度会累积）
5. 在训练期间使用 `torch.no_grad()`（会禁用梯度计算）
6. 将 `num_workers` 设置得过高（会生成过多进程，导致内存抖动（memory thrashing））
7. 在 GPU 上训练时未使用 `pin_memory=True`
8. 保存整个模型对象而非 `state_dict`（代码重构时会导致报错）