# 调试与性能分析 (Debugging and Profiling)

> 最糟糕的 AI 缺陷（AI bugs）并不会导致程序崩溃。它们会在垃圾数据上默默训练，并呈现一条看似完美的损失曲线（loss curve）。

**类型：** 项目构建 (Build)
**语言：** Python
**前置条件：** 第 1 课（开发环境），具备 PyTorch 基础知识
**时长：** 约 60 分钟

## 学习目标

- 使用条件式 `breakpoint()` 和 `debug_print` 在训练过程中检查张量形状 (tensor shapes)、数据类型 (dtypes) 以及 NaN 值
- 使用 `cProfile`、`line_profiler` 和 `tracemalloc` 对训练循环 (training loops) 进行性能剖析，以定位瓶颈
- 检测常见的 AI 缺陷 (AI bugs)：形状不匹配、NaN 损失值、数据泄露 (data leakage) 以及设备不匹配的张量
- 配置 TensorBoard 以可视化损失曲线 (loss curves)、权重直方图 (weight histograms) 和梯度分布 (gradient distributions)

## 问题

AI 代码的失败方式与常规代码不同。Web 应用崩溃时会抛出堆栈跟踪 (stack trace)。而配置错误的训练循环 (training loop) 可能运行 8 小时，消耗 200 美元的 GPU 计算时间，最终却只生成一个对所有输入都预测均值的模型。代码本身从未报错。真正的缺陷可能是张量 (tensor) 被放在了错误的设备 (device) 上、忘记调用 `.detach()`，或是标签泄露到了特征中。

你需要调试工具 (debugging tools) 来捕获这些静默失败 (silent failures)，以免它们白白浪费你的时间和计算资源。

## 概念

AI 调试（AI Debugging）分为三个层级：

graph TD
    L3["3. Training Dynamics<br/>Loss curves, gradient norms, activations"] --> L2
    L2["2. Tensor Operations<br/>Shapes, dtypes, devices, NaN/Inf values"] --> L1
    L1["1. Standard Python<br/>Breakpoints, logging, profiling, memory"]

大多数人会直接跳到第 3 层（盯着 TensorBoard 看）。但实际上，80% 的 AI 错误（AI Bugs）都存在于第 1 层和第 2 层。

## 构建它

### Part 1: Print Debugging (Yes, It Works)

Print debugging gets dismissed. It shouldn't. For tensor code, a targeted print statement beats stepping through a debugger because you need to see shapes, dtypes, and value ranges all at once.

```python
def debug_print(name, tensor):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, "
          f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
          f"mean={tensor.mean().item():.4f}, "
          f"has_nan={tensor.isnan().any().item()}")
```

Call this after every suspicious operation. When the bug is found, remove the prints. Simple.

### Part 2: Python Debugger (pdb and breakpoint)

The built-in debugger is underrated for AI work. Drop `breakpoint()` into your training loop and inspect tensors interactively.

```python
def training_step(model, batch, criterion, optimizer):
    inputs, labels = batch
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    if loss.item() > 100 or torch.isnan(loss):
        breakpoint()

    loss.backward()
    optimizer.step()
```

When the debugger drops you in, useful commands:

- `p outputs.shape` to check shapes
- `p loss.item()` to see the loss value
- `p torch.isnan(outputs).sum()` to count NaNs
- `p model.fc1.weight.grad` to check gradients
- `c` to continue, `q` to quit

This is conditional debugging. You only stop when something looks wrong. For a 10,000-step training run, that matters.

### Part 3: Python Logging

Replace print statements with logging when your debugging goes beyond a quick check.

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting training: lr=%.4f, batch_size=%d", lr, batch_size)
logger.warning("Loss spike detected: %.4f at step %d", loss.item(), step)
logger.error("NaN loss at step %d, stopping", step)
```

Logging gives you timestamps, severity levels, and file output. When a training run fails at 3 AM, you want a log file, not terminal output that scrolled off screen.

### Part 4: Timing Code Sections

Knowing where time goes is the first step to optimization.

```python
import time

class Timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[{self.name}] {elapsed:.4f}s")

with Timer("data loading"):
    batch = next(dataloader_iter)

with Timer("forward pass"):
    outputs = model(batch)

with Timer("backward pass"):
    loss.backward()
```

Common finding: data loading takes 60% of training time. The fix is `num_workers > 0` in your DataLoader, not a faster GPU.

### Part 5: cProfile and line_profiler

When you need more than manual timers:

```bash
python -m cProfile -s cumtime train.py
```

This shows every function call sorted by cumulative time. For line-by-line profiling:

```bash
pip install line_profiler
```

```python
@profile
def train_step(model, data, target):
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    return loss

# Run with: kernprof -l -v train.py
```

### Part 6: Memory Profiling

#### CPU Memory with tracemalloc

```python
import tracemalloc

tracemalloc.start()

# your code here
model = build_model()
data = load_dataset()

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")
for stat in top_stats[:10]:
    print(stat)
```

#### CPU Memory with memory_profiler

```bash
pip install memory_profiler
```

```python
from memory_profiler import profile

@profile
def load_data():
    raw = read_csv("data.csv")       # watch memory jump here
    processed = preprocess(raw)       # and here
    return processed
```

Run with `python -m memory_profiler your_script.py` to see line-by-line memory usage.

#### GPU Memory with PyTorch

```python
import torch

if torch.cuda.is_available():
    print(torch.cuda.memory_summary())

    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

When you hit OOM (Out of Memory):

1. Reduce batch size (first thing to try, always)
2. Use `torch.cuda.empty_cache()` to free cached memory
3. Use `del tensor` followed by `torch.cuda.empty_cache()` for large intermediates
4. Use mixed precision (`torch.cuda.amp`) to halve memory usage
5. Use gradient checkpointing for very deep models

### Part 7: Common AI Bugs and How to Catch Them

#### Shape Mismatch

The most frequent bug. A tensor has shape `[batch, features]` when the model expects `[batch, channels, height, width]`.

```python
def check_shapes(model, sample_input):
    print(f"Input: {sample_input.shape}")
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            in_shape = inp[0].shape if isinstance(inp, tuple) else inp.shape
            out_shape = out.shape if hasattr(out, "shape") else type(out)
            print(f"  {name}: {in_shape} -> {out_shape}")
        return hook

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()
```

Run this once with a sample batch. It maps every shape transformation in your model.

#### NaN Loss

NaN loss means something exploded. Common causes:

- Learning rate too high
- Division by zero in custom loss
- Log of zero or negative number
- Exploding gradients in RNNs

```python
def detect_nan(model, loss, step):
    if torch.isnan(loss):
        print(f"NaN loss at step {step}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    print(f"  Inf gradient in {name}")
        return True
    return False
```

#### Data Leakage

Your model gets 99% accuracy on the test set. Sounds great. It's a bug.

```python
def check_data_leakage(train_set, test_set, id_column="id"):
    train_ids = set(train_set[id_column].tolist())
    test_ids = set(test_set[id_column].tolist())
    overlap = train_ids & test_ids
    if overlap:
        print(f"DATA LEAKAGE: {len(overlap)} samples in both train and test")
        return True
    return False
```

Also check for temporal leakage: using future data to predict the past. Sort by timestamp before splitting.

#### Wrong Device

Tensors on different devices (CPU vs GPU) cause runtime errors. But sometimes a tensor silently stays on CPU while everything else is on GPU, and training just runs slowly.

```python
def check_devices(model, *tensors):
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    for i, t in enumerate(tensors):
        if t.device != model_device:
            print(f"  WARNING: tensor {i} on {t.device}, model on {model_device}")
```

### Part 8: TensorBoard Basics

TensorBoard shows you what's happening inside training over time.

```bash
pip install tensorboard
```

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment_1")

for step in range(num_steps):
    loss = train_step(model, batch)

    writer.add_scalar("loss/train", loss.item(), step)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

    if step % 100 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(f"weights/{name}", param, step)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, step)

writer.close()
```

Launch it:

```bash
tensorboard --logdir=runs
```

What to look for:

- **Loss not decreasing**: Learning rate too low, or model architecture issue
- **Loss oscillating wildly**: Learning rate too high
- **Loss goes to NaN**: Numerical instability (see NaN section above)
- **Train loss decreasing, val loss increasing**: Overfitting
- **Weight histograms collapsing to zero**: Vanishing gradients
- **Gradient histograms exploding**: Need gradient clipping

### Part 9: VS Code Debugger

For interactive debugging, configure VS Code with a `launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Training",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

Set breakpoints by clicking the gutter. Use the Variables pane to inspect tensor properties. The Debug Console lets you run arbitrary Python expressions mid-execution.

Useful for stepping through data preprocessing pipelines where you want to see each transformation.

## 使用方法

以下是能够捕获大多数 AI 错误 (AI bugs) 的调试工作流：

1. **训练 (training) 前**：使用样本批次 (batch) 运行 `check_shapes`。验证输入和输出维度 (dimensions) 是否符合预期。
2. **前 10 步**：对损失 (loss)、输出和梯度 (gradients) 使用 `debug_print`。确认没有出现 NaN，且数值处于合理范围内。
3. **训练期间**：记录损失、学习率 (learning rate) 和梯度范数 (gradient norms)。使用 TensorBoard 进行可视化。
4. **发生故障时**：在故障点插入 `breakpoint()`。交互式检查张量 (tensors)。
5. **性能调优**：对数据加载 (data loading)、前向传播 (forward pass) 与反向传播 (backward pass) 分别进行耗时统计。若接近内存溢出 (OOM)，请进行内存分析。

## 发布

运行调试工具包（debugging toolkit）脚本：

python phases/00-setup-and-tooling/12-debugging-and-profiling/code/debug_tools.py

请参阅 `outputs/prompt-debug-ai-code.md`，其中提供了一个用于诊断 AI 特定错误（AI-specific bugs）的提示词（prompt）。

## 练习

1. 运行 `debug_tools.py` 并逐节查看输出。修改虚拟模型以引入非数字 (NaN)（提示：在前向传播 (forward pass) 中执行除以零操作），并观察检测器如何捕获该异常。
2. 使用 `cProfile` 对训练循环 (training loop) 进行性能剖析，并定位最耗时的函数。
3. 使用 `tracemalloc` 找出数据加载流水线 (data loading pipeline) 中内存分配最多的代码行。
4. 为一次简单的训练任务配置 TensorBoard，并判断模型是否出现过拟合 (overfitting)。
5. 在训练循环中插入 `breakpoint()`。练习在调试器提示符下检查张量 (tensor) 形状、计算设备以及梯度 (gradient) 值。