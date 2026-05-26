# 深度Q网络 (Deep Q-Networks, DQN)

> 2013年：Mnih 直接在原始像素上训练了一个 Q学习 (Q-learning) 网络，在七款 Atari 游戏中击败了所有经典强化学习 (Reinforcement Learning, RL) 智能体。2015年：该方法扩展至49款游戏，成果发表于《自然》(Nature) 杂志，由此开启了深度强化学习 (Deep-RL) 时代。DQN 本质上就是 Q学习 加上三项让函数近似 (Function Approximation) 保持稳定的技巧。

**类型：** 实战构建
**语言：** Python
**前置知识：** 第3阶段 · 03 (反向传播, Backpropagation)，第9阶段 · 04 (Q学习, Q-learning；SARSA)
**预计耗时：** 约75分钟

## 核心问题

表格型 Q学习 (Tabular Q-learning) 需要为每一个（状态，动作）对存储独立的 Q值 (Q-value)。国际象棋棋盘的状态数约为 10⁴³。一帧 Atari 游戏画面包含 210×160×3 = 100,800 个特征。当状态数达到数千时，表格型强化学习 (Tabular RL) 就会失效，更不用说数十亿的状态了。

事后看来，解决方案显而易见：用神经网络 `Q(s, a; θ)` 替代 Q表 (Q-table)。但这一“事后之明”却耗费了数十年。在 Q学习 中直接进行朴素的函数近似会在“致命三元组 (Deadly Triad)”——函数近似 + 自举 (Bootstrapping) + 异策略学习 (Off-policy Learning)——的共同作用下导致算法发散。Mnih 等人（2013, 2015）提出了三项工程技巧来稳定学习过程：

1. **经验回放 (Experience Replay)**：打破状态转移之间的相关性。
2. **目标网络 (Target Network)**：冻结自举目标值。
3. **奖励裁剪 (Reward Clipping)**：归一化梯度幅值。

在 Atari 游戏上的 DQN 首次实现了仅凭单一网络架构和一组超参数，就能直接从原始像素中解决数十种控制问题。此后所有“深度强化学习”的进展——如 DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57 等——均建立在这三项基础技巧之上。

## 核心概念

![DQN 训练循环：环境、经验回放缓冲区、在线网络、目标网络、贝尔曼时序差分损失](../assets/dqn.svg)

**目标。** DQN 旨在最小化神经 Q 函数（neural Q-function）上的单步时序差分损失（one-step TD loss）：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络（online network），每一步通过梯度下降（gradient descent）进行更新。`θ^-` = 目标网络（target network），定期从 `θ` 复制而来（约每 10,000 步一次）。`D` = 存储历史状态转移（transitions）的经验回放缓冲区（replay buffer）。

**三大技巧（按重要性排序）：**

**经验回放（Experience Replay）。** 一个容量约为 `~10⁶` 个状态转移的环形缓冲区。每个训练步骤从中均匀随机采样一个小批量（minibatch）。这打破了时间相关性（temporal correlation，即连续帧几乎完全相同），使网络能够多次从稀疏的奖励转移中学习，并解耦了连续的梯度更新。若不采用此方法，在 Atari 游戏上使用神经网络进行同策略时序差分（on-policy TD）学习会导致发散。

**目标网络（Target Network）。** 在贝尔曼方程（Bellman equation）两侧使用同一个网络 `Q(·; θ)` 会导致目标值在每次更新时都发生变化——如同“追逐自己的尾巴”。解决方法是：保留第二个权重冻结的网络 `Q(·; θ^-)`。每隔 `C` 步，将 `θ → θ^-` 进行复制。这能在数千次梯度更新期间稳定回归目标。软更新（Soft updates）`θ^- ← τ θ + (1-τ) θ^-`（用于 DDPG、SAC）是一种更平滑的变体。

**奖励裁剪（Reward Clipping）。** Atari 游戏的奖励幅度从 1 到 1000+ 不等。将其裁剪至 `{-1, 0, +1}` 可防止单一游戏主导梯度方向。当奖励幅度本身很重要时，这种做法是错误的；但在 Atari 中仅奖励符号有意义，因此完全适用。

**Double DQN（双 DQN）。** Hasselt (2016) 修复了最大化偏差（maximization bias）：使用在线网络来*选择*动作，使用目标网络来*评估*该动作。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

可直接无缝替换，且效果始终更优。建议默认使用。

**其他改进（Rainbow, 2017）：** 优先经验回放（Prioritized Replay，更频繁地采样高 TD 误差的状态转移）、决斗网络架构（Dueling Architecture，分离 `V(s)` 和优势函数头）、噪声网络（Noisy Networks，用于学习探索）、n 步回报（n-step returns）、分布型 Q 学习（Distributional Q，如 C51/QR-DQN）、多步自举（multi-step bootstrapping）。每项改进都能带来几个百分点的提升，且收益大致可叠加。

## 动手实现

此处的代码仅使用标准库且不依赖 NumPy。我们在一个微型连续网格世界（GridWorld）中手动实现了一个单隐藏层的多层感知机（MLP），因此每个训练步骤仅需微秒级时间即可完成。该算法在扩展后与用于 Atari 游戏的深度 Q 网络（DQN）完全一致。

### 步骤 1：经验回放缓冲区（Replay Buffer）

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)

Atari 环境通常需要约 50,000 的容量；而对于我们的玩具环境，5,000 已足够。

### 步骤 2：微型 Q 网络（Q-Network）（手动实现的 MLP）

class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h

前向传播（Forward Pass）流程：线性层 → ReLU 激活函数 → 线性层。这就是整个网络的全部结构。

### 步骤 3：DQN 更新

def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))

其整体架构与第 04 课中的 Q 学习（Q-Learning）一致，但有两处区别：(a) 我们通过可微的 `Q(·; θ)` 进行反向传播（Backpropagation），而非直接查表；(b) 目标值计算使用了 `Q(·; θ^-)`。

### 步骤 4：外层循环

在每个回合（Episode）中，基于 `Q(·; θ)` 采用 ε-贪婪（ε-Greedy）策略执行动作，将状态转移（Transition）推入缓冲区，采样一个小批量（Minibatch），执行一次梯度更新步骤，并定期同步 `θ^- ← θ`。具体模式如下：

for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next

在我们使用 16 维独热编码（One-Hot）状态的微型 GridWorld 中，智能体（Agent）仅需约 500 个回合即可学习到接近最优的策略（Policy）。若应用于 Atari 环境，则需将数据规模扩展至 2 亿帧，并加入卷积神经网络（CNN）特征提取器。

## 常见陷阱（Pitfalls）

- **致命三角 (Deadly Triad)。** 函数近似 (Function Approximation) + 异策略 (Off-policy) + 自举 (Bootstrapping) 可能导致算法发散。DQN 通过目标网络 (Target Network) 和经验回放 (Experience Replay) 缓解此问题；切勿移除其中任何一项。
- **探索 (Exploration)。** ε 必须衰减，通常在训练的前 ~10% 阶段从 1.0 降至 0.01。若早期探索不足，Q 网络 (Q-Network) 将收敛至局部最优区域。
- **高估偏差 (Overestimation)。** 对含噪声的 Q 值取 `max` 会产生向上偏差。在生产环境中务必使用 Double DQN。
- **奖励尺度 (Reward Scale)。** 对奖励进行裁剪或归一化；梯度幅值与奖励幅值成正比。
- **经验回放冷启动 (Replay Buffer Coldstart)。** 在缓冲区积累数千条状态转移 (Transitions) 数据前，切勿开始训练。仅基于约 20 个样本计算的早期梯度极易导致过拟合。
- **目标网络同步频率 (Target Sync Frequency)。** 过于频繁 ≈ 失去目标网络的作用；过于稀疏 ≈ 目标值滞后。Atari DQN 设定为每 10,000 个环境步 (Environment Steps) 同步一次。经验法则：每训练总时长的 ~1/100 同步一次。
- **观测预处理 (Observation Preprocessing)。** Atari DQN 通过堆叠 4 帧图像使状态满足马尔可夫性 (Markov Property)。任何包含速度信息的环境都需要采用帧堆叠 (Frame-stacking) 或循环状态 (Recurrent State) 表示。

## 使用场景 (Use It)

到 2026 年，DQN 已很少作为最先进 (State-of-the-Art) 算法，但它仍是异策略算法的基准参考：

| 任务 | 首选方法 | 为何不选 DQN？ |
|------|------------------|--------------|
| 离散动作类 Atari 任务 | Rainbow DQN 或 Muesli | 框架相同，但包含更多优化技巧。 |
| 连续控制 | SAC / TD3 (Phase 9 · 07) | DQN 缺乏策略网络。 |
| 同策略 (On-policy) / 高吞吐量 | PPO (Phase 9 · 08) | 无需经验回放缓冲区；更易于扩展。 |
| 离线强化学习 (Offline RL) | CQL / IQL / Decision Transformer | 采用保守 Q 目标，避免自举发散。 |
| 大规模离散动作空间（推荐系统） | 带动作嵌入的 DQN 或 IMPALA | 可行；细节装饰更重要。 |
| 大语言模型强化学习 (LLM RL) | PPO / GRPO | 序列级优化而非步级优化；损失函数不同。 |

这些经验依然适用。经验回放与目标网络广泛见于 SAC、TD3、DDPG、SAC-X、AlphaZero 的自对弈缓冲区以及所有离线强化学习方法中。奖励裁剪在 PPO 中演变为优势函数归一化 (Advantage Normalization)。该架构已成为后续算法的设计蓝图。

## 交付 (Ship It)

保存为 `outputs/skill-dqn-trainer.md`：

---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.

## 练习

1. **简单。** 运行 `code/main.py`。绘制每回合回报（per-episode return）曲线。滑动平均值（running mean）超过 -10 需要多少个回合？
2. **中等。** 禁用目标网络（target network）（在贝尔曼目标（Bellman target）的两侧均使用在线网络（online network））。评估训练的不稳定性——回报会出现震荡还是发散？
3. **困难。** 引入 Double DQN：使用在线网络选择 `argmax a'`，使用目标网络进行价值评估。在带噪声奖励的 GridWorld 环境中，对比使用与不使用 Double DQN 训练 1,000 个回合后，`Q(s_0, best_a)` 与真实最优状态价值 `V*(s_0)` 之间的偏差。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| DQN | “深度 Q 学习” | 结合神经网络 Q 函数、经验回放（Experience Replay）和目标网络（Target Network）的 Q 学习算法。 |
| 经验回放（Experience Replay） | “打乱的转移样本” | 每个梯度步均匀采样的环形缓冲区；用于消除数据间的相关性。 |
| 目标网络（Target Network） | “冻结的自举目标” | 定期复制的 Q 网络，用于计算贝尔曼目标；旨在稳定训练过程。 |
| 致命三要素（Deadly Triad） | “强化学习为何发散” | 函数近似（Function Approximation）+ 自举（Bootstrapping）+ 异策略（Off-policy）= 无法保证收敛。 |
| Double DQN | “解决最大化偏差的改进” | 在线网络负责选择动作，目标网络负责评估该动作。 |
| Dueling DQN | “V 和 A 分支” | 将 Q 值分解为 Q = V + A - mean(A)；输出维度相同，但梯度流动更优。 |
| Rainbow | “技巧大合集” | 将 DDQN、PER、Dueling、n-step、Noisy Nets 和分布型强化学习（Distributional RL）整合于一体的算法。 |
| PER | “优先经验回放” | 根据 TD 误差（TD-error）的大小按比例采样转移样本。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) —— 2013 年 NeurIPS 研讨会论文，开启了深度强化学习（Deep RL）的研究浪潮。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) —— 发表于《Nature》的论文，介绍了在 49 款 Atari 游戏上训练的 DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) —— DDQN 算法。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) —— Dueling DQN 架构。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) —— 整合多项改进技巧的论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) —— 清晰且现代的算法讲解。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) —— 教科书级讲解“致命三要素”（函数近似 + 自举 + 异策略），DQN 的目标网络与经验回放缓冲区正是为了解决该问题而设计。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) —— 用于消融实验（Ablation Studies）的单文件 DQN 参考实现；建议与本课程的从零实现版本对照阅读。