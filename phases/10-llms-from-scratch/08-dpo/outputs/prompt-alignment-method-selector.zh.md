---
name: 提示词对齐方法选择器
description: 为您的用例选择合适的对齐方法（SFT、RLHF、DPO、KTO、ORPO、SimPO）
version: 1.0.0
phase: 10
lesson: 8
tags: [对齐, dpo, rlhf, kto, orpo, simpo, 偏好优化, 微调]
---

# 对齐方法选择器 (Alignment Method Selector)

在为语言模型选择对齐方法时，请使用此框架评估您的数据、算力 (Compute) 和质量要求，然后选择最符合您约束条件的方法。

## 输入要求

提供：
- **基座模型 (Base Model)**（例如 Llama 3 8B、Mistral 7B、Qwen 2.5 72B）
- **起点 (Starting Point)**（基座模型，还是已经进行过 SFT？）
- **可用数据 (Available Data)**（指令对 (Instruction Pairs)、偏好对 (Preference Pairs)、非配对评分 (Unpaired Ratings)，或无）
- **算力预算 (Compute Budget)**（GPU 小时数、GPU 数量）
- **质量目标 (Quality Target)**（满足原型需求、与开源模型竞争力相当、达到业界最先进水平）
- **时间线 (Timeline)**（天、周、月）

## 决策矩阵

### 快速选择

| 您的情况 | 推荐方法 | 原因 |
|---------------|-------------------|-----|
| 无偏好数据，仅有指令对 | 仅使用 SFT | 缺乏偏好信号无法进行对齐 |
| 偏好对少于 5,000 条，算力有限 | DPO | 流程更简单，在小规模数据上表现良好 |
| 非配对反馈（仅有点赞/点踩） | KTO | 唯一无需成对比较即可工作的方法 |
| 希望在单次训练运行中完成对齐 | ORPO | 结合 SFT 与对齐，无需参考模型 (Reference Model) |
| 内存受限（无法加载参考模型） | SimPO | 无需参考模型 |
| 大规模、多目标对齐 | RLHF (PPO) | 独立的奖励模型 (Reward Model) 可捕捉复杂偏好 |
| 使用在线数据进行迭代对齐 | RLHF (PPO) | 可在循环中生成、评分并重新训练 |
| RLHF 后的微调优化 | DPO | 在特定偏好上对 RLHF 模型进行微调 |

### 详细对比

| 方法 | 数据要求 | 内存中的模型数量 | 训练循环 | 稳定性 | 适用规模 |
|--------|-----------------|-----------------|----------------|-----------|------------|
| SFT | 指令对（10K+） | 1 | 1 | 高 | 任意 |
| RLHF | 偏好对（20K+） | 3-4 | 3 | 低 | 大型（70B+） |
| DPO | 偏好对（5K+） | 2 | 2（SFT + DPO） | 高 | 中小型（7B-70B） |
| KTO | 非配对评分（5K+） | 2 | 2（SFT + KTO） | 高 | 任意 |
| ORPO | 偏好对（10K+） | 1 | 1 | 高 | 中小型 |
| SimPO | 偏好对（5K+） | 1 | 2（SFT + SimPO） | 高 | 中小型 |

## 各方法特定配置

### SFT

- **停止时机**：训练 1-3 个 epoch 后，或验证集损失不再下降时
- **关键超参数**：学习率 (Learning Rate)（1e-5 至 5e-5，模型越大取值越低）
- **关键细节**：在计算损失时掩码 (Mask) 指令部分的 token
- **避坑指南**：超过 3 个 epoch 会导致模型记忆化 (Memorization)；建议混入 2-5% 的预训练数据

### RLHF (PPO)

- **适用场景**：拥有 20K+ 对比对，需要多目标对齐，或希望进行迭代式在线学习
- **关键超参数**：KL 散度系数 (KL Coefficient)（0.01-0.05）、PPO 裁剪比率 (Clip Ratio)（0.1-0.3）、学习率（5e-6 至 3e-5）
- **关键细节**：奖励模型的参数量应大于或等于策略模型 (Policy Model)
- **避坑指南**：PPO 训练不稳定；需持续监控 KL 散度 (KL Divergence) 和奖励曲线

### DPO

- **适用场景**：拥有偏好对，且希望采用比 RLHF 更简单的流程
- **关键超参数**：Beta 值（0.1-0.5；值越低表示允许偏离参考模型的程度越大）
- **关键细节**：参考模型必须是 SFT 检查点的冻结副本
- **避坑指南**：对 Beta 值极其敏感；建议在 [0.05, 0.1, 0.2, 0.5] 范围内进行参数扫描 (Sweep)

### KTO

- **适用场景**：仅有“好”或“坏”的标签，缺乏成对比较数据
- **关键超参数**：Beta 值（同 DPO）、损失厌恶乘数 (Loss Aversion Multiplier)（对负面响应施加 1.5 倍权重）
- **关键细节**：需要正负样本大致平衡（比例约为 40%-60%）
- **避坑指南**：缺乏成对数据会导致梯度信号较弱；可能需要比 DPO 更多的数据

### ORPO

- **适用场景**：希望完全跳过监督微调（Supervised Fine-Tuning, SFT），直接从基座模型（Base Model）过渡到对齐（Aligned）状态
- **关键超参数**：Lambda（偏好项与 SFT 项的权重比例）
- **关键细节**：需要在同一个数据集中同时包含指令标签（Instruction Labels）和偏好对（Preference Pairs）
- **注意事项**：联合目标函数难以平衡；若 SFT 损失占主导，对齐效果会较弱

### SimPO

- **适用场景**：内存受限的环境，无法加载参考模型（Reference Model）
- **关键超参数**：Beta、Gamma（长度归一化指数）
- **关键细节**：长度归一化（Length Normalization）可防止模型倾向于生成过短的回复
- **注意事项**：缺乏参考模型作为锚点，模型可能发生更严重的漂移；需密切监控

## 训练流程模板

### 模板 1：快速原型（1-2 天）

Base Model -> SFT (1 epoch, 10K examples) -> DPO (3 epochs, 5K pairs)

计算资源：在 A100 上训练 7B 模型约需 4 GPU 小时
质量：具备扎实的指令遵循能力，实现基础的偏好对齐

### 模板 2：生产级质量（1-2 周）

Base Model -> SFT (2 epochs, 50K examples) -> DPO (5 epochs, 20K pairs) -> Eval -> Iterate

计算资源：7B 模型约需 40 GPU 小时，70B 模型约需 200 GPU 小时
质量：与开源的基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）模型具有竞争力

### 模板 3：前沿水平（1-3 个月）

Base Model -> SFT (2 epochs, 100K+ examples) -> RLHF (PPO, 50K+ pairs) -> DPO (targeted refinement) -> Eval -> Iterate

计算资源：70B 模型需 500+ GPU 小时
质量：接近前沿模型的对齐水平

### 模板 4：极简数据（1-2 天）

Base Model -> SFT (1 epoch, 5K examples) -> KTO (unpaired thumbs up/down from users)

计算资源：7B 模型约需 2 GPU 小时
质量：在数据收集开销极低的情况下，效果优于仅使用 SFT

## 评估流程

完成对齐后，请从以下维度进行评估：

1. **偏好胜率（Preference Win Rate）**：由人工评审在 200+ 测试提示词（Prompts）上对比对齐模型与 SFT 模型。目标：胜率 > 60%。
2. **基准测试保留率（Benchmark Retention）**：MMLU、HumanEval 或领域特定基准测试。性能下降幅度不应超过 SFT 基线的 5%。
3. **MT-Bench 或 AlpacaEval**：标准的对齐质量基准测试。与已发布的基线模型进行对比。
4. **安全性评估（Safety Evaluation）**：针对对抗性提示词（Adversarial Prompts）、越狱攻击（Jailbreaks）及有害请求类别进行测试。
5. **回复多样性（Response Diversity）**：测量 100 个提示词回复的熵（Entropy）值。低熵值意味着模式崩溃（Mode Collapse）。

## 常见失败模式

| 症状 | 原因 | 针对性修复方案 |
|---------|-------|-------------------|
| 回复冗长、填充内容过多 | 奖励模型（Reward Model）/ 隐式奖励偏好长文本 | DPO：增大 beta。RLHF：添加长度惩罚。SimPO：调整 gamma。 |
| 模型盲目附和所有观点 | 偏好数据偏差导致的讨好倾向（Sycophancy） | 添加正确回复与用户观点相左的偏好对 |
| 拒绝良性请求 | 安全数据过度对齐 | 降低安全示例比例，增加更多良性-拒绝偏好对 |
| 输出结果与 SFT 几乎一致 | Beta 过高（DPO/KTO）或 KL 系数（KL Coefficient）过高（PPO） | 降低 beta / KL 系数；当前模型未有效学习 |
| 训练损失震荡 | 学习率（Learning Rate）过高或数据不足 | 将 lr 降低 2-3 倍；增加偏好数据 |