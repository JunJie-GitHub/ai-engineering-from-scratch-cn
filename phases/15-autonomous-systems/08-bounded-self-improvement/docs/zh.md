# 有界自我改进设计 (Bounded Self-Improvement Designs)

> 研究已收敛于四种用于约束自我改进循环 (self-improvement loop) 的基本原语 (primitives)。形式化不变量 (formal invariants) 必须在每次编辑中保持成立。对齐锚点 (alignment anchors) 不可被修改。多目标约束 (multi-objective constraints) 要求每个维度（安全性、公平性、鲁棒性）都必须满足，而不仅仅是性能。回归检测 (regression detection) 会在历史指标表明能力下降时暂停循环。它们都不是安全性的证明——信息论结果 (information-theoretic results)（柯尔莫哥洛夫复杂度 Kolmogorov complexity、洛布定理 Löb's theorem）限制了任何系统对其自身后继系统所能证明的范围。它们是缓解措施，旨在提高静默失败 (silent failure) 的成本。

**类型：** 学习
**语言：** Python（标准库，带不变量检查的有界循环）
**前置条件：** 第15阶段 · 07 (RSI)，第15阶段 · 04 (DGM)
**时长：** 约60分钟

## 核心问题

第7课的竞赛模拟器表明，微小的速率差异会不断累积，最终形成巨大差距。第4课的DGM案例研究则显示，循环机制可能会主动针对其自身的评估器进行博弈。这两项结果共同指向了同一个工程问题：应当为自我改进循环设置怎样的约束条件，才能确保这些约束不会被循环自身暗中削弱？

ICLR 2026 RSI研讨会总结 (openreview.net/pdf?id=OsPQ6zTQXV) 明确了这四种原语。Anthropic的RSP v3.0（第19课）和DeepMind的FSF v3（第20课）均在能力阈值中引用了它们。Meta的HyperAgents工作以及SAHOO（2026年3月）等社区框架已在生产环境中实现了其中的部分子集。

需要坦诚地指出：这些仅仅是缓解措施。信息论结果从根本上限制了任何系统对其自身后继版本所能证明的范围，且目前没有任何设计能从形式化层面彻底解决该问题。一个约束良好的循环确实比无约束循环更安全，但这绝不等同于绝对安全。

## 核心概念

### 原语 1：形式化不变量 (Formal Invariants)

不变量 (Invariant) 是指在每次自我修改前后都必须保持成立的属性。例如：

- 输出分布 (Output Distribution) 受固定的宪法头部 (Constitution Header) 约束（第 17 课）。
- 没有任何工具调用 (Tool Call) 会发送至未授权的端点。
- 内存写入 (Memory Writes) 必须经过已记录日志且经过签名的路径。
- 评估器 (Evaluator) 的模块哈希值 (Module Hash) 与已批准的版本一致。

不变量由循环 (Loop) 无法编辑的外部代码进行检查。如果提议的修改违反了不变量，则会被拒绝。这能够捕获类似 DGM 风格的评估器篡改模式（第 4 课）。

难点在于选择那些对安全性至关重要且能在合理时间内计算验证的不变量。过度规定会阻碍进展；规定不足则会让循环绕过重要属性。

### 原语 2：对齐锚点 (Alignment Anchors)

对齐锚点是对循环核心目标的不可变表示，固定在循环的编辑范围之外。例如：

- 每次调用时都从只读位置加载的宪法文本（第 17 课）。
- 位于独立仓库中且具有独立访问控制权限的评估器。
- 由人类签名并在每个周期重新验证的工具权限清单 (Tool-Permission Manifest)。

锚点的作用是防止目标漂移 (Objective Drift)。循环可以优化其追求目标的方式，但无法更改目标本身。

隐蔽的失效模式：即使循环无法编辑锚点，如果循环在读取锚点的方式上发生漂移，仍可能对其重新解释。宪法式人工智能 (Constitutional AI)（第 17 课）明确基于推理来处理新情况；而解释漂移 (Interpretation Drift) 正是发生在这个推理层。因此，锚点是必要条件，而非充分条件。

### 原语 3：多目标约束 (Multi-Objective Constraints)

仅优化单一标量分数 (Scalar Score) 的循环会寻找捷径。而必须同时满足多个硬性约束的循环，可用的捷径则少得多。典型维度包括：

- 性能（任务级基准测试）
- 安全性（红队评估、对已知恶意输入的拒绝率）
- 公平性（对敏感子群体的差异化影响边界）
- 鲁棒性（分布外测试集、对抗性输入处理）

只有当所有约束均满足时，修改才会被接受。第 13 课的成本控制器 (Cost Governor) 在此基础上叠加了财务约束。第 18 课的 Llama Guard 则作为安全性维度接入。

### 原语 4：回归检测 (Regression Detection)

每个周期的得分都会与历史分布进行对比。如果下降幅度超出配置的容差范围，循环将暂停。这能够捕获静默的能力退化，否则这种退化会被循环“超越式改进”的滑动平均值所掩盖。

一种实际实现方案：存储最近 N 个周期的每项任务得分。每个新周期计算每项任务的增量 (Delta)。如果任何增量低于阈值，则该周期被拒绝，并交由人工审查。

### 信息论限制 (Information-Theoretic Limits)

柯尔莫哥洛夫复杂度 (Kolmogorov Complexity) 和洛布定理 (Löb's Theorem) 为系统能够证明自身属性的能力设定了上限。Schmidhuber 提出的形式化哥德尔机 (Gödel Machine)（第 4 课）旨在追求这一上限的最高值；但至今无人完成过非平凡的证明。洛布定理指出：如果一个系统可证明地相信“如果我证明我应该做 X，我就会做 X”，那么它会在未证明应该做的情况下直接执行 X，这是一种著名的自指失效 (Self-Reference Failure)。

这对我们上述原语的启示在于：它们无法彻底解决安全问题，但会让静默失效的代价变得更高。一个恶意或发生漂移的循环原本可以悄悄绕过缺失的检查，现在则必须主动破坏显式的检查，而这种行为会留下更容易被检测到的特征。

### 实例演示

假设一个智能体 (Agent) 提议进行一项编辑。门控栈 (Gating Stack) 流程如下：

1. 不变量检查：模块哈希值、工具权限清单、宪法头部。
2. 锚点检查：目标声明与已批准版本匹配（按字节或语义）。
3. 多目标评估：性能、安全性、公平性、鲁棒性维度。
4. 回归检测：任何维度的下降幅度均不超过容差。

只有全部四项通过，编辑才会生效。任何一项失败都会导致循环暂停。

## 使用它

`code/main.py` 在第 4 课的 DGM 风格玩具模型（toy model）上运行一个有界自我改进循环（bounded self-improvement loop），并在其基础上叠加了四个原语（primitives）。每个原语均可独立启用或禁用。本演示旨在证明：每个原语都能拦截特定类型的故障，而移除其中任意一个都会导致该类故障漏过。

## 交付它

`outputs/skill-bounded-loop-review.md` 用于审查提议的有界循环，并对其实际实现的原语与声称实现的原语进行对比评分。

## 练习

1. 在启用所有原语的情况下运行 `code/main.py`。确认循环在主要指标上仍能取得提升，且不会让指标投机（hack）得逞。

2. 禁用回归检测（regression detection）。构造一个输入示例，展示这将如何导致系统接受静默能力退化（silent capability loss）。

3. 禁用多目标约束（multi-objective constraint）。展示循环如何在性能轴上收敛，同时安全轴指标出现下降。

4. 为编程智能体（coding agent）设计一个对齐锚点（alignment anchor）。具体包含什么文本？存储在哪里？如何进行检查？

5. 阅读 ICLR 2026 RSI 研讨会总结。从四个原语中任选其一，针对当前最先进水平（state of the art）提出一项具体的改进方案。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| 不变量（Invariant） | “恒真属性” | 每次编辑前后均由外部代码检查的属性 |
| 对齐锚点（Alignment anchor） | “固定目标” | 位于循环编辑界面之外、不可变的核心目标表示 |
| 多目标约束（Multi-objective constraint） | “所有维度必须达标” | 性能、安全性、公平性、鲁棒性——全部为必需条件 |
| 回归检测（Regression detection） | “指标下降即暂停” | 当历史指标变化量暗示能力退化时暂停循环 |
| 柯尔莫哥洛夫界（Kolmogorov bound） | “信息论极限” | 限制系统对其自身后继版本所能证明的内容 |
| 洛布定理（Löb's theorem） | “自指陷阱” | 系统可基于“我应当”采取行动，而无需证明其确实应当 |
| 门控栈（Gate stack） | “分层检查” | 多个原语组合；任一原语失败即拒绝该编辑 |
| 有界改进（Bounded improvement） | “缓解而非证明” | 提高静默失败的成本；但并未彻底解决安全问题 |

## 延伸阅读

- [ICLR 2026 RSI 研讨会总结 (OpenReview)](https://openreview.net/pdf?id=OsPQ6zTQXV) —— 四个原语的收敛性。
- [Anthropic 负责任扩展政策 v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) —— 多目标能力阈值。
- [DeepMind 前沿安全框架 v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) —— 将欺骗性对齐（deceptive alignment）监控作为不变量原语。
- [Schmidhuber (2003). 哥德尔机](https://people.idsia.ch/~juergen/goedelmachine.html) —— 这些原语的形式化证明先驱。
- [Anthropic — Claude 宪法（2026 年 1 月）](https://www.anthropic.com/news/claudes-constitution) —— 基于推理的对齐锚点。