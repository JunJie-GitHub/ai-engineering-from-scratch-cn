# 故障模式（Failure Modes）— MAST、群体思维（Groupthink）、单一文化（Monoculture）与级联错误（Cascading Errors）

> 2026 年的参考分类体系为 **MAST**（Cemri 等人，NeurIPS 2025，arXiv:2503.13657），该体系基于 7 个最先进的开源多智能体系统（Multi-Agent System, MAS）的 1642 条执行轨迹（Execution Traces）得出，显示**故障率高达 41%–86.7%**。三大根因类别包括：**规范问题（Specification Problems）**（41.77%）——角色模糊、任务定义不清；**协调失败（Coordination Failures）**（36.94%）——通信中断、状态不同步；**验证缺口（Verification Gaps）**（21.30%）——缺乏验证、缺失质量检查。**群体思维（Groupthink）**系列研究（arXiv:2508.05687）进一步补充了：单一文化崩溃（Monoculture Collapse，使用相同基座模型导致故障高度相关）、从众偏差（Conformity Bias，智能体相互强化彼此的错误）、心智理论（Theory of Mind）缺失、混合动机动态（Mixed-Motive Dynamics）以及级联可靠性故障（Cascading Reliability Failures）。级联故障示例：重试风暴（Retry Storms），即支付失败触发订单重试，进而引发库存重试，最终压垮库存服务（数秒内负载激增 10 倍——需引入熔断器（Circuit Breakers））。记忆投毒（Memory Poisoning）：单个智能体的幻觉（Hallucination）进入共享内存，下游智能体将其当作事实处理；系统准确率逐渐衰减，导致根因诊断极为困难。**STRATUS**（NeurIPS 2025）报告指出，通过部署专用的检测、诊断与验证智能体，可将故障缓解成功率提升 1.5 倍。本课程将故障模式视为一等工程目标（First-Class Engineering Targets）。

**Type:** 学习
**Languages:** Python (stdlib)
**Prerequisites:** 第 16 阶段 · 13（共享内存（Shared Memory））、第 16 阶段 · 14（共识与拜占庭容错（Consensus and BFT））、第 16 阶段 · 15（投票与辩论拓扑（Voting and Debate Topology））
**Time:** 约 75 分钟

## 问题

多智能体系统（Multi-Agent System, MAS）在实际任务中的故障率高达 41%–86.7%（Cemri 等人于 2025 年对 7 个开源 MAS 进行了测量）。这类问题无法通过“单纯增加智能体数量”来调试。这些故障具有结构性成因。MAST 分类体系为你提供了明确的类别划分。本课程将把每个类别映射到具体的检测、诊断与缓解模式，从而让这些数字不再显得随意。

2026 年的生产实践要求将故障模式作为设计输入。只有当你能明确指出每个 MAST 类别并说出已部署的相应缓解措施时，你的架构才算“足够好”。

## 概念

### MAST 分类

**规范问题（占故障的 41.77%）**。智能体 (Agent) 的任务定义不够严密。示例：

- 角色模糊：两个智能体都认为自己担任评审员角色。
- 任务描述不充分：用户需要特定视角的总结，但指令仅为“总结这个”。
- 成功标准隐含：智能体无法判断任务是否成功完成。

缓解措施：
- 编写明确的角色契约。每个智能体的提示词 (Prompt) 需明确说明其职责范围*以及不负责的边界*。
- 为每个任务设定验收测试。在智能体启动前，明确定义“完成状态应呈现为 X”。
- 预检规范审查：在分发任务前，由独立的智能体审查任务定义。

**协调故障（占 36.94%）**。通信或状态同步中断。

示例：
- 两个智能体在未同步的情况下更新共享状态。
- 智能体间消息丢失（队列故障、超时）。
- 状态漂移 (State Drift)：智能体 A 认为任务已完成，而智能体 B 仍在执行。

缓解措施：
- 采用乐观并发控制 (Optimistic Concurrency Control) 的版本化共享状态。
- 关键消息需显式确认（重试直至收到确认回执）。
- 定期设置状态同步检查点；尽早检测漂移。

**验证缺口（占 21.30%）**。缺乏对输出结果的独立检查。

示例：
- 某个智能体声称成功，但无人验证。
- 智能体链中每个节点都盲目信任前序节点的输出。
- 缺乏对涌现组合行为的测试覆盖。

缓解措施：
- 独立验证智能体（见第 13 课）。具备只读权限与独立的数据源访问能力。
- 明确的交接契约：“A 的输出必须通过检查器 C 的验证后，B 才能开始执行。”
- 记录结果日志以供事后分析。

### 群体思维家族 (Groupthink Family) (arXiv:2508.05687)

当智能体趋于同质化或相互模仿时，会出现以下五类相关故障：

**单一文化崩溃 (Monoculture Collapse)**。使用相同的基础模型或训练数据 → 产生相关性错误。当三个智能体共享同一个大语言模型 (LLM) 时，它们也会共享该模型的幻觉 (Hallucination)。

**从众偏差 (Conformity Bias)**。智能体会倾向于调整自身输出以迎合声音最大或最自信的同伴，即使对方是错误的。

**心智理论 (Theory of Mind, ToM) 缺失**。智能体无法建模彼此的信念状态；导致协调机制崩溃（见第 18 课）。

**混合动机动态 (Mixed-Motive Dynamics)**。激励部分对齐的智能体会逐渐向折中方案靠拢，最终导致各方都不满意。

**级联可靠性故障 (Cascading Reliability Failure)**。一个组件的错误模式会触发依赖组件的连锁错误。

### 级联示例 —— 重试风暴 (Retry Storm)

典型的 2026 年事故模式：

payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load
   ↓
inventory service starts timing out
   ↓
every order retries inventory check
   ↓
inventory service sees 10x normal load
   ↓
cluster goes down

修复方案是经典的：**熔断器 (Circuit Breaker)**。当下游错误率超过阈值时，直接短路并返回缓存或默认结果。此外，为每个请求设置重试预算上限。

熔断器是少数无需修改即可直接从分布式系统 (Distributed System) 中借鉴用于多智能体故障缓解的机制之一。

### 记忆投毒 (Memory Poisoning)（回顾）

源自第 13 课：某个智能体的幻觉被当作共享内存中的事实；下游智能体基于该被污染的事实进行推理。用 MAST 的术语来说，这是共享内存层的验证缺口。

其症状是准确率逐渐衰减。系统不会直接崩溃，而是出现难以追溯根因的缓慢漂移。

缓解措施：仅追加日志 (Append-only Log)、数据溯源 (Provenance)、不可写入的验证器。已在第 13 课中详述。

### STRATUS —— 专用于故障检测的智能体

STRATUS（NeurIPS 2025）指出，部署以下组件可使缓解成功率提升 1.5 倍：

- **检测智能体**。监控症状模式（如高分歧率、重试激增、准确率漂移）。
- **诊断智能体**。根据症状，从 MAST 分类体系中推断可能的根本原因。
- **验证智能体**。在实施缓解措施后，检查症状是否消除。

这是将站点可靠性工程 (Site Reliability Engineering, SRE) 风格的事故响应机制应用于智能体系统。这三个角色均可由配备专用提示词的大语言模型智能体担任。

### 故障模式审计 (Failure-Mode Audit)

2026 年的一项最佳实践是每年（或每次重大版本发布时）进行一次故障模式审计：

1. **追踪采样**。收集约 1000 条真实执行轨迹 (Execution Trace)。
2. **分类**。针对每条轨迹中的故障，映射至 MAST 与群体思维分类体系。
3. **计算分类故障率**。哪些类别在你的系统中占主导地位？
4. **排序缓解措施**。哪种修复方案能消除最多的故障？
5. **选取 2-3 项缓解措施**。实施后，于下一季度重新审计。

建立审计纪律比具体选择何种措施更重要。若不进行审计，故障将淹没在噪声中，永远无法得到系统性解决。

### 系统静默故障 (Silent Failure)

最危险的故障类别是静默正确性故障。发生显式故障（崩溃、异常、告警）的系统易于监控。而输出看似合理实则错误的系统，无法通过异常日志被检测到。这就是为什么尽管验证缺口仅占故障总数的 21.30%，但其单次故障成本却是最高的。

建议投入以下方面：
- 基于抽样的人工审查。
- 黄金数据集 (Golden Dataset) 回归测试。
- 针对关键输出进行跨智能体交叉验证。

### 即时故障与缓慢故障 (Failure vs Slow Failure)

部分故障是即时发生的，部分则是缓慢演进的。即时故障（超时、模式不匹配、认证错误）的检测成本较低。缓慢故障（记忆投毒、单一文化漂移、角色模糊）的检测与预防成本则较高。

2026 年的工程实践方向：为缓慢故障部署代理指标 (Proxy Metric)，以便在漂移演变为可见错误前将其捕获。一致率、重试率、输出长度分布，以及连续智能体版本间的编辑距离 (Edit Distance) 都是有用的代理指标。

## 构建它

`code/main.py` 实现了：

- `FailureTaxonomy`（故障分类体系）—— 将模拟事件归类至 MAST（多智能体系统故障分类法）与 Groupthink（群体思维）类别。
- `CircuitBreaker`（熔断器）—— 经典设计模式；当错误率超过阈值时触发开启。
- `RetryStormSimulator`（重试风暴模拟器）—— 展示级联故障（cascading failure）；可切换熔断器的开启/关闭状态。
- `DetectionAgent`（检测智能体）—— 基于脚本的 STRATUS（症状追踪与响应框架）风格症状匹配器。

运行：

python3 code/main.py

预期输出：
- 无熔断器时的重试风暴：库存错误激增（模拟）。
- 启用熔断器：错误率被限制在阈值内；系统返回降级模式（degraded-mode）响应。
- 检测智能体标记该模式并指出对应的 MAST 分类。

## 使用它

`outputs/skill-mast-auditor.md` 对多智能体系统（multi-agent system）执行 MAST 风格的故障模式审计。流程为：追踪（Traces）→ 分类（categorization）→ 缓解措施排序（mitigation ranking）。

## 发布它

生产环境中的故障管理规范：

- **每季度执行一次 MAST 审计。** 而非每年一次。随着系统演进，故障类别也会发生变化。
- **全面部署熔断器。** 针对所有对外部依赖服务的调用。默认开启阈值设为 5-10% 的错误率。
- **黄金数据集（Golden datasets）。** 规模小、质量高、经人工审核。每周针对它们进行回归测试。
- **STRATUS 三件套。** 由检测（Detection）、诊断（Diagnosis）与验证（Validation）智能体共同监控生产环境。初期仅部署检测智能体；当症状信号存在较多噪声时，再加入诊断智能体。
- **故障预算（Failure budget）。** 为各类别故障率设定明确的服务等级目标（SLO）。超出预算将触发暂停发布（stop-shipping）的讨论。

## 练习

1. 运行 `code/main.py`。验证熔断器是否有效限制了重试风暴。调整故障阈值并观察其中的权衡关系。
2. 实现一个**慢速故障代理指标（slow-failure proxy）**：计算 3 个并行智能体之间的输出一致率。当该指标骤降时触发告警。通过逐渐增加智能体输出的相关性，模拟单一文化漂移（monoculture drift）。
3. 阅读 Cemri 等人的论文（arXiv:2503.13657）。选取其研究的 7 个多智能体系统（MAS）之一，映射出排名前 3 的故障类别。这些结果与 MAST 的预测有何异同？
4. 阅读群体思维（Groupthink）相关论文（arXiv:2508.05687）。找出五种模式中在生产环境中最难检测的一种。提出一个代理指标（proxy metric）。
5. 为你所熟悉的某个特定多智能体系统设计一套 STRATUS 风格的“检测-诊断-验证”三件套。检测模块关注哪些症状？诊断模块推荐哪些缓解措施？验证模块如何确认这些措施有效？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 多智能体系统分类法（MAST） | “2026 分类法” | Cemri 等人（2025）提出；包含 3 个根类别与 14 种故障子类型。 |
| 规范问题（Specification Problem） | “角色模糊” | 任务或角色定义不充分；智能体（Agent）不清楚具体该执行什么操作。 |
| 协调失败（Coordination Failure） | “状态漂移” | 智能体之间的通信或同步机制发生中断。 |
| 验证缺口（Verification Gap） | “无人复核” | 输出结果未经独立验证即被直接采纳。 |
| 群体思维家族（Groupthink Family） | “同质化故障” | 涵盖单一文化、从众效应、心智理论（Theory of Mind, ToM）缺失、混合动机及级联效应。 |
| 单一文化崩溃（Monoculture Collapse） | “同源模型，同源幻觉” | 因共享基础模型或训练数据而引发的关联性错误。 |
| 重试风暴（Retry Storm） | “级联错误放大” | 单次故障触发重试机制，进而导致下游系统负载被放大。 |
| 熔断器（Circuit Breaker） | “错误率超标时快速失败” | 当错误率超过设定阈值时触发熔断，并返回默认值进行短路处理。 |
| STRATUS（多智能体事件响应框架） | “事件响应三剑客” | 由检测、诊断与验证智能体组成。可将故障缓解成功率提升 1.5 倍。 |
| 记忆投毒（Memory Poisoning） | “幻觉传播” | 共享内存中的事实数据遭到污染；下游智能体基于被污染的数据进行推理。 |

## 延伸阅读

- [Cemri 等人 — 为什么多智能体大语言模型系统会失败？](https://arxiv.org/abs/2503.13657) — MAST 分类法，NeurIPS 2025
- [多智能体大语言模型中的群体思维故障](https://arxiv.org/abs/2508.05687) — 单一文化、从众效应及五大类故障分类体系
- [STRATUS — 面向多智能体系统（Multi-Agent System, MAS）事件响应的专用智能体](https://neurips.cc/) — NeurIPS 2025 会议论文收录（检测 + 诊断 + 验证）
- [Release It! — 稳定性模式（Nygard）](https://pragprog.com/titles/mnee2/release-it-second-edition/) — 熔断器模式的权威参考指南
- [Anthropic — 多智能体研究系统](https://www.anthropic.com/engineering/multi-agent-research-system) — 生产环境故障模式记录