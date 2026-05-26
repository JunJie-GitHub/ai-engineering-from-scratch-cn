# 大语言模型（LLM）的影子流量（Shadow Traffic）、金丝雀发布（Canary Rollout）与渐进式部署（Progressive Deployment）

> 大语言模型的发布（Rollout）汇集了软件部署中最棘手的难题：缺乏单元测试（Unit Tests）、故障模式（Failure Modes）分散、反馈信号延迟。标准流程为：（1）影子模式（Shadow Mode）——将生产环境（Production）请求复制一份发送给候选模型，记录日志并与原模型对比，对用户零影响；能捕捉明显的数据分布问题（Distribution Issues），但无法作为质量保障；（2）金丝雀发布——按 10% → 25% → 50% → 75% → 100% 的比例逐步切换流量，每一步设置检查点（Gates）；需监控延迟分位数（Latency Percentiles）、单次请求成本（Cost/Request）、错误/拒绝率（Error/Refusal Rate）、输出长度分布（Output Length Distribution）及用户反馈率（User-Feedback Rate）；（3）在稳定性确认后，针对明显不同的替代方案进行 A/B 测试（A/B Testing）。非确定性（Non-determinism）无法消除——由于 GPU 浮点运算非结合性（GPU FP Non-associativity）及批次大小方差（Batch-size Variance），相同输入在不同运行间的准确率波动可达 15%。成本是变量而非常量——性能提升 20% 的模型，单次调用成本可能高出 3 倍。回滚（Rollback）速度至关重要：如果回滚需要重新部署（Redeploy），说明你的流程太慢了。路由策略（Policy）应保存在配置/功能开关（Config/Flags）中；模型应存放在模型注册表（Registry）中并固定摘要（Pinned Digests）；回滚 = 秒级切换策略 + 恢复阈值 + 锁定旧模型。

**Type:** 学习
**Languages:** Python（标准库，简易金丝雀进度模拟器）
**Prerequisites:** 第 17 阶段 · 13（可观测性（Observability）），第 17 阶段 · 21（A/B 测试）
**Time:** 约 60 分钟

## 学习目标

- 区分影子模式（零影响对比）、金丝雀发布（线上流量渐进切换）与 A/B 测试（稳定性确认后的对比）。
- 列举五项针对大语言模型的金丝雀监控指标（延迟、单次请求成本、错误/拒绝率、输出长度分布、用户反馈）。
- 解释为何大语言模型的非确定性（波动可达 15%）会改变发布过程中对“稳定”的定义。
- 设计一条秒级回滚路径（通过策略切换），而非耗时数小时（重新部署）。

## 问题背景

你上线了一个新模型。离线评估（Offline Evals）显示准确率提升了 3%。你在生产环境中将其启用。24 小时内，成本飙升 40%，用户“踩”（Thumbs-down）比例上升 8%，三张客户工单反馈“回答很奇怪”。你决定回滚。重新部署耗时 3 小时。你的周末彻底毁了。

上述所有问题本都可以避免。影子模式能在任何用户察觉前就捕捉到 40% 的成本飙升。金丝雀发布会在“踩”比例上升时，于 10% 流量阶段自动拦截。基于策略开关的回滚只需 30 秒。正是这种工程规范，填补了“离线评估表现良好”与“真实用户满意”之间的鸿沟。

## 核心概念

### 影子模式（Shadow Mode）

候选模型接收与生产环境相同的请求；输出仅记录日志，不返回给用户。对用户零影响。记录内容：

- 输出内容（与生产环境的差异对比）。
- Token 数量（成本差异）。
- 延迟（Latency）。
- 拒绝响应与错误。

可捕获：成本激增、输出长度退化、明显的拒绝策略变更、硬性错误。无法捕获：用户可感知的质量差异。影子模式仅用于冒烟测试（Smoke Test），而非质量测试。

### 金丝雀发布（Canary Rollout）

带有关卡控制的渐进式流量切换。典型进度：1% → 10% → 25% → 50% → 75% → 100%。在每个阶段基于 5 项指标进行关卡控制：

1. **延迟百分位数（Latency Percentiles）** — P50、P95、P99。触发阈值：金丝雀版本的 P99 延迟超过基线的 1.5 倍。
2. **单次请求成本（Cost per Request）** — 综合美元成本。触发阈值：高于基线 20% 以上。
3. **错误/拒绝率（Error/Refusal Rate）** — 5xx 状态码及显式拒绝响应。触发阈值：达到基线的 2 倍。
4. **输出长度分布（Output Length Distribution）** — 均值与 P99。触发阈值：分布发生显著偏移。
5. **用户反馈率（User-Feedback Rate）** — 点踩数/工单提交数。触发阈值：达到基线的 1.5 倍。

### 非确定性是新的方差来源

相同的输入会产生不同的输出。原因包括：

- GPU 浮点运算非结合性（GPU FP Non-associativity）（浮点归约顺序随批次变化）。
- 批次大小差异（Batch-size Variance）（相同提示词在 128 批次与 16 批次中的处理差异）。
- 采样（Sampling）（温度参数 temperature > 0）。

实测数据：在相同的评估集上，多次运行间的准确率波动最高可达 15%。发布过程中的“稳定”意味着指标处于预期方差范围内，而非与基线完全一致。应将关卡阈值设置在噪声基底（Noise Floor）之上。

### 成本是动态变量

性能提升 20% 的模型，单次调用成本可能高出 3 倍。单次请求成本是五大关卡指标之一。如果发布一个“更好”的模型却破坏了单位经济效益（Unit Economics），则必须执行回滚。

### 回滚是核心武器

- 策略开关（Policy Flag，基于功能开关系统）：在配置中切换流量百分比；耗时仅需数秒。
- 模型锁定（Model Pinning，基于注册表摘要 Registry Digest）：锁定的模型不会自动升级。
- 回滚操作 = 恢复开关配置 + 将锁定摘要回退至上一版本。耗时数秒，而非数小时。

如果你的技术栈需要重新部署才能回滚，请在发布前修复此问题。

### 工具链

**Argo Rollouts** / **Flagger** — Kubernetes 渐进式交付控制器。可与 Istio/Linkerd 的加权路由集成。

**Istio 加权路由（Istio Weighted Routing）** — 服务网格（Service Mesh）层级的流量拆分。

**KServe / Seldon Core** — 内置金丝雀发布功能的模型服务框架。

**功能开关（Feature Flags）** — LaunchDarkly、Flagsmith、Unleash。策略级切换，无需重新部署。

### 指标监控频率（Metrics Cadence）

金丝雀关卡检查频率根据流量大小设定为每 5-15 分钟一次。1% 的流量配合 10 req/min 的请求速率，每个时间窗口可产生 50-150 个数据点——足以评估延迟，但用于用户反馈则噪声较大。10% 的流量可提供约 10 倍的数据量。流量推进应在每个阶段暂停足够长的时间，以积累充足的样本。

### A/B 测试步骤为可选

如果新模型存在显著差异（行为不同、成本曲线不同、语气风格不同），在通过金丝雀发布后，可将其置于 50% 流量进行 A/B 测试（A/B Testing）。如果只是迭代优化版本，则在通过金丝雀关卡后直接切换至 100% 流量。

### 关键数据备忘

- 金丝雀发布进度：1% → 10% → 25% → 50% → 75% → 100%。
- 非确定性上限（Non-determinism Ceiling）：相同输入下多次运行的方差最高可达 15%。
- 五大金丝雀指标：延迟、成本、错误/拒绝率、输出长度、用户反馈。
- 成本关卡：高于基线 20% 即触发阈值。
- 回滚耗时：数秒，而非数小时。

## 实践操作

`code/main.py` 模拟了注入性能回退（regression）的灰度发布（canary rollout）流程。该脚本会报告发布流程在哪个阶段暂停，以及具体是哪个检查门限（gate）触发了拦截。

## 发布上线

本课时将生成 `outputs/skill-rollout-runbook.md`。根据候选模型（candidate model）、基线模型（baseline）和风险容忍度（risk tolerance），设计从影子模式（shadow mode）到灰度发布再到全量（100%）的发布计划。

## 练习题

1. 运行 `code/main.py`。注入 25% 的成本回退。灰度发布会在哪个阶段暂停？
2. 你的新模型在离线评估中准确率提升了 3%，但单次请求成本增加了 18%。是否应该发布上线（ship）？这取决于策略（policy）——请分别写出两种决策路径。
3. 设计一个端到端耗时低于 60 秒的回滚（rollback）方案。列出所需的基础设施。
4. 你的评估结果显示出 ±7% 的非确定性（non-determinism）波动。设置灰度检查门限，以避免误报（false alarm）。你会使用什么阈值系数（multipliers）？
5. 影子模式在灰度发布前捕获到了 40% 的成本激增（cost spike）。编写在影子模式下触发的告警规则（alert rule）。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 影子模式（Shadow mode） | “复制到新环境” | 零影响地将请求发送至候选模型以进行日志记录 |
| 灰度发布（Canary） | “渐进式流量” | 带有检查门限的、逐步向真实用户开放的发布流程 |
| 检查门限（Gates） | “发布检查” | 用于拦截流程推进的指标阈值 |
| 非确定性（Non-determinism） | “大语言模型波动” | 无法消除的多次运行之间的固有差异 |
| 策略开关（Policy flag） | “开关翻转回滚” | 配置级别的回滚机制，耗时以秒计而非小时 |
| 模型锁定（Model pin） | “注册表摘要” | 指向特定模型版本的不可变引用 |
| Argo Rollouts | “K8s 渐进式发布” | 原生支持 Kubernetes 的灰度发布与回滚控制器 |
| KServe | “K8s 推理服务” | 内置灰度发布原语的模型服务框架 |
| Istio 权重路由（Istio weighted） | “网格分流” | 服务网格流量拆分组件 |

## 延伸阅读

- [TianPan — 在不破坏生产环境的情况下发布 AI 功能](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — 安全部署机器学习模型](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — 高级大语言模型部署模式](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts 官方文档](https://argo-rollouts.readthedocs.io/)
- [Flagger 官方文档](https://docs.flagger.app/)