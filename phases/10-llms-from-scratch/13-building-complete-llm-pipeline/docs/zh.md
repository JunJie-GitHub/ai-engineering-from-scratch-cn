# 构建完整的大语言模型（LLM）流水线

> 第 01 至 12 课的内容分别对应流水线中的各个独立阶段。本课将作为支撑框架，把这些阶段串联成一次完整的端到端（end-to-end）运行流程：分词（tokenize）、预训练（pre-train）、扩展（scale）、监督微调（SFT）、对齐（align）、评估（evaluate）、量化（quantize）与服务部署（serve）。你无法在笔记本电脑上训练一个 700 亿参数的模型。你将实际构建的是编排层（orchestration layer）、运行清单（manifest）、评估门禁（eval gate）以及回滚计划（rollback plan）——这些正是 2026 年前沿团队用于决策模型是否发布的核心组件。本课是综合压轴项目（capstone）。

**类型：** 构建实践
**编程语言：** Python（标准库）
**前置要求：** 第 10 阶段第 01-12 课全部内容
**预计耗时：** 约 120 分钟

## 学习目标

- 将前十一课的内容（分词器、数据处理、预训练、扩展、监督微调、基于人类反馈的强化学习（RLHF）、直接偏好优化（DPO）、宪法式 AI（CAI）、评估、量化、推理）整合为一份单一且可复现的流水线规范（pipeline spec）
- 定义各阶段之间的工件契约（artifact contract）：明确每个阶段的输入消耗、输出产物，以及下一阶段如何验证输入数据
- 构建一个编排器（orchestrator），用于追踪实验、计算工件哈希值，并基于评估阈值（eval thresholds）控制发布决策
- 设计回滚计划（rollback plan）：明确哪些工件重新运行的成本较低，哪些成本较高，以及检查点（checkpoint）损坏所带来的代价

## 问题背景

此前的每一课都能独立运行。分词器已训练完毕。小型 GPT 已完成预训练。监督微调数据集已组装。奖励模型已训练。直接偏好优化已执行。评估指标已测量。量化权重已导出。推理服务器已启动。但每一课都只是一个独立的 Notebook。它们各自遵循不同的规范，拥有独立的输出路径和随机种子。

前沿模型的训练绝非几个 Notebook 就能搞定。Llama 3 405B 在约 54 天内消耗了 3000 万小时的 H100 算力。DeepSeek-V3 则使用了约 280 万小时的 H800 算力。在此期间，一个损坏的检查点、一次数据污染或一项评估指标的回退，都可能让团队损失数周的实际运行时间（wall-clock time）和长达一个月的 GPU 预算。团队能够应对这种挑战的关键在于严格的流水线卫生规范（pipeline hygiene）：每个阶段都必须具备确定性的输入、确定性的输出、运行清单、哈希校验以及发布门禁。

本课是综合压轴项目。你不需要在笔记本电脑上端到端地运行整个流水线。你将编写用于协调各阶段的编排器、描述运行过程的清单文件、控制发布决策的验证器，以及允许第三方仅凭单个文件即可复现你工作的重放计划（replay plan）。代码量虽小，但所需的工程纪律却极为严苛。

该架构模式在从 1 亿参数扩展到 1 万亿参数时保持不变。相同的四大组件——清单文件、编排器、评估门禁、工件存储（artifact store）——既能驱动 Llama 3 的训练，也能运行你的个人兴趣 GPT 项目。两者的区别仅在于各阶段配置中的数值规模，而非流水线本身的结构形态。

## 核心概念

### 十二个阶段 (The Twelve Stages)

Phase 10 的每一课都是一个阶段。以下是完整的依赖关系图。

graph TD
    S1["01 Tokenizer vocab"] --> S2["02 Trained tokenizer"]
    S2 --> S3["03 Sharded dataset"]
    S3 --> S4["04 Base model checkpoint"]
    S4 --> S5["05 Scaled training recipe"]
    S5 --> S6["06 SFT checkpoint"]
    S6 --> S7["07 Reward model + PPO policy"]
    S6 --> S8["08 DPO policy"]
    S7 --> S9["09 CAI / GRPO refined policy"]
    S8 --> S9
    S9 --> S10["10 Eval report"]
    S9 --> S11["11 Quantized weights"]
    S11 --> S12["12 Inference server"]
    S10 --> GATE["Ship gate"]
    S12 --> GATE

    style S1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style S4 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style S9 fill:#1a1a2e,stroke:#0f3460,color:#fff
    style GATE fill:#1a1a2e,stroke:#51cf66,color:#fff

阶段 07 和 08 可以并行运行。其余所有阶段均为强依赖。阶段 02（分词器/Tokenizer）的变更会使所有下游产物失效。阶段 10（评估/Eval）的变更仅会影响发布决策。

### 清单文件 (Manifest)

清单文件（Manifest）是一个单一文件，它完整描述了一次运行过程，足以支持完全复现。流水线产生的任何内容都不应依赖于清单之外的状态。这些字段虽然枯燥，但都是强制性的。

pipeline_version: 1.2.3
seed: 42
git_commit: a1b2c3d4
stages:
  01_tokenizer:
    recipe: bpe_32k
    input_hash: sha256:...
    output_hash: sha256:...
    wall_clock_sec: 3600
    cost_usd: 12

阶段 N 的输出哈希值即为阶段 N+1 的输入哈希值。任何偏差都会导致流水线暂停。这是早期发现数据损坏的方法。同时，这也是身处不同大洲的队友验证其复现结果是否与你生成相同产物的依据。

在实际操作中，团队会使用一个精简的 YAML 模式（Schema）配合清单检查器，将其与上一次成功运行的结果进行差异比对。任何超出预期字段（如成本、实际耗时/Wall Clock）的变动都会被视为危险信号。

### 产物类型化 (Artifact Typing)

每个阶段的输出都是一个类型化产物（Typed Artifact）。它不是目录数据块，也不是 Pickle 文件，而是具有已知模式（Schema）的命名类型。

| 阶段 | 产物类型 | 关键字段 |
|-------|--------------|-----------|
| 01-02 | 分词器 (Tokenizer) | vocab.json, merges.txt, config.json, hash |
| 03 | 数据集 (Dataset) | shards[], 行数, Token 数量, 去重统计 |
| 04-05 | 检查点 (Checkpoint) | weights.safetensors, config.json, 优化器状态, 步数 |
| 06 | SFT 模型 | 检查点 + SFT 配方 + 数据混合比例 |
| 07 | 奖励模型 (Reward Model) | RM 检查点 + 偏好数据哈希 |
| 08-09 | 策略模型 (Policy) | 检查点 + 参考模型哈希 + beta 值 + 已消耗 KL 预算 |
| 10 | 评估报告 (Eval Report) | 基准测试分数 + 回归差异 + 评估数据哈希 |
| 11 | 量化模型 (Quantized Model) | 量化权重 + 校准数据 + 相较于 FP16 的精度差异 |
| 12 | 服务器规格 (Server Spec) | 端点 + 模型哈希 + 配置 + 可观测性钩子 |

类型化机制避免了最常见的故障模式：例如将阶段 08 的输出误用作阶段 06 的输入，或将经过 DPO 训练的模型通过 SFT 路径发布。类型化的产物与阶段签名（Stage Signatures）能让这类错误在“编译期”就被拦截，而不是等到上线数日后才暴露。

### 评估门禁 (Eval Gate)

发布（Shipping）不等于“训练完成”。发布意味着“训练完成且通过了评估门禁”。该门禁在运行开始前就已定义好。

gates:
  mmlu:      >= baseline + 0.5   # no regression
  humaneval: >= baseline + 1.0
  truthfulqa: >= baseline         # no drop
  safety_refusal_rate: <= 0.05
  kl_from_reference: <= 25.0
  cost_total_usd: <= 50000

每个门禁都是一个数值阈值。没有“看起来不错”这种主观门禁，也没有主观审批。只有当所有门禁全部通过时，产物才会被标记为可发布。若任一门禁失败，运行将被挂起，等待指定评审员进行显式覆盖（Override），该操作本身也会记录在清单文件中。

两个门禁能拦截绝大多数灾难性情况。**回归门禁**（Regression Gate，新模型在核心基准测试上的表现必须至少与上一版持平）用于捕获训练缺陷。**KL 预算门禁**（KL Budget Gate，对齐后的策略模型与参考模型的偏离度不得超过 X）用于防止对齐过度（Alignment Overcooking）。每个生产级流水线都必须包含这两者。

### 编排器 (Orchestrator)

这是一段小型代码，负责读取清单文件、分发阶段任务、追踪产物，并在任何契约违规时立即暂停。它不是 Airflow，也不是 Kubeflow。为了保持流水线的整洁与可控，你最好自己写一个枯燥但可靠的工具。

编排器的职责非常明确：

1. 从清单文件中解析有向无环图（DAG）。
2. 针对每个阶段，检查预期输出是否已存在且哈希值正确（若是则跳过）。
3. 运行该阶段，捕获标准输出/标准错误（stdout/stderr），并记录实际耗时与成本。
4. 验证输出哈希值是否与下游阶段的预期输入哈希值匹配。
5. 若失败，则写入包含确切失败阶段的局部清单文件，并以非零状态码退出。

这大约只需要 200 行 Python 代码。在本课程中，它将类似于 `code/main.py` 文件。在底层，真实的流水线会使用 `torchrun` 或 `ray` 在集群上执行各个阶段，但编排器本身仅运行在单台机器上。

### 实验追踪与产物存储 (Experiment Tracking and Artifact Storage)

两个外部系统构成了流水线的基石。

**实验追踪器（Experiment Tracker，如 wandb、neptune、mlflow）**：按阶段记录损失曲线、评估指标和系统遥测数据。当你需要在三周后对比运行 A 和运行 B 时，追踪器就是你的去处。团队几乎总是使用托管型追踪器——自己开发会浪费本应用于模型训练的时间。

**产物存储（Artifact Store，如 S3、R2、GCS）**：用于存放检查点、数据集、分词器和评估报告的不可变对象存储。产物通过哈希值寻址，而非文件名。像 `latest.pt` 这样的文件名是隐患（Foot-gun）；而 `ckpt-7b-step-20000-sha256:abc123.safetensors` 则是一份契约。

编排器会同时向这两个系统写入数据。追踪器供人类查看图表，产物存储供下一阶段查找输入。

### 成本核算 (Costing)

前沿模型的每次运行都伴随着明确的美元成本。预算纪律体现在两个环节。

**运行前估算**：根据清单文件计算预期浮点运算次数（FLOPs）（预训练阶段公式为：6 × 参数量 × Token 数量）、预期 GPU 小时数（FLOPs / 峰值吞吐量 / 利用率），以及按当前租赁费率计算的美元成本。若估算值超出预算门禁，流水线将拒绝启动。

**运行中追踪**：各阶段的实际耗时与成本会记录到清单文件中。每个阶段结束后，都会检查剩余预算。若某阶段超支，下一阶段门禁将基于新的剩余预算进行评估。你绝不应该等到风投（VC）打电话来时才意识到资金耗尽。

Llama 3 的公开成本为 6100 万美元。DeepSeek-V3 主预训练运行的公开成本为 560 万美元。这一差距主要源于硬件效率与混合专家模型（Mixture-of-Experts, MoE）架构——但之所以能看清具体成本，是因为两个团队都是按阶段而非按整次运行来追踪成本的。

### 可复现性与确定性 (Reproducibility vs Determinism)

这两者并不相同。**可复现（Reproducible）**是指：相同的清单文件、相同的代码与相同的基础设施，能够生成具有等效下游指标的检查点。**确定性（Deterministic）**则指输出在比特级别完全一致。

现代大语言模型（LLM）训练是可复现的，但并非确定性的。分布式训练中的归约顺序（Reduce-order）、GPU 内核的非确定性（如 cuBLAS、flash-attn）以及混合精度舍入误差，共同导致不同运行间的浮点数在 1e-5 级别存在差异。这对最终指标没有影响，指标依然稳定。但如果你试图通过比特级差异进行调试，这将是致命的。解决方法是记录每个阶段的输入哈希、输出哈希和核心指标——只要这些匹配，即使权重并非比特级一致，该运行也被视为“已复现”。

graph LR
    M["Manifest v1.2.3"] --> O["Orchestrator"]
    O --> S["Stages 01 → 12"]
    S --> AS["Artifact Store\n(content-addressed)"]
    S --> ET["Experiment Tracker\n(metrics, curves)"]
    AS --> GATE["Eval Gate"]
    ET --> GATE
    GATE -->|pass| SHIP["Ship"]
    GATE -->|fail| ROLL["Rollback plan"]

    style M fill:#1a1a2e,stroke:#0f3460,color:#fff
    style GATE fill:#1a1a2e,stroke:#e94560,color:#fff
    style SHIP fill:#1a1a2e,stroke:#51cf66,color:#fff
    style ROLL fill:#1a1a2e,stroke:#c0392b,color:#fff

### 回滚计划 (Rollback Plan)

在运行开始前，需明确每个阶段失败时的应对措施。主要分为三类：

- **重跑成本低**（数小时）：分词器、评估、量化、推理服务器。直接重跑即可。
- **中等成本**（数天）：SFT、DPO、CAI。保留基础模型，仅重跑对齐阶段。
- **成本高昂**（数周及数百万美元）：预训练。此处的回滚计划不是“重跑”，而是“使用上一个良好的检查点，并基于修正后的数据重跑成本较低的下游阶段”。

由于阶段依赖关系已类型化并附带哈希值，编排器可以自动计算回滚范围：使失败阶段及其所有下游阶段失效。阶段 06（SFT）失败会使 06、07、08、09、10、11、12 全部失效。阶段 11（量化）失败仅会使 11 和 12 失效。提前明确这些规则，可以避免团队在凌晨 4 点精疲力竭时临时抱佛脚。

### 2026 年观察到的生产级配方 (Production Recipes Observed in 2026)

大多数前沿团队已收敛至相同的架构骨架。

- 分词器（Tokenizer）：128k BPE 并带字节回退（Byte Fallback）。在小型、均衡的多语言数据切片上训练。
- 预训练（Pre-training）：10-20T Token，主要为网页数据、代码及合成数据。使用 Muon 或 AdamW 优化器。采用 FSDP2 或 DeepSpeed ZeRO-3。启用梯度检查点（Gradient Checkpointing）。权重使用 BF16，主副本使用 FP32。
- 监督微调（SFT）：50万-200万指令对，混合人工与合成数据，并与评估集进行严格去重。
- 对齐（Alignment）：DPO 或 CAI + GRPO。仅在偏好信号过于多维、DPO 无法处理时才使用 RLHF。
- 评估（Eval）：MMLU-Pro、MATH、HumanEval+、GPQA、SWE-Bench Verified、LiveBench，外加一个公众永远无法看到的私有保留集（Held-out Set）。
- 量化（Quantization）：服务部署采用 4-bit GPTQ 或 AWQ；安全评估采用 8-bit（此时精度差异至关重要）。
- 服务部署（Serving）：vLLM、TensorRT-LLM 或自研方案。支持连续批处理（Continuous Batching）、投机解码（Speculative Decoding）与 KV 缓存淘汰（KV Cache Eviction）。

具体数值每六个月就会更新一次，但架构骨架始终不变。

## 构建

本课程的代码实现了一个编排器（orchestrator）和清单检查器（manifest checker），而非十二个独立的训练脚本。每个阶段均通过占位符进行模拟，该占位符会生成具有正确形状（shape）和哈希值（hash）的输出产物（artifact）。在将 GPU 算力成本投入到实际阶段之前，端到端运行该编排器可验证流水线（pipeline）的底层链路是否畅通。

完整实现请参见 `code/main.py`。核心组件包括：

- `Manifest` 数据类（dataclass）：流水线版本、随机种子（seed）、Git 提交记录、阶段（stages）、门控（gates）。
- `Stage` 数据类：名称、类型、输入（哈希值）、输出（哈希值）、实际耗时（wall clock）、成本。
- `Orchestrator.run()`：解析有向无环图（DAG）、分发阶段、验证哈希值、更新清单。
- `EvalGate.check()`：读取阈值、与最新评估报告进行对比、返回通过/失败结果。
- `ArtifactStore`（内存存根）：按哈希值进行存取，用于模拟 S3。
- `CostTracker`：记录单阶段及累计成本，超出上限时自动终止。

`main.py` 中的流水线会运行十二个占位阶段，生成一份清单，并触发一个失败的评估门控，以演示被挂起（held）的运行状态。将每个占位符替换为对应课程中的真实训练脚本，即可得到前沿流水线所使用的核心骨架。

## 使用

标准工作流包含三个命令。

python code/main.py plan    # validate manifest, compute cost estimate, print DAG
python code/main.py run     # execute stages, writing to manifest.out.yaml
python code/main.py gate    # read manifest.out.yaml, apply eval gates, ship-or-hold

每次务必先运行 `plan`。大多数流水线问题都会在规划阶段暴露出来——例如缺失门控阈值、哈希值过期或预算超支。运行 `plan` 是免费的，而运行 `run` 则成本高昂。在低成本阶段提前发现漏洞，能有效节省开支。

`gate` 的输出结果为 `SHIP` 或 `HOLD: <reason>`。被挂起的运行并非失败，而是一个决策节点。指定的审核人员可以选择强制覆盖（该操作会被记录日志），或者批准回滚。

## 发布

本课程会生成 `outputs/skill-llm-pipeline-reviewer.md` 文件。向其输入拟定的流水线清单，它将验证所有契约（contracts）：阶段类型、哈希链、门控、回滚计划及成本估算。如果清单缺失评估门控、KL 散度预算（KL budget）无上限，或运行过程中混合了评估数据与训练数据，它将拒绝批准。

## 练习

1. 扩展编排器（orchestrator），以支持阶段 07 与阶段 08 的并行执行。请使用标准库 `concurrent.futures` 模块。验证最终清单（manifest）是否完整记录了这两个阶段的输出，并确保阶段 09 的输入哈希（input hash）是由两者生成的确定性组合。

2. 添加“数据污染检查”门控（contamination check gate）。给定评估数据集（eval dataset）的哈希值与训练数据集分片（training dataset shards），计算两者的重叠部分（精确字符串匹配或 13-gram 匹配）。若重叠率超过 0.1%，则门控判定失败。向其输入受污染的训练集，验证该门控能否成功拦截运行。

3. 基于第一性原理实现成本估算器（cost estimator）。针对阶段 04（预训练），按 `6 x params x tokens` 估算浮点运算次数（FLOPs）。假设在 H100 上运行 BF16 精度时，理论峰值为 989 TFLOPs，模型浮点运算利用率（MFU）为 40%，GPU 成本为 2.50 美元/小时。请报告一个 7B 模型在 2T tokens 上训练的成本估算值，并与已发布的 Llama 2 数据进行对比。

4. 构建部分回滚（partial rollback）机制。模拟阶段 09（CAI）发生故障，随后重新运行阶段 09 至 12，同时保留阶段 01 至 08 的缓存。编排器应能通过哈希值识别已缓存的制品（artifacts）并跳过它们。测量相较于完整重跑所节省的实际耗时（wall-clock time）。

5. 增加可观测性（observability）。为每个阶段发射 OpenTelemetry 跨度（spans），并附带参数量、已处理词元数、损失值（loss）及成本等属性。将这些跨度数据管道传输至本地收集器。核心目的并非构建仪表盘，而是确保每个阶段的健康状态均可通过单一追踪 ID（trace ID）进行溯源。

## 关键术语

| 术语 | 常见叫法 | 技术定义 |
|------|----------------|----------------------|
| 清单（Manifest） | “配方文件” | 描述流水线版本、随机种子、各阶段配置及门控阈值的 YAML 或 JSON 文件——足以完整复现一次运行 |
| 内容寻址（Content-addressed） | “按哈希而非名称” | 制品以其内容的 SHA-256 哈希值进行存储，从而彻底避免版本 A 与版本 B 混淆 |
| 评估门控（Eval gate） | “发布标准” | 基准指标与安全评分的数值阈值，制品必须达标后方可标记为可发布状态 |
| KL 散度预算（KL budget） | “对齐偏离程度” | 对齐各阶段累积 KL(policy || reference) 的上限，作为门控强制执行 |
| 模型浮点运算利用率（MFU） | “GPU 实际利用率” | 模型浮点运算利用率（Model FLOPs Utilization）——实际达到的 FLOPs 除以理论峰值。70B 规模通常为 40%，7B 规模约为 55% |
| 回滚预案（Rollback plan） | “故障应对方案” | 针对各阶段故障预先制定的操作集：重跑、降级回退或使用修订后的输入重新训练 |
| 编排器（Orchestrator） | “指挥家” | 负责读取清单、调度各阶段、验证哈希值，并在任何契约违规时立即中止的进程 |
| 制品存储（Artifact store） | “带版本控制的权重 S3” | 不可变的内容寻址对象存储——作为检查点、数据集与评估报告的唯一可信源 |
| 可复现（Reproducible） | “重跑指标一致” | 底层权重比特位可能不同，但下游指标等效——这是分布式大语言模型训练的现实目标 |
| 成本门控（Cost gate） | “预算红线” | 运行前成本估算与运行中追踪器相结合——若估算值超出预算，流水线将拒绝启动 |

## 延伸阅读

- [Dubey 等人，2024 -- "The Llama 3 Herd of Models"](https://arxiv.org/abs/2407.21783) -- 对前沿大模型开发流水线（frontier pipeline）最详尽的公开描述，涵盖数据处理、模型训练、对齐（alignment）与评估（evaluation）
- [DeepSeek-AI，2024 -- "DeepSeek-V3 Technical Report"](https://arxiv.org/abs/2412.19437) -- 采用效率优先的流水线（efficiency-first pipeline），训练成本仅为 Llama 3 同量级模型的约十分之一
- [Kaplan 等人，2020 -- "Scaling Laws for Neural Language Models"](https://arxiv.org/abs/2001.08361) -- 首次提出算力、数据量与参数量之间的缩放定律（Scaling Laws）
- [Hoffmann 等人，2022 -- "Training Compute-Optimal Large Language Models (Chinchilla)"](https://arxiv.org/abs/2203.15556) -- 对 Kaplan 等人研究的修正，重新校准了现代大模型训练的数据预算配比
- [PyTorch FSDP2 文档](https://pytorch.org/docs/stable/fsdp.html) -- 在 PyTorch 2.4+ 中取代 FSDP1 的分布式训练原语（primitive）
- [Weights & Biases LLM Reports](https://wandb.ai/site/llms) -- 开源大语言模型（LLM）训练任务的真实运行清单（manifests）与实验追踪器（experiment tracker）输出，非常适合作为可直接复用的参考模板