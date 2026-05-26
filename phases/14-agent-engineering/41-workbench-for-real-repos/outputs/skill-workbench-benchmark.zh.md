---
name: 工作台基准测试
description: 在项目的自有示例应用上，通过仅提示（prompt-only）和工作台引导（workbench-guided）流水线运行相同任务，并生成包含五种结果的“前后对比”报告。
version: 1.0.0
phase: 14
lesson: 41
tags: [基准测试, 前后对比, 评估, 工作台, 示例应用]
---

给定一个代码仓库、一个智能体（Agent）产品以及一个小型示例应用，构建一个可移植的评估工具（evaluation harness），用于对比仅提示（prompt-only）流水线与工作台引导（workbench-guided）流水线。

产出内容：

1. `eval/sample_app/` — 一个源自项目领域的最小可行示例应用（minimum-viable sample app）。
2. `eval/run_prompt_only.py` 与 `eval/run_workbench.py`，两者均接收任务描述并返回 `TaskOutcome`。
3. `eval/report.py`，用于运行两条流水线并生成 `before-after-report.md` 及 `comparison.json`。
4. CI 工作流（CI workflow），当工作台流水线在固定任务集上的结果出现退化（regression）时触发失败。
5. `docs/benchmark.md`，用于解释五种结果类型以及何种情况被视为退化。

硬性拒绝条件：

- 仅包含单条流水线的基准测试。对比才是核心目的。
- 缺乏分母的百分比形式结果。必须始终报告 `n / m`。
- 智能体产品训练时已见过的示例应用。请使用针对领域定制的测试夹具（fixture）。
- 掩盖假阴性（false negative）的报告。必须逐一列出仅提示流水线更快的任务。

拒绝规则：

- 若项目缺乏验收命令（acceptance command），则拒绝交付该基准测试。因为无从度量。
- 若工作台流水线在中等难度任务上的耗时超过仅提示流水线的 3 倍，需明确暴露该发现；此时需要简化的是工作台，而非模型。
- 若评估工具无法离线运行，则拒绝将其接入 CI。网络波动会破坏对比结果的准确性。

输出结构：

<repo>/
├── eval/
│   ├── sample_app/
│   ├── run_prompt_only.py
│   ├── run_workbench.py
│   └── report.py
├── outputs/eval/
│   ├── before-after-report.md
│   └── comparison.json
├── docs/benchmark.md
└── .github/workflows/benchmark.yml

结尾附上“下一步阅读”指引，指向：

- 第 42 课：包含工作台流水线所用所有交互界面（surface）的综合项目包（capstone pack）。
- 第 19 课（SWE-bench、GAIA、AgentBench）：本测试所补充的宏观基准测试（macro benchmarks）。
- 第 30 课（评估驱动的智能体开发 / Eval-Driven Agent Development）：基准测试接入后用于持续评估循环（eval loops）的指南。