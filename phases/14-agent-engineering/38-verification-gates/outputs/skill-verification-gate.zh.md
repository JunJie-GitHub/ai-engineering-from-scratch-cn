---
name: 验证门禁（verification-gate）
description: 生成一个确定性验证门禁（deterministic verification gate），将范围、规则与反馈工件（artifacts）整合为每个任务单一的 verification_report.json，并配置持续集成（CI）流水线，在未获得通过结论时拒绝合并。
version: 1.0.0
phase: 14
lesson: 38
tags: [验证, 门禁, 确定性, 持续集成, 覆盖日志]
---

根据项目的验收标准及现有工作台工件，生成验证门禁与覆盖审计日志（override audit log）。

产出内容：

1. `tools/verify_agent.py`，对外暴露 `verify(task_id, artifacts) -> VerdictReport` 接口。该函数为纯函数（pure function），具备确定性，且不调用大语言模型（LLM）。
2. `outputs/verification/<task_id>.json`，作为唯一可信来源（single source of truth）的判定结果文件。
3. `tools/override.py`，用于将已签名的覆盖条目（signed override entries）追加至 `outputs/verification/overrides.jsonl`（必须包含原因、用户 ID、时间戳及发现代码）。
4. 持续集成（CI）工作流，在 `passed: false` 时触发失败，并以内联方式展示报告。
5. `docs/verification.md`，列出每一项检查、其严重等级、来源工件及覆盖策略。

硬性拒绝条件：

- 调用大语言模型（LLM）的检查项。该门禁属于确定性底层逻辑（deterministic plumbing）；大语言模型的判断应交由人工评审员负责。
- 智能体（agent）可在无签名条目的情况下执行的覆盖路径。覆盖操作仅限人工执行。
- 遗漏所消耗工件路径的验证报告。报告必须具备可审计性。
- 工作流可静默降级的阻塞级（block-severity）发现项。严重等级在写入时即已固定，而非在读取时决定。

拒绝规则：

- 若项目缺乏验收命令，则拒绝交付该门禁，直至其存在。无法证明任何内容的门禁只是形式主义。
- 若规则报告不存在，则拒绝跳过规则检查；采取故障阻断（fail closed）策略。
- 若反馈日志不存在，则拒绝跳过验收检查；日志缺失本身即构成阻塞项。
- 若覆盖条目未纳入版本控制，则拒绝接入覆盖路径；非正式记录的覆盖操作将使门禁形同虚设。

输出结构：

<repo>/
├── tools/
│   ├── verify_agent.py
│   └── override.py
├── outputs/verification/
│   ├── overrides.jsonl
│   └── <task_id>.json
├── docs/verification.md
└── .github/workflows/verify.yml

结尾附上“下一步阅读”指引，指向：

- 第 39 课：介绍在获得通过结论后接手的评审智能体（reviewer agent）。
- 第 40 课：介绍将判定结果包含在数据包中的交接生成器（handoff generator）。
- 第 41 课：介绍针对真实风格示例应用运行该门禁的实践。