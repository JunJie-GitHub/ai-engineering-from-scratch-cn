---
name: 反馈运行器
description: 封装 Shell 命令，以确定性方式捕获标准输出/标准错误/退出码/耗时，为每条命令持久化一条 JSONL（JSON Lines）记录，并在缺少反馈时拒绝推进智能体循环（agent loop）。
version: 1.0.0
phase: 14
lesson: 37
tags: [反馈, 子进程, 运行器, JSONL, 循环控制]
---

针对在智能体循环（agent loop）中执行 Shell 命令的项目，请生成一个反馈运行器（feedback runner）及其写入的 JSONL 文件。

需产出以下内容：

1. `tools/run_with_feedback.py`，对外暴露 `run_with_feedback(command: list[str], agent_note: str, timeout_s: float) -> FeedbackRecord` 函数。
2. 位于工作台（workbench）目录下的 `feedback_record.jsonl` 文件，每行存储一条记录。
3. `tools/feedback_loader.py`，用于返回当前活跃任务最近的 N 条记录。
4. 一个 `loop_can_advance(record) -> bool` 辅助函数，供智能体循环在宣告成功前调用。
5. 测试用例需覆盖：成功路径（success path）、非零退出码（non-zero exit）、超时（timeout）、二进制文件缺失（missing binary）、确定性首尾截断（deterministic head/tail truncation）。

硬性拒绝条件（Hard rejects）：

- 运行器中任何位置均不得使用 `shell=True`。仅限使用参数列表（argv-only）。
- 截断逻辑不得依赖系统时钟（wall clock）或随机采样（random sampling）。相同输入必须生成相同的记录。
- 记录中不得缺失 `duration_ms` 字段。响应缓慢的探针（probe）通常是工作台卡死（wedged workbench）的首要征兆。
- 加载器不得返回无界列表（unbounded list）。需限制为最近 N 条或采用分页机制（paginate）。

拒绝规则（Refusal rules）：

- 若项目通过标准输出（stdout）传递敏感信息（secrets），在未加入脱敏（redaction）步骤前，拒绝交付该运行器。需明确展示原本会被捕获的行。
- 若项目包含可能无限期挂起（hang indefinitely）的命令，在未设置默认超时（default timeout）及显式覆盖列表（explicit override list）前，拒绝交付。
- 若运行器在共享状态（shared state）的 Worker 进程内执行，在向 JSONL 追加记录时不得省略文件锁（file lock）。多写入者并发操作会导致文件损坏。

输出结构：

<repo>/
├── feedback_record.jsonl
└── tools/
    ├── run_with_feedback.py
    ├── feedback_loader.py
    └── test_feedback_runner.py

文末需附上“下一步阅读”指引，指向：

- 第 38 课：讲解消费这些记录的验证关卡（verification gate）。
- 第 39 课：讲解在评估运行结果时读取反馈的评审智能体（reviewer agent）。
- 第 23 课：讲解在反馈机制稳固后，可添加至遥测（telemetry）侧的 OTel GenAI 规范（OTel GenAI conventions）。