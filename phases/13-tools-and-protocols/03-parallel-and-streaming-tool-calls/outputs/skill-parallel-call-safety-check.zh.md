---
name: 并行调用安全检查
description: 审计工具注册表 (tool registry) 以确保安全并行化 (parallelization)。标记每个工具的 `parallel_safe` 状态，注明排序依赖关系，并标记下游速率限制 (rate-limit) 风险。
version: 1.0.0
phase: 13
lesson: 03
tags: [并行工具调用, 流式传输, 关联分析, 速率限制]
---

给定一个工具注册表（包含工具名称、描述和执行器的列表），返回一个添加了 `parallel_safe: bool`、`ordering_deps: [tool_name]` 和 `rate_limit_group: name` 字段的带注释副本。

生成以下内容：

1. 逐工具分类。针对每个工具进行判定：在同一轮次 (turn) 内并行运行是否安全（纯读取操作、不同资源）；不安全（涉及状态变更、共享资源、外部速率限制）。
2. 依赖图。识别输出需作为另一工具输入的工具对。此类工具无法在同一轮次内并行化。使用 `ordering_deps` 进行标记。
3. 速率限制分组。调用同一下游 API 的工具应归入同一组。宿主应限制每组的并发 (concurrency) 上限，而非单个工具。
4. 安全建议。针对每个不安全的工具，明确指出应在该轮次禁用并行、加入队列，还是按资源进行分片 (shard)。
5. 供应商特定标志。当集合中包含任何不安全工具时，建议在 OpenAI 上设置 `parallel_tool_calls=false`，或在 Anthropic 上设置 `disable_parallel_tool_use=true`。

硬性拒绝条件：
- 审计后未进行分类的任何注册表。默认拒绝；未知即视为不安全。
- 任何在共享资源上执行写入路径且被标记为 `parallel_safe: true` 的工具。存在竞态条件 (race condition)。
- 任何调用受速率限制的外部 API 但未分配 `rate_limit_group` 的工具。

拒绝规则：
- 若要求在不进行检查的情况下将所有工具标记为并行安全，则予以拒绝。
- 若注册表包含针对同一资源的关键操作工具（如针对同一路径的 `delete_file` 和 `write_file`），则拒绝并行化，并指引至第 14 阶段 · 09 课进行沙箱级序列化 (sandbox-level serialization)。
- 若用户声称其工具绝不会发生竞态，则予以拒绝并要求提供证明（测试用例、日志或形式化论证）。竞态在生产环境中往往是静默发生的。

输出：一个修订后的注册表 JSON 数据块 (JSON blob)，其中每个工具包含上述三个新字段；随后附上一段简短摘要，指出风险最高的并行化选项及推荐的缓解措施。最后，为当前轮次建议一个 `tool_choice` 覆盖配置。