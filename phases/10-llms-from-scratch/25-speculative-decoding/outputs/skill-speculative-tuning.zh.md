---
name: speculative-tuning
description: 分析解码工作负载，并为推测解码（speculative decoding）选择草稿模型（draft model）、草稿长度 K（draft length K）、温度门限（temperature gate）和回退策略（fallback policy）。
version: 1.0.0
phase: 10
lesson: 25
tags: [推测解码, 草稿模型, 接受率, 吞吐量, 推理, 解码延迟]
---

给定目标模型（规模、系列、分词器（tokenizer））、工作负载遥测数据（telemetry）（任务混合比例、提示词与解码词元比例（prompt-vs-decode token ratio）、p50/p99 解码延迟、加速器与高带宽内存（HBM）余量（HBM headroom）、平均批次大小（batch size）、采样温度分布）以及可用的草稿检查点（checkpoints），输出以下内容：

1. **草稿模型选择（Draft choice）**。从同系列小型模型（如 Llama-70B 对应 Llama-3.2-1B）、蒸馏草稿模型（如 Qwen3-0.6B-spec）、附加在目标模型上的 Medusa 头（Medusa heads），或“不使用推测解码”中进行选择（前提是没有草稿模型的浮点运算（FLOP）成本比率（FLOP cost ratio）低于 30%）。逐字节确认分词器与目标模型完全匹配；拒绝不匹配的分词器。
2. **草稿长度 K（Draft length K）**。求 E[tokens] / (1 + K x c) 的最大值（Argmax），其中 c 为草稿模型与目标模型的成本比率。使用在 5_000 个同分布数据（in-distribution data）词元上校准运行所测得的接受率（alpha），展示 K 取 2、3、4、5、6 时的计算过程。默认值：对话任务 K=4，代码任务 K=6，高温创意写作任务 K=2。
3. **温度门限（Temperature gate）**。设定一个温度阈值，超过该阈值则禁用推测解码。默认值为 0.8；若校准显示接受率（alpha）更早崩溃，则降至 0.6。拒绝任何依赖单次请求检查且增加延迟超过 50 微秒的温度门限机制。
4. **树预算（Tree budget）**。若服务栈（serving stack）支持树状草稿生成（tree drafting），则对于批次大小小于 8 的情况选择小型固定树（深度 2，分支 3-2）；对于批次大小大于 32 的情况使用扁平链（flat chain）。说明验证器（verifier）的键值（KV）暂存区（KV scratch）大小（以字节为单位），并确认其适配高带宽内存（HBM）余量。
5. **回退策略（Fallback policy）**。明确指标名称（基于最近 1_000 次验证的滑动窗口（sliding-window）实测接受率 alpha）及阈值（alpha 低于 0.4），当达到该阈值时，服务器将针对该请求流回退至普通自回归解码（autoregressive decode）。需包含回退决策在单次请求中的生命周期。

当批次大小超过验证器处于计算瓶颈（compute-bound）的临界点时，拒绝使用推测解码。超过该点后，推测器（speculator）本应利用的闲置浮点运算（FLOP）将不复存在，导致吞吐量下降。对于任何实测接受率（alpha）低于 0.4 的任务族，拒绝使用推测解码；此时草稿开销将占据主导，导致实际延迟（wall-clock latency）恶化。拒绝未在目标模型上针对预留的 1_000 词元样本进行验证的草稿模型：未经验证的草稿模型会导致隐式的 KL 散度漂移（KL drift）。

示例输入：“在 8xH100 上运行 Llama-3.3-70B，对话工作负载，批次大小 16，p50 解码延迟 28 毫秒，p99 解码延迟 60 毫秒，温度分布均值 0.4 / 最大值 1.2，校准显示对话任务接受率（alpha）为 0.78，代码任务为 0.61。”

示例输出：
- 草稿模型：Llama-3.2-1B-Instruct-spec。分词器相同，系列相同，成本比率 c 约为 0.03。
- K：4。E[tokens/verify] = 3.4（对话），2.5（代码）。K=5 在对话任务中仅增加 0.1 个词元，但需额外付出 0.03 的 c 成本；故拒绝。
- 温度门限：0.8。在 0.8 以上时，校准集上的接受率（alpha）降至 0.45 以下。
- 树预算：深度 2，分支 (3, 2)。批次大小 16 时 KV 暂存区为 480 MB，符合容量要求。
- 回退策略：基于最近 1_000 次验证的滑动窗口接受率（alpha）若低于 0.40，则针对该请求流禁用推测解码 30 秒，随后重新探测。