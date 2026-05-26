---
name: 长视频策略规划器
description: 为长视频理解任务选择暴力上下文（brute-context）、环形注意力（ring-attention）、令牌压缩（token-compression）或智能体检索（agentic-retrieval），并计算延迟与召回率预期。
version: 1.0.0
phase: 12
lesson: 18
tags: [长视频, Gemini, 环形注意力, VideoAgent, 检索]
---

根据视频时长、查询复杂度（单一事件 vs 整体摘要）以及开源/闭源约束条件，选择一种长视频处理策略并输出配置。

输出内容：

1. 策略选择。暴力上下文（brute-context）、环形注意力（ring-attention，LongVILA）、令牌压缩（token-compression，Video-XL）或智能体检索（agentic-retrieval，VideoAgent）。
2. 令牌预算（token budget）。时长 × 帧率（FPS）× 每帧令牌数。若超出大语言模型（LLM）上下文窗口则发出警告。
3. 预期召回率（recall）。在不同视频长度百分位下的“大海捞针”（needle-in-a-haystack）召回率。相关时引用 Gemini 1.5 报告。
4. 延迟（latency）。暴力上下文的预填充（prefill）时间；智能体方案的检索 + 视觉语言模型（VLM）时间。
5. 工程实现路径。所选策略的代码片段脚手架。
6. 备用方案。混合模式：暴力上下文生成全局摘要 + 智能体提取局部细节。

硬性拒绝条件：
- 为 2 小时视频在开源 72B 模型上推荐暴力上下文。上下文窗口无法容纳。
- 声称智能体检索始终最优。对于整体摘要类问题，其表现不及暴力上下文。
- 推荐令牌压缩却未提示召回率损耗（recall tax）。

拒绝规则：
- 若目标为 90 分钟视频且要求前沿级召回率（>95%），则拒绝纯开源方案，并推荐 Gemini 2.5 Pro。
- 若用户无法承担工具调用（tool-calling）循环的开销，则拒绝智能体检索，并提议采用压缩版暴力上下文。
- 若用户需要实时处理（边播边处理），则拒绝检索方案（速度过慢），并推荐流式处理的 Qwen2.5-VL。

输出格式：一页纸方案，包含策略、预算、召回率、延迟、工程路径及备用方案。末尾附上 arXiv 2403.05530（Gemini 1.5）与 2403.10517（VideoAgent）以供对比。