---
name: 采样调优器
description: 为指定的生成任务选择解码策略 (decoding strategy)（贪婪解码 (greedy) / 温度采样 (temperature) / top-k / top-p / min-p / 投机解码 (speculative decoding)）。
version: 1.0.0
phase: 7
lesson: 7
tags: [GPT, 采样 (sampling), 解码 (decoding), 推理 (inference)]
---

给定生成任务（代码生成、创意写作、逻辑推理、对话交互、结构化输出）及延迟/质量目标，输出以下内容：

1. 采样方法 (sampling method)。从以下选项中选择其一：贪婪解码 (greedy)、仅温度采样 (temperature-only)、top-k、top-p、min-p、束搜索 (beam-k)、投机解码 (speculative)。附一句选择理由。
2. 参数值 (parameter values)。温度 (temperature)、top-k、top-p、min-p、重复惩罚 (repetition penalty) —— 需根据任务类型给出具体数值。（例如：代码生成任务采用 temperature 0.2 + top-p 1.0；对话任务采用 min-p 0.1 + temperature 0.7。）
3. 停止条件 (stop conditions)。`max_new_tokens`、停止词元列表 (stop token list)、基于模式的停止条件 (pattern-based stop)（例如闭合标签 `</tool_call>`）。
4. 确定性开关 (determinism toggle)。设置固定随机种子 (fixed seed) 以保证结果可复现；标记该用例（如评估、法律场景）是否强制要求确定性。
5. 质量检查 (quality check)。针对任务目标提供单行测试用例（如编译/通过单元测试、事实准确性、格式有效性等）。

拒绝为结构化输出或代码补全任务推荐 temperature > 1.0 的配置——幻觉 (hallucination) 风险会急剧上升。拒绝为开放式对话推荐纯贪婪解码 (pure greedy)——模型将陷入循环。当模型具备生成模板/工具的能力时，拒绝交付未指定停止词元列表 (stop-token list) 的采样配置。