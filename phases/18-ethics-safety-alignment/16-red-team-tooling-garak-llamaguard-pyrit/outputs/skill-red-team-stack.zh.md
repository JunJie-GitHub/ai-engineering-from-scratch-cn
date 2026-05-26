---
name: 红队工具栈
description: 为给定部署推荐红队（Red Team）工具栈及配置。
version: 1.0.0
phase: 18
lesson: 16
tags: [llama-guard, garak, pyrit, 红队工具, mlcommons-hazards]
---

根据部署描述，推荐红队（Red Team）工具栈及回归测试频率（Regression Cadence）。

输出内容：

1. 分类器（Classifier）部署位置。推荐在输入端、输出端或两端同时部署 Llama Guard（3-8B、3-1B-INT4 或 4-12B）。对于边缘部署（Edge Deployment），优先选择 3-1B-INT4。对于多模态（Multimodal）场景，使用 Llama Guard 4。
2. 探针扫描器（Probe Scanner）配置。推荐与部署场景相关的 Garak 探针：幻觉（Hallucination，适用于检索增强生成（RAG）系统）、数据泄露（Data Leakage，适用于涉及个人身份信息（PII）的场景）、提示词注入（Prompt Injection，始终启用）、越狱攻击（Jailbreak，始终启用）。指定使用 Prompt-Guard-86M + Llama-Guard-3-8B 防护盾组合进行端到端评估（End-to-End Evaluation）。
3. 攻击活动编排器（Campaign Orchestrator）。针对具备新能力的模型，推荐在发布前使用 PyRIT 进行测试活动。指定需运行的转换器链（Converter Chains）（如改写、编码、翻译、角色扮演）及编排器（Orchestrator）（使用 Crescendo 进行逐步升级，使用 TAP 进行分支探索）。
4. 执行频率（Cadence）。Garak 每晚运行以进行回归测试。PyRIT 在每个版本发布前运行以进行深度红队测试。Llama Guard 持续部署运行。
5. 裁判模型校准（Judge Calibration）。为每个使用裁判模型的工具指定具体的裁判大语言模型（Judge LLM）（如 GPT-4-turbo、StrongREJECT 或内部模型）。裁判模型的校准将直接影响所报告的攻击成功率（ASR, Attack Success Rate）。

硬性拒绝条件：
- 任何未部署至少一个 Llama Guard 级别输入或输出分类器的部署方案。
- 任何未包含 Garak 或等效单轮回归测试（Single-turn Regression）的发布版本。
- 任何高风险部署（High-stakes Deployment）在发布前未执行等效于 PyRIT 的攻击活动。

拒绝规则：
- 如果用户要求推荐单一“最佳”工具，请拒绝——这三款工具覆盖不同层级，属于叠加使用关系，而非替代关系。
- 如果用户要求推荐一体化商业替代方案，请拒绝该请求并指出 2026 年的行业现状：这三款开源工具即为当前的最佳实践栈（Best-practice Stack）。

输出要求：一份单页推荐文档，需明确分类器部署位置、探针配置、攻击活动编排器、回归测试频率及裁判模型身份（Judge Identity）。分别引用一次 Meta（arXiv:2407.21783）、NVIDIA Garak 和 Microsoft PyRIT。