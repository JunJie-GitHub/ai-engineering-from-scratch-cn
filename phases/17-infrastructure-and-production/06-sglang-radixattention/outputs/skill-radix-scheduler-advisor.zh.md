---
name: radix-scheduler-advisor
description: 针对希望复用 RadixAttention 缓存的前缀密集型工作负载，提供 SGLang 采用建议与提示词排序规范指导。
version: 1.0.0
phase: 17
lesson: 06
tags: [sglang, radixattention, prefix-caching, scheduler, prompt-ordering]
---

根据工作负载描述（提示词模板结构、检索模式、对话长度、并发租户数量、硬件配置），生成一份 SGLang / RadixAttention 采用建议书。

输出内容应包含：

1. **工作负载指纹（Workload Fingerprint）**。将其分类为前缀密集型（Prefix-heavy，例如带有重复前言的检索增强生成（RAG）、带有重复工具模式的智能体（Agent）、带有重复上下文的语音任务）或前缀轻量型（Prefix-light，例如独立的单次提示词（Single-shot Prompt））。标明共享前缀长度与重复率。
2. **提示词排序审计（Prompt-ordering Audit）**。自上而下检查当前提示词模板。标记出任何穿插在不可变部分中的动态内容。推荐标准顺序：系统提示（System）→ 工具/模式（Tools/Schemas）→ 检索上下文（Retrieval Context）→ 对话历史（Conversation History）→ 用户输入（User Input）。
3. **预期命中率（Expected Hit Rate）**。根据工作负载指纹，估算可实现的缓存命中率（Cache Hit Rate）。通用聊天场景为 10-30%。模板一致的 RAG 场景为 60-85%。带有固定前言的语音/视觉场景为 80-95%。
4. **SGLang 与 vLLM 选型决策**。若预期命中率 > 40% 且工作负载非单次请求，推荐 SGLang。若 < 30%，使用带有 `--enable-prefix-caching` 参数的 vLLM 更为简便。若处于 30-40% 之间，则在样本数据上同时运行两者并择优选择。
5. **上线计划（Rollout Plan）**。使用当前提示词模板在 SGLang 上进行 48 小时的影子基准测试（Shadow Benchmark）。记录命中率。修复提示词排序问题。重新进行基准测试。若命中率达标，则正式发布。

**硬性拒绝条件（Hard Rejects）：**
- 未在实际流量中测量前缀共享情况就推荐 SGLang。必须拒绝。
- 未结合工作负载特征就声称 6.4 倍的数值。该数据具有工作负载特异性。
- 忽视提示词排序规范。模板即缓存键（Cache Key）；缺乏规范，调度器将无法发挥作用。

**拒绝规则（Refusal Rules）：**
- 若工作负载为单次请求（无重复的系统提示词），则拒绝 SGLang 并推荐 vLLM。
- 若团队无法控制提示词模板（例如面向第三方消费者），则拒绝并建议在重新评估前，先在代理层（Proxy-level）进行模板标准化。
- 若多租户隔离要求为每个租户分配独立的 KV 池（KV Pool），需注明 SGLang 虽支持该功能，但树状分支淘汰（Tree-branch Eviction）可能导致小型租户资源饥饿；建议按租户分配预算配额。

**输出要求：** 一份单页的 SGLang 建议书，需列出工作负载指纹、提示词排序修复方案、预期命中率、引擎选型及上线计划。结尾需附加一段“下一步阅读建议”，根据当前最大的知识短板，指引读者查阅 SGLang 论文、vLLM 前缀缓存文档或本课的提示词排序练习。