# 聊天机器人（Chatbots）：从基于规则（Rule-Based）到神经网络（Neural）再到大语言模型智能体（LLM Agents）

> ELIZA 通过模式匹配进行回复。DialogFlow 负责意图映射。GPT 基于模型权重生成回答。Claude 则调用工具并验证结果。每一个时代都解决了前一个时代最致命的缺陷。

**Type:** 学习
**Languages:** Python
**Prerequisites:** 第 5 阶段 · 13（问答系统），第 5 阶段 · 14（信息检索）
**Time:** 约 75 分钟

## 核心问题

用户说：“我想改签航班。”系统必须弄清楚他们的意图、缺失哪些信息、如何获取这些信息，以及如何完成操作。接着用户又说：“等等，如果我取消航班呢？”此时系统必须记住上下文、切换任务并保持状态。

对话对机器学习（Machine Learning, ML）系统来说极具挑战。输入是开放式的，输出必须在多轮交互中保持连贯。系统可能需要在现实世界中执行操作（如改签航班、扣款）。每一个错误的步骤都会直接暴露给用户。

聊天机器人架构经历了四种范式的演进，每一种的引入都是因为前一种的失败过于明显。本课程将按顺序逐一讲解。2026 年的生产环境架构是后两种范式的混合体。

## 核心概念

![聊天机器人演进：基于规则 → 检索式 → 神经网络 → 智能体](../assets/chatbot.svg)

**基于规则（Rule-Based）（ELIZA、AIML、DialogFlow）。** 人工编写的模式匹配用户输入并生成回复。意图分类器（Intent Classifiers）将请求路由至预定义流程。槽位填充（Slot-filling）状态机负责收集必要信息。在其设计的狭窄范围内表现卓越，但一旦超出范围便会立即失效。目前仍广泛应用于对安全性要求极高的领域（如银行身份验证、机票预订），因为这些场景绝不允许出现幻觉（Hallucination）。

**检索式（Retrieval-Based）。** 一种类似 FAQ 的系统。对每一组（用户话语，系统回复）进行编码。在运行时，对用户消息进行编码，并检索最相似的已存储回复。可以将其理解为 Zendesk 经典的“相似文章”功能。相比规则系统，它能更好地处理同义改写。由于不涉及文本生成，因此不会产生幻觉。

**神经网络（Neural）（Seq2Seq）。** 基于对话日志训练的编码器-解码器（Encoder-Decoder）架构。从零开始生成回复。语言流畅，但容易输出泛泛而谈的内容（如“我不知道”）并出现事实性漂移（Factual Drift）。始终无法可靠地紧扣主题。这也是 Google、Facebook 和 Microsoft 在 2016-2019 年间推出的聊天机器人均令人失望的原因。

**大语言模型智能体（LLM Agents）。** 将语言模型封装在一个循环中，使其能够进行规划、调用工具并验证结果。它不是仅仅依靠超长提示词（Prompt）的聊天机器人，而是一个智能体循环：规划 → 调用工具 → 观察结果 → 决定下一步。优先检索的事实锚定技术（检索增强生成，Retrieval-Augmented Generation, RAG）可防止其产生幻觉。工具调用（Tool Calls）使其能够真正执行操作。这就是 2026 年的主流架构。

这四种范式并非简单的线性替代关系。2026 年的生产级聊天机器人会综合路由至这四种架构：基于规则用于身份验证和破坏性操作，检索式用于 FAQ，神经网络生成用于自然语言措辞，大语言模型智能体用于处理模糊的开放式查询。

## 动手实践

### 步骤 1：基于规则的模式匹配 (Rule-based Pattern Matching)

import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."

仅用 20 行代码实现的 ELIZA。其中的“反射技巧” (Reflection Trick，例如将“I feel sad”转换为“Why do you feel sad”）源自 Weizenbaum 1966 年的经典心理治疗师演示程序。至今仍有教学意义。

### 步骤 2：基于检索的问答 (Retrieval-based FAQ)

此示例代码片段需要运行 `pip install sentence-transformers`（该命令会连带安装 torch）。本课程的完整可运行文件 `code/main.py` 实际上使用了 Python 标准库中的 Jaccard 相似度 (Jaccard Similarity) 作为替代，因此无需安装外部依赖即可运行。

from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]

基于阈值 (Threshold) 的拒绝机制是关键设计选择。如果最佳匹配结果的相似度不够高，则返回 `None`，交由系统进行升级处理 (Escalation)。

### 步骤 3：神经生成基线 (Neural Generation Baseline)

使用小型指令微调 (Instruction-tuned) 的编码器-解码器 (Encoder-Decoder) 模型（如 FLAN-T5）或经过微调的对话模型。在 2026 年，此类模型若单独部署于生产环境仍不可用（存在自相矛盾、话题漂移、事实性错误等问题），但通常作为混合系统的一部分，用于生成更自然的表述。DialoGPT 风格的仅解码器 (Decoder-only) 模型需要显式的对话轮次分隔符 (Turn Separators) 和结束符 (EOS) 处理才能生成连贯回复；而 FLAN-T5 的 text2text 管道 (Pipeline) 开箱即用，非常适合作为教学示例。

from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])

### 步骤 4：大语言模型智能体循环 (LLM Agent Loop)

2026 年生产环境的标准架构如下：

def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."

需要明确三个核心概念。工具 (Tools) 是大语言模型可调用的函数。当大语言模型返回最终答案而非工具调用请求时，循环终止。步骤预算 (Step Budget) 用于防止在处理模糊任务时陷入无限循环。

实际生产环境还会增加以下模块：检索优先的事实锚定 (Retrieval-first Grounding，在每次调用大语言模型前注入相关文档)、安全护栏 (Guardrails，未经确认则拒绝执行破坏性操作)、可观测性 (Observability，记录每一步操作日志) 以及评估机制 (Evaluations，自动检查智能体行为是否符合规范)。

### 步骤 5：混合路由 (Hybrid Routing)

def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)

核心模式如下：对任何破坏性操作使用确定性规则 (Deterministic Rules)，对预设常见问题使用检索匹配，其余情况交由大语言模型智能体处理。这正是 2026 年客户支持系统中实际部署的架构。

## 实际应用

2026 年技术栈：

| 使用场景 | 架构 |
|---------|---------------|
| 预订、支付、身份验证 | 基于规则的状态机（Rule-based state machine）+ 槽位填充（Slot filling） |
| 客服常见问题解答 | 基于精选答案的检索（Retrieval over curated answers） |
| 开放式帮助对话 | 结合检索增强生成（Retrieval-Augmented Generation, RAG）与工具调用（Tool call）的大语言模型智能体（LLM agent） |
| 内部工具 / IDE 助手 | 具备工具调用能力的大语言模型智能体（搜索、读取、写入） |
| 陪伴型 / 角色扮演聊天机器人 | 经过微调（Fine-tuned）的大语言模型，配备角色设定系统提示词（Persona system prompt）及知识库检索 |

在生产环境中务必使用混合路由（Hybrid routing）。没有任何单一架构能完美处理所有请求。路由层本身通常是一个轻量级的意图分类器（Intent classifier）。

## 实际部署中仍存在的故障模式

- **确信型捏造（Confident fabrication）。** 大语言模型智能体声称已完成某项操作，但实际上并未执行。缓解措施：验证执行结果、记录工具调用日志，绝不允许大语言模型在未收到工具成功返回结果的情况下声称已完成操作。
- **提示词注入（Prompt injection）。** 用户插入文本以覆盖系统提示词。在《2025 年 OWASP 大语言模型应用十大安全风险》中位列 LLM01。分为两种形式：直接注入（直接粘贴到聊天窗口中）和间接注入（隐藏在智能体读取的文档、电子邮件或工具输出中）。

  攻击成功率因场景而异。在通用工具使用和编程基准测试中，前沿模型的实测成功率约为 0.5% 至 8.5%。在特定的高风险配置下（如针对 AI 编程智能体的自适应攻击、存在漏洞的编排系统），成功率可高达约 84%。生产环境中的已知漏洞包括 EchoLeak（CVE-2025-32711，CVSS 评分 9.3）——这是 Microsoft 365 Copilot 中的一个零点击数据外泄漏洞，由攻击者控制的电子邮件触发。

  缓解措施：在整个交互循环中将用户输入视为不可信数据；在调用工具前进行输入清洗；将工具输出与主提示词隔离；采用“计划-验证-执行”（Plan-Verify-Execute, PVE）模式，即智能体先制定计划，在执行前对照该计划验证每一步操作（此举可防止工具结果注入新的未计划操作）；对破坏性操作要求用户确认；对工具权限范围实施最小权限原则（Least-privilege）。

  无论进行多少提示词工程（Prompt engineering）优化，都无法完全消除该风险。必须引入外部运行时防御层（如 LLM Guard、白名单验证、语义异常检测）。
- **范围蔓延（Scope creep）。** 智能体因工具调用返回了边缘相关信息而偏离原定任务。缓解措施：收紧工具契约（Tool contract）；保持系统提示词聚焦；增加对任务偏离率的评估。
- **无限循环（Infinite loop）。** 智能体持续重复调用同一工具。缓解措施：设置步骤预算（Step budget）、工具调用去重、使用大语言模型作为裁判（LLM judge）评估“当前是否取得进展”。
- **上下文窗口耗尽（Context window exhaustion）。** 长对话会将最早的对话轮次挤出上下文。缓解措施：对较早的对话轮次进行摘要、通过相似度检索相关的历史轮次，或使用长上下文模型（Long-context model）。

## 部署发布

Save as `outputs/skill-chatbot-architect.md`:

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## Exercises

1. **Easy.** Implement the rule-based respond above with 10 patterns for a coffee-shop ordering bot. Test edge cases: double orders, modifications, cancellation, unclear intent.
2. **Medium.** Build a hybrid FAQ + LLM fallback. 50 canned FAQ entries for a SaaS product, LLM fallback with retrieval over the docs site. Measure refusal rate and accuracy on 100 real support questions.
3. **Hard.** Implement the agent loop above with three tools (search, read-user-data, send-email). Run an evaluation with 50 test scenarios including prompt injection attempts. Report off-task rate, failed task rate, and any injection success.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Intent | What the user wants | Categorical label (book_flight, reset_password). Routed to a handler. |
| Slot | A piece of info | Parameter the bot needs (date, destination). Slot filling is the sequence of asks. |
| RAG | Retrieval plus generation | Retrieve relevant docs, then ground the LLM's response. |
| Tool call | Function invocation | LLM emits a structured call with name + args. Runtime executes, returns result. |
| Agent loop | Plan, act, verify | Controller that runs LLM calls interleaved with tool calls until task complete. |
| Prompt injection | User attacks prompt | Malicious input that tries to override the system prompt. |

## Further Reading

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) —— 基于规则（rule-based）聊天机器人（chatbot）的开创性论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) —— Google 在大型语言模型智能体（LLM agents）接管主流之前发布的神经聊天机器人（neural chatbot）末期论文。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) —— 正式命名智能体循环（agent loop）模式的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) —— 2024 年发布的生产级（production）指南，其核心原则在 2026 年依然适用。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) —— 提示词注入（prompt injection）领域的标志性论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) —— 该安全榜单将提示词注入列为首要安全威胁。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) —— 编排层（orchestration layer）的实用防御方案，涵盖“规划-验证-执行”（Plan-Verify-Execute）流程与用户确认（user-confirmation）机制。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) —— 由间接提示词注入（indirect prompt injection）引发的典型零点击数据外泄（zero-click data exfiltration）CVE 漏洞。该案例充分说明了为何具备写入权限（write-access）的智能体需要运行时（runtime）防御机制。