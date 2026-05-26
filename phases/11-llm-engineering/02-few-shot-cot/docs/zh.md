# 少样本（Few-Shot）、思维链（Chain-of-Thought）与思维树（Tree-of-Thought）

> 告诉模型该做什么属于提示（Prompting），而教会它如何思考则属于工程。在相同模型、相同任务和相同数据下，准确率从 78% 提升到 91% 的差距并非源于更强大的模型，而是源于更优的推理策略。

**类型：** 构建
**语言：** Python
**前置课程：** 第 11.01 课（提示工程）
**时长：** 约 45 分钟

## 学习目标

- 通过选择和格式化示例演示来实现少样本提示（Few-Shot Prompting），以最大化任务准确率
- 应用思维链（Chain-of-Thought, CoT）推理，提升数学应用题等多步问题的准确率
- 构建思维树（Tree-of-Thought）提示，探索多条推理路径并选择最优解
- 在标准基准测试上，测量零样本（Zero-Shot）、少样本与思维链之间的准确率提升幅度

## 问题背景

你开发了一款数学辅导应用。你的提示词写着：“解答这道应用题。”在 GSM8K（标准小学数学基准测试）上，GPT-5 的正确率为 94%。你以为已经触及天花板了，但实际上并没有——思维链仍能再提升 3 到 4 个百分点。

只需加上五个字——“让我们一步步思考”——准确率就能跃升至 91%。如果再补充几个已解答的示例，准确率可达 95%。模型相同、温度参数（Temperature）相同、API 调用成本也相同。唯一的区别在于，你给了模型一张草稿纸。

这并非什么取巧的捷径，而是推理运作的本质。人类无法仅凭一次思维跳跃就解决多步问题，Transformer 架构同样如此。当你强制模型生成中间词元（Token）时，这些词元会成为下一个词元的上下文。每一个推理步骤都在为下一步提供输入。模型正是通过这种逐步计算的方式得出最终答案。

但“一步步思考”仅仅是起点，而非终点。如果采样五条推理路径并进行多数投票会怎样？如果让模型探索一棵可能性树，评估并剪枝无效分支会怎样？如果将推理过程与工具调用交替进行又会怎样？这些并非空想，而是已有论文发表且经过实测验证的技术。在本节课中，你将亲手实现所有这些方法。

## 核心概念

### 零样本（Zero-Shot）与少样本（Few-Shot）：何时示例胜过指令

零样本提示（Zero-shot prompting）仅向模型提供任务，不包含其他信息。少样本提示（Few-shot prompting）则会先提供示例。

Wei 等人（2022）在 8 个基准测试中对此进行了评估。对于情感分类等简单任务，零样本和少样本的表现差异在 2% 以内。对于多步算术和符号推理等复杂任务，少样本提示将准确率提升了 10-25%。

其核心直觉在于：示例就是压缩后的指令。与其描述输出格式，不如直接展示它；与其解释推理过程，不如直接演示它。模型在示例上进行模式匹配（pattern-matching）的可靠性，远高于其解读抽象指令的能力。

graph TD
    subgraph Comparison["Zero-Shot vs Few-Shot"]
        direction LR
        Z["Zero-Shot\n'Classify this review'\nModel guesses format\n78% on GSM8K"]
        F["Few-Shot\n'Here are 3 examples...\nNow classify this review'\nModel matches pattern\n85% on GSM8K"]
    end

    Z ~~~ F

    style Z fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#51cf66,color:#fff

**少样本提示胜出的场景：** 对格式敏感的任务、分类任务、结构化信息提取、领域特定术语，以及任何需要模型匹配特定模式的任务。

**零样本提示胜出的场景：** 简单的事实性问题、示例会限制创造力的创意类任务，以及寻找优质示例比编写优质指令更困难的任务。

### 示例选择：相似性优于随机性

并非所有示例都同等有效。在分类任务中，选择与目标输入相似的示例比随机选择的效果高出 5-15%（Liu 等人，2022）。遵循以下三个原则：

1. **语义相似性（Semantic similarity）**：在嵌入空间（embedding space）中选择与输入最接近的示例
2. **标签多样性（Label diversity）**：确保示例覆盖所有输出类别
3. **难度匹配（Difficulty matching）**：示例的复杂度应与目标问题相匹配

对于大多数任务，最佳示例数量为 3-5 个。少于 3 个时，模型缺乏足够的信号来提取模式；超过 5 个时，收益递减且会浪费上下文窗口（context window）的 token。对于多标签分类任务，建议每个标签使用一个示例。

### 思维链（Chain-of-Thought, CoT）：为模型提供草稿纸

思维链（Chain-of-Thought, CoT）提示由 Google Brain 的 Wei 等人（2022）提出。其理念很简单：与其直接要求模型给出答案，不如先让它展示推理步骤。

graph LR
    subgraph Standard["Standard Prompting"]
        Q1["Q: Roger has 5 balls.\nHe buys 2 cans of 3.\nHow many balls?"] --> A1["A: 11"]
    end

    subgraph CoT["Chain-of-Thought Prompting"]
        Q2["Q: Roger has 5 balls.\nHe buys 2 cans of 3.\nHow many balls?"] --> R2["Roger starts with 5.\n2 cans of 3 = 6.\n5 + 6 = 11."] --> A2["A: 11"]
    end

    style Q1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style Q2 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style R2 fill:#1a1a2e,stroke:#ffa500,color:#fff
    style A2 fill:#1a1a2e,stroke:#51cf66,color:#fff

从机制上看，它为何有效？Transformer 生成的每个 token 都会成为下一个 token 的上下文。在没有 CoT 的情况下，模型必须将所有推理过程压缩到单次前向传播（forward pass）的隐藏状态中。而使用 CoT 时，模型将中间计算过程外化为 token。每个推理 token 都有效扩展了计算深度。

**GSM8K 基准测试（小学数学题，共 8.5K 道题）：**

| 模型 | 零样本（Zero-Shot） | 零样本 CoT | 少样本 CoT |
|-------|-----------|---------------|--------------|
| GPT-4o | 78% | 91% | 95% |
| GPT-5 | 94% | 97% | 98% |
| o4-mini (reasoning) | 97% | — | — |
| Claude Opus 4.7 | 93% | 97% | 98% |
| Gemini 3 Pro | 92% | 96% | 98% |
| Llama 4 70B | 80% | 89% | 94% |
| DeepSeek-V3.1 | 89% | 94% | 96% |

**关于推理模型的说明。** OpenAI 的 o 系列（o3、o4-mini）和 DeepSeek-R1 等模型在输出答案前，会在内部自动运行思维链。向推理模型添加“让我们一步步思考”是多余的，有时甚至适得其反——因为它们已经内置了该过程。

CoT 的两种变体：

**零样本 CoT（Zero-shot CoT）**：在提示词末尾附加“让我们一步步思考”。无需示例。Kojima 等人（2022）证明，仅凭这一句话就能提升算术、常识和符号推理任务的准确率。

**少样本 CoT（Few-shot CoT）**：提供包含推理步骤的示例。其效果优于零样本 CoT，因为模型能直接看到你期望的确切推理格式。

**CoT 适得其反的场景：** 简单的事实回忆（“法国的首都是哪里？”）、单步分类任务，以及速度比准确率更重要的任务。CoT 会为每次查询增加 50-200 个 token 的推理开销。对于高吞吐量、低复杂度的任务，这纯属浪费成本。

### 自洽性（Self-Consistency）：多次采样，一次投票

Wang 等人（2023）提出了自洽性（Self-Consistency）方法。其核心洞察是：单条 CoT 路径可能包含推理错误。但如果采样 N 条独立的推理路径（使用温度参数 temperature > 0），并对最终答案进行多数投票，错误就会相互抵消。

graph TD
    P["Problem: 'A store has 48 apples.\nThey sell 1/3 on Monday\nand 1/4 of the rest on Tuesday.\nHow many are left?'"]

    P --> Path1["Path 1: 48 - 16 = 32\n32 - 8 = 24\nAnswer: 24"]
    P --> Path2["Path 2: 1/3 of 48 = 16\nRemaining: 32\n1/4 of 32 = 8\n32 - 8 = 24\nAnswer: 24"]
    P --> Path3["Path 3: 48/3 = 16 sold\n48 - 16 = 32\n32/4 = 8 sold\n32 - 8 = 24\nAnswer: 24"]
    P --> Path4["Path 4: Sell 1/3: 48 - 12 = 36\nSell 1/4: 36 - 9 = 27\nAnswer: 27"]
    P --> Path5["Path 5: Monday: 48 * 2/3 = 32\nTuesday: 32 * 3/4 = 24\nAnswer: 24"]

    Path1 --> V["Majority Vote\n24: 4 votes\n27: 1 vote\nFinal: 24"]
    Path2 --> V
    Path3 --> V
    Path4 --> V
    Path5 --> V

    style P fill:#1a1a2e,stroke:#ffa500,color:#fff
    style Path1 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style Path2 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style Path3 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style Path4 fill:#1a1a2e,stroke:#e94560,color:#fff
    style Path5 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style V fill:#1a1a2e,stroke:#51cf66,color:#fff

在最初的 PaLM 540B 实验中，当 N=40 时，自洽性将 GSM8K 的准确率从 56.5%（单条 CoT）提升至 74.4%。在 GPT-5 上，提升幅度较小（97% 至 98%），因为基础准确率已接近饱和。该技术在基础 CoT 准确率为 60-85% 的模型上表现最为出色——此时单路径错误频繁但并非系统性错误。对于推理模型（o 系列、R1），自洽性已被其内置的内部采样机制所涵盖。

权衡取舍：N 次采样意味着 N 倍的 API 成本和延迟。在实践中，N=5 即可捕获大部分收益。N=3 是进行有效投票的最低要求。对于大多数任务，N > 10 的收益会递减。

### 思维树（Tree-of-Thought, ToT）：分支探索

Yao 等人（2023）提出了思维树（Tree-of-Thought, ToT）。与 CoT 遵循单一线性推理路径不同，ToT 会探索多个分支，并在继续推进前评估哪些分支最有希望。

graph TD
    Root["Problem"] --> B1["Thought 1a"]
    Root --> B2["Thought 1b"]
    Root --> B3["Thought 1c"]

    B1 --> E1["Eval: 0.8"]
    B2 --> E2["Eval: 0.3"]
    B3 --> E3["Eval: 0.9"]

    E1 -->|Continue| B1a["Thought 2a"]
    E1 -->|Continue| B1b["Thought 2b"]
    E3 -->|Continue| B3a["Thought 2a"]
    E3 -->|Continue| B3b["Thought 2b"]

    E2 -->|Prune| X["X"]

    B1a --> E4["Eval: 0.7"]
    B3a --> E5["Eval: 0.95"]

    E5 -->|Best path| Final["Solution"]

    style Root fill:#1a1a2e,stroke:#ffa500,color:#fff
    style E2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style X fill:#1a1a2e,stroke:#e94560,color:#fff
    style E5 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style Final fill:#1a1a2e,stroke:#51cf66,color:#fff
    style B1 fill:#1a1a2e,stroke:#808080,color:#fff
    style B2 fill:#1a1a2e,stroke:#808080,color:#fff
    style B3 fill:#1a1a2e,stroke:#808080,color:#fff
    style B1a fill:#1a1a2e,stroke:#808080,color:#fff
    style B1b fill:#1a1a2e,stroke:#808080,color:#fff
    style B3a fill:#1a1a2e,stroke:#808080,color:#fff
    style B3b fill:#1a1a2e,stroke:#808080,color:#fff
    style E1 fill:#1a1a2e,stroke:#808080,color:#fff
    style E3 fill:#1a1a2e,stroke:#808080,color:#fff
    style E4 fill:#1a1a2e,stroke:#808080,color:#fff

ToT 包含三个核心组件：

1. **思维生成（Thought generation）**：生成多个候选的下一步骤
2. **状态评估（State evaluation）**：对每个候选步骤进行评分（可直接使用大语言模型作为评估器）
3. **搜索算法（Search algorithm）**：在树结构中执行广度优先搜索（BFS）或深度优先搜索（DFS），并剪枝低分分支

在“24点游戏”任务（使用算术运算将 4 个数字组合成 24）中，使用标准提示的 GPT-4 仅能解决 7.3% 的问题。使用 CoT 时降至 4.0%（由于搜索空间广阔，CoT 在此反而产生负面影响）。而使用 ToT 时，解决率高达 74%。

ToT 的成本较高。树中的每个节点都需要一次 LLM 调用。分支因子（branching factor）为 3、深度为 3 的树最多需要 39 次 LLM 调用。仅建议将其用于搜索空间大但可评估的问题——如规划、解谜以及带约束条件的创造性问题解决。

### ReAct：思考与行动结合

Yao 等人（2022）将推理轨迹与行动相结合。模型在思考（生成推理）和行动（调用工具、搜索、计算）之间交替进行。

graph LR
    Q["Question:\nWhat is the\npopulation of the\ncountry where\nthe Eiffel Tower\nis located?"]
    T1["Thought: I need to\nfind which country\nhas the Eiffel Tower"]
    A1["Action: search\n'Eiffel Tower location'"]
    O1["Observation:\nParis, France"]
    T2["Thought: Now I need\nFrance's population"]
    A2["Action: search\n'France population 2024'"]
    O2["Observation:\n68.4 million"]
    T3["Thought: I have\nthe answer"]
    F["Answer:\n68.4 million"]

    Q --> T1 --> A1 --> O1 --> T2 --> A2 --> O2 --> T3 --> F

    style Q fill:#1a1a2e,stroke:#ffa500,color:#fff
    style T1 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style A1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style O1 fill:#1a1a2e,stroke:#808080,color:#fff
    style T2 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style A2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style O2 fill:#1a1a2e,stroke:#808080,color:#fff
    style T3 fill:#1a1a2e,stroke:#51cf66,color:#fff
    style F fill:#1a1a2e,stroke:#51cf66,color:#fff

在知识密集型任务上，ReAct 的表现优于纯 CoT，因为它能将推理建立在真实数据之上。在 HotpotQA（多跳问答）基准测试中，结合 GPT-4 的 ReAct 实现了 35.1% 的精确匹配率，而纯 CoT 仅为 29.4%。其真正强大之处在于：推理错误可通过观察结果得到纠正——模型能够在执行过程中动态更新计划。

ReAct 是现代 AI 智能体（AI agents）的基石。每个智能体框架（LangChain、CrewAI、AutoGen）都实现了某种形式的“思考-行动-观察”循环（Thought-Action-Observation loop）。你将在第 14 阶段构建完整的智能体。本节主要讲解该提示词模式。

### 结构化提示（Structured Prompting）：XML 标签、分隔符与标题

随着提示词变得复杂，结构化设计能防止模型混淆不同部分。以下是三种常用方法：

**XML 标签**（在 Claude 上效果最佳，在其他模型上也表现稳定）：
<context>
You are reviewing a pull request.
The codebase uses TypeScript and React.
</context>

<task>
Review the following diff for bugs, security issues, and style violations.
</task>

<diff>
{diff_content}
</diff>

<output_format>
List each issue with: file, line, severity (critical/warning/info), description.
</output_format>

**Markdown 标题**（通用）：


## 角色

金融科技公司的资深安全工程师。

## 任务

分析此 API 端点 (API endpoint) 是否存在漏洞。

## 输入

{api_code}

## 规则

- 重点关注 OWASP Top 10
- 对每个发现进行评级：严重 (critical)、高 (high)、中 (medium)、低 (low)
- 包含修复步骤 (remediation steps)

**分隔符 (Delimiters)**（极简但有效）：
---INPUT---
{user_text}
---END INPUT---

---INSTRUCTIONS---
Summarize the above in 3 bullet points.
---END INSTRUCTIONS---

### 提示词链 (Prompt Chaining)：顺序分解 (Sequential Decomposition)

某些任务过于复杂，无法通过单个提示词 (prompt) 完成。提示词链将其拆分为多个步骤，其中前一个提示词的输出将作为下一个提示词的输入。

graph LR
    I["Raw Input"] --> P1["Prompt 1:\nExtract\nkey facts"]
    P1 --> O1["Facts"]
    O1 --> P2["Prompt 2:\nAnalyze\nfacts"]
    P2 --> O2["Analysis"]
    O2 --> P3["Prompt 3:\nGenerate\nrecommendation"]
    P3 --> F["Final Output"]

    style I fill:#1a1a2e,stroke:#808080,color:#fff
    style P1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style O1 fill:#1a1a2e,stroke:#ffa500,color:#fff
    style P2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style O2 fill:#1a1a2e,stroke:#ffa500,color:#fff
    style P3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#51cf66,color:#fff

链式处理优于单提示词 (single-prompt) 的原因有三：

1. **每个步骤更简单**：模型只需处理单一聚焦的任务，而无需同时兼顾所有环节
2. **中间输出可审查**：你可以在步骤之间进行验证和修正
3. **不同步骤可使用不同模型**：例如使用低成本模型进行信息提取，使用高成本模型进行逻辑推理

### 性能对比

| 技术 (Technique) | 最佳适用场景 (Best For) | GSM8K 准确率 (GPT-5) | API 调用次数 (API Calls) | Token 开销 (Token Overhead) | 复杂度 (Complexity) |
|-----------|----------|------------------------|-----------|----------------|------------|
| 零样本 (Zero-Shot) | 简单任务 | 94% | 1 | 无 | 极低 |
| 少样本 (Few-Shot) | 格式匹配 | 96% | 1 | 200-500 tokens | 低 |
| 零样本思维链 (Zero-Shot CoT) | 快速提升推理能力 | 97% | 1 | 50-200 tokens | 极低 |
| 少样本思维链 (Few-Shot CoT) | 最大化单次调用准确率 | 98% | 1 | 300-600 tokens | 低 |
| 自洽性 (Self-Consistency, N=5) | 关键/高风险推理 | 98.5% | 5 | 5倍 token 成本 | 中 |
| 推理模型 (Reasoning model, o4-mini) | 直接替代思维链 | 97% | 1 | 隐藏 (内部 2-10倍) | 极低 |
| 思维树 (Tree-of-Thought) | 搜索/规划类问题 | 不适用 (24点游戏 74%) | 10-40+ | 10-40倍 token 成本 | 高 |
| ReAct | 基于知识的推理 | 不适用 (HotpotQA 35.1%) | 3-10+ | 可变 | 高 |
| 提示词链 (Prompt Chaining) | 复杂多步任务 | 96% (流水线) | 2-5 | 2-5倍 token 成本 | 中 |

选择合适的技术取决于三个因素：准确率要求 (accuracy requirement)、延迟预算 (latency budget) 和成本容忍度 (cost tolerance)。对于大多数生产系统而言，采用少样本思维链 (Few-Shot CoT) 并辅以 3 样本自洽性 (Self-Consistency) 回退机制，即可覆盖 90% 的使用场景。

## 动手构建

我们将构建一个数学问题求解器，将少样本提示（few-shot prompting）、思维链推理（chain-of-thought reasoning）和自一致性投票（self-consistency voting）整合到单一流水线（pipeline）中。随后，我们将为难题引入思维树（tree-of-thought）。

完整实现位于 `code/advanced_prompting.py`。以下是核心组件。

### 步骤 1：少样本示例存储库

第一个组件负责管理少样本示例，并为给定问题筛选最相关的示例。

GSM8K_EXAMPLES = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every egg at the farmers' market for $2. How much does she make every day at the farmers' market?",
        "reasoning": "Janet's ducks lay 16 eggs per day. She eats 3 and bakes 4, using 3 + 4 = 7 eggs. So she has 16 - 7 = 9 eggs left. She sells each for $2, so she makes 9 * 2 = $18 per day.",
        "answer": "18"
    },
    ...
]

每个示例包含三个部分：问题、推理链和最终答案。正是推理链将普通的少样本示例转化为思维链（CoT）少样本示例。

### 步骤 2：思维链提示构建器

提示构建器将系统消息、带有推理链的少样本示例以及目标问题组合成一个完整的提示词。

def build_cot_prompt(question, examples, num_examples=3):
    system = (
        "You are a math problem solver. "
        "For each problem, show your step-by-step reasoning, "
        "then give the final numerical answer on the last line "
        "in the format: 'The answer is [number]'."
    )

    example_text = ""
    for ex in examples[:num_examples]:
        example_text += f"Q: {ex['question']}\n"
        example_text += f"A: {ex['reasoning']} The answer is {ex['answer']}.\n\n"

    user = f"{example_text}Q: {question}\nA:"
    return system, user

格式约束（“The answer is [number]”）至关重要。如果没有它，自一致性机制将无法跨样本提取并比较答案。

### 步骤 3：自一致性投票

采样 N 条推理路径，并采用多数票答案。

def self_consistency_solve(question, examples, client, model, n_samples=5):
    system, user = build_cot_prompt(question, examples)

    answers = []
    reasonings = []
    for _ in range(n_samples):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.7
        )
        text = response.choices[0].message.content
        reasonings.append(text)
        answer = extract_answer(text)
        if answer is not None:
            answers.append(answer)

    vote_counts = Counter(answers)
    best_answer = vote_counts.most_common(1)[0][0] if vote_counts else None
    confidence = vote_counts[best_answer] / len(answers) if best_answer else 0

    return best_answer, confidence, reasonings, vote_counts

温度参数（temperature）设为 0.7 非常关键。若设为 0.0，所有 N 个样本将完全相同，从而失去该机制的意义。你需要足够的随机性以生成多样化的推理路径，但又不能过高导致模型输出无意义的内容。

### 步骤 4：思维树求解器

对于线性推理失效的问题，思维树（ToT）会探索多种解题思路，并评估哪个方向最具潜力。

def tree_of_thought_solve(question, client, model, breadth=3, depth=3):
    thoughts = generate_initial_thoughts(question, client, model, breadth)
    scored = [(t, evaluate_thought(t, question, client, model)) for t in thoughts]
    scored.sort(key=lambda x: x[1], reverse=True)

    for current_depth in range(1, depth):
        next_thoughts = []
        for thought, score in scored[:2]:
            extensions = extend_thought(thought, question, client, model, breadth)
            for ext in extensions:
                ext_score = evaluate_thought(ext, question, client, model)
                next_thoughts.append((ext, ext_score))
        scored = sorted(next_thoughts, key=lambda x: x[1], reverse=True)

    best_thought = scored[0][0] if scored else ""
    return extract_answer(best_thought), best_thought

评估器本身也是一次大语言模型（LLM）调用。你会向模型提问：“在 0.0 到 1.0 的范围内，该推理路径解决此问题的潜力有多大？”这正是思维树的核心洞见——让模型评估自身的部分解。

### 步骤 5：完整流水线

该流水线结合所有技术，并采用动态升级策略。

def solve_with_escalation(question, examples, client, model):
    system, user = build_cot_prompt(question, examples)
    single_response = call_llm(client, model, system, user, temperature=0.0)
    single_answer = extract_answer(single_response)

    sc_answer, confidence, _, _ = self_consistency_solve(
        question, examples, client, model, n_samples=5
    )

    if confidence >= 0.8:
        return sc_answer, "self_consistency", confidence

    tot_answer, _ = tree_of_thought_solve(question, client, model)
    return tot_answer, "tree_of_thought", None

升级逻辑如下：首先尝试低成本方案（单次思维链）。如果自一致性置信度低于 0.8（即 5 个样本中少于 4 个达成一致），则升级至思维树。这种设计在成本与准确率之间取得了平衡——大多数问题以低成本解决，而难题则分配更多计算资源。

## 使用方法

### 结合 LangChain 使用

LangChain 内置了对提示词模板（Prompt Templates）和输出解析（Output Parsing）的支持，可简化少样本（Few-Shot）和思维链（Chain of Thought, CoT）模式的使用：

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

example_prompt = PromptTemplate(
    input_variables=["question", "reasoning", "answer"],
    template="Q: {question}\nA: {reasoning} The answer is {answer}."
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Q: {input}\nA: Let's think step by step.",
    input_variables=["input"]
)

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
chain = few_shot_prompt | llm
result = chain.invoke({"input": "If a train travels 120 km in 2 hours..."})

LangChain 还提供了用于语义相似度（Semantic Similarity）选择的 `ExampleSelector` 类：

from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    k=3
)

### 结合 DSPy 使用

DSPy 将提示词策略（Prompting Strategies）视为可优化的模块。与其手动编写思维链提示词，不如定义一个签名（Signature），然后让 DSPy 自动优化提示词：

import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o", temperature=0.7))

class MathSolver(dspy.Module):
    def __init__(self):
        self.solve = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.solve(question=question)

solver = MathSolver()
result = solver(question="Janet's ducks lay 16 eggs per day...")

DSPy 的 `ChainOfThought` 会自动添加推理轨迹（Reasoning Traces）。`dspy.majority` 则实现了自一致性（Self-Consistency）：

result = dspy.majority(
    [solver(question=q) for _ in range(5)],
    field="answer"
)

### 对比：从零实现 vs 框架

| 特性 | 从零实现（本课程） | LangChain | DSPy |
|---------|--------------------------|-----------|------|
| 提示词格式控制 | 完全控制 | 基于模板 | 自动 |
| 自一致性（Self-Consistency） | 手动投票 | 手动 | 内置（`dspy.majority`） |
| 示例选择 | 自定义逻辑 | `ExampleSelector` | `dspy.BootstrapFewShot` |
| 思维树（Tree-of-Thought） | 自定义树搜索 | 社区链 | 非内置 |
| 提示词优化 | 手动迭代 | 手动 | 自动编译 |
| 适用场景 | 学习、自定义流水线 | 标准工作流 | 研究、优化 |

## 交付成果

本课程将产出两份交付物。

**1. 推理链提示词**（`outputs/prompt-reasoning-chain.md`）：一份可用于生产环境的提示词模板，支持带自一致性的少样本思维链。只需填入你的示例和问题领域即可。

**2. CoT 模式选择指南**（`outputs/skill-cot-patterns.md`）：一套决策框架，可根据任务类型、精度要求和成本约束，帮助你选择合适的推理技术。

## 练习

1. **评估性能差距**：选取 10 道 GSM8K 题目。分别使用零样本（zero-shot）、少样本（few-shot）、零样本思维链（zero-shot CoT）和少样本思维链（few-shot CoT）进行求解。记录每种方法的准确率。哪种技术能为你的模型带来最大的性能提升？

2. **示例选择实验**：针对同样的 10 道题目，对比随机选择示例与人工挑选相似示例的效果。测量两者的准确率差异。在什么临界点上，示例质量的重要性会超过示例数量？

3. **自一致性（self-consistency）成本曲线**：在 20 道 GSM8K 题目上，分别设置 N=1、3、5、7、10 运行自一致性算法。绘制准确率与成本（总 token 数）的关系图。你的模型在该曲线的拐点（knee of the curve）位于何处？

4. **构建 ReAct 循环**：在现有流水线中集成计算器工具。当模型生成数学表达式时，使用 Python 的 `eval()`（在沙箱环境中）执行计算，并将结果反馈回模型。评估基于工具的推理（tool-grounded reasoning）是否优于纯思维链（pure CoT）。

5. **面向创意任务的思维树（Tree-of-Thought, ToT）**：将思维树求解器适配至创意写作任务：“写一个仅含 6 个单词、既幽默又悲伤的故事。”使用大语言模型（LLM）作为评估器。分支探索（branching exploration）能否比单次生成（single-shot generation）产出更优质的创意内容？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 少样本提示（Few-shot prompting） | “给它一些示例” | 在提示词中包含输入-输出演示，以锚定模型的输出格式和行为 |
| 思维链（Chain-of-Thought） | “让它一步步思考” | 引导模型生成中间推理 token，在输出最终答案前扩展其有效计算过程 |
| 自一致性（Self-Consistency） | “多次运行它” | 在 temperature > 0 时采样 N 条不同的推理路径，并通过多数投票选出最常见的最终答案 |
| 思维树（Tree-of-Thought） | “让它探索不同选项” | 对推理分支进行结构化搜索，评估每个部分解，仅扩展有潜力的路径 |
| ReAct | “思考 + 工具使用” | 在“思考-行动-观察”循环中，将推理轨迹与外部操作（搜索、计算、API 调用）交替执行 |
| 提示词链（Prompt chaining） | “将其拆分为多个步骤” | 将复杂任务分解为顺序提示词，使每个输出作为下一个提示词的输入 |
| 零样本思维链（Zero-shot CoT） | “只需加上‘一步步思考’” | 在不提供任何示例的提示词末尾附加推理触发短语，依赖模型潜在的推理能力 |

## 进一步阅读

- [思维链提示激发大语言模型的推理能力](https://arxiv.org/abs/2201.11903) -- Wei 等人，2022 年。来自 Google Brain 的原始思维链（Chain-of-Thought, CoT）论文。阅读第 2-3 节以了解核心结果。
- [自一致性提升语言模型中的思维链推理能力](https://arxiv.org/abs/2203.11171) -- Wang 等人，2023 年。自一致性（Self-Consistency）论文。表 1 包含了你所需的所有数据。
- [思维树：大语言模型的深思熟虑式问题解决](https://arxiv.org/abs/2305.10601) -- Yao 等人，2023 年。思维树（Tree of Thoughts, ToT）论文。第 4 节中的“24点游戏”（Game of 24）结果是本文亮点。
- [ReAct：在语言模型中协同推理与行动](https://arxiv.org/abs/2210.03629) -- Yao 等人，2022 年。现代 AI 智能体（AI Agents）的基石。第 3 节详细解释了“思考-行动-观察”（Thought-Action-Observation）循环。
- [大语言模型是零样本（Zero-Shot）推理器](https://arxiv.org/abs/2205.11916) -- Kojima 等人，2022 年。提出“让我们一步步思考”（Let's think step by step）的论文。尽管方法极其简单，但效果出奇地好。
- [DSPy：将声明式语言模型调用编译为自改进流水线](https://arxiv.org/abs/2310.03714) -- Khattab 等人，2023 年。将提示（Prompting）视为编译问题。如果你想超越手动提示工程（Prompt Engineering），推荐阅读此文。
- [OpenAI — 推理模型指南](https://platform.openai.com/docs/guides/reasoning) -- 厂商指南，说明了思维链何时会转变为内部按 Token 计费的“推理”模式，而非仅停留在提示层面的技巧。
- [Lightman 等人，《让我们逐步验证》（2023）](https://arxiv.org/abs/2305.20050) -- 过程奖励模型（Process Reward Models, PRM），用于对推理链中的每一步进行评分；这是一种取代仅结果奖励（Outcome-Only Rewards）的推理监督信号。
- [Snell 等人，《最优扩展大语言模型测试时计算（Test-Time Compute）》（2024）](https://arxiv.org/abs/2408.03314) -- 对 CoT 长度、自一致性采样以及蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）的系统性研究；探讨了当准确性比延迟更重要时，“逐步思考”策略的演进方向。