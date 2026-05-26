# 评估 (Evaluation)：基准测试 (Benchmarks)、评测 (Evals) 与 LM Harness

> 古德哈特定律 (Goodhart's Law)：当一项指标成为目标时，它就不再是一项好指标。每家前沿实验室都在针对基准测试进行优化（刷分）。MMLU 分数不断攀升，但模型依然无法可靠地数出“strawberry”中有几个字母“R”。唯一重要的评测是你自己的评测——针对你的任务，使用你的数据。

**类型：** 构建 (Build)
**语言：** Python
**前置条件：** 第 10 阶段，第 01-05 课（从零构建大语言模型）
**时长：** 约 90 分钟

## 学习目标

- 构建自定义评估框架 (Evaluation Harness)，用于在语言模型上运行多项选择和开放式基准测试
- 解释为何标准基准测试（如 MMLU、HumanEval）会出现分数饱和，且无法有效区分前沿模型 (Frontier Models)
- 实现针对特定任务的评测，并采用合适的评估指标：精确匹配 (Exact Match)、F1 分数 (F1 Score)、BLEU 分数 (BLEU Score) 以及 LLM 作为裁判 (LLM-as-Judge) 评分
- 设计针对你特定用例的自定义评估套件，而非仅仅依赖公开排行榜

## 问题所在

MMLU 于 2020 年发布，包含涵盖 57 个学科的 15,908 道题目。短短三年内，前沿模型便使其分数趋于饱和。GPT-4 得分为 86.4%，Claude 3 Opus 为 86.8%，Llama 3 405B 为 88.6%。排行榜的分数被压缩在 3 分的极小区间内，这些差异仅是统计噪声 (Statistical Noise)，而非真实的能力差距。

与此同时，这些模型却在一些 10 岁孩子不假思索就能完成的任务上频频失误。在 MMLU 上取得 88.7% 高分的 Claude 3.5 Sonnet，最初甚至无法数清“strawberry”中的字母数量——这项任务不需要任何世界知识或推理能力，只需进行字符级遍历 (Character-level Iteration)。HumanEval 通过 164 道题目测试代码生成能力。模型在该测试中得分超过 90%，但生成的代码依然会在任何初级开发者都能发现的边界情况 (Edge Cases) 下崩溃。

基准测试表现与实际应用可靠性之间的差距，是大语言模型 (LLM) 评估的核心问题。基准测试只能告诉你模型在基准测试本身上的表现。它几乎无法告诉你该模型在你的特定任务、特定数据以及特定故障模式 (Failure Modes) 下的表现。如果你正在构建客服机器人，MMLU 毫无参考价值。如果你正在构建代码助手，HumanEval 仅涵盖函数级别的代码生成——它完全无法反映跨文件调试、重构或代码解释的能力。

你需要自定义评测。这并不是因为基准测试毫无用处——它们在初步筛选模型时确实有效——而是因为最终的评估必须与你的实际部署条件完全一致。

## 核心概念

### 评估（Evaluation）全景

评估主要分为三类，每类的成本与信号质量（signal quality）各不相同。

**基准测试（Benchmarks）**是标准化的测试套件，例如 MMLU、HumanEval、SWE-bench、MATH、ARC 和 HellaSwag。将模型在基准测试上运行即可得到分数。其优势在于：所有人使用相同的测试，便于模型间横向对比。劣势在于：模型和训练数据正日益污染这些基准测试。实验室使用的训练数据中包含了基准测试的题目。分数上去了，但实际能力未必提升。

**自定义评估（Custom evals）**是针对特定用例构建的测试套件。你需要定义输入、预期输出以及评分函数。例如，法律文档摘要模型应在法律文档上进行评估；SQL 生成器应在你的数据库模式（schema）上进行评估。这类评估构建成本高昂，但却是唯一能预测生产环境（production）表现的评估方式。

**人工评估（Human evals）**通过付费标注员，根据有用性、正确性、流畅度和安全性等标准对模型输出进行评判。对于自动化评分失效的开放式任务，这是黄金标准。Chatbot Arena 已收集了超过 200 万条针对 100 多个模型的人类偏好投票。其缺点在于：成本高（每次评判 0.10-2.00 美元）且速度慢（耗时数小时至数天）。

graph TD
    subgraph Eval["Evaluation Landscape"]
        direction LR
        B["Benchmarks\n(MMLU, HumanEval)\nCheap, standardized\nGameable, stale"]
        C["Custom Evals\nYour task, your data\nHighest signal\nExpensive to build"]
        H["Human Evals\n(Chatbot Arena)\nGold standard\nSlow, costly"]
    end

    B -->|"rough model selection"| C
    C -->|"ambiguous cases"| H

    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff

### 基准测试为何失效

三种机制导致基准测试分数不再反映真实能力。

**数据污染（Data contamination）。**训练语料库（corpora）从互联网上抓取数据，而基准测试题目也存在于互联网上。模型在训练过程中“见过”答案。这并非传统意义上的作弊——实验室并非有意将基准数据纳入训练。但网络规模的抓取使得几乎无法将其完全排除。

**应试训练（Teaching to the test）。**实验室会针对基准测试表现优化训练数据配比。如果训练集中有 5% 是 MMLU 风格的选择题，模型就会学习其格式和答案分布。MMLU 是四选一选择题。模型会学到 A/B/C/D 的答案分布大致均匀，这即使在模型不知道正确答案时也能提供帮助。

**分数饱和（Saturation）。**当所有前沿模型在某基准测试上的得分都达到 85-90% 时，该测试就失去了区分度。剩余 10-15% 的题目可能存在歧义、标签错误，或需要冷门的领域知识。在 MMLU 上从 87% 提升到 89%，可能仅仅意味着模型多记住了两道冷门题目，而非变得更聪明。

### 困惑度（Perplexity）：快速健康检查

困惑度衡量模型对一系列词元（tokens）的“惊讶”程度。形式上，它是平均负对数似然（negative log-likelihood）的指数：

PPL = exp(-1/N * sum(log P(token_i | context)))

困惑度为 10 意味着，平均而言，模型在每个词元位置上的不确定性相当于从 10 个选项中随机选择。数值越低越好。GPT-2 在 WikiText-103 上的困惑度约为 30，GPT-3 约为 20，Llama 3 8B 约为 7。

困惑度适用于在同一测试集上对比模型，但也存在盲区。模型可能因为擅长预测常见模式而获得低困惑度，却在罕见但重要的模式上表现糟糕。此外，它无法反映指令遵循（instruction following）、推理或事实准确性。应将其用作健全性检查（sanity check），而非最终定论。

### 大模型即裁判（LLM-as-Judge）

使用强模型来评估弱模型的输出。思路很简单：让 GPT-4o 或 Claude Sonnet 对回答的正确性、有用性和安全性进行 1-5 分打分。使用 GPT-4o-mini 时，每次评判成本约 0.01 美元，且与人类评判的相关性出人意料地高——在大多数任务上的一致性约为 80%。

评分提示词（prompt）比模型本身更重要。模糊的提示词（如“给这个回答打分”）会产生噪声较大的分数。而带有明确评分标准（rubric）的结构化提示词（如“若答案事实正确且引用来源得 5 分，正确但无来源得 4 分，部分正确得 3 分……”）则能产出一致且可复现的分数。

失效模式：裁判模型会表现出位置偏差（position bias，在成对比较中偏好第一个回答）、冗长偏差（verbosity bias，偏好更长的回答）以及自我偏好（self-preference，GPT-4 给 GPT-4 输出的打分高于同等质量的 Claude 输出）。缓解措施：随机化顺序、对长度进行归一化处理、使用与被评估模型不同的裁判模型。

### 基于成对比较的 ELO 评分

这是 Chatbot Arena 采用的方法。向人类（或大模型裁判）展示不同模型对同一提示词的两个回答，由裁判选出更优者。基于成千上万次此类比较，为每个模型计算 ELO 评分——与国际象棋使用的系统相同。

ELO 的优势：相对排名比绝对评分更可靠，能妥善处理平局，且相比独立评估每个输出，只需更少的比较次数即可收敛。截至 2026 年初，Chatbot Arena 排行榜显示，GPT-4o、Claude 3.5 Sonnet 和 Gemini 1.5 Pro 在顶部的 ELO 分差均在 20 分以内。

graph LR
    subgraph ELO["ELO Rating Pipeline"]
        direction TB
        P["Prompt"] --> MA["Model A Output"]
        P --> MB["Model B Output"]
        MA --> J["Judge\n(Human or LLM)"]
        MB --> J
        J --> W["A Wins / B Wins / Tie"]
        W --> E["ELO Update\nK=32"]
    end

    style P fill:#1a1a2e,stroke:#0f3460,color:#fff
    style J fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#51cf66,color:#fff

### 评估框架

**lm-evaluation-harness**（EleutherAI）：标准的开源评估框架。支持 200 多个基准测试。只需一条命令即可将任意 Hugging Face 模型在 MMLU、HellaSwag、ARC 等测试上运行。Open LLM Leaderboard 即采用此框架。

**RAGAS**：专为检索增强生成（RAG）流水线设计的评估框架。主要衡量忠实度（答案是否与检索到的上下文一致？）、相关性（检索到的上下文是否与问题相关？）以及答案正确性。

**promptfoo**：面向提示词工程（prompt engineering）的配置驱动型评估工具。在 YAML 中定义测试用例，针对多个模型运行，并生成通过/失败报告。适用于提示词的回归测试——确保提示词的修改不会破坏现有测试用例。

### 构建自定义评估

这是唯一对生产环境有意义的评估。构建流程如下：

1. **定义任务。**模型究竟需要做什么？务必精确。“回答问题”过于模糊。“给定一封客户投诉邮件，提取产品名称、问题类别和情感倾向”才是可评估的任务。

2. **创建测试用例。**原型评估至少需要 50 个，生产环境需 200 个以上。每个测试用例是一个（输入，预期输出）对。需包含边界情况：空输入、对抗性输入、歧义输入、其他语言输入。

3. **定义评分标准。**结构化输出使用精确匹配（Exact match）。文本相似度使用 BLEU/ROUGE。开放式质量使用大模型即裁判。抽取任务使用 F1 分数。可通过加权组合多个指标。

4. **自动化。**每次评估均通过一条命令运行，无需人工干预。将结果存储为支持随时间对比的格式。

5. **持续追踪。**孤立的评估分数毫无意义，你需要的是趋势线。上次修改提示词后分数是否提升？切换模型后是否出现性能回退？将评估版本与提示词版本同步管理。

| 评估类型 | 单次评判成本 | 与人类一致性 | 最佳适用场景 |
|-----------|------------------|----------------------|----------|
| 精确匹配（Exact match） | ~0 美元 | 100%（适用时） | 结构化输出、分类任务 |
| BLEU/ROUGE | ~0 美元 | ~60% | 翻译、摘要生成 |
| 大模型即裁判（LLM-as-judge） | ~0.01 美元 | ~80% | 开放式生成 |
| 人工评估（Human eval） | 0.10-2.00 美元 | 不适用（即真实基准） | 歧义任务、高风险任务 |

## 开始构建

### 步骤 1：最小化评估框架 (Minimal Eval Framework)

定义核心抽象概念。一个评估用例（eval case）包含输入文本、预期输出以及可选的元数据字典（metadata dict）。评分器（scorer）接收模型预测结果与参考答案，并返回一个介于 0 到 1 之间的分数。

import json
from collections import Counter

class EvalCase:
    def __init__(self, input_text, expected, metadata=None):
        self.input_text = input_text
        self.expected = expected
        self.metadata = metadata or {}

class EvalSuite:
    def __init__(self, name, cases, scorers):
        self.name = name
        self.cases = cases
        self.scorers = scorers

    def run(self, model_fn):
        results = []
        for case in self.cases:
            prediction = model_fn(case.input_text)
            scores = {}
            for scorer_name, scorer_fn in self.scorers.items():
                scores[scorer_name] = scorer_fn(prediction, case.expected)
            results.append({
                "input": case.input_text,
                "expected": case.expected,
                "prediction": prediction,
                "scores": scores,
            })
        return results

### 步骤 2：评分函数 (Scoring Functions)

构建精确匹配（exact match）、词元 F1（token F1）以及模拟的大语言模型裁判（LLM-as-judge）评分器。

def exact_match(prediction, expected):
    return 1.0 if prediction.strip().lower() == expected.strip().lower() else 0.0

def token_f1(prediction, expected):
    pred_tokens = set(prediction.lower().split())
    exp_tokens = set(expected.lower().split())
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = pred_tokens & exp_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def llm_judge_simulated(prediction, expected):
    pred_words = set(prediction.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    overlap = len(pred_words & exp_words) / len(exp_words)
    length_penalty = min(1.0, len(prediction) / max(len(expected), 1))
    return round(overlap * 0.7 + length_penalty * 0.3, 3)

### 步骤 3：ELO 评分系统 (ELO Rating System)

实现基于 ELO 更新机制的成对比较（pairwise comparisons）。这正是 Chatbot Arena 用于对模型进行排名的系统。

class ELOTracker:
    def __init__(self, k=32, initial_rating=1500):
        self.ratings = {}
        self.k = k
        self.initial_rating = initial_rating
        self.history = []

    def _ensure_player(self, name):
        if name not in self.ratings:
            self.ratings[name] = self.initial_rating

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def record_match(self, player_a, player_b, outcome):
        self._ensure_player(player_a)
        self._ensure_player(player_b)

        ea = self.expected_score(self.ratings[player_a], self.ratings[player_b])
        eb = 1 - ea

        if outcome == "a":
            sa, sb = 1.0, 0.0
        elif outcome == "b":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        self.ratings[player_a] += self.k * (sa - ea)
        self.ratings[player_b] += self.k * (sb - eb)

        self.history.append({
            "a": player_a, "b": player_b,
            "outcome": outcome,
            "rating_a": round(self.ratings[player_a], 1),
            "rating_b": round(self.ratings[player_b], 1),
        })

    def leaderboard(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1])

### 步骤 4：困惑度计算 (Perplexity Calculation)

使用词元概率（token probabilities）计算困惑度（perplexity）。在实际应用中，这些数据通常直接从模型的逻辑值（logits）中获取。此处我们通过概率分布进行模拟。

import numpy as np

def perplexity(log_probs):
    if not log_probs:
        return float("inf")
    avg_neg_log_prob = -np.mean(log_probs)
    return float(np.exp(avg_neg_log_prob))

def token_log_probs_simulated(text, model_quality=0.8):
    np.random.seed(hash(text) % 2**31)
    tokens = text.split()
    log_probs = []
    for i, token in enumerate(tokens):
        base_prob = model_quality
        if len(token) > 8:
            base_prob *= 0.6
        if i == 0:
            base_prob *= 0.7
        prob = np.clip(base_prob + np.random.normal(0, 0.1), 0.01, 0.99)
        log_probs.append(float(np.log(prob)))
    return log_probs

### 步骤 5：汇总结果 (Aggregate Results)

计算单次评估运行的汇总统计信息：均值、中位数、达到特定阈值的通过率（pass rate），以及各指标的细分数据。

def summarize_results(results, threshold=0.8):
    all_scores = {}
    for r in results:
        for metric, score in r["scores"].items():
            all_scores.setdefault(metric, []).append(score)

    summary = {}
    for metric, scores in all_scores.items():
        arr = np.array(scores)
        summary[metric] = {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "pass_rate": round(float(np.mean(arr >= threshold)), 3),
            "n": len(scores),
        }
    return summary

def print_summary(summary, suite_name="Eval"):
    print(f"\n{'=' * 60}")
    print(f"  {suite_name} Summary")
    print(f"{'=' * 60}")
    for metric, stats in summary.items():
        print(f"\n  {metric}:")
        print(f"    Mean:      {stats['mean']:.3f}")
        print(f"    Median:    {stats['median']:.3f}")
        print(f"    Std:       {stats['std']:.3f}")
        print(f"    Range:     [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    Pass rate: {stats['pass_rate']:.1%} (threshold >= 0.8)")
        print(f"    N:         {stats['n']}")

### 步骤 6：运行完整流水线 (Run the Full Pipeline)

将所有组件串联起来。定义任务、创建测试用例、模拟两个模型、运行评估、基于成对比较计算 ELO 分数，并打印排行榜（leaderboard）。

def demo_model_good(prompt):
    responses = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Who wrote Hamlet?": "William Shakespeare",
        "What language is PyTorch written in?": "Python and C++",
        "What is the boiling point of water?": "100 degrees Celsius",
    }
    return responses.get(prompt, "I don't know")

def demo_model_bad(prompt):
    responses = {
        "What is the capital of France?": "Paris is the capital city of France",
        "What is 2 + 2?": "The answer is four",
        "Who wrote Hamlet?": "Shakespeare",
        "What language is PyTorch written in?": "Python",
        "What is the boiling point of water?": "212 Fahrenheit",
    }
    return responses.get(prompt, "Unknown")

cases = [
    EvalCase("What is the capital of France?", "Paris"),
    EvalCase("What is 2 + 2?", "4"),
    EvalCase("Who wrote Hamlet?", "William Shakespeare"),
    EvalCase("What language is PyTorch written in?", "Python and C++"),
    EvalCase("What is the boiling point of water?", "100 degrees Celsius"),
]

suite = EvalSuite(
    name="General Knowledge",
    cases=cases,
    scorers={
        "exact_match": exact_match,
        "token_f1": token_f1,
        "llm_judge": llm_judge_simulated,
    },
)

results_good = suite.run(demo_model_good)
results_bad = suite.run(demo_model_bad)

print_summary(summarize_results(results_good), "Model A (concise)")
print_summary(summarize_results(results_bad), "Model B (verbose)")

“优秀”模型会给出精确答案，而“较差”模型则会给出冗长的改写表述。精确匹配指标会对冗长模型进行严厉惩罚，而词元 F1 和 LLM 裁判指标则相对宽容。这充分说明了指标选择的重要性：同一个模型的表现是优异还是糟糕，完全取决于你采用的评分方式。

### 步骤 7：ELO 锦标赛 (ELO Tournament)

在多轮次中运行模型间的成对比较。

elo = ELOTracker(k=32)

for case in cases:
    pred_a = demo_model_good(case.input_text)
    pred_b = demo_model_bad(case.input_text)

    score_a = token_f1(pred_a, case.expected)
    score_b = token_f1(pred_b, case.expected)

    if score_a > score_b:
        outcome = "a"
    elif score_b > score_a:
        outcome = "b"
    else:
        outcome = "tie"

    elo.record_match("model_a_concise", "model_b_verbose", outcome)

print("\nELO Leaderboard:")
for name, rating in elo.leaderboard():
    print(f"  {name}: {rating:.0f}")

### 步骤 8：困惑度对比 (Perplexity Comparison)

对比不同质量水平“模型”的困惑度。

test_text = "The quick brown fox jumps over the lazy dog in the garden"

for quality, label in [(0.9, "Strong model"), (0.7, "Medium model"), (0.4, "Weak model")]:
    log_probs = token_log_probs_simulated(test_text, model_quality=quality)
    ppl = perplexity(log_probs)
    print(f"  {label} (quality={quality}): perplexity = {ppl:.2f}")


## 使用方法

### lm-evaluation-harness (EleutherAI)

用于在任何模型上运行基准测试（Benchmark）的标准工具。

# pip install lm-eval
# Command line:
# lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks mmlu --batch_size 8

# Python API:
# import lm_eval
# results = lm_eval.simple_evaluate(
#     model="hf",
#     model_args="pretrained=meta-llama/Llama-3.1-8B",
#     tasks=["mmlu", "hellaswag", "arc_easy"],
#     batch_size=8,
# )
# print(results["results"])

### promptfoo

面向提示词工程（Prompt Engineering）的配置驱动型评估（Evaluation）工具。支持在 YAML 中定义测试用例，并针对多个模型提供商（Provider）运行测试。

# promptfoo.yaml
providers:
  - openai:gpt-4o-mini
  - anthropic:claude-3-haiku

prompts:
  - "Answer in one word: {{question}}"

tests:
  - vars:
      question: "What is the capital of France?"
    assert:
      - type: contains
        value: "Paris"
  - vars:
      question: "What is 2 + 2?"
    assert:
      - type: equals
        value: "4"

### 用于检索增强生成（RAG）评估的 RAGAS

# pip install ragas
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
#
# result = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision],
# )
# print(result)

RAGAS 能够衡量通用评估工具所忽略的指标：模型的回答是否严格基于检索到的上下文（Context），而不仅仅是抽象意义上的“正确性”。

## 交付上线

本章节将生成 `outputs/prompt-eval-designer.md` 文件——这是一个可复用的提示词（Prompt），能够为任意任务设计自定义评估套件（Eval Suite）。只需输入任务描述，它即可自动生成测试用例、评分函数以及通过/失败阈值的推荐值。

同时还会生成 `outputs/skill-evaluation.md` 文件——这是一个决策框架，可根据你的任务类型、预算和延迟（Latency）要求，帮助你选择最合适的评估策略。

## 练习

1. 添加一个“一致性 (consistency)”评分器，将相同输入传递给模型运行 5 次，并测量输出结果匹配的频率。在确定性输入 (deterministic inputs) 上出现不一致的回答，通常表明提示词 (prompts) 设计脆弱或温度参数 (temperature) 设置过高。

2. 扩展 ELO 追踪器 (ELO tracker)，使其支持多种评判函数 (judge functions)（如精确匹配 (exact match)、F1 分数 (F1)、大语言模型作为裁判 (LLM-as-judge)），并为它们分配权重。对比在侧重精确匹配与侧重 F1 分数时，排行榜 (leaderboard) 的变化情况。

3. 为特定任务构建一个评估套件 (eval suite)：将电子邮件分类为 5 个类别。创建 100 个测试用例，包含多样化的示例及边界情况 (edge cases)（如可能属于多个类别的邮件、空白邮件、其他语言的邮件）。测量不同“模型”（基于规则 (rule-based)、关键词匹配 (keyword matching)、模拟大语言模型 (simulated LLM)）的表现。

4. 实现数据污染检测 (contamination detection)：给定一组评估问题 (eval questions) 和训练语料库 (training corpus)，检查评估问题（或其近似改写 (close paraphrases)）在训练数据中出现的百分比。这是研究人员审计基准测试 (benchmark) 有效性的标准方法。

5. 构建一个“模型差异对比 (model diff)”工具。给定两个模型版本的评估结果，高亮显示哪些具体测试用例有所提升、哪些出现退化、哪些保持不变。这相当于代码差异对比 (code diff) 在评估领域的对应物——对于理解某项改动是带来增益还是造成损害至关重要。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| MMLU | “基准测试” | 大规模多任务语言理解 (Massive Multitask Language Understanding) —— 涵盖 57 个学科的 15,908 道选择题，到 2025 年得分率已饱和至 88% 以上 |
| HumanEval | “代码评估” | 来自 OpenAI 的 164 个 Python 函数补全问题，仅测试独立函数的生成能力 |
| SWE-bench | “真实编程评估” | 来自 12 个 Python 仓库的 2,294 个 GitHub 问题 (issues)，用于衡量端到端的缺陷修复能力（包括测试用例生成） |
| 困惑度 (Perplexity) | “模型有多困惑” | exp(-avg(log P(token_i given context))) —— 值越低，表示模型对实际词元 (token) 赋予的概率越高 |
| ELO 评分 (ELO rating) | “模型的象棋排名” | 根据两两胜负记录计算出的相对技能评分，Chatbot Arena 使用它来对 100 多个模型进行排名 |
| 大语言模型作为裁判 (LLM-as-judge) | “用 AI 给 AI 打分” | 由能力较强的模型根据评分标准对较弱模型的输出进行打分，与人类裁判的一致性约为 80%，每次评判成本约 0.01 美元 |
| 数据污染 (Data contamination) | “模型见过考题” | 训练数据中包含了基准测试题目，导致分数虚高，但并未真正提升模型能力 |
| 评估套件 (Eval suite) | “一堆测试” | 用于衡量特定能力的版本化集合，包含（输入、预期输出、评分器）三元组 |
| 通过率 (Pass rate) | “答对的比例” | 得分超过阈值的评估用例所占比例——比平均分更具可操作性，因为它衡量的是可靠性 |
| Chatbot Arena | “模型排名网站” | LMSYS 平台，拥有超过 200 万次人类偏好投票，通过 ELO 评分生成最受信任的大语言模型排行榜 |

## 进一步阅读

- [Hendrycks 等人，2021 -- "Measuring Massive Multitask Language Understanding"](https://arxiv.org/abs/2009.03300) -- MMLU 论文，尽管该基准测试（Benchmark）已趋于饱和，但仍是引用量最高的大语言模型（Large Language Model）评估基准
- [Chen 等人，2021 -- "Evaluating Large Language Models Trained on Code"](https://arxiv.org/abs/2107.03374) -- OpenAI 发布的 HumanEval 论文，确立了代码生成（Code Generation）的评估方法体系
- [Zheng 等人，2023 -- "Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685) -- 系统分析了使用大语言模型作为裁判（LLM-as-a-Judge）来评估其他模型的方法，并揭示了位置偏差（Position Bias）与冗长偏差（Verbosity Bias）等关键发现
- [LMSYS Chatbot Arena](https://chat.lmsys.org/) -- 拥有超 200 万次投票的众包（Crowdsourced）模型对比平台，也是目前业界最值得信赖的真实场景大语言模型排行榜