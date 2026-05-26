# 为什么需要多智能体 (Multi-Agent)？

> 单个智能体遇到瓶颈时，明智的做法不是打造一个更大的智能体，而是使用更多的智能体。

**类型：** 学习
**语言：** TypeScript
**前置条件：** 第 14 阶段（智能体工程）
**预计时间：** 约 60 分钟

## 学习目标

- 识别单智能体瓶颈 (Single-Agent Ceiling)（上下文溢出、混合专业技能、串行瓶颈），并解释在何种情况下拆分为多个智能体是更优选择
- 比较编排模式 (Orchestration Patterns)（流水线 Pipeline、并行扇出 Parallel Fan-out、主管模式 Supervisor、层级模式 Hierarchical），并为给定的任务结构选择合适的模式
- 设计具有明确角色边界、共享状态和通信契约 (Communication Contract) 的多智能体系统
- 分析多智能体复杂性（延迟、成本、调试难度）与单智能体简洁性之间的权衡

## 问题所在

你在第 14 阶段构建了一个单智能体。它运行良好，能够读取文件、执行命令、调用 API 并对结果进行推理。但当你将它应用于真实的代码库时：200 个文件、三种编程语言、依赖基础设施的测试，以及在编写代码前需要调研外部 API 的需求。

智能体开始“卡壳”。这并非因为大语言模型 (LLM) 不够聪明，而是因为该任务超出了单个智能体循环的处理能力。上下文窗口 (Context Window) 被文件内容填满。智能体会忘记 40 次工具调用前读取的内容。它试图同时扮演研究员、程序员和代码审查员的角色，结果三者都表现不佳。

这就是单智能体瓶颈。每当任务需要满足以下条件时，你就会遇到它：

- **超出单个窗口容量的上下文** - 读取 50 个文件会轻松突破 20 万 token 的限制
- **不同阶段需要不同的专业技能** - 调研所需的提示词 (Prompting) 与代码生成截然不同
- **可并行执行的工作** - 既然可以同时读取，为何还要按顺序读取三个文件？

## 核心概念

### 单智能体天花板 (Single-Agent Ceiling)

单个智能体 (Single Agent) 就是一个循环、一个上下文窗口 (Context Window) 和一个系统提示词 (System Prompt)。想象一下：

┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘

三个问题随之出现：

1. **上下文饱和 (Context Saturation)**——工具调用结果不断堆积。到第 30 轮对话时，智能体已经消耗了 15 万 Token 的文件内容、命令输出和先前的推理过程。第 5 轮的关键细节早已丢失。

2. **角色混淆 (Role Confusion)**——如果系统提示词写着“你既是研究员、程序员，又是代码审查员和测试员”，最终产出的智能体只会半吊子地做研究和写代码，永远无法完成审查工作。

3. **串行瓶颈 (Sequential Bottleneck)**——智能体先读取文件 A，再读文件 B，最后读文件 C。三次串行的 LLM 调用，三次串行的工具执行。毫无并行能力。

### 多智能体解决方案 (Multi-Agent Solution)

拆分工作。为每个智能体分配单一任务、独立的上下文窗口，以及针对该任务专门调优的系统提示词：

┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘

每个智能体具备：
- 专注的系统提示词（“你是一名代码审查员。你的唯一职责就是查找 Bug。”）
- 独立的上下文窗口（不会被其他智能体的工作污染）
- 清晰的输入/输出契约（接收研究笔记，输出代码）

### 实际应用此架构的系统

**Claude Code 子智能体 (Subagents)**——当 Claude Code 使用 `Task` 生成子智能体时，会创建一个具有明确任务范围的子智能体。父智能体保持上下文整洁，子智能体执行专注任务并返回摘要。

**Devin**——运行规划智能体 (Planner Agent)、编码智能体 (Coder Agent) 和浏览器智能体 (Browser Agent)。规划智能体将工作拆解为步骤，编码智能体负责编写代码，浏览器智能体负责查阅文档。每个智能体拥有独立的上下文。

**多智能体编程团队 (SWE-bench)**——在 SWE-bench 上表现顶尖的系统通常采用：负责阅读代码库的研究员智能体、负责设计修复方案的规划智能体，以及负责具体实现的编码智能体。单智能体系统的得分普遍较低。

**ChatGPT Deep Research**——并行生成多个搜索智能体，各自探索不同方向，最后综合汇总结果。

### 架构光谱

多智能体 (Multi-Agent) 并非非黑即白，而是一个连续的光谱：

SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles

**单智能体 (Single Agent)**——单一循环，单一提示词。适用于简单任务。

**子智能体 (Subagents)**——父智能体为专注的子任务生成子智能体。父智能体维护整体计划，子智能体完成后汇报。这正是 Claude Code 的工作方式。

**流水线 (Pipeline)**——智能体按顺序运行。智能体 A 的输出作为智能体 B 的输入。适用于分阶段工作流：研究 -> 编码 -> 审查 -> 测试。

**团队 (Team)**——智能体通过共享消息总线 (Message Bus) 并行运行。每个智能体有明确角色，由编排器 (Orchestrator) 进行协调。适用于需要同时调用不同技能的场景。

**蜂群 (Swarm)**——大量相同或高度相似的智能体共享状态。没有固定的编排器。智能体从队列中自主领取任务。适用于高吞吐量的并行任务。

### 四种多智能体模式

#### 模式 1：流水线 (Pipeline)

Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)

每个智能体对数据进行转换并传递给下一个。逻辑清晰易于推理。但任一阶段失败都会阻塞后续流程。

#### 模式 2：扇出/扇入 (Fan-out / Fan-in)

                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘

将工作拆分给多个并行智能体，最后合并结果。适用于可拆解为独立子任务的工作。

#### 模式 3：编排器-工作者 (Orchestrator-Worker)

                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘

智能编排器负责决策、将任务委派给工作者，并综合最终结果。编排器本身也是一个智能体，具备生成工作者智能体的工具。

#### 模式 4：对等蜂群 (Peer Swarm)

         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘

没有中心编排器。智能体之间进行点对点 (Peer-to-Peer) 通信。决策通过交互自然涌现。调试难度较高，但能轻松扩展至大量智能体。

### 何时不应使用多智能体

多智能体会增加系统复杂度。智能体之间的每条消息都可能成为故障点。调试工作从“阅读一段对话”变成了“追踪五个智能体之间的消息流转”。

**在以下情况保持单智能体：**
- 任务能容纳在单个上下文窗口内（工作数据约低于 10 万 Token）
- 不同阶段不需要使用不同的系统提示词
- 串行执行的速度已足够快
- 任务足够简单，拆分带来的额外开销大于其收益

**复杂度代价：**
- 每个智能体边界都是一次有损压缩步骤：智能体 A 的完整上下文会被压缩摘要成一条消息传给智能体 B
- 协调逻辑（谁在何时以何种顺序做什么）本身就会引入新的 Bug
- 延迟增加：N 个智能体意味着至少 N 次串行 LLM 调用，如果需要来回交互则更多
- 成本成倍增加：每个智能体都会独立消耗 Token

经验法则：如果任务所需的工具调用少于 20 次，且能容纳在 10 万 Token 以内，请坚持使用单智能体。

## 构建

### 步骤 1：过载的单智能体 (Single Agent)

这是一个试图包揽所有任务的单智能体。它仅依赖一个庞大的系统提示词（System Prompt）和一个同时承载研究资料、代码与审查意见的上下文窗口（Context Window）：

type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}

该方案存在的问题：
- 上下文窗口会随着每个阶段的推进而不断膨胀。到了代码审查阶段，其中已混杂了研究笔记、代码以及先前的推理过程。
- 系统提示词过于通用，无法针对每个阶段进行专项优化。
- 所有任务均串行执行，无法并行处理。

### 步骤 2：专家智能体 (Specialist Agents)

现在将其拆分。每个智能体只负责一项特定任务：

type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);

每个专家智能体都拥有高度聚焦的提示词，并且各自获得一个干净的上下文窗口，其中仅包含其所需的输入信息。

### 步骤 3：通过消息进行协调

通过显式的消息传递（Message Passing）机制将各专家智能体连接起来：

type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}

每个智能体仅接收发给自己的消息，彻底避免了上下文污染（Context Pollution）。例如，研究智能体读取的 5 万 Token 的文档资料永远不会进入审查智能体的上下文中。

### 步骤 4：对比分析

async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}

多智能体架构的总 Token 消耗量更高（涉及三个智能体及三次独立的大语言模型（LLM）调用），但每个智能体的上下文始终保持干净。由于系统提示词实现了专业化定制，每个阶段的输出质量均得到显著提升。

## 实践应用

本课程将生成一个可复用的提示词（prompt），用于判断何时应采用多智能体（multi-agent）架构。请参阅 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个专家角色：一个“测试员”智能体（agent），它接收来自编码员的代码和评审员的反馈，然后编写测试用例
2. 修改流水线（pipeline），使评审员能够将反馈发回给编码员以进行修订循环（最多 2 轮）
3. 将顺序流水线转换为扇出（fan-out）模式：并行运行研究员和“需求分析员”智能体，然后在将结果传递给编码员之前合并它们的输出

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 群体（Swarm） | “AI 智能体的群体思维” | 一组具有共享状态且无固定领导者的对等智能体。其行为由局部交互涌现而出。 |
| 编排器（Orchestrator） | “老板智能体” | 一种智能体，其工具集包含创建和管理其他智能体。它负责规划与任务分发，但通常不直接执行具体工作。 |
| 协调器（Coordinator） | “交通警察” | 一种非智能体组件（通常仅为代码而非大语言模型（LLM）），负责根据规则在智能体之间路由消息。 |
| 共识（Consensus） | “智能体达成一致” | 一种协议，要求多个智能体在继续执行前必须达成一致。常用于解决输出冲突的场景。 |
| 涌现行为（Emergent behavior） | “智能体自己摸索出来的” | 由智能体交互产生的系统级模式，但并非显式编程设定。可能有益，也可能有害。 |
| 扇出/扇入（Fan-out / fan-in） | “智能体的 Map-Reduce” | 将任务拆分给多个并行智能体处理（扇出），随后合并它们的结果（扇入）。 |
| 消息传递（Message passing） | “智能体相互对话” | 智能体之间的通信机制：将结构化数据从一个智能体发送至另一个智能体，以此替代共享上下文窗口（context windows）。 |

## 延伸阅读

- [新兴 AI 智能体架构全景](https://arxiv.org/abs/2409.02977) - 多智能体模式综述
- [AutoGen：赋能下一代大语言模型应用](https://arxiv.org/abs/2308.08155) - 微软的多智能体对话框架
- [Claude Code 子智能体文档](https://docs.anthropic.com/en/docs/claude-code) - Claude Code 如何通过 Task 进行任务委派
- [CrewAI 文档](https://docs.crewai.com/) - 基于角色的多智能体框架