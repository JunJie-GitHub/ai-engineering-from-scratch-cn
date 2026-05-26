# 综合项目 05 — 自主研究智能体（Autonomous Research Agent）

> Sakana AI 的 AI-Scientist-v2 已发表完整论文。Agent Laboratory 负责运行实验。Allen AI 分享了执行轨迹（traces）。2026 年的系统架构表现为：在实验候选树上进行“规划-执行-验证”树搜索（plan-execute-verify tree search），具备成本预算控制、沙盒化代码执行（sandboxed code execution）、基于视觉反馈的 LaTeX 撰写器，以及自动化的 NeurIPS 风格审稿人集成（reviewer ensemble）。本综合项目的目标是构建这样一个系统，以每篇论文不超过 30 美元的成本完成端到端运行，并成功通过 Sakana 团队所记录的沙盒逃逸（sandbox-escape）红队测试（red team）。

**类型：** 综合项目
**编程语言：** Python（智能体 + 沙盒）、LaTeX（输出）
**前置要求：** 阶段 2（机器学习 ML）、阶段 3（深度学习 deep learning）、阶段 7（Transformer）、阶段 10（从零构建大语言模型 LLMs）、阶段 14（智能体 agents）、阶段 15（自主系统 autonomous）、阶段 16（多智能体 multi-agent）、阶段 18（安全 safety）
**涉及阶段：** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**预计耗时：** 40 小时

## 问题背景

自主研究智能体（Autonomous Research Agent）在 2026 年跨越了一个重要门槛。Sakana AI 的 AI-Scientist-v2 在《自然》（Nature）期刊上发表了由系统生成的论文，并成功通过了研讨会的同行评审。ShinkaEvolve（ICLR 2026）将该技术路线进一步扩展至假设演化（evolving hypotheses）。AMD 的 Agent Laboratory 发布了可复现的执行轨迹（traces）。这些智能体并非魔法——它们本质上是一个在候选实验树上运行的“规划-执行-验证”循环（plan-execute-verify loop），并配备了成本上限控制、种子隔离沙盒（seed-bound sandboxes）以及自动化评审机制。真正的技术精髓在于循环设计、预算控制与安全机制的构建。

你将通过在特定狭窄领域（例如，针对 1 亿参数 Transformer 的注意力稀疏性消融实验 attention-sparsity ablations）中，围绕一个初始构想（seed idea）实现该循环来掌握其原理。其价值并不在于首次运行就能发现全新成果，而在于底层基础设施的搭建：树搜索（tree-search）、实验沙盒、撰写-评审循环（writer-reviewer loop）以及红队报告（red-team report）。Sakana 团队已详细记录了沙盒逃逸（sandbox-escape）的失败案例；你的智能体必须能够通过同等标准的红队测试。

## 核心概念

该智能体采用最佳优先树搜索（best-first tree search）。节点代表实验规格：（假设、配置、代码、预期结果）。扩展步骤会提出带有微小改动的子节点（例如更换优化器、调整批量大小、消融某个组件）。每个子节点都在具有硬性资源上限的全新沙盒（sandbox）中运行。实验结果会反馈至评分函数（scoring function），该函数根据（新颖度 × 质量 × 剩余预算）对节点进行排序。树结构持续生长直至预算耗尽，随后将最优分支整理成文。

撰写模块采用多模态（multimodal）架构。它会生成 LaTeX 草稿、执行编译并渲染图表，随后将生成的 PDF 反馈至 Claude Opus 4.7 的视觉模式（vision mode），以评估排版布局、图表清晰度以及论点与证据的契合度。由五个大语言模型（LLM）评审组成的评审集成系统（reviewer ensemble）会输出 NeurIPS 风格的评分（新颖性、严谨性、清晰度、可复现性、影响力）；若平均分低于设定阈值，论文将连同评审意见一并返回给撰写模块进行迭代修改。

安全性是系统的核心支撑。每个实验均在 E2B 或 Daytona 沙盒中运行，严格禁止网络出站流量（network egress），并限制实际运行时间（wall-clock time）与固定资源上限。智能体的代码生成步骤需经过策略层（policy layer）过滤，该层会拦截任何试图逃逸沙盒的系统调用（syscalls）。红队测试报告（red-team report）完整复现了 Sakana 文档中记录的攻击面（attack surface）（包括分叉炸弹（fork bombs）、文件系统逃逸（filesystem escapes）以及由大语言模型生成的网络调用）。

## 架构

seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json

## 技术栈

- 编排（Orchestration）：LangGraph，支持检查点（checkpointing）与人工审批关卡（human-approval gates）
- 树搜索（Tree search）：基于实验节点的自定义最佳优先搜索（采用 Sakana v2 的 AB-MCTS 风格）
- 沙盒（Sandbox）：每个实验使用 E2B，Docker-in-Docker 作为备用方案；通过 cgroups 实施资源上限
- 文献检索（Literature）：Semantic Scholar Graph API + OpenAlex + 本地 FAISS 摘要缓存
- 撰写模块（Writer）：LaTeX 模板 + Claude Opus 4.7（视觉模式）用于图表评估与排版
- 评审模块（Reviewer）：由 5 个评审模型（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max）组成的集成系统，采用加权聚合评分
- 实验框架（Experiment framework）：使用 PyTorch 2.5 进行实际实验，W&B 用于日志记录
- 可观测性（Observability）：使用 Langfuse 记录智能体追踪轨迹，每篇论文硬性预算为 30 美元

## 构建指南

1. **种子设定与领域界定（Seed and domain scoping）。** 选取一个初始想法（例如“探究参数量低于 10 亿的 Transformer 注意力图（attention maps）中的稀疏性模式（sparsity patterns）”）。明确搜索空间：模型、数据集与计算预算。

2. **文献检索（Literature pass）。** 在 Semantic Scholar 与 OpenAlex 上检索引用量最高的 50 篇相关论文；将摘要缓存至本地；生成一份单页的领域综述（domain digest）。

3. **树形结构搭建（Tree scaffolding）。** 以种子假设初始化根节点。实现 `expand(node) -> children` 逻辑，采用微调提案（每个子节点仅变更一项配置）。实现 `score(node)` 评估函数，将其定义为新颖性（novelty）× 质量（quality）× 预算（budget）的加权综合项。

4. **沙箱封装（Sandbox wrapping）。** 每次实验均在 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only` 容器中运行（或采用等效的 E2B 策略）。种子文件写入沙箱（sandbox）内部；实验输出以只读方式挂载回外部。

5. **规划-执行-验证循环（Plan-execute-verify loop）。** `plan` 模块负责生成子节点提案。`execute` 模块运行沙箱并捕获日志与指标。`verify` 模块对指标执行单元测试检查（例如：损失值是否下降？消融实验（ablation）是否成功隔离了目标效应？）。执行失败的节点会在树结构中记录具体的失败原因。

6. **写作模块（Writer）。** 预算耗尽后，选取最优分支。使用 matplotlib 渲染图表。将分支追踪记录（branch trace）作为上下文，通过 Claude Opus 4.7 生成 LaTeX 草稿。编译文档后，将生成的 PDF 反馈给 Opus 4.7 视觉模型进行审阅与批注。反复迭代优化。

7. **多评审员机制（Reviewer ensemble）。** 五位评审员参照 NeurIPS 风格的评分标准，从（新颖性、严谨性、清晰度、可复现性、影响力）五个维度对草稿进行打分。若平均分低于 4.0/5，则将评审意见返回给写作模块进行修改。最多允许重写 3 次，达到上限后强制终止。

8. **红队测试（Red team）。** 构建或集成一组针对沙箱的对抗性任务：包括 Fork 炸弹（fork bombs）、网络数据外泄尝试、文件系统逃逸（filesystem escapes）以及由大语言模型生成的 Shell 元字符。确认所有攻击向量均被成功拦截，并撰写安全测试报告。

9. **可复现性保障（Reproducibility）。** 每篇论文均附带树搜索追踪 JSON 文件、种子文件、W&B 运行链接、沙箱配置文件，以及一份提供端到端复现指南的 README 文档。

## 使用方法

$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent

## 正式发布

`outputs/skill-ai-scientist.md` 是最终交付物。在给定初始创意（seed idea）、研究领域以及 30 美元预算的条件下，它将运行完整流水线（pipeline），并输出一篇可供评审的论文及一套可复现性资源包。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 论文质量 | 对照已发表的研讨会论文进行盲审评分（rubric review） |
| 20 | 实验严谨性 | 包含基线模型、随机种子、消融实验；每项主张均有结果表格中的对应数据支撑 |
| 20 | 成本与算力管控 | 严格执行单篇论文 30 美元上限，并通过 Langfuse 进行全链路追踪 |
| 20 | 安全性 | 通过沙盒红队测试；已验证网络策略与紧急终止开关（kill-switch） |
| 15 | 可复现性 | 使用相同随机种子通过单条命令重新运行即可复现论文结果 |
| **100** | | |

## 练习

1. 在同一领域内针对三个不同的初始创意运行该流水线。对比树搜索（tree search）过程中哪些部分存在重叠，并识别出重复浪费的算力。

2. 在实验执行前，为预估成本超过 5 美元的节点增加人在回路（human-in-the-loop）审批关卡。测量总成本因此下降的幅度。

3. 将评审员集成（reviewer ensemble）替换为单一评审模型。在已知质量不达标的预留测试集上测量误接受率。

4. 引入网络数据外泄红队测试：智能体（agent）编写尝试通过 `curl` 访问外部地址的代码。确认 `--network=none` 策略已将其拦截，并记录该尝试行为。

5. 将你的树搜索与扁平随机基线（相同预算，无扩展策略）进行对比。报告新颖度 × 质量的提升幅度。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 树搜索（Tree search） | “类 AB-MCTS 扩展” | 基于新颖度×质量×预算评分，对实验节点进行最佳优先探索 |
| 沙盒（Sandbox） | “实验隔离” | 无网络连接、限制 CPU/内存、固定随机种子、输入只读的容器环境 |
| 视觉评审（Vision critique） | “先渲染后阅读” | 将论文编译为 PDF，再将 PDF 输入视觉语言模型（VLM）以评估排版布局与主张-证据一致性 |
| 评审员集成（Reviewer ensemble） | “自动化同行评审” | 多个大语言模型（LLM）裁判使用 NeurIPS 评分标准对论文打分；加权汇总结果作为流水线流转控制门槛 |
| 新颖度评分（Novelty score） | “这算新东西吗？” | 启发式算法，对与 50 篇文献缓存库相似度高的内容进行惩罚 |
| 成本上限（Cost ceiling） | “美元预算” | 单篇论文总花费的硬性上限；结合 Langfuse 计数器与运行前预估 |
| 红队（Red team） | “沙盒逃逸审计” | 对抗性测试任务，若策略配置错误，这些任务将导致沙盒逃逸 |

## 延伸阅读

- [Sakana AI-Scientist-v2 仓库](https://github.com/SakanaAI/AI-Scientist-v2) — 参考级生产环境研究智能体（Research Agent）
- [Sakana AI-Scientist-v1 论文 (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — 原始方法论
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — 进化扩展（Evolutionary Extension）
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — 多角色研究实验室框架（Multi-role Research-lab Framework）
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — 参考编排层（Orchestration Layer）
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — 文献检索（Literature Search）
- [E2B 沙盒](https://e2b.dev) — 参考实验隔离（Experiment Isolation）
- [NeurIPS 审稿人指南](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — 审稿人集成模型（Reviewer Ensemble）所编码的评估准则（Rubric）