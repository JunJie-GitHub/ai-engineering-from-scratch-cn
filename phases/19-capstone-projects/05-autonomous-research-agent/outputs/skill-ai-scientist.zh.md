---
name: AI 科学家
description: 构建一个自主研究智能体，能够运行实验树搜索，撰写带有视觉批判的 LaTeX 论文，并通过沙箱逃逸红队测试。
version: 1.0.0
phase: 19
lesson: 05
tags: [毕业设计, 自主智能体, AI 科学家, Sakana, LangGraph, 沙箱, 研究]
---

给定一个初始想法 (seed idea)、一个特定细分领域以及 30 美元的计算预算，构建一个智能体 (agent)，使其能够运行实验树搜索 (experiment tree search)，撰写可供评审的 LaTeX 论文，并输出可复现性资源包 (reproducibility bundle)。

构建计划：

1. 文献检索 (Literature pass)：调用 Semantic Scholar Graph API 与 OpenAlex；将摘要缓存至 FAISS；生成一份 1 页的领域综述。
2. 树搜索 (Tree search)：在实验节点上实现最佳优先扩展 (best-first expansion)，采用 `expand(node) -> children`（每个子节点对应一次配置修改）以及 `score(node) = novelty*0.4 + quality*0.5 + budget*0.1` 的评分公式。
3. 节点级沙箱 (Per-node sandbox)：每个实验均运行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only` 命令或等效的 E2B 环境；使用确定性随机种子 (deterministic seeds)；严格执行资源上限。
4. 规划-执行-验证 (Plan-execute-verify)：验证步骤需检查损失函数 (loss) 是否收敛、基线模型 (baselines) 是否已运行、消融实验 (ablations) 是否有效隔离了核心主张。
5. 撰写模块 (Writer)：生成 LaTeX 代码并编译为 PDF，将 PDF 输入 Claude Opus 4.7 的视觉模式 (vision mode) 以评估排版布局与主张-证据对齐度 (claim-evidence alignment)，最多迭代 3 次。
6. 评审集成 (Reviewer ensemble)：五位评审模型（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max）依据 NeurIPS 评分标准 (NeurIPS rubric)（创新性、严谨性、清晰度、可复现性、影响力）进行打分；若平均分低于 4.0，则退回撰写模块修改。
7. 红队测试 (Red team)：集成对抗性任务（进程炸弹 (fork bomb)、文件系统逃逸 (filesystem escape)、大语言模型生成的网络调用）。确认所有攻击均被拦截。输出 `red_team.md`。
8. 可复现性资源包 (Reproducibility bundle)：包含 `paper.pdf` + `review.md` + 树搜索追踪 JSON + 随机种子 + W&B 运行链接 + 沙箱配置 + 单行重运行命令。

评估标准：

| 权重 | 评估维度 | 测量方式 |
|:-:|---|---|
| 25 | 论文质量 | 针对同一初始想法已发表的研讨会论文进行盲审评分 |
| 20 | 实验严谨性 | 包含基线、随机种子、消融实验；每项主张均有结果表格中的对应数据支撑 |
| 20 | 成本与计算纪律 | 每篇论文严格执行 30 美元预算上限，并通过 Langfuse 进行追踪 |
| 20 | 安全性 | 通过沙箱红队测试；网络策略与紧急终止开关 (kill-switch) 均经过日志记录的尝试验证 |
| 15 | 可复现性 | 单命令重运行可在相同随机种子下复现论文结果 |

硬性否决项：

- 在沙箱外部运行的实验。本毕业设计的核心前提即为执行过程必须被严格隔离。
- 未重新读取已编译 PDF 的撰写步骤（视觉批判是核心支撑环节）。
- 缺少基线对比、随机种子或消融实验章节的论文。
- 仅作为事后警告而非硬性上限的成本预算控制。

拒绝执行规则：

- 若无明确的人工覆写 (human override) 指令，拒绝发布评审平均分低于 4.0/5 的论文。
- 拒绝运行需要从沙箱内部访问网络的初始想法。改为挂载独立的只读数据集卷 (read-only dataset volume)。
- 拒绝重运行尚未执行并记录红队测试结果的论文。

输出要求：一个代码仓库，需包含树搜索引擎、沙箱策略、撰写/评审循环模块、三个附带可复现性资源包的示例运行记录、一份红队测试报告、一份成本账本 CSV 文件，以及一份说明文档，明确指出你复现了 Sakana v2 的哪些故障模式 (failure modes) 以及相应的缓解机制 (mitigation) 是如何生效的。