---
name: ecosystem-map
description: 将对齐（Alignment）声明或评估（Evaluation）映射至相关机构、方法论及交叉验证（Cross-checks）。
version: 1.0.0
phase: 18
lesson: 28
tags: [mats, redwood, apollo, metr, eleos, 生态系统]
---

给定一项对齐声明或评估，将其来源映射至研究生态系统中，并识别交叉验证机会。

产出内容：

1. 来源识别。该声明由哪家机构提出（实验室、MATS、Redwood、Apollo、METR、Eleos 或学术实验室）？
2. 方法论风格。该研究是否符合该机构已记录的方法论风格——Redwood 的控制协议（Control Protocols）、Apollo 的三支柱谋划（Three-pillar Scheming）、METR 的任务视界（Task-horizon）或 Eleos 的福利评估（Welfare）？
3. 对标机构。还有哪些机构在研究相邻问题？它们是否发表过互补或相悖的结果？
4. 多机构协作信号。该论文是单一实验室的成果还是联合发表（例如 Apollo + OpenAI、Redwood + Anthropic）？多机构合作的论文通常具有更高的外部可信度。
5. 发表渠道。仅限 arXiv 的预印本、NeurIPS/ICML/ICLR 会议论文集、实验室博客，还是监管提交文件？发表渠道是衡量审查严格程度的信号。

硬性拒绝条件：
- 任何未明确产出机构的对齐声明。
- 任何缺乏外部复现（Replication）或验证的单一机构安全声明。
- 任何忽略 MATS 人才输送管道（Talent-pipeline）结构的生态图谱。

拒绝规则：
- 若用户询问“哪家研究机构最值得信赖”，请拒绝进行排名，并引导其关注多机构复现结果。
- 若用户询问生态系统内部的政治动态，请予以拒绝，并严格围绕已发表的方法论进行讨论。

输出要求：一份单页图谱，需完整填写上述五个部分，列出交叉验证机会，并指出最强证据与最强反驳论点。