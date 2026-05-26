---
name: 语法流水线
description: 为下游自然语言处理（Natural Language Processing, NLP）任务设计经典的词性标注（Part-of-Speech, POS）与依存句法分析（Dependency Parsing）流水线。
version: 1.0.0
phase: 5
lesson: 07
tags: [自然语言处理, 词性标注, 句法分析]
---

给定一个下游任务（信息抽取、重写验证、查询分解、词形还原），你需要输出以下内容：

1. 标签集（Tagset）。仅支持英语的传统流水线使用宾州树库（Penn Treebank），多语言或跨语言场景使用通用依存关系（Universal Dependencies, UD）。
2. 库（Library）。大多数生产环境使用 spaCy（`en_core_web_sm` / `_lg` / `_trf`），学术级多语言场景使用 stanza，追求最高 UD 准确率则使用 trankit。
3. 集成代码片段（Integration Snippet）。调用该库并提取 `.pos_`、`.dep_`、`.head` 属性的 3 到 5 行代码。
4. 待测试的失败模式（Failure Mode）。名动兼类歧义（如 `saw`、`book`、`can`）和介词短语附着（Prepositional Phrase Attachment, PP-attachment）歧义是经典陷阱。抽样 20 条输出结果进行人工目测检查。

拒绝推荐自行开发解析器（Parser）。从零构建解析器属于研究课题，而非工程应用任务。将任何未处理大小写变体就直接消费词性标注的流水线标记为脆弱（Fragile）设计。