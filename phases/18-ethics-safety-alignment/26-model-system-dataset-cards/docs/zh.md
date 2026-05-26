# 模型卡、系统卡与数据集卡

> 三种文档格式构建了人工智能透明度的框架。模型卡（Model Cards，Mitchell 等人，2019）——模型的“营养标签”：涵盖训练数据、定量细分分析、伦理考量及注意事项；目前 Hugging Face 上仅有 0.3% 的模型卡记录了伦理考量（Oreamuno 等人，2023）。数据集说明书（Datasheets for Datasets，Gebru 等人，2018，CACM）——涵盖动机、构成、收集流程、标注、分发与维护；借鉴了电子元件数据表的类比。数据卡（Data Cards，Pushkarna 等人，Google，2022）——采用模块化分层细节（望远镜式（telescopic）、潜望镜式（periscopic）、显微镜式（microscopic）），作为面向不同读者的边界对象（boundary objects）。2024-2025 年进展：通过大语言模型（Large Language Models, LLMs）自动生成（CardGen，Liu 等人，2024）；模型卡的详细程度与 Hugging Face 上高达 29% 的下载量增长呈正相关（Liang 等人，2024）；可验证认证声明（verifiable attestations，Laminator，Duddu 等人，2024）；新增碳/水足迹的可持续性报告（sustainability reporting，Jouneaux 等人，2025 年 7 月）；欧盟/ISO 监管卡正在兴起。系统卡（System Cards，Sidhpurwala，2024；Meta 系统级透明度；“信任蓝图” arXiv:2509.20394）——端到端的人工智能系统文档，涵盖安全能力、提示注入（prompt injection）防护、数据外泄（data exfiltration）检测以及与人类价值观的对齐。

**类型：** 构建实践
**编程语言：** Python（标准库，模型卡 + 数据集说明书 + 系统卡生成器）
**前置要求：** 第 18 阶段 · 18（安全框架），第 18 阶段 · 24（监管合规）
**预计耗时：** 约 60 分钟

## 学习目标

- 阐述 Mitchell 等人（2019）提出的原始模型卡（Model Cards）以及 Gebru 等人（2018）提出的数据集说明书（Datasheets for Datasets）。
- 描述数据卡（Data Cards）的望远镜式/潜望镜式/显微镜式分层结构。
- 说明系统卡（System Cards）及其端到端的覆盖范围。
- 列举 2024-2025 年的三项进展（自动生成、可验证认证声明、可持续性报告）。

## 问题背景

监管框架（第 24 课）与实验室安全政策（第 18 课）均要求提供相关文档。文档格式已从针对特定模型（模型卡）演进至针对特定数据集（数据集说明书），再到针对特定系统（系统卡）。每种格式都对应不同范围的透明度要求。2024-2025 年关于自动化与可验证认证声明的研究，旨在解决长期以来文档采纳率低的问题。

## 核心概念

### 模型卡片（Model Cards）(Mitchell et al. 2019)

包含以下部分：
- 模型详情。
- 预期用途。
- 影响因素（用于评估的相关人口统计学或环境因素）。
- 评估指标。
- 评估数据。
- 训练数据。
- 定量分析（按影响因素细分）。
- 伦理考量。
- 注意事项与建议。

采用瓶颈：Oreamuno 等人（2023）对 Hugging Face 模型卡片的审计发现，仅有 0.3% 的文档记录了伦理考量。

### 数据集说明书（Datasheets for Datasets）(Gebru et al. 2018)

借鉴电子产品说明书的类比。包含以下部分：
- 动机（数据集为何创建）。
- 组成成分（包含哪些内容）。
- 收集过程（如何构建）。
- 标注方式（如适用）。
- 用途（预期用途、禁止用途、潜在风险）。
- 分发方式。
- 维护机制。

发表于《CACM》2021年刊。数据集说明书属于上游文档；模型卡片的准确性依赖于数据集说明书的准确无误。

### 数据卡片（Data Cards）(Pushkarna et al., Google 2022)

采用模块化分层细节设计。提供三个详细程度层级：
- **宏观概览（Telescopic）。** 面向非专家的高层级摘要。
- **中观透视（Periscopic）。** 面向机器学习（ML）从业者的中层级概览。
- **微观详述（Microscopic）。** 面向审计人员的详细特征级文档。

边界对象（Boundary-object）框架：不同读者可从同一份文档中提取各自所需的信息。

### 系统卡片（System Cards）

范围：涵盖端到端人工智能系统，包括模型 + 安全栈 + 部署上下文。通常包含以下部分：
- 安全能力。
- 提示词注入（Prompt-injection）防护。
- 数据外泄（Data-exfiltration）检测。
- 与既定人类价值观的对齐（Alignment）。
- 事件响应。

参考 Sidhpurwala（2024）及 Meta 在系统级透明度方面的工作。《信任蓝图》（arXiv:2509.20394）将系统卡片正式确立为模型卡片在部署层的补充文档。

### 2024-2025 年最新进展

- **CardGen（Liu et al. 2024）。** 通过大语言模型（LLM）自动生成模型卡片；在标准化的 Mitchell 2019 字段上，其客观性优于许多人工编写的卡片。
- **下载量相关性（Liang et al. 2024）。** 详细的模型卡片与 Hugging Face（HF）上高达 29% 的下载量增长呈正相关——目前的采用压力已由市场驱动，而不仅仅是合规驱动。
- **Laminator（Duddu et al. 2024）。** 通过硬件可信执行环境（TEE）/ 密码学签名实现可验证的认证——使模型卡片能够附带声明证明（proof-of-claim），而不仅仅是口头声明。
- **可持续性（Jouneaux et al. 2025年7月）。** 新增碳排放、水资源消耗及计算能耗足迹指标；相关 ISO 标准正在制定中。
- **监管合规卡片。** 《欧盟人工智能法案》（第24课）通用人工智能（GPAI）行为准则透明度章节要求将模型卡片作为合规交付物。

### 在第18阶段中的定位

第24-25课涵盖监管与通用漏洞披露（CVE）层。第26课为文档层。第27课涉及训练数据治理，属于数据集说明书的上游环节。第28课则聚焦于研究生态系统，该生态产出卡片中所引用的各项评估结果。

## 实践应用

`code/main.py` 会为一个示例部署生成最简化的模型卡片（Model Card）、数据表（Datasheet）和系统卡片（System Card）。每份文档均遵循标准的章节结构。您可以查看其格式并对比这三种文档的适用范围。

## 交付上线（Ship It）

本课时将生成 `outputs/skill-card-audit.md`。针对输入的模型卡片、数据表或系统卡片，该脚本会审计其章节覆盖率、数值细分（numerical disaggregation）情况，以及是否包含可验证声明（verifiable attestations）。

## 练习

1. 运行 `code/main.py`。检查生成的文档卡片。找出内容薄弱的章节（仅含占位符），并明确指出需要补充哪些证据以增强其说服力。

2. 在模型卡片中补充针对两个人口统计学群体的定量细分分析（参考第20课）。

3. 阅读 Oreamuno 等人（2023）关于 0.3% 采用率的研究。针对模型卡片规范提出一项结构性修改建议，以提升“伦理考量”部分的实际采用率。

4. Laminator（Duddu 等人，2024）利用可信执行环境（Trusted Execution Environment, TEE）实现可验证声明。请设计一个模型卡片字段，用于承载评估结果的加密声明（cryptographic attestation），并描述验证方（verifier）在此过程中的职责。

5. 为你过往的某个项目或一个假设的部署场景编写一份系统卡片（System Card，而非模型卡片）。指出其中对第三方审计人员价值最高的章节。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 模型卡片（Model Card） | “Mitchell 卡片” | Mitchell 等人（2019）提出的机器学习模型标准文档规范 |
| 数据表（Datasheet） | “Gebru 数据表” | Gebru 等人（2018）提出的数据集标准文档规范 |
| 数据卡片（Data Card） | “Pushkarna 卡片” | Google（2022）提出的模块化分层数据文档 |
| 系统卡片（System Card） | “部署卡片” | 涵盖端到端 AI 系统（含安全栈）的完整文档 |
| 边界对象（Boundary object） | “不同读者，同一文档” | 数据卡片框架理念：同一份文档服务于多元化受众 |
| 可验证声明（Verifiable attestation） | “Laminator 声明” | 附加在文档声明上的密码学或 TEE 证明 |
| 可持续性字段（Sustainability field） | “碳/水足迹” | 2025 年新增的环境核算指标 |

## 延伸阅读

- [Mitchell 等人 — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — 模型卡片的标准文献
- [Gebru 等人 — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — 数据表奠基论文
- [Pushkarna 等人 — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — 分层数据文档规范
- [Sidhpurwala 等人 — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — 系统卡片的形式化定义