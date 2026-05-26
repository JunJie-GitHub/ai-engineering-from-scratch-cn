# 数据溯源（Data Provenance）与训练数据治理（Training-Data Governance）

> 《欧盟人工智能法案》（EU AI Act）要求通用人工智能模型（GPAI）在2025年8月前采用机器可读的退出（opt-out）标准（依据《欧盟版权指令》文本与数据挖掘（TDM）例外条款）。加州AB 2013法案（2024年签署）——生成式AI训练数据透明度要求开发者发布包含12个必填字段的数据集摘要。2025年数据保护机构（DPA）在合法利益（legitimate interest）上的立场趋同：爱尔兰数据保护委员会（DPC，2025年5月21日）在欧洲数据保护委员会（EDPB）出具意见后，认可Meta在采取保障措施的前提下，使用第一方公开的欧盟/欧洲经济区成人内容训练大语言模型（LLM）；科隆高等地方法院（2025年5月23日）驳回禁令申请；汉堡数据保护机构取消紧急措施；英国信息专员办公室（ICO，2025年9月23日）对LinkedIn的AI训练保障措施（透明度、简化的退出机制、延长的异议窗口期）作出积极的监管回应并持续监控——但这并非正式批准。巴西国家数据保护局（ANPD，2024年7月2日）曾因信息透明度不足暂停Meta的数据处理活动；在Meta提交合规计划后，该预防性措施于2024年8月30日解除。核心不可逆问题：Cookie同意框架专为实时、可逆的追踪而设计；数据一旦融入模型权重（model weights），精准擦除便不再可能——对于已训练的神经网络，GDPR的删除权（right-to-erasure）在实践中无法落实。合规窗口期位于数据采集阶段。数据溯源倡议（Data Provenance Initiative，dataprovenance.org，Longpre、Mahari、Lee等人，《危机中的同意》，2024年7月）的大规模审计显示，随着内容发布商添加robots.txt限制，AI数据共享池（AI data commons）正在迅速萎缩。

**类型：** 学习
**语言：** Python（标准库，12字段加州AB 2013脚手架生成器）
**前置要求：** 第18阶段 · 24（监管），第18阶段 · 26（模型卡片）
**时长：** 约60分钟

## 学习目标

- 阐述加州AB 2013法案针对生成式AI训练数据透明度规定的12个必填字段。
- 说明2025年数据保护机构（DPA）关于基于合法利益训练大语言模型（LLM）的立场（爱尔兰DPC、英国ICO、汉堡、科隆）。
- 解释不可逆问题：为何GDPR的删除权（right-to-erasure）在已训练的神经网络中缺乏实际对应的执行机制。
- 陈述数据溯源倡议（Data Provenance Initiative）《危机中的同意》报告的核心发现。

## 核心问题

训练数据治理（Training-data governance）是构建每个模型卡片（第26课）和履行监管义务（第24课）的上游基础。在2024至2025年间，监管格局已围绕三大原则达成共识：退出（opt-out）基础设施、按数据集披露，以及针对公开数据的合法利益（legitimate interest）豁免机制。若提供商在数据采集阶段未能合规，后续将无法进行补救。

## 核心概念

### 加州 AB 2013 法案 (California AB 2013)

该法案于 2024 年签署。对于 2022 年 1 月 1 日及之后发布的系统，相关文档必须在 2026 年 1 月 1 日或之前发布。第 3111(a) 条要求开发者发布训练所用数据集的高层摘要，包含 12 项法定内容：
1. 数据集的来源或所有者。
2. 描述数据集如何促进实现 AI 系统的预期目的。
3. 数据集中的数据点数量（可接受大致范围；动态数据集可提供估算值）。
4. 描述数据点的类型（标注数据集的标签类型；未标注数据集的一般特征）。
5. 数据集是否包含受版权、商标或专利保护的数据，或是否完全属于公共领域。
6. 数据集是否为购买或授权获取。
7. 数据集是否包含个人信息（依据《加州民法典》第 1798.140(v) 条）。
8. 数据集是否包含汇总的消费者信息（依据《加州民法典》第 1798.140(b) 条）。
9. 开发者进行的清洗、处理或其他修改操作及其预期目的。
10. 数据收集的时间段，若收集仍在进行需予以说明。
11. 数据集在开发过程中首次使用的日期。
12. 系统是否使用或持续使用合成数据生成 (synthetic data generation)。

第 12 项（合成数据）相较于 Gebru 等人 2018 年的数据表 (datasheets) 是新增内容。第 7 项（个人信息）会触发《加州隐私权法案》(California Privacy Rights Act, CPRA) 的相关义务。该法案豁免了安全/完整性、航空器运行以及仅限联邦的国家安全系统（第 3111(b) 条）。

### 欧盟《人工智能法案》(EU AI Act)（第 24 课）与文本与数据挖掘 (Text and Data Mining, TDM) 退出机制

欧盟《版权指令》(EU Copyright Directive) 中的文本与数据挖掘例外条款允许使用公开可用内容进行训练，除非权利人选择退出。欧盟《人工智能法案》通用人工智能 (General Purpose AI, GPAI) 行为准则的版权章节要求 GPAI 提供商遵守机器可读的退出信号（如 robots.txt、C2PA 的“禁止 AI 训练”声明等）。

### 2025 年数据保护机构 (Data Protection Authority, DPA) 在合法利益 (legitimate interest) 上的趋同

爱尔兰数据保护委员会 (Data Protection Commission, DPC)（2025 年 5 月 21 日）：在欧洲数据保护委员会 (European Data Protection Board, EDPB) 发表意见后，Meta 利用第一方公开的欧盟/欧洲经济区成年用户内容进行训练的计划在附加保障措施的前提下获得认可。科隆高等地方法院（2025 年 5 月 23 日）驳回了对 Meta 的禁令：退出机制已足够。汉堡数据保护机构出于欧盟范围内一致性的考虑，取消了紧急程序。英国信息专员办公室 (Information Commissioner's Office, ICO)（2025 年 9 月 23 日）对 LinkedIn 在类似保障措施和持续监控下恢复 AI 训练一事作出了积极的监管回应——但这并非正式批准。

趋同原则：在提供退出机制的前提下，合法利益可作为使用公开可用的第一方内容进行训练的正当理由。无需征得同意。

### 巴西国家数据保护局 (Autoridade Nacional de Proteção de Dados, ANPD)（2024 年 6 月）

因信息透明度不足，暂停了 Meta 处理巴西用户数据用于 AI 训练的行为。结果与欧盟各数据保护机构不同——ANPD 将透明度置于合法利益的可接受性之上。

### 不可逆性问题 (The irreversibility problem)

Cookie 同意机制 (Cookie-consent) 是为实时、可逆的跟踪而设计的。训练数据则不同：一旦数据融入模型权重 (model weights)，就无法进行精确擦除。从头重新训练是唯一彻底的补救措施，但其成本极其高昂。

部分补救措施：
- **机器遗忘 (Machine Unlearning)**。近似移除；通过成员推理攻击 (Membership Inference Attack, MIA) 进行评估（第 22 课）。
- **基于影响函数 (Influence Function) 的定位**。识别受数据影响最大的权重；进行选择性更新。
- **微调抑制 (Fine-tune-suppression)**。训练模型拒绝生成源自该数据的输出。

这些方法均无法彻底解决问题。合规的关键窗口期在于数据收集阶段。

### 数据溯源倡议 (Data Provenance Initiative)

dataprovenance.org。Longpre、Mahari、Lee 等人发表的《危机中的同意》(Consent in Crisis)（2024 年 7 月）：对 AI 训练数据共享池进行的大规模审计。研究发现：出版商添加 robots.txt 限制的速度正在加快。可供公开训练的数据共享池正在迅速萎缩。从 2023 年到 2024 年，约 25% 的顶级训练来源增加了某种限制。启示：未来训练数据的可用性将取决于新的获取范式（授权许可、合成生成、激励性参与）。

### 本内容在第 18 阶段 (Phase 18) 中的定位

第 26 课侧重于模型级文档 (model-level documentation)。第 27 课侧重于数据集级治理 (dataset-level governance)。两者共同定义了透明度层 (transparency layer)。第 28 课则梳理了致力于解决这些问题的研究生态。

## 使用方法

`code/main.py` 会针对示例数据集生成一份符合加州 AB 2013 法案（California AB 2013）要求的 12 字段数据集摘要模板。你可以填写各字段内容，并观察哪些字段会触发隐私或版权相关的后续合规义务。

## 交付产出

本课时将生成 `outputs/skill-provenance-check.md`。针对用于训练的指定数据集，该文件会检查 AB 2013 法案的 12 字段覆盖情况、退出机制（opt-out）基础设施的合规性、与数据保护协议（Data Protection Agreement, DPA）的对齐程度，以及模型不可逆性（irreversibility）风险评估。

## 练习

1. 运行 `code/main.py`。为示例数据集生成 12 字段摘要，并找出哪些字段定义不够明确（under-specified）。
2. 欧盟版权指令中的文本与数据挖掘（Text and Data Mining, TDM）退出机制是机器可读的。请提出一种退出信号（opt-out signal）的标准格式，并将其与 `robots.txt` 以及 C2PA 的“No AI Training”声明进行对比。
3. 阅读数据溯源倡议组织（Data Provenance Initiative）发布的《危机中的同意》（Consent in Crisis，2024 年 7 月）。描述限制速度最快的三个内容类别，并论述其中一项经济影响。
4. 2025 年的数据保护协议（DPA）对齐方案接受将“合法利益”（legitimate interest）作为公开内容训练的依据。请构建一个“合法利益”不足以支撑训练的场景，并指出服务提供商此时需要依赖的其他法律依据。
5. 草拟一份训练数据溯源清单（training-data-provenance manifest），使其能够与 AB 2013 字段以及每个数据集的 C2PA 签名溯源链相兼容。指出其中一项技术障碍和一项法律障碍。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| AB 2013 | “加州那部法律” | 生成式 AI 训练数据透明度法案；强制要求填写 12 个字段 |
| TDM 例外（TDM exception） | “文本与数据挖掘” | 欧盟版权指令中的训练数据例外条款，附带退出机制（opt-out） |
| 合法利益（Legitimate interest） | “欧盟的法律依据” | 《通用数据保护条例》（GDPR）第 6 条规定的法律依据，可作为在公开内容上进行训练的正当理由 |
| 退出信号（Opt-out signal） | “机器可读的禁止训练标记” | `robots.txt`、C2PA 的“No AI Training”声明、TDM.Reservation |
| 不可逆性（Irreversibility） | “无法撤销训练” | 模型权重中的数据无法通过外科手术式的方法精准移除 |
| 机器遗忘（Unlearning） | “近似移除” | 训练后采取的干预措施，旨在降低模型对特定数据的依赖程度 |
| 《危机中的同意》（Consent in Crisis） | “DPI 审计报告” | 数据溯源倡议组织（DPI）于 2024 年 7 月发布的报告，指出 `robots.txt` 限制正在加速收紧 |

## 延伸阅读

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — 生成式 AI 训练数据透明度法案
- [EU AI Act + GPAI Code of Practice (Lesson 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — 版权相关章节
- [Longpre, Mahari, Lee et al. — Consent in Crisis (dataprovenance.org, July 2024)](https://www.dataprovenance.org/consent-in-crisis-paper) — DPI 审计报告
- [IAPP — EU Digital Omnibus GDPR amendments (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — 监管背景解读