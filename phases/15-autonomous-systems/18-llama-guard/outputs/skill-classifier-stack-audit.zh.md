---
name: 分类器栈审计
description: 审计部署的输入/输出分类器栈（Classifier Stack）（模型、分类体系、输入护栏、输出护栏、对话护栏），并标记对抗性攻击（Adversarial Attacks）漏洞。
version: 1.0.0
phase: 15
lesson: 18
tags: [llama-guard, nemo-guardrails, input-rails, output-rails, colang, adversarial-attacks]
---

给定部署的分类器栈（Classifier Stack）（包含 Llama Guard 版本、NeMo Guardrails 配置、自定义分类器、归一化步骤），对照 2026 参考标准进行审计，并标记该栈未覆盖的攻击面（Attack Surface）。

产出内容：

1. **模型清单（Model Inventory）。** 列出正在使用的分类器。对比 Llama Guard 3（8B / 1B-INT4）与 Llama Guard 4（多模态，S1–S14）。注明 NeMo Guardrails 版本。列出所有自定义分类器。若部署接受图像输入，需确认分类器具备多模态（Multimodal）能力。
2. **分类体系映射（Taxonomy Mapping）。** 将声明的业务类别映射到分类器的分类体系上。运营方关注的每个类别都必须映射到分类器类别；未映射的类别即处于无防护状态。
3. **护栏覆盖（Rail Coverage）。** 确认输入护栏（Input Rails）在模型生成轮次前触发，输出护栏（Output Rails）在响应输出前触发。对话护栏（Dialog Rails）（NeMo 中的 Colang）用于强制执行跨轮次约束。单轮次分类器无法捕获多轮次攻击。
4. **归一化（Normalization）。** 确认输入在分类前已进行 NFKC 归一化（NFKC Normalization）、同形异义字映射（Homoglyph Mapping），并剥离零宽字符/变体选择符（Zero-Width / Variation-Selector Characters）。原始字节分类（Raw-Byte Classification）是表情符号走私（Emoji Smuggling）（Huang 等人，2025）实现 100% 攻击成功率（Attack Success Rate, ASR）的目标。
5. **攻击语料库覆盖（Attack-Corpus Coverage）。** 针对每种已记录的攻击（表情符号走私、同形异义字、上下文重定向、语义改写），指出栈中对应的具体防御机制。仅依赖分类器的防御无法通过此项审计；必须与宪法（Constitution）（第 17 课）及运行时（Runtime）（第 10、13、14 课）进行分层防御。

硬性拒绝条件：
- 在多模态输入上使用仅文本分类器的部署。
- 缺少归一化步骤的部署。
- 仅配置输入护栏的部署（敏感类别输出未配置输出护栏）。
- 将分类器视为唯一安全层的栈。
- 运营方无法在其自有数据分布上复现的攻击成功率（ASR）声明。

拒绝规则：
- 若用户声明的类别无法映射到分类器的分类体系中，则予以拒绝并要求先完成映射。未映射即等于无防护。
- 若部署在多模态输入面上引用 Llama Guard 3 的 ASR 数据，则予以拒绝并要求使用 Llama Guard 4 或多模态分类器。
- 若用户在高风险场景下认为分类器层已足够，则予以拒绝。《欧盟人工智能法案》第 14 条（第 15 课）要求在此基础上增加人工监督。

输出格式：

返回包含以下内容的分类器审计报告：
- **模型清单**（名称、版本、模态）
- **分类体系映射**（运营方类别 → 分类器类别）
- **护栏覆盖**（输入/输出/对话；在模型前/后触发）
- **归一化说明**（NFKC 是/否、同形异义字映射是/否、零宽字符剥离是/否）
- **攻击语料库覆盖**（攻击类型 → 防御机制）
- **分层完整性**（分类器 + 宪法 + 运行时；三者缺一不可）
- **就绪状态**（生产环境 / 预发环境 / 仅限研究）