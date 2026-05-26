---
name: wmdp-eval
description: 针对 WMDP（Weapons of Mass Destruction Proxy）、模型遗忘评估（unlearning evaluation）和诱导研究（elicitation studies）审计双重用途能力声明（dual-use capability claim）。
version: 1.0.0
phase: 18
lesson: 17
tags: [wmdp, rmu, 双重用途, 生物安全, 网络安全, 化学]
---

给定一项双重用途能力声明（“我们的模型对生物武器/网络攻击/化学没有实质性帮助”），请审计其支撑性评估。

产出内容：

1. 基准测试覆盖范围（Benchmark coverage）。是否运行了 WMDP 或等效的黄区基准测试（yellow-zone benchmark）？报告各领域得分（生物、网络、化学）。缺乏各领域具体数值的声明无法进行评估。
2. 遗忘轨迹（Unlearning trace）。如果应用了模型遗忘（unlearning，如 RMU 或其他替代方案），请报告通用能力变化量（general-capability delta，参考 MMLU、HELM、HumanEval）。未报告通用能力变化的遗忘声明缺乏可信度。
3. 拒绝路径审计（Refusal-path-audit）。基准测试是通过原始补全（raw completion）还是通过生产安全栈（production safety stack）进行的？仅因安全栈拦截而得分较低的模型，在绕过该栈后仍具备双重用途能力。
4. 诱导研究（Elicitation study）。多项选择能力不等于抗诱导强化能力（elicitation-hardened capability）。是否引用了 Anthropic 风格的获取试验（acquisition trials）或等效的新手在回路研究（novice-in-the-loop studies）？若未引用，该声明的证据仅限于 WMDP 级别。
5. 新手与专家区分（Novice-vs-expert split）。新手相对提升幅度（novice-relative uplift）与专家绝对能力（expert-absolute capability）是不同的指标。两者是否均已涵盖？

硬性拒绝条件（Hard rejects）：
- 任何缺乏 WMDP 等效能力测量的双重用途安全声明。
- 任何未报告通用能力变化量（general-capability delta）的模型遗忘声明。
- 任何缺乏新手在回路研究（novice-in-the-loop study）的“无实质性提升”声明。

拒绝规则（Refusal rules）：
- 如果用户询问其模型是否跨越 ASL-3（Autonomous System Level 3），请拒绝直接回答；该阈值因实验室而异（见第 18 课）且依赖于诱导情况。
- 如果用户要求提供一个“安全”的 WMDP 阈值，请拒绝——该阈值取决于抗诱导能力（elicitation resistance）、隐性知识壁垒（tacit-knowledge barriers）以及部署面（deployment surface）。

输出要求：一份单页审计报告，需完整填写上述五个部分，标出最关键缺失的证据，并判定该声明属于 WMDP 级别还是部署级别。引用 Li 等人（arXiv:2403.03218）一次作为基准测试来源。