---
name: 社会风险审查
description: 使用CAIS四风险框架（Four-Risk Framework）及CAISI/SB-53监管背景，审查部署的社会级风险态势。
version: 1.0.0
phase: 15
lesson: 22
tags: [cais, caisi, 四风险框架, 组织风险, sb-53, 社会风险]
---

针对拟议或已运行的AI部署（AI Deployment），生成一份社会级风险审查（Societal-Scale-Risk Review）报告。该报告需根据CAIS四风险框架对部署进行标记，盘点组织风险（Organizational Risk）的子杠杆，并明确适用的监管范围（Regulatory Surface）。

输出内容：

1. **四风险标记（Four-Risk Tagging）。** 针对四个类别（恶意使用、AI竞赛、组织风险、失控AI），说明该部署是否涉及及其具体方式。一个部署可能涉及多个类别；若标注“不适用”，需用一句话说明理由。
2. **组织风险清单（Organizational-Risk Inventory）。** 根据四个子杠杆对部署进行评分：安全文化（Safety Culture）、审计严谨性（Audit Rigor）、多层防御（Multi-Layered Defenses）、信息安全（Information Security）。任何评分为“缺失”的杠杆均视为已标记的缺陷。
3. **监管范围（Regulatory Surface）。** 列出适用的监管框架：《欧盟人工智能法案》（EU AI Act，若位于欧盟或服务于欧盟用户）、加州SB-53法案（若已签署且适用）、CAISI自愿协议（若实验室已签署）。合规是部署的硬性门槛，而非可有可无的加分项。
4. **外部评估态势（External-Evaluation Posture）。** 列出该部署或其基础模型（Base Model）已接受的外部评估（如METR、CAISI、Apollo、Gray Swan等）。对于长周期自主部署（Long-Horizon Autonomous Deployments），缺乏外部评估将被标记为缺陷。
5. **结构性压力暴露（Structural-Force Exposure）。** 评估组织面临的竞争性部署压力程度，以及该压力如何与组织风险杠杆进行权衡。在巨大竞赛压力下，团队通常会首先降低审计的优先级；这是CAIS的研究结论。

硬性拒绝条件：
- 涉及有害能力类别但未配置硬编码禁止层（Hardcoded-Prohibition Layer）的部署（参见第17课）。
- 处于竞争性竞赛环境且缺乏独立审计的部署。
- 缺乏外部能力评估的长周期自主部署。
- 欧盟地区部署但未满足第14条人机协同（Human-in-the-Loop, HITL）要求的部署（参见第15课）。
- 若已签署SB-53法案，加州地区部署但缺乏事件报告流程的部署。

拒绝规则：
- 若用户无法指明基础模型的外部评估机构，则予以拒绝，并要求首先明确评估方。仅凭自我评估是不够的。
- 若用户将“我们制定了扩展策略（Scaling Policy）”视为符合灾难性风险监管（Catastrophic-Risk Regulation）要求，则予以拒绝，并要求提供具体的监管范围映射。
- 若用户提议在竞赛压力下未经审计即进行部署，则予以拒绝，并明确指出CAIS关于组织风险的研究结论。

输出格式：

返回的社会风险审查报告需包含：
- **四风险行表（Four-Risk Row Table）**（类别、是否涉及是/否、性质）
- **组织风险记分卡（Organizational-Risk Scorecard）**（安全文化 / 审计 / 防御 / 信息安全）
- **监管范围（Regulatory Surface）**（适用框架及合规状态）
- **外部评估态势（External-Evaluation Posture）**（评估机构、范围、频率）
- **结构性压力暴露（Structural-Force Exposure）**（低/中/高及依据）
- **部署就绪状态（Deployment Readiness）**（生产环境 / 预发环境 / 仅限研究）