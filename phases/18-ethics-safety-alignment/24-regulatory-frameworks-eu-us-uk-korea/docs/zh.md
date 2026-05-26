# 监管框架 — 欧盟、美国、英国、韩国

> 四大主要监管体系构成了2026年人工智能（AI）治理格局。《欧盟人工智能法案》（EU AI Act，2024年8月1日生效）——自2025年2月2日起实施禁止性实践与AI素养要求；自2025年8月2日起履行通用人工智能（General-Purpose AI, GPAI）义务；2026年8月2日全面适用并执行第50条透明度规定；2027年8月2日适用于遗留GPAI及嵌入式高风险系统；违规罚款最高可达1500万欧元或全球营业额的3%。《GPAI实践守则》（GPAI Code of Practice，2025年7月10日）：包含透明度、版权、安全与安保三个章节，共12项承诺；执法将于2026年8月启动。英国人工智能安全研究所（UK AISI）更名为AI安全研究所（AI Security Institute，2025年2月）：更名标志着其管辖范围收窄。美国人工智能安全研究所（US AISI）更名为AI标准与创新中心（Center for AI Standards and Innovation, CAISI，2025年6月）：该机构隶属于国家标准与技术研究院（NIST）；政策立场转向促进增长。韩国《人工智能框架法》（Korean AI Framework Act，2024年12月通过，2026年1月生效）：第12条规定在韩国科学技术信息通信部（MSIT）下设立AISI；强制要求外国AI公司指定本地代表，开展风险评估，并为高影响及生成式AI制定安全措施。

**Type:** 学习
**Languages:** 无
**Prerequisites:** 第18阶段 · 18（前沿框架）, 第18阶段 · 27（数据治理）
**Time:** 约75分钟

## 学习目标

- 描述《欧盟人工智能法案》（EU AI Act）的风险分级（禁止类、高风险类、通用人工智能类、有限风险类）以及2025年8月/2026年8月/2027年8月的时间线。
- 描述《GPAI实践守则》（GPAI Code of Practice）的三个章节及其分别约束的提供商。
- 描述2025年的机构更名：英国AISI更名为AI安全研究所（AI Security Institute）；美国AISI更名为CAISI；以及每次更名所暗示的政策方向。
- 阐述韩国《人工智能框架法》（Korean AI Framework Act）的核心条款。

## 问题背景

实验室框架（第18课）属于自愿性质，而监管框架则具有强制性。2024至2026年间，首批综合性AI监管法规正式生效。部署方必须将技术控制措施与监管义务进行映射对齐；该映射关系因司法管辖区而异。

## 核心概念

### 《欧盟人工智能法案》（EU AI Act）

**2024年8月1日生效。** 风险分级结构（Risk-tier structure）：

- **禁止性实践（Prohibited practices）**（第5条）。社会评分（Social scoring）、公共场所实时远程生物特征识别（real-time remote biometric identification，含执法例外）、对弱势群体的剥削性操纵。2025年2月2日适用。
- **高风险系统（High-risk systems）**（附件三）。就业、教育、信贷、执法、司法、移民领域。需进行合格评定（conformity assessment）、风险管理、日志记录与透明度要求。
- **通用人工智能（General-Purpose AI, GPAI）模型**。2025年8月2日适用。所有GPAI提供商均负有义务；具有系统性风险的GPAI（systemic-risk GPAI，训练算力超过1e25 FLOP）需承担额外义务。
- **有限风险系统（Limited-risk systems）**。根据第50条承担透明度义务（AI生成内容标识）。2026年8月2日适用。

时间线：
- 2025年2月2日：禁止性实践 + AI素养（AI literacy）。
- 2025年8月2日：GPAI + 治理。
- 2026年8月2日：全面适用 + 第50条透明度要求 + 最高1500万欧元或全球营业额3%的罚款。
- 2027年8月2日：存量GPAI（legacy GPAI） + 嵌入式高风险系统（embedded high-risk）。

欧盟委员会（European Commission）于2025年底提议将高风险系统的适用时间线调整为16个月。

### GPAI行为准则（GPAI Code of Practice）

2025年7月10日发布。共三章：

- **透明度（Transparency）**。面向所有GPAI提供商。
- **版权（Copyright）**。面向所有GPAI提供商。
- **安全与安保（Safety and Security）**。面向具有系统性风险的GPAI提供商（预计5-15家公司）。

共计12项承诺。由人工智能办公室（AI Office）主席领导的签署方工作组（Signatory Taskforce）负责推进实施。执法将于2026年8月2日开始；在此之前，接受善意合规（good-faith compliance）。

### 第50条透明度准则（Transparency Code for Article 50）

初稿于2025年12月17日发布，二稿于2026年3月发布，最终版于2026年6月定稿。涵盖包括深度伪造（deepfakes）在内的AI生成内容标识——该监管层要求使用第23课中的水印技术（watermarking technology）。

### 英国人工智能安全研究所（UK AI Security Institute）（2025年2月）

由人工智能安全研究所（AI Safety Institute）更名而来。此次品牌重塑缩小了范围：不再强调算法偏见（algorithmic bias）和言论自由框架；转而聚焦前沿能力安全（frontier capability security）。已于2024年5月开源 `Inspect` 评估工具。与 Redwood（第10课）在控制安全案例（control safety cases）方面开展合作。

### 美国人工智能标准与创新中心（US CAISI）（2025年6月）

特朗普政府将美国国家标准与技术研究院（NIST）的人工智能安全研究所转型为人工智能标准与创新中心（Center for AI Standards and Innovation）。根据副总统万斯在巴黎人工智能行动峰会上的发言，政策转向“促进增长的AI政策（pro-growth AI policies）”。降低对部署前评估（pre-deployment evaluation）的重视，转而强调标准制定与创新支持。作为国内力量，旨在制衡《欧盟人工智能法案》的监管姿态。

### 韩国人工智能框架法（Korean AI Framework Act）

2024年12月通过，2025年1月颁布，2026年1月生效。整合了19项独立的AI法案。

第12条规定在科学与信息通信技术部（MSIT）下设立人工智能安全研究所（AISI）。强制要求：
- 在韩国运营的外国AI公司需设立本地代表。
- 对“高影响力”AI系统（high-impact AI systems）进行风险评估。
- 针对生成式AI（generative AI）和高影响力AI采取安全措施。

亚洲首个出台综合性横向AI监管法规的司法管辖区。

### 跨司法管辖区动态（Cross-jurisdiction dynamics）

- 欧盟：严格、风险分级、处罚严厉。隐私相关监管（privacy-adjacent regulation）的基准。
- 美国：偏向创新、去中心化，各州（如加州 AB 2013 法案——第27课）填补联邦空白。
- 英国：聚焦狭窄的安全领域，拥有强大的评估基础设施。
- 韩国：由 MSIT 主导，重点关注外国提供商。

监管理念相互竞争。在多个司法管辖区部署的实体必须遵守最严格的规定，而在2026年，这通常指《欧盟人工智能法案》。

### 本内容在第18阶段（Phase 18）中的定位

第18课属于实验室自愿治理（lab-voluntary governance）；第24课属于监管层面；第25课涉及AI系统新兴的一类通用漏洞披露（CVEs）；第26-27课涵盖文档（模型卡等）与训练数据治理（training-data governance）。

## 使用它

无需编写代码。请查阅《欧盟人工智能法案》（EU AI Act）的原始资料：法规正文、通用人工智能（General-Purpose AI, GPAI）行为准则，以及英国人工智能安全研究所（UK AI Security Institute, UK AISI）的 Inspect 框架。将您的部署方案映射至各司法管辖区（jurisdiction）适用的合规义务。

## 交付成果

本课时将生成 `outputs/skill-regulatory-map.md` 文件。根据提供的部署描述，该文件会映射适用的司法管辖区、各辖区内的风险分级（tier classifications）、各辖区对应的合规义务以及截止日期结构。

## 练习

1. 阅读《欧盟人工智能法案》（法规 2024/1689）与《通用人工智能行为准则》（2025年7月10日）。找出适用于所有通用人工智能（GPAI）提供商的三项义务，以及仅适用于具有系统性风险（systemic-risk）的通用人工智能的三项义务。

2. 某项部署由美国公司发起，运行于欧盟基础设施之上，并为韩国用户提供服务。适用哪三个司法管辖区的规则？针对每个实质性问题，分别受哪条规则约束？

3. 英国人工智能安全研究所（UK AISI）的更名缩小了其管辖范围。请分别阐述支持与反对这一更窄定位的理由，并指出每种立场所依赖的政策假设。

4. 美国人工智能标准中心（CAISI）的“促增长”定位偏离了2022-2024年的人工智能安全研究所模式。请指出由此定位可能引发的两项可衡量的政策转变。

5. 韩国《人工智能框架法》要求外国提供商设立本地代表。请描述这对一家服务于韩国用户的湾区公司在运营层面的具体影响。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 《欧盟人工智能法案》（EU AI Act） | “该法规” | 基于风险分级的横向人工智能监管法规；2024年8月生效 |
| 通用人工智能（GPAI） | “通用型AI” | 大型基础模型；具有系统性风险的子集需承担额外合规义务 |
| 第50条（Article 50） | “透明度义务” | AI生成内容标识要求；2026年8月起适用 |
| 英国人工智能安全研究所（UK AISI） | “AI安全研究所” | 2025年2月更名；聚焦范围更窄的前沿安全领域 |
| 美国人工智能标准中心（CAISI） | “美国AI标准中心” | 2025年6月由AI安全研究所更名而来；采取促增长立场 |
| 韩国《人工智能框架法》（Korean AI Framework Act） | “MSIT横向法规” | 亚洲首部综合性人工智能法律；2026年1月生效 |
| 系统性风险通用人工智能（Systemic-risk GPAI） | “1e25 FLOP 阈值” | 额外义务层级；预计约束5至15家公司 |

## 延伸阅读

- [《欧盟人工智能法案》正文（法规 2024/1689）](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — 法规文本与时间线
- [《通用人工智能行为准则》（2025年7月10日）](https://digital-strategy.ec.europa.eu/en/library/final-version-general-purpose-ai-code-practice) — 三章准则
- [英国人工智能安全研究所（2025年2月更名）](https://www.gov.uk/government/organisations/ai-security-institute) — 官方页面
- [CSET — 韩国《人工智能框架法》分析（2025年）](https://cset.georgetown.edu/publication/south-korea-ai-law-2025/) — 韩国框架法案解读