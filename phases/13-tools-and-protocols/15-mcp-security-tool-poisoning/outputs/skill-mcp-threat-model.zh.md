---
name: mcp-威胁模型
description: 为 MCP（Model Context Protocol）部署生成威胁模型（threat model），列出适用的攻击类别（attack classes）、现有防御措施以及违反双重规则（Rule-of-Two）的情况。
version: 1.0.0
phase: 13
lesson: 15
tags: [mcp, 安全, 工具投毒, 威胁模型, 双重规则]
---

给定一个 MCP 部署（服务器列表、工具列表、权限列表），生成一份威胁模型。

生成内容：

1. 攻击适用性。针对七类攻击（工具投毒（tool poisoning）、服务撤资（rug pull）、影子劫持（shadowing）、MPMA、寄生工具链（parasitic toolchain）、采样攻击（sampling attacks）、供应链伪装（supply-chain masquerade）），评估其适用性为高/中/低，并附一句理由。
2. 防御清单。列出已部署的防御措施（哈希固定（hash pinning）、静态检测器（static detector）、网关（gateway）、已签名注册表（signed registry）、MELON、双重规则强制执行）。
3. 双重规则审计。对每个工具进行分类：不可信（untrusted）/ 敏感（sensitive）/ 关键后果（consequential），并标记在单次交互中同时涉及这三类的情况。
4. 缺失的防御措施。根据威胁特征，指出尚未应用且收益最高的防御手段。
5. 运行手册（runbook）。团队在未来一周内为提升安全态势（security posture）应采取的三项行动。

硬性拒绝条件：
- 任何声称“由于我们信任该服务器，因此攻击类别 X 不适用”的威胁模型。必须假设至少有一台服务器会被攻破。
- 任何使用静默覆盖命名空间解析（silent-overwrite namespace resolution）的部署。
- 任何启用了采样但未配置单会话速率限制器（per-session rate limiter）的部署。

拒绝规则：
- 如果部署缺乏已批准工具描述的文档，则拒绝并强制要求首先实施哈希固定。
- 如果部署使用公共未签名的 MCP 注册表，需标记供应链风险，并建议迁移至已验证的注册表。
- 如果任何工具同时处理不可信输入、敏感数据和关键操作，则拒绝批准并要求进行拆分。

输出：一份单页威胁模型，包含攻击适用性表格、防御清单、双重规则标记列表以及三项行动的运行手册。最后附上针对该部署价值最高的单一安全增强措施。