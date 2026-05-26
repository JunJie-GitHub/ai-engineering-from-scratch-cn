# 安全 — 机密管理 (Secrets)、API 密钥轮换、审计日志与护栏 (Guardrails)

> 通过集中式密钥库（如 HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除机密泛滥 (secret sprawl)。切勿将凭据存储在配置文件、版本控制系统 (VCS) 的环境文件或电子表格中。优先使用身份与访问管理 (IAM) 角色而非静态密钥；持续集成/持续交付 (CI/CD) 流程采用开放身份连接 (OIDC)。AI 网关模式是 2026 年的标准解决方案：应用 → 网关 → 模型提供商，网关在运行时从密钥库拉取凭据。在密钥库中执行轮换，所有应用可在几分钟内自动生效——无需重新部署，也无需在 Slack 上询问“谁拿到了新密钥”。轮换策略周期 ≤90 天；每次提交均使用 TruffleHog / GitGuardian / Gitleaks 进行扫描。零信任 (Zero-trust)：多因素认证 (MFA)、单点登录 (SSO)、基于角色/属性的访问控制 (RBAC/ABAC)、短期令牌、设备姿态评估。个人身份信息清理 (PII scrubbing) 利用实体识别在转发前遮蔽受保护健康信息 (PHI)/个人身份信息 (PII)；一致性令牌化 (consistent tokenization，采用 Mesh 方法) 将敏感值映射为稳定的占位符，使大语言模型 (LLM) 能够保留代码/关系语义。网络出站流量控制：将 LLM 服务部署在专用的虚拟私有云/虚拟网络 (VPC/VNet) 子网中，仅白名单放行 `api.openai.com`、`api.anthropic.com` 等地址；阻止所有其他出站连接。2026 年典型事件驱动因素：Vercel 供应链攻击，攻击者通过泄露的 CI/CD 凭据窃取了数千个客户部署中的环境变量。

**类型:** 学习
**语言:** Python（标准库，简易 PII 清理器 + 审计日志写入器）
**前置条件:** 第 17 阶段 · 19（AI 网关），第 17 阶段 · 13（可观测性）
**时长:** 约 60 分钟

## 学习目标

- 列举四种机密管理反模式（版本控制系统中的配置文件、硬编码环境变量、电子表格、静态密钥），并说明其替代方案。
- 解释“AI 网关从密钥库拉取凭据”模式为何成为 2026 年的生产环境标准。
- 实现具备一致性令牌化功能（相同值映射为相同占位符）的 PII 清理器，以确保语义得以保留。
- 指出 2026 年 Vercel 供应链事件，并总结其对 CI/CD 凭据安全规范的启示。

## 问题背景

实习生误将包含 API 密钥的 `.env` 文件提交至代码库。虽然他们迅速删除了该文件，但密钥已留存于 Git 历史记录中——GitGuardian 扫描捕获了此次泄露，而你们的密钥轮换流程却是“在 Slack 通知团队、手动更新 40 个配置文件、重新部署所有服务”。8 小时后，一半的服务已上线，另一半仍在等待部署窗口。

另一方面，用户提示词中包含“我的社会安全号码 (SSN) 是 123-45-6789”。该提示词被直接发送至 OpenAI。尽管你们已签署业务伙伴协议 (BAA)，且内部政策要求在转发前遮蔽个人身份信息 (PII)，但你们并未执行。

此外，你们弹性 Kubernetes 服务 (EKS) 集群中的 LLM Pod 能够访问任意互联网主机。攻击者通过向受控域名发起 DNS 查询窃取了数据，而没有任何机制对此进行拦截。

LLM 服务的安全防护必须同时应对上述三个攻击面。具体包括：基于密钥库的凭据管理、PII 清理、网络出站流量过滤以及审计日志。

## 核心概念

### 集中式密钥库 (Centralized Vault) + IAM 角色拉取 (IAM-Role Pull)

**密钥库 (Vault)**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。作为单一事实来源。

**IAM 角色 (IAM Role)**：应用程序/网关通过其 IAM 身份进行认证，而非使用静态密钥。密钥库仅在令牌有效期内返回密钥。

**AI 网关模式 (AI-Gateway Pattern)**：网关在每次请求时从密钥库拉取 `OPENAI_API_KEY`。在密钥库中轮换密钥后，下一次请求即可获取新密钥。无需重新部署。

### 轮换策略 (Rotation Policy) ≤ 90 天

涵盖所有 API 密钥、密钥库根令牌以及 CI/CD 凭证 (CI/CD Credentials)。尽可能采用自动化轮换。手动轮换需记录日志并跟踪。

### 密钥扫描 (Secret Scanning)

- **TruffleHog** — 基于正则表达式 (Regex) 与信息熵 (Entropy) 的提交扫描。
- **GitGuardian** — 商业工具，准确率高。
- **Gitleaks** — 开源软件 (OSS)，在持续集成 (CI) 中运行。

每次代码提交时运行。若检测到新密钥，则阻止合并请求 (PR)。

### 零信任架构 (Zero-Trust Posture)

- 所有账户必须启用多因素认证 (MFA)。
- 通过 SAML/OIDC 实现单点登录 (SSO)。
- 采用基于角色的访问控制 (RBAC) 或基于属性的访问控制 (ABAC) 实现细粒度权限管理。
- 使用短期令牌（有效期为数小时而非数天）。
- 设备合规性检查 — 仅限启用磁盘加密的企业设备。

### 个人身份信息/受保护健康信息脱敏 (PII/PHI Scrubbing)

在提示词 (Prompt) 离开你的基础设施之前：

1. 实体识别 (Entity Recognition)（使用 spaCy NER、Presidio 或商业工具）。
2. 掩码匹配到的实体：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. 一致性标记化 (Consistent Tokenization，采用 Mesh 方法)：相同值映射到相同的占位符，以便大语言模型 (LLM) 保留数据间的关联关系。
4. 可选的反向映射，用于还原 LLM 的响应内容。

静态正则过滤器可捕获基础模式；NER 能识别更多内容。建议两者结合使用。

### 输入与输出护栏 (Input + Output Guardrails)

输入端：拦截已知的越狱攻击 (Jailbreaks) 与禁止话题；实施基于用户的速率限制 (Rate-Limit)。

输出端：使用正则表达式过滤泄露的密钥（如 API 密钥模式、拒绝上下文中的邮箱模式），并使用分类器检测策略违规行为。

### 网络出站白名单 (Network Egress Whitelist)

将 LLM 服务部署在专用子网中：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、密钥库端点。
- 其他所有流量：丢弃。
- DNS 解析仅通过白名单解析器（防止 DNS 隧道数据外泄）。

### 审计日志 (Audit Log)

记录每次 LLM 调用的不可变日志，包含：
- 时间戳。
- 用户 / 租户。
- 提示词哈希值（出于隐私保护，不记录原始提示词）。
- 模型及版本。
- Token (词元) 数量。
- 成本。
- 响应哈希值。
- 触发的护栏规则。

根据合规要求保留日志（SOC 2 为 1 年，HIPAA 为 6 年）。

### 2026 年 Vercel 安全事件

供应链攻击：遭泄露的 CI/CD 凭证导致数千个客户部署的环境变量被窃取。教训：CI/CD 凭证等同于生产环境权限。必须存入密钥库。严格限制权限范围。积极执行轮换。

### 关键数据备忘

- 轮换策略：≤ 90 天。
- 每次提交扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026 事件：CI/CD 凭证泄露 → 数千客户环境变量外泄。
- 审计日志保留期：SOC 2 = 1 年，HIPAA = 6 年。

## 使用指南

`code/main.py` 实现了一个简易的 PII 脱敏工具，包含一致性标记化与仅追加 (Append-Only) 的审计日志功能。

## 部署上线

本课时将生成 `outputs/skill-llm-security-plan.md`。结合监管范围与当前系统状态，规划密钥库（Vault）迁移、敏感信息脱敏器（Scrubber）、出站流量管控（Egress）及审计日志（Audit Log）方案。

## 练习

1. 运行 `code/main.py`。发送两条引用相同社会安全号码（SSN）的提示词（Prompt），确认两者均返回相同的占位符（Placeholder）。
2. 为部署在 EKS 上的 vLLM 服务设计网络出站策略（Egress Policy），该服务需调用 OpenAI、Anthropic 和 Weaviate。
3. 你在 Git 历史记录中发现了一个密钥（已存在 2 年）。正确的应对措施是什么——轮换密钥（Rotate）、清理历史记录（Scrub），还是两者兼施？请说明理由。
4. 你的审计日志每天增长 10 GB。请设计数据留存分级策略（热数据 30 天、温数据 12 个月、冷数据 6 年）。
5. 论证反向令牌化（Reverse-tokenization，即将真实值重新替换回大语言模型响应中）所带来的复杂性是否值得，相较于直接保留可见的占位符。

## 核心术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| 密钥库（Vault） | “密钥存储” | 集中式凭证管理服务 |
| IAM 角色（IAM Role） | “基于身份的身份验证” | 由应用程序承担的角色；返回短期凭证 |
| CI/CD 的 OIDC | “云颁发的令牌” | CI 流程中无需静态密钥——通过 OIDC 进行身份验证 |
| TruffleHog / GitGuardian / Gitleaks | “密钥扫描器” | 提交时的密钥检测工具 |
| RBAC / ABAC | “访问控制” | 基于角色 vs 基于属性的访问控制 |
| PII 清洗（PII Scrubbing） | “数据脱敏” | 移除或令牌化敏感实体 |
| 一致性令牌化（Consistent Tokenization） | “稳定占位符” | 相同值每次生成相同的令牌 |
| Mesh 方法（Mesh Approach） | “Mesh 令牌化” | 保持语义的令牌化模式 |
| 出站白名单（Egress Whitelist） | “出站允许列表” | 仅允许访问已授权的域名 |
| 审计日志（Audit Log） | “不可变历史记录” | 仅追加的记录，用于合规审计 |

## 延伸阅读

- [Doppler — 高级 LLM 安全实践](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — 使用密钥引用管理 LLM API 密钥](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM 护栏最佳实践](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — 2026 年密钥管理最佳实践](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — 个人身份信息（PII）检测与匿名化。
- [HashiCorp Vault 文档](https://developer.hashicorp.com/vault/docs)