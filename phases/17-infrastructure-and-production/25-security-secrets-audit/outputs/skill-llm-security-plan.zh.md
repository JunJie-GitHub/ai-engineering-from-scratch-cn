---
name: llm-security-plan
description: 生成一份大语言模型（LLM）安全计划，涵盖密钥保险箱（Secrets Vault）、采用一致化令牌化（Consistent Tokenization）的个人身份信息（PII）清理、网络出站（Egress）白名单、审计日志保留策略以及零信任（Zero-Trust）架构。
version: 1.0.0
phase: 17
lesson: 25
tags: [安全, 密钥保险箱, hashicorp, aws-secrets-manager, 个人身份信息, presidio, 网络出站, 审计日志, 零信任, ci-cd-供应链]
---

结合合规范围（SOC 2、HIPAA、GDPR）、当前凭证（Credential）状态以及网络/出站（Egress）架构，制定一份安全计划。

输出内容：

1. 密钥保险箱（Vault）迁移。选择保险箱方案（HashiCorp、AWS Secrets Manager、Azure Key Vault 或 GCP Secret Manager）。网关模式：应用 → 网关 → 运行时访问保险箱。弃用硬编码的环境变量和配置文件凭证。
2. 密钥扫描（Secret Scanning）。在每次提交（Commit）时启用 TruffleHog / GitGuardian / Gitleaks。一旦检测到密钥，立即拦截拉取请求（PR）。
3. 轮换策略（Rotation Policy）。周期 ≤ 90 天。尽可能实现自动化。为 CI/CD 凭证设置专用轮换周期（建议更短，如 30 天）。
4. 个人身份信息（PII）清理。实体识别（Presidio + 正则表达式）。采用一致化令牌化（Consistent Tokenization）（相同值映射为相同占位符）以保留语义。
5. 出站（Egress）白名单。将大语言模型（LLM）提供商域名、向量数据库（Vector DB）及保险箱端点加入白名单。配置 DNS 白名单解析器。
6. 审计日志（Audit Log）。仅追加（Append-only）、不可变。必填字段：用户、租户、提示词/响应哈希值、Token 数量、成本、安全护栏（Guardrail）触发次数。保留期限遵循合规框架要求（SOC 2 为 1 年 / HIPAA 为 6 年）。
7. CI/CD 安全规范（Hygiene）。采用开放身份连接（OIDC）身份联合（Identity Federation）（禁用静态云密钥）。严格限制 CI/CD 凭证的权限范围。以 2026 年 Vercel 供应链事件作为实施背景说明。

硬性拒绝项：
- 配置文件中包含静态密钥。直接拒绝。
- 在审计日志中存储原始提示词。拒绝——仅存储哈希值，除非合规框架明确要求例外。
- 允许出站访问 `*` 或“整个互联网”。拒绝——必须使用白名单。

拒绝规则：
- 若客户无法接受任何保险箱方案（如物理隔离/Air-gapped 要求），则拒绝常规计划，并设计基于文件轮换的降级方案。需明确注明其安全性较低。
- 若客户以“延迟”为由拒绝 PII 清理，则拒绝——该操作延迟通常 <20 毫秒，且合规风险远高于此延迟影响。
- 若要求保险箱根令牌（Root Token）的轮换周期 >90 天，则拒绝——这将演变为安全漏洞突破口。

输出要求：一份单页计划，涵盖保险箱、扫描、轮换、清理、出站控制、审计日志及 CI/CD 安全态势。结尾附上单一核心指标：每月密钥扫描命中次数；目标为零。