# 综合项目 06 — Kubernetes DevOps 故障排查智能体 (DevOps Troubleshooting Agent)

> AWS 的 DevOps Agent 已正式发布 (General Availability, GA)，Resolve AI 发布了其 Kubernetes 运维手册 (playbooks)，NeuBird 演示了语义监控 (semantic monitoring)，而 Metoro 将 AI 站点可靠性工程 (Site Reliability Engineering, SRE) 与单服务服务等级目标 (Service Level Objective, SLO) 进行了绑定。生产环境的架构形态已趋于定型：告警 Webhook 触发后，智能体读取遥测数据 (telemetry)，遍历 Kubernetes 对象图谱，对根因假设 (root-cause hypotheses) 进行排序，并在 Slack 中发布包含审批按钮的简报。默认采用只读模式。所有修复操作均需人工审批。本综合项目正是构建此类智能体，将在 20 个合成故障场景中进行评估，并与 AWS Agent 在三个共享案例上进行对比。

**类型：** 综合项目 (Capstone)
**编程语言：** Python（智能体）、TypeScript（Slack 集成）
**前置要求：** 第 11 阶段（大语言模型工程 (LLM Engineering)）、第 13 阶段（工具与模型上下文协议 (Model Context Protocol, MCP)）、第 14 阶段（智能体 (Agents)）、第 15 阶段（自主运行 (Autonomous)）、第 17 阶段（基础设施 (Infrastructure)）、第 18 阶段（安全 (Safety)）
**涉及阶段：** P11 · P13 · P14 · P15 · P17 · P18
**预计耗时：** 30 小时

## 问题背景

2025-2026 年的 SRE 行业共识已演变为：“AI 智能体负责故障分诊，人类负责审批修复方案。”AWS DevOps Agent、Resolve AI、NeuBird、Metoro 以及 PagerDuty AIOps 均已在生产环境中交付此类架构。该智能体读取 Prometheus 指标、Loki 日志、Tempo 链路追踪、kube-state-metrics 以及 Kubernetes 对象知识图谱 (Knowledge Graph)。它能在五分钟内生成带有遥测数据引用的排序根因假设。未经 Slack 明确的人工审批，它绝不会执行任何破坏性命令。

大部分核心难点在于范围界定与安全控制，而非推理能力。该智能体需要具备默认只读的基于角色的访问控制 (Role-Based Access Control, RBAC) 接口、经过加固的 MCP 工具服务器，以及记录所有“已考虑”与“已执行”命令的审计日志。它必须能够识别自身能力边界并及时升级上报。此外，其运行成本必须足够低廉，以避免因内存溢出终止 (Out-Of-Memory Kill, OOM-kill) 级联故障而产生高达 5000 美元的智能体账单。

## 核心概念

该智能体基于知识图谱运行。节点包含 Kubernetes 对象（Pod、Deployment、Service、Node、水平自动扩缩容 (Horizontal Pod Autoscaler, HPA)、持久卷声明 (Persistent Volume Claim, PVC)）以及遥测数据源（Prometheus 指标序列、Loki 日志流、Tempo 链路追踪）。边用于编码所有权关系（Pod -> ReplicaSet -> Deployment）、调度关系（Pod -> Node）以及观测关系（Pod -> Prometheus 指标序列）。图谱通过 kube-state-metrics 同步保持实时更新，并在每次告警触发时重新采样。

当告警触发时，智能体将从受影响对象出发进行根因分析。它沿图谱边遍历，拉取相关遥测数据切片（最近 15 分钟），并起草假设。假设将根据证据进行排序：支持它的遥测引用数量、数据时效性以及具体程度。排名前 3 的假设将连同图谱路径可视化结果及修复操作的审批按钮一并发送至 Slack。

修复操作受到严格管控。默认允许的操作均为只读。破坏性操作（缩容、回滚、删除 Pod）必须经过 Slack 审批；ArgoCD 回滚钩子需要身份验证令牌，而智能体绝不会持有该令牌。审计日志会记录智能体*考虑过*的每条命令——而不仅仅是已执行的命令——从而确保审查流程能够捕捉到潜在的误操作风险。

## 架构设计

PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command

## 技术栈（Stack）

- 可观测性数据源（Observability sources）：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱（Knowledge graph）：基于 Kubernetes 对象与遥测边（telemetry edges）构建的 Neo4j（托管版）或 kuzu（嵌入式）
- 智能体（Agent）：基于 LangGraph 构建，默认只读，并为每个工具配置允许列表（allow-list）
- 工具传输（Tool transport）：基于 StreamableHTTP 的 FastMCP；破坏性工具需通过审批网关（approval gate），并部署在独立服务器上
- 模型（Models）：使用 Claude Sonnet 4.7 进行根因推理（root-cause reasoning），使用 Gemini 2.5 Flash 进行日志摘要（log summarization）
- 故障修复（Remediation）：ArgoCD 回滚 Webhook、PagerDuty 升级告警、Slack 审批卡片
- 审计（Audit）：仅追加的结构化日志（记录评估、执行、审批及结果状态）
- 部署（Deployment）：K8s 部署，配备独立的最小权限 RBAC 角色；使用独立命名空间

## 构建指南（Build It）

1. **图数据摄入（Graph Ingestion）。** 每 30 秒将 kube-state-metrics 同步至 Neo4j/kuzu。节点（Nodes）：Pod、Deployment、Node、Service、PVC、HPA。边（Edges）：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测覆盖边（Telemetry Overlay Edges）：OBSERVED_BY（表示 Pod 被 Prometheus 指标序列观测）。

2. **告警接收器（Alert Receiver）。** 提供 FastAPI 端点以接收 PagerDuty 或 Alertmanager 的 Webhook（网络钩子）。提取受影响对象及服务等级目标（SLO）违规信息。

3. **只读工具接口（Read-only Tool Surface）。** 通过 FastMCP 封装 kubectl、Prometheus 查询、Loki LogQL 与 Tempo TraceQL。每个工具仅配置最小权限的基于角色的访问控制（RBAC）动词（如 "get"、"list"、"describe"）。默认服务器中不提供 "delete"、"exec"、"scale" 等写操作。

4. **根因分析智能体（Root-cause Agent）。** 基于 LangGraph 构建，包含三个节点：`sample` 负责拉取过去 15 分钟的遥测数据（Telemetry）切片，`walk` 负责查询图谱中的相邻对象，`hypothesize` 负责生成附带遥测数据引用的排序根因候选方案。

5. **证据评分（Evidence Scoring）。** 每个假设的得分计算公式为：时效性 × 特异性 × 图路径（Graph Path）长度倒数 × 引用数量。系统返回得分最高的前 3 项。

6. **Slack 简报（Slack Brief）。** 推送包含假设结论、图路径可视化（服务端渲染的子图图片）以及最多一个修复操作审批按钮的附件。

7. **修复操作门禁（Remediation Gate）。** 破坏性工具（缩容、回滚、删除）部署于独立的第二个 MCP 服务器，并受审批令牌保护。智能体仅在人工审批 Slack 卡片后方可调用这些工具。

8. **审计日志（Audit Log）。** 采用仅追加的 JSON Lines（JSONL）格式：针对每个候选命令，记录其是否被评估、是否被执行以及审批人信息。每日将日志归档至 S3。

9. **合成故障测试套件（Synthetic Incident Suite）。** 构建 20 个故障场景：内存溢出终止（OOMKill）级联、DNS 抖动（DNS Flap）、水平自动扩缩容震荡（HPA Thrash）、PVC 空间耗尽、嘈杂邻居（Noisy Neighbor）、故障边车容器（Sidecar）、错误的配置映射发布（ConfigMap Rollout）、证书轮换（Certificate Rotation）、镜像拉取退避（Image-Pull Backoff）等。根据根因定位准确率与生成假设所需时间对智能体进行评分。

## 使用示例 (Use It)

webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)

## 部署发布 (Ship It)

`outputs/skill-devops-agent.md` 为最终交付物。在给定 Kubernetes (K8s) 集群与告警源的情况下，该智能体 (Agent) 将生成按优先级排序的根本原因假设 (Root-Cause Hypotheses)，并提供需经 Slack 审批的修复流程。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 场景套件上的根因分析 (RCA) 准确率 | 在 20 起模拟事件中，正确定位根本原因的比例 ≥80% |
| 20 | 安全性 | 审计日志显示，未经 Slack 审批，破坏性操作防护机制绝不会触发 |
| 20 | 假设生成耗时 | 从告警触发到 Slack 简报生成的 p50（中位数）时间低于 5 分钟 |
| 20 | 可解释性 | 每个假设均附带图路径与遥测数据引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端正常运行 |
| **100** | | |

## 练习

1. 在 AWS DevOps Agent 演示所用的相同三起事件上运行你的智能体。发布并排对比结果，并报告智能体输出存在差异的具体环节。
2. 增加一项“险兆事件 (Near-miss)”审计，用于标记智能体*曾考虑执行*但未经审批即具破坏性的任何命令。统计为期一周的险兆事件发生率。
3. 将假设生成模型从 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量根因分析 (RCA) 准确率的差值以及单起事件的处理成本。
4. 构建因果过滤器：区分具有相关性的遥测数据突增与真正的根本原因。基于 20 个场景的标签训练一个轻量级分类器。
5. 增加回滚预演 (Dry-run)：使用相同的清单 (Manifest) 在预发集群上执行 ArgoCD 回滚。在触发 Slack 审批按钮前，于生产集群中验证回滚方案。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| K8s 知识图谱 (K8s Knowledge Graph) | “集群图谱” | 节点 = K8s 对象 + 遥测数据序列；边 = 归属关系、调度关系、观测关系 |
| 默认只读 (Read-only-by-default) | “限定范围 RBAC” | 智能体的服务账户仅拥有 get/list/describe 动词权限；破坏性动词权限位于独立的审批服务器之后 |
| 审计日志 (Audit Log) | “已考虑 vs 已执行” | 仅追加记录每个候选命令的状态（是否执行）及审批人 |
| 假设排序 (Hypothesis Ranking) | “证据得分” | 时效性 × 特异性 × 图路径长度倒数 × 引用次数 |
| Slack 审批卡片 (Slack Approval Card) | “人机协同 (HITL) 关卡” | 包含修复按钮的交互式 Slack 消息；在人工点击确认前，智能体无法继续推进 |
| 遥测引用 (Telemetry Citation) | “证据指针” | 用于支撑结论的 Prometheus 查询、Loki 选择器或 Tempo 追踪 URL |
| MTTR (平均恢复时间) | “解决耗时” | 从告警触发到服务等级目标 (SLO) 恢复的实际耗时 |

## 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 年权威参考 (Canonical Reference)
- [Resolve AI K8s 故障排查](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞品参考 (Competitor Reference)
- [NeuBird 语义监控](https://www.neubird.ai) — 语义图 (Semantic Graph) 方法
- [Metoro AI SRE](https://metoro.io) — 以 SLO (Service Level Objective) 优先的生产环境架构
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态 (Cluster State) 数据源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 智能体编排器 (Agent Orchestrator) 参考实现
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP (Model Context Protocol) 服务器框架
- [ArgoCD 回滚](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 受控修复 (Gated Remediation) 目标