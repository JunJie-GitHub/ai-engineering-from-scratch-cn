---
name: devops-agent
description: 构建一个 Kubernetes（K8s）故障排查智能体（Agent），该智能体能够遍历集群知识图谱、对根本原因（Root Cause）进行排序，并通过 Slack 审批流程控制所有修复操作。
version: 1.0.0
phase: 19
lesson: 06
tags: [综合项目, devops, sre, kubernetes, langgraph, fastmcp, aiops]
---

给定一个 Kubernetes（K8s）集群和告警源（PagerDuty 或 Alertmanager），构建一个智能体（Agent），使其能在五分钟内生成按优先级排序的根本原因假设，并通过 Slack 审批卡片控制所有修复操作。

构建计划：

1. 每 30 秒将 `kube-state-metrics` 数据摄入 Neo4j 或 kuzu。构建包含 Pods、Deployments、Services、Nodes、PVCs、HPAs 的图谱，并添加指向 Prometheus、Loki 和 Tempo 数据源的遥测（Telemetry）关联边。
2. 搭建一个 FastAPI Webhook 接收器，用于接收 PagerDuty 和 Alertmanager 的请求。
3. 通过 FastMCP 以 StreamableHTTP 传输协议暴露只读工具：`kubectl get/describe`、`promql`、`logql`、`traceql`。
4. 构建一个包含三个节点的 LangGraph 根本原因分析智能体：`sample`（拉取 15 分钟遥测数据）、`walk`（遍历图谱邻居节点）、`hypothesize`（根据时效性 × 特异性 × 引用次数对候选原因进行排序）。
5. 将排名前 3 的假设及其图谱路径可视化结果发布到 Slack，并附带审批按钮。
6. 将破坏性工具（扩缩容、回滚、删除）部署在独立的 FastMCP 服务器上，并设置审批令牌保护，智能体仅在获得 Slack 签批后才能获取该令牌。
7. 维护仅追加（Append-only）的审计日志：记录每一条*被考虑*的命令、是否获批、是否执行以及审批人。
8. 构建 20 个合成故障场景（OOMKill、DNS 抖动、HPA 震荡、PVC 占满、嘈杂邻居、故障 Sidecar、ConfigMap 错误发布、证书轮换、镜像拉取退避、探针失败，以及另外 10 个）。根据根本原因分析（Root Cause Analysis, RCA）准确率和假设生成时间对智能体进行评分。

评估标准：

| 权重 | 评估标准 | 测量指标 |
|:-:|---|---|
| 25 | 场景套件上的 RCA 准确率 | 在 20 个合成故障中，根本原因判断正确率至少达到 80% |
| 20 | 安全性 | 审计日志中显示，未经 Slack 审批，破坏性操作防护机制绝不会触发 |
| 20 | 假设生成时间 | 从告警到 Slack 简报的 p50（中位数）时间低于 5 分钟 |
| 20 | 可解释性 | 每个假设均附带图谱路径和遥测数据引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端正常运行 |

硬性拒绝条件：

- 使用单一 MCP 服务器混合部署只读工具与破坏性工具的智能体。
- 任何未附带遥测数据引用的 RCA 结果。缺乏引用的假设必须被拒绝。
- 仅记录已执行命令的审计日志。日志必须记录每一条被考虑过的命令。
- 未使用固定随机种子（Seeds）在 20 个场景套件上运行智能体，却声称具备高准确率的报告。

拒绝规则：

- 拒绝在未获得人类值班人员（On-caller）通过 Slack 审批的情况下执行修复操作。即使假设非常明显也不例外。
- 拒绝通过只读 MCP 暴露 `kubectl exec`、`kubectl port-forward` 或任何交互式工具。这些工具在实际效果上具有破坏性。
- 拒绝在未提供逐个 Deployment 审批卡片的情况下，跨多个 Deployment 批量应用修复操作。

输出：一个代码仓库，包含 FastAPI 接收器、LangGraph 智能体、只读与破坏性 MCP 服务器、Slack 集成模块、20 场景测试套件、在三个共享故障场景下与 AWS DevOps Agent 的对比分析，以及为期一周观察窗口内的“险兆命令”（Near-miss commands，即智能体*考虑过*但未执行的命令）说明文档。