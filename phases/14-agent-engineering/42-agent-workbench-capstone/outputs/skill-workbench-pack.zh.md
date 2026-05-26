---
name: workbench-pack
description: 生成针对项目调优的即插即用型智能体工作台包（agent workbench pack）——规则根据团队历史进行打磨，作用域通配符（scope globs）与代码仓库匹配，评估量规（rubric）维度扩展了一个特定领域条目。
version: 1.0.0
phase: 14
lesson: 42
tags: [综合项目 (capstone), 工作台包 (workbench-pack), 安装程序 (installer), 数据模式 (schemas), 即插即用 (drop-in)]
---

给定一个代码仓库（repo）、团队的事故历史记录（incident history）以及在其中运行的智能体产品（agent product），输出一个经过调优的智能体工作台包（agent-workbench-pack）及其安装程序（installer）。

产出内容：

1. 符合标准布局（canonical layout）的 `agent-workbench-pack/` 目录：包含 `AGENTS.md`、`docs/`、`schemas/`、`scripts/`、`bin/`、`README.md` 和 `VERSION`。
2. 一个 `bin/install.sh` 脚本，在未提供 `--force` 参数时拒绝覆盖（clobber）现有包，并将 `.workbench-version` 写入目标仓库。
3. 针对项目调优的 `agent-rules.md`（每个类别至少包含一条基于团队最近六次事故推导出的规则）、`reviewer-rubric.md`（增加第六个领域维度）以及 `scope_contract.schema.json`（包含项目特定的通配符）。
4. 一个 `lint_pack.py` 脚本，当脚本与模式（schemas）之间发生版本漂移（drift），或 `VERSION` 与模式中的 `schema_version` 不一致时，该脚本将报错退出。
5. 可选的持续集成（CI）集成方案，用于在演示分支上安装该包，并针对已知良好的任务运行验证门禁（verification gate）。

硬性拒绝条件（Hard rejects）：

- 包含项目特定任务的包。任务应保留在目标仓库的看板（board）上。
- 绑定单一供应商软件开发工具包（SDK）的包。必须保持框架无关性（framework-agnostic）；SDK 的集成与配置由目标仓库负责。
- 会修改状态文件的安装程序。安装程序必须是仅作用于表层且具备幂等性（idempotent）的；状态管理属于智能体和人类用户的职责。
- 缺乏对应检查函数（check function）的规则。愿景型规则应放在入职引导（onboarding）中，而非本包内。

拒绝执行规则（Refusal rules）：

- 如果事故历史记录为空，则拒绝交付调优后的 `agent-rules.md`。应使用标准默认版本，并明确暴露该差距（surface the gap）。
- 如果目标仓库的持续集成（CI）环境与安装流程不兼容（例如缺少 `.github/workflows/` 目录或等效配置），则跳过可选的 CI 步骤，并记录手动操作路径。
- 如果团队使用的是该包的私有分支（private fork），则拒绝编写公开的安装程序。私有安装程序需承载私有不变量（private invariants）。

输出结构：

agent-workbench-pack/
├── AGENTS.md
├── docs/
├── schemas/
├── scripts/
├── bin/install.sh
├── lint_pack.py
├── VERSION
└── README.md

结尾附上“下一步阅读建议”，指向以下内容：

- 第 41 课：了解该包所改进的前后基准测试（before/after benchmark）。
- 第 30 课（评估驱动的智能体开发 / Eval-Driven Agent Development）：了解消耗该包裁决结果（verdicts）的评估循环（eval loop）。
- [SkillKit](https://github.com/rohitg00/skillkit)：用于将该包分发至 32 个 AI 智能体。