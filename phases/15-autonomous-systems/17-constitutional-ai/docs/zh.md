# 宪法式AI (Constitutional AI) 与规则覆写 (Rule Overrides)

> Anthropic 于 2026 年 1 月 22 日发布的 Claude 宪法 (Claude Constitution) 长达 79 页，采用 CC0 协议。它从基于规则的对齐 (Rule-Based Alignment) 转向基于推理的对齐 (Reason-Based Alignment)，并建立了四层优先级体系 (Four-Tier Priority Hierarchy)：(1) 安全与支持人类监督，(2) 伦理，(3) Anthropic 指南，(4) 有用性。行为被分为两类：操作员和用户均无法覆写的硬编码禁令 (Hardcoded Prohibitions)（如生物武器能力增强、儿童性虐待材料 CSAM），以及操作员可在限定范围内调整的软编码默认设置 (Soft-Coded Defaults)。2022 年的原始版本（Bai 等人）通过自我审查 (Self-Critique) 和基于 AI 反馈的强化学习 (RLAIF) 对照宪法来训练无害性。需要坦诚说明的是：基于推理的对齐依赖于模型将原则泛化到未预见情境的能力。Anthropic 自身在 2023 年开展的参与式实验显示，公众来源的原则与企业原则之间存在约 50% 的分歧；而 2026 年版本并未纳入这些发现。

**类型：** 学习
**语言：** Python（标准库，四层优先级解析器）
**前置条件：** 第 15 阶段 · 06（自动化对齐研究），第 15 阶段 · 10（权限模式）
**时间：** 约 60 分钟

## 核心问题

实际部署的智能体 (Agent) 会接触到其设计者从未见过的输入。没有任何规则列表能长到足以覆盖所有情况。也没有任何规则列表能短到在计算压力下快速执行。实际面临的挑战是：如何让智能体对齐到既能应对长尾 (Long Tail) 案例，又能满足快速推理 (Inference) 需求的原则？

基于规则的对齐 (Rule-Based Alignment, RBA)：列出所有禁止事项。检查速度快，易于审计，但无法保持时效性，且经常对未预见到的相似情况产生过度拒绝 (Over-refusal)。基于推理的对齐（如 2026 年 Claude 宪法）：编码原则，让模型自行推理。能够泛化到未见过的案例，但审计难度更大；其失效模式 (Failure Mode) 是原则误用 (Principle Misapplication)，而非规则遗漏。

2026 年宪法采取了明确的中间立场。硬编码禁令——其错误性不依赖于上下文的事项（如生物武器能力增强、CSAM）——采用 RBA：绝对禁止，无论操作员或用户如何指令。其余所有事项均在四层优先级体系内采用基于推理的方式：安全与支持人类监督优先；伦理次之；Anthropic 声明的指南排第三；有用性最后。操作员可以在软编码区域内调整默认设置，但无法触碰硬编码禁令。

## 核心概念

### 四级优先级体系

1. **安全与支持人类监督（Safety and supporting human oversight）**。最高优先级。模型的首要原则是绝不能削弱人类与 Anthropic 对 AI 进行监督和纠正的能力。这并非泛泛的“保持谨慎”，而是特指“不得采取任何增加人类监督难度的行为”。
2. **伦理（Ethics）**。诚实、避免对人造成伤害、不欺骗、不操纵。当与 Anthropic 的指南发生冲突时，伦理原则优先。
3. **Anthropic 指南（Anthropic guidelines）**。Anthropic 确定的运营规范：产品范围、交互模式、何时使用何种工具等。
4. **有益性（Helpfulness）**。最低优先级。在满足更高优先级的前提下，尽可能提供有用的帮助。

当各层级发生冲突时，高优先级胜出。其逻辑与 Unix 进程优先级或网络服务质量（QoS）相同——这种设计旨在实现可预测的冲突解决机制，而非保证在单一维度上达到最优表现。

### 硬编码（Hardcoded）禁令与软编码默认值（Soft-coded defaults）

**硬编码：**
- 生物武器 / 化学、生物、放射性和核（CBRN）能力提升
- 儿童性虐待材料（CSAM）
- 针对关键基础设施的攻击
- 在被直接询问时，就模型身份欺骗用户

操作员和用户均无法覆盖这些限制。在可能的情况下，它们会在模型权重（model weights）层面通过人类反馈强化学习（RLHF）/ 宪法 AI（Constitutional AI）训练来强制执行；在无法通过权重实现的情况下，则在推理层（inference layer）进行拦截。

**软编码默认值（操作员可调）：**
- 响应长度默认值
- 主题范围（模型可拒绝操作员部署范围之外的主题）
- 风格（正式或随意）
- 工具使用模式

操作员的调整必须在声明的边界内进行。操作员无法通过重命名的方式移除硬编码禁令。

### 2022 年宪法 AI（CAI）训练

最初的宪法 AI（Constitutional AI, CAI）（Bai 等人，2022）专注于无害性（harmlessness）训练：

1. 针对一组提示词（prompts）生成响应。
2. 要求模型根据宪法（即明确的原则）对每个响应进行批判性评估。
3. 根据评估结果修改响应。
4. 对修改后的配对数据执行基于 AI 反馈的强化学习（RLAIF）。

结果：得到一个能够基于原则性解释拒绝有害请求的模型，而非采取一刀切的拒绝策略。2026 版宪法采用了该训练方法的演进版本，并额外增加了针对明确优先级体系的后训练（post-training）。

### 基于推理的对齐（reason-based alignment）能捕获与遗漏的内容

**能捕获：**
- 允许的基础能力（primitives）出现未预期的组合，但相关原则依然明确适用的情况。
- 与已禁止请求高度相似的新型请求。
- 依赖“你并未明确禁止 X”这一逻辑的社会工程学攻击（social-engineering attacks）。

**会遗漏：**
- 利用原则模糊性的攻击（例如“用户提出了该请求，因此有益性原则要求同意”）。
- 两个原则以未预期的方式发生冲突，且层级顺序不明确的情景。
- 在多个训练周期中，对原则的解释发生缓慢偏移（重新解释）。

### 2023 年参与式实验

Anthropic 在 2023 年进行了一项实验，对比了由企业起草的宪法与通过公众意见（约 1000 名美国受访者）生成的宪法。两个版本在约 50% 的原则上达成一致。在存在分歧的领域，公众生成的版本在某些议题上更为严格（如政治内容处理），而在另一些议题上则更为宽松（如 AI 身份的自我披露）。2026 版宪法并未采纳公众调研的结果。这是该方法中已记录在案的一种内在张力。

### 为何需要硬编码禁令

仅靠基于推理的对齐无法覆盖长尾风险（close the tail）。如果攻击者能让模型接受某个前提（例如“我们是一家持有执照的生物武器研究实验室”），通常就能绕过依赖个案推理的原则。硬编码禁令不会因前提设定的框架而妥协。它们正是对齐层（alignment layer）中第 14 课所指的“硬性宪法限制”。

### 宪法在技术栈中的位置

宪法并非第 14 课中的终止开关（kill switch）。它位于模型层（model layer）：决定模型权重被训练为何种偏好。终止开关与金丝雀令牌（canary tokens）则位于运行时层（runtime layer）：决定运行时环境允许执行的操作。两者缺一不可。如果因为模型权重过于宽松而导致运行时执行了所有错误操作，这是运行时层的问题。如果因为运行时限制过严而导致模型拒绝了所有正确操作，这同样是运行时层的问题。不同层级负责防御不同类型的风险。

## 使用它

`code/main.py` 实现了一个极简的四层优先级解析器（four-tier priority resolver）。该解析器接收一个拟议动作和一组原则评估（principle-evaluations，涵盖安全性、伦理、准则、有益性），并返回原动作、拒绝执行或修改后的动作。驱动脚本（driver）运行一组小型测试用例：明确允许、明确禁止、硬编码禁令（hardcoded prohibition）以及跨层级模糊案例。

## 交付它

`outputs/skill-constitution-review.md` 用于审计部署中的宪法层（constitutional layer）：哪些是硬编码的，哪些是软编码的（soft-coded），操作员可在何处进行调整，以及四层层级结构（four-tier hierarchy）是否真正作为解析顺序。

## 练习

1. 运行 `code/main.py`。确认即使有益性（helpfulness）评分很高，硬编码禁令（hardcoded prohibition）依然会触发。修改解析器，将有益性的权重置于伦理（ethics）之上；观察其失效模式（failure mode）。

2. 阅读《Claude 宪法》（公开版，79页，CC0协议）。找出你认为定义不够明确的一项原则。撰写两段文字，说明具体的模糊之处，并提出更严谨的表述方案。

3. 为客服智能体（customer-support agent）设计一套软编码默认配置（soft-coded default set）。操作员可以调整哪些内容？哪些内容操作员无法修改？请为每项边界设定提供理由。

4. 阅读 Bai 等人 2022 年关于宪法式人工智能（Constitutional AI, CAI）的论文。描述一个具体案例，说明在该案例中，宪法式人工智能的“批评-修订”循环（critique-and-revise loop）产生的结果会比一刀切规则（blanket rule）更差。指出该案例所属的类别。

5. Anthropic 2023 年的参与式实验（participatory experiment）发现，公众原则与企业原则之间存在约 50% 的分歧。挑选一个对生产环境部署（production deployment）至关重要的类别（例如政治中立性）。提出一种设计方案，使操作员能够表达自身的价值观，同时保持硬编码禁令（hardcoded prohibitions）不受影响。

## 关键术语

| 术语 | 大众说法 | 实际含义 |
|---|---|---|
| Constitutional AI（宪法式人工智能） | “Anthropic 的对齐方法” | 针对成文宪法进行自我批评 + RLAIF |
| Reason-based alignment（基于推理的对齐） | “原则而非规则” | 模型基于原则进行推理，以处理未见案例 |
| Hardcoded prohibition（硬编码禁令） | “绝不做 X” | 基于规则的禁令，任何操作员或用户均无法覆盖 |
| Soft-coded default（软编码默认配置） | “操作员可调” | 在声明边界内的行为，由操作员控制 |
| Four-tier hierarchy（四层优先级结构） | “优先级顺序” | 安全性 > 伦理 > 准则 > 有益性 |
| RLAIF（基于 AI 反馈的强化学习） | “AI 反馈强化学习” | 奖励信号来自模型生成批评的强化学习 |
| Participatory constitution（参与式宪法） | “公众征集的原则” | 2023 年 Anthropic 实验；与企业原则存在约 50% 的分歧 |
| Principle drift（原则漂移） | “解读偏差” | 模型对固定原则文本的解读发生缓慢变化 |

## 延伸阅读

- [Anthropic — Claude 宪法（2026年1月）](https://www.anthropic.com/news/claudes-constitution) — 一份长达79页的 CC0（Creative Commons Zero）协议文档。
- [Bai 等人 — 宪法AI（Constitutional AI）：基于AI反馈的无害性](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — 2022年原始论文。
- [Anthropic — 集体宪法AI（2023年）](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — 一项公众参与式实验。
- [Anthropic — 负责任扩展政策（Responsible Scaling Policy）v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 阐明宪法在 RSP 技术栈中的定位。
- [Anthropic — 实践中智能体（Agent）自主性的测量](https://www.anthropic.com/research/measuring-agent-autonomy) — 宪法在长周期（long-horizon）部署中的作用。