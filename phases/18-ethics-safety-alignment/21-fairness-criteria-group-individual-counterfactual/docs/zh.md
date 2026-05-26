# 公平性准则（Fairness Criteria）—— 群体（Group）、个体（Individual）与反事实（Counterfactual）

> 公平性（Fairness）文献主要由三大类准则构成。群体公平性（Group Fairness）：包括人口统计学均等（Demographic Parity）、机会均等（Equalized Odds）、条件使用准确率均等（Conditional Use Accuracy Equality）—— 即受保护群体（Protected Groups）在平均层面上具有相等的比率。个体公平性（Individual Fairness，Dwork 等人，2012）：相似的个体应获得相似的决策；决策映射需满足利普希茨条件（Lipschitz Condition）。反事实公平性（Counterfactual Fairness，Kusner 等人，2017）：当敏感属性（Sensitive Attributes）在反事实情境下发生改变时，若对个体的决策保持不变，则该决策对该个体是公平的。2024 年理论成果（NeurIPS 2024）：反事实公平性与准确率之间存在固有的权衡（Trade-off）；一种模型无关（Model-Agnostic）的方法可将最优但不公平的预测器转换为反事实公平的预测器，且准确率损失有界。回溯反事实（Backtracking Counterfactuals，arXiv:2401.13935，2024 年 1 月）：一种新范式，避免了对法律受保护属性进行干预的要求。哲学层面的调和（ICLR Blogposts 2024）：借助因果图（Causal Graphs），满足特定的群体公平性度量即意味着满足反事实公平性。

**Type:** 学习
**Languages:** Python（标准库，三项准则对比）
**Prerequisites:** 第 18 · 20 阶段（偏差），第 02 阶段（经典机器学习）
**Time:** 约 60 分钟

## 学习目标

- 阐述三项群体公平性准则（人口统计学均等、机会均等、条件使用准确率均等）以及一项不可能性定理（Impossibility Result）。
- 基于 Dwork 等人（2012）的利普希茨公式描述个体公平性。
- 描述反事实公平性及其对因果图的依赖关系。
- 解释回溯反事实方法，以及它们为何能规避对受保护属性进行干预的问题。

## 问题背景

第 20 课侧重于偏差（Bias）的度量。第 21 课则致力于定义度量工作所应遵循的公平性标准。这三大类准则在结构上提供了不同的标准——一个模型可能满足群体公平但违背个体公平，或满足反事实公平但违背群体公平。选择何种标准属于策略性决策；没有任何一种标准是普遍最优的。

## 核心概念

### 群体公平性 (Group Fairness)

- **统计均等 (Demographic Parity)。** 对所有群体，P(Y=1 | A=a) = P(Y=1 | A=a')。即各群体的接受率相等。
- **机会均等 (Equalized Odds)。** P(Y=1 | Y*=y, A=a) = P(Y=1 | Y*=y, A=a')。即各群体的真正例率 (TPR) 和假正例率 (FPR) 相等。
- **条件使用准确率均等 (Conditional Use Accuracy Equality)。** P(Y*=y | Y=y, A=a) = P(Y*=y | Y=y, A=a')。即各群体的预测值 (Predictive Value) 相等。

不可能性定理 (Impossibility Theorem)（Chouldechova, Kleinberg-Mullainathan-Raghavan 2017）：在基础比率 (Base Rates) 不相等的情况下，这三项指标无法同时满足。

### 个体公平性 (Individual Fairness)

Dwork 等人 (2012)。若存在某个利普希茨常数 (Lipschitz Constant) L，使得对于任务特定的相似度度量 (Similarity Metric) d 满足 |f(x) - f(x')| <= L * d(x, x')，则决策映射 f 是个体公平的。即相似的个体应获得相似的决策。

这需要明确定义 d。这是一个政策与规范问题，而非统计学问题。

### 反事实公平性 (Counterfactual Fairness)

Kusner 等人 (2017)。在总体的因果模型 (Causal Model) 下，如果当个体 i 的敏感属性 (Sensitive Attributes) 发生反事实改变时，其决策保持不变，则该决策对个体 i 是反事实公平的。

这需要构建因果有向无环图 (Causal DAG)。DAG 的选择属于建模决策。反事实公平性的合理性完全取决于所构建 DAG 的合理性。

### 反事实公平性与准确率之间的权衡 (CF-vs-Accuracy Trade-off)

NeurIPS 2024 理论研究表明：反事实公平性与预测准确率之间存在固有的权衡关系。一种与模型无关 (Model-Agnostic) 的方法可以将最优但不公平的预测器转换为反事实公平的预测器，且准确率损失是有界的。该准确率损失取决于最优不公平预测器中敏感属性系数的大小。

### 回溯反事实 (Backtracking Counterfactuals)

arXiv:2401.13935（2024年1月）。传统的反事实方法要求对敏感属性进行干预 (Intervention)——“如果此人性别不同，决策是否会改变”。在法律上这存在问题：在自动化分类相关法律中，受保护属性 (Protected Attributes) 不允许被干预。

回溯反事实方法反转了这一方向：不再干预属性，而是探究个体实际特征的何种组合本应产生该反事实结果。这巧妙地规避了法律上的争议。

### 哲学层面的调和 (Philosophical Reconciliation)

ICLR 博客文章 (2024)。在拥有因果图 (Causal Graph) 的前提下，满足某些群体公平性指标必然蕴含反事实公平性。这三大类公平性并非正交（互不相关）；它们是同一底层因果结构的不同侧面。

这并未解决不可能性定理（基础比率不相等仍然会阻碍群体公平性的同时实现）。但它表明，“群体”与“个体/反事实”之间表面上的对立，部分源于未明确指定因果模型所导致的人为假象。

### 在第18阶段中的定位

第20课讲解偏差测量 (Bias Measurement)。第21课讲解公平性定义。第22课讲解隐私（差分隐私 Differential Privacy）。第23课讲解水印技术 (Watermarking)。这些是与资源分配相关的课程 (Allocation-adjacent)，与第7-11课（与欺骗相关的课程 Deception-adjacent）形成互补。

## 实践应用

`code/main.py` 构建了一个包含敏感属性（sensitive attribute）且基础比率（base rates）不均的玩具二分类数据集（toy binary-classification dataset）。在一个简单分类器上计算人口统计学均等（demographic parity）、机会均等（equalized odds）和条件使用准确率均等（conditional use accuracy equality）。观察这三个指标如何产生分歧。应用针对人口统计学均等的重加权（re-weighting）方法，并观察其对另外两个指标造成的代价。

## 交付成果
本课将生成 `outputs/skill-fairness-criterion.md`。给定一项公平性主张或政策，识别其所声称的公平性准则，判断模型在声称的基础比率不均条件下能否满足其余准则，以及该主张依赖于何种因果有向无环图（causal DAG）。

## 练习
1. 运行 `code/main.py`。报告默认数据上的三个群体指标。应用针对人口统计学均等的重加权方法后，重新报告结果。
2. 使用非敏感特征上的 L2 范数实现 Dwork 等人（2012）的个体公平性（individual fairness）指标。报告有多少样本对违反了常数 L=1 的利普希茨条件（Lipschitz condition）。
3. 阅读 Kusner 等人（2017）的论文。为简历评分构建一个简单的双特征因果有向无环图，并指出其隐含的反事实公平性（counterfactual fairness）条件。
4. 2024 年关于回溯反事实（backtracking counterfactuals）的论文避免了对受保护属性进行干预。描述一个该特性对法律合规性至关重要的场景。
5. ICLR 2024 的一篇调和性论文指出，群体公平性（group fairness）与反事实公平性是同一结构的两个方面。从 `code/main.py` 的三个准则中任选两个，并说明能使它们等价的因果假设。

## 关键术语
| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| 人口统计学均等（Demographic parity） | “比率相等” | 各组间的 P(Y=1 | A=a) 相等 |
| 机会均等（Equalized odds） | “TPR/FPR 相等” | 各组间的真正例率（TPR）和假正例率（FPR）相等 |
| 条件使用准确率（Conditional use accuracy） | “PPV/NPV 相等” | 各组间的预测值（PPV/NPV）相等 |
| 个体公平性（Individual fairness） | “利普希茨条件” | 相似的个体获得相似的决策 |
| 反事实公平性（Counterfactual fairness） | “因果改变不变性” | 在反事实属性改变下决策保持不变 |
| 回溯反事实（Backtracking counterfactual） | “通过实际结果解释” | 从结果反向推导的反事实，而非从属性正向推导 |
| 不可能性定理（Impossibility theorem） | “三者冲突” | Chouldechova / KMR 2017：在基础比率不均时，群体准则互斥 |

## 延伸阅读
- [Dwork 等人 — Fairness through Awareness (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913) — 个体公平性
- [Kusner, Loftus, Russell, Silva — Counterfactual Fairness (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856) — 反事实公平性
- [Chouldechova — Fair prediction with disparate impact (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056) — 不可能性定理
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935) — 受保护属性干预的新范式