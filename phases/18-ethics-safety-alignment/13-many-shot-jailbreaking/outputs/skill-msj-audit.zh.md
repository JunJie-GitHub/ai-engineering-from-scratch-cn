---
name: msj-audit
description: 审核长上下文安全评估是否涵盖多轮越狱（many-shot jailbreaking）测试。
version: 1.0.0
phase: 18
lesson: 13
tags: [多轮越狱, 上下文窗口, 幂律, Anthropic]
---

针对长上下文模型的安全评估，审核该评估是否涵盖多轮越狱（many-shot jailbreaking）测试。

输出内容：

1. 提示轮次覆盖率（shot-count coverage）。报告测试的提示轮次（shot counts）（应包含 1、5、16、64、256，且对于上下文长度 >= 1M 的模型，至少包含一个 >= 512 的轮次）。若评估仅测试单一提示轮次，则攻击成功率（Attack Success Rate, ASR）缺乏参考价值——多轮越狱（MSJ）的效果呈现为一条曲线。
2. 幂律拟合（power-law fit）。报告每个行为类别的拟合指数。较平缓的指数表明模型在该类别上具备上下文学习（In-Context Learning, ICL）鲁棒性；较陡峭的指数表明多轮越狱（MSJ）的效果异常显著。
3. 类别细分（category breakdown）。多轮越狱（MSJ）的有效性因类别而异：暴力内容、欺骗、自残、生物武器。根据 Anil 等人（2024）的研究，暴力/欺骗类内容所需的越狱提示轮次更少。若评估中缺失任何类别，需予以标记。
4. 防御机制识别（defense identification）。是否部署了基于分类器的提示词修改（prompt modification）？该分类器本身是否经过对抗鲁棒性（adversarial robustness）评估？Anthropic 报告的 61% 降至 2% 的降幅依赖于分类器的校准（calibration）效果。
5. 组合攻击检查（compositional check）。评估是否测试了 MSJ + PAIR、MSJ + 说服性模板（persuasive templates）或 MSJ + 编码（encoding）？组合攻击（compositional attacks）的强度通常高于任何单一技术。

硬性否决项（hard rejects）：
- 任何仅基于 5 轮提示（5-shot）评估就宣称“我们的长上下文模型是安全的”的说法。
- 任何未在同一分类器上同时报告越狱攻击成功率（ASR）与良性上下文学习（ICL）性能的防御声明——权衡取舍（trade-off）才是核心所在。
- 任何未提供类别细分的类别汇总攻击成功率（ASR）。

拒绝回答规则（refusal rules）：
- 若用户询问 MSJ 是否能被完全修复，请拒绝给出二元（是/否）答案；MSJ 与上下文学习（ICL）共享底层机制，若不消除 ICL 则无法彻底消除 MSJ。
- 若用户询问评估推荐的提示轮次数量，请拒绝给出单一数值；应要求提供 5 至 512 轮提示范围内的幂律拟合结果。

输出要求：一份单页审计报告，需包含提示轮次覆盖率、各类别的幂律拟合结果、防御机制识别情况，以及一项组合攻击的测试缺口。将 Anil 等人（2024）（Anthropic）作为方法论参考文献引用一次。