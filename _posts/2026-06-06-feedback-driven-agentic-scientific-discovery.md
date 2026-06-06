---
layout: blog-post
title: "从反馈回路看 Agent 如何进入科学发现"
title_en: "Feedback Loops for Agentic Scientific Discovery"
date: 2026-06-06 20:00:00 +0800
categories: [Agent 系统]
tags: [Agents, Evaluation, Methodology]
author: Hyacehila
excerpt: "Gemini 解宇宙弦积分提供了一个清楚的观察入口：当候选生成、搜索控制和外部验证连成闭环时，Agent 才有机会把模型能力转化为可改进的工作过程。"
excerpt_en: "Gemini's cosmic string case shows how candidate generation, search control, and external validation can turn model capability into an iterative scientific workflow."
featured: true
math: false
---

# 从反馈回路看 Agent 如何进入科学发现

Shunyu Yao 在 [The Second Half](https://ysymyth.github.io/The-Second-Half/) 里有一个判断我一直记得：AI 的下半场会从“解决问题”转向“定义问题”，evaluation 会比 training 更重要。放到 Agent 语境里，这句话可以再往前推一点：问题定义不只是写清楚 prompt，还包括为系统安排一个会持续返回信号的环境。

模型已经足够会说、会写、会联想，也能在很多问题上快速生成候选方案。但如果没有反馈，这些候选很难从“看起来合理的解释”变成可迭代的工作。一个问题一旦被放进能验证、能比较、能回滚、能反复试错的系统里，模型的生成能力才开始有了搜索的形状。

我想用 AI for Science 当切口，是因为科学场景对验证特别敏感，也更容易看清反馈回路的结构。但这篇文章的落点并不只在科学发现。科学里的 Agent 可以被理解为一个放进反馈回路里的候选生成器；换到编码、检索、浏览器操作、数据分析等场景，这个理解同样成立。人类的角色也没有消失，只是更多转向定义问题、设计验证器、控制搜索空间，并判断结果是否真的有意义。

## 宇宙弦论文值得看的地方

最近这篇 [Solving an Open Problem in Theoretical Physics using AI-Assisted Discovery](https://arxiv.org/abs/2603.04735) 表面上在讲宇宙弦引力辐射功率谱里的一个球面积分。我更关注的是它的方法部分：作者把一个理论物理问题组织成了一个可搜索、可验证、可反馈的 Agent 系统。

他们先搭了一个 neuro-symbolic system，再把模型放进这个系统里工作。Gemini 负责生成数学假设、展开路径和 Python 验证代码；Tree Search 负责组织候选解法空间；高精度数值计算负责检查候选公式；执行错误、数值误差和不稳定现象再被回注到模型上下文里。整个过程大约探索了 600 个候选节点，超过八成的分支因为代数错误、数值发散或灾难性抵消被剪掉。

这套结构可以压缩成一条很短的链：

```text
LLM 生成候选推导
→ Tree Search 展开解法空间
→ Python verifier 执行和打分
→ traceback / error / instability 回注
→ 剪枝、修正、换路线
→ 继续搜索
```

这和普通聊天式使用模型的差别很大。普通聊天里，模型输出一个答案，人读一读，觉得对或不对。这里的系统把模型输出放在候选节点的位置：它需要能被运行、被比较、被反驳，最后接受 verifier 检查。

negative prompting 也很有意思。系统找到一条有效路线之后，作者会明确禁止模型继续用它，引导它寻找别的方法。最后模型给出了六类解析方法：单项式展开、Gaussian lifting、Legendre 谱方法、Volterra 递推、Gegenbauer 展开等。这里更值得借鉴的是搜索治理，而非某个提示词模板：既要避免模型太早收敛到一条路线，也要避免它长时间在同一类表示里打转。

## Feedback 在这里起什么作用

这篇论文给我的提醒是：LLM 的能力并不直接等于发现能力。模型擅长生成候选，发现则来自候选在环境里被反复筛选。

什么样的反馈才撑得起这个筛选？我自己琢磨下来，大概有几个绕不开的要求。它得够快，也得够局部。如果每改一版都要等几个月湿实验，Agent 很难高频迭代；如果反馈只是一句“错了”，下一轮生成也很难发生实质变化。更有用的反馈，是告诉它哪个参数区间发散、哪个公式数值不稳定、哪个 Python traceback 出错。宇宙弦那套之所以转得起来，正是因为数值积分和公式验证能在程序里反复跑，每次失败都带着具体位置。

反馈还需要尽量客观，并且可比较。让另一个 LLM 评价“这个推导看起来严谨”，可以作为辅助意见，但它的强度不够。更可靠的信号来自数值基准、形式验证、真实实验数据、仿真器、类型检查、单元测试、守恒律或统计检验。光有对错也不够，搜索系统需要知道哪个候选更好、哪条分支值得继续。没有一个可比较的 score，Tree Search、evolutionary search、ranking tournament 都会失去抓手。最后，这些反馈不能只躺在日志里，它要能回到下一轮上下文里，成为模型修正推导、替换表示、放弃错误假设的材料。

换句话说，feedback 把模型的语言生成变成了环境里的试错过程。没有它，模型更多是在一次性回答；有了它，系统才开始表现得像一个会不断筛选候选的搜索过程。

## 可控搜索让单轮能力变成持续改进

把视野放宽，这篇宇宙弦论文其实和过去几年几个重要 AI for Science 系统共享同一类骨架。

[FunSearch](https://www.nature.com/articles/s41586-023-06924-6) 把数学问题改写成程序搜索问题：LLM 生成候选程序，evaluator 自动执行和打分，高分程序进入数据库，再被拿来提示下一轮生成。Google DeepMind 的官方介绍里也强调，FunSearch 的关键是把 LLM 的创造性和 automated evaluator 配对，用 evaluator 过滤幻觉和错误想法。

[AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) 也是类似结构，只是规模更工程化。它是一个 evolutionary coding agent，由 Gemini 生成和修改算法，由自动评估器验证答案，再用演化框架保留有希望的程序。它能处理算法发现和优化问题，前提仍然是用户能定义 evaluation function。

[Co-Scientist](https://www.nature.com/articles/s41586-026-10644-y) 把这套机制放进生命科学假设生成里。它不依赖单一模型一次性给出假设，而采用多 Agent 系统：Generation agent 产生假设，Reflection agent 扮演虚拟 peer reviewer，Ranking agent 组织 idea tournament，Evolution agent 重组和改写高分假设，Proximity agent 控制多样性，Meta-review agent 综合反馈。这里的 feedback 不再只是数值误差，也包括文献证据、假设新颖性、可测试性、虚拟辩论和排序。

[The AI Scientist](https://arxiv.org/abs/2408.06292) 更靠近端到端自动化，试图覆盖 idea、code、experiment、paper writing 和 automated review 的全流程。它展示了自动研究流程的想象力，也提醒我们另一个问题：当系统开始自动写论文，automated reviewer 的可信度会变成主要瓶颈。如果 evaluator 不够可靠，自动科学很容易生产出形式上像论文、但验证不足的产物。

[Robin](https://www.nature.com/articles/s41586-026-10652-y) 则把多 Agent 系统接到实验生物学流程里：文献检索、假设生成、实验建议、数据分析、再生成更新后的假设。它的重要性在于把反馈从纯计算扩展到了 lab-in-the-loop。只是反馈成本明显更高，所以系统需要更仔细地安排每一次实验机会。

这些系统领域不同，但结构上很接近：

```text
候选生成
→ 搜索或排序
→ 外部工具执行
→ 验证器打分
→ 反馈进入下一轮
```

模型提出可能性，系统让可能性接受选择压力。这个分工一旦成立，Agent 就不再只是一次性给建议，而是在一个受控环境里持续推进候选。

## 科学发现 Agent 的通用结构

如果要从这些系统里抽象出一个可迁移架构，我会写成这样：

```text
Problem Interface
→ Candidate Generator
→ Search Controller
→ Domain Tools / Simulators / Databases
→ Verifier / Evaluator
→ Feedback Memory
→ Human Scientist
```

`Problem Interface` 决定问题能不能被 Agent 处理。科学问题不能只以一句自然语言存在，它要被写成输入、输出、约束、资源预算和成功标准。FunSearch 要求你提供问题描述、seed program 和 evaluator；AlphaEvolve 要求你定义初始程序和评价函数；宇宙弦论文把物理问题压成 `I(N, α)` 的可计算积分。这一步做不好，后面的 Agent 很容易失去方向。

`Candidate Generator` 是 LLM 最擅长的位置。它可以快速提出公式、程序、实验方案、证明路线、机制假设，也可以从跨学科知识里做联想。Gemini 案例集 [Accelerating Scientific Research with Gemini](https://arxiv.org/abs/2602.03837) 反复强调，模型在 cross-pollination、counterexample search、formalization、iterative refinement 上有明显价值。

`Search Controller` 负责让生成保持方向。它决定什么时候 exploitation，什么时候 exploration；什么时候保留高分候选，什么时候强制多样性；什么时候让模型继续局部修复，什么时候换表示。Tree Search、evolutionary search、Elo tournament、negative prompting，本质上都在做搜索控制。

`Domain Tools` 是科学世界的接口。数学里是数值积分、符号计算、定理库；生物里是文献数据库、序列分析、实验数据处理；材料里是 DFT、结构稳定性预测、合成可行性检查；软件和算法里是测试集、benchmark、profiling 和形式化 checker。

`Verifier` 是这套结构里最需要认真设计的部分。它把候选从叙述拉回外部约束。不同学科的 verifier 形态不同：数值误差、反例、实验读数、仿真结果、统计显著性、类型系统、Lean/Coq 证明、专家复核，都可以是反馈。但无论哪一种，它都应该尽量独立于模型自评。

`Feedback Memory` 负责沉淀失败。哪些路线不稳定，哪些表示有用，哪些假设被反例杀死，哪些局部引理可以复用，这些都应该进入系统记忆。否则很多失败会在不同轮次里反复出现。

`Human Scientist` 仍然是不可缺席的一层。人类负责定义问题是否值得做，判断验证器是否真的衡量了科学目标，识别结果是否只是 benchmark trick，以及把系统发现转化成学科共同体能理解、能审查、能复现的知识。

## 哪些问题适合这种方法

这类系统更适合那些“难求解、易验证”的问题，不适合覆盖所有科学问题。FunSearch 的 Nature 论文里也强调，很多数学和计算机科学问题虽然难以求解，但候选解的质量容易评估。宇宙弦积分也是如此：解析式难找，但给定 `N` 和 `α` 后，可以用高精度数值积分做基准。

把范围放宽一点，适合 Agent 的问题往往长得有点像。它能被拆成中间步骤，每一步都可检查，避免拖到最后才发现整件事错了；它有清楚的候选表示，无论那是程序、公式、分子、材料结构、实验方案、因果图、证明草稿还是参数化模型；它至少在早期筛选阶段有便宜的局部反馈，不必每个候选都押上一次昂贵实验；它有明确的失败信号，比如发散、反例、测试失败、物理约束违反、实验无效、统计不显著，这些都比“看起来不太好”有用得多。还有一点容易被忽略：它得允许搜索。候选空间足够大，模型的联想才有价值；但空间也不能完全无边界，否则搜索就会变成随机尝试。

反过来，如果一个问题没有可靠 verifier、反馈极慢、成功标准模糊，只能靠长篇解释自圆其说，就不宜过早把它称为自动发现。在这种场景里，AI 仍然可以做文献整理、假设启发和写作辅助，只是还不能被当作一个发现系统。

## 不要把 AI for Science 都混成 Agent

现在市面上 AI for Science 的研究很多，但它们不都属于同一种东西。

[AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w) 和 [GNoME](https://www.nature.com/articles/s41586-023-06735-9) 是非常重要的科学 AI 系统。前者预测生物分子复合物结构，后者用图网络扩展材料发现空间。它们说明 AI 可以极大压缩候选空间，改变科学工作流。但严格说，它们更像强大的预测模型或发现工具，不是本文讨论的那种“生成候选、执行搜索、吸收反馈、迭代行动”的 Agent 系统。

FunSearch、AlphaEvolve、Co-Scientist、Robin 和宇宙弦论文里的系统更接近 Agentic Science，因为它们显式拥有候选生成、搜索控制、外部验证和反馈闭环。差别只是在反馈形态不同：FunSearch 和 AlphaEvolve 的反馈来自程序执行和评分；宇宙弦案例来自数值验证；Co-Scientist 来自多 Agent critique、文献证据和实验可行性排序；Robin 进一步接入实验结果。

所以，与其问“用了 AI 没有”，不如问：它有没有把科学问题变成一个反馈足够丰富的环境。如果没有反馈闭环，它可能仍然有用，但还不是这里讨论的科学发现 Agent。

## Feedback 不只是科学发现的事

写到这里我得把话说回来。前面一直在谈科学，但 candidate → search → verify → feedback 这条骨架，跟科学并不绑定。它描述的是任何一个想让模型反复试错的 Agent 系统。换个领域，变的只是 verifier 长什么样。

很直接的对照是 coding agent。它的 verifier 是测试、编译器、类型检查、lint，反馈是 traceback 和失败用例，和宇宙弦那套几乎一一对应，只是把数值积分换成了 `pytest`。所以现在好用的编码 Agent，体验的稳定性往往来自它背后那套能让模型看到自己哪行代码挂了的回路。RAG 和检索类系统的反馈藏得深一些：检索到的片段相不相关、答案有没有被证据支撑、引用对不对得上，这些就是它的“数值误差”，只是更难量化。computer-use 和浏览器 Agent 的反馈是 DOM 状态、截图、动作到底有没有生效；multi-agent 系统则把反馈做成了 critique、投票和 tournament 排序，让一个 Agent 的输出先过另一批 Agent 这一关。

这些系统都没有“高精度数值积分”那么好的 verifier，反馈也更脏、更模糊、更贵。但缺了它，结局很相似：Agent 很容易停留在生成层，看起来像在干活，却很难判断它到底对不对。科学发现之所以是个好例子，恰恰因为它的 verifier 更好、反馈闭环更清楚，把这件事的结构照得更亮。需要认真设计反馈的远不止科学，只是在别的地方，这个缺口常常被一句“模型能力还不够”盖过去了。

## 可以带走的方法论

要把这套东西搬到一个新领域，不管是新学科还是一个新的 Agent 产品，我认为第一步通常不该急着选模型，可以先把几个问题问清楚。

候选对象到底是什么？公式、程序、机制假设、实验方案、分子结构、材料配方，还是一次浏览器操作、一段检索结果？这些候选又靠什么快速验证，有没有现成的模拟器、数据库、代理指标、形式化规则、历史数据、测试集或专家标注能提供反馈？光有反馈还不够，关键是它能不能局部化，能不能指出是哪一步错、哪个约束被违反、哪个区域不稳定，而不是只给一个笼统的“失败”。接着才轮到搜索怎么控制：是要 Tree Search、evolutionary search、beam search、tournament ranking，还是一个更朴素的 generate-and-rerank 就够了。最后别忘了人在哪：定义任务、设计 verifier、审查高分候选、批准真实实验，还是最终把它写成能发表、能交付的东西。

这几个问题，通常比“用哪个 Agent 框架”更重要。框架只是运行时，更影响成败的是问题接口和反馈设计。

绕了一圈，我想把这篇宇宙弦论文的启发收回到最开始那句话上：

**Agent 能否持续变强，更多取决于我们能否把问题工程化成一个可搜索、可验证、可反馈的环境。科学发现只是把这件事演示得很清楚的一个例子。**

在这样的环境里，LLM 不应该被放在最终裁判的位置。它是高通量的候选生成器，是跨学科的联想器，是不会疲倦的推导助手。让系统可靠的是 verifier，让它强大的是 search，而让它能持续运转的是 feedback。三者里面，反馈往往最容易被省掉，最后也最容易变成系统的短板。

我猜未来好用的 Agent，大概都不会长成一个万能聊天框。它更像一组围绕具体任务搭起来的工作台：有结构化的问题入口，有领域工具，有实验、仿真或线上环境的接口，有一个尽量独立于模型自评的验证器，有可回放的轨迹，还有人能随时审查和接管的中间工件。模型越强，这套外部结构反而越重要，因为生成越快，验证的瓶颈就越明显。

所以下半场我更惦记的，不是“AI 会不会想出新东西”。它当然会，这点我并不担心。更麻烦的是另一头：我们能不能给它一个足够好的反馈世界，让那些新东西被筛选、被反驳、被改进，最后沉淀成可信的知识，而不只是又一批漂亮但不可验证的文本。

## 参考资料

- Shunyu Yao, [The Second Half](https://ysymyth.github.io/The-Second-Half/).
- Michael P. Brenner, Vincent Cohen-Addad, David P. Woodruff, [Solving an Open Problem in Theoretical Physics using AI-Assisted Discovery](https://arxiv.org/abs/2603.04735), arXiv:2603.04735, 2026.
- David P. Woodruff et al., [Accelerating Scientific Research with Gemini: Case Studies and Common Techniques](https://arxiv.org/abs/2602.03837), arXiv:2602.03837, 2026.
- Bernardino Romera-Paredes et al., [Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6), Nature, 2023.
- Google DeepMind, [FunSearch: Making new discoveries in mathematical sciences using Large Language Models](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/), 2023.
- Google DeepMind, [AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/), 2025.
- Alexander Novikov et al., [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131), arXiv:2506.13131, 2025.
- Vivek Natarajan et al., [Accelerating scientific discovery with Co-Scientist](https://www.nature.com/articles/s41586-026-10644-y), Nature, 2026.
- Google DeepMind, [Co-Scientist: A multi-agent AI partner to accelerate research](https://deepmind.google/blog/co-scientist-a-multi-agent-ai-partner-to-accelerate-research/), 2026.
- Chris Lu et al., [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292), arXiv:2408.06292, 2024.
- Sakana AI, [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://sakana.ai/ai-scientist/), 2024.
- Ali Essam Ghareeb et al., [Robin: A multi-agent system for automating scientific discovery](https://arxiv.org/abs/2505.13400), arXiv:2505.13400, 2025.
- FutureHouse, [Demonstrating end-to-end scientific discovery with Robin](https://www.futurehouse.org/research-announcements/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system), 2025.
- Peter Jansen et al., [DISCOVERYWORLD: A Virtual Environment for Developing and Evaluating Automated Scientific Discovery Agents](https://arxiv.org/abs/2406.06769), NeurIPS Datasets and Benchmarks, 2024.
- Allen Institute for AI, [DiscoveryWorld project page](https://allenai.github.io/discoveryworld/).
- Google DeepMind and Isomorphic Labs, [Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w), Nature, 2024.
- Google DeepMind, [Scaling deep learning for materials discovery](https://www.nature.com/articles/s41586-023-06735-9), Nature, 2023.
