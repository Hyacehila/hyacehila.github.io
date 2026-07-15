---
title: "从反馈回路看 Agent 如何把生成变成搜索"
title_en: "Feedback Loops for Agentic Search"
date: 2026-06-06 20:00:00 +0800
categories: ["Agent Systems", "Agent Evaluation & Governance"]
tags: ["Feedback Loops", "Evaluation"]
author: Hyacehila
excerpt: "宇宙弦积分和 Trae Agent 分别来自科学发现和编码任务，却都指向同一件事：LLM 生成候选以后，系统要能搜索、验证、剪枝和选择。"
excerpt_en: "The cosmic string case and Trae Agent come from different domains, but both ask the same design question: after an LLM generates candidates, how does the system search, verify, prune, and choose?"
permalink: '/blog/2026/06/06/feedback-driven-agentic-scientific-discovery/'
---

Shunyu Yao 在 [The Second Half](https://ysymyth.github.io/The-Second-Half/) 里有个判断我一直记得：AI 的下半场会从解决问题转向定义问题，evaluation 与 RL 的 generalization 会比训练算法更重要。我把这个判断放到 Agent 上，会得到一个更具体的系统问题：先定义问题，再给系统搭一个能不断返回信号的环境。这样，Agent 才有机会在不进行参数训练的情况下解决新问题。

模型已经很会说、会写、会联想，也能在很多问题上快速给出候选方案。可是一旦没有反馈，这些候选很容易停在“看起来合理的解释”。把问题放进一个能验证、能比较、能回滚的系统里，生成才会慢慢变成搜索。

这篇文章原本从 AI for Science 切入，因为科学场景对验证特别敏感，反馈回路也更容易看清。后来我发现，同一套问题也出现在编码任务里。宇宙弦积分论文采用科学任务的 `tree search + verifier`，Trae Agent 采用编码任务的 `ensemble search + pruning + selector`。两个例子看起来很远，其实都在追问同一个麻烦：LLM 把候选空间撑大以后，系统怎么管住这些候选。

人的角色没有消失，只是换了位置：定义问题，设计验证器，控制搜索空间，剪掉坏分支。最后还得有人判断，结果到底有没有意义。

## 宇宙弦论文值得看的地方

[Solving an Open Problem in Theoretical Physics using AI-Assisted Discovery](https://arxiv.org/abs/2603.04735) 表面上在讲宇宙弦引力辐射功率谱里的一个球面积分。我更关注的是它的方法部分：作者把一个理论物理问题组织成了一个可搜索、可验证、可反馈的 Agent 系统。

他们先搭了一个 neuro-symbolic system，再把模型放进这个系统里工作。Gemini 负责生成数学假设、展开路径和 Python 验证代码；tree search 负责组织候选解法空间；高精度数值计算负责检查候选公式；执行错误、数值误差和不稳定现象再被回注到模型上下文里。整个过程大约探索了 600 个候选节点，超过八成的分支因为代数错误、数值发散或灾难性抵消被剪掉。

这个过程大概是这样的：

```text
LLM 生成候选推导
→ tree search 展开解法空间
→ Python verifier 执行和打分
→ traceback / error / instability 回注
→ 剪枝、修正、换路线
→ 继续搜索
```

这和普通聊天式使用模型的差别很大。普通聊天里，模型输出一个答案，人读一遍，再判断对不对。这个系统把模型输出放在候选节点的位置：候选要能运行、比较和被反驳，之后才交给 verifier。

negative prompting 也很有意思。系统找到一条有效路线之后，作者会明确禁止模型继续用它，引导它寻找别的方法。最后模型给出了六类解析方法：单项式展开、Gaussian lifting、Legendre 谱方法、Volterra 递推、Gegenbauer 展开等。对我更有用的是其中的搜索治理：延缓模型向单一路线收敛，同时推动它离开已经反复使用的表示。

## 编码任务里的同一条回路

[Trae Agent](https://arxiv.org/abs/2507.23370) 把类似的问题搬到了软件工程里。它讨论的是 repository-level issue resolution：给定一个代码库和一个自然语言描述的 issue，系统通过 test-time scaling 生成一组 candidate patches，再从里面挑一个最终补丁。这个任务在形式上很像科学发现里的候选公式搜索，只是候选对象从公式和推导路径换成了 patch，验证器从数值积分换成了测试、执行环境和代码理解。

我会把 Trae 的结构记成三步。第二步里的两个闸门是平级的，这点容易写错：

```text
Patch Generation：多样化生成候选 patch
→ Patch Pruning：deduplication 与 regression testing 平级剪枝
→ Patch Selection：repo-level understanding + majority voting
```

多样化采样放在最前面。Trae 一开始就用 coder agent 并行生成多个 patch，而不是赌一个 Agent 一次写对。多样性来自高温采样、多次独立运行，也来自多个模型的 round-robin mixture。这个设计背后的判断很朴素：同一个 issue 反复跑，整体通过率可能差不多，但每次解决的是不同子集。候选空间里确实有互补信息，后续系统要做的是把它捞出来。

候选数量也有代价。ensemble size 增大时，oracle 上界会上升，也就是说只要能选中正确 patch，理论潜力更高。与此同时，adversary 下界会下降，因为候选里混入更多错误、冗余和干扰项。候选越多，单轮 prompt selector 看到的信息越长，context dilution 越严重，选择本身反而会变难。

多样化生成接上候选治理，才会真正提高找到有效方案的机会。

接下来是 patch pruning。它承担中途过滤，和最终选择是两个独立的治理层。Trae 里面的 pruning 由两个平级策略组成：

```text
patch deduplication：移除 redundant candidates
regression testing：移除 faulty / low-quality candidates
```

deduplication 解决的是冗余。多个 patch 可能表面不同，实际修改等价；如果全丢给 selector，只会浪费上下文窗口，还会让投票和比较被重复方案污染。Trae 先用 `unidiff` 把候选转成结构化对象，再在结构化对象层面比较，避免让 LLM 只凭自然语言表面判断两个补丁是否相近。

regression testing 解决的是不可行或低质量候选。Trae 会从原代码库中运行测试，保留原本就能通过的 tests，再由 tester agent 选择更像 true regression tests 的子集。随后每个 candidate patch 都要在这些 regression tests 上单独跑；失败的 patch 被剪掉，能通过的才进入 selection。这里的可验证反馈是搜索中途的过滤器。

还有一个很小心的设计：如果所有候选都没通过 regression tests，Trae 会保留全集进入后续选择。自动选出来的 regression tests 可能不准，误杀正确 patch 的代价太高。这条回退路径承认 verifier 也可能出错，让剪枝保留自我修正的余地。

patch selection 依赖 repository-level 的程序理解。Trae 的 selector agent 会读相关代码片段，查看 issue 描述里提到的文件、patch 修改的文件和依赖相关代码；同时生成和执行测试，收集 execution traces，用静态理解和动态验证拼出 repository-level understanding。最后再用多数投票压一压单次 selector 的不稳定。

复杂任务里，剪枝和最终选择分两轮引入 feedback。pruning 先把候选空间压小，让系统能够深入搜索。selector 再做更贵、更细的判断，给出最终答案。

## Feedback 在这里起什么作用

`feed-forward` 和 `feedback` 是 Agent 外层 harness 中两套平级、互补的机制。这篇文章把重点放在后者。Thoughtworks 在 [Harness engineering and agent feedback: Exploring AI coding sensors](https://www.thoughtworks.com/en-us/insights/blog/generative-ai/harness-engineering-agent-feedback-exploring-ai-coding-sensors) 里把外层 harness 分成两类。Skills、规则和参考文档在生成前提供知识与约束；测试、静态分析和执行结果观察实际产出。两者缺一不可：前者告诉模型该怎么做，后者告诉它刚才做得怎么样。

我过去写 Prompt、RAG、Skills 和上下文工程时，花了很多篇幅研究知识怎样进入模型。对执行后的信号怎样回来、是否足够快、能不能指出错误，我写得少得多。这种失衡会直接影响系统：它可能装进了很多知识，却拿不到刚才那一步的质量信号。coding agent 和科学发现 Agent 的进展提醒我，生成能力与知识注入只是系统的一半。测试、执行环境与 verifier 把结果送回循环之后，Agent 才能修正路线，继续往下走。

把宇宙弦论文和 Trae Agent 合在一起看，feedback 从生成阶段就开始进入任务。模型负责提出候选，环境持续运行、筛选并返回信号，发现就在这轮往复里发生。

生成时有执行反馈。宇宙弦案例里，模型写出的 Python 验证代码会暴露 traceback、数值发散、灾难性抵消；Trae 的 coder agent 也会探索代码库、复现 bug、生成 patch、重跑 reproduction tests。这类反馈直接帮助模型修正当前路线。

能支撑筛选的反馈至少要快，也要局部。每改一版都要等待几个月湿实验，Agent 就很难高频迭代；反馈如果只剩一句“错了”，下一轮生成也很难真正变好。宇宙弦的 tree search 会剪掉代数错误、数值不稳定和误差太大的分支。Trae 的 patch pruning 通过结构化去重剪掉 redundant candidates，再通过 regression testing 剪掉 faulty candidates。搜索系统需要知道哪个候选更好、哪条分支值得继续，tree search、evolutionary search 和 ranking tournament 都需要一个可比较的 score 作为抓手。

到了选择阶段，feedback 又变成证据。科学场景里，这可能是数值基准、形式验证、实验结果、文献证据和专家复核；编码场景里，则是静态代码理解、依赖关系、测试执行结果和 execution traces。它们能帮助 selector 判断哪个候选更值得留下。


## 可控搜索让单轮能力变成持续改进

宇宙弦论文、Trae Agent 和过去几年出现的几个 AI for Science 系统，其实都在用相近的骨架。

[FunSearch](https://www.nature.com/articles/s41586-023-06924-6) 把数学问题改写成程序搜索问题：LLM 生成候选程序，evaluator 自动执行和打分，高分程序进入数据库，再被拿来提示下一轮生成。Google DeepMind 的官方介绍把它概括为 LLM 创造性与 automated evaluator 的组合，由 evaluator 过滤幻觉和错误想法。

[AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) 也是类似结构，只是规模更工程化。它是一个 evolutionary coding agent，由 Gemini 生成和修改算法，由自动评估器验证答案，再用演化框架保留有希望的程序。它能处理算法发现和优化问题，但用户得先定义 evaluation function。

[Co-Scientist](https://www.nature.com/articles/s41586-026-10644-y) 把这套机制放进生命科学假设生成里。它把一次性假设生成拆成多 Agent 系统：Generation agent 产生假设，Reflection agent 扮演虚拟 peer reviewer，Ranking agent 组织 idea tournament，Evolution agent 重组和改写高分假设，Proximity agent 控制多样性，Meta-review agent 综合反馈。这里的 feedback 除了数值误差，还包括文献证据、假设新颖性、可测试性、虚拟辩论和排序。

[The AI Scientist](https://arxiv.org/abs/2408.06292) 更靠近端到端自动化，试图覆盖 idea、code、experiment、paper writing 和 automated review 的全流程。它让人看到自动研究流程可以长成什么样，也把另一个问题推到台前：当系统开始自动写论文，automated reviewer 的可信度会变成瓶颈。如果 evaluator 不够可靠，自动科学很容易产出形式上像论文、但验证不足的东西。

[Robin](https://www.nature.com/articles/s41586-026-10652-y) 则把多 Agent 系统接到实验生物学流程里：文献检索、假设生成、实验建议、数据分析、再生成更新后的假设。它把反馈从纯计算推到了 lab-in-the-loop。代价也很直接：实验反馈更贵，每一次实验机会都得省着用。

领域不同，骨架却很像：

```text
候选生成
→ 搜索或排序
→ 外部工具执行
→ 验证/剪枝/集成
→ 反馈进入下一轮
```

LLM 负责提出可能性，系统负责施加选择压力。这个分工一旦成立，Agent 就从一次性建议器变成了在受控环境里持续推进候选的系统。

`tree search` 依赖几个前提：动作空间相对有限，状态可以被表示，下一步动作能被枚举，候选结果也能由某种反馈比较。棋类、程序搜索和组合优化通常具备这些条件，自然语言问题则往往缺少明确边界。一个问题可以沿着无数方向展开，每个节点都能继续写成一大段话，节点之间很难比较，失败也常常没有明确位置。

LLM 带来的变化更微妙：自然语言空间仍然很大，但它很擅长临时造出一组可操作的候选结构。一个问题可以被拆成子问题，模糊想法可以被写成假设列表，推理路线可以被整理成几个分支，实验设计可以落到步骤，数学推导可以变成可执行代码，软件修改可以变成 patch，回答质量也可以被拆成几个评价维度。原本连续、松散、不可枚举的语言空间，到了这里，至少在局部有了可以展开、可以选择、可以回退的形状。搜索也就有了下手的地方。

## Feedback-driven Agent 的通用结构

如果从这些系统里抽一层可迁移的结构，我现在会写成这样：

```text
Problem Interface
→ Candidate Generator
→ Diversity / Sampling Controller
→ Pruning Gates
→ Domain Runtime / Tools
→ Verifier / Evaluator
→ Selector / Aggregator
→ Feedback Memory
→ Human Owner
```

`Problem Interface` 负责把问题写成输入、输出、约束、资源预算和成功标准。编码任务还要说明 issue、代码库状态、测试环境、允许修改范围和验收标准。FunSearch 需要问题描述、seed program 和 evaluator；AlphaEvolve 需要初始程序和评价函数；宇宙弦论文把物理问题压成 `I(N, α)` 的可计算积分；Trae 则把软件 issue resolution 写成从候选 patch 集合里选出一个能通过 golden tests 且满足 issue 要求的 patch。

`Candidate Generator` 是 LLM 最擅长的位置：提出公式、程序、实验方案、证明路线、机制假设，也可以生成多个 patch、测试和重构路径。它的位置是高通量候选源，最终裁判交给后面的验证与选择层。

`Diversity / Sampling Controller` 管覆盖面。宇宙弦论文里的 negative prompting 会阻止模型重复已有路线；Trae 用高温采样、独立运行和多模型 mixture 扩大 patch 多样性；Co-Scientist 里还有 Proximity agent 控制假设之间的距离。足够的多样性才能让搜索离开同一答案的反复改写。

`Pruning Gates` 把候选空间压回可选择范围。Trae 在这里给了一个很清楚的样子：deduplication 和 regression testing 都属于 pruning，但它们是平级闸门。一个处理冗余，一个处理不可行或低分。换到别的领域，dedup 可以是分子结构等价、证明路线等价、检索证据重复；testing 可以是仿真失败、约束违反、统计不显著、引用不支持结论。

`Domain Runtime / Tools` 是 Agent 接触外部世界的接口。数学里是数值积分、符号计算、定理库；生物里是文献数据库、序列分析、实验数据处理；材料里是 DFT、结构稳定性预测、合成可行性检查；软件里是测试集、benchmark、profiling、类型系统和真实代码执行环境。

`Verifier / Evaluator` 把候选从叙述拉回外部约束，理想情况下独立于模型自评。数值误差、反例、实验读数、仿真结果、统计显著性、类型检查、单元测试、Lean/Coq 证明、专家复核，都可以成为反馈。verifier 也有边界。Trae 对 regression testing 的保守策略提醒我们：只要存在误杀风险，系统就要保留回退路径。每个学科和问题需要的 `Verifier` 也各不相同。

`Selector / Aggregator` 留到最后，和 pruning 各有分工。剪枝负责缩小候选集，最终答案还要经过选择。简单任务可以 generate-and-rerank，复杂任务则需要更贵的 selector。Trae 的 selector agent 要建立 repository-level understanding，还要通过多数投票降低单次判断不稳定性；Co-Scientist 用 ranking tournament 和 meta-review 做类似的聚合。

`Feedback Memory` 用来记住失败。哪些路线不稳定，哪些表示有用，哪些假设被反例杀死，哪些 patch 类型经常破坏兼容性，哪些测试容易误判，这些都该进系统记忆。否则很多失败会在不同轮次里反复出现。

`Human Owner` 负责价值判断与最终责任。人类要判断问题值不值得做，验证器是否量到了真正的目标，结果是否只是在钻 benchmark 的空子。最后也要有人把系统发现转成共同体或工程团队能审查、能复现、能交付的东西。

## 哪些问题适合这种方法

这类系统更适合“难求解、易验证”的问题，不适合覆盖所有科学或工程问题。FunSearch 的 Nature 论文里也强调，很多数学和计算机科学问题虽然难以求解，但候选解的质量容易评估。宇宙弦积分也是如此：解析式难找，但给定 `N` 和 `α` 后，可以用高精度数值积分做基准。编码任务也类似：真实 issue 的修复很难一次写对，但 patch 能不能编译、测试是否通过、是否破坏既有行为，至少有一部分可以验证。

适合 Agent 的问题通常有几项共同特征。任务可以拆成中间步骤，每一步都能检查，避免拖到最后才发现整件事错了。候选表示也要清楚，无论那是程序、公式、分子、材料结构、实验方案、因果图、证明草稿、参数化模型，还是一个 patch。早期筛选最好能获得便宜的局部反馈，把昂贵实验和人工 review 留给更少的候选。失败信号也要明确，比如发散、反例、测试失败、物理约束违反、实验无效、统计不显著、patch parse failure。这些信号都比“看起来不太好”有用得多。

这类问题还得允许搜索。候选空间要足够大，模型的联想才有价值；同时也要有边界，避免搜索退化成随机尝试。Trae 的经验尤其提醒我，候选空间扩大以后，选择难度会同步上升。搜索系统既要看生成上界，也要算选择成本。

反过来，如果一个问题没有可靠 verifier、反馈极慢、成功标准模糊，只能靠长篇解释自圆其说，就不宜过早把它称为自动发现或自动解决。在这种场景里，AI 仍然可以做文献整理、假设启发、代码草稿和写作辅助，只是还不能被当作一个闭环搜索系统。

## 不要把所有 AI 系统都混成 Agent

现在 AI for Science 和 coding agent 的研究都很多，这些系统其实分属不同类型。

[AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w) 和 [GNoME](https://www.nature.com/articles/s41586-023-06735-9) 都是真正有用的科学 AI 系统。前者预测生物分子复合物结构，后者用图网络扩展材料发现空间。它们说明 AI 可以大幅压缩候选空间，改变科学工作流。但严格说，它们更像强大的预测模型或发现工具，不是本文讨论的那种“生成候选、执行搜索、吸收反馈、迭代行动”的 Agent 系统。

软件工程里也一样。能生成代码只是起点。一个稳定的 coding agent 通常还要包含代码库探索、工具调用、测试执行、错误恢复、轨迹记录和最终选择。Trae 的价值不只在 Pass@1 数字，而在于它把 repository-level issue resolution 组织成 generation、pruning、selection 三个相互独立又能衔接的阶段。

判断一个系统是否属于 feedback-driven agentic search，可以看五件事：候选是否经过多样化生成，是否有结构化表示，是否有可验证的剪枝，是否有独立 selector，以及是否有人能够审查和接管。缺少这条反馈回路的系统可能仍然有用，只是属于别的类型。

## 可以带走的方法论

要把这套东西搬到一个新领域，不管是新学科还是一个新的 Agent 产品，我会先把几个问题问清楚，再选模型。

先定义候选对象：公式、程序、机制假设、实验方案、分子结构、材料配方、一次浏览器操作、一段检索结果，或者一个 patch。候选对象越清楚，后面的去重、验证和选择才越有抓手。

让模型一次生成多条路线。高温采样、多次独立运行、多模型 mixture、negative prompting、proximity control 都可以用来扩大候选差异。

剪枝可以有多个闸门，彼此不一定是从属关系，但 feedback 都要可靠。Trae 里的 deduplication 和 regression testing 就是平级策略：一个移除 redundant candidates，一个移除 faulty candidates。科学任务里也可以同时有数值误差、物理约束、仿真结果、统计检验、文献证据等多个剪枝面。

剪枝之后仍然需要 selector。这个 selector 可以是 ranking tournament、majority voting、repo-level selector agent、专家复核，或者多种方式组合。通过局部 verifier 只代表候选跨过了一道闸门，最终正确还需要更多证据。

还要控制 ensemble size。候选越多，上界可能越高，但上下文、执行、去重、测试和选择成本都会上升。test-time scaling 要求采样、剪枝和选择同步扩展，任何一层跟不上都会成为新的瓶颈。

**绕了一圈，我还是会回到最开始那句话。Agent 能不能持续变强，很大程度上取决于我们能不能把问题工程化成一个可搜索、可验证、可剪枝、可选择的反馈环境。科学发现和编码任务就是很好的例子。**

LLM 更适合生成候选。测试和 verifier 先排除明显错误，search 组织候选空间，pruning 控制候选规模，剩下的判断再交给 selector 和 human owner。模型生成得越快，验证与选择越容易成为瓶颈。

到了下半场，AI 能不能想出新东西反而没那么让我担心。我更惦记的是：我们能否给它一个足够好的反馈世界，让那些新东西经过筛选和反驳，最终沉淀成可信的知识和可交付的软件。

## 参考资料

- Thoughtworks, [Harness engineering and agent feedback: Exploring AI coding sensors](https://www.thoughtworks.com/en-us/insights/blog/generative-ai/harness-engineering-agent-feedback-exploring-ai-coding-sensors), 2026.
- Shunyu Yao, [The Second Half](https://ysymyth.github.io/The-Second-Half/).
- Pengfei Gao, Zhao Tian, Xiangxin Meng et al., [Trae Agent: An LLM-based Agent for Software Engineering with Test-time Scaling](https://arxiv.org/abs/2507.23370), arXiv:2507.23370, 2025.
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
