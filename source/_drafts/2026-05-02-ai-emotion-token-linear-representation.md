---
title: "把模型当作新生物来研究：从 AI 孪生脑、激活组件到可解释性显微镜"
date: 2026-05-02 20:30:00 +0800
categories: ["Foundation Models"]
tags: [LLM, Interpretability, Mechanistic-Interpretability, Neuroscience, MoE, Prompt-Engineering]
author: Hyacehila
excerpt: "神经科学和大模型可解释性正在靠近同一种方法：先训练一个能完成任务的代理系统，再通过扰动、激活、路由和低维投影去观察它。真正值得研究的不是模型把思考大声说出来的文本，而是内部组件何时亮起、信息如何流动、抽象概念如何被压缩，以及这些结构怎样影响幻觉、置信和 prompt 工程。"
mathjax: true
---

这篇笔记想讨论一个越来越清楚的方法论变化：**我们正在把模型当作一种新生物来研究。**

在神经科学里，研究者开始先训练一个能模拟脑活动或行为的代理神经网络，然后不再只看它的预测分数，而是把它当作实验对象：扰动某个脑区、某个节点、某个隐藏层，观察其他部分如何响应。在大模型可解释性里，研究者也在做相似的事：找到模型内部哪些 feature 会 active，哪些 circuit 在传递信息，哪些方向能控制拒答、情绪、真假判断、幻觉倾向，哪些专家在处理哪些 token。

这和传统的黑盒评测不同。评测只问模型最后答得对不对；可解释性想问的是：模型为什么这样答，内部哪些组件亮了，信息经过了哪条路，哪些抽象概念被压缩成了可读、可干预的内部变量。

如果这个方向继续推进，prompt engineering 也会发生变化。今天的 prompt 工程很多时候还像经验调参：改一句话、换一个格式、加一个例子，看输出有没有变好。下一步更有价值的做法应该是：**写 prompt 时同步观察模型内部状态，看看任务相关组件有没有亮起，幻觉相关信号有没有升高，专家路由有没有跑偏，内部轨迹有没有进入正确区域。** 这就是我理解的“可解释性显微镜”。

## 研究问题与范围

这篇文章围绕四个问题展开。

第一，为什么生物研究会开始使用深度学习模型？例如先用脑活动或行为数据训练一个 surrogate model，再研究这个模型的内部结构。

第二，为什么大模型可解释性也越来越像一种“模型生物学”？我们不只关心输出文本，而要研究 active components、features、circuits、hidden objectives 和 hallucination-related signals。

第三，prompt engineering 能不能从经验技巧走向内部观测？也就是不只比较哪个 prompt 输出更好，还要看 prompt 怎样改变模型内部表示、路由和生成轨迹。

第四，如果要打造一台稳定、全局、一键可用的“模型显微镜”，它应该观察什么？PCA、UMAP、线性探针、SAE feature、activation patching、MoE routing 分别能提供哪类证据？

范围上，这不是一篇完整 survey，而是一篇研究笔记。文中把证据分为三层：

- `confirmed`：来自期刊论文、ACL/NAACL 论文、官方研究页或技术报告，能支撑具体方法与结论。
- `early`：新近 arXiv 或刚出现的研究方向，值得关注，但还没有形成稳定共识。
- `speculative`：基于已有证据提出的研究假设，例如“幻觉可能与事实知识、置信校准和输出策略信号的错位有关”。

## 一页式结论

**第一，神经科学和大模型可解释性正在共享同一种范式：先构造代理系统，再扰动和观察它。**  
南方科技大学团队在 [Nature Methods 2025](https://www.nature.com/articles/s41592-025-02654-x) 的 NPI 工作中，先用 ANN 学习脑网络动力学，再虚拟扰动代理脑来推断 effective connectivity。浙江大学团队在 [Cell Reports 2026](https://www.sciencedirect.com/science/article/pii/S2211124725016092) 的 handwriting 工作中，用任务驱动的 surrogate DNN 帮助解释运动皮层中的层级书写编码。这两类工作都说明：模型不只是预测器，也可以是实验对象。

**第二，大模型可解释性的核心单位正在从“单个神经元”转向 feature、circuit、activation trajectory 和 population geometry。**  
Anthropic 的 [Towards Monosemanticity](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning)、[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) 和 [Mapping the Mind of a Large Language Model](https://www.anthropic.com/research/mapping-mind-language-model) 都在推进一种思路：用 sparse autoencoder 把混叠的激活拆成更可解释的 feature，再研究 feature 如何组合成可读行为。

**第三，“组件亮起”是一个值得研究的概念，但不能被过度拟人化。**  
某个 feature active，不等于模型拥有一个人类式概念；某个专家被路由到，也不等于它就是“数学专家”或“中文专家”。更稳妥的说法是：这些组件在特定输入、层、位置和任务条件下携带可测量的信息，并且有时可以通过干预影响输出。

**第四，模型幻觉可能与事实知识、置信校准和输出策略信号的错位有关，但这仍是研究假设。**  
Anthropic 的 [Language Models Mostly Know What They Know](https://www.anthropic.com/research/language-models-mostly-know-what-they-know) 说明模型在行为层面可以通过 `P(True)` / `P(IK)` 一类信号表达自我评估；[H-Neurons](https://arxiv.org/abs/2512.01797) 这类工作则把幻觉与特定激活组件联系起来。它们还不能直接证明“幻觉只有一个开关”，更合理的研究问题是：模型在生成幻觉前，哪些内部信号已经和正确回答分叉？

**第五，外显 chain-of-thought 不是模型真正思考的全部。**  
NAACL 2024 的 [How Interpretable are Reasoning Explanations from Prompting Large Language Models?](https://aclanthology.org/2024.findings-naacl.138/) 等工作已经提醒我们，模型说出来的解释和内部导致答案的机制可能不一致。真正需要观察的是隐藏状态、注意力、MLP feature、残差流和生成过程中的轨迹变化。

**第六，MoE 模型给可解释性提供了一个更天然的观察窗口。**  
MoE 有 gating network 和 expert routing，研究者可以直接看 token 被分给哪些专家、专家输出如何变化、不同任务是否激活不同专家。[DeepSeekMoE](https://aclanthology.org/2024.acl-long.70/) 关注专家专门化，[Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts](https://aclanthology.org/2024.findings-emnlp.620/) 则把 MoE 用在可解释 reward modeling 中；它们共同说明专家路由本身就是值得研究的内部行为。

## 信息源渠道图

| 主题 | 代表来源 | 证据等级 | 这篇文章如何使用 |
| --- | --- | --- | --- |
| 代理脑与有效连接 | [Nature Methods: NPI](https://www.nature.com/articles/s41592-025-02654-x) | confirmed | 作为“先训练代理脑，再虚拟扰动”的神经科学模板 |
| 代理网络与书写编码 | [Cell Reports: Surrogate DNN handwriting](https://www.sciencedirect.com/science/article/pii/S2211124725016092) | confirmed | 作为“用任务驱动模型解释人脑层级表征”的案例 |
| SAE 与 feature | [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) | confirmed / early | 作为“组件亮起”的主要技术背景 |
| Circuit tracing | [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | early | 作为从 feature 走向计算图的路径 |
| Hidden objectives | [Auditing Language Models for Hidden Objectives](https://assets.anthropic.com/m/317564659027fb33/original/Auditing-Language-Models-for-Hidden-Objectives.pdf) | early | 作为构造性 hidden-objective 审计测试床的案例 |
| 幻觉相关组件 | [H-Neurons](https://arxiv.org/abs/2512.01797) | early | 作为“幻觉是否有可定位内部信号”的研究线索 |
| CoT faithfulness | [NAACL 2024 reasoning explanations](https://aclanthology.org/2024.findings-naacl.138/) | confirmed | 作为“说出来的推理不等于真正机制”的证据 |
| Attribution / patching | [BlackboxNLP 2024 attribution patching](https://aclanthology.org/2024.blackboxnlp-1.25/) | confirmed | 作为 activation / attribution patching 与 circuit recovery 的工具背景 |
| MoE 解释性 | [DeepSeekMoE](https://aclanthology.org/2024.acl-long.70/) / [MoE reward modeling](https://aclanthology.org/2024.findings-emnlp.620/) | confirmed | 作为专家专门化与专家路由解释性的入口 |

## 1. 神经科学模板：先训练代理脑，再研究代理脑

传统神经科学很依赖“记录”和“扰动”。记录告诉我们某个脑区、神经元群体或信号通道在什么时候活动；扰动告诉我们改变这个部分后，系统会发生什么。这就是为什么光遗传、深部脑刺激、皮层电刺激、损伤研究都很重要。它们不只是看相关性，而是试图看到因果链。

问题在于，人脑很难像小模型那样随便扰动。侵入式实验昂贵、受伦理限制，也很难做到全脑尺度。于是出现了一条新的路线：先训练一个足够拟合脑活动或行为的代理模型，再在模型里做虚拟实验。

[Mapping effective connectivity by virtually perturbing a surrogate brain](https://www.nature.com/articles/s41592-025-02654-x) 就是一个非常清晰的例子。NPI 的做法可以概括成三步：

1. 用自监督方式训练 ANN，让它根据前几个时间点的脑区活动预测下一个时间点。
2. 把训练后的 ANN 当作个体化 AI 孪生脑，检查它是否能恢复真实 fMRI 数据中的功能依赖。
3. 在代理脑中对每个节点施加虚拟扰动，观察其他节点响应，从而推断方向、强度和正负属性的 effective connectivity。

这件事的关键不是“用了 AI”，而是研究对象发生了转换。模型先被训练成一个可工作的替身，然后研究者开始对替身做实验。它不像传统回归模型那样只输出一个预测值，而像一个可以被剖开、扰动、记录的实验系统。

浙江大学团队的 [Surrogate deep neural networks reveal hierarchical handwriting encoding in the human motor cortex](https://www.sciencedirect.com/science/article/pii/S2211124725016092) 提供了另一个角度。汉字书写不是简单的手部轨迹问题，它包含字符类别、字形结构、笔画顺序、运动计划和低级执行。传统运动学变量很难完整描述这些高层特征。研究团队训练一个能完成相似书写任务的 surrogate DNN，再比较模型内部层级和真实运动皮层神经信号之间的对应关系。

这类研究的意义在于：**当人类缺乏手工定义抽象变量的能力时，深度模型可以先学习一个可工作的内部表征，然后我们再反过来研究这些表征。** 这不是让模型替代科学，而是让模型变成新的实验仪器。

## 2. 从 Sherringtonian 到 Hopfieldian：模块路径与群体几何

[Nature Reviews Neuroscience: Two views on the cognitive brain](https://www.nature.com/articles/s41583-021-00448-6) 提供了一个很有用的方法论对照。粗略地说，Sherringtonian 风格更关注局部模块、连接路径和功能分工；Hopfieldian 风格更关注群体状态、动力学、吸引子、低维结构和整体计算性质。

把这个对照放到大模型上，会发现两种研究路线都存在。

Sherringtonian 路线会问：这个 attention head 在做什么？这个 MLP neuron 是否检测括号？这个 feature 是否对应拒答？信息从哪一层传到哪一层？某个 token 的事实信息经过了哪条路径？

Hopfieldian 路线会问：模型在高维激活空间里形成了怎样的低维轨迹？正确答案和幻觉答案是否在某一层已经分叉？不同 prompt 是否把模型推到同一个吸引区域？某个任务是否对应一个稳定子空间？MoE 专家路由是否形成可重复的几何结构？

两条路线不是互斥的。机制可解释性早期常从模块、head、neuron 入手，因为这些对象更容易命名。但大模型内部有 superposition、polysemanticity、残差流共享通信、注意力与 MLP 的复杂交互，只靠“每个模块负责什么”很容易解释不完。Hopfieldian 视角提醒我们：很多重要计算可能不是某个单点组件完成的，而是由群体激活的几何形状、流动轨迹和状态空间结构承载。

因此，对大模型来说，一个更现实的显微镜不应只显示“哪个神经元亮了”，还要显示：

- 哪些 feature 在当前 token 位置共同 active；
- 这些 feature 组成了什么样的低维轨迹；
- 正确、错误、幻觉、拒答、过度拒答分别落在哪些区域；
- prompt 改写是否改变了内部状态，而不只是改变了输出文本；
- 干预某个方向、feature 或专家后，后续生成是否进入新轨迹。

## 3. LLM 可解释性：从神经元到 active features

早期“神经元解释”很容易受到一个问题困扰：单个神经元往往不是单义的。一个维度可能同时响应多个不相关特征，一个概念也可能分散在多个维度上。这就是 Anthropic 在 [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) 和后续 monosemanticity 工作中反复讨论的问题。

Sparse autoencoder 的基本想法是：模型原始激活空间里有很多特征叠在一起，我们训练一个稀疏字典，把这些混叠特征拆成更稀疏、更可解释的 feature。这样，研究对象从“某个 neuron 是否对应某个概念”变成“某个 learned feature 是否在某类上下文中 active”。

这就是“亮起”这个概念的技术含义。一个 feature 亮起，通常表示输入在某个层、某个位置触发了一个可读方向或稀疏组件。它可能对应城市、代码漏洞、拒答语气、情绪概念、事实关系，也可能是更抽象的策略或语境状态。Anthropic 的 [Mapping the Mind of a Large Language Model](https://www.anthropic.com/research/mapping-mind-language-model) 把这种 feature map 推向更大模型，说明许多抽象概念可以在模型内部被定位和分析。

但这里要避免两个误区。

第一个误区是把 feature 当成真实世界概念本身。feature 只是模型内部某种可读表征，未必和人类概念一一对应。

第二个误区是把 active 当成充分因果解释。一个 feature 在某个输出前 active，不代表它导致了这个输出。要证明因果作用，还需要做 activation patching、feature steering、ablation 或反事实输入实验。

所以，“组件亮起”最适合被当作显微镜视野里的线索，而不是最终结论。

## 4. 幻觉、置信与“模型是否知道自己没把握”

一个很值得展开的假设是：**模型幻觉可能与事实知识、置信校准和输出策略信号的错位有关。**

这个说法不能直接当作结论，但它是一个非常好的研究问题。LLM 幻觉不只是“知识库里没有这个事实”。很多时候模型似乎有足够线索知道自己不确定，却仍然给出流畅、确定、细节丰富的错误回答。这说明生成过程里至少有三类信号可能彼此错位：

1. 事实知识或检索线索：模型是否在内部有正确事实；
2. 置信或不确定性信号：模型是否知道自己把握不足；
3. 输出策略信号：模型是否仍然选择给出看似完整的答案。

Anthropic 的 [Language Models Mostly Know What They Know](https://www.anthropic.com/research/language-models-mostly-know-what-they-know) 已经提示，语言模型在某些条件下能通过输出层面的自我评估信号表达知识状态。另一方面，[H-Neurons](https://arxiv.org/abs/2512.01797) 这类关于 hallucination-associated neurons 的工作试图把幻觉行为和特定内部组件联系起来。即使这些工作还处在 early 阶段，它们也说明了一个方向：不要只在输出层统计幻觉率，而要追问幻觉生成前内部发生了什么。

一个可执行的实验可以这样做：

1. 准备同一类问题的三组样本：模型确定知道、模型不知道但承认不知道、模型不知道却幻觉回答。
2. 在每一层、每个关键 token 位置记录 residual stream、MLP activations、attention outputs 和 SAE feature activations。
3. 训练线性探针或稀疏探针，预测“模型是否会承认不确定”与“模型是否会幻觉”。
4. 对候选 feature 做 patching：把非幻觉样本中的内部状态 patch 到幻觉样本上，观察是否降低幻觉概率。
5. 反过来把幻觉样本中的状态 patch 到正常样本上，观察是否诱发错误自信。

如果某些 feature 不只相关，而且 patch 后能稳定改变输出策略，我们才可以更强地说它们参与了幻觉机制。否则，它们只能算伴随信号。

## 5. 在生成过程中干预，并继续观察模型

可解释性最有意思的地方在于，它不只是看显微镜照片，还可以做扰动实验。对 LLM 来说，生成过程本身就是一条时间轨迹：每一步 token 生成前，模型都有一组 hidden states、attention patterns、MLP activations、logits 和采样分布。

“在生成过程中干预，并继续观察模型”可以拆成几类实验。

第一类是 activation patching。把一个干净运行中的激活替换到另一个运行中，看输出是否恢复。例如事实问答中，正确样本的某层 residual stream patch 到错误样本后，答案是否被纠正。BlackboxNLP 2024 的 [Attribution Patching Outperforms Automated Circuit Discovery](https://aclanthology.org/2024.blackboxnlp-1.25/) 就是在讨论如何更有效地用 patching 和 attribution 找到重要计算部分。

第二类是 feature steering。先找到某个 SAE feature 或线性方向，再在生成过程中增强或抑制它，观察后续 token 分布如何变化。Anthropic 的情绪向量研究 [Emotion concepts and their function in a large language model](https://www.anthropic.com/research/emotion-concepts-function) 就属于更接近“概念方向可干预”的路线。这里仍要谨慎：干预改变输出，不等于模型有主观情绪；它说明模型内部存在与情绪概念相关的功能性表征。

第三类是 token / hidden-state intervention。比如在模型写到某个中间步骤时，替换某个关键 token 的 hidden state，或者改变某层中与不确定性、拒答、事实检索相关的方向，然后继续 decoding。观察模型是否回到正确轨迹，或者是否从正确轨迹滑向幻觉轨迹。

这类实验对应神经科学里的“扰动-记录”。区别只是对象从脑区变成了 hidden state，从神经刺激变成了 activation intervention，从行为反应变成了后续 token 和内部轨迹。

## 6. Prompt engineering：外显 CoT 不是内部思考

Prompt engineering 很容易陷入一个误区：把模型输出的 chain-of-thought 当成模型真正的思考过程。实际上，外显 CoT 更像模型生成的一段解释文本。它可能和内部计算过程有关，也可能只是事后合理化。

NAACL 2024 的 [How Interpretable are Reasoning Explanations from Prompting Large Language Models?](https://aclanthology.org/2024.findings-naacl.138/) 讨论的正是这个问题：模型给出的推理解释并不总是忠实反映它得出答案的内部依据。更广泛的 CoT faithfulness 研究也在提醒我们，模型可以生成看起来很合理的思考文本，但真正决定答案的信号可能藏在激活空间、注意力路径和中间表示里。

这对 prompt 工程有两个影响。

第一，prompt 的目标不应该只是让模型“说出更像推理的文本”。如果 prompt 只是在诱导一种解释风格，而没有改变内部任务状态，那么它可能提高可读性，却没有提高可靠性。

第二，好的 prompt 应该被内部观测验证。比如同样是要求模型逐步推理，我们可以观察：

- 关键事实 feature 是否更早、更稳定地 active；
- 错误分支是否被抑制；
- 不确定性信号是否在低把握题目上升高；
- 最终答案前的 logits 是否更集中在正确候选；
- MoE 专家路由是否转向在任务样本中稳定出现的路由模式；
- 正确样本和错误样本的 PCA 轨迹是否被 prompt 拉开。

这样，prompt engineering 就从“改字句看输出”变成“改输入看内部机制是否进入正确状态”。

## 7. 可解释性显微镜：PCA、UMAP、探针和 SAE

如果我们真的想打造一个稳定、全局、一键可用的模型显微镜，它至少应该有四层能力。

第一层是采集。对同一批任务样本，记录模型每层、每个 token 位置的 hidden states、attention outputs、MLP activations、logits、SAE feature activations 和 MoE routing。采集对象不能只限于最终 token，因为很多分叉发生在中间步骤。

第二层是投影。人类很难直接理解几千维、上万维激活，所以需要 PCA、UMAP、t-SNE、线性判别、CCA/RSA 等方法把高维状态投到可观察空间。以 PCA 的前三主成分，或 UMAP / t-SNE 的二维、三维嵌入作为起点，看分布里是否出现椭圆、聚类或轨迹结构，是一个很朴素但有效的起点。我们不应该期待这些低维投影解释一切，但它能帮助发现轨迹、聚类、环、分叉、边界和异常点。

第三层是解释。低维图形本身不是解释。看到一个椭圆、两个簇或一条分叉轨迹之后，还要回到原始样本、token、feature 和输出行为，标注这些结构对应什么：正确/错误、会幻觉/不会幻觉、承认不确定/强行回答、拒答/过度拒答、不同 prompt 模板、不同专家路由。

第四层是干预。只有当我们能改变某个方向、feature 或路由，并稳定改变后续输出时，解释才从相关走向因果。否则，PCA 图只能算探索性可视化。

一个最低可行的显微镜实验可以这样设计：

| 步骤 | 操作 | 观察目标 |
| --- | --- | --- |
| 任务选择 | 选择事实问答、数学推理、医疗建议或代码安全任务 | 让正确、错误、幻觉、拒答都有足够样本 |
| 激活采集 | 记录每层 residual stream 和关键 token 位置 | 找到内部状态开始分叉的位置 |
| 低维投影 | 对每层做 PCA / UMAP | 观察是否出现聚类、椭圆、轨迹分叉 |
| 探针训练 | 预测正确性、幻觉、拒答、不确定性 | 判断哪些层最可读 |
| SAE 分解 | 找 active feature 并标注触发样本 | 把几何结构拆成可读组件 |
| 干预验证 | patch / ablate / steer 候选组件 | 检查是否因果影响输出 |

这里要特别小心：低维可视化容易产生幻觉式解释。高维数据投到二维或三维后，结构可能来自投影算法本身，也可能来自样本采样偏差。因此，任何漂亮图形都必须配合 holdout 样本、反事实 prompt、干预实验和定量指标。

## 8. 抽象概念、压缩和线性表征

为什么大模型在抽象概念领域更值得研究？一个原因是，语言模型本质上被迫压缩世界。它要从海量文本中学会实体、关系、风格、事实、任务、规范、情绪、意图、时间、空间、代码结构和社会语境，然后把这些东西压缩进有限维度的参数和激活空间。

这种压缩可能让一些高层概念变得可读。比如 sentiment、truth、refusal、emotion、toxicity、uncertainty 这类变量，经常能被线性探针、activation steering 或 SAE feature 捕捉到。所谓“情绪 token 线性化”可以在这里重新理解：不是某个 token 本身线性，而是情绪相关的内部表征在某些层和位置上呈现出可读、可干预的方向。

[Linear Representations of Sentiment in Large Language Models](https://arxiv.org/abs/2310.15154)、[The Geometry of Truth](https://arxiv.org/abs/2310.06824) 和 [Language Models Represent Space and Time](https://arxiv.org/abs/2310.02207) 都指向同一个现象：非线性网络内部可能形成相对线性的高层变量。Anthropic 的情绪向量研究进一步把这类变量和行为干预联系起来。

但线性方向不是全部机制。它更像显微镜下的一条可见轴。真正的计算可能是多方向、局部线性、分层组合、跨 token 传播、受上下文门控的。线性探针能读出一个变量，不代表模型只用这个变量；steering 能改变行为，也不代表这个方向没有副作用。

所以，抽象概念研究最好的问题不是“模型有没有一个情绪神经元”或“模型有没有一个真理方向”，而是：

- 哪些抽象变量在什么层、什么位置、什么语境下可读；
- 它们是否跨 prompt、跨语言、跨任务泛化；
- 它们能否通过干预改变输出；
- 它们是否和其他变量混叠；
- 它们是否参与模型真实使用的计算路径。

## 9. MoE：专家路由也是可解释性对象

Mixture-of-Experts 给可解释性提供了一个很自然的入口。普通 dense Transformer 每个 token 基本经过同一组参数；MoE 模型则会用 gating network 把 token 路由到少数专家。这样，模型内部多了一类显式可观测变量：哪个 token 激活了哪些专家，gate 分数是多少，专家输出如何改变 residual stream。

这让 MoE 看起来很像模块化系统，但不能太快下结论。专家不是人类命名的功能模块，gate 的目标也不是让专家语义整齐，而是优化训练损失、负载均衡和计算效率。一个专家可能处理多种表面不相关的 token，也可能在不同层承担不同角色。

即便如此，专家激活仍然值得研究。[DeepSeekMoE](https://aclanthology.org/2024.acl-long.70/) 直接讨论了 MoE 语言模型中的专家专门化；[Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts](https://aclanthology.org/2024.findings-emnlp.620/) 则把多目标 reward modeling 与 MoE gate 结合起来，让不同偏好目标的选择更可解释。后续关于 MoE 专家专门化、路由偏差和 expert collapse 的研究，也在把专家路由当成模型行为的一部分。

一个 MoE 显微镜可以观察：

- 不同任务类型是否稳定激活不同专家；
- 幻觉样本和正确样本是否有不同路由模式；
- 某些专家是否和拒答、安全、数学、代码、事实检索相关；
- gate entropy 是否和模型不确定性相关；
- 专家输出做 PCA 后是否形成任务结构；
- 强制改路由是否改变输出质量或幻觉率。

尤其值得做的是专家输出降维。对每个 token，不只记录被选中的专家编号，还记录各专家输出向量。然后在同一任务上对这些输出做 PCA / UMAP，看专家是否形成可解释的几何分工。如果某个专家输出区域总是在幻觉前出现，或者某个路由组合总是在正确推理中出现，那就有了进一步 patching 和 ablation 的候选。

## 10. 大模型生物学家会研究什么

如果把大模型当成新生物来研究，那么未来的研究者可能不只是“模型工程师”。他们更像模型生物学家：不一定每天训练新模型，而是长期观察一个复杂系统，给它做实验，建立概念、分类、测量仪器和干预方法。

这类研究者会关心的问题包括：

- 哪些内部结构对应稳定的抽象概念；
- 哪些结构只是在特定数据集上看起来可解释；
- 模型什么时候知道自己不知道；
- 幻觉从哪一层开始偏离；
- prompt 怎样改变内部轨迹；
- CoT 文本和内部状态什么时候一致，什么时候不一致；
- MoE 专家是否形成可复用功能分工；
- 安全对齐是否只是输出约束，还是改变了内部机制；
- 后训练增强了哪些已有机制，又抑制了哪些机制。

这里最需要的不是又一个漂亮 demo，而是一台稳定的显微镜。它应该能对任意任务批量采集中间状态，自动做低维投影，标注 active features，显示专家路由，支持 patching 和 steering，并把相关性证据与因果证据分开。

如果这样的工具成熟，解释性研究的工作流会变成：

1. 选一个行为现象，例如幻觉、拒答、数学错误、prompt 敏感性。
2. 采集大量成功和失败轨迹。
3. 自动定位内部分叉层和候选 feature / expert / direction。
4. 对候选组件做干预验证。
5. 把稳定结果沉淀成可复用的机制地图。

到那时，prompt 工程也会更像实验科学：不是凭感觉改一句话，而是看 prompt 是否把模型推入了正确的内部状态。

## 争议与局限

第一，可解释性结果很容易被过度解释。一个 feature 有人类可读标签，不代表它就是人类概念；一个 PCA 图有结构，不代表模型真的按这张图思考。

第二，干预实验可能有副作用。steering 一个方向能提升某个行为，也可能破坏其他任务。拒答方向、情绪方向、truth 方向都可能和其他变量混叠。

第三，外部行为和内部机制不一定一一对应。同一个输出可以由不同内部路径产生，同一个内部 feature 也可能在不同上下文里服务不同功能。

第四，当前显微镜还很不稳定。SAE feature 的训练、解释、对齐、跨模型迁移都还在发展中；activation patching 对层和位置很敏感；MoE 路由受训练目标和负载均衡影响；低维可视化容易误导。

第五，生物类比有启发性，但不能滥用。大模型不是大脑，没有神经递质、身体、演化史和主观体验。把它叫作“新生物”，只是强调研究方法上的相似：复杂、可训练、可扰动、可观察、需要新的测量语言。

## 可行动研究想法

**想法一：幻觉前的内部置信分叉。**  
在事实问答中采集正确、承认不知道、幻觉三类轨迹，定位哪一层最早把三者分开。用 patching 验证候选 feature 是否能把幻觉改成不确定表达。

**想法二：prompt 是否真的改变内部思考。**  
比较直接回答、CoT、few-shot、self-consistency、工具调用提示在同一任务上的激活轨迹。判断它们是只改变输出格式，还是改变关键 feature 和轨迹分叉。

**想法三：MoE 专家路由与幻觉。**  
在 MoE 模型上记录 gate entropy、expert id、expert output PCA，研究幻觉样本是否有稳定路由模式。进一步强制路由或屏蔽专家，看幻觉率是否改变。

**想法四：PCA 显微镜基线。**  
为每个任务自动生成每层 PCA 前三主成分图，必要时补充 UMAP / t-SNE 的二维或三维嵌入，叠加正确性、prompt 模板、输出长度、置信度、专家路由和 SAE feature 标签。先不追求完整解释，只建立稳定观测面板。

**想法五：CoT faithfulness 与内部轨迹一致性。**  
把模型写出的 CoT 分成关键推理节点，再检查内部激活是否在对应位置出现事实、变量、关系或分支 feature。如果 CoT 文本和内部轨迹不一致，就标记为低忠实解释。

## 推荐阅读路径

如果从神经科学侧进入，可以先读 [Nature Methods NPI](https://www.nature.com/articles/s41592-025-02654-x) 和 [Cell Reports handwriting surrogate DNN](https://www.sciencedirect.com/science/article/pii/S2211124725016092)，理解 surrogate model 怎样成为实验对象。

如果从机制可解释性侧进入，可以按 [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)、[Towards Monosemanticity](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning)、[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)、[Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) 这条线读。

如果关心 CoT 忠实性，可以先读 [How Interpretable are Reasoning Explanations from Prompting Large Language Models?](https://aclanthology.org/2024.findings-naacl.138/)。如果关心 prompt 解释与输入归因，再读 [PromptExp](https://arxiv.org/abs/2410.13073) 和 [TokenSHAP](https://arxiv.org/abs/2407.10114)。

如果关心 MoE，可以从 [DeepSeekMoE](https://aclanthology.org/2024.acl-long.70/) 和 [Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts](https://aclanthology.org/2024.findings-emnlp.620/) 开始，再继续看专家路由、expert specialization 和 expert activation 的后续工作。

## 暂定结论

我现在越来越相信，可解释性不是大模型研究的附属工具，而可能成为下一阶段的基础科学工具。

原因很简单：当模型越来越大、行为越来越复杂、外显回答越来越会伪装成合理解释时，只看输入输出已经不够了。我们需要看到内部组件什么时候亮起，看到信息怎样流动，看到抽象概念怎样被压缩，看到 prompt 是否真的改变了内部状态，看到幻觉在生成前是否已经露出迹象。

神经科学已经给了一个模板：复杂系统不能只靠观察行为，也要记录、扰动、建模和解释。大模型研究现在也在走向同一个方向。

真正的目标不是把模型解释成几个漂亮词，而是做出一台可靠的显微镜。它能让我们在模型生成时看见内部世界：哪些 feature active，哪些专家接管，哪些轨迹分叉，哪些信号在说“我知道”，哪些信号在说“我其实没把握”。

到那个时候，prompt engineering、模型安全和机制解释会合到一起。我们不再只是问“怎么写 prompt 才能让模型输出好答案”，而是问：

**怎样把模型推入一个内部结构上更可靠、更诚实、更可控的状态。**
