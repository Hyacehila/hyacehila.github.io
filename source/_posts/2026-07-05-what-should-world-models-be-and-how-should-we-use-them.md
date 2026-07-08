---
title: 世界模型应该是什么样子，又该如何被利用？
title_en: "What Should World Models Be, and How Should We Use Them?"
date: 2026-07-05 00:00:00 +0800
categories: ["Foundation Models", "Model Mechanics"]
tags: [World Models, Multimodality, Model Mechanics]
author: Hyacehila
excerpt: "世界模型最有价值的部分，是让智能体在行动前模拟未来。本文从几条路线的差异讲到 JEPA，再回到 Critiques of World Models 与 PAN。"
excerpt_en: "World models matter less as beautiful video generators than as internal simulators for agents. This post compares the main routes, explains why JEPA is appealing, and revisits Critiques of World Models and PAN."
mathjax: true
---

## 为什么要搞世界模型？

LLM 的成功，很大程度上来自一个简单而强的目标：预测下一个词。可是智能体面对真实环境时，只预测下一个词不够。它需要知道如果自己往左走会发生什么，如果把杯子倾斜水会不会洒，如果现在安慰一个正在哭的人，对方会停止哭泣、继续崩溃，还是把这种安慰理解成打扰。

相比一台更大的看图说话机器，或者一个更强的视频生成器，它更像智能体脑内的试验场：给定当前状态和一个可能的动作，模型模拟接下来会出现的状态。用更形式化一点的话说，就是从 $s$ 和 $a$ 出发，估计 $s' \sim p(s' | s,a)$。问题从识别世界变成了在行动前预测世界。

[Critiques of World Models](https://arxiv.org/abs/2507.05169) 是本文的核心参考资料，作者把重心放在 purposeful reasoning and acting 上：世界模型要模拟所有可行动的可能性，让智能体能据此选择下一步。换句话说，世界模型首先服务决策，观赏性排在后面。

对智能体来说，这个差别很关键。一个没有世界模型的 agent 往往只能在真实环境里试错，或者依赖语言模型（当然语言模型带来的 prior 非常有价值，这也是目前语言模型智能体的核心）给出听起来合理的建议。一个有世界模型的 agent 则可以先在内部跑几条分支：这一步会不会撞上障碍，后续收益会不会更高，有没有更安全的替代动作。AlphaGo 里的搜索、自动驾驶里的轨迹预测、机器人里的动作规划，都可以看作这种思路在不同约束下的局部实现。

所以，讨论世界模型时，我们更希望知道这些被模拟出来的未来，对行动到底有没有用？画面精细度当然有价值，但它不该压过这个问题。水杯、球、车辆、网页、队友的情绪、长期策略，表面上完全不同，背后都是同一个问题：模型能否把当前状态、候选动作和后果接起来。

## Landscape：JEPA 之外的几条路

世界模型这几年突然热起来，一个麻烦也随之出现：很多东西都叫 world model，但它们想解决的问题并不相同。考虑到我比较喜欢 Lecun 的 JEPA ，先粗略看几条不以 JEPA 为中心的路线。

### 游戏与交互世界模型

游戏路线的代表包括 Google DeepMind 的 [Genie 2](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/)、Microsoft 的 [WHAM / Muse](https://www.nature.com/articles/s41586-025-08600-3)，以及 Decart 和 Etched 的 [Oasis](https://oasis-model.github.io/)。这些系统的共同点是把世界模型放进可交互环境里：输入动作后，模型继续生成接下来的画面。Genie 2 可以从单张提示图生成可玩的 3D 环境，Oasis 则展示了类似 Minecraft 的实时生成式游戏环境。

这条路线的好处很清楚。游戏天然有状态、动作、反馈，也容易让人直观看到世界是否跟着动作演化。问题是，游戏环境通常被规则、视角和动作接口强烈限制。它们很适合训练和评估某类具身智能体，却还不能直接说明模型理解了开放现实世界。一个能在 Minecraft 风格环境里接键盘输入的模型，离能处理厨房、街道、办公室和社交场景，还有很长一段距离。这一类专用模型很有价值，虽然游戏是虚拟环境，但训练很多特殊用途的具身智能体，这样的环境可能就足够了。

### 3D 场景与空间智能

第二条是 3D 与空间智能路线。World Labs 的 [Marble](https://www.worldlabs.ai/blog/marble-world-model) 就属于这一类（或者说目前好像只有World labs关注这个问题）：从文本、图像或视频生成可浏览、可编辑的 3D 世界。这个方向背后的直觉很强，现实世界首先是空间性的。物体有位置、尺度、遮挡、深度和可达性；如果模型没有稳定的空间表征，很多推理都会变得飘。

我赞同 3D 表示很重要，尤其是机器人、AR、游戏、仿真这类场景。但如果把理解世界完全等同于重建三维空间，问题又会变窄。空间结构只是世界的一层。行动后果还涉及物理、意图、任务目标、社会关系和时间尺度。更现实的判断是：3D 世界生成会成为世界模型的重要组件，但不太可能单独承担完整的智能体推理。当然 3D 场景的生成非常有价值，当我们以后去探索一个空间智能体的时候，一个可用的稳定的 3D 环境可能是一切的基础。

### Physical AI 与自动驾驶世界模型

第三条路线面向 Physical AI，尤其是自动驾驶和机器人。NVIDIA 的 [Cosmos](https://arxiv.org/abs/2501.03575) 把 world foundation model 定位成可为机器人和自动驾驶定制的基础平台；Wayve 的 [GAIA-2](https://arxiv.org/abs/2503.20523) 则更明确地面向自动驾驶，生成多视角、可控的驾驶视频，并用道路语义、车辆动态、天气和 agent 配置来控制场景。

这类模型是现实世界版本的游戏交互世界，核心在于将游戏规律变成物理规律，因此他们的使用场景都类似，难度也是接近的。任务边界比较清楚：车怎么动，行人在哪里，天气如何影响传感器，某个罕见场景是否值得加入训练。它们的问题也来自这种清楚的边界。模型往往深度绑定特定传感器、任务和控制接口，泛化到家庭机器人、网页 agent 或战略规划时，很多结构要重做。它们是很好的专用世界模型，不一定是通用世界模型的最终形态。

### 通用视频生成模型

第四条是最特殊的世界模型路线：视频生成。OpenAI 在 Sora 技术报告 [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) 里明确提出，扩展视频生成模型可能是通向物理世界通用模拟器的一条路。Google DeepMind 的 [Veo](https://deepmind.google/models/veo/) 系列也在不断增强视频生成的控制、质量和一致性。

视频生成的优势很直接：输出可见，进展也容易被感知。可从世界模型角度看，它的弱点同样明显。多数视频生成模型仍是 prompt-to-video，它们生成一条固定轨迹，不方便让 agent 在中途插入动作、比较多个后果，也未必有清晰的状态和动作表示。它们可以学到一部分世界规律，但如果缺少可交互、可分支、可评价的结构，就更像世界的影像模型，离行动的世界模型还差一步。

这几条路线都很有价值。游戏路线强调交互，3D 路线强调空间，Physical AI 路线强调可控物理场景，视频生成路线强调视觉动态。world model 这个词下面，其实藏着一组围绕如何模拟未来的工程选择。

在视频生成领域还有数字人这个分支，不过他们的研发思路并不太一样，通用视频生成强调是覆盖不同的场景，电影，短剧或者新闻报道，通用和泛化是这里的关键，因此往往依赖大规模的 End2End 训练与下一帧生成。而数字人更像是一个特化的场景，我们只关注人的表情、神态以及他们和声音的同步，当然也有 Vidu S1 这样的将单纯视频生成与 Interaction Model 衔接的产物，强调模型根据人的回答进行即时反馈，其背后的技术暂不明确，可能是分离的 Interaction Model 与数字人层，也可能是独立的 End2End 训练的数字人 Interaction Model，不过这是一个缺少想象力的技术，关于 Interaction Model 本身就足够了。（数字人的分支技术还包括实时语音交互，TTS and ASR 等语言专门技术，但更不值得在此进行讨论了）

接下来要讲的 JEPA，把问题切到了另一层：也许理解世界不必从生成完整世界开始。

## JEPA：我更看好的抽象预测路线

LeCun 在 [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) 里提出的核心判断是：智能系统需要在多个抽象层级上学习表征、预测和规划。然后他提出了 JEPA，Joint Embedding Predictive Architecture。它的出发点很简单：绕开原始数据本身，预测原始数据背后的表征。

这点和人类的直觉很接近。我们预测松手后球会落地时，脑内不会逐像素渲染球的每一帧运动，也不会复原背景墙面的纹理。我们抓住的是少数有用变量：球、手、重力、支撑关系、时间。JEPA 想让模型在抽象嵌入空间里学会这类预测，把每个像素、每段措辞、每个难以处理的细枝末节都放到次要位置。

典型 JEPA 结构可以简化成三部分：context encoder 把可见部分编码成上下文表征，target encoder 把目标区域编码成目标表征，predictor 根据上下文去预测目标表征。训练时比较表征，像素与 token 留在后面。这样做的一个好处是，模型可以忽略低层噪声，把注意力放到语义结构、物体关系和动态变化上。

I-JEPA 是这条路线在图像上的起点。[I-JEPA](https://arxiv.org/abs/2301.08243) 从一张图的上下文块预测目标块的表征，不依赖手工设计的数据增强，也不要求模型补全像素。相比 masked autoencoder 之类重建式方法，它更想学到高层语义；缺失纹理只是干扰项。

V-JEPA 把这个想法推到视频。[V-JEPA](https://arxiv.org/abs/2404.08471) 只用 feature prediction 训练视频表征，不用文本、负样本、重建或预训练图像编码器。到 [V-JEPA 2](https://arxiv.org/abs/2506.09985)，路线又往物理行动迈了一步：先用大规模视频做 action-free 预训练，再用少量机器人轨迹数据做 action-conditioned world model，让视频表征进入预测和规划环节。

音频方向也有类似扩展。[A-JEPA](https://arxiv.org/pdf/2311.15830) 把 I-JEPA 的思想移到音频频谱上，预测被遮住区域的潜在表征，避开原始波形或频谱细节的重建。

视觉语言方向的 [VL-JEPA](https://arxiv.org/abs/2512.10942) 更接近传统VLM的嵌入形式改进。传统 VLM 大多沿着“视觉编码器 + 对齐层 + LLM decoder”的接口往前走：LLaVA 用 MLP projector 把视觉 patch 特征映射到语言模型能接收的空间，InstructBLIP 用 Q-Former 从图像里抽取和问题相关的视觉信息，再交给语言模型逐 token 生成答案。这条路线很有效，VQA、caption、多模态对话都靠它跑起来。但它也把很多轻量任务推向了完整语言生成。只是判断画面里有没有异常、检索一段视频、回答一个短分类问题，也常常要启动大 decoder 一字一句吐出文本。

VL-JEPA 主要调整这个接口。它可以被拆成 X-Encoder、Y-Encoder、Predictor 和 Y-Decoder 四块：X-Encoder 负责图像或视频输入，Y-Encoder 把目标文本编码成连续语义嵌入，Predictor 根据视觉表征和查询去预测这个目标嵌入，Y-Decoder 只在需要可读答案时出场。模型先在语义空间里靠近答案，再决定是否把答案翻成 token。这个顺序值得注意。它把 VLM 里最贵、最容易受语言先验牵引的生成接口往后移，让理解、匹配、监测和轻量问答先停留在嵌入层。

这样做还有一个作用，和传统的文本嵌入一样：同义表达会自然靠近。“灯灭了”和“房间变暗了”在 token 层差得很远，在语义嵌入里可以指向同一个事件。对世界模型来说，这比复述成某句固定文本更有用。智能体关心的是场景状态变了、风险升高了、下一步该避开某个区域；至于最后说成哪句话，经常是后处理问题。

论文里还报告了选择性解码可以减少约 2.85 倍解码操作，同时保持相近性能。它说明了一个方向：视觉语言模型未必每次都要把理解变成完整文本。VL-JEPA 仍要面对细节保真、表征接地和复杂推理的问题，可它把什么时候需要语言这个问题重新摆上了桌面。

这就是我偏向 JEPA 的原因。它把理解世界从重建世界的路径里移开了。视频生成和 3D 生成当然有用，但它们容易把大量算力花在对决策没有帮助的细节上。JEPA 更像是在问：如果智能体真正需要的是下一步的可行动信息，那我们为什么不直接学习这层信息？将表征学习做到极致，未尝不是比学习人类的信息架构更合适的方案。

JEPA 的困难同样明显：表征空间是否稳定，latent prediction 是否真的保留了任务所需信息，动作条件如何加入，长时预测误差怎么处理，如何把抽象表征接到真实行动。这些问题都没有被彻底解决。但作为世界模型路线，我觉得它抓住了一条值得押注的线索：世界模型不应该被人类牵着走。

### 路线对比

把几条路线放在一起，可以看到真正的分歧：预测发生在哪一层。

| 路线 | 代表工作 | 预测对象 | 优势 | 主要问题 |
|:---|:---|:---|:---|:---|
| 游戏/交互世界模型 | Genie 2, Muse, Oasis | 可交互游戏状态和画面 | 有动作输入，容易评估交互 | 场景和动作空间偏窄 |
| 3D/空间智能 | World Labs Marble | 空间结构和可浏览世界 | 空间一致性强，适合仿真和编辑 | 不等于完整行动推理 |
| Physical AI | Cosmos, GAIA-2 | 物理/驾驶/机器人场景 | 任务约束清楚，工程价值高 | 领域绑定强 |
| 通用视频生成 | Sora, Veo | 视频帧或潜变量 | 视觉质量高，覆盖面广 | 多为固定轨迹，缺少可行动分支 |
| JEPA | I-JEPA, V-JEPA, V-JEPA 2, VL-JEPA | 抽象表征 | 避免重建无关细节，适合语义预测 | 表征接地和长时控制仍难 |

### JEPA 适合什么场景

JEPA 的应用场景不该被写成万能清单，只预测抽象表征其实意味着很大的应用局限性。目前他应该被用在几类确实符合其优势的地方。

第一类是实时感知和交互。智能眼镜、监控、车载系统、机器人很多时候并不需要每一帧都生成文字说明，它们更需要知道语义是否发生了变化。VL-JEPA 的选择性解码正适合这个思路：先在嵌入空间里监测和预测，到需要汇报、解释或交互时再输出语言。

第二类是端侧和边缘部署。连续嵌入预测通常比完整 token 生成更轻，如果任务只是分类、检索、异常检测或轻量 VQA，就没有必要每次都启动一个大语言解码器。这里的收益主要是更快、更省、更稳定，聊天能力排在后面。

第三类是具身智能。V-JEPA 2 这类视频表征模型如果能和少量动作数据结合，就有机会在机器人规划里充当世界动态的压缩表征。它不需要在脑内生成一段好看的视频，只要能判断哪个动作更可能让杯子被拿起、物体被推到目标位置，就已经有价值。

第四类是内容理解和检索。嵌入空间天然适合相似性计算，也更容易把同义说法压到相近区域。VL-JEPA 对文本语义嵌入的预测，刚好把这件事变成训练目标的一部分。对检索、审核、去重和开放词汇分类来说，这种鲁棒性比漂亮的生成结果更重要。

## Critiques of World Modeling：争论和 PAN

[Critiques of World Models](https://arxiv.org/abs/2507.05169) 对 JEPA 这一派并不完全买账。它承认当前很多世界模型讨论过度围绕视频生成，但也认为 LeCun/JEPA 路线里有一组值得细查的假设。我读下来，它更像是在问：如果世界模型最终要服务通用智能体，那么只做固定维度的连续表征预测，会不会太窄？

第一个分歧是数据。LeCun 经常强调感知数据，尤其是视频和行动经验，因为真实世界的信息量远大于文本。Critiques 作者的回应是：数据量不等于信息密度。视频里有大量冗余像素，而语言是人类长期压缩经验的结果，里面包含因果、社会规则、反事实、计划、价值判断这些不容易直接从视觉里观察到的东西。世界模型如果只盯着感知流，很容易学到世界长什么样，却漏掉人在这个世界里为什么这样行动。JEPA-VL 引入语言表征，也一定程度说明了这一点，我们还是不能抛弃语言。

第二个分歧是表征。JEPA 倾向于连续嵌入，因为连续空间适合梯度优化，也能承载细腻感知差异。Critiques 作者则强调离散 token 的价值：语言、符号、概念、可组合的记忆结构，都是长期推理非常需要的东西。这个问题很难现在就给出一个答案。连续表征适合低层感知，离散表征适合稳定概念和长程推理。真正有前途的世界模型大概率需要混合表征。

第三个分歧是架构。JEPA 反感直接生成原始观测，因为像素重建会引入大量不可预测细节。Critiques 作者则说，完全去掉生成式 decoder 会带来接地问题：模型在 latent space 里预测得很接近，不代表它预测的东西在真实观察空间里有意义。换句话说，下一表征预测不能完全替代下一观测约束。这并不推翻 JEPA，但提醒我们：latent space 不能自说自话，它必须经常被现实校准。保留生成观测带来的校准能力，对于训练嵌入表征也是有价值。

第四个分歧是训练目标。JEPA 使用 latent-space objective，试图绕开原始数据空间的复杂性。Critiques 作者担心 latent loss 有坍缩和不可识别风险，需要很多额外正则去维持表征质量。他们更推崇以观测数据为锚的生成式损失，因为它至少让模型的内部状态对外部世界负责。这里我会保留一点怀疑：生成式损失确实更接地，但也可能把模型拉回细节重建。更好的做法也许是在不同抽象层上设置不同强度的生成约束。

第五个分歧是使用方式。LeCun 体系里常见的想法是把世界模型放进 MPC，让 agent 在推理时滚动模拟若干步，选择代价最低的动作。Critiques 作者认为 MPC 适合短视野控制，但通用智能体还需要从模拟经验中学习，把世界模型当成训练场，通过 RL 或其他学习信号把策略内化下来。或许我们还是要混合训练？ 不过现在的 JEPA 生成嵌入表征对我来说就已经足够有用了。

这组批评最后引出了 PAN。[PAN: A World Model for General, Interactable, and Long-Horizon World Simulation](https://arxiv.org/abs/2511.09057) 介绍了具体的架构：用语言动作控制世界演化，用 LLM 风格的 latent dynamics backbone 维持抽象状态和长程知识，再用视频 diffusion decoder 把状态还原成可观察的未来片段。

PAN 的特色可以概括成几条。它吃多模态经验，同时使用连续和离散表征；它把自回归生成放进分层架构里，高层动态和低层视觉细节分开处理；它用可观察数据给内部状态上锚，也把世界模型当作 agent 学习和试错的模拟环境。PAN 和 Critiques 的观点是匹配的。作者先批评 JEPA 可能过度依赖感知、连续表征、latent loss 和短视野 MPC，然后给出一个几乎逐项相反的方案：文本也重要，离散 token 也重要，生成式接地也重要，世界模型也应该参与训练智能体。它对 JEPA 的态度更接近补全，重点在补上缺掉的几个维度。

我个人仍然更偏向 JEPA 的基本直觉：智能体需要预测有用表征，复刻世界表象排在后面。但 Critiques 和 PAN 提醒得很及时。如果 JEPA 的抽象表征不能接地、不能行动、不能长程积累，那它就会停在漂亮的 representation learning。世界模型最终要回到 agent：它能不能让系统更好地选择动作，更少地真实试错，更稳地处理没有见过的场景。

## 结语

世界模型最有价值的部分，是给智能体一个可行动的想象空间。视频、3D、游戏、Physical AI 都在补这块拼图，JEPA 则提醒我们：有时最该预测的未来落在抽象表征里，像素只是其中一层。

我更看好 JEPA，是因为它把理解世界从重建表象里解放出来。但我也不想忽略 Critiques 和 PAN 的问题意识：世界模型不能只在 latent space 里自洽，它还要被真实观测校准，被行动接口调用，并最终帮助 agent 学会更好的策略。

也许真正成熟的世界模型不会只属于某一条路线。它会有 JEPA 式的抽象预测，也会有 PAN 式的生成接地；会利用文本里的压缩经验，也会利用视频和交互里的物理经验。最后还是要回到智能体每天都会遇到的问题：如果我这么做，世界接下来会怎样？

## 参考资料

- Eric Xing, Mingkai Deng, Jinyu Hou, Zhiting Hu, [Critiques of World Models](https://arxiv.org/abs/2507.05169), arXiv:2507.05169, 2025.
- Yann LeCun, [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf), 2022.
- David Ha, Jürgen Schmidhuber, [World Models](https://arxiv.org/abs/1803.10122), arXiv:1803.10122, 2018.
- Google DeepMind, [Genie 2: A large-scale foundation world model](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/), 2024.
- Anssi Kanervisto et al., [World and Human Action Models towards gameplay ideation](https://www.nature.com/articles/s41586-025-08600-3), Nature, 2025.
- Decart and Etched, [Oasis: A Universe in a Transformer](https://oasis-model.github.io/), 2024.
- World Labs, [Marble: A Multimodal World Model](https://www.worldlabs.ai/blog/marble-world-model), 2025.
- NVIDIA, [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575), arXiv:2501.03575, 2025.
- Wayve, [GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving](https://arxiv.org/abs/2503.20523), arXiv:2503.20523, 2025.
- OpenAI, [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/), 2024.
- Google DeepMind, [Veo](https://deepmind.google/models/veo/).
- Mahmoud Assran et al., [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243), arXiv:2301.08243, 2023.
- Adrien Bardes et al., [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471), arXiv:2404.08471, 2024.
- Mido Assran et al., [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985), arXiv:2506.09985, 2025.
- Zhengcong Fei, Mingyuan Fan, Junshi Huang, [A-JEPA: Joint-Embedding Predictive Architecture Can Listen](https://arxiv.org/abs/2311.15830), arXiv:2311.15830, 2023.
- Delong Chen et al., [VL-JEPA: Joint Embedding Predictive Architecture for Vision-language](https://arxiv.org/abs/2512.10942), arXiv:2512.10942, 2025.
- PAN Team, [PAN: A World Model for General, Interactable, and Long-Horizon World Simulation](https://arxiv.org/abs/2511.09057), arXiv:2511.09057, 2025.
