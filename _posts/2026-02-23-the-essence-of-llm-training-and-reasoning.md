---
layout: blog-post
title: LLM 推理与训练的本质：从 Surrogate 到强化学习的几何空间
date: 2026-02-23 20:00:00 +0800
series: LLM ESSENCE
categories: [基础模型]
tags: [Reasoning, Alignment, Reinforcement Learning]
author: Hyacehila
excerpt: 从Loss 只是 Surrogate的视角出发，回顾 Test-Time Compute (TTS) 如何控制泛化误差，并结合 CMU 的能力边缘理论揭示 RL 后训练的边界；微观剖析策略熵坍缩现象与 Meta 的 Three-Gate 理论，了解RL和SFT在微观的差异。
featured: true
math: true
---

> **本文融合近期多篇核心研究，旨在系统性地回应一个问题：当下的 LLM 后训练（如 RLHF / RLVR）到底是在创造全新的推理能力，还是仅仅解锁了预训练的固有封印？从宏观泛化到微观几何，希望这篇文章能帮你更好的理解RL**

# LLM 推理与训练的本质：从 Surrogate 到强化学习的几何空间

## 楔子：回归训练的代理本质

在探讨大模型推理 (Reasoning)之前，我们必须首先端正对模型训练的认知态度。

很多时候，我们在不断追求将模型的某个 Loss（损失函数）逼近到无穷小，仿佛只要收敛了就找到了真理。但事实并非如此。引用田渊栋的一句话：“很多 Loss Function 都是 surrogate（代理），比如预测下一个词 (Predict the next token)……它的核心目的是产生一个梯度流，让表征往正确的方向走，这是最重要的逻辑。”

**Loss Function 是且仅是 Surrogate**。如果真理真的存在且唯一，就不会有如此多花样百出的损失函数都能在各自的领域生效并起作用。

至于这个目标函数在数学上长什么样，其实并不重要。优化损失只是我们寻找大模型正确参数表示的手段，而非终极目的。如果我们不在乎模型究竟学到了怎样的数据结构，而仅仅盯着表面的 Loss，就极易陷入陷阱。最典型的例子就是 0-1 评测机制：如果我们只教导模型“答对 = 1 分、回答我不知道 (I don't know) = 0 分”，这种看似合理的 Surrogate 实际上就是一种鼓励投机的激励结构系统，自然催生幻觉模型。

因此，与其盯着 loss，不如从我们希望表征学习到的数据结构出发，去设计符合需求的训练信号；与其问 loss 有没有收敛，不如问这个 surrogate 是否奖励了你真正想要的行为（以及是否系统性惩罚了不确定性表达）。把 loss 当成 surrogate，会更自然地把注意力放回数据、约束与评测，他们才是训练的真正核心。

基于这层视角，我们可以用一个更包容的态度去重新审视 SFT (监督微调) 与 RL (强化学习)：**它们之间不存在本质的训练质量高下之别**。两者都是在调整庞大而超高维的参数空间，仅仅是因为使用的数据和提供的 Surrogate 引导方向不同，才让它们走向了不同的道路。训练方法的区别可能影响速度，但大概率不决定终点。**RL的有效，来自于RL的data，奖励这件事情本身不引入新的知识，但增强特定能力，这是一个有效的surrogate。**

顺着这个代理的思路，让我们把目光投向当前万众瞩目的推理 (Reasoning)。不管是 SFT 的数据，还是 RL 的奖励信号，这些 Surrogate 究竟是如何从宏观和微观两个层面，重塑模型的潜能的？


## 宏观视角的泛化与突破：从 TTS 到能力边缘

为了解析大语言模型的宏观泛化能力，我们绕不开思维链 (Chain-of-Thought, CoT)。

当前主流的大语言模型推理方法，无论是显式的 CoT 还是内隐的计算，都可以被宽泛地归类为一种 **Test-Time Compute (TTS，测试时扩展计算)**。它的理论源头其实可以追溯到很久之前——比如 Alex Graves (2016) 年提出的 Adaptive Computation Time (自适应计算时间) 以及 Ling et al. (2017) 提出通过生成分布步骤来解决问题的奠基性工作。

### CoT 为何有效？

如果沿着 [Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning](https://arxiv.org/abs/2502.04667v3) 在可控 two-hop compositional 任务上的建模去看，CoT 的价值并不神秘。作者先把测试分布写成 ID 与 OOD 的混合；在 ID 部分已经被充分拟合，也就是 $D_{\text{KL}}(P_{\text{test}}^{\text{ID}} \Vert P_{\text{train}}) \to 0$ 时，系统的期望泛化误差主要就会被 OOD 成分主导：

$$
\overline{\text{error}} \leq \sqrt{\frac{2R^2\alpha}{N} D_{\text{KL}}(P_{\text{test}}^{\text{OOD}}(Y \mid X) \Vert P_{\text{train}}(Y \mid X))}
$$

这里需要收紧一句：这不是所有 CoT 任务的普适定理，而是该文在受控组合设定下给出的建模与误差上界。它表达的核心直觉是：如果模型只是在学一个粗糙的 $X \to Y$ 映射，那么一旦测试题目换成新的组合模式，OOD 散度就会立刻成为瓶颈，使得总体的 KL 散度根本无法控制。

如何打破困局？这篇论文给出的思路，是显式引入中间推理步骤 $C_i$。

在引入 $C_i$ 后，原始的条件概率 $P(Y \mid X)$ 可以被分解为 $\sum_i P(Y \mid X, C_i) \cdot P(C_i \mid X)$，相应地，泛化误差的上界也被拆成了两个更细的部分：第一是从 $X$ 生成中间推理 $C$ 的散度，第二是给定中间推理 $C$ 走向结果 $Y$ 的期望散度。

$$
\overline{\text{error}}^2 \leq \frac{2R^2\alpha}{N} \left[ D_{\text{KL}}(P_{\text{test}}^{\text{OOD}}(C_i \mid X) \Vert P_{\text{train}}(C_i \mid X)) + \mathbb{E}_{C_i \sim P_{\text{test}}^{\text{OOD}}(C_i \mid X)} \left[ D_{\text{KL}}(P_{\text{test}}^{\text{OOD}}(Y \mid X, C_i) \Vert P_{\text{train}}(Y \mid X, C_i)) \right] \right]
$$

简言之：**CoT 通过显式生成的中间步骤 $C$，把一个原本难以直接外推的 OOD 组合问题，转化为若干个更接近 ID 分布、也更容易控制散度的局部子问题。** 如果把视角再往 latent reasoning 推一步，也可以把 latent CoT 当作一种启发式理解：中间步骤未必作为输出 token 出现，也可能被压进隐藏层表征里。

### RL 如何延伸能力边缘？

当我们知道了 TTS 可以拆解难度后，一个近年来备受争议的终极问题来了：我们投入海量算力使用强化学习（如 PPO 和 GRPO），到底是凭空让模型悟出了新的推理法则（提升了上限 pass@128），还是仅仅训练它学会了更麻利地找到其本来就懂的答案（仅提升一次测试通过率 pass@1）？即**RL 是否能扩展基础模型的推理能力**。

卡内基梅隆大学 (CMU) 的论文 [On the Interplay of Pre-Trai](https://arxiv.org/abs/2512.07783v1) 通过严谨的控制变量实验给出的回答是：**存在一组阶段博弈的动态边界**。

RL 所提供的奖励也是一种 Surrogate 且这种 Surrogate 的效果和模型与任务均相关。
*   **对于已掌握的 ID 任务**：无论加多少强化学习，提升的主要都是首发命中率 (pass@1)。此时的 RL 更多是在利用既有能力，甚至会催生熵坍缩；它并不真正扩充绝对能力边界 (即 pass@128 没有提升)。
*   **对于彻底未知的 OOD 任务**：预训练阶段如果完全没有暴露过相应的原子运算操作（哪怕只有 1% 的曝光度都会成为种子），单靠后训练的 RL 仍然无能为力，因为 RL 并没有注入新知识。
*   **边界的突破（Capability Edge）**：RL 这个强力的 Surrogate **唯有在能力边缘才会展现作用**。所谓能力边缘，指的是那些模型在 pass@1 上难以成功，但在多次实验的 pass@128 上偶尔能够成功的艰深任务。只有针对这些靶向任务设计训练信号，大规模强化学习才能实质性提升 pass@128，实现真正的外延泛化。

这个研究处理了之前研究上的矛盾：有的研究认为 RL 只会带来熵塌缩，但是有的研究认为 RL 扩展了能力边缘（如RLVR），其核心原因就在于训练难度谱的不同区域。对于基础模型已能解决的分布内任务，随着 pass@k 中 k 值增大，性能会趋于饱和，RL 很难再带来优势；而当 RL 瞄准基础模型表现不佳的真正分布外任务，且 RL 数据贴近当前模型的能力边缘时，就能观察到明显的外延泛化提升。

综上，我们建议**围绕模型的能力边缘设计 RL 数据**；即筛选 RL 数据集，聚焦模型在 pass@1 上失败但在 pass@k 上能成功的任务。这种策略可避免在高 pass@1 任务上的冗余训练，同时防止在零 pass@k 任务上出现奖励稀疏问题。该过程也可以迭代进行：定期重新评估能力边缘任务池，随着模型能力增强，原本的分布外任务会逐渐进入可解区间，重新评估能力边缘即可。

**这个 RL 思路已经逐渐成为当前训练的共识：使用分层的训练数据，而不是静态分布的数据，在清洗训练数据的时候就要把能力边缘考虑进去。**

**如果需要的话，还要把“问题是否可回答”纳入奖励设计，对拒绝回答进行奖励来抑制幻觉。**

不过，上面的结论还有一个很重要的前提：你使用的评测指标，真的在衡量“推理边界”，而不是在衡量“答案是否容易被多次猜中”。这在纯数学问答里尤其关键，因为最终答案往往短小、格式固定，多次采样时猜中的概率会被迅速放大。于是，数学任务上的 Pass@K 有时衡量的并不只是 reasoning boundary，也混入了 guessing boundary。

### 当 Pass@K 不再可靠：CoT-Pass@K、LLM-as-Rubric 与 CoT 奖励

MSRA 的 [Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs](https://arxiv.org/abs/2506.14245v2)  则从另一条线索补上了这个故事：它与 CMU 那篇关于 capability edge 的视角互补，讨论的不是CoT training 机制，而是 **answer-only RLVR 会不会隐式偏好更正确的 CoT**。作者的结论是：在数学推理和代码任务上，只使用结果奖励的强化学习也能改善模型能力，同时也暴露了纯数学场景里 Pass@K 的局限性。

作者重新审视了纯数学问答里的 Pass@K，指出基础模型在 AIME 这类答案空间有限的任务上，确实可能靠多次采样把正确答案“撞”出来；于是他们提出 `CoT-Pass@K`：只有最终答案正确，且整条 CoT 也基本正确时，这次采样才算真正成功。用这个指标再看，RLVR 在数学任务上的能力边界扩展就重新变得可见。相对地，在代码任务里，正确性由程序执行来验证，“猜中”的空间被大幅压缩，所以传统 Pass@K 仍然更接近真实的 reasoning boundary。

为了近似判断一条数学 CoT 是否站得住脚，这篇论文采用的是一个相当轻量的 LLM-as-rubric：用更强的数学推理模型去审阅完整解题过程，而不是只看最终答案；同时对同一条 CoT 做多次独立判分，再分别用 `any-correct`、`majority-correct` 和 `all-correct` 三种聚合方式降低单次误判带来的噪声。它当然不是完美裁判，但已经足以把“答案碰巧对了”和“推理本身是对的”区分开来。

这里还需要区分三件不同的事：第一，只奖励 `final answer` 的 RLVR；第二，对整条 CoT 给一个单一 reward；第三，对每一个 token / step 分别打分的过程奖励模型 (PRM)。微软这篇论文的主实验真正证明的是第一件事：**哪怕只看答案，answer-only RLVR 也会在统计上隐式偏好更正确的 CoT**。而这篇博客由此进一步推出的，是第二件事的潜力：这不是传统意义上的 token 级过程奖励，而仍然是一个单一 reward，只是把判分对象从最终答案扩展到了整条 CoT 的正确性。在纯数学问答里，这种 CoT 级的单一奖励，比只奖励最终答案更贴近我们真正想优化的行为。

这也修正了一个常见误解：这里真正被支持的，并不是结果奖励已经等价于显式过程监督，而是在模型已有 logic prior 的前提下，只看答案的 RLVR 已经会隐式改善 CoT 质量。而如果我们进一步把 CoT 正确性本身直接纳入 reward 设计，那才是在此基础上更进一步的 surrogate 设计，用更贴近推理质量的信号去压制捷径、猜测与 reward hacking。而分 step 的奖励或者 token 级的奖励，则是抑制捷径、猜测与 reward hacking 的更进一步方法。

### 预训练、中期训练与后训练阶段之间的相互作用

既然 RL 的核心在于能力边缘，那么大模型又是如何走到这个边界的？这其实离不开预训练（Pre-Training）、中期训练（Mid-Training）与后训练（Post-Training）三个阶段的精密互接。

**预训练：无暴露则无泛化**
虽然 RL 极具边界扩展能力，但它本身并不给模型灌输全新的知识，而是在现有知识范围内对能力边缘进行外推。研究表明，**若预训练阶段针对某个概念或长尾语境完全是 0 暴露，无论后续投入多少探索，RL 都将无能为力，无法诱导向该未知语境的迁移**。然而，只要预训练里哪怕只有极少量相关暴露（如最基础的原子运算片段），就能为后续 RL 提供至关重要的能力种子。依靠这颗种子，基础模型便获得了必要的初步推理原语，后续 RL 才能将其精准放大，实现稳健的跨语境泛化。因此，预训练的核心不在于教会模型严密推演，而在于尽可能大地铺设认知与基础概念的覆盖。

**中期训练：连接原语与能力边缘**
在横跨预训练的海量数据和 RL 的精准打击之间，当前行业逐渐意识到引入中期训练阶段（Mid-Training）是一根极具影响力的隐形杠杆。面向推理的中期训练主要目的是打底：它能利用适度的资源和衔接分布，大幅增强模型对后续 RL 的适配能力，巩固稳定相关的基础原语。在这个阶段，数据结构就已经开始瞄准通往能力边缘的方向。**对于 Mid-training，一般是精选高质量、强针对性的领域数据，进行 next-token prediction 的持续预训练，为后续的后训练铺路。**

这也牵涉到一个基于约束的硬核算力博弈——在固定的计算预算下，应该如何分配中期训练与后训练 (RL) 的资源比重？
*   **保证核心能力**：若当前的交付首要任务是保障分布内 (ID) 任务的高分表现，计算预算就应当向中期训练大举倾斜，后续只需搭配较轻量级的 RL 去强化既定认知即可。
*   **扩展外部能力**：若意图攀登更困难的分布外 (OOD) 推理泛化高峰，则只需留给中期训练“刚够建立必要先验”的压缩预算，而后将绝大部分算力倾注于重度 RL 探索之中。

### 后训练的 SFT 与 RL 改进同一个目标

RLVR 则进一步验证了 RL 和 SFT 在 surrogate 视角上的同质性。它给出的口语化直觉其实很朴素：如果预训练已经让模型具备基本的 knowledge / logic prior，那么**正确 CoT 比错误 CoT 更容易导向正确答案**。于是即便 RLVR 只看最终答案，梯度在期望上也会更常强化那些更干净的推理路径。换句话说，answer-only RLVR 不只是把已有答案采样得更快，它还会逐步把 rollout 里的 CoT 洗得更像可学习的数据。

这件事的实践意义非常直接：这些被 RLVR 改善过并且经过验证后有效的 CoT，不应该只停留在 on-policy rollout 里，它们还可以被回收成新的 SFT 数据。论文里就观察到，用这些 CoT 再做 SFT，能够在相当程度上复现后训练模型的表现。这可以被看作一个更便宜的离线/离策略学习闭环：`Pre-Training` 提供原语，`Mid-Training` 组织原语，`RLVR` 在能力边缘筛出更好的推理轨迹，而 `SFT` 再把这些高质量 CoT 沉淀回模型。

**综上所述，RL 想要发挥其不可取代的作用，既要保证目标任务没有被完全死记硬背以至于仍留有探索弹性，也要保证训练数据贴合当前模型的能力边缘；而从更长的流水线看，SFT 还可以继续回收 RLVR 产生的高质量 reasoning data。**

这也更接近今天顶尖模型背后的流水线分工：预训练尽可能提供暴露；**中期训练 (Mid-Training)** 承上启下，搭建通用的认知桥梁并坚固特定推理原语；**后训练的 SFT** 提供更好的冷启动效果；**RLVR** 在能力边缘上筛出更好的推理；最后再将模型的能力 distill 给下一阶段的训练。

到这里，宏观部分回答的主要是奖励该瞄准什么题；接下来的问题则是：哪怕只有一个看似粗糙的 reward，它又是如何穿过 token 分叉、策略熵与参数几何，真的把这些偏好写进模型里的？

## 微观视角的动力学：策略熵，令牌分叉与三门理论

宏观部分讨论的是数据与任务边界，微观部分则要分两步看：先看 CoT 训练如何把推理步骤写进模型内部，再看 RL 如何通过熵管理、token 分叉与参数几何真正改变模型行为。

### CoT 如何被内部化为两阶段电路

先看 [Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning
](https://arxiv.org/abs/2502.04667v3) 自己最独特的结构分析。作者用 logit lens 与 causal tracing 观察到一种 **two-stage compositional circuit**：在显式 CoT 训练下，模型会先在较浅层解出桥接性的中间结果，再把这个中间结果交给后续层去完成下一跳推理。

这件事的重要性不只在于“可解释”。与不带 CoT 的训练相比，中间结果能够在更浅层被读出，意味着更深层还保留着处理后续步骤的容量。因此，论文里那句很有传播力的总结——CoT 更像是在教模型 *how to think*，而不只是 *what to think*——放在这里是成立的，但语境需要收紧：它说的是显式步骤训练下的组合式 reasoning，而不是所有推理任务的无条件定律。

作者也把这套观察和有限的现实实验连了起来：在 MetaMathQA 上做 CoT SFT，再到 GSM8K 上验证时，带 CoT 的训练显著优于 answer-only SFT；而当推理步骤里掺入一定噪声时，模型依旧保留了可观的泛化能力。所以回答的了 CoT 的内部化机制与有限噪声鲁棒性。

### 策略熵的双刃剑效应与经验定律

如果把视角切到 RL for reasoning 的阶段，信息论里的熵就开始变得关键。[The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.22617v1)  讨论的正是这件事：当模型在一个高预期回报的动作上首次尝到甜头时，它会迅速提高这个选择的概率，并同步压低其他可能性的概率。当这个正反馈循环运转过快，我们就会遇到 **策略熵崩溃 (Policy Entropy Collapse)**，也就是模型越来越自信，却越来越不愿探索。

策略熵（平均 token 级熵）用于描述模型输出时 token 概率分布的不确定性。定义为：
$$
H(\pi_\theta, D) = -\mathbb{E}_{D, \pi_\theta}[\log \pi_\theta(y_t \mid y_{<t})].
$$

**直观地理解：当一个高概率动作获得高奖励时，模型会进一步强化这个选择，从而降低整体熵；反之，当一个低概率动作意外获得高奖励时，模型会抬升其概率，从而增加整体熵。**

这篇论文从理论和实证两端都说明，策略熵的变化与 advantage 驱动的 logit 更新密切相关；经验上，作者还观察到一个近似关系：$R = -a \exp(H) + b$。它指向一个很现实的结论：不少 RL 性能提升，本质上是在消耗探索熵。如果缺少额外干预——例如文中讨论的 KL-Cov 选择性约束，或更传统的熵正则化——模型就很容易沉迷于局部高分捷径，最终出现 reward hacking 式的短视行为。

### 高熵少数 Token 与令牌分叉

那么，在生成长长的 CoT 时，每一次 token 预测都有相同的战略重量吗？显然不是。

一篇近期工作 [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2506.01939v2) 把这个问题压到了 token 级别。作者发现，在 CoT rollout 中，熵的分布是高度不均匀的：

*   **极少数“高熵” Token**：它们通常位于逻辑路线的分叉口、假设反转点或关键运算节点。这里每一步选择都可能把模型带向完全不同的推理分支。
*   **绝大多数“低熵” Token**：它们更多承担铺垫句法、补全连接词或延续既有表述的职责，在这些位置上模型往往已经非常笃定。

也正因为如此，对所有 token 一刀切地回传梯度很可能是资源错配。这篇论文的核心发现甚至更进一步：如果把策略梯度主要集中在这少数高熵 token 上，效果往往可以逼近甚至超过全量 token 更新。它支持的是“RLVR 的有效性主要来自关键分叉 token”，而不是 CoT training 本身的理论机制。

### Three-Gate Theory：RLVR 走向稀疏更新

前一节谈的是 token-level / policy-level 的熵与分叉；这一节转向另一类问题：参数几何如何把 RL 更新筛成稀疏、局部且偏向非主方向的改动。

当我们审视高成本、高收益的特定强化学习（如 Meta AI 论文重点探讨的基于程序验证奖励的 RLVR）时，不可避免地会发现一个极度反直觉的现象：**这种昂贵且强大的 RL 方法引发的参数变化居然比简单的 SFT 还要稀疏并且集中于特定区域。这种只针对极少数神经元矩阵进行的微调，为何能撬动整个系统的推理涌现？**

Meta 的研究论文 *"The Path Not Taken: RLVR Provably Learns Off the Principals"*，用极为优雅的 **三门理论 (Three-Gate Theory)** 一定程度上回答了这个问题。

1.  **Gate I（策略约束门，KL Anchor）**：在每一轮 on-policy 更新的端口，KL 锚点强行对更新步伐做了限制。它确保每一次尝试都在预训练既有模型的熟悉辐射带内。
2.  **Gate II（模型几何门，Model Geometry）**：这是整个理论的精髓，大语言模型在预训练时，参数矩阵其实已经被雕刻成了一个无比巨大的“高维Landscape”。在这个壮阔的空间里，存在那些主方向（即曲率最大、模型原有知识权重最稠密的通道）。当 RLVR 的梯度开始传播着修改参数的时候，大模型的底层几何结构迫使或者说主动引导这些更新梯度避开已夯实的主骨干方向（Off the principal directions），而是去修稿那些曲率极低、不破坏既定光谱特征的特殊子空间边缘。这不仅解释了为何特定模型对特定微调路径有本能的选择倾向，更说明了这种偏见源于预训练造就的地貌，而非某个数据集。
3.  **Gate III（精度掩码门，Precision）**：最终，由于受到低精度参数存储的过滤限制，那些原本就没有获得几何偏好的极其微小的改动，在此阶段被抹去。这就导致最终从观测层面，RL 的改动呈现出**局部与稀疏**。

稀疏性不是结论，是由于参数精度有限以及参数更新选择性共同产生的。那些没有被选中的更新区域的参数变动被精度不足隐藏了。

也正因为这种动力机制的存在，它警醒着当下的开源研发者：在走向后训练时代时，直接照搬 SFT 时代的 PEFT (参数高效微调，比如只依赖于低秩分解的标准 LoRA) 是相当危险的。传统的低秩主权重更新路径可能与这种偏向非主曲率方向、高熵边缘、微小调整的需求背道而驰。新的参数效能技术方向应当考虑对这部分流形的针对性保护和冻干处理。

## 结尾：不同的代理，不同的数据，不同的目标

所以，大模型的推理并非魔法，也无神秘可言。我们使用预训练、中期训练、SFT 与 RLVR 这些不同阶段的数据和损失代理，去不断塑形模型的能力。**模型能力改进的本质，仍然是一场关于设计更优 Surrogate、让表征目标更好被学习的工程。**

但大模型的终点不该只有推理。无论是工具使用、角色扮演、记忆模块还是规划能力，只要有了更好的 surrogate 坐标，那些潜藏在参数中的复杂流形都可以被逐步发现，而不一定非要等待更玄妙的算法。算法很多时候决定的，更多只是我们发现这些潜藏能力的速度。

## 参考资料

- [Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning](https://arxiv.org/abs/2502.04667v3)
- [On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models](https://arxiv.org/abs/2512.07783v1)
- [Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs](https://arxiv.org/abs/2506.14245v2)
- [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.22617v1)
- [Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2506.01939v2)
- [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/abs/2210.03350v3)
- [Do Large Language Models Latently Perform Multi-Hop Reasoning?](https://arxiv.org/abs/2402.16837v2)
- [Hopping Too Late: Exploring the Limitations of Large Language Models on Multi-Hop Queries](https://arxiv.org/abs/2406.12775v2)
- [From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step](https://arxiv.org/abs/2405.14838v1)


