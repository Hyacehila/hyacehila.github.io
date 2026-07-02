---
title: LLM 对齐中的强化学习：从奖励信号到优势估计
title_en: "Reinforcement Learning in LLM Alignment: From Reward Signals to Advantage Estimation"
date: 2026-03-16 17:30:00 +0800
categories: ["AI & Agents", "Training & Alignment"]
tags: ["Reinforcement Learning", "Alignment", "Reward Modeling", "Advantage Estimation", "Training Dynamics"]
author: Hyacehila
excerpt: 从 reward、baseline、advantage 与 normalization 这条信号链出发，解释为什么 LLM 对齐中的 RL 算法总在重写奖励信号。
excerpt_en: "Following the signal chain from reward to baseline, advantage, and normalization, this post explains why RL algorithms for LLM alignment keep rewriting reward signals."
mathjax: true
permalink: '/blog/2026/03/16/rl-alignment-from-reward-to-advantage/'
---

过去很长一段时间里，我们谈论 LLM 对齐中的强化学习，常常是按算法名字来记忆的：PPO、DPO、RLOO、GRPO、REINFORCE++。这么记当然方便，但也很容易失去对这个领域的全面认识，好像这个领域的进展主要就是不断换缩写、换 loss、换训练配方，涨点（也不一定真涨了），pub，然后下一个。各种策略优化（PO）层出不穷，但是好像外行不了解他们在折腾什么。

换个角度看会更清楚。**LLM 对齐里持续变化的，是奖励信号进入优化器的方式。** 一条原始 reward，到底怎样经过约束、分配、减基线、归一化，最后变成可以稳定更新策略的 advantage，才是这些方法的主要分歧。

抓住这条主线后，原本彼此割裂的方法就能连起来：

- PPO 提供早期 LLM RLHF 中影响很大的 reward-to-advantage 工程模板；
- DPO 绕开在线 RL，把偏好对直接写进优化目标；
- GRPO 和 REINFORCE++ 重新追问：**没有 critic 之后，baseline 和 normalization 应该从哪里来。**

这里讨论的是 reward signal 如何被优化器消费的概念框架；如果你关心 TRL `GRPOTrainer` 的代码级原理和训练接口，可以对照 [Re0-04：TRL GRPOTrainer（原理篇）](/blog/2025/12/30/Re0HF-04/)。

## 引子：为什么 LLM 里的 RL 难点不是有没有 reward，而是 reward 能不能成为可靠训练信号

传统强化学习里的 reward，至少在很多经典环境中，是相对明确的：游戏得分、到达目标、是否碰撞、是否成功完成任务。可在 LLM 对齐里，reward 通常不是这样。

首先，它往往是**稀疏**的。许多场景里，模型生成完整段回答之后，奖励模型才给一个总分，或者验证器才告诉你答案对不对。[Learning to Summarize from Human Feedback](https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf) 与 [InstructGPT](https://arxiv.org/abs/2203.02155) 这类经典 RLHF 流程，很大程度上都在处理这种终局反馈。

其次，它往往是**延迟**的。一个回答前面几十个 token 的价值，很可能只有在整段答案写完之后才能被判断出来。策略梯度的难点之一，就是要把这个迟到的信号再分配回前面的生成过程。[Williams 1992](https://doi.org/10.1007/BF00992696) / [Policy Gradient Theorem](https://proceedings.neurips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation)

第三，它常常是**带噪声的**。奖励模型并不是真理判官，它只是一个 surrogate。它可以学习到人类偏好的某些模式，也可以学偏。OpenAI 在总结任务上很早就观察到，如果让策略过度优化 reward model，最终得到的结果会偏离真实人类偏好。

第四，它还可能被**利用**。对开放式任务来说，模型并不一定真的变得更有帮助，它也可能只是更擅长顺着奖励模型拿分。这也是为什么不少 RLHF 文章都要引入 KL 约束、长度控制、参考模型等机制：不是因为这些东西看起来更优雅，而是因为没有它们，reward 太容易被 hack。

所以，LLM 里的 RL 难点从来不只是有没有 reward，而是：

**这个 reward 能不能被加工成一个低方差、不过拟合、不轻易被 exploit、还能跨 prompt 保持尺度稳定的训练信号。**

## TLDR

先把我想放在最前面的判断写出来。

1. **LLM-RL 的演化，是在不断重写 `reward -> advantage` 这条链。** 变化的不是算法名，而是谁更擅长把原始反馈变成稳定的优势估计。[PPO](https://arxiv.org/abs/1707.06347) / [Back to Basics](https://aclanthology.org/2024.acl-long.662/) / [REINFORCE++](https://arxiv.org/abs/2501.03262)
2. **经典 RLHF 里的 reward，从一开始就不是裸分数，而是“任务奖励 + 行为约束”的复合信号。** [Learning to Summarize from Human Feedback](https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf) / [InstructGPT](https://arxiv.org/abs/2203.02155)
3. **critic-free 方法并没有消灭 advantage estimation 的问题，只是把它从 critic 网络转移到了 reward 处理与统计构造上。** 你不再训练 value head，但仍然要回答：baseline 从哪里来、尺度怎么定、局部噪声怎么控。[Back to Basics](https://aclanthology.org/2024.acl-long.662/) / [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
4. **GRPO 的关键创新，不只是不用 critic，还包括把学习信号改写成组内相对优势。** 这在推理任务上很有效，但也把问题暴露了出来：局部标准差太小怎么办？组内更好是否等于全局更好？[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
5. **REINFORCE++ 值得单独重视，因为它把尺度问题推到了全局归一化上。** 它争论的不再只是 baseline，而是：advantage 的尺度究竟应该在 prompt 内定义，还是在整个 batch 上定义。[REINFORCE++](https://arxiv.org/abs/2501.03262)
6. **如果你以后再读新的 RL 对齐论文，最值得先问的不是它叫什么 PO，而是五个问题：reward 从哪里来、KL 放哪里、baseline 从哪里来、normalization 在哪里做、最后优化的到底是什么。** 这比背算法缩写更有用。

## 先把概念拆开：reward、return、baseline、advantage、KL、normalization

如果不先把这几个词拆开，后面几乎一定会看混。

### reward：原始反馈，不一定已经适合训练

reward 是任务最原始的反馈来源。在 LLM 对齐里，它可能来自：

- 奖励模型的分数；
- 规则验证器的对错判定；
- 过程奖励模型的步骤分；
- LLM as Judge 的评价；
- 外部环境返回的成功/失败信号。

但从 RLHF 的历史看，**reward 通常不是直接拿来就用的。** 它还要和 KL 约束、长度惩罚、停止惩罚等机制组合，才会成为进入优化的信号。虽然 reward 本身的质量很关键，但要让 reward 可用于训练，RL 算法层面的信号处理同样要投入大量工作。

### return：把未来 reward 累起来之后的回报

策略梯度不只关心眼前一步的 reward，而是关心当前动作会不会把后面的奖励也带起来。最朴素的累计回报写法是：

$$
G_t = \sum_{t' = t}^{T} \gamma^{t' - t} r_{t'}
$$

如果在 LLM 任务里只有终局奖励，这个式子就意味着：**虽然 reward 是在最后给的，但前面的 token 仍然能通过累计回报拿到训练信号。**

return 的思路很简单，但它是把稀疏奖励还原为 token 级信号、用于优化的基础。有了 return 之后，下一步就是怎么用它更新参数——最朴素的做法就是 REINFORCE（后面会展开）：用 $G_t$ 乘以 log 概率的梯度，高回报的 token 提升概率，低回报的压低概率。但这种粗糙的分配也导致了信号本身质量不足（整条轨迹的随机性都叠在里面），方差极大，后面的每一步——baseline、advantage、normalization——都在弥补这个粗糙信号。

### baseline：不是额外奖励规则，而是降方差参考系

不少初学者会把 baseline 理解成另一个奖励函数，但更准确地说，它更像一个统计参考系。你不是在问这次得了多少分，而是在问这次比通常水平高多少。

[Greensmith et al. 2004](https://www.jmlr.org/papers/volume5/greensmith04a/greensmith04a.pdf) 清楚地说明了这一点：baseline 的主要作用是做 control variate，也就是降方差，而不是改变梯度期望。单纯的奖励大小意义有限，进入更新的是 advantage，而 baseline 是它的基础。

### advantage：真正进入更新的，通常不是 reward 本身

一旦引入 baseline，训练器真正关心的就不再是裸回报，而是优势：

$$
A_t = G_t - b_t
$$

**在引入 baseline / advantage 的策略梯度方法里，更新信号通常不是裸分数，而是相对参考水平的优势。**

在 PPO 时代，advantage 最常用的估计方法是 **GAE（Generalized Advantage Estimation）**。它的思路是：只看一步 TD error 方差低但严重依赖 critic 准不准（bias 高），看到底的 Monte Carlo 无 bias 但方差大，GAE 通过一个参数 $\lambda$ 在两者之间插值，兼顾两端。

### KL：它不是附件，而是行为边界

RLHF 里另一个特别容易被说轻的对象是 KL。不少人会把它理解成一个顺手加上的 regularizer，但从早期 RLHF 工作来看，KL 的角色更接近行为边界：

- reward model 告诉你想朝哪里走；
- KL 告诉你不能偏离 reference model 太远。

没有这层边界，reward model 很容易被过度优化。在 LLM 场景中，它可能表现为行为漂移、偏好退化、格式或长度投机、能力退化，或者更典型的 reward hacking。若从优化视角理解，可以把它看作策略逐渐远离 reference model 所代表的高质量行为区域；但这不是唯一机制，也不必把所有问题都归因到灾难性遗忘或某个 loss basin。

这里需要盯住的不是 KL 这个超参数本身，而是**优化压力**。同一个 reward signal，在轻量 reranking 或短轮数偏好优化里可能还算可靠；一旦进入长时间 online RL、多候选采样、低 KL 或高权重 reward shaping，policy 就会更频繁地探测这个信号的边界。DPO、rejection sampling、online RL 对 reward 的利用方式并不相同，但它们都可能增加对代理信号缺陷的选择压力；真正高风险的是强在线优化持续把一个有盲区的 signal 当成目标推进。

所以 reward hacking 的临界点通常不是“reward 写错了”这么简单，而是：**policy 的探索能力已经超过 reward signal 的判别和泛化能力，但优化器仍然把这个 signal 当成目标继续推进。** 这里先作为 KL / 过优化风险的伏笔；更系统的拆解放到第三篇。训练监控上，KL、reward 分布、人评或验证指标，以及长度等行为分布要一起看；只看平均 reward，很容易把过优化误判成能力提升。

### normalization：它决定模型是在和谁比较自己

normalization 看起来像一个工程细节，但它决定了比较尺度。因为它决定了高于平均和低于平均是在哪个尺度上定义的：

- 是对同一 prompt 的几个样本来说高于平均？
- 还是对整个 batch 的所有样本来说高于平均？

这个差别，正是 GRPO 与 REINFORCE++ 分歧的中心。

下面这张总地图，是我觉得最值得先建立的。

| 环节 | 它在回答什么问题 | 常见对象 | 代表做法 | 常见失败模式 |
| --- | --- | --- | --- | --- |
| 原始 reward | 什么样的回答算更好 | RM、verifier、环境反馈 | 奖励模型打分、规则验证、过程奖励 | 奖励噪声、错标、不可泛化 |
| 约束项 | 能离参考策略多远 | KL、长度约束、停止惩罚 | KL penalty、长度/截断惩罚 | reward hacking、风格漂移 |
| credit assignment | 终局信号怎么回传给前面 token | return、token reward | 累计回报、末端 reward 回灌 | 稀疏、延迟、长程噪声 |
| baseline / advantage | 这次比基准水平高多少 | critic、group mean、leave-one-out | GAE、leave-one-out、group mean | 高方差、基线失真 |
| normalization | 优势尺度在哪定义 | local group、global batch | group std、batch std、z-score | 局部爆炸、跨 prompt 不可比 |
| policy update | 怎样稳住每次参数更新 | actor / old policy / reference policy | PPO clip、KL loss、critic-free update | 策略抖动、训练不稳定 |

如果把这几层分开，后面很多争论就会清楚很多。说白了，很多 paper 表面上在改 reward，真正动手最多的其实是第四行和第五行，这种内容应该属于算法本身。

用一条线把整条信号链串起来：

```
raw reward → +KL约束 → return(分配到token) → -baseline → advantage → normalize → ×∇log π → 更新参数
```

后面每个算法的分歧，都可以对照这条线，看它在哪一步做了不同的选择。

## 第一代答案：PPO 式 RLHF 如何把 reward model、critic 和 KL 拼成可训练系统

[PPO](https://arxiv.org/abs/1707.06347) 原本不是为 LLM 设计的，但它给了 RLHF 一个特别重要的工程模板：

- 用 critic 来估计 advantage；
- 用 clipping 控制策略更新不要过猛——具体来说，PPO 通常用 $\epsilon \approx 0.2$ 裁剪新旧策略概率比，移除继续朝过大有利更新方向移动的激励，防止单次更新步子太大；
- 用 old policy / reference policy 给训练提供稳定参照。

到了 [Learning to Summarize from Human Feedback](https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf) 与 [InstructGPT](https://arxiv.org/abs/2203.02155) 这条 RLHF 路线里，reward model 与 PPO 开始真正融合。一个足够好用的心智模型是：

$$
R(x, y) = r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}
$$

它不必被理解成所有论文都逐字逐句使用的统一公式，但它准确抓住了经典 RLHF 的精神：

- 第一部分 $r_\phi(x, y)$ 表示奖励模型或任务反馈；
- 第二部分表示偏离参考策略所付出的代价。

这一步非常关键，因为它说明**经典 RLHF 从一开始就在做复合 reward 设计**。reward model 不是全部，KL 也不是附件，它们共同构成了真正的优化目标。

critic 在这里的角色也很重要。它不是在告诉模型什么是真理，而是在尽可能稳定地估计 advantage，让训练不要因为奖励噪声太大而发散。这就是 PPO 式 RLHF 的第一代答案：

**用 reward model 提供方向，用 KL 提供边界，用 critic 提供低方差的优势估计，再用 PPO 的更新机制把整个系统稳住。**

这套方案很强，但也确实不轻：

- 结构重——在 LLM 场景下，critic 通常和 actor 共享 backbone 加一个 value head，再加上 reference policy 和 reward model，训练时需要同时维护四个大模型规模的网络，显存和计算量几乎翻倍；
- critic 不好训时，会成为新的误差源；
- 训练配方相当复杂（学习率、GAE 参数、clipping 范围、KL 系数都要调）。

也正因为这些问题，后面 critic-free 的路线才会重新兴起。

## 旁支：DPO 为什么不是这篇文章的主线

[DPO](https://arxiv.org/abs/2305.18290) 发现了一个关键关系：在 KL 约束下，最优策略与隐式 reward 之间存在闭式联系，于是可以直接在偏好对上做分类式优化，绕开在线 rollout、critic 和优势估计。它让很多人第一次意识到**对齐不一定非得通过在线 RL 来做**。但正因为它绕开了 `reward -> advantage` 这条链，而不是继续重写它，所以在本文中只作为对照项。如果你关心的是在线 RL 本身该怎样处理奖励信号，主线仍是 PPO、GRPO 和 REINFORCE++。

## critic-free 主线：从 REINFORCE 原型到 GRPO 的组内相对优势

一旦把 critic 去掉，问题并不会消失，只是重新摆到了台面上。新的问题变成：

- 没有 value head 了，baseline 从哪里来？
- 没有 GAE 了，advantage 怎么估？
- 没有 critic 帮你平滑尺度了，normalization 靠什么做？

### REINFORCE：最小原型，直接把结果回传给轨迹

[Williams 1992](https://doi.org/10.1007/BF00992696) 的 REINFORCE 是这一切的起点。它最朴素的更新直觉是：高回报轨迹提升 logprob，低回报轨迹压低 logprob。

$$
\nabla_\theta J(\theta) \approx \sum_t G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

在 LLM 上，这可以被直观地理解为：如果一整段回答最后得分很高，那么组成这段回答的 token 选择，在相似上下文中就应该更容易再次被采样出来。

REINFORCE 的价值，不在于它今天还能不能直接拿来训大模型，而在于它把问题说明白了：**策略学习就是把结果反向归因给整条采样轨迹。**

但它也立即暴露出最老的问题：方差太大，分数本身就有噪声，分配给 token 以后会引入更多噪声；如果不减掉 baseline，而是让绝对得分直接影响输出概率，训练通常很难稳定。朴素无 baseline 的 REINFORCE 很少直接用于现代 LLM 对齐，但 REINFORCE-style critic-free 方法重新活跃，正是在这个思想原型上继续改造。

### GRPO：把局部相对化推到组均值 + 组标准差

[Back to Basics](https://aclanthology.org/2024.acl-long.662/) 这类工作把 REINFORCE 风格方法重新放回 LLM RLHF 的讨论中心，也提醒我们：critic-free 不是不要 baseline，而是把 baseline 从 value head 转移到采样统计上。比如同题多样本的 leave-one-out baseline，会用同一个 prompt 下其他回答的平均分作为参考系。GRPO 则把这条路继续推进：它不只减去组均值，还会除以组标准差。

[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) 的典型写法是，对同一个 prompt 采样 $k$ 个回答，拿到一组奖励 $r_{1:k}$，然后构造：

$$
A_i = \frac{r_i - \mathrm{mean}(r_{1:k})}{\mathrm{std}(r_{1:k}) + \epsilon}
$$

这背后的想法很直观：**对同一道题来说，值得学习的不是绝对得分，而是相对组内水平的优势。** 如果把它拆开看，GRPO 同时完成了三件事：

- group mean 是 baseline，回答不再和零分或全局平均比较，而是和同题候选的平均水平比较；
- group std 是 local normalization，优势尺度由这个 prompt 自己的一组候选决定；
- 同题多采样定义了比较空间，训练信号首先回答“这几个候选里谁更值得提高概率”。

这就是 critic-free 阶段最关键的转向：**少训练一套 critic，并不意味着 advantage estimation 消失了，而是它被压进了 rollout 组的统计结构里。** 你不再需要 value head，也减少了一套可能训偏的估计器；代价是更依赖同题多采样的质量、更依赖 verifier / reward model 能在组内拉开差异，也更依赖组内统计本身足够稳定。

这在数学推理这类同题多采样、同题排序明显的任务上很有效。也正因为如此，GRPO 才会在 reasoning 场景里迅速成为有影响力的代表方法。它特别适合那类可以为同一道题生成多个候选、再用规则验证器或奖励模型判断相对好坏的训练设置。

但如果从 reward-signal 的视角看，GRPO 也把局部相对化的问题暴露得很清楚：

1. 组太小时，均值和标准差本身不稳，advantage 会强依赖这一次采样到的候选。
2. 如果组内奖励几乎没有差异，group std 会让信号对 $\epsilon$ / clipping 非常敏感；在某些设置里接近零信号，在另一些设置里也可能放大噪声。
3. 在这个 prompt 的候选里更好，不一定等于“在整个数据分布上更好”。

我更愿意把这些问题理解成 GRPO 的历史价值：它不是把问题解决完了，而是把 **local normalization** 这件事推到了舞台中央。后来的 REINFORCE++，就是沿着这条线继续追问下去。

## REINFORCE++：从局部归一化争论到全局尺度

如果说 GRPO 的关键在于局部相对化，那么 [REINFORCE++](https://arxiv.org/abs/2501.03262) 的关键就在于：它认为**不稳定的不只是 baseline 本身，还有局部归一化的尺度。**

在作者的叙述里，像 GRPO 这样的 prompt-level normalization 至少有三个风险：局部组很小，均值和标准差都不稳；分子和分母来自同一小组，统计上彼此耦合；模型学到的更像是在组内赢过自己，不一定形成跨 prompt 的全局尺度。

于是 REINFORCE++ 给出的回答不是换一个 reward model，而是把 normalization 的范围从 local group 拉到 global batch：

$$
A^{\text{norm}} = \frac{A - \mathrm{mean}_{\text{batch}}(A)}{\mathrm{std}_{\text{batch}}(A) + \epsilon}
$$

这一步看起来只是把 std 换个地方算，但它改写了训练信号的比较对象：

- 在 GRPO 里，模型主要是在跟同一道题的其他候选比较；
- 在 REINFORCE++ 里，模型是在跟整个 batch 的优势分布比较。

论文里的普通 REINFORCE++ 和 REINFORCE++-Baseline 可以放在这个框架里理解：前者更强调即使没有同题多样本 baseline，也可以在 batch 尺度上归一化 advantage；后者保留组均值减法，让同题比较继续提供 baseline，但把 std 从局部组移到 batch 级别，避免用同一个小组同时决定“比谁高”和“高多少”。

这里更值得记住的不是术语，而是三个决策维度：

- baseline 从哪里来；
- normalization 在哪里做；
- KL 到底是混进 advantage，还是单独成 loss。

这也正是 REINFORCE++ 在这条主线里的价值：它不是新增一个复杂分支，而是把 critic-free 的核心争论从“baseline 从哪里来”继续推进到“advantage 的尺度在哪里定义”。

## 端到端走一遍：一个 prompt 的信号是怎样从 reward 变成 advantage 的

前面拆了不少概念，这里用一个具体的数值例子把整条链串一遍。

假设有一个 prompt："用一句话解释量子纠缠"，我们对它采样了 3 个回答，奖励模型分别打分：

| 回答 | RM 分数 $r_i$ |
| --- | --- |
| 回答 A | 8.2 |
| 回答 B | 6.5 |
| 回答 C | 9.1 |

**Return/Return**：在传统 RL 直觉里，如果只有终局 reward，可以把 $G_t = \gamma^{T-t} \cdot r_T$ 分配给不同时间步。但在 LLM RLHF / GRPO 的常见实现里，更常见的是把 sequence-level reward 转成同一条 completion 的 advantage，并广播到参与更新的 completion tokens，同时再叠加 token-level KL 或 ratio 约束。这里保留 return 概念，是为了说明“结果如何反向归因给轨迹”，不是说所有 LLM 实现都会按折扣终局 reward 逐 token 分配。

**Baseline（组内均值 / leave-one-out 风格）**：看回答 A 时，一种同题参考系可以取其余回答的均值，baseline = $(6.5 + 9.1) / 2 = 7.8$。如果按 GRPO 的组均值写法，则直接使用三条回答的平均值 $7.93$ 作为这个 prompt 的 baseline。两者细节不同，但共同点是：baseline 不再来自 critic，而是来自同题采样统计。

**Advantage**：如果用 leave-one-out 写法，回答 A 的 $A = 8.2 - 7.8 = +0.4$（比同题参考水平稍好，小幅提升概率）；回答 B 的 $A = 6.5 - 8.65 = -2.15$（明显不如同题其他样本，压低概率）。如果用 GRPO 写法，则先统一减去组均值，再进入组内标准差归一化。

**Normalization（GRPO 风格 vs REINFORCE++ 风格）**：

- GRPO：用这 3 个回答自己的均值和标准差做 z-score，$A_i^{\text{norm}} = (r_i - 7.93) / 1.07$。这是 prompt 内局部尺度，组只有 3 个样本时 std 不太稳。
- REINFORCE++：把这 3 个回答的 advantage 和整个 batch（比如 256 个 prompt × 3 = 768 个回答）的 advantage 一起算均值和标准差，尺度更稳定。

**Policy Update**：最后，REINFORCE 式更新用 normalized advantage 乘以 $\nabla_\theta \log \pi_\theta$——回答 A 的 token 概率被小幅提升（advantage 为正），回答 B 的 token 概率被明显压低（advantage 为负），回答 C 获得最大的正向更新。

这就是一条完整的 `reward → return → baseline → advantage → normalization → policy update` 链。后面所有算法的分歧，都可以回到这个例子里，看它们在哪一步做了不同的选择。

## 读论文检查清单：先问哪五个接口

如果整篇文章最后只留一个可复用框架，大概就是这一节。

以后再看到新的 RL 对齐算法，先别急着记它叫什么。先问下面五个问题：

| 问题 | 它真正决定什么 | 典型答案 |
| --- | --- | --- |
| 原始 reward 从哪里来？ | 你到底在优化什么 | 奖励模型、规则验证器、过程奖励、环境反馈 |
| KL 写进 reward 还是单独成 loss？ | 约束是否直接参与 advantage 构造 | 经典 RLHF 更偏前者，GRPO / REINFORCE++ 某些变体更偏后者 |
| baseline 从哪里来？ | 方差如何降低 | critic、moving average、leave-one-out、group mean |
| normalization 在什么范围上做？ | 模型在和谁比较自己 | local group、global batch、running statistics |
| 最终进入更新的是什么？ | 真正被优化的对象是什么 | sequence reward、cumulative return、normalized advantage |

这五个问题几乎可以把所有主流方法重新排列一遍：

- **PPO 式 RLHF**：critic + KL + PPO update，把 reward model 的信号组织成稳定更新；
- **DPO**：干脆绕开在线 advantage estimation；
- **GRPO**：group mean baseline + group std，把信号定义成局部相对优势；
- **REINFORCE++**：把归一化尺度推进到 batch / global 级别。

说到底，本文讨论的不是完整的 reward production，而是 reward signal shaping / reward consumption：奖励信号进入优化器前后，怎样被约束、比较、归一化并解释成 advantage。它至少包括：

**原始反馈怎样经过一系列统计与优化处理，最后被解释成可以训练策略的优势信号。**

换个更直白的说法，这条主线就是 `reward-to-advantage design`：名义上是 `reward`，实际是在改进 RL 算法本身。

## 小结

如果把 LLM 对齐中的强化学习看成一串算法名，那么这几年发生的事情像是在不断换缩写；但如果把它看成一条从 reward 到 advantage 的信号链，许多东西就会突然变得连贯。

- PPO 式 RLHF 解决的是：怎样把 reward model、critic 和 KL 组合成一套可大规模工作的系统模板；
- DPO 告诉我们：有些问题可以绕开在线 RL；
- GRPO 说明：没有 critic 之后，baseline 与 normalization 会被转移到组内统计上；
- REINFORCE++ 则进一步把问题收敛到一个具体争论上：**advantage 的尺度，到底应该局部定义，还是全局定义。**

这条研究线可以概括为一句话：

**至少在 RL 算法这条线上，改进的重点不是不停发明新的奖励，而是在重写奖励信号进入优化器前后应该怎样被解释。**

如果这条链处理得足够成熟，后续瓶颈才会进一步转向 rollout 选择、样本效率与系统吞吐。PODS 这类训练系统问题先放到附录；主线下一步先回到更上游：reward 自己到底如何被生产出来。

## 附录：PODS 与训练系统旁支——并不是所有 rollout 都值得用于更新

[PODS](https://arxiv.org/abs/2504.13818) 是 training-system 侧的旁支，不是本系列下一篇的主线。它有意思的地方，不是在重写 reward，也不是在重写 advantage，而是在追问一个更靠近训练系统本身的问题：**哪些 rollout 值得送进一次昂贵的参数更新。** 在在线 RL 里，生成 rollout 往往更容易并行，真正贵的是后续的反向传播、跨卡同步和优化器状态维护，所以问题开始从"怎么定义信号"转向"哪些样本值得消耗更新预算"。

它的思路也很朴素：先生成更多 rollout，再筛掉信息量较低的样本，只把更有训练价值的子集交给原有的 GRPO / PPO 目标去更新。这里看的不是单纯谁分高，而是谁更能拉开训练信号；论文里强调的 max-variance down-sampling，更倾向于同时保留一部分高奖励样本和低奖励样本，而不是只做 top-k 式保留。

如果把它放回这篇文章的主线里看，PODS 讨论的已经不是 reward-to-advantage 这条链本身，而是这条链足够成熟之后，瓶颈怎样继续转向 rollout 选择、样本效率与系统吞吐。主线仍然是信号怎样被解释，PODS 讨论的则是解释完成之后，哪些样本值得用于更新模型。

## 参考资料

### 基础理论

- [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://doi.org/10.1007/BF00992696)
- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation)
- [Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning](https://www.jmlr.org/papers/volume5/greensmith04a/greensmith04a.pdf)

### 经典 RLHF / PPO

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Learning to Summarize from Human Feedback](https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

### critic-free 与归一化

- [Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs](https://aclanthology.org/2024.acl-long.662/)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Normalization](https://arxiv.org/abs/2501.03262)

### 系统旁支

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning](https://arxiv.org/abs/2504.13818)
