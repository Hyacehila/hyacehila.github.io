---
layout: blog-post
title: "从数据整理到 Agentic RL：把训练真正接成一个闭环"
date: 2026-03-21 22:00:00 +0800
categories: [LLM]
tags: [Agentic RL, Training Systems, Data Curation, SFT, RL, Post-Training]
series: "Reward and Training"
author: Hyacehila
excerpt: "Agentic RL 真正要解决的，不是把 RL 接到 tool use 上，而是把任务整理、环境合同、反馈栈、offline shaping、online rollout 与部署回流接成一条完整训练飞轮。"
featured: false
math: false
---

# 从数据整理到 Agentic RL：把训练真正接成一个闭环

[上一篇文章]({% post_url 2026-03-19-reward-design-evolution-from-rlhf-to-rlvr %})主要在讲 reward 自己是怎样被生产出来的。这一篇我想把问题再往后推一步：**这些 verifier、judge、tool feedback 和环境状态，一旦都准备好了，它们到底怎样从数据整理开始，一路长成一条真正可运行的 agent 训练闭环。**

同时我也想先把边界说清楚。下一篇[《Reward 与 Training 在真实 Agent 中如何闭环：从数据治理到在线 RL》]({% post_url 2026-03-22-reward-and-training-in-agent-k-paperbench-amap %})会更多讨论 `Agent K`、`PaperBench`、`AMAP` 这类具体系统和具体 recipe，所以这一篇我会刻意保持在一个更宏观的视角：**不急着问某篇论文用了什么 trainer，而是先问 agentic RL 到底把训练对象、训练顺序和训练飞轮重写成了什么。**

如果只保留一句话，我现在的压缩判断是：**Agentic RL 不是把 RL 接到 tool use 上，而是把任务、数据、环境、反馈、训练和部署回流接成一条完整飞轮。** 真正决定系统上限的，通常不是某个 loss 的名字，而是这条链是不是从一开始就被接对了。

## 为什么 Agentic RL 需要重写训练对象

对我帮助最大的一手材料，还是 [The Landscape of Agentic Reinforcement Learning for LLMs: A Survey](https://arxiv.org/abs/2509.02547)。这篇综述真正重要的地方，不只是又总结了一批论文，而是它把很多原本混在一起讨论的对象强行拆开了：传统的 preference-based reinforcement fine-tuning 更接近一个 `T = 1` 的退化 MDP；而 agentic RL 则是一个 **部分可观测、时序展开、带真实状态转移的 POMDP**。

这件事一旦说清楚，很多原本看起来像“功能增强”的东西就会自动落位。规划、工具使用、记忆、自我改进、反思，甚至不同任务域里的各种 agent 形态，并不是平白多出来的模块；它们之所以突然变得重要，是因为训练对象不再是“单次回答质量”，而是**policy 在环境里走完整条轨迹时的行为质量**。

换句话说，过去很多 LLM-RL 的讨论，默认的训练对象仍然是一段文本：给一个 prompt，生成一个 answer，然后在终点给一个分数。可一旦走到 agentic RL，状态不再是静态 prompt，动作也不再只是 token；动作会改变环境，环境会产生新的 observation，reward 会沿着轨迹延迟返回，credit assignment 也会从“最后一句答对没”变成“哪一步真的推动了任务完成”。

所以我现在越来越不喜欢把 Agentic RL 写成“RFT 的升级版”或者“工具调用时代的 RL”。真正的问题不是模型会不会调用工具，而是**训练目标本身已经从回答优化，变成了环境中的策略学习**。只要这一步没有说清楚，后面关于环境、数据、SFT、online RL 的所有讨论都会显得像在补配件。

## 先看全流程：从数据到训练飞轮

很多文章一讲 agentic RL 就从 PPO、GRPO 或者某个 rollout 系统开始。但我现在越来越觉得，这个起点通常太晚了。因为你看到 optimizer 之前，系统其实已经做完了一连串更关键的决定：哪些任务值得学，哪些日志要被保留，工具边界怎样写，失败怎么记录，哪些反馈先做数据闸门，哪些信号才真的该进入 policy update。

如果把一条更像真实世界的 agent 训练链压成最短形态，我会写成：

```text
raw tasks / logs
-> curated task pool
-> environment contract
-> verifier / reward stack
-> seed SFT
-> offline shaping
-> online RL rollout
-> replay / curriculum / environment scaling
-> distill / deploy
```

这条链最重要的不是顺序本身有多神圣，而是它提醒我们：**reward 不是孤零零的一行分数，RL 也不是一句“后面再训一下”。** 在 agent 体系里，reward 往往先是数据筛选器，然后才是训练信号；环境不是 benchmark 的配套，而是训练对象的一部分；而 distillation 也不是最后的善后，而是把 rollout 中发现的高价值行为沉淀回部署策略的必要环节。

所以这一篇我会按一条更实用的顺序来讲：先从任务整理和数据治理开始，再写环境合同和反馈栈，然后再进入 cold-start SFT、offline shaping、online rollout，以及后半段真正决定飞轮能不能持续转下去的 replay、curriculum、environment scaling 和 distill back。

## 第 0 层：任务整理、数据治理与轨迹 schema

很多人一谈 agent 训练，默认前面已经有一批“可训练数据”。但对真实 agent 系统来说，这往往是最不真实的假设。因为原始世界给你的通常不是整齐的 `(prompt, response)`，而是一堆杂乱的 query、历史日志、工具调用记录、用户修正、失败轨迹和不完整的环境状态。**训练闭环的第一步，往往不是优化，而是整理。**

为什么我这次要把数据整理放到最前面？因为 agentic RL 里的“数据”本来就不只是文本样本，而是任务空间的切法。你首先得知道：哪些请求属于同一种任务，哪些约束是这个任务的核心难点，哪些请求虽然高频但根本不该交给当前 agent，哪些失败是低价值噪声，哪些失败恰恰是最值得保留的恢复样本。没有这一步，后面的 reward、curriculum 和 online RL 很容易只是在噪声上打转。

这一层至少有四件事必须先被做掉。

**第一，先把任务池切出来。** 你需要某种 taxonomy，去区分任务类型、约束密度和能力边界。不是因为分类本身优雅，而是因为后面你要讨论覆盖度、长尾、难度分布和 curriculum，前提都是任务空间先被切成可管理的形状。

**第二，给任务写难度。** 难度不是附属标注，而是后面 curriculum 的地基。一个系统如果不知道什么叫简单、什么叫长程依赖、什么叫多工具协调、什么叫高约束低容错，它通常也很难知道该先让模型学什么、该把哪些失败当成正常探索、又该把哪些失败视作协议崩坏。

**第三，负样本和边界样本必须保留。** 很多 agent 不是不会做事，而是不会说“不该做”、不会承认信息不足、不会在超出工具边界时停下来。也正因为如此，out-of-scope 请求、不可解请求、危险请求和高价值失败样本，都不应该被简单清洗掉。它们是后面 refusal、recovery 和 trustworthiness 的一部分训练地基。

**第四，尽早统一 trajectory schema。** 一条轨迹到底按什么切，是按 utterance、tool call、code execution、environment transition，还是按更抽象的 step？这个问题如果前面不决定，后面的 verifier、masking、replay、process reward 和 credit assignment 都会跟着漂。很多系统到后面越训越乱，本质上不是 optimizer 不好，而是根本没有统一“什么叫一条可回放、可验证、可学习的轨迹”。

所以我现在越来越愿意把“数据治理”理解成 agentic RL 的第 0 步，而不是附属清洗流程。真正的问题不是样本够不够，而是**你有没有把原始世界整理成一个可训练的任务池，以及一套后面能够反复回放的轨迹语言。** 下一篇我会拿更具体的系统例子去展开这件事，但在宏观层面，这已经足够决定一条训练线能不能起飞。

## 第 1 层：环境合同、工具边界与反馈栈

任务池被整理出来以后，下一步不是立刻上 SFT，而是先把环境合同写清楚。因为 agentic RL 和普通 post-training 最大的分水岭之一，就在于**环境不再只是评测舞台，而是训练对象的一部分。** 这也是为什么综述会专门拿出一整章讨论 environment simulator 和 RL frameworks，而不是把它们当成附录素材。

所谓环境合同，我现在更愿意把它理解成一套最小但刚性的训练接口：模型到底能观察到什么，哪些动作只是文本，哪些动作会真正改变外部世界，tool output 怎么进入上下文，环境怎样 reset，失败码是什么，哪些 side effect 允许出现，哪些必须被 sandbox 限住。

这一层如果想看比较具体的工程锚点，[`verl` 的 agentic RL 文档](https://verl.readthedocs.io/en/latest/start/agentic_rl.html) 和 [agent loop 文档](https://verl.readthedocs.io/en/latest/advance/agent_loop.html) 很有代表性。它们反复强调 async rollout、sticky session、message fidelity 和 token-level consistency，本质上都在说明同一件事：**一旦 agent 开始多轮调用工具，rollout 就不再等于“同步生成一段文本”。** 如果轨迹边界、消息还原和 token 对齐做不稳，训练会在最基础的地方先失真。

我现在越来越倾向于把 feedback 也写成一个分层的 stack，而不是一个总分：

- **verifier** 负责能程序化验证的部分，比如测试是否通过、答案是否满足规则、格式是否合规；
- **process signal** 负责中间步骤是否推进了任务，比如工具选择、子目标完成、局部修复是否有效；
- **judge** 负责 verifier 暂时覆盖不到、但又不得不评估的开放部分；
- **trace / audit** 负责让你在训练后知道模型到底是“不会做”、”乱做“还是”在错误的地方被奖励了“。

上一篇在讲 reward 时我已经说过，reward production pipeline 从来不是一个模型吐出一行分数这么简单。到了 agentic RL，这句话会变得更硬：**很多反馈首先应该被用来定义环境边界和数据闸门，其次才应该进入优化器。**

而且 trustworthiness 也不是最后一章才需要补的安全条款。综述在 challenge 章节把安全、幻觉、sycophancy 放得很靠前，我认为是对的，因为 agent 的攻击面从一开始就比普通 LLM 大得多。工具、记忆、外部 API、网页、数据库、跨 agent 通信，全都会变成状态转移的一部分。也正因为如此，sandbox、权限边界、可审计 trace 和 refusal policy，本来就应该从环境合同阶段开始写，而不是等 RL 把危险策略学出来之后再补丁式修理。

## 第 2 层：cold-start SFT 与 offline shaping

等环境合同和反馈栈站稳以后，cold-start SFT 才真正开始有意义。它在 agent 训练里最重要的作用，不是再重复一遍“模型已经会的东西”，而是**先把合法动作先验、基本节奏和交互协议写进 policy。**

很多时候，大家会把 SFT 和 RL 写成一种此消彼长的关系，好像只要 online RL 足够强，前面的 demonstrations 就都不重要了。但越看真实系统，我越觉得这是一种非常单轮任务的想象。对 agent 来说，SFT 压进去的不是抽象能力，而是很具体的行为习惯：什么时候先读 observation，什么时候先调用工具，什么时候该追问，工具失败时是重试还是回退，什么时候应该停止继续探索并交付结果。

这也是为什么很多看上去“RL 很强”的系统，前面其实都做了相当扎实的 cold-start。像 [DeepSeek-R1](https://www.nature.com/articles/s41586-025-09422-z) 最终的完整 recipe，也不是简单的 RL-only 传说，而是回到了 cold-start、筛选、再 SFT、再 RL 的闭环。工具环境里的系统更是如此：如果模型连合法调用都不会，online RL 往往只会让它在稀疏噪声里浪费大量 rollout 预算。

我也越来越愿意把 offline shaping 单独从 SFT 后面拎出来讲。因为很多系统真正需要的，并不是一上来就学长程探索，而是先把策略分布拉回一个更可控的区域：格式修正、局部协议遵守、调用语法稳定、事实保真、短程恢复，这些问题通常更适合在离线阶段先做干净。DPO、reward-guided filtering、rejection sampling、基于 verifier 的回流数据，都属于这一层。

所以如果只保留一句话，我对这一层的理解是：**SFT 负责把模型送进可学习区域，offline shaping 负责把这个区域清干净。** 很多“后面 RL 训得很稳”的系统，本质上不是突然学会了探索，而是前面已经把不必要的噪声和低级错误压掉了。

## 第 3 层：online rollout、credit assignment 与稳定训练

等到任务池、环境合同、反馈栈和 cold-start prior 都有了，online RL 才真正值得登场。它在 agent 里的职责也和单轮偏好优化不一样：不是单纯让回答更像人喜欢的文本，而是让 policy 学会**何时行动、何时追问、何时切换策略、何时停止，以及怎样在长程交互里恢复错误。**

这一层最容易被低估的，其实不是算法名，而是 rollout 本身。agent 的 rollout 不是“吐一段长文本”这么简单，而是一串持续和环境交换 observation、action、feedback 的交互历史。你如果没有可靠的异步执行、消息复原、状态跟踪、masking 和 trace，所谓 agentic RL 很容易只剩论文里的 loss，而没有真正可用的训练系统。

也正因为如此，我现在越来越不把 online RL 的核心问题理解成“该用 PPO 还是 GRPO”，而是先问三件更底层的事：

- **哪些 token 或 action 真该被训练？**
- **一条成功轨迹的 credit 应该往前分到哪里？**
- **哪些中间行为值得直接给过程奖励，哪些只该在终局结算？**

这也是 agentic RL 比单轮 RFT 更难的地方。单轮任务里，终局 0/1 reward 往往还勉强够用；但在多轮 agent 里，成功很可能是若干局部决策共同造成的。一次正确的工具选择、一次及时的放弃、一次有效的错误恢复，往往比最后一句漂亮答案更重要。像 [Agent Lightning](https://arxiv.org/abs/2508.03680) 去做 trace-to-transition 的重写，或者 [GiGPO](https://arxiv.org/abs/2505.10978) 去拆更细的 group-based advantage，本质上都在回答同一个问题：**长轨迹的 credit 到底怎样才能被分回真正关键的局部动作。**

与此同时，稳定训练也不只是数值稳定。对 agent 来说，它还包括 reward stack 的稳定、工具调用吞吐的稳定、环境响应延迟的稳定、以及策略探索不要把系统推到危险区域的稳定。综述里把 trustworthiness 和 scaling up training 放在开放挑战里，我觉得很准确，因为从 online rollout 开始，这两件事就不再是额外话题，而会直接决定训练能不能持续进行。

如果只保留一句更尖锐的判断，我会说：**online RL 真正稀缺的，不是再给模型一个更高的总分，而是让它在环境里学会探索、恢复和时机。** 只有当你的瓶颈真的落在这些地方时，online RL 才配得上它那套昂贵的系统代价。

## 第 4 层：replay / curriculum / environment scaling / distill back

一旦模型开始在环境里真正学习，后半条链就会立刻出现：经验要不要回流，难度怎么调，环境要不要跟着模型一起变，最后又怎样把 rollout policy 沉淀回部署策略。也正是在这里，Agentic RL 才真正从“一次训练”变成“一个飞轮”。

**replay** 的价值不只是节省样本，而是决定成功经验、边界失败和高价值恢复路径会不会继续留在系统里。**curriculum** 的价值也不只是从易到难，而是让 agent 始终待在可学习边界附近，而不是被长程、稀疏、噪声过大的任务直接打成零奖励。

再往后一步，就是综述里专门强调的 **scaling up environment**。这是我在读那篇 survey 时最有共鸣的一部分：agent 的上限经常不只被 model size 和 training steps 限制，也被它所处的训练世界限制。环境如果太贫、太静态、太不安全、太难 reset，模型很快就会学满；而可扩张、可合成、可程序化验证的环境，则会把数据和训练一起变成更持久的飞轮。像 [VeriEnv](https://arxiv.org/abs/2603.10505) 或其他 environment generation 方向的重要性，也正在这里。

但我现在越来越觉得，后半段最容易被低估的仍然是 **distill back**。很多 rollout policy 在训练时很强，却不适合直接部署：成本太高、上下文太长、对 sandbox 和 trace 依赖太重、风格太像探索中间态，或者安全边界还不够稳。所以更常见也更现实的路线，是把 online RL 中挖出来的高价值轨迹、恢复策略、停止条件和工具调用时机重新组织成更稳定的数据，再蒸馏回更便宜、更鲁棒的 deployment policy。

也就是说，Agentic RL 的后半段真正要管理的，从来不只是“继续训”，而是四件绑在一起的事：

- 经验回流能不能持续发生；
- 难度调度是否贴着模型能力边界；
- 环境会不会随着模型一起扩张；
- 训练时学到的高价值行为能不能沉淀回真正可部署的策略。

如果只允许我保留这一篇最短的一句话，我会写成：**Agentic RL 真正的分水岭，不是谁先把 LLM 接上 RL，而是谁能把“可训练任务池、可执行环境、可分层反馈、可回放轨迹、可持续环境扩张，以及可部署蒸馏回流”同时搭起来。**

