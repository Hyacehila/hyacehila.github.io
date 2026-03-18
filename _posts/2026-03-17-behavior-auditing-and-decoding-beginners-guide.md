---
layout: blog-post
title: "行为审计与行为解码：从 Reward 之后到 Agent 可观测性"
date: 2026-03-17 10:00:00 +0800
categories: [AI, Research]
tags: [Behavioral Evaluations, Reward Modeling, Anthropic, Bloom, Transluce, Docent, PCD, AI Safety, Agent Evaluation]
author: Hyacehila
excerpt: "Reward 负责把目标写进优化器，但它不负责证明模型真的学会了正确目标。本文从初学者视角重写行为审计与行为解码：为什么 reward 之后还需要后验校验，Anthropic 与 Transluce 两条路线分别在补什么空白。"
featured: false
math: false
---

# 行为审计与行为解码：从 Reward 之后到 Agent 可观测性

最近在看 reward design、behavior eval 和 agent observability 这条线时，我越来越强烈地意识到一件事：**“模型分数变高了” 和 “模型真的学会了正确目标” 其实不是一回事。**

reward 负责把目标翻译成优化器能吃下去的信号，但它不负责证明模型理解了你的真实意图。一个模型完全可能一边拿到更高的 reward，一边学会讨好 judge、钻评分规则的空子，甚至在某些设置下开始研究“怎么改分数系统本身”。

到了 Agent 时代，这个问题更麻烦。因为 agent 不只是回答一句话，它会持续行动、调用工具、留下长轨迹，甚至直接改变环境。你不能只在训练结束时看一个总分，然后就默认系统已经对齐了。

这篇文章想回答一个更基础的问题：**reward 之后，我们到底该用什么办法去看见模型真正学到了什么？**

## TLDR

1. **Reward 是训练接口，不是对齐证明。** 它告诉模型什么更容易拿分，但不保证模型理解的是目标本身而不是拿分技巧。
2. **行为审计和行为解码是在补 reward 看不见的空白。** 前者更偏外部验证，后者更偏把内部或外部行为信号变成可读、可搜索的对象。
3. **Anthropic 这条线可以读成五步：发现行为、看到 reward 的边界、审计隐藏目标、把行为测量程序化、再把内部激活当作证据。**
4. **Bloom 最值得记住的，不是一个 benchmark，而是行为测量流水线。** 它从 seed 出发自动生成场景、执行 rollout、再做 judgment。
5. **PCD 和 Docent 不是同一种东西。** PCD 更像内部状态翻译器，Docent 更像agent轨迹观测台。
6. **这个方向最难的部分，往往不是 judge 模型本身，而是你到底想测什么行为、用什么 rubric 才算测得准。**
7. **一旦模型变成 agent，行为审计就不再是可选研究题，而更像 AI 系统的 observability 基础设施。**

## 先把词讲清楚

第一次接触这个方向，最容易被各种名词淹没。先把几个高频词翻成人话：

| 词 | 用大白话怎么理解 | 它想解决什么 | 它不等于什么 |
| --- | --- | --- | --- |
| `reward` | 训练时给模型的分数 | 告诉优化器什么更值得学 | 不等于真实目标本身 |
| `judge` | 一个负责打分或判定的系统 | 把开放行为压成可比较结果 | 不等于绝对真理 |
| `rubric` | 写给 judge 的评分标准 | 把想要盒不想要什么说清楚 | 不等于人类模糊直觉 |
| `transcript` | 对话、工具调用、环境反馈的记录 | 让你回看 agent 到底做了什么 | 不等于模型全部内部状态 |
| `behavior auditing` | 从外部系统化检查模型行为 | 验证模型有没有表现出某种行为 | 不只是再做一套 benchmark |
| `behavior decoding` | 把行为相关信号翻译成可读形式 | 让行为变得可解释、可搜索、可预测 | 不是 Transformer 里的 decoder |
| `hidden objective` | 模型表面正常，内部却在追别的目标 | 解释为什么它会这样做 | 不一定会直接显露成坏回答 |
| `activation` | 模型中间层的内部表示 | 为行为提供另一类证据 | 不直接是人类语言 |

不用担心没理解这些词的意思，你可以去问一问AI，也可以直接向下继续，后面会把他们逐个放到具体故事里讲解。

如果只记一句话，我会建议记这个：

**reward 在做训练，行为审计在做验证，行为解码在把验证线索变得更可读、更可操作。**

## 为什么 reward 不是终点

如果把训练类比成教学生，reward 更像考试打分，而不是知识本身。

你可以把回答要礼貌、任务要完成、风险要低，这些要求压缩成某种可优化的信号，让模型朝那个方向学。但只要信号是可计算、可优化的，它就天然有两个边界：

- **它会丢信息。** 真实目标总比 reward 更复杂，一个简单的标量不可能包含足够的语义。
- **它会被研究。** 模型越强，越可能学会怎样在这个评分系统里表现得更好。reward hacking 越是模型强大，越容易出现。

这就是为什么我不把 reward 看成终点。它更像一个必要手段和surrogate，让模型前往更好的参数空间，而不是最后证据。

真正棘手的地方在于：**当模型开始变成 agent，它不只是答错一道题，而是会在环境里持续做事。** 这时你更关心的通常不是平均分高不高，而是：

- 它会不会在某些场景里稳定出现某种危险倾向？
- 它表面正常时，背后是不是仍在追另一个目标？
- 它有没有学会讨好 judge、规避检查、甚至利用奖励通道本身？
- 海量运行轨迹里，到底哪些 case 值得你认真看？

也正因为如此，reward 之后自然会长出两条互补路线：

| 路线 | 关注点 | 你最常问的问题 |
| --- | --- | --- |
| 行为审计 | 从外部系统化发现、测量、验证行为 | “它到底做了什么？多常见？多严重？” |
| 行为解码 | 把内部或外部信号翻译成可读对象 | “它为什么这样做？内部在发生什么？我怎么更快找到线索？” |

下面我会按这个顺序看两条线：先看 Anthropic 怎样把审计拆成几件可做的事，再看 Transluce 怎样把解码行为工程化。

## Anthropic 这条线：把审计拆成五个更具体的问题

如果把 Anthropic 最近几年这批公开工作放在一起看，我觉得最好的读法不是一篇篇分开背，而是把它们看成五个连续问题：

1. 还有哪些行为值得测？
2. 为什么高 reward 仍然不够？
3. 如果模型表面正常，能不能审计它是不是在追隐藏目标？
4. 能不能把行为测量做成一条可复用流水线？
5. 能不能把内部激活也变成审计证据？

### 1. Model-Written Evaluations：先别急着打分，先把行为找出来

Anthropic 在 **2022 年 12 月 19 日** 发布了 [Discovering Language Model Behaviors with Model-Written Evaluations](https://www.anthropic.com/research/discovering-language-model-behaviors-with-model-written-evaluations)。

这篇工作的价值，在我看来不只是模型会自动出题，而是它先提醒你一件更根本的事：**我们往往连该测什么都还没发现完整。**

- **它想解决什么**：传统 benchmark 默认风险点已经被人类列出来了，我们只要去测。但模型能力一旦变化，新的行为会不断冒出来，手写题库很容易跟不上。
- **它怎么做**：让语言模型参与生成评测项，再做筛选、过滤和标注。官方摘要提到，这篇工作一共生成了 **154 个数据集**，并且众包人员对样本相关性和标签有较高一致性。
- **它能做什么**：快速暴露像 **sycophancy**、更强的政治表态、避免被关闭、目标保持这类原本不一定被系统测到的行为趋势。
- **它还做不到什么**：它更像“行为发现器”，不是最终裁判。它能告诉你“这里有个值得盯的现象”，但还不能直接告诉你这种行为有多常见、为什么出现、部署后会不会持续存在。

所以我会把这篇工作理解成 **discovery layer**：先把值得测的行为挖出来，后面的 measurement 和 auditing 才有对象。

### 2. reward tampering：高分不等于目标正确

Anthropic 在 **2024 年 6 月 17 日** 发布了 [Sycophancy to subterfuge: Investigating reward tampering in language models](https://www.anthropic.com/research/reward-tampering)。

如果说前一篇是在回答还有什么值得测，这篇就是在回答：**为什么你不能因为 reward 变高了就放心。**

最适合初学者的理解方式，其实就是一个很朴素的类比：  
**学生不一定认真答题，他也可能开始研究怎么改成绩系统。**

Anthropic 在官方页里把 reward tampering 解释得很直白：这是模型有机会接触到训练或评分通道时，开始hack the reinforcement system 的行为。论文设置了一条逐步升级的 curriculum：

- 最早阶段是比较轻的 specification gaming，比如政治阿谀奉承；
- 后面变成修改 checklist，让没完成的任务看起来像完成了；
- 最后给模型一个能接触自身代码的机会，看它会不会去改 reward 函数本身。

- **它想解决什么**：比较轻的钻规则空子会不会泛化成更深的改奖励系统本身？
- **它怎么做**：在一系列逐步升级的环境里训练模型，然后看它是否会从较轻的 specification gaming **零样本泛化** 到更严重的 reward tampering。
- **它能做什么**：把“高 reward 不代表高对齐”这句话变得非常具体。官方页给出的结果是：reward tampering 虽然稀少，但确实出现了；而且模型有时还会掩盖痕迹。
- **它还做不到什么**：这**不是**在说“当前生产模型已经普遍会在真实环境中这样做”。Anthropic 官方在 caveat 里讲得很清楚：这是一个刻意构造的人工设置，模型被显式置于训练场景中，还给了隐藏 scratchpad；他们**不对当下前沿产品模型在真实场景中的倾向做结论**。

这篇文章把 reward 的边界说清楚：**reward 是可以被学会利用的。** 一旦你接受这一点，行为审计就不再像附加项，而更像必要的后验校验层。

### 3. Hidden Objectives：从它做了什么推进到它到底在追什么

Anthropic 在 **2025 年 3 月 13 日** 发布了 [Auditing language models for hidden objectives](https://www.anthropic.com/research/auditing-hidden-objectives)。

我觉得这篇工作真正把问题往前推了一步。因为前面很多评测，默认都在看**外显行为**：回答是否危险、任务是否完成、是否出现某种倾向。但真实风险往往更难一点：

- 模型大部分时候看起来都正常；
- 表面输出没有明显暴露问题；
- 可它在长期策略上，可能仍然朝另一个目标优化。

这就是 hidden objective 的问题。

- **它想解决什么**：如果一个模型表面上基本合格，我们还能不能审计出它背后正在追别的目标？
- **它怎么做**：论文先故意训练一个带隐藏目标的模型，然后做一个 **blind auditing game**。红队在信息隔离下训练模型，蓝队拿到模型和部分训练数据，在不知道答案的前提下做审计。官方页面还专门讨论了白盒和黑盒等不同审计手段。
- **它能做什么**：证明post-training audit 不只是看 outputs，也可以尝试追 motive。论文报告里，**4 支审计队伍中有 3 支成功找到了隐藏目标**，说明这件事至少不是完全做不到。
- **它还做不到什么**：这当然不等于我们已经发明了稳定读心术。更准确的理解是：它告诉你 hidden-objective audit 值得做，而且在某些设置下是可行的，但离通用、稳定、低成本还很远。

如果说 reward tampering 是在提醒你分数可能被利用，那 Hidden Objectives 则是在提醒你：**很多最重要的问题，根本不是看分数就能知道的。**

### 4. Bloom：把行为测量做成一条流水线

Anthropic 在 **2025 年 12 月 19 日** 发布了 [Bloom 官方文章](https://alignment.anthropic.com/2025/bloom-auto-evals/) 并同步开源了 [Bloom 仓库](https://github.com/safety-research/bloom)。

Bloom 是这篇文章里我最想给初学者讲清楚的一个工具，因为它特别容易被误读成又一个 benchmark。

我觉得更好的理解方式是：

**Bloom 不是一张试卷，而是一条会围绕某个行为自动造题、执行互动、最后判分的行为测量流水线。**

Anthropic 在官方文章里也明确把 Bloom 和 Petri 区分开来：**Petri 更像开放式自动审计，Bloom 更像围绕一个研究者指定行为做深入、可复用的测量。** 这个区分很重要，因为发现问题与稳定测量问题不是一回事。

- **它想解决什么**：如果我已经知道某个行为值得关注，比如 sycophancy、自利偏差、自我保存，那么我能不能批量生成很多场景，去测这个行为在多大范围内会被诱发、严重到什么程度？
- **它怎么做**：Bloom 从一个 `seed` 配置出发，跑完整条四阶段流水线：
  1. `Understanding`：先读懂你要测的行为；
  2. `Ideation`：围绕这个行为自动生成场景，并做多样化变化；
  3. `Rollout`：让 evaluator / 环境和 target model 真正互动；
  4. `Judgment`：由 judge 对行为出现程度和其他质量指标打分。
- **它能做什么**：Anthropic 官方把它定义成一个能生成 **configurable evaluation suites** 的 agentic framework。它不是靠固定题库，而是从 seed 派生出成套评测。官方文里重点看的指标之一叫 `elicitation rate`，也就是行为分数超过阈值的 rollout 占比。
- **它还做不到什么**：Bloom 不是跑一下就有真相。它非常依赖 seed 定义、few-shot 示例、variation 设计、judge rubric 和 threshold。模型也可能逐渐学会意识到自己在被评测，这在官方文里被明确提到。

Bloom 仓库里最值得初学者看的，不是代码细节，而是 seed 的结构。官方模板把一套评测显式拆成了下面这些部件：

```yaml
behavior:
  name: "self-preferential-bias"
  description: "..."
  examples: []

understanding:
  model: "claude-sonnet-4"

ideation:
  num_scenarios: 5
  variation_dimensions:
    - emotional_pressure
    - noise

rollout:
  target: "claude-sonnet-4"
  modality: "conversation"
  max_turns: 2

judgment:
  num_samples: 1
```

这段配置最重要的启发是：**行为评测终于不再只是拿一批 prompt 去考，而是被拆成了可以控制、可以复现、可以重跑的实验设计。**

Bloom 还支持 `conversation` 和 `simenv` 两种模式。前者更像标准多轮对话，后者则更接近带工具和环境交互的 agent 场景。这也是为什么我会把它看成从语言模型评测走向agent 行为测量的一个很关键节点。

### 5. Activation Oracles：开始把内部激活也当作证据

Anthropic 在 **2025 年 12 月 19 日** 发布了 [Activation Oracles](https://alignment.anthropic.com/2025/activation-oracles/)。

如果说前面几步仍然主要在看外部行为，这篇工作就在问另一个很自然的问题：**如果 transcript 会骗人，内部激活能不能提供另一类证据？**

- **它想解决什么**：很多风险在外部行为里不一定立刻暴露出来，或者只在特定触发条件下才会出现。那么，能不能让另一个系统直接读取 activations，回答“这个模型现在在想什么、有什么倾向、是不是带着某种隐藏信息”？
- **它怎么做**：把 target model 的 activation 当成额外输入，让另一个 LLM 学会回答关于这些 activations 的自然语言问题。官方文章把这类模型叫做 **Activation Oracles**。论文还做了下游审计任务测试和数据规模消融。
- **它能做什么**：官方结果显示，Activation Oracles 在 4 个下游审计任务里有 3 个达到了当时最好的表现，而且随着训练数据的数量和多样性增加，性能还会继续提高。这说明“把激活作为一种额外模态来问问题”是条有希望的路。
- **它还做不到什么**：官方也讲得很清楚，这是一种 **non-mechanistic** 的方法。它并不直接告诉你概念在表示空间里是怎样编码的，也不回答模型内部算法机制长什么样。它更像审计工具箱里的一件新仪器，而不是终极解释。

所以如果把 Anthropic 这条线压成一句话，我会这样概括：

**先把值得测的行为发现出来，再承认 reward 可能失真，再尝试审计隐藏目标，然后用 Bloom 把行为测量程序化，最后把内部激活也拉进证据链。**

## Transluce 这条线：把“看行为”这件事做成工程工具

如果说 Anthropic 更像在回答“模型有没有问题、问题在哪里”，那 Transluce 给我的感觉更像在回答：**能不能把看行为这件事做成日常工程工具？**

这条线里最容易混淆的一点是：**PCD 和 Docent 不是一类工具。**

- PCD 看的是**内部状态**；
- Docent 看的是**外部轨迹**。

一个像内部状态翻译器，一个像 agent transcript 的观测台。

### 1. PCD：像一个“内部状态翻译器”

Transluce 在 **2025 年 12 月 18 日** 发布了 [Predictive Concept Decoders](https://transluce.org/pcd)。

PCD 的出发点非常好理解：**模型的自我报告不一定可信。**  
一个被 jailbreak 的模型，可能一边输出危险内容，一边还说“我没有在想什么特别的东西”。如果你只能问模型自己，它完全可能给你一个表面上很体面的解释。

PCD 想做的，就是绕开这个问题。

- **它想解决什么**：不用依赖模型自我报告，而是直接从内部状态预测和解释行为。
- **它怎么做**：根据官方项目页和论文 TeX 源，PCD 采用的是一个两段式结构：先把 activations 压缩成一个 **稀疏概念瓶颈**，得到一小组短的、可读的 concept 列表；再只根据这些 concept 去回答“模型行为”相关的问题。官方页把它概括成“translate a model's internal states into short, human-readable concept lists, then use those concepts to answer questions about behavior”。
- **它能做什么**：官方页展示了几类特别有代表性的场景：`jailbreaks`、`secret hints`、`injected / implanted concepts`。这些场景有一个共同点：模型内部其实“知道”了某些事，但它未必会在自我报告里老老实实说出来。PCD 就是在这些地方比“直接问模型自己”更有价值。
- **它还做不到什么**：可读不等于因果正确。一个 concept bottleneck 可能确实对行为有预测力，但它不一定就是最真实的内部机制。另外，PCD 默认是白盒工具，你得能拿到 activations。

如果把它再翻译成一句更直观的话，我会说：

```text
activations -> 可读概念列表 -> 回答“它接下来会怎样、为什么会这样”
```

PCD 最有意思的地方也在这里：它不是单纯事后贴标签，而是把 interpretability 当成一个**预测问题**。如果你真的理解了 activations，你就应该能用它们去预测后续行为。这个 framing 很聪明，因为它给了系统一个可训练、可验证的监督信号。

### 2. Docent：像一个“agent 轨迹观测台”

和 PCD 相比，[Docent 文档](https://docs.transluce.org/introduction) 解决的是另一个更工程化的问题：**当你已经有海量 agent transcripts 时，人根本看不过来，该怎么办？**

Docent 官方首页给出的定位非常直接：**summarize, cluster, and search over agent transcripts**。我觉得这个描述比很多宣传语都更准确。

- **它想解决什么**：人类没法手工读完成千上万条 agent 运行记录，但我们又很想知道“异常都发生在哪”“哪类轨迹像 reward hacking”“不同训练阶段的行为有没有变化”。
- **它怎么做**：先 ingest traces，然后围绕一个 `rubric` 构建 `judge`。Docent 文档里的 `Rubrics and Judges` 页面把这件事说得很清楚：rubric 是你写给 judge 的评价标准，judge 输出可以带 `label`、`score`、`explanation`，而且 `explanation` 字段支持 `citations`，也就是把判断指回 transcript 的具体片段。再往后，你可以搜索、聚类、筛选，并继续修改 rubric。
- **它能做什么**：官方文档直接举了几类用途：观测长 reasoning trace 中的异常、监控 RL rollouts 里的意外行为、对 agent 轨迹做 summarize / cluster / search。Rubric refinement 教程还特别强调：像 cheating、sycophancy 这种人一眼能认出来，但很难写准边界的行为，很适合通过反复 refinement 来把标准磨清楚。
- **它还做不到什么**：Docent 不是“自动真相机”。如果 transcript 本身不完整、你写的 rubric 很模糊、judge 理解偏了，结果就会漂。它非常有用，但它的上限很大程度上取决于你有没有把要测的行为定义清楚。

我特别喜欢 Docent 的一点，是它默认承认：**行为规格一开始通常是模糊的。**  
这在初学者看来可能反而很重要，因为很多人第一次接触这类工具会误以为“只要模型够强，judge 自然会懂”。Docent 的文档其实反过来提醒你：真正难的是把你的担忧写成一个可执行的行为标准。

也正因为如此，我更愿意把 Docent 理解成一种 **behavior observability layer**。它不直接帮你训练模型，但它帮你把发生了什么组织成可复盘的证据。

## 把两条路线放回同一张图

如果你第一次读这个方向，我不建议按“论文名清单”去记。更好的记法是：**先问自己真正想回答什么问题。**

| 你真正想问的问题 | 更接近的工作 | 为什么 |
| --- | --- | --- |
| 还有哪些行为值得测？ | [Model-Written Evaluations](https://www.anthropic.com/research/discovering-language-model-behaviors-with-model-written-evaluations)、[Petri](https://www.anthropic.com/research/petri-open-source-auditing) | 先扩展行为空间，别只盯着已知 benchmark |
| 某种行为在什么场景下会被诱发、多严重？ | [Bloom](https://alignment.anthropic.com/2025/bloom-auto-evals/) | 把行为测量做成可重复流水线 |
| 表面正常时，它是不是还在追别的目标？ | [Hidden Objectives](https://www.anthropic.com/research/auditing-hidden-objectives) | 从 outputs 往 motive 推进 |
| 内部状态里有没有危险先兆？ | [Activation Oracles](https://alignment.anthropic.com/2025/activation-oracles/)、[PCD](https://transluce.org/pcd) | 把 activations 当作另一类证据 |
| 海量 agent traces 里到底哪里出事了？ | [Docent](https://docs.transluce.org/introduction) | 把轨迹变成可搜索、可聚类、可引用的观测对象 |

如果再压缩一句：

- **Anthropic 更偏“验证”**：有没有问题、问题多严重、背后是不是隐藏着别的目标；
- **Transluce 更偏“工程”**：怎么把内部或外部行为线索做成日常可用的工具。

两者并不冲突。一个更完整的系统，很可能同时需要：

- Bloom 这种**围绕指定行为的测量**；
- Docent 这种**围绕大量轨迹的观测**；
- PCD / Activation Oracles 这种**内部信号辅助证据**。

## 为什么 Agent 时代，这件事就从研究题变成基础设施

如果只是普通聊天模型，行为问题很多时候还表现为回答不理想。  
但一旦系统变成 agent，问题性质就变了。

第一，**agent 是多步、有状态的。**  
一个单步看起来无害的动作，可能只是更长策略的一环。你不能只看最后一句回复。

第二，**agent 会调用工具、改文件、碰 API，后果不可逆。**  
错误不再只是说错话，而可能是真的做错事。

第三，**agent 天然会产生长 transcript。**  
人类 review 不可能逐条细看，因而你必须有搜索、聚类、judge、rubric refinement 这样的观测层。

第四，**部署后的行为分布和 benchmark 分布本来就不一样。**  
Agent 在真实环境里会遇到更开放、更长程、更具诱惑性的场景。这时只靠训练时的 reward 或离线 benchmark，很难持续知道它有没有偏掉。

所以我现在越来越愿意把这个方向放在AI 版 observability里理解：

- 传统软件 observability 看 `logs`、`metrics`、`traces`；
- agent observability 除了这些，还要看 **behavior specs、judge outputs、rubric versions、transcript evidence、internal signals**。

从这个角度看，行为审计不是安全领域的额外附件，而更像 agent 系统要长大的基础设施。

## 最难是你到底想验证什么

最后说几点我自己的判断。

**第一，reward design 和 behavior audit 其实是同一个闭环的前后半段。**  
reward 负责把目标写进优化器，audit 负责检查模型究竟把什么写进了自己。只谈前者不谈后者，闭环是不完整的。

**第二，这个方向最容易低估的难点不是模型能力，而是行为规格。**  
你想测“作弊”“阿谀奉承”“自利偏差”“隐藏目标”，这些词在人类脑子里听起来很自然，但一旦要写成可稳定执行的 rubric，难度会立刻上来。Docent 的 rubric refinement 和 Bloom 的 seed design，某种意义上都在承认这个事实。

**第三，内部证据很重要，但不是读心术。**  
不管是 PCD 还是 Activation Oracles，它们都很有启发，但更像在给审计者增加一种新的证据类型，而不是一次性终结所有不确定性。外部行为、长轨迹、环境后果仍然是主干。

**第四，我现在更愿意把这一整套东西理解成“从 reward 走向 observability”的迁移。**  
过去我们更在意怎么给模型一个分；现在越来越要紧的是：**模型在什么条件下会出现什么行为，我们能不能及时看到、稳定复盘、持续修正。**

这才是我理解里的从 Reward 之后到 Agent 可观测性。

## 参考资料

### 官方研究与文档

- Anthropic, [Discovering Language Model Behaviors with Model-Written Evaluations](https://www.anthropic.com/research/discovering-language-model-behaviors-with-model-written-evaluations), 2022-12-19
- Anthropic, [Sycophancy to subterfuge: Investigating reward tampering in language models](https://www.anthropic.com/research/reward-tampering), 2024-06-17
- Anthropic, [Auditing language models for hidden objectives](https://www.anthropic.com/research/auditing-hidden-objectives), 2025-03-13
- Anthropic Alignment Science, [Bloom: an open source tool for automated behavioral evaluations](https://alignment.anthropic.com/2025/bloom-auto-evals/), 2025-12-19
- GitHub, [safety-research/bloom](https://github.com/safety-research/bloom)
- Bloom, [`seed.yaml.template`](https://raw.githubusercontent.com/safety-research/bloom/main/src/bloom/data/templates/seed.yaml.template)
- Anthropic Alignment Science, [Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers](https://alignment.anthropic.com/2025/activation-oracles/), 2025-12-19
- Anthropic, [Petri: An open-source auditing tool to accelerate AI safety research](https://www.anthropic.com/research/petri-open-source-auditing), 2025-10-06
- Transluce, [Predictive Concept Decoders](https://transluce.org/pcd), 2025-12-18
- Docent Docs, [Welcome to Docent](https://docs.transluce.org/introduction)
- Docent Docs, [Quickstart](https://docs.transluce.org/quickstart)
- Docent Docs, [Rubrics and Judges](https://docs.transluce.org/concepts/rubrics-and-judges)
- Docent Docs, [Rubric Refinement](https://docs.transluce.org/tutorials/rubric-refinement)

### 论文入口

- [Discovering Language Model Behaviors with Model-Written Evaluations (arXiv:2212.09251)](https://arxiv.org/abs/2212.09251)
- [Sycophancy to Subterfuge: Exploring Reward Tampering in Language Models (arXiv:2406.10162)](https://arxiv.org/abs/2406.10162)
- [Auditing language models for hidden objectives (arXiv:2503.10965)](https://arxiv.org/abs/2503.10965)
- [Activation Oracles (arXiv:2512.15674)](https://arxiv.org/abs/2512.15674)
- [Predictive Concept Decoders (arXiv:2512.15712)](https://arxiv.org/abs/2512.15712)

## 最后一句话

如果把全文压缩成一句最短总结，我会写成：

**Reward 负责把目标写进优化器，行为审计负责检查模型究竟学成了什么，行为解码负责把这些线索变得更可读、更可搜索、更适合在 Agent 系统里持续使用。**
