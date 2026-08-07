---
title: "AI Agent 的来时路：我们曾经以为瓶颈在哪"
title_en: "How AI Agents Got Here: Where We Thought the Bottleneck Was"
date: 2026-07-25 20:00:00 +0800
categories: ["Agent Systems", "Agent Architecture"]
tags: ["AI Agent", "Agent Architecture", "AI Engineering"]
author: Hyacehila
excerpt: "从 2024 年到 2026 年，AI Agent 走过的不是一条向前的直线，而是一串猜测：我们不停地猜瓶颈在哪，然后把大部分人力和预算押在猜中的那个位置上。Prompt、RAG、Workflow、Tools、Context、模型与新词、评测，七层猜测，七批工具、名词与项目。这篇文章按时间顺序回看这条来时路。"
excerpt_en: "From 2024 to 2026, AI agents did not advance in a straight line. Each year was a guess about where the bottleneck was, and most of the budget and headcount went to whichever layer we guessed. Prompt, RAG, Workflow, Tools, Context, models and new vocabulary, evaluation—seven guesses, each growing its own tools, terms, and projects. This essay revisits that road."
mathjax: false
permalink: '/blog/2026/07/25/the-road-here-of-ai-agents/'
hidden: false
---

## 写在前面

这篇文章原本是[《当我们构建 AI Agent 时，我们究竟在解决什么问题？》](/blog/2026/07/28/what-problems-are-we-really-solving-when-building-ai-agents/)里的一章。那篇文章想回答三个问题：

* 当你在构建一个 AI Agent 的时候，你究竟在做什么？
* 当你想用 AI Agent 去解决一个真实场景里的问题时，你得先解决 AI Agent 自己的什么问题？
* 一个 AI Agent 项目为什么成功，又为什么失败？

这三个问题光盯着最近半年是看不出来的，所以在回答之前，需要先回到 2024 年，看看 AI Agent 的来时路。这一章后来长到了不适合放在一篇文章里的程度，于是拆了出来单独成篇。如果你想看那三个问题的答案，请回到原文；这里聊聊过去两年的 AI Agent 本身。

## 来时路

过去两年，AI Agent 并不是一条向前的直线。它更像一串猜测：我们不停地猜瓶颈在哪，然后把大部分人力和预算押在猜中的那个位置上。

每一次猜测都会长出一批工具、一批名词、一批岗位，也会立起一批项目。等这一层被填得差不多，或者模型的进步消灭这一层问题，下一层问题就露出来。上一批东西里有一小部分沉淀成基础设施，剩下的迅速贬值。

让我们来看看过去两年大家都发现了什么，做了什么的猜测，提出了什么方法。他们都是对前面三个问题的一个回答，并不是最终答案。

如果你精通 Agent，不妨跳过这一章，看看标题就好。我之所以这样写，更多是为了清理思绪。

### 咒语与 Prompt Engineering

ChatGPT 的发布是划时代的一刻。我们花了很多年，找到了一个适合语言序列建模的数学结构，还发明了配合其并行运算的硬件基础设施，同时把这些数学结构变成了一种入手门槛极低、泛化能力极强的交互方式 —— 聊天。少了任何一个基础，ChatGPT 都不会取得如此的成功。

既然我们所拥有的只是一个聊天机器人，那我们还能做些什么呢？好像只能决定说些什么话。于是去相信是自己的话没说对，才让 AI 输出得不好；毕竟如果你觉得是模型训练不足，那也没什么办法。说话因此成了一项技术，让 AI 输出更好结果的咒语成了一门科学，我们称之为 Prompt Engineering，专门研究它的人被称为 Prompt Engineer，好像曾经工资还不错。

Prompt Engineering 持续了不短的时间。看起来只是说话，但各种各样的技术都基于此被提出，比较经典的有 CoT、Few-shot、Role-Play；至于那些比较好玩的，take a deep breath、小费、威胁、PUA，也在 Skill 时代重出江湖。把任务讲清楚这件事本身从来没有过时，我们现在仍旧在做 Prompt Engineering，只是再也不提这个词了。

Prompt Engineering 里藏着一条到今天仍然成立的 Agent 开发原则：把问题讲清楚。这个名词的退场，不是因为它错了，而是因为我们不再需要专门的 Prompt Engineer，人人都是 Prompt Engineer。

这一段的技术细节可以参考[《提示工程与上下文学习》](/blog/2024/09/20/prompt-engineering-and-in-context-learning/)，如果你对 Spec 与本节的内容关系好奇，可以参考[《Spec 不是新范式》](/blog/2026/04/07/spec-is-not-the-new-paradigm/)。

### 知识增强与 RAG

RAG 是做 AI Agent 不得不聊的话题。如果你在 2023 年到 2025 年第三季度之间做过 AI Agent，那么 RAG 是 70% 的项目里都绕不开的一环。

它的基础想法很简单。模型的训练数据有截止日期，公司内部的数据也大多不能拿去训练；而把一份相关的材料 Prompt 给模型，输出大概率会更好。既然如此，一个自动去找材料、再把材料塞进 Prompt 的工具，就会很自然地生长出来。当时的判断同样自然：模型不是不会做，是不知道，知识可能才是系统最大的瓶颈。

RAG 因此成了那几年的默认答案。向量数据库拿到大笔融资，几乎所有传统数据库都补上了向量检索，每家公司都在自建内部知识库。切分策略、rerank、hybrid search、GraphRAG、RAPTOR、Agentic RAG，各种技术至今层出不穷。

检索本身没有错。在 Agent 动手之前把它需要的材料放到面前，今天仍然是最有效的做法之一。在 2026 年或者更遥远的未来，只要 AI 训练技术本身不发生大革命，RAG 就不会消亡。

但 RAG 从设计上就是单向的（Agentic RAG 好一些，但能流回去的信息仍然很有限）。文档、切片、向量、召回、拼进 Prompt，信息一路向前流进上下文，然后停在那里。Agent 做完这一次任务之后，没有任何东西流回去。这次召回的三段材料里哪一段真的有用，哪一段是噪音，用户最后是接受了还是整个重写了，下次遇到同类问题该不该换一种检索方式，这些全都没有出口。

我们花了两年时间，把“怎么把知识送进去”做得极其精细，却几乎没有人做“怎么把结果拿回来”。这条不对称，我们会在主文再次提到。

如果你对 RAG 本身好奇，可以参考[《文本嵌入：从词袋模型到 Qwen3 Embedding》](/blog/2024/09/25/text-embedding-from-bow-to-qwen3/)、[《Embedding Atlas：用可视化理解 RAG 的嵌入空间》](/blog/2026/05/29/embedding-atlas-rag-embedding-visualization/)和[《AI Agent 如何从互联网获取信息》](/blog/2026/06/10/ai-agent-retrieval-tools/)；至于知识被送进去之后，如何在上下文和记忆里被组织，可以先看[《Context is All You Need》](/blog/2026/06/11/agent-context-engineering/)与[《从记忆形成到记忆治理》](/blog/2026/03/21/agent-memory-panorama/)，后面我们还会回到这个问题。

### LangGraph、LangChain 与 Workflow

Workflow 编排是 AI Agent Dev 同样绕不开的话题。这一节想聊的是，过去两年整个社区捣鼓出了什么样的 Agent Framework，以及为什么需要它。

最早的一批系统是 n8n 和 LangChain。那时候我们其实还活在过去的工作流自动化范式里：搭一条线性的流程，连接不同的外部服务，计算，然后决策。LLM 在这条流程里只是一个节点，负责那些传统脚本解决不了的部分，去写一封邮件，理解一段自然语言然后决策进入新的分支等等。

再下一步呢？LangGraph、Dify 这类用循环图代替 DAG 的系统开始出现，ReAct 让 LLM 可以在一个局部流程里反复循环，直到把问题解决。Very Powerful Tools，在 LLM 出现之前，一台循环机器不会拥有这样的能力。Loop 在最近两个月又回归了，某种意义上也是类似的原因。LLM 变强了，于是循环变得更大更强了。

作为 Workflow 编排的对照，多智能体框架也有过一阵热度。既然模型这么像人，那就让它们像人一样工作吧。MetaGPT、ChatDev、AutoGen 给 Agent 发下产品经理、架构师、程序员和测试的工牌，让它们按 SOP 开会、交付文档、互相评审。热了一阵，后面很快就没什么人用了，原因我们留在主文中。

为什么 Workflow 的开发者数量远多于多智能体？为什么企业做产品时几乎不选多智能体？在能够给出流程的场景里，一个 Runtime 的流程可以很好地吸收 LLM 本身的不确定性。把不确定性挡在系统之外，这在模型还会犯错的年代是非常好的工程思想，后来的 Agent Harness 做的也是类似的事情，只是 LLM 的不确定性少了，我们加的约束自然也少了。

参考 [《从智能体的认知结构到智能体框架》](/blog/2026/03/03/cognitive-architecture-to-agent-framework/) 讨论 Workflow、Agent、Supervisor、Agent Team 与 MAS 的边界在哪里，以及 MetaGPT、AutoGen 这些框架的工程价值和抽象代价。阅读 [《给 LLM 戴上确定性枷锁的外围工程》](/blog/2026/03/20/building-agent-deterministic-constraints/) 思考把不确定性挡在系统之外应该怎么做。

### Function Calling 与 MCP

一个只能 Chat 的机器人不是我们想要的。收集信息固然是人类工作里很大的一部分，但仍旧有另一部分，是拿着这些信息去做决策并执行某些 Action。把 Next Token Prediction 变成能与外界交互的智能体，让它直接帮我们把活干了，人动嘴，AI 动手，看起来就是一个非常美好的未来。

既然有想法，那就要去做。Function Calling 率先诞生，靠简单的 JSON Schema 为模型补上了基本的外部工具调用能力：修改文件，执行代码，搜索资料。这些对人而言司空见惯的操作，开始进入 AI 的能力范围。为了统一各家互不兼容的接口，Model Context Protocol 诞生，并逐渐成为一个被广泛兼容的事实标准。随着模型能力继续进步，Agent Skills 这种把 Prompt 与 Tool 打包在一起的更轻量封装成为主流，被大量使用。在标准化接口的另一面，GUI Agent 随着 VLM 与 Reasoning 的成熟浮出水面，Manus 这类项目层出不穷，但受限于隐私与权限管理，一直难以进一步推广。

但仅仅把工具标准化、再堆上更多工具，就足够了吗？

SWE-agent 以及它一并提出的 ACI 值得我们进一步思考。同一个模型，接一套专门为 Agent 设计的接口，和直接扔给它一个裸 shell，成绩差得不像同一个系统。模型没变，变的是它能看到什么、能做什么，以及做完之后会拿到什么新的信息。MCP 僵化的 Tool Calling 机制对于 2024 年底的模型可能是恰当的，但大量的工具注入信息与一轮轮工具调用逐渐开始腐蚀上下文，自适应加载、程序化函数调用以及 Skill Scripts 的思路出现来缓解这些问题。

SWE-agent 对工业界很难说掀起了多大的改变，因为它不像 MCP 那样能变成一个协议、一个生态、一个可以立项的东西。接口质量是手艺问题，很难标准化，也很难写进汇报材料。但它是一个极高性价比的改进方向：工具不是越多越好，工具的质量对于 Agent 甚至可能比基础模型更重要。Tools 从来不是一份工具列表，而是 AI Agent 能看到、也能影响的整个世界。

这条线的技术演进可以参考[《LLM 工具使用的技术演进》](/blog/2026/03/05/llm-tool-use-evolution/)与[《MCP (Model Context Protocol)》](/blog/2026/02/16/mcp-model-context-protocol/)，Skills 为什么会在 MCP 之后又赢一次，可以参考[《从 MCP 到 Agent Skills》](/blog/2026/03/10/from-mcp-to-agent-skills/)。至于本节最后那个判断，可以参考[《AEnvironment：Agent Dev 为什么需要交互环境层？》](/blog/2026/03/16/aenvironment-everything-as-environment/)和[《Harness 到底是什么》](/blog/2026/04/04/understanding-agent-harness/)。

### 长上下文军备竞赛与 Context Engineering

聊一聊在 Prompt Engineering 之后的 Context Engineering 吧。这应该也是大部分 Agent 开发者听说过的词。当我们想让一个 AI Agent 去帮我们干更多的事，无论是在 Chat 中给出更长的输出，还是在多轮工具调用里解决问题，Context 都是我们绕不过去的问题。最早期的模型只有 1k 到 8k 的 Context，很难想象一个人类眼中的复杂任务能靠几千个字去描述并解决，CoT/RAG 技术的出现也加剧了上下文不足的问题。做不好复杂任务或许是因为上下文窗口太小，那我们就要尝试去解决这个问题。

如果从第一性原理出发，那么研究长上下文的第一考虑一定是去修改模型。位置编码从绝对位置换成 RoPE 这类相对位置编码，再用位置插值、NTK-aware、YaRN 之类的手段在不训练的情况下扩展上下文长度；注意力那一侧则想办法绕开序列长度的平方复杂度，考虑滑动窗口、局部与全局交替的稀疏注意力；并做 FlashAttention 这种 IO 工程层面的优化。在一系列工作和更多算力的加持下，上下文窗口从 8k 一路涨到 128k，再到今天的 1M，几乎每半年翻一倍，Context 预算逐渐变得阔绰了。

Context Engineering 当然不只是长上下文。随着我们拥有越来越长的 Context，Context Rot 的问题浮出水面：仅仅把窗口做长、往里面堆内容，并不能带来更好的效果。我们需要把上下文视作有限资源，去调度而不是去填满。Compaction、Subagent、Memory 系统应运而生，Skills 通过文件系统引入渐进式加载，以减轻提示词的输入负担，MCP 从全量的工具描述注入改为按需的局部注入，用文件系统实现分层 Memory 系统也成为现实，Context 开始被动态地管理。从 2025 年底到 2026 年上半年，Context Engineering 的进步是这场智能体大跃进的基础，人们也认真地开始想：知识究竟应该住在参数里、上下文里，还是外部，又应该如何被正确调度。

这一节里那些训练细节，可以先看[《自注意力机制与 Transformer 架构》](/blog/2024/11/14/self-attention-and-transformer-architecture/)与[《LLM 生命周期总览》](/blog/2024/08/15/llm-lifecycle-overview/)补一下底子，长窗口在推理侧的代价则可以参考[《为什么 Output Token 更贵：从 KV Cache 到 Agent 成本工程》](/blog/2026/04/26/output-token-pricing-kv-cache-agent-cost/)。窗口变长之后真正的那套工程，在[《Context is All You Need：智能体的上下文工程》](/blog/2026/06/11/agent-context-engineering/)里写得比这里细得多；Memory 那一侧，可以看[《从记忆形成到记忆治理：Agent Memory 的全景图》](/blog/2026/03/21/agent-memory-panorama/)与[《Agent Memory 与 Runtime 技术盘点》](/blog/2026/06/07/agent-runtime-teardown/)，这两篇也正是 RAG 那节回来的地方。至于渐进式加载为什么算 Context Engineering 而不只是工具协议的改良，可以回看[《从 MCP 到 Agent Skills》](/blog/2026/03/10/from-mcp-to-agent-skills/)。

### 回到模型本身与那些新词

这一章快到结尾啦，我们来看看 2026 年这半年我们都在搞一些什么。

先来看看模型。LLM 从 2023 年至今的几年时间里，关于要不要在某些领域做进一步的训练（一般是 Mid-Training 去补充知识），其实经历了大起大落。在最早期，还没有前面聊到的那么多东西的时候，Training 是增强能力的唯一选择，做领域适配的 QA 和做 Training 几乎等价。这也是 LLM 之前的深度学习时代留下的习惯，因为那时候模型的泛化能力难以被信任。

随着 LLM 泛化能力的提升，人们的观点开始逐渐变化。我们更少地在做工程问题的时候提到 SFT、LoRA、RL，大量训练工作重新前置，成为基础模型的一部分。The Second Half 则提出我们进入了新时代，训练 Infra 逐渐成熟，训练从算法问题变成了闭环工程问题。模型迭代变快，也变得更加适合用户的使用，我们好像确实可以更多地相信基础模型，做适合自己的评估，而不是去先考虑训练。

2026 年的 Agent 大跃进是 Agent 本身的技术大跃进吗？我的回答是没有，那更多是焦虑和自上而下的驱动。但 Agent 的能力其实真的在飞速进步，只是这些进步不来自 Agent Harness，而是更多地源自模型。其实大量的 Harness 都是补丁，而不是必须的。你在三个月前做的 Harness 工作被下个版本的模型免费移除，才是 Agent Dev 的常态。

我们从 Workflow 编排转而去开发 Claw，ReAct 循环从诞生就没怎么变化过，变的是循环体里那个模型，在新模型的加持下，极简的 Loop 就能打赢 Workflow。

Multiple Agents 开始退场，因为模型 + 上下文工程让单个 Agent 能一口气做完，分工的收益开始小于通信复杂度带来的损耗与开销。人们开始去尝试一下更为动态的具有一定约束的自主智能体，而不是完全的分工。

Loop 与 Goal 模式出现，人们希望越来越少地介入，让一个具备极强自主能力的 Agent 去实现那个最终目标。这是一个更大的 ReAct，也是去年的 Ralph，Loop 从来不是新玩法。

Harness 是全新的概念吗？其实曾经的 Workflow 也是一种 Harness，只是大家现在不开发 Workflow 了，更关注那个自主的 Agent，总需要一个新的词来强调我们在做什么，我们做的和之前不一样。

2026 上半年不是新东西涌现的半年，是旧东西终于露出形状、被命名的半年。模型吃掉框架之后，每家公司不同的抽象模式找到了最大公约数。这半年里大量新人的涌入，也需要我们找一些新名字，方便讨论和汇报。Marketing 本身就不是纯粹的问题，是人与技术的协同。

关于训练那条线，Mid-Training 到底在补什么可以参考[《LLM 推理与训练的本质》](/blog/2026/02/23/the-essence-of-llm-training-and-reasoning/)，它为什么从算法问题变成闭环工程问题可以参考[《Agentic RL：为什么训练闭环比训练算法更重要》](/blog/2026/03/21/from-sft-to-agentic-rl-training-loop/)，而 The Second Half 这个说法本身，可以参考[《从 RL Agent 到 LLM Agent》](/blog/2026/03/09/from-rl-agent-to-language-agent-v2/)。至于模型如何反过来吃掉框架，[《Model Is Good Enough》](/blog/2026/03/18/model-is-good-enough/)讲的就是这件事，而[《Claude Code or Codex》](/blog/2026/04/10/how-to-choose-the-right-model-for-developers/)是它在产品层面的一个具体切片：同样的极简 Loop，换一个模型，体验就完全不同。[《Harness 到底是什么》](/blog/2026/04/04/understanding-agent-harness/)整篇都在论证这个词罩住的是一堆旧问题；多智能体这笔账的完整版本，则在[《从智能体的认知结构到智能体框架》](/blog/2026/03/03/cognitive-architecture-to-agent-framework/)里。

### Benchmark 与 Evaluation

来时路的最后一节，是一个很少人关注的地方。

前面每一层都有一个响亮的名字，有工具，有生态，有可以拿去立项的东西。这一层的名字是 Evals，听起来是交付前要走的一道流程，所以它长期是整条链路上最不被重视的一环。大跃进真正的教训是指标替代了目标，而 Evals 恰恰是唯一一件在正面回答目标的事。

半路入行的开发者尤其容易跳过这一层。教程里 Evals 永远排在最后一章，它不产生 Demo 和能看到的收益，做完了也没有什么可以拿去汇报的东西。但跳过它的代价会在后面一次性还回来。

Evals 其实和很多问题相关。要不要换一个模型，Prompt 改完是变好了还是变坏了，上下文压缩掉的那部分到底重不重要，工具描述改一个词有没有用。你凭什么说现在这一版比上一版好。没有 Evals，这些判断全部只能靠感觉，而感觉在一个概率系统上非常不可靠，你会长期在原地打转，还以为自己在迭代。

所以这一层不起眼，也确实没有一个听起来很重要的概念，但它是 Agent Dev 里的一部分。如果可以的话，别跳过 Evals，起码在不忙的时候考虑一下。

关于 Evals 应该怎么做，推荐直接读 Anthropic 的那篇[《Demystifying evals for AI agents》](/blog/2026/07/07/demystifying-evals-for-ai-agents/)，从基础概念到从零搭一套 eval suite 讲得相当完整；至于工程上如何抉择评些什么、如何迭代，我在[《当我们构建 AI Agent 时，我们究竟在解决什么问题？》](/blog/2026/07/28/what-problems-are-we-really-solving-when-building-ai-agents/)的 Evals 一节里聊了自己的做法。

## 写在最后

七层猜测走完了。每一层在当时都是对的，也都长出过一批工具、一批名词和一批项目；其中一小部分沉淀成了今天的基础设施，剩下的迅速贬值。

回看这条路是为了回答那三个问题：当你在构建一个 AI Agent 的时候，你究竟在做什么？你得先解决 AI Agent 自己的什么问题？一个 AI Agent 项目为什么成功，又为什么失败？我的答案在[《当我们构建 AI Agent 时，我们究竟在解决什么问题？》](/blog/2026/07/28/what-problems-are-we-really-solving-when-building-ai-agents/)里。现在是回去的时候了。
