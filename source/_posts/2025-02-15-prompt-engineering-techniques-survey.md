---
title: "提示工程综述：从思维链到自我优化的技术全景"
title_en: "Prompt Engineering Survey: From Chain-of-Thought to Self-Optimization"
date: 2025-02-15 20:00:00 +0800
categories: ["Foundation Models", "Model Mechanics"]
tags: ["Prompt Engineering", "CoT", "RAG", "Survey"]
author: Hyacehila
excerpt: "基于 2025 年综述，系统整理推理增强提示（CoT/ToT/GoT/Self-Consistency）、幻觉缓解提示（RAG/ReAct/CoVe）和代码相关提示技术。"
excerpt_en: "Based on a 2025 survey, systematically covers reasoning-enhancing prompts (CoT/ToT/GoT/Self-Consistency), hallucination-reduction prompts (RAG/ReAct/CoVe), and code-related prompting techniques."
mathjax: true
hidden: true
permalink: '/blog/2025/02/15/prompt-engineering-techniques-survey/'
---

## A Systematic Survey of Prompt Engineering in Large Language Models

本文是关于提示词工程（Prompt Engineering）的综述，发表于 2025 年 5 月，内容本身并不复杂。关于提示词工程的基础概念，可参考 [提示学习与上下文学习](/blog/2024/09/20/llm-prompting-and-in-context-learning/) 中的相关讨论。

### New Tasks Without Extensive Training

这里主要介绍模型如何通过提示词工程激活预训练能力，以及利用提示词中的输入输出对。下面两类方法在 [提示学习与上下文学习](/blog/2024/09/20/llm-prompting-and-in-context-learning/) 中已有详细介绍：

* Zero-Shot Prompting
* Few-Shot Prompting

### Reasoning and Logic

激活模型的思考与推理能力主要依赖 CoT 方法，基本方法包括（同样在 [提示学习与上下文学习](/blog/2024/09/20/llm-prompting-and-in-context-learning/) 中已有介绍）：

* 人工编写 CoT
* Auto-CoT（仅提示模型进行思考）

下面逐个介绍更复杂的提示技术。

#### Self-Consistency

**Self-Consistency（自我一致性）技术**处理贪婪解码策略的问题：复杂问题往往有多条路径可以到达答案，而普通输出只会给出其中一条。同时，它可以缓解 CoT 中"一步错步步错"的问题（该问题也可通过训练策略处理）。

**Self-Consistency 是一种提示词策略，或者说解码策略。**我们对同一个提示词多次推理，获得多个结果与多条 CoT，最后再从多个推理结果中以投票方式提取最终答案。

Tips：

* Self-Consistency 是上层技术，可与多种提示词策略结合
* 一般需要调高温度来获取更多样的推理路径
* 最终结果由投票产生：可由 LLM 投票，也可转化为精准结果后投票
* 至少需要 5 条以上路径，意味着 5 倍的推理开销
* 建议增加置信度阈值，路径多样性过强时人工核查
* 对小模型（难以输出多样性）可人工增加提示词指定思考角度

#### Logical CoT (LogiCoT) Prompting

Logical Chain-of-Thought (LogiCoT) Prompting 是一种融合**符号逻辑规则**的提示工程技术。它先让语言模型将问题转化为符号逻辑，再交给模型推导，推导过程中用逻辑检查器验证正确性，确认无误后再转化为自然语言输出。

LogiCoT **侧重于可验证推理**，适合高严谨性领域。对于一般化问题，很难转换为逻辑语言，也难以取得好效果，**不是一个足够泛用的提示词思路。**

同类研究还有 Logic-of-thought Prompting。

#### Chain-of-Symbol (CoS) Prompting

Chain-of-Symbol (CoS) Prompting 面向**空间推理与规划任务**，核心思想是通过**符号化表示**替代自然语言来描述中间推理步骤，提升 LLM 在复杂空间环境理解中的性能。

环境中的**空间关系**（方位、距离）被抽象为**符号表达式**，例如：

- 自然语言描述："银行在超市的东侧 200 米处，学校在银行北侧 100 米处"
- CoS 符号表示：`[Bank → East(Supermarket, 200m)]`, `[School → North(Bank, 100m)]`

纯 Prompting 的 CoS 策略先进行自然语言推理，再提取信息转化为预定义的空间符号，最后输出符号化的逻辑链条。它希望利用符号结构减少理解偏差与词元消耗。

CoS 专为空间规划任务设计，在导航、机器人操作等具身智能领域有一定应用潜力。

#### Tree-of-Thoughts And Graph-of-Thought

两者的作者相同，且有一定关联性。

ToT 是一种模拟人类深思熟虑、探索多种可能性的提示框架。它允许 LLM 在解决问题的每一步生成多个"思想"（thoughts）或中间步骤，并对这些思想进行评估，从而构建树状推理结构。

ToT 的实现包含以下几个关键步骤：

1. 问题拆解：把复杂问题拆为可控步骤
2. 生成思想：针对每个小步骤，让模型生成多个可行的推理方案
3. 状态评估：评估不同方案的价值
4. 最终方案搜索：通过搜索算法在整个 ToT 上搜索，找到较优方案

GoT 是 ToT 的进一步扩展，将语言模型的思想表示为一个图（Graph），其中节点（vertices）代表思想单元，边（edges）表示思想之间的依赖关系。

GoT 框架定义了一系列操作来构建和修改思想图，包括生成、聚合、改进等步骤，允许对单个问题进行更细致的讨论。

**GoT 和 CoT 是具备泛用性的解码框架，可以提升模型的综合能力。**

#### System 2 Attention (S2A) Prompting

相比前面的解码策略，System 2 Attention (S2A) Prompting 侧重于处理上下文信息。它希望通过 LLM 重新生成输入上下文，让模型有选择地关注相关内容。

S2A 采用两步流程：首先用上下文再生（context regeneration）提炼输入信息，然后基于提炼后的上下文生成回应——这提升了注意力的精准度和回应质量。

S2A 是具备泛用性的方法，旨在提升综合能力。

#### Thread of Thought (ThoT) Prompting

Thread of Thought (ThoT) 和 System 2 Attention 在思想上有近似之处，都用于处理混乱的上下文。但后者侧重"混乱"，前者侧重"长"。

ThoT 的灵感来自人类认知：它将冗长上下文系统性地分解为易于管理的小片段，并进行增量式分析。

该方法采用两阶段处理：LLM 先对每个片段总结和审查，再将信息提炼后形成最终回应。

ThoT 是具备泛用性的方法，旨在提升综合能力。

#### Chain-of-Table Prompting

如方法名所示，Chain-of-Table 用于处理表格问题。它通过动态生成并执行常见的 SQL 或 DataFrame 操作，在表格上逐步进行表格化推理。这个过程是迭代性的，使 LLM 能够通过逻辑上可视化的推理链条进行预测。

同类研究还有图相关任务的 End-to-End DAG-Path (EEDP) Prompting。

#### Self-Refine Prompting

Self-Refine 通过自我生成的反馈来迭代精炼输出，模拟人类的修改和完善过程。它循环执行"生成输出→自我批判→自我修正→再输出"，用这种模拟反复思考的方式加强推理能力。

Self-Refine Prompting 是具备泛用性的思考框架。

#### Code Prompting

代码预训练能增强 LLM 的推理能力，但其背后机制尚不明确。研究者希望探究将自然语言问题重构为代码形式，是否能激发模型的条件推理能力。

该技术**将自然语言任务重构为结构化的代码形式**，从而可以直接提示那些能处理文本+代码的 LLM。

#### Self-Harmonized Chain-of-Thought (ECHO)

Auto-CoT 这类自动生成示例的方法，其生成质量堪忧，可能包含错误推理过程和过多无关示例。

ECHO 将原始问题分组，从每个聚类中选取代表性问题，用 Zero-Shot-CoT 生成推理过程，最后通过动态提示机制迭代优化这些推理过程、使其模式对齐——以此获得质量更好的 Auto-CoT。

#### Instance-adaptive Prompting (IAP)

这是一个由显著性驱动的框架，旨在为每个具体实例动态定制提示词。通过分析注意力层的信息流，研究者发现有效推理与特定的信息流动模式相关。IAP 通过两种自适应策略优化推理：

- **IAP-ss（顺序替换）**：迭代测试提示词以满足显著性阈值，提升效率
- **IAP-mv（多数投票）**：聚合多个提示词的显著性分数来确定共识答案，优先考虑鲁棒性

#### Layer-of-Thoughts (LoT) Prompting

Layer-of-Thoughts 提示引入分层框架，利用约束层次结构构建推理过程，提高检索准确度和可解释性。

在法律文档检索中，LoT 将推理分为"层思想"（概念阶段）和"选项思想"（部分解决方案），并应用顺序约束迭代筛选和优化候选答案。

#### Narrative-of-Thought (NoT) Prompting

时间推理对 LLM 来说是一个重大挑战。NoT 通过将事件封装在 Python 类中处理，配合 NoT 提示模板和少量叙事示例，提升了叙事的结构连续性。

#### Buffer of Thoughts (BoT) Prompting

BoT 通过引入"元缓冲区"（meta-buffer）来解决单查询方法（依赖手动示例）和多查询方法（计算效率低）的局限。这个元缓冲区可以从不同任务中提炼出"思想模板"，并由动态的缓冲区管理器在新问题解决时持续优化模板。BoT 能够检索并自适应地实例化特定任务的思想模板，模拟人类的类比推理，从而无需手动设计提示词或进行递归探索。

#### Contrastive Denoising with Noisy Chain-of-Thought (CD-CoT)

CoT 提示中的"带噪推理过程"（noisy rationales）——即不相关或错误的中间推理步骤——会降低 LLM 性能。

"带噪思想链的对比去噪"（CD-CoT）通过将带噪推理过程与干净推理过程进行对比来缓解这个问题。它会对有缺陷的示例进行改写，选择最优推理路径，并对最一致的答案投票。

#### Chain of Draft (CoD) Prompting

Chain of Draft (CoD) 是一种旨在提高复杂推理任务效率的新型提示策略。与强调详细分步推理的 CoT 不同，CoD 每一步都生成简洁、信息密集的输出——这类似于人类解决问题时只记录关键要点的做法。通过限制每个推理步骤的用词量，CoD 在不牺牲准确性的前提下减少了延迟和词元消耗。

### Reduce Hallucination

#### Retrieval Augmented Generation (RAG)

减少幻觉、增强模型可信度的最佳技术是 Retrieval Augmented Generation (RAG)，它已经成为一个值得单独研究的学科分支。参考 RAG（检索增强生成）。

#### ReAct Prompting

ReAct 使 LLM 能够同时生成推理轨迹和特定于任务的行动。这种推理与行动交错进行的过程增强了两者的协同，通过行动获得的信息可以有效减少幻觉。

ReAct 通过与简单的维基百科 API 交互，解决问答和事实核查任务中的幻觉与错误传播问题，生成更具可解释性的任务解决路径。

ReAct 已经成为一个非常重要的思路，被广泛用于当前模型：模型思考→检索外部信息→用这些数据增强回答准确性。

#### Chain-of-Verification (CoVe) Prompting

CoVe 通过审慎的多步骤验证过程，使 LLM 即便面对矛盾信息也能增强逻辑推理能力并减少错误。其思想类似于前面介绍的多种推理方法：让模型自行检查输出，再进一步获得更好的结果。

#### Chain-of-Note (CoN) Prompting

CoN 是对 RAG 技术的改进，旨在提高 RALM 在处理嘈杂、不相关文档时的鲁棒性，并准确处理未知情况。笔记链（CoN）系统地评估文档相关性，强调关键和可靠信息以过滤无关内容，从而产生更精确、与上下文更相关的回应。

#### Chain-of-Knowledge (CoK) Prompting

受人类解决问题方式启发，知识链（CoK）将复杂任务系统性地分解为协调良好的步骤。该过程始于全面的推理准备阶段——建立上下文并框定问题；随后进入动态知识适应阶段，从多种来源（内部知识库、外部数据库和给定提示词）细致地收集证据。

### CODE

#### Scratchpad Prompting

着眼于任务设计而非模型修改，允许模型在提供最终答案之前生成任意序列的中间字符（即推理过程）。

#### Program of Thoughts (PoT) Prompting

提倡使用外部语言解释器来执行计算步骤。PoT 使 Codex 这类模型能够通过可执行的 Python 程序来表达推理过程。

#### Structured Chain-of-Thought (SCoT) Prompting

这是专门为代码生成量身定制的提示技术。通过将程序结构（顺序、分支和循环结构）融入推理步骤，SCoT 提示增强了 LLM 生成结构化源代码的性能。

#### Chain-of-Code (CoC) Prompting

CoC 鼓励 LLM 将语义子任务格式化为灵活的伪代码，并允许一个解释器（LMulator）捕捉和模拟未定义的行为。这种"用代码思考"的方法扩展了 LLM 正确回答推理问题的能力。

### ELSE

#### Active-Prompting

与依赖固定人工标注示例的现有 CoT 方法不同，Active-Prompting 引入了一种机制来确定哪些问题最值得标注。该方法从基于不确定性的主动学习中获取灵感，利用多种指标描述不确定性，选择最不确定的问题进行标注。

#### Automatic Prompt Engineer (APE)

APE 摆脱了静态手工设计提示的限制，通过动态生成并选择最有影响力的提示词来服务特定任务。该方法分析用户输入，构建候选指令，利用强化学习选择最优提示词，并根据不同上下文即时调整。

#### Automatic Reasoning and Tool-use (ART)

ART 通过结构化程序自动完成推理步骤，无需费力手工制作。其动态工具集成确保顺畅协作——能在需要时暂停生成以整合外部工具输出，再无缝恢复流程。通过集成外部工具获取专业知识和计算能力，ART 为 LLM 带来了更大的通用性。

#### Contrastive Chain-of-Thought (CCoT) Prompting

对比思想链提示（CCoT）在原始提示旁边同时提供有效和无效的推理示例，这种双重视角促使 LLM 进行逐步推理。

#### Emotion Prompting

从心理学研究中获取灵感，将 11 种情感刺激句附加到提示词中，以增强 LLM 的情商。

#### Optimization by Prompting (OPRO)

利用 LLM 作为优化器的新颖方法。与传统方法不同，OPRO 使用自然语言提示词，根据问题描述迭代生成解决方案，从而快速适应不同任务并定制优化过程。

#### Rephrase and Respond (RaR) Prompting

RaR 允许 LLM 在单个提示词中重新表述和扩展问题，提高理解力和回应准确性。包含改述和回应两个 LLM 的两步 RaR 变体，在各种任务上实现了显著的性能提升。

#### Take a Step Back Prompting

这种创新方法使模型能够进行抽象，从具体实例中提取高层概念和基本原则。该方法包含两步过程：抽象（Abstraction）和推理（Reasoning）。
