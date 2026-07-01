---
title: LLM 工具使用的技术演进：从 Toolformer 到 ToolLLM
title_en: "The Evolution of LLM Tool Use: From Toolformer to ToolLLM"
date: 2026-03-05 13:20:00 +0800
categories: ["Agent Infrastructure"]
tags: [Agents, Tool Use]
author: Hyacehila
excerpt: 梳理 LLM Tool Use 领域的研究脉络：Toolformer 的自监督学习、Gorilla 的微调+检索、Tulip Agent 的递归分解、ToolLLM 的大规模框架，以及这条路线如何转向协议、运行时和工程实践。
excerpt_en: "A survey of LLM tool-use research, from Toolformer's self-supervision and Gorilla's fine-tuning plus retrieval to Tulip Agent and ToolLLM."
permalink: '/blog/2026/03/05/llm-tool-use-evolution/'
---

## 为什么关注工具使用

大语言模型在不少经典 NLP 领域表现突出，但在算术问题和事实回答等场景上仍不稳定——模型无法及时更新内部参数，也存在幻觉问题。让 LLM 使用外部工具，可以使其访问实时、准确的知识库，并完成计算任务。

关于 Tool Use，**最值得工程界关注的可能是 MCP（Model Context Protocol）协议**。MCP 提供了一个标准化的工具定义与描述通信协议，让模型能够以统一的方式发现和调用工具。关于 MCP 的详细介绍，可以参考本博客此前的文章。了解 Tool Use 的相关论文，更多是为了理解其中的发展脉络与技术思路的演变。

## Toolformer：自监督学习工具调用

> Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools*, NeurIPS 2023.

Toolformer 考虑对原始模型进行微调，借助微调来增强其在 Tool Use 上的能力，同时学习给出的工具。其微调用数据集使用了自监督的方式来生成，并且在单次解码中一次性完成对工具的使用。

- **关于自监督生成的微调数据集**：使用上下文学习的方式让模型主动请求使用工具。当模型使用工具后，通过对比使用工具/不使用工具的解码损失来衡量工具使用是否有价值，然后将有价值的结果组织成微调用数据集
- **关于解码中的工具调用**：执行常规解码，当解码得到请求 API 调用的 token 符号时，请求调用 API，并将结果返回到解码序列中，然后继续解码流程

**Toolformer 是对引入外部工具的一种早期尝试，但由于没有能够结合现在模型的推理能力而依赖解码上的修改，效果有限。同时由于单次解码的策略，Toolformer 无法链式地利用工具，无法满足多智能体的工具调用需求。其最值得学习的部分可能是自监督学习的策略。**

## Gorilla：微调与检索的结合

> Patil et al., *Gorilla: Large Language Model Connected with Massive APIs*, 2023.

先前的大部分工作在将工具集成到 LLM 时，考虑的是一组小型、文档完善的 API，这些 API 可以轻松注入到提示中。然而，支持一个超大型的、具有重叠功能的 API 仓库需要新的技术来解决。

本文考虑了使用 Self-Instruct Fine-Tuning（对原始模型进行参数微调）以及 Retrieval（通过检索手段来获得上下文提示，类似 MCP 但没有全部注入 Prompt）来加强模型对大规模 API 库的正确调用能力，以规避提示词注入的方法。

将直接 Fine-Tune 和 Retrieval 结合，是本文的主要改进，即 **Retrieval-Aware Training**。实验证明：
- 有好的检索器时，Retrieval-Aware Training 比单纯的微调更好
- Retrieval-Aware Training 可以适应 API 文档的快速更改
- 涉及 API 的调用约束（需权衡多种需求来决定选择哪个 API）时，所有模型的性能都产生了显著下降

**考虑使用微调以及简单的检索来实现工具调用。由于不涉及多轮推理和调用，对指令检索后得到的相关 API 会作为上下文交给模型处理。**

## Tulip Agent：递归任务分解与语义检索

> Ruis et al., *Tulip Agent: Enabling LLM-Based Agents to Solve Tasks Using Large Tool Libraries*, 2024.

Tulip Agent 不会把所有可用工具的描述都编码进系统提示（这会占用模型的上下文窗口），也不会嵌入整个提示来检索工具。它会把任务递归分解成多个子任务，再让每个子任务单独做语义级别的向量数据库检索，匹配合适的工具，并允许动态管理工具。

相比于前面的技术：
1. **放弃了将全部工具描述以提示词的形式注入 LLM**，以规避过长上下文的问题
2. 放弃将提示一次性嵌入寻找工具，改为先对**任务进行推理规划，再嵌入、检索，并继续推理思考**
3. 使用**向量数据库、Embedding 以及类似 RAG 的技术**进行工具的检索
4. 允许工具的动态管理

**本技术中，检索器在每个推理规划的步骤结束后被激活来检索相关 API，然后以上下文的形式提供给模型参考应该调用哪些 API。**

## ToolLLM：通用工具使用框架

> Qin et al., *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs*, ICLR 2024.

ToolLLM 引入了一个涵盖数据构建、模型训练和评估的通用工具使用框架，工程设计较完整。作为开源项目，它在 Tool Use 方向获得了大量关注，也是 LLM Agents 工具使用方向的代表项目之一。

闭源模型已经具备较强的工具调用能力，但开源社区现有研究存在不足：
1. API 数量有限，可能覆盖范围太小、多样性不足
2. 局限于单工具调用，经常假设用户手动给定理想的 API 集
3. 规划和推理能力不足，包括 CoT 推理或 ReAct 的推理与行动

### 核心组件

**API 收集**：从 RapidAPI 平台收集了 16,464 个 REST API，涵盖 49 个不同类别，包含详细文档供 LLM 学习。

**指令生成**：从整个 API 集合中采样，提示 ChatGPT 生成多样化的指令，涉及单工具和多工具场景（Self-Instruct）。

**解决方案路径标注**：每个解决方案路径可能包含多轮模型推理和实时 API 调用。为此开发了基于深度优先搜索的决策树（DFSDT）来增强 LLM 的计划和推理能力。

**评估（ToolEval）**：开发了自动评估器 ToolEval 来评估 LLM 的工具使用能力。

**微调（ToolLLaMA）**：通过在 ToolBench 上微调 LLaMA 得到指令生成模型。

### 关键发现

- ToolLLaMA 展示了处理单工具和多工具指令的能力
- ToolLLaMA 对未见过的 API 表现出强大的泛化能力，只需 API 文档即可有效适应新 API
- DFSDT 通过考虑多个推理轨迹扩展了搜索空间，比 ReAct 取得更好的性能——**这是对推理策略的研究**

**在本研究中，检索只对用户指令进行一次，搜索得到相关 API 会在多轮推理中交给模型作为上下文参考，方便其思考推理策略。**

**本文给出的 DFSDT 是一种决策推理策略，和 ReAct 联系紧密。本文关注的不只是工具调用性能，也包括带有工具调用的推理解题流程。**

## 结语

回顾 Tool Use 的研究脉络，从 Toolformer 的自监督微调、Gorilla 的检索增强训练、Tulip Agent 的递归分解检索，到 ToolLLM 的大规模系统工程，这条研究路线始终围绕一个问题：**如何让模型在海量工具中找到正确的那一个，并正确地调用它。**

不过，**随着基础模型能力提升，以及 MCP（Model Context Protocol）这类协议出现，Tool Calling 作为独立训练方向的边际收益正在下降。** 当前前沿模型已经具备较强的函数调用能力，很多场景不再需要额外微调或复杂检索管线来“教会”模型用工具。MCP 处理的是另一层问题：工具如何描述、发现、调用，以及如何在客户端和 Server 之间形成稳定接口。

过去 Tool Use 研究里的很多问题，如上下文膨胀、工具检索、多轮调用，现在更多转向协议、运行时和 Agent 工程来处理。**后续值得关注的重点，可能不再只是训练一个更会用工具的模型，而是如何把工具、权限、上下文和执行过程做成稳定系统。** 本文不再更新。
