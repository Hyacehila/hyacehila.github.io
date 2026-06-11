---
layout: blog-post
title: "拆开 Agent Runtime：记忆、上下文与隔离在真实系统里如何被装配"
title_en: "Taking Apart the Agent Runtime: How Memory, Context, and Isolation Are Assembled in Real Systems"
date: 2026-06-07 17:00:00 +0800
categories: [Agent 系统]
tags: [Agents, Agent Runtime, Context Engineering]
author: Hyacehila
excerpt: "用前三篇建立的坐标系（CoALA、L1/L2/L3 三层记忆、harness 三分）去测绘真实系统：Mem0、Letta、Zep、LangGraph、OpenAI Agents SDK、Claude Code、OpenClaw 等如何把记忆、上下文与隔离装配在一起，以及它们在哪里真正分歧。"
excerpt_en: "Using the coordinate system built across the prior three posts—CoALA, the L1/L2/L3 memory layers, the harness trichotomy—to survey real systems: how Mem0, Letta, Zep, LangGraph, the OpenAI Agents SDK, Claude Code, OpenClaw, and others assemble memory, context, and isolation, and where they genuinely diverge."
featured: false
math: false
---

> **写作状态：本文为骨架草稿。** 系列前三篇已经把概念坐标系搭好，这一篇是"测绘"——需要我逐个读真实系统的代码和文档后再填实。先把结构、定位和待补的格子立在这里，内容随后逐步补全。下文标注 `TODO` 的小节均为占位。

# 拆开 Agent Runtime：记忆、上下文与隔离在真实系统里如何被装配

如果把 Mem0、Letta、Zep、LangGraph、OpenAI Agents SDK、Claude Code、OpenClaw、Hermes、PI 这些系统排成一列，很容易写成又一篇"工具百科"：每个产品一小节，讲功能、讲卖点。但这件事，我在[《从智能体的认知结构到智能体框架》](/blog/2026/03/03/cognitive-architecture-to-agent-framework/)里已经对框架做过一轮（LangGraph / Dify / Coze / AutoGen / OpenAI SDK 的抽象对比），在[《给 LLM 戴上确定性枷锁的外围工程》](/blog/2026/03/20/building-agent-deterministic-constraints/)和[《Harness 到底是什么》](/blog/2026/04/04/understanding-agent-harness/)里又对 Claude Code 的 harness 做过一轮拆解。**再做一次逐产品横评，只会是前几篇的合订重印。**

所以这篇文章换一个动作：不是横评，而是**测绘（reconciliation）**。前三篇加上这篇之前的两篇，已经留下了一整套坐标系——

- **CoALA 的记忆层次**（工作记忆 / 情景 / 语义 / 程序记忆），来自认知结构篇；
- **L1/L2/L3 三层成本框架**（全上下文 / 外部记忆 / 参数记忆），来自[《Agent Memory 全景图》](/blog/2026/03/21/agent-memory-panorama/)；
- **运行时上下文手法**（compaction、reset、subagent、checkpoint、外部化、recitation），来自[《Context is All You Need》](/blog/2026/06/11/agent-context-engineering/)；
- **harness 三分**（工程 / 产品 / 用户友好外壳），来自 harness 篇。

这篇文章要做的，就是**把这套坐标系盖到真实系统上**，回答两个前几篇没回答的问题：第一，当 L1 + L2 + L3 + 工具 + 隔离 + 治理被**装配进一个具体系统**时，接缝长什么样、谁和谁打架？第二，对照坐标系，**哪些格子是理论说该有、而某个产品没做的**？

## 一个前提：不要头对头比，要看 stack 怎么拼

测绘之前，先立一个会贯穿全文的判断：**这些系统不在同一层，硬把它们排成一张排行榜是错的。**

- **记忆层产品**（Mem0、Letta、Zep）：解决"记什么、怎么存、怎么召回"，本身不编排工具、不跑 agent loop。
- **框架 / runtime**（LangGraph、OpenAI Agents SDK）：解决"怎么编排"，记忆只是其中一个子系统。
- **端到端产品**（Claude Code、OpenClaw、Hermes、PI）：把上面两层 + 工具 + UI 装配成一个用户能直接用的东西。

所以正确的问法不是"Mem0 和 LangGraph 谁强"，而是"**Mem0 + LangGraph 这个 stack 怎么拼、拼起来有没有重复或冲突**"。本文的对比维度因此是**装配关系**，而不是功能多少。

先用一张表把三层压住（**TODO：表格内容待逐项核实后填实**）：

| 层 | 代表系统 | 核心抽象 | 在坐标系里的位置 | 典型缺口 |
| --- | --- | --- | --- | --- |
| 记忆层产品 | Mem0 / Letta / Zep | 抽取-更新-检索 / memory blocks / 时间图 | L2 外部记忆 | TODO |
| 框架 / runtime | LangGraph / OpenAI Agents SDK | state+checkpoint+store / sessions+handoffs | L1 调度 + 编排 | TODO |
| 端到端产品 | Claude Code / OpenClaw / Hermes / PI | CLAUDE.md+subagent+/compact / ... | L1+L2+工具+治理 | TODO |

## 记忆层产品：L2 的三种不同答案

> **TODO（读文档/代码后填实）。** 这一节按"对账"展开，每个产品对应回 L2 框架的一个侧面：

- **Mem0**：抽取 → 更新 → 检索的管线 + 自动去重（"最新事实胜出"）；graph memory 为可选项。对账点：它对应 L2 写入侧的"对齐/版本化"，但冲突消解做得有多深？
- **Letta（MemGPT）**：memory blocks + 自编辑记忆 + LLM-OS 隐喻（把上下文当 OS 管理的内存分页）。对账点：它把 MemGPT 的"分页交换"产品化，对应 Context 篇的 compaction/reset，但它的边界在哪？
- **Zep（Graphiti）**：时间感知知识图谱 + fact-invalidation（旧事实标记失效但保留历史）。对账点：它正面回应了 Memory 篇反复强调的"时间有效性"难题——这是它最独特的格子。

## 框架 / runtime：L1 调度与编排怎么被抽象

> **TODO。** 这一节对账"框架怎么把 L1 调度做成 API"：

- **LangGraph**：state + checkpointer + store（短期线程态 vs 跨线程长期记忆）。对账点：checkpoint/fork 在这里是一等公民，正好对应 Context 篇的状态管理原语，但它把 compaction 留给了开发者。
- **OpenAI Agents SDK**：sessions + handoffs + 模型侧 compaction。对账点：handoff 对应 Context 篇的 reset/handoff artifact，但 compaction 绑定在 Responses API 上。

## 端到端产品：装配的接缝在哪里

> **TODO。** 这一节是全文重点——真实产品怎么把前两层 + 工具 + 治理拼起来：

- **Claude Code**：CLAUDE.md（人写的持久上下文）+ auto-memory（模型写的）+ subagent 隔离 + /compact + checkpoint/fork。对照坐标系，它几乎每个 L1 格子都填了，但 L2 长期跨会话存储相对薄。
- **OpenClaw**：TODO —— 面向异构个人环境（聊天/语音/设备/自动化）的装配，记忆与权限的接缝。
- **Hermes**：TODO（作者后续读代码补，先占位）。
- **PI**：TODO（作者后续读代码补，先占位）。Harness-as-a-Service 的形态。

## 它们在哪里真正分歧

> **TODO。** 把对账结果收成几条真实分歧轴：

- **记忆更新策略**：手动编辑（Letta）vs 自动去重（Mem0）vs 时间失效（Zep）vs 确定性快照（LangGraph）——"谁/什么来更新记忆，为什么"的根本分歧。
- **长上下文换出策略**：DB 分页 / 压缩 / 时间剪枝 / checkpoint / 模型侧 compaction，各自的代价。
- **多智能体取舍**：这里承接 Context 篇留下的 Anthropic-vs-Cognition 辩论，但落在"产品实际怎么选"上。

## 对比维度表（骨架）

> **TODO：逐格填实。** 行=系统，列=坐标系维度。

| 系统 | L1 调度 | L2 记忆层 | L3 参数记忆 | 隔离模型 | 治理 / HITL | 开放/闭源 |
| --- | --- | --- | --- | --- | --- | --- |
| Mem0 | — | TODO | — | — | TODO | 开源+托管 |
| Letta | TODO | TODO | — | TODO | TODO | 开源+托管 |
| Zep | — | TODO | — | — | TODO | 闭源 SaaS |
| LangGraph | TODO | TODO | — | TODO | TODO | 开源 |
| OpenAI Agents SDK | TODO | TODO | — | TODO | TODO | 闭源 |
| Claude Code | TODO | TODO | — | TODO | TODO | 闭源 |
| OpenClaw | TODO | TODO | — | TODO | TODO | TODO |

## 作者偏好与开放问题

> **TODO。** 读完代码后写：我自己更喜欢哪种装配方式、为什么；以及测绘暴露出的、整个领域仍然空着的格子。

## 参考资料

- [《从智能体的认知结构到智能体框架》](/blog/2026/03/03/cognitive-architecture-to-agent-framework/)
- [《从记忆形成到记忆治理：Agent Memory 的全景图》](/blog/2026/03/21/agent-memory-panorama/)
- [《Context is All You Need：智能体的上下文工程》](/blog/2026/06/11/agent-context-engineering/)
- [《给 LLM 戴上确定性枷锁的外围工程：从 Claude Code 看 Agent Harness》](/blog/2026/03/20/building-agent-deterministic-constraints/)
- [《Harness 到底是什么：从 model + harness 到工程、产品与用户友好外壳》](/blog/2026/04/04/understanding-agent-harness/)
- TODO：Mem0 / Letta / Zep / LangGraph / OpenAI Agents SDK / Claude Code 官方文档链接，读后补全
