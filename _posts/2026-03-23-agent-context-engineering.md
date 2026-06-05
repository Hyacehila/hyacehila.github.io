---
layout: blog-post
title: Context is All You Need：智能体的上下文工程
title_en: "Context Is All You Need: Context Engineering for Agents"
date: 2026-03-23 12:00:00 +0800
categories: [Agent 系统]
tags: [Agents, Context Engineering]
author: Hyacehila
excerpt: 把上下文当成有限资源来调度：从 context rot 与注意力预算出发，讨论读写路径编排、存储载体的工程实现，以及 compaction、reset、subagent、checkpoint 等在运行时缓解上下文不足的具体手法。
excerpt_en: "Treating context as a finite resource: starting from context rot and the attention budget, this post covers read/write path orchestration, the engineering of storage carriers, and concrete runtime techniques—compaction, reset, subagents, checkpointing—for coping with limited context."
featured: true
math: false
---

# Context is All You Need：智能体的上下文工程

## 引言：从地图到手册

在[《从记忆形成到记忆治理：Agent Memory 的全景图》](/blog/2026/03/21/agent-memory-panorama/)里，我把 Agent Memory 拆成了三层成本结构：**L1 全上下文**（知识直接住在当前窗口里，靠长上下文与 KV-cache 支撑）、**L2 外部记忆**（向量库、文件系统、知识图谱里的非参数记忆）、**L3 参数记忆**（编码进权重的隐式知识）。那篇文章的主角是 L2，它明确把 L1——也就是"当前推理现场如何选择、压缩、隔离和调度信息"——划成了另一条独立的线，留到这里展开。

**这篇文章就是那条线。** 如果说 Memory 全景图回答的是"长期记忆在研究上长成了什么样"，是一张**地图**；那么这篇文章换一个问题：**在一个真实的 Agent runtime 里，有限的注意力预算下，我们怎么把信息读进来、按住、再隔离出去？** 它是一本**手册**。

需要先说明一句范围。"上下文工程"（Context Engineering）这个词常常被用得很大，大到几乎等同于"关于 Agent 信息组织的一切"。本文不取这个伞义。这里的 Context Engineering 专指 **L1 这条线上的运行时调度**：它的主角是 token 预算和注意力，而不是长期记忆该怎么分类、该用什么结构永久存储——那些属于 Memory 全景图。按 CoALA 的语言（见[《从智能体的认知结构到智能体框架》](/blog/2026/03/03/cognitive-architecture-to-agent-framework/)），本文几乎全部围绕 **Working Memory** 的工程化管理：在有限窗口里，决定什么该进场、什么该留在场、什么该退场、什么该在需要时被重新召回。只要底层还是 attention 机制和有限窗口，这四个动作就一直是 Agent 工程绕不开的核心。

## 严格区分 LLM Memory 与 Agent Memory

在讨论运行时调度之前，有一个必须先澄清的混淆：**KV-Cache、RoPE、Attention 变种、长上下文架构，它们解决的是模型如何更高效地利用窗口，而不是智能体如何跨任务维护记忆。**

LLM Memory 关注的是模型架构层的能力边界，比如 Flash Attention、Ring Attention、KV-Cache 压缩等；Agent Memory 关注的是智能体在多轮决策中如何积累、检索、更新和遗忘知识。前者是底层推理基础设施，后者是上层系统设计。把两者混在一起，就很容易把 Context 工程问题误判成"把窗口做得更长"。

这条区分对本文尤其重要，因为本文谈的恰好是 L1。L1 的物理基底确实是 LLM Memory——窗口能放多少、prompt cache 命中率多高、decode 多贵，这些都由模型架构和推理基础设施决定。**但本文不展开这层成本机理**：为什么 output token 比 input 贵、prefill 与 decode 的不对称、KV-Cache 如何吃掉显存和调度槽位，这些我在[《为什么 Output Token 更贵：从 KV Cache 到 Agent 成本工程》](/blog/2026/04/26/output-token-pricing-kv-cache-agent-cost/)里已经单独讲过。这篇文章接受这层基础设施作为给定约束，只讨论**在这个约束之上，Agent 该怎么组织上下文**。换句话说：LLM Memory 决定了工作台有多大、多贵，Agent Context Engineering 决定了在这张工作台上怎么摆东西。

## 上下文是有限资源：context rot 与注意力预算

Context Engineering 的全部动机，可以收敛成一句话：**上下文是有限资源，而且它的退化比你想象的来得早。**

第一个直觉错误是"窗口够大就行"。但更大的窗口不等于更好的利用。早在 [Lost in the Middle](https://arxiv.org/abs/2307.03172) 里就有一个被反复验证的发现：模型对长上下文不同位置的信息利用并不均匀，放在开头和结尾的信息更容易被用上，夹在中间的信息则经常被忽略——即便对号称长上下文的模型也是如此。我在认知结构那篇里提到过这个现象；这里要补的是它的**量化版本**。

Chroma 的 [Context Rot](https://www.trychroma.com/research/context-rot) 报告测了 18 个模型，结论相当一致：**性能随输入长度增长而退化，而且是非均匀、常常出人意料地退化。** 几个具体数字值得记住：Gemini 系列在 ~500–750 词附近就开始出现随机输出，GPT-4 Turbo 的局部性能峰值在 ~500 词，Claude Opus 4 大约能撑到 ~2500 词才显著退化。更反直觉的是，他们发现**结构连贯的 haystack 反而不如打乱顺序的 haystack**——也就是说，让上下文"读起来更通顺"有时会损害检索性能。这说明退化不是一个简单的"装满就崩"的阈值问题，而是和内容如何呈现深度耦合。

Anthropic 在 [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) 里把这件事框成两个概念：**context rot**（随 token 增长，召回准确率下降）和 **attention budget**（注意力预算有限，每多一个 token 都在消耗它，根源是 transformer 的 n² 两两关系）。由此引出的指导原则很简洁：**找到能最大化目标达成概率的、最小的高信号 token 集合。**

这一节是后面所有手法的地基。**正因为上下文会腐化、注意力有预算，"读什么进来""怎么压缩""什么时候隔离"才不是可选优化，而是必须做的工程。** 下面分三块展开：读路径、写路径、以及运行时维护的手法。

## 读路径编排：把该在场的信息带进来

检索是 Context 的入口。真正的问题从来不是"能不能检索"，而是**用什么路径检索、检索到什么粒度、以及如何把结果安全地暴露给 Working Memory。**

需要先和 Memory 全景图划清一道分工：那篇文章列过长期记忆的四类激活路径（字符匹配、语义检索、结构化信源、图遍历），但它明确"不展开成工程 pipeline"。**这里就是展开 pipeline 的地方。** 激活路径"有哪些、对应什么记忆载体"是 Memory 篇的事；"运行时怎么编排它们、怎么把候选压进当前窗口"是这里的事。

### 混合检索，而不是单一路线

目前最实用的方案不是在 RAG 和 Grep 之间二选一，而是接受它们各自擅长不同问题，并把它们统一放进一个检索编排层：

- **Grep / 关键词匹配（BM25）**：适合已知术语、配置名、错误信息、规则条目等精确定位任务。
- **向量检索 / RAG**：适合自然语言查询、模糊需求、跨表述召回。
- **LSP / 结构化索引**：适合代码、配置、符号引用等天然带类型和层次的信息源。
- **图遍历 / 关系查询**：适合需要多跳关联、时间条件、实体关系的记忆内容。

编排层的职责，是让 Agent 先判断当前任务属于哪一种信息需求，再选择合适的检索组合，而不是默认所有问题都走同一个向量库。工程上最有价值的不是押注单一技术，而是把路由决策本身做对。

### Query Rewrite 与 Rerank

原始输入通常不是合适的查询。用户说的是任务语言，记忆库存的是摘要、事实、代码片段、规则、图节点和历史上下文，它们之间天然存在语义鸿沟。因此检索系统至少要有两步增强：

1. **Query rewrite**：把用户问题改写成更适合 memory store 的检索语句，必要时拆成多个子查询。
2. **Rerank / compression**：把候选结果重新排序，并整理成可进入 Working Memory 的最小必要片段。

如果没有这两步，所谓的"检索增强"常常只是把更多噪声搬进上下文。真正有效的检索系统，不是召回一百条候选，而是能稳定挑出那三条真正该在场的信息——这正好呼应上一节的"最小高信号 token 集合"。

### 从检索结果到 Working Memory：promotion 的实现

Memory 全景图反复强调过一个判断：长期记忆不会直接参与行动，它必须经过检索、筛选、压缩，**以足够克制的形式进入 Working Memory** 才能生效。那篇文章说的是"必须经过 Working Memory"这个原则；这里讲的是**它具体怎么经过**。

promotion 不应该把长期记忆的存储单位原样塞进当前上下文，而要经过一个最小必要暴露的工序：

- 提取和当前目标直接相关的字段，丢掉其余。
- 在需要时保留 provenance 和时间信息（这条记忆来自哪、什么时候生效）。
- 对重复、冲突、低置信度内容做折叠或标记。
- 超预算时，按优先级保留约束、计划、当前子任务和高价值事实。

**Working Memory 不是长期记忆的镜像，而是长期记忆在当前任务切面上的受限投影。** promotion 就是做这次投影的运行时动作——它的输入是"当前任务意图 + 剩余窗口预算"，这两个量都是纯粹的运行时变量，这也是为什么 promotion 的实现属于 L1 而不属于 L2。

## 存储载体的工程实现深度

接下来谈存储。这里要先立一道**深度切割**：Memory 全景图已经从研究视角说明了"为什么不同记忆倾向不同载体"——关系密集的语义知识倾向图，操作性经验和规范倾向文件，事实性记录倾向数据库。**本文不重复这套选型理由**，而是接着往下问一层：**选定了载体之后，工程上到底怎么把读写真正建出来？** 那篇文章在这层是浅的（它有意为之，因为它判断"结构本身不是目的"）；这篇文章要在这层变深。

### 向量库：chunking、索引与混合召回

把文本塞进向量库远不是"embed 一下"那么简单，几个决定召回质量的工程选择：

- **Chunking 策略**：定长切分简单但会割裂语义；按结构切分（段落、函数、章节）更稳，但需要解析器。chunk 太大则单块信号被稀释、检索精度下降，太小则丢失上下文、召回后还要拼。实践中常配合 chunk overlap 和 parent-document 回溯（命中小 chunk，但回填它所在的大块）。
- **Embedding 选择**：不同 embedding 模型在领域文本上的表现差异很大，代码、法律、医疗都可能需要专门模型；维度越高召回略好但存储和检索成本上升。
- **索引类型**：精确检索用暴力 KNN，规模一大就不可行。生产里基本是 ANN——**HNSW**（图结构，召回率高、内存占用大）或 **IVF**（倒排 + 聚类，省内存、需训练、召回略低），二者是延迟、内存、召回率之间的三角权衡。
- **稀疏 + 稠密混合**：纯向量检索对精确术语（错误码、API 名、人名）经常打不准，把 BM25 稀疏信号和稠密向量信号融合（如 RRF 倒数排名融合）几乎总是比单路好。
- **Metadata 过滤**：在向量检索前/后用 metadata（scope、type、时间、来源）做硬过滤，能大幅缩小搜索空间、避免跨域污染——这也要求写入时就把 metadata 结构化好（见下一节）。

### 图存储：抽取与运行时遍历

图的难点不在查询，在**怎么把非结构化文本变成图**，以及**图怎么随新信息增量更新**。实体关系抽取（谁、和谁、什么关系、何时）通常要靠一轮 LLM 调用，这一步的质量直接决定图的可用性；抽取错了，再漂亮的多跳遍历也是在错误的边上推理。运行时遍历则要控制跳数和扇出——多跳关联是图的长处，但不加约束的遍历会瞬间把成百上千个节点拉进候选集，反而违背"最小高信号集合"。增量更新还要处理实体消解（"老王"和"王经理"是不是同一个节点）和边的时效（旧关系是否已失效）。

### 文件系统：目录布局与导航

文件系统是被低估的记忆载体，而且它对 Agent 异常友好——Agent 天生会读写文件、跑 glob 和 grep。它的工程要点不在"存"，在"**怎么让 Agent 高效导航而不必把全部内容拉进上下文**"：

- **目录布局与命名约定**：层次化目录本身就是一种廉价索引，清晰的命名让 Agent 能靠路径猜内容，不必逐个打开。
- **glob / grep / LSP 导航**：让 Agent 用 `glob` 定位文件、`grep` 精确匹配、`LSP` 做符号跳转，按需读取，而不是预先全量加载。这正是"just-in-time 检索"的物理实现（后面手法章会再回到它）。
- **层级加载**：像 `CLAUDE.md` 这类约定，是把项目级的稳定上下文放在固定位置、会话开始时加载一部分、其余按需展开。

### Memory Object 与 Metadata

无论最终落在向量库、数据库、图还是文件系统，运行时读写的基本单位都不应该是一大坨未经治理的文本，而应该是带元数据的 **memory object**。一个 memory object 至少应该有这些字段：

- `type`：persona、conversation、experience、knowledge、rule 等。
- `scope`：用户级、会话级、项目级、全局级。
- `content`：被保留的事实、规则、经验或摘要。
- `source`：来自哪次对话、哪次工具调用、哪份外部材料。
- `timestamp`：写入时间和最后更新时间。
- `confidence`：这条记忆有多确定。
- `status`：active、stale、conflicted、deleted 等。
- `links`：它与哪些记忆重复、冲突、继承或 supersede。

Memory 全景图列过写入侧要回答的六个问题（写什么、写到哪、怎么写、如何更新、何时失效、何时写）——**那是问题定义，这里是承载这些决定的字段表。** `status` 和 `links` 承载"如何更新"的版本与冲突语义，`timestamp` 和 `confidence` 承载"何时失效"的时效与不确定性。只要这些元数据不存在，后面的版本更新、冲突消解、删除与回滚几乎都无从谈起。而且这些字段不是白存的——它们直接喂给上一节的 metadata 过滤和 promotion 的 provenance 保留，是连通读写两条路径的接缝。

## 写入：把行动经验变成可用记忆

如果读路径回答"什么该被带进来"，写入回答的就是"什么值得被留下来"。后者往往更难，因为写入错误会在未来持续污染检索结果。一个可用的写路径至少要经过四步：

1. **抽取**：从对话、工具调用、环境反馈里识别值得保留的候选信息，过滤噪声。
2. **分类**：判断它应该进入 persona、经验、语义知识还是对话记忆，决定写入哪个载体。
3. **对齐**：检查是否已有相同或相近记忆，决定新增、merge、覆盖（supersede），还是仅标记冲突。
4. **版本化**：保留更新链路（写进 `links` 和 `status`），而不是简单覆盖旧值。

比如"用户喜欢拿铁"与"用户最近开始戒咖啡"就不是简单的二选一覆盖。系统需要知道两条记忆的时间关系、适用条件和冲突状态，而不是把它们合并成一条失真的静态偏好。

**何时写**同样关键。很多系统把写入拖到会话结束再统一处理，但这通常太晚了。更合理的是按事件触发：用户明确给出稳定偏好或身份事实时、工具调用产生可迁移经验时、用户纠正系统推翻旧事实时、某个规则在短时间内反复被激活时。写入质量直接决定未来检索质量——如果写入时就没做好结构化、去重和版本关系，再强的检索也只能从一堆冗余和矛盾里徒劳筛选。

## 缓解上下文不足的工程手法

前面三节解决的是"信息怎么进出存储"。但 Context Engineering 真正困难、也最能拉开差距的部分，是在**单次任务执行过程中**持续维护工作台的可用性。当一个任务跑了几十轮、上下文逼近窗口上限时，你该怎么办？

这一节集中讨论这组手法。需要先和[《给 LLM 戴上确定性枷锁的外围工程》](/blog/2026/03/20/building-agent-deterministic-constraints/)做一个分工说明：那篇文章也讲了 subagent、checkpoint、fork、worktree，但它的角度是"**限制即可靠性**"——把这些当作收窄失败半径的可靠性原语。**这里从另一个角度重新展开它们：把它们当作上下文管理手段**——在有限预算下，决定什么留下、什么换出、什么隔离。同一组原语，两种视角，互不重复。

把这些手法摆在一起，会发现它们沿着一条张力线分布：**丢信息 vs 不丢信息。**

### 压缩派：Compaction 与 Context Reset

最直接的办法是**压缩（compaction）**：把接近窗口上限的对话摘要掉，用摘要重启一个新窗口。Anthropic 给的实践要点是：保留架构决策、未解决的 bug、关键实现细节，丢掉冗余的工具输出；调参顺序是**先最大化 recall（别漏掉相关内容），再优化 precision（去掉多余的）**。Claude Code 的自动 compact 在上下文超过 **95%** 时触发，先清理较早的工具输出，必要时再摘要对话，并保留最近访问的 5 个文件。

其中最轻量的一种形式是 **tool-result clearing**：当接近 token 上限时，自动清掉上下文里陈旧的工具调用和结果。Anthropic 在 [context management 的发布](https://claude.com/blog/context-management) 里给了一组数字——在一个 100 轮的 web 搜索测试里，单是 context editing 就**削减了 84% 的 token 消耗**、带来 **29% 的性能提升**，配合 memory 工具一起用则达到 **39%**。

但压缩有个绕不开的风险：**过度压缩会丢掉当时看不出重要、后来才显出关键的细节。** 这就引出和 compaction 相对的另一条路——**context reset**。两者的区别值得讲清楚：compaction 是原地压缩，**同一个 agent 带着摘要继续跑**，保留了连续性，但没有给一个真正干净的开始；reset 则是**给下一个 agent 一个全新的干净上下文**，只通过一份结构化的 handoff artifact 交接必要状态。reset 要付出 handoff 的成本，但它能真正切断"上下文焦虑"——前面几十轮的噪音不再跟着走。选 compaction 还是 reset，本质是在"连续性"和"干净起点"之间权衡。

### 外部化派：把记忆挪到窗口外面

另一条思路是干脆不在窗口里硬扛，把信息**外部化**到文件系统。Manus 在 [Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) 里把文件系统当作"终极上下文"——容量无限、天然持久、Agent 可直接操作。

这里的关键工程原则是**压缩必须可恢复**：丢掉一个网页的正文，但留下它的 URL；省略一份文档的内容，但保留它的路径。这样需要时还能把内容重新拉回来，避免了 compaction 那种不可逆的信息损失。Anthropic 的结构化笔记（写 `NOTES.md`、维护 to-do 列表）和 Claude 的 file-based memory 工具走的是同一条路：把记忆持久化到窗口外，需要时再读回来，且跨会话存活。

### 抗遗忘派：Recitation

长循环任务还有一个独立的问题：**目标漂移**。Manus 观察到一个典型任务平均要 ~50 次工具调用，跑到后半程模型很容易忘了最初的目标——这正是 Lost-in-the-Middle 在 agentic 场景的体现。他们的对策是 **recitation**：维护一个 `todo.md`，每完成一步就重写一次，把当前目标不断推到上下文的**最末端**，也就是模型注意力最强的位置。这不需要任何架构改动，纯粹靠"把重要的东西放在模型最近的注意力范围里"来对抗遗忘。

### 反直觉派：把错误留在上下文里

一个违反直觉但很有效的手法：**别清理失败。** Manus 主张把失败的动作和它产生的 stack trace 留在上下文里，因为模型看到失败会隐式更新自己的信念，从而避免重复同样的错误。他们甚至认为，**错误恢复能力是"真正 agentic 行为"最清晰的指标之一**。相关的还有 **don't get few-shotted**：如果上下文里全是格式整齐、节奏一致的动作-观察对，模型会陷入模仿这种节奏的惯性，导致漂移和过度泛化；适当注入一些格式变化反而更稳。

### 隔离派：Subagent 与 Checkpoint / Fork

最后一条路是**隔离**——与其在一个上下文里硬塞，不如把任务切开。

**Subagent** 的核心工程价值，不是多开几个 agent 提速，而是**上下文隔离**：子 agent 可以为了探索烧掉数万 token，但只向主 agent 返回一份 **1000–2000 token 的压缩摘要**（Anthropic 的数字）。主 agent 因此始终保持干净，detailed 的搜索/探索上下文被挡在外面。Claude Code 的 [subagent](https://code.claude.com/docs/en/sub-agents) 给每个子 agent 独立的上下文窗口、定制的 system prompt、受限的工具集和独立权限——一个只读的 reviewer subagent，"不许改文件"就不再是一句靠模型自觉的提示词，而是它的动作空间里根本没有 Edit。

和隔离配套的是**状态管理原语**。Claude Code 的 [checkpointing](https://code.claude.com/docs/en/checkpointing) 在每次用户 prompt 时自动快照文件状态、跨会话持久、30 天后清理，支持五种 rewind（恢复代码 / 恢复对话 / 两者都恢复 / 从某点摘要 / 摘要到某点）。会话本身是本地 JSONL（见 [how Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)），`--continue`/`--resume` 在同一会话后面追加消息，`--fork-session`/`/branch` 则把历史复制成一个新会话、原会话不动。从上下文管理的角度看，这些原语提供的是"换出"和"分叉"能力：把一段上下文存到窗口外、需要时再换回来，或者从一个干净分支重新开始。

### 张力的极端：要不要上多智能体？

把"隔离"推到极致，就撞上了 Agent 工程里最尖锐的一场争论：**到底该不该用多智能体并行？** 这场争论的两方都有硬数据，而且结论看似相反。

**Anthropic 力挺。** 在 [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) 里，他们用 orchestrator-worker 模式（一个 lead agent 协调、若干 subagent 并行），报告在内部研究评测上**比单 agent Claude Opus 4 高出 90.2%**，复杂查询的研究时间**最多缩短 90%**（lead 同时起 3–5 个 subagent，每个再并行调 3+ 工具）。代价是 token：agent 本身比聊天多烧 ~4×，多智能体系统则多烧 **~15×**，而且他们发现 **token 用量本身就能解释 80% 的性能方差**。结论是：多智能体适合**高价值、可大量并行、信息量超出单窗口、要对接众多复杂工具**的任务——典型是广度优先的研究。

**Cognition 唱反调。** 在 [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents) 里，他们主张对**编码任务**默认排除多智能体架构，理由是两条原则：一是"**要共享完整的 agent trace，而不只是单条消息**"，二是"**行动隐含着决策，而冲突的决策导致糟糕的结果**"。他们举的例子是做一个 Flappy Bird 克隆：一个 subagent 画背景和管道、另一个画小鸟，因为各自独立地隐式决定了视觉风格，最后拼出来风格不搭。他们的处方是**单线程线性 agent**——上下文连续，每个动作都看得到此前所有决策；对于长到溢出窗口的任务，再上一个**专门的历史压缩模型**（必要时甚至微调一个小模型）来把历史压成关键细节和决策。

**这两方其实没有真的矛盾，他们在为不同的任务画像优化。** Anthropic 的场景是研究——子任务天然可并行、彼此独立，广度优先；Cognition 的场景是编码——子任务高度耦合、后一步依赖前一步的隐含决策，顺序执行。Anthropic 自己也承认"大多数编码任务的可并行子任务比研究少"。从上下文管理的角度看，这场争论的本质是**两条压缩历史的工程路径之争**：subagent 摘要隔离（把历史压成子 agent 返回的摘要）vs 单线程 + 专门压缩模型（把历史压成一份连续的关键记录）。至于"到底该选单体还是多体"这个**架构选型**问题，已经超出上下文管理、进入 Agent 架构本身——我在认知结构那篇的 "Single Agent or MAS" 一节里讨论过它的边界。

### 收束：Write / Select / Compress / Isolate

上面这些手法看似零散，但 LangChain 在 [Context Engineering for Agents](https://www.langchain.com/blog/context-engineering-for-agents) 里给了一个很好用的归档框架，把它们收进四类动作：

- **Write（写出去）**：把信息存到窗口外——scratchpad、外部化笔记、跨会话记忆。
- **Select（选进来）**：把相关信息拉进窗口——检索、`CLAUDE.md` 这类规则文件、记忆召回。
- **Compress（压缩）**：只留执行任务所需的 token——compaction、摘要、trimming。
- **Isolate（隔离）**：把上下文切分到不同 agent 或环境——subagent、沙箱。

对照来看：本节的压缩派是 Compress，外部化派是 Write，读路径那节是 Select，隔离派是 Isolate。这个框架的好处是提醒我们：**Context Engineering 不是单一技巧，而是这四个动作的组合编排。** Drew Breunig 还总结过四种典型的上下文失败模式——poisoning（幻觉进入上下文并被反复引用）、distraction（上下文多到压过模型自身能力）、confusion（无关内容干扰了回答）、clash（上下文里有互相矛盾的部分）——这四种失败，恰好是上面四个动作没做好时会掉进的坑。

## 怎么用好 KV-Cache 友好的上下文组织

前面反复提到 token 成本。这里要单独说一个**怎么用**的层面——注意，不是 KV-Cache 的成本机理（那在 [output token 那篇](/blog/2026/04/26/output-token-pricing-kv-cache-agent-cost/) 里讲过 prefill/decode 不对称、稳定前缀为什么省钱），而是在已知这层机理的前提下，运行时组织上下文有哪些实践要点。

核心原则是**让前缀稳定、让上下文 append-only**。Manus 给的几条很实用：

- **前缀要稳定**：哪怕只改动一个 token，它后面的缓存就全部失效。一个常见的坑是在 system prompt 里放秒级精确的时间戳——它会让每次请求的前缀都不一样，cache 命中率归零。
- **上下文 append-only**：不要回头去修改前面已经发生的动作和观察；序列化要确定（比如 JSON 的 key 顺序要稳定，否则会悄悄破坏缓存）。
- **Mask，而不是增删工具**：动态地往上下文里加减工具定义是双重灾难——工具定义通常在上下文很靠前的位置，一改就让 KV-Cache 大面积失效；而且上下文里若引用了已经不存在的工具，模型会困惑甚至幻觉出调用。更好的做法是用一个状态机在解码时 **mask 掉工具的 logits**，工具定义始终不变，只是让某些工具在当前步"不可选"。配合一致的工具名前缀（如 `browser_`、`shell_`），可以按组约束动作空间。

这些要点的回报是实打实的：缓存命中的 token 和未命中的，价格能差到 **10×**。把上下文组织得 cache-friendly，几乎是 Agent 成本优化里性价比最高的一件事。

## 治理、隔离边界与评估闭环

最后收束到治理。对一个长期在线的 Agent 来说，遗忘和召回同等重要——把不该继续在场的信息移出去，和把该在场的带进来，是同一个问题的两面。

**删除不等于物理抹除。** 工程上至少要区分几种动作：软删除（对当前检索不可见，但保留审计记录）、硬删除（为隐私合规执行物理移除）、降级（从 active 降到 stale，不再默认召回但可追溯）、回滚（更新错误时恢复上一版本）。如果系统只有 append 和 overwrite 两种操作，它几乎不可能被部署到长周期、强约束的环境里。这些动作全都依赖前面 memory object 的 `status`、`links`、`timestamp` 字段——治理不是事后补的，是写入时就要埋好的。

**隔离边界**也属于治理。很多所谓的"上下文污染"，本质不是压缩失败，而是隔离失败：

- **Subagent 隔离**：子智能体保留中间推理，主智能体只接收结果摘要。
- **Tool-call 隔离**：工具执行日志进入工具工作区，而不是直接污染主会话。
- **Session 隔离**：不同任务或不同用户的工作记忆默认不共享，只通过显式 memory object 交换。

至于**怎么评估**一个 Context 系统做得好不好，我在 Memory 全景图里给过一套 L1–L6 框架（写入正确性、更新正确性、调用及时性、行为一致性、经验迁移可控性、不确定性处理），这里直接沿用、不再重复定义。需要补的只有一句运行时视角的话：评估不能只测"答得对不对"，还要能测"它知不知道自己什么时候不该自信"——记忆模糊时主动 abstain 并留痕，是一个成熟 Context 系统的标志。

## 结论：Context 管理是一个可治理的系统

回头看，Context Engineering 落地时至少要同时守住三件事：

- 信息是否够用，同时没有把注意力预算和调用成本吃光。
- 规则是否足够稳定，同时允许经验和用户状态继续更新。
- 自动写入与压缩是否可审计、可恢复、可回滚，而不是把决策藏进黑盒。

所以，Context Engineering 的重点从来不是单一技巧，而是把读、写、压缩、隔离、遗忘、审计这些动作纳入一个可治理的系统设计里。更强的 Agent 不只是窗口更长——它要能说明哪些信息进入了当前上下文、哪些被压缩或换出、哪些被隔离在子 agent 里、哪些被写入长期记忆，以及这些决定为什么发生。窗口会一直有限，注意力会一直是预算；只要这个前提不变，把上下文当成有限资源来精心调度，就一直是 Agent 工程的核心。

## 参考资料

- Anthropic, [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- Anthropic, [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- Anthropic / Claude, [Managing context on the Claude Developer Platform](https://claude.com/blog/context-management)
- Manus, [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- Cognition, [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents)
- Chroma, [Context Rot: How Increasing Input Tokens Impacts LLM Performance](https://www.trychroma.com/research/context-rot)
- LangChain, [Context Engineering for Agents](https://www.langchain.com/blog/context-engineering-for-agents)
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- Anthropic Docs, [Subagents](https://code.claude.com/docs/en/sub-agents)、[Checkpointing](https://code.claude.com/docs/en/checkpointing)、[How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)
- [《从记忆形成到记忆治理：Agent Memory 的全景图》](/blog/2026/03/21/agent-memory-panorama/)
- [《从智能体的认知结构到智能体框架》](/blog/2026/03/03/cognitive-architecture-to-agent-framework/)
- [《给 LLM 戴上确定性枷锁的外围工程：从 Claude Code 看 Agent Harness》](/blog/2026/03/20/building-agent-deterministic-constraints/)
- [《为什么 Output Token 更贵：从 KV Cache 到 Agent 成本工程》](/blog/2026/04/26/output-token-pricing-kv-cache-agent-cost/)
