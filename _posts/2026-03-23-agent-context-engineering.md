---
layout: blog-post
title: Context is All You Need：智能体的上下文工程
title_en: "Context Is All You Need: Context Engineering for Agents"
date: 2026-03-23 12:00:00 +0800
categories: [Agent 系统]
tags: [Agents, Context Engineering]
author: Hyacehila
excerpt: 把上下文当成有限资源来调度：从 context rot 与注意力预算出发，讨论存储结构、检索 pipeline、写入时的时间对齐与版本治理，以及 compaction、reset、subagent、checkpoint 等运行时手法。
excerpt_en: "Treating context as a finite resource: from context rot and attention budgets to storage structures, retrieval pipelines, time-aligned writes, version governance, and runtime techniques such as compaction, reset, subagents, and checkpointing."
featured: true
math: false
---

# Context is All You Need：智能体的上下文工程

## 引言：从地图到手册

在[《从记忆形成到记忆治理：Agent Memory 的全景图》](/blog/2026/03/21/agent-memory-panorama/)里，我把 Agent Memory 拆成了三层结构：L1 全上下文（知识直接在当前窗口里，靠长上下文与 KV-cache 支撑）、L2 外部记忆（向量库、文件系统、知识图谱里的外接非参数记忆）、L3 参数记忆（编码进权重的隐式知识）。那篇文章的主角是 L2，也就是长期记忆如何形成、组织、更新和失效。L1 这部分，我当时只留下了一个入口：当前推理现场里，信息到底怎么选择、压缩、隔离和调度。

这篇文章接着讲这个入口。Memory 全景图更像一张地图，回答的是"长期记忆都在研究什么"；本文换成一个更工程化的问题：在真实的 Agent runtime 里，注意力预算有限，信息怎么读进来、怎么留住、又怎么从当前窗口里换出去？

## 写在正文开始之前

正文会从存储结构讲到运行时手法。先把几件事讲清楚：本文在整个系列里处在哪一层，"上下文工程"在这里具体指什么，以及为什么上下文应该被当成一种需要调度的有限资源。

### 三篇的分工：地图、手册、拆解

这篇文章和前后两篇是一组，但分工不同：《Agent Memory 全景图》偏研究视角，讨论长期记忆怎么形成、组织、更新、失效和评估。本文偏工程落地，讨论在一个 runtime 里，有限注意力预算下信息怎么进出当前窗口。《Agent Runtime Teardown》偏产品和框架拆解，讨论市面上常见的 Agent Runtime 和独立记忆系统怎么装配这些方案。

所以本文会停在"工程方案"这一层，把常见手法讲清楚，但不逐个产品横评。读到某个机制时，如果你想知道某个具体系统怎么实现，代码的每一行是怎么写的，答案在 teardown 篇里。

按 CoALA 的语言（见[《从智能体的认知结构到智能体框架》](/blog/2026/03/03/cognitive-architecture-to-agent-framework/)），本文讨论的基本都是 Working Memory 的工程化管理：什么该进场，什么该留在场上，什么该退场，什么要在需要时重新召回。只要底层还是 attention 机制和有限窗口，这几个动作就绕不开。

为了更好的理解记忆调度问题，我们也必须对 Memory 所使用的数据结构进行一定的介绍，广义的 RAG 以及一些工程上的 Context 调度手段是本文的后面将要讨论的核心。

### 严格区分 LLM Memory 与 Agent Memory

讨论运行时调度前，先拆开一个常见混淆：KV-Cache、RoPE、Attention 变种、长上下文架构，解决的是模型如何更高效地利用窗口；Agent Memory 解决的是智能体如何跨任务积累、检索、更新和遗忘知识。前者是推理基础设施，后者是系统设计。解决他们的思路完全不同，相关研究也大相径庭。

这条区分对本文尤其重要，因为本文谈的正好是 L1。L1 的物理基底确实是 LLM Memory：窗口能放多少、prompt cache 命中率多高、decode 多贵，都由模型架构和推理基础设施决定。但本文不展开这层成本机理。为什么 output token 比 input 贵、prefill 与 decode 的不对称、KV-Cache 如何吃掉显存和调度槽位，以及怎么把上下文组织得 KV-Cache 友好，我在[《为什么 Output Token 更贵：从 KV Cache 到 Agent 成本工程》](/blog/2026/04/26/output-token-pricing-kv-cache-agent-cost/)里已经单独讲过。本文接受这些基础设施约束，只讨论 Agent 在约束之上怎么组织上下文。简单说：LLM Memory 决定工作台有多大、多贵；Agent Context Engineering 决定工作台上该放什么。

### 上下文是有限资源：Context Rot 与注意力预算

Context Engineering 的动机很直接：上下文是有限资源，而且退化通常来得比直觉更早。许多研究都指出了直觉与现实之间的差距。

更大的窗口不等于更好的利用。早在 [Lost in the Middle](https://arxiv.org/abs/2307.03172) 里就有一个被反复验证的现象：模型对长上下文不同位置的信息利用并不均匀，开头和结尾的信息更容易被用上，中间的信息经常被忽略，即便是长上下文模型也一样。

Chroma 的 [Context Rot](https://www.trychroma.com/research/context-rot) 报告测了 18 个模型，结论相当一致：性能会随输入长度增长而退化，而且退化不均匀，常常出现在很意外的位置。大海捞针（NIAH）测试不一定能够准确反应LLM的长上下文能力，性能退化可能远比我们想象的更早。更反直觉的是，他们发现结构连贯的 haystack 反而不如打乱顺序的 haystack。也就是说，让上下文"读起来更顺"有时会损害检索性能。这说明问题没有一个简单的"装满就崩"阈值，内容呈现方式也会影响模型取用信息。（P.s. 这家公司是做 Vector Retrival Infra 的，好用）

Anthropic 在 [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) 里把这件事拆成两个协作的概念：模型在充足的信息中受益，受限于 Transformer 的结构产生上下文腐烂。那么无论是压缩，笔记还是 Subagent 他们都是为了找到能最大化目标达成概率的、最小的高信号 token 集合。

这就是后面所有方案的前提。上下文会腐化，注意力有预算，所以"信息放在哪""怎么取出来""怎么写进去""什么时候隔离"都不是锦上添花。而是处理复杂任务的必需。我们将分别讨论记忆的结构，检索与写入的手段，和一些工程上的有趣 tricks。

## 存储结构：把记忆放在哪、长什么样

先看存储。Memory 全景图已经解释过为什么不同记忆适合不同载体：关系密集的语义知识适合图，操作性经验和规范适合文件，事实记录适合数据库。本文不再重复这套选型理由，直接往下一层看：载体选定之后，工程上怎么把读写建出来？

### 向量库：chunking、embedding 与索引

把文本塞进向量库远不是"embed 一下"。几个选择会直接影响召回质量：

- Chunking 策略：定长切分简单，但容易割裂语义；按结构切分（段落、函数、章节）更稳，不过需要解析器。chunk 太大，单块信号会被稀释；太小，又会丢上下文，召回后还要拼。实践中常配合 chunk overlap 和 parent-document 回溯：命中小 chunk，再回填它所在的大块。
- Embedding 选择：不同 embedding 模型在领域文本上的表现差异很大，代码、法律、医疗都可能需要专门模型。维度越高，召回可能略好，但存储和检索成本也会上去。
- 索引类型：精确检索可以用暴力 KNN，规模一大就不可行。生产里通常用 ANN。HNSW 是图结构，召回率高、内存占用大；IVF 是倒排加聚类，省内存、需要训练、召回略低。最后仍然是在延迟、内存和召回率之间取舍。

这一节讲的是怎么把库建出来。建好之后怎么取准，比如稀疏加稠密混合、metadata 过滤，放到下一章讲。

### 图存储：抽取与增量更新

图的难点不在查询，而在怎么把非结构化文本变成图，以及图怎么随新信息增量更新。实体关系抽取要回答谁、和谁、什么关系、何时，通常还要靠一轮 LLM 调用。这一步质量决定图能不能用；抽取错了，再漂亮的多跳遍历也只是在错误的边上推理。

运行时遍历还要控制跳数和扇出。多跳关联是图的长处，但不加限制地遍历，很快就会把成百上千个节点拉进候选集，最后又违背"最小高信号集合"。增量更新也麻烦：要做实体消解，判断"老王"和"王经理"是不是同一个节点；还要处理边的时效，判断旧关系是否已经失效。

### 文件系统：目录布局与导航

文件系统是被低估的记忆载体。它对 Agent 也很友好，因为 Agent 天生会读写文件、跑 glob 和 grep。工程重点不在"存进去"，而在让 Agent 高效导航，不必把全部内容一次性拉进上下文。

- 目录布局与命名约定：层次化目录本身就是廉价索引。清晰命名能让 Agent 靠路径猜内容，少开很多无关文件。
- glob / grep / LSP 导航：用 `glob` 定位文件、`grep` 精确匹配、`LSP` 做符号跳转，按需读取，而不是预先全量加载。这就是 just-in-time 检索的物理实现。
- 层级加载：像 `CLAUDE.md` 这样的约定，把项目级稳定上下文放在固定位置。会话开始先加载一部分，其余内容按需展开。

### Memory Object 与 Metadata

无论最终落在向量库、数据库、图还是文件系统，运行时读写的基本单位都不应该是一整块未经治理的文本，而应该是带元数据的 memory object。一个 memory object 至少需要这些字段：

- `type`：persona、conversation、experience、knowledge、rule 等。
- `scope`：用户级、会话级、项目级、全局级。
- `content`：被保留的事实、规则、经验或摘要。
- `source`：来自哪次对话、哪次工具调用、哪份外部材料。
- `timestamp`：写入时间和最后更新时间。
- `confidence`：这条记忆有多确定。
- `status`：active、stale、conflicted、deleted 等。
- `links`：它与哪些记忆重复、冲突、继承或 supersede。

Memory 全景图列过写入侧要回答的六个问题：写什么、写到哪、怎么写、如何更新、何时失效、何时写。这里这张字段表，是承载这些决定的工程结构。`status` 和 `links` 表达版本与冲突语义，`timestamp` 和 `confidence` 表达时效与不确定性。没有这些元数据，后面的版本更新、冲突消解、删除与回滚都很难做。它们还会直接进入下一章的 metadata 过滤，也会被 promotion 用来保留 provenance。读写两条路径，其实在这里接上了。

## 从存储结构中检索

有了存储结构，检索就是把外部信息接回当前推理现场。除了"能不能检索"，还要决定走哪条检索路径、取到什么粒度，以及怎么把结果克制地暴露给 Working Memory。

这里也和 Memory 全景图分一下工。那篇文章列过长期记忆的四类激活路径：字符匹配、语义检索、结构化信源、图遍历。但它没有展开成工程 pipeline。本文讲的就是 runtime 怎么编排这些路径，怎么把候选压进当前窗口。

### 混合检索：按信息需求路由

目前最实用的方案，通常不是在 RAG 和 grep 之间二选一。更常见的做法是承认它们擅长的问题不同，并把它们放进一个检索编排层：

- Grep / 关键词匹配（BM25）：适合已知术语、配置名、错误信息、规则条目等精确定位任务。
- 向量检索 / RAG：适合自然语言查询、模糊需求、跨表述召回。
- LSP / 结构化索引：适合代码、配置、符号引用等天然带类型和层次的信息源。
- 图遍历 / 关系查询：适合需要多跳关联、时间条件、实体关系的记忆内容。

编排层要做的，是让 Agent 先判断当前任务属于哪类信息需求，再选择检索组合，避免所有问题都默认走向量库。工程上最值钱的地方，经常不在某一种检索技术本身，而在路由是否做对。

### 稀疏 + 稠密混合与 Metadata 过滤

路由决定走哪条路。进入向量检索这条路以后，还有两个增强几乎总该做：

- 稀疏 + 稠密混合：纯向量检索对精确术语（错误码、API 名、人名）经常打不准。把 BM25 稀疏信号和稠密向量信号融合，比如用 RRF 倒数排名融合，通常比单路更稳。
- Metadata 过滤：在向量检索前后用 metadata（scope、type、时间、来源）做硬过滤，可以缩小搜索空间，也能减少跨域污染。这要求写入时就把 metadata 结构化好，也就是上一章 memory object 里的字段。

### Query Rewrite 与 Rerank

原始输入通常不是好的查询。用户说的是任务语言，记忆库存的是摘要、事实、代码片段、规则、图节点和历史上下文，两边天然有语义落差。检索系统至少需要两步增强：

1. Query rewrite：把用户问题改写成适合 memory store 的检索语句，必要时拆成多个子查询。
2. Rerank / compression：把候选结果重新排序，并整理成能进入 Working Memory 的最小必要片段。

没有这两步，"检索增强"很容易变成把更多噪声搬进上下文。有效的检索系统不追求召回一百条候选，而要稳定挑出那几条当前任务真正需要的信息。

### 从检索结果到 Working Memory：promotion 的实现

Memory 全景图反复讲过一个判断：长期记忆不会直接参与行动。它必须经过检索、筛选、压缩，以足够克制的形式进入 Working Memory 才能影响下一步。那篇文章讲原则，这里讲动作。

promotion 不应该把长期记忆的存储单位原样塞进当前上下文，而要做一次最小必要暴露：

- 提取和当前目标直接相关的字段，丢掉其余内容。
- 必要时保留 provenance 和时间信息，比如这条记忆来自哪里、什么时候生效。
- 对重复、冲突、低置信度内容做折叠或标记。
- 超预算时，优先保留约束、计划、当前子任务和高价值事实。

Working Memory 不是长期记忆的镜像，而是它在当前任务切面上的受限投影。promotion 做的就是这次投影。它的输入是当前任务意图和剩余窗口预算，这两个量都是运行时变量，所以 promotion 的实现属于 L1，而不是 L2。

## 存入：经验落库，以及时间对齐与版本

如果检索回答"什么该被带进来"，存入回答的就是"什么值得留下来"。后者往往更难，因为写错的记忆会在未来持续污染检索结果。

### 写路径四步：抽取、分类、对齐、版本化

一个可用的写路径至少要经过四步：

1. 抽取：从对话、工具调用、环境反馈里识别值得保留的候选信息，过滤噪声。
2. 分类：判断它应该进入 persona、经验、语义知识还是对话记忆，决定写入哪个载体。
3. 对齐：检查是否已有相同或相近记忆，决定新增、merge、supersede，还是只标记冲突。
4. 版本化：保留更新链路，写进 `links` 和 `status`，而不是简单覆盖旧值。

前两步相对直接，难点在对齐和版本化。记忆带着时间维度：一条记忆什么时候生效、什么时候被另一条取代、两条冲突时该信谁，这些都不是"覆盖"两个字能解决的。

### 何时写：事件触发而非会话结束

"何时写"和"写什么"一样重要。很多系统把写入拖到会话结束再统一处理，但这通常太晚。更合理的是按事件触发：用户明确给出稳定偏好或身份事实，工具调用产生可迁移经验，用户纠正系统并推翻旧事实，或者某条规则在短时间内反复被激活。

写入质量会决定未来检索质量。如果写入时没有做好结构化、去重和版本关系，再强的检索也只能从一堆冗余和矛盾里筛。

### 时间对齐与版本调整

写入最容易出错的地方，是把"新信息"直接当成"对旧信息的覆盖"。比如"用户喜欢拿铁"和"用户最近开始戒咖啡"就不是简单二选一。后者不一定推翻前者，因为戒咖啡可能只是阶段性的；但两者也不能合并成一条静态偏好。系统需要知道两条记忆的时间关系、适用条件和冲突状态，而不是把它们压平。

工程上，这落在 memory object 的几个字段上：

- supersede 而非 overwrite：新事实出现时，旧事实不删，标成 `status: stale`，用 `links` 指向取代它的新版本。需要追溯时，更新链路完整可见。
- 冲突标记而非强行合并：两条记忆矛盾且无法判断时序时，标成 `status: conflicted`，留给后续证据或用户澄清，不要当场赌一个。
- 时效靠 `timestamp` 承载：写入时间、最后更新时间、必要时的过期时间都记下来。召回时才能按时间加权，或者硬过滤掉过期信息。

写入更像一次带时间戳的版本提交，不是给全局变量赋值。只要把记忆当成"当前最优快照"去 overwrite，时间维度上的信息就被抹掉了。

### 删除不等于物理抹除

记忆生命周期的另一端是删除。对长期在线的 Agent 来说，遗忘和召回一样重要：把不该继续在场的信息移出去，和把该在场的信息带进来，是同一个问题的两面。工程上至少要区分几种动作：

- 软删除：对当前检索不可见，但保留审计记录。
- 硬删除：为隐私合规执行物理移除。
- 降级：从 active 降到 stale，不再默认召回，但仍可追溯。
- 回滚：更新错误时恢复上一版本。

如果系统只有 append 和 overwrite 两种操作，就很难部署到长周期、强约束的环境里。这些动作都依赖前面 memory object 的 `status`、`links`、`timestamp` 字段。治理不是事后补丁，写入时就要留下结构。

## 缓解上下文不足的工程手法

前面几章解决的是信息怎么进出存储。Context Engineering 更难的部分，在单次任务执行过程中出现：任务跑了几十轮，上下文逼近窗口上限，工作台开始变脏，这时该怎么办？

这一节讨论这些运行时手法。先和[《给 LLM 戴上确定性枷锁的外围工程》](/blog/2026/03/20/building-agent-deterministic-constraints/)划一下边界：那篇文章也讲 subagent、checkpoint、fork、worktree，但角度是"限制即可靠性"，把它们看成收窄失败半径的原语。这里换一个角度，只讨论它们怎么帮助管理上下文：哪些信息留下，哪些换出，哪些隔离到别处。

这些手法大致分成几类：压缩当前窗口、把信息外部化、反复刷新目标、保留失败痕迹，以及把任务隔离到不同上下文里。

### 压缩派：Compaction 与 Context Reset

最直接的办法是压缩（compaction）：把接近窗口上限的对话摘要掉，用摘要重启一个新窗口。Anthropic 给的实践要点是：保留架构决策、未解决的 bug、关键实现细节，丢掉冗余工具输出；调参顺序是先最大化 recall（别漏掉相关内容），再优化 precision（去掉多余内容）。Claude Code 的自动 compact 在上下文超过 95% 时触发，先清理较早的工具输出，必要时再摘要对话，并保留最近访问的 5 个文件。

其中最轻量的一种形式是 tool-result clearing：接近 token 上限时，自动清掉上下文里陈旧的工具调用和结果。Anthropic 在 [context management 的发布](https://claude.com/blog/context-management) 里给了一组数字：在一个 100 轮 web 搜索测试里，单是 context editing 就削减了 84% 的 token 消耗，带来 29% 的性能提升；配合 memory 工具一起用则达到 39%。

压缩的风险也明显：过度压缩会丢掉当时看不出重要、后来才显出关键的细节。context reset 是另一种处理方式。compaction 是原地压缩，同一个 agent 带着摘要继续跑，连续性更好；reset 则给下一个 agent 一个干净上下文，只通过一份结构化 handoff artifact 交接必要状态。reset 要付出 handoff 成本，但能切断前面几十轮噪音的影响。选哪一个，本质是在连续性和干净起点之间取舍。

### 外部化派：把记忆挪到窗口外面

另一条路是不在窗口里硬扛，把信息外部化到文件系统。Manus 在 [Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) 里把文件系统称作"终极上下文"：容量大、天然持久，Agent 也能直接操作。

压缩最好是可恢复的。可以丢掉网页正文，但留下 URL；可以省略文档内容，但保留路径。这样需要时还能重新拉回来，避免 compaction 带来的不可逆损失。Anthropic 的结构化笔记（写 `NOTES.md`、维护 to-do 列表）和 Claude 的 file-based memory 工具也是这个思路：把记忆放到窗口外，需要时再读回来，并且跨会话存活。

### 抗遗忘派：Recitation

长循环任务还有一个独立问题：目标漂移。Manus 观察到一个典型任务平均要 ~50 次工具调用，跑到后半程模型很容易忘了最初目标。这就是 Lost-in-the-Middle 在 agentic 场景里的表现。他们的对策是 recitation：维护一个 `todo.md`，每完成一步就重写一次，把当前目标不断推到上下文最末端，也就是模型注意力最强的位置。这个办法不需要改架构，只是把重要信息放到模型最近能看到的地方。

### 反直觉派：把错误留在上下文里

还有一个违反直觉但很有效的手法：别急着清理失败。Manus 主张把失败动作和对应 stack trace 留在上下文里，因为模型看到失败会隐式更新自己的信念，从而避免重复同样的错误。他们甚至认为，错误恢复能力是判断 agentic 行为的清晰指标之一。

相关的还有 don't get few-shotted：如果上下文里全是格式整齐、节奏一致的动作-观察对，模型会开始模仿这种节奏，导致漂移和过度泛化。适当注入一些格式变化，反而更稳。

### 隔离派：Subagent 与 Checkpoint / Fork

最后一类是隔离。与其在一个上下文里硬塞，不如把任务切开。

Subagent 的工程价值不只是提速，更重要的是上下文隔离：子 agent 可以为了探索烧掉数万 token，但只向主 agent 返回一份 1000-2000 token 的压缩摘要（Anthropic 的数字）。主 agent 因此保持干净，详细搜索和探索过程被挡在外面。Claude Code 的 [subagent](https://code.claude.com/docs/en/sub-agents) 给每个子 agent 独立的上下文窗口、定制 system prompt、受限工具集和独立权限。一个只读 reviewer subagent 如果没有 Edit 工具，"不许改文件"就不只是提示词，而是动作空间限制。

和隔离配套的是状态管理原语。Claude Code 的 [checkpointing](https://code.claude.com/docs/en/checkpointing) 在每次用户 prompt 时自动快照文件状态、跨会话持久、30 天后清理，支持五种 rewind（恢复代码 / 恢复对话 / 两者都恢复 / 从某点摘要 / 摘要到某点）。会话本身是本地 JSONL（见 [how Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)），`--continue` / `--resume` 在同一会话后面追加消息，`--fork-session` / `/branch` 则把历史复制成一个新会话、原会话不动。从上下文管理角度看，这些原语提供的是换出和分叉能力：把一段上下文存到窗口外，需要时再换回来，或者从一个干净分支重新开始。

边界再说一次：subagent 隔离、tool-call 隔离、session 隔离作为可靠性原语的系统展开，在确定性枷锁那篇里讲过。这里只取它们的上下文管理视角。

### 进一步的问题：要不要上多智能体？

把隔离推到极致，就会撞上 Agent 工程里最尖锐的一场争论：到底该不该用多智能体并行？这场争论两边都有硬数据，而且结论看起来相反。

Anthropic 支持多智能体。在 [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) 里，他们使用 orchestrator-worker 模式：一个 lead agent 协调若干 subagent 并行。他们报告，在内部研究评测上，这套系统比单 agent Claude Opus 4 高出 90.2%，复杂查询的研究时间最多缩短 90%（lead 同时起 3-5 个 subagent，每个再并行调 3+ 工具）。代价是 token：agent 本身比聊天多烧 ~4x，多智能体系统则多烧 ~15x，而且他们发现 token 用量本身就能解释 80% 的性能方差。结论是，多智能体适合高价值、可大量并行、信息量超出单窗口、需要对接众多复杂工具的任务，典型就是广度优先研究。

Cognition 反对默认多智能体。在 [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents) 里，他们主张编码任务默认排除多智能体架构，理由有两条：一是要共享完整的 agent trace，而不只是单条消息；二是行动隐含决策，冲突的决策会带来糟糕结果。他们举的例子是做一个 Flappy Bird 克隆：一个 subagent 画背景和管道，另一个画小鸟。因为各自独立地决定了视觉风格，最后拼出来就不搭。他们的处方是单线程线性 agent：上下文连续，每个动作都能看到此前决策；如果任务长到溢出窗口，再上一个专门的历史压缩模型，必要时甚至微调小模型，把历史压成关键细节和决策。

这两方其实在优化不同任务。Anthropic 的场景是研究，子任务天然可并行、彼此独立，适合广度优先；Cognition 的场景是编码，子任务高度耦合，后一步经常依赖前一步的隐含决策，顺序执行更稳。Anthropic 自己也承认，大多数编码任务的可并行子任务比研究少。从上下文管理角度看，这场争论是在比较两条压缩历史的路径：subagent 摘要隔离，把历史压成子 agent 返回的摘要；单线程加专门压缩模型，把历史压成一份连续的关键记录。至于到底选单体还是多体，已经进入 Agent 架构选型本身。我在认知结构那篇的 "Single Agent or MAS" 一节里讨论过它的边界。

### 收束：Write / Select / Compress / Isolate

这些手法看起来分散，但 LangChain 在 [Context Engineering for Agents](https://www.langchain.com/blog/context-engineering-for-agents) 里给了一个好用的归档框架，把它们收成四类动作：

- Write（写出去）：把信息存到窗口外，比如 scratchpad、外部化笔记、跨会话记忆。
- Select（选进来）：把相关信息拉进窗口，比如检索、`CLAUDE.md` 这类规则文件、记忆召回。
- Compress（压缩）：只留执行任务所需的 token，比如 compaction、摘要、trimming。
- Isolate（隔离）：把上下文切分到不同 agent 或环境，比如 subagent、沙箱。

对照前文，压缩派是 Compress，外部化派是 Write，检索那章是 Select，隔离派是 Isolate。这个框架的价值在于提醒我们：Context Engineering 不是某个单点技巧，需要组合编排这些动作。Drew Breunig 还总结过四种常见上下文失败模式：poisoning（幻觉进入上下文并被反复引用）、distraction（上下文多到压过模型自身能力）、confusion（无关内容干扰回答）、clash（上下文里有互相矛盾的部分）。这些坑，基本都能对应到 write、select、compress、isolate 的某个动作没做好。

## 结论：Context 管理是一个可治理的系统

Context Engineering 落地时，至少要同时守住几件事：

- 信息够用，但不把注意力预算和调用成本吃光。
- 规则足够稳定，同时允许经验和用户状态继续更新。
- 自动写入与压缩可审计、可恢复、可回滚，不能把关键决策藏进黑盒。

"怎么评估一个 Context 系统做得好不好"，我在 Memory 全景图里给过一套 L1-L6 框架：写入正确性、更新正确性、调用及时性、行为一致性、经验迁移可控性、不确定性处理。这里不重复。至于真实 runtime 怎么把存储、检索、写入、隔离这些动作装配起来，留给 teardown 篇逐个拆。

Context Engineering 的重点不是把窗口做长，也不是堆一个向量库。更实用的标准是：系统能不能说明哪些信息进入了当前上下文，哪些被压缩或换出，哪些被隔离在子 agent 里，哪些被写入长期记忆，以及这些决定为什么发生。窗口会继续有限，注意力也仍然是预算。只要这个前提不变，把上下文当成有限资源来调度，就是 Agent 工程的一项基础能力。

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
- [《拆开 Agent Runtime：记忆、上下文与隔离在真实系统里如何被装配》](/blog/2026/06/07/agent-runtime-teardown/)
