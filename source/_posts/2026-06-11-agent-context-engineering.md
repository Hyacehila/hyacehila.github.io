---
title: Context is All You Need：智能体的上下文工程
title_en: "Context Is All You Need: Context Engineering for Agents"
date: 2026-06-11 12:00:00 +0800
categories: ["AI & Agents", "Agent Architecture"]
tags: ["Context Engineering", "Retrieval", "Memory"]
author: Hyacehila
excerpt: 把上下文当成有限资源来调度：从 context rot 与注意力预算出发，讨论存储结构、检索 pipeline、写入时的时间对齐与版本治理，以及 compaction、reset、subagent、checkpoint 等运行时手法。
excerpt_en: "Treating context as a finite resource: from context rot and attention budgets to storage structures, retrieval pipelines, time-aligned writes, version governance, and runtime techniques such as compaction, reset, subagents, and checkpointing."
permalink: '/blog/2026/06/11/agent-context-engineering/'
---

## 引言：从地图到手册

在[《从记忆形成到记忆治理：Agent Memory 的全景图》](/blog/2026/03/21/agent-memory-panorama/)里，我把 Agent Memory 拆成了三层结构：L1 全上下文（知识直接在当前窗口里，靠长上下文与 KV-cache 支撑）、L2 外部记忆（向量库、文件系统、知识图谱里的外接非参数记忆）、L3 参数记忆（编码进权重的隐式知识）。

那篇文章的主角是 L2，也就是长期记忆如何形成、组织、更新和失效。L1 这部分，我当时只留下了一个简单的解释：**当前推理现场里，信息到底怎么选择、压缩、隔离和调度。**

这篇文章接着讲这个入口。Memory 全景图更像一张地图，回答的是"长期记忆都在研究什么"；本文换成一个更工程化的问题：在真实的 Agent runtime 里，注意力预算有限，信息怎么读进来、怎么留住、又怎么从当前窗口里离开？

## 正文开始之前

正文会从存储结构讲到运行时手法。正式展开前，先交代三件事：本文在整个系列里处在哪一层，上下文工程在这里具体指什么，以及为什么上下文应该被当成一种需要调度的有限资源。

### 三篇的分工：地图、手册、拆解

这篇文章和前后两篇是一组，但分工不同。《Agent Memory 全景图》偏研究视角，讨论长期记忆怎么形成、组织、更新、失效和评估。本文偏工程落地，讨论在一个 runtime 里，有限注意力预算下信息怎么进出当前窗口；《Agent Runtime Teardown》偏产品和框架拆解，讨论市面上常见的 Agent Runtime 和独立记忆系统怎么装配这些方案。

所以本文会停在工程实现，把常见手法讲清楚，但不逐个产品横评。读到某个机制时，如果你想知道某个具体系统怎么实现，代码的每一行是怎么写的，答案在 teardown 篇里。

按 CoALA 的语言（见[《从智能体的认知结构到智能体框架》](/blog/2026/03/03/cognitive-architecture-to-agent-framework/)），本文讨论的基本都是 Working Memory 的工程化管理：什么该进场，什么该留在场上，什么该退场，什么要在需要时重新召回，以及这些动作背后的附带约束和工程手法。只要底层还是 attention 机制和有限窗口，这几个动作就绕不开。

### 严格区分 LLM Memory 与 Agent Memory

讨论运行时调度前，先拆开一个常见混淆：KV-Cache、RoPE、Attention 变种、长上下文架构，这些 LLM Memory 解决的是模型如何拥有更长更有效的上下文窗口；Agent Memory 解决的是智能体如何跨任务积累、检索、更新和遗忘知识。前者是推理基础设施，后者是系统设计。解决这两类问题的思路完全不同，相关研究也大相径庭。

L1 的物理基底是 LLM Memory：窗口能放多少、prompt cache 命中率多高、decode 多贵，都由模型架构和推理基础设施决定。为什么 output token 比 input 贵、prefill 与 decode 的不对称、KV-Cache 如何吃掉显存和调度槽位，以及怎么把上下文组织得 KV-Cache 友好，我在[《为什么 Output Token 更贵：从 KV Cache 到 Agent 成本工程》](/blog/2026/04/26/output-token-pricing-kv-cache-agent-cost/)里已经单独聊过。

L2 是 Agent 的附属结构，但它要通过 L1 才能发挥作用：任何外部记忆，最后都必须进入推理上下文窗口，才能影响下一步行动。于是讨论 Agent Memory 时，就不能绕开 L1 的基础设施约束。简单说，LLM Memory 决定工作台有多大、多贵；Agent Context Engineering 决定外部信息什么时候进来、什么时候离开，以及工作台上到底该放什么。

### 上下文是有限资源：Context Rot 与注意力预算

Context Engineering 的动机很直接：上下文是有限资源，而且退化通常来得比直觉更早。许多研究都指出了直觉与现实之间的差距。

更大的窗口不等于更好地利用。早在 [Lost in the Middle](https://arxiv.org/abs/2307.03172) 里就有一个被反复验证的现象：模型对长上下文不同位置的信息利用并不均匀，开头和结尾的信息更容易被用上，中间的信息经常被忽略，即便是长上下文模型也一样。

Chroma 的 [Context Rot](https://www.trychroma.com/research/context-rot) 报告测了 18 个模型，结论相当一致：性能会随输入长度增长而退化，而且退化不均匀，常常出现在很意外的位置。大海捞针（NIAH）测试不一定能准确反映 LLM 的长上下文能力，性能退化可能比我们想象得更早。

更反直觉的是，他们发现结构连贯的 haystack 反而不如打乱顺序的 haystack。也就是说，让上下文读起来更顺，有时会损害检索性能。这里没有一个简单的"装满就崩"阈值，注意力召回本来就是很难预测的问题。

Anthropic 在 [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) 里把这件事拆成两个拮抗的概念：模型受益于充足信息，也会受 Transformer 结构限制出现上下文腐烂。

无论是 compaction、notes 还是 subagent，本质上都在找一组更小、更高信号的 token，让目标达成概率最大化。

这就是后面所有方案的前提。上下文会腐化，注意力有预算，所以"信息放在哪""怎么取出来""怎么写进去""什么时候隔离"都不是锦上添花，而是处理复杂任务的必需。后面会分别讨论记忆结构、检索与写入，以及一些工程手法。

## Object Path：先确定记忆对象和记忆载体

先看存储。但别一上来就问用哪个数据库，先问一个更底层的问题：Agent 在运行时读写的到底是什么对象。Memory 全景图已经解释过为什么不同记忆适合不同载体：关系密集的语义知识适合图，操作性经验和规范适合文件，事实记录适合关系数据库。本文不再重复这套选型理由，而是讨论工程上该怎么实现，以及这些载体如何一起服务于当前窗口的调度。

如果直接从向量库、图数据库、文件系统开始聊，很容易把问题说成工具选型。Context Engineering 真正关心的是另一件事：信息如何以可治理的形态留在窗口外，又如何在需要时以最小高信号集合回到窗口里。所以这一节先定义 memory object，再讨论它如何投影到向量索引、全文索引、图、文件和关系数据库。

### Memory Object 与 Metadata 联系起各种数据结构

无论最终落在向量库、数据库、图还是文件系统，运行时读写的基本单位都不应该是一整块未经治理的文本，而应该是带元数据的 memory object。一个 memory object 至少需要这些字段：

- `type`：persona、conversation、experience、knowledge、rule 等。
- `scope`：用户级、会话级、项目级、全局级。
- `content`：被保留的事实、规则、经验或摘要。
- `source`：来自哪次对话、哪次工具调用、哪份外部材料。
- `timestamp`：写入时间和最后更新时间。
- `confidence`：这条记忆有多确定。
- `status`：active、stale、conflicted、deleted 等。
- `links`：它与哪些记忆重复、冲突、继承或 supersede。

Memory 全景图列过写入侧要回答的六个问题：写什么、写到哪、怎么写、如何更新、何时失效、何时写。这张字段表承载的，就是这些决定。

`status` 和 `links` 表达版本与冲突语义，`timestamp` 和 `confidence` 表达时效与不确定性。没有这些元数据，后面的版本更新、冲突消解、删除与回滚都很难做。

这些字段还会进入下一章的 metadata 过滤，也会被 promotion 用来保留 provenance。读写两条路径在这里接上了。

确定了 memory object 之后，再看存储载体会清楚很多。关系数据库、向量索引、全文索引、图和文件系统不一定是彼此替代的选项，更像同一份记忆的不同投影。关系数据库提供业务事实源负责保存对象本身、权限、状态和版本；全文索引负责精确词和编号；向量索引负责跨表述召回；图负责关系和路径；文件系统负责可读、可编辑和可恢复。

不要让向量库成为唯一事实源。向量库能回答什么和这个 query 相似，但它不天然知道这条记忆是否过期、是否被新版本取代、是否属于当前用户、是否可以被引用到答案里。存储结构要先保住治理语义，再把不同检索能力做成投影。后面无论用 SQLite、LanceDB、Postgres、图数据库还是文件系统，核心都是这个边界。

特别地，像 LLM Wiki 这样的系统也研究知识结构，也会思考如何更好地存储和召回，但这里不展开。它们更像知识沉淀系统或独立技术实验，可以单独写文章分析。本文只关心更基础的工程问题：一个 runtime 如何把外部记忆存成可治理对象，又如何在当前窗口里选择性使用它们。

### 向量嵌入语义检索

向量索引是 memory object 的一种重要投影，也是传统 RAG 的入口。但别把它想成一个把文本丢进去就会让 AI 变聪明的"语义搜索盒子"。真正落地时，效果往往不取决于数据库名字，而取决于几件更朴素的事：文档怎么切，向量怎么生成，索引怎么建，原文和元数据有没有留住，以后换模型、换库时会不会把自己锁死。

先讲这些底层选择，再看三档工程方案。

#### 向量嵌入检索的注意事项

向量嵌入检索解决的是一类很具体的问题：把 query 和文档片段映射到同一个向量空间里，再用距离或相似度找出语义相近的候选。自然语言改写、模糊需求、跨表述召回，是它擅长的地方。错误码、函数名、合同条款编号、人名、项目代号这类精确匹配，单靠 dense embedding 经常会漏掉。后面会讲混合检索，这里先只看向量库本身怎么建。

Chunking 最容易被低估。固定长度切分省事，但容易把标题、表格、代码块和段落关系切断；按结构切分更稳，比如按 Markdown 标题、段落、函数、章节、页面或表格单元切。

chunk 太大，向量会变成一团平均语义，真正有用的句子被稀释；chunk 太小，又会丢上下文，召回后还得靠相邻片段或 parent document 回填。比较稳的做法是让小 chunk 负责命中，再用标题路径、页码、章节、父文档 ID 把上下文补回来。chunk 不该是一段裸文本，它应该是带出处的检索单元。

Embedding 模型也不能随手挑。通用模型可以覆盖很多文档，但代码、医学、法律、内部制度这类文本有自己的术语密度。维度越高不等于效果越好，它会直接影响存储、内存、索引构建和检索延迟。

更麻烦的是版本治理。embedding 模型一换，旧向量和新向量通常不能混在一起比较。至少要记录 `embedding_model`、`embedding_dim`、`distance_metric` 和索引版本，否则后面排查召回问题会很痛苦。

索引层也是取舍。小数据量可以 exact KNN，也就是把所有向量算一遍距离，简单、可解释，规模上来后就慢了。ANN 用近似换速度。HNSW 通常召回好、查询快，但吃内存，构建和参数也更敏感；IVF 先把向量分桶，再在相关桶里找，内存和速度更可控，但需要训练或聚类，召回更依赖参数。这里没有神奇按钮。你调的是延迟、内存、构建成本和召回率之间的平衡。

Metadata 和 provenance 也要在写入时留住。向量库只返回相似，不天然知道这段话属于哪个租户、哪个项目、哪个版本、哪一页、是否已经删除、用户是否有权限看。写入时没有把这些信息结构化，检索时就只能靠补丁。至少要保留 `doc_id`、`chunk_id`、标题路径、source URI、页码或行号、更新时间、权限标签、删除状态和 parser 版本。RAG 的答案要能追溯，不然召回再准也很难进入可靠系统。

评估也要早做。别只看 demo 问题。准备一组真实 query，覆盖语义改写、精确术语、跨文档、无答案、旧版本、新版本、表格和代码块。看 Recall@k、MRR、引用是否支持答案，也看删除和权限是否生效。向量检索的很多问题其实和模型无关，而是文档解析错、chunk 切坏、metadata 缺失，或者评估样本太干净。

#### 三档检索底座一：Local-first：SQLite + FTS5 + sqlite-vec / sqlite-vss / vec1

如果目标是本地 Agent、个人知识库、离线工具，或者小团队先跑通传统 RAG，SQLite 是一个很实用的起点。它不靠性能天花板取胜，胜在把事情做小：文档表、chunk 表、元数据、索引状态、删除状态、任务记录都可以放在一个本地数据库里，随应用一起分发。

SQLite FTS5 是官方全文检索扩展，带 `bm25()` 排序函数。它正好补 dense embedding 的短板：精确词、编号、异常码、产品名、制度条款。

向量侧有三个方向，成熟度不一样。`sqlite-vec` 是第三方 SQLite 向量扩展，纯 C、依赖少，适合本地嵌入；但它仍是 pre-v1，要接受接口变化，项目文档也把它称作 `sqlite-vss` 的 successor。`sqlite-vss` 是更早基于 Faiss 的路线，作者已经把主要精力转向 `sqlite-vec`，新项目不太适合启用它。

`vec1` 是 SQLite 官方新的向量扩展，文档已经列出 ANN、L2 和 cosine 距离接口，也提醒测试和优化还不充分。方向值得关注，但首版采用前要验证语言绑定、发布节奏、平台打包和索引能力。

这一路线最好用在小边界里：每个用户一个库，或者每个小项目一个库。查询时可以并行跑 FTS5 和向量检索，再在业务层做简单融合和 rerank。边界也很窄。SQLite 不适合作为高并发、多租户、集中审计的检索服务；向量扩展还在演进；权限和索引一致性都要自己治理。它最舒服的位置，是 local-first 的基础设施，不是企业搜索平台。

升级信号也很清楚：多人共享、集中权限、后台索引队列、查询审计、跨项目统计，或者本地文件分发已经管不住版本。到了这一步，就该往服务化走。

#### 三档检索底座二：轻服务化：LanceDB

LanceDB 可以放在 SQLite 和重型搜索基础设施之间。它仍然能本地跑，也方便轻量部署，但向量检索、全文检索、hybrid search、rerank 的路径更直接。官方文档里的 hybrid search，就是把 vector search 和 full-text search 的候选合并，再通过 reranking 算法排序。比起自己从零拼 FTS、向量索引和融合逻辑，这条路省不少力气。

我更愿意把它理解成先把检索层服务化一点。原始文档、chunk 元数据、权限和任务状态仍然可以放在 Postgres 或业务数据库里，LanceDB 专心做检索索引。这样边界比较清楚：业务事实源归业务库，检索能力交给检索组件，答案生成层只拿统一格式的候选。

但它不会替你处理权限、审计和图谱。多租户隔离、ACL 过滤、删除传播、索引重建状态，最好仍由外层服务控制。如果团队已经决定要做内部知识库服务，而不是单机工具，LanceDB 是一个不错的过渡层：比纯 SQLite 更像服务，又比一上来引入大规模向量数据库轻。

继续往上走的信号，是数据量、并发、租户隔离、SLA、备份恢复、搜索运营开始成为日常工作。到那时，检索层就不再只是一个库了，才有必要考虑更专门的向量数据库或搜索平台。

#### 三档检索底座三：Postgres-centered service：Postgres + pgvector + FTS / ParadeDB

到了内部服务阶段，我会优先认真看 Postgres。原因很朴素：很多团队本来就用它管理用户、权限、文档元数据、任务、审计、版本和删除状态。把传统 RAG 的一部分能力放进同一个数据库生态，通常比额外引入几套服务更稳。

`pgvector` 把向量相似度检索带进 Postgres，支持 exact search，也支持 HNSW、IVFFlat 这类 approximate search 索引。Postgres 自带全文检索，可以用 `tsvector`、`tsquery`、ranking 函数处理关键词召回。这样一来，文档事实源、权限过滤、事务一致性、向量检索、全文检索都能留在一套数据库和 SQL 边界里。对内部知识库来说，这比多画一个组件更有用。

但 Postgres 也不是完整搜索引擎。Postgres FTS 的体验和 Elasticsearch / OpenSearch 那类搜索平台不是一回事；dense + lexical 的融合通常还要自己写 SQL 或业务编排；数据量特别大时，专用向量库的扩展边界会更高。

如果希望在 Postgres 内得到更接近搜索引擎的 BM25、faceted search、hybrid search 体验，可以关注 ParadeDB 的 `pg_search` 路线。它的吸引力在于不急着把系统拆成好几套服务，同时补上 Postgres 原生搜索的一些短板。

这一层适合小团队到中等规模的内部服务，尤其适合已经有 Postgres 运维经验、又很在意权限和审计的场景。再往后，Qdrant、Milvus、OpenSearch、Elasticsearch、Neo4j 这些选择当然存在，但那基本进入专门搜索基础设施或图基础设施的领域，需要更明确的规模压力和运维投入。对本文讨论的适度工程化 RAG 来说，先把这三层吃透，已经够用了。

这三档讲的是怎么把传统 RAG 的存储和索引底座建出来。它们不是三套互斥方案，而是工程边界随规模变化的三个阶段：本地时把事情做小，轻服务化时把检索层拆出来，内部服务阶段把权限、审计和事务一致性放回稳定的业务数据库生态。底座建好之后，真正决定当前窗口质量的，是怎么取准，比如稀疏加稠密混合、metadata 过滤、query rewrite 和 rerank，放到下一章讲。

### 图存储：先验证图检索收益，再选择图数据库

图的难点不在查询，而在怎么把非结构化文本变成图，以及图怎么随新信息增量更新。实体关系抽取要回答谁、和谁、什么关系、何时，通常还要靠一轮 LLM 调用。这一步质量决定图能不能用；抽取错了，再漂亮的多跳遍历也只是在错误的边上推理。

运行时遍历要控制跳数和扇出。多跳关联是图的长处，但不加限制地遍历，很快就会把成百上千个节点拉进候选集，最后违背"最小高信号集合"。

增量更新也麻烦：要做实体消解，判断"老王"和"王经理"是不是同一个节点；还要处理边的时效，判断旧关系是否已经失效。

#### 阶段一：用 LightRAG 先验证图检索收益

所以图存储应该从"关系型问题是否真的需要图检索"开始。早期可以只在 SQLite 或 Postgres 里保留简单的 `entity`、`relation`、`mention`、`chunk_link` 表，把它们当作可审计的关系投影。也可以更临时一点，用 JSON、NetworkX、内存图、LlamaIndex 的默认存储，或者 LightRAG 的轻量存储先跑实验。这样做不优雅，但足够验证很多问题：实体是否稳定、关系是否可抽取、用户是否真的会问多跳问题、图召回是否比 BM25 + dense + rerank 更好。

LightRAG 适合做第一轮收益验证。它把实体、关系和 chunk 检索包装成比较直接的 GraphRAG 通道，并提供 `local`、`global`、`hybrid`、`naive`、`mix` 这类查询模式。

这里最有用的是它能让你很快比较"只走传统 chunk 向量检索"和"把图结构也纳入召回"之间的差异。比如 `naive` 可以近似传统 RAG baseline，`mix` 则会把 local、global 和 naive 结果合并，更适合观察图检索是否真的给当前语料带来额外信号。

它本身也是一个很有趣的独立项目。如果只需要尽快跑一个能用的 RAG MVP，或者希望做一些 RAG 研究，它是很好的选择。

这个阶段不该以"已经接入图数据库"为验收，而应该输出一组可复现的判断：哪些 query 真的需要图，哪些实体和关系抽得稳定，哪些多跳路径可以解释答案，哪些错误来自抽取而不是检索。验收也要朴素：在关系型、多跳、跨文档问题上，GraphRAG 是否比 BM25 + dense + rerank 有稳定提升；如果只在少数 demo 问题上好看，就让它继续留在实验里。

从 Context Engineering 的角度看，轻量图检索实验的价值，是让系统学会一件事：什么时候应该把关系路径提升进 Working Memory，什么时候只需要原文 chunk。只有这个选择做对，图才是在节省注意力预算，而不是把更多结构化噪声搬进窗口。

#### 阶段二：Neo4j 作为真正的图基础设施

如果图检索被证明有稳定收益，下一步也不一定立刻上 Neo4j。只有当图谱开始成为核心资产，或者它会被多个功能复用时，Neo4j 这类图数据库才真正有意义。这里的核心变化是：图不再只是 RAG 的一个召回通道，而是系统里可查询、可维护、可解释、可运营的一类基础数据。

Neo4j 的价值在成熟度。它有完整的 property graph 建模、Cypher 查询语言、可视化工具、图算法生态和相对成熟的运维路径。对组织、人、项目、客户、合同、系统、风险、流程这些长期存在的实体来说，Neo4j 能把关系从抽出来辅助检索提升为可以被产品和业务系统共同使用的结构化资产。

它也在向 GraphRAG 靠近。Neo4j 支持向量索引，可以把 embedding 放在节点属性上做向量相似度查询；Neo4j GraphRAG Python 则提供面向 Neo4j 的 retriever、Text2Cypher、hybrid 检索和 LLM 组合方式。也就是说，Neo4j 不只是一个图遍历后端，它可以同时承载一部分向量召回、图召回和 Cypher 查询，把 GraphRAG 从实验工具推进到更稳定的服务边界里。

但它确实重。引入 Neo4j 意味着要认真设计 schema、约束、索引、权限、备份、迁移、增量同步和删除传播，会带来很高的运维成本。Neo4j 的合理位置是"图谱已经被证明是长期资产，所以需要真正的图基础设施"。在那之前，LightRAG 更像低成本探针；Neo4j 则是验证通过后的承载层。FalkorDB、Memgraph 也可以作为 Neo4j 之外的轻量图服务候选，不过这里就不再展开了。

### 文件系统：目录布局与导航

文件系统是被低估的记忆载体，但它的价值不只是便宜存文本。对 Agent 来说，文件系统更像一种 Agent-native 的记忆协议层：Agent 天生会 `read` / `write`，会跑 `glob` 和 `grep`，会顺着路径、目录、文件名和 Markdown 标题导航。

路径本身可以成为索引，Markdown 本身可以成为人和 Agent 都能读写的记忆格式。工程重点也就不只是把内容存进去，而是让 Agent 高效导航，不必把全部内容一次性拉进上下文。

这也是为什么很多真实工具都在往仓库里的 Markdown 上收敛，把它当作一种上下文协议。Claude Code 用 `CLAUDE.md` 承载项目上下文，也支持 auto memory；Codex 用 `AGENTS.md` 做分层指令；Gemini CLI 使用 `GEMINI.md`。

GitHub Copilot 支持 `.github/copilot-instructions.md` 和路径级 instructions；Cursor Rules 则把规则放进 `.cursor/rules/*.mdc`。

这些文件名不同，但抽象很接近：把稳定规则、项目约定、目录知识和工作偏好放在 Agent 启动或按需检索时能找到的位置。Claude 的 memory tool 更进一步，直接把跨会话记忆做成一个可读写的文件目录，让 Agent 在需要时读回，而不是预先把所有记忆塞满窗口。

文件型记忆最好分层，不要把所有 Markdown 都混成一锅。

- 稳定指令层：比如 `CLAUDE.md`、`AGENTS.md`、`GEMINI.md`、`.github/copilot-instructions.md`、`.cursor/rules/*.mdc`，适合放项目约定、架构边界、测试命令、代码风格和长期有效的操作规则。
- 演化笔记层：比如 `NOTES.md`、auto memory、`/memories/*`，适合放 Agent 从纠错、用户偏好、项目事实和反复出现的问题里沉淀出的跨会话知识。
- 运行态工作层：比如 `todo.md`、scratchpad、handoff artifact、evidence 或 log 摘要，适合放当前任务目标、阶段状态、失败痕迹和可恢复引用。

这三层的生命周期不同。稳定指令应该由人类明确维护，不能被一次任务里的临时结论随手改写；演化笔记可以由 Agent 辅助更新，但需要 provenance、时间戳和冲突处理；运行态工作文件则更接近 Working Memory 的外溢，任务结束后可以归档、压缩或丢弃。分层的意义，是防止临时 scratchpad 升级成长期事实，也防止 Agent 自动写入的记忆污染团队共享规范。

文件系统还有一个向量库很难替代的优势：它天然可治理。文件可以 diff，可以 code review，可以回滚，可以被权限隔离，也可以被人类直接编辑。对规则、经验、操作手册、项目约定这类需要共同维护的记忆来说，这种可见性很重要。黑盒索引也许更擅长相似度召回，但它不擅长解释"这条规则为什么存在、谁改过、什么时候生效、能不能删"。文件系统至少把这些问题放回了一个成熟的工程工作流里。

但文件系统记忆也不是把所有 Markdown 自动加载进上下文。好的文件系统记忆不是更长的 prompt，而是更清楚的导航：入口文件要短，目录结构要清楚，命名要可预测，正文要能被 `grep` 命中，长文档要保留目录、标题路径和引用链接。

会话开始只加载稳定入口，需要细节时再按路径展开；不需要时只保留文件名、行号、URL 或摘要。这种 just-in-time 的读法，和向量检索、图遍历一样，都是把 L2 记忆投影进 L1 工作台的一种方式。

它的特殊优势在于，人类和 Agent 可以共同维护同一套文本结构，维护成本远比数据库低。它的缺陷也明显：基本只适合本地的 Single Agent 治理。

## Read Path：从外部记忆到 Working Memory

有了存储结构，下一步就是把外部信息接回当前推理现场。前一章讲的是外部记忆怎么存成可治理对象，又怎么投影到关系库、全文索引、向量索引、图和文件系统里。Read Path 关心的是另一侧：任务发生时，runtime 怎么从这些投影里挑出少量证据，让它们带着来源、权限和版本信息进入 Working Memory，去指导真实决策。相比于 Memory 全景图的简单介绍，这里我们需要将整个工程问题讲清楚，

### 检索是 Pipeline，不是一次 Search 调用

从 Context Engineering 的角度看，检索不是一次 Search API 调用，它对应的是 Rewrite / Select / Compress 的整个流程。外部记忆里也许有很多相关内容，但当前窗口只需要少数高信号片段。

一条稍微完整一点的 read path，大概长这样：

```text
query / task state / scope / ACL / time
  -> query plan
  -> retrievers
  -> hard filter / fusion / dedup / rerank
  -> compression / evidence pack
  -> Working Memory
```

这里每一步都接着前文的存储结构。`scope`、`ACL`、`status`、`timestamp`、`links` 决定候选能不能被看见；全文索引、向量索引、图和文件系统决定候选能不能被找到；provenance 决定候选能不能进入答案；版本和冲突信息决定它进入 Working Memory 时要不要被标记或者在进入之前就被筛选。

RAG 最早被提出时，重点是把参数记忆和可更新的非参数记忆结合起来。但真正的检索增强不止是"向量库 + 生成模型"，还需要一条可以拆分、替换、评估和治理的 pipeline。

这也解释了为什么 Read Path 后面会接 Write Path。Promotion 只是把外部证据临时投影进 Working Memory，服务眼前任务；Persistence 才是把当前任务产生的新事实、新规则、新经验写回长期记忆。读和写在 metadata 与 working memory 上接壤。

### 检索方式：Read Path 的最小动作单元

讨论混合检索前，先把最小动作单元讲清楚。很多系统看起来复杂，底层通常就是几种能力的组合：按路径找、按字符串找、按词项找、按向量找、按结构找、按关系找。数据库和索引系统只是这些能力的载体，不是能力本身。

#### 文件路径、目录导航与 grep/glob

最朴素的检索其实是导航。已知文件名、目录、扩展名、模块路径、配置名、错误信息、函数名时，路径检索、glob、regex、grep 往往比向量检索更可靠。

它的工作方式很直接：先根据路径、文件类型或 ignore 规则缩小搜索空间，再用字符串或正则表达式匹配内容。`ripgrep` 这类工具的价值不在语义，而在确定性和速度。递归搜索目录、支持正则、默认尊重 `.gitignore`，这些特性很适合在大代码库或文件树里快速定位候选。

这类检索的缺点也明显：它需要字面线索。用户问"登录失败的处理逻辑在哪里"，但代码里实际叫 `AuthChallenge`，grep 可能找不到。可一旦 query 里有错误码、配置键、接口名、日志片段、文档标题，确定性搜索通常应该先于语义搜索。

尤其是 Coding Agent ，一套优雅的 Embedding 方案不一定有价值，根据一点初始信息进行反复多轮 glob/grep 往往就能定位错误。这就是 Claude Code 一直以来的做法，搭配上专用针对性的训练，能解决非常复杂的代码任务。

#### 倒排索引、关键词匹配与 BM25

grep 是逐文件扫描。全文检索系统则会提前建倒排索引：把 token 映射到出现过它的文档或 chunk。查询时不必重新扫所有文本，而是直接从词项表里找到候选，再按相关性排序。

BM25 是这条路线上最常见的排序方法，也是传统搜索引擎使用多年的技术。它大致综合三类信号：query 词在文档里出现得多不多，词本身在整个语料里常不常见，文档长度会不会让长文天然占便宜。术语、编号、异常码、人名、接口名、制度条款、产品代号，这些查询 BM25 往往被 dense embedding 更好用。

所以 SQLite FTS5 (SQLite Extension - Full-Text Search)、Postgres 全文检索、OpenSearch、Elasticsearch 这类系统仍然重要。即使系统已经有向量库，也不该把关键词检索当成旧时代技术。对 Agent 来说，BM25 是低成本、高可解释、容易和权限过滤结合的候选生成器。

#### n-gram 与 sparse n-gram

n-gram 检索介于 grep 和全文检索之间。它不理解语义，也不一定按自然语言分词，而是把文本拆成连续的字符片段。trigram 就是长度为 3 的字符片段。这样做的好处是，它适合子串匹配、模糊拼写、相似字符串和正则候选剪枝。至于 sparse n-gram 就是抛弃一些不怎么有区分度的片段，增强检索效率。

Postgres 的 `pg_trgm` 就是典型例子：它用 trigram 相似度支持文本相似匹配，也可以通过 GiST 或 GIN 索引加速查询。对产品名拼写不稳定、文件名搜索、短字符串匹配、日志片段定位这类问题，trigram 往往比完整分词更合适。

Cursor 的 fast regex search 也可以放在这个家族里理解。它为 Agent 工具加速大仓库 regex 搜索：先用 sparse n-gram 之类的索引方法筛掉大量不可能匹配的文件，再对候选做精确正则匹配。它不替代 grep，只是让精确工具在大规模代码库里仍然能被 Agent 高频调用。并和 Cursor 引以为傲的代码向量语义数据库协同工作。

#### 向量语义检索

向量语义检索解决的是跨表述召回。它通常用 encoder 把 query 和 document chunk 分别编码到同一个向量空间里，再用 cosine、dot product 或 L2 distance 找最近邻。用户说"客户退款流程"，文档写"售后退费 SOP"，只要 embedding 空间学到了相近语义，就可能被召回。

Dense Passage Retrieval 是这条路线的经典节点。它把开放域问答里的问题和 passage 分别编码成 dense vector，再用向量相似度召回候选。工程上，规模小可以 exact KNN，所有向量算一遍相似度；规模上来就要 ANN（近似最邻近），用近似换速度。

HNSW、IVF、PQ 分别是三种最为核心的解决这个问题的算法。HNSW (Hierarchical Navigable Small World) 基于多层图结构的近似最近邻（ANN）搜索。有着极高的查询速度和极高的召回率，也伴随极高的内存消耗用来存储图结构。IVF (Inverted File Index) 则是经典的分桶然后组内暴力索引的方法。PQ (Product Quantization) 则是唯一的有损压缩方法，通过降低维数减少开销，但精度损失严重，一般和 IVF 联合使用。Faiss (Facebook AI Similarity Search) 则是开源的，用于组织这些算法的有效高层工具。

随着 RAG 架构的普及，开发者倾向于“少引入新组件”，因此传统关系型数据库纷纷通过插件拥抱向量能力。这非常适合那些需要在单一数据库中同时保证 ACID 事务与向量检索的场景。pgvector (PostgreSQL Extension) 以及 sqlite-vec (SQLite Extension) 就是其中的经典扩展。至于需要更大规模的服务，Elasticsearch / OpenSearch / Redis 这种原生就提供混合检索的框架是值得选择的，他们的侧重点略有不同。

向量检索的误区是把"语义相近"当成"答案正确"。它擅长自然语言改写、模糊问题、跨语言或跨表达召回，但对错误码、函数名、合同编号、短 query、罕见实体并不天然可靠。它返回的是相似候选，不是经过验证的证据。因此向量检索最好被看作候选生成器，而不是最终答案选择器。

#### LSP 与代码索引

代码不是普通文本。函数、类、接口、引用、定义、类型、调用关系、模块边界，本来就是结构化信息。如果只把代码切成 chunk 再做 embedding，就会丢掉很多编译器和语言服务器已经知道的东西。

LSP 的价值在这里很明显。Language Server Protocol 把补全、跳转定义、查找引用、文档符号、workspace symbols、诊断等语言能力标准化，让编辑器和工具可以复用语言服务器。对 Agent 来说，LSP 检索回答的是这个符号在哪里定义、哪里被引用、当前文件有哪些结构化实体，不是哪段文本看起来更相似。

如果有 LSP、AST 这样的符号关系可以利用，那为什么不呢？

#### 图检索与关系路径

图检索处理的是实体和关系。它关心的不是某个 chunk 是否相似，而是实体之间如何连接：人属于哪个团队，系统依赖哪个服务，合同对应哪个客户，故障影响哪些组件，某个概念在哪些文档里以不同名字出现。

GraphRAG 不是图数据库。图数据库只是承载层，真正的检索单元包括实体邻域、关系路径、多跳遍历、社区摘要、Text-to-Cypher、模板查询等。LightRAG 是相对轻量且适合研究的技术选型。当然我们也可以选择自己研究图数据库的算法，然后自己考虑编排（Llamaindex 是非常常用的工具）。不过图检索不一定有价值且大概率很昂贵，建议把图和 BM25、dense、reranker 放在同一评估集上比较。只有关系型、多跳、跨文档问题有稳定收益时，再把图谱升级成基础设施。


### 从单路检索到混合检索与 Agentic RAG

理解了这些最小单元，再看工程系统会清楚很多。所谓混合检索，就是让不同检索器各做自己擅长的事，然后在统一候选层合并。随着各种加速算法的成熟，把所有策略全跑一遍也没那么贵，难点在于如何聚合，给 Agent 带来信息而不是噪声。

混合检索里最容易被低估的是 hard filter。权限、租户、项目、时间、删除状态、版本状态，这些不该交给 reranker 猜，而应该在召回前后作为硬约束生效，我们有 metadata 就应该去利用。企业知识库里，泄露无权限文档比少召回一个 chunk 严重得多。

候选合并之后，通常还要 dedup、fusion 和 rerank。Dedup 处理同一个 chunk 被多条路径召回、同一段文本在不同版本里重复出现的问题。Fusion 处理不同检索器分数不可比的问题。RRF 很适合这个阶段，因为它按排名融合，不要求 BM25 分数、向量相似度、图路径分数处在同一尺度上。各个数据库 hybrid search 文档，处理的基本就是这类稀疏、稠密和排序融合问题。虽然数据来源变得越来越多，现有的融合方式往往不那么够用，结合项目的需求本身设计合适的融合方案变得很重要。

Rerank 是另一层。第一阶段召回的目标是别漏，所以可以宽一点；第二阶段 rerank 的目标是排准，可以慢一点。以前的 reranker 往往还是一个神经网络模型，通常比 bi-encoder 更准，但成本更高。而现在，直接让一个 LLM 充当 Reranker 也是常见的情况，毕竟 token 在越来越便宜，还训练干嘛。

再往前走，就是 Agentic RAG。传统 RAG 是一次 retrieve-then-generate；Agentic RAG 则把检索变成多轮动作：计划、检索、阅读、发现缺口、改写查询、再检索、对照证据、生成答案。它用更多 test-time compute 换召回覆盖、多跳能力和自我校验。换句话说，这也是一种检索侧的 test-time scaling。

Agentic RAG 是一个蛮有意思的话题，检索从一个确定性行为变成了多轮次的 Agent Action。ReAct 在思想上毫无疑问是他的开篇之作，Thinking 和 Acting 交织，反馈指引下一轮行动。IRCoT 和 Search-o1 将同样的思想扩展到了检索领域。FLARE 用即将生成的内容预测自己缺什么，再主动触发检索。Self-RAG 也是类似的思路，但将“检索与反思自己的检索结果”放到了训练阶段学习，而不是依赖外部启发式方法。 检索在 Agentic RAG 中从一个前置步骤变成了一段自我往复的循环。从 Feedback 信息的角度来看，这绝对是正确的选择。

工程上要小心两个问题。第一，Agentic RAG 不是免费收益，它会增加延迟、成本和不确定性。第二，多轮检索不等于把更多内容塞进窗口。每一轮都应该产出更好的候选、更窄的 scope 或更明确的缺口；做不到这一点，只是在扩大噪声。

### Agentic RAG 之后：让结构重新进入检索

Agentic RAG 解决的是“什么时候检索、检索什么、检索后如何继续行动”，但它没有自动解决另一个老问题：文档并不是一堆无序 chunk。如果底层仍是扁平文本块，Agent 只是用更高成本反复搜索，仍可能漏掉长文档里集中、连续、有层级分布的证据。很多答案并不是随机散落在全文，而是沿着标题层级、段落顺序和局部叙事组织起来。人读文档通常先看目录、标题和小节位置，再进入某个区域连续阅读；RAG 如果只反复猜关键词、命中零散片段，就很容易错过同一节里没有被 query 明确说出的要求。

RAPTOR 是较早把这种结构显式做进检索的一条路线。它先把原始文本切成叶子节点，再递归 embedding、聚类、摘要，构造一棵从局部片段到高层摘要的树；查询时可以在树上逐层走，也可以把全树节点摊平后在不同抽象层级中一起检索。这样做的好处是，系统不只拿到最相似的短 chunk，也能召回主题性、跨段落的上层节点，适合需要全局理解和多步综合的问题。不过 RAPTOR 的树是后处理生成的结构，不是文档原生结构；聚类和摘要会重写信息，可能压缩掉细节，甚至引入轻微幻觉。换句话说，它利用了结构，但结构来自模型对文档的再组织。

DeepRead 更接近把文档原本的结构还给 Agent。它把标题和段落都视为一等实体：标题构成全局导航骨架，带有子标题集合、直属段落数和 token 数，只有这层轻量骨架进入系统提示；段落则作为基本检索和阅读单元，避免答案被滑动窗口切碎。运行时只有两个工具：`Retrieve` 接收查询字符串，在段落级语义检索后返回段落坐标，并按上下文窗口适当外扩，给 Agent 一个带位置的预览；`ReadSection` 则按 doc_id、section_id 和段落范围返回连续原文。这样，Agent 可以先定位，再沿着确定区域顺序阅读，而不是在长文档里不断重新搜索关键词。

这两篇工作放在一起，其实说明 Agentic RAG 之后还有一条很重要的路线：不只是让模型多检索几轮，而是把检索对象从扁平文本块恢复成有位置、有层级、有阅读顺序的文档。RAPTOR 证明了多层抽象结构可以提升长文档 QA，但也提示我们，重写式树结构会带来信息破坏风险；DeepRead 则更保守地利用原生标题和段落顺序，把结构变成 Agent 的导航接口。**比 RAG 更好，往往也会比 RAG 更贵，这是 Agentic Search 带来的直接结果**；但当搜索开始结合意图理解、结构导航和连续阅读，它也会在一定程度上改变搜索本身。

### 工程增强：让检索更像 Runtime 决策

有了基本检索器和混合检索层，真正影响系统质量的，往往是一些工程增强。它们不一定像论文里的主贡献，但在生产系统里很要命。

第一是信息需求判断。Agent 要先判断当前问题属于哪类：精确定位、语义探索、结构化查询、关系多跳、时效版本、代码符号、无答案判断。不同信息需求应该走不同路线。问"错误码 E1027 是什么"时，BM25 和 grep 比向量更该先出场；问"有没有类似的客户退款政策"时，dense 更有价值；问"这个函数改了会影响谁"时，LSP 与 AST 更重要。

这就是 router 的价值。它把"该用什么检索器"变成 runtime 决策。LlamaIndex Router Retriever 这类工具提供了一个具体实现：让 LLM 根据 query 选择一个或多个 retriever。更成熟的系统也可以不用 LLM router，而用规则、分类器、查询日志和失败样本训练出路由策略。

第二是 query rewrite。用户输入通常不是好的检索语句。用户说的是任务，索引里存的是文档片段、代码符号、规则条目、实体关系和历史记录。中间有天然语义落差。很多时候 query rewrite 不是一改一，而是一改多：把一个问题拆成多个检索子问题，扩展不同说法，抽取关键实体，生成结构化 filter，再分别送进不同 retriever。关于 query rewrite 的研究不多（有趣的更是没有），RAG-Fusion/HyDE/Self-Retrieval 可以作为参考。

第三是 index-side recall enhancement。前文讲 chunking 时说过，chunk 太小会丢上下文，chunk 太大又会稀释语义。很多增强其实都是在修这个问题。Anthropic 的 Contextual Retrieval 是一个很好的工程例子：在写入索引前，为每个 chunk 生成一段 chunk-specific context，再把这段上下文和原 chunk 一起用于 embedding 和 BM25。Jina 的 Late Chunking 则反过来，先在长文本上产生 contextualized token embeddings，再池化出 chunk embedding，让 chunk embedding 保留更长上下文的信息。Parent document retrieval、sentence window retrieval 也是同一个思路：小粒度负责命中，大粒度负责回填。命中时用小 chunk 保持召回敏感，进入 Working Memory 前再带回父段落、相邻句子、标题路径或页码。

第四是 promotion。检索结果不应该原样进入上下文，而应该被整理成 evidence pack。一个好的 evidence pack 至少包括：和当前目标直接相关的摘录，必要的父级上下文，source URI、标题路径、页码或行号，版本和时间，置信度，冲突或过期标记，权限和可引用状态。promotion 同时提供本轮次的信息和 Agentic RAG 的未来决策。从而方便他的行动。

### 标准候选对象与 LlamaIndex 的价值

多路检索要想可治理，最好别让业务层直接依赖某个数据库或框架的返回值。BM25、dense、graph、LSP、文件导航，都应该被规整成一种候选对象。可以叫它 `RetrievalResult`：

```json
{
  "query_id": "q_20260611_001",
  "retriever": "grep|bm25|dense|lsp|graph|file",
  "doc_id": "doc_123",
  "chunk_id": "chunk_456",
  "source_uri": "file://policy/leave_policy.pdf#page=3",
  "title_path": ["人事制度", "休假政策", "年假"],
  "text": "召回文本...",
  "snippet": "高亮片段...",
  "score": 0.83,
  "rank": 4,
  "metadata": {
    "tenant_id": "tenant_a",
    "acl_tags": ["hr", "cn-office"],
    "status": "active",
    "version": "v3"
  },
  "provenance": {
    "indexed_at": "2026-06-11T10:00:00+08:00",
    "source_updated_at": "2026-06-10T18:00:00+08:00",
    "parser": "docling|ragflow|custom"
  }
}
```

这个对象的意义不是 JSON 格式本身，而是边界。检索器负责找候选，编排层负责过滤、融合、去重、重排和压缩，生成层只接收已经带来源和治理信息的证据。以后把 Chroma 换成 LanceDB，把 LanceDB 换成 Qdrant，把 LightRAG 换成 Neo4j GraphRAG，或者把 grep 换成更快的 sparse n-gram search，引入中间层来降低整个系统的耦合是面向未来所必须的。

LlamaIndex 的价值也可以放在这里理解。它不一定是最终生产系统的唯一框架，但它把很多检索编排抽象产品化了：retriever、router、query fusion、node postprocessor、response synthesizer、property graph retriever。读它的设计，能帮助我们把 RAG 从"调用一个向量库"提升到"组织一组可替换的 read path 组件"。当然，框架仅仅是编排系统的工具，不要让它取代我们对系统的理解以及我们对系统的要求。

## Write Path：从运行时事件到可重建知识资产

如果检索回答"什么该被带进来"，写路径回答的就是"当前发生的事和新增补的信息怎样变成以后可信的记忆"。后者往往更难，因为写错的记忆会在未来持续污染检索结果。

工程里的 Write 不应该被理解成一次 `add_memory` 调用，也不应该被理解成把文本 append 到向量库里。更准确的说法是：先决定谁是事实源，哪些只是 projection；再把事实源的一次变化提交成可审计状态；最后让向量索引、BM25、图、文件摘要这些投影异步更新或重建。投影是用于提升 read path 的召回和组织能力，但不应该单独承担真实记忆的职责。

换句话说，Write Path 的核心不是"把新信息存下来"，而是判断一条新信息如何改变系统对世界的当前解释：它是新增事实、旧事实的新版本、条件化偏好、冲突证据、短期事件，还是根本不该进入长期记忆。

### 写入什么：Promotion 与 Persistence 不是对等概念

进入 Working Memory 的内容，不一定值得写入长期记忆。Promotion 是把外部信息临时带进当前窗口，服务眼前任务；persistence 是把运行时产生的新事实、新规则或新经验写回长期记忆。前者可以相对激进一点，错了最多污染当前轮；后者必须保守，因为写错会污染未来很多次检索。

这个区分能避免一个常见误区：只要某条内容帮当前任务答对了，就把它沉淀成长期记忆。很多信息只是局部、临时、上下文相关，应该随着任务结束退场。真正适合 persistence 的，是稳定偏好、明确事实、可迁移经验、被反复激活的规则，或者用户主动纠正后形成的新约束。Persistence 的目标不是多记一点，而是让未来的 Read Path 只读到可解释、可回滚、可失效的信息。

### 何时写入：运行时记忆写入与知识库写入

讨论 Write Path 时，最容易混在一起的其实是两类完全不同的写入。

一类是运行时 Agent memory 写入：用户在对话里纠正系统，工具调用暴露出一个可迁移经验，某个项目约定被反复触发，或者用户明确给出偏好和身份事实。它的触发点是事件，重点是判断这条信息是否稳定、是否跨任务有用、是否应该更新 persona、rule、experience 或 episodic summary。

另一类是知识库 / RAG 写入：连接器拉到新文档，Wiki 页面改了，代码仓库有新 commit，业务数据库导出了一批记录，制度文件换了版本。它的触发点是 source change，重点不是让 LLM 自动记住，而是把外部事实源重新解析、规范化、切 chunk、计算 hash、更新 metadata，再刷新各种检索投影。

| 维度 | 运行时 Agent memory | 知识库 / RAG 写入 |
| --- | --- | --- |
| 输入来源 | 对话、工具调用、用户纠错、环境反馈 | 文档、Wiki、代码、工单、业务库、网页抓取 |
| 触发方式 | 事件触发：偏好出现、错误被纠正、经验可迁移、规则反复激活 | 来源变化：文件更新、connector 同步、定时刷新、CDC、人工发布 |
| 事实源 | memory object、用户 profile、规则文件、审计日志 | document store、对象存储、业务库快照、Git/Wiki 历史 |
| 写入对象 | persona、rule、experience、semantic fact、episodic summary | document、chunk、metadata、source hash、index manifest |
| 更新策略 | 保守写入，必要时请求确认或标记低置信度 | 默认可重建；增量更新是规模和新鲜度压力下的优化 |

这两类写入都叫 Write，但风险形态不同。运行时写入的危险，是把临时语境、幻觉、用户玩笑或一次性失败固化成长期偏见。知识库写入的危险，则是旧 chunk、旧向量、旧图边和旧摘要残留在投影里，让系统以为自己读到了最新事实。如果我们是在构建一个用户侧的 Agent，那么最常见的是运行时来记住偏好，而如果我们在构建服务，那么往往知识库的稳定与正确就是核心。

### Write Pipeline：同一个抽象，两条实现路径

一个可治理的写路径可以共用同一套抽象，但实现上最好拆成两条路径：

```text
runtime memory:
  event
    -> memory proposal
    -> gate
    -> normalize
    -> align
    -> commit
    -> project
    -> verify

KB / RAG ingestion:
  source change
    -> parse / chunk / hash
    -> staging index
    -> evaluate
    -> publish manifest
    -> retire old index
```

运行时记忆要先生成 `memory proposal`。proposal 是写入候选，不是已经落库的记忆。它至少要带上候选内容、来源、scope、type、confidence、TTL 或有效期、privacy / ACL、idempotency key，以及触发它的事件。这样系统才能区分用户明确说了一个稳定偏好，和模型从一次对话里猜到一个偏好。

`gate` 决定要不要写。这里要检查稳定性、复用价值、权限边界和来源可靠性。用户主动纠正、外部事实源变更、工具失败复盘，通常比模型自己总结出来的偏好更值得写。很多系统把写入拖到会话结束统一总结，但这会把许多细粒度证据混在一起。更稳的做法是按事件生成 proposal，再由 gate 决定是否提交、降级为 episodic summary，或者只留在当前任务日志里。

`normalize` 把候选转成稳定 schema：时间、实体、项目、用户、source URI、版本、权限、证据片段都要结构化。对知识库来说，这一步还包括解析文档、保留标题路径和页码、生成 chunk、计算 source hash、记录 parser version 和 embedding model version。

`commit` 才是真正的写入。它应该写 audit log，而不是只写向量库。commit 需要留下 `active`、`stale`、`conflicted`、`deleted` 等状态，记录 `supersedes` / `superseded_by`，必要时保留 `valid_at`、`invalid_at`、`observed_at`、`source_updated_at`、`committed_at` 和 `index_version`。写入更像一次小型状态提交，不是给全局变量赋值。

`project` 负责刷新投影。语义知识重新生成 embedding，术语和编号进入全文索引，实体关系更新图边，文件型记忆更新目录或摘要，Wiki 页面重新入库。投影可以异步，但必须能追踪；否则 read path 会召回旧向量、旧关键词或已经失效的关系。

`verify` 则是写入后的回归检查。新文档是否能被召回，旧版本是否被降级，删除是否传播到所有检索通道，权限过滤是否仍然生效，引用是否还能追到原始 source。即便是用户侧 runtime，也不应该只把核查责任丢给用户；至少要提供写入 diff、来源链接、撤销入口和 memory review，让自动写入保持可见。

知识库写入更像数据发布流程，而不是单条记忆提交。LlamaIndex 的 document management pipeline 用 `doc_id` 和 hash 判断文档是否变化，避免重复写入，并把 update / delete 做成文档级操作。工程上也常用 staging index、离线评估、index manifest 和蓝绿切换来发布新索引。这样做不花哨，但它把"这次知识库到底发布了什么"变成可验证状态。

### 写入最难的是对齐旧世界

`align` 是 Write Path 里最容易被低估的一步。新知识不自然是对旧知识的覆盖，"用户喜欢拿铁"和"用户最近开始戒咖啡"不是一个简单的二选一问题。系统需要知道两条记忆的时间关系、适用条件和冲突状态。把记忆当成"当前最优快照"去 overwrite，时间维度上的信息就被抹掉了。

这里至少有三类问题。

第一类是时间变化。旧事实未必是错的，它可能只是过了有效期。用户过去在上海工作，后来搬到杭州，这不是要把"上海"从历史里删掉，而是要标记它在什么时间段成立。Zep / Graphiti 这类 temporal graph 把 fact validity 和 provenance 放到中心位置，就是为了让系统能回答"现在什么是真的""当时什么是真的"，而不是只保留最后一次写入的结果。

第二类是事实矛盾。新旧事实如果不能同时成立，就应该进入 `conflicted`、`superseded` 或待确认状态，而不是强行合并。MemConflict 这类评测把 memory conflict 单独拿出来测，原因也在这里：长期记忆系统真正难的不是记住更多，而是在新证据到来时不把矛盾写成一致。

第三类是条件适用。偏好、规则和经验很少是全局真理。"这个项目用 pnpm"、"这位用户喜欢简短回答"、"这个客户不接受周五发布"都可能有 scope、时间和场景限制。没有条件字段，运行时记忆就会从帮助变成偏见。

所以 `align` 不只是 duplicate 检查和相似记忆合并。它要做实体消解、冲突检测、supersede 判断、条件收窄、时间对齐和遗忘传播。Mem0 这类记忆层产品把 add / update / search 做成显式 API，本质上是在产品化这段 pipeline；LangMem 把 memory operation 表达成"输入 conversation 和当前 memory state，再生成更新后的 memory state"，也说明写入不是单点插入，而是对已有状态的重写和整合。

由于我们需要 `align` ，那么如果 action space 只有 append 和 overwrite，系统很快会把历史事实、当前事实、冲突事实和临时结论混在一起。更稳的做法，是把写入动作显式化。

- `ADD`：新增一条没有直接冲突的记忆。
- `UPDATE`：在同一个事实或 profile 内更新字段。
- `SUPERSEDE`：新记忆取代旧记忆，但保留旧记忆的历史身份。
- `CONFLICT`：新旧信息互相矛盾，先标记冲突而不是合并。
- `ARCHIVE`：旧记忆不再默认召回，但仍可审计和恢复。
- `DELETE`：因错误、隐私、权限或用户请求彻底删除。
- `NOOP`：信息太临时、太不确定或已有等价记忆，不写入。
- `ASK_USER`：影响长期行为但证据不足时，请用户确认。

这组动作的意义，不是把 schema 变复杂，而是让系统承认写入有多种结局。长期记忆系统最危险的状态，是所有新信息都被包装成"更新成功"。

时间字段也应该服务于这个动作空间。`valid_at` / `invalid_at` 表达事实在世界中何时成立或失效；`observed_at` / `source_updated_at` 表达来源何时说过或更新过这件事；`committed_at` 表达系统何时把它写入记忆。前两者是世界时间，后者是系统时间。把它们混在一起，系统就很难解释为什么一条旧记忆仍然被召回，或者为什么新信息没有立刻覆盖旧信息。

### 索引是 Projection：工程上为什么经常重建知识库

这也是为什么真实工程里，知识库经常不是一直聪明地增量生长（当然研究增量生长的项目也是非常多，这绝对是一个值得思考的能力，尤其是对于用户侧 runtime 的较大知识库），而是反复重建、分区重建或蓝绿切换。听起来笨，但它更可验证。

增量更新真正难的不是插入一条新 chunk，而是处理一整串连锁问题：旧 chunk 是否已经 stale，删除是否传到向量索引、FTS 和图，旧 chunk 能不能和新 chunk 共存，embedding 模型换了以后旧向量还能不能比较，实体抽取模型改变后旧图边是否还可信。只要这些问题没有被显式建模，增量就会把知识库变成一层层历史残留。

所以小规模、本地、个人知识库、小团队 PoC，默认全量重建或按目录 / 文档集合分区重建往往更合理。重建时重新解析、重新 chunk、重新 embedding、重新跑 BM25 / graph extraction，然后用新的 index manifest 替换旧版本。旧索引先不删，验证通过再切流量；出问题就回滚。这比在旧索引上不断打补丁更容易定位问题。

增量更新适合另一类条件：数据规模大到全量重建太贵；freshness 要求高到不能等下一轮批处理；来源系统能提供可靠变更流；系统已经有 doc_id、source hash、delete tombstone、index version 和 projection lag 监控。Graphiti 则选择实时增量图谱，适合对话和事件流里持续变化的 temporal facts。增量有价值，但不是所有系统一开始都该追的目标。

图谱尤其应该谨慎。图检索收益还没被验证时，自动增量图谱很容易变成错误关系的放大器。实体消解、关系抽取、边的时间有效性、旧关系失效，全都比普通 chunk upsert 更难。很多时候，先把图作为可重建 projection，用固定语料离线生成和评估，比让运行时持续改图更稳。

### 写入评估：不要只测召回

最后，Write Path 的评估不能只看未来能不能召回。Memory 全景图里那套 L1-L6 框架，在这里可以落成更工程化的指标。

写入正确性看 `write precision`：写进去的候选到底有多少是真正稳定、有用、来源可靠的记忆。更新正确性看 duplicate rate、conflict detection、supersede accuracy：新旧事实有没有对齐，有没有重复堆积，有没有把冲突强行合并。投影质量看 projection lag、stale recall rate、delete consistency：提交以后多久能被读到，旧版本是否还在召回，删除是否传到所有索引。

治理能力还要看 rollback success、audit completeness、citation correctness 和 permission correctness。HaluMem 这类评测已经开始把 memory hallucination 拆到 extraction、updating、QA 这些操作层；LongMemEval 把长期多 session 里的知识更新、时间推理和 abstention 拉进评测；MemoryAgentBench 进一步强调 accurate retrieval、test-time learning、long-range understanding 和 selective forgetting；MemConflict 则把冲突记忆单独拿出来测。方向是对的：不要只问系统记不记得，而要问它写得准不准、改得对不对、删得干不干净、不确定时敢不敢停。

## 缓解上下文不足的工程手法

前面几章解决的是信息怎么进出存储。Context Engineering 更难的部分，在单次任务执行过程中出现：任务跑了几十轮，上下文逼近窗口上限，工作台开始变脏，这时该怎么办？

这一节只讨论运行时手法。先和[《给 LLM 戴上确定性枷锁的外围工程》](/blog/2026/03/20/building-agent-deterministic-constraints/)划清边界：那篇文章也讲 subagent、checkpoint、fork、worktree，但角度是限制即可靠性，把它们看成收窄失败半径的原语。

这里换一个角度，只看它们怎么帮助管理上下文：哪些信息留下，哪些换出，哪些隔离到别处。

### 压缩派：Compaction 与 Context Reset

最直接的办法是压缩（compaction）：把接近窗口上限的对话摘要掉，用摘要重启一个新窗口。

Anthropic 给的实践要点是：**保留架构决策、未解决的 bug、关键实现细节，丢掉冗余工具输出**；先最大化 recall（别漏掉相关内容），再优化 precision（去掉多余内容）。

Claude Code 的自动 compact 在上下文超过 95% 时触发，先清理较早的工具输出，必要时再摘要对话，并保留最近访问的 5 个文件。其中最轻量的一种形式是 tool-result clearing：接近 token 上限时，自动清掉上下文里陈旧的工具调用和结果。

Anthropic 在 [context management 的发布](https://claude.com/blog/context-management) 里给了一组数字：在一个 100 轮 web 搜索测试里，单是 context editing 就削减了 84% 的 token 消耗，带来 29% 的性能提升；配合 memory 工具一起用则达到 39%。

压缩的风险也明显：过度压缩会丢掉当时看不出重要、后来才显出关键的细节。context reset 是另一种处理方式。compaction 是原地压缩，同一个 agent 带着摘要继续跑，连续性更好；reset 则给下一个 agent 一个干净上下文，只通过一份结构化 handoff artifact 交接必要状态。reset 要付出 handoff 成本，但能切断前面几十轮噪音的影响。选哪一个，取决于你更想保留连续性，还是更需要一个干净起点。Claude 在 Runtime 默认使用 Compact，但对于 Dynimic Workflow 使用了后者，你的选择和你的任务本身息息相关。

错误也不一定需要立刻清理，虽然注意力的失效并非线性，但开发共识依旧认为我们应该在 Context 窗口稀缺后再考虑压缩或重置。把失败动作和对应 stack trace 留在上下文里，因为模型看到失败会隐式更新自己的信念，从而避免重复同样的错误。从某种意义上讲，错误恢复能力是判断 agentic 行为的清晰指标之一。

### 外部化派：把记忆挪到窗口外面

另一条路是不在窗口里硬扛，把信息外部化到文件系统。Manus 在 [Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) 里把文件系统称作"终极上下文"：容量大、天然持久，Agent 也能直接操作。

压缩最好是可恢复的。可以丢掉网页正文，但留下 URL；可以省略文档内容，但保留路径。这样需要时还能重新拉回来，避免 compaction 带来的不可逆损失。Anthropic 的结构化笔记（写 `NOTES.md`、维护 `to-do.md`）和 Claude 的 file-based memory 工具（如 Claude.md*2，rules，skills）也是这个思路：把记忆放到窗口外，需要时再读回来，并且跨会话存活。

长循环任务还有一个独立问题：目标漂移。Manus 观察到一个典型任务平均要 ~50 次工具调用。跑到后半程，最初目标会开始掉出注意力。这就是 Lost-in-the-Middle 在 agentic 场景里的表现。此时外部化一个`todo.md`，每完成一步就重写一次，把当前目标不断推到上下文最末端，也就是模型注意力最强的位置。这也是外部化的思想之一。

### 隔离派：Subagent 与 Checkpoint / Fork

最后一类是隔离。与其在一个上下文里硬塞，不如把任务切开。

Subagent 的工程价值不只是提速，更重要的是上下文隔离。子 agent 可以为了探索烧掉数万 token，但只向主 agent 返回一份 1000-2000 token 的压缩摘要（Anthropic 的数字）。主 agent 因此保持干净，详细搜索和探索过程被挡在外面。

Claude Code 的 [subagent](https://code.claude.com/docs/en/sub-agents) 给每个子 agent 独立的上下文窗口、定制 system prompt、受限工具集和独立权限。一个只读 reviewer subagent 如果没有 Edit 工具，不许改文件就不只是提示词，而是动作空间限制。

和隔离配套的是状态管理原语。Claude Code 的 [checkpointing](https://code.claude.com/docs/en/checkpointing) 在每次用户 prompt 时自动快照文件状态、跨会话持久、30 天后清理，支持五种 rewind（恢复代码 / 恢复对话 / 两者都恢复 / 从某点摘要 / 摘要到某点）。

会话本身是本地 JSONL（见 [how Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)）。`--continue` / `--resume` 在同一会话后面追加消息，`--fork-session` / `/branch` 则把历史复制成一个新会话、原会话不动。

从上下文管理角度看，这些原语提供的是换出和分叉能力：把一段上下文存到窗口外，需要时再换回来，或者从一个干净分支重新开始。

边界再说一次：subagent 隔离、tool-call 隔离、session 隔离作为可靠性原语的系统展开，在确定性枷锁那篇里讲过。这里只取它们的上下文管理视角，也就是把 subagent 当作 `Isolate` 原语。至于到底选 MAS、Single Agent 还是 Dynamic Workflow，已经进入 Agent 架构选型本身，我放到[认知结构那篇的 "MAS or Single Agent and Dynamic Workflow" 一节](/blog/2026/03/03/cognitive-architecture-to-agent-framework/#mas-or-single-agent-and-dynamic-workflow)里讨论。

## 结语：Write / Select / Compress / Isolate

这些手法看起来分散，但 LangChain 在 [Context Engineering for Agents](https://www.langchain.com/blog/context-engineering-for-agents) 里给了一个好用的分类框架，把它们收成四类动作：

- Write（写出去）：把信息存到窗口外，比如 scratchpad、外部化笔记、跨会话记忆。
- Select（选进来）：把相关信息拉进窗口，比如检索、`CLAUDE.md` 这类规则文件、记忆召回。
- Compress（压缩）：只留执行任务所需的 token，比如 compaction、摘要、trimming。
- Isolate（隔离）：把上下文切分到不同 agent 或环境，比如 subagent、沙箱。

对照前文，压缩派是 Compress，外部化派是 Write，检索那章是 Select，隔离派是 Isolate。

**Context Engineering 不是某个单点技巧，需要组合编排这些动作。**

Drew Breunig 还总结过四种常见上下文失败模式：poisoning（幻觉进入上下文并被反复引用）、distraction（上下文多到压过模型自身能力）、confusion（无关内容干扰回答）、clash（上下文里有互相矛盾的部分）。这些坑，基本都能对应到 write、select、compress、isolate 的某个动作没做好。

"怎么评估一个 Context 系统做得好不好"，我在 Memory 全景图里给过一套 L1-L6 框架：写入正确性、更新正确性、调用及时性、行为一致性、经验迁移可控性、不确定性处理。这里不重复。至于真实 runtime 怎么把存储、检索、写入、隔离这些动作装配起来，留给 teardown 篇逐个介绍。

Context Engineering 的重点不是把窗口做长，也不是堆一个向量库。更实用的标准是：**系统能不能说明哪些信息进入了当前上下文，哪些被压缩或换出，哪些被隔离在子 agent 里，哪些被写入长期记忆，以及这些决定为什么发生**。窗口会继续有限，注意力也仍然是预算。只要这个前提不变，把上下文当成有限资源来调度，就是 Agent 工程的一项基础能力。

## 参考资料

- Anthropic, [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- Anthropic / Claude, [Managing context on the Claude Developer Platform](https://claude.com/blog/context-management)
- Anthropic Docs, [Claude Code Memory](https://code.claude.com/docs/en/memory)、[Memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)
- Manus, [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- Chroma, [Context Rot: How Increasing Input Tokens Impacts LLM Performance](https://www.trychroma.com/research/context-rot)
- LangChain, [Context Engineering for Agents](https://www.langchain.com/blog/context-engineering-for-agents)
- LangChain Docs, [Memory overview](https://docs.langchain.com/oss/python/concepts/memory)、[Long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory)
- LangMem, [Conceptual Guide](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)
- OpenAI Help Center, [Memory FAQ](https://help.openai.com/articles/8590148-memory-faq)
- OpenAI Codex, [Custom instructions with AGENTS.md](https://developers.openai.com/codex/guides/agents-md)
- GitHub Docs, [Custom instructions for GitHub Copilot](https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions)
- Gemini CLI, [Provide context with GEMINI.md files](https://google-gemini.github.io/gemini-cli/docs/cli/gemini-md.html)、[Memory Tool](https://google-gemini.github.io/gemini-cli/docs/tools/memory.html)
- Cursor Docs, [Rules](https://cursor.com/docs/rules)；Cursor, [SemSearch](https://cursor.com/blog/semsearch)、[Fast Regex Search](https://cursor.com/blog/fast-regex-search)
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)
- ripgrep, [ripgrep](https://github.com/BurntSushi/ripgrep)
- Stephen Robertson and Hugo Zaragoza, [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
- SQLite, [FTS5](https://sqlite.org/fts5.html)、[vec1](https://sqlite.org/vec1.html)
- Alex Garcia, [sqlite-vec](https://github.com/asg017/sqlite-vec)、[sqlite-vss](https://github.com/asg017/sqlite-vss)
- PostgreSQL Docs, [Full Text Search](https://www.postgresql.org/docs/current/textsearch.html)、[pg_trgm](https://www.postgresql.org/docs/current/pgtrgm.html)
- Microsoft, [Language Server Protocol](https://microsoft.github.io/language-server-protocol/)
- Tree-sitter, [Tree-sitter](https://tree-sitter.github.io/tree-sitter/)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
- [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)
- pgvector, [pgvector](https://github.com/pgvector/pgvector)
- LanceDB, [Hybrid Search](https://docs.lancedb.com/search/hybrid-search/)；Qdrant, [Hybrid Queries](https://qdrant.tech/documentation/search/hybrid-queries/)；Weaviate, [Hybrid Search](https://docs.weaviate.io/weaviate/search/hybrid)；Elasticsearch, [Reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
- [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
- [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)；[Document Ranking with a Pretrained Sequence-to-Sequence Model](https://arxiv.org/abs/2003.06713)；BAAI, [BGE Reranker](https://bge-model.com/tutorial/5_Reranking/5.1.html)
- ParadeDB, [ParadeDB / pg_search](https://github.com/paradedb/paradedb)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)；[Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://arxiv.org/abs/2212.10509)
- [Forward-Looking Active REtrieval augmented generation](https://arxiv.org/abs/2305.06983)；[Self-RAG](https://arxiv.org/abs/2310.11511)；[Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
- [RAG-Fusion](https://arxiv.org/abs/2402.03367)；[Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)；[Query Rewriting for Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2305.14283)
- Anthropic, [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)；Jina AI, [Late Chunking](https://arxiv.org/abs/2409.04701)；LangChain, [Parent Document Retriever](https://python.langchain.com/docs/how_to/parent_document_retriever/)；LlamaIndex, [Sentence Window Retrieval](https://developers.llamaindex.ai/python/examples/node_postprocessor/metadatareplacementdemo/)
- [DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search](https://arxiv.org/abs/2602.05014)；[RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)
- LightRAG, [LightRAG](https://github.com/HKUDS/LightRAG)
- [SPLADE v2](https://arxiv.org/abs/2109.10086)；[ColBERT](https://arxiv.org/abs/2004.12832)
- LlamaIndex, [Router Retriever](https://developers.llamaindex.ai/python/framework/integrations/retrievers/router_retriever/)、[Reciprocal Rerank Fusion](https://developers.llamaindex.ai/python/framework/integrations/retrievers/reciprocal_rerank_fusion/)、[PropertyGraphIndex](https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/)、[Response Synthesizers](https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/)、[Node Postprocessors](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/)、[Document Management Pipeline](https://developers.llamaindex.ai/python/examples/ingestion/document_management_pipeline/)
- Microsoft GraphRAG, [CLI](https://microsoft.github.io/graphrag/cli/)；[Indexing Overview](https://microsoft.github.io/graphrag/index/overview/)
- Mem0, [Docs](https://docs.mem0.ai/overview)、[Quickstart](https://docs.mem0.ai/platform/quickstart)、[GitHub](https://github.com/mem0ai/mem0)
- Zep / Graphiti, [Graphiti docs](https://help.getzep.com/graphiti/getting-started/overview)、[GitHub](https://github.com/getzep/graphiti)、[Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956)
- Letta, [Memory](https://docs.letta.com/guides/agents/memory)、[Context hierarchy](https://docs.letta.com/guides/core-concepts/memory/context-hierarchy)
- Andrej Karpathy, [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- [HaluMem](https://arxiv.org/abs/2511.03506)；[LongMemEval](https://arxiv.org/abs/2410.10813)；[MemoryAgentBench](https://openreview.net/forum?id=DT7JyQC3MR)；[MemConflict](https://arxiv.org/abs/2605.20926)
- [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110)；[MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250)
- Neo4j, [Vector Indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)；[Neo4j GraphRAG Python](https://neo4j.com/docs/neo4j-graphrag-python/current/)
- Anthropic Docs, [Subagents](https://code.claude.com/docs/en/sub-agents)、[Checkpointing](https://code.claude.com/docs/en/checkpointing)、[How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)
- [《从记忆形成到记忆治理：Agent Memory 的全景图》](/blog/2026/03/21/agent-memory-panorama/)
- [《从智能体的认知结构到智能体框架》](/blog/2026/03/03/cognitive-architecture-to-agent-framework/)
- [《给 LLM 戴上确定性枷锁的外围工程：从 Claude Code 看 Agent Harness》](/blog/2026/03/20/building-agent-deterministic-constraints/)
- [《为什么 Output Token 更贵：从 KV Cache 到 Agent 成本工程》](/blog/2026/04/26/output-token-pricing-kv-cache-agent-cost/)
- [《拆开 Agent Runtime：记忆、上下文与隔离在真实系统里如何被装配》](/blog/2026/06/07/agent-runtime-teardown/)
