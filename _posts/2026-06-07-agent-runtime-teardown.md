---
layout: blog-post
title: "Agent Memory 与 Runtime 技术盘点：外挂记忆、运行时研究与框架内建能力"
title_en: "Agent Memory and Runtime Atlas: External Memory, Runtime Systems, and Framework Memory"
date: 2026-06-07 17:00:00 +0800
categories: [Agent 系统]
tags: [Agent Memory, Agent Runtime, Context Engineering]
author: Hyacehila
excerpt: "本文暂时只保留结构占位。后续会按外挂 Memory、Agent Runtime、框架内建记忆能力三类，逐个拆解真实系统的工作机理。"
excerpt_en: "A placeholder structure for future technical teardowns of external memory systems, agent runtimes, and framework-provided memory capabilities."
featured: false
math: false
---

> **写作状态：本文为占位草稿。** 现在只保留文章定位和标题坑位，不展开产品索引，不做长内容盘点。

# Agent Memory 与 Runtime 技术盘点：外挂记忆、运行时研究与框架内建能力

这篇文章不再沿用 L1/L2/L3 的测绘写法，也不做产品排行榜。后续真正要回答的是：一个系统到底怎么记、怎么取、怎么改、怎么忘。在这里我们用真实运行的系统的例子，来解释一个完整的 Agent Runtime、一个 Memory Framework 以及 Agent 开发工具，都是怎么来帮助用户管理上下文与长期记忆的。这将是对前面讨论 Research 和 Context Engineering 的实例化研究。

## 外挂 Memory 产品与研究（占位）

### Mem0 / OpenMemory

### Zep / Graphiti

### Letta / MemGPT

### TencentDB Agent Memory

### Cognee

### Supermemory

### MemOS

### Generative Agents

### Reflexion

### MemoryBank

### A-MEM

### LongMemEval

### HaluMem

### MemoryAgentBench

### MemConflict

### LongMemEval-V2

## Agent Runtime 研究与产品（占位）

### AIOS

### SWE-agent

### OpenHands

### Agent S

### OSWorld / BrowserGym

### Magentic-One / Magentic-UI

### Claude Code

### Codex CLI



### OpenClaw

### Hermes Agent

### Pi.dev

## Agent 框架提供的记忆能力（占位）

这一节讨论的是 Agent Framework 自己提供的记忆能力，而不是专门的 Memory Layer，也不是模型厂商在产品里做好的个人记忆。换句话说，这里关心的是：当开发者选了一个框架之后，它到底帮你处理了多少记忆问题，又把多少复杂度留给了开发者。

我会按五个维度拆它。**短期与裁剪**看的是会话历史如何进入上下文窗口，框架是否提供截断、摘要、compaction 或 history reducer。**长期记忆**看的是历史信息能不能跨会话沉淀成可复用事实，还是只提供一个外接 provider 的接口。**用户记忆**看的是有没有 user、session、app、agent 这类作用域隔离，避免不同人、不同任务、不同产品入口的记忆串味。**工作记忆**看的是框架是否支持任务白板、TODO、中间决策、结构化运行状态，而不是只把所有东西塞进聊天记录。**RAG**则看它如何处理外部知识召回，以及它到底把 RAG 当作知识库，把 Memory 当作用户经验，还是干脆把两者混成同一个 context injection 问题。

这五个维度不是为了给每个框架打分，而是为了把它们的记忆哲学暴露出来。它们都能叫 memory，但开发时承担的工程复杂度完全不同。同样，这样的解读可以方便你根据你的需求和框架的契合程度，去选择适合的 Framework。

### LangGraph/LangMem

### AgentScope

### CrewAI

### Mastra

### Agno

### LlamaIndex

### OpenAI Agents SDK

### AutoGen/Microsoft Agent Framework/Semantic Kernel

微软 Agent 家族路线有点乱：AutoGen、Semantic Kernel、Microsoft Agent Framework 三个名字都能讲 Agent，但它们其实代表了三代不同抽象。AutoGen 更像研究型多智能体框架，Semantic Kernel 更像企业集成和 RAG 框架，MAF 则是微软试图收束后的统一运行时。微软确实是一家没有品味的公司，喜欢自己和自己竞争。

#### AutoGen

**短期与裁剪**：短期记忆落在 `model_context`。`BufferedChatCompletionContext` 保留最近 N 条消息；`TokenLimitedChatCompletionContext` 按 token 预算控制上下文，超预算时会从中间删除消息；`HeadAndTailChatCompletionContext` 保留最早的 head 和最近的 tail，并用 skipped message 标记中间被跳过的历史。这些策略解决的是上下文窗口预算，并且没有什么压缩，就是截断。

**长期记忆**：长期记忆通过 `Memory` protocol 外挂。基本链路是：把事实、文本或其他内容包装成 `MemoryContent` 写入 memory store；推理前由 memory 根据当前上下文或查询执行 `query`；再通过 `update_context` 把召回结果写回模型上下文。需要增加长期记忆则使用 `add`。 AutoGen 只规定这组接口，不规定后端的具体工作流。

**用户记忆**：AutoGen 没有 `UserMemory` 作用域。用户偏好、身份事实、跨会话信息可以用 `ListMemory` 保存，也可以放进 Mem0、Redis、Chroma 或自定义后端；`ListMemory` 更像示例级实现：按时间顺序保存内容，`update_context` 时把记忆追加进上下文。当我们在真实开发的过程中，需要使用类似长期记忆的方法处理 `UserMemory` 的问题。

**工作记忆**：`TextCanvasMemory` 是比较典型的工作记忆。它保存持续变化的文档或画布，类似于 `todo.md`，让 agent 编辑长文本、草稿或计划时，将中间状态外置，不是自动抽取用户事实。他的工作方式和经典的 `todo.md` 基本一样。

**RAG**：综上，Autogen 根本不提供自己的 RAG 设计，仅仅是又一套接口，你可以自由的选择和编排自己喜欢的 RAG 系统。Chroma、Redis、Mem0 等只是不同 memory implementation。它们可以用 embedding、关键词、metadata 或外部服务做召回，但在 AutoGen 看来都走同一个模式：memory 自己决定如何查询、排序、过滤，再通过 `update_context` 把结果注入上下文。

这就是 Autogen，从现在的眼光来看已经完全落伍而不值得使用的框架，我们能看到他在 Memory 的编排上基本同样一事无成。

#### Semantic Kernel

**短期与裁剪**：Semantic Kernel 的短期记忆落在 `ChatHistory` 和 `AgentThread`。`ChatHistory` 是模型实际看到的消息流，`AgentThread` 是会话状态抽象（服务端 thread id ）。SK 有 `ChatHistoryReducer` 这条线，可以做消息数截断与旧消息摘要。这套能力主要在 chat history 层工作，需要开发者决定什么时候 reduce、保留哪些 system/tool/function-call 结构，只提供了适当的工具封装。

**长期记忆**：长期记忆主要通过 `Mem0Provider` 补齐（也有其他方案，我们在 RAG 那里会聊到）。它从 thread 消息中抽取 memories，后续 invocation 前再按当前请求召回。这里的长期记忆不是原始聊天归档，而是被抽取后的可复用事实。SK 通过 provider 把外部记忆能力挂到 agent 调用链上。但起码没有只提供抽象，不提供解决方案。

**用户记忆**：Semantic Kernel 还是使用类似 `Mem0Provider`  的方案，但是将记忆进行了类别拆分。`UserId` 可以表示跨 thread 的用户偏好和长期事实，`ThreadId` 可以把记忆限定在单个任务里。它把“这个人长期如此”和“这个任务暂时如此”分开了。它比 AutoGen 清楚的一点是，SK 至少把 user/thread 这两个作用域暴露了出来，记忆系统本身仍旧依赖外部 Provider。

**工作记忆**：`WhiteboardProvider` 也是类似 `todo.md` 的产物，提供了相当完善的封装和开箱即用的体验。它从对话中抽取 requirements、proposals、decisions、actions，保存任务推进过程中真正要稳定保留的结构化状态。即使 chat history 被截断，agent 仍然知道已经确认了什么、下一步该做什么。

**RAG**：RAG 是 SK 成熟的地方。Vector Store 是外部知识库，`TextSearchProvider` 负责检索并注入 agent 上下文。它不是像 AutoGen 那样只给一个 memory interface，而是围绕企业知识库、向量存储和搜索 provider 做了比较完整的连接器设计。这也是 SK 我认为为数不多的完整系统化设计。

这就是 Semantic Kernel：它不是一个统一的 Memory runtime，而是把 thread、history、Mem0、Whiteboard、Vector Store 这些能力拼成一组 provider。它比 AutoGen 更接近可用的企业 RAG 和任务白板，但整体上仍然是能力组合，而不是统一记忆系统。

#### Microsoft Agent Framework

**短期与裁剪**：Microsoft Agent Framework 的短期记忆落在 `AgentSession` 和 `ChatHistoryProvider`。`AgentSession` 保存会话状态，用于对话复用和恢复。`ChatHistoryProvider` 管历史持久化与上下文管理：负责上下文的装配以及长上下文时，由 reducer 和 compaction 压缩成可继续推理的上下文预算。开发者只需要简单配置即可。

**长期记忆**：长期记忆统一放到 `AIContextProvider` 后面。provider 可以在调用前搜索相关上下文，也可以在调用后从新消息中抽取记忆。这里的相关机制完全类似于 SK，我们可以建立适当的 scope，然后让 `AIContextProvider` 根据对应 scope 召回，这就是我们在其他 blog 里提到的关于 metadata 的重要性。`AIContextProvider` 本身也不和固定后端绑定。

**用户记忆**：用户记忆通过分 scope 的 `AIContextProvider` 实现。app、agent、user、session 都可以成为边界，避免不同用户和任务串味。虽然没有将用户记忆系统单列，但是 MAF 基本已经找到了一个成熟的抽象，能够提供相对稳定的记忆层，只是仍旧需要不少开发者的工作量，后台的 provider 需要为了前端的简洁，实现不少复杂的记忆存储与召回逻辑。

**工作记忆**：MAF 没有像 SK WhiteboardProvider 那样开箱即用的结构化工作记忆。它提供的是 session state、context provider、workflow checkpoint 这些底层积木。你可以用它们实现项目白板、任务计划、中间决策和 TODO，但抽取 schema、更新策略、冲突处理、何时注入上下文，都要开发者自己设计。相当不友好，工作记忆太常用了，重复开发会消耗不少时间。

**RAG**：`TextSearchProvider`负责外部知识召回。它们既可以自动注入上下文，也可以作为工具由 agent 按需调用。完全继承了 SK 提供的成熟外挂知识库方案。

这就是 Microsoft Agent Framework：它试图把 AutoGen 的多智能体遗产和 SK 的企业 RAG 遗产收束到一套 session、history、context provider、RAG、compaction 的运行时里。整体而言确实更加成熟了。

### Dify / Coze / n8n

#### Dify

**短期与裁剪**：Dify 的短期记忆主要落在 Agent 节点和 LLM 节点的会话上下文里。Agent 节点的 Memory 参数本质上是 `TokenBufferMemory`，开发者可以控制保留多少历史消息，窗口越大，模型看到的上下文越完整，token 成本也越高。LLM 节点也可以启用 Memory，把之前的 user-assistant 消息作为上下文拼回 prompt。这里解决的是低代码应用里的多轮连续性，不是一个完整的 Agent runtime 记忆系统。Dify 会自动处理系统的避免爆上下文，当然还是经典的裁剪+压缩思路。

**长期记忆**：Dify 默认没有真正的跨会话长期记忆。LLM 节点的 Memory 是 node-specific，而且不在不同 conversation 之间持久。想要长期保存用户事实、历史偏好或任务经验，需要开发者自己接外部数据库、知识库、插件、API 节点或变量更新逻辑。Dify 提供的是应用编排平台，不是一个帮你自动抽取、合并、遗忘和治理长期记忆的框架。

**用户记忆**：Dify 里比较接近用户记忆的是 Conversation Variables。它可以在同一个 chat session 的多轮对话中持续保存状态，比如用户选择、表单字段、任务阶段、临时偏好。但这个作用域仍然是会话级的，不是天然跨会话的 `UserMemory`。如果你要做“这个用户长期喜欢什么”“这个客户以前买过什么”，就需要把变量和外部存储打通，你需要用长期记忆那一套自己去搞。

**工作记忆**：Dify 的工作记忆更像 workflow 变量和节点状态。变量赋值节点、条件分支、知识检索结果、工具返回值可以共同构成一次任务的中间状态，但它不是 SK Whiteboard 那种结构化白板，也不是 Mastra working memory 那种用户画像表。这就是 Workflow 的程序记忆，schema、更新策略和冲突处理都散落在 workflow 设计里。

**RAG**：RAG 是 Dify 最成熟的方向之一。它把知识库、检索、重排、LLM 节点和应用发布放进同一个平台，对企业问答、客服助手、表单处理和内容生成非常友好。在 Dify 里，RAG 更多是外部知识进入上下文，Memory 更多是会话连续性和变量状态，两者可以在工作流里组合，却没有统一成一个长期记忆模型。

这就是 Dify：它适合交付，不适合研究长期记忆机制。它把 Agent 应用里最常见的知识库、工具、变量、发布和运营后台做好了，但 memory 的核心问题，尤其是跨会话用户事实抽取、冲突合并、过期、删除和治理，仍然要靠开发者自己补。

#### Coze

**短期与裁剪**：Coze 的短期记忆主要体现在 Bot 对话、变量、工作流节点和平台托管的上下文里。作为产品化 Agent 工厂，它更关心的是用户在聊天入口里是否能连续互动，而不是让开发者直接控制每条消息如何进入模型上下文。上下文裁剪、历史拼接和节点状态被平台藏起来了，专注产品设计而无需关心这些细节。当然其后台大概率还是裁剪+压缩的那一套，维护单一会话内记忆，这是最经济有效的方案。

**长期记忆**：Coze 的长期记忆不是纯黑盒，但它暴露的是产品化配置，而不是完整 memory runtime。开发者可以在智能体编排页面开启长期记忆；开启后，系统会自动从对话中抽取用户画像、用户记忆点和用户手动编辑的信息，并写入平台系统数据库。开发者还可以配置是否“支持在 Prompt 中调用长期记忆”：开启后，用户可以在普通对话中通过提问召回记忆；关闭后，长期记忆只能通过工作流里的长期记忆节点召回。新版记忆库还支持绑定多个低代码智能体和工作流，并用用户 UID 与渠道 ID 组合作为隔离标识。这就是他暴露的长期记忆能力，但其后台具体如何工作不得而知。

**用户记忆**：Coze 的用户记忆边界比 Dify 清楚。新版长期记忆会按用户 UID 与渠道 ID 隔离，同一记忆库下不同用户、不同渠道的数据不会混在一起。记忆内容也有内置分类：用户画像、用户记忆点、用户编写的信息；其中用户编写的信息优先级更高，和自动抽取内容冲突时会优先采纳。开发者和调试者可以在 Memory > 长期记忆 页面查看、编辑、删除或清空自己的长期记忆，用户也可以通过自然语言要求系统修改或记录记忆。长期记忆节点可以被用于召回长期记忆。考虑到我们在 Dify 里将数据库和变量放到了用户记忆范畴，那么 Coze 也有类似的能力。

**工作记忆**：Coze 的工作记忆主要藏在工作流、变量、数据库、知识库和插件编排里。它可以承载任务状态、表单信息、中间结果和流程节点输出。工作流编排就是这样，没什么额外可以说的。

**RAG**：Coze 也有知识库和插件体系，RAG 在这里服务于 Bot 能不能回答业务知识、调用外部能力、完成产品入口任务。它和长期记忆的关系更偏产品分工：知识库保存公共知识，长期记忆保存用户个性化信息。这个产品分工很清楚，但它没有把知识库、长期记忆、变量、数据库统一抽象成一个可编程 memory runtime。你只能通过可视化节点去使用他们。

这就是 Coze：它不是 code-first Agent framework，而是产品化 Agent 工厂。它的长期记忆比 Dify 更像真正面向用户的个性化记忆，但工程透明度和平台绑定也更强。你可以很快做出记得用户的 Bot，却不一定能细致控制这套记忆系统如何写入、合并、审计和迁移。

#### n8n

**短期与裁剪**：n8n 的短期记忆主要通过 AI cluster 里的 memory 子节点实现，最典型的是 Simple Memory。它保存当前 session 中可配置长度的聊天历史，让 AI Agent 或 chain 能够知道用户前几轮说了什么。这个能力解决的是对话连续性，是一个可视化 workflow 里的 chat history buffer。依旧是裁剪+压缩，三个工作流平台都这样。

**长期记忆**：n8n 不内建完整的长期记忆治理系统，但它很擅长把外部记忆服务接进流程。除了 Simple Memory 之外，n8n 还可以使用 Redis Chat Memory、Postgres Chat Memory、Zep 等 memory service；如果需要更复杂的消息加载、插入、删除和压缩，还可以用 Chat Memory Manager。n8n 把记忆当成一个可以接入 workflow 的外部节点”。

**用户记忆**：n8n 的用户记忆边界依赖 session key、外部存储和 workflow 设计。你可以用不同 session id 区分不同用户，也可以把用户信息写进数据库、CRM、表格或向量库，再在后续流程里召回。但这不是 Dify/Coze 那种面向终端产品的用户画像记忆，更不是框架级的 `UserMemory` 作用域。用户隔离、权限、生命周期和删除策略，都要由 workflow 设计者自己负责。

**工作记忆**：n8n 真正强的是工作流执行状态。触发器、节点输入输出、条件分支、循环、错误处理、外部 SaaS 连接器共同构成了它的工作记忆。它能很自然地保存“这次自动化流程走到了哪一步、哪个系统返回了什么、下一步该调用哪个服务”。工作流编排就是这样，没什么额外可以说的。

**RAG**：n8n 可以通过向量库节点、检索节点、工具节点和 AI Agent 节点拼出 RAG 流程。这里的 RAG 仍然是 workflow 拼装结果：你把数据进入向量库、检索、重排、生成这些步骤连起来，它就能跑。它不是统一 memory runtime，也不会替你决定外部知识、聊天历史和用户记忆之间的优先级。

这就是 n8n：它擅长把 AI 嵌进自动化流程，而不是把流程演化成完整 Agent runtime。它的 memory 能力很实用，但本质上是节点化的会话记忆和外部存储连接。对于流程自动化 + 一点智能的任务，它非常顺手；对于复杂长期记忆、用户画像治理和多 Agent 工作记忆隔离，它仍然要靠开发者自己搭。

### Pydantic AI / PocketFlow

Pydantic AI 没有必要按上面五类展开，因为它本来就不提供完整的记忆管理层。它真正提供的是 typed agent 编程体验：message history 可以通过 `all_messages()`、`new_messages()` 和 `message_history` 在多次运行之间传递，history processors 可以在模型调用前做裁剪、过滤或摘要；除此之外，长期记忆、用户记忆、RAG、工作白板、冲突合并、权限隔离都要开发者自己接数据库、向量库、检索器或外部 memory service。换句话说，Pydantic AI 只把消息历史作为结构化数据这件事做得很漂亮，并不替你决定什么该被记住、怎么写入、怎么召回、怎么删除。

PocketFlow 也一样，因为它是极简流程编排底座，不是 Agent memory framework。它的核心是 Node、Flow、Action 和 `shared store`，所谓记忆最多就是节点之间共享的一次运行状态：中间结果、工具返回、分支输出、trace、最终产物都可以放进去。但 `shared store` 不是长期记忆，也不是用户画像，更不是语义检索层。要做跨会话保存、用户隔离、RAG、记忆抽取、过期删除或冲突处理，开发者必须自己把 `shared store` 接到外部数据库、文件、向量库或业务系统里。PocketFlow 的优点是透明轻量，缺点也正是它几乎不替你做记忆治理。但记忆有时候也不一定是你需要的。

## 参考资料

### AutoGen / Semantic Kernel / Microsoft Agent Framework

- [AutoGen README](https://github.com/microsoft/autogen)
- [AutoGen Memory and RAG](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/memory.html)
- [AutoGen Model Context](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components/model-context.html)
- [Microsoft Agent Framework overview](https://learn.microsoft.com/en-us/agent-framework/overview/)
- [Microsoft Agent Framework conversations](https://learn.microsoft.com/en-us/agent-framework/agents/conversations/)
- [Microsoft Agent Framework context providers](https://learn.microsoft.com/en-us/agent-framework/agents/conversations/context-providers)
- [Microsoft Agent Framework context compaction](https://learn.microsoft.com/en-us/agent-framework/agents/conversations/compaction)
- [Semantic Kernel Agent Memory](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-memory)
- [Semantic Kernel Agent RAG](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-rag)
- [Semantic Kernel Vector Store connectors](https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors/)

### Dify / Coze / n8n

- [Dify Agent node](https://docs.dify.ai/en/use-dify/nodes/agent)
- [Dify LLM node](https://docs.dify.ai/en/use-dify/nodes/llm)
- [Dify Variable Assigner](https://docs.dify.ai/en/use-dify/nodes/variable-assigner)
- [Coze 长期记忆](https://www.coze.cn/open/docs/guides/long_memory)
- [n8n What's memory in AI?](https://docs.n8n.io/advanced-ai/examples/understand-memory/)
- [n8n Simple Memory node](https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.memorybufferwindow/)
- [n8n AI Agent node](https://docs.n8n.io/integrations/builtin/cluster-nodes/root-nodes/n8n-nodes-langchain.agent/)
- [n8n RAG in n8n](https://docs.n8n.io/advanced-ai/rag-in-n8n/)

### Pydantic AI / PocketFlow

- [Pydantic AI Message History](https://pydantic.dev/docs/ai/core-concepts/message-history/)
- [PocketFlow Documentation](https://the-pocket.github.io/PocketFlow/)
