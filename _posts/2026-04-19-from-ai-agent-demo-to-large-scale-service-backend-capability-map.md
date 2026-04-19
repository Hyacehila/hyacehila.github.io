---
layout: blog-post
title: "从 AI Agent Demo 到大规模服务：后端能力全景图"
date: 2026-04-19 21:30:00 +0800
series: 后端学习笔记
categories: [智能体系统]
tags: [Backend, Agents, Databases, Go, Concurrency, Distributed Systems, Reliability]
author: Hyacehila
excerpt: 这是一份后端学习笔记的第一篇，研究把 AI Agent 从 Demo 变成大规模服务，到底需要哪些后端能力，今天的后端技术又分别如何解决这些问题。后面将会开始逐步学习相关技术。
featured: false
math: false
---

# 从 AI Agent Demo 到大规模服务：后端能力全景图

**把 `AI Agent` 从 Demo 变成大规模服务，不是把一个模型接口包成 API，而是在补齐一整套后端能力栈。**

当我在本地把 prompt、tool calling、memory、RAG、workflow 等各个组件接起来，Agent也许已经能完成任务，于是就很容易得出一个错觉：剩下的只是处理一下上线的相关工作了。可一旦这个 Agent 要服务真实用户，要同时接住多人请求，要跑分钟级甚至小时级任务，要在失败后恢复，要控制成本，要限制权限，要支持多租户，要解释为什么这次成功率下降了，问题的中心就会立刻从“Agent够不够聪明”转向“系统能不能稳定地承载这种聪明”。

为了让这个问题更具体，我先假设一个贯穿全文的统一案例：你正在做一个通用 Agent 平台。它有在线入口，用户可以提交任务；有异步长任务，Agent 可能运行几分钟；会调用外部工具；会读写状态；而且要服务多用户、多租户。

它的最小生产链路大概会长成这样：

```text
用户请求
  -> API Gateway / 鉴权 / 限流
  -> Web/API 服务
  -> 数据库写入任务与状态
  -> 队列 / 工作流系统分发任务
  -> Worker 执行 Agent loop / 调工具 / 写结果
  -> 缓存 / 对象存储 / 检索系统参与读写
  -> 日志 / 指标 / Tracing / 评测系统持续观测
  -> 轮询 / Webhook / WebSocket 把结果回给用户
```

请注意，这条链路里多出来的东西，大部分都和Agent与LLM无关，而是**后端为了让任务在多人、多机、多进程、多故障条件下依然成立所加上的结构**。

## AI Agent 规模化所需能力全览

| 所需能力 | 典型后端技术/系统 | 解决的问题 | 为什么对 Agent 特别重要 | 后续对应篇章 |
| --- | --- | --- | --- | --- |
| 接住请求与流量 | `API Gateway`、负载均衡、无状态服务、副本扩缩容、限流、缓存 | 高并发访问、突发流量、低延迟、鉴权、配额 | Agent 产品经常同时面对实时请求和长任务启动流量 | 第 2、8 篇 |
| 保存状态与保证数据正确性 | 关系型数据库、对象存储、缓存、搜索/向量索引 | 用户状态、任务状态、审计记录、恢复能力、一致性 | Agent 不能只靠内存记住会话和任务，否则一重启就丢世界 | 第 2、3 篇 |
| 让长任务可靠执行 | 消息队列、任务队列、工作流引擎、重试、死信队列 | 异步执行、失败恢复、暂停续跑、任务编排 | 很多 Agent 根本不可能在一个 HTTP 请求里跑完 | 第 4 篇 |
| 在并发下不把系统写乱 | 事务、锁、乐观并发、悲观并发、幂等键、条件更新 | 重复执行、乱序、覆盖写、竞争 | Agent 服务天然会遇到多 worker、多副本、重复消费 | 第 3、6 篇 |
| 让服务之间协作而不失控 | RPC/API、事件驱动、服务发现、超时、熔断、退避重试 | 多服务协作、依赖故障、隔离边界 | Agent 平台很快就会拆出认证、调度、工具执行、检索、计费 | 第 4、8 篇 |
| 看见系统真实状态 | 日志、指标、Tracing、任务审计、评测、SLO、告警 | 慢在哪里、错在哪里、贵在哪里、退化在哪里 | Agent 的失败不只是一条 500，还包括任务质量退化和工具误用 | 第 7 篇 |
| 治理风险、权限与成本 | 鉴权授权、租户隔离、预算、配额、沙箱、策略引擎、审计 | 越权、滥用、资源抢占、成本失控 | Agent 能行动，所以它的风险也比普通聊天接口更强 | 第 8 篇 |
| 选择合适的执行语言与并发模型 | `Go`、协程/线程模型、连接池、上下文取消、并发原语 | 高并发 I/O、资源控制、取消传播、吞吐与复杂度平衡 | Agent 后端常常既是 API 服务，也是 worker 系统 | 第 5、6 篇 |

这一张表基本概括了本文想要讨论的问题，它把后续数据库、Go、并发、锁、队列、观测这些看似分散的话题，重新收束成了同一个问题：**后端并不是在给 Agent 做外围优化，后端是在定义 Agent 能否成为服务。**

## 为什么 AI Agent 一旦服务化，就变成后端问题

本地可跑的 Agent 不是服务。

真正的服务至少意味着几件事情同时成立：

- 它能被很多用户稳定调用，而不是只在你自己的机器上成功一次。
- 它能跨分钟甚至跨小时执行，而不是只适合短请求。
- 它出错以后能恢复，而不是重来一遍祈祷这次能过。
- 它的状态能被保存、查询、审计，而不是只存在于进程内存。
- 它的成本、权限和吞吐是可控的，而不是“先跑起来再说”。

Anthropic 在 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 里把 agent 看成“模型 + 工具 + 环境反馈”的循环系统，并明确提醒：Agent 往往是在用更高的延迟和成本换更高的任务能力。OpenAI 在 [Background mode](https://developers.openai.com/api/docs/guides/background) 里也已经把“长任务异步执行”做成了平台级能力，因为 reasoning model 在复杂任务里本来就可能跑上几分钟。到这里你应该已经能看到问题转换：**一旦任务不是同步、短促、单用户、单进程，你面对的就不再只是模型调用，而是完整的服务系统设计。**

**AI Agent 的大规模服务化，本来就是一个后端问题。**

## 接住请求与流量的能力

任何生产系统的第一道门槛，都是它能不能接住请求。

这听起来像一句废话，但 Agent 服务在这里比普通 CRUD 应用更容易失控。原因很简单：普通接口通常是几十毫秒到几百毫秒的短调用；Agent 平台往往同时有两类流量，一类是实时入口流量，另一类是会诱发长任务、工具调用和后台资源占用的任务启动流量。如果入口层没有把流量、配额和执行模式拆开，系统就会迅速从偶尔慢一点变成全站被拖死。

今天后端通常靠下面这些技术来解决这类问题：

- `API Gateway` 或接入层先做鉴权、限流、路由、配额控制。
- 负载均衡把流量分发到多个服务副本，避免单点打满。
- Web/API 服务尽量做成无状态，让副本可以随时增减、重启、迁移。
- 自动扩缩容根据 CPU、QPS、队列长度或自定义指标增加/减少实例。
- 连接池和缓存把数据库与下游依赖保护起来，不让每个请求都直接打穿后端。

为什么“无状态”几乎是规模化前提？因为一旦服务副本可以被任意拉起和销毁，你就不能把真正重要的状态藏在本地内存里。Google Cloud 在 [Patterns for scalable and resilient apps](https://cloud.google.com/architecture/scalable-and-resilient-apps) 里明确建议 `Aim for statelessness`，因为无状态服务能独立处理每个请求，便于扩缩容和故障恢复；同一份架构文档也强调 `load-balance at each tier` 和基于指标做 autoscaling。AWS 在 [Reliability Pillar 的设计原则](https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/design-principles.html) 里也把“水平扩展”“停止猜容量”“用自动化管理变化”放在基础原则位置。

这件事放到 Agent 场景里尤其重要。因为你的 Web 层真正应该做的，通常不是在一个请求里把整个 Agent 跑完，而是：

- 验证请求是否合法。
- 记录任务元数据。
- 决定它是同步短任务还是异步长任务。
- 尽快返回任务 ID、状态链接或流式回执。

换句话说，**入口层的责任是接住流量，不是吞下整个复杂性。**

## 保存状态与保证数据正确性的能力

一旦你不再只服务自己，数据库就会从存点数据的地方变成系统正确性的中枢。

一个真实 Agent 平台里，需要持久化的东西比很多初学者想象得多得多：

- 用户与租户信息
- 会话与任务状态
- 工具调用记录
- 任务输入输出
- 配额与计费数据
- 审计日志
- 失败与重试历史

如果这些东西只放在内存里，系统一重启，世界就断裂了。你不仅丢任务、丢状态，还会丢掉“这次到底执行到哪里、为什么失败、有没有重复扣费、是不是重复调用工具”这些更关键的信息。

今天后端通常这样分层存储：

- 关系型数据库负责核心事务状态，比如用户、任务、账单、配额、状态机。
- 对象存储负责大对象，比如长文本、附件、日志归档、工具输出快照。
- 缓存负责热点读取、会话加速、短期去重和降压。
- 搜索索引或向量索引负责检索类能力，但通常不承担最核心的事务真相。

Google 在 [Patterns for scalable and resilient apps](https://cloud.google.com/architecture/scalable-and-resilient-apps) 的数据库部分明确指出，关系型数据库的价值在于事务、强一致性、引用完整性和跨表查询；而 PostgreSQL 在 [MVCC 文档](https://www.postgresql.org/docs/current/mvcc-intro.html) 里更直接地说明，多版本并发控制的目标就是在多用户环境里维护一致性、隔离并尽量减少锁竞争。换句话说，数据库不是“把数据放进去以后再取出来”的工具，而是**把多用户、多事务、多并发条件下的真实世界压成可维护状态的机器。**

这也是为什么从这一章开始，你就必须开始接触这些词：

- 事务
- `schema`
- 索引
- `MVCC`
- 幂等
- 一致性

在开篇里你还不需要掌握它们的底层实现，但你必须先知道它们分别在做什么：

- 事务在兜“几步更新要么一起成功，要么一起失败”。
- 索引在兜“数据多了以后查询还能不能快”。
- `schema` 在兜“系统是否知道自己保存的到底是什么”。
- `MVCC` 和锁在兜“多人同时改数据时会不会互相踩坏”。
- 幂等在兜“同一个任务被重复执行时，结果能不能还是对的”。

## 让长任务可靠执行的能力

普通 Web 请求最喜欢的世界，是“请求进来，几十毫秒后结果返回”。但 Agent 不行了。

很多 Agent 任务天然就不是短请求：

- 需要多轮 reasoning
- 需要调用多个工具
- 需要等待外部依赖
- 需要在失败后重试
- 需要把中间状态持久化
- 需要长时间执行却不能占住前端连接

OpenAI 在 [Background mode](https://developers.openai.com/api/docs/guides/background) 里已经把这个事实写得很直白：复杂 reasoning 任务可能要跑几分钟，所以平台需要异步启动、轮询状态、脱离前端连接地执行。Anthropic 在 [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps) 里也把问题进一步收束成更耐久的运行时结构：规划、生成、评估这些环节需要被放进能持续恢复和持续验证的外部系统里。队列系统这边，Amazon SQS 在 [standard queues 文档](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/standard-queues.html) 中明确提醒，标准队列是 `at-least-once delivery`，消息可能重复、也可能乱序。

这三件事放在一起，其实就是长任务后端的第一堂课：

- 任务要和请求解耦。
- 后台执行一定会失败、重试、重复。
- 所以系统必须设计成“允许任务被重新拿起”，而不是指望“它一次就跑完”。

今天后端通常靠这些结构解决：

- 消息队列负责把入口请求和后台执行拆开。
- 任务队列负责把工作分配给多个 worker。
- 工作流引擎负责保存进度、协调步骤、处理恢复。
- 重试策略和死信队列负责把“坏一次”变成“可诊断的失败”。
- 轮询、Webhook 或 WebSocket 负责把异步任务状态再反馈给用户。

对于 Agent，这一层尤其关键。因为 Agent 不是普通异步任务，它经常是**一个带状态、会调用工具、会分叉决策、可能被人类打断、也可能要恢复继续跑的长生命周期过程**。这也是为什么 Durable Execution 会比“简单开个后台线程跑一下”重要得多。

如果把这层抽象得再简化一点：

- 请求层回答“你要做什么”
- 工作流层回答“它现在做到哪了”
- worker 层回答“这一步具体怎么跑”

后端把这三层分开，Agent 才有机会真正扩到多用户和长任务。

## 在并发下不把系统写乱的能力

在 Agent 服务里，并发是一个**正确性问题**。

设想几个非常普通的场景：

- 两个 worker 同时拿到了同一个任务。
- 一个工具调用超时后被重试，但第一次其实已经部分成功。
- 一个用户连续点了两次“重新运行”。
- 同一个会话状态被两个副本同时更新。
- 队列因为 `at-least-once delivery` 把同一条消息又送来了一遍。

如果系统没有并发控制，这些情况不会只让系统慢一点，而会让结果直接错掉：重复扣费、重复发消息、状态回退、覆盖写、乱序写入、幽灵结果。

今天后端通常靠这些机制兜底：

- 事务，把一组必须原子完成的修改绑在一起。
- 锁，在必要时显式地保护冲突点。
- 乐观并发控制，用版本号或条件更新判断“别人有没有先改过”。
- 悲观并发控制，在高冲突点上直接串行化访问。
- 幂等键和去重表，确保重复请求不会重复生效。
- 原子操作和条件更新，避免“先读后写”之间被别的并发插队。

PostgreSQL 在 [MVCC 文档](https://www.postgresql.org/docs/current/mvcc-intro.html) 里强调，多版本并发控制的核心收益之一就是在多用户环境里保证隔离并降低读写冲突；在 [Explicit Locking](https://www.postgresql.org/docs/current/explicit-locking.html) 中又明确指出，当 `MVCC` 不能给出你想要的行为时，应用可以使用显式锁来管理冲突。队列这边，SQS 官方文档已经告诉你消息可能重复到达；Go 这边，官方在 [Pipelines and cancellation](https://go.dev/blog/pipelines) 以及 [Context](https://go.dev/blog/context) 里一再强调并发流水线、取消传播和 goroutine 退出管理，因为并发程序不仅会竞争锁，还会因为取消处理不当而泄漏资源、挂住上游、拖垮系统。

所以你后面为什么一定要学并发、锁、线程竞争？

不是因为这些词“更底层、更硬核、更高级”，而是因为没有它们，你根本没法回答这些最基本的问题：

- 同一份状态谁有权改？
- 两次修改谁先谁后？
- 同一个任务被执行两遍怎么办？
- 一个请求取消后，后台 goroutine 还要不要继续跑？
- 多副本扩容以后，系统如何保证不是把错误放大？

在真正的服务系统里，**正确地并发** 比 **尽量地并发** 更难，也更重要。

## 让服务之间协作而不失控的能力

系统规模一大，复杂性不会待在原地，它会自然外溢。

你一开始也许只有一个服务：接请求、调用模型、返回结果。可只要真的开始服务用户，职责很快就会分裂出来：

- 认证服务
- 任务调度服务
- worker 执行层
- 检索服务
- 工具代理层
- 配额与计费服务
- 通知与回调服务

这不是架构师的审美问题，而是规模带来的现实分工。因为这些部分的流量模型、故障模型、延迟要求和扩展方式本来就不一样。Google Cloud 在 [Patterns for scalable and resilient apps](https://cloud.google.com/architecture/scalable-and-resilient-apps) 里把 `loose coupling` 和 `modular architectures` 放在非常靠前的位置，并明确指出，独立服务可以分别发布、扩展和管理。换句话说，分布式并不是什么“更高级的形态”，它更像是复杂性上来以后系统被迫暴露出的自然边界。

今天后端通常这样让这些部分协作：

- 用 RPC 或 HTTP API 做同步调用。
- 用事件和消息做异步解耦。
- 用服务发现和配置中心管理依赖关系。
- 用超时、熔断、隔离和退避重试控制故障扩散。
- 用明确的数据边界和 ownership 防止所有服务共同写一张烂表。

这一章真正要建立的直觉是：**服务拆分解决的不是优雅，而是把不同故障模型和扩展需求隔开。**

对 Agent 平台来说尤其如此。因为它很容易同时拥有三种完全不同的部件：

- 低延迟入口
- 高不确定性的模型/工具执行
- 强一致性的状态与计费系统

## 看见系统真实状态的能力

如果你看不见系统在做什么，你就不是真的在运营服务，你只是在祈祷它别出事。

而 Agent 服务比普通后端更需要观测，因为它的失败模式更多：

- 请求失败
- worker 崩掉
- 工具调用异常
- 模型输出质量下降
- 任务卡在中间状态
- 重试次数暴涨
- token 成本异常飙升
- 某个租户正在持续打爆系统

今天后端通常靠下面这些体系来建立可见性：

- 日志，回答“发生了什么”。
- 指标，回答“趋势如何、哪里异常”。
- `Tracing`，回答“一个请求或任务穿过了哪些组件，慢在了哪里”。
- 任务审计，回答“这个 Agent 到底做了哪些动作”。
- 质量评测，回答“结果是不是还达标”。
- `SLO` 和告警，回答“哪些问题已经影响用户体验，应该立刻处理”。
- 成本监控，回答“这次上线为什么把 token / GPU / I/O 开销推高了”。

对 Agent 来说，观测不能只停留在传统的 CPU、内存和接口延迟：

- 你要看任务成功率。
- 你要看工具失败率。
- 你要看平均重试次数。
- 你要看中间状态停留时长。
- 你要看每类任务的单位成本。
- 你要看质量指标是否在悄悄退化。

**没有观测，Agent会变成神秘学炼金术。**

## 治理风险、权限与成本的能力

很多 Agent 产品真正先死掉的，不是模型不够强，而是权限、安全或成本先失控。

原因非常简单：普通聊天接口主要在“说”；Agent 系统开始在“做”。一旦它能做事，风险模型就完全变了：

- 它可能调用危险工具。
- 它可能越权读取别人的数据。
- 它可能在多租户环境里抢占资源。
- 它可能因为失控重试把成本打爆。
- 它可能在工具链里制造大规模副作用。

Anthropic 在 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 里明确提醒，Agent 的自治会带来更高成本和复合错误风险，因此需要在沙箱环境里做广泛测试，并配上合适的 guardrails。Anthropic 在 [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) 里又进一步说明，工具设计、参数约束、返回结构和 token 效率会直接塑造 Agent 的行为边界。OpenAI 在 [Harness engineering](https://openai.com/index/harness-engineering/) 中强调的也正是同一件事：真正关键的是环境、反馈和控制系统。

今天后端通常靠这些能力做治理：

- 鉴权授权，决定谁能访问什么。
- 租户隔离，防止一个客户影响整个平台。
- 预算与配额，控制每个用户、模型、工具、任务的资源上限。
- 沙箱与权限边界，把高风险动作关进受限环境。
- 策略引擎和审计日志，确保系统的关键动作可追踪、可复核。
- 限速和资源调度，防止某类任务把整个平台拖垮。

从服务运营角度看，这一章还有一个常被忽略但极其现实的点：**成本本身也是系统约束。**

Agent 服务很容易出现一种假繁荣：功能看起来很强，演示也很好看，但后台是高频重试、超长上下文、低命中缓存、粗暴工具调用和不设预算的模型执行。

**权限、限流、配额、预算和审计，不是等系统成熟以后再补的企业功能，而是 Agent 从一开始就必须拥有的后端骨架。**

## 后端技术如何分别解决这些问题

| 问题类型 | 典型技术 | 它真正解决的边界 |
| --- | --- | --- |
| 多人同时访问，流量有峰值 | 网关、负载均衡、无状态服务、自动扩缩容 | 让系统能接住需求变化，而不是靠单机硬扛 |
| 状态不能丢，数据不能乱 | 关系型数据库、事务、索引、对象存储 | 让系统有持久真相，并能在多用户环境里保持正确 |
| 任务太长，不能一直占住请求 | 队列、后台 worker、工作流引擎、轮询/Webhook | 把“用户请求”和“后台执行”拆开，并支持恢复和重试 |
| 消息会重复，worker 会竞争 | 幂等、锁、条件更新、乐观并发、悲观并发 | 让重复执行和并发修改不会直接写坏状态 |
| 服务越拆越多，依赖越来越复杂 | RPC、事件、超时、熔断、服务发现、配置中心 | 让协作发生，但把故障扩散控制在边界内 |
| 不知道哪里慢、哪里错、哪里贵 | 日志、指标、Tracing、审计、评测、SLO | 让服务变得可运营，而不是靠个人直觉维护 |
| 模型和工具会越权，也会烧钱 | 鉴权、沙箱、租户隔离、预算、配额、策略引擎 | 让 Agent 的行动能力落在可控边界之内 |

这张表其实也解释了为什么后端学习笔记这个系列会长成接下来的样子。因为你后面学的每一个主题，都不是孤立知识点，而是在回答全景图里的某一个缺口。

## 系列路线图

接下来这个系列我会按下面的顺序继续写：

1. 数据库为什么是 Agent 服务的状态中枢，对应本篇第三章“保存状态与保证数据正确性的能力”
2. 事务、幂等与一致性，为什么重复执行会把系统写坏，对应本篇第三章和第五章
3. 缓存、队列与工作流，如何把长任务跑得更稳，对应本篇第二章和第四章
4. Go 为什么适合写高并发 Agent 后端，对应本篇第五章和第六章里的执行模型与服务协作
5. 并发、锁与线程竞争，吞吐量和正确性如何冲突，对应本篇第五章
6. 日志、指标、Tracing 与评测，如何真正看见 Agent 系统，对应本篇第七章
7. 限流、权限、多租户与成本控制，如何把 Agent 做成可运营产品，对应本篇第二章和第八章

如果把它们映射回这篇文章的八个章节，其实就是一条很清晰的学习路线：

- 先学“状态”
- 再学“长任务”
- 再学“并发正确性”
- 再学“可观测与治理”


## 最后的判断

我现在对这个问题的判断已经很明确了。

如果你只是想让一个 Agent 在本地跑出结果，那么你最该关心的是model、prompt、tool use 和 harness。

但如果你想把它做成一个真正的大规模服务，那么问题中心一定会转移。真正重要的，会变成：

- 请求怎么接
- 状态怎么存
- 长任务怎么跑
- 并发下怎么保证正确
- 服务之间怎么协作
- 系统怎么被观测
- 风险、权限和成本怎么被治理

也就是说，**从 Demo 到服务，Agent 最大的变化不是更会推理，而是开始被后端接管。**

后端不是 AI Agent 的配角。  
后端是把 Agent 的能力变成稳定系统的那套现实约束。
大量的技术只解决一件事，也就是处理 Scaling Up。

## 参考文献与官方入口

- [OpenAI: Harness engineering](https://openai.com/index/harness-engineering/)
- [OpenAI API: Background mode](https://developers.openai.com/api/docs/guides/background)
- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Anthropic: Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- [Anthropic: Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [Anthropic: Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [Google Cloud: Patterns for scalable and resilient apps](https://cloud.google.com/architecture/scalable-and-resilient-apps)
- [Google Cloud Blog: How to design good SLOs, according to Google SREs](https://cloud.google.com/blog/products/devops-sre/how-to-design-good-slos-according-to-google-sres)
- [AWS Well-Architected: Reliability Pillar design principles](https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/design-principles.html)
- [Amazon SQS: Standard queues](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/standard-queues.html)
- [PostgreSQL: MVCC introduction](https://www.postgresql.org/docs/current/mvcc-intro.html)
- [PostgreSQL: Explicit locking](https://www.postgresql.org/docs/current/explicit-locking.html)
- [Temporal Docs: Workflows](https://docs.temporal.io/workflows)
- [Go Blog: Pipelines and cancellation](https://go.dev/blog/pipelines)
- [Go Blog: Context](https://go.dev/blog/context)
