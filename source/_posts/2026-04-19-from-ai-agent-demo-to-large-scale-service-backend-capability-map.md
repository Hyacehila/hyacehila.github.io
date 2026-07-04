---
title: "从 AI Agent Demo 到大规模服务"
title_en: "From AI Agent Demo to Large-Scale Service"
date: 2026-04-19 21:30:00 +0800
categories: ["Programming", "Backend Engineering"]
tags: ["Backend Engineering", "Reliability Engineering"]
author: Hyacehila
excerpt: 这是一篇 Agent 基础设施总览，梳理 AI Agent 从 Demo 走向大规模服务时依赖的能力，以及这些能力分别解决的工程问题。
excerpt_en: "An agent infrastructure overview that maps the backend capabilities needed for AI agents to move from demos to large-scale services and the engineering problems each solves."
permalink: '/blog/2026/04/19/from-ai-agent-demo-to-large-scale-service-backend-capability-map/'
---

把 `AI Agent` 从 Demo 变成大规模服务，难点通常不在把模型接口包成 API，而在补齐一整套后端能力栈。

当我在本地把 prompt、tool calling、memory、RAG、workflow 等组件接起来，Agent 也许已经能完成任务，于是很容易产生一个错觉：剩下的只是上线相关工作。

可一旦这个 Agent 要服务真实用户，要同时接住多人请求，要跑分钟级甚至小时级任务，要在失败后恢复、控制成本、限制权限、支持多租户，还要解释失败并留下足够的记录，问题中心就会变成：系统能不能稳定承载这种能力。

为了让这个问题更具体，我先假设一个贯穿全文的案例：你正在做一个通用 Agent 平台。它有在线入口，用户可以提交任务；有异步长任务，Agent 可能运行几分钟；会调用外部工具；会读写状态；还要服务多用户、多租户。

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

这条链路里新增的环节，大部分并不直接属于 Agent 或 LLM。它们是后端为了让任务在多人、多机、多进程、多故障条件下依然成立而提供的结构。

## AI Agent 规模化所需能力全览

| 所需能力 | 典型后端技术/系统 | 解决的问题 | 为什么对 Agent 特别重要 |
| --- | --- | --- | --- |
| 接住请求与流量 | `API Gateway`、负载均衡、无状态服务、副本扩缩容、限流、缓存 | 高并发访问、突发流量、低延迟、鉴权、配额 | Agent 经常同时面对实时请求和长任务启动流量 |
| 保存状态与保证数据正确性 | 关系型数据库、对象存储、缓存、搜索/向量索引 | 用户状态、任务状态、审计记录、恢复能力、一致性 | Agent 不能只靠内存记住会话和任务，否则一重启就丢世界 |
| 让长任务可靠执行 | 消息队列、任务队列、工作流引擎、重试、死信队列 | 异步执行、失败恢复、暂停续跑、任务编排 | 多轮 Agent 任务通常无法在一个 HTTP 请求里跑完 |
| 在并发下不把系统写乱 | 事务、锁、乐观并发、悲观并发、幂等键、条件更新 | 重复执行、乱序、覆盖写、竞争 | Agent 服务天然会遇到多 worker、多副本、重复消费 |
| 让服务之间协作而不失控 | RPC/API、事件驱动、服务发现、超时、熔断、退避重试 | 多服务协作、依赖故障、隔离边界 | Agent 平台很快就会拆出认证、调度、工具执行、检索、计费 |
| 看见系统真实状态 | 日志、指标、Tracing、任务审计、评测、SLO、告警 | 慢在哪里、错在哪里、贵在哪里、退化在哪里 | Agent 的失败可能是 500，也可能是任务质量退化和工具误用 |
| 治理风险、权限与成本 | 鉴权授权、租户隔离、预算、配额、沙箱、策略引擎、审计 | 越权、滥用、资源抢占、成本失控 | Agent 能行动，所以它的风险也比普通聊天接口更强 |
| 执行语言与运行时基座 | `Go`、`goroutine`/`channel`、`context` 取消、连接池、`Redis`、消息队列、工作流引擎、对象存储 | 高并发 I/O、资源控制、取消传播、吞吐与复杂度平衡，以及哪些能力自己写、哪些交给独立基础设施 | Agent 后端常常既是 API 服务，也是 worker 系统，还要把状态、队列、缓存托付给成熟组件 |

这张表概括了本文要讨论的问题：数据库、Go、并发、锁、队列、观测等话题看似分散，其实都在回答同一个问题。Agent 能不能成为服务，取决于后端能不能接住它的状态、流量、故障和成本。

## 为什么 AI Agent 一旦服务化，就变成后端问题

本地可跑的 Agent 不是服务。

生产级服务至少意味着几件事情同时成立：

- 它能被多用户稳定调用，而不是只在你自己的机器上成功一次。
- 它能跨分钟甚至跨小时执行，而不是只适合短请求。
- 它出错以后能恢复，而不是靠重跑一次碰运气。
- 它的状态能被保存、查询、审计，而不是只存在于进程内存。
- 它的成本、权限和吞吐是可控的，而不是先跑起来再说。

一旦任务会跨分钟执行、会调用工具、会在失败后恢复，问题就从模型怎么回答变成系统怎么承载。Anthropic 在 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 里把 agent 看成模型 + 工具 + 环境反馈的循环系统，并提醒开发者注意延迟和成本。只要任务不再是同步、短促、单用户、单进程，它就已经进入服务系统设计的范畴。

## 接住请求与流量的能力

任何生产系统的第一道门槛，都是它能不能接住请求。

这听起来像一句废话，但 Agent 服务在这里比普通 CRUD 应用更容易失控。原因很简单：普通接口通常是几十毫秒到几百毫秒的短调用；Agent 平台往往同时有两类流量，一类是实时入口流量，另一类是会诱发长任务、工具调用和后台资源占用的任务启动流量。如果入口层没有把流量、配额和执行模式拆开，系统就会迅速从偶尔慢一点变成全站被拖死。

今天后端通常靠下面这些技术来解决这类问题：

- `API Gateway` 或接入层先做鉴权、限流、路由、配额控制。
- 负载均衡把流量分发到多个服务副本，避免单点打满。
- Web/API 服务尽量做成无状态，让副本可以随时增减、重启、迁移。
- 自动扩缩容根据 CPU、QPS、队列长度或自定义指标增加/减少实例。
- 连接池和缓存把数据库与下游依赖保护起来，避免每个请求都直接打穿后端。

为什么“无状态”几乎是规模化前提？因为一旦服务副本可以被任意拉起和销毁，你就不能把任务、会话和计费状态藏在本地内存里。Google Cloud 在 [Patterns for scalable and resilient apps](https://cloud.google.com/architecture/scalable-and-resilient-apps) 里建议 `Aim for statelessness`，并强调分层负载均衡和基于指标的 autoscaling。AWS 在 [Reliability Pillar 的设计原则](https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/design-principles.html) 里也把水平扩展、停止猜容量和自动化管理变化放在基础原则位置。这样的设计也能让我们更好地利用云服务商的能力，控制开发复杂度和成本。

放到 Agent 场景里，这点会更明显。Web 层通常不该在一个请求里跑完整个 Agent，而应当：

- 验证请求是否合法。
- 记录任务元数据。
- 决定它是同步短任务还是异步长任务。
- 尽快返回任务 ID、状态链接或流式回执。

入口层负责接住流量，不该吞下整个 Agent 的复杂性。

## 保存状态与保证数据正确性的能力

一旦你不再只服务自己，数据库就会从存点数据的地方变成系统正确性的中枢。

一个真实 Agent 平台里，需要持久化的对象比初学者常见预期更多：

- 用户与租户信息
- 会话与任务状态
- 工具调用记录
- 任务输入输出
- 配额与计费数据
- 审计日志
- 失败与重试历史

这里还要说一句 Agent 记忆。工作记忆、情节记忆、长期记忆听起来像模型话题，可一旦要跨会话、跨重启地留存，它本质上仍是存储问题：要有 `schema`，要能检索，也要能治理。这条线我在[《从记忆形成到记忆治理：Agent Memory 的全景图》](/blog/2026/03/21/agent-memory-panorama/)里单独展开过。

如果这些东西只放在内存里，系统一重启，世界就断裂了。任务和状态会丢，连“这次到底执行到哪里、为什么失败、有没有重复扣费、是不是重复调用工具”这些信息也会一起消失。

今天后端通常这样分层存储：

- 关系型数据库负责核心事务状态，比如用户、任务、账单、配额、状态机。
- 对象存储负责大对象，比如长文本、附件、日志归档、工具输出快照。
- 缓存负责热点读取、会话加速、短期去重和降压。
- 搜索索引或向量索引负责检索类能力，但通常不承担最核心的事务真相。

Google 在 [Patterns for scalable and resilient apps](https://cloud.google.com/architecture/scalable-and-resilient-apps) 的数据库部分明确指出，关系型数据库的价值在于事务、强一致性、引用完整性和跨表查询；PostgreSQL 在 [MVCC 文档](https://www.postgresql.org/docs/current/mvcc-intro.html) 里也说明，多版本并发控制的目标是在多用户环境里维护一致性、隔离并尽量减少锁竞争。数据库不是把数据放进去再取出来这么简单。它负责把多用户、多事务、多并发条件下的世界状态维持在一个可管理的范围里。

从这一章开始，需要先接触这些词：

- 事务
- `schema`
- 索引
- `MVCC`
- 幂等
- 一致性

开篇还不需要掌握它们的底层实现，先知道各自负责什么就够了：

- 事务负责把几步更新绑成一个整体。要么一起成功，要么一起失败，不能留下半截状态。
- `schema` 负责说明系统到底在存什么。字段、类型、约束、表之间的关系，都靠它把边界画清楚。
- 索引负责让查询在数据变多以后仍然跑得动。没有合适索引，很多查询最后都会退化成一行行扫描。
- `MVCC` 和锁负责处理并发读写。多人同时改数据时，系统要知道谁能看见哪个版本，谁必须等待。
- 幂等负责处理重复执行。同一个任务、消息或请求被重试时，结果应该仍然是对的，而不是重复扣费、重复创建、重复调用。
- 一致性负责兜住最终状态。订单、账单、任务、审计记录之间不能互相打架。

这些机制的底层实现值得单独拆。这里只先建立一个判断：对 Agent 来说，数据库保存的是唯一可信的世界状态，不是一块随手写入的附属存储。

## 让长任务可靠执行的能力

普通 Web 请求最喜欢的世界，是请求进来，几十毫秒后结果返回。但 Agent 往往不是这样。

许多 Agent 任务天然就不是短请求：

- 需要多轮 reasoning
- 需要调用多个工具
- 需要等待外部依赖
- 需要在失败后重试
- 需要把中间状态持久化
- 需要长时间执行却不能占住前端连接

复杂 reasoning 任务可能要跑几分钟，所以平台需要异步启动、轮询状态、脱离前端连接地执行。Anthropic 在 [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps) 里也把问题进一步收束成更耐久的运行时结构：规划、生成、评估这些环节需要被放进能持续恢复和持续验证的外部系统里。队列系统这边，Amazon SQS 在 [standard queues 文档](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/standard-queues.html) 中明确提醒，标准队列是 `at-least-once delivery`，消息可能重复、也可能乱序。

归到工程上，就是几条朴素但很硬的要求：

- 任务要和请求解耦。
- 后台执行一定会失败、重试、重复。
- 系统必须设计成“允许任务被重新拿起”，不能指望“它一次就跑完”。

今天后端通常靠这些结构解决：

- 消息队列负责把入口请求和后台执行拆开。这里有个常被忽略的细节：写数据库和发消息这两步如果不在同一事务里，就会出现“任务记下了却没投递”或“投递了却没落库”的裂缝，工程上常用 `transactional outbox` 处理。
- 任务队列负责把工作分配给多个 worker。
- 工作流引擎负责保存进度、协调步骤、处理恢复。
- 重试策略和死信队列负责把“坏一次”变成“可诊断的失败”。
- 轮询、Webhook 或 WebSocket 负责把异步任务状态再反馈给用户。

对 Agent 来说，这一层很容易出问题。Agent 不是普通异步任务，它经常带状态、会调工具、会分叉决策，可能被人类打断，也可能要恢复后继续跑。Durable Execution 比“简单开个后台线程跑一下”重要得多。像 Temporal 这样的工作流引擎在 [Workflows 文档](https://docs.temporal.io/workflows) 里把工作流定义成可持久、可重放、能在进程崩溃后从上次进度继续的执行单元，正好对上 Agent “中途不能丢”的诉求。至于这套耐久运行时如何与 Agent 的规划-执行循环咬合，我在[《Harness 到底是什么：从 model + harness 到工程、产品与用户友好外壳》](/blog/2026/04/04/understanding-agent-harness/)里有更靠近 Agent 侧的讨论。

如果把这层抽象得再简化一点：

- 请求层回答“你要做什么”
- 工作流层回答“它现在做到哪了”
- worker 层回答“这一步具体怎么跑”

后端把这三层分开，Agent 才有机会扩到多用户和长任务。

## 在并发下不把系统写乱的能力

在 Agent 服务里，并发首先是正确性问题。

设想几个常见场景：

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

落到 Agent 后端，最常见的形态往往就两招：入口处给每个任务发一个幂等键（`idempotency key`），重复提交直接命中去重表，而不是再跑一遍；状态机推进时使用带版本号的条件更新（`compare-and-set`），让“我以为状态还是 A”在别人已经改成 B 时安全失败，而不是覆盖写。这两招怎么和具体存储、队列语义配合，值得另开一篇细讲。

并发问题不是某一个技术栈独有。PostgreSQL 的 [MVCC 文档](https://www.postgresql.org/docs/current/mvcc-intro.html) 讨论多用户环境里的隔离与读写冲突，[Explicit Locking](https://www.postgresql.org/docs/current/explicit-locking.html) 则说明应用何时需要显式锁。队列这边，SQS 官方文档提醒标准队列可能重复投递消息。Go 的 [Pipelines and cancellation](https://go.dev/blog/pipelines) 与 [Context](https://go.dev/blog/context) 反复强调取消传播和 `goroutine` 退出管理，因为并发程序会竞争锁，也会因为取消处理不当而泄漏资源、挂住上游、拖垮系统。

后面为什么要学并发、锁、线程竞争？因为没有它们，你没法回答这些最基本的问题：

- 同一份状态谁有权改？
- 两次修改谁先谁后？
- 同一个任务被执行两遍怎么办？
- 一个请求取消后，后台 `goroutine` 还要不要继续跑？
- 多副本扩容以后，系统如何保证不是把错误放大？

服务系统真正难的是正确地并发，不是尽量地并发。

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

这和架构审美关系不大，更多是规模带来的现实分工。这些部分的流量模型、故障模型、延迟要求和扩展方式本来就不一样。Google Cloud 在 [Patterns for scalable and resilient apps](https://cloud.google.com/architecture/scalable-and-resilient-apps) 里把 `loose coupling` 和 `modular architectures` 放在靠前位置，也指出独立服务可以分别发布、扩展和管理。分布式不是“更高级”的形态，很多时候只是复杂性上来以后，系统把边界暴露了出来。

今天后端通常这样让这些部分协作：

- 用 RPC 或 HTTP API 做同步调用。
- 用事件和消息做异步解耦。
- 用服务发现和配置中心管理依赖关系。
- 用超时、熔断（`circuit breaker`）、隔离和退避重试控制故障扩散，必要时用背压（`backpressure`）把过载向上游传导，而不是让某个慢依赖把调用方拖垮。
- 用明确的数据边界和 ownership 防止所有服务共同写一张烂表。

这一章要建立的直觉很简单：服务拆分的价值，在于隔开不同故障模型和扩展需求。

Agent 平台很容易同时拥有三种完全不同的部件：

- 低延迟入口
- 高不确定性的模型/工具执行
- 强一致性的状态与计费系统

把这三种东西塞进同一个进程，等于让最不稳定的部分决定整个系统的下限。拆分的价值在这里很直接：把模型与工具的不确定性挡在状态与计费系统外面。

## 看见系统真实状态的能力

如果你看不见系统在做什么，你就不是真的在运营服务，只是在等它出事。

而 Agent 服务比普通后端更需要观测，因为它的失败模式更多：

- 请求失败
- worker 崩掉
- 工具调用异常
- 模型输出质量下降
- 任务卡在中间状态
- 重试次数暴涨
- `token` 成本异常飙升
- 某个租户正在持续打爆系统

今天后端通常靠下面这些体系来建立可见性：

- 日志，回答“发生了什么”。
- 指标，回答“趋势如何、哪里异常”。
- `Tracing`，回答“一个请求或任务穿过了哪些组件，慢在了哪里”。这里要把 trace context 一路透传到 worker、工具调用和下游服务，否则一条 Agent 任务的链路会在异步边界上断开。
- 任务审计，回答“这个 Agent 到底做了哪些动作”。
- 质量评测，回答“结果是不是还达标”。Anthropic 在 [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) 里强调，Agent 的评测要盯住真实任务的完成度，而不是只看单步输出，因此质量指标本身就该是生产观测的一部分。
- `SLO` 和告警，回答“哪些问题已经影响用户体验，应该立刻处理”。Google SRE 在 [How to design good SLOs](https://cloud.google.com/blog/products/devops-sre/how-to-design-good-slos-according-to-google-sres) 里提醒，好的 `SLO` 要贴着用户真正在意的体验来定，而不是堆一堆没人会据此行动的指标。
- 成本监控，回答“这次上线为什么把 `token` / GPU / I/O 开销推高了”。

对 Agent 来说，观测不能只停留在传统的 CPU、内存和接口延迟：

- 你要看任务成功率。
- 你要看工具失败率。
- 你要看平均重试次数。
- 你要看中间状态停留时长。
- 你要看每类任务的单位成本。
- 你要看质量指标是否在悄悄退化。

观测本身还不够。Agent 服务需要把关键中间状态暴露成可以干预和替换的工程对象：检索结果能不能被复核，工具调用能不能重放，失败步骤能不能单独重试，某个模型或路由策略退化时能不能切换。否则系统表面上是在运行，实际上一旦出错就只能从最终答案倒推黑盒。

没有观测，Agent 服务很快就会变成靠直觉维护的黑盒。

## 治理风险、权限与成本的能力

许多 Agent 产品最先暴露的问题，不一定是模型能力，而是权限、安全或成本失控。

原因很直接：普通聊天接口主要在说；Agent 系统开始在做。一旦它能做事，风险模型就完全变了：

- 它可能调用危险工具。
- 它可能越权读取别人的数据。
- 它可能在多租户环境里抢占资源。
- 它可能因为失控重试把成本打爆。
- 它可能在工具链里制造大规模副作用。

Anthropic 在 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 里明确提醒，Agent 的自治会带来更高成本和复合错误风险，因此需要在沙箱环境里做广泛测试，并配上合适的 guardrails。Anthropic 在 [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) 里又进一步说明，工具设计、参数约束、返回结构和 `token` 效率会直接塑造 Agent 的行为边界。OpenAI 在 [Harness engineering](https://openai.com/index/harness-engineering/) 中强调的也是同一件事：环境、反馈和控制系统决定了 Agent 的可用边界。

今天后端通常靠这些能力做治理：

- 鉴权授权，决定谁能访问什么。
- 租户隔离，防止一个客户影响整个平台。
- 预算与配额，控制每个用户、模型、工具、任务的资源上限。
- 沙箱与权限边界，把高风险动作关进受限环境。
- 策略引擎和审计日志，确保系统的关键动作可追踪、可复核。
- 限速和资源调度，防止某类任务把整个平台拖垮。

这套治理属于 Agent 信任边界。什么动作算可信、在哪一层被拦下、越界后谁负责，本身就是架构决策。我在[《BettaFish、MiroFish、OpenClaw 与 Agent 的信任边界》](/blog/2026/04/12/agent-trust-boundary-openclaw-bettafish/)里专门讨论过这条边界该画在哪里。

从服务运营角度看，这里还有一个常被忽略但很现实的点：成本本身也是系统约束。

Agent 服务很容易出现一种假繁荣：功能看起来很强，演示也很好看，但后台是高频重试、超长上下文、低命中缓存、粗暴工具调用和不设预算的模型执行。

权限、限流、配额、预算和审计，不是上线后再补的东西。Agent 从一开始就需要这些后端骨架。

## 选择合适的执行语言与运行时基座

前面七项能力回答的都是“系统该做什么”。但还有一个绕不开的问题：这些东西到底用什么写、跑在什么之上。

我个人在服务端的默认选择是 `Go`。一个 Agent 平台天生是两栖的：它既是一个要扛实时请求的 API 服务，又是一个要跑长任务的 worker 系统。这两件事都高度 I/O 密集、都需要同时管理成百上千个并发执行、都需要在超时和取消时干净地收手。Go 的 `goroutine` 让为每个任务起一个轻量执行流变得廉价，`channel` 和 `select` 让这些执行流之间的协调有章可循，`context` 则把超时与取消沿着调用链一路传下去。

`context` 取消传播这件事，在 Agent 后端很要命。一个用户取消了任务，或者上游请求超时了，正在后台跑的 `goroutine`、正在等待的工具调用、正在持有的连接，都必须随之退出。否则它们会泄漏资源、挂住上游，慢慢把系统拖垮。Go 官方那两篇 [Pipelines and cancellation](https://go.dev/blog/pipelines) 和 [Context](https://go.dev/blog/context) 反复讲的就是这件事：并发程序的难点往往不在怎么并行起来，而在怎么干净地停下来。

但语言只解决一部分问题。还要判断哪些能力适合自己用 Go 写，哪些应该直接交给成熟的独立基础设施项目。

适合留在 Go 进程里自己处理的，通常是：

- 服务编排与请求处理：路由、鉴权中间件、把同步短任务和异步长任务分流。
- 并发原语与生命周期：`goroutine` 池、`context` 取消与超时、优雅退出（`graceful shutdown`）。
- 连接管理：数据库连接池、下游客户端复用、背压控制。

而下面这些，几乎都不该自己造，而要引入独立的基础设施项目。

- `Redis`：缓存、会话加速、分布式锁、幂等键与去重计数、限流计数器，乃至轻量队列。Agent 后端里大量“快而临时”的状态都落在这里。
- `PostgreSQL`：核心事务真相、任务状态机、审计记录。需要强一致性和引用完整性的地方，通常应该落在这里。
- 消息/任务系统（`NATS`、`Kafka`、云上的 `SQS` 等）：把入口请求和后台执行解耦，扛住峰值。
- 工作流引擎（`Temporal`）：把“可持久、可重放、崩溃后能续跑”的 durable execution 交给它，而不是自己用数据库手搓一套半成品状态机。
- 对象存储（`S3`、`MinIO`）：长文本、附件、工具输出快照、日志归档。
- 可观测栈（`OpenTelemetry` + `Prometheus` / `Grafana` / `Jaeger`）：指标、追踪、告警的事实标准底座。

这层分工可以说得朴素一点：语言决定你怎么写，基础设施决定你能不能少重造一些正确性。

这一章我刻意没有深挖任何一个组件。Go 的并发模型、连接池与背压的调参、`Redis` 与 `PostgreSQL` 各自的适用边界、`Temporal` 工作流的写法，每一个都值得单开一篇细讲，也是我接下来想顺着这张全景图展开的方向。在这里只需要先建立一个判断：选型本质上是在决定正确性该落到哪一层。

## 后端技术如何分别解决这些问题

| 问题类型 | 典型技术 | 它解决的边界 |
| --- | --- | --- |
| 多人同时访问，流量有峰值 | 网关、负载均衡、无状态服务、自动扩缩容 | 让系统能接住需求变化，而不是靠单机硬扛 |
| 状态不能丢，数据不能乱 | 关系型数据库、事务、索引、对象存储 | 让系统有持久真相，并能在多用户环境里保持正确 |
| 任务太长，不能一直占住请求 | 队列、后台 worker、工作流引擎、轮询/Webhook | 把“用户请求”和“后台执行”拆开，并支持恢复和重试 |
| 消息会重复，worker 会竞争 | 幂等、锁、条件更新、乐观并发、悲观并发 | 让重复执行和并发修改不会直接写坏状态 |
| 服务越拆越多，依赖越来越复杂 | RPC、事件、超时、熔断、服务发现、配置中心 | 让协作发生，但把故障扩散控制在边界内 |
| 不知道哪里慢、哪里错、哪里贵 | 日志、指标、Tracing、审计、评测、SLO | 让服务变得可运营，而不是靠个人直觉维护 |
| 模型和工具会越权，也会烧钱 | 鉴权、沙箱、租户隔离、预算、配额、策略引擎 | 让 Agent 的行动能力落在可控边界之内 |
| 写得对，也要落得下去 | `Go` + `goroutine`/连接池/`context` 取消、`Redis`、消息队列、`Temporal`、对象存储、OTel | 让前面所有能力真正被实现与运维，而不是停在架构图上 |

## 最后的判断

我现在对这个问题的判断很明确。

如果你只是想让一个 Agent 在本地跑出结果，最该关心的是 `model`、`prompt`、`tool use` 和 `harness`。

但如果你想把它做成一个大规模服务，问题中心一定会转移。你需要回答：

- 请求怎么接
- 状态怎么存
- 长任务怎么跑
- 并发下怎么保证正确
- 服务之间怎么协作
- 系统怎么被观测
- 风险、权限和成本怎么被治理
- 用什么语言和基础设施把它们落下去

从 Demo 到服务，Agent 最大的变化是开始被后端约束。

这些技术共同解决一件事：`Scaling Up`。

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
