---
title: From AI Agent Demo to Large-Scale Service
title_zh: 从 AI Agent Demo 到大规模服务
date: 2026-04-19 21:30:00 +0800
categories:
- Programming
- Full Stack Development
tags:
- Backend Engineering
- Reliability Engineering
author: Hyacehila
excerpt: An agent infrastructure overview that maps the backend capabilities needed for AI agents to move from demos to large-scale
  services and the engineering problems each solves.
description: An agent infrastructure overview that maps the backend capabilities needed for AI agents to move from demos to
  large-scale services and the engineering problems each solves.
excerpt_zh: 这是一篇 Agent 基础设施总览，梳理 AI Agent 从 Demo 走向大规模服务时依赖的能力，以及这些能力分别解决的工程问题。
permalink: /blog/2026/04/19/from-ai-agent-demo-to-large-scale-service-backend-capability-map/
lang: en
translation_key: 2026-04-19-from-ai-agent-demo-to-large-scale-service-backend-capability-map
translation_status: machine
translation_source_hash: 48d6ab6ababb285a13f5f38c3ece07a3df5dde25e46b475dd9cba52cf1daa12b
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>Put <code>AI Agent</code> From Demo to large-scale service, the difficulty is not to wrap the model interface into an API, but to complete a set of back-end capacity.</p>
<p>When I put together the components of prompt, tool calling, memoory, RAG, workflow, and so on, Agent might have been able to do it, and it was easy to create a illusion: the rest was just on-line related work.</p>
<p>But once this Agent is a true user, who is requested by many people, who are running minutes or even hours, who are to recover after failure, control costs, limit privileges, support multi-tenant households, and who is to explain the failure and keep sufficient records, the problem centre becomes: can the system stabilize the capacity?</p>
<p>To make this issue more specific, I would like to assume that you are making a generic Agent platform as a case through the whole text. It has an online portal where users can submit assignments; has a long walk, Agent may run for a few minutes; calls external tools; reads and writes; and serves multiple users and tenants.</p>
<p>Its smallest production chain is likely to grow like this:</p>
<pre><code class="language-text">用户请求
  -&gt; API Gateway / 鉴权 / 限流
  -&gt; Web/API 服务
  -&gt; 数据库写入任务与状态
  -&gt; 队列 / 工作流系统分发任务
  -&gt; Worker 执行 Agent loop / 调工具 / 写结果
  -&gt; 缓存 / 对象存储 / 检索系统参与读写
  -&gt; 日志 / 指标 / Tracing / 评测系统持续观测
  -&gt; 轮询 / Webhook / WebSocket 把结果回给用户
</code></pre>
<p>Most of the new links in this chain do not directly belong to Agent or LLM. They are the structures that the back end provides to keep missions multi-person, multi-engineered, multi-processed, multi-facility.</p>
<h2>A complete overview of the capabilities required for the size of AI Agent</h2>
<table>
<thead>
<tr>
<th>Capacity requirements</th>
<th>Typical back-end technology/system</th>
<th>Problems addressed</th>
<th>Why is it so important to Agent?</th>
</tr>
</thead>
<tbody><tr>
<td>Catch the requests and traffic.</td>
<td><code>API Gateway</code>, load balance, non-state service, copy amplification, restricted flow, cache</td>
<td>High-mix visits, sudden flows, low delays, rights of assurance, quotas</td>
<td>Agent often faces both real-time requests and long-term start-up traffic.</td>
</tr>
<tr>
<td>Save status and ensure data validity</td>
<td>Relationship database, object storage, cache, search/ vector index</td>
<td>User status, task status, audit records, resilience, consistency</td>
<td>Agent, you can't just remember the conversation and the mission from memory, or lose the world as soon as we restart.</td>
</tr>
<tr>
<td>Let the long mission be carried out with certainty.</td>
<td>Message queue, task queue, workflow engine, retest, dead line</td>
<td>Step execution, failure recovery, suspension of running, tasking</td>
<td>Multiwheel Agent missions cannot normally be completed in a HTTP request</td>
</tr>
<tr>
<td>Not to mess with the system in the simultaneous distribution.</td>
<td>Business, locks, optimism, pessimism, cynicism, updates.</td>
<td>Repeated execution, disorder, overwriting, competition</td>
<td>Agent services are naturally subject to multiple worker, multiple copy, double consumption.</td>
</tr>
<tr>
<td>Let the services work together without losing control.</td>
<td>RCC/API, event driven, service discovery, time overtime, melting, retreat from re-testing</td>
<td>Multi-service collaboration, dependence on malfunctions, border isolation</td>
<td>Agent will soon be decorated, dispatch, tool implementation, retrieval, billing</td>
</tr>
<tr>
<td>See the system's real state.</td>
<td>Logs, indicators, Tracing, mission audits, evaluations, SLOs, alarms</td>
<td>Where is slow, where is wrong, where is expensive, where is degradation?</td>
<td>Agent's failure could be 500, or it could be the deterioration of mission quality and the misuse of tools.</td>
</tr>
<tr>
<td>Governance risks, competencies and costs</td>
<td>Authority, segregation of tenants, budget, quotas, sandboxes, strategic engines, audit</td>
<td>Overstepping authority, misuse, resource grabs, out of control.</td>
<td>Agent is capable of action, so it's more risky than a regular chat interface.</td>
</tr>
<tr>
<td>Execute Language and Runtime Base</td>
<td><code>Go</code>、<code>goroutine</code>/<code>channel</code>、<code>context</code> Cancel, connect the pool,<code>Redis</code>, message queue, workflow engine, object storage</td>
<td>High-level co-activity I/O, resource control, undispersed, insulated and complex balance, and which capacity to write and which to hand over to independent infrastructure</td>
<td>Agent backends often are both API services and worker systems, and pay the state, queue, cache to mature components</td>
</tr>
</tbody></table>
<p>This table summarizes the issues to be addressed in this paper: databases, Go, co-opt, locks, queues, observations, etc., which seem to be scattered, and are in fact answering the same question. Agent's service depends on the back end catching up with its state, traffic, failure and cost.</p>
<h2>Why, AI Agent, once it's serviced, it becomes a back end problem.</h2>
<p>Locally run Agent is not a service.</p>
<p>Production-level services mean at least a few things are set up simultaneously:</p>
<ul>
<li>It can be called by multiple users, not only once on your own machine.</li>
<li>It can be executed in minutes or even hours, rather than only for short requests.</li>
<li>It recovers when it is wrong, not by running again.</li>
<li>Its state can be preserved, queried, audited, not merely in the process memory.</li>
<li>Its costs, privileges and staleness are manageable, not to run.</li>
</ul>
<p>Once the task is performed over a minute, the tool is called and it recovers after failure, the question is how to go from model to system. Anthropic is here. <a href="https://www.anthropic.com/engineering/building-effective-agents">Building effective agents</a> . Sees angent as a + tool + loop system for environmental feedback and alerts developers to delays and costs. As long as the task is no longer synchronized, short-lived, single-user, single-process, it is already in the design of the service system.</p>
<h2>Capacity to capture requests and traffic</h2>
<p>The first threshold for any production system is whether it can accommodate the request.</p>
<p>It sounds like a piece of crap, but Agent services here are easier to lose than normal CRD applications. The reason is simple: ordinary interfaces are usually short calls of several dozen milliseconds to hundreds of milliseconds; the Agent platform often has two types of traffic at once, one for real-time access and the other for start-up traffic that induces longer missions, tool calls and back-office resource occupancy. If the entrance level is not unconnected with traffic, quotas and implementation modes, the system will quickly be dragged from a slower and occasional pace to a full station.</p>
<p>Today, the back end usually relies on these technologies to solve such problems:</p>
<ul>
<li><code>API Gateway</code> Or access layers are first identified, restricted, routed, quota controlled.</li>
<li>The load balance distributes traffic to multiple copies of services to avoid single-point filling.</li>
<li>Web/API services are as non-existent as possible, allowing copies to be added, re-established and migrated at any time.</li>
<li>Automatic scaling up is based on examples of increases/decreases in CPU, QPS, queue length or customized indicators.</li>
<li>Connecting pools and caches protect databases from downstream dependence and avoid each request hitting the back end directly.</li>
</ul>
<p>Why is the “state of absence” almost a prerequisite for scale? Because once a copy of the service can be pulled up and destroyed, you cannot hide the task, session and billing status in the local memory. Google Cloud is here. <a href="https://cloud.google.com/architecture/scalable-and-resilient-apps">Patterns for scalable and resilient apps</a> It's a suggestion. <code>Aim for statelessness</code>and emphasis is placed on layer-based load balance and indicator-based autoscaling. AWS is in <a href="https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/design-principles.html">Principles for the design of Relibility Pilar</a> It also places horizontal expansion, stop guessing capacity and automated management changes in the underlying principles. Such a design would also allow us to make better use of the capacity of cloud service providers to control the complexity and cost of development.</p>
<p>Put it in the Agent scene, and that's even more obvious. Web level should not normally run the entire Agent in a request, but should:</p>
<ul>
<li>Validation of the request.</li>
<li>Records task metadata.</li>
<li>It is decided whether it is a simultaneous short or a long one.</li>
<li>Returns the task ID, status link or streamback as soon as possible.</li>
</ul>
<p>The entrance level is responsible for the flow, and should not swallow the entire Agent complex.</p>
<h2>Capacity to preserve status and to ensure data accuracy</h2>
<p>Once you stop serving yourself, the database becomes the hub for system correctness from where you store the data.</p>
<p>In a real Agent platform, more people than beginners usually expect to be sustained:</p>
<ul>
<li>User and Tenant Information</li>
<li>Session & Task Status</li>
<li>Tool Call Record</li>
<li>Task Input Output</li>
<li>Quotas and billing data</li>
<li>Audit log</li>
<li>Failed and Retry History</li>
</ul>
<p>Here's another one: Agent memory. Work memory, plot memory, long-term memory sounds like a model topic, but once it is left in between sessions and re-starts, it is still essentially a storage problem: Yes. <code>schema</code>And if you can access it, you have to be able to manage it. I'm on the line.<a href="/en/blog/2026/03/21/agent-memory-panorama/">From Memory Generation to Memory Governance: A Panorama of Age Memoory</a>It's been done alone.</p>
<p>If these things are only in memory, the world will break once the system is restarted. The mission and state of affairs will disappear, as will the information on “where exactly this time was going, why failed, whether there was a double charge, whether there was a double call tool”.</p>
<p>Today's end is usually stored in layers:</p>
<ul>
<li>The relationship database is responsible for the core service status, such as user, task, billing, quota, status machine.</li>
<li>Object storage is responsible for large objects, such as long text, attachments, log filings, tool output snapshots.</li>
<li>Cache is responsible for reading hot spots, speeding up sessions, short-term weight removal and pressure relief.</li>
<li>Searching indexes or vector indexes is responsible for searching for type capabilities, but usually does not assume the central truth of the matter.</li>
</ul>
<p>Google's here. <a href="https://cloud.google.com/architecture/scalable-and-resilient-apps">Patterns for scalable and resilient apps</a> The database section clearly indicates that the value of the relationship database lies in services, strong consistency, citation integrity and cross-table queries; PostgreSQL is <a href="https://www.postgresql.org/docs/current/mvcc-intro.html">MVCC Document</a> It also indicates that multiple versions of the simultaneous control are aimed at maintaining consistency, segregation and minimizing lock competition in a multi-user environment. The database is not as simple as putting data in and taking it out. It is responsible for maintaining the state of the world in a manageable context, with multiple users, multiple services and multiple conditions.</p>
<p>Starting with this chapter, these words need to be approached first:</p>
<ul>
<li>Service</li>
<li><code>schema</code></li>
<li>Index</li>
<li><code>MVCC</code></li>
<li>Wait.</li>
<li>Coherence</li>
</ul>
<p>The opening of the article does not require a mastery of their bottom line, but it is enough to know what each is responsible for:</p>
<ul>
<li>The service is responsible for the consolidation of the steps of the upgrade. Either we succeed together or we fail together, and we cannot leave a half-state.</li>
<li><code>schema</code> It's responsible for explaining what the system is really doing. The relationship between fields, types, constraints and tables is based on it to draw the boundaries.</li>
<li>The index is responsible for keeping the query running after the data has grown. Without a suitable index, many queries eventually degenerate into a line scan.</li>
<li><code>MVCC</code> And locks handle and read and write. When multiple people change the data at the same time, the system needs to know who can see which version and who has to wait.</li>
<li>The latter are responsible for dealing with duplicate implementation. When the same task, message or request is retried, the result should remain correct, not duplicate deductions, creations, calls.</li>
<li>Coherence is responsible for ending the state of affairs. Orders, bills, assignments, audit records cannot fight each other.</li>
</ul>
<p>The bottom-up of these mechanisms is worth breaking. Here's only one judgment: for Agent, the database is the only credible state of the world, not a piece of collateral that is written off.</p>
<h2>Capacity to deliver a long-term mandate with reliability</h2>
<p>Normal Web requests the world most favorite, to come in, and then return in a few dozen milliseconds. But Agent is often not.</p>
<p>Many Agent missions are naturally not short-requested:</p>
<ul>
<li>Need multiple rounds</li>
<li>Multiple tools to call</li>
<li>Need to wait for external dependence</li>
<li>We need to try again after failure.</li>
<li>Need to make the middle state permanent.</li>
<li>It takes a long time to execute, but not to take over the front-end connection.</li>
</ul>
<p>Complex resonating missions may take several minutes, so the platform needs to be a step-start, a rotational state, and to be implemented without a front-end connection. Anthropic is here. <a href="https://www.anthropic.com/engineering/harness-design-long-running-apps">Harness design for long-running application development</a> It also further binds the problem into a more durable operating-time structure: planning, generating, assessing the links need to be placed in an external system that can sustain recovery and continuous validation. The queue system, Amazon SQS, is here. <a href="https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/standard-queues.html">Standard queues documents</a> The standard queue is <code>at-least-once delivery</code>The information may be repetitive or confusing.</p>
<p>Back to the project, there are a few simple but hard-on requirements:</p>
<ul>
<li>The mission is to be decomposed with the request.</li>
<li>The implementation backstage will fail, try again, and repeat.</li>
<li>The system must be designed to “enable the mission to be taken back” and cannot be expected to “exit at once”.</li>
</ul>
<p>Today, the back end is usually addressed by these structures:</p>
<ul>
<li>The news queue is responsible for the opening of the entrance request and the execution of the backstage. There's a lot of missing details: if you don't write databases and send messages in the same business, there's a crack that's "tasked without delivery" or "dropped without a library" and it's common in engineering. <code>transactional outbox</code> Deal.</li>
<li>Task queues are responsible for allocating tasks to multiple work-offs.</li>
<li>The workflow engine is responsible for preserving progress, coordinating steps and processing recovery.</li>
<li>The retest strategy and the Death Line are responsible for turning "bad once" into "diagnostic failure".</li>
<li>The questioner, Webbook or WebSocket is responsible for re-referraling the different task to the user.</li>
</ul>
<p>For Agen, this layer is easy to get into. Agent is not a common walker, but often carries a state, transfers tools, cross-decisions, may be interrupted by humans, or may be restored and continue running. Dureable Exchange is much more important than "just running backstage." Workstream engines like Temporal are in <a href="https://docs.temporal.io/workflows">Workflows documents</a> It defines the flow of work as a sustainable, re-alignable implementation module that can continue from the last progress after the process has collapsed, in line with the request of Agent to “not lose” the way. And how this durable operation is going to fit into Agent's plan-implementation cycle, I'm here.<a href="/en/blog/2026/04/04/understanding-agent-harness/">"Harness: What is it? From model + shelter to engineering, product-user-friendly shell?"</a>There is a discussion closer to Agent.</p>
<p>If the layer is abstracted and simplified:</p>
<ul>
<li>Requesting level answer: "What are you doing?"</li>
<li>The work stream says, "Where is it now?"</li>
<li>The workingr layer answers "How exactly to run this step"</li>
</ul>
<p>The back end separates the three layers, so Agent has the opportunity to expand to multiple users and long assignments.</p>
<h2>The ability to keep the system in the same place.</h2>
<p>In the Agent service, the question of correctness is first and foremost.</p>
<p>The following are some of the most common scenarios:</p>
<ul>
<li>Two people working together got the same job.</li>
<li>A tool was retried after the time elapsed, but the first was partially successful.</li>
<li>One user has requested "re-run" twice in a row.</li>
<li>The same session state is updated simultaneously with two copies.</li>
<li>Queue because <code>at-least-once delivery</code> The same message was sent again.</li>
</ul>
<p>If the system is not controlled in parallel, these situations do not just slow down the system, but result in a direct error: double deduction, repeat message, back-state, overwriting, unsequenced writing, ghost results.</p>
<p>Today, these mechanisms are usually used to cover the back end:</p>
<ul>
<li>Questions, which tie together a set of modifications that must be completed at the atom.</li>
<li>Locked, where necessary, to protect the conflict point visibly.</li>
<li>Optimistic and controlled, with the version number or condition updated to determine “any other person has changed first”.</li>
<li>Pessimism and control are being exercised by direct and serial visits to high-conflict points.</li>
<li>The hyphenation key and the de-weighting table ensure that repeated requests do not duplicate the validity.</li>
<li>Atomic operation and condition update to avoid being separated from “read-and-write” by a different side.</li>
</ul>
<p>At the back end of Agent, the most common pattern is often two moves: a key is given to each mission at the entrance (or a key is given).<code>idempotency key</code>) repeats the direct hit-to-heavy table instead of running it again; updates the condition machine with a version number when advancing (<code>compare-and-set</code>And let's say "I think state or A" fail when someone else has changed to B, not overwhelm. How these two approaches fit into specific storage and line language is worth a separate detail.</p>
<p>The problem of co-existence is not unique to a particular technology. PostgreSQL <a href="https://www.postgresql.org/docs/current/mvcc-intro.html">MVCC Document</a> The blog is a forum for discussion of segregation and the conflict between reading and writing in a multi-user environment.<a href="https://www.postgresql.org/docs/current/explicit-locking.html">Explicit Locking</a> This indicates when the application requires a visible lock. This side of the queue, the SQS official file reminds the standard queue that you may repeat the delivery message. Go's. <a href="https://go.dev/blog/pipelines">Pipelines and cancellation</a> and <a href="https://go.dev/blog/context">Context</a> Repeated emphasis on de-communication and <code>goroutine</code> Exiting management because the co-processing process would compete for locks and would release resources, hang upstream and bring down systems as a result of the cancellation of mishandling.</p>
<p>Why do you learn to co-op, lock, and thread? Because without them, you can't answer the most basic questions:</p>
<ul>
<li>Who has the right to change the same status?</li>
<li>Who's first and who's first?</li>
<li>What if the same mission is twice carried out?</li>
<li>After a request is cancelled, backstage. <code>goroutine</code> You want to keep running?</li>
<li>How does the system ensure that errors are not magnified when multiple copies are expanded?</li>
</ul>
<p>What is really difficult for service systems is to do it in parallel, not as much as possible.</p>
<h2>Capacity to coordinate services without losing control</h2>
<p>The system is large and complex and it will spill naturally.</p>
<p>You may have had one service at first: to take requests, to call models, to return results. But as long as you really start serving the users, the duties will soon be divided:</p>
<ul>
<li>Certification services</li>
<li>Mission dispatch services</li>
<li>workingker Executive</li>
<li>Retrieving services</li>
<li>Tool Proxy Layer</li>
<li>Quotas and billing services</li>
<li>Notification and callback services</li>
</ul>
<p>This has little to do with the aesthetics of the architecture, and more with the real division of labour brought about by size. The flow models, failure models, delay requirements and extension patterns of these components are inherently different. Google Cloud is here. <a href="https://cloud.google.com/architecture/scalable-and-resilient-apps">Patterns for scalable and resilient apps</a> Liang! <code>loose coupling</code> and <code>modular architectures</code> Positioning ahead also indicates that independent services can be issued, extended and managed separately. The distribution is not a “higher” form, and in many cases it is only complicated and the system exposes the boundary.</p>
<p>Today, at the back end, these parts are usually coordinated:</p>
<ul>
<li>Synchronize calls with RPC or HTTP API.</li>
<li>The first step of the project is to create a process of decoupling the event and the news.</li>
<li>The service discovery and deployment centre is used to manage the dependency relationship.</li>
<li>♪ Use timeout, melt (<code>circuit breaker</code>), quarantine and avoidance re-test to control the spread of malfunctions, with back pressure, if necessary (<code>backpressure</code>) Transmit overload upstream, rather than drag down the caller by some slow dependence.</li>
<li>Use clear data boundaries and ownership to prevent all services from writing a bad watch together.</li>
</ul>
<p>The intuition that this chapter is to build is simple: the value of split services lies in separating different failure models from expansion needs.</p>
<p>The Agent platform can easily have three completely different components at the same time:</p>
<ul>
<li>Low delay entrance</li>
<li>Implementation of high uncertainty models/tools</li>
<li>A robust and consistent status and accounting system</li>
</ul>
<p>To shove these three things into the same process would be to let the most unstable parts determine the floor of the system. The value of the split is direct here: to keep the uncertainty of models and tools outside the state and the billing system.</p>
<h2>The ability to see the system's real state.</h2>
<p>If you can't see what the system is doing, you're not really running the service, just waiting for it to happen.</p>
<p>And the Agent service needs more observation than the normal back end because it has more failure patterns:</p>
<ul>
<li>Request failed</li>
<li>♪ Worker, blow ♪</li>
<li>Tool Call Abnormal</li>
<li>Model output quality is down</li>
<li>Task Card in Middle</li>
<li>The number of retries has soared.</li>
<li><code>token</code> The cost is soaring.</li>
<li>Some tenant is breaking up the system.</li>
</ul>
<p>Today, the back end is usually built on these systems:</p>
<ul>
<li>Log, answer "What happened."</li>
<li>Indicator, answer “how and where the trend is abnormal”.</li>
<li><code>Tracing</code>, answers “what components a request or mission passes through and where it is slow”. Here's the path path to worker, tool call and downstream service, or a link to the Agent mission will be cut off on the line.</li>
<li>Mission Audit, answer "This is exactly what Agent did."</li>
<li>Quality assessment, reply: “Is the result still on track?”. Anthropic is here. <a href="https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents">Demystifying evals for AI agents</a> It was emphasized that Agent ' s assessment should focus on the completion of the real mission, rather than on the single-step output, so that quality indicators themselves should be part of the production observations.</li>
<li><code>SLO</code> And the alarm, the answer is “which questions have affected the user experience and should be addressed immediately”. Google SRE is in <a href="https://cloud.google.com/blog/products/devops-sre/how-to-design-good-slos-according-to-google-sres">How to design good SLOs</a> - The reminder. - Okay. <code>SLO</code> It is to be determined by the true user-conscious experience, rather than by a pile of indicators that no one will act on.</li>
<li>Cost control. Answer, "Why is this on the line?" <code>token</code> / GPU / I/O costs are pushed up."</li>
</ul>
<p>For Agent, observations cannot be limited to the traditional CPU, memory and interface delays:</p>
<ul>
<li>You'll have to see the success of the mission.</li>
<li>You have to see the failure rate of the tool.</li>
<li>You have to see the average number of retries.</li>
<li>You have to see the length of the stopover.</li>
<li>You have to look at the unit cost of each type of mission.</li>
<li>You have to see whether quality indicators are quietly declining.</li>
</ul>
<p>Observations themselves are not enough. Agent services need to expose critical intermediate states as subject to intervention and replacement: Whether the results can be reviewed, whether the tool is replayed, whether the failure steps can be retested alone, and whether a model or route strategy can be converted when it is degraded. Otherwise, the system is operating on its face, and in fact, once an error has occurred, it can only push the box from the final answer.</p>
<p>Without observation, the Agent service will soon become a box maintained by instinct.</p>
<h2>Capacity to manage risks, competencies and costs</h2>
<p>The first problems that many Agent products are exposed are not necessarily modelling capabilities, but are beyond control over authority, safety or cost.</p>
<p>The reason is straightforward: the normal chat interface is mainly talking; the Agent system is starting to do it. Once it can do something, the risk model changes completely:</p>
<ul>
<li>It could call dangerous tools.</li>
<li>It may over-read someone's data.</li>
<li>It may rob resources in a multi-tenant environment.</li>
<li>It could blow costs because it was out of control.</li>
<li>It can create large-scale side effects in the tool chain.</li>
</ul>
<p>Anthropic is here. <a href="https://www.anthropic.com/engineering/building-effective-agents">Building effective agents</a> It is a clear reminder that Agent ' s autonomy entails higher costs and complex error risks, and therefore requires extensive testing in the sandbox environment, with the right guardrails. Anthropic is here. <a href="https://www.anthropic.com/engineering/writing-tools-for-agents">Writing effective tools for agents</a> It further describes the design of the tool, parameters constraints, the structure of the return <code>token</code> Efficiency directly shapes the boundaries of Agent's behaviour. OpenAI is in <a href="https://openai.com/index/harness-engineering/">Harness engineering</a> The same thing was emphasized: the environment, the feedback and the control system determined the available boundaries of Agent.</p>
<p>Today, the back end is usually governed by these capabilities:</p>
<ul>
<li>The power of attorney determines who can access what.</li>
<li>Tenants are segregated to prevent a client from influencing the entire platform.</li>
<li>Budget and quotas, with maximum resources for each user, model, tool, mission.</li>
<li>Sandboxes and permission borders, and locking high-risk movements into restricted environments.</li>
<li>Strategic engines and audit logs to ensure that key actions of the system are traceable and subject to review.</li>
<li>Speed limit and resource mobilization to prevent certain types of tasks from dragging the entire platform down.</li>
</ul>
<p>This system of governance belongs to the Agent Trust Border. What is credible, what is stopped on what level, who is responsible for crossing borders, is in itself a framework decision. I'm here.<a href="/en/blog/2026/04/12/agent-trust-boundary-openclaw-bettafish/">"BettaFish, Mirofish, OpenClaw and Agent's Trust Border"</a>It's been devoted to this border. Where is it?</p>
<p>From the point of view of service operations, there is also a frequently overlooked but realistic point: costs themselves are also system-constrained.</p>
<p>Agent services are prone to a false boom: they look powerful and show good performance, but backstage is a high frequency retest, a hyper-long context, low-life cache, rough tools call and budgetless model implementation.</p>
<p>Authority, restricted flow, quotas, budgets and audits are not added after they are online. Agent needs these back-end skeletons from the start.</p>
<h2>Select the appropriate executing language and runtime base</h2>
<p>The first seven capabilities answer “what the system should do”. But there's a question of how hard to get around: what is it written and what is running over?</p>
<p>My personal default at the service level is <code>Go</code>I'm sorry. An Agent platform is born amphibious: it is an API service that carries real time requests, and a workingrer system that runs long missions. Both are highly concentrated I/O, need to manage hundreds of simultaneous execution, need to stop overtime and cancel. Go's. <code>goroutine</code> The government has been making it cheaper to start a light stream for each mission.<code>channel</code> and <code>select</code> The government has been able to provide a better understanding of the situation and the need for a better understanding of the situation.<code>context</code> The timeout and cancellation are being passed down along the call chain.</p>
<p><code>context</code> Cancel spreading this thing, it's pretty deadly at the Agent backend. A user canceled the mission, or the request was timed out, running backstage. <code>goroutine</code>The tools that are waiting for them to be called and the connections that are being held must be withdrawn. Otherwise they leak resources, hang up upstream and slow down the system. Go, the official two. <a href="https://go.dev/blog/pipelines">Pipelines and cancellation</a> and <a href="https://go.dev/blog/context">Context</a> This is what is said repeatedly: the difficulty of the simultaneous procedure is often not how to go in parallel, but how to stop cleanly.</p>
<p>But language solves only part of the problem. It is also necessary to determine which competencies fit in Go and which should be directly transferred to a mature independent infrastructure project.</p>
<p>What is appropriate for staying in the Go process is usually:</p>
<ul>
<li>Service organization and processing of requests: route, intermediate assurance, synchronizing short and long-range tasks.</li>
<li>The Queen of the West is a great man.<code>goroutine</code> Ike.<code>context</code> Cancel and exit with extra-time, elegantness (%1)<code>graceful shutdown</code>）。</li>
<li>Connectivity management: database connectors pool, downstream customer re-use, back pressure control.</li>
</ul>
<p>And almost all of this should be built on its own, but rather introduce independent infrastructure projects.</p>
<ul>
<li><code>Redis</code>Cache, speed of session, distributed lock, swirling and dereduced count, limit flow counters, and even light queues. The "fast and temporary" state of the backend of Agent is all here.</li>
<li><code>PostgreSQL</code>: Core business truth, task status machine, audit records. Where there is a need for greater consistency and references to integrity, this should usually be the place.</li>
<li>Message/Task System (Performance)<code>NATS</code>、<code>Kafka</code>On the clouds. <code>SQS</code> • Decoupling of the entry request and backstage to hold the peak.</li>
<li>Workstream engine (%1)<code>Temporal</code>): Hand over to it the "sustainable, re-playable, failed" dirable excursion instead of using the database hand to rub a semi-finished machine.</li>
<li>Object Storage (<code>S3</code>、<code>MinIO</code>): long text, attachments, tool output snapshots, log archive.</li>
<li>Observable stacks (in %2)<code>OpenTelemetry</code> + <code>Prometheus</code> / <code>Grafana</code> / <code>Jaeger</code>): The bottom of the factual standards for indicators, tracking, and alerting.</li>
</ul>
<p>This division of labour is a little more simple: language determines how you write, infrastructure determines whether you can recreate some of the correctness.</p>
<p>I didn't deliberately dig any of the components in this chapter. The co-production model for Go, the interaction of the connect pool and the back pressure,<code>Redis</code> and <code>PostgreSQL</code> their respective applicable borders,<code>Temporal</code> Each of the workstreams is worth a single piece of detail, and I'm going to follow this landscape. Here, just a judgement is needed: the selection is essentially about deciding which level of correctness should go.</p>
<h2>How back-end technologies can address these issues separately</h2>
<table>
<thead>
<tr>
<th>Type of problem</th>
<th>Typical technology</th>
<th>It solves the boundaries.</th>
</tr>
</thead>
<tbody><tr>
<td>Multiple visitors are visiting, traffic peaks.</td>
<td>Gateway, load balance, non-state service, automatic build-up</td>
<td>Let the system catch changes in demand, not on a single machine.</td>
</tr>
<tr>
<td>We can't lose our status. We can't mess up our data.</td>
<td>Relationship database, services, index, object storage</td>
<td>It's a system that has a lasting truth and that can be correct in a multi-user environment.</td>
</tr>
<tr>
<td>It's too long. We can't keep taking over requests.</td>
<td>Queue, backstage worker, workflow engine, rounding/Webhuk</td>
<td>Dismantling user requests and backstage execution and supporting recovery and re-testing</td>
</tr>
<tr>
<td>The message will repeat, the worker will compete.</td>
<td>♪ Wait, lock, conditions are renewed, optimism is shared, pessimism is shared ♪</td>
<td>Make repeated and simultaneous changes not directly bad state</td>
</tr>
<tr>
<td>The more services are being cut down, the more complex the dependence is.</td>
<td>RPC, event, timeout, melting, service discovery, configuration of centre</td>
<td>Let the collaboration take place, but keep the malfunctions at the border. Internal</td>
</tr>
<tr>
<td>I don't know where it's slow, where it's wrong, where it's expensive.</td>
<td>Logs, indicators, Tracing, Audit, Evaluation, SLO</td>
<td>Make the service operational, not personal instincts.</td>
</tr>
<tr>
<td>Models and tools will overstep power and burn money.</td>
<td>Secrecy, sandbox, tenant segregation, budget, quotas, strategic engines</td>
<td>Lets the power of Agent fall within the controllable boundary.</td>
</tr>
<tr>
<td>You're right. You're gonna get it down.</td>
<td><code>Go</code> + <code>goroutine</code>/ Connect pool/<code>context</code> Cancel.<code>Redis</code>♪ News lines, ♪<code>Temporal</code>, Object Storage, Otel</td>
<td>Let all the capabilities in the front get real and good, not stop on the map.</td>
</tr>
</tbody></table>
<h2>Final judgment.</h2>
<p>I am now making a clear judgement on this issue.</p>
<p>If you're just trying to get a person who's actually going to run out of town, the most important thing is that... <code>model</code>、<code>prompt</code>、<code>tool use</code> and <code>harness</code>。</p>
<p>But if you want to make it a big service, the problem center will be moved. You need to answer:</p>
<ul>
<li>How do you answer the request?</li>
<li>How do we keep it?</li>
<li>How do you run a long mission?</li>
<li>How do you guarantee it's right?</li>
<li>How does the service work?</li>
<li>How did the system get observed?</li>
<li>How risks, competencies and costs are being managed</li>
<li>What language and infrastructure are you using to drop them?</li>
</ul>
<p>From Demo to service, the biggest change for Agent is to start being bound by the back end.</p>
<p>These technologies solve one thing together:<code>Scaling Up</code>。</p>
<h2>References and official entry points</h2>
<ul>
<li><a href="https://openai.com/index/harness-engineering/">OpenAI: Harness engineering</a></li>
<li><a href="https://developers.openai.com/api/docs/guides/background">OpenAI API: Background mode</a></li>
<li><a href="https://www.anthropic.com/engineering/building-effective-agents">Anthropic: Building effective agents</a></li>
<li><a href="https://www.anthropic.com/engineering/harness-design-long-running-apps">Anthropic: Harness design for long-running application development</a></li>
<li><a href="https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents">Anthropic: Demystifying evals for AI agents</a></li>
<li><a href="https://www.anthropic.com/engineering/writing-tools-for-agents">Anthropic: Writing effective tools for agents</a></li>
<li><a href="https://cloud.google.com/architecture/scalable-and-resilient-apps">Google Cloud: Patterns for scalable and resilient apps</a></li>
<li><a href="https://cloud.google.com/blog/products/devops-sre/how-to-design-good-slos-according-to-google-sres">Google Cloud Blog: How to design good SLOs, according to Google SREs</a></li>
<li><a href="https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/design-principles.html">AWS Well-Architected: Reliability Pillar design principles</a></li>
<li><a href="https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/standard-queues.html">Amazon SQS: Standard queues</a></li>
<li><a href="https://www.postgresql.org/docs/current/mvcc-intro.html">PostgreSQL: MVCC introduction</a></li>
<li><a href="https://www.postgresql.org/docs/current/explicit-locking.html">PostgreSQL: Explicit locking</a></li>
<li><a href="https://docs.temporal.io/workflows">Temporal Docs: Workflows</a></li>
<li><a href="https://go.dev/blog/pipelines">Go Blog: Pipelines and cancellation</a></li>
<li><a href="https://go.dev/blog/context">Go Blog: Context</a></li>
</ul>
