---
title: "从最小 PPO/GRPO Trainer 到 Agentic Tool Calling：读懂 verl 的训练主线"
date: 2026-03-19 22:00:00 +0800
categories: [AI, Research]
tags: [verl, Agentic RL, Tool Use, PPO, GRPO, ReTool]
author: Hyacehila
excerpt: "从手写最小 PPO/GRPO-style trainer 的心智模型切入，解释 verl 为什么适合做 agentic RL、multi-turn rollout 与工具使用优化。"
---

# 从最小 PPO/GRPO Trainer 到 Agentic Tool Calling：读懂 verl 的训练主线

很多人第一次看 `verl`，会觉得它有点“怪”：它不是那种把 agent、tool、memory、workflow 全都包成黑盒的一站式框架，但它也不是只会跑单轮 PPO 的训练脚手架。它更像一个故意把训练链路拆开的 RL runtime：你可以从最小 trainer 的角度读它，也可以一路读到 async rollout、multi-turn tool use、Agent Loop 乃至 ReTool 这样的 recipe。

这篇文章的目标，就是把这条线串起来。

## 研究问题与范围 / 一页式结论

本文只回答一个问题：**如果你已经能手写一个最小 PPO/GRPO-style trainer，那么应当怎样理解 `verl`，以及为什么它会自然长到 agentic tool calling？**

证据标准先说清楚：

- 关键事实优先引用官方仓库、官方文档和论文，如 [`verl` README](https://github.com/verl-project/verl)、[`Agentic RL Training`](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)、[`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)、[`Multi-turn Rollout Support`](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)、[`verl-recipe/retool`](https://github.com/verl-project/verl-recipe/tree/main/retool) 与 [`ReTool` 论文](https://arxiv.org/abs/2504.11536)。
- 解释性判断会显式写成“分析”，避免把推断写成官方结论。
- 文章不扩成泛 agent 综述；只在最后用很轻量的 2025 背景做定位。

如果只看结论，我的判断是：

1. `verl` 的核心身份仍然是 **RL training library**，而不是通用 agent SDK；官方 README 直接把它定义为“flexible, efficient and production-ready RL training library for LLMs”，并说明它是 [`HybridFlow`](https://arxiv.org/abs/2409.19256v2) 的开源版本。[`verl` README](https://github.com/verl-project/verl)
2. 之所以说它“底层”，是因为它把训练真正关心的边界暴露出来：trainer、rollout、reward、worker、inference engine、server manager、tool parser、trace，而不是把这些细节全部藏在一个高层 agent runtime 后面。[`verl` README](https://github.com/verl-project/verl), [`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)
3. 之所以说它“开放”，是因为这些边界几乎都允许替换：训练后端可以是 FSDP/Megatron-LM，rollout 可以接 vLLM/SGLang/HF，reward 可以是函数式，tool 与 interaction 也都能从配置和类注册扩展进去。[`verl` README](https://github.com/verl-project/verl), [`Multi-turn Rollout Support`](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
4. `verl` 对 tool calling 的关键贡献，不只是“支持多轮对话”，而是把 **异步 rollout、server/client split、token-based generate、sticky session、response masking、trace** 这些 RL 训练真正需要的机制放在了同一个系统里。[`Agentic RL Training`](https://verl.readthedocs.io/en/latest/start/agentic_rl.html), [`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html), [`Trace Function Usage Instructions`](https://verl.readthedocs.io/en/latest/advance/rollout_trace.html)
5. `ReTool` 是理解这条主线最好的案例：论文把问题定义为“战略性工具使用”，而官方 recipe 则把它具体接成 `cold-start SFT -> multi-turn async rollout -> sandbox execution -> reward shaping -> RL optimization` 的完整训练闭环。[`ReTool` 论文](https://arxiv.org/abs/2504.11536), [`verl-recipe/retool`](https://github.com/verl-project/verl-recipe/tree/main/retool)

**分析：** 如果你从“我想找一个最省心的 agent 框架”出发，`verl` 不是第一选择；但如果你从“我想精确控制 RL 训练中的 trajectory、mask、tool response、rollout concurrency 和 infra placement”出发，它的设计就会显得非常顺手。

## 为什么说 `verl` “底层但开放”

先看官方自我定义。README 给出的关键词是：**flexible**、**efficient**、**production-ready**，并且明确强调三件事：

- 用 hybrid-controller programming model 扩展 PPO、GRPO 等 RL dataflow；[`verl` README](https://github.com/verl-project/verl)
- 用模块化 API 无缝接现有 LLM infra，例如 FSDP、Megatron-LM、vLLM、SGLang；[`verl` README](https://github.com/verl-project/verl)
- 用灵活 device mapping 做训练与生成阶段的资源编排。[`verl` README](https://github.com/verl-project/verl)

“底层”主要体现在两个层面。

第一，它关心的是**训练数据流**，而不是“对话产品”的抽象。你在 `verl` 里最常看到的是 `trainer`、`rollout`、`ref`、`critic`、`reward`、`worker group`、`server manager` 这类对象，而不是 workflow node、business toolchain、memory graph 这类应用层对象。[`verl` README](https://github.com/verl-project/verl), [`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

第二，它对“可训练 token 边界”极度敏感。官方文档在 agentic RL 部分反复强调：如果 decode-encode 之后 token 不一致，PPO 训练会偏离策略分布，甚至不收敛；这也是它坚持提供 token-in token-out `generate` 接口，而不是只依赖 chat completion 的原因。[`Agentic RL Training`](https://verl.readthedocs.io/en/latest/start/agentic_rl.html), [`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

“开放”则体现在它没有把任何一层写死。

- 算法层：README 列出了 PPO、GRPO、GSPO、ReMax、RLOO、DAPO、PRIME、DrGRPO 等多种训练方式，且不少 recipe 已拆到独立的 [`verl-recipe`](https://github.com/verl-project/verl-recipe) 仓库维护。[`verl` README](https://github.com/verl-project/verl)
- rollout 层：同一套训练主线可以接 vLLM、SGLang 甚至 HF Transformers。[`verl` README](https://github.com/verl-project/verl)
- tool 层：官方 `multiturn` 文档要求工具继承 `BaseTool`，通过 YAML 配置装配；MCP 工具也有独立配置路径。[`Multi-turn Rollout Support`](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
- interaction 层：除了工具响应，还可以接“模拟用户/环境互动”；在代码里，`ToolAgentLoop` 对 tool 和 interaction 是两套初始化与状态流转逻辑。[`tool_agent_loop.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/tool_agent_loop.py)

**分析：** 所谓“底层但开放”，并不是说 `verl` 提供了一个更薄的 API，而是说它把训练系统真正重要的可替换边界公开了出来。你看到的复杂度，不是多余复杂度，而是它没有替你提前做掉那些训练时往往必须自己决定的系统选择。

## 从最小 PPO/GRPO trainer 到 `verl` 主训练链路：dataset -> trainer -> rollout -> reward -> worker/server

如果先不看 `verl`，一个最小 PPO/GRPO-style trainer 的心智模型其实很简单：

```python
for step in training_steps:
    batch = sample(dataset)
    responses = policy.generate(batch.prompts)
    rewards = reward_fn(batch, responses)
    advantages = estimate_advantage(rewards, responses)
    update_policy(policy, responses, advantages)
```

真正复杂的地方，从来都不是这五行伪代码，而是每一步背后的工程拆分：prompt 怎么组织、response 怎样对齐、reward 在哪算、rollout 如何并发、训练和推理如何共享或切换资源。

`verl` 的价值，就在于它把这些“最小 trainer 背后真实存在的系统问题”拆成了清晰链路。

### 1. dataset

在 agentic RL 文档里，官方要求至少打开两个开关：

- `data.return_raw_chat=True`
- `actor_rollout_ref.rollout.mode=async`

如果是 Tool Agent Loop，数据里还需要额外的 `agent_name` 字段，用来决定选择 `tool_agent_loop` 还是默认的 `single_turn_agent`。[`Agentic RL Training`](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)

这一步的意思很直白：进入 agentic RL 之后，dataset 不再只是“prompt token ids 的来源”，而要把原始 chat 消息、额外字段、甚至 interaction kwargs 都留给 rollout 侧处理。

### 2. trainer

`Agent Loop` 文档把单个 PPO step 拆成 rollout phase 与 train phase。在 rollout phase 中，`PPOTrainer` 先从 dataset 采样，然后调用 `AgentLoopManager.generate_sequences`；等所有 trajectory 回来后，再进入训练更新阶段。[`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

这跟“手写最小 trainer”的关系非常直接：`PPOTrainer` 仍然是总调度者，只是它不再直接把 prompt 喂给一个同步 `generate`，而是把 rollout 交给 Agent Loop 系统去并发完成。

### 3. rollout

在 `verl` 的 agentic RL 架构里，rollout 不只是“模型生成一段回答”，而是“运行一段多轮轨迹”。官方文档把它拆成三层：

- `AgentLoop`：客户端，负责 agent 逻辑；[`Agentic RL Training`](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)
- `AsyncLLMServerManager`：推理网关，提供 generate 接口；[`Agentic RL Training`](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)
- `AsyncServer`：真正连接推理引擎的服务端实例。[`Agentic RL Training`](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)

这就是从“最小 trainer”走向真实系统的第一大分叉：一旦工具调用会阻塞，单个同步 rollout 就会让 GPU 大量空转，所以必须改成 server-based asynchronous rollout。[`Agentic RL Training`](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)

### 4. reward

在单轮 RLHF 里，reward 往往是“给完整回答打分”；但在多轮 agentic RL 里，reward 可能来自最终答案、interaction turn、tool call shaping，甚至额外 trace 字段。`AgentLoopOutput` 除了 `prompt_ids`、`response_ids`、`response_mask`，还可以带 `reward_score`、`num_turns`、`metrics` 与 `extra_fields`。[`agent_loop.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/agent_loop.py)

这意味着 reward 不再只是 trainer 末端的一个函数，而是贯穿 rollout 设计的结果。

### 5. worker/server

官方 `Agent Loop` 文档对 rollout phase 的描述非常重要：

1. `PPOTrainer` 采样 batch；
2. `AgentLoopManager` 唤醒 async LLM servers，同步权重；
3. 它把 batch 切块发给 `AgentLoopWorker`；
4. 每个 worker 对每个 prompt 启动一个 `AgentLoopBase.run` coroutine；
5. 需要模型生成时再通过 `AsyncLLMServerManager.generate` 跟 server 通信；
6. rollout 完成后 gather 结果，再让 server 进入 sleep/offload。[`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

从“最小 trainer”视角看，这里真正新增的不是算法，而是**把 rollout 从函数调用变成了并发系统**。

下面这个映射表，是我觉得最适合初读 `verl` 的读法：

| 最小 trainer 里的问题 | 在 `verl` 里对应什么 |
| --- | --- |
| 从哪里拿 prompt | dataset / `raw_chat` / `agent_name` |
| 谁来组织 rollout | `PPOTrainer` + `AgentLoopManager` |
| 谁真正生成 token | `AsyncServer` 背后的 vLLM / SGLang |
| 谁维护多轮轨迹 | `AgentLoopBase` / `ToolAgentLoop` |
| 谁知道哪些 token 可训练 | `response_mask` |
| 谁执行工具与环境交互 | tool / interaction config + `ToolAgentLoop` |
| 奖励最终喂回哪里 | trainer update path / custom reward function |

## `verl` 如何做 `tool calling` 优化：async rollout、server/client split、sticky session、token-based generate、`response_mask`、`ToolParser`、tool 与 interaction 分层、trace/debug

这一节是全文核心。因为 `verl` 真正跟一般“支持 tool calling 的框架”拉开差距的地方，不在于能不能调工具，而在于它**如何为了 RL 训练去处理工具调用**。

### 1. async rollout 与 server/client split

官方 `Agentic RL Training` 文档直接给出动机：agent 需要等待工具返回时，如果 rollout 仍然是同步执行，GPU 会空转，因此需要 asyncio-based co-routing 和 server-based asynchronous rollout。[`Agentic RL Training`](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)

同时，文档明确把 agent 和 inference engine 架构分离，目标包括：

- 用多 GPU 之间的负载均衡缓解长尾请求；
- 避免 tracing 等 agent 特性污染推理引擎本身。[`Agentic RL Training`](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)

这一步非常关键：`verl` 不是在一个高层对话循环里“顺手”加了 tool call，而是先把 rollout 系统改写成异步客户端/服务端结构，然后再把工具调用塞进去。

### 2. sticky session 与 load balancing

`Agent Loop` 文档说得很明确：`AsyncLLMServerManager` 提供两件事——**load balance** 和 **sticky session**。第一次 turn 选择负载最轻的 server；后续 turn 保持同一个 `request_id` 绑定到同一 server。[`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

代码里这个逻辑由 `GlobalRequestLoadBalancer` 实现：它维护 `request_id -> server_id` 的缓存，以及每个 server 的 in-flight request 计数。[`agent_loop.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/agent_loop.py)

这样做的直接收益，是多轮会话可以更稳定地复用同一 server 上的前缀缓存，同时避免所有请求都堆到单个实例上。

### 3. token-based generate，而不是只靠 chat completion

这是 `verl` 最值得注意的设计点之一。

官方文档指出，很多 agent 框架习惯直接调用 OpenAI chat completion API，并把 history 当消息列表维护；但在 RL 训练里，如果最终消息重新 apply chat template 后得到的 token，跟逐轮真实生成出来的 token 不一致，就会让 trajectory 偏离模型分布。官方甚至明确写到：这种不一致会导致 PPO 训练不收敛。[`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html), [`Agentic RL Training`](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)

所以 `AsyncServer` 除了 chat completion，还提供 token-in token-out 的 `generate(prompt_ids, sampling_params, request_id)` 接口，并把它作为训练时的核心接口。[`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

**分析：** 这也是为什么我会说 `verl` 是“从 trainer 侧长出来的 tool calling”。它优先保护的不是消息层 API 美观，而是策略优化所依赖的 token 真实边界。

### 4. `response_mask`

官方文档把 `AgentLoopOutput.response_mask` 定义为：**LLM 生成 token 记为 1，tool response token 记为 0**。[`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

实际代码更进一步：

- 在模型生成后，`ToolAgentLoop` 会把对应位置追加成 `[1] * len(response_ids)`；[`tool_agent_loop.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/tool_agent_loop.py)
- 在工具响应或 interaction/user 响应追加到 prompt 时，则补成 `[0] * len(response_ids)`。[`tool_agent_loop.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/tool_agent_loop.py)

**分析：** 所以从实现效果上看，`response_mask` 不是单纯的“工具 token mask”，而是一个更一般的“哪些 token 来自策略模型、哪些 token 来自环境/工具/交互”的边界标记。对 RL 来说，这个边界比“消息角色”更重要。

### 5. `ToolParser`

`ToolParser` 的角色，是**从模型原始生成 token 里提取 tool calls，而不是从事后整理过的消息文本里逆推**。官方代码里它是一个抽象基类，当前至少注册了 `hermes`、`gpt-oss`、`qwen3_coder` 三种 parser。[`tool_parser.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/tool_parser.py)

这背后的问题，官方文档也专门解释了：tool parser 在抽取 `<tool_call>...</tool_call>` 或其他 function-call 结构时，可能会修改消息内容；一旦你把“抽取后的消息”重新 encode，得到的 token 就不一定等于模型原始生成 token。因此训练必须以原始 token 为准。[`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

这也是为什么 `ToolParser` 被放在 rollout 内核附近，而不是一个无关紧要的后处理组件。

### 6. tool 与 interaction 分层

`ToolAgentLoop` 的代码说明了一件很重要的事：`verl` 没把“工具调用”和“环境交互”混成一类东西。

- tool 来自 `tool_config_path`，通过 `initialize_tools_from_config` 初始化；[`tool_agent_loop.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/tool_agent_loop.py)
- interaction 来自 `interaction_config_path`，通过 `initialize_interactions_from_config` 初始化；[`tool_agent_loop.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/tool_agent_loop.py)
- 状态机也分成 `PROCESSING_TOOLS` 与 `INTERACTING` 两条路径。[`tool_agent_loop.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/tool_agent_loop.py)

这意味着在 `verl` 里，“给模型一个函数调用接口”和“让模型跟环境继续多轮互动”是两个正交扩展点。

### 7. trace/debug

多轮 agentic RL 很容易出现“训练在跑，但我不知道轨迹里究竟发生了什么”的问题。官方为此提供了 `rollout trace`，支持至少 `mlflow` 和 `weave` 两类后端，并暴露了：

- `actor_rollout_ref.rollout.trace.backend`
- `actor_rollout_ref.rollout.trace.token2text`
- `actor_rollout_ref.rollout.trace.max_samples_per_step_per_worker`

等配置项。[`Trace Function Usage Instructions`](https://verl.readthedocs.io/en/latest/advance/rollout_trace.html)

文档还说明，trace 的目标是记录多轮对话、工具调用、函数输入输出及时间戳，帮助你理解整条 trajectory 如何形成最终结果。[`Trace Function Usage Instructions`](https://verl.readthedocs.io/en/latest/advance/rollout_trace.html)

**分析：** 在单轮 RLHF 里，日志往往够用；但到了 tool use / multi-turn 阶段，没有 trace 基本就等于盲训。`verl` 把 trace 做成训练系统的一部分，而不是外挂 debug 工具，这点非常实用。

## `Agent Loop` 机制：`AgentLoopBase.run`、`AgentLoopOutput`、multi-turn coroutine、当前 non-goals

官方 `Agent Loop` 文档开头就把 design goal 和 non-goal 写得很坦白。

design goal 有三条：

- 可插拔的用户自定义 agent loop；
- 标准化的 request generate API；
- 多 inference servers 的请求级负载均衡。[`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

而 non-goal 只有一句：**它不规定 tool 应该如何定义、如何调用。**[`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

### 1. `AgentLoopBase.run`

`AgentLoopBase` 是抽象基类，`run(self, sampling_params, **kwargs) -> AgentLoopOutput` 是用户真正需要实现的核心接口。文档明确说，`run` 拿到 prompt messages 与 dataset fields 之后，可以自由去做：

- LLM generate
- tool call
- environment interaction
- reflection

等等。[`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

这基本就是 `verl` 对 agent loop 的哲学：**给你统一训练边界，但不替你规定智能体行为。**

### 2. `AgentLoopOutput`

在文档版定义里，`AgentLoopOutput` 最关键的三个字段是：

- `prompt_ids`
- `response_ids`
- `response_mask`

并说明当前一次只输出一个 trajectory，多 trajectory output 仍在讨论中。[`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

实际代码里的 `AgentLoopOutput` 还包含 `response_logprobs`、`routed_experts`、`multi_modal_data`、`reward_score`、`num_turns`、`metrics` 和 `extra_fields`。[`agent_loop.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/agent_loop.py)

也就是说，文档定义的是最小接口，代码实现则已经朝更完整的训练载体扩展。

### 3. multi-turn coroutine

官方文档对并发模型也给了清楚描述：`AgentLoopWorker` 会并发调度多个 coroutine；如果 worker 数量等于 batch size，那么每个 worker 基本可以只负责一个 prompt。[`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

而在 `ToolAgentLoop` 代码里，多轮过程本身又是个状态机：

`PENDING -> GENERATING -> PROCESSING_TOOLS / INTERACTING -> TERMINATED`

其中工具执行会用 `asyncio.gather` 并发跑多次 tool call。[`tool_agent_loop.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/tool_agent_loop.py)

这就是 `Agent Loop` 真正的工程意义：它不是单纯“支持多轮”，而是把多轮 rollout 明确定义成一个 coroutine-driven 的可训练系统。

### 4. 当前 non-goals

官方文档明说 non-goal 是“如何定义和调用工具”。[`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

我会把这句话进一步读成两层意思：

- `verl` 不打算跟 LangGraph/CrewAI 一类框架在“工作流编排语义”上正面竞争；
- 它更关心的是，只要你已经定义好了 loop，系统如何把这条 loop 稳定地变成 RL 可训练 trajectory。

**分析：** 这也是为什么 `Agent Loop` 看起来像 agent abstraction，但骨子里仍然是 rollout abstraction。

## `ReTool` 案例：cold-start SFT、sandbox/code execution loop、reward shaping、`run_qwen2-32b_dapo.sh` 如何把 recipe 接进 `verl`

如果说前几节还是“系统设计”，那 `ReTool` 就是这套系统如何落成一条真实 recipe 的最好样本。

### 1. 论文要解决什么问题

`ReTool` 论文的问题设定很清楚：纯文本 RL reasoning model 在需要结构化求解、几何推理、复杂方程求解时，会遇到工具使用瓶颈，因此要让模型学会在长链路推理中**动态穿插实时代码执行**，并通过 outcome feedback 学会“何时调用工具、如何调用工具”。[`ReTool` 论文](https://arxiv.org/abs/2504.11536)

论文摘要还给出了标志性结果：32B 模型在 AIME 上用 400 个训练 step 达到 67% accuracy，而文本 RL baseline 需要 1080 step 只有 40%。[`ReTool` 论文](https://arxiv.org/abs/2504.11536)

### 2. cold-start SFT

官方 `verl-recipe/retool` README 把工作流分成两个阶段：

1. Cold Start + SFT
2. Dynamic Interaction + RL optimization

其中第一阶段会构造“带代码增强推理轨迹”的高质量数据，再通过 SFT 让模型掌握基本的 tool call 和执行结果分析能力。[`verl-recipe/retool`](https://github.com/verl-project/verl-recipe/tree/main/retool)

README 对应的命令也很直接：

```bash
python3 recipe/retool/retool_sft_preprocess.py
bash recipe/retool/run_qwen2-32b_sft.sh
```

所以 `ReTool` 不是“直接拿 base model 做 RL”，而是先让模型学会最基础的工具语法和执行反馈阅读。

### 3. sandbox / code execution loop

README 对 RL 阶段的描述，是一个典型的“think-execute-feedback”闭环：

- 模型在推理中动态插入代码块；
- 检测到代码终止标记时，把代码异步送进 sandbox；
- sandbox 返回执行结果或错误；
- 结果再反馈回模型，指导下一轮推理。[`verl-recipe/retool`](https://github.com/verl-project/verl-recipe/tree/main/retool)

`retool.py` 里的 `CustomSandboxFusionTool` 把这件事写得更具体：

- 它继承 `SandboxFusionTool`；[`retool.py`](https://github.com/verl-project/verl-recipe/blob/main/retool/retool.py)
- 会用正则提取 ```python ... ``` 代码块；[`retool.py`](https://github.com/verl-project/verl-recipe/blob/main/retool/retool.py)
- 如果最后一行没有 `print`，会自动补一个 `print(...)`，避免脚本算完但不显式输出；[`retool.py`](https://github.com/verl-project/verl-recipe/blob/main/retool/retool.py)
- 最终通过远程 execution pool 执行代码，并把结果作为工具响应返回。[`retool.py`](https://github.com/verl-project/verl-recipe/blob/main/retool/retool.py)

这正好对应前面说的：`verl` 的工具层不是为了给 demo 增添功能，而是为了把外部执行结果稳定接回 trajectory。

### 4. reward shaping

`retool.py` 里的 `compute_score` 先调用 `math_dapo.compute_score(..., strict_box_verify=True)` 对最终答案打分；如果结果小于 0，则按 `num_turns` 给一个额外的 tool call reward，并把分数截到不高于 `-0.6` 的区间内。[`retool.py`](https://github.com/verl-project/verl-recipe/blob/main/retool/retool.py)

这段逻辑很值得注意，因为它说明 `ReTool` 并没有把“调用工具”当成独立监督标签，而是把它写进 reward shaping：

- 最终还是看答案对不对；
- 但当结果不好时，会鼓励模型通过更多有效 turn 去探索工具使用。

**分析：** 这类 shaping 非常符合 `verl` 的系统取向：工具不是目的，能提升 outcome 的策略才是目的。

### 5. `run_qwen2-32b_dapo.sh` 如何接进 `verl`

`run_qwen2-32b_dapo.sh` 把 recipe 接到 `verl` 主训练链路的方式非常典型：

- `data.custom_cls.path=recipe/retool/retool.py` + `data.custom_cls.name=CustomRLHFDataset`：接入自定义数据读取；[`run_qwen2-32b_dapo.sh`](https://github.com/verl-project/verl-recipe/blob/main/retool/run_qwen2-32b_dapo.sh)
- `custom_reward_function.path=recipe/retool/retool.py` + `custom_reward_function.name=compute_score`：接入自定义 reward；[`run_qwen2-32b_dapo.sh`](https://github.com/verl-project/verl-recipe/blob/main/retool/run_qwen2-32b_dapo.sh)
- `actor_rollout_ref.rollout.name=vllm` + `actor_rollout_ref.rollout.mode=async`：使用异步 rollout；[`run_qwen2-32b_dapo.sh`](https://github.com/verl-project/verl-recipe/blob/main/retool/run_qwen2-32b_dapo.sh)
- `actor_rollout_ref.rollout.multi_turn.enable=True`：打开多轮；[`run_qwen2-32b_dapo.sh`](https://github.com/verl-project/verl-recipe/blob/main/retool/run_qwen2-32b_dapo.sh)
- `actor_rollout_ref.rollout.multi_turn.tool_config_path=...`：注入 sandbox tool 配置；[`run_qwen2-32b_dapo.sh`](https://github.com/verl-project/verl-recipe/blob/main/retool/run_qwen2-32b_dapo.sh)
- `actor_rollout_ref.rollout.multi_turn.format=hermes`：指定 tool parser 格式；[`run_qwen2-32b_dapo.sh`](https://github.com/verl-project/verl-recipe/blob/main/retool/run_qwen2-32b_dapo.sh)

换句话说，`ReTool` 并没有另造一套训练框架；它做的是把论文所需的 dataset、reward、tool、rollout 选择，全部作为 recipe 接回 `verl` 的标准入口。

## 官方 recipe 地图与阅读路径：`ppo/grpo/dapo`、`sglang_multiturn`、`sandbox_fusion`、`retool`

如果你想系统读 `verl`，我不建议一上来就啃 `ReTool`。更好的路线是从“单轮 trainer”一路读到“多轮工具轨迹”。

### 第一层：先把单轮训练骨架读顺

- `examples/ppo_trainer`
- `examples/grpo_trainer`

这两块更接近“最小 trainer”的世界：先理解 dataset、reward function、rollout backend、policy update 的基本接线方式，再读多轮。[`verl` README](https://github.com/verl-project/verl)

### 第二层：再读 agentic RL 的最小官方入口

- [`Agentic RL Training`](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)
- [`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)
- [`Multi-turn Rollout Support`](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
- `examples/sglang_multiturn`

这一步会把“为什么需要 async rollout”“为什么要 token-based generate”“tool config 和 interaction config 怎么接”这些概念都串起来。

### 第三层：看 sandbox 与外部执行环境怎么进入主链路

- [`Sandbox Fusion Example`](https://verl.readthedocs.io/en/latest/examples/sandbox_fusion_example.html)
- `examples/sglang_multiturn` 下的 search / sandbox 相关配置

这一层的重点是明白：`verl` 的 tool use 不等于本地函数调用，它从设计上就允许把外部执行环境当成 rollout 的一部分。

### 第四层：最后再看 recipe

- `verl-recipe/dapo`
- `verl-recipe/retool`

这里你会看到真正可复现的研究 recipe 是怎样把前面所有模块组合起来的。需要注意的是，官方 README 已说明 `recipe` 目录在 2026 年 1 月迁移到了独立的 [`verl-recipe`](https://github.com/verl-project/verl-recipe) 仓库。[`verl` README](https://github.com/verl-project/verl)

**分析：** 这条阅读路径背后的原则很简单：先把“trainer 为什么要这样拆”搞明白，再读“agent 为什么能这样接”，最后才读“论文 recipe 如何落地”。反过来读，往往会只看到很多 YAML 参数，而看不到系统主线。

## 轻量背景对照：把 `verl` 放进 2025 年 tool-use / agentic RL 背景里，但不扩成泛综述

这里只做一个非常轻量的定位。

从官方资料看，`verl` 在 2025 年有一个非常清晰的扩展轨迹：

- 2025 年 3 月，README 已把 DAPO 作为重要里程碑列出来，并强调其训练完全由 `verl` 驱动；[`verl` README](https://github.com/verl-project/verl)
- 2025 年 6 月到 7 月，官方文档陆续补上 `Multi-turn Rollout Support`、`Agentic RL Training`、`Agent Loop`、`rollout trace` 等文档；[`Multi-turn Rollout Support`](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html), [`Agentic RL Training`](https://verl.readthedocs.io/en/latest/start/agentic_rl.html), [`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)
- 2025 年 7 月，README 又专门宣布 `ReTool` recipe fully open sourced。[`verl` README](https://github.com/verl-project/verl)

这条时间线说明的不是“`verl` 突然开始做 agent”，而是它从一开始就站在 **post-training / RL runtime** 这一层；当 2025 年 tool-use、multi-turn reasoning、agentic RL 变成真实训练需求时，它只是顺着原有抽象继续往外推。

`ReTool` 论文则代表了同一时期更明确的一种研究口径：模型不只是要会 reasoning，还要会**战略性地使用工具**，也就是 outcome-driven 地决定何时执行代码、何时继续文本推理。[`ReTool` 论文](https://arxiv.org/abs/2504.11536)

**分析：** 所以把 `verl` 放进 2025 年背景里，我更愿意把它看成“agentic RL 的训练底座”，而不是“又一个 agent 开发框架”。它最强的地方，是把 tool use 还原成 rollout、mask、reward、server placement 和 trace 这些训练系统问题。

## 参考文献与官方入口

### 官方入口

1. [`verl` GitHub 仓库](https://github.com/verl-project/verl)
2. [`verl` 官方文档首页](https://verl.readthedocs.io/en/latest/)
3. [`Agentic RL Training`](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)
4. [`Agent Loop`](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)
5. [`Multi-turn Rollout Support`](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
6. [`Trace Function Usage Instructions`](https://verl.readthedocs.io/en/latest/advance/rollout_trace.html)
7. [`Sandbox Fusion Example`](https://verl.readthedocs.io/en/latest/examples/sandbox_fusion_example.html)
8. [`verl-recipe` 仓库](https://github.com/verl-project/verl-recipe)
9. [`verl-recipe/retool`](https://github.com/verl-project/verl-recipe/tree/main/retool)

### 代码入口

1. [`agent_loop.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/agent_loop.py)
2. [`tool_agent_loop.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/tool_agent_loop.py)
3. [`single_turn_agent_loop.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/single_turn_agent_loop.py)
4. [`tool_parser.py`](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/tool_parser.py)
5. [`examples/sglang_multiturn`](https://github.com/verl-project/verl/tree/main/examples/sglang_multiturn)
6. [`run_qwen2-32b_dapo.sh`](https://github.com/verl-project/verl-recipe/blob/main/retool/run_qwen2-32b_dapo.sh)
7. [`retool.py`](https://github.com/verl-project/verl-recipe/blob/main/retool/retool.py)

### 论文

1. [`HybridFlow: A Flexible and Efficient RLHF Framework`](https://arxiv.org/abs/2409.19256v2)
2. [`ReTool: Reinforcement Learning for Strategic Tool Use in LLMs`](https://arxiv.org/abs/2504.11536)

如果后面要继续深入，我建议下一步只做两件事：**一是顺着 `examples/grpo_trainer` 手写一个最小版单轮 trainer；二是再回到 `ToolAgentLoop`，逐行读一次 `response_mask` 是怎么长出来的。** 前者帮你抓主链路，后者帮你抓 `verl` 真正区别于普通 agent framework 的地方。
