---
layout: blog-post
title: "为什么 Output Token 更贵：从 KV Cache 到 Agent 成本工程"
title_en: "Why Output Tokens Are More Expensive: From KV Cache to Agent Cost Engineering"
date: 2026-04-26 15:00:00 +0800
categories: [Agent 基础设施]
tags: [Agents, Backend, Model Mechanics]
author: Hyacehila
excerpt: "Output token 贵，主要因为 decode 串行、KV Cache 占显存和调度槽位；Agent 成本优化要控制输出预算和稳定前缀。"
excerpt_en: "Output tokens are expensive mainly because decoding is serial, KV cache consumes memory, and generation occupies scheduler slots. Agent cost optimization requires controlling output budgets."
featured: false
math: false
---

# 为什么 Output Token 更贵：从 KV Cache 到 Agent 成本工程

本文源自一个简单的面试题：

**为什么大模型 API 里，input token 的价格通常远低于 output token？从技术上看，这个定价合理吗？**

这个问题有意思，是因为它看起来简单，却可以继续追到推理系统和应用架构层面。一个 Agent 开发者如果真的处理过成本和延迟，就不能只回答“output 更贵是因为生成更慢”。从 GPU 资源到推理侧 batching 调度，这个问题并不轻。

如果只从产品层面看，input 和 output 好像都是 token。一个 token 进来，一个 token 出去，为什么价格能差这么多？

但从推理系统看，它们并不是同一种工作。

以 [OpenAI API Pricing](https://platform.openai.com/docs/pricing/) 官方页面的展示为例，Standard 文本 token 已经把 `input`、`cached input` 和 `output` 拆成不同价格档位。**同一个模型的 input、cached input、output 被明确拆成了不同资源形态。**

这是因为推理阶段本身不对称。

## Prefill 和 Decode

大模型推理至少可以粗略拆成两个阶段：`Prefill` 和 `Decode`。

`Prefill` 处理的是用户已经给出的 prompt。模型一次性读入一段输入，把这些 input token 的隐藏状态算出来，并生成后续解码要用的 KV Cache。

`Decode` 处理的是模型自己正在生成的 token。它每次只能生成下一个 token，然后把这个新 token 接回上下文，再继续生成下一个。

这两个阶段最大的差异，是并行度。

Prefill 阶段，模型可以并行处理一整段输入。底层大量操作接近矩阵乘矩阵，也就是 GPU 最喜欢的工作形态。现代 GPU 的 Tensor Core 本来就是为这种高吞吐矩阵计算设计的。只要 batch 和序列长度足够，模型权重被大量 token 复用。权重从显存搬出来以后，可以参与大量矩阵计算。算力利用率可以被打得比较高。

Decode 阶段完全不同。自回归生成决定了模型必须先知道第 N 个 token，才能生成第 N+1 个 token。即使你有再强的 GPU，也不能把未来 token 提前一次性算完。每生成一个 token，都要重新过一遍模型层，读取权重，读取历史 KV Cache，写入新的 KV Cache，再采样下一个 token。计算本身并不一定足以把 Tensor Core 喂饱，很多时候 GPU 不是不会算，而是在等数据从 HBM 搬到计算单元。

所以一个很重要的直觉是：

```text
Prefill:  一次读很多 token，尽量把 GPU 算力吃满
Decode:   一次吐一个 token，被自回归顺序和显存访问拖住
```

这就是 input token 和 output token 成本差异的第一层来源。Input token 在当前请求里主要对应 prefill，output token 主要对应 decode。同样是一张显卡，近似的电量消耗，Prefill 花的时间比 Decode 少得多，而显卡折旧又是语言模型推理成本里很重的一部分。

## KV Cache 既是优化，也带来状态成本

如果没有 KV Cache，模型每生成一个新 token，都要重新计算前面所有 token 的 attention key/value。

KV Cache 的作用，是把历史 token 在每一层 attention 里产生的 key 和 value 保存下来。下次生成新 token 时，模型不用重算所有历史 token，只需要计算新 token 的 query/key/value，然后让新 query 去看历史 KV。

这当然是巨大的优化。没有它，长上下文生成基本不可用。

但 KV Cache 也不是免费的。使用 KV Cache 持续占用显存、带宽和调度资源的状态成本。它把一次请求变成了一个持续占用显存的状态对象。

每个仍在生成的请求，都需要在显存里保留自己的 KV Cache。输出越长，KV Cache 越长；并发请求越多，KV Cache 占用越大。Decode 每走一步，不仅要读模型权重，还要读历史 KV，并把新 token 的 KV 写回去。

所以 KV Cache 同时有两面：

| 视角 | 它带来的收益 | 它带来的成本 |
| --- | --- | --- |
| 计算 | 避免重复计算历史 token | 不能消除自回归串行生成 |
| 显存 | 让长上下文可用 | 每个活跃请求都占用持续增长的状态 |
| 调度 | 支持多请求连续生成 | batch size 会被 KV 容量和带宽限制 |
| Agent | 可复用稳定前缀 | 上下文一旦组织混乱，缓存命中率会下降 |

**LLM serving 的最小系统对象不是请求本身，而是请求携带的 KV 状态。**

## 为什么 Decode 下 batching 会变慢

Batching 在 Prefill 阶段很好理解。多个请求一起进来，拼成大 batch，GPU 做大矩阵计算，吞吐上升，单 token 成本下降。

Decode 阶段也可以 batch，但它的收益没有这么理想。

原因有三层。

第一，Decode 的基本单位是一轮一 token。你可以把多个请求的“下一 token”放在同一个 iteration 里生成，但每个请求仍然只能向前走一步。一个请求生成 200 token，就要经历大约 200 轮 decode。Batching 提高的是多请求吞吐，不会把单个请求的自回归链条变成并行链条。

第二，batch 越大，需要同时读取和维护的 KV Cache 越多。模型权重读取可以在 batch 内共享一部分收益，但 KV Cache 基本跟请求数和上下文长度一起涨。Decode 已经容易被显存带宽卡住，更多 KV 访问会继续挤占带宽。

第三，请求长度不一致。真实服务里，有的请求很快生成结束，有的请求还在继续，有的请求刚完成 prefill 准备加入 decode。现代引擎会用 continuous batching 或 iteration-level scheduling，把完成的请求踢出去，把新请求加进来，让 GPU 尽量不断流。这能显著提升吞吐，但它解决的是调度空洞，不是消除 decode 的物理瓶颈。

所以更准确的说法是：

**Continuous batching 让 Decode 更不浪费，但不能让 Decode 变成 Prefill。**

这也是很多在线对话系统会同时关心两个指标的原因：`TTFT` 和 `tokens/sec`。

`TTFT` 是 time to first token，主要受排队、prefill 和调度影响。用户第一次看到模型开口之前，等待的就是这段时间。

`tokens/sec` 更接近 decode 阶段的持续生成速度。用户已经看到模型开始输出了，但它每秒能吐多少 token，主要受 decode 路径影响。

如果你的 Agent 很喜欢先塞一大堆工具文档、仓库摘要和历史消息，TTFT 会变差。如果你的 Agent 又喜欢输出几千字思考过程和重复总结，decode 成本会继续放大。

## 定价与 Agent Dev

Input 不是完全免费，因为 prefill 仍然要算。但只要输入能够被并行处理，它的单位成本就更容易被摊薄。

`Cached input` 更便宜，通常对应 prompt cache 命中后的前缀复用。以 OpenAI [Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) 文档为例，cache hit 依赖可匹配的长前缀、路由和保留时间等条件。官方价格页把 `Cached input` 单独列出来，实际上是在把前缀复用变成用户可见的成本信号。

Output 更贵，是因为 decode 消耗的是更难摊薄的串行生成时间和显存带宽。长输出不只是多几个字，而是在持续的占用显存和显卡的工时。更长上下文下往往会有更贵的 Output 价格，是为了弥补长上下文常见下带来的过大 KV Cache 对显存的消耗。

如果把它放进 Agent 工程里，价格其实在提醒开发者三件事：

1. 能缓存的输入，尽量做成稳定前缀。
2. 不需要模型说出来的内容，不要让它输出。思考可以解决复杂问题，但复杂问题没那么多。
3. 长链路 Agent 的成本优化，不能只看总 token 数，还要看哪些 token 是 prefill，哪些 token 是 decode，哪些 token 命中了 cache。

对普通聊天产品来说，KV Cache 很多时候是推理服务商内部的事情。开发者只看到 token 账单和延迟，也很难组织一套利用定价和 Cache 的成本优化方案，毕竟没人能够预测用户说什么以及模型会输出什么。

但对 Agent 开发者来说，KV Cache 会反过来影响你应该怎样组织上下文。

先把上下文分成四类会更清楚：稳定前缀，例如系统指令、工具说明和长期任务规则；半稳定状态，例如同一文档、同一仓库或同一会话的摘要；动态内容，例如本轮工具结果和错误信息；面向用户的输出，例如最终回复和解释。开发者通常能直接控制的是这些内容的顺序、稳定性和长度，而不一定能控制服务商底层 KV 是否驻留在同一实例。

不同服务商的 `cached input` 可能由 prompt caching、prefix caching 或其他内部机制实现。开发者通常只能通过**稳定前缀提高命中概率**；只有在自托管或明确暴露 cache-aware routing 的系统里，才更接近直接管理 runtime KV Cache。

### 稳定前缀要稳定

prefix caching 可以拆成三点理解：

- **前提**：多个请求共享相同或可哈希匹配的 token 前缀。
- **典型场景**：长文档反复问答、多轮对话复用历史。
- **工程边界**：不同系统的实现细节不同，所以应把它理解为尽量制造稳定前缀，而不是任何重复内容都必然命中。

这会直接影响 Agent 的提示词组织。

很多 Agent 每轮请求都会重新拼 prompt，但拼接顺序很随意：时间戳放在最前面，动态 trace 放在 system prompt 前面，工具列表顺序不稳定，memory 每次插入位置不同。这样做会破坏共享前缀。哪怕大段内容其实相同，只要前面插入了一点动态内容，token prefix 就不再一致。因此越是稳定的内容越应该靠前，而越是动态的内容越应该靠后，从而复用 Cache 来优化成本。

更合理的结构通常是：

```text
稳定前缀:
  system prompt
  developer policy
  tool schema / tool descriptions
  repo instructions / docs index

相对稳定的会话状态:
  compressed memory
  selected files / selected docs

高度动态内容:
  latest user message
  latest tool result
  transient trace
```

### 工具输出不要污染长前缀

Agent 很容易把工具输出直接塞回上下文，尤其是日志、网页、搜索结果、测试输出、数据库记录。

如果这些内容未经压缩就进入历史，它们会带来两类问题。

第一，它们会增加后续成本。每一轮都要重新处理越来越长的上下文，价格会随着上下文的增长水涨船高。

第二，上下文会快速腐烂，大量的工具输出填满了 Context，模型很快就会被迫压缩，然后重新 prefill，多轮下去就会遗忘最初的目标。

所以 Agent 的工具层应该尽量先做筛选和压缩。不要把完整工具结果都交给模型，让工具服务端先返回结构化摘要、关键字段、错误码、可验证状态。真正需要原文时，再按需展开。

一个实用原则是：

**工具返回给模型的内容，应该是下一步决策所需的最小充分状态，而不是外部世界的完整复制。**

这和我之前讨论 MCP / Harness 时的观点是一致的：工具接口不是越大越好，工具输出也不是越全越好。它们都会进入模型的上下文预算，最终变成 prefill、cache 和 decode 的成本。

### 长文档问答要显式制造 shared prefix

长文档问答是 prefix caching 最容易发挥价值的场景。

如果用户围绕同一篇论文、同一个代码仓库、同一份财报连续提问，不应该每次都把文档随机切块、随机排序、随机塞进 prompt。更好的做法是让文档上下文成为稳定前缀，然后把不同问题放在后面。

比如：

```text
[固定任务说明]
[固定文档内容或固定文档摘要]
[固定引用格式要求]
[本轮用户问题]
```

这样多个问题之间共享大量 prefix，服务端更容易复用缓存。

当然，这不是说永远把整篇长文档塞进去。RAG 与渐进的上下文加载仍然重要。这里的重点是：当你已经决定让一段材料反复进入上下文时，就应该把它组织成可复用的稳定前缀，而不是每轮重排。


## 应该记录哪些指标

如果你真的在做 Agent 成本工程，只记录总 token 数是不够的。

至少应该把这些指标拆开：

| 指标 | 说明 | 主要对应的问题 |
| --- | --- | --- |
| input tokens | 本轮实际进入模型的输入 | 上下文是否过大 |
| cached input tokens | 命中缓存的输入 | 稳定前缀是否设计得好 |
| output tokens | 模型生成的 token | decode 成本是否失控 |
| TTFT | 首 token 延迟 | prefill、排队、调度是否过慢 |
| decode latency | 持续生成耗时 | output 是否过长，带宽是否吃紧 |
| output/input ratio | 输出与输入比例 | Agent 是否过度解释或空转 |
| cache hit rate | prefix / KV 复用情况 | 上下文结构是否破坏缓存 |

有了这些指标，你才能判断优化该落在哪里。

如果 TTFT 高、cached input 低，问题可能在上下文结构和 prefix 复用。

如果 TTFT 还行，但总耗时高，问题可能在 output 太长或 decode 太慢。

如果 input 很大但 cached input 占比高，未必是坏事，因为长稳定前缀可能已经被复用。

如果 output/input ratio 长期很高，尤其发生在分类、工具路由、schema 生成这类中间步骤，就说明 Agent 可能在用最贵的 token 做最不该做的事。

## 回答最开始的问题

一个短答案：

**这个定价从技术上是合理的。Input token 主要消耗 prefill 阶段的并行计算，容易被 batch 和矩阵计算摊薄；output token 主要消耗 decode 阶段的串行生成时间、显存带宽和持续增长的 KV Cache 状态。一种合理的技术解释是：这种定价反映了更稀缺、更难摊薄的 GPU 资源成本。**

然后再展开三点。

第一，Prefill 和 Decode 的并行度不同。Prefill 更接近大矩阵计算，Decode 被自回归顺序限制。

第二，Decode 更容易 memory-bound。每生成一个 token，都要访问模型权重和历史 KV Cache，显存带宽会成为核心瓶颈。

第三，KV Cache 改善了重复计算，却引入了持续占用显存和调度资源的状态成本。batching 能提升吞吐，但无法把串行生成变成完全并行。Continuous Batching 在尽力压榨显卡，但和 KV-Cache 协同作用下，会把显存逐渐塞爆。

三点回到同一个问题：Decode 需要占用显卡更长时间，而 GPU 折旧很贵。

最后是工程理解：

**对 Agent 开发者来说，这个问题的启发是要把上下文组织成可缓存的稳定前缀，把动态工具结果压缩在后面，并严格控制中间步骤的输出预算。**

这时你回答的就不只是模型原理，而是推理系统和应用架构之间的连接。
