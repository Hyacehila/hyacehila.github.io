---
title: 'Why Output Tokens Are More Expensive: From KV Cache to Agent Cost Engineering'
title_zh: 为什么 Output Token 更贵：从 KV Cache 到 Agent 成本工程
date: 2026-04-26 15:00:00 +0800
categories:
- Agent Systems
- Agent Infrastructure
tags:
- Cost Engineering
- Backend Engineering
author: Hyacehila
excerpt: Output tokens are expensive mainly because decoding is serial, KV cache consumes memory, and generation occupies
  scheduler slots. Agent cost optimization requires controlling output budgets.
description: Output tokens are expensive mainly because decoding is serial, KV cache consumes memory, and generation occupies
  scheduler slots. Agent cost optimization requires controlling output budgets.
excerpt_zh: Output token 贵，主要因为 decode 串行、KV Cache 占显存和调度槽位；Agent 成本优化要控制输出预算和稳定前缀。
permalink: /blog/2026/04/26/output-token-pricing-kv-cache-agent-cost/
lang: en
translation_key: 2026-04-26-output-token-pricing-kv-cache-agent-cost
translation_status: machine
translation_source_hash: d7dda4120c7a293501cf4384a740701150ead7567913c73e0d939b34db30093b
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>This post is based on a simple interview:</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/04/25/llm-semantic-routing-compound-ai-systems/">What exactly is the model route solving: from Agent cost, delay to reasoning control?</a>、<a href="/en/blog/2026/02/16/mcp-model-context-protocol/">MCP (Model Context Protocol)</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p><strong>Why is the price of input token in the big model usually much lower than output token? Technically, is this a reasonable price?</strong></p>
<p>This is interesting because it looks simple, but it can continue to follow the reasoning system and application architecture. If a person who is an Agent developer has really handled costs and delays, it cannot simply answer "output is more expensive because it is slower to produce." From GPU resources to the reasoning side of the watching schedule, the problem is not light.</p>
<p>If it's just product-level, input and output seem to be token. One token come in, one token go out, why is the price so low?</p>
<p>But from the reasoning system, they are not the same kind of work.</p>
<p>Here. <a href="https://platform.openai.com/docs/pricing/">OpenAI API Pricing</a> The official page display is an example of the Standard text token that has been used <code>input</code>、<code>cached input</code> and <code>output</code> Dismantling into different price slots.<strong>The same model, input, capched input, unput, was explicitly detached into different resource forms.</strong></p>
<p>This is because the reasoning stage itself is asymmetrical.</p>
<h2>Prefill and Decode</h2>
<p>Large model reasoning can be broken down into at least two stages:<code>Prefill</code> and <code>Decode</code>。</p>
<p><code>Prefill</code> The process is for the user to give the prompt. The model is read in a single input to calculate the hidden status of these input token and generate the KV Cache that will be used for subsequent decode.</p>
<p><code>Decode</code> It's about the model itself being generated token. It can only generate the next token each time, then reconnect this new token back to the context and continue to generate the next one.</p>
<p>The largest difference between the two phases is parallelity.</p>
<p>Prefill phase, where the model can process an entire input in parallel. And the bottom mass of operations is close to the matrix, the GPU's favorite work pattern. Tensor Core of the modern GPU was designed for this high-volume matrix calculation. The model weight is reused in large quantities token as long as the watch and sequence length are sufficient. The weights are moved from the visible to the large array calculations. The utilization of the algorithm can be higher.</p>
<p>Decode is completely different. Self-regression generation determines that the model must be N token to produce N+1 token. Even if you had the strongest GPU, you wouldn't have finished counting the future token in advance. Each token generation is re-engineered, reading weights, reading history KV Cache, writing new KV Cache, and sampling the next token. The calculations themselves are not necessarily enough to feed Tensor Core, and most times GPU is waiting for data to move from HBM to the computing unit.</p>
<p>So one important instinct is:</p>
<pre><code class="language-text">Prefill:  一次读很多 token，尽量把 GPU 算力吃满
Decode:   一次吐一个 token，被自回归顺序和显存访问拖住
</code></pre>
<p>This is the first source of the cost difference between input token and output token. Input token is the main counterpart in the current request prefill, output token. It's also a graphic card, which consumes almost electricity, and Prefil spends much less time than Decode, and the depreciation of the graphic card is a significant part of the cost of language model reasoning.</p>
<h2>KV Cache is both optimized and cost-effective</h2>
<p>If KV Cache is not available, the model recalculates all previous token's comments key/value for each new token.</p>
<p>KV Cache functions to save the key and value that history has generated in each layer of attitudinal. Next time you generate a new token, the model does not recalculate all history token, but only the new token query/key/value, and then the new query looks at the history KV.</p>
<p>This is certainly a great optimisation. Without it, context generation is largely unavailable.</p>
<p>But KV Cache is not free. State-of-the-art costs of using KV Cache to continuously occupy visible, bandwidth and movement resources. It turns a request into a continuously occupied visible object.</p>
<p>Each request that is still being generated requires that its own KV Cache be kept in the visible memory. The longer the output, the longer the KV Cache; the more the request is made, the larger the KV Cache. Decode, at each step, not only reads model weights, but also history KV, and writes back the new token KV.</p>
<p>So KV Cache has two sides:</p>
<table>
<thead>
<tr>
<th>Perspective</th>
<th>The benefits it brings.</th>
<th>The cost of it.</th>
</tr>
</thead>
<tbody><tr>
<td>Calculate</td>
<td>Token</td>
<td>Cannot eliminate self-returning serial generation</td>
</tr>
<tr>
<td>Organisation</td>
<td>Allows context to be used</td>
<td>Every active request takes over the sustained growth.</td>
</tr>
<tr>
<td>Movement control</td>
<td>Support for multi-request continuous generation</td>
<td>Watch size is limited by KV capacity and bandwidth</td>
</tr>
<tr>
<td>Agent</td>
<td>Reuse Stable Prefix</td>
<td>If the context is in disarray, the C.O.C. rate will drop.</td>
</tr>
</tbody></table>
<p><strong>The smallest system object for LLM service is not the request itself, but the KV status that is requested to be carried.</strong></p>
<h2>Why, Decode, watching slows.</h2>
<p>Batching is very understanding in the Prefill phase. Multiple requests come in, put together big bats, GPU do big matrix calculations, swallow up, single token costs drop.</p>
<p>Decode could also catch, but its returns were not as ideal.</p>
<p>There are three levels of reason.</p>
<p>First, the basic unit of Decode is a round token. You can create "next token" in the same project, but each request can only move forward. A request to generate 200 token will experience about 200 rounds of decode. The batting is increasing the number of requests for ingestion, which does not transform the self-return chain of individual requests into a parallel chain.</p>
<p>Second, the bigger the bat, the more KV Cache needs to be read and maintained. Model weight reading can share a share of the proceeds in the bat, but KV Cache basically increases with the requested number and the length of the context. Decode is already easily stuck in visible bandwidth, and more KV visits will continue to squeeze bandwidth.</p>
<p>Third, the length of the request was inconsistent. In real-life services, some requests are soon closed, others are continuing, and others are just ready to join the decode. Modern engines will use their own data or its operation-level production to kick out the completed requests, add new requests and keep GPU flowing as much as possible. This can significantly increase the amount of vomiting, but it solves the control hole, not the physical bottlenecks of decode.</p>
<p>So more precisely:</p>
<p><strong>Continuous watching makes Decode less waste, but not Decode becomes Prefill.</strong></p>
<p>This is why many online dialogue systems are concerned about two indicators:<code>TTFT</code> and <code>tokens/sec</code>。</p>
<p><code>TTFT</code> It is time to first token, mainly affected by queues, prefills and movement movements. This is the time that the user waited for before the first time he saw the model opening.</p>
<p><code>tokens/sec</code> Closer to the continuous generation rate of the decode phase. Users have seen how much token it can throw up every second, mainly influenced by the decode path.</p>
<p>If you're a lot like Agent to plug a bunch of tool files, warehouse summaries and historical messages, TTFT will get worse. If you're still into the mind-scrutinizing and re-suming, the decode costs will continue to expand.</p>
<h2>Pricing and Agent Dev</h2>
<p>Input is not completely free because prefill still counts. However, as long as input can be processed in parallel, its unit cost is more easily absorbed.</p>
<p><code>Cached input</code> cheaper, usually for prefix reuse after the impact of prompt cape. With OpenAI <a href="https://platform.openai.com/docs/guides/prompt-caching">Prompt Caching</a> Document is an example where the cache hit depends on matching conditions such as long prefixes, route and retention time. Official price page. <code>Cached input</code> Separately, it is actually making prefix reuse a user-visible cost signal.</p>
<p>Output is more expensive because decode consumes a more difficult serial generation time and visible bandwidth. Long output is not just a few words, but is continuously working hours in the presence of both the visible and the visible cards. There are often more expensive Outlook prices in the longer context to compensate for the excess KV Cache consumption of the visible.</p>
<p>If you put it in the Agent project, the price is actually a reminder to developers of three things:</p>
<ol>
<li>Cacheable input, try to make a stable prefix.</li>
<li>Don't need the model to say what it says. Don't let it out. Thinking can solve complex problems, but not so many.</li>
<li>Agent cost optimization, not just total token numbers, but also what token is prefill, which token is decode, which token hits the cache.</li>
</ol>
<p>For ordinary chat products, KV Cache is often a matter of reasoning service providers. Developers see only token bills and delays, and it is difficult to organize a cost-optimization scheme for using pricing and Cache, after all, no one can predict what users say and what models will output.</p>
<p>But for Agen developers, KV Cache will in turn influence how you organize the context.</p>
<p>The first four categories of context would be clearer: stabilization prefixes, such as system instructions, tool descriptions and long-term mission rules; semi-stabilization, such as summaries of the same document, the same warehouse or the same session; dynamic content, such as the results of this round of tools and erroneous information; and user-oriented outputs, such as final responses and explanations. The sequence, stability and length of these elements are usually directly controlled by developers, and not necessarily the presence of the service provider ' s bottom KV in the same example.</p>
<p>Different service providers. <code>cached input</code> It could be achieved by prompt caching, prefix caching or other internal mechanisms. The developers usually only go through.<strong>Stable prefix increases the probability of hit.</strong>; only in systems that host or explicitly expose the cache-aware route are direct management of runtime KV Cache more closely.</p>
<h3>Steady prefix. Steady.</h3>
<p>I'm not sure if I'm gonna get a chance to get a better picture of the situation.</p>
<ul>
<li><strong>Preconditions</strong>: Multiple requests share the same or a matchable token prefix.</li>
<li><strong>Typical scene</strong>: Long document, question and answer, multi-cycle conversation history.</li>
<li><strong>Engineering boundary</strong>: The details of the different systems are different, so it should be understood as creating stabilization prefixes as far as possible, and not as any repetition is inevitable.</li>
</ul>
<p>This will directly affect the hint organization of Agent.</p>
<p>Many Agent requests are re-spelled prompt, but the order is random: time stamp is at the top, dynamic track is at the front of system program, tool list is unstable, memoory is different at each insertion. This would undermine the shared prefix. Even if the big paragraphs are the same, if a bit of dynamic is inserted, token prefix is no longer the same. The more stable the content, the more forward, the more dynamic the content, the more backward, and thus the Cache, which is used to optimize costs.</p>
<p>A more rational structure is usually:</p>
<pre><code class="language-text">稳定前缀:
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
</code></pre>
<h3>Do not contaminate the tool output with long prefixes</h3>
<p>Agent can easily plug the tool output directly back into the context, especially in the log, web page, search results, test output, database records.</p>
<p>If these elements enter history without compression, they pose two types of problems.</p>
<p>First, they increase the cost of follow-up. Each round is reprocessed to the longer context, and prices will rise as the context increases.</p>
<p>Second, the context is rapidly decomposed, and a large number of tool outputs are filled with Context, and the model will soon be compressed and re-press, and then the original target will be forgotten by the next round.</p>
<p>So the Age tool layer should be filtered and compressed as much as possible. Do not turn the full tool results over to the model, and let the tool service provider first return to structured summary, key fields, error code, verifiable state. When the original language is really needed, it should be expanded as needed.</p>
<p>A practical principle is:</p>
<p><strong>The return of the tool to the model should be the minimum adequacy required for the next decision-making, rather than a complete reproduction of the outside world.</strong></p>
<p>This is in line with my earlier views on MCP/Harness: the tool interface is not as big as it is good, and the tool output is not as good as it is. They all go into the context budget of the model, and eventually become the cost of prefill, cache and decode.</p>
<h3>Long file questions and answers to make visible</h3>
<p>Long file questions and answers are the most valuable scene of prefix caching.</p>
<p>If the user questions the same paper, the same code repository, the same financial paper, it is not always appropriate to randomly slice, sort and pompt the document. It would be better to have the context of the document as a stable prefix and then to put different issues behind it.</p>
<p>For example:</p>
<pre><code class="language-text">[固定任务说明]
[固定文档内容或固定文档摘要]
[引用/证据规则]
[本轮用户问题]
</code></pre>
<p>This is a large number of prefixes shared between multiple issues, making it easier for the service to re-use the cache.</p>
<p>Of course, that's not to say always stuff the whole long file in. RAG and incremental context loading remain important. The point here is that when you have decided to re-enter a material, it should be organized into a stable prefix that can be used again, rather than re-aligning it.</p>
<h3>Mask tool instead of adding or deleting, keep application-only</h3>
<p>I've been talking about this.&quot;How do you get the contents?&quot;And there is another dimension that will also quietly destroy the CLA:&quot;How does the context change?&quot;I'm sorry. Manus is here. <a href="https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus">Context Engineering for AI Agents</a> It gives two engineering disciplines, pre-empting this stability.</p>
<p>Number one: <strong>Mask, not dynamic add-and-delet tool</strong>I'm sorry. Many Agent prefers to add and reduce the definition of tools to the context in a dynamic way at the current stage, which is a double disaster: the definition of tools is usually placed in a position that is very forward to the context (part of the prefix for stabilization), and the change is to render the KV-Cache behind it large-scale invalid, each round of which is re-prefilled; and the model is confused and even hallucinating if a tool that has been removed is cited in the context. Better to have tools defined.<strong>It's always the same.</strong>..and using a status machine to decode <strong>Mask logits for the wrong tool</strong>— The tool is still in context, just this step.&quot;Unoptional&quot;I'm sorry. Compatibility prefixes for tool names (e.g.) <code>browser_</code>、<code>shell_</code>) can also be bound by group motion space without moving the prefix at all.</p>
<p>Number two:<strong>Context</strong>I'm sorry. Do not turn back to the actions and observations that have already taken place — any rewriting of history would render the prefix after that all ineffective. And that's what's going on.<strong>Sequence to be sure.</strong>: The same content must be sequenced into the same bytes, for example, the key of JSON must be stable, or an invisible field can be reordered quietly to destroy the cache. In other words, prefix stability is not just required.&quot;Same content.&quot;And I'm asking you to...&quot;Meaning same&quot;。</p>
<p>These two and the front.&quot;The stable content is ahead, the dynamic content is back.&quot;It's an extension of the same principle:<strong>The whole context is operated as a stability prefix, which is not re-packaged as a temporary buffer zone.</strong></p>
<h2>Which indicators should be recorded</h2>
<p>If you're really doing the Age Cost Project, recording the total number of tokens is not enough.</p>
<p>At a minimum, these indicators should be removed:</p>
<table>
<thead>
<tr>
<th>Indicators</th>
<th>Annotations</th>
<th>Main counterpart issues</th>
</tr>
</thead>
<tbody><tr>
<td>input tokens</td>
<td>Inputs for current round of physical access to models</td>
<td>Is the context too big?</td>
</tr>
<tr>
<td>cached input tokens</td>
<td>Inputs for Cache Hit</td>
<td>Is the prefix designed well?</td>
</tr>
<tr>
<td>output tokens</td>
<td>Modelled token</td>
<td>Decode, is the cost out of control?</td>
</tr>
<tr>
<td>TTFT</td>
<td>First token delay</td>
<td>- Yes, sir. - Yes, sir.</td>
</tr>
<tr>
<td>decode latency</td>
<td>Time-consuming generation on a continuous basis</td>
<td>Is output too long and the bandwidth tight?</td>
</tr>
<tr>
<td>output/input ratio</td>
<td>Output-Input Ratio</td>
<td>Agent overexplains or emptys</td>
</tr>
<tr>
<td>cache hit rate</td>
<td>Prefix / KV reuse</td>
<td>Whether context structure disrupts cache</td>
</tr>
</tbody></table>
<p>With these indicators, you can judge where the optimization is.</p>
<p>If TTFT is high, caught input is low, the problem may be re-used in context structure and prefix.</p>
<p>If TTFT is okay, but it's always high, the problem is probably output too long or decode too slow.</p>
<p>If input is large but capched input is high, it may not be bad, as long stabilization prefixes may have been reused.</p>
<p>If output/input radio is long-term high, especially in classification, tool route, and schema generation of such intermediate steps, it means that Agent may be doing the least of what is supposed to be done with the most expensive token.</p>
<h2>Answer the first question.</h2>
<p>A short answer:</p>
<p><strong>This pricing is technically reasonable. Input token 's main consumption prefill calculation is easily measured by the bat and matrix; unput token 's lead time, visible bandwidth and sustained KV Cache status of the major consumption decode phase. A reasonable technical explanation is that such pricing reflects the more scarce and less affordable costs of GPU resources.</strong></p>
<p>And then we'll expand on three points.</p>
<p>First, Prefil and Decode are different parallels. Prefill is closer to the big matrix calculation, Decode is restricted by the order of return.</p>
<p>Second, Decode is easier to memoory-born. For each token generated, the model weight and history KV Cache is accessed, and the visible bandwidth becomes the core bottleneck.</p>
<p>Third, KV Cache improved the double counting, but introduced the current cost of the continued occupation of the visible and movement control resources. But watching can lift the stale, but it can't turn the swab generation into a complete parallel. Continuous Batching, trying to squeeze the graphic cards, but with a synergy with KV-Cache, it's gonna be a gradual build-up.</p>
<p>Three points back to the same question: Decode needs to take a card longer, and GPU depreciation is expensive.</p>
<p>Finally, the project understands:</p>
<p><strong>For Agent developers, the idea is to organize context into a cached, stable prefix, to compress the results of dynamic tools and to strictly control the output budget of intermediate steps.</strong></p>
<p>And then you answer not just the model theory, but the connection between the reasoning system and the application architecture.</p>
