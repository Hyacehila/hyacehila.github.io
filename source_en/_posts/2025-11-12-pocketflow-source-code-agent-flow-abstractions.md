---
title: 'PocketFlow Source Walkthrough: Understanding Agent Flow Abstractions in 100 Lines'
title_zh: PocketFlow 源码解读：用百行代码理解 Agent Flow 抽象
date: 2025-11-12 21:49:13 +0800
categories:
- Agent Systems
- Agent Architecture
tags:
- PocketFlow
- Python
author: Hyacehila
mathjax: false
excerpt: Walks through PocketFlow's BaseNode, Node, Flow, Batch, and Async implementations to explain how a minimal LLM agent
  framework organizes workflows with graphs, shared state, and node protocols.
description: Walks through PocketFlow's BaseNode, Node, Flow, Batch, and Async implementations to explain how a minimal LLM
  agent framework organizes workflows with graphs, shared state, and node protocols.
excerpt_zh: 从 PocketFlow 的 BaseNode、Node、Flow、Batch 与 Async 实现出发，梳理一个极简 LLM Agent 框架如何用图、共享存储和节点协议组织工作流。
permalink: /blog/2025/11/12/pocketflow-source-code-agent-flow-abstractions/
lang: en
translation_key: 2025-11-12-pocketflow-source-code-agent-flow-abstractions
translation_status: machine
translation_source_hash: a2dd6b6b4def927ac0d837a88a9d3bbca3bc8ea83ad9c22192fe4b127cba3afd
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>What's PocketFlow?</h2>
<p>The questions in this article can also be addressed<a href="/en/blog/2025/01/15/agent-ai-multimodal-survey/">Multi-modular Agent AI Overview: Model, Learning Mechanisms and Applications</a>、<a href="/en/blog/2026/03/03/cognitive-architecture-to-agent-framework/">From the cognitive structure of an intelligent body to the framework of an intelligent body</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>PocketFlow is one <a href="https://github.com/The-Pocket/PocketFlow/blob/main/pocketflow/__init__.py">100 Line Code</a>The very simple LLM framework.</p>
<ul>
<li>It is completely light, with zero swelling, zero dependence, zero supplier locking.</li>
<li>New abstract models are used, but many of the work streams that have been proposed can be easily achieved.</li>
<li>Better for Argentina Coding. Because it's very simple enough, LLM can easily understand all the documents and core codes.</li>
</ul>
<p>Why would there be PocketFlow? After a year of struggle with a swollen framework, the author decided to remove all unnecessary things, meaning meaningless wrapper. The result is PocketFlow: A very simple LLM framework with a core of 100 lines.</p>
<p>Existing frameworks such as Langchain are helpful when simple demands are consistent with their use assumptions, but too many abstract layers make code difficult to understand and to maintain. These frameworks also raise the issue of dependence, version conflict and changing interfaces for developers.</p>
<p>The judgement behind PocketFlow is that the LLM system can be seen as simple and oriented in essence. By stripping off unnecessary layers, a framework of zero redundancy, zero dependence and zero supplier lock-in can be obtained.</p>
<p>The relevant entrances are available. <a href="https://the-pocket.github.io/PocketFlow/">PocketFlow Document</a>、<a href="https://github.com/The-Pocket/PocketFlow">Community warehouse</a> and <a href="https://github.com/The-Pocket/PocketFlow-Go">Go Language Version</a>。</p>
<h2>Charts, Nodes and Shared Storage</h2>
<p>PocketFlow modeling LLM workflows as maps and sharing storage:</p>
<ul>
<li>Node handles simple LLM tasks.</li>
<li>Flow connects nodes by operation, i.e., by the side with the label.</li>
<li>Store allows nodes in the process to communicate with each other.</li>
</ul>
<p>On the Agent system, Node performs three simple operations:</p>
<ol>
<li>Prep: Retrieval of required content from shared storage.</li>
<li>Exec: Performing professional tasks.</li>
<li>Post: returns the result to the shared storage and determines the next operation.</li>
</ol>
<p>Flow is executed according to conditions, that is, Orch, as stated in the PocketFlow document.</p>
<p>It also supports bulk processing, step execution and parallel processing of nodes and processes:</p>
<ul>
<li>Catch Node/ Flow processing data-intensive tasks.</li>
<li>Async Node/ Flow awaits a walk-in.</li>
<li>Parallel Node/ Flow handles I/O intensive tasks.</li>
</ul>
<p>PocketFlow specifically avoids binding a supplier-specific API, so the original code does not contain unnecessary wrapper. If API wrapper is needed, it can be prepared by itself at any time, or it can be used by any company.</p>
<h2>Document Home and Design Mode</h2>
<p>The following figure shows the core abstraction of PocketFlow.</p>
<p><a href="https://github.com/The-Pocket/.github/raw/main/assets/abstraction.png"><img src="https://github.com/The-Pocket/.github/raw/main/assets/abstraction.png" alt="PocketFlow core abstraction"></a></p>
<p>From here on, it's more direct to achieve common sense of the pactor.</p>
<p><a href="https://github.com/The-Pocket/.github/raw/main/assets/design.png"><img src="https://github.com/The-Pocket/.github/raw/main/assets/design.png" alt="PocketFlow design patterns"></a></p>
<p>PocketFlow does not provide built-in examples of use, but rather examples:</p>
<ul>
<li><a href="https://the-pocket.github.io/PocketFlow/utility_function/llm.html">LLM Wrapper</a></li>
<li><a href="https://the-pocket.github.io/PocketFlow/utility_function/viz.html">Viz and Debug</a></li>
<li><a href="https://the-pocket.github.io/PocketFlow/utility_function/websearch.html">Web Search</a></li>
<li><a href="https://the-pocket.github.io/PocketFlow/utility_function/chunking.html">Chunking</a></li>
<li><a href="https://the-pocket.github.io/PocketFlow/utility_function/embedding.html">Embedding</a></li>
<li><a href="https://the-pocket.github.io/PocketFlow/utility_function/vector.html">Vector Databases</a></li>
<li><a href="https://the-pocket.github.io/PocketFlow/utility_function/text_to_speech.html">Text-to-Speech</a></li>
</ul>
<p>The tools needed to build freely, based on the examples, preserve the bottom-up optimization space and facilitate the construction of various transmission structures.</p>
<h2>Source Structure Overview</h2>
<p>The core code for PocketFlow requires four Python standard library packages.</p>
<pre><code class="language-python">import asyncio, warnings, copy, time
</code></pre>
<p>They are used for the stept, warning information, object copying and time-related functions.</p>
<p>The main abstractions in the source code include:</p>
<ul>
<li><code>BaseNode</code>: all node and flow base protocols.</li>
<li><code>Node</code>: implementable nodes, add retry and fallback.</li>
<li><code>Flow</code>: Process organizer, also special Node.</li>
<li><code>BatchNode</code> / <code>BatchFlow</code>: Batch execution.</li>
<li><code>AsyncNode</code> / <code>AsyncFlow</code>: Step execution.</li>
<li><code>AsyncParallelBatchNode</code> / <code>AsyncParallelBatchFlow</code>: Parallel batching.</li>
</ul>
<h2>BaseNode: Node Agreement and Transfer Relationship</h2>
<p><code>BaseNode</code> is the basis of all Node and Flows. In the abstract structure of PocketFlow, Flow is also a special Node, thus supporting the mix of Node and Flow and the Flow-to-Flow nesting.</p>
<pre><code class="language-python">class BaseNode:
    def __init__(self): self.params,self.successors={},{}
    def set_params(self,params): self.params=params
    def next(self,node,action=&quot;default&quot;):
        if action in self.successors: warnings.warn(f&quot;Overwriting successor for action &#39;{action}&#39;&quot;)
        self.successors[action]=node; return node
    def prep(self,shared): pass
    def exec(self,prep_res): pass
    def post(self,shared,prep_res,exec_res): pass
    def _exec(self,prep_res): return self.exec(prep_res)
    def _run(self,shared): p=self.prep(shared); e=self._exec(p); return self.post(shared,p,e)
    def run(self,shared):
        if self.successors: warnings.warn(&quot;Node won&#39;t run successors. Use Flow.&quot;)
        return self._run(shared)
    def __rshift__(self,other): return self.next(other)
    def __sub__(self,action):
        if isinstance(action,str): return _ConditionalTransition(self,action)
        raise TypeError(&quot;Action must be a string&quot;)
</code></pre>
<p>Construct Functions for All <code>BaseNode</code> And the following inheritance categories provide two basic attributes:<code>params</code> and <code>successors</code>I'm sorry. The former describes node parameters, while the latter describes the node's subsequent relationship.</p>
<pre><code class="language-python">def __init__(self): self.params,self.successors={},{}
</code></pre>
<p>The method for setting the parameters is straightforward.</p>
<pre><code class="language-python">def set_params(self,params): self.params=params
</code></pre>
<p>To construct a link between nodes, you need to define the next node.<code>next</code> Accept action and its proxies, modify <code>successors</code>and return <code>node</code> Self-in order to achieve a chain call. Repeats the settings by giving the waning to avoid unexpectedly overwrite the already completed node transfer.</p>
<pre><code class="language-python">def next(self,node,action=&quot;default&quot;):
        if action in self.successors: warnings.warn(f&quot;Overwriting successor for action &#39;{action}&#39;&quot;)
        self.successors[action]=node; return node
</code></pre>
<p>The logic of the specific nodes is carried out by three placeholder methods. They are the core of the need to rewrite when building a node.</p>
<pre><code class="language-python">def prep(self,shared): pass
def exec(self,prep_res): pass
def post(self,shared,prep_res,exec_res): pass
</code></pre>
<p>Node execution is broken <code>_exec</code>、<code>_run</code> and <code>run</code> Three.<code>_exec</code> Only call. <code>exec</code>I'm sorry. This internal approach is needed to freely re-write the implementation logic in the following subcategory without modifying the developers' own definition. <code>exec</code>。<code>_run</code> Implementation process in-house.<code>run</code> is the open interface to initiate the implementation logic, and also to preserve space for a single node, and to warn when a subsequent node exists.</p>
<pre><code class="language-python">def _exec(self,prep_res): return self.exec(prep_res)
def _run(self,shared): p=self.prep(shared); e=self._exec(p); return self.post(shared,p,e)
def run(self,shared):
    if self.successors: warnings.warn(&quot;Node won&#39;t run successors. Use Flow.&quot;)
</code></pre>
<h2>Syntax: Construct Flow with an operator</h2>
<p>PocketFlow builds Flow with the following syntax sugar.</p>
<pre><code class="language-python">node &gt;&gt; next_node  # 设置一个节点的默认后继节点，即 default action 下的 next node
node - &quot;action&quot; &gt;&gt; next_node  # 设置一个节点在某个 action 下的 next node
</code></pre>
<p>The government is not going to be able to do this.<code>BaseNode</code> It's overloading. <code>__rshift__</code> Operator, that's... <code>&gt;&gt;</code>and <code>__sub__</code> Operator, that's... <code>-</code>I'm sorry. The former is called. <code>next</code> way to set the next node. The latter returns one after checking the validity of the string <code>_ConditionalTransition</code> Internal category, to facilitate next use.</p>
<pre><code class="language-python">def __rshift__(self,other): return self.next(other)
def __sub__(self,action):
    if isinstance(action,str): return _ConditionalTransition(self,action)
    raise TypeError(&quot;Action must be a string&quot;)
</code></pre>
<p>Supporting Internal Classes <code>_ConditionalTransition</code> For temporary reloading <code>__sub__</code> As a result, more complex grammar sugars are achieved.</p>
<pre><code class="language-python">class _ConditionalTransition:
    def __init__(self,src,action): self.src,self.action=src,action
    def __rshift__(self,tgt): return self.src.next(tgt,self.action)
</code></pre>
<h2>Node: retry with fallback</h2>
<p><code>Node</code> It is the basic implementable node. It needs to include automatic retry, avoid. <code>exec</code> ; The requested LLM output is unreliable; it also requires a fallback, and do not cause the entire process to collapse by error.</p>
<p>As the core definition,<code>Node</code> Inherits the parent parameter and adds two new parameters.<code>super().__init__()</code> Executes a parent construction function that ensures reliable initialization, and then loads the external parameters into the class parameters.</p>
<pre><code class="language-python">class Node(BaseNode):
    def __init__(self,max_retries=1,wait=0): super().__init__(); self.max_retries,self.wait=max_retries,wait
</code></pre>
<p>The backup method is used to retreat after an error. Default realization is just a throwout error.</p>
<pre><code class="language-python">def exec_fallback(self,prep_res,exc): raise exc
</code></pre>
<p>As a true enforceable class,<code>Node</code> Rewriting of the implementation process in conjunction with the retry and retreat mechanisms. Here you see the advantages of the logical separation of the execution and function: the developer achieves <code>exec</code>,Framework Rewrite <code>_exec</code>。<code>_exec</code> Automatically retry and record the number of re-tests, and call Fallback after too many errors, not cyclical, and not directly exit the program.</p>
<pre><code class="language-python">    def _exec(self,prep_res):
        for self.cur_retry in range(self.max_retries):
            try: return self.exec(prep_res)
            except Exception as e:
                if self.cur_retry==self.max_retries-1: return self.exec_fallback(prep_res,e)
                if self.wait&gt;0: time.sleep(self.wait)
</code></pre>
<h2>Flow: organiser and parameter dissemination</h2>
<p><code>Flow</code> It's the whole process controller. From the abstract design point of view, Flow is the collection of Node; from the code design point of view, Flow is the entry point for the implementation of a series of nodes. It's inherited from... <code>BaseNode</code>, therefore supports complex nesting. To show the special, Flow has increased. <code>start_node</code> properties, and the function to set this parameter.</p>
<pre><code class="language-python">class Flow(BaseNode):
    def __init__(self,start=None): super().__init__(); self.start_node=start
    def start(self,start): self.start_node=start; return start
</code></pre>
<p>To get Flow running, it needs to know who the next node is. And so... <code>get_next_node</code> It's based on the current node. <code>curr</code> Yes. <code>action</code>Yes. <code>successors</code> ; if a non-terminated node jumps out of the process, give the waning. If no follow-up node is available, this method returns <code>None</code>。</p>
<pre><code class="language-python">    def get_next_node(self,curr,action):
        nxt=curr.successors.get(action or &quot;default&quot;)
        if not nxt and curr.successors: warnings.warn(f&quot;Flow ends: &#39;{action}&#39; not found in {list(curr.successors)}&quot;)
        return nxt
</code></pre>
<p>PocketFlow provides the logical implementation of the whole Flow <code>_orch</code> Alternatives <code>_exec</code>I'm sorry. Although Flow also inherited from <code>BaseNode</code>However, its implementation logic and individual nodes are inconsistent and therefore require separate process organization.</p>
<pre><code class="language-python">    def _orch(self,shared,params=None):
        curr,p,last_action =copy.copy(self.start_node),(params or {**self.params}),None
        while curr: curr.set_params(p); last_action=curr._run(shared); curr=copy.copy(self.get_next_node(curr,last_action))
        return last_action
</code></pre>
<p>This code can be broken down into steps:</p>
<ul>
<li><code>curr, p, last_action = ...</code> Initializes three variables.</li>
<li><code>curr</code> is the current node to be implemented.<code>copy.copy(self.start_node)</code> Creates a copy of the starting node to avoid interference with the state of the node while running the process repeatedly.</li>
<li><code>p</code> is the parameter for the current node. It's coming in. <code>params</code> With Process itself <code>self.params</code> .</li>
<li><code>last_action</code> Records the action returned from the previous node, initially as <code>None</code>。</li>
<li><code>while curr:</code> Means that if there is a next node, it's always circular.</li>
<li><code>curr.set_params(p)</code> Sets the parameters for the current node.</li>
<li><code>last_action = curr._run(shared)</code> Runs the current node and saves the next action that you want to return.</li>
<li><code>curr = copy.copy(self.get_next_node(curr, last_action))</code> Finds the next node according to the current node and returns action and creates a copy.</li>
<li><code>return last_action</code> Returns the results of the last node after the end of the cycle.</li>
</ul>
<p>Collapse the whole Flow and find it modified the logic of the portal. <code>_run</code> Methodology <code>post</code> Methodology to be implemented <code>_exec</code> For implementation read implementation <code>_orch</code>, and set the whole Flow return value to the last node after it is executed.</p>
<pre><code class="language-python">    def _run(self,shared): p=self.prep(shared); o=self._orch(shared); return self.post(shared,p,o)
    def post(self,shared,prep_res,exec_res): return exec_res
</code></pre>
<p><code>params</code> Yes. <code>BaseNode</code> One of the basic attributes. It provides an independent <code>shared</code> Dictionary, a layer of parameters that can be accessed by nodes and solidified during running. If you need to use <code>params</code>, the parameters access is considered when handwritten Node.</p>
<p>Because <code>(params or {**self.params})</code> , for a Flow, the external input parameter has higher priority and will overwrite the Flow internal parameter.<code>Flow</code> Classes are also reserved for rewriting. <code>prep</code> and <code>post</code>, preserve space for subsequent Flow nesting and special needs.</p>
<h2>BatchNode and BatchFlow: BatchFlow</h2>
<p><code>BatchNode</code> Succession <code>Node</code>, for processing large amounts of duplicate data on a case-by-case basis. It naturally got it. <code>Node</code> . Retry with fallback capability. Because of the manual distribution of the area in front of you. <code>exec</code>, logic of implementation of internal nodes <code>_exec</code>Overall logic <code>_run</code> And start the interface. <code>run</code>, lot processing only requires rewrite the logic of the internal nodes <code>_exec</code>。</p>
<pre><code class="language-python">class BatchNode(Node):
    def _exec(self,items): return [super(BatchNode,self)._exec(i) for i in (items or [])]
</code></pre>
<p>This is what we're asking. <code>BatchNode</code> It's manual. <code>prep</code> Step to generate an iterative object without modification of the manual <code>exec</code>I'm sorry. Of concern are:<code>post</code> Step needs to be addressed. <code>BatchNode</code> list of the items. New realization by list-based extrapolation and parent <code>_exec</code> (c) The implementation of bulk processing to address data-intensive tasks.</p>
<p><code>BatchFlow</code> The volume execution structure is allowed to be fully consistent but with different content. Different times <code>params</code>I'm sorry. It can be understood as a cycle: it runs the Flow over each parameter set. All right. <code>shared</code> The changes to the dictionary need to be made in Node, in principle <code>BatchFlow</code> Just a dispatcher.</p>
<p><code>BatchFlow</code> Request rewrite <code>prep</code> Step, and let <code>prep</code> method returns a list of parameters, i.e. a list of dictionaries. Each element is a set of parameters that run the process.<code>BatchFlow</code> Run once for each group of parameters <code>_orch</code>, the running time parameter is the combination of the process's own parameters and this group of specific parameters. Just change. <code>_run</code> The way we do it. <code>BatchFlow</code>。</p>
<pre><code class="language-python">class BatchFlow(Flow):
    def _run(self,shared):
        pr=self.prep(shared) or []
        for bp in pr: self._orch(shared,{**self.params,**bp})
        return self.post(shared,pr,None)
</code></pre>
<p>One. <code>BatchFlow</code> Or you can embed it in another. <code>BatchFlow</code> Medium. Because <code>BatchFlow</code> special design, which consolidates all the parameters in the BatchFlow layer and then passes them to the innermost node. When actually implemented, the first external parameter will be used to run through all the parameters of the inner layer and then the most basic Flow-by-case.<code>BatchFlow</code> You can embed a single node internally or a multiple node-based Flow.</p>
<p>Use <code>BatchNode</code> and <code>BatchFlow</code> The first question is: What parameters does Node have to recycle, not fix Node only to change? <code>shared</code> Data.</p>
<h2>AsyncNode and AsyncFlow: Step execution</h2>
<p>Next, the code goes into the world of the opposite. The core difference is use <code>async</code> / <code>await</code> Keywords.</p>
<ul>
<li><code>async def</code> Defines the co-ordinate function, which is a stept function. It may suspend its execution and give way to control.</li>
<li><code>await</code> Only <code>async def</code> function is used internally, meaning that you wait for the walk-in to be completed. During the waiting period, the procedure may perform other tasks.</li>
</ul>
<p>In the process of the different steps,<code>await</code> The real meaning is to suspend the current mission, hand over CPU control, and let the other missions run first. Notifys that the current task continues down after I/O operations have been completed. This way, the whole program can be avoided by some I.O. Cardon.</p>
<p>In conducting the async programming, the following rules need to be noted:</p>
<ol>
<li>Use in definition <code>async def</code>I'm sorry. Any function or method, if internal use is made <code>await</code>, the definition must be <code>def</code> Add <code>async</code>。</li>
<li>Use when calling <code>await</code>I'm sorry. Call one. <code>async def</code> , the function must be used <code>await</code> Keywords.</li>
<li>Transmissible. If the function <code>A</code> Internal <code>await</code> Another function <code>B</code>, then function <code>A</code> It must be defined as well. <code>async def</code>I'm sorry. This rule will be passed up until the top caller.</li>
</ol>
<p>Rewrite as each method is different when using the step nodes and stepts Flow <code>prep</code>、<code>exec</code>、<code>post</code> , all the positions that I/O waiting should be increased <code>await</code>I'm sorry. Use to create functions that contain aniso nodes and arcae <code>async</code> Keywords, and yes. <code>flow.run_async</code> Use <code>await</code>I'm sorry. If you want to start an aniso-function from the sync function, you need to use:</p>
<pre><code class="language-python"># asyncio.run 是连接同步世界和异步世界的桥梁
asyncio.run(main())
</code></pre>
<p><code>AsyncNode</code> Succession from the ordinary nodes and rewrite all the methods associated with the walk.</p>
<pre><code class="language-python">class AsyncNode(Node):
    async def prep_async(self,shared): pass
    async def exec_async(self,prep_res): pass
    async def exec_fallback_async(self,prep_res,exc): raise exc
    async def post_async(self,shared,prep_res,exec_res): pass
</code></pre>
<p>These methods are reserved for rewrite business logic. The method names were modified accordingly to avoid confusion with the synchronized version. When rewriting these methods, care needs to be taken to use them while waiting for a mission <code>await</code>Jean. <code>AsyncNode</code> Reads data more efficiently, calls LLM, waits for user feedback or coordinates multiple Agents.</p>
<p>The logic of retrying and fallback is unchanged, but is used extensively because of the contagious nature of the arctic function <code>await</code>。<code>asyncio.sleep</code> It's a cosmobilized hibernation function, which does not block the whole program.</p>
<pre><code class="language-python">    async def _exec(self,prep_res):
        for self.cur_retry in range(self.max_retries):
            try: return await self.exec_async(prep_res)
            except Exception as e:
                if self.cur_retry==self.max_retries-1: return await self.exec_fallback_async(prep_res,e)
                if self.wait&gt;0: await asyncio.sleep(self.wait)
</code></pre>
<p>I'm gonna need to get it. <code>run</code> The stale version. The logic itself is unchanged, but it needs to be introduced because of the insularity of the infection. <code>await</code>, and limit the user 's need to pass <code>run_async</code> Starts the node. If using previous sync method, throw directly <code>RuntimeError</code>。</p>
<pre><code class="language-python">    async def run_async(self,shared):
        if self.successors: warnings.warn(&quot;Node won&#39;t run successors. Use AsyncFlow.&quot;)
        return await self._run_async(shared)
    async def _run_async(self,shared): p=await self.prep_async(shared); e=await self._exec(p); return await self.post_async(shared,p,e)
    def _run(self,shared): raise RuntimeError(&quot;Use run_async.&quot;)
</code></pre>
<p>Run only one <code>AsyncNode</code>, and when no other parallel task is added, the async does not cause the async to be swallowed up, but only the unblocked effect. Run to corresponding <code>AsyncNode</code> , the program does not block the I/O while waiting, thus leaving the CPU for other tasks.</p>
<p><code>AsyncFlow</code> Multiple inheritance, existing <code>Flow</code> And the ability to organize, and the ability to organize, <code>AsyncNode</code> The stept properties.</p>
<pre><code class="language-python">class AsyncFlow(Flow,AsyncNode):
    async def _orch_async(self,shared,params=None):
        curr,p,last_action =copy.copy(self.start_node),(params or {**self.params}),None
        while curr: curr.set_params(p); last_action=await curr._run_async(shared) if isinstance(curr,AsyncNode) else curr._run(shared); curr=copy.copy(self.get_next_node(curr,last_action))
        return last_action
</code></pre>
<p>It's almost identical to the sync version, but it just increased. <code>await</code>and supports the mixing of synchronous nodes and hexeronodes in an all-step Flow.</p>
<ul>
<li><code>isinstance(curr, AsyncNode)</code> Checks whether the current node is an anecdotal.</li>
<li>If it's a stale node, it's a stale node. <code>await curr._run_async(shared)</code>;, if not, call directly <code>curr._run(shared)</code>。</li>
<li>This allows for the use of a hybrid of synchronous and heteronodes in a walk process.</li>
</ul>
<h2>AsyncBatch and AsyncParalBattch: Batch in sequence and in parallel</h2>
<p>The staggered capacity also uses multiple inheritances, combining Node with Flow.</p>
<p><code>AsyncBatchNode</code> The text reads as follows:</p>
<pre><code class="language-python">class AsyncBatchNode(AsyncNode,BatchNode):
    async def _exec(self,items): return [await super(AsyncBatchNode,self)._exec(i) for i in items]
</code></pre>
<p>It's a succession. <code>AsyncNode</code> and <code>BatchNode</code>I'm sorry. It's... <code>_exec</code> Method through List, for each item <code>await</code> Parent <code>_exec</code>, therefore is sequentially executed.</p>
<p><code>AsyncBatchFlow</code> The text reads as follows:</p>
<pre><code class="language-python">class AsyncBatchFlow(AsyncFlow,BatchFlow):
    async def _run_async(self,shared):
        pr=await self.prep_async(shared) or []
        for bp in pr: await self._orch_async(shared,{**self.params,**bp})
        return await self.post_async(shared,pr,None)
</code></pre>
<p>This is a bulk process in a stept version and will also be the case for multiple processes in sequence.</p>
<p><code>AsyncParallelBatchNode</code> Use <code>asyncio.gather</code> Perform parallel batching.</p>
<pre><code class="language-python">class AsyncParallelBatchNode(AsyncNode,BatchNode):
    async def _exec(self,items): return await asyncio.gather(*(super(AsyncParallelBatchNode,self)._exec(i) for i in items))
</code></pre>
<p>The key here is... <code>asyncio.gather(...)</code>I'm sorry. It receives a list of the courses, while initiating them and awaiting completion of all the courses.<code>(... for i in items)</code> Is the generator expression,<code>*</code> It's going to be extended into multiple parameters, which is equivalent to <code>asyncio.gather(coro1, coro2, coro3, ...)</code>。</p>
<p><code>AsyncParallelBatchFlow</code> Several examples of processes are initiated in parallel in the same way.</p>
<pre><code class="language-python">class AsyncParallelBatchFlow(AsyncFlow,BatchFlow):
    async def _run_async(self,shared):
        pr=await self.prep_async(shared) or []
        await asyncio.gather(*(self._orch_async(shared,{**self.params,**bp}) for bp in pr))
        return await self.post_async(shared,pr,None)
</code></pre>
<p>This is the parallel batch process: use <code>asyncio.gather</code> Several examples of processes are initiated simultaneously.</p>
