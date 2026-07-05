---
title: "PocketFlow 源码解读：用百行代码理解 Agent Flow 抽象"
title_en: "PocketFlow Source Walkthrough: Understanding Agent Flow Abstractions in 100 Lines"
date: 2025-11-12 21:49:13 +0800
categories: ["Agent Systems", "Agent Architecture"]
tags: ["PocketFlow", "Agent Frameworks", "Source Code", "Python"]
author: Hyacehila
excerpt: "从 PocketFlow 的 BaseNode、Node、Flow、Batch 与 Async 实现出发，梳理一个极简 LLM Agent 框架如何用图、共享存储和节点协议组织工作流。"
excerpt_en: "Walks through PocketFlow's BaseNode, Node, Flow, Batch, and Async implementations to explain how a minimal LLM agent framework organizes workflows with graphs, shared state, and node protocols."
mathjax: false
permalink: "/blog/2025/11/12/pocketflow-source-code-agent-flow-abstractions/"
---

## PocketFlow 是什么

PocketFlow 是一个 [100 行代码](https://github.com/The-Pocket/PocketFlow/blob/main/pocketflow/__init__.py)的极简主义 LLM 框架。

* 完全轻量化，零臃肿、零依赖、零供应商锁定。
* 使用新的抽象模式，但可以轻松实现许多已经被提出的工作流。
* 更适合 Agent Coding。由于它足够极简，LLM 可以轻松理解全部文档和核心代码。

为什么会有 PocketFlow？在挣扎于臃肿的框架一年后，作者决定去除所有不必要的东西，也就是各种无意义的 wrapper。结果就是 PocketFlow：一个极简 LLM 框架，核心只有 100 行代码。

现有框架如 LangChain，在简单需求与其使用假设一致时是有帮助的，但过多抽象层会让代码难以理解、难以维护。这些框架也给开发者带来了依赖臃肿、版本冲突和接口不断变化的问题。

PocketFlow 背后的判断是：LLM 系统本质上可以被看成简单的有向图。通过剥离不必要的层级，就能得到一个零冗余、零依赖、零供应商锁定的框架。

相关入口可以查看 [PocketFlow 文档](https://the-pocket.github.io/PocketFlow/)、[社区仓库](https://github.com/The-Pocket/PocketFlow) 和 [Go 语言版本](https://github.com/The-Pocket/PocketFlow-Go)。

## 图、节点与共享存储

PocketFlow 将 LLM 工作流建模为图和共享存储：

* Node 处理简单的 LLM 任务。
* Flow 通过操作，也就是带标签的边，连接节点。
* Store 让流程中的节点能够彼此通信。

在 Agent 系统中，Node 执行三种简单操作：

1. Prep：从共享存储中检索所需内容。
2. Exec：执行专业任务。
3. Post：将结果返回到共享存储中，并确定下一步操作。

Flow 根据条件编排执行，也就是 PocketFlow 文档中说的 Orch。

它还支持节点和流程的批量处理、异步执行和并行处理：

* Batch Node / Flow 处理数据密集型任务。
* Async Node / Flow 等待异步任务。
* Parallel Node / Flow 处理 I/O 密集型任务。

PocketFlow 专门避免捆绑特定供应商的 API，因此原始代码中不包含不必要的 wrapper。如果需要 API wrapper，可以随时自己编写，也可以使用任何公司的 API。

## 文档 Home 与设计模式

下面的图展示了 PocketFlow 的核心抽象。

[![PocketFlow core abstraction](https://github.com/The-Pocket/.github/raw/main/assets/abstraction.png)](https://github.com/The-Pocket/.github/raw/main/assets/abstraction.png)

从这里开始，实现常见 design pattern 会比较直接。

[![PocketFlow design patterns](https://github.com/The-Pocket/.github/raw/main/assets/design.png)](https://github.com/The-Pocket/.github/raw/main/assets/design.png)

PocketFlow 不提供内置 utility function，而是提供示例：

- [LLM Wrapper](https://the-pocket.github.io/PocketFlow/utility_function/llm.html)
- [Viz and Debug](https://the-pocket.github.io/PocketFlow/utility_function/viz.html)
- [Web Search](https://the-pocket.github.io/PocketFlow/utility_function/websearch.html)
- [Chunking](https://the-pocket.github.io/PocketFlow/utility_function/chunking.html)
- [Embedding](https://the-pocket.github.io/PocketFlow/utility_function/embedding.html)
- [Vector Databases](https://the-pocket.github.io/PocketFlow/utility_function/vector.html)
- [Text-to-Speech](https://the-pocket.github.io/PocketFlow/utility_function/text_to_speech.html)

基于示例自由构建需要的工具，可以保留底层优化空间，也便于构建各种传输结构。

## 源码结构总览

PocketFlow 的核心代码需要四个 Python 标准库包。

```python
import asyncio, warnings, copy, time
```

它们分别用于异步、警告信息、对象复制和时间相关函数。

源码中的主要抽象包括：

* `BaseNode`：所有 node 和 flow 的基础协议。
* `Node`：可执行节点，增加重试和 fallback。
* `Flow`：流程编排器，也是特殊的 node。
* `BatchNode` / `BatchFlow`：批量执行。
* `AsyncNode` / `AsyncFlow`：异步执行。
* `AsyncParallelBatchNode` / `AsyncParallelBatchFlow`：并行批处理。

## BaseNode：节点协议与转移关系

`BaseNode` 是所有 Node 和 Flow 的基础。在 PocketFlow 的抽象结构中，Flow 也是一种特殊的 Node，从而支持 Node 与 Flow 混合嵌套，以及 Flow 对 Flow 的嵌套。

```python
class BaseNode:
    def __init__(self): self.params,self.successors={},{}
    def set_params(self,params): self.params=params
    def next(self,node,action="default"):
        if action in self.successors: warnings.warn(f"Overwriting successor for action '{action}'")
        self.successors[action]=node; return node
    def prep(self,shared): pass
    def exec(self,prep_res): pass
    def post(self,shared,prep_res,exec_res): pass
    def _exec(self,prep_res): return self.exec(prep_res)
    def _run(self,shared): p=self.prep(shared); e=self._exec(p); return self.post(shared,p,e)
    def run(self,shared): 
        if self.successors: warnings.warn("Node won't run successors. Use Flow.")  
        return self._run(shared)
    def __rshift__(self,other): return self.next(other)
    def __sub__(self,action):
        if isinstance(action,str): return _ConditionalTransition(self,action)
        raise TypeError("Action must be a string")
```

构造函数为所有 `BaseNode` 以及后面的继承类提供两个基本属性：`params` 和 `successors`。前者描述节点参数，后者描述节点的后继关系。

```python
def __init__(self): self.params,self.successors={},{}
```

设置参数的方法很直接。

```python
def set_params(self,params): self.params=params
```

为了构造节点之间的链接，需要定义下一个节点。`next` 接受动作及其对应节点，修改 `successors`，并返回 `node` 自身，以便实现链式调用。重复设置时会给出 warning，避免意外覆盖已经完成的节点转移。

```python
def next(self,node,action="default"):
        if action in self.successors: warnings.warn(f"Overwriting successor for action '{action}'")
        self.successors[action]=node; return node
```

具体节点逻辑由三个占位方法承载。它们是自行构建 node 时需要重写的核心。

```python
def prep(self,shared): pass
def exec(self,prep_res): pass
def post(self,shared,prep_res,exec_res): pass
```

节点执行被拆成 `_exec`、`_run` 和 `run` 三层。`_exec` 只负责调用 `exec`。之所以需要这个内部方法，是为了在后面的子类中自由重写执行逻辑，而不修改开发者自己定义的 `exec`。`_run` 实现内部执行流程。`run` 是启动执行逻辑的公开接口，也为单节点执行保留空间，并在存在后继节点时给出警告。

```python
def _exec(self,prep_res): return self.exec(prep_res)
def _run(self,shared): p=self.prep(shared); e=self._exec(p); return self.post(shared,p,e)
def run(self,shared): 
    if self.successors: warnings.warn("Node won't run successors. Use Flow.") 
```

## 语法糖：用运算符构造 Flow

PocketFlow 用下面的语法糖构建 Flow。

```python
node >> next_node  # 设置一个节点的默认后继节点，即 default action 下的 next node
node - "action" >> next_node  # 设置一个节点在某个 action 下的 next node
```

为实现这个效果，`BaseNode` 重载了 `__rshift__` 运算符，也就是 `>>`，以及 `__sub__` 运算符，也就是 `-`。前者翻译为调用 `next` 方法来设置 next node。后者在检查字符串合法性后，返回一个 `_ConditionalTransition` 内部类，方便下一步使用。

```python
def __rshift__(self,other): return self.next(other)
def __sub__(self,action):
    if isinstance(action,str): return _ConditionalTransition(self,action)
    raise TypeError("Action must be a string")
```

辅助内部类 `_ConditionalTransition` 用于暂存重载后的 `__sub__` 结果，从而实现更复杂的语法糖。

```python
class _ConditionalTransition:
    def __init__(self,src,action): self.src,self.action=src,action
    def __rshift__(self,tgt): return self.src.next(tgt,self.action)
```

## Node：重试与 fallback

`Node` 是可执行的基本节点。它需要包含自动重试功能，避免 `exec` 中请求的 LLM 函数输出结果不可靠；同时也需要 fallback，不要因为出错直接让整个流程崩溃。

作为最核心的定义部分，`Node` 继承父类参数，并增加两个新参数。`super().__init__()` 执行父类构造函数，保证可靠初始化，然后将外部参数加载到类参数中。

```python
class Node(BaseNode):
    def __init__(self,max_retries=1,wait=0): super().__init__(); self.max_retries,self.wait=max_retries,wait
```

备用方法用于出错后的回退。默认实现只是抛出错误。

```python
def exec_fallback(self,prep_res,exc): raise exc
```

作为真实可执行的类，`Node` 结合重试和回退机制重写执行流程。这里能看到执行逻辑与函数逻辑分离的优势：开发者实现 `exec`，框架重写 `_exec`。`_exec` 实现自动重试并记录重试次数，出错次数过多后调用 fallback，不会死循环，也不会直接退出整个程序。

```python
    def _exec(self,prep_res):
        for self.cur_retry in range(self.max_retries):
            try: return self.exec(prep_res)
            except Exception as e:
                if self.cur_retry==self.max_retries-1: return self.exec_fallback(prep_res,e)
                if self.wait>0: time.sleep(self.wait)
```

## Flow：编排器与参数传播

`Flow` 是整个流程的控制器。从抽象设计角度看，Flow 是 Node 的集合；从代码设计角度看，Flow 是执行一系列节点的入口。它继承自 `BaseNode`，因此支持复杂嵌套。为了展现特殊性，Flow 增加了 `start_node` 属性，以及设置这个参数的函数。

```python
class Flow(BaseNode):
    def __init__(self,start=None): super().__init__(); self.start_node=start
    def start(self,start): self.start_node=start; return start
```

为了让 Flow 正常运行，它需要知道下一个节点是谁。因此 `get_next_node` 会根据当前节点 `curr` 的 `action`，在 `successors` 中寻找下一个节点；如果某个非终止节点跳出流程，则给出 warning。如果没有后继节点，此方法返回 `None`。

```python
    def get_next_node(self,curr,action):
        nxt=curr.successors.get(action or "default")
        if not nxt and curr.successors: warnings.warn(f"Flow ends: '{action}' not found in {list(curr.successors)}")
        return nxt
```

为了编排整个 Flow 的执行逻辑，PocketFlow 提供了 `_orch` 方法来替代 `_exec`。虽然 Flow 也继承自 `BaseNode`，但它的执行逻辑和单个节点不一致，因此需要单独的流程编排方法。

```python
    def _orch(self,shared,params=None):
        curr,p,last_action =copy.copy(self.start_node),(params or {**self.params}),None
        while curr: curr.set_params(p); last_action=curr._run(shared); curr=copy.copy(self.get_next_node(curr,last_action))
        return last_action
```

这段代码可以拆成几步理解：

- `curr, p, last_action = ...` 初始化三个变量。
- `curr` 是当前要执行的节点。`copy.copy(self.start_node)` 创建起始节点副本，避免多次运行流程时节点状态互相干扰。
- `p` 是当前节点的参数。它是传入 `params` 与流程自身 `self.params` 的合并结果。
- `last_action` 记录上一个节点返回的动作，初始为 `None`。
- `while curr:` 表示只要还有下一个节点，就一直循环。
- `curr.set_params(p)` 为当前节点设置参数。
- `last_action = curr._run(shared)` 运行当前节点，并把它返回的下一步动作存起来。
- `curr = copy.copy(self.get_next_node(curr, last_action))` 根据当前节点和返回动作找到下一个节点，并创建副本。
- `return last_action` 在循环结束后返回最后一个节点的执行结果。

整理整个 Flow，会发现它修改了入口执行逻辑的 `_run` 方法和 `post` 方法，将执行 `_exec` 改为执行 `_orch`，并将整个 Flow 的返回设置为最后一个节点执行后的返回值。

```python
    def _run(self,shared): p=self.prep(shared); o=self._orch(shared); return self.post(shared,p,o)
    def post(self,shared,prep_res,exec_res): return exec_res
```

`params` 是 `BaseNode` 的基本属性之一。它提供了一个独立于 `shared` 字典、可以被节点访问、在运行时固化的参数层。如果需要使用 `params`，就要在手写 Node 时考虑参数访问。

由于 `(params or {**self.params})` 的逻辑，对于一个 Flow 来说，外部输入参数拥有更高优先级，会覆盖 Flow 内部参数。`Flow` 类也预留了可重写的 `prep` 和 `post`，为后续 Flow 嵌套和特殊需求保留空间。

## BatchNode 与 BatchFlow：批量执行

`BatchNode` 继承 `Node`，用于逐个处理大量重复数据。它自然获得了 `Node` 中的重试与 fallback 能力。由于前面清晰地区分了人工实现的 `exec`、内部节点执行逻辑 `_exec`、整体逻辑 `_run` 以及启动接口 `run`，批量处理只需要重写内部节点执行逻辑 `_exec`。

```python
class BatchNode(Node):
    def _exec(self,items): return [super(BatchNode,self)._exec(i) for i in (items or [])]
```

这里要求 `BatchNode` 在人工的 `prep` 步中生成一个可迭代对象，并且无需修改人工实现的 `exec`。值得关注的是：`post` 步需要处理 `BatchNode` 的列表返回结构。新实现通过列表推导式和父类 `_exec` 实现批量处理，从而解决数据密集型任务。

`BatchFlow` 则允许批量执行结构完全一致但内容不同的 Flow。每次使用不同的 `params`。可以把它理解成一个循环：它会针对每个参数集重复运行该 Flow。所有对 `shared` 字典的修改都需要在 Node 中实现，原则上 `BatchFlow` 只是一个调度器。

`BatchFlow` 要求重写 `prep` 步，并让 `prep` 方法返回一个参数列表，也就是由字典组成的列表。每个元素都是一组用于运行流程的参数。`BatchFlow` 对每一组参数运行一次 `_orch`，运行时参数是流程自身参数和这组特定参数的合并。只需要修改 `_run` 方法就可以实现 `BatchFlow`。

```python
class BatchFlow(Flow):
    def _run(self,shared):
        pr=self.prep(shared) or []
        for bp in pr: self._orch(shared,{**self.params,**bp})
        return self.post(shared,pr,None)
```

一个 `BatchFlow` 也可以嵌套在另一个 `BatchFlow` 中。由于 `BatchFlow` 的特殊设计，它会将所有 BatchFlow 层中的参数合并后传给最内层节点。实际执行时，会从外部第一个参数开始遍历内层全部参数，然后逐个执行最基础的 Flow。`BatchFlow` 内部可以嵌套单个节点，也可以嵌套由多个节点组成的 Flow。

使用 `BatchNode` 和 `BatchFlow` 的首要问题是：Node 需要以什么样的参数循环往复运行，而不是固定 Node 只变换 `shared` 数据。

## AsyncNode 与 AsyncFlow：异步执行

接下来代码进入异步世界。核心区别是使用 `async` / `await` 关键字。

- `async def` 定义协程函数，也就是异步函数。它可以在执行过程中暂停，让出控制权。
- `await` 只能在 `async def` 函数内部使用，表示等待异步操作完成。在等待期间，程序可以执行其他任务。

在异步编程中，`await` 的真正含义是：暂停当前任务，把 CPU 控制权交出去，让其他任务先跑。等耗时 I/O 操作完成后，再通知当前任务继续向下执行。这样可以避免整个程序因为某个 I/O 卡顿而阻塞。

进行 async 编程时，需要注意下面几条规则：

1. 定义时用 `async def`。任何函数或方法，只要内部使用了 `await`，定义时就必须在 `def` 前加上 `async`。
2. 调用时用 `await`。调用一个 `async def` 定义的函数时，必须使用 `await` 关键字。
3. 传染性。如果函数 `A` 内部 `await` 了另一个函数 `B`，那么函数 `A` 本身也必须被定义为 `async def`。这个规则会一直向上传递，直到顶层调用者。

在使用异步节点和异步 Flow 时，由于各个方法都是异步的，因此重写 `prep`、`exec`、`post` 时，所有涉及 I/O 等待的位置都应该增加 `await`。创建包含异步节点和异步流的函数时，需要使用 `async` 关键字，并对 `flow.run_async` 使用 `await`。如果要从同步函数启动异步函数，则需要使用：

```python
# asyncio.run 是连接同步世界和异步世界的桥梁
asyncio.run(main())
```

`AsyncNode` 从普通节点继承，并重写所有和异步相关的方法。

```python
class AsyncNode(Node):
    async def prep_async(self,shared): pass
    async def exec_async(self,prep_res): pass
    async def exec_fallback_async(self,prep_res,exc): raise exc
    async def post_async(self,shared,prep_res,exec_res): pass
```

这些方法预留给用户重写业务逻辑。方法名称都做了对应修改，以避免和同步版本混淆。在自己重写这些方法时，需要注意在等待任务上使用 `await`，让 `AsyncNode` 以更性能友好的方式读取数据、调用 LLM、等待用户反馈或协调多个 Agent。

重试和 fallback 的逻辑本身不变，只是因为异步函数的传染性而大量使用 `await`。`asyncio.sleep` 是协程休眠函数，不会阻塞整个程序。

```python
    async def _exec(self,prep_res): 
        for self.cur_retry in range(self.max_retries):
            try: return await self.exec_async(prep_res)
            except Exception as e:
                if self.cur_retry==self.max_retries-1: return await self.exec_fallback_async(prep_res,e)
                if self.wait>0: await asyncio.sleep(self.wait)
```

接下来需要提供 `run` 的异步版本。逻辑本身不变，但因为异步传染性，需要引入 `await`，并限制用户必须通过 `run_async` 启动节点。如果使用以前的同步方法，则直接抛出 `RuntimeError`。

```python
    async def run_async(self,shared): 
        if self.successors: warnings.warn("Node won't run successors. Use AsyncFlow.")  
        return await self._run_async(shared)
    async def _run_async(self,shared): p=await self.prep_async(shared); e=await self._exec(p); return await self.post_async(shared,p,e)
    def _run(self,shared): raise RuntimeError("Use run_async.")
```

只运行一个 `AsyncNode`，且没有添加其他并行任务时，不会获得 async 带来的吞吐提升，而只是获得不阻塞的效果。运行到对应 `AsyncNode` 时，程序在等待 I/O 时不会阻塞，从而为其他任务预留 CPU。

`AsyncFlow` 使用多重继承，既有 `Flow` 的编排能力，又有 `AsyncNode` 的异步特性。

```python
class AsyncFlow(Flow,AsyncNode):
    async def _orch_async(self,shared,params=None):
        curr,p,last_action =copy.copy(self.start_node),(params or {**self.params}),None
        while curr: curr.set_params(p); last_action=await curr._run_async(shared) if isinstance(curr,AsyncNode) else curr._run(shared); curr=copy.copy(self.get_next_node(curr,last_action))
        return last_action
```

它和同步版本几乎一致，只是增加了 `await`，并支持在一个异步 Flow 中混合同步节点和异步节点。

- `isinstance(curr, AsyncNode)` 检查当前节点是不是异步节点。
- 如果是异步节点，就用 `await curr._run_async(shared)`；如果不是，就直接调用 `curr._run(shared)`。
- 这使得一个异步流程中可以混合使用同步和异步节点。

## AsyncBatch 与 AsyncParallelBatch：顺序批处理和并行批处理

异步批量能力也采用多重继承，同时结合 Node 与 Flow。

`AsyncBatchNode` 如下：

```python
class AsyncBatchNode(AsyncNode,BatchNode):
    async def _exec(self,items): return [await super(AsyncBatchNode,self)._exec(i) for i in items]
```

它同时继承 `AsyncNode` 和 `BatchNode`。它的 `_exec` 方法遍历列表，对每个项目 `await` 父类 `_exec`，因此是顺序执行。

`AsyncBatchFlow` 如下：

```python
class AsyncBatchFlow(AsyncFlow,BatchFlow):
    async def _run_async(self,shared):
        pr=await self.prep_async(shared) or []
        for bp in pr: await self._orch_async(shared,{**self.params,**bp})
        return await self.post_async(shared,pr,None)
```

这是异步版本的批量流程，也会顺序执行多个流程实例。

`AsyncParallelBatchNode` 则使用 `asyncio.gather` 执行并行批处理。

```python
class AsyncParallelBatchNode(AsyncNode,BatchNode):
    async def _exec(self,items): return await asyncio.gather(*(super(AsyncParallelBatchNode,self)._exec(i) for i in items))
```

这里的关键是 `asyncio.gather(...)`。它接收一个协程列表，同时启动这些协程，并等待所有协程完成。`(... for i in items)` 是生成器表达式，`*` 会将其展开成多个参数，相当于 `asyncio.gather(coro1, coro2, coro3, ...)`。

`AsyncParallelBatchFlow` 也用同样的方式并行启动多个流程实例。

```python
class AsyncParallelBatchFlow(AsyncFlow,BatchFlow):
    async def _run_async(self,shared): 
        pr=await self.prep_async(shared) or []
        await asyncio.gather(*(self._orch_async(shared,{**self.params,**bp}) for bp in pr))
        return await self.post_async(shared,pr,None)
```

这就是并行版本的批量流程：使用 `asyncio.gather` 同时启动多个流程实例。
