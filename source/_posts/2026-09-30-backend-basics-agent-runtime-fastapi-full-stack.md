---
title: "后端基础：一个问答 Bot 如何变成互联网上的 API"
title_en: "Backend Basics: How a Question-Answer Bot Becomes an Internet API"
date: 2026-09-30 00:25:00 +0800
categories: ["Programming", "Full Stack Development"]
tags: [Backend, FastAPI, Python, Agent Runtime, HTTP, JavaScript]
author: Hyacehila
excerpt: "从一个把用户问题交给语言模型的最小问答 Bot 出发，对照 curl 和浏览器发出的请求，理解 Python 函数如何通过 HTTP、FastAPI 路由与数据约定，成为客户端可以调用的后端 API。"
excerpt_en: "Starting with a minimal question-answer bot, compare requests from curl and a browser to see how HTTP, FastAPI routes, and data contracts make a Python function available as a backend API."
description: "从一个把用户问题交给语言模型的最小问答 Bot 出发，对照 curl 和浏览器发出的请求，理解 Python 函数如何通过 HTTP、FastAPI 路由与数据约定，成为客户端可以调用的后端 API。"
mathjax: false
hidden: true
permalink: '/blog/2026/09/30/backend-basics-agent-runtime-fastapi-full-stack/'
---

我过去写 Agent 时，通常先关心模型能不能回答、工具能不能调用、任务循环能不能跑通。到了要给别人用的时候，问题突然变了：一个只在我电脑上的 Python 进程里存在的函数，怎样才能让另一台机器上的浏览器调用？为什么在终端里敲一条 `curl` 命令，就能得到那个函数的结果？

在 AI Coding 已经可以很快拼出一个页面和一套接口的今天，我反而更想把这件事想明白。如果只知道让 AI 生成 `@app.post` 和 `fetch`，项目一旦出现任何问题我都不知道该从哪里开始排查。下面我用一个简单得有些无聊的问答 Bot，把这条链路从头走一遍。它没有工具、记忆和规划，只为了给后面的服务化问题找一个最小的业务入口。

## 先有一个只能在 Python 里调用的 Bot

用户给出一个问题，我们把它交给语言模型，再把答案还给用户。下面直接看代码。示例假定运行进程已有 `OPENAI_API_KEY`；`OpenAI()` 从环境中读取密钥，代码只负责创建客户端并调用模型。先看同步版：

```python
from openai import OpenAI


def run_bot(question: str) -> str:
    with OpenAI() as client:
        response = client.responses.create(
            model="gpt-5.6-luna",
            input=question,
        )
    return response.output_text


if __name__ == "__main__":
    print(run_bot("什么是 FastAPI？"))
```

`question` 是输入，`run_bot()` 是我们自己的业务函数，模型服务是它调用的外部 API，返回值是答案。这里固定使用 [`gpt-5.6-luna`](https://developers.openai.com/api/docs/models/gpt-5.6-luna)。`client.responses.create(...)` 发起请求，程序在模型回答之前会停在这一行，拿到响应后再读取 `output_text`；`with` 代码块结束时关闭客户端。

同一件事也可以用异步客户端来写。模型、输入与结果都没有换，只是等待的方式变了：

```python
import asyncio

from openai import AsyncOpenAI


async def run_bot(question: str) -> str:
    async with AsyncOpenAI() as client:
        response = await client.responses.create(
            model="gpt-5.6-luna",
            input=question,
        )
    return response.output_text


if __name__ == "__main__":
    print(asyncio.run(run_bot("什么是 FastAPI？")))
```

这时 `run_bot()` 是异步函数，调用它需要 `await run_bot(question)`；在独立脚本的最外层，我们用 `asyncio.run(...)` 启动它。直接写 `run_bot(question)` 只会得到一个协程对象，不会得到答案。`AsyncOpenAI()` 创建异步客户端，`async with` 在退出时关闭它。我们先记住同步版的 `def` 与异步版的 `async def` 在调用方式上不同。两者怎样影响服务器处理多个请求，后面讲后端并发时再展开。

但此时的函数只能由能运行这段 Python 代码的程序调用。网页中的 JavaScript 无法拿着字符串直接执行服务器内存里的 `run_bot()`。它与 Python 进程不在同一处，甚至不在同一台机器上。我们需要给这个能力约定一个远程入口：客户端发来什么，服务器返回什么。

先从最简单的 API 开始：

```text
POST /chat
请求 JSON：{"question": "什么是 FastAPI？"}
响应 JSON：{"answer": "……模型生成的答案……"}
```

`run_bot(question)` 是 Python 函数接口；`POST /chat` 是网络接口。两个接口处理的是同一个问题，但说的是两种语言。后端要做的第一件事，就是在它们之间建立一条可重复、可检查的转换路径。

## 地址：请求究竟发到哪里

假设 Bot 跑在一台云服务器上，它对外的 IPv4 地址是 `203.0.113.10`，服务监听 `8000` 端口；前端页面则由另一台机器 `198.51.100.20:5173` 提供。于是浏览器或终端要访问的接口地址是：

```text
http://203.0.113.10:8000/chat
```

这里 `203.0.113.10` 负责找到服务器，`8000` 帮服务器找到正在监听的进程，`/chat` 则在 Web 应用里找到我们定义的入口。前端页面在另一台机器上，来体现前后端的分离特性，不使用 Web 前端而是考虑 App 或者服务器也是一样的。

这时再看 URL，可以把它拆成几个有不同归属的部分：

| URL 中的部分 | 在这个例子里 | 谁主要负责 |
| --- | --- | --- |
| 协议 | `http` | 客户端与服务器按 HTTP 交换请求和响应 |
| 主机 | `203.0.113.10` | 网络把请求送到目标机器 |
| 端口 | `8000` | 操作系统把连接交给监听该端口的进程 |
| 路径 | `/chat` | FastAPI 在应用里匹配路由 |

示例用明文 HTTP，是为了让协议结构容易看清。真正面向公众时，通常会让 HTTPS 入口处理 TLS，再把请求交给应用；尤其当前端页面本身通过 HTTPS 打开时，不应让它去调用这个 HTTP 示例地址。

## 给 Bot 一个 FastAPI 入口

现在把同步版的业务函数放进一个完整的 `main.py`。我特意让 `run_bot()` 继续只收一个 Python 字符串：它不需要知道 HTTP 方法、URL 或 JSON 长什么样。

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


app = FastAPI()

# 一个简单的跨域访问规则，授权这些源对相关 API 的请求，是业务无关的浏览器安全配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://198.51.100.20:5173"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


def run_bot(question: str) -> str:
    with OpenAI() as client:
        response = client.responses.create(
            model="gpt-5.6-luna",
            input=question,
        )
    return response.output_text


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    answer = run_bot(request.question)
    return ChatResponse(answer=answer)
```

这里的 `app = FastAPI()` 创建的是应用对象，还没有让机器监听端口。`@app.post("/chat")` 把 **POST 方法 + `/chat` 路径**登记到这个应用；下面的 `chat()` 是接住 HTTP 请求的函数。它读出 `request.question`，交给 `run_bot()`，再把答案装进 `ChatResponse`。我们并没有把 Python 函数直接“暴露到互联网”；我们定义了一个 HTTP 入口，由入口函数来调用它。

`ChatRequest` 与 `ChatResponse` 则是边界上的两份约定。前者告诉 FastAPI：请求体应当是带有 `question` 字段的 JSON 对象；后者描述成功响应的结构。请求缺少 `question` 或格式不合要求时，框架会在业务函数运行之前返回验证错误。`question: str` 由网络传输，本地解析并经过 `pydantic` 校验格式的正确性。

这个 `main.py` 选用同步版，所以路由函数也是普通 `def`。模型调用期间，它要等到远程 API 返回才能继续。FastAPI 会把这种同步路由放到线程池中执行，避免它直接阻塞负责处理异步请求的事件循环。同步 SDK 直接放到 FastAPI 里不算是非常严重的阻塞，只是会较多的侵占线程资源。

异步版的 HTTP 接口也写成完整示例。这里把它视为另一个文件 `main_async.py`，与同步版对照：

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://198.51.100.20:5173"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


async def run_bot(question: str) -> str:
    async with AsyncOpenAI() as client:
        response = await client.responses.create(
            model="gpt-5.6-luna",
            input=question,
        )
    return response.output_text


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    answer = await run_bot(request.question)
    return ChatResponse(answer=answer)
```

对客户端而言，这两个版本仍是同一个 `POST /chat`：URL、请求 JSON 和响应 JSON 都一样。变化发生在服务器等待模型回答时。同步版由线程池承接等待；异步版在 `await` 处让出执行机会，等响应到来再继续。这里用的是真正支持异步调用的 SDK，不能只把同步版的 `def` 改成 `async def`、却继续在里面调用阻塞的同步方法。

至于线程、事件循环和并发请求究竟怎么配合，我们后面会单独讨论。使用 FastAPI 的时候不再需要手动在最外层用 `asyncio.run()` 开启事件循环，它自己就会启动，当然不再异步函数里塞同步阻塞也是此时的基本素养，它依旧会导致卡死，哪怕 FastAPI 支持正常异步和自动线程池也不会拯救这一点。

两个示例都在每次请求中创建并关闭客户端，便于看清调用边界。长期运行的服务还会考虑复用客户端、在应用关闭时释放连接；那是下一步的工程安排。

## 先用 curl：把一个请求摊在桌上

假设服务已经部署并可达，在另一台机器的终端运行：

```bash
curl -i -X POST "http://203.0.113.10:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"什么是 FastAPI？"}'
```

`-X POST` 指定方法，URL 指出目标，`-H` 设置请求头，`-d` 放入请求体，`-i` 让我们同时看到响应头。成功时，响应体的形状大概是：

```json
{"answer":"……模型生成的答案……"}
```

把 `curl` 命令展开，HTTP 请求在概念上接近这样：

```http
POST /chat HTTP/1.1
Host: 203.0.113.10:8000
Content-Type: application/json

{"question":"什么是 FastAPI？"}
```

第一行给出方法和路径；`Host` 表示目标主机与端口；`Content-Type` 说明请求体按 JSON 解释；空行后面才是正文。`curl` 并不知道 `run_bot()`、Pydantic 或语言模型，它只会按我们写的选项构造 HTTP 请求、等待 HTTP 响应。正因为这个边界足够普通，他才足够通用。

前面加了 `-i`，因此终端上看到的不只是答案，而是**状态行、响应头、空行、响应体**连在一起。成功时可以把它理解成下面这个形状；实际答案由模型生成，不会固定：

```http
HTTP/1.1 200 OK
content-type: application/json

{"answer":"……模型生成的答案……"}
```

`200` 表示这次 HTTP 请求成功，`content-type` 告诉客户端正文是 JSON。空行之后的 `{"answer":"..."}` 才是 FastAPI 根据 `ChatResponse` 发回的数据。`curl` 默认只是把收到的正文打印出来；`-i` 让它把响应头也显示出来。如果只想读取答案，可以去掉 `-i`，把纯 JSON 正文交给相关终端命令或脚本解析。

## 再用页面：换了客户端，接口没有换

现在想给 Bot 加一个页面：输入问题，点按钮，显示答案。假设下面的页面由 `http://198.51.100.20:5173` 提供，核心 JavaScript 只有一段 `fetch`：

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>问答 Bot</title>
</head>
<body>
  <textarea id="question" placeholder="问 Bot 一个问题"></textarea>
  <button id="send">发送</button>
  <pre id="answer"></pre>

  <script>
    const questionInput = document.querySelector("#question");
    const answerBox = document.querySelector("#answer");

    document.querySelector("#send").addEventListener("click", async () => {
      answerBox.textContent = "正在等待回答……";

      try {
        const response = await fetch("http://203.0.113.10:8000/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: questionInput.value }),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        answerBox.textContent = data.answer;
      } catch (error) {
        answerBox.textContent = `请求失败：${error.message}`;
      }
    });
  </script>
</body>
</html>
```

`fetch()` 做的事情和 `curl` 很接近：指定 URL、方法、请求头和请求体，再拿到响应。但页面还得处理**人的交互**：从输入框读值、在等待模型时给出反馈、把 `answer` 显示在页面上。`curl` 把响应打印到终端就完成了任务；浏览器的 JavaScript 则要决定怎么把响应变成界面状态。

把三份代码并排看，会发现它们写法不同，但遵守的是同一份接口约定：

| 要对齐的内容 | curl 命令 | 浏览器 JS | 后端代码 |
| --- | --- | --- | --- |
| 找到服务器 | URL 中的 `203.0.113.10:8000` | `fetch()` 的 URL 中同一地址 | Uvicorn 默认监听 `8000` 端口 |
| 找到接口 | `-X POST`，URL 路径为 `/chat` | `method: "POST"`，URL 路径为 `/chat` | `@app.post("/chat")` 登记这组方法与路径 |
| 说明正文格式 | `-H "Content-Type: application/json"` | `headers` 中的 `Content-Type` | FastAPI 按 JSON 读取请求体 |
| 传入问题 | `-d '{"question":"…"}'` | `JSON.stringify({ question: questionInput.value })` | `ChatRequest.question` → `request.question` |
| 调用 Bot | 不涉及 Python 函数名 | 不涉及 Python 函数名 | `chat()` 调用 `run_bot(request.question)` |
| 读出答案 | 输出响应 JSON，再交给 Python 解析 `answer` | `response.json()` 后读取 `data.answer` | `ChatResponse(answer=answer)` 生成响应 |

这里没有“函数名相同就自动联通”的魔法。`203.0.113.10:8000` 把请求送到运行服务的机器和端口；**POST + `/chat`** 让 FastAPI 选中 `chat()`；JSON 里的 **`question`** 与 `ChatRequest` 的字段对上，才成为 `request.question`。然后是后端自己写的普通 Python 调用，把这个字符串交给 Bot。

客户端既不知道 `chat()` 叫什么，也不知道 `run_bot()` 存在。如果前端把字段改成 `message`，请求就过不了数据校验；如果把 `POST` 改成 `GET`，也不会进入这条路由。异步版只是在后端调用 Bot 时多写一个 `await`，这份对外约定不变。

`JSON.stringify({ question: ... })` 把页面里的 JavaScript 对象编码成请求体字符串；后端反向解析并校验它；响应回来后，`response.json()` 再把响应体读成 JavaScript 对象。这与刚才把 `curl` 正文交给 `json.load(sys.stdin)` 做的是同一类解析，只是发生在浏览器里：先用 `response.ok` 判断状态，再读 JSON，最后取 `data.answer`。

这两个方向看起来像在搬运同一个字段，实际每跨过一次网络边界都要重新编码或解码。前端的 `await fetch(...)` 等的是后端 HTTP 响应；后端同步版由线程等待模型，异步版则写成 `await run_bot(...)`。前端与后端都可能出现 `await`，但它们发生在不同的进程和调用链里。

## FastAPI 为什么能调用到那个 Python 函数

看到这里，`curl` 与 `fetch` 已经对上了。还有一段隐藏在 `@app.post` 背后的链路：那条 HTTP 请求怎样走到 `chat()`？

**Uvicorn：让应用真正接到网络请求**

`app = FastAPI()` 只是在 Python 里创建了一个应用对象，它自己不会占用服务器的 `8000` 端口。运行 `uvicorn main:app --host 0.0.0.0 --port 8000`，才会启动负责监听的服务器进程。`main:app` 让 Uvicorn 找到我们创建的应用；客户端连接到服务器的 IP 和端口后，先接到请求的是 Uvicorn。它处理网络连接和 HTTP，把请求的方法、路径、头部与正文交给应用；等应用给出结果，它再把响应送回客户端。所以 Uvicorn 不会根据 `POST /chat` 自己去选择 `chat()`，也不知道 `run_bot()` 该怎样问模型。路由匹配、请求数据校验和调用业务函数，是交给 FastAPI 之后发生的事。

**ASGI：Uvicorn 与 FastAPI 怎样说话**

那么“交给应用”具体是什么意思？Uvicorn 和 FastAPI 之间遵守一套叫 **ASGI** 的 Python 接口约定。它不是浏览器要学习的新协议，也不是另外启动的一项服务：浏览器和 `curl` 发送的仍是 HTTP。ASGI 规定服务器怎样把一次连接的信息和收到的事件交给 Python 应用，应用又怎样把响应交还给服务器。换一个符合这套约定的服务器，FastAPI 应用仍有共同的接入方式；换一个符合约定的应用，Uvicorn 也知道该怎样调用它。

在规范里，这套交接会出现 `scope`、`receive` 和 `send`：可以先把它们分别理解为连接信息、接收请求内容、发送响应内容。我们平时写 `@app.post("/chat")` 时不需要亲自处理这些对象，FastAPI 已经站在 ASGI 的应用这一侧接住它们，再把请求带到路由函数。

我更愿意把它看作一条逐步缩小问题范围的链路：

```text
公网 IP:端口 → Uvicorn 接收 HTTP → 按 ASGI 约定交给应用
             → FastAPI 按 POST /chat 找到 chat()
             → JSON 变成 ChatRequest → run_bot(question)
             → ChatResponse 变成 JSON → HTTP 响应回到客户端
```

到了 FastAPI 这一层，框架才会按方法和路径匹配路由，把 JSON 请求体校验成 `ChatRequest`，调用 `chat()`，随后进入 `run_bot()`。这里不必继续追问 ASGI 消息内部有哪些字段，但要知道问题发生在哪一层：端口连不上，先看地址、监听与防火墙；只有页面失败而 `curl` 成功，先看浏览器控制台与 CORS；收到 422，先对照 JSON 字段；进入 `run_bot()` 后模型报错，再看模型服务与服务器环境变量。这样排查，比盯着 `fetch` 或 `@app.post` 猜原因可靠得多。

反过来，结果也要穿过同样的边界。`run_bot()` 返回 Python 字符串，`chat()` 将它放进 `ChatResponse.answer`，框架按响应模型把它序列化为 JSON，再通过 ASGI 与 Uvicorn 发回客户端。成功响应通常是 `200`；输入不符合模型时，FastAPI 会返回 `422`；业务函数内部的未处理异常则会导致服务器错误。浏览器的 `fetch()` 即使收到 `422` 或 `500`，也通常会得到一个 `Response` 对象，所以页面代码用 `response.ok` 自己判断；`curl -i` 则可以直接看状态行和响应体。接口是否通了，还得区分网络能否到达、HTTP 是否有响应、业务是否真正成功。

## 这篇文章想留下的后端直觉

到这里，我们只是把一个极小的问答 Bot 接到了互联网上。它还没有会话、数据库、可靠重试、限流和部署治理，也还算不上一个完善的 Agent 系统。但这条最短路径已经足够解释后端最基础的设计：内部函数解决业务问题，API 把这份能力约定成外部能够发送和理解的消息，服务器与框架负责把两边接起来。客户端可以换成 `curl`、浏览器或别的程序，后端的 `/chat` 约定仍是同一份。

我觉得在 AI Coding 时代尤其需要这种理解。让工具生成一个接口并不难，难的是知道它究竟注册了什么、监听在哪里、接受什么字段、何时会拒绝请求、又在哪一步调用了模型。看清这些边界之后，我们再去讨论更复杂的 Agent Runtime、异步任务、持久化状态和完整的全栈工程，才能更加清楚的知道问题究竟在哪。

后端框架已经极大的封装了复杂度，在目前的体系下我们的代码更多的只关注业务而不关注各种基础实现，理解这些构建的同时理解一些数据如何流通的逻辑，对于我们理解业务和工程本身也是由一定帮助的。这是理解服务化的第一步。
