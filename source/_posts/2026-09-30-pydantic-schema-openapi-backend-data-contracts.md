---
title: "后端的数据边界：一个问题怎样进入问答 Bot"
title_en: "Backend Data Boundaries: How a Question Reaches a Q&A Bot"
date: 2026-09-30 12:30:00 +0800
categories: ["Programming", "Full Stack Development"]
tags: [Backend, FastAPI, Python, Pydantic, OpenAPI, JSON Schema]
author: Hyacehila
excerpt: "沿着上一篇问答 Bot 的 /chat 接口，看看客户端送来的 JSON 怎样变成可信的 Python 数据，答案又怎样按约定返回；再由此理解 Pydantic、Schema 和 OpenAPI 在后端各自做什么。"
excerpt_en: "Following the Q&A bot's /chat endpoint, see how incoming JSON becomes validated Python data, how the answer takes a defined shape, and where Pydantic, schemas, and OpenAPI fit in."
description: "沿着上一篇问答 Bot 的 /chat 接口，看看客户端送来的 JSON 怎样变成可信的 Python 数据，答案又怎样按约定返回；再由此理解 Pydantic、Schema 和 OpenAPI 在后端各自做什么。"
mathjax: false
hidden: true
permalink: '/blog/2026/09/30/pydantic-schema-openapi-backend-data-contracts/'
---

上一篇里，我们让一个只能在 Python 进程里调用的问答 Bot，有了 `POST /chat` 这个网络入口。客户端发来 `question`，后端把问题交给语言模型，再返回 `answer`。从 `curl` 到浏览器里的 `fetch`，只要按同一套 HTTP 约定发请求，都能走到这个入口。

可是，能收到请求还不等于能放心地使用它。`question` 如果没有传，或者传来的是一个数组，后端怎么办？假如我把模型调用后的整个对象直接返回，客户端又能不能稳定地从里面找到答案？这一篇我想沿着这两个问题，看看后端是怎样给数据划边界的。我们仍然用那个最简单的 Bot：用户问一个问题，模型答一句话。

## JSON 进来时，还不是我们想要的 Python 对象

先不用 Pydantic，直接拿一个 `dict`，代码似乎也能跑：

```python
@app.post("/chat")
def chat(data: dict):
    answer = run_bot(data["question"])
    return {"answer": answer}
```

正常请求当然没问题。但 `data["question"]` 假定这个键一定存在；即使存在，也没有保证值是我们准备交给 `run_bot(question: str)` 的字符串。Python 函数参数后面的 `str` 是类型标注，本身不会在运行时拦住 `run_bot([1, 2, 3])`。更麻烦的是，等到模型 SDK 报错，我们看到的是业务流程中途的一次异常，很难一眼看出真正的问题在客户端输入。除非你在每一个入口处都写满 `try` 去捕获各种异常。

我希望入口能先回答几个明确的问题：这个请求必须有哪些字段？字段是什么类型？字符串能不能是空的？多传了不认识的字段要怎么办？这些规则放在一起，就是这条接口的**输入约定**，也常被叫作 schema 或数据契约，规定了这个接口愿意接受什么样的数据。

## 用 Pydantic 把约定写成代码

FastAPI 通常用 Pydantic 模型描述请求体。把上一段改成下面这样：

```python
from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[str, Field(min_length=1, max_length=4000)]


app = FastAPI()


@app.post("/chat")
def chat(request: ChatRequest):
    answer = run_bot(request.question)
    return {"answer": answer}
```

这里的 `ChatRequest` 是请求体的形状，不是模型服务的请求对象。FastAPI 看到参数 `request: ChatRequest`，就会读取 HTTP 请求体中的 JSON，交给 Pydantic 解析和校验。通过以后，路由函数拿到的是一个 `ChatRequest` 实例，可以用 `request.question` 取值；没有通过，函数本身就不会执行，客户端会得到说明错误位置的 **422** 响应。

这段代码里，`question` 必须出现，而且必须是长度在 1 到 4000 之间的字符串。`extra="forbid"` 让未知字段也报错，**Pydantic 默认会忽略多余字段**，不会自动拒绝它们。[Pydantic 的配置说明](https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.extra)也展示了 `ignore`、`forbid`、`allow` 这几种处理方式。

现在，类型标注不再只是给编辑器看的提示。`BaseModel` 会依据它在运行时处理输入，而 FastAPI 把这个处理步骤放在路由函数之前。客户端用 `curl` 还是 `fetch` 发请求都一样，这套校验并不认识也不关心客户端是用什么语言写的。

## 必填、可空和默认值

写 schema 时，我最容易混在一起的，是“可以不传”和“可以传 `null`”。看下面几个字段：

```python
from pydantic import BaseModel, Field


class ExampleRequest(BaseModel):
    question: str
    context: str | None
    note: str | None = None
    limit: int = Field(default=3, ge=1, le=10)
```

`question` 必须传，且不能是 `null`。`context` 也必须出现在请求中，不过它的值可以是字符串或 `null`。`note` 可以完全不传，没传时得到 `None`。`limit` 不传时用 `3`，传了就得满足 1 到 10 的范围。`str | None` 只说明值允许为 `None`，**有没有默认值**才决定客户端能否省略字段。

还有一个容易被忽略的细节：Pydantic 默认有时会转换输入。例如某些数字字段收到字符串形式的数字，可能会被解析成数字。如果这不是我们想要的，可以给字段或模型设置严格模式。我们可以在这里自己设计相关转换规则，不过细则就交给 AI 写吧。

请求再复杂一些时，也不必回到一个巨大的 `dict`。比如以后 Bot 要接收会话消息，我们可以让 `ChatRequest` 包含 `list[Message]`，由 `Message` 模型规定每条消息的 `role` 和 `content`。校验会顺着嵌套结构往里走，报错也能指向具体是哪一条消息、哪一个字段。schema 可以随着数据结构一起组合，而不是只检查最外面一层，这就是 Pydantic 数据契约的价值。

## 入口校验之后，业务判断才开始

假设请求的 JSON 合法，`question` 也确实是非空字符串，后端是不是就可以无条件调用模型？先别急。我们至少还想知道：是谁在调用这个接口？一个最小的办法是给调用方一串 API Key，约定它在请求头里发送 `X-API-Key: ...`。后端保存预期的密钥，收到请求后取出请求头里的值，与它比较。没带或者带错了，就停在这里，不调用模型。这是鉴权；Pydantic 对 `question` 的校验则是在检查数据，两件事解决的问题不同。

把这个逻辑接到上一篇的 Bot，可以写成一个完整的 `main.py`：

```python
import os
import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Annotated[str, Field(min_length=1, max_length=4000)]


app = FastAPI()
key_from_header = APIKeyHeader(name="X-API-Key", auto_error=False)
expected_key = os.environ["CHAT_API_KEY"]


def require_api_key(
    key: Annotated[str | None, Depends(key_from_header)],
) -> None:
    if key is None or not secrets.compare_digest(key, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "APIKey"},
        )


def run_bot(question: str) -> str:
    with OpenAI() as client:
        response = client.responses.create(
            model="gpt-5.6-luna",
            input=question,
        )
    return response.output_text


@app.post("/chat", dependencies=[Depends(require_api_key)])
def chat(request: ChatRequest):
    return {"answer": run_bot(request.question)}
```

先停在 `key_from_header = APIKeyHeader(name="X-API-Key", auto_error=False)` 这一行。`APIKeyHeader` 是 FastAPI 在 `fastapi.security` 里提供的类；我们调用它的构造函数，得到一个从请求头取 API Key的对象，并把它存在 `key_from_header` 变量里。`name="X-API-Key"` 指定客户端必须使用的请求头名字。**创建对象时还没有任何 HTTP 请求进来，也不会在这一行检查密钥。** 它既不会生成 API Key，也不知道 `CHAT_API_KEY` 的值；它只负责在每次请求到达时取出请求头里的字符串，同时把这个认证方式写进 OpenAPI。

`auto_error=False` 控制的是“没带这个请求头时怎么办”：这里让 `APIKeyHeader` 返回 `None`，把决定权留给我们的 `require_api_key()`。如果使用默认的 `True`，缺少请求头时，FastAPI 的这个工具会直接返回认证错误，我们就进不到自己统一处理“没带”和“带错”的判断里。

再看 `Depends`。它是从 FastAPI 导入的**依赖声明工具**，不是 Python 关键字。`Depends(key_from_header)` 不是立刻取一次请求头：它告诉 FastAPI，“等有请求进来时，请调用 `key_from_header`，把结果交给需要它的地方”。虽然 `key_from_header` 是对象而不是我们写的 `def` 函数，它也是可调用的，所以能放进 `Depends(...)`。这里传的是对象本身，不写成 `Depends(key_from_header())`；后一种写法会试图当场调用它。这个“声明现在写好、实际调用留到每次请求时做”的区别，是理解 FastAPI 依赖的关键。[FastAPI 的依赖说明](https://fastapi.tiangolo.com/tutorial/dependencies/)有同样的函数传递规则。

因此，`key: Annotated[str | None, Depends(key_from_header)]` 可以从左往右读：`key` 这个参数最后会是字符串，或者在没带请求头时是 `None`；`Annotated[..., Depends(...)]` 额外告诉 FastAPI 这个值从哪里来，框架会把取到的请求头值传给它。

外面还有一层：路由装饰器里的 `dependencies=[Depends(require_api_key)]` 告诉 FastAPI，调用 `chat()` 前先执行 `require_api_key()`。这里用列表写在装饰器上，是因为这个函数只负责检查，返回 `None`，`chat()` 不需要接收它的结果。于是传值链路就是：**请求头 → `APIKeyHeader` 取出 Key → FastAPI 把 Key 传给 `require_api_key()` → 检查通过才运行 `chat()`**。内层的 `Depends(key_from_header)` 是给函数参数提供值；外层的 `Depends(require_api_key)` 是要求路由先完成检查。

这里有**两把不同的钥匙**。`CHAT_API_KEY` 是我们这个后端用来核对*调用方*的；`OpenAI()` 读取的 `OPENAI_API_KEY` 是后端调用*模型服务*时使用的。前一把由 `/chat` 的客户端放进 `X-API-Key` 请求头，后一把只留在运行后端的机器上。`secrets.compare_digest()` 负责比较请求头里的 Key 与后端保存的预期值。鉴权失败时，`raise HTTPException(...)` 让 FastAPI 返回 **401**，`run_bot()` 不会执行；请求体里的 `question` 不符合 `ChatRequest` 时，FastAPI 返回 **422**。这版只有一把供客户端使用的 Key，只能判断“拿到了这把钥匙”，还不能区分具体是哪位用户。

客户端发来的东西可以这样对上代码：

```http
POST /chat HTTP/1.1
X-API-Key: <调用这个后端的密钥>
Content-Type: application/json

{"question": "你好"}
```

`X-API-Key` 在**请求头**中，`question` 在**请求体**中。

其他状态码可以先按“请求走到了哪一步”来理解：

| 情形 | 返回什么 | 谁来决定 |
| --- | --- | --- |
| API Key 缺失或错误 | **401**，尚未通过身份验证 | `require_api_key()` 抛出 `HTTPException` |
| Key 有效，但无权访问某个会话 | **403**，身份已知但权限不足 | 以后查询会话归属的业务代码 |
| 请求的会话 ID 格式正确，但会话不存在 | **404**，找不到资源 | 以后查询会话的业务代码 |
| 类型或长度不符合约定 | **422**，请求数据校验失败 | FastAPI 和 Pydantic |
| 调用次数超过接口规定的速率限制 | **429**，请求太频繁 | 以后实现的限流逻辑 |
| 都通过，模型也返回答案 | **200**，正常返回 | FastAPI 默认的成功响应 |

眼下这个 `/chat` 没有会话 ID，也没有用户权限表，所以代码里只有 401 和 422 的处理。如果将来增加 `/sessions/{session_id}/chat`，就可以在确认调用方身份后查找会话：查不到时 `raise HTTPException(status_code=404, detail="Session not found")`；查得到但调用方无权访问时，`raise HTTPException(status_code=403, detail="Forbidden")`。Pydantic 只能判断 `session_id` 是否符合声明的类型，不能替我们查询这个会话属于谁。

再看请求里数据放在哪里。我们把接口扩成 `/sessions/42/chat?max_output_tokens=256`。`42` 在**路径**里，`max_output_tokens=256` 在**查询字符串**里，`{"question": "你好"}` 在 **JSON 请求体**里，而 API Key 仍在**请求头**里。FastAPI 会根据路由和函数参数分别读取这些位置。客户端必须按约定放数据；在查询字符串里写一个 `question=你好`，不会自动变成请求体里的 `ChatRequest.question`。下面把这几个位置真正接到同一个函数上。

## 返回答案也需要一个形状

入口有约定，出口也应该有。否则今天返回 `{"answer": "..."}`，明天不小心把 SDK 的原始响应、内部请求记录甚至不该暴露的字段一起返回，前端就只能猜。我们可以另外定义一个响应模型：

```python
class ChatResponse(BaseModel):
    answer: str


@app.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_api_key)],
)
def chat(request: ChatRequest):
    answer = run_bot(request.question)
    return ChatResponse(answer=answer)
```

这里是在上一节的路由上增加 `response_model`，所以 `dependencies=[Depends(require_api_key)]` 仍然保留。`run_bot()` 还是上一篇那个调用真实模型的函数。路由负责把已经校验的 `question` 交给它，再把结果装进 `ChatResponse`。`response_model` 会让 FastAPI 按响应模型处理和输出数据，也会把响应结构写进接口描述。即使内部对象还有别的字段，最终对外的结果也应该按这个边界来设计；[FastAPI 的响应模型说明](https://fastapi.tiangolo.com/tutorial/response-model/)还展示了返回数据的过滤行为。

我会尽量把 `ChatRequest` 和 `ChatResponse` 分开。它们现在只有一两个字段，看着像多写了两个类；但它们代表不同方向的承诺。输入里可以有用户的问题，输出里只该有公开的答案。以后接入数据库时，内部可能还有主键、费用、日志或密钥，都需要通过数据契约来更好的解析和返回。

Pydantic 模型在 Python 里很好用，不过它也不是 JSON 或者 Python dict 本身。`ChatResponse(answer="你好").model_dump()` 得到 Python `dict`，`model_dump_json()` 得到 JSON 字符串。FastAPI 返回普通模型时会负责后续的 HTTP 响应序列化，通常不需要我们自己先调用 `model_dump_json()`。要理解对象形式的不同和变化，这是面向对象语言的要点之一。

## 一个完整的会话接口

这里的 session 先指一个**问答会话资源**：它有 ID，也有能访问它的调用方。我们先用一个内存字典代替将来的会话表，让路径、查询、请求体、鉴权和响应模型在同一个例子里工作：

```python
import os
import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.security import APIKeyHeader
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Annotated[str, Field(min_length=1, max_length=4000)]


class ChatResponse(BaseModel):
    session_id: int
    answer: str


app = FastAPI(title="Q&A Bot API")
key_from_header = APIKeyHeader(name="X-API-Key", auto_error=False)
expected_key = os.environ["CHAT_API_KEY"]
session_owners = {42: "demo-client", 43: "other-client"}


def identify_client(
    key: Annotated[str | None, Depends(key_from_header)],
) -> str:
    if key is None or not secrets.compare_digest(key, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    return "demo-client"


@app.post(
    "/sessions/{session_id}/chat",
    summary="Ask a question in a session",
    response_model=ChatResponse,
    responses={
        401: {"description": "Invalid API key"},
        403: {"description": "Session belongs to another client"},
        404: {"description": "Session not found"},
    },
)
def chat(
    session_id: int,
    request: ChatRequest,
    client_id: Annotated[str, Depends(identify_client)],
    max_output_tokens: Annotated[int, Query(ge=16, le=1024)] = 256,
) -> ChatResponse:
    owner_id = session_owners.get(session_id)
    if owner_id is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if owner_id != client_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    with OpenAI() as client:
        response = client.responses.create(
            model="gpt-5.6-luna",
            input=request.question,
            max_output_tokens=max_output_tokens,
        )
    return ChatResponse(session_id=session_id, answer=response.output_text)
```

代码仍然调用真实模型；运行它的后端进程需要上一篇的 `OPENAI_API_KEY`，也需要供这个后端核对调用方的 `CHAT_API_KEY`。`session_owners` 则只是为了把会话归属这个判断写清楚：`42` 属于 `demo-client`，`43` 属于另一个调用方。

现在把一个请求从外向里读：

```http
POST /sessions/42/chat?max_output_tokens=256 HTTP/1.1
X-API-Key: <与 CHAT_API_KEY 相同的值>
Content-Type: application/json

{"question": "什么是 FastAPI？"}
```

`{session_id}` 与函数参数 `session_id: int` 同名，FastAPI 从路径里取出 `42` 并转成整数。`max_output_tokens` 由 `Query(...)` 声明在查询字符串里，没传时用默认的 `256`，传了就检查 16 到 1024 的范围。`request: ChatRequest` 让 JSON 请求体变成经过校验的对象。`Depends(identify_client)` 则先从请求头读 Key、核对它，再把返回的 `demo-client` 作为 `client_id` 交给路由。前面见过的几种参数位置，到这里终于对上了同一组 Python 参数。

接着才是业务判断：找不到 `session_id` 就返回 **404**；找到了但 `owner_id` 与 `client_id` 不同，就返回 **403**；都通过后才会调用模型。用上面的 Key 请求 `42`，会得到包含 `session_id` 和模型答案的 **200** 响应；请求 `43` 会走到 403，请求 `999` 会走到 404。Key 不对则在进入路由前得到 401，`question` 或查询参数不合规则得到 422。

装饰器中的 `responses={...}` 还有另一层作用：它**描述**了 401、403、404 可能出现，供 OpenAPI 文档展示；它本身不会执行鉴权或查会话。真正产生这些响应的是 `identify_client()` 和路由里的 `raise HTTPException(...)`。这样我们就能区分“告诉别人接口可能返回什么”和“请求来了以后代码实际做什么”。

我把它写在同一个文件里，是为了能顺着一次请求把流程看完。以后项目长大，`ChatRequest` 和 `ChatResponse` 可以放在 schema 模块，`identify_client()` 放在鉴权模块，会话查询交给数据访问层，调用模型交给服务层；路由只负责把它们接起来。代码分到几个文件，接口接收什么、返回什么以及业务规则本身并不会因此改变。

### 问号前后：路径和查询怎么分工

URL 里的 `?` 是分界线：前面的 `/sessions/42/chat` 是**路径**，用来找到“对 42 号会话发起聊天”这个接口；后面的 `max_output_tokens=256` 是**查询参数**，用 `名字=值` 给这次调用附加一个选项。比较 `/sessions/42/chat?max_output_tokens=256` 和 `/sessions/42/chat?max_output_tokens=512`，会话仍是 42，只是生成长度的上限变了；换成 `/sessions/43/chat?max_output_tokens=256`，才是换了会话。一个 URL 有多个查询参数时，从 `?` 开始，用 `&` 连接后面的参数。

这是一种很实用的设计习惯：路径通常放“要访问哪一个资源”，查询通常放“这次怎样访问它”，比如筛选、排序、分页或这里的生成上限。它们都在 URL 中，却不会被 FastAPI 混成一个值。上面的代码里，`{session_id}` 对应 `session_id: int`，而 `max_output_tokens` 由 `Query(...)` 读取；它能省略，是因为我们给了 `= 256` 的默认值，并不是所有查询参数天生都可省略。路径里缺了 `42`，这条会话路由就匹配不上。

## OpenAPI 是给接口画一张可读的地图

到这里，我们已经在 Python 代码中写下请求和响应的形状。那浏览器前端、另一个服务，或者一个刚加入项目的人，怎么知道 `/sessions/{session_id}/chat` 接受什么、返回什么？FastAPI 会利用这些模型生成 **JSON Schema**，再把它们连同路径、HTTP 方法、参数、鉴权方式和响应一起放进 **OpenAPI** 描述里。

JSON Schema 关心“这份 JSON 数据是什么形状”；OpenAPI 关心“这个 API 有哪些入口、如何调用、会收到什么”。前者可以成为后者的一部分。默认配置下，我们可以查看 `/openapi.json` 中的机器可读描述，也可以打开 `/docs` 看交互式文档；用规范的语言开发，就可以让机器自动根据这些规范生成交接文档方便其他人接受项目。

下面这张图就是上面那段代码启动后，打开 `/docs`、展开 `POST /sessions/{session_id}/chat` 时的实际页面。它不是我们手写的前端页面，而是 FastAPI 根据接口声明生成的 Swagger UI：

![问答 Bot 的 FastAPI 交互式 API 文档，展示路径参数、查询参数、请求体和鉴权入口](/assets/images/full-stack-development/fastapi-session-chat-openapi-docs.png)

看图时先找四处：`session_id` 标为 **path**，`max_output_tokens` 标为 **query**，`question` 出现在 **Request body** 的 JSON 示例里，右上角的 **Authorize** 用来填写 `X-API-Key`。点 **Try it out** 可以在页面里填参数并发起真实请求。文档下面还会列出 200、401、403、404、422 等响应；前三个错误描述来自我们写的 `responses={...}`，422 则来自 FastAPI 的请求校验。这个界面背后读取的是 `/openapi.json`，调用的仍是同一条后端接口。

于是同一个 `ChatRequest.question` 会出现在两条链路中。一条是**运行时**：客户端发送 JSON，FastAPI 和 Pydantic 校验，路由调用 Bot，响应模型约束返回内容。另一条是**描述时**：Pydantic 模型提供数据结构，FastAPI 把接口拼成 OpenAPI，文档页面和客户端工具据此知道怎样发请求。两条链路用的是同一套声明，所以改动字段时，运行规则和接口描述可以一起更新。

这和给 Agent 的工具写参数 schema 有点像：调用方需要知道工具叫什么、参数是什么形状；真正执行时，还得检查收到的参数能否使用。不过类比到这里就够了。工具调用与对公网开放的 HTTP API 各有自己的执行环境，不能因为都出现了 schema，就把认证、权限和业务判断也交给数据模型。

## 回到那个最简单的问题

如果现在再看从 `POST /chat` 扩展出来的会话接口，它已经不只是“把 JSON 读出来，调用一个函数”。外部请求先经过 Key 的核对，再把各个位置的数据转换成 Python 参数；代码查到会话并确认归属后，才取出 `question` 调用模型，最后用 `ChatResponse` 决定对外说什么。OpenAPI 则把这段约定交给接口使用者阅读和复用。

这套结构看起来比直接用 `dict` 啰嗦一些，但它让错误更早出现，也让前后端各自知道自己该遵守什么。下一步如果要把路由、业务逻辑和数据访问拆开，我们至少已经知道了最外面这层边界在哪里。
