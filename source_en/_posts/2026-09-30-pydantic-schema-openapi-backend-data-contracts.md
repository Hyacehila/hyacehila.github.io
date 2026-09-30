---
title: "Backend Data Boundaries: How a Question Reaches a Q&A Bot"
title_zh: "后端的数据边界：一个问题怎样进入问答 Bot"
date: 2026-09-30 12:30:00 +0800
categories: ["Programming", "Full Stack Development"]
tags: [Backend, FastAPI, Python, Pydantic, OpenAPI, JSON Schema]
author: Hyacehila
excerpt: "Following the Q&A bot's /chat endpoint, see how incoming JSON becomes validated Python data, how the answer takes a defined shape, and where Pydantic, schemas, and OpenAPI fit in."
description: "Following the Q&A bot's /chat endpoint, see how incoming JSON becomes validated Python data, how the answer takes a defined shape, and where Pydantic, schemas, and OpenAPI fit in."
excerpt_zh: "沿着上一篇问答 Bot 的 /chat 接口，看看客户端送来的 JSON 怎样变成可信的 Python 数据，答案又怎样按约定返回；再由此理解 Pydantic、Schema 和 OpenAPI 在后端各自做什么。"
mathjax: false
hidden: true
permalink: '/blog/2026/09/30/pydantic-schema-openapi-backend-data-contracts/'
lang: en
translation_key: 2026-09-30-pydantic-schema-openapi-backend-data-contracts
translation_status: machine
translation_source_hash: 0b4caf5a59266b44dbe98bdf313c358a2f7a6b509205c2f5aeb20d081d9db2a0
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

In the previous post, we gave a Q&A bot that could only be called inside a Python process a network entry point: `POST /chat`. A client sends `question`, the backend passes it to a language model, and then returns `answer`. Whether the client uses `curl` or `fetch` in a browser, it can reach this endpoint by following the same HTTP agreement.

Receiving a request, though, does not mean we can safely use it. What if `question` is missing or contains an array? If I return the entire object from the model call, can the client reliably find the answer? I want to follow these two questions to see how a backend draws boundaries around its data. We are still using the simplest bot: a user asks a question, and the model answers.

## Incoming JSON Is Not Yet the Python Object We Want

Without Pydantic, taking a `dict` seems to work:

```python
@app.post("/chat")
def chat(data: dict):
    answer = run_bot(data["question"])
    return {"answer": answer}
```

A valid request works. But `data["question"]` assumes the key exists. Even when it does, nothing here guarantees that its value is the string we intend to pass to `run_bot(question: str)`. The `str` annotation on a Python function parameter does not, by itself, reject `run_bot([1, 2, 3])` at runtime. If the model SDK fails later, we see an exception in the middle of the business operation when the original problem was the client's input. We could write `try` blocks at every entry point to catch all sorts of errors, but that quickly becomes tedious.

At the entry point, I want answers to more precise questions: which fields are required, what type does each field hold, can a string be empty, and what happens to unknown fields? Together, these rules form the endpoint's **input contract**, often called a schema or data contract. They specify what kind of data this endpoint is willing to accept.

## Put the Contract into Code with Pydantic

FastAPI commonly uses a Pydantic model to describe a request body. Here is the same endpoint with one:

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

`ChatRequest` describes the HTTP request body, not the request object used by the model provider. When FastAPI sees `request: ChatRequest`, it reads the JSON body and gives it to Pydantic for parsing and validation. If validation succeeds, the route receives a `ChatRequest` instance and can use `request.question`. If it fails, the route does not run; the client receives a **422** response pointing to the error.

Here, `question` must be present and contain a string from 1 to 4000 characters long. `extra="forbid"` also rejects unknown fields. **Pydantic ignores extra fields by default**; it does not reject them automatically. The [Pydantic configuration reference](https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.extra) describes the `ignore`, `forbid`, and `allow` options.

The type annotation is now more than an editor hint. `BaseModel` uses it to process incoming data at runtime, and FastAPI places that step before the route function. Whether the client uses `curl` or `fetch`, validation does not know or care which language the client was written in.

## Required, Nullable, and Default Values

When writing a schema, it is easy to mix up “may be omitted” and “may be `null`.” Consider these fields:

```python
from pydantic import BaseModel, Field


class ExampleRequest(BaseModel):
    question: str
    context: str | None
    note: str | None = None
    limit: int = Field(default=3, ge=1, le=10)
```

`question` is required and cannot be `null`. `context` is also required, but its value may be a string or `null`. `note` may be omitted entirely, in which case it becomes `None`. `limit` becomes `3` if omitted and must be between 1 and 10 when supplied. `str | None` says that `None` is an allowed **value**; the presence or absence of a **default** determines whether the field may be omitted.

Another detail is easy to miss: by default, Pydantic sometimes **converts** input. A numeric field, for example, may accept a number written as a string and parse it. If that is not what we want, we can enable strict mode for a field or model. We can design conversion rules ourselves here, and leave the details of implementing them to AI.

More complicated requests do not require a giant `dict`. If the bot later accepts conversation messages, `ChatRequest` could contain a `list[Message]`, with a separate `Message` model defining each message's `role` and `content`. Validation follows the nested structure and can point to the particular message and field that failed. Schemas can be composed as the data grows instead of checking only the outermost layer; that is the value of Pydantic's data contract.

## Business Decisions Begin After Input Validation

Suppose the JSON is valid and `question` is a nonempty string. Can the backend immediately call the model? Not so fast. We at least want to know who is calling the endpoint. A minimal approach is to give the caller an API key and ask it to send `X-API-Key: ...` in a request header. The backend keeps the expected key, reads the header, and compares the two. If the header is missing or wrong, it stops before calling the model. That is authentication; Pydantic's check of `question` is data validation. The two checks answer different questions.

Here is a complete `main.py` that adds this check to the bot from the previous post:

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

First, consider `key_from_header = APIKeyHeader(name="X-API-Key", auto_error=False)`. `APIKeyHeader` is a class provided by FastAPI in `fastapi.security`. Calling its constructor gives us an object that knows how to extract an API key from a request header; we store that object in `key_from_header`. `name="X-API-Key"` specifies the header name the client must use. **There is no HTTP request when this object is created, and no key is checked on this line.** It does not issue keys or know the value of `CHAT_API_KEY`. Its job is to read the header for each incoming request and describe this authentication scheme in OpenAPI.

`auto_error=False` controls what happens when that header is missing. Here it makes `APIKeyHeader` return `None`, leaving the decision to `require_api_key()`. With the default `True`, the FastAPI tool would produce an authentication error immediately for a missing header, before our function could handle missing and incorrect keys in one place.

Now consider `Depends`. It is a **dependency declaration** imported from FastAPI, not a Python keyword. `Depends(key_from_header)` does not immediately read a request header. It tells FastAPI to call `key_from_header` when a request arrives and pass its result where needed. Although `key_from_header` is an object rather than a function we wrote with `def`, it is callable and can therefore be passed to `Depends(...)`. We pass the object itself, not `Depends(key_from_header())`, which would try to call it immediately. The important distinction is that we declare the work now and FastAPI performs it for each request. The [FastAPI dependency guide](https://fastapi.tiangolo.com/tutorial/dependencies/) explains the same rule for passing functions.

Read `key: Annotated[str | None, Depends(key_from_header)]` from left to right. The parameter `key` will be a string, or `None` if the header is absent. `Annotated[..., Depends(...)]` tells FastAPI where the value comes from, and the framework passes the extracted header value into this parameter.

There is one more layer: `dependencies=[Depends(require_api_key)]` in the route decorator asks FastAPI to run `require_api_key()` before `chat()`. We put it in the decorator's list because this function only checks the key and returns `None`; `chat()` does not need its return value. The data flow is **request header → `APIKeyHeader` extracts the key → FastAPI passes it to `require_api_key()` → `chat()` runs only if the check succeeds**. The inner `Depends(key_from_header)` supplies a function argument; the outer `Depends(require_api_key)` requires a check before the route runs.

There are **two different keys** here. `CHAT_API_KEY` lets our backend check its *caller*. The `OPENAI_API_KEY` read by `OpenAI()` lets the backend call the *model service*. The client of `/chat` sends the first one in the `X-API-Key` header; the second stays on the machine running the backend. `secrets.compare_digest()` compares the supplied key with the expected value held by the backend. If authentication fails, `raise HTTPException(...)` makes FastAPI return **401**, and `run_bot()` is not called. If `question` fails the `ChatRequest` contract, FastAPI returns **422**. This version has only one key for clients: it tells us someone holds that key, but not which individual user they are.

The client's HTTP message maps to the code like this:

```http
POST /chat HTTP/1.1
X-API-Key: <the key for calling this backend>
Content-Type: application/json

{"question": "Hello"}
```

`X-API-Key` is in a **request header**; `question` is in the **body**.

The other status codes are easier to understand by asking how far the request got:

| Situation | Response | Decided by |
| --- | --- | --- |
| API key missing or incorrect | **401**: caller not authenticated | `require_api_key()` raises `HTTPException` |
| Valid key, but no permission to access a conversation | **403**: identity known, access denied | Future business code checking ownership |
| A well-formed conversation ID names no existing conversation | **404**: resource not found | Future business code looking it up |
| The type or length violates the contract | **422**: request validation failed | FastAPI and Pydantic |
| Too many calls under the endpoint's rate limit | **429**: too many requests | Future rate-limiting logic |
| All checks pass and the model answers | **200**: normal result | FastAPI's default success response |

Our current `/chat` has no conversation ID or user permission database, so the example only needs the 401 and 422 paths. If we later add `/sessions/{session_id}/chat`, we can look up the conversation after identifying the caller: `raise HTTPException(status_code=404, detail="Session not found")` if it does not exist, or `raise HTTPException(status_code=403, detail="Forbidden")` if it exists but the caller lacks access. Pydantic can check the declared type of `session_id`; it cannot look up who owns that conversation.

Finally, consider where the different values are placed. Suppose we extend the endpoint to `/sessions/42/chat?max_output_tokens=256`: `42` is in the **path**, `max_output_tokens=256` is in the **query string**, `{"question": "Hello"}` is in the **JSON body**, and the API key remains in a **header**. FastAPI reads each location according to the route and function parameters. The client must put values where the API expects them: `question=Hello` in the query string does not automatically become `ChatRequest.question` in the body. Next, let us connect these locations to one function.

## The Answer Needs a Shape Too

The entry point has a contract; the exit should have one as well. Otherwise, an endpoint that returns `{"answer": "..."}` today might accidentally return the raw SDK response, an internal request record, or a field that should stay private tomorrow. We can define a separate response model:

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

This adds `response_model` to the route from the previous section, so `dependencies=[Depends(require_api_key)]` remains in place. `run_bot()` is still the function from the previous post that calls a real model. The route gives it the validated `question` and wraps the answer in `ChatResponse`. FastAPI uses `response_model` to process the outgoing data and to describe the response in the API specification. The [FastAPI response model guide](https://fastapi.tiangolo.com/tutorial/response-model/) also shows how the declared model filters fields from returned data.

I prefer separate `ChatRequest` and `ChatResponse` classes even though they are tiny here. They represent promises in opposite directions. Input contains the user's question; output should contain only the public answer. Once a database is involved, internal data may also include primary keys, costs, logs, or secrets. A data contract helps us decide how that data should be interpreted and returned.

A Pydantic model is useful inside Python, but it is neither JSON nor a Python `dict` itself. `ChatResponse(answer="Hello").model_dump()` gives a Python `dict`; `model_dump_json()` gives a JSON string. When we return a regular model from a FastAPI route, FastAPI handles the subsequent HTTP response serialization, so we normally do not call `model_dump_json()` first. Understanding how these object forms differ and change is one of the important ideas in object-oriented programming.

## A Complete Session Endpoint

Here, a session is a **Q&A conversation resource** with an ID and a caller who may access it. For now, a dictionary in memory stands in for a future session table so that path, query, body, authentication, and response model can work together in one example:

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

This still calls the real model. The backend process needs the `OPENAI_API_KEY` from the previous post and the `CHAT_API_KEY` it uses to check callers. `session_owners` exists only to make ownership visible in the example: session `42` belongs to `demo-client`, while `43` belongs to another caller.

Now read one request from the outside in:

```http
POST /sessions/42/chat?max_output_tokens=256 HTTP/1.1
X-API-Key: <the value of CHAT_API_KEY>
Content-Type: application/json

{"question": "What is FastAPI?"}
```

`{session_id}` has the same name as the `session_id: int` function parameter, so FastAPI takes `42` from the path and converts it to an integer. `Query(...)` declares `max_output_tokens` in the query string; it defaults to `256` and accepts values from 16 to 1024. `request: ChatRequest` turns the JSON body into a validated object. `Depends(identify_client)` reads and checks the key from the header, then passes the returned `demo-client` to the route as `client_id`. The different parts of the HTTP request now line up with one set of Python parameters.

Only then does the business decision happen. An unknown `session_id` gets **404**; a known session whose `owner_id` differs from `client_id` gets **403**. The model is called only after both checks pass. With the key above, session `42` produces a **200** response containing `session_id` and the model's answer; `43` produces 403, and `999` produces 404. An incorrect key fails with 401 before the route runs, while an invalid `question` or query parameter gets 422.

The decorator's `responses={...}` has a different job: it **describes** the possible 401, 403, and 404 outcomes for OpenAPI. It does not authenticate anyone or look up a session. The actual responses come from `identify_client()` and the route's `raise HTTPException(...)` branches. That is the distinction between telling callers what an API may return and writing the code that decides what happens to an actual request.

I kept the example in one file so we can follow a request from beginning to end. As a project grows, `ChatRequest` and `ChatResponse` can move into a schema module, `identify_client()` into an authentication module, session lookup into a data access layer, and the model call into a service layer. The route then connects those parts. Splitting code into files does not change what the API accepts and returns or the business rules themselves.

### Before and After the Question Mark: Path and Query

The `?` in a URL separates two parts. Before it, `/sessions/42/chat` is the **path**: it locates the endpoint for chatting in session 42. After it, `max_output_tokens=256` is a **query parameter**, a `name=value` option for this call. Compare `/sessions/42/chat?max_output_tokens=256` with `/sessions/42/chat?max_output_tokens=512`: the session is still 42, but the output limit changes. With `/sessions/43/chat?max_output_tokens=256`, we are talking to a different session. If a URL has several query parameters, the first follows `?` and the others are joined with `&`.

This is a useful API design convention: the path usually says **which resource** we want, while the query says **how to access it this time**, such as filtering, sorting, pagination, or the output limit here. Both live in the URL, but FastAPI does not merge them into one value. In the code above, `{session_id}` maps to `session_id: int`, while `Query(...)` reads `max_output_tokens`. That query parameter may be omitted because we gave it a default of `256`; query parameters are not automatically optional. Remove `42` from the path and this session route no longer matches.

## OpenAPI Is a Readable Map of the API

We have now written the input and output shapes in Python. How can a browser frontend, another service, or a new teammate know what `/sessions/{session_id}/chat` accepts and returns? FastAPI generates **JSON Schema** from these models, then includes those schemas alongside paths, HTTP methods, parameters, authentication, and responses in an **OpenAPI** description.

JSON Schema describes the shape of JSON data. OpenAPI describes an API: its endpoints, how to call them, and what to expect back. JSON Schema can be part of an OpenAPI description. With FastAPI's default configuration, `/openapi.json` exposes the machine-readable description, while `/docs` presents interactive documentation. Writing declarations in a standard form lets a tool generate a handoff document that another developer can use to understand the project.

The screenshot below shows the actual `/docs` page after starting the complete example and expanding `POST /sessions/{session_id}/chat`. We did not hand-build this frontend page; FastAPI generated its Swagger UI from the endpoint declarations:

![FastAPI interactive API docs for the Q&A bot, showing path and query parameters, request body, and the authentication control](/assets/images/full-stack-development/fastapi-session-chat-openapi-docs.png)

Four parts are worth finding: `session_id` is labeled **path**, `max_output_tokens` is labeled **query**, `question` appears in the **Request body** JSON example, and **Authorize** at the top is where a caller can enter `X-API-Key`. **Try it out** lets us fill in the fields and send a real request. Farther down, the page also lists responses such as 200, 401, 403, 404, and 422. Our `responses={...}` supplied the descriptions of the three authentication and resource errors; FastAPI added 422 for request validation. The page reads `/openapi.json` and ultimately calls the same backend endpoint.

The same `ChatRequest.question` thus takes part in two flows. At **runtime**, a client sends JSON, FastAPI and Pydantic validate it, the route calls the bot, and the response model shapes the output. At **description time**, the Pydantic model supplies a data shape, FastAPI assembles the OpenAPI description, and documentation or client tools use it to learn how to call the endpoint. Both flows come from the same declarations, so a field change can update the behavior and its description together.

There is a resemblance to writing a parameter schema for an agent tool: the caller needs to know the tool's name and argument shape, and the receiver must check the actual arguments. The analogy is useful up to that point. A tool call and a public HTTP API run in different settings; a schema does not take over authentication, authorization, or business decisions in either one.

## Back to the Original Question

The session endpoint that grew out of `POST /chat` is now more than “read JSON and call a function.” The external request first passes the key check, then its values in different locations become Python parameters. The code looks up the session and checks ownership before sending `question` to the model. Finally, `ChatResponse` decides what we say to the outside world. OpenAPI makes that agreement available for clients and developers to read and reuse.

This structure takes more code than a bare `dict`, but it surfaces mistakes earlier and gives both sides of the API a clearer agreement. If we later separate routes, business logic, and data access, we already know where the outermost boundary lies.
