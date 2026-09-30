---
title: "Backend Basics: How a Question-Answer Bot Becomes an Internet API"
title_zh: "后端基础：一个问答 Bot 如何变成互联网上的 API"
date: 2026-09-30 00:25:00 +0800
categories: ["Programming", "Full Stack Development"]
tags: [Backend, FastAPI, Python, Agent Runtime, HTTP, JavaScript]
author: Hyacehila
excerpt: "Starting with a minimal question-answer bot, compare requests from curl and a browser to see how HTTP, FastAPI routes, and data contracts make a Python function available as a backend API."
description: "Starting with a minimal question-answer bot, compare requests from curl and a browser to see how HTTP, FastAPI routes, and data contracts make a Python function available as a backend API."
excerpt_zh: "从一个把用户问题交给语言模型的最小问答 Bot 出发，对照 curl 和浏览器发出的请求，理解 Python 函数如何通过 HTTP、FastAPI 路由与数据约定，成为客户端可以调用的后端 API。"
mathjax: false
hidden: true
permalink: '/blog/2026/09/30/backend-basics-agent-runtime-fastapi-full-stack/'
lang: en
translation_key: 2026-09-30-backend-basics-agent-runtime-fastapi-full-stack
translation_status: machine
translation_source_hash: cf12dd119dd14b76c3741383eeb517a464eeee6469e336f9c3fbd3acedfe7de8
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

When I worked on agents before, I usually started by asking whether the model could answer, whether tools could be called, and whether the task loop could run. Once I wanted other people to use one, the question changed: how can a function that exists only inside a Python process on my computer be called by a browser on another machine? Why can a single `curl` command in a terminal obtain the result of that function?

AI Coding can now assemble a page and an API quickly, which makes me want to understand this path more carefully. If I only ask AI to generate `@app.post` and `fetch`, I may have no idea where to begin when something breaks. I will follow a deliberately simple question-answer bot through the whole path. It has no tools, memory, or planning. It is just the smallest useful business function for asking what it means to turn a capability into a service.

## A Bot That Only Python Can Call

A user supplies a question, we pass it to a language model, and we return the answer. Here is the synchronous version. Assume the process already has `OPENAI_API_KEY` available; `OpenAI()` reads it from the environment, while our code creates the client and calls the model.

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
    print(run_bot("What is FastAPI?"))
```

`question` is the input, `run_bot()` is our business function, and the model service is an external API it calls. The example uses [`gpt-5.6-luna`](https://developers.openai.com/api/docs/models/gpt-5.6-luna). `client.responses.create(...)` sends the request. Execution waits there until the model service replies, then reads `output_text`. The `with` block closes the client when the call is done.

The same operation has an asynchronous form. The model, input, and result are unchanged; what changes is how Python waits:

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
    print(asyncio.run(run_bot("What is FastAPI?")))
```

Here `run_bot()` is an asynchronous function. Another asynchronous function must use `await run_bot(question)` to obtain its result; calling `run_bot(question)` alone produces a coroutine object, not an answer. At the top level of a standalone script, `asyncio.run(...)` starts it. `AsyncOpenAI()` creates the asynchronous client, and `async with` closes it. We can leave the question of how these two forms affect multiple simultaneous requests for a later discussion of backend concurrency.

At this point, though, either function can only be called by a program that can run this Python code. JavaScript in a web page cannot directly execute `run_bot()` in another process's memory, possibly on another machine. We need a remote entry point with an agreed input and output. The smallest contract for this example is:

```text
POST /chat
Request JSON: {"question": "What is FastAPI?"}
Response JSON: {"answer": "...the model's answer..."}
```

`run_bot(question)` is a Python function interface; `POST /chat` is a network interface. Both represent the same capability, but they speak different languages. The backend provides a repeatable and inspectable path between them.

## Address: Where Does the Request Go?

Suppose the bot runs on a cloud machine whose public IPv4 address is `203.0.113.10`, with the service listening on port `8000`. A separate machine serves the frontend at `198.51.100.20:5173`. A browser or terminal sends its API request to:

```text
http://203.0.113.10:8000/chat
```

The address `203.0.113.10` identifies the destination machine, `8000` identifies the listening process on it, and `/chat` identifies an entry point inside the web application. Putting the page on another machine makes the frontend and backend boundary visible. A mobile app or another server could take the frontend's place without changing this idea.

| Part of the URL | In this example | Main role |
| --- | --- | --- |
| Scheme | `http` | The client and server exchange HTTP requests and responses |
| Host | `203.0.113.10` | The network routes the request to the destination machine |
| Port | `8000` | The operating system delivers the connection to the listening process |
| Path | `/chat` | FastAPI matches a route inside the application |

I use plain HTTP so the request is easy to inspect. A public service would normally have an HTTPS entry point terminate TLS before passing the request to the application. In particular, a page opened over HTTPS should not call this plain HTTP example URL.

## Give the Bot a FastAPI Entry Point

Now put the synchronous business function in a complete `main.py`. I still want `run_bot()` to accept only a Python string: it should not have to know what an HTTP method, URL, or JSON request looks like.

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

# A simple CORS rule for browser access, separate from the bot's business logic
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

`app = FastAPI()` creates an application object; it does not start listening on a port. `@app.post("/chat")` registers the combination of the **POST method and `/chat` path** with that application. The `chat()` function receives the HTTP request, takes `request.question`, gives it to `run_bot()`, and puts the answer in a `ChatResponse`. We have not exposed the Python function itself to the Internet. We have defined an HTTP entry point whose handler calls it.

`ChatRequest` and `ChatResponse` are contracts at the boundary. The former says that the request body should be a JSON object with a `question` field; the latter describes a successful response. If `question` is missing or the input has the wrong shape, the framework returns a validation error before the business function runs. `question: str` is declared in Python, while the value arrives over the network and is parsed and checked locally by Pydantic.

This version of `main.py` uses ordinary `def` for both the bot and the route. The code waits for the remote model API to return. FastAPI runs a synchronous route in a thread pool so this wait does not directly block its event loop. The synchronous SDK therefore works here, although waiting requests occupy thread resources.

For comparison, here is the complete asynchronous HTTP version as a separate `main_async.py`:

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

The client still sees the same `POST /chat` URL, request JSON, and response JSON. The difference lies inside the server while it waits for the model. The synchronous version waits in a thread; the asynchronous version yields at `await` and resumes when the response arrives. This works because the SDK itself offers an asynchronous client. Merely changing the route from `def` to `async def` while keeping a blocking synchronous model call would not have the same effect.

We can discuss threads, the event loop, and concurrent requests later. With FastAPI, we do not manually call `asyncio.run()` around the route: the server manages that execution context. Blocking an asynchronous route with synchronous I/O can still hold up its event loop despite the framework's support for asynchronous code and thread pools.

Both examples create and close a model client for every request so the boundary stays visible. A long-running service may instead reuse a client and close its connections when the application shuts down; that is a separate engineering decision.

## First, curl: Put One Request on the Table

Assume the service is reachable. From a terminal on another machine, the request looks like this:

```bash
curl -i -X POST "http://203.0.113.10:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is FastAPI?"}'
```

`-X POST` chooses the method, the URL identifies the destination, `-H` supplies a request header, `-d` supplies the body, and `-i` also shows the response headers. A successful response body has this shape:

```json
{"answer":"...the model's answer..."}
```

Conceptually, the corresponding HTTP request looks like:

```http
POST /chat HTTP/1.1
Host: 203.0.113.10:8000
Content-Type: application/json

{"question":"What is FastAPI?"}
```

The first line gives the method and path; `Host` identifies the target host and port; `Content-Type` says how to interpret the body after the blank line. `curl` knows nothing about `run_bot()`, Pydantic, or the model. It constructs an HTTP request from the options we supplied and waits for an HTTP response. That ordinary boundary is what makes it useful across languages and platforms.

Because the command includes `-i`, the terminal shows more than an answer. It shows the **status line, response headers, a blank line, and the response body** together:

```http
HTTP/1.1 200 OK
content-type: application/json

{"answer":"...the model's answer..."}
```

`200` indicates an HTTP success, and `content-type` says that the body is JSON. Only the text after the blank line is the data FastAPI returned according to `ChatResponse`. By default, `curl` prints the response body; `-i` makes it show the headers too. To extract only the answer, a command or script would parse the JSON body without including those headers.

## Then a Page: A Different Client, the Same Interface

Now add a page with a question field, a button, and a place to show the answer. Suppose the page is served from `http://198.51.100.20:5173`. Its JavaScript uses `fetch`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Question-Answer Bot</title>
</head>
<body>
  <textarea id="question" placeholder="Ask the bot a question"></textarea>
  <button id="send">Send</button>
  <pre id="answer"></pre>

  <script>
    const questionInput = document.querySelector("#question");
    const answerBox = document.querySelector("#answer");

    document.querySelector("#send").addEventListener("click", async () => {
      answerBox.textContent = "Waiting for an answer...";

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
        answerBox.textContent = `Request failed: ${error.message}`;
      }
    });
  </script>
</body>
</html>
```

`fetch()` does something close to `curl`: it specifies a URL, method, headers, and body, then obtains a response. But the page must also handle **human interaction**: read the input, show that it is waiting for the model, and display `answer`. `curl` can finish by printing to the terminal; browser JavaScript must turn the response into page state.

These three pieces of code use different syntax but follow one interface contract:

| What must match | curl command | Browser JavaScript | Backend code |
| --- | --- | --- | --- |
| Find the server | `203.0.113.10:8000` in the URL | The same address in the `fetch()` URL | Uvicorn listens on port `8000` by default |
| Find the endpoint | `-X POST` and path `/chat` | `method: "POST"` and path `/chat` | `@app.post("/chat")` registers the method and path |
| State the body format | `-H "Content-Type: application/json"` | `Content-Type` in `headers` | FastAPI reads a JSON body |
| Pass the question | `-d '{"question":"..."}'` | `JSON.stringify({ question: questionInput.value })` | `ChatRequest.question` becomes `request.question` |
| Call the bot | No Python function name | No Python function name | `chat()` calls `run_bot(request.question)` |
| Read the answer | Print the response JSON, then parse `answer` | Call `response.json()` and read `data.answer` | `ChatResponse(answer=answer)` produces the response |

There is no magic matching of function names. `203.0.113.10:8000` gets the request to the machine and port running the service. **POST plus `/chat`** lets FastAPI select `chat()`. The JSON field **`question`** matches the field of `ChatRequest`, becoming `request.question`. Only then does the backend make an ordinary Python call to pass that string to the bot.

The client does not know what `chat()` or `run_bot()` is called. If the frontend changes the field name to `message`, validation fails; if it changes POST to GET, this route will not be selected. The asynchronous version only adds `await` to the internal bot call. It does not change the external contract.

`JSON.stringify({ question: ... })` encodes a JavaScript object as a request-body string. The backend parses and validates it. On the way back, `response.json()` reads the response body into a JavaScript object. This is the same kind of parsing a terminal script would do with `curl` output and `json.load(sys.stdin)`: check the response status, parse JSON, then read `data.answer`.

The same field appears to move in both directions, but each network boundary requires encoding or decoding. The frontend's `await fetch(...)` waits for the backend HTTP response. In the backend, the synchronous version waits in a thread for the model; the asynchronous version writes `await run_bot(...)`. Both frontend and backend may use `await`, but they run in different processes and call chains.

## Why Can FastAPI Call That Python Function?

`curl` and `fetch` now line up. One part of the path remains hidden behind `@app.post`: how does an HTTP request reach `chat()`?

**Uvicorn: letting the application receive network requests**

`app = FastAPI()` creates an application object in Python. It does not occupy port `8000` by itself. Running `uvicorn main:app --host 0.0.0.0 --port 8000` starts the server process that listens there. `main:app` tells Uvicorn where to find our application. When a client connects to the server's IP and port, Uvicorn receives the request first. It handles the network connection and HTTP, passes the method, path, headers, and body to the application, and sends the application's response back to the client. Uvicorn does not choose `chat()` from `POST /chat` or know how `run_bot()` should ask the model. Route matching, request validation, and the business call happen after the request reaches FastAPI.

**ASGI: how Uvicorn and FastAPI communicate**

What does it mean to pass a request to the application? Uvicorn and FastAPI follow a Python interface convention called **ASGI**. It is neither a new protocol the browser must speak nor another service to start. The browser and `curl` still send HTTP. ASGI specifies how a server passes information about a connection and incoming events to a Python application, and how that application sends a response back. A different server that follows the same convention has a common way to run the FastAPI application; Uvicorn likewise knows how to call another compatible application.

The interface uses `scope`, `receive`, and `send`. For now, think of them as connection information, receiving request content, and sending response content. When we write `@app.post("/chat")`, we do not handle those objects ourselves. FastAPI accepts them on the application side of ASGI and brings the request to our route function.

I find it useful to picture the path as a series of increasingly specific decisions:

```text
Public IP:port → Uvicorn receives HTTP → passes it to the app via ASGI
               → FastAPI matches POST /chat to chat()
               → JSON becomes ChatRequest → run_bot(question)
               → ChatResponse becomes JSON → HTTP response returns to the client
```

At the FastAPI layer, the framework matches a route by method and path, validates the JSON body into `ChatRequest`, calls `chat()`, and finally enters `run_bot()`. We do not need the fields inside an ASGI message yet, but knowing the layers helps us locate a failure. If the port cannot be reached, inspect the address, listener, and firewall. If `curl` succeeds but the page fails, inspect the browser console and CORS. A 422 response points toward the JSON fields. If the model call fails inside `run_bot()`, inspect the model service and server environment. That gives us a better starting point than guessing at `fetch` or `@app.post`.

The result crosses the boundaries in reverse. `run_bot()` returns a Python string. `chat()` puts it in `ChatResponse.answer`. The framework serializes it as JSON, then ASGI and Uvicorn deliver the HTTP response. A successful request usually returns `200`; invalid input returns `422`; an unhandled exception in the business function causes a server error. A browser's `fetch()` will usually give us a `Response` object even for a 422 or 500 status, so the page checks `response.ok` itself. `curl -i` shows the status line and body directly. “The API works” can mean three different things: the network is reachable, HTTP returned a response, or the business operation actually succeeded.

## The Backend Intuition I Want to Keep

We have only connected a tiny question-answer bot to the Internet. It has no sessions, database, reliable retries, rate limits, or deployment controls; it is not a complete agent system. Yet this shortest path already shows the basic design of a backend: an internal function solves a business problem, an API expresses that capability as messages other programs can send and understand, and the server and framework connect the two. Replace `curl` with a browser or another program, and the backend's `/chat` contract stays the same.

I think this understanding matters especially in the AI Coding era. Generating an endpoint is easy; understanding what it registers, where it listens, which fields it accepts, when it rejects a request, and where it calls the model is harder. Once these boundaries are clear, we can discuss more complex agent runtimes, asynchronous tasks, persistent state, and full-stack engineering while knowing more precisely where a problem lies.

Backend frameworks have already hidden much of the underlying complexity. Much of our code can focus on the business problem instead of reimplementing those mechanisms. Understanding how the pieces are built and how data moves through them still helps us reason about both the business logic and the engineering. This is a first step toward understanding what it means to make a capability into a service.
