---
title: "REST APIs: From Calling Functions to Working with Resources"
title_zh: "REST API：从调用函数到操作资源"
date: 2026-10-01 12:30:00 +0800
categories: ["Programming", "CS Foundations"]
tags: [Backend, HTTP, FastAPI, Python]
author: Hyacehila
excerpt: "Understand REST APIs through resources, HTTP methods, and parameter locations, then put those ideas into a minimal FastAPI user API."
description: "Understand REST APIs through resources, HTTP methods, and parameter locations, then put those ideas into a minimal FastAPI user API."
excerpt_zh: "从资源、HTTP 方法和请求中的参数位置理解 REST API，再用一个最小的 FastAPI 用户接口看看这些约定怎样落到代码里。"
mathjax: false
hidden: true
permalink: '/blog/2026/10/01/rest-api-resource-design-fastapi/'
lang: en
translation_key: 2026-10-01-rest-api-resource-design-fastapi
translation_status: machine
translation_source_hash: 19cf43ffd655c78dc16181dbd94cba0592e6876f3daec6033f2b554b91b98a55
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

Once an API works, there is still a design question: should an endpoint be called `/getUser` or `/users/123`? Do reading and deleting a user each need a separate function name? These are questions that REST API design addresses. REST stands for **Representational State Transfer**. Rather than dwell on the name, I prefer to understand it through a practical change: when designing an API, first identify the system's resources, then define how to access and modify them.

## From Remote Functions to Resources

Suppose we want to retrieve user 123. Writing `/getUser?id=123` feels like moving Python's `getUser(123)` onto the network: the client knows the name of an operation and passes it some arguments.

A REST-style API would typically use:

```http
GET /users/123
```

Here, `/users/123` identifies a user resource, and `GET` means to read it. Using `DELETE` with the same address means to delete that user. The path identifies the object, while the HTTP method expresses the operation, so each action does not need a new URL.

A resource does not have to correspond to a database table. Orders in an e-commerce system, or conversations and messages in an agent system, can all be resources. For example, `POST /conversations/42/messages` can be defined as adding a message to conversation 42.

Thinking in terms of resources is one entry point into REST. The complete architecture also includes constraints such as statelessness, caching, and a uniform interface; see [Fielding's original description of REST](https://ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm) for the details.

## Addresses and Methods Each Have a Role

For user resources, we can arrange the endpoints like this:

| Request | Meaning in this API |
| --- | --- |
| `GET /users/123` | Read user 123 |
| `POST /users` | Create a user in the user collection |
| `PUT /users/123` | Replace user 123's entire representation; the API can also allow creating a resource at that address |
| `PATCH /users/123` | Partially modify user 123 |
| `DELETE /users/123` | Delete user 123 |

These choices must follow the semantics of the HTTP methods themselves. For example, `GET` is for reading and should not also perform a deletion. `POST` has broader uses than creation: it submits data for the target resource to process. The [HTTP specification](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.3) defines these methods' meanings.

Compare `/users/123` with `/users?age=20`: the first identifies one user, while the second filters the user collection by age. This also corresponds to the different parameter locations in an HTTP request: **Path usually identifies the resource being accessed; Query usually specifies options such as filtering, sorting, and pagination; Body contains the data being submitted; Header carries request metadata such as authentication information and content type.** Query parameters can also specify operation options, rather than just filters.

## Transfer Representations, and Make Requests Independently Understandable

When reading `/users/123`, a client usually receives JSON like this:

```json
{"id": 123, "name": "Alice", "age": 20}
```

This JSON is a **representation** of the user resource. It contains the information we choose to expose, rather than sending the database object exactly as it is.

Another easily misunderstood principle is **statelessness**. Each request must carry the information needed to understand the operation, rather than relying on the server to remember the context of the caller's previous request. For example, a request to access a user includes the resource address and any necessary authentication information. It cannot simply say "keep viewing that last user" and leave the server to guess which user "last" refers to.

This does not mean the server cannot store data. User profiles, orders, and chat histories can certainly live on the server. The same applies to a chat system: conversation history can be stored as a resource. The client explicitly includes the conversation ID and authentication information with each request, and the backend looks up the history by ID. Persistent resource state and client session context retained to understand requests are different things.

## A Small Example with FastAPI

The example implements only creation, retrieval, and filtering. It uses an in-memory dictionary to demonstrate API design: data disappears when the process restarts, and concurrent ID allocation is not handled. A real project typically leaves persistence and ID generation to a database. Save the code as `main.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    name: str = Field(min_length=1)
    age: int = Field(ge=0)


class User(UserCreate):
    id: int


app = FastAPI()
users: dict[int, User] = {}


@app.post("/users", response_model=User, status_code=201)
def create_user(data: UserCreate):
    user_id = len(users) + 1
    user = User(id=user_id, **data.model_dump())
    users[user_id] = user
    return user


@app.get("/users", response_model=list[User])
def list_users(age: int | None = None):
    return [u for u in users.values() if age is None or u.age == age]


@app.get("/users/{user_id}", response_model=User)
def get_user(user_id: int):
    user = users.get(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

In this code, `user_id` matches the path placeholder, so it comes from Path. `age` is a regular parameter that does not appear in the path, so it comes from Query. `data: UserCreate` tells FastAPI to read and validate JSON from Body. This also demonstrates the input and output data contracts: the client supplies a name and age when creating a user, while the server includes an ID in the response.

FastAPI helps us implement HTTP APIs; REST influences how we design them. The framework can receive requests, but we still need to decide what resources the system has and whether each method expresses the intended operation.
