---
title: "REST API：从调用函数到操作资源"
title_en: "REST APIs: From Calling Functions to Working with Resources"
date: 2026-10-01 12:30:00 +0800
categories: ["Programming", "CS Foundations"]
tags: [Backend, HTTP, FastAPI, Python]
author: Hyacehila
excerpt: "从资源、HTTP 方法和请求中的参数位置理解 REST API，再用一个最小的 FastAPI 用户接口看看这些约定怎样落到代码里。"
excerpt_en: "Understand REST APIs through resources, HTTP methods, and parameter locations, then put those ideas into a minimal FastAPI user API."
description: "从资源、HTTP 方法、资源表示和无状态理解 REST API，并用一个简单的 FastAPI 示例实现用户的创建、读取与筛选。"
mathjax: false
hidden: true
permalink: '/blog/2026/10/01/rest-api-resource-design-fastapi/'
---

接口能跑起来以后，还有一个设计问题：地址应该叫 `/getUser`，还是 `/users/123`？读取和删除用户，要不要各起一个函数名字？这是 REST API 关心的事情。REST 的全称是 **Representational State Transfer**，通常译作“表述性状态转移”。先不纠结这个名字，我更愿意从一个具体变化理解它：设计接口时，先找出系统里的资源，再约定怎样访问和修改它们。

## 从远程函数，换成资源

假设我们要获取用户 123。写成 `/getUser?id=123`，很像把 Python 里的 `getUser(123)` 搬到网络上：客户端知道一个操作的名字，再把参数交给它。

REST 风格通常会写成：

```http
GET /users/123
```

这里 `/users/123` 标识一个用户资源，`GET` 表示读取。同一个地址换成 `DELETE`，就表示删除这个用户。路径负责标识对象，HTTP 方法负责表达操作，不必为每个动作重新起一个 URL。

资源也不一定对应数据库中的一张表。电商里的订单、Agent 系统里的会话和消息，都可以作为资源。比如 `POST /conversations/42/messages`，可以约定为向 42 号会话中添加一条消息。

资源导向是理解 REST 的一个入口，完整的 REST 架构还包含无状态、缓存和统一接口等约束，具体可以看 [Fielding 对 REST 的原始说明](https://ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm)。

## 地址和方法，各自说清一件事

围绕用户资源，可以先这样安排接口：

| 请求 | 在这个接口中的含义 |
| --- | --- |
| `GET /users/123` | 读取用户 123 |
| `POST /users` | 在用户集合中创建用户 |
| `PUT /users/123` | 整体替换用户 123 的表示，也可以约定在该地址创建资源 |
| `PATCH /users/123` | 部分修改用户 123 |
| `DELETE /users/123` | 删除用户 123 |

这些安排要遵守 HTTP 方法本身的语义。例如，`GET` 用来读取，不应该顺便执行删除；`POST` 的用途比创建更广，它表示把数据交给目标资源处理。[HTTP 规范](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.3)定义了这些方法的含义。

再比较 `/users/123` 和 `/users?age=20`：前者定位一个用户，后者是在用户集合上按年龄筛选。这也对应了 HTTP 请求中几种参数的位置：**Path 通常标识访问的资源，Query 通常表达筛选、排序、分页等访问选项，Body 放要提交的数据，Header 携带认证信息、内容类型等请求元信息。** 查询参数也可以是操作选项，并不只用于筛选。

## 传输的是表示，请求要能独立理解

读取 `/users/123` 时，客户端拿到的通常是这样的 JSON：

```json
{"id": 123, "name": "Alice", "age": 20}
```

这个 JSON 是用户资源的一种**表示（representation）**。它是我们选择对外提供的信息，不是把数据库对象原封不动地传过去。

另一个容易误会的原则是**无状态（stateless）**。它要求每次请求带齐理解这次操作所需的信息，而不是依赖服务器记住调用方上一次请求的上下文。例如，访问用户时带上资源地址和必要的认证信息，不能只发一句“继续查看刚才那个用户”，让服务器猜“刚才”是谁。

这不等于服务器不能保存数据。用户资料、订单、聊天记录当然可以存在服务器上。对聊天系统也是如此：会话历史可以保存为资源；客户端每次明确带上会话 ID 和认证信息，后端再按 ID 查询历史。资源的持久状态与为了理解请求而保留的客户端会话上下文，是两回事。

## 用 FastAPI 写一个小例子

下面只实现创建、读取和筛选。这里用内存字典演示接口设计，重启后数据会丢失，也没有处理并发分配 ID；实际项目通常把持久化和 ID 生成交给数据库。把代码保存为 `main.py`：

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

代码里 `user_id` 与路径占位符同名，所以来自 Path；`age` 是路径中没有出现的普通参数，所以来自 Query；`data: UserCreate` 则让 FastAPI 从 Body 读取并校验 JSON。这里也体现了输入与输出的数据契约：创建时客户端提供姓名和年龄，返回时服务端还会提供 ID。

FastAPI 帮我们实现 HTTP API，REST 则影响我们怎样设计这些接口。框架能把请求接进来，但系统里有哪些资源、方法是否表达了正确的动作，仍然需要我们自己想清楚。
