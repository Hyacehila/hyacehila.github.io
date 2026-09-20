---
title: "浏览器交互如何运行：从事件与表单到前后端通信"
title_en: "How Browser Interactions Work: From Events and Forms to Client-Server Communication"
date: 2026-09-20 12:00:00 +0800
categories: ["Programming", "Full Stack Development"]
tags: [JavaScript, DOM, Frontend, Async, HTTP, Software Engineering]
author: Hyacehila
excerpt: "以注册表单为主线，串起事件、输入校验、Promise 与 async/await、HTTP 请求和 Fetch，理解浏览器如何等待并展示远程结果，再通过存储、登录状态和 F12 排查建立前后端边界的基本认识。"
excerpt_en: "Follow a registration form through events, input validation, Promise and async/await, HTTP, and Fetch. Understand how browsers wait for remote results and update the page, then use storage, login state, and developer tools to identify where problems occur."
description: "以注册表单为主线，串起事件、输入校验、Promise 与 async/await、HTTP 请求和 Fetch，理解浏览器如何等待并展示远程结果，再通过存储、登录状态和 F12 排查建立前后端边界的基本认识。"
mathjax: false
hidden: true
permalink: '/blog/2026/09/20/browser-interactions-events-forms-and-http/'
---

## 从一次注册看浏览器里的分工

一个注册页面看起来只是几个输入框和一个按钮，实际却连接了几套机制：HTML 和 CSS 提供内容与样式，DOM 表示运行中的页面，事件把用户操作交给 JavaScript，表单组织输入，异步代码等待结果，HTTP 则承载前后端之间的请求与响应。

前面的[《Web 前端基础概述》](/blog/2025/05/15/web-frontend-basics-overview/)、[《JavaScript 基础：变量、函数、数组与事件回调》](/blog/2026/09/18/javascript-basics-variables-functions-arrays-callbacks/)和[《DOM 基础：文档树、元素操作与动态页面》](/blog/2026/09/17/dom-basics-document-tree-and-element-operations/)已经分别整理了语法与页面操作。这里继续把它们放回一个交互过程，先看浏览器内部怎样响应操作，再看数据怎样离开页面、经过后端处理并返回。

我希望建立的是一种能够解释问题的理解：点击之后代码有没有执行，数据有没有收集对，请求有没有发出去，服务器又返回了什么。读懂这些环节，才更容易判断 AI 生成的代码是否完成了需求，并把实际遇到的问题描述清楚。下面只讲到浏览器与 HTTP 接口这一层，后端内部如何实现暂时作为另一个系统看待。

## Event：浏览器如何把操作交给代码

DOM 让 JavaScript 能够修改页面，事件系统则决定这些修改什么时候发生。用户可能刚打开页面就点击按钮，也可能过了很久才输入文字。代码可以先告诉浏览器“发生这件事时，请运行这个函数”，然后继续执行其他工作。之后事件触发，浏览器再调用对应的处理函数。这就是事件驱动的基本思想。

### 注册监听与事件对象

注册监听的常见写法是 `element.addEventListener("click", handleClick)`：在这个元素上监听 `click` 事件，由 `handleClick` 处理。这里传入的是函数本身，不能随手写成 `handleClick()`，否则会立即执行函数，再把它的返回值传进去。箭头函数 `event => { ... }` 也是同样的道理，只是把处理逻辑直接写在了注册的位置。

注册完成不意味着函数马上执行，也不会让程序停在那里一直等待。监听器通常会持续生效，后续每次符合条件的事件都会调用它。事件处理函数也不会因此自动变成后台线程，长时间的同步计算仍可能让页面交互卡住。

浏览器调用处理函数时，会传入一个 `event` 对象，告诉代码这次发生了什么。`event.type` 是事件类型，`event.target` 是事件的目标，`event.currentTarget` 是当前正在执行监听器的对象。参数名可以写成 `event` 或 `e`，它只是接收这个对象的变量名。

现阶段认得几种常见事件就够了：`click` 用于按钮等元素的点击或激活，`input` 用于响应用户编辑输入内容，`submit` 用于处理表单提交，`keydown` 用于处理按键。选择事件时，先想清楚自己关心的是“按钮被点击”“输入内容改变”，还是“用户提交了表单”。

### 为什么点了按钮，外层列表也能收到事件

事件可以沿着页面的层级传播。以普通 DOM 中的 `click` 为例，它的传播过程可以简化为：

```text
捕获阶段：从外层祖先沿路径向目标传递
目标阶段：执行目标上的监听器
冒泡阶段：从目标沿路径向外层祖先传递
```

比如按钮放在 `li` 中，`li` 又放在 `ul` 中。按钮和列表都注册了普通 `click` 监听器时，点击按钮会先执行按钮上的处理函数，再通过冒泡执行列表上的处理函数。一次操作可以被不同层级处理，不一定是用户点了两次。

`addEventListener` 默认不使用捕获监听，所以祖先上的监听器通常在冒泡阶段处理事件；目标自身的监听器在目标阶段执行。传入 `{ capture: true }` 可以监听捕获阶段，先知道有这一区别即可。也不是所有事件都会冒泡，不能把 `click` 的表现直接套到每一种事件上。

在列表的监听器中，`event.currentTarget` 是列表，而 `event.target` 可能是里面的按钮。如果按钮中还有一个图标，点击图标时目标还可能是那个内部元素。因此，“监听器挂在哪里”和“这次具体点到了哪里”是两个不同的问题。

这个区别可以用来做**事件委托**：只在列表上注册一个监听器，再根据事件目标判断用户点击了哪一项。之后新增的列表项，只要事件能冒泡到这个列表，同一个监听器就能处理，不必每增加一项都重新绑定。

### 默认行为与事件传播是两回事

有些操作自带浏览器行为，例如点击链接会导航，提交表单通常会按表单配置发送请求并导航。注册一个监听器，并不会自动取消这些行为。如果要由 JavaScript 接管表单提交，可以在 `submit` 处理函数里调用 `event.preventDefault()`，取消这次可取消事件的默认行为，再执行自己的逻辑。

`event.stopPropagation()` 处理的是另一个问题：阻止事件沿传播路径继续传递，既可能用于捕获阶段，也可能用于冒泡阶段。它不等于取消链接跳转或表单提交；反过来，`preventDefault()` 也不会自动阻止冒泡。事件委托依赖传播，因此不需要为了“防止重复”就在每个处理函数里加 `stopPropagation()`，先确认到底是哪一层不该接收事件。

### 一个例子：添加和删除用户

把下面的片段放进 HTML 页面的 `body` 中即可尝试。它沿用前面学过的 DOM 操作，只增加两个监听器：表单负责添加用户，列表负责处理删除。

```html
<form id="user-form">
    <label for="username">用户名</label>
    <input id="username" required>
    <button type="submit">添加用户</button>
</form>

<ul id="users"></ul>

<script>
    const form = document.querySelector("#user-form");
    const input = document.querySelector("#username");
    const list = document.querySelector("#users");

    form.addEventListener("submit", event => {
        event.preventDefault();

        const username = input.value.trim();
        if (!username) return;

        const item = document.createElement("li");
        const name = document.createElement("span");
        name.textContent = username;

        const button = document.createElement("button");
        button.type = "button";
        button.classList.add("delete-user");
        button.textContent = "删除";

        item.append(name, button);
        list.append(item);
        input.value = "";
    });

    list.addEventListener("click", event => {
        if (!(event.target instanceof Element)) return;

        const button = event.target.closest(".delete-user");
        if (!button || !list.contains(button)) return;

        const item = button.closest("li");
        if (item) item.remove();
    });
</script>
```

输入 Alice 后点击添加，表单会触发 `submit`，处理函数读取输入框、创建节点并插入列表。监听表单提交，也能覆盖这个简单表单中按 Enter 提交的情况。`required` 提供浏览器的非空检查，`trim()` 再排除只有空格的输入；姓名通过 `textContent` 写入，作为普通文字显示。

点击删除按钮时，`click` 冒泡到列表，由列表的处理函数统一处理。`closest(".delete-user")` 从目标元素自身开始向上寻找删除按钮，所以以后按钮里加入图标，也能找到它所属的按钮。代码先确认目标是元素、按钮位于当前列表内，再找到对应的 `li` 并移除。点击姓名或列表空白处不会执行删除。

这段代码没有单独维护用户数组，添加和删除只改变当前 DOM，刷新后列表会恢复为空。若要保留用户数据，还需要存储或后端接口，监听器不会自动替我们完成这些工作。

## Form：把用户输入整理成可以提交的数据

上一节的“添加用户”已经用到了表单：输入框收集姓名，`submit` 事件把处理过程交给 JavaScript。把它扩展到注册、搜索或修改资料，基本过程仍然相同：收集一组相关输入，检查是否符合要求，再决定如何提交，以及向用户显示什么结果。

### 表单如何组织输入

`form` 把输入控件组织成一次提交，`input` 等控件保存当前输入，`button type="submit"` 发起提交。监听表单的 `submit`，可以统一处理点击提交按钮和表单支持的 Enter 提交，而不必分别给两种操作写一份业务逻辑。

一个输入框常常同时有 `id` 和 `name`，它们的用途不同。`id` 用于定位元素，也可以通过 `label for="..."` 将文字说明与输入框关联；`name` 则规定收集表单数据时的字段名。例如，`id="register-username" name="username"` 表示页面上通过 `#register-username` 找到它，整理数据时则把它放在 `username` 字段下。输入框能显示、能输入，不代表它一定会被收集；缺少 `name` 就是字段丢失的一种常见原因。

`type` 决定控件的输入方式和部分检查规则，例如 `email` 可以检查基本邮箱格式，`password` 遮住屏幕上的密码字符。文字输入通常读取 `value`，复选框是否选中则看 `checked`。这些输入方式和基础检查主要用于帮助用户填写。接入真实接口后，后端仍需要独立校验，因为请求也可以绕过页面直接发出。

### 从输入到校验：一个注册表单

另建一个测试页面，把下面的片段放进 `body` 中即可尝试。这个例子还没有后端，只演示收集数据和本地校验，不会真正注册账号。

```html
<form id="register-form">
    <div>
        <label for="register-username">用户名</label>
        <input id="register-username" name="username" required>
    </div>
    <div>
        <label for="register-email">邮箱</label>
        <input id="register-email" name="email" type="email" required>
    </div>
    <div>
        <label for="register-password">密码</label>
        <input id="register-password" name="password" type="password"
               autocomplete="new-password" required>
    </div>
    <label>
        <input name="agree" type="checkbox" value="yes" required>
        同意条款
    </label>
    <button type="submit">检查注册信息</button>
</form>
<p id="register-message" role="status"></p>

<script>
    const registerForm = document.querySelector("#register-form");
    const message = document.querySelector("#register-message");

    registerForm.addEventListener("input", () => {
        message.textContent = "";
    });

    registerForm.addEventListener("submit", event => {
        event.preventDefault();
        message.textContent = "";

        const formData = new FormData(registerForm);
        const data = Object.fromEntries(formData.entries());
        data.username = data.username.trim();

        if (data.username.length < 3) {
            message.textContent = "用户名去掉首尾空格后至少需要 3 个字符。";
            return;
        }
        if (data.password.length < 8) {
            message.textContent = "本例要求密码至少 8 个字符。";
            return;
        }

        message.textContent =
            data.username + " 的信息已通过本地校验，尚未发送给服务器。";
    });
</script>
```

这里有两层前端检查。浏览器先根据 HTML 的 `required` 和 `type="email"` 检查必填项、邮箱格式，以及是否勾选条款；按这个例子的正常提交方式，如果这些检查不通过，就会提示用户，`submit` 处理函数也不会执行。

通过浏览器检查后，JavaScript 再收集数据，检查去掉首尾空格后的用户名长度和密码长度。这两条只是本例选择的演示规则。用户名做了 `trim()`，密码则保留原始输入，不随意删除用户输入的字符。检查不通过时，代码显示原因并 `return`；通过时，也只说明“本地校验通过”，不能据此判断账号已经注册成功。

上面的 `input` 监听器只负责在用户继续编辑时清除旧提示，真正的检查仍放在 `submit` 中。实时搜索等场景可以用 `input` 响应内容变化；文本框的 `change` 通常在值改变并失去焦点时触发，`focus` 和 `blur` 则表示进入和离开输入框。这里不需要给所有事件都绑定处理函数，按需要选择反馈时机即可。

### FormData 如何把控件变成字段

`new FormData(registerForm)` 会按控件的 `name` 收集当时可提交的数据。例如，`formData.get("username")` 读取用户名，`formData.get("email")` 读取邮箱。它是一份收集时的数据快照，用户随后继续编辑，不会自动修改已经创建的这份 `FormData`。

本例的字段名各不相同，所以用 `Object.fromEntries(formData.entries())` 把字段和值转换成普通对象，后面就能通过 `data.username`、`data.email` 访问。到了前后端通信阶段，这些字段还需要与接口约定一致。

这里先记住几个能帮助排查问题的规则。没有 `name` 或被 `disabled` 禁用的控件通常不会进入这份数据；未勾选的复选框也不会提交对应字段。例子里的 `agree` 被勾选后得到字符串 `"yes"`，并不是布尔值 `true`。普通输入字段收集到的通常是字符串，数字输入框也不意味着 `FormData` 会自动给出数字类型。若以后遇到同名多选字段，需要考虑 `getAll()`，不能直接套用这里转成单值对象的做法。

### 提交以后，才开始连接异步与后端

表单本身可以在没有 JavaScript 的情况下提交。`action` 指定目标地址，`method` 指定提交方式，浏览器收集字段后发送请求，通常再导航到响应页面。我们调用 `preventDefault()`，只是取消这次默认提交；它不会自动发请求，也不会自动保存数据。上面的例子执行到显示提示就结束了。

之后接入接口时，处理函数还需要承担后半段工作：

```text
用户填写并提交
    ↓
浏览器基础检查 → submit 处理函数 → 业务校验与数据整理
    ↓
显示“提交中”，发出请求
    ↓
等待后端检查、处理并返回结果
    ↓
成功：显示结果或跳转
失败：保留必要输入并说明原因，允许修改或重试
```

等待请求期间，页面还需要管理“提交中、成功、失败”等状态，例如避免用户反复点击造成重复请求。网络失败和后端明确拒绝请求也不是同一回事，不能只要收到了响应就显示成功。接下来先看这段等待怎样发生，再把表单接到一个明确约定的接口上。

## 异步：等待服务器时，页面为什么还能响应

### 等待和计算是两种不同的事情

页面里的 JavaScript 通常在主线程上执行，调用栈记录当前进入了哪些函数、执行到哪里。一个函数调用另一个函数，就要先完成里面的调用，再返回外层继续执行。这个阶段如果一直进行耗时计算，用户输入和页面更新就可能被拖住。

网络请求却有大量时间花在等待。浏览器可以负责发送请求、接收数据，JavaScript 不必用一个循环守着结果。代码先发起操作，再约定结果可用以后如何继续。异步减少的是这种等待对执行流程的占用。

用计时器可以先模拟这种等待。下面没有发出网络请求，只是让结果稍后出现：

```javascript
function waitForReply() {
    return new Promise(resolve => {
        setTimeout(() => resolve("模拟结果已到达"), 800);
    });
}

async function run() {
    console.log("开始等待");
    const result = await waitForReply();
    console.log(result);
}

run();
console.log("外面的代码继续执行");
```

输出先是“开始等待”，然后是“外面的代码继续执行”，最后才是“模拟结果已到达”。`await` 暂停的是 `run` 中后面的那段流程，外面的代码仍然能继续。计时器的 `800` 毫秒表示请求的延迟，实际什么时候执行还受调度影响，并不是精确的完成时间。

### Promise、await 与事件循环怎样配合

Promise 表示一个操作的结果，开始可能处于 `pending`，之后变成 `fulfilled` 或 `rejected`，也就是成功或失败。成功后可以用 `then` 接着处理，失败则需要相应的错误处理。创建 Promise 时传入的函数本身会立即执行，上例真正延后的是计时器的回调；给一段耗时计算套上 Promise，并不能让它不阻塞页面。

`async` 函数总会返回 Promise，`await` 用于等到结果可用后继续当前函数。若等待的 Promise 失败，异常会在 `await` 处抛出，可以通过 `try/catch` 接住。函数被调用以后，遇到第一个 `await` 之前的代码仍按同步方式执行。它是一种组织异步流程的写法，不是“开启后台任务”的开关。

结果准备好了，也不能随意打断正在执行的同步代码。事件循环负责安排后续工作，其中需要先认识两类：计时器等产生的任务，以及 Promise 后续处理所使用的微任务。看一个小例子：

```javascript
console.log("同步开始");

setTimeout(() => console.log("计时器回调"), 0);
Promise.resolve().then(() => console.log("Promise 后续"));

console.log("同步结束");
```

这段代码的输出顺序是“同步开始 → 同步结束 → Promise 后续 → 计时器回调”。当前这段同步代码先完成，已经排入队列的微任务随后执行，再轮到后面的任务。这里的 Promise 已经兑现，所以它的后续可以先排入微任务队列。

可以把常见情况概括为“执行当前任务 → 处理微任务 → 浏览器在合适时机更新画面 → 继续其他任务”。浏览器不保证每完成一个任务都绘制一帧，连续的同步工作或大量微任务也可能推迟页面响应。理解到这一层，已经足以解释为什么 `await` 网络请求通常允许页面继续交互，而一个很长的循环仍会卡住页面。

## HTTP：前端和后端交换什么

### 一次请求与响应的结构

回到注册表单，JavaScript 现在能拿到用户名、邮箱和密码，但这些数据还只存在于浏览器中。要让后端处理它们，就需要发送请求。HTTP 为请求和响应约定了结构；这里先关注目标地址、操作方式、附加说明和数据内容。

假设后端提供 `POST /api/register`，接受 JSON，成功时返回新用户的公开信息。下面用 HTTP/1.1 的文本形式示意一次交换，省略一些头部，字段值仅用于说明：

```http
POST /api/register HTTP/1.1
Host: example.com
Content-Type: application/json

{"username":"Alice","email":"alice@example.com","password":"<测试输入>","agree":"yes"}
```

```http
HTTP/1.1 201 Created
Content-Type: application/json

{"user":{"id":123,"username":"Alice"}}
```

请求中的 `POST` 是方法，`/api/register` 是路径，`Content-Type` 说明请求体使用 JSON 格式，空行之后才是具体数据。响应则用状态码说明处理结果，并返回自己的头部和响应体。JSON 只是双方约定的一种数据格式，HTTP 也可以传递 HTML、图片或文件。

`/api/register` 需要后端实际实现。前端写出这个地址，并不会自动生成注册功能，也不会自动连接数据库。后端负责解析请求、重新校验、执行注册逻辑，再返回响应；它内部是否访问数据库或其他服务，暂时可以看成方框里的实现。

### 方法、地址与状态码怎样帮助判断问题

GET 通常用来获取资源，POST 用来提交数据或执行某个动作。之后还会看到 PUT、PATCH、DELETE，先知道它们常用于更新和删除即可，具体行为始终要以接口约定为准。

地址也有分工：`/api/users/123` 中的 `123` 常用于指定某个用户，`/api/users?keyword=alice&page=2` 则用查询参数表达筛选和分页条件。站点内的 `/api/register` 会相对于当前页面的源解析，页面在本地开发服务器上时，它并不会自动指向另一个端口上的后端；这时需要正确的接口地址或开发代理。

状态码不需要一次记完，可以先按下面几类定位：

| 状态 | 初步理解 | 接下来检查什么 |
| --- | --- | --- |
| 200、201 等 2xx | 请求按 HTTP 层面的含义成功 | 响应内容是否符合业务预期 |
| 3xx | 重定向或缓存相关处理 | 最终请求去了哪里，是否使用了缓存 |
| 400、422 等 | 请求数据或语义不符合要求 | 字段名、类型和校验反馈 |
| 401 | 缺少有效身份认证 | 登录凭据是否有效、是否随请求发送 |
| 403 | 服务器拒绝访问 | 权限或访问策略；不一定代表已经确认了身份 |
| 404 | 没找到相应资源或路由 | 地址、路径和后端路由 |
| 5xx | 服务器处理发生问题 | 响应说明与后端日志 |

状态码是一条线索，不能独自说明全部原因。有些接口还在 JSON 中约定业务状态，200 也不能直接等同于“注册一定成功”。反过来，204 表示成功但没有响应体，就不应该继续按非空 JSON 解析。

这里不继续展开底层网络传输。

## Fetch：让表单真正接上请求与结果

### 发起请求、读取响应和处理失败

`fetch` 是 JavaScript 调用浏览器请求能力的入口。它返回 Promise；`await fetch(...)` 得到的是 `Response` 对象，里面有状态和头部信息，还不是已经解析好的业务数据。

如果接口约定返回 JSON，就继续 `await response.json()`，读取响应体并解析成 JavaScript 对象。这也需要等待，因为拿到响应的状态和头部时，响应体未必已经全部读完。请求体方向则相反：通过 `JSON.stringify(data)` 把对象转换成 JSON 文本，并用 `Content-Type: application/json` 说明格式。

有一个容易影响错误处理的区别：404、500 等 HTTP 响应通常不会让 `fetch` 自己抛出异常，需要检查 `response.ok`；网络故障、请求取消或浏览器策略限制等才可能让 `fetch` 拒绝。`response.json()` 也可能因内容不是合法 JSON 而失败。这几种情况都需要处理，但原因并不相同。

### 用同一张表单观察完整过程

保留 Form 一节的表单 HTML 和 `register-message` 段落，用下面的代码**替换原来的整段 `<script>`**，不要把新旧两个提交监听器同时保留。这个版本默认开启模拟模式，能直接体验等待、成功提示和失败提示，不会发送请求或创建账号。

真实模式需要先由后端实现下面的约定：`POST /api/register` 接受前面那几个字段；成功时返回带有 `user.username` 的 JSON；不成功时使用相应的 HTTP 错误状态。准备好接口后，再将 `demoMode` 改成 `false`，并通过 HTTP 开发服务器打开页面，而不是把本地文件路径当作接口地址。

```html
<script>
    const registerForm = document.querySelector("#register-form");
    const message = document.querySelector("#register-message");
    const submitButton = registerForm.querySelector('[type="submit"]');
    const demoMode = true;
    const defaultLabel = demoMode ? "模拟提交" : "提交注册";
    submitButton.textContent = defaultLabel;
    let submitting = false;

    async function requestRegistration(data) {
        if (demoMode) {
            await new Promise(resolve => setTimeout(resolve, 800));
            if (data.username.toLowerCase() === "taken") {
                throw new Error("模拟失败：用户名已存在，请换一个用户名。");
            }
            return { user: { username: data.username } };
        }

        const response = await fetch("/api/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error("服务器返回 HTTP " + response.status);
        }

        const result = await response.json();
        if (!result || !result.user ||
            typeof result.user.username !== "string") {
            throw new Error("响应数据与约定不一致：缺少 user.username。");
        }
        return result;
    }

    registerForm.addEventListener("input", () => {
        if (!submitting) message.textContent = "";
    });

    registerForm.addEventListener("submit", async event => {
        event.preventDefault();
        if (submitting) return;

        const data = Object.fromEntries(new FormData(registerForm).entries());
        data.username = data.username.trim();

        if (data.username.length < 3 || data.password.length < 8) {
            message.textContent = "用户名至少 3 个字符，密码至少 8 个字符。";
            return;
        }

        submitting = true;
        submitButton.disabled = true;
        submitButton.textContent = "提交中…";
        message.textContent = demoMode ? "正在模拟等待…" : "正在提交…";

        try {
            const result = await requestRegistration(data);
            message.textContent = demoMode
                ? "模拟完成：" + result.user.username + "；没有创建真实账号。"
                : "注册成功：" + result.user.username;
        } catch (error) {
            message.textContent = "本次操作未完成：" + error.message;
        } finally {
            submitting = false;
            submitButton.disabled = false;
            submitButton.textContent = defaultLabel;
        }
    });
</script>
```

填写符合规则的内容后提交，按钮会暂时禁用并显示“提交中”。模拟模式下，大约等待一小段时间后会显示模拟完成；把用户名填成 `taken`，则会进入失败分支。输入框仍然可以编辑，但这次处理使用的是提交时收集的 `data`，不是后来改动的内容。

`submitting` 记录当前是否已有操作在等待，避免这一张页面里重复发起同一轮提交；禁用按钮则把这个状态显示给用户。`try` 处理正常流程，`catch` 显示失败，`finally` 不论成功或失败都会恢复按钮。只在成功分支恢复按钮，会让一次失败之后的页面一直停在“提交中”。

这个示例的模拟分支没有调用 `fetch`，所以 Network 面板中不会出现注册请求。切换真实模式后才会发生 HTTP 通信。真实网络里，请求报错也不一定说明服务器什么都没做，例如服务端处理完后响应丢失；涉及支付等重要操作时，重复提交还需要后端配合防重，不能只靠一个禁用按钮。

对于搜索框，还会出现另一种顺序问题：用户先搜 Alice，再搜 Bob，但 Alice 的响应晚回来，覆盖了 Bob 的结果。代码执行没有报错，页面却显示了过时数据。这里先记住“请求发出的顺序不保证等于完成的顺序”，后续可以通过请求编号或取消旧请求，只采用仍然有效的结果。

## 浏览器边界：存储与登录状态

### 为什么刷新后，有些信息还在

DOM 和普通 JavaScript 变量属于当前页面运行时，刷新后会重新建立。如果信息还能恢复，通常是因为它来自浏览器存储，或者页面重新从后端读取了数据。

`localStorage` 可以跨刷新保存一些本地数据，`sessionStorage` 则属于当前标签页的会话范围。它们不会像请求参数一样自动发送给后端，存进去也不等于已经保存到服务器。页面偏好、尚未提交的草稿和后端账号记录，分别保存在哪里，需要由应用决定。

登录状态还涉及身份凭据。Cookie 是浏览器保存并按规则随请求发送的数据载体；Session 通常指服务端维护的会话状态，Cookie 中可以只放一个会话标识；Token 则是供服务端验证的凭据，也可以通过 Cookie 或请求头携带。它们不是三套完全互斥的选项。某些 Cookie 设置了 `HttpOnly`，页面脚本读不到，但浏览器仍可按规则将它随请求发送。

所以，“页面显示已经登录”与“服务器认可这次请求的身份”也是两个问题。仅在本地保存 `isLoggedIn = true`，不能获得后端权限。若刷新后登录丢失，或跨源请求突然得到 401，可以沿着“凭据保存在哪里 → 请求是否携带 → 后端是否仍认可”检查，而不是只修改页面上的登录提示。

## 用 F12 把问题定位到具体一层

现在可以把一次注册完整地读出来：用户输入产生表单状态，提交事件触发处理函数，JavaScript 收集并校验数据，通过 Fetch 发出 HTTP 请求；等待后端处理以后，读取响应，更新提示和按钮状态。

调试时，可以把浏览器的三个面板对应到这条流程：

| 面板 | 主要观察什么 | 一个具体问题 |
| --- | --- | --- |
| Elements | 当前 DOM、控件和样式 | 错误提示是否已经写进页面，只是被样式隐藏了？ |
| Console | 执行错误、断点配合的变量观察 | 选择器是否返回了 null，响应字段是否不存在？ |
| Network | 请求地址、方法、请求数据、状态和响应 | 表单数据究竟发到了哪里，后端返回的真的是 JSON 吗？ |

可以从最靠近现象的地方向前追。如果根本没有请求，先确认是否仍在模拟模式、原生校验是否放行、监听器是否执行、代码是否走到 `fetch`。如果请求已经发出，查看 URL、方法和 Payload，再看状态码与 Response；请求到了错误的页面，返回的可能是 HTML，后续就会在 JSON 解析处报错。如果响应符合约定但界面不对，再检查字段读取、状态更新以及 DOM 操作。

这篇文章到这里完成的是浏览器侧的一次交互流程。再往后，模块化、构建工具和框架解决代码如何组织，后端、数据库和部署解决服务如何实现与运行，可以分别展开。当前先做到：面对一个交互问题，能够指出它发生在事件、表单、等待、请求、后端响应还是页面更新这一层。
