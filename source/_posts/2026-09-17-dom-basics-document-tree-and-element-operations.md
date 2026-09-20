---
title: "DOM 基础：文档树、元素操作与动态页面"
title_en: "DOM Basics: Document Trees, Element Operations, and Dynamic Pages"
date: 2026-09-18 12:00:00 +0800
categories: ["Programming", "Full Stack Development"]
tags: [DOM, HTML, JavaScript, Frontend]
author: Hyacehila
excerpt: "从 HTML 如何形成 DOM 树讲起，理解 document、window、Node 与 Element，再通过用户列表示例学习元素查找、内容与属性修改、节点创建和删除，以及数据与页面之间的关系。"
excerpt_en: "Understand how HTML becomes a DOM tree and how document, window, Node, and Element relate. Use a user list to explore element queries, content and attribute updates, node creation and removal, and the relationship between data and the page."
description: "从 HTML 如何形成 DOM 树讲起，理解 document、window、Node 与 Element，再通过用户列表示例学习元素查找、内容与属性修改、节点创建和删除，以及数据与页面之间的关系。"
mathjax: false
hidden: true
permalink: '/blog/2026/09/17/dom-basics-document-tree-and-element-operations/'
---

这篇文章衔接[《Web 前端基础概述》](/blog/2025/05/15/web-frontend-basics-overview/)和[《JavaScript 基础：变量、函数、数组与事件回调》](/blog/2026/09/18/javascript-basics-variables-functions-arrays-callbacks/)，从文档树讲到元素的查找、修改、创建和删除，并用一个用户列表串起 DOM 的基本操作。

## 为什么需要 DOM

HTML 文件保存的是描述页面的文本字符串，但程序操作页面时，需要找到具体的元素，知道它们之间的关系，并读取或修改其中的内容。浏览器解析 HTML 时，会把文档组织成内存中的节点树，其中包括元素节点、文本节点等。这就是 DOM（Document Object Model，文档对象模型）：它把 HTML 描述的文档结构变成了程序可以访问和操作的对象。例如，一个列表包含哪些列表项、某个按钮位于哪个容器中，都可以通过这棵树来确定。

有了 DOM，JavaScript 就能通过统一的接口找到并修改页面中的对象。比如，用户点击“添加任务”后，代码可以创建一个新的列表项，把它加入现有列表；修改标题时，可以找到对应元素并更新文字。浏览器再根据更新后的文档结构和样式呈现页面，我们不必每次都重新拼接整份 HTML 或刷新网页。DOM 因而连接了页面结构与交互代码，让前面学到的变量、函数和事件回调能够作用于具体的页面内容。

参考我之前做的游戏 UI 系统，HTML 类似原本的工程文件，在编辑器打开工程以后会解析为类似 DOM 树的形式，只是采用私有编码，并允许编辑器的工具结构去修改他们并保存得到新的文件。如果要考虑程序逻辑的相关接入，那么游戏 UI 其实和 WebUI 一样，都是先完成版式设计，在进行逻辑的接入，只是前者实现了岗位分离并且要处理更多游戏引擎内逻辑。

## DOM 到底是什么

先看一个简单的用户列表页面。把下面的内容保存为 UTF-8 编码的 HTML 文件，再用浏览器打开，就能看到标题、两个用户名和一个按钮：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>DOM Demo</title>
</head>
<body>
    <h1 id="title">用户列表</h1>

    <div class="users">
        <p>Alice</p>
        <p>Bob</p>
    </div>

    <button id="add-btn">添加用户</button>
</body>
</html>
```

我们在文件里写下的是 HTML 文本。浏览器读取它时，会通过 HTML 解析器识别标签、属性和文字，并建立对应的 DOM 树。这里只关注文档结构的形成，可以把过程写成 `HTML 文本 → HTML 解析器 → DOM 树`。上面这份文档得到的结构大致如下，图中省略了缩进和换行产生的空白文本节点：

```text
Document
├── DocumentType: html
└── html
    ├── head
    │   ├── meta
    │   └── title
    │       └── #text "DOM Demo"
    └── body
        ├── h1#title
        │   └── #text "用户列表"
        ├── div.users
        │   ├── p
        │   │   └── #text "Alice"
        │   └── p
        │       └── #text "Bob"
        └── button#add-btn
            └── #text "添加用户"
```

`Document` 是整份文档的节点，`html`、`body`、`p` 等是元素节点，元素内部的文字则由文本节点表示。例如，第一个 `p` 元素包含一个内容为 `Alice` 的文本节点；两个 `p` 都是 `div` 的子节点。图中的 `h1#title` 和 `div.users` 只是为了方便辨认而附上了 `id` 或类名，这些属性不是图中额外的子节点。

从 JavaScript 一侧看，树上的节点都是可以访问的对象。例如，页面中的按钮对应一个 `HTMLButtonElement` 对象，它暴露了用于读取或修改按钮的属性，以及可以调用的方法：

```text
HTMLButtonElement（这个按钮可访问的部分属性和方法）
├── id          → "add-btn"
├── textContent → "添加用户"
├── disabled    → false
├── style       → 读取或修改内联样式
├── classList   → 读取或修改类名
└── click()     → 以程序方式触发点击
```

打开这个页面的开发者工具，在 Console 中运行下面的代码，就可以取得并修改这个按钮：

```javascript
const button = document.querySelector("#add-btn");

console.log(button instanceof HTMLButtonElement); // true
console.log(button.textContent);                  // "添加用户"

button.textContent = "暂时无法添加";
button.disabled = true;
```

`document.querySelector("#add-btn")` 会在 DOM 中找到第一个符合选择器的元素，并返回这个元素对象；找不到时会返回 `null`。这里的 `button` 就指向页面中现有的按钮，因此修改它的 `textContent` 和 `disabled` 后，页面上的文字会改变，按钮也会变为不可用。这个按钮示例还没有绑定添加用户的业务逻辑，点击它并不会自动增加用户。

所以，查询按钮时，JavaScript 操作的是浏览器已经建立的文档对象，而不是在 HTML 字符串里搜索和替换文字。DOM 保留了文档的层级关系，又提供了可以操作这些节点的接口。之后学习查找元素、添加节点和监听事件，都是在这套对象结构上继续展开。

## document 与 window，网页和它的运行环境

上一节用 `document.querySelector(...)` 找到了按钮，其中的 `document` 就是当前网页对应的文档对象，也是我们访问这份 DOM 的入口。打开前面的 DOM Demo，在 F12 开发者工具的 Console 中输入 `document`，通常能看到可以展开查看的文档结构。它代表整份文档，包括 `head` 中的页面信息和 `body` 中的页面内容。

例如，文档的标题可以通过 `document.title` 读取和修改：

```javascript
console.log(document.title); // "DOM Demo"

document.title = "新的标题";
```

执行后，浏览器标签页上的标题会变成“新的标题”，文档中 `<title>` 的文字也会随之更新。不过，页面里 `<h1>` 显示的“用户列表”仍然保留，因为它是另一个元素。

浏览器还提供了一个经常出现的对象：`window`。可以先把它理解为当前页面运行环境的总入口。在普通网页脚本中，它是浏览器提供的全局对象，通过它既能访问文档，也能使用与地址、导航、存储和计时等有关的功能：

```text
window（当前页面运行环境的入口）
├── document     → 当前页面的文档对象
├── location     → 当前地址与页面跳转
├── history      → 当前浏览上下文的会话历史
├── navigator    → 浏览器、设备及部分能力的访问入口
├── localStorage → 按源隔离的本地存储
├── console      → 控制台输出与调试
└── setTimeout   → 注册延后执行的回调
```

这表示通过 `window` 可以访问哪些属性和方法，并不是另一棵 DOM 树；`window` 本身也不是 `document` 的 DOM 父节点。它对应当前的窗口或框架环境，不代表整个浏览器程序。页面中嵌入的 `iframe` 也有自己的 `window` 和 `document`。

因此，`window` 和 `document` 的分工可以按操作对象来理解：修改页面中的文字、按钮和列表，主要从 `document` 进入；查看当前 URL、安排计时任务等，则会用到 `window` 提供的其他能力。对于前面的示例，可以在 Console 中继续观察：

```javascript
console.log(window.document === document); // true
console.log(window.location.href);         // 当前页面的完整 URL

console.log(document.documentElement);     // <html> 元素
console.log(document.head);                // <head> 元素
console.log(document.body);                // <body> 元素
```

平时直接写 `document`、`console.log(...)` 或 `setTimeout(...)`，是因为这些名字在普通网页环境中可以直接访问，常常省略了前面的 `window.`。它们由运行环境提供，并不意味着 JavaScript 语言在所有地方都自带网页和浏览器。例如，在 Node.js 中运行 JavaScript 时，就没有浏览器默认提供的 `window` 和 `document`。

接下来学习 DOM 时，可以先沿着 `window.document → document.documentElement → head / body → 具体元素` 理解访问路径：`window.document` 取得文档，`document.documentElement` 取得它的根元素 `<html>`，再往下才是页面里的各个节点。实际写代码时，通常直接从 `document` 开始查询需要的元素即可。

## Node 和 Element，节点与元素有什么区别

前面画 DOM 树时，树上既有 `div`、`button`，也有文字和整份文档。它们都属于节点，英文叫 **Node**。其中，代表 HTML 元素的节点叫 **Element**。因此，一个按钮既是 Element，也是 Node；一段文字也是 Node，但它属于 Text，不是 Element。

可以先按类型把常见节点分成几类：

```text
Node（节点的共同基础类型）
├── Element       元素节点，例如 div、p、button
├── Text          文本节点，例如 "Hello"
├── Comment       注释节点，例如 <!-- 备注 -->
├── Document      文档节点，例如 document
└── DocumentType  文档类型节点，例如 <!DOCTYPE html>
```

这些节点有共同的基础能力，比如访问自己的 `parentNode`、查看 `childNodes`，但它们的具体用途不同。元素可以有 `id`、`class` 等属性，文本节点负责保存文字，注释节点保存注释内容。

看这段 HTML：

```html
<p>Hello</p>
```

只看这个片段，浏览器会建立一个 `p` 元素节点，并在它下面放一个文本节点：

```text
p（Element）
└── "Hello"（Text）
```

`<p>` 和 `</p>` 是 HTML 源码中标记元素边界的写法，不会分别变成两个元素节点。它们共同描述一个 `p` 元素；其中的 `Hello` 则是这个元素包含的文本内容。

把文字单独表示为节点，是因为一个元素内部可以混合出现文字和其他元素。例如，`<p>Hello <strong>Alice</strong>!</p>` 中，`p` 的直接子节点依次是文本 `"Hello "`、元素 `strong`、文本 `"!"`；`strong` 内部还包含文本 `"Alice"`。这样的结构既保留了内容顺序，也让浏览器知道只有 `Alice` 位于 `strong` 元素中。

如果为了排版，把 HTML 写成下面这样：

```html
<div>
    Hello
</div>
```

这个 `div` 中的文本节点实际包含 `"\n    Hello\n"`：开头是换行，接着是四个空格和 `Hello`，最后还有一个换行。这里的 `\n` 表示换行字符。页面在默认样式下通常会折叠这些空白，所以视觉上看起来仍然只是 `Hello`，但 DOM 中的文本并没有因此消失。

这也是第一节的 DOM 树需要注明“省略空白文本节点”的原因。只看页面效果，很容易以为一个容器的第一个子节点就是里面的按钮；实际检查时，第一个节点却可能是按钮前面的换行和空格。

把下面这一行放进之前 DOM Demo 的 `body` 中，再打开 Console。这里特意把内容写在一行，避免额外的换行和缩进干扰观察：

```html
<div id="node-demo">Hello<span>World</span><!--备注--></div>
```

它的结构是：

```text
div#node-demo（Element）
├── "Hello"（Text）
├── span（Element）
│   └── "World"（Text）
└── "备注"（Comment）
```

`div` 有三个直接子节点，其中只有 `span` 是元素节点。`World` 是 `span` 的子节点，是 `div` 的后代，但不是 `div` 的直接子节点。可以用下面的代码确认：

```javascript
const demo = document.querySelector("#node-demo");

console.log(demo.childNodes.length); // 3：文本、span 元素、注释
console.log(demo.children.length);   // 1：只有 span 元素

console.log(demo.firstChild.nodeName);        // "#text"
console.log(demo.firstElementChild.tagName);  // "SPAN"

console.log(demo instanceof Node);              // true
console.log(demo instanceof Element);           // true
console.log(demo.firstChild instanceof Node);   // true
console.log(demo.firstChild instanceof Element); // false
```

`childNodes` 包含所有类型的直接子节点，`children` 只包含直接子元素。同样，`firstChild` 取得第一个子节点，`firstElementChild` 则取得第一个子元素。代码里的 `instanceof` 用来检查对象是否属于某个类型，因此 `demo` 的两次检查都为 `true`，而文本节点只满足其中的 Node 检查。

也可以观察同一个元素的文字内容：

```javascript
console.log(demo.textContent); // "HelloWorld"
```

对于这个元素，`textContent` 会汇集后代文本节点的文字，不包含注释，也不会自动在 `Hello` 和 `World` 之间加空格。它读到的是这段子树里的文本，而不是给整个 `div` 额外附上了一个“总文本节点”。

后面修改按钮文字、切换样式、读取输入值时，经常操作的是 Element，因为这些工作针对的是具体元素。需要遍历完整文档结构或处理文字时，Node 与 Element 的区别就会直接影响结果。遇到 `childNodes`、`firstChild` 这类名字，先记得它们可能返回文本或注释；如果只关心元素，就选择 `children`、`firstElementChild` 或元素选择器。

## 找到元素，并修改它

前面已经用过 `querySelector`，现在把查找和修改连起来。下面换成一个稍完整的用户列表，把这段 HTML 放进测试页面的 `body` 中。后面的第四、第五节示例按顺序在这个页面的 Console 中运行；开始一轮新实验前刷新页面即可。

```html
<h1 id="list-title">用户列表</h1>
<label for="username">用户名</label>
<input id="username" value="Alice">

<ul id="user-list">
    <li class="user-card">
        <span class="user-name">Alice</span>
        <button type="button">删除</button>
    </li>
    <li class="user-card">
        <span class="user-name">Bob</span>
        <button type="button">删除</button>
    </li>
</ul>
```

`querySelector` 接收 CSS 选择器，返回第一个匹配的元素。`#list-title` 按 `id` 查找，`.user-card` 按类名查找，`button` 按标签名查找。选择器还可以组合，例如 `#user-list .user-name` 表示列表里面的用户名元素。也可以先取得一个容器，再在它内部查询：

```javascript
const listTitle = document.querySelector("#list-title");
const list = document.querySelector("#user-list");
const firstCard = list.querySelector(".user-card");
const firstName = firstCard.querySelector(".user-name");

console.log(firstName.textContent); // "Alice"

const cards = list.querySelectorAll(".user-card");
cards.forEach(card => {
    console.log(card.querySelector(".user-name").textContent);
}); // 依次输出 "Alice"、"Bob"
```

`querySelectorAll` 返回一组匹配元素，类型是 `NodeList`，可以用下标取其中一项，也可以用 `forEach` 遍历，但它不是普通数组。这次查询保存的是当时匹配到的元素集合，之后新增卡片，不会自动让 `cards` 多出一项，需要重新查询。

如果合法的选择器没有匹配到元素，`querySelector` 返回 `null`，`querySelectorAll` 返回长度为 `0` 的集合。遇到“无法读取 null 的属性”时，先检查选择器、查询范围和执行时机，确认元素真的已经存在，再继续操作。另一种常见写法是 `document.getElementById("list-title")`，它直接接收 `id`，前面不加 `#`。

取得元素对象以后，可以修改它的文字：

```javascript
listTitle.textContent = "员工列表";
firstName.textContent = "Alicia";
```

这里我们只修改了姓名所在的 `span`，旁边的删除按钮仍然存在。如果改成 `firstCard.textContent = "Alicia"`，整张卡片原来的子节点都会被替换，包括 `span` 和按钮，最后只剩文字。`textContent` 赋值的范围由你选中的元素决定，所以找到正确的操作对象，比记住赋值语法更重要。

输入框的当前内容则要通过 `value` 读取。下面可以看到 HTML 属性（Attribute）和对象属性（Property）之间的区别：

```javascript
const input = document.querySelector("#username");

console.log(input.getAttribute("value")); // "Alice"
console.log(input.value);                 // "Alice"

input.value = "Bob";

console.log(input.value);                 // "Bob"
console.log(input.getAttribute("value")); // 仍然是 "Alice"
```

HTML 中的 `value="Alice"` 是内容属性，提供这个输入框的默认值；`input.value` 是对象属性，表示当前值。用户在输入框中打字，也是在改变当前值。因此，提交表单时需要读取 `input.value`，不能用 `getAttribute("value")` 代替。不同属性有不同的对应规则，不能把所有 Attribute 都理解成永远不变的初始值。

内容属性本身也可以修改。例如，`firstCard.setAttribute("title", "用户资料")` 设置鼠标悬停时的提示文字，`getAttribute("title")` 读取它，`hasAttribute("title")` 检查它是否存在，`removeAttribute("title")` 删除它。而输入框是否可用这类状态，可以直接用 `input.disabled = true` 这样的对象属性来设置。

改变外观时，可以先在页面的 `head` 中加入样式：

```html
<style>
    .user-card { padding: 8px; border: 1px solid #ccc; }
    .user-card.selected { background-color: #e8f2ff; }
</style>
```

再让 JavaScript 操作类名：

```javascript
firstCard.classList.add("selected");
console.log(firstCard.classList.contains("selected")); // true

firstCard.classList.toggle("selected"); // 已经有这个类，所以将它移除
firstCard.classList.add("selected");    // 再次加上，方便观察选中效果
```

`classList.remove("selected")` 也可以明确移除这个类。这里的 `selected` 是我们自己约定的类名，浏览器没有内置“选中卡片”的规则；具体显示什么，由对应的 CSS 决定。对于一组固定的外观变化，用类名连接 JS 和 CSS，后面调整样式时就不必同时改动逻辑代码。

也可以直接写 `firstCard.style.backgroundColor = "lightyellow"`，它修改的是元素的内联样式。CSS 的 `background-color` 在这种点号访问中写成 `backgroundColor`。`style` 适合设置运行时计算出来的具体值，但它不代表元素的全部最终样式：即使卡片通过样式表有背景色，`firstCard.style.backgroundColor` 也可能仍是空字符串。

## 创建、插入与删除元素

修改现有卡片以后，接下来尝试增加一个 Charlie。浏览器可以先创建一个独立的元素对象，再由我们决定把它放到文档的什么位置：

```javascript
const newCard = document.createElement("li");
newCard.classList.add("user-card");

const newName = document.createElement("span");
newName.classList.add("user-name");
newName.textContent = "Charlie";

const deleteButton = document.createElement("button");
deleteButton.type = "button";
deleteButton.textContent = "删除";

newCard.append(newName, deleteButton);

console.log(newCard.isConnected); // false：尚未连接到页面文档
list.append(newCard);
console.log(newCard.isConnected); // true：已经插入当前页面
```

`createElement("li")` 返回一个 `li` 元素对象，给它设置类名或文字时，它已经可以正常接受操作，但还不在当前文档树里。`newCard.append(newName, deleteButton)` 先把姓名和按钮组装成一棵小树，`list.append(newCard)` 再把整棵小树接到页面列表下面。创建、配置和插入是几个独立步骤，创建以后并不会自动显示在页面中。

这里还可以观察第四节保存的查询结果：

```javascript
console.log(cards.length); // 2：之前那次查询得到的集合
console.log(list.querySelectorAll(".user-card").length); // 3：重新查询
```

插入位置不同，使用的方法也有所不同：

| 写法 | 插入的位置 |
| --- | --- |
| `list.append(newCard)` | 作为 `list` 的最后一个子节点 |
| `list.prepend(newCard)` | 作为 `list` 的第一个子节点 |
| `firstCard.before(newCard)` | 与 `firstCard` 同级，放在它前面 |
| `firstCard.after(newCard)` | 与 `firstCard` 同级，放在它后面 |

移除这个节点时，可以直接调用 `remove`：

```javascript
newCard.remove();

console.log(newCard.isConnected); // false
console.log(newCard.textContent); // "Charlie删除"
```

卡片从文档树中移除了，但 `newCard` 变量仍然引用着这个对象，因此还能读取内容，甚至可以再次 `list.append(newCard)` 把它放回去。节点离开页面，不等于这个对象立即消失。

到这里，代码已经能控制页面结构，但这些操作并没有替我们维护业务数据。如果卡片最初来自一个用户数组，删除卡片不会自动删除数组中的那条记录；反过来，修改数组也不会让这里手写的 DOM 自动更新。这两部分如何保持一致，会成为后续学习事件和状态时需要处理的问题。

## 从数据到页面，串起 DOM 基础

下面把前面的操作放进一个完整页面。将它单独保存为 `index.html`，用浏览器打开即可运行，不需要与前面的 Console 实验混在一起。代码先根据数组生成三张卡片，再修改 Alice 的显示名称，移除 Bob 的卡片：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>从数据到 DOM</title>
    <style>
        body { font-family: sans-serif; margin: 24px; }
        #user-list { padding: 0; list-style: none; }
        .user-card {
            display: flex;
            gap: 16px;
            align-items: center;
            margin: 8px 0;
            padding: 12px;
            border: 1px solid #ccc;
        }
        .user-card.selected { background-color: #e8f2ff; }
    </style>
</head>
<body>
    <h1>用户列表</h1>
    <ul id="user-list"></ul>
    <p id="summary"></p>

    <script>
        const users = [
            { id: 1, name: "Alice" },
            { id: 2, name: "Bob" },
            { id: 3, name: "Charlie" }
        ];

        const list = document.querySelector("#user-list");
        const summary = document.querySelector("#summary");

        function createUserCard(user) {
            const card = document.createElement("li");
            card.classList.add("user-card");

            const name = document.createElement("span");
            name.classList.add("user-name");
            name.textContent = user.name;

            const button = document.createElement("button");
            button.type = "button";
            button.textContent = "删除";

            card.append(name, button);
            return card;
        }

        users.forEach(user => {
            list.append(createUserCard(user));
        });

        const firstCard = list.firstElementChild;
        firstCard.querySelector(".user-name").textContent = "Alice（已修改）";
        firstCard.classList.add("selected");

        const secondCard = list.children[1];
        secondCard.remove();

        summary.textContent =
            "页面显示 " + list.children.length +
            " 位用户；数组仍有 " + users.length + " 条记录。";
    </script>
</body>
</html>
```

最终页面显示 Alice（已修改）和 Charlie 两张卡片，其中第一张带有选中背景。下方显示“页面显示 2 位用户；数组仍有 3 条记录”。`users[0].name` 仍然是 `"Alice"`，Bob 也仍然留在数组中，因为我们后面的修改只作用于 DOM。卡片上的“删除”按钮目前只是一个元素，还没有注册事件处理函数，点击不会执行删除。

`createUserCard` 接收一条用户数据，返回组装好的元素；循环负责把这些元素插入列表。先把一张卡片如何创建写成函数，再用数组驱动重复操作，就把之前的 JavaScript 语法和页面结构连接了起来。创建三张卡片、修改姓名和移除一张卡片都在这段脚本中顺序执行，浏览器不一定会把每一步中间状态分别画出来，打开页面时通常看到的是执行后的结果。

这个例子的脚本放在 `body` 末尾。浏览器执行到这里时，前面的列表和说明段落已经被解析，可以被查询到。如果把同样的普通内联脚本放到这些元素之前，查询时它们还没进入 DOM，就可能得到 `null`。后面把脚本拆成 `app.js` 时，也可以在 `head` 中这样加载：

```html
<script src="app.js" defer></script>
```

对这种外部普通脚本，`defer` 让浏览器在解析 HTML 的同时加载文件，等文档解析完成后再执行。它不用于延迟普通内联脚本。

现在再用 F12 看这个页面，Elements 面板里已经有两张卡片，但源文件中的 `ul` 仍然是空的。因为 Elements 展示的是当前 DOM，JavaScript 运行后创建的节点也会出现在里面。选中一张卡片，在 Console 中运行 `console.dir($0)`，则可以从对象属性的角度观察它；`$0` 是 Chrome、Edge 开发者工具提供的选中元素引用，不是页面脚本里的通用变量。

也可以在 Console 中运行 `document.querySelector("h1").textContent = "我修改了页面"`，然后刷新。标题会恢复为文件中的“用户列表”，脚本也会重新执行，得到上面那两张卡片。这次实验没有保存数据：我们修改的是运行时的文档对象，没有改写磁盘里的 HTML 文件。

到这里，DOM 基础可以串成一条具体的过程：

```text
HTML 被解析为文档树
        ↓
JavaScript 查询现有元素，或根据数据创建新元素
        ↓
读取内容、修改属性、插入或移除节点
        ↓
浏览器根据当前 DOM 和样式更新页面显示
```

接下来要补上的，是这些操作由谁、在什么时候触发。比如点击“添加”后读取输入框，创建一张新卡片；点击某张卡片的“删除”后移除它。这就进入了事件系统：浏览器把用户操作交给事件处理函数，函数再更新数据和页面。
