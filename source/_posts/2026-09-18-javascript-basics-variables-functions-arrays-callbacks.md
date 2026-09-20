---
title: "JavaScript 基础：变量、函数、数组与事件回调"
title_en: "JavaScript Basics: Variables, Functions, Arrays, and Event Callbacks"
date: 2026-09-17 12:00:00 +0800
categories: ["Programming", "Programming Languages"]
tags: [JavaScript, Frontend]
author: Hyacehila
excerpt: "以用户列表为例，逐步理解 JavaScript 的变量与类型、对象与数组、普通函数与箭头函数、回调和常用数组操作，并衔接浏览器事件与 React 列表渲染。"
excerpt_en: "Learn JavaScript through a user list: variables and types, objects and arrays, regular and arrow functions, callbacks, and array methods, then connect them to browser events and React list rendering."
description: "以用户列表为例，逐步理解 JavaScript 的变量与类型、对象与数组、普通函数与箭头函数、回调和常用数组操作，并衔接浏览器事件与 React 列表渲染。"
mathjax: false
hidden: true
permalink: '/blog/2026/09/18/javascript-basics-variables-functions-arrays-callbacks/'
---

HTML 和 CSS 描述了页面的内容与样式。页面需要响应点击、筛选列表或处理后端返回的数据时，就会用到 JavaScript。这篇文章从一个简单的用户列表出发，整理变量、对象、数组、函数和回调的基本写法，再看看它们如何用于页面交互。

文中的数组处理示例沿用同一份 `users` 数据，适合按顺序阅读和运行。涉及浏览器事件和 React 的部分会分别说明运行环境。

## 变量与类型：先把页面需要的信息存下来

假设我们要做一个显示用户信息的页面，最先需要保存的是姓名、年龄和登录状态。JavaScript 可以用变量给这些值命名：

```javascript
let name = "Alice";
const age = 20;
let isLogin = true;
let currentUser = null;
let errorMessage;

name = "Bob";
console.log(name); // "Bob"
```

这里的 `let` 和 `const` 都用于声明变量。`let` 声明的变量可以重新赋值，所以 `name` 能从 `"Alice"` 改成 `"Bob"`；`const` 声明时必须给出值，此后不能再给这个变量赋另一个值。写代码时，可以先用 `const`，确定需要重新赋值时再使用 `let`。

`=` 表示赋值，`console.log(...)` 用来把值输出到控制台，方便观察代码的执行结果。示例中的 `//` 后面是注释，不参与执行。

这些变量保存的值也有不同类型：

| 类型 | 示例 | 在页面中可能表示什么 |
| --- | --- | --- |
| `string` | `"Alice"` | 姓名、输入框里的文字、提示信息 |
| `number` | `20`、`19.5` | 年龄、数量、价格 |
| `boolean` | `true`、`false` | 是否登录、是否选中、是否正在加载 |
| `null` | `null` | 主动表示“当前没有用户”等空值 |
| `undefined` | 上面尚未赋值的 `errorMessage` | 尚未获得值，也可能来自不存在的属性 |
| `object` | `{ id: 1, name: "Alice" }` | 包含多个字段的用户、商品或任务 |

这里列的是入门时经常遇到的类型，JavaScript 还有 `bigint` 和 `symbol`。对象与前面的几种原始值不同，后面会用它来组织相关信息。

类型会影响运算结果。例如 `20 + 1` 得到数字 `21`，而 `"20" + 1` 得到字符串 `"201"`。以后从输入框读取年龄或数量时，就需要留意拿到的是文字还是数字，不能只根据页面上显示的样子判断。

## 对象与数组：把零散的值组织成数据

如果一个用户有编号、姓名和年龄，分别声明三个变量当然可以，但把它们放到同一个对象中，更容易看出这些信息属于谁：

```javascript
const user = {
    id: 1,
    name: "Alice",
    age: 20
};

console.log(user.name); // "Alice"
console.log(user.age);  // 20

user.age = 21;
console.log(user.age);  // 21
```

对象里的 `id`、`name` 和 `age` 都是属性，冒号后面是属性的值。`user.name` 表示读取 `user` 的 `name` 属性，`user.age = 21` 则修改了其中的年龄。

这里会遇到一个容易误解的地方：虽然 `user` 使用 `const` 声明，里面的属性仍然可以修改。`const` 限制的是给变量重新赋值，并不会自动冻结对象。上面的修改有效，但 `user = { id: 2 }` 会报错。这个区别在保存页面状态时也会用到。

页面通常还需要显示多个用户，这时可以用数组把多个对象放在一起。后面的数组示例都以这份数据为基础：

```javascript
const users = [
    { id: 1, name: "Alice", age: 20 },
    { id: 2, name: "Bob", age: 17 },
    { id: 3, name: "Carol", age: 23 }
];

console.log(users.length);  // 3
console.log(users[0].name); // "Alice"
```

数组用方括号表示，按顺序保存一组值，下标从 `0` 开始。因此，`users[0]` 是第一个用户对象，`users[0].name` 是这个用户的姓名。数组本身也是一种对象。

“一个对象描述一条记录，一个数组保存多条记录”会贯穿很多前端功能。例如用户列表、商品列表和聊天消息，都可以按这个方式理解。后端接口也常通过 JSON 传递类似的数据。不过，JSON 是数据交换格式；前端需要先解析响应内容，才能得到这里可以访问和操作的 JavaScript 对象。

## 函数：把一段处理过程写成可以调用的行为

有了数据，下一步就是处理数据。函数可以接收输入、执行操作，再把结果返回给调用它的地方：

```javascript
function add(a, b) {
    return a + b;
}

const result = add(2, 3);
console.log(result); // 5
```

`a` 和 `b` 是参数，调用 `add(2, 3)` 时分别接收到 `2` 和 `3`。`return` 把计算结果返回，并结束这次函数执行，所以 `result` 得到 `5`。如果函数执行结束时没有返回值，调用结果就是 `undefined`。

放回用户列表，我们可以把“是否成年”写成一个函数：

```javascript
function isAdult(user) {
    return user.age >= 18;
}

if (isAdult(users[0])) {
    console.log("Alice 已成年");
} else {
    console.log("Alice 未成年");
}
```

`>=` 用于比较，`user.age >= 18` 的结果是布尔值。`if` 根据条件决定执行哪个代码块，`else` 处理条件不成立的情况。以后判断是否显示某个提示、输入内容是否符合要求，都可以从这样的条件判断开始。

函数让判断规则有了名字。需要判断其他用户时，只要传入另一个对象，不必重新写一遍同样的逻辑。上面的判断只用于演示前端语法；涉及真实权限时，后端仍需要独立校验。

## 箭头函数：读懂前端代码里常见的简写

前端代码里经常出现 `=>`，这是箭头函数的语法。前面的加法函数也可以分别写成下面两种形式：

```javascript
const addWithBlock = (a, b) => {
    return a + b;
};

const addShort = (a, b) => a + b;

console.log(addWithBlock(2, 3)); // 5
console.log(addShort(2, 3));     // 5
```

箭头左侧是参数，右侧是函数体。如果右侧只有一个表达式，可以省略大括号和 `return`，直接返回表达式的结果。如果使用了大括号，就需要显式写 `return` 才能返回这里的计算结果。

只有一个普通参数时，参数外面的括号也可以省略。因此，前面的成年判断可以写成：

```javascript
const checkAdult = user => user.age >= 18;

console.log(checkAdult(users[0])); // true
```

读 `user => user.age >= 18` 时，可以把它理解为“接收一个用户，返回这个用户的年龄是否大于等于 18”。先把这种读法掌握好，后面的数组操作就容易理解了。

## 函数可以作为参数，也可以作为返回值

上面的 `checkAdult` 保存的就是一个函数。在 JavaScript 中，函数也可以作为值传来传去。这里尤其需要分清：`checkAdult` 表示函数本身，`checkAdult(users[0])` 表示现在调用它，得到返回结果。

例如，我们可以写一个函数，接收“用户”和“如何判断这个用户”两个参数：

```javascript
function checkUser(user, rule) {
    return rule(user);
}

console.log(checkUser(users[0], checkAdult)); // true
```

调用时，`checkAdult` 被传给了参数 `rule`；执行到 `rule(user)` 时，才真正调用这个判断函数。以后需要更换规则，可以传入另一段函数，而不必改写 `checkUser` 的调用规则。

像这样，作为参数交给其他函数，由接收方调用的函数，称为**回调函数（callback）**。它的执行时机取决于接收它的函数：有的会立刻调用，有的会等到后续事件发生。回调与异步不是同一个概念。

函数还可以返回另一个函数。例如，想创建几种不同的年龄判断规则：

```javascript
function createAgeChecker(minAge) {
    return user => user.age >= minAge;
}

const atLeast18 = createAgeChecker(18);
const atLeast21 = createAgeChecker(21);

console.log(atLeast18(users[0])); // true
console.log(atLeast21(users[0])); // false
```

`createAgeChecker(18)` 返回的是一个判断函数，之后调用 `atLeast18` 时才会判断某个用户。第一次确定规则，第二次用规则处理数据。

## 数组操作：把处理规则应用到一组数据

现在再看数组方法，就能把其中的箭头函数读出来了。比如，只保留成年用户：

```javascript
const adults = users.filter(user => user.age >= 18);

console.log(adults.map(user => user.name)); // ["Alice", "Carol"]
```

`filter` 负责遍历数组，把每一项交给回调函数。这里参数名 `user` 表示当前正在处理的那一项；判断结果为 `true` 时，这一项就被保留到新数组中。`map` 则把每一项转换为回调返回的值，所以 `adults.map(user => user.name)` 得到的是姓名数组。

前面定义的函数也可以直接传进去：`users.filter(checkAdult)` 与这里的筛选规则相同。箭头函数只是把这段规则就地写在了参数位置。

处理列表时，下面几个方法分别对应不同的需求：

| 方法 | 返回结果 | 对用户列表的用途 |
| --- | --- | --- |
| `map` | 转换后的新数组 | 提取所有姓名，或生成每一项的展示内容 |
| `filter` | 符合条件的元素组成的新数组 | 筛选成年用户、搜索匹配项 |
| `find` | 第一个匹配项，找不到时返回 `undefined` | 按编号找到某个用户 |
| `some` | 是否至少有一项符合条件的布尔值 | 检查是否存在未成年用户 |
| `reduce` | 逐项累计得到的结果 | 计算总数、总价或其他汇总值 |

`find` 与 `some` 可以这样使用：

```javascript
const selectedUser = users.find(user => user.id === 2);
const hasMinor = users.some(user => user.age < 18);

console.log(selectedUser.name); // "Bob"；本例确定存在 id 为 2 的用户
console.log(hasMinor);          // true
```

这里的 `===` 是严格相等比较，不会先把字符串和数字转换成同一种类型；例如 `2 === "2"` 是 `false`。实际项目里，如果编号不一定存在，就需要先判断 `selectedUser` 是否为 `undefined`，再读取它的属性。

`reduce` 的写法稍长一些。以计算所有用户的年龄总和为例：

```javascript
const totalAge = users.reduce((sum, user) => {
    return sum + user.age;
}, 0);

console.log(totalAge); // 60
```

末尾的 `0` 是累计结果的初始值。第一次处理 Alice，返回 `0 + 20`；第二次把上一次返回的 `20` 作为 `sum`，再加上 Bob 的 `17`；最后加上 Carol 的 `23`，得到 `60`。同样的写法可以用于计算购物车总价。显式提供初始值后，空数组也能返回这个初始值。

这些方法在这里都同步执行回调。`map` 和 `filter` 返回新数组，但新数组不等于把内部对象全部复制了一遍，例如 `filter` 保留的仍是原有对象。上述写法没有修改原数组；如果在回调里给对象属性赋值，仍然可能修改原数据。

## 事件回调：让代码在用户操作时运行

前面的代码由我们直接调用。放到网页里，很多代码需要等用户点击按钮后才执行。可以把下面这个片段放进 HTML 文件的 `body` 中试一下：

```html
<button id="hello-button" type="button">打个招呼</button>
<p id="message">还没有点击按钮</p>

<script>
    const button = document.querySelector("#hello-button");
    const message = document.querySelector("#message");

    button.addEventListener("click", () => {
        message.textContent = "你好，Alice";
        console.log("clicked");
    });
</script>
```

`document.querySelector` 按选择器找到页面元素，`#hello-button` 对应按钮的 `id`。脚本放在这两个元素之后，执行查询时它们已经被解析。

`addEventListener` 为按钮注册点击处理函数。执行注册这一行时，浏览器保存这个函数；用户点击按钮后，浏览器再调用它，修改段落文字，并向控制台输出 `"clicked"`。这里的 DOM 查询和事件监听都是浏览器提供的 API，我们用 JavaScript 调用它们。

如果把箭头函数单独命名，这部分也可以写成：

```javascript
function handleClick() {
    message.textContent = "你好，Alice";
    console.log("clicked");
}

button.addEventListener("click", handleClick);
```

这段写法用于替换前面示例中的监听注册。传入的是 `handleClick`，因为我们要交给浏览器一个可以调用的函数。如果写成 `handleClick()`，就会在注册时立即执行，并把执行结果传进去。这也是“函数本身”与“函数调用结果”的一个实际区别。

计时器也接收回调：

```javascript
setTimeout(() => {
    console.log("延迟执行");
}, 1000);

console.log("先执行这一行");
```

这里会先输出“先执行这一行”，之后才输出“延迟执行”。`1000` 的单位是毫秒，表示请求的延迟；回调实际运行时间还受浏览器调度影响，不能理解为精确到一秒就执行。`map` 的回调在数组处理过程中执行，按钮回调在点击事件发生时执行，计时器回调则延后执行。以后学习事件循环和异步请求时，需要继续追踪的就是这些执行时机。

## 这些语法如何出现在后面的前端开发中

以一个用户列表页面为例，前端获得数据后，可以用对象和数组保存用户信息，用 `filter` 筛选，用 `find` 找到选中的用户，再通过事件回调响应操作。到这里，数据和处理规则已经能够接起来了，还需要学习如何把结果显示到页面上。

以后接触 React 时，会看到类似这样的列表代码：

```jsx
function UserList({ users }) {
    return (
        <div>
            {users.map(user => (
                <UserCard key={user.id} user={user} />
            ))}
        </div>
    );
}
```

这是 React 中的示意片段，假定项目已经定义或导入了 `UserCard` 组件。`UserList` 接收参数时，`{ users }` 从传入的对象中取出 `users` 属性，这种写法叫对象解构。`map` 的作用没有变化：为每个用户返回一项结果，只是这里返回的是描述用户卡片的 JSX。

`<UserCard ... />` 是 JSX 写法，需要相应的转换环境，不能直接当作普通 JavaScript 粘贴到控制台里运行。`user={user}` 把当前用户对象传给组件，`key={user.id}` 则用稳定的编号帮助 React 识别列表项。具体写法可以参照 [React 的列表渲染文档](https://react.dev/learn/rendering-lists)。

现在回看这段代码，里面已经有不少熟悉的部分：`users` 是数组，`user` 是其中的对象，箭头函数负责描述每一项如何转换。接下来要补上的，是 DOM 如何表示页面，以及 JavaScript 如何通过它更新页面。弄清这一步以后，再理解框架如何根据数据更新界面，就有了可以对照的基础。
