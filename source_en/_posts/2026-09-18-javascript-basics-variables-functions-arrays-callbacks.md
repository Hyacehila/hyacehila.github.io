---
title: "JavaScript Basics: Variables, Functions, Arrays, and Event Callbacks"
title_zh: "JavaScript 基础：变量、函数、数组与事件回调"
date: 2026-09-17 12:00:00 +0800
categories: ["Programming", "Programming Languages"]
tags: [JavaScript, Frontend]
author: Hyacehila
excerpt: "Learn JavaScript through a user list: variables and types, objects and arrays, regular and arrow functions, callbacks, and array methods, then connect them to browser events and React list rendering."
description: "Learn JavaScript through a user list: variables and types, objects and arrays, regular and arrow functions, callbacks, and array methods, then connect them to browser events and React list rendering."
excerpt_zh: "以用户列表为例，逐步理解 JavaScript 的变量与类型、对象与数组、普通函数与箭头函数、回调和常用数组操作，并衔接浏览器事件与 React 列表渲染。"
mathjax: false
hidden: true
permalink: '/blog/2026/09/18/javascript-basics-variables-functions-arrays-callbacks/'
lang: en
translation_key: 2026-09-18-javascript-basics-variables-functions-arrays-callbacks
translation_status: machine
translation_source_hash: 88560542d95f383b05bc95a85b0de70fc9a9162efcf8defabe8a08e34ceaa7b9
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

HTML and CSS describe a page's content and appearance. JavaScript comes into play when the page needs to respond to clicks, filter a list, or process data returned by a backend. This article uses a simple user list to introduce variables, objects, arrays, functions, and callbacks, then connects them to page interactions.

The array examples share the same `users` data and are intended to be read and run in order. The sections on browser events and React explain their respective execution environments.

## Variables and Types: Storing the Information a Page Needs

Suppose we are building a page that displays user information. We first need to store a name, an age, and a login status. JavaScript variables give names to these values:

```javascript
let name = "Alice";
const age = 20;
let isLogin = true;
let currentUser = null;
let errorMessage;

name = "Bob";
console.log(name); // "Bob"
```

Both `let` and `const` declare variables. A variable declared with `let` can be reassigned, so `name` can change from `"Alice"` to `"Bob"`. A `const` declaration requires an initial value, and the variable cannot subsequently be assigned another value. When writing code, you can start with `const` and use `let` when reassignment is needed.

`=` performs assignment, and `console.log(...)` prints a value to the console so that we can inspect the result. Text after `//` is a comment and does not execute.

The values stored in these variables have different types:

| Type | Example | What It Might Represent on a Page |
| --- | --- | --- |
| `string` | `"Alice"` | A name, text from an input, or a message |
| `number` | `20`, `19.5` | An age, a quantity, or a price |
| `boolean` | `true`, `false` | Whether someone is logged in, an item is selected, or data is loading |
| `null` | `null` | An explicitly empty value, such as "no current user" |
| `undefined` | The unassigned `errorMessage` above | A value that has not been supplied; it can also result from accessing a nonexistent property |
| `object` | `{ id: 1, name: "Alice" }` | A user, product, or task with several fields |

These are common types encountered when getting started. JavaScript also has `bigint` and `symbol`. Objects differ from the primitive values listed above; we will use them to group related information.

Types affect the result of an operation. For example, `20 + 1` produces the number `21`, while `"20" + 1` produces the string `"201"`. When reading an age or quantity from an input field, we therefore need to check whether we have text or a number. Its appearance on the page is not enough to tell.

## Objects and Arrays: Organizing Related Values

A user may have an ID, a name, and an age. We could declare three separate variables, but placing them in one object makes it clearer that they describe the same person:

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

`id`, `name`, and `age` are properties, with each property's value written after the colon. `user.name` reads the `name` property of `user`, while `user.age = 21` changes its age.

One distinction is easy to miss: although `user` is declared with `const`, its properties can still change. `const` prevents reassignment of the variable; it does not automatically freeze the object. The property update above is valid, but `user = { id: 2 }` would throw an error. This distinction also matters when storing page state.

A page often displays several users. An array can hold their objects together. The later array examples use this data:

```javascript
const users = [
    { id: 1, name: "Alice", age: 20 },
    { id: 2, name: "Bob", age: 17 },
    { id: 3, name: "Carol", age: 23 }
];

console.log(users.length);  // 3
console.log(users[0].name); // "Alice"
```

Arrays use square brackets to hold an ordered collection of values. Indexes start at `0`, so `users[0]` is the first user object, and `users[0].name` is that user's name. An array is itself a kind of object.

Using an object for one record and an array for several records is a useful way to understand user lists, product lists, and chat messages. Backend APIs often exchange similarly structured data using JSON. JSON is a data interchange format; the frontend must parse the response content to obtain JavaScript objects that it can access and manipulate.

## Functions: Giving a Processing Step a Name

Once we have data, we need to process it. A function can accept inputs, perform an operation, and return a result to the place that called it:

```javascript
function add(a, b) {
    return a + b;
}

const result = add(2, 3);
console.log(result); // 5
```

`a` and `b` are parameters. In the call `add(2, 3)`, they receive `2` and `3` respectively. `return` sends back the calculated result and ends that function call, so `result` receives `5`. If execution reaches the end of a function without returning a value, the call returns `undefined`.

Returning to the user list, we can write a function that checks whether a user is at least 18:

```javascript
function isAdult(user) {
    return user.age >= 18;
}

if (isAdult(users[0])) {
    console.log("Alice is an adult");
} else {
    console.log("Alice is not an adult");
}
```

`>=` performs a comparison, and `user.age >= 18` produces a boolean. `if` uses the condition to decide which block to execute; `else` handles the case where the condition is false. This kind of conditional logic can later determine whether to show a message or whether input meets a requirement.

The function gives the rule a name. To check another user, we pass in a different object rather than repeat the same logic. This check illustrates frontend syntax; any real access-control decision still needs independent validation on the backend.

## Arrow Functions: Reading a Common Shorthand

The `=>` syntax appears frequently in frontend code. It defines an arrow function. Our addition function can also be written in either of these forms:

```javascript
const addWithBlock = (a, b) => {
    return a + b;
};

const addShort = (a, b) => a + b;

console.log(addWithBlock(2, 3)); // 5
console.log(addShort(2, 3));     // 5
```

The parameters are on the left of the arrow, and the function body is on the right. When the right-hand side is a single expression, we can omit the braces and `return`; the expression's value is returned directly. With a block body in braces, an explicit `return` is needed to return the calculation's result.

For a single ordinary parameter, the parentheses around the parameter can also be omitted. The age check can therefore be written as:

```javascript
const checkAdult = user => user.age >= 18;

console.log(checkAdult(users[0])); // true
```

Read `user => user.age >= 18` as "accept a user and return whether their age is at least 18." Becoming comfortable with that reading makes the array operations below easier to follow.

## Functions as Arguments and Return Values

`checkAdult` holds a function. In JavaScript, functions can be passed around as values. The distinction to keep in mind is that `checkAdult` refers to the function itself, while `checkAdult(users[0])` calls it now and produces its return value.

For example, we can write a function that accepts both a user and a rule for checking that user:

```javascript
function checkUser(user, rule) {
    return rule(user);
}

console.log(checkUser(users[0], checkAdult)); // true
```

When this runs, `checkAdult` is passed into the `rule` parameter. The check actually executes when the function reaches `rule(user)`. To change the check, we can supply a different function without changing how `checkUser` invokes the rule.

A function passed as an argument to another function and invoked by the receiving code is called a **callback**. When it runs depends on the code receiving it: some callbacks execute immediately, while others wait for a later event. A callback is not necessarily asynchronous.

A function can also return another function. For example, we might want to create several age checks with different thresholds:

```javascript
function createAgeChecker(minAge) {
    return user => user.age >= minAge;
}

const atLeast18 = createAgeChecker(18);
const atLeast21 = createAgeChecker(21);

console.log(atLeast18(users[0])); // true
console.log(atLeast21(users[0])); // false
```

`createAgeChecker(18)` returns a checking function. It only checks a particular user when we subsequently call `atLeast18`. The first call establishes the rule; the second applies it to data.

## Array Methods: Applying Rules to a Collection

We can now read the arrow functions used in array methods. To keep only users aged 18 or older:

```javascript
const adults = users.filter(user => user.age >= 18);

console.log(adults.map(user => user.name)); // ["Alice", "Carol"]
```

`filter` iterates through the array and passes each item to the callback. Here, the parameter `user` represents the item being processed. When the check returns `true`, that item is retained in the new array. `map` instead transforms each item into the callback's return value, so `adults.map(user => user.name)` produces an array of names.

We can also pass the previously defined function directly: `users.filter(checkAdult)` applies the same rule. The arrow-function form simply writes that rule directly where the argument is supplied.

These methods serve different list-processing needs:

| Method | Return Value | Use with a User List |
| --- | --- | --- |
| `map` | A new array of transformed values | Extract every name or produce display content for each item |
| `filter` | A new array containing matching items | Keep adult users or search matches |
| `find` | The first match, or `undefined` if none is found | Locate a user by ID |
| `some` | A boolean indicating whether at least one item matches | Check whether any user is under 18 |
| `reduce` | A result accumulated across the items | Calculate a count, total price, or another aggregate |

Here is how `find` and `some` can be used:

```javascript
const selectedUser = users.find(user => user.id === 2);
const hasMinor = users.some(user => user.age < 18);

console.log(selectedUser.name); // "Bob"; this example contains a user with ID 2
console.log(hasMinor);          // true
```

`===` performs strict equality comparison. It does not first convert strings and numbers to the same type; for example, `2 === "2"` is `false`. In an actual project, if the ID may not exist, we should check whether `selectedUser` is `undefined` before reading its properties.

`reduce` takes a little more explanation. Consider calculating the sum of all users' ages:

```javascript
const totalAge = users.reduce((sum, user) => {
    return sum + user.age;
}, 0);

console.log(totalAge); // 60
```

The final `0` is the accumulator's initial value. Processing Alice returns `0 + 20`. On the second iteration, that returned `20` becomes `sum`, and Bob's `17` is added. Finally, Carol's `23` brings the result to `60`. The same pattern can calculate a shopping cart total. Providing an initial value also lets an empty array return that initial value.

The callbacks in these examples execute synchronously. `map` and `filter` return new arrays, but that does not mean every object inside has been copied. For example, `filter` retains references to the existing objects. These examples do not modify the original array; assigning to an object's properties inside a callback could still modify the original data.

## Event Callbacks: Running Code When a User Acts

So far, we have called the code directly. On a web page, much of it needs to wait for the user to click a button. Try placing this fragment inside an HTML file's `body`:

```html
<button id="hello-button" type="button">Say hello</button>
<p id="message">The button has not been clicked</p>

<script>
    const button = document.querySelector("#hello-button");
    const message = document.querySelector("#message");

    button.addEventListener("click", () => {
        message.textContent = "Hello, Alice";
        console.log("clicked");
    });
</script>
```

`document.querySelector` finds a page element using a selector. `#hello-button` matches the button's `id`. The script appears after both elements, so they have already been parsed when the queries run.

`addEventListener` registers a click handler for the button. When the registration statement runs, the browser stores the function. When the user clicks the button, the browser calls it, updates the paragraph's text, and prints `"clicked"` to the console. DOM queries and event listeners are APIs provided by the browser, which we call using JavaScript.

We can give the handler a name instead of writing it as an inline arrow function:

```javascript
function handleClick() {
    message.textContent = "Hello, Alice";
    console.log("clicked");
}

button.addEventListener("click", handleClick);
```

This replaces the listener registration in the previous example. We pass `handleClick` because we want to give the browser a function it can call. Writing `handleClick()` would execute it during registration and pass its result instead. This is a practical example of the difference between a function and the result of calling it.

Timers also accept callbacks:

```javascript
setTimeout(() => {
    console.log("Runs later");
}, 1000);

console.log("This line runs first");
```

This prints "This line runs first" before "Runs later." `1000` is the requested delay in milliseconds. Browser scheduling also affects when the callback actually runs, so it does not guarantee execution at precisely one second. A `map` callback runs during array processing, a button handler runs when a click event occurs, and a timer callback runs later. These execution times are what we will need to follow when studying the event loop and asynchronous requests.

## Where These Patterns Appear in Frontend Development

On a user-list page, the frontend can store incoming data in objects and arrays, use `filter` to narrow the list, use `find` to locate a selected user, and respond to actions through event callbacks. We now have a connection between data and processing rules; displaying the results is the next part to learn.

When learning React, we will encounter list code like this:

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

This is an illustrative React fragment that assumes the project has already defined or imported `UserCard`. In `UserList`'s parameter list, `{ users }` extracts the `users` property from the supplied object, a syntax called object destructuring. `map` still produces one result for each user; here, that result is JSX describing a user card.

`<UserCard ... />` is JSX and requires a suitable transformation environment. It cannot be pasted into the console as ordinary JavaScript. `user={user}` passes the current user object to the component, while `key={user.id}` gives React a stable ID for identifying the list item. The [React documentation on rendering lists](https://react.dev/learn/rendering-lists) explains this pattern further.

Several pieces are now familiar: `users` is an array, `user` is one of its objects, and the arrow function describes how each item is transformed. The next step is to understand how the DOM represents a page and how JavaScript updates the page through it. That provides a concrete point of comparison when learning how frameworks update interfaces from data.
