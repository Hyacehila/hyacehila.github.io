---
title: "DOM Basics: Document Trees, Element Operations, and Dynamic Pages"
title_zh: "DOM 基础：文档树、元素操作与动态页面"
date: 2026-09-18 12:00:00 +0800
categories: ["Programming", "Full Stack Development"]
tags: [DOM, HTML, JavaScript, Frontend]
author: Hyacehila
excerpt: "Understand how HTML becomes a DOM tree and how document, window, Node, and Element relate. Use a user list to explore element queries, content and attribute updates, node creation and removal, and the relationship between data and the page."
description: "Understand how HTML becomes a DOM tree and how document, window, Node, and Element relate. Use a user list to explore element queries, content and attribute updates, node creation and removal, and the relationship between data and the page."
excerpt_zh: "从 HTML 如何形成 DOM 树讲起，理解 document、window、Node 与 Element，再通过用户列表示例学习元素查找、内容与属性修改、节点创建和删除，以及数据与页面之间的关系。"
mathjax: false
hidden: true
permalink: '/blog/2026/09/17/dom-basics-document-tree-and-element-operations/'
lang: en
translation_key: 2026-09-17-dom-basics-document-tree-and-element-operations
translation_status: machine
translation_source_hash: 009d492a8ee7d51906c25ef8be1c59fac9fdb153af43312a8079ce78fe79c653
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

This article builds on [Web Frontend Basics Overview](/en/blog/2025/05/15/web-frontend-basics-overview/) and [JavaScript Basics: Variables, Functions, Arrays, and Event Callbacks](/en/blog/2026/09/18/javascript-basics-variables-functions-arrays-callbacks/). It starts with the document tree, then uses a user list to connect element queries, updates, creation, and removal.

## Why We Need the DOM

An HTML file stores text describing a page. To work with that page, a program needs to find specific elements, understand their relationships, and read or change their contents. As the browser parses HTML, it organizes the document into an in-memory tree of nodes, including element nodes and text nodes. This is the DOM, or Document Object Model: it turns the document structure described in HTML into objects that programs can access and manipulate. The tree tells us, for example, which items belong to a list or which container holds a button.

With the DOM, JavaScript can use a consistent interface to find and modify page objects. When a user clicks "Add task," code can create a list item and insert it into the existing list. To change a heading, it can find the corresponding element and update its text. The browser then presents the page using the updated document structure and styles. We do not need to reconstruct the entire HTML string or reload the page for every change. The DOM connects page structure to interaction code, giving the variables, functions, and callbacks we learned earlier concrete page content to work with.

Thinking back to the game UI systems I have worked on, HTML is somewhat like a project file. Opening the project in an editor produces a structure resembling a DOM tree, represented in the system's own format. The editor's tools can modify that structure and save a new file. When connecting application logic, game UI and web UI follow a similar progression from layout design to behavior, although game UI work involves separate production roles and more logic within the game engine.

## What the DOM Is

Start with a simple user-list page. Save the following as a UTF-8 HTML file and open it in a browser. You will see a heading, two usernames, and a button:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DOM Demo</title>
</head>
<body>
    <h1 id="title">User list</h1>

    <div class="users">
        <p>Alice</p>
        <p>Bob</p>
    </div>

    <button id="add-btn">Add user</button>
</body>
</html>
```

What we write in the file is HTML text. The browser's HTML parser identifies tags, attributes, and text, then constructs the corresponding DOM tree. Focusing just on document structure, the process is `HTML text → HTML parser → DOM tree`. The document above produces approximately this structure, omitting whitespace text nodes introduced by indentation and line breaks:

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
        │   └── #text "User list"
        ├── div.users
        │   ├── p
        │   │   └── #text "Alice"
        │   └── p
        │       └── #text "Bob"
        └── button#add-btn
            └── #text "Add user"
```

`Document` is the node for the whole document. `html`, `body`, and `p` are element nodes, while the text inside elements is represented by text nodes. For example, the first `p` contains a text node whose content is `Alice`, and both `p` elements are children of `div`. Labels such as `h1#title` and `div.users` include the ID or class to make the elements easier to identify; those attributes are not additional child nodes in this diagram.

From JavaScript, the tree's nodes are accessible objects. The button, for example, is represented by an `HTMLButtonElement` object that exposes properties for reading or updating it and methods that can be called:

```text
HTMLButtonElement (some available properties and methods)
├── id          → "add-btn"
├── textContent → "Add user"
├── disabled    → false
├── style       → Read or modify inline styles
├── classList   → Read or modify classes
└── click()     → Trigger a click programmatically
```

Open the page's developer tools and run the following in the Console to retrieve and modify the button:

```javascript
const button = document.querySelector("#add-btn");

console.log(button instanceof HTMLButtonElement); // true
console.log(button.textContent);                  // "Add user"

button.textContent = "Adding is unavailable";
button.disabled = true;
```

`document.querySelector("#add-btn")` finds the first element matching the selector in the DOM and returns that element object. If there is no match, it returns `null`. Here, `button` refers to the existing button on the page, so changing its `textContent` and `disabled` properties changes its label and makes it unavailable. No user-adding logic is attached to this example button yet; clicking it does not automatically add a user.

When JavaScript queries the button, it works with the document objects the browser has already created, rather than searching and replacing text in an HTML string. The DOM preserves the document's hierarchy and provides interfaces for operating on its nodes. Element queries, node insertion, and event listeners all build on this object structure.

## document and window: The Page and Its Environment

The previous section found a button using `document.querySelector(...)`. Here, `document` is the document object for the current page and our entry point into its DOM. Open the DOM Demo, press F12 to open developer tools, and enter `document` in the Console. The tools usually display an expandable document structure. It represents the whole document, including page information in `head` and page content in `body`.

For example, `document.title` reads and updates the document's title:

```javascript
console.log(document.title); // "DOM Demo"

document.title = "New title";
```

After this runs, the browser tab's title becomes "New title," and the text in the document's `<title>` element changes as well. The `<h1>` on the page still reads "User list," because it is a different element.

The browser also provides another frequently used object: `window`. A useful starting point is to treat it as the main entry point to the current page's execution environment. In ordinary page scripts, it is the global object supplied by the browser. Through it, we can access both the document and facilities for URLs, navigation, storage, and timers:

```text
window (entry point to the page's environment)
├── document     → The current document
├── location     → The current URL and navigation
├── history      → Session history for the current browsing context
├── navigator    → Access to browser, device, and capability information
├── localStorage → Local storage separated by origin
├── console      → Console output and debugging
└── setTimeout   → Schedule a callback for later
```

This diagram shows properties and methods accessible through `window`. It is not another DOM tree, and `window` is not the DOM parent of `document`. It corresponds to the current window or frame environment, rather than the entire browser application. An embedded `iframe` also has its own `window` and `document`.

The distinction depends on what we want to operate on. For page text, buttons, and lists, we mainly start with `document`. For the current URL or a timer, we use other facilities exposed through `window`. Continue exploring the demo in the Console:

```javascript
console.log(window.document === document); // true
console.log(window.location.href);         // The page's full URL

console.log(document.documentElement);     // The <html> element
console.log(document.head);                // The <head> element
console.log(document.body);                // The <body> element
```

We often write `document`, `console.log(...)`, or `setTimeout(...)` directly because those names are accessible in an ordinary page environment, allowing the `window.` prefix to be omitted. They are supplied by the runtime environment; JavaScript does not inherently include a web page and browser everywhere it runs. For example, JavaScript running in Node.js does not have the browser-provided `window` and `document` by default.

As we continue studying the DOM, the access path can be read as `window.document → document.documentElement → head / body → individual elements`. `window.document` gives us the document, and `document.documentElement` gives us its root element, `<html>`. The page's other nodes sit below it. In practice, we usually start directly with `document` and query the element we need.

## Node and Element: How Nodes Differ from Elements

Our DOM diagrams include `div` and `button`, but also text and the document itself. All of them are **nodes**, or `Node` objects. Nodes representing HTML elements are **elements**, or `Element` objects. A button is therefore both an Element and a Node. A piece of text is also a Node, but its type is Text, not Element.

The common node types can be grouped like this:

```text
Node (the common base type for nodes)
├── Element       Element nodes, such as div, p, and button
├── Text          Text nodes, such as "Hello"
├── Comment       Comment nodes, such as <!-- Note -->
├── Document      Document nodes, such as document
└── DocumentType  Document type nodes, such as <!DOCTYPE html>
```

These nodes share basic capabilities such as accessing their `parentNode` or inspecting their `childNodes`, but they serve different purposes. Elements can have attributes such as `id` and `class`; text nodes store text; comment nodes store comment content.

Consider this HTML:

```html
<p>Hello</p>
```

For this fragment, the browser creates a `p` element node with a text node inside it:

```text
p (Element)
└── "Hello" (Text)
```

`<p>` and `</p>` mark the element's boundaries in the HTML source. They do not become two separate element nodes. Together, they describe one `p` element, whose text content is represented by the `Hello` node.

Text has its own nodes because an element can contain a mixture of text and other elements. In `<p>Hello <strong>Alice</strong>!</p>`, the direct children of `p` are the text `"Hello "`, the `strong` element, and the text `"!"`. The `strong` element contains another text node, `"Alice"`. This preserves the order of the content and tells the browser that only `Alice` is inside `strong`.

If we format the HTML across several lines:

```html
<div>
    Hello
</div>
```

The text node inside this `div` contains `"\n    Hello\n"`: a newline, four spaces, `Hello`, and another newline. Here, `\n` represents a newline character. Default styling usually collapses this whitespace, so the page appears to show just `Hello`. That does not mean the whitespace has disappeared from the DOM.

This is why the first DOM diagram explicitly omitted whitespace text nodes. From the page's appearance, we might assume a container's first child is its button. On inspection, that first child may instead contain the newline and spaces before the button.

Add this line inside the earlier DOM Demo's `body` and open the Console. It is deliberately written on one line to avoid extra indentation and line breaks:

```html
<div id="node-demo">Hello<span>World</span><!--Note--></div>
```

Its structure is:

```text
div#node-demo (Element)
├── "Hello" (Text)
├── span (Element)
│   └── "World" (Text)
└── "Note" (Comment)
```

The `div` has three direct child nodes, but only `span` is an element. `World` is a child of `span` and a descendant of `div`, not a direct child of `div`. We can confirm the distinction with:

```javascript
const demo = document.querySelector("#node-demo");

console.log(demo.childNodes.length); // 3: text, a span element, and a comment
console.log(demo.children.length);   // 1: only the span element

console.log(demo.firstChild.nodeName);        // "#text"
console.log(demo.firstElementChild.tagName);  // "SPAN"

console.log(demo instanceof Node);              // true
console.log(demo instanceof Element);           // true
console.log(demo.firstChild instanceof Node);   // true
console.log(demo.firstChild instanceof Element); // false
```

`childNodes` includes direct children of every node type, while `children` includes only direct child elements. Similarly, `firstChild` returns the first child node, and `firstElementChild` returns the first child element. `instanceof` checks whether an object belongs to a type: both checks on `demo` return `true`, while the text node satisfies the Node check but not the Element check.

We can also inspect the same element's text:

```javascript
console.log(demo.textContent); // "HelloWorld"
```

For this element, `textContent` collects text from descendant text nodes. It excludes the comment and does not automatically insert a space between `Hello` and `World`. It reads text from the subtree; it does not add an extra "combined text node" to `div`.

When changing button labels, switching styles, or reading input values, we frequently work with Elements because those tasks target specific elements. The distinction between Node and Element becomes especially relevant when traversing the full document or handling text. Names such as `childNodes` and `firstChild` can include text or comments. If we only need elements, we can use `children`, `firstElementChild`, or an element selector.

## Finding and Modifying Elements

We have already used `querySelector`; now we can connect queries to updates. Put the following, slightly more complete user list inside a test page's `body`. Run the examples in this section and the next in order in that page's Console. Refresh before starting a new round of experiments.

```html
<h1 id="list-title">User list</h1>
<label for="username">Username</label>
<input id="username" value="Alice">

<ul id="user-list">
    <li class="user-card">
        <span class="user-name">Alice</span>
        <button type="button">Delete</button>
    </li>
    <li class="user-card">
        <span class="user-name">Bob</span>
        <button type="button">Delete</button>
    </li>
</ul>
```

`querySelector` accepts a CSS selector and returns the first matching element. `#list-title` searches by ID, `.user-card` by class, and `button` by tag name. Selectors can also be combined: `#user-list .user-name` finds a username element inside the list. We can also retrieve a container first, then query within it:

```javascript
const listTitle = document.querySelector("#list-title");
const list = document.querySelector("#user-list");
const firstCard = list.querySelector(".user-card");
const firstName = firstCard.querySelector(".user-name");

console.log(firstName.textContent); // "Alice"

const cards = list.querySelectorAll(".user-card");
cards.forEach(card => {
    console.log(card.querySelector(".user-name").textContent);
}); // Prints "Alice", then "Bob"
```

`querySelectorAll` returns a `NodeList` of matching elements. We can access items by index or iterate with `forEach`, but it is not an ordinary array. The collection contains the elements that matched when the query ran. Adding a card later does not automatically add an item to `cards`; we need to query again.

If a valid selector matches nothing, `querySelector` returns `null` and `querySelectorAll` returns an empty collection. When an error says a property of null cannot be read, check the selector, the query's scope, and execution timing. Confirm that the element exists before operating on it. Another common form is `document.getElementById("list-title")`, which accepts the ID directly, without a leading `#`.

Once we have the element object, we can change its text:

```javascript
listTitle.textContent = "Employee list";
firstName.textContent = "Alicia";
```

Here, we update only the `span` holding the name, so the adjacent Delete button remains. If we instead wrote `firstCard.textContent = "Alicia"`, all the card's existing child nodes would be replaced, including its `span` and button, leaving only text. The element we select determines the scope of a `textContent` assignment. Choosing the right object matters more than memorizing the assignment syntax.

An input's current content is read through `value`. This example shows the distinction between an HTML attribute and an object property:

```javascript
const input = document.querySelector("#username");

console.log(input.getAttribute("value")); // "Alice"
console.log(input.value);                 // "Alice"

input.value = "Bob";

console.log(input.value);                 // "Bob"
console.log(input.getAttribute("value")); // Still "Alice"
```

The HTML attribute `value="Alice"` supplies the input's default value. The object property `input.value` represents its current value, which also changes when a user types. When submitting a form, we therefore need `input.value` rather than `getAttribute("value")`. Different attributes follow different correspondence rules; not every attribute is an unchanging initial value.

Content attributes can also be modified. For example, `firstCard.setAttribute("title", "User details")` sets the hover tooltip, `getAttribute("title")` reads it, `hasAttribute("title")` checks whether it exists, and `removeAttribute("title")` removes it. States such as whether an input is available can be set through object properties, for example `input.disabled = true`.

To change the appearance, first add styles inside the page's `head`:

```html
<style>
    .user-card { padding: 8px; border: 1px solid #ccc; }
    .user-card.selected { background-color: #e8f2ff; }
</style>
```

Then let JavaScript manipulate the class names:

```javascript
firstCard.classList.add("selected");
console.log(firstCard.classList.contains("selected")); // true

firstCard.classList.toggle("selected"); // The class is present, so remove it
firstCard.classList.add("selected");    // Add it again to observe the selected appearance
```

`classList.remove("selected")` explicitly removes the class as well. `selected` is a class name we chose; the browser has no built-in rule for a "selected card." Its appearance comes from the corresponding CSS. For a fixed set of visual changes, connecting JavaScript and CSS through classes lets us adjust the appearance later without also changing the logic.

We can also write `firstCard.style.backgroundColor = "lightyellow"` directly. This modifies the element's inline style; CSS `background-color` is written as `backgroundColor` with this dot-access syntax. `style` is useful for assigning specific values calculated at runtime, but it does not represent the element's complete final styling. Even if a stylesheet gives the card a background color, `firstCard.style.backgroundColor` can still be an empty string.

## Creating, Inserting, and Removing Elements

After updating an existing card, try adding Charlie. The browser can first create a separate element object, and we can then decide where to put it in the document:

```javascript
const newCard = document.createElement("li");
newCard.classList.add("user-card");

const newName = document.createElement("span");
newName.classList.add("user-name");
newName.textContent = "Charlie";

const deleteButton = document.createElement("button");
deleteButton.type = "button";
deleteButton.textContent = "Delete";

newCard.append(newName, deleteButton);

console.log(newCard.isConnected); // false: not yet connected to the document
list.append(newCard);
console.log(newCard.isConnected); // true: inserted into the current page
```

`createElement("li")` returns an `li` element object. We can set its class and content even while it is outside the current document tree. `newCard.append(newName, deleteButton)` first assembles the name and button into a small subtree, and `list.append(newCard)` attaches that subtree beneath the page's list. Creation, configuration, and insertion are separate steps; creating an element does not automatically display it on the page.

We can now compare the query result saved in the previous section with a fresh query:

```javascript
console.log(cards.length); // 2: the collection from the earlier query
console.log(list.querySelectorAll(".user-card").length); // 3: a fresh query
```

Different methods choose different insertion positions:

| Expression | Insertion Position |
| --- | --- |
| `list.append(newCard)` | The last child of `list` |
| `list.prepend(newCard)` | The first child of `list` |
| `firstCard.before(newCard)` | Immediately before `firstCard`, as its sibling |
| `firstCard.after(newCard)` | Immediately after `firstCard`, as its sibling |

To remove the node, call `remove`:

```javascript
newCard.remove();

console.log(newCard.isConnected); // false
console.log(newCard.textContent); // "CharlieDelete"
```

The card has left the document tree, but the `newCard` variable still refers to the object. We can read its content or even put it back with `list.append(newCard)`. Leaving the page does not mean the object immediately disappears.

We can now control the page structure, but these operations do not maintain business data for us. If the cards originally came from a user array, removing a card does not automatically remove that record from the array. Conversely, changing the array does not automatically update the DOM we wrote by hand. Keeping the two consistent becomes a question to address when learning events and state.

## From Data to the Page: Putting DOM Basics Together

The following complete page combines the earlier operations. Save it separately as `index.html` and open it in a browser; it does not need to be mixed with the previous Console experiments. The code generates three cards from an array, changes Alice's displayed name, and removes Bob's card:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>From data to DOM</title>
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
    <h1>User list</h1>
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
            button.textContent = "Delete";

            card.append(name, button);
            return card;
        }

        users.forEach(user => {
            list.append(createUserCard(user));
        });

        const firstCard = list.firstElementChild;
        firstCard.querySelector(".user-name").textContent = "Alice (updated)";
        firstCard.classList.add("selected");

        const secondCard = list.children[1];
        secondCard.remove();

        summary.textContent =
            "The page shows " + list.children.length +
            " users; the array still has " + users.length + " records.";
    </script>
</body>
</html>
```

The final page displays two cards, Alice (updated) and Charlie, with the first card highlighted. Below them, it says "The page shows 2 users; the array still has 3 records." `users[0].name` is still `"Alice"`, and Bob remains in the array, because our later changes affect only the DOM. The Delete buttons are currently just elements with no registered event handlers; clicking them does not remove anything.

`createUserCard` accepts a user record and returns the assembled element. The loop inserts those elements into the list. Describing how to build one card as a function, then repeating that operation over an array, connects the JavaScript syntax we learned to page structure. Creating three cards, changing a name, and removing a card all happen sequentially within this script. The browser does not necessarily paint each intermediate state separately; when opening the page, we usually see the result after execution.

The script is at the end of `body`. By the time the browser executes it, the list and summary paragraph have been parsed and can be queried. Placing the same ordinary inline script before those elements may return `null` because they have not yet entered the DOM. Later, if we move the script into `app.js`, we can also load it from `head` like this:

```html
<script src="app.js" defer></script>
```

For this external classic script, `defer` allows the browser to load the file while parsing HTML and execute it after document parsing finishes. It does not defer an ordinary inline script.

Now inspect the page with F12. The Elements panel contains two cards, although the `ul` in the source file is still empty. Elements shows the current DOM, including nodes created by JavaScript after loading. Select a card and run `console.dir($0)` in the Console to inspect it as an object with properties. `$0` is a selected-element reference supplied by Chrome and Edge developer tools, not a general-purpose variable available to page scripts.

We can also run `document.querySelector("h1").textContent = "I changed the page"` in the Console, then refresh. The heading returns to "User list" from the file, and the script runs again to produce the same two cards. This experiment saves no data: we modified the document objects at runtime, not the HTML file on disk.

The DOM operations now form a concrete sequence:

```text
HTML is parsed into a document tree
        ↓
JavaScript finds existing elements or creates new ones from data
        ↓
Read content, update properties, insert or remove nodes
        ↓
The browser updates the display using the current DOM and styles
```

What remains is to decide who triggers these operations and when. Clicking Add could read an input and create a card; clicking a card's Delete button could remove it. That leads into the event system: the browser passes user actions to event handlers, which then update data and the page.
