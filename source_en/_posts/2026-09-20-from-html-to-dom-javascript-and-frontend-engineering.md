---
title: "How Browser Interactions Work: From Events and Forms to Client-Server Communication"
title_zh: "浏览器交互如何运行：从事件与表单到前后端通信"
date: 2026-09-20 12:00:00 +0800
categories: ["Programming", "Full Stack Development"]
tags: [JavaScript, DOM, Frontend, Async, HTTP, Software Engineering]
author: Hyacehila
excerpt: "Follow a registration form through events, input validation, Promise and async/await, HTTP, and Fetch. Understand how browsers wait for remote results and update the page, then use storage, login state, and developer tools to identify where problems occur."
description: "Follow a registration form through events, input validation, Promise and async/await, HTTP, and Fetch. Understand how browsers wait for remote results and update the page, then use storage, login state, and developer tools to identify where problems occur."
excerpt_zh: "以注册表单为主线，串起事件、输入校验、Promise 与 async/await、HTTP 请求和 Fetch，理解浏览器如何等待并展示远程结果，再通过存储、登录状态和 F12 排查建立前后端边界的基本认识。"
mathjax: false
hidden: true
permalink: '/blog/2026/09/20/browser-interactions-events-forms-and-http/'
lang: en
translation_key: 2026-09-20-from-html-to-dom-javascript-and-frontend-engineering
translation_status: machine
translation_source_hash: 0867a6e84a41f6732a56a7f169a7039a0e61ea6c7839642dee8e6d9147b45131
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

## Understanding the Browser's Roles Through Registration

A registration page may look like a few inputs and a button, but several mechanisms work together underneath. HTML and CSS provide content and styling, the DOM represents the running page, events pass user actions to JavaScript, forms organize input, asynchronous code waits for results, and HTTP carries requests and responses between the frontend and backend.

[Web Frontend Basics Overview](/en/blog/2025/05/15/web-frontend-basics-overview/), [JavaScript Basics: Variables, Functions, Arrays, and Event Callbacks](/en/blog/2026/09/18/javascript-basics-variables-functions-arrays-callbacks/), and [DOM Basics: Document Trees, Element Operations, and Dynamic Pages](/en/blog/2026/09/17/dom-basics-document-tree-and-element-operations/) covered syntax and page operations separately. Here, we put them back into an interaction: first how the browser responds to an action, then how data leaves the page, is processed by the backend, and returns.

I want an understanding that helps explain problems: did the code run after the click, was the right data collected, did the request leave the browser, and what did the server return? Understanding these steps makes it easier to judge whether AI-generated code meets a requirement and describe problems accurately. This article stays at the browser and HTTP interface level; the backend's internal implementation is treated as a separate system.

## Events: How the Browser Passes Actions to Code

The DOM lets JavaScript change the page; the event system determines when those changes happen. A user might click immediately after opening a page or wait a long time before typing. Code can first tell the browser, "When this happens, run this function," then continue with other work. When the event occurs, the browser calls the corresponding handler. That is the basic idea of event-driven programming.

### Registering Listeners and Receiving Event Objects

A common registration is `element.addEventListener("click", handleClick)`: listen for `click` on this element and handle it with `handleClick`. We pass the function itself. Writing `handleClick()` would execute it immediately and pass its return value instead. An arrow function such as `event => { ... }` follows the same principle, with the handling logic written directly at the registration site.

Registration does not execute the function immediately or stop the program to wait. A listener generally remains active, and subsequent matching events call it again. The handler does not automatically become a background thread; lengthy synchronous computation can still make the page unresponsive.

The browser passes an `event` object to the handler to describe what happened. `event.type` is the event type, `event.target` is its target, and `event.currentTarget` is the object whose listener is currently running. The parameter can be named `event` or `e`; either is simply a variable receiving that object.

A few common events are enough for now: `click` handles clicks or activation of elements such as buttons, `input` responds to user edits, `submit` handles form submission, and `keydown` handles key presses. Choose the event according to whether you care about a button being clicked, input changing, or a form being submitted.

### Why an Outer List Can Receive a Button's Event

Events can propagate through the page's hierarchy. For a `click` in the ordinary DOM, the process can be simplified as:

```text
Capture phase: travel from outer ancestors toward the target
Target phase: run listeners on the target
Bubble phase: travel from the target toward outer ancestors
```

Suppose a button is inside an `li`, which is inside a `ul`. If the button and list both have ordinary `click` listeners, clicking the button first runs its handler, then runs the list's handler through bubbling. One action can be handled at several levels; it does not necessarily mean the user clicked twice.

`addEventListener` does not use capture by default, so ancestor listeners generally handle events during bubbling, while listeners on the target run during the target phase. Passing `{ capture: true }` enables capture listening; knowing that distinction is enough for now. Not every event bubbles, so the behavior of `click` cannot be assumed for every event type.

In the list's handler, `event.currentTarget` is the list, while `event.target` may be the button inside it. If the button contains an icon, clicking the icon may make that inner element the target. Where the listener is attached and what was actually clicked are separate questions.

This enables **event delegation**: register one listener on the list, then inspect the target to determine which item was clicked. New items can use the same listener as long as their events bubble to the list, without registering a listener for every added item.

### Default Behavior and Propagation Are Different

Some actions have built-in browser behavior: clicking a link navigates, and submitting a form normally sends a request according to its configuration and navigates. Registering a listener does not automatically cancel those behaviors. To take over submission in JavaScript, the `submit` handler can call `event.preventDefault()` to cancel this cancelable event's default action before running its own logic.

`event.stopPropagation()` addresses a different problem: it prevents the event from continuing along its propagation path, during either capture or bubbling. It does not cancel link navigation or form submission. Conversely, `preventDefault()` does not automatically stop bubbling. Because delegation depends on propagation, we should not add `stopPropagation()` to every handler merely to "prevent duplicates." First identify which level should not receive the event.

### An Example: Adding and Removing Users

Place this fragment inside an HTML page's `body` to try it. It reuses the DOM operations introduced earlier and adds two listeners: the form handles adding users, and the list handles deletion.

```html
<form id="user-form">
    <label for="username">Username</label>
    <input id="username" required>
    <button type="submit">Add user</button>
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
        button.textContent = "Delete";

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

Enter Alice and click Add user. The form triggers `submit`, and the handler reads the input, creates nodes, and inserts them into the list. Listening for submission also covers pressing Enter in this simple form. `required` provides the browser's nonempty check, while `trim()` rejects input containing only spaces. The name is assigned through `textContent` and displayed as plain text.

Clicking Delete causes the `click` to bubble to the list, where its handler processes the action. `closest(".delete-user")` searches upward starting with the target itself, so adding an icon inside the button later still lets the handler find the containing button. The code checks that the target is an element and the button belongs to this list, then finds and removes the corresponding `li`. Clicking a name or empty list space does not delete anything.

This code maintains no separate user array. Adding and removing users changes only the current DOM, and refreshing restores an empty list. Persistence requires storage or a backend interface; the listeners do not provide it automatically.

## Forms: Turning User Input into Submittable Data

The previous example already used a form: an input collected a name, and `submit` passed control to JavaScript. Registration, search, and profile editing follow the same basic process: collect related inputs, check their requirements, then decide how to submit them and what result to show.

### How Forms Organize Input

`form` organizes controls into one submission, controls such as `input` hold current values, and `button type="submit"` initiates submission. Listening for the form's `submit` handles both the submit button and supported Enter-key submission without duplicating business logic for each action.

An input often has both `id` and `name`, which serve different purposes. `id` locates an element and can associate a label through `label for="..."`. `name` specifies the field name used when collecting data. With `id="register-username" name="username"`, we find the element through `#register-username` but collect its value under `username`. A visible, editable input is not necessarily included in collected data; a missing `name` is one common reason for a missing field.

`type` determines the input behavior and some validation rules. `email` checks basic email formatting, while `password` masks the displayed characters. Text inputs are usually read through `value`, and checkbox selection through `checked`. These controls and checks primarily help users fill in the form. A real backend must validate independently because requests can also be sent without going through the page.

### From Input to Validation: A Registration Form

Create another test page and place this fragment inside `body`. There is no backend yet: it demonstrates collection and local validation without creating an account.

```html
<form id="register-form">
    <div>
        <label for="register-username">Username</label>
        <input id="register-username" name="username" required>
    </div>
    <div>
        <label for="register-email">Email</label>
        <input id="register-email" name="email" type="email" required>
    </div>
    <div>
        <label for="register-password">Password</label>
        <input id="register-password" name="password" type="password"
               autocomplete="new-password" required>
    </div>
    <label>
        <input name="agree" type="checkbox" value="yes" required>
        I agree to the terms
    </label>
    <button type="submit">Check registration details</button>
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
            message.textContent = "The trimmed username must contain at least 3 characters.";
            return;
        }
        if (data.password.length < 8) {
            message.textContent = "This example requires a password of at least 8 characters.";
            return;
        }

        message.textContent =
            data.username + " passed local validation; the data has not been sent to the server.";
    });
</script>
```

There are two layers of frontend validation. The browser first checks required fields, email formatting, and agreement to the terms using `required` and `type="email"`. With this example's normal submission flow, a failed check produces browser feedback, and the `submit` handler does not run.

Once browser validation passes, JavaScript collects the data and checks the trimmed username and password lengths. These are demonstration rules chosen for this example. The username is trimmed, but the password is preserved rather than having characters silently removed. Failed checks display a reason and `return`. Passing them means only that local validation succeeded, not that an account was registered.

The `input` listener merely clears old feedback when the user edits again; validation remains in `submit`. Live search can use `input` to react to changes. For text fields, `change` typically fires when a changed value loses focus, while `focus` and `blur` indicate entering and leaving the field. There is no need to handle every event; choose when feedback is useful.

### How FormData Turns Controls into Fields

`new FormData(registerForm)` collects the currently submittable values under each control's `name`. For example, `formData.get("username")` reads the username, and `formData.get("email")` reads the email. It is a snapshot taken at collection time; later edits do not automatically change an existing `FormData` object.

The field names in this example are distinct, so `Object.fromEntries(formData.entries())` converts them into a plain object. We can then access `data.username` and `data.email`. When communicating with a backend, these field names must match the API contract.

A few rules help diagnose missing or unexpected values. Controls without a `name` or disabled with `disabled` are generally omitted, as are unchecked checkboxes. Checking `agree` in this example produces the string `"yes"`, not the boolean `true`. Ordinary input fields are generally collected as strings; a number input does not make `FormData` return a number automatically. Repeated field names, such as multiple selected options, may require `getAll()` rather than conversion into a single-value object.

### Submission Leads into Asynchronous Work and the Backend

Forms can submit without JavaScript. `action` specifies the destination and `method` the submission method. The browser collects fields, sends a request, and usually navigates to the response page. Calling `preventDefault()` only cancels that default submission; it does not send a request or save data on its own. The example above ends after displaying feedback.

Once an API is connected, the handler also needs to manage the rest of the process:

```text
The user fills in and submits the form
    ↓
Browser validation → submit handler → business checks and data preparation
    ↓
Show a submitting state and send the request
    ↓
Wait for the backend to validate, process, and return a result
    ↓
Success: display the result or navigate
Failure: retain necessary input, explain the issue, and allow correction or retry
```

While waiting, the page needs states such as submitting, success, and failure, including measures to avoid repeated clicks producing duplicate requests. A network failure and an explicit backend rejection are different, and receiving a response alone is not enough to show success. We will first examine this wait, then connect the form to an explicitly defined interface.

## Asynchronous Work: Why the Page Can Respond While Waiting

### Waiting and Computing Are Different Activities

Page JavaScript normally executes on the main thread. The call stack records which functions have been entered and where execution stands. When one function calls another, the inner call must finish before execution returns to the outer function. Continuous expensive computation can therefore delay user input and page updates.

Network requests spend much of their time waiting. The browser can send requests and receive data without JavaScript polling in a loop. Code initiates an operation and arranges how to continue when the result is available. Asynchronous execution reduces the time that waiting occupies the execution flow.

A timer can simulate this wait. No network request is made below; the result simply becomes available later:

```javascript
function waitForReply() {
    return new Promise(resolve => {
        setTimeout(() => resolve("The simulated result has arrived"), 800);
    });
}

async function run() {
    console.log("Start waiting");
    const result = await waitForReply();
    console.log(result);
}

run();
console.log("Code outside the function continues");
```

The output is "Start waiting," followed by "Code outside the function continues," and finally "The simulated result has arrived." `await` pauses the remainder of `run`, while outside code continues. The timer's `800` milliseconds specify a requested delay, not an exact completion time; scheduling also affects when it executes.

### How Promise, await, and the Event Loop Work Together

A Promise represents an operation's result. It may begin as `pending` and later become `fulfilled` or `rejected`, meaning success or failure. `then` can continue after success, and failure needs corresponding handling. The function passed when constructing a Promise executes immediately; the delayed part above is the timer callback. Wrapping expensive computation in a Promise does not make it nonblocking.

An `async` function always returns a Promise. `await` continues the current function when its result is available. If the awaited Promise rejects, an exception is thrown at `await` and can be caught with `try/catch`. Code before the first `await` still runs synchronously when the function is called. This organizes asynchronous flow; it is not a switch that starts background work.

A ready result cannot arbitrarily interrupt running synchronous code. The event loop schedules subsequent work, including tasks from timers and microtasks used by Promise continuations. Consider:

```javascript
console.log("Synchronous start");

setTimeout(() => console.log("Timer callback"), 0);
Promise.resolve().then(() => console.log("Promise continuation"));

console.log("Synchronous end");
```

The output order is "Synchronous start → Synchronous end → Promise continuation → Timer callback." The current synchronous code completes first, followed by queued microtasks, then later tasks. The Promise here is already fulfilled, so its continuation can be queued as a microtask immediately.

A useful outline is "run the current task → process microtasks → update the display when appropriate → continue other tasks." The browser does not guarantee one painted frame after every task; continuous synchronous work or many microtasks can delay responsiveness. This is enough to explain why awaiting a network request usually lets the page keep responding, while a long loop can still block it.


## HTTP: What the Frontend and Backend Exchange

### The structure of a request and response

Back in the registration form, JavaScript can collect a username, email address, and password, but that data still exists only in the browser. To have the backend process it, we need to send a request. HTTP defines the structure of requests and responses. For now, focus on the destination, the operation, accompanying information, and the data itself.

Suppose the backend provides `POST /api/register`, accepts JSON, and returns the new user's public information on success. The following exchange uses the text format of HTTP/1.1, omits some headers, and uses sample values for illustration:

```http
POST /api/register HTTP/1.1
Host: example.com
Content-Type: application/json

{"username":"Alice","email":"alice@example.com","password":"<test input>","agree":"yes"}
```

```http
HTTP/1.1 201 Created
Content-Type: application/json

{"user":{"id":123,"username":"Alice"}}
```

In the request, `POST` is the method, `/api/register` is the path, and `Content-Type` identifies the request body as JSON. The data comes after the blank line. The response uses a status code to describe the outcome and includes its own headers and body. JSON is just one data format the two sides can agree on; HTTP can also carry HTML, images, or files.

The backend must actually implement `/api/register`. Writing that address in frontend code does not create registration functionality or connect a database. The backend parses the request, validates it again, runs the registration logic, and returns a response. Whether it accesses a database or another service can remain an implementation detail for now.

### Using methods, addresses, and status codes to understand problems

`GET` is usually used to retrieve resources, while `POST` submits data or performs an action. You will also encounter `PUT`, `PATCH`, and `DELETE`. For now, knowing that they are commonly used for updates and deletion is enough; the API contract determines the exact behavior.

Parts of an address also serve different purposes. In `/api/users/123`, `123` commonly identifies a user. In `/api/users?keyword=alice&page=2`, query parameters express filtering and pagination. A path such as `/api/register` is resolved against the current page's origin. If the page runs on a local development server, that path does not automatically point to a backend on another port. You need the correct API address or a development proxy.

You do not need to memorize every status code. Start with these groups:

| Status | General meaning | What to check |
| --- | --- | --- |
| 2xx, such as 200 and 201 | Success according to HTTP semantics | Whether the response meets the application's expectations |
| 3xx | Redirects or cache-related handling | The final destination and whether cached content was used |
| 400, 422, and similar codes | The request data or its meaning does not meet requirements | Field names, types, and validation feedback |
| 401 | Valid authentication is missing | Whether credentials are valid and included in the request |
| 403 | The server refuses access | Permissions or access policies; the user's identity is not necessarily established |
| 404 | The resource or route was not found | The address, path, and backend routes |
| 5xx | A problem occurred during server-side processing | The response details and backend logs |

A status code is a clue, not a complete explanation. Some APIs also define application-level status fields in JSON, so `200` does not necessarily mean registration succeeded. Conversely, `204` means success without a response body; you should not try to parse it as nonempty JSON.

We will leave the underlying network transport outside this article.

## Fetch: Connecting the Form to Requests and Results

### Sending a request, reading the response, and handling failure

`fetch` gives JavaScript access to the browser's request capabilities. It returns a Promise. `await fetch(...)` gives you a `Response` object containing status and header information, not already-parsed application data.

If the API returns JSON, follow it with `await response.json()` to read the body and parse it into a JavaScript object. That also involves waiting: receiving the response status and headers does not mean the entire body has been read. In the other direction, `JSON.stringify(data)` converts an object into JSON text for the request body, and `Content-Type: application/json` identifies its format.

One distinction matters for error handling: HTTP responses such as `404` and `500` normally do not make `fetch` throw. You need to check `response.ok`. Network failures, cancellation, or browser policy restrictions can cause `fetch` to reject. `response.json()` can also fail if the body is not valid JSON. All of these need handling, but they have different causes.

### Following the complete process with the same form

Keep the form HTML and the `register-message` paragraph from the Form section, and **replace its entire original `<script>` block** with the code below. Do not keep both submit listeners. This version starts in simulation mode, so you can observe waiting, success, and failure without sending a request or creating an account.

Real mode requires a backend that implements this contract: `POST /api/register` accepts the fields described earlier, returns JSON containing `user.username` on success, and uses an appropriate HTTP error status on failure. Once the endpoint is ready, change `demoMode` to `false` and open the page through an HTTP development server rather than treating a local file path as an API address.

```html
<script>
    const registerForm = document.querySelector("#register-form");
    const message = document.querySelector("#register-message");
    const submitButton = registerForm.querySelector('[type="submit"]');
    const demoMode = true;
    const defaultLabel = demoMode ? "Simulate submission" : "Submit registration";
    submitButton.textContent = defaultLabel;
    let submitting = false;

    async function requestRegistration(data) {
        if (demoMode) {
            await new Promise(resolve => setTimeout(resolve, 800));
            if (data.username.toLowerCase() === "taken") {
                throw new Error("Simulated failure: this username already exists. Choose another.");
            }
            return { user: { username: data.username } };
        }

        const response = await fetch("/api/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error("The server returned HTTP " + response.status);
        }

        const result = await response.json();
        if (!result || !result.user ||
            typeof result.user.username !== "string") {
            throw new Error("Unexpected response structure: missing user.username.");
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
            message.textContent = "The username needs at least 3 characters and the password at least 8.";
            return;
        }

        submitting = true;
        submitButton.disabled = true;
        submitButton.textContent = "Submitting…";
        message.textContent = demoMode ? "Simulating a wait…" : "Submitting…";

        try {
            const result = await requestRegistration(data);
            message.textContent = demoMode
                ? "Simulation completed: " + result.user.username + "; no real account was created."
                : "Registration succeeded: " + result.user.username;
        } catch (error) {
            message.textContent = "The operation did not complete: " + error.message;
        } finally {
            submitting = false;
            submitButton.disabled = false;
            submitButton.textContent = defaultLabel;
        }
    });
</script>
```

Submit valid values, and the button is temporarily disabled and displays "Submitting…". In simulation mode, a short wait is followed by a completion message. Enter `taken` as the username to trigger the failure branch. The fields remain editable, but the operation uses the `data` collected at submission time, not any later edits.

`submitting` records whether an operation is already waiting, preventing this page from starting another submission while it is pending. Disabling the button makes that state visible. `try` handles the normal flow, `catch` displays failure, and `finally` restores the button whether the operation succeeds or fails. Restoring it only on success would leave the page stuck on "Submitting…" after a failure.

The simulated branch does not call `fetch`, so no registration request appears in the Network panel. HTTP communication starts only in real mode. In a real network, a request error does not necessarily mean the server did nothing: the server may finish processing before its response is lost. Important operations such as payments also need backend protection against duplicates; a disabled button alone is insufficient.

Search fields introduce another ordering problem. A user searches for Alice, then Bob, but Alice's response arrives later and overwrites Bob's results. No code error occurs, yet the page shows stale data. Remember that requests do not necessarily finish in the order they were sent. Later, request IDs or cancellation of older requests can help ensure that only a still-relevant result is used.

## Browser Boundaries: Storage and Login State

### Why some information survives a refresh

The DOM and ordinary JavaScript variables belong to the current page's runtime and are recreated on refresh. If information can be restored, it usually comes from browser storage or is fetched again from the backend.

`localStorage` can preserve local data across refreshes, while `sessionStorage` belongs to the current tab's session. Neither is automatically sent to the backend as request data, and saving something there does not save it on the server. The application decides where page preferences, unsubmitted drafts, and backend account records belong.

Login state also involves credentials. A Cookie is a browser-stored data carrier that the browser sends with requests according to its rules. A Session usually means server-maintained session state; a Cookie may contain only its identifier. A Token is a credential the server can verify, and it can be carried in a Cookie or a request header. These are not three mutually exclusive choices. Some Cookies use `HttpOnly`: page scripts cannot read them, but the browser can still send them with requests under the applicable rules.

A page displaying "logged in" and a server accepting the request's identity are therefore separate matters. Saving `isLoggedIn = true` locally cannot grant backend permissions. If login disappears after a refresh, or a cross-origin request suddenly receives `401`, check "where the credentials are stored → whether the request carries them → whether the backend still accepts them," rather than only changing the login indicator.

## Using F12 to Locate the Problem

We can now read a registration interaction from start to finish. The user's input becomes form state; a submit event triggers a handler; JavaScript collects and validates data, then uses Fetch to send an HTTP request. After waiting for backend processing, it reads the response and updates the message and button state.

Three browser panels correspond to different parts of this flow:

| Panel | What it helps you inspect | Example question |
| --- | --- | --- |
| Elements | The current DOM, controls, and styles | Has the error message already been added but hidden by CSS? |
| Console | Execution errors and, with breakpoints, variable inspection | Did the selector return `null`? Is a response field missing? |
| Network | Request addresses, methods, data, status, and responses | Where did the form data go? Did the backend actually return JSON? |

Start near the symptom and work backward. If no request appears, check whether simulation mode is still enabled, native validation passed, the listener ran, and execution reached `fetch`. If a request was sent, inspect its URL, method, and Payload, followed by the status and Response. A request sent to the wrong page may receive HTML, causing a later JSON parsing error. If the response matches the contract but the interface looks wrong, inspect field access, state updates, and DOM operations.

This article completes one browser-side interaction flow. Modules, build tools, and frameworks address how code is organized; backends, databases, and deployment address how services are implemented and run. Those can be explored separately. For now, the goal is to identify whether an interaction problem occurs in an event, a form, waiting, a request, the backend response, or the page update.
