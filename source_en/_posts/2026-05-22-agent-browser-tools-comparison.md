---
title: 'Letting Agents Use Browsers: From Automation Scripts to Browser Infrastructure'
title_zh: 让 Agent 操作浏览器：从自动化脚本到浏览器基础设施的演进
date: 2026-05-22 17:00:00 +0800
categories:
- Agent Systems
- Agent Infrastructure
tags:
- MCP
- Tool Use
author: Hyacehila
hidden: true
excerpt: A discussion of how browser-agent tools evolved from official host-native browser bridges to automation scripts and
  browser infrastructure, covering Codex Chrome extension, Claude in Chrome, Playwright, Chrome DevTools MCP, Browser-Use,
  Stagehand, Skyvern, Lightpanda, and cloud browser platforms.
description: A discussion of how browser-agent tools evolved from official host-native browser bridges to automation scripts
  and browser infrastructure, covering Codex Chrome extension, Claude in Chrome, Playwright, Chrome DevTools MCP, Browser-Use,
  Stagehand, Skyvern, Lightpanda, and cloud browser platforms.
excerpt_zh: 从 Codex Chrome extension、Claude in Chrome 到 Playwright、Chrome DevTools MCP、Browser-Use、Stagehand、Skyvern 和云端浏览器基础设施，梳理浏览器
  Agent 工具如何围绕官方宿主边界、确定性控制、可读感知、行动编排和生产化承载逐步演进。
permalink: /blog/2026/05/22/agent-browser-tools-comparison/
lang: en
translation_key: 2026-05-22-agent-browser-tools-comparison
translation_status: machine
translation_source_hash: 22beb2908db90368f514cec7b0ce986988857b30b9a73e9cc165279405fbf9a4
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>If the tools of the Codex Chrome extension, Claude in Chrome, Playwright, Chome DevTools MCP, Browner-Use, Vercel angent-browser, Stagehand, Skyvern, Lightpanda, Brownserbase, Steel.dev, MultiOn are listed, it is easy to write into tools encyclopedia: a small section of each tool, describing functions, scenes and limits. This helps to identify nouns, but it does not explain a more interesting question: why are these tools evolving like this?</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/02/16/mcp-model-context-protocol/">MCP (Model Context Protocol)</a>、<a href="/en/blog/2026/05/17/agent-resource-collection/">Agent Extra Resource Collection: Skills, MCP Server, Plugins and Practical Tools</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>Let us start with the sequence of de-activation: first, the ability of the official host to pack the real browser, log-in, rights confirmation and secure boundaries into boxes, then the browser can be stabilized, the web site can be read by models, then action can be saved, validated and organized, and finally the browser itself can become a fully available running time resource.</p>
<h2>Why the browser became the Agent infrastructure</h2>
<p>A lot of real missions don't clean API. Checking data, filling out forms, logging backstage, downloading invoices, reproducing front-end bugs, checking network requests, running performance analysis, and eventually returning to the browser. For humans, browsers are visual interfaces; for Agent, browsers are complex situations: Dom, barrier-free trees, cut-off charts, web requests, Console logs, cookies, localStorage, farama, bullet windows, authentication codes, loading time series and permission boundaries.</p>
<p>Traditional browser automation concerns how the program controls the browser. The Agent browser tool addresses a wider issue: what the model sees, how to decide on the action, how to judge whether the action is successful, how to recover after failure, and whether successful experience can be reused.</p>
<p>I'll use a watch to hold the pulse.</p>
<table>
<thead>
<tr>
<th>Phase</th>
<th>Representative Tool</th>
<th>Core bottlenecks addressed</th>
<th>Changes relative to previous period</th>
<th>The problem that still exists.</th>
</tr>
</thead>
<tbody><tr>
<td>Official Host Browser Bridge</td>
<td>Codex Chrome extension、Claude in Chrome</td>
<td>Login re-use, permission confirmation, sensitive action confirmation, prompt exposure and real browser operations</td>
<td>From "A browser tool for Agent" to "Cholme" directly accessed by host product and placed a safe boundary on the product layer</td>
<td>Still dependent on UI; structured API and official connector safer; testing, debugging, low token expression, workflow and production still require follow-up tools</td>
</tr>
<tr>
<td>Call browser from script to Argentina</td>
<td>Playwright、Playwright MCP、Chrome DevTools MCP</td>
<td>Load user operation automation and DevTools debugging into an Agent tool</td>
<td>From author to developer to Agent, who can play "end user" or "front engineer"</td>
<td>The context is highly contaminated, and traditional automation and DevTools outputs are still being tested/debugged for humans, not the original context boundary for Agent</td>
</tr>
<tr>
<td>From full web to Agent Readable</td>
<td>Vercel agent-browser、Browser-Use</td>
<td>Controls what Agent can see, not just to make it work for browsers.</td>
<td>Semantic compression, referencing and task-related filtering of page status</td>
<td>The cost of autonomous exploration is high and implementation certainty is inadequate</td>
</tr>
<tr>
<td>From open exploration to industrial landing</td>
<td>Stagehand、Skyvern</td>
<td>Move autonomous browser operations to programable, reusable processes</td>
<td>Staagehand, Discovery Developer Mixed Taking over and Cache Self-Rehabilitation, Skyvern Business Visual Workstream and Target Check</td>
<td>Long process still depends on browser sessions, agents, playbacks and production traffic</td>
</tr>
<tr>
<td>From single operation to production level</td>
<td>Lightpanda、Browserbase、Steel.dev、MultiOn</td>
<td>Carrying of simultaneous distribution, sessions, isolation, agency, anti-crawling, cost and APIization</td>
<td>Browser from local tools to cloud infrastructure and reuseable operations Time</td>
<td>Chrome compatibility, security of access, login and wind control remain long-term challenges</td>
</tr>
</tbody></table>
<p>The focus is not on the sorting of tools, but on the level of interface they are re-engineered: from the host 's official boundary to page click and run-time diagnostics; from DOM to model readable expression; from free exploration to cache and workflow; from a single machine browser to cloud session and light runtime.</p>
<h2>Phase Zero: Official browser bridge, first handing the browser over to the host</h2>
<p>Before discussing Playwright, MCP, angent-browser or Browner-Use, there is a more advanced choice today: the browser bridge provided by Agent host itself. They are not an additional layer of CDP sealing, or allow developers to maintain a browser session, but instead allow Codex or Claude to access the already-used, log-in Chrome.</p>
<p><a href="https://developers.openai.com/codex/app/chrome-extension">Codex Chrome extension</a> It's this way. It expands the connection to real Chome profile by Codex plugins and Chrome, allowing Codex to operate Gmail, Salesforce, the company's intranet or other pages that need to be accessed using the user's login browser status. More importantly, it places many of the boundaries that were previously to be filled by engineers themselves into the product layer: domain name authorizations before accessing the new website, anallowlist and a blocklist, a visible mandate to view high-risk capabilities such as history, and a security model that treats web content as untrustworthy context.</p>
<p><a href="https://code.claude.com/docs/en/chrome">Claude in Chrome</a> It's similar (he's earlier). Claude Code can pass. <code>claude --chrome</code> Start, or use in session <code>/chrome</code> Connect to the Chrome extension. It returns the user's browser login, executes click, input, navigation and page reading in the visible Chrome window; encounters such events as login pages or CAPTCHA that require human intervention, and suspends user processing. Anthropic has been discussed separately. <a href="https://support.claude.com/en/articles/12902428-using-claude-in-chrome-safely">Safe use of Claude in Chloe</a>The focus is on the problem injection, site access, sensitive action recognition and high-risk mission boundaries.</p>
<p>So if the question is just "I want Agent to open the browser and use my login to do a web mission for me," these two official tools should be almost the first priority. They make login, authorization, visible operation, manual takeover and security confirmation part of the host product. They are more like a product-oriented road than taking Playwright, profile, running CDP, or connecting a general-purpose browser anent to a real account.</p>
<p>But this does not mean that the browser should become the first option for all tasks. The browser is still, in essence, a temporary intermediary when structured API and official connector fit incompletely. Priority to be given to these structured interfaces is usually safer, more stable and more auditable, as long as the target systems are stable API, MCP, app connéctor or official integration.</p>
<p>The official browser bridge is about "can you open the box and let Agent work in my real browser?" The tools behind this address the more detailed issues: Playwright is needed to write back tests; Chrome DevTools MCP is needed to diagnose frontend operations; angent-browser or Browner-Use is needed to compress the web page; Stagehand or Skyvern is needed to automate the process as a reusable business; and if mass mail is to be carried, it will go to Brownserbase, Steel.dev or Lightpanda.</p>
<h2>Phase 1: Call browser from script to Agent</h2>
<p>If the official host browser bridge is considered phase zero, the first phase will be attended by Playwright, Playwright MCP and Chrome DevTools MCP. They are not typical autonomous Agent frames, but they are the starting point of the browser Agent's subdivision control and debugging tool chain. Understanding their relationship cannot be based on who can point the button, but on the bottom line: CDP.</p>
<p><a href="https://chromedevtools.github.io/devtools-protocol/index.html">Chrome DevTools Protocol</a> It is the debugging and automating protocol for the Chrome, Chromium and other Blink-based browsers. It breaks down the browser's internal capabilities into domains, DOM, CSS, Network, Runtme, Debugger, Performance, Tracing, Input, Page, etc. The Chrome DevTools itself uses this protocol, and many browser automation and debugging tools are directly or indirectly related to it.</p>
<p><a href="https://playwright.dev/docs/intro">Playwright</a> It's a certainty baseline for modern Web automation. It supports Chromium, WebKit and Firefox, which can open pages, locate elements, clicks, enters, waits for network requests, intercepts, records and tracks, and can also work with test assertions, CI, reporting systems. In the Chromium/Chromome scene, Playwright and CDP are linked to the ecological bloodline and support the existing Chrome connection through CDP; it does not give developers original CDP, but higher-level locator, action, auto-wait, trace and test API.</p>
<p>This is Playwright's value: transform bottom browser protocols into stable tests and automated abstractions. As long as the page design is relatively fixed and E2E tests are done using Playwright, you can pass through the entire system process like a true user without having to call at each handwritten DOM, Network or Runtme level protocol.</p>
<p>And many times, we don't really need Agent to understand the web. The criteria for success are clear with respect to login tests, return of closing processes, checklist verification, and front-end changes to E2E. Instead, a layer of uncertainty is added to the freedom of exploration with models.</p>
<p>The new change in Age is that these certainty capabilities are beginning to be exposed through MCP. But Playwright MCP and Chrome DevTools MCP, while both are on the browser automation/debugging spectrum, have allowed models to play two roles.</p>
<h3>Different roles from same source: end user vs frontend project Division</h3>
<p><a href="https://playwright.dev/mcp/introduction">Playwright MCP</a>Pack Playwright 's automation capability into MCP Server, so that LLM can operate the web page by structuring access and elements. It treats the model as an "end user": what buttons, input boxes, check boxes are on the page, where the point is next, what is to be filled, and what is to be done. Official documents also emphasize that it is based on access snapshot and does not require visual models; typical tools include navigation, click, input, form, tab, dialog, storage, network, tracing, etc.</p>
<p>This step is not to turn Playwright into a full-fledged Argentina, but to allow browser control into an Internet loop:</p>
<pre><code class="language-text">用户目标
  -&gt; 模型判断需要浏览器
  -&gt; 调用 Playwright MCP
  -&gt; 获取页面状态和元素引用
  -&gt; 执行点击、输入、截图、等待等动作
  -&gt; 把结果返回模型继续推理
</code></pre>
<p>Chrome DevTools MCP uses the model as a "front-end engineer". It also has input automation and navigation tools, which can click, enter, navigate, and intercept; but this is not the most valuable place. It's strong in handing DevTools perspective to the coding anent.</p>
<p>Google's here. <a href="https://developer.chrome.com/blog/chrome-devtools-mcp?hl=en">Brome DevTools MCP publishing article</a>It's very straightforward: coding anent can't see what actually happened in the browser with the front-end code that he wrote. It can change files and run commands, but without browser feedback, it is not known whether the page is white screen, whether the resource is 404, whether the COREs failed, why the buttons are not moving, and why the LPC is high.</p>
<p>Based on <a href="https://github.com/ChromeDevTools/chrome-devtools-mcp/blob/main/docs/tool-reference.md">Chrome DevTools MCP tool reference</a>It exposes not only clicks and fillers, but also automating console, network, personal track, Lighthouse, heap snapshot, DOM/CSS, page navigation, screenshots and input. In other words, it makes Agent not only operate the browser, but also read DevTools.</p>
<h3>Context difference: Interactive tree vs running time evidence</h3>
<p>Playwright MCP gives the model a context more like the operational semantic structure of the page. It will convert the page to access snapshot and assign interactive elements. Models see titles, buttons, text boxes, check boxes and their references, and are therefore more suitable for page operations at the logical level: login, filling in, clicking on next steps, capturing visible data.</p>
<p>The Chrome DevTools MCP gives the model a context more like the browser running time evidence. It is suitable for checking the reasons for the configation of console, network requet, DOM/CSS details, performance track, Lighthouse results, heap snapshot. It's not about the next point, but about why the page is in this state.</p>
<p>So it is more natural to want the AI auto-entry system, fill out the report forms, and grab the visible data on the page; it is more natural to want the AI React memory leak, analyze the API loading performance, locate the API request failed or the CORS problem.</p>
<p>The problem with this set of tools is here. They allow the browser to finally be called by Agent, but still send the browser in the form of "test engineer" and "front engineer" in the context. Playwright MCP has been compressed with access snapshot, ref and visual mode, but it still inherits the perspective of the E2E automation tool. The Chrome DevTools MCP has a larger amount of information, many logs, requests, styles, strips that are relevant only for a particular issue; if they are not screened, the model is easily taken away by low-value details.</p>
<p>The first stage addresses the question of whether Agent can use the browser tool, and does not address the question of the form in which the browser should enter the Agent context. This is where the second phase begins.</p>
<h2>Phase 2: From full web page to Agent readable expression</h2>
<p>The second phase is not intended to encapsulate another CDP or add an additional operational interface to the browser, but to re-formulate the browser state into the context of the Agent-consumptionable task.</p>
<p>Playwright provides a strong browser control base, but if Playwright is handed over directly to AI, a few problems will arise.</p>
<p>The first is context-related disaster. The real Web application has a large number of layout containers, style class names, frame status, hydration data, burial nodes and invisible elements; and the data will expand rapidly. Humans can be screened when debugging, not in the context of the model.</p>
<p>The second is that the state is at different rhythms. The traditional Playwright script is more like a one-time run process, while Agent needs to look at, think about, and walk. It requires a browser session to be permanent and to read the current status at every step before deciding on the next step.</p>
<p>The third is the vulnerability of the position. CSS self-prospect and XPath are suitable for developers to write definitive scripts, but not for models to fix themselves after failure. Once the DOM hierarchy, class name or component structure has changed, the model is easily in an awkward state: know where it wants to be, but how to locate it.</p>
<p>So the question is, how do we turn Playwright into a user that can control it, into a space that Agent can understand and continue to operate? Vercel Labs <code>agent-browser</code> And Browner-Use gives two different routes: one to text down, one to Agent closed.</p>
<h3>Vercel anent-browser: Text downside and CLI philosophy</h3>
<p>The name of Vercel Labs. <a href="https://github.com/vercel-labs/agent-browser">agent-browser</a> It represents a light quantitative direction. It is an automated CLI browser for AI delegates, working through the CDP of the Chrome/Chromium, with the core form being local CLI plus permanent presence in daemon. The common process is first. <code>agent-browser snapshot</code> Get the access tree, get it back. <code>agent-browser click @e2</code>、<code>fill @e3</code> This short command completes the action.</p>
<p>The first thing to think about is... <code>context budget</code>I'm sorry. The model does not need to read the full DOM, nor does it need to see the screenshot, if you see the titles, links, buttons, input boxes on the page, and the corresponding refs for these elements. For the code anent in the terminal, the page is compressed into short text and the action is compressed into short references.</p>
<p>A typical snapshot is probably:</p>
<pre><code class="language-text">- heading &quot;Example Domain&quot; [ref=e1]
- link &quot;More information...&quot; [ref=e2]
</code></pre>
<p>The model does not need to generate complex self-portor, just output:</p>
<pre><code class="language-text">agent-browser click @e2
</code></pre>
<p>This ref-based operation changes the interface between Agent and the web page. Past models may generate a selector:</p>
<pre><code class="language-text">click(&quot;button.submit.primary:nth-child(2)&quot;)
</code></pre>
<p>Now it just needs to choose a semantic node:</p>
<pre><code class="language-text">click @e2
</code></pre>
<p>This is a small step, but it has a great impact on the cost and the stability of the actions of the token. Agent does not need to guess which Selector is the most appropriate in a bunch of CSS names, but is based on the accessibility syntax and the referencing id operating page.</p>
<p>CLI forms are also critical.<code>agent-browser</code> More like a terminal capability that can be called by such tools as Claude Code, Cursor, Codex, instead of complete autonomous Browner anent. The browser maintains a session in daemon, and AI can explore the web page as if it were a shell command; if AI is stuck, humans can take over the command directly from the same terminal, and light it through a step and let it continue.</p>
<p>This route can be summed up as "de-dimensional": from complex DOM, Trace and DevTools panels down to a short set of text snapshots and action commands. It is suitable for working ant-page light-checking, information search, functional validation and local interaction. The border is clear:<code>agent-browser</code> Address low token controls and terminals that can be taken over and do not have the responsibility for complete mission planning. It makes models work on web pages cheaper, but it's not Web Age brain.</p>
<h3>Browner-Use: Organize browser into Agent environment</h3>
<p>Browner-Use is going another route.<a href="https://github.com/browser-use/browser-use">browser-use</a> The location is to make the site accessible to the AI delegates. It is not just about page snapshots, but about packaging browsers into a model that can circulate observations, reasoning, actions, and the environment of the results. The model is not facing a fixed script, but a goal and changing web state.</p>
<p>This frame turns the browser into an active browser:</p>
<pre><code class="language-text">观察页面
  -&gt; 推理和规划下一步
  -&gt; 执行动作
  -&gt; 读取页面结果
  -&gt; 判断任务是否完成或是否需要重试
</code></pre>
<p>And this and... <code>agent-browser</code> The difference is great.<code>agent-browser</code> A sharp command line tool, the model requires a decision on how to organize multistep missions; Browner-Use is more like a browser mission framework, combining observation space, action space, state summaries and task loops. It combines DOM/HTML status, interactive elements, using a screenshot or visual information, where necessary, to help the model to move on around natural language targets.</p>
<p>In contrast to Playwright, the Browner-Use progress is not a different way of clicking, but a change in the task pattern. Playwright is a process that developers have written in advance; Browner-Use is faced with an open goal like "Do this web mission for me." The model is no longer just a call to a fixed script, but a loop decision is made in the page state.</p>
<p>It also integrates visual understanding, allowing VLM to support understanding of the page structure, rather than relying entirely on DOM. More precisely, the centrepiece of Browner-Use is the browser state and action cycle; the page structure, interactive elements, transects, visual abilities are all just for this cycle.</p>
<p>The costs are also direct: context and model calls are more costly, implementation paths are more uncertain and long missions are more easily drifting. The more open, the more necessary it is to validate, cache, heal and bind business processes.</p>
<h3>A comparison of the two routes</h3>
<table>
<thead>
<tr>
<th>Dimensions</th>
<th>Playwright</th>
<th>Vercel agent-browser</th>
<th>Browser-Use</th>
</tr>
</thead>
<tbody><tr>
<td>Core philosophy</td>
<td>Human Developer writing certainty automation</td>
<td>Token CLI tool to reduce the web page to low</td>
<td>Organize browser as Agent task environment</td>
</tr>
<tr>
<td>Page Representation</td>
<td>Dom, Locator, Trace, screenshot, incident</td>
<td>Accessibility snapshot、<code>@eN</code> ref, short text</td>
<td>Page structure, interactive elements, state summaries, combined with screenshots/visual</td>
</tr>
<tr>
<td>Positioning</td>
<td>CSS selector、locator、XPath</td>
<td>semantic nodes, e. g. <code>@e2</code></td>
<td>Agent selects actions based on observations</td>
</tr>
<tr>
<td>Token Cost</td>
<td>It's very high for the model.</td>
<td>Very low, pre-empting context</td>
<td>Medium to High, depending on task cycle and visual use</td>
</tr>
<tr>
<td>Autonomy</td>
<td>Low, dependent on developers scripts</td>
<td>, provide comboable commands</td>
<td>High, built-in multi-step job cycle</td>
</tr>
<tr>
<td>Human taking over.</td>
<td>Change code or debug script</td>
<td>Click directly on the terminal for the CLI command</td>
<td>Relative loop more like taking over a running</td>
</tr>
<tr>
<td>Typical scene</td>
<td>CI/E2E, Stable Process</td>
<td>Lightweight web operation for coding anent</td>
<td>Open web missions, multiple steps to implement autonomously</td>
</tr>
<tr>
<td>Legacy issues</td>
<td>Context and positioning are not appropriate for models</td>
<td>Not responsible for complete planning and operational validation</td>
<td>Cost, drift and reliability pressure are higher</td>
</tr>
</tbody></table>
<p>The second phase is not really a direction, but a two-part cut. The progress of angent-browser is in the down-dimensional compression of browsers into end tools that are easy to use in large text models. Browner-Use progresses in organizing the rings, packaging the browser into an enabling environment for observation, reasoning, action and inspection.</p>
<p>Both answered the contextual questions left by the first phase and left new questions.<code>agent-browser</code> Solving the context economy without being responsible for the full mission planning; Browner-Use provides an autonomous cycle, but brings costs, drift and validation pressure. Phase III, Stagehand/Skyvern, is in the process of re-enacting autonomy into caches, validation, visual recognition and business process constraints.</p>
<h2>Phase III: from open exploration to industrial development Land</h2>
<p>Phase II addressed the question of how Agent viewed the web page and how to recycle decision-making, but not how to land the industry.</p>
<p><code>agent-browser</code> Low token operations and terminals are resolved but they are not responsible for complete mission planning. Browner-Use provides an autonomous cycle, which also brings costs, drift and stability pressures. Phase 3 moves from "self-contained" to "land-defeating": how to put AI into a manageable process, how to reduce repetition of reasoning, and how to allow non-developers to configure complex web tasks.</p>
<p>Staagehand and Skyvern are two different routes. Staagehand, for developers, mix Playwright and AI into a more productive SDK; Skyvern, for business automation, package visual priority browsers into workstream platforms.</p>
<h3>Staagehand: Developer's route, mixed takeover and implementation cache</h3>
<p><a href="https://www.browserbase.com/stagehand/">Brownserbase's introduction to Stagehand</a>It's called the open source for Browner treaties. SDK, the core originals include <code>act</code>、<code>observe</code>、<code>extract</code> and <code>agent</code>I'm sorry. The idea behind these original words is clear: not to choose between Playwright and Argentina, but to mix definitive scripts with AI reasoning.</p>
<p>Staagehand is more like an AI enhanced Playwright than a black box antroneous anent. Developer still writes codes, defines processes, processes input output and business validation; AI intervenes only when the page is unstable, semantics difficult to locate, and data extraction complex.</p>
<p>A typical Stagehand style process is:</p>
<pre><code class="language-text">用代码控制主流程
  -&gt; 用 observe 找当前页面可行动作
  -&gt; 用 act 执行自然语言动作
  -&gt; 用 extract 按 schema 提取结构化数据
  -&gt; 必要时交给 agent 处理更长流程
  -&gt; 成功路径尽量缓存和复用
</code></pre>
<p>That's the mix takeover. Opens the web page, fills the fixed account number, enters a backstage path, and these definitive steps continue to be coded. Click a button that always changes the script, extract structured fields from complex pages, and determine what action is possible on the current page, which is handed over to the unstable links <code>act</code>、<code>observe</code> and <code>extract</code>。</p>
<p>Another key point of Stagehand is the cache. Fully autonomous Agent often re-examines the model at every step, which is wasted in a repetitive process. Staged's Cacheline is: First lets the model resolve action or angent step, saves the reusable action results after success; and, when a similar page structure is encountered, first uses the Cache, and then reduces or skips the LLM call. Repositioning and repairing the model when the cache fails or the page changes.</p>
<p>This design is practical: AI is no longer deduced at every step, but only intervenes when there is uncertainty or the cache is not working. It is more suitable for professional developers, for existing business streams, and for stable but small-change web-page tasks. Staagehand is not replacing Playwright, but making Playwright more resilient to real changes in the web.</p>
<h3>Skyvern: Businessline, Visual Priority Workstream Platform</h3>
<p>Skyvern deals with another category of issues: complex business portals and non-developer automation.</p>
<p>Such websites often have dynamic forms, bullet windows, flame, break pages, upload downloads, unsync buttons and strange layouts. It's often not enough for self or barrier-free trees because key controls are visible and in Dom they're probably just a norole nest. <code>div</code>。</p>
<p><a href="https://www.skyvern.com/docs/developers/getting-started/introduction">Skyvern Document</a>Specifically, it uses LLM and computer vision automating Browner-based workworkworks. The official implementation cycle described is: screenshot, extraction DOM, LLM reasoning, execution action, inspection of targets, repetition. Understand it as a more accurate view-first: visual and transective are core sensory portals, Dom are auxiliary messages, and the ultimate goal is not to treat vulnerable XPath or self-ecter as the only truth.</p>
<p>This route is closer to AI RPA or workworkloading platform. Skyvern can be triggered by dashboard, API or workflow; workflow block can express the steps of login, navigation, downloading, extraction, loop, code, etc. For business users, it doesn't require you to be well compiled in code as Staagehand did. Line <code>act</code> and <code>extract</code>, instead of better configuration of " login backstage " with natural language targets and visualization processes -&gt; Find outstanding invoices -&gt; Download PDF -&gt; Sending mail's such a job.</p>
<p>Skyvern automating browsers from "developer scripts" to "business process organization". It is more friendly to complex portals, unsync pages and dynamic UIs, and is closer to automated tool forms that non-developers can understand.</p>
<p>The costs are also evident. Visual and multi-modular reasoning is slower and more expensive; the longer the flow of work, the more it requires clear target checks, manual points of intervention, authority management and abnormal treatment. When it comes to login, 2FA, authentication codes, payments or sensitive data, it is not simply possible to believe that “the visual model will solve everything”, and that workflow design and authority governance are still needed.</p>
<h3>Coordinates for Phase III</h3>
<table>
<thead>
<tr>
<th>Dimensions</th>
<th>Playwright</th>
<th>Browser-Use</th>
<th>Stagehand</th>
<th>Skyvern</th>
</tr>
</thead>
<tbody><tr>
<td>Technical positioning</td>
<td>Automation of certainty API</td>
<td>Open</td>
<td>AI Enhancement Playwright SDK</td>
<td>WorkFlow integration plan</td>
</tr>
<tr>
<td>Human intervention point</td>
<td>Prepare a full script</td>
<td>Initial objectives and operational monitoring</td>
<td>Combining Codes with AI</td>
<td>Dashboard / workflow blocks / Natural Language Target</td>
</tr>
<tr>
<td>Implementation speed</td>
<td>Come on.</td>
<td>Slow to medium, relying on model cycles.</td>
<td>Quick, close to certainty on Cache.</td>
<td>Slower, dependent on visual and multistep checks</td>
</tr>
<tr>
<td>Token Cost</td>
<td>Low, but not suitable for direct modeling</td>
<td>Medium to High</td>
<td>Low to Medium, repeat process can be reduced by cache</td>
<td>High, more costly for visual and long processes</td>
</tr>
<tr>
<td>Retrofitting</td>
<td>Low, dependent on self/loctor</td>
<td>Medium to high, depending on observation and action design</td>
<td>High, with AI positioning, self-rehabilitation and cache verification</td>
<td>High, based on visual priority and target check</td>
</tr>
<tr>
<td>Target users</td>
<td>Test/Automation Engineer</td>
<td>Agent Developer</td>
<td>Professional developers and production of water currents</td>
<td>Business Automation Users, Operating Teams, RPA scene</td>
</tr>
<tr>
<td>Legacy issues</td>
<td>Weakness, context, inappropriate models</td>
<td>Cost, drift, insufficient validation</td>
<td>Still need code and browser running time</td>
<td>High cost, slow pace, complex authority and unusual governance</td>
</tr>
</tbody></table>
<p>This phase moves the browser Agent from "can explore" to "can complete business processes." Staagehand addresses the controlled, cache and low-cost side of the developers; Skyvern addresses the visual workflow and non-developer availability of the business side.</p>
<p>However, phase III still defaults that a browser can be stabilized. As long as it's going to production, the assumption becomes heavier: how does the browser session manage? How does the simultaneous distribution expand? How does the login become isolated? What about agent and authentication codes? How do we put back the track of failure? That is the fourth stage.</p>
<h2>Phase 4: from single operation to production browser operation Time</h2>
<p>When the browser Agent changes from a demo to a service, the question goes from "Can you operate a web page" to "Can you stabilize the browser session, co-dispatch, agent, login, replay, isolate and cost"? The browser is no longer a tool, but a running time resource.</p>
<ul>
<li><a href="https://lightpanda.io/docs/">Lightpanda</a>: to solve the problem of the browser runningtime too heavy, for the lighter, higher-than-supplendent AI-native/headless Browner.</li>
<li><a href="https://docs.browserbase.com/use-cases/agents">Browserbase</a>: Solve the hosting browser session, debugging, replaying, Agent Identity and Stagehand ecological carrying.</li>
<li><a href="https://steel.dev/">Steel.dev</a>: solving the infrastructure problems of the browser, the proxy, the back-up, the captcha, the cloud session, etc.</li>
<li><a href="https://docs.multion.ai/">MultiOn</a>: further aPI-based browser operation, allowing developers to submit a Web action in the natural language, rather than managing every click.</li>
</ul>
<p>This phase is only one example of the fact that the browser Agent will eventually be called from tools to the infrastructure layer. The product selection is not the focus of this paper, but it is important to see that the production load itself becomes a stand-alone issue.</p>
<h2>Two real technology main lines.</h2>
<p>After reading these tools in stages, two more questions can be abstracted. The first question is: What did Agent see? The second question is: How does Agent do it? The evolution of the browser Agen is essentially a re-interface between the two layers.</p>
<p>But before the sensor level and the action level, the official browser bridge adds a product boundary: whose browser, whose login status, who authorizes, who confirms sensitive actions and what security model the web content is read. The value of Codex Crome extension and Claude in Crome is first here, not just the extra set of clicks and fills.</p>
<p>The sensory layer answers: in what form the browser states into the context of the model.</p>
<p>The earliest automations were based on Dom, Locator and DevTools telemetry. CDP exposed the browser's internal state, Playwright packaged them into tests and automation abstractions, while Chome DevTools packaged them into debugging perspectives. The information here is very strong: Dom, CSS, Console, network, Trace, performance, heap snapshot. They are, however, first for human engineers and testing frameworks, not for the context of the model, so that both the robust observational capacity and the context pollution occur simultaneously.</p>
<p>Playwright MCP starts to compress the first layer: it turns the page into an interactive semantic tree with accessibility snapshot and ref. The model is no longer looking at complete HTML, but rather the button, input box, title, link and reference ID. This is the model that works like an end user: "I want which buttons" "I want which input boxes."</p>
<p><code>agent-browser</code> Push that direction even more extreme. It compresses the page snapshot into a very short text, turning the elements into <code>@eN</code> Reference, priority service token economy. For local coding parties, this is the equivalent of turning the browser into a low-noise, low-cost text interface.</p>
<p>Browner-Use senses not just a single snapshot, but a state summary in the mission cycle. It needs to tell the model at every step: what the current page is, what interactive elements are, what happens after the last step is completed, and if necessary, can be combined with a screenshot or visual information. It does not focus on the smallest token, but rather on supporting continuous decision-making.</p>
<p>Staagehand's perception serves the uncertain nodes in the code process.<code>observe</code> The blogger says that the government is not a party to the law, but that it is not a party.<code>extract</code> The answer is, "What structured data can I get from here?" It does not give the entire web page to Agent, but instead transforms local uncertainties into structures that can be addressed by models in the determination of the code skeleton.</p>
<p>Skyvern's perception is more selective. It understands complex business pages with screenshots and visual information, while reducing errors with DOM support and target checks. This sense is closer to humans than pure semantics.</p>
<p>This can be written in:</p>
<pre><code class="language-text">host-native browser bridge / signed-in state / permission gates
  -&gt; CDP / DOM / DevTools telemetry
  -&gt; Accessibility Tree / ref
  -&gt; compressed text snapshot / @eN refs
  -&gt; task-oriented browser state
  -&gt; 视觉 + DOM
  -&gt; engine-native browser state
</code></pre>
<p>The second level is the operational level. The action level answers: how the model turns the intent into a browser's action.</p>
<p>The official browser bridge puts the action back in the host product: the action occurs in a real browser that is visible to the user, and access, authentication, sensitive submission and site authorization can be returned to human identification. It does not seek programming at the bottom, but rather the availability and competence of ordinary tasks.</p>
<p>Playwright's actions are script: self, locator, click, bill, haw, auto-wait. It is stable, provided that the process and page structure are known to the developers.</p>
<p>Playwright MCP and Chrome DevTools MCP turn these capabilities into tools for call. Playwright MCP user-actions: navigation, clicks, input, waiting, screenshot. Chrome DevTools MCP Project-wise Diagnostic: Read the console, network, Dom/CSS, personale, and then add to the automated recurrence of the necessary input. They all access the browser capacity to the model, but they still do not fully address the problem of how the context of the action is condensed.</p>
<p><code>agent-browser</code> Compress action into CLI/ref command. Models don't need to create complex models. Just... <code>click @e2</code> or <code>fill @e3</code>I'm sorry. This action interface is suitable for local coding anent and facilitates direct human takeover at the terminal.</p>
<p>Browner-Use put the action in ant loop. The model is not a call for a single click, but is a continuous observation, planning, execution, inspection until the mission is completed or a re-test is required. This changes the browser operation from a one-step tool to a multi-step task execution.</p>
<p>Staagehand takes the operation back into the developer's control. The certainty code is responsible for the main process, AI for unstable actions; successful actions or angent step can cache and then the model is repaired when the cache fails. This allows for both AI flexibility and as close as possible to the cost and speed of the script.</p>
<p>Skyvern organizes the action into a workflow. It does not require users to manage each self-portor like developers, but to organize missions through workflows, natural language targets and visual checks, closer to operations RPA.</p>
<p>Brownserbase, Steel.dev and MultiOn push the action to the cloud-side executive level. Brownserbase/Steel.dev cares about session, agent, playback, hair and isolation; MultiOn further encapsulates the action as Web action API, allowing developers to submit assignments, rather than manage each click.</p>
<p>This can be written in:</p>
<pre><code class="language-text">host-native browser action / human confirmation
  -&gt; script / selector
  -&gt; MCP tools
  -&gt; ref / action primitives
  -&gt; agent loop
  -&gt; hybrid cached actions / workflow blocks
  -&gt; cloud execution / Web action API
</code></pre>
<p>The sensory layer determines whether the model is visible and the operational layer determines whether the system is working smoothly. The official browser bridge places these two layers within the host product's login, access and security boundaries. Tool evolution is not a substitute for B from A, but is constantly changing interfaces on these layers.</p>
<h2>The tool selection should go back to the level of your card.</h2>
<p>So the selection should not start with "What tool is hotter" but with "What level is my bottleneck?"</p>
<p>If the target system already has structured API, official app connéctor, MCP server or stable integration, look at these interfaces first. They are usually safer, more stable, more auditable than browsers and are not easily carried away by the UI re-edited and prompt injection.</p>
<p>If you're just going to have Agent do your job on a log-in page, look at Codex Crope extension and Claude in Crope. They are not the most sophisticated browser automated language, but as a real browser operating capability, open-boxing, access reuse, authorization confirmation, manual takeover and secure borders are more complete.</p>
<p>If you're missing more solid controls, then playwright and playwright MCP. Known processes, E2E tests, CI returns, do not need to be forced to be given to autonomous Agent. Further, Playwright MCP is more natural if the goal is to complete the process like a normal user.</p>
<p>If you're missing debug feedback, look at Chrome DevTools MCP. It's best for the coding anent to read console, network, Dom/CSS, performance and Lighthouse in real Shrome, back to the code fixes. If the goal is to get the model to diagnose the page like a front-end engineer, then the Chrome DevTools MCP is more natural.</p>
<p>If you lack the Agent Readable Pages, or you have found Playwright/ DevTools output contaminating the context, see Vercel angent-browser and Browner-Use. The former are lightweight, low token, local terminal workflows; the latter are full browser mission environments and autonomous exploration. This question does not stop at Playwright MCP or Chrome DevTools MCP to continue stacking more DevTools, Trace or DOM outputs, but instead turns to a clearer context compression and task-related filter.</p>
<p>If you're missing action caches and process layout, look at Stagagehand and Skyvern. The former is suitable for professional developers to mix Playwright with AI reasoning and reduce the cost of duplicate processes by cacheing; the latter is appropriate for the complex business portal and visual priority workflow automation, closer to AI RPA.</p>
<p>If you're missing a production load, look at Lightpanda, Brownserbase, Steel.dev, MultiOn. Lightpanda solves runtime costs, Brownserbase and Steel.dev solves cloud browser infrastructure, MultiOn solves higher-level Web action API.</p>
<p>So, instead of choosing one in a row, you can start by locating the stage of the browser Agent's tool chain:</p>
<pre><code class="language-text">有结构化 API / 官方 connector -&gt; API / connector
只想让 Agent 操作已登录浏览器 -&gt; Codex Chrome extension / Claude in Chrome
控制不了浏览器        -&gt; Playwright / Playwright MCP
看不到真实运行时      -&gt; Chrome DevTools MCP
上下文被浏览器噪声污染 -&gt; agent-browser / Browser-Use
行动需要被缓存/编排   -&gt; Stagehand（开发者混合接管）/ Skyvern（业务视觉工作流）
生产运行扛不住        -&gt; Lightpanda / Browserbase / Steel.dev / MultiOn
</code></pre>
<h2>Concluding remarks</h2>
<p>The browser was a visible application and is now becoming the Internet-based environmental interface for Agent.</p>
<p>The official browser bridge has made the Browner anent from engineering experiments to direct-use product capacity. The Codex Cropesion and Claude in Cropme solves the problem of "Let's let Agent work in my real browser." This matter: re-entry, access confirmation, retention of manual takeovers and integration of prompt action and sensitive actions into the safe boundaries of the host product.</p>
<p>After this line, Playwright resolves certainty control, while Chome DevTools MCP addresses real running feedback, but they also bring a lot of browser information into Agent's context. Agent-browser and Browner-Use solver model readable web-page status, Staagehand resolves the hybrid take-over and execution cache on the side of the developers, Skyvern addresses the visual workflow on the side of the operation, Lightpanda, Brownserbase, Steel.dev, MultiOn addresses the running time and platformization of production.</p>
<p>So this is not the story of "AI Alternative Automation Script." More precisely, the browser was dismantled into layers of interfaces: the host ' s official security boundaries, a controlled environment, readable status, verifiable actions, and loadable running time.</p>
<p>Once these interfaces stabilize, the Agent Operator Browser will no longer be a fragile automated script, but will be an AI system access to the real Web world.</p>
