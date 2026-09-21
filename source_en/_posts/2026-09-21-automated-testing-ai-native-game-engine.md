---
title: "Automatic Testing and AI-Native Game Engines"
title_zh: "Automatic Testing 与 AI-Native Game Engine"
date: 2026-09-21 20:00:00 +0800
categories: ["Creative Media & Games", "Game AI & Production"]
tags: ["Automated Testing", "Game Engine", "Game AI", "AI Agent"]
author: Hyacehila
excerpt: "An agentic testing system—or an interface that lets AI genuinely experience a game and draw conclusions from it—may be central to the next stage of AI4Game, and may even determine the shape of an AI-native game engine."
description: "An agentic testing system—or an interface that lets AI genuinely experience a game and draw conclusions from it—may be central to the next stage of AI4Game, and may even determine the shape of an AI-native game engine."
excerpt_zh: "Agentic Testing System，或者说，一套让 AI 真正进入游戏、体验游戏并形成判断的接口，或许会成为下一步 AI4Game 的核心，也可能决定 AI-Native Game Engine 应该是什么样子。"
mathjax: false
hidden: false
permalink: '/blog/2026/09/21/automated-testing-ai-native-game-engine/'
lang: en
translation_key: 2026-09-21-automated-testing-ai-native-game-engine
translation_status: machine
translation_source_hash: 26b7928f136b48b86c800f2be31ff4d157cf5d8f7651aba7174ed9720310456e
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

An Agentic Testing System—in other words, an interface that lets AI genuinely enter a game, experience it, and form judgments—may become central to the next stage of AI4Game. It may even determine what an AI-Native Game Engine should look like.

## Why Talk About Testing?

Why would I suddenly discuss testing and AI-native game engines together?

### Feedback Is All You Need

As AI coding rapidly spreads through frontend, backend, client, and server development, generating an app demo from a single sentence is gradually becoming a reality. Yet the same expectation has not transferred smoothly to game development. Today's coding agents can quickly write character controls and combat logic, or even scaffold a complete game project. But when asked to finish a game independently, the result often remains a demo that merely runs. The model usually does not know whether the game is playable, whether the visuals are correct, whether the full flow works, or whether the game is actually fun.

During my three-month internship at NetEase, I developed a vague but increasingly persistent feeling: what AI game coding currently lacks may not be generation, but feedback.

The development of coding agents over the past few years has already shown that the same model becomes far more capable when it can run its own code, read errors, execute tests, observe results, and keep revising its work from that feedback. [SWE-agent](https://arxiv.org/abs/2405.15793) calls tools and interaction formats designed for models an Agent-Computer Interface (ACI). In game development, however, the equivalent loop is still immature. Important game state is distributed across scenes, assets, animation, physics, UI, and the live runtime. Many errors never become a clear `AssertionError`; they appear as “the character did not reach the right position,” “the camera clipped through the environment,” or “this level cannot actually be completed.”

If a model can only generate, it remains a generator. It begins to become a Game Coding Agent only when it can complete the loop of generation, execution, observation, verification, and modification.

That is why I became interested in game testing. Here, “testing” does not refer only to test cases in the traditional QA sense. It means an entire layer of feedback infrastructure built for agents: enabling a model to observe game state, operate the runtime, complete player flows automatically, detect anomalies, and turn those results into feedback it can understand during the next revision.

My intuition is that this may become a basic capability in AI-native game development: **giving coding agents the eyes they need to understand games.**

### Game Benchmarks and the Current State

[GameDevBench](https://arxiv.org/abs/2602.11103), an ICML 2026 work, tests whether coding agents can complete game-development tasks in the modern game engine Godot. It is somewhat like SWE-bench for game development. Its tasks come from web and video tutorials and cover common scenarios including gameplay, 2D and 3D graphics, and UI.

Unfortunately, even leading agents that perform strongly on SWE-bench and many of its variants still falter in games. The paper reports that average success falls from 51.4% on gameplay tasks to 33.0% on 2D graphics tasks, which place greater demands on visual understanding. **Pure gameplay logic is not the hardest part.** Coding agents are good at manipulating symbolic worlds, but much of a game's real state exists in visual and runtime worlds.

Researchers then added two forms of visual feedback, Editor Screenshot MCP and Runtime Video. With them, GPT-5.4's success rate rose from 41.1% to 52.0%. At minimum, this result shows that visual and runtime feedback can substantially improve game-development performance. **The perception-action-verification loop of today's coding agents was not designed for game editors.**

The gap between browser-based game development and mature commercial engines makes the problem even clearer. `Three.js` renders everything through a `canvas`, but an agent can still access the underlying content through JavaScript, the language it handles best. An Unreal project contains far more than code scripts; huge amounts of state exist only in editor assets and runtime changes. [GameEngineBench](https://arxiv.org/abs/2607.03525) collects 110 C++ tasks from nine real Unreal Engine 5 projects. Its strongest configuration reaches only 55.5% pass@1, and 31 tasks are not solved by any configuration. General models keep improving, but performance on engine-level tasks does not rise automatically or linearly with them. That is the current state of game-development agents. **When code is no longer the entire world, we need a new way to represent the world.**

### Becoming Model-Progress-Resistant

The release of [GPT-6 Astra](https://openai.com/index/gpt-6-astra/) made quite a few developers nervous. OpenAI showed it building a house model in Blender and turning it into a walkable scene in Unreal Engine 5; it can also create websites, web apps, and games directly. Each new generation of foundation models consumes part of the space previously occupied by intermediate generation layers, including some of my earlier work on UI workflows.

This jump in model capability has also made me more cautious about what I choose to build. If a direction derives most of its value from filling the final 5% to 10% gap in today's generation capabilities, the next model upgrade may quickly compress that value.

Finding a direction that becomes more useful as models improve—a Model-Progress-Resistant direction—would also help me keep my job.

Every round of model progress tests the value of intermediate generation layers. Verification infrastructure still matters because stronger models can do more and therefore produce more states that need to be checked. Better models do not necessarily make testing less important. They may instead enable larger and more autonomous engineering systems, expanding the value of automatic verification.

> “LLMs made generating code cheap. The real bottleneck is verification.” — [OpenHands](https://www.openhands.dev/blog/20260305-learning-to-verify-ai-generated-code)

## Starting with the Web to Understand Agentic Testing

Let us begin with the Web. It has already separated several parts of agent testing relatively clearly, giving us a set of concepts we can reuse later.

Two types of tools are worth discussing here. The first is an automation framework such as [Playwright](https://playwright.dev/). It emphasizes repeatable flows: open a page, locate an element, click or type, wait for state to change, and finally use an assertion to judge the result. The second is a general browser tool for agents, such as the [ChatGPT browser extension](https://learn.chatgpt.com/docs/chrome-extension). It emphasizes active operation. The agent decides what to do next from the current page and continues working across signed-in browser sessions, multiple tabs, and tasks that keep changing.

Their product goals are different, but both answer the same question: **how should an agent understand and use a web page?** Playwright wants the test process to be stable and repeatable. A browser agent has to continue toward a goal without a fixed script. Underneath, both need an interface between the browser and the model that controls what the agent can see, what it can do, and what feedback it receives after each action.

### The Mediation Layer Is First an Information Filter

A browser contains far too much information. Giving a model the complete DOM, styles, layout, event listeners, network requests, Console, Storage, and JavaScript Runtime wastes context and makes it harder to find what matters. The first job of the mediation layer is filtering. It compresses the browser's complicated and continuously changing state into a representation the agent can understand and operate on.

[Playwright MCP](https://playwright.dev/docs/next/getting-started-mcp) uses structured Accessibility Snapshots by default. The model does not read the full HTML document. It sees buttons, textboxes, headings, text, accessible names, and element references it can use in later actions. A login page, for example, may be reduced to a username field, a password field, and a sign-in button. Many nodes that exist only for layout and styling disappear, leaving a structure closer to how people understand the page's functions.

This is more stable than asking an agent to read the full HTML, and cheaper than relying on screenshots alone. Filtering always loses information, however. An element may appear as a “Submit” button in the semantic tree even when it is covered by something else. The tree also cannot tell us whether its color, position, and animation match the design. Semantic representations answer “what is it?” Visual observation still has to answer “what does it look like?”

### Agents Operate the Browser Through Different Channels

An agent generally operates a browser through two different channels. It usually does not modify the DOM directly, but instead uses interfaces that are closer to real-world use.

A conventional Playwright test locates a real element and sends clicks, keyboard input, drags, and other actions. The page's own event handlers and business logic receive those inputs and then change the DOM and application state. For testing, this path is closer to what a real user does.

Browser tools can, of course, execute JavaScript directly, read page variables, call functions, modify Storage, or use the [Chrome DevTools Protocol (CDP)](https://chromedevtools.github.io/devtools-protocol/) to inspect Console, Network, and Runtime information. CDP's Input domain can dispatch mouse, keyboard, touch, and drag events. These capabilities are closer to full debugging than to using a page like a user.

These two modes of operation should remain distinct. Clicking a button is user-level control, while changing a variable or calling an internal function is privileged control. The first is closer to the real user experience; the second makes diagnosis easier. A testing system often needs both, but it cannot treat them as the same thing.

### Three Planes We Will Reuse Later

At this point, agent testing can be divided into three planes. The same division will carry directly into the discussion of game engines.

| Plane | Question | Typical Web capabilities |
| --- | --- | --- |
| **Observation Plane** | What can the agent see? | Accessibility Snapshots, DOM, screenshots, Console, Network, Storage, Runtime state |
| **Control Plane** | What can the agent do? | Click, type, scroll, drag, navigate, execute JavaScript, modify the environment, or mock requests |
| **Verification Plane** | How does the system know the result is correct? | Assertions, URL and DOM state checks, network responses, screenshot comparison, business invariants, and human evaluation |

Observation lets an agent understand what is happening on the page. It needs textual input to avoid the huge token cost of continuous screenshots, but it also needs visual checks for rendered results that text cannot describe.

Control lets an agent change page state. Keyboard and mouse input can simulate a user, while executing JavaScript, changing the environment, or using debugging interfaces can bring the agent to a target state more quickly.

Verification turns an operation into a test. A Test Oracle cannot be ignored here: only reliable final and intermediate signals can produce an executable test. Some signals are difficult to express as assertions and require methods such as LLM-as-Judge together with visual inspection. Questions such as “can a user understand this tutorial?” or “is this process fun?” are even harder to solve with one rule and may still require model evaluation and human sampling.

### Black Box, White Box, and the More Practical Gray Box

Another set of concepts worth fixing early is Black-box, White-box, and Grey-box testing. They do not describe the size of a test. They describe how much of the implementation the testing system can see.

| Type | What the test can see | Typical approach |
| --- | --- | --- |
| **Black-box Testing** | External inputs and outputs only, without relying on implementation details | Simulate mouse and keyboard input, observe the screen or public page state, and judge whether the user flow completes |
| **White-box Testing** | Code structure and internal logic directly | Call functions, check branches and coverage, and verify internal module state |
| **Grey-box Testing** | A real end-to-end flow plus selected internal state | Operate the page with real input while inspecting the DOM, Network, Console, Storage, test hooks, or Runtime state |

Black-box testing that relies only on pixels and coordinates, or on public interfaces such as the DOM, is closest to how a person uses a page. But it is not realistic as the only strategy for large-scale CI. It is slow, sensitive to animation, resolution, and rendering changes, and gives little help in locating a failure.

Grey-box E2E keeps real input and the complete flow while collecting diagnostic signals from the DOM, network requests, logs, and Runtime. It is less “pure” than black-box testing, but usually easier to reproduce and better at telling developers which layer failed.

### Why Canvas Suddenly Makes a Web Page Look Like a Game

Ordinary web pages fit this approach well because buttons, textboxes, and text naturally exist in the DOM and Accessibility Tree. Canvas-based games change the situation abruptly. In [Three.js](https://threejs.org/docs/pages/WebGLRenderer.html), for example, `WebGLRenderer` ultimately renders a Scene and Camera into a Canvas. The browser's semantic layer may see only one canvas. Characters, enemies, buttons, and collision relationships inside it do not automatically become DOM nodes.

People can read a great deal from the image, while an Accessibility Snapshot may contain little more than a `canvas`. An agent can fall back to vision, inspect screenshots, and continue with mouse and keyboard input. But if that is the testing system's only path, it inherits all the problems of visual black-box testing: low information density, difficult diagnosis, and expensive feedback.

A web game still runs in a JavaScript Runtime. Through a controlled JavaScript interface, a test can read scene objects, character coordinates, quest state, and collision results while preserving real keyboard and mouse input so the game updates through its normal path. The external operations remain real, while internal state becomes observable. This is **Grey-box E2E**. Losing the Accessibility Tree does not mean vision is the only option. It means the game world needs another machine-readable state representation.

This is why web testing is worth discussing first. Agents on the Web already have relatively mature Observation and Control Planes, plus a Verification Plane made from assertions, screenshots, and runtime logs. The gap exposed by Canvas leads directly into games: what should an agent read when the DOM no longer represents the whole world; how should it act when clicking a button is no longer enough; and how should the system verify an outcome when navigation no longer means that the task is complete?

None of these three questions disappears inside a game engine. They only become harder.

## Mobile Apps: A New Platform, but the Same Problem

Moving from the Web to mobile apps does not fundamentally change the problem. [Appium](https://appium.io/docs/en/latest/intro/drivers/) does not directly understand every phone and application. It provides a unified WebDriver-based interface, then passes commands to a platform driver. Android commonly uses UiAutomator2, while iOS commonly uses XCUITest. Test code sees the same general commands for finding elements, clicking, typing, and taking screenshots; the platform's native automation technology still performs the actual device operation.
This is the same three-plane model repeated on mobile.

Mobile testing adds permission dialogs, foreground and background transitions, device variation, and Native/WebView contexts, but the basic shape of the testing system remains the same. The real break still occurs when the semantic layer disappears. Ordinary native controls can form a queryable UI hierarchy. Once an interface is rendered through a custom Canvas, OpenGL, Metal, or a game engine, Appium may see only a surface. The system must then either fall back to visual black-box testing or expose internal observable state and return to Grey-box E2E.

For this article, Appium matters less as proof that mobile devices can be clicked automatically than as evidence that the same testing model works beyond browsers: give the agent a machine-readable interface, give it controls close to how a user acts, and provide an independent judgment of correctness. Games still lack a unified, agent-friendly version of these three things.

## Gaming: Testing Has to Enter the Engine

Appium brings the WebDriver approach to mobile operating systems, and in doing so it exposes the boundary of that approach. With an ordinary app, the platform automation framework can still read the control tree and locate buttons, text fields, and lists. With a Unity or Unreal game, the operating system usually sees only the native window that hosts the player and a rendering surface continuously updated by the engine. Characters, enemies, levels, colliders, and quest state still exist, but they exist inside the engine's world. They do not automatically become controls that Android or iOS can query. The two sides operate at different abstraction layers.

This brings the discussion back to Gaming. Browsers can expose the DOM and Accessibility Tree for web pages, while mobile operating systems can expose a UI hierarchy for native apps. Only the game engine can expose a semantic interface for the game world. What the engine allows a testing system to observe and control, and how it lets that system judge an outcome, determines whether the rest of the system can work at all.

### What Game Engines Can Do Today

The same three planes still apply; only the object being manipulated has changed from a page to a game world. The Observation Plane needs a non-visual state representation through which an agent can query scene objects, properties, relationships, physics, animation, quest, and UI state. The Control Plane should preserve player-like keyboard, mouse, controller, and touch input while also allowing controlled debug commands and test hooks. The Verification Plane must bring internal state, runtime logs, screenshot comparison, and judgments about experience into the same testing flow.

None of these ideas began with AI. Long before the current generation of agents, giving a game a queryable object tree similar to the DOM was already a familiar approach. [AltTester](https://alttester.com/docs/sdk/latest/home.html) instruments a Unity game with an SDK so an external test can find and manipulate objects in the Editor or on a real device. NetEase's [Airtest](https://airtest.doc.io.netease.com/en/tutorial/2_Airtest_introduction/) and [Poco](https://airtest.doc.io.netease.com/en/IDEdocs/poco_framework/6_poco_sdk/) provide a more general combination. Airtest locates content through image recognition and simulates input, while Poco integrates an SDK into Unity, Cocos, or Unreal to retrieve the runtime UI hierarchy and its properties. One leans toward visual black-box testing; the other brings engine semantics back into the testing system. Together they already resemble the Observation and Control Planes described above. However, they do not seem to be used much anymore.

The engines also have formal testing systems of their own. The [Unity Test Framework](https://docs.unity3d.com/Packages/com.unity.test-framework@1.4/manual/index.html) supports tests in Edit Mode, Play Mode, and on target platforms. Unreal's [Automation Test Framework](https://dev.epicgames.com/documentation/unreal-engine/automation-test-framework-in-unreal-engine) covers unit, feature, content stress, and screenshot-comparison tests, while [Gauntlet](https://dev.epicgames.com/documentation/unreal-engine/gauntlet-automation-framework-overview-in-unreal-engine) launches and monitors more complex game sessions. Developers can use these frameworks to arrange input, read runtime state, inspect logs, and write assertions. In that sense, writing a game test is not so different from writing a Web flow in Playwright.

But none of this means that an agent can already test a game well. These frameworks solve questions that can be specified in advance: a person first imagines a path, then writes down its actions and expected result. They are effective at blocking known bugs and protecting stable flows from later changes, but they cannot cover combinations no one has thought of yet. Games keep adding levels, mechanics, and new forms of interaction, so test cases usually follow the content rather than precede it. Human exploratory testing therefore remains an expensive part of the process that is difficult to remove. Test Oracles are also harder in games than in conventional software testing.

**The remaining limitation is that games are too complex and their combinations of states grow too quickly. It is difficult for AI to write enough tests from development requirements alone, while the absence of test feedback also makes continuous iteration difficult. The game industry's shortage of automated testing and the Coding Agent's dependence on feedback meet at this point, limiting how well Coding Agents work in game development.**

Agentic Testing still needs the ability for an agent to enter the game world, keep exploring changes in state and image, and find problems that have not yet been written into a test case. The agent can explore and form a judgment first, then leave the final review and diagnosis to a person. This is how it can relieve part of the Test Oracle problem.

### How Can an Agent Play a Game?

The question now is how to make AI play a game. More precisely, it is how to make an agent explore edge conditions like a test engineer.
If the only question is whether it can keep playing, there are already several answers. A testing system has to ask more: can it accept a different objective at any time, reproduce the path precisely, and explain where a failure occurred?

Perhaps we should begin from first principles. What does a person need to play a game: the complete image, or some representation of state? How do people make decisions inside a game? Much of the work in reinforcement learning and model-based agents is choosing among these variables. My intuition is that people understand the world through vision, but retain concepts in their minds. That is also my answer.

#### Reinforcement Learning: Training One Goal to a High Level

Reinforcement learning was once one of the most important routes for game AI. [OpenAI Five](https://openai.com/index/dota-2-with-large-scale-deep-reinforcement-learning/) defeated the Dota 2 world champions through self-play, showing that RL can handle long horizons and complex action spaces. It also required ten months of training and roughly 45,000 years of simulated play, and it consumed structured state exposed by the Dota API.
A combinatorial state space in an AAA game does not make convergence theoretically impossible, but it makes data collection, exploration, reward design, long-horizon planning, and migration across versions expensive. RL works well when training one capability to a high level in an environment with a stable objective. It is a poor fit for a general QA agent that faces different testing conditions every day.

#### VLA: Seeing and Acting Like a Player

ByteDance Seed's [Lumine](https://arxiv.org/abs/2511.08892) is a representative example of this route. It reads raw pixels at 5 Hz and produces keyboard and mouse actions at 30 Hz. In Genshin Impact, it completed the five-hour Mondstadt storyline and transferred zero-shot to Wuthering Waves and Honkai: Star Rail. This shows that an end-to-end vision-language-action model can develop substantial behavioral generalization. But it is first of all a player. Natural-language instructions can tell it where to go and what to do, yet they are a weak interface for enforcing a tester's preconditions, state invariants, coverage targets, and failure evidence. Pixels alone also cannot say whether an anomaly came from quest state, collision, or an animation state machine.

#### Rules and Search: From Handwritten Rules to Learned Policy

Rules remain effective in games with clear boundaries. Traditional chess engines encode legal moves, position evaluation, and Alpha-Beta Search. Quest scripts, state machines, and navigation bots in game testing follow the same idea: they are inexpensive, stable, and easy to trace when they fail. Their coverage ends where the developer's written rules end.

[AlphaZero](https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/) learned Policy and Value through self-play under known legal actions and win conditions, then used Monte Carlo Tree Search to choose actions. MuZero learned a latent dynamics model for planning. Asking an agent to read code, extract environmental constraints, and construct testing rules is another route worth studying. Whichever approach we choose, it still needs precise state, available actions, and a clear objective signal. Board games define these conditions cleanly. Open-world games usually do not, and neither their rules nor their environments necessarily transfer to the next game.

#### LLM Chat Decision

An LLM or VLM can also use the same interface as a person. [Cradle](https://arxiv.org/abs/2403.03186) consumes screenshots and emits keyboard and mouse operations, and has completed missions lasting about forty minutes in Red Dead Redemption 2. This black-box interface requires almost no change to the game and generalizes broadly. The cost is that every step has to infer state again from pixels. Resolution, observation history, and frame count increase visual-token cost and inference latency, while obscured values and internal flags never appear on screen. Vision is useful for verifying what the player finally saw. It is a poor sole source of truth for large-scale CI.

Pure vision does not require sending every frame unchanged to a VLM. A long-running flow can maintain a rolling Visual Window containing the current full frame, keyframes with significant changes, task-relevant crops, and the latest Action Trace, while repeated frames are reduced to their differences. This is the agent's approximate estimate of the game's complete short-term state. [LongVU](https://arxiv.org/abs/2410.17434) shows that long video can be compressed according to inter-frame similarity and task queries. Testing adds one more requirement: the system should retain a raw rolling buffer and freeze the evidence before and after an anomaly. The decision window may be compressed with loss. The evidence delivered to a Test Oracle or a person cannot consist only of a model-generated summary.

Detectors such as YOLO can also participate in this compression. They can locate buttons, health bars, quest markers, or enemies and turn them into objects with classes, positions, and confidence scores. [OmniParser](https://arxiv.org/abs/2408.00203) already uses YOLOv8, OCR, and an icon-description model to reconstruct structured elements from general UI screenshots. A game could similarly build a probabilistic Visual DOM and feed events such as `health_drop`, `dialog_open`, and `player_stuck` into `if/else` rules. Unfortunately, generalization with YOLO is not a simple problem either.

The other route directly compresses the game world into a language the model can read reliably. Lap, introduced in [Towards LLM-Based Automatic Playtest](https://arxiv.org/abs/2507.09490), does not ask an LLM to guess directly from a match-3 screenshot. It first converts the board into a numeric matrix, then asks the model to choose a swap. In the study, it achieved higher code coverage and triggered more crashes. [TITAN](https://arxiv.org/abs/2509.22170) goes further on two large commercial MMORPGs: it filters and discretizes raw state such as location, quest stage, health, nearby NPCs, and inventory before giving it to the LLM for high-level action selection. Screenshots are not removed; they are used mainly for reflection and diagnosis after progress stalls.

Vision and text are therefore not mutually exclusive routes. High-frequency detectors and control policies can handle continuous images and local operations. A lower-frequency LLM can interpret goals, plan paths, and handle anomalies. Engine state provides the primary facts, while a Visual Window supplies evidence about the actual rendering and short-term changes. If games can also expose a general interaction representation similar to the DOM, and models receive more pretraining on that representation, an LLM has a better chance of driving an agent that can explore the game world freely and complete tasks.

### Test Oracles and Agentic Testing

The routes above eventually converge on the same shape: expose the system as controllable textual signals, then add selected visual signals. The engine has to provide enough support for this, but playing a game well is not the same as testing it well. What else do we need beyond the Observation and Control Planes exposed by the engine?

Letting an agent enter the game can relieve part of the Test Oracle problem. With good prompting, training, and a complete testing structure, an agent can already automate parts of an in-game workflow. Even when state combinations explode and complete coverage remains impossible, a system can learn from existing test cases and specifications, expand coverage where it can, and inspect selected scenarios more deeply. When its strategy is effective, an agent can achieve greater coverage than a person, easing the trade-off between broad coverage and targeted testing. An LLM with broad world knowledge is also well suited to handling open-ended semantic questions in a Test Oracle.

An agent's Test Oracle capabilities can be divided into three levels. The first contains relatively deterministic machine signals: crashes, NaN values in invalid states, explicit assertion failures, and deterministic hash divergence can trigger a failure directly. The second contains context-dependent anomalies. A stuck player, unreachable objective, exceeded frame budget, or clipping event has to be judged against its preconditions, duration, and design intent. The third concerns aesthetics and design experience. A model may participate in that evaluation, but the final judgment still belongs to people.

The testing system therefore has to separate three questions: what should the agent do next, is this a bug, and which layer caused it? Otherwise, the agent is only an automated player. State rules, the Visual Window, the Action Trace, and engine logs have to come together here. The model handles the open and ambiguous parts, while a person performs the final review.

These problems eventually converge. A Gameplay Agent first explores the game and produces candidate testing signals. An Oracle confirms the issue, and a diagnostic system locates its likely cause. Once confirmed, an open-ended exploration should be compressed into minimal reproduction steps, an Action Trace, and a test assertion or test-case description before entering the regression suite. When a similar problem appears later, the system can replay an existing case first and then expand new paths from it. Expensive exploration should leave behind more than a bug report; it should continue to guide later development.

## The Answer

AI Native Games, Automatic Testing, and the AI-Native Game Engine converge on the same problem here.

AI Native Games need a certain degree of intelligence in their NPCs and worlds, which requires agents that can operate game characters and participate in world evolution. Automatic Testing still needs deterministic assertions and regression CI, but fixed cases cannot exhaust a game with explosive state combinations and a timeline. Intelligent bots can supply the exploratory part, run as executors inside CI, and return the discovered paths, states, and evidence to the testing system.

By an AI-Native Game Engine, I do not mean an editor with an LLM attached. I mean an engine and its surrounding infrastructure that let an agent observe, operate, verify, and replay the game world in a structured form from the runtime upward. It should let an agent provide input like a player while reading state and debugging the system like a developer. Both Game NPCs and Testing Bots need these capabilities.

If this is going to enter CI for real, it also needs several pieces of infrastructure that do not look particularly like AI: fixed random seeds, controllable time, saved and restorable state, and replayable Action Traces. This is also a classic problem in game testing.

That is my answer.
