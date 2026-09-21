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
translation_source_hash: 8028f9cb562d0b86f1cf69c12b626a78d8a13f4eb36449b4ef534f0d1a00ee42
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

An Agentic Testing System—in other words, an interface that lets AI genuinely enter a game, experience it, and form judgments—may become central to the next stage of AI4Game. It may even determine what an AI-Native Game Engine should look like.

## Why Talk About Testing?

Why would I suddenly discuss testing and AI-native game engines together?

Before getting into the topic, I want to start with what led me to write this article.

### Feedback Is All You Need

As AI coding rapidly spreads through frontend, backend, client, and server development, generating an app demo from a single sentence is gradually becoming a reality. Yet the same expectation has not transferred smoothly to game development. Today's coding agents can quickly write character controls and combat logic, or even scaffold a complete game project. But when asked to finish a game independently, the result often remains a demo that merely runs. The model usually does not know whether the game is playable, whether the visuals are correct, whether the full flow works, or whether the game is actually fun.

During my three-month internship at NetEase, I developed a vague but increasingly persistent feeling: what AI game coding currently lacks may not be generation, but feedback.

The development of coding agents over the past few years has already shown that the same model becomes far more capable when it can run its own code, read errors, execute tests, observe results, and keep revising its work from that feedback. SWE-agent calls this model-facing mode of interaction an Agent-Computer Interface. In game development, however, the equivalent loop is still immature. Important game state is distributed across scenes, assets, animation, physics, UI, and the live runtime. Many errors never become a clear `AssertionError`; they appear as “the character did not reach the right position,” “the camera clipped through the environment,” or “this level cannot actually be completed.”

If a model can only generate, it remains a generator. It begins to become a Game Coding Agent only when it can complete the loop of generation, execution, observation, verification, and modification.

That is why I became interested in game testing. Here, “testing” does not refer only to test cases in the traditional QA sense. It means an entire layer of feedback infrastructure built for agents: enabling a model to observe game state, operate the runtime, complete player flows automatically, detect anomalies, and turn those results into feedback it can understand during the next revision.

My intuition is that this may become a basic capability in AI-native game development: **giving coding agents the eyes they need to understand games.**

### Game Benchmarks and the Current State

GameDevBench was published at ICML 2026 to test whether coding agents can complete game-development tasks in a modern game engine, Godot. It is somewhat like SWE-bench for game development: the tasks are not complicated, but they cover common development scenarios.

Unfortunately, even leading agents that have nearly saturated SWE-bench and many of its variants falter in games. The success rate is about 51.4% for gameplay tasks, while 2D graphics tasks, which place greater demands on visual understanding, reach only about 33.0%. **Pure gameplay logic is not the hardest part.** Coding agents are good at manipulating symbolic worlds, but much of a game's real state exists in visual and runtime worlds.

Researchers then gave the agent two additional capabilities: an editor MCP and runtime video. With them, GPT-5.4's success rate rose from 41.1% to 52.0%. This provides early evidence of a weakness in generative AI for game development: **the perception-action-verification loop of today's coding agents was not designed for game editors.**

The gap between browser-based game development and mature commercial engines makes the problem even clearer. `Three.js` renders everything through a `canvas`, but an agent can still access the underlying content through JavaScript, the language it handles best. An Unreal project contains far more than code scripts; huge amounts of state exist only in binary editor files and runtime changes. GameEngineBench is a benchmark for Unreal Engine, and performance can actually regress as models improve. That is the current state of game-development agents. **When code is no longer the entire world, we need a new way to represent the world.**

### Becoming Model-Progress-Resistant

The release of GPT6-Astra made quite a few developers nervous. It can use SVG to imitate human brushwork, operate Blender to create high-quality models, and reconstruct object placement in scene-building tools from reference images. Each new generation of foundation models consumes part of the space previously occupied by intermediate generation layers, including some of my earlier work on UI workflows.

This jump in model capability has also made me more cautious about what I choose to build. If a direction derives most of its value from filling the final 5% to 10% gap in today's generation capabilities, the next model upgrade may quickly compress that value.

Finding a direction that becomes more useful as models improve—a Model-Progress-Resistant direction—would also help me keep my job.

Every round of model progress tests the value of intermediate generation layers. Verification infrastructure still matters because stronger models can do more and therefore produce more states that need to be checked. Better models do not necessarily make testing less important. They may instead enable larger and more autonomous engineering systems, expanding the value of automatic verification.

> “LLMs made generating code cheap. The real bottleneck is verification.” — OpenHands

## The Current State of Web Apps and Games

### Starting with the Foundations of Web Automation
