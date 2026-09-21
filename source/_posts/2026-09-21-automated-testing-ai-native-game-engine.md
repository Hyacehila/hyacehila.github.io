---
title: "Automatic Testing 与 AI-Native Game Engine"
title_en: "Automatic Testing and AI-Native Game Engines"
date: 2026-09-21 20:00:00 +0800
categories: ["Creative Media & Games", "Game AI & Production"]
tags: ["Automated Testing", "Game Engine", "Game AI", "AI Agent"]
author: Hyacehila
excerpt: "Agentic Testing System，或者说，一套让 AI 真正进入游戏、体验游戏并形成判断的接口，或许会成为下一步 AI4Game 的核心，也可能决定 AI-Native Game Engine 应该是什么样子。"
excerpt_en: "An agentic testing system—or an interface that lets AI genuinely experience a game and draw conclusions from it—may be central to the next stage of AI4Game, and may even determine the shape of an AI-native game engine."
mathjax: false
hidden: false
permalink: '/blog/2026/09/21/automated-testing-ai-native-game-engine/'
---

Agentic Testing System，或者说，一套让 AI 真正进入游戏、体验游戏并形成判断的接口，或许会成为下一步 AI4Game 的核心，也可能决定 AI-Native Game Engine 应该是什么样子。

## 为什么聊到测试？
为什么我会突然把“测试”和“AI 原生游戏引擎”放在一起讨论？

### Feedback is all you need
当 AI Coding 正快速渗透前端、后端、客户端和服务端开发，“一句话生成一个 App Demo”已经逐渐成为现实。但当同样的期待来到游戏开发，事情却没有那么顺利。今天的 Coding Agent 可以很快写出角色控制、战斗逻辑，甚至搭起一个完整的游戏工程，可真正让它独立完成一款游戏时，结果往往仍然停留在“能够运行”的 Demo——至于它能不能玩、画面是否正确、流程能不能走通，甚至到底好不好玩，模型自己常常并不知道。

在网易实习的三个月里，我逐渐产生了一个很模糊但越来越强烈的感觉：AI Game Coding 当前缺少的，也许并不是生成能力，而是反馈。

Coding Agent 过去几年的发展已经证明，只要允许模型真正运行自己的代码，读取报错、执行测试、观察结果，再根据反馈不断修改，同一个模型的能力就会发生巨大的变化。[SWE-agent](https://arxiv.org/abs/2405.15793) 将这种面向模型设计的工具与交互格式称为 Agent-Computer Interface（ACI）；而在游戏开发中，这样的闭环却远没有成熟。游戏的重要状态分散在场景、资产、动画、物理、UI 和实时运行过程中，很多错误不会变成一条清晰的 AssertionError，而只会表现为“角色没有走到正确的位置”“镜头穿模了”或者“这一关实际上根本无法通关”。

如果模型只能生成，那么它仍然只是一个生成器；只有当它能够完成“生成—运行—观察—验证—修改”的循环时，它才真正开始成为一个 Game Coding Agent。

这也是我为什么开始对游戏测试产生兴趣。这里所谓的“测试”并不只是传统 QA 意义上的测试用例，而是一整套面向 Agent 的反馈基础设施：让模型能够观察游戏状态、操作运行时、自动完成玩家流程、发现异常，并把这些结果转化成下一轮修改可以理解的反馈。

我的直觉是，这可能会成为 AI 原生游戏开发中非常基础的一层能力。**让 Coding Agent 拥有理解游戏的眼睛。**

### Game Benchmark 与现状
[GameDevBench](https://arxiv.org/abs/2602.11103) 是一项发表于 ICML 2026 的工作，测试 Coding Agent 能否在现代游戏引擎 Godot 中完成游戏开发任务。有点像游戏世界的 SWE-bench，任务来自网页与视频教程，覆盖 Gameplay、2D/3D Graphics 和 UI 等常见开发场景。

遗憾的是，已经在 SWE-bench 及其各种变体上表现很强的顶尖 Agent，到了游戏领域还是有一点失灵。论文报告的平均成功率从 Gameplay 类任务的 51.4%，下降到强调视觉理解的 2D Graphics 类任务的 33.0%。加入 Editor Screenshot MCP 与 Runtime Video 两种视觉反馈以后，GPT-5.4 在该测试中的成功率从 41.1% 上升到 52.0%。这说明了**纯 Gameplay Logic 不是最难的。** 代码 Agent 很擅长操作符号世界，游戏大量真实状态存在于视觉世界和运行时世界。**现有 Coding Agent 的感知—行动—验证闭环，并不是为游戏编辑器设计的。**

基于浏览器，也就是 Web 界面的游戏开发与成熟商业引擎之间的差距进一步显现了这个问题。`Three.js` 全是 `canvas`，但 Agent 依旧可以通过它最擅长的 JavaScript 去访问内容；一个 Unreal 项目内部远不止一些代码脚本，海量状态只存在于编辑器资产以及运行时变化中。[GameEngineBench](https://arxiv.org/abs/2607.03525) 在 9 个真实 Unreal Engine 5 项目中整理了 110 个 C++ 任务，最强配置的 pass@1 也只有 55.5%，还有 31 个任务没有被任何配置解决。通用模型一直在进步，引擎内任务却不会自动随之线性提升，这就是 Game Dev Agent 的现状。**当 Code 不再是整个世界，我们需要重新表征 World。**

### 变得 Model Progress Resistant
> “LLMs made generating code cheap. The real bottleneck is verification.” — [OpenHands](https://www.openhands.dev/blog/20260305-learning-to-verify-ai-generated-code)

[GPT-6 Astra](https://openai.com/zh-Hans-CN/index/gpt-6-astra/) 的发布让不少开发者有点害怕。OpenAI 展示了它在 Blender 中建立房屋模型，再放进 Unreal Engine 5 变成可行走场景；它也能直接生成网站、Web App 和游戏。一代新的基础模型发布，会吞噬很多生成中间层原本占据的位置，也包括我之前做的 UI 工作流。

如果一个方向的主要价值只是弥补今天基础模型在生成能力上的 5%～10% 缺口，那么下一代模型升级就可能迅速压缩它的价值空间。一个 Model Progress Resistant 的方向，对保住我的饭碗也有不小的价值。

模型的每一轮进步对于生成能力的中间层都是考验，但随着模型能做的越来越多，需要验证的状态更多，Verification infrastructure 仍然存在。模型越强，测试不一定越不重要；它可能反而让更大规模、更高自主性的工程成为可能，从而扩大自动验证的价值。


## 从 Web 开始理解 Agentic Testing

先从 Web 开始。它已经把 Agent 测试需要的几块东西做得相对清楚，我们可以先借它建立一套后面反复使用的概念。

这里比较值得聊的有两类工具。第一类是 [Playwright](https://playwright.dev/) 这样的自动化测试框架。它强调可重复的流程：打开页面、找到元素、点击或输入、等待状态变化，最后通过 Assertion 判断结果。第二类是 [ChatGPT 的浏览器扩展](https://learn.chatgpt.com/docs/chrome-extension) 这类面向通用 Agent 的浏览器工具。它更强调主动操作：Agent 根据当前页面决定下一步，在已经登录的浏览器、多个标签页和不断变化的任务中继续工作。

它们的产品定位并不一样，但都在回答同一个问题：**Agent 应该怎样理解和使用一个网页？** Playwright 希望测试过程稳定、可重复，浏览器 Agent 则要在没有固定脚本的情况下继续完成目标。到了底层，两边都需要给 Agent 一个介于浏览器和模型之间的接口，控制它能看见什么、能做什么，以及每次操作之后得到什么反馈。

### 中介层首先是一种信息过滤

浏览器内部的信息实在太多。完整 DOM、样式、布局、事件监听、网络请求、Console、Storage 和 JavaScript Runtime 全部交给模型，既浪费 Context，也不利于它找到真正重要的东西。中介层的价值首先是过滤。它把浏览器里复杂、连续变化的状态，压缩成 Agent 当前能够理解和操作的表示。

[Playwright MCP](https://playwright.dev/docs/next/getting-started-mcp) 默认使用结构化的 Accessibility Snapshot。模型看到的不是整份 HTML，而是按钮、输入框、标题、文本及其 accessible name，再加上可以用于后续操作的元素引用。例如，一个登录页可以被压缩成“用户名输入框、密码输入框、登录按钮”。大量只负责布局和样式的节点被省略了，剩下的结构更接近人类对页面功能的理解。

这比让 Agent 阅读完整 HTML 更稳定，也比只看截图便宜。但过滤一定会丢信息。一个元素在语义树里叫作“提交按钮”，不代表它没有被遮挡，也不代表它的颜色、位置和动效符合设计。语义表示回答“它是什么”，视觉仍然负责回答“它看起来怎么样”。

### Agent 通过不同的渠道操作浏览器本身

Agent 一般通过两种不同的渠道操作浏览器。它通常不直接修改 DOM，而是走更接近真实场景的接口。

常规的 Playwright 测试会定位一个真实元素，然后发送点击、键盘输入、拖拽等操作。页面自己的事件处理和业务代码收到输入以后，再去改变 DOM 与应用状态。对测试而言，这条路径更接近真实用户。

浏览器工具当然也可以直接执行 JavaScript，读取页面变量、调用函数、修改 Storage，或者通过 [Chrome DevTools Protocol（CDP）](https://chromedevtools.github.io/devtools-protocol/) 读取 Console、Network 与 Runtime 信息。CDP 的 Input domain 还能派发鼠标、键盘、触摸和拖拽事件。这里更倾向于完整的调试，而不是像用户一样使用网页。

这两种操作需要分开。点击按钮属于用户层控制；直接改变量或调用内部函数属于特权控制。前者更接近真实体验，后者更容易定位问题。测试系统往往两边都需要，且不能混为一谈。

### 三个后面还会反复出现的平面

到这里，可以先把 Agentic Testing 拆成三个平面。这套划分后面也可以直接带到游戏引擎里。

| 平面 | 回答的问题 | Web 中的典型能力 |
| --- | --- | --- |
| **Observation Plane（观察平面）** | Agent 能看见什么？ | Accessibility Snapshot、DOM、截图、Console、Network、Storage、Runtime 状态 |
| **Control Plane（控制平面）** | Agent 能做什么？ | 点击、输入、滚动、拖拽、导航、执行 JavaScript、修改环境或 Mock 请求 |
| **Verification Plane（验证平面）** | 系统怎样知道结果正确？ | Assertion、URL 与 DOM 状态检查、网络响应、截图对比、业务不变量与人工评价 |

观察让 Agent 理解页面发生了什么。它既需要文本化输入，避免连续截图带来的巨额 Token 开销，也需要视觉检查，确认文本信号无法描述的真实渲染结果。

控制让 Agent 改变页面状态。键盘和鼠标输入可以模拟用户操作；执行 JavaScript、修改环境或使用调试接口，则能让 Agent 更快到达目标状态。

验证让操作变成测试。Test Oracle 是这里绕不开的问题，只有找到可靠的终局与过程信号，才能形成一条可执行的测试。有些信号很难用断言描述，需要结合类似 LLM as Judge 的方法与视觉检查。“这个引导是否让人看得懂”或者“这个过程是否有趣”则更难只靠一条规则解决，可能还需要模型评价与人工抽检。

### 黑盒、白盒和更实用的灰盒

另一组需要提前说清楚的概念，是 Black-box、White-box 与 Grey-box。它们描述的不是测试规模，而是测试系统能够看到多少实现细节。

| 类型 | 测试能看到什么 | 典型做法 |
| --- | --- | --- |
| **Black-box Testing（黑盒测试）** | 只看外部输入与输出，不依赖内部实现 | 模拟鼠标键盘、观察屏幕或公开页面状态，判断用户流程能否完成 |
| **White-box Testing（白盒测试）** | 直接理解代码结构与内部逻辑 | 调用函数、检查分支与覆盖率、验证模块内部状态 |
| **Grey-box Testing（灰盒测试）** | 保留真实端到端流程，同时读取一部分内部状态 | 使用真实输入操作页面，同时查看 DOM、Network、Console、Storage、测试 Hook 或 Runtime 状态 |

纯粹依赖屏幕与坐标或者使用 DOM 等暴露给用户的操作接口的黑盒测试最接近人在使用网页，但把它作为超大规模 CI 的唯一方案并不现实。它慢，容易受到动画、分辨率和渲染变化影响，失败以后也很难知道问题发生在哪里。

灰盒 E2E 保留了真实输入和完整流程，同时从 DOM、网络请求、日志和 Runtime 中拿到诊断信号。它不如纯黑盒干净，但通常更容易复现，也更容易告诉开发者到底坏在了哪一层。

### Canvas 让网页突然变得像游戏

普通网页非常适合这套方法，因为按钮、输入框和文本天然存在于 DOM 与 Accessibility Tree 中。到了基于 Canvas 的游戏，情况会突然变化。以 [Three.js](https://threejs.org/docs/pages/WebGLRenderer.html) 为例，`WebGLRenderer` 最终把 Scene 与 Camera 渲染到一个 Canvas。人从画面里能看到很多信息，Accessibility Snapshot 却可能只剩下一个 `canvas`。Agent 当然可以退回视觉，观察截图，再用鼠标键盘继续操作。但如果测试系统只能走这条路，它也会继承纯视觉黑盒测试的所有问题：信息密度低、定位困难、反馈昂贵。

Web 游戏仍然运行在 JavaScript Runtime 中。测试可以通过受控的 JS 接口读取场景对象、角色坐标、任务状态和碰撞结果，也可以保留真实的键鼠输入，让游戏按照正常路径更新。外部操作是真实的，内部状态又能被观察，这正是一种 **Grey-box E2E**。没有 Accessibility Tree 不等于只能依赖视觉；它意味着我们需要为游戏世界另外提供一份机器可读的状态表示。

这也是 Web 测试值得先讲的地方。网页上的 Agent 已经拥有比较成熟的观察平面和控制平面，也有 Assertion、截图和运行日志组成的验证平面。Canvas 暴露出的断层，则刚好把问题带向游戏：当 DOM 不再代表整个世界，Agent 应该读什么；当点击一个按钮不再足够，Agent 应该怎样操作；当页面跳转不再代表任务完成，系统又该如何验证结果。

到了游戏引擎里，这三个问题一个都不会消失，只会变得更难。

### 移动端：换了平台，问题没有换

从 Web 再走到移动 App，问题其实没有发生根本变化。[Appium](https://appium.io/docs/en/latest/intro/drivers/) 自己并不直接理解每一种手机和应用，它在外面提供一套基于 WebDriver 的统一接口，再把命令交给平台 Driver。Android 常用 UiAutomator2，iOS 常用 XCUITest。测试代码看到的是同一组查找元素、点击、输入和截图命令，真正操作设备的仍然是各个平台原生的自动化能力。这就是前面三个平面在移动端的一次重复。

移动端当然会多出权限弹窗、前后台切换、设备型号和 Native/WebView Context 等麻烦，但测试系统的基本形状没有变。真正的断层仍然发生在语义层消失以后。普通原生控件还能形成可查询的 UI hierarchy；一旦界面由自绘 Canvas、OpenGL、Metal 或游戏引擎统一渲染，Appium 看到的也可能只剩下一块表面。此时要么退回视觉黑盒，要么由应用暴露可观测的内部状态，重新进入 Grey-box E2E。

先给 Agent 一份机器可读的界面，再给它接近用户的操作能力，最后提供独立的正确性判断。到了游戏里，我们依旧缺少一个统一且 Agent-friendly 的版本。

## Gaming：测试需要进入引擎内部

Appium 把 WebDriver 这套方法带到了移动操作系统，也很自然地把我们带到了它的边界。面对普通 App，平台自动化框架还能读取控件树，找到按钮、输入框和列表；面对 Unity 或 Unreal 游戏，操作系统通常只能看到承载 Player 的原生窗口，以及一块由引擎持续更新的渲染表面。角色、敌人、关卡、碰撞体和任务状态依旧存在，但它们存在于游戏引擎的世界里，不会自动变成 Android 或 iOS 可以查询的控件节点。两边并不在同一个抽象层。

这也把讨论重新带回 Gaming。浏览器可以替网页暴露 DOM 和 Accessibility Tree，移动系统可以替原生 App 暴露 UI hierarchy；游戏世界的语义接口，只能由游戏引擎自己提供。引擎愿意让测试系统看到什么、控制什么，又用什么方式判断结果，才是后面整套系统能否成立的核心。

### 游戏引擎现在做到了什么

前面的三个平面仍然成立，只是操作对象从页面变成了游戏世界。Observation Plane 需要一份不依赖视觉的状态表示，让 Agent 能查询场景中的对象、属性、关系以及物理、动画、任务和 UI 状态；Control Plane 既要保留键鼠、手柄和触摸这类接近真人的输入，也要允许受控的调试命令与测试 Hook；Verification Plane 则要把内部状态、运行日志、截图比较和体验判断放到同一条测试流程中。

这些思路并不是 AI 出现以后才有人想到。在前 AI 时代，给游戏提供一棵类似 DOM 的可查询对象树并不稀奇。面向 Unity 的 [AltTester](https://alttester.com/docs/sdk/latest/home.html) 会把 SDK 嵌入游戏，让外部测试在编辑器或真机上查找并操作对象。更通用一些的方案是网易的 [Airtest](https://airtest.doc.io.netease.com/en/tutorial/2_Airtest_introduction/) + [Poco](https://airtest.doc.io.netease.com/en/IDEdocs/poco_framework/6_poco_sdk/)：Airtest 通过图像识别定位画面并模拟输入，Poco 则把 SDK 接进 Unity、Cocos 或 Unreal，从 Runtime 中取出 UI 控件树和属性。一个偏视觉黑盒，一个把引擎内部语义带回了测试系统，组合起来已经很接近前面讨论的 Observation 与 Control Plane。不过现在看起来没什么人用了。

引擎自己也有正式的测试系统。[Unity Test Framework](https://docs.unity3d.com/Packages/com.unity.test-framework@1.4/manual/index.html) 支持 Edit Mode、Play Mode 和目标平台上的测试；Unreal 的 [Automation Test Framework](https://dev.epicgames.com/documentation/unreal-engine/automation-test-framework-in-unreal-engine) 覆盖单元、功能、内容压力与截图比较，[Gauntlet](https://dev.epicgames.com/documentation/unreal-engine/gauntlet-automation-framework-overview-in-unreal-engine) 则负责启动和监控更复杂的游戏会话。开发者可以基于这些框架编排输入、读取运行状态、检查日志并写下 Assertion，整体上和用 Playwright 编写一条 Web 流程测试没有那么不同。

但这还不意味着 Agent 已经能测好一个游戏。这些框架解决的是可以预先描述的问题：人先想到一条路径，再写下操作步骤和期望结果。它们很适合挡住已经遇到过的 Bug，也能保护稳定流程不被下一次修改破坏，却覆盖不了人还没有想到的组合。游戏一直在增加新关卡、新机制和新的交互方式，测试用例通常只能跟在内容后面，人工探索因此仍然是测试工作里昂贵而且很难省掉的一部分。Test Oracle 在游戏测试中也比传统软件测试更棘手。

**目前的不足仍然在于游戏过于复杂，状态组合又极易爆炸。AI 很难只根据开发需求写出足够的测试，没有测试反馈也就很难持续迭代。游戏行业自动化测试系统难以覆盖各种 Corner Case，而 Coding Agent 又依赖反馈改进，两件事在这里交汇，最终限制了 Coding Agent 在游戏开发中的表现。**

让 Agent 自己进入游戏世界，持续探索状态与画面的变化，并找到尚未被写进用例的问题，才是 Agentic Testing 还需要补上的那块能力。Agent 先探索世界并形成判断，最终再交给人工复核和诊断，这才能在一定程度上缓解 Test Oracle。

### 如何让 Agent 玩游戏？

现在，问题变成了怎样让 AI 玩游戏。再说得准确一点，是怎样让它像测试工程师一样探索游戏的边界条件。

只问“能不能玩下去”，今天已经有不少答案；测试系统还要问，能不能随时换一个目标，能不能准确复现过程，失败以后能不能说清楚坏在了哪里。

或许我们可以从第一性原理开始，人类想要玩一个游戏需要什么？完整画面？一些状态？人类又是如何在游戏中进行决策？无论是强化学习还是模型智能体，我们的很多工作就是在这些变量中选择。从直觉来看，人类通过视觉理解世界，但脑内只存储概念，这也是我认为的答案。

#### 强化学习：把一个目标练到很强

强化学习曾经是游戏 AI 最重要的路线之一。[OpenAI Five](https://openai.com/index/dota-2-with-large-scale-deep-reinforcement-learning/) 用自我博弈击败了 Dota 2 世界冠军，证明 RL 可以处理长时间跨度与复杂动作空间；代价是持续十个月、约四万五千年模拟游戏时间的训练，而且它读取的也是 Dota API 提供的结构化状态。AAA 游戏的状态组合爆炸，不代表 RL 在理论上无法收敛，却会把样本采集、探索、Reward 设计、长时程规划和版本迁移的成本推得很高。RL 很适合在目标稳定的环境里把一项能力练到很强，却不太适合直接充当每天面对不同测试条件的通用 QA Agent。

#### VLA：像人一样看，也像人一样按

字节跳动 Seed 的 [Lumine](https://arxiv.org/abs/2511.08892) 是这条路线很有代表性的例子。它从原始画面出发，以 5 Hz 读取像素并生成 30 Hz 的键鼠动作，在《原神》中完成了五小时的蒙德主线，还能零样本迁移到《鸣潮》和《崩坏：星穹铁道》。这说明端到端的视觉—语言—动作模型已经能形成相当不错的操作泛化。不过它首先是一名玩家。自然语言可以告诉它去哪里、做什么，却很难稳定承载测试工程师关心的前置条件、状态不变量、覆盖率目标和失败证据；只看像素也无法直接告诉它某次异常来自任务状态、碰撞系统还是动画状态机。

#### 规则与搜索：从手工规则到 Learned Policy

纯规则在边界明确的游戏里一直有效。传统棋类引擎把合法走法、局面评估与 Alpha-Beta Search 写进程序。游戏测试中的任务脚本、状态机和导航 Bot 也是同一种思路：便宜、稳定，失败后容易追踪。它们能走到哪里，取决于开发者事先写进了多少规则。

[AlphaZero](https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/) 在已知合法动作与胜负条件的前提下，用自我博弈学习 Policy 与 Value，再通过 Monte Carlo Tree Search 选择动作；MuZero 学到的是用于规划的隐空间动态模型。让 Agent 阅读代码，从中抽取环境约束并构建测试规则，是另一条可以继续研究的路线。无论采用哪一种方法，它们仍然需要精确的状态、可用动作和清楚的目标信号。棋盘游戏能把这些条件说得很干净，开放世界游戏则很难，规则和环境也未必能迁移到下一个游戏。

#### LLM Chat Decision

LLM 或 VLM 也可以沿用人的接口。[Cradle](https://arxiv.org/abs/2403.03186) 只接收截图，再输出键鼠操作，已经能在《荒野大镖客 2》中连续完成约四十分钟的任务。这种纯黑盒接口几乎不用修改游戏，泛化范围也大。代价是每一步都要重新从像素推断状态，画面分辨率、观察历史和帧数都会增加视觉 Token 与推理延迟，一些被遮挡的数值和内部标记更不会出现在屏幕上。视觉适合验证玩家最终看到了什么，却不适合作为大规模 CI 唯一的事实来源。

纯视觉也不等于把每一帧原样交给 VLM。一个长流程可以维护滚动的 Visual Window，只保留当前完整画面、发生明显变化的关键帧、与任务有关的局部裁剪以及最近的 Action Trace，重复画面则只记录差异，这是 Agent 对一个游戏完整短期状态的近似估计。[LongVU](https://arxiv.org/abs/2410.17434) 已经证明长视频可以按帧间相似度与任务查询压缩时空 Token；不过测试还要额外保留一段原始画面缓冲，在异常发生时冻结前后证据。用于决策的窗口可以有损压缩，最后交给人和 Test Oracle 的证据不能只剩一段模型摘要。

YOLO 这类检测器也能参与压缩。它可以从画面中找出按钮、血条、任务标记或敌人，再把它们转换成带有类别、位置和置信度的对象。[OmniParser](https://arxiv.org/abs/2408.00203) 已经在通用 UI 上用 YOLOv8、OCR 和图标描述模型把截图还原成结构化元素。游戏里也可以尝试构建一棵概率性的 Visual DOM，让 `health_drop`、`dialog_open` 或 `player_stuck` 这样的事件进入 `if/else` 规则。可惜 YOLO 的泛化也不是一个简单的问题。

另一条路线是直接把游戏世界压缩成模型能稳定读取的语言。[Towards LLM-Based Automatic Playtest](https://arxiv.org/abs/2507.09490) 中的 Lap 没有让 LLM 直接猜三消截图，而是先把棋盘转换成数值矩阵，再让模型选择交换动作；实验里它取得了更高的代码覆盖率，也触发了更多 Crash。[TITAN](https://arxiv.org/abs/2509.22170) 在两个大型商业 MMORPG 上走得更远：它把位置、任务阶段、生命值、附近 NPC 和物品等原始状态筛选、离散化，再交给 LLM 选择高层动作。截图没有被完全丢掉，而是更多用于卡住后的反思和诊断。

所以，视觉与文本并不是两条互斥的路线。高频的检测器和控制策略可以处理连续画面与局部操作，低频的 LLM 负责理解目标、规划路径和处理异常；引擎状态提供主要事实，Visual Window 则补充真实渲染和短期变化。如果游戏还能提供类似 DOM 的通用交互表示，并增加针对这些表示的预训练，LLM 才更有机会驱动 Agent 在游戏世界里自由地探索并完成任务。

### Test Oracle 与 Agentic Testing

前面的技术路线最终会落到同一个形状：把系统暴露为可控的文本信号，再搭配经过筛选的视觉信号。这需要引擎本身提供足够的能力，但能玩好游戏并不等于做好测试。那么，除了引擎暴露的观察与控制平面以外，我们还需要什么？

让 Agent 进入游戏，可以在一定程度上缓解 Test Oracle。一个经过良好提示与训练对齐的 Agent，搭配完整的测试结构，已经可以自动化地测试游戏中的工作场景。哪怕状态爆炸，哪怕我们永远不可能实现完整覆盖，也可以通过学习现有测试样例和规范，尝试构建一个尽可能扩大覆盖、同时针对特定场景深入检查的自动化测试系统。Agent 在策略有效的时候总比人类能够实现更大的覆盖，覆盖面积与局部针对性之间的 trade-off 也会因此缓解。具备世界先验知识的 LLM，则是目前很适合处理开放语义问题、缓解 Test Oracle 的工具。

Agent 处理 Test Oracle 的能力可以分成三个层级。第一层是相对确定的机器信号，例如 Crash、非法状态中的 NaN、明确的 Assertion 失败和确定性的哈希分歧，它们可以直接触发失败。第二层是依赖上下文的异常：玩家卡住、目标不可达、帧预算超标或者穿模，都需要结合任务前置条件、持续时间和设计意图再判断。第三层才是审美和设计体验，这类问题可以让模型参与评价，但最后仍然需要人来判断。

测试系统因此要把“接下来怎么玩”“这里是不是 Bug”和“问题发生在哪一层”拆成三个问题，否则 Agent 最多只是一个自动玩家。状态规则、Visual Window、Action Trace 与引擎日志需要在这里合并，模型负责处理其中开放而模糊的部分，人工则负责最后的复核。

这些问题最终会收束。Gameplay Agent 先在游戏中探索并产出候选测试信号，Oracle 确认问题，诊断系统再定位可能的原因。确认无误以后，一次开放探索应该被压缩成最小复现步骤、Action Trace，以及一条测试断言或测试样例描述，随后进入回归测试。以后遇到相似问题，系统可以先重放已有样例，再基于它扩展新的路径。这样，昂贵的探索不会只留下一个 Bug 报告，而会继续指导后面的开发。

## 答案

AI Native Game、Automatic Testing 与 AI-Native Game Engine 在这里收敛为同一个问题。

AI Native Game 需要 NPC 与世界具有一定的智能，这要求 Agent 能操作游戏角色并参与世界演化。Automatic Testing 仍然需要确定性的测试断言和回归 CI，只是固定用例不可能穷尽状态爆炸且包含时间轴的游戏。智能化的 Bot 可以补上探索部分，再作为执行者进入 CI，把发现的路径、状态和证据交还给测试系统。

我在这里所说的 AI-Native Game Engine，并不只是给编辑器接上一个 LLM，而是一套从运行时开始就允许 Agent 结构化地观察、操作、验证和回放游戏世界的引擎与基础设施。它要让 Agent 既能像玩家一样输入，也能像开发者一样读取状态、调试系统。无论是 Game NPC 还是 Testing Bot，都需要这些能力。

如果它真的要进入 CI，还需要一些看起来不那么 AI 的基础工作：固定随机种子、控制时间、保存与恢复状态，以及重放 Action Trace。这也是游戏测试领域的一个经典问题。

这就是我的答案。
