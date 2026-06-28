---
title: "让 Agent 操作浏览器：从自动化脚本到浏览器基础设施的演进"
title_en: "Letting Agents Use Browsers: From Automation Scripts to Browser Infrastructure"
date: 2026-05-22 17:00:00 +0800
categories: [Agent 基础设施]
tags: [Agents, MCP, Browser Automation]
author: Hyacehila
excerpt: 从 Codex Chrome extension、Claude in Chrome 到 Playwright、Chrome DevTools MCP、Browser-Use、Stagehand、Skyvern 和云端浏览器基础设施，梳理浏览器 Agent 工具如何围绕官方宿主边界、确定性控制、可读感知、行动编排和生产化承载逐步演进。
excerpt_en: "A discussion of how browser-agent tools evolved from official host-native browser bridges to automation scripts and browser infrastructure, covering Codex Chrome extension, Claude in Chrome, Playwright, Chrome DevTools MCP, Browser-Use, Stagehand, Skyvern, Lightpanda, and cloud browser platforms."
permalink: '/blog/2026/05/22/agent-browser-tools-comparison/'
---

# 让 Agent 操作浏览器：从自动化脚本到浏览器基础设施的演进

如果把 Codex Chrome extension、Claude in Chrome、Playwright、Chrome DevTools MCP、Browser-Use、Vercel agent-browser、Stagehand、Skyvern、Lightpanda、Browserbase、Steel.dev、MultiOn 这些工具排成一列，很容易写成工具百科：每个工具一小节，讲功能、场景和限制。这样能帮人认名词，但解释不了一个更有意思的问题：这些工具为什么会这样演进？

让我们按照用户解除的顺序来展开：先看官方宿主如何把真实浏览器、登录态、权限确认和安全边界打包成开箱即用的能力，再让浏览器能被稳定控制，让网页状态能被模型读懂，然后让行动可以缓存、验证和编排，最后把浏览器本身变成可承载的运行时资源。

## 浏览器为什么会成为 Agent 基础设施

很多真实任务没有干净 API。查数据、填表单、登录后台、下载发票、复现前端 bug、检查网络请求、跑性能分析，最后都会回到浏览器里。对人来说，浏览器是视觉界面；对 Agent 来说，浏览器是一堆复杂状态：DOM、无障碍树、截图、网络请求、console 日志、cookie、localStorage、iframe、弹窗、验证码、加载时序和权限边界。

传统浏览器自动化关心的是程序怎么控制浏览器。Agent 浏览器工具要处理的问题更宽：模型看见什么，怎么决定动作，怎么判断动作是否成功，失败以后怎么恢复，成功经验能不能复用。

先用一张表把脉络压住：

| 阶段 | 代表工具 | 解决的核心瓶颈 | 相对上一阶段的变化 | 仍然留下的问题 |
| --- | --- | --- | --- | --- |
| 官方宿主浏览器桥 | Codex Chrome extension、Claude in Chrome | 登录态复用、权限确认、敏感动作确认、prompt injection 风险和真实浏览器操作 | 从“给 Agent 一把浏览器工具”，变成宿主产品直接接入用户已登录的 Chrome，并把安全边界放到产品层 | 仍然依赖 UI；结构化 API 和官方 connector 更安全；测试、调试、低 token 表示、工作流和生产化仍需要后续工具 |
| 从人写脚本到 Agent 调用浏览器 | Playwright、Playwright MCP、Chrome DevTools MCP | 把用户操作自动化和 DevTools 调试封装成 Agent 工具 | 从开发者写脚本，变成 Agent 能扮演“终端用户”或“前端工程师” | 上下文污染严重，传统自动化与 DevTools 输出仍面向人类测试/调试，不是原生面向 Agent 的上下文边界 |
| 从完整网页到 Agent 可读表示 | Vercel agent-browser、Browser-Use | 控制 Agent 能看到什么，而不只是让它能操作浏览器 | 对页面状态做语义压缩、引用化和任务相关过滤 | 自主探索成本高，执行确定性不足 |
| 从开放探索到工业落地 | Stagehand、Skyvern | 把自主浏览器操作推向可编排、可复用的流程 | Stagehand 走开发者混合接管与缓存自愈，Skyvern 走业务视觉工作流与目标检查 | 长流程仍然依赖浏览器会话、代理、回放和生产运维 |
| 从单次操作到生产级运行时 | Lightpanda、Browserbase、Steel.dev、MultiOn | 承载并发、会话、隔离、代理、反爬、成本和 API 化 | 浏览器从本地工具变成云端基础设施和可复用运行时 | Chrome 兼容性、权限安全、登录态和风控仍是长期挑战 |

重点不在于给工具排座次，而是看它们在改哪一层接口：从宿主官方边界，到页面点击和运行时诊断；从 DOM 解析，到模型可读表示；从自由探索，到缓存和工作流；从单机浏览器，到云端 session 和轻量 runtime。

## 第零阶段：官方浏览器桥，先把浏览器交给宿主

在讨论 Playwright、MCP、agent-browser 或 Browser-Use 之前，今天已经有一类更靠前的选择：由 Agent 宿主自己提供的浏览器桥。它们不是又一层 CDP 封装，也不是让开发者自己维护一套浏览器 session，而是让 Codex 或 Claude 直接接入用户正在使用、已经登录的 Chrome。

[Codex Chrome extension](https://developers.openai.com/codex/app/chrome-extension) 走的是这个方向。它通过 Codex 插件和 Chrome 扩展连接真实 Chrome profile，让 Codex 可以使用用户已经登录的浏览器状态去操作 Gmail、Salesforce、公司内网或其他需要登录的网页。更重要的是，它把很多过去需要工程师自己补的边界放进了产品层：访问新网站前的域名授权、allowlist 和 blocklist、浏览历史这类高风险能力的显式授权，以及把网页内容当成不可信上下文处理的安全模型。

[Claude in Chrome](https://code.claude.com/docs/en/chrome) 也类似（他更早一些）。Claude Code 可以通过 `claude --chrome` 启动，或在会话中使用 `/chrome` 连接 Chrome 扩展。它复用用户的浏览器登录态，在可见 Chrome 窗口里执行点击、输入、导航和读取页面状态；遇到登录页或 CAPTCHA 这类需要人类介入的场景，会暂停让用户处理。Anthropic 也单独讨论了 [Claude in Chrome 的安全使用](https://support.claude.com/en/articles/12902428-using-claude-in-chrome-safely)，重点就是 prompt injection、站点权限、敏感动作确认和高风险任务边界。

所以，如果问题只是“我希望 Agent 打开浏览器，利用我的登录态，替我完成一个网页任务”，这两个官方工具几乎应该是第一优先级。它们把登录态、授权、可见操作、人工接管和安全确认做成了宿主产品的一部分。相比自己拿 Playwright 连 profile、临时跑 CDP、或者把一个通用 browser agent 接进真实账号，它们更像一条被产品化过的道路。

但这并不意味着浏览器应该变成所有任务的第一选择。浏览器本质上仍然是结构化 API 和官方 connector 适配不完整时的临时中介。只要目标系统有稳定 API、MCP、app connector 或官方集成，优先走这些结构化接口通常更安全、更稳定、更可审计。

官方浏览器桥解决的是“能不能开箱即用地让 Agent 在我的真实浏览器里干活”。后面的工具解决的是更细的问题：如果要写回归测试，需要 Playwright；如果要诊断前端运行时，需要 Chrome DevTools MCP；如果要压缩网页上下文，需要 agent-browser 或 Browser-Use；如果要把流程做成可复用业务自动化，需要 Stagehand 或 Skyvern；如果要大规模承载 session，就会走向 Browserbase、Steel.dev 或 Lightpanda。

## 第一阶段：从人写脚本到 Agent 调用浏览器

如果把官方宿主浏览器桥看成第零阶段，那么第一阶段的代表是 Playwright、Playwright MCP 和 Chrome DevTools MCP。它们不是典型的自主 Agent 框架，但它们是浏览器 Agent 细分控制和调试工具链的起点。理解它们之间的关系，不能只看谁能点按钮，要先看底层那条线：CDP。

[Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/index.html) 是 Chrome、Chromium 和其他 Blink-based 浏览器的调试与自动化协议。它把浏览器内部能力拆成 DOM、CSS、Network、Runtime、Debugger、Performance、Tracing、Input、Page 等 domain。Chrome DevTools 自己就在用这套协议，许多浏览器自动化和调试工具也和它有直接或间接关系。

[Playwright](https://playwright.dev/docs/intro) 是现代 Web 自动化的确定性基线。它支持 Chromium、WebKit 和 Firefox，可以打开页面、定位元素、点击、输入、等待网络请求、截图、录制 trace，也能和测试断言、CI、报告系统配合。在 Chromium/Chrome 场景里，Playwright 和 CDP 生态血脉相连，也支持通过 CDP 连接已有 Chrome；但它给开发者的不是原始 CDP，而是更高层的 locator、action、auto-wait、trace 和 test API。

这就是 Playwright 的价值：把底层浏览器协议变成稳定的测试和自动化抽象。只要页面设计相对固定，用 Playwright 做 E2E 测试，就能像真实用户一样穿过整个系统流程，不必每次手写 DOM、Network 或 Runtime 级别的协议调用。

很多时候，我们确实不需要 Agent 自主理解网页。登录测试、结账流程回归、表单校验、前端改动后的 E2E 检查，成功标准都很明确。用模型自由探索，反而多了一层不确定性。

Agent 时代的新变化，是这些确定性能力开始通过 MCP 暴露出来。不过 Playwright MCP 和 Chrome DevTools MCP 虽然都站在浏览器自动化/调试这条谱系上，却让模型扮演了两个角色。

### 同源不同角色：终端用户 vs 前端工程师

[Playwright MCP](https://playwright.dev/mcp/introduction)把 Playwright 的自动化能力包装成 MCP Server，让 LLM 可以通过结构化 accessibility snapshot 和元素 ref 操作网页。它把模型当成“终端用户”：页面上有什么按钮、输入框、复选框，下一步该点哪里、填什么、等什么。官方文档也强调，它基于 accessibility snapshot 工作，不需要视觉模型；典型工具包括导航、点击、输入、表单、tab、dialog、storage、network、tracing 等。

这一步不是让 Playwright 变成完整自主 Agent，而是让浏览器控制能力进入 Agent loop：

```text
用户目标
  -> 模型判断需要浏览器
  -> 调用 Playwright MCP
  -> 获取页面状态和元素引用
  -> 执行点击、输入、截图、等待等动作
  -> 把结果返回模型继续推理
```

Chrome DevTools MCP 则把模型当成“前端工程师”。它当然也有 input automation 和 navigation tools，可以点击、输入、导航、截图；但这不是它最有价值的地方。它强在把 DevTools 视角交给 coding agent。

Google 在 [Chrome DevTools MCP 发布文章](https://developer.chrome.com/blog/chrome-devtools-mcp?hl=en)里说得很直接：coding agent 看不到自己写出来的前端代码在浏览器里实际发生了什么。它可以改文件，可以运行命令，但如果没有浏览器反馈，就不知道页面是否白屏、资源是否 404、CORS 是否失败、按钮为什么点不动、LCP 为什么很高。

根据 [Chrome DevTools MCP tool reference](https://github.com/ChromeDevTools/chrome-devtools-mcp/blob/main/docs/tool-reference.md)，它暴露的不只是点击和填表，还包括 console、network、performance trace、Lighthouse、heap snapshot、DOM/CSS、页面导航、截图和输入自动化。换句话说，它让 Agent 不只会操作浏览器，还会读 DevTools。

### 上下文差异：交互树 vs 运行时证据

Playwright MCP 给模型的上下文更像页面的可操作语义结构。它会把页面转成 accessibility snapshot，并给交互元素分配 ref。模型看到的是标题、按钮、文本框、复选框和它们的引用，因此更适合做逻辑层面的页面操作：登录、填表、点击下一步、抓取肉眼可见的数据。

Chrome DevTools MCP 给模型的上下文更像浏览器运行时证据。它适合拿 console 报错、network request、DOM/CSS 细节、performance trace、Lighthouse 结果、heap snapshot 来排查原因。它回答的不是下一步点哪里，而是为什么页面现在是这个状态。

所以，如果希望 AI 自动登录系统、填报表单、抓取页面上可见数据，Playwright MCP 更自然；如果希望 AI 排查 React 内存泄漏、分析页面加载性能、定位 API 请求失败或 CORS 问题，Chrome DevTools MCP 更自然。

这组工具的问题也在这里。它们让浏览器终于能被 Agent 调用，却仍然把浏览器以“测试工程师”和“前端工程师”的信息形态送进上下文。Playwright MCP 已经用 accessibility snapshot、ref 和无视觉模式做了压缩，但它仍然继承了 E2E 自动化工具的视角。Chrome DevTools MCP 的信息量更大，很多日志、请求、样式、trace 片段只对特定问题有意义；如果没有筛选，模型很容易被低价值细节牵走。

第一阶段解决了 Agent 能不能使用浏览器工具的问题，没有解决浏览器应该以什么形态进入 Agent 上下文的问题。第二阶段就是从这里开始的。

## 第二阶段：从完整网页到 Agent 可读表示

第二阶段不是为了再封装一层 CDP，也不是给浏览器多加一个操作接口，而是把浏览器状态重新整理成 Agent 可消费的任务上下文。

Playwright 提供了很强的浏览器控制基座，但如果直接把 Playwright 式浏览器状态交给 AI，会遇到几个问题。

第一是上下文灾难。真实 Web 应用的 DOM 里有大量布局容器、样式类名、框架状态、hydration 数据、埋点节点和不可见元素；trace、日志、截图、网络信息也会迅速膨胀。人类调试时可以筛选，模型上下文不行。

第二是状态节奏不同。传统 Playwright 脚本更像一次性运行一段流程，而 Agent 需要看一步、想一步、走一步。它需要浏览器 session 常驻，需要每一步都能读取当前状态，再决定下一步。

第三是定位脆弱。CSS selector 和 XPath 适合开发者写确定性脚本，但不适合模型在失败后自我修复。一旦 DOM 层级、class name 或组件结构变了，模型很容易陷入一种尴尬状态：知道自己要点哪里，但不知道该怎么定位。

于是问题变成：如何把 Playwright 能控制的浏览器，变成 Agent 能理解和连续操作的环境？Vercel Labs `agent-browser` 和 Browser-Use 给出了两条不同路线：一条做文本降维，一条做 Agent 闭环。

### Vercel agent-browser：文本降维与 CLI 哲学

Vercel Labs 的 [agent-browser](https://github.com/vercel-labs/agent-browser) 代表了轻量化方向。它是面向 AI agents 的浏览器自动化 CLI，通过 Chrome/Chromium 的 CDP 工作，核心形态是本地 CLI 加常驻 daemon。常见流程是先用 `agent-browser snapshot` 拿到带 ref 的 accessibility tree，再用 `agent-browser click @e2`、`fill @e3` 这类短命令完成动作。

这套设计首先考虑的是 `context budget`。模型不需要读完整 DOM，也不需要看截图，只要看到页面上有哪些标题、链接、按钮、输入框，以及这些元素对应的 ref。对终端里的 coding agent 来说，页面状态被压缩成短文本，动作也被压缩成短引用。

一个典型快照大概是这样：

```text
- heading "Example Domain" [ref=e1]
- link "More information..." [ref=e2]
```

模型不需要生成复杂 selector，只要输出：

```text
agent-browser click @e2
```

这种 ref-based 操作改变了 Agent 和网页之间的接口。过去模型可能要生成 selector：

```text
click("button.submit.primary:nth-child(2)")
```

现在它只需要选择一个语义节点：

```text
click @e2
```

这一步看似很小，但对 token 成本和动作稳定性影响很大。Agent 不再需要在一堆 CSS 类名里猜测哪个 selector 最合适，而是基于无障碍语义和引用 ID 操作页面。

CLI 形态也很关键。`agent-browser` 更像一个可以被 Claude Code、Cursor、Codex 这类工具调用的终端能力，而不是完整的 autonomous browser agent。浏览器在 daemon 里保持会话状态，AI 可以像敲 shell 命令一样逐步探索网页；如果 AI 卡住，人类也可以直接在同一个终端里接管命令，帮它点过某一步，再让它继续。

这条路线可以概括为“降维”：把浏览器从复杂 DOM、trace 和 DevTools 面板，降成一组很短的文本快照和动作命令。它适合 coding agent 做轻量页面检查、资料查阅、功能验证和局部交互。边界也清楚：`agent-browser` 解决低 token 控制和终端可接管，不负责完整任务规划。它让模型更便宜地操作网页，但它自己不是 Web Agent 大脑。

### Browser-Use：把浏览器组织成 Agent 环境

Browser-Use 走的是另一条路线。[browser-use](https://github.com/browser-use/browser-use) 的定位是让网站可以被 AI agents 使用。它关心的不只是页面快照，而是把浏览器包装成模型可以循环观察、推理、行动、检查结果的环境。模型面对的不是一段固定脚本，而是一个目标和不断变化的网页状态。

这类框架把浏览器变成 Agent environment：

```text
观察页面
  -> 推理和规划下一步
  -> 执行动作
  -> 读取页面结果
  -> 判断任务是否完成或是否需要重试
```

这和 `agent-browser` 的区别很大。`agent-browser` 是一把锋利的命令行工具，模型需要自己决定如何把多步任务组织起来；Browser-Use 更像浏览器任务框架，把观察空间、动作空间、状态摘要和任务循环放到一起。它可以结合 DOM/HTML 状态、可交互元素，必要时使用截图或视觉信息，帮助模型围绕自然语言目标持续推进。

相对 Playwright，Browser-Use 的进步不是换一种点击方式，而是改变任务形态。Playwright 执行的是开发者提前写好的流程；Browser-Use 面对的是“帮我完成这个网页任务”这种开放目标。模型不再只是调用固定脚本，而是在页面状态中循环决策。

它也可以融合视觉理解，让 VLM 辅助理解页面结构，而不是完全依赖 DOM。更准确地说，Browser-Use 的核心是浏览器状态和动作循环；页面结构、交互元素、截图、视觉能力，都只是为这个循环服务。

代价也很直接：上下文和模型调用成本更高，执行路径更不确定，长任务更容易漂移。越开放，越需要验证、缓存、自愈和业务流程约束。

### 两条路线的对比

| 维度 | Playwright | Vercel agent-browser | Browser-Use |
| --- | --- | --- | --- |
| 核心哲学 | 人类开发者写确定性自动化 | 把网页降维成低 token CLI 工具 | 把浏览器组织成 Agent 任务环境 |
| 页面表示 | DOM、locator、trace、截图、事件 | Accessibility snapshot、`@eN` ref、短文本 | 页面结构、交互元素、状态摘要，可结合截图/视觉 |
| 定位方式 | CSS selector、locator、XPath | 语义节点引用，如 `@e2` | Agent 根据观察结果选择动作 |
| Token 成本 | 直接给模型很高 | 极低，优先压缩上下文 | 中到高，取决于任务循环和视觉使用 |
| 自主性 | 低，依赖开发者脚本 | 中，提供可组合命令 | 高，内置多步任务循环 |
| 人类接管 | 改代码或调试脚本 | 直接在终端敲 CLI 命令 | 相对更像接管一个运行中的 Agent loop |
| 典型场景 | CI/E2E、稳定流程 | coding agent 的轻量网页操作 | 开放网页任务、多步骤自主执行 |
| 遗留问题 | 上下文和定位不适合模型 | 不负责完整规划和业务验证 | 成本、漂移和可靠性压力更大 |

第二阶段其实不是一个方向，而是两条分叉。agent-browser 的进步在于降维，把浏览器压缩成文本大模型容易使用的终端工具。Browser-Use 的进步在于组织闭环，把浏览器包装成可以观察、推理、行动和检查的 Agent 环境。

二者都回答了第一阶段留下的上下文问题，也都留下了新的问题。`agent-browser` 解决上下文经济，却不负责完整任务规划；Browser-Use 提供自主循环，却带来成本、漂移和验证压力。第三阶段的 Stagehand / Skyvern，就是在把自主性重新收进缓存、验证、视觉确认和业务流程约束里。

## 第三阶段：从开放探索到工业落地

第二阶段解决了 Agent 怎么看网页、怎么循环决策的问题，但还没有解决工业落地。

`agent-browser` 解决了低 token 操作和终端可接管，但它不负责完整任务规划。Browser-Use 提供了自主循环，也带来了成本、漂移和稳定性压力。第三阶段从“能自主”转向“能落地”：怎么把 AI 放进可控流程，怎么减少重复推理，怎么让非开发者也能配置复杂网页任务。

Stagehand 和 Skyvern 是两条不同路线。Stagehand 面向开发者，把 Playwright 和 AI 混合成一个更容易生产化的 SDK；Skyvern 面向业务自动化，把视觉优先的浏览器代理包装成工作流平台。

### Stagehand：开发者路线，混合接管与执行缓存

[Browserbase 对 Stagehand 的介绍](https://www.browserbase.com/stagehand/)把它称为面向 browser agents 的开源 SDK，核心原语包括 `act`、`observe`、`extract` 和 `agent`。这几个原语背后的思想很清楚：不要在 Playwright 和 Agent 之间二选一，而是把确定性脚本和 AI 推理混合起来。

Stagehand 更像 AI 增强型 Playwright，而不是一个黑盒 autonomous agent。开发者仍然写代码、定义流程、处理输入输出和业务校验；AI 只在页面不稳定、语义定位困难、数据提取复杂时介入。

一个典型 Stagehand 风格的流程是：

```text
用代码控制主流程
  -> 用 observe 找当前页面可行动作
  -> 用 act 执行自然语言动作
  -> 用 extract 按 schema 提取结构化数据
  -> 必要时交给 agent 处理更长流程
  -> 成功路径尽量缓存和复用
```

这就是混合接管。打开网页、填固定账号、进入某个后台路径，这些确定性步骤继续用代码。点击一个经常改文案的按钮、从复杂页面里提取结构化字段、判断当前页面有哪些可行动作，这些不稳定环节交给 `act`、`observe` 和 `extract`。

Stagehand 的另一个关键点是缓存。完全自主 Agent 往往每一步都要重新问模型，这在重复流程里很浪费。Stagehand 的缓存思路是：第一次让模型解析动作或 agent step，成功后把可复用的动作结果保存下来；后续遇到相似页面结构时，优先使用缓存，减少或跳过 LLM 调用。缓存失效或页面变化时，再唤起模型重新定位和修复。

这个设计很实际：AI 不再每一步都推理，而是只在不确定或缓存失效时介入。它更适合专业开发者、已有业务流水线、稳定但会小幅变化的网页任务。Stagehand 不是要取代 Playwright，而是让 Playwright 在真实网页变化面前更有弹性。

### Skyvern：业务路线，视觉优先工作流平台

Skyvern 处理的是另一类问题：复杂业务门户和非开发者自动化。

这类网站经常有动态表单、弹窗、iframe、分页、文件上传下载、非语义化按钮和奇怪布局。只靠 selector 或无障碍树经常不够，因为关键控件在视觉上很明显，在 DOM 里却可能只是一个没有 role 的嵌套 `div`。

[Skyvern 文档](https://www.skyvern.com/docs/developers/getting-started/introduction)明确说，它使用 LLM 和 computer vision 自动化 browser-based workflows。官方描述的执行循环是：截图、提取 DOM、LLM 推理、执行动作、检查目标、重复。把它理解成 vision-first 更准确：视觉和截图是核心感知入口，DOM 是辅助信息，最终目标是不把脆弱的 XPath 或 selector 当作唯一真相。

这条路线更接近 AI RPA 或 workflow automation platform。Skyvern 可以通过 dashboard、API 或工作流触发；workflow block 可以表达登录、导航、下载、提取、循环、代码等步骤。对业务用户来说，它不像 Stagehand 那样要求你在代码里精细编排 `act` 和 `extract`，而是更适合用自然语言目标和可视化流程去配置“登录后台 -> 查找未付款发票 -> 下载 PDF -> 发邮件”这类任务。

Skyvern 把浏览器自动化从“开发者写脚本”推向“业务流程编排”。它对复杂门户、非语义化页面和动态 UI 更友好，也更接近非开发者能理解的自动化工具形态。

代价也明显。视觉和多模态推理更慢、更贵；工作流越长，越需要明确的目标检查、人工介入点、权限管理和异常处理。遇到登录、2FA、验证码、支付或敏感数据时，不能简单相信“视觉模型会解决一切”，仍然需要工作流设计和权限治理。

### 第三阶段的坐标

| 维度 | Playwright | Browser-Use | Stagehand | Skyvern |
| --- | --- | --- | --- | --- |
| 技术定位 | 确定性自动化 API | 开放式 browser agent loop | AI 增强型 Playwright SDK | 视觉优先 workflow automation platform |
| 人类介入点 | 编写完整脚本 | 初始目标和运行中监督 | 代码与 AI 混合编排 | Dashboard / workflow blocks / 自然语言目标 |
| 执行速度 | 快 | 慢到中等，依赖模型循环 | 快，缓存命中时接近确定性执行 | 较慢，依赖视觉和多步检查 |
| Token 成本 | 低，但不适合直接给模型 | 中到高 | 低到中，重复流程可通过缓存降低 | 高，视觉和长流程成本更重 |
| 抗改版能力 | 低，依赖 selector/locator | 中到高，取决于观察和动作设计 | 高，靠 AI 定位、自愈和缓存验证 | 高，靠视觉优先和目标检查 |
| 目标用户 | 测试/自动化工程师 | Agent 开发者 | 专业开发者和生产流水线 | 业务自动化用户、运营团队、RPA 场景 |
| 遗留问题 | 脆弱、上下文不适合模型 | 成本、漂移、验证不足 | 仍需代码和浏览器运行时 | 成本高、速度慢、权限和异常治理复杂 |

这一阶段把浏览器 Agent 从“能探索”推向“能完成业务流程”。Stagehand 解决开发者侧的可控、缓存和低成本；Skyvern 解决业务侧的视觉工作流和非开发者可用性。

不过第三阶段仍然默认有一个浏览器可以被稳定运行。只要走向生产，这个假设就会变重：浏览器 session 怎么管理？并发怎么扩？登录态怎么隔离？代理和验证码怎么办？失败轨迹怎么回放？这就是第四阶段的问题。

## 第四阶段：从单次操作到生产级浏览器运行时

当浏览器 Agent 从 demo 变成服务，问题就从“能不能操作网页”变成“能不能稳定承载浏览器 session、并发、代理、登录态、回放、隔离和成本”。浏览器不再只是工具，而是运行时资源。

- [Lightpanda](https://lightpanda.io/docs/)：解决浏览器 runtime 太重的问题，面向更轻量、更高并发的 AI-native/headless browser。
- [Browserbase](https://docs.browserbase.com/use-cases/agents)：解决托管浏览器 session、调试、回放、Agent Identity，以及 Stagehand 生态承载问题。
- [Steel.dev](https://steel.dev/)：解决浏览器 fleet、代理、反爬、captcha、云端 session 等执行基础设施问题。
- [MultiOn](https://docs.multion.ai/)：把浏览器操作进一步 API 化，让开发者提交自然语言 Web action，而不是管理每一次点击。

这一阶段只说明一个事实：浏览器 Agent 最终会从工具调用进入基础设施层。具体产品选型不是本文重点，重要的是看见生产化承载本身会成为独立问题。

## 两条真正的技术主线

把这些工具按阶段读完后，可以再抽象成两个问题。第一个问题是：Agent 看见什么？第二个问题是：Agent 怎么行动？浏览器 Agent 的演进，本质上是在改这两层接口。

不过在感知层和行动层之前，官方浏览器桥先补上了一层产品边界：谁的浏览器、谁的登录态、谁来授权、谁来确认敏感动作，以及网页内容在什么安全模型里被读取。Codex Chrome extension 和 Claude in Chrome 的价值首先在这里，而不只是多提供一组 click 和 fill。

感知层回答的是：浏览器状态以什么形态进入模型上下文。

最早的自动化主要依赖 DOM、locator 和 DevTools telemetry。CDP 把浏览器内部状态暴露出来，Playwright 把它们包装成测试与自动化抽象，Chrome DevTools 把它们包装成调试视角。这里的信息很强：DOM、CSS、console、network、trace、performance、heap snapshot 都能看到。但它们首先是给人类工程师和测试框架用的，不是给模型上下文设计的，所以强观测能力和上下文污染是同时出现的。

Playwright MCP 开始做第一层压缩：它用 accessibility snapshot 和 ref 把页面变成可交互语义树。模型看到的不再是完整 HTML，而是按钮、输入框、标题、链接和引用 ID。这适合模型像终端用户一样推理：“我要点哪个按钮”“我要填哪个输入框”。

`agent-browser` 把这个方向推得更极端。它把页面快照压缩成极短文本，把元素变成 `@eN` 引用，优先服务 token economy。对本地 coding agent 来说，这等于把浏览器变成了一个低噪声、低成本的文本界面。

Browser-Use 的感知不只是单次快照，而是任务循环里的状态摘要。它需要在每一步告诉模型：当前页面是什么、有哪些可交互元素、上一步做完后发生了什么，必要时还可以结合截图或视觉信息。它的重点不是最小 token，而是支撑连续决策。

Stagehand 的感知服务于代码流程里的不确定节点。`observe` 回答“当前有哪些可行动作”，`extract` 回答“我能从这里拿到什么结构化数据”。它不是把整个网页交给 Agent，而是在确定性代码骨架里，把局部不确定性转成模型可以处理的结构。

Skyvern 的感知更偏 vision-first。它用截图和视觉信息理解复杂业务页面，同时用 DOM 辅助和目标检查降低误判。对非语义化页面、动态表单、视觉上明显但 DOM 结构混乱的控件，这种感知方式比纯 selector 更接近人类。

可以写成：

```text
host-native browser bridge / signed-in state / permission gates
  -> CDP / DOM / DevTools telemetry
  -> Accessibility Tree / ref
  -> compressed text snapshot / @eN refs
  -> task-oriented browser state
  -> 视觉 + DOM
  -> engine-native browser state
```

第二层是行动层。行动层回答的是：模型如何把意图变成浏览器里的动作。

官方浏览器桥把行动先放回宿主产品里：动作发生在用户可见的真实浏览器中，登录、验证码、敏感提交和站点授权都能回到人类确认。它不追求最底层的可编程性，而是追求普通任务的开箱可用和权限治理。

Playwright 的行动是脚本化的：selector、locator、click、fill、hover、auto-wait。它很稳定，但前提是开发者已经知道流程和页面结构。

Playwright MCP 和 Chrome DevTools MCP 把这些能力变成工具调用。Playwright MCP 偏用户动作：导航、点击、输入、等待、截图。Chrome DevTools MCP 偏工程诊断：读取 console、network、DOM/CSS、performance，再配合必要的输入自动化复现问题。它们都把浏览器能力接入了模型，但仍然没有完全解决动作上下文如何收敛的问题。

`agent-browser` 把动作压缩成 CLI/ref 命令。模型不需要生成复杂 selector，只需要 `click @e2` 或 `fill @e3`。这种行动接口适合本地 coding agent，也方便人类在终端里直接接管。

Browser-Use 把动作放进 agent loop。模型不是只调用一次 click，而是持续观察、规划、执行、检查，直到任务完成或需要重试。这让浏览器操作从单步工具调用变成了多步任务执行。

Stagehand 把行动重新收回开发者控制。确定性代码负责主流程，AI 负责不稳定动作；成功动作或 agent step 可以缓存，缓存失效时再让模型修复。这让行动既有 AI 的弹性，又尽量接近脚本的成本和速度。

Skyvern 把行动编排成工作流。它不要求用户像开发者一样管理每个 selector，而是通过 workflow blocks、自然语言目标和视觉检查来组织任务，更接近业务 RPA。

Browserbase、Steel.dev 和 MultiOn 则把行动推向云端执行层。Browserbase/Steel.dev 关心 session、代理、回放、并发和隔离；MultiOn 进一步把动作封装成 Web action API，让开发者提交任务，而不是管理每一次点击。

可以写成：

```text
host-native browser action / human confirmation
  -> script / selector
  -> MCP tools
  -> ref / action primitives
  -> agent loop
  -> hybrid cached actions / workflow blocks
  -> cloud execution / Web action API
```

感知层决定模型是否看得清，行动层决定系统是否做得稳。官方浏览器桥则把这两层放进宿主产品的登录态、权限和安全边界里。工具演进不是从 A 替代 B，而是在这些层上不断改接口。

## 工具选型应该回到你卡在哪一层

所以选型不应该从“哪个工具最火”开始，而应该从“我的瓶颈在哪一层”开始。

如果目标系统已经有结构化 API、官方 app connector、MCP server 或稳定集成，先看这些接口。它们通常比浏览器更安全、更稳定、更可审计，也不容易被网页 UI 改版和 prompt injection 牵着走。

如果你的目标只是让 Agent 在已经登录的网页里替你完成任务，先看 Codex Chrome extension 和 Claude in Chrome。它们不是最细的浏览器自动化原语，但作为开箱即用的真实浏览器操作能力，登录态复用、权限确认、人工接管和安全边界都更完整。

如果你缺的是更细的稳定控制，再看 Playwright 和 Playwright MCP。已知流程、E2E 测试、CI 回归，不需要强行交给自主 Agent。进一步说，如果目标是让模型像普通用户一样完成流程，Playwright MCP 更自然。

如果你缺的是调试反馈，看 Chrome DevTools MCP。它最适合 coding agent 在真实 Chrome 里读取 console、network、DOM/CSS、performance 和 Lighthouse，再回到代码里修问题。如果目标是让模型像前端工程师一样诊断页面，Chrome DevTools MCP 更自然。

如果你缺的是 Agent 可读页面表示，或者你已经发现 Playwright/DevTools 输出正在污染上下文，看 Vercel agent-browser 和 Browser-Use。前者偏轻量、低 token、本地终端工作流；后者偏完整浏览器任务环境和自主探索。这个问题不要停留在 Playwright MCP 或 Chrome DevTools MCP 上继续堆更多 DevTools、trace 或 DOM 输出，而要转向更明确的上下文压缩和任务相关过滤。

如果你缺的是行动缓存和流程编排，看 Stagehand 和 Skyvern。前者适合专业开发者把 Playwright 与 AI 推理混合，并通过缓存降低重复流程成本；后者适合复杂业务门户和视觉优先的工作流自动化，更接近 AI RPA。

如果你缺的是生产化承载，看 Lightpanda、Browserbase、Steel.dev、MultiOn。Lightpanda 解决 runtime 成本，Browserbase 和 Steel.dev 解决云端浏览器基础设施，MultiOn 解决更高层的 Web action API 化。

这样看，工具选型就不再是一排工具里挑一个，而是先定位你处在浏览器 Agent 工具链的哪个阶段：

```text
有结构化 API / 官方 connector -> API / connector
只想让 Agent 操作已登录浏览器 -> Codex Chrome extension / Claude in Chrome
控制不了浏览器        -> Playwright / Playwright MCP
看不到真实运行时      -> Chrome DevTools MCP
上下文被浏览器噪声污染 -> agent-browser / Browser-Use
行动需要被缓存/编排   -> Stagehand（开发者混合接管）/ Skyvern（业务视觉工作流）
生产运行扛不住        -> Lightpanda / Browserbase / Steel.dev / MultiOn
```

## 结语

浏览器过去是给人看的应用，现在正在变成 Agent 使用互联网的环境接口。

官方浏览器桥让 browser agent 从工程实验变成可直接使用的产品能力。Codex Chrome extension 和 Claude in Chrome 解决的是“让 Agent 在我的真实浏览器里干活”这件事：复用登录态、放进权限确认、保留人工接管，并把 prompt injection 和敏感动作纳入宿主产品的安全边界。

在这条线之后，Playwright 解决确定性控制，Chrome DevTools MCP 解决真实运行时反馈，但它们也会把大量浏览器信息带进 Agent 上下文。agent-browser 和 Browser-Use 解决模型可读的网页状态，Stagehand 解决开发者侧的混合接管和执行缓存，Skyvern 解决业务侧的视觉工作流，Lightpanda、Browserbase、Steel.dev、MultiOn 解决生产运行时和平台化。

所以这不是“AI 替代自动化脚本”的故事。更准确地说，浏览器被拆成了几层接口：宿主官方安全边界、可控制的环境、可读的状态、可验证的动作、可承载的运行时。

这些接口一旦稳定下来，Agent 操作浏览器就不再是一段脆弱的自动化脚本，而会成为 AI 系统访问真实 Web 世界的一条基础通道。
