---
layout: blog-post
title: "BettaFish、MiroFish、OpenClaw 与 Agent 的信任边界"
title_en: "BettaFish, MiroFish, OpenClaw, and Agent Trust Boundaries"
date: 2026-04-12 20:00:00 +0800
categories: [Agent 系统]
tags: [Agents, Security, Product]
author: Hyacehila
excerpt: "BettaFish/MiroFish 和 OpenClaw 分别把 Agent 推到两条信任边界上：我们能相信 AI 说到什么程度，以及我们愿意让 AI 在自己的环境里做到什么程度。"
excerpt_en: "BettaFish/MiroFish and OpenClaw push agents against two trust boundaries: how much we can trust what AI says, and how much we let AI do inside our environments."
featured: false
math: false
---

# BettaFish、MiroFish、OpenClaw 与 Agent 的信任边界

语言模型智能体的边界，不只是能力边界，更是委托边界。

只问模型能不能规划、能不能调用工具、能不能看浏览器、能不能长期记忆，还没有碰到 Agent 的核心。更关键的是：用户愿意把什么交给它。

BettaFish 和 MiroFish 触碰的是认识论边界：我们愿意在多大程度上相信 AI 对世界的描述、归纳和推演。OpenClaw 触碰的是操作边界：我们愿意在多大程度上让 AI 进入电脑、浏览器、通信入口和长期记忆，替我们做真实动作。

这两条边界最后会汇到同一个问题上：Agent 的价值，经常出现在用户愿意让它跨过某条原本不该轻易跨过的线之后。跨过去会产生收益，也会放大风险。好的 Agent 产品不是消灭这条线，而是把这条线画清楚。

## BettaFish 与 MiroFish：相信 AI 到什么程度

> 聊到这个项目仅仅是因为几乎同时我也在做类似舆情相关的工作，但是从Stars的角度看并没有这么成功。

BettaFish 的中文名叫“微舆”，定位是人人可用的多 Agent 舆情分析助手。它不是简单把 LLM 接到搜索 API 上，而是把舆情分析拆成一组 Agent 和工程模块：Query Agent 理解问题和检索方向，Media Agent 处理媒体与素材，Insight Agent 做分析抽象，Report Agent 生成报告；ForumEngine 监听不同 Agent 的日志，由主持模型组织阶段性讨论。

这些复杂度对用户基本不可见。用户面对的是一个更自然的问题入口：这件事现在舆论怎么看，谁在说什么，情绪在哪里，趋势可能往哪走。BettaFish 把原本属于舆情公司、数据分析团队和专业研究流程的工作，变成了普通用户可以直接使用的入口。

MiroFish 是他的迭代产物。BettaFish 偏向回答“现在怎么看”，MiroFish 进一步回答“接下来会怎样”。它让用户上传种子材料，比如数据报告、故事、事件背景，再输入预测需求；系统从材料中抽取实体和关系，构建知识图谱，生成带有人格、记忆和行为逻辑的智能体，把它们放入并行演化的模拟环境，最后由 ReportAgent 生成预测报告，用户还可以继续和模拟世界里的角色互动。

MiroFish 更接近世界模拟。它使用 Zep 这样的记忆/图谱能力承载 seed materials 里的实体和关系，再把实体转换为可用的智能体画像。报告阶段也不是一次性总结，而是一个带工具的 ReACT 式 ReportAgent，可以检索图谱、访谈模拟角色、提炼洞察。

BettaFish 和 MiroFish 的热度和其背后的技术无关。普通用户并不关心系统里有几个 Agent、有没有 GraphRAG、用了什么仿真引擎。真正有传播性的，是它们把复杂技术翻译成了两个本能问题：**怎么看，和会怎样。**

这也是它们的风险。界面越自然，报告越完整，模拟越具体，用户越容易相信他，但这些内容百分百都是依靠语言在约束一个概率生成模型去生成的。多 Agent 争论、知识图谱、群体仿真和报告模板都能增加解释层次，但不会自动把不确定性变成确定性。它们最多让假设更显性、过程更可讨论、结论更容易被质疑。

BettaFish/MiroFish 因而是一个很有价值、也很危险的方向：它们让 AI 从回答问题变成组织认知。它们越好用，越需要提醒用户，自己看到的是一个被系统加工过的世界模型，而不是世界本身。

## OpenClaw：授权 AI 到什么程度

OpenClaw 触碰的是另一条边界。

它的前身包括换了一堆名字，最后改名 OpenClaw。项目目标是做一个运行在用户自己设备上的个人 AI 助手，支持广泛平台。这个定位不新，但 OpenClaw 的工程形态足够完整。

OpenClaw 的核心是 Gateway。Gateway 连接消息入口、会话、工具、节点和客户端。WhatsApp、Telegram、Slack、Discord、Signal、iMessage/BlueBubbles、Matrix、Teams、Feishu、LINE、Mattermost、WeChat、QQ、WebChat 等入口都可以成为和 Agent 对话的地方。它不是把用户拉进一个新 App，而是把 Agent 放进用户已经在使用的即使通信软件。

IM 入口不是渠道数量问题，而是关系位置问题。网页、IDE、终端和独立 App 都要求用户主动进入一个工具；IM 入口让 Agent 更像待在用户日常环境里的执行者。它离用户更近，也更容易被当成我的助手，或者说被当作一个 IM 工具里的人。

OpenClaw 的系统设计可以写作几层：

| 层次 | OpenClaw 的做法 | 对 Agent 开发者的意义 |
| --- | --- | --- |
| 入口 | 多 IM / WebChat / 语音 / Companion app | 入口是任务发生的位置，不只是 UI |
| 控制面 | Gateway daemon + WebSocket API + Nodes | Agent 变成本地控制平面 |
| 会话 | DM 默认共享、群组隔离、cron 新会话、session transcript | 会话承载上下文、权限和任务连续性 |
| 记忆 | `AGENTS.md`、`SOUL.md`、`USER.md`、`MEMORY.md`、`memory/YYYY-MM-DD.md` | 长期记忆被做成工作区状态 |
| 工具 | shell、文件、浏览器、canvas、cron、外部节点 | Agent 能力来自工具和权限 |
| 扩展 | Skills、plugins、ClawHub | 可复用能力变成可安装、可分享的生态对象 |
| 安全 | host-first 默认、可选 sandbox、trusted-operator model | 押注个人信任边界 |

最后一行是理解 OpenClaw 的关键。OpenClaw 文档明确说明，workspace 是 Agent 的家和私有记忆，但不是 hard sandbox；相对路径默认落在 workspace，绝对路径在未启用 sandbox 时仍可能访问主机其他位置。sandbox 是可选配置，模式包括 `off`、`non-main`、`all`，后端可以是 Docker、SSH 或 OpenShell。

这不是小注脚，而是产品判断。真正有用的个人助手不能永远待在无害但无能的框里。它需要在某些场景下接触真实电脑、真实文件、真实浏览器、真实通信入口和真实长期记忆。问题因此从“如何完全避免风险”变成“如何把风险放在用户能理解、能配置、能收回的边界里”。可能这个边界控制的并不好，OpenClaw 的用户规模意味着大量的非专业开发者涌入，而他们对于这些内容并不熟悉，更多的将一切都授权，去获得最大程度的能力。

这和很多企业级 Agent 平台的默认路径不同。企业平台首先问隔离、审计、审批、最小权限；OpenClaw 首先问一个个人用户如何把 Agent 养在自己的环境里。前者更稳妥，后者更容易制造普通用户第一次看到时的震动感。

## OpenClaw 的爆火原因

OpenClaw 在 GitHub 上已经是几十万 star 量级，npm 最近一个月下载量达到数百万量级。数字不能证明长期成功，但足以说明它踩中了 2026 年初的强情绪：用户已经不满足于一个会聊天的 AI，也不满足于一个只会在浏览器里跑任务的 Agent。他们开始想要一个真的在身边、真的能做事、真的属于自己的 AI。

OpenClaw 至少同时踩中了六件事。

第一，IM 入口把 Agent 放进日常动作里。用户不需要打开 IDE、后台或专门工具，可以在 Telegram、Slack、微信、QQ 或 WebChat 里像发消息一样调度 Agent。入口越贴近日常，Agent 越不像工具，越像生活的一部分。

第二，长期记忆和人格文件拉近了情感距离。`AGENTS.md`、`SOUL.md`、`USER.md`、`MEMORY.md`、每日 memory 文件让 Agent 不再只是当前对话里的模型，而是一个有持续状态的本地对象。对开发者来说这是 context engineering，对普通用户来说更接近它认识我。默认 Memory 对普通用户的价值远大于对专业用户的价值，后者愿意去审计自己的 `Claude.md` ， 前者可能不知道什么叫 `.md`

第三，高权限带来能力。Claude Code 已经让开发者接受“AI 可以在仓库和终端里工作”，但 Claude Code 仍然属于开发工具，普通用户基本不接触。OpenClaw 把类似能力搬到个人助手场景：shell、文件、浏览器、外部节点、定时任务都可能进入它的能力范围。普通用户第一次看到的是“这个 AI 真的可以动我的环境”，而不仅限于聊天。特别的，CLI相较于GUI也是一个在目前的场景下更稳妥的入口，能力的下限更高。

第四，Skills 和 ClawHub 让能力具备分享属性。Skills 把可复用操作过程打包成可安装、可传播、可治理的能力单元。没有生态，高权限 Agent 更像危险玩具；有了 ClawHub，个人实践才可能变成可传播的技能资产。至于Skills为什么可用，那么和基础模型能力的提升也有着很重要的关系。

第五，local-first 和 own-your-data 降低了心理负担。一个能进入个人环境的 Agent，如果完全运行在远端云里，用户会天然紧张。OpenClaw 的本地 Gateway、工作区、私有记忆、可选 sandbox，让它在叙事上更接近我的助手，而不是某家公司在云上代我操作。

OpenClaw 的爆火不是单点创新，而是入口、记忆、权限、生态、所有权和基础模型能力的进步共同推动的结果。

## 能操作环境，不等于产品成立

Manus、豆包手机、GLM Agent / Open-AutoGLM 这类产品或框架，都在证明模型可以在 GUI、浏览器或手机里完成相当多动作。但能操作环境只是能力，不等于产品。

Manus 更像远端任务门户。用户把任务交给它，它在云端或浏览器环境里执行。这种形态适合展示 autonomous agent 的能力，但它和用户本人的日常入口、私人长期记忆、本地文件系统和工具生态仍有距离。它能做事，却不一定像我的环境里的助手。而一个 Agent 能做多少事，和一个 Agent 对我有多了解是密不可分的，一个编码任务可以放到云端，但回微信不行。

至于豆包手机， 它的信任成本更高。通讯录、相册、支付、验证码、私聊、位置、App 登录态都在手机上，这意味着隐私信息的完全暴露。手机 UI 自动化也更脆弱，App 反自动化机制多，权限提示密集。同时豆包手机的ToC属性明显也更弱，用户用不到自然难以继续裂变传播。

Open-AutoGLM 作为开源 Phone Agent model & framework 有技术意义，甚至说是这些框架里看起来最受欢迎的一个（当然Manus的饼也很大，20亿是真贵）。但更像能力展示和研究/开发框架。它证明AI Phone可行，却不解决入口、记忆、生态和个人归属感。

Claude Code 是另一个参照。它的高权限被仓库、终端、git diff、测试和代码审查这些开发者熟悉的边界吸收了。开发者知道它可能改坏文件，也知道如何回滚、测试、看 diff。OpenClaw 想做更普通、更私人、更泛化的版本，难度也正在这里：普通用户没有开发者那套天然边界。当然这也意味着普通用户不会去用Claude Code，这不是面向他们的产品。

Agent 产品的竞争不只是看谁最早让模型点按钮，而是看谁能把“点按钮”安放在一个用户愿意持续使用、愿意承担风险、愿意分享能力的系统里。

## Agent 的边界与打破边界

Agent 的边界不应该只按能力划，也不应该只按安全恐惧划。更准确地说，它应该按委托关系划：用户交出什么，系统拿去做什么，换回什么收益，出错时如何停下、回滚和收回。

边界不是一道静态墙，而是一份隐含契约。BettaFish/MiroFish 让用户交出一部分认识论主权。用户原本要自己读材料、找信息、判断情绪、比较观点、想象未来，现在把其中一部分交给 AI。回报是结构化认知、趋势解释和可交互模拟。风险是，用户可能把系统生成的叙述当成现实本身，过于相信Agent所说的话。

OpenClaw 让用户交出一部分操作主权。用户原本要自己打开 App、读文件、跑命令、整理信息、维护长期上下文，现在把其中一部分交给 Agent。回报是直接执行、跨入口响应、长期记忆和可扩展技能。风险是，Agent 可能会出错，而出错将不仅限于错误的回答问题，而是直接做错事，搞坏文件。

一个是让 AI 替我看世界，一个是让 AI 替我动世界。它们最终都在问同一件事：**我愿意把多少判断和行动交给一个不完全可靠的系统。**

这件事需要辩证地看。把 Agent 留在边界之内，当然更安全：只读不写，只建议不执行，只总结不预测，只在沙箱里做，不碰真实环境。这也更符合工程师本能，因为错误被限制在文本、报告或临时环境里。但边界之内的 Agent 很容易变成笨重的建议机器。它会告诉你可以做什么，却把真正的执行摩擦留给你；它会给出谨慎结论，却不一定让人信任；它降低风险，也降低收益。

我自己做的 [AnalysisPosts](https://github.com/Hyacehila/AnalysisPosts) 就很接近这个反例。它同样面向舆情分析，也同样引入了 Agent：Stage 1 做数据增强，Stage 2 做深度分析，Stage 3 生成报告；中间有 QuerySearchFlow、DataAgent 使用统计分析函数来给出结论、SearchAgent 基于搜索引擎补充信息、ForumHost 动态循环辩论、图表分析、洞察生成和trace 证据链。还增加了一套完整的监控系统来帮助核验结论从哪个请求给出，从哪里出错。

这些设计都是为可靠性服务的。Stage2 会把搜索、数据分析和论坛讨论的过程写进 `trace.json`；Stage3 要求段落引用 `[E#]` 证据角标，章节生成后还有 ReviewChapters 的硬性检查；报告里还会注入方法论附录和证据索引。换句话说，它一直在努力回答一个问题：这份结论凭什么可信。

但这也说明了边界内系统的代价。AnalysisPosts 更像给分析者使用的严肃流水线，而不是普通用户会自然打开的产品。用户要理解阶段、使用爬虫工具、选择入口、等待多轮循环、查看 trace、阅读报告、溯源证据并考虑针对性的介入Agent的思考流程。它把不确定性拆得很细，把证据链做得很重，却没有像 BettaFish/MiroFish 那样把复杂性收束为“我问一个问题，你给我一个可用判断”，也没有像 OpenClaw 那样把能力直接放进用户每天所在的环境。

它在边界之内，所以安全、可追溯、可审计。也正因为在边界之内，它没有给普通用户足够强的“交出一点信任就立刻换回巨大收益”的感觉。这个项目对我最大的提醒是：可靠性本身不是产品体验。把结论做得更可信，和让用户愿意把任务交给系统，是两件相关但不同的事。用户不一定需要这个可靠性。

强烈的 Agent 产品体验，往往会跨出这条边界。BettaFish/MiroFish 如果永远只做资料摘要，就不会产生未来沙盘的想象力；OpenClaw 如果永远不能碰文件、浏览器、IM 和长期记忆，就只会退化成另一个聊天窗口。打破边界带来的潜在收益并不低：它把建议变成委托，把一次性问答变成长期关系，把 AI 能说什么推进到 AI 能替我持续做什么。

危险也正来自这里。认识论边界被打破后，流畅叙述会掩盖假设、证据缺口和模拟性质；操作边界被打破后，模型错误、提示注入、权限误配和插件信任都会进入真实环境。一个能读你长期记忆、控制已登录浏览器、响应 IM、调用 shell 的 Agent，不再只是一个答错题的系统，而是一个可能改变环境状态的执行者。

打破边界不是为鲁莽越权辩护。越靠近边界，越要认真设计边界。用户是否知道自己交出了什么，系统是否清楚展示它能看什么、改什么、记住什么，行动是否可观察、可中断、可回滚，错误爆炸半径是否被限制，记忆和技能是否有来源、版本和撤销路径，个人场景进入多人场景时信任边界是否重新划分，这些问题决定了跨边界到底是产品能力，还是危险幻术。

没有这些答案，跨边界只是在制造风险。有了这些答案，跨边界才可能成为产品能力。边界不是一次性推倒的东西，而是每次授权、每个场景都要重新画清楚。

## 结语

BettaFish、MiroFish 和 OpenClaw 可以放在一起看，是因为它们分别从“相信 AI 的叙述”和“允许 AI 的行动”两侧，把 Agent 推到同一个中心：**Agent 不是单纯的智能问题，而是委托问题。**

模型能力继续提升当然重要，但在 ASI 到来之前，Agent 产品的分野会越来越出现在边界设计上。BettaFish 和 MiroFish 的问题不只是准确率，而是如何让用户理解假设、证据和不确定性；OpenClaw 的问题不只是能不能执行，而是如何让用户清楚地管理权限、记忆和信任。

OpenClaw 的争议也正在这里。它把个人 AI 助手做成了一个可以安装、可以接 IM、可以写记忆、可以装 skills、可以运行工具的系统。它没有完全解决 Agent 边界问题，却把边界推到了桌面上：当 Agent 进入真实个人环境时，产品吸引力和安全风险来自同一个地方。它越有用，就越接近危险；它越被限制，就越容易失去魔法感。

下一代 Agent 产品的关键，不只是让 AI 更聪明，而是让用户知道自己交出了什么，并且能够负责地把一部分世界交给它。

## 参考资料与注释

- [OpenClaw GitHub 仓库](https://github.com/openclaw/openclaw)
- [OpenClaw 文档：Architecture](https://docs.openclaw.ai/concepts/architecture)
- [OpenClaw 文档：Agent workspace](https://docs.openclaw.ai/concepts/agent-workspace)
- [OpenClaw 文档：Sandboxing](https://docs.openclaw.ai/gateway/sandboxing)
- [OpenClaw SECURITY.md](https://github.com/openclaw/openclaw/blob/main/SECURITY.md)
- [OpenClaw 文档：Skills](https://docs.openclaw.ai/tools/skills)
- [ClawHub](https://clawhub.ai/)
- [npm: openclaw](https://www.npmjs.com/package/openclaw)
- [BettaFish GitHub 仓库](https://github.com/666ghj/BettaFish)
- [MiroFish GitHub 仓库](https://github.com/666ghj/MiroFish)
- [OASIS: Open Agent Social Interaction Simulations](https://github.com/camel-ai/oasis)
- [Open-AutoGLM GitHub 仓库](https://github.com/zai-org/Open-AutoGLM)
- [AnalysisPosts GitHub 仓库](https://github.com/Hyacehila/AnalysisPosts)
- [Manus 官方入口](https://manus.im/)
- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code/overview)
