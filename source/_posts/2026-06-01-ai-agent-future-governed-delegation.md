---
title: "AI Agent 的未来不是完全自治，而是可治理的委托"
title_en: "The Future of AI Agents Is Governed Delegation, Not Full Autonomy"
date: 2026-06-01 21:00:00 +0800
categories: [随笔与观察]
tags: [Agents, Society, AI Safety]
author: Hyacehila
excerpt: "Agent 的未来不会从全自主替代人类开始，而会先变成一套可治理的委托关系：人设目标、控权限、看后果，AI 承担越来越多执行。"
excerpt_en: "The future of AI agents will not begin with full autonomy replacing humans, but with governed delegation: humans set goals and boundaries while AI takes on more execution."
permalink: '/blog/2026/06/01/ai-agent-future-governed-delegation/'
---

# AI Agent 的未来不是完全自治，而是可治理的委托

Agent 这个词已经快被用烂了。

浏览器里能点按钮的是 Agent，IDE 里能改代码的是 Agent，客服后台里能分单的是 Agent，手机里能替你打开 App 的也是 Agent。再往远处说，就变成完全自主智能体、AGI、ASI。故事听起来很顺：模型继续变强，工具越接越多，某一天它自然从聊天窗口里站起来，接管人类手里的大部分工作。

我不太相信这个叙事。至少不相信它会以这种方式发生。

Agent 的变化当然重要。Chatbot 主要在回答，Agent 开始行动。它看环境、调工具、读写状态、连续执行任务。从这一刻起，我们讨论的就不只是智能，还有行动权、授权边界和后果由谁承担。

一个只会说错话的模型，最坏情况通常是一段错误文本。一个能操作浏览器、手机、文件系统、邮箱、CRM、支付页面和公司后台的 Agent，说错话只是风险里最轻的一层。它可能点错按钮，发错邮件，删错文件，把不该上传的内容交给第三方服务，或者在你还没真正理解时替你做了一个不可逆动作。

**通用 AI 已经足够强，但还不够可信。它已经能做很多事，麻烦的是，我们还不知道哪些事真的可以放心交给它。** 我在 OpenClaw 的那篇文章里聊过这个问题，但那包含了个人的一些经历和无关的技术细节，现在让我们抛开过多的技术细节来继续讨论这个问题

## 从手机和浏览器开始的不安

豆包手机风波是一个很好的切口。那篇 [《从豆包手机风波看通用 AI Agent 的困局与出路》](https://www.gm7.org/archives/23470) 里讲得很好：通用 Agent 不只是技术问题，还会撞上安全、法律、平台生态和用户授权。手机不是一个普通界面。通讯录、相册、私聊、验证码、支付、定位、App 登录态都在里面。让 AI 进入手机，就不是让它“帮我总结一下网页”那么简单。

AI 浏览器也是同一类问题。浏览器里有登录态，有 Cookie，有公司后台，有邮箱，有各种 SaaS 系统。传统浏览器扩展已经足够麻烦，Agentic Browser 又多了一层“它会替你行动”。Gartner 那些关于 AI 浏览器的安全提醒，语气可能偏保守，但担心本身并不奇怪：网页内容可以被攻击者控制，Agent 又会把网页内容当成任务上下文。间接提示注入不是什么遥远的实验室问题，它天然适合发生在浏览器里。

Agent 真正危险的地方，不是会出错，而是会带着权限出错。

普通软件的权限通常比较清楚：这个按钮会提交表单，那个 API 会写数据库。Agent 的权限更像一团动态东西。它读到什么、理解成什么、接下来调用哪个工具，很多时候不是预先写死的。你当然可以加审批、加沙箱、加策略，但只要目标是通用和自主，系统就会不断向边界外伸手。

我能理解用户为什么兴奋。第一次看到 AI 自己打开网页、填表、查资料、整理结果，确实会有一种“它终于能做事了”的感觉。可问题也从这里开始：如果它能替我做事，我到底愿意把多少东西交给它？

## 几种不同的未来想象

现在关于 Agent 的未来，市场上并没有一个统一版本。大家嘴上都在说 agent，心里想的东西可能很不一样。

企业软件公司喜欢把 Agent 讲成数字员工。Microsoft 2026 Work Trend Index 里的“Frontier Firm”就是这个方向：人和 Agent 混在一起工作，Agent 承担更多执行，人类负责意图、判断和质量。这套叙事很顺，也最容易被企业接受。它没有直接说人会消失，而是说组织会重新分工。

创业叙事往往更激进。autonomous workforce、多 Agent 公司、无人值守业务流程，这些词都很诱人。一个能 24 小时工作的数字员工，不抱怨、不请假、不需要办公室。如果成本足够低，企业没有理由不试。问题是，这个想象经常把“能跑 demo”直接跳到“能扛责任”。

安全团队看见的是另一种东西：一个高权限软件主体。OpenAI 在 [Practices for Governing Agentic AI Systems](https://openai.com/index/practices-for-governing-agentic-ai-systems/) 里讨论各方责任和安全实践，Anthropic 在 [trustworthy agents](https://www.anthropic.com/research/trustworthy-agents) 相关研究里反复谈人类控制、透明度、隐私、安全和身份归属。它们没有否定 Agent，只是把问题拉回了责任：当 AI 能行动时，谁来负责？

AI Safety 那边更担心长期目标。Yoshua Bengio 近年一直提醒，人类真正该小心的不是会回答问题的 AI，而是有自主目标、能长期行动、能获取资源的 superintelligent agent。他提出 Scientist AI 这类更审慎的路径，本质上是在说：别急着把最强的智能做成最强的行动者。

我更关注后两种。Agent 会来，而且会很深地进入工作流。但完全自主智能体不会那么快成为主流。模型会继续变聪明，可聪明和可信之间隔着一整套工程、制度和责任问题。

## 为什么长期完全自治很难

今天的模型已经能完成很多以前想不到的任务。METR 的 [Task-Completion Time Horizons of Frontier AI Models](https://metr.org/time-horizons/) 用 time horizon 来衡量模型能完成多长的人类专家任务，这比普通 benchmark 更接近 Agent 讨论。它把问题从“模型会不会做题”推进到“模型能不能连续做事”。

但这个指标不能读得太远。METR 自己也提醒，time horizon 不是模型能连续自治多久。它衡量的是特定任务、特定环境、特定可靠性下，AI agent 能完成相当于人类专家多长时间的任务。真实世界里的长期任务要脏得多。

真实任务经常没有干净目标。用户一开始也说不清自己要什么，中途又会改主意。上下文散在聊天记录、文档、会议、邮件、公司习惯和人的偏好里。反馈也慢。今天做的决策，可能两周后才知道有没有坑。更麻烦的是，很多错误并不会立刻报错。它们会变成错误承诺、错误库存、错误审批、错误沟通，然后在系统里慢慢发酵。

长期自治还会遇到治理问题。一个 Agent 如果只是帮我总结网页，出错了我最多骂它两句。可如果它要持续经营一个广告账户、管理一个供应链、处理客户投诉、修改生产系统配置，事情就变了。它需要预算边界、权限边界、异常上报、审计记录、回滚机制，还需要有人对结果负责。

这里最麻烦的不是“它会不会犯错”。人也会犯错。麻烦的是，Agent 犯错的方式不太像人。它可能在看起来很合理的文本里埋下一个错误假设，也可能被网页里的恶意指令带偏，还可能因为工具权限太大，把一个小误解放大成真实损失。模型越像一个能干的人，用户越容易忘记它其实没有人的常识、责任感和处境理解。

我不太认同“模型再强一点，完全自治自然就到了”。长期自治不是把模型运行更久，而是让系统在更久的时间里仍然能被治理。它需要稳定目标、可靠反馈、可控权限、可解释过程和责任闭环。少一块都不舒服。

## User in the loop 不是临时补丁

很多人说 Human in the loop 时，脑子里想的是一个很笨的画面：AI 每做一步都问人，用户不停点确认。这当然很烦，也很低效。如果 HITL 只是这样，它确实会被淘汰。

我说的 User in the loop，不是让人当橡皮图章。更接近的画面是：人还握着方向盘、刹车和责任，只是不用亲手拧每一颗螺丝。

人提出目标，Agent 给出计划。人不一定检查每一步，但应该能看见它准备碰哪些资源、可能产生什么后果。低风险动作可以自动执行，高风险动作要批准。执行过程中，人可以暂停、修改目标、接管任务。结束以后，系统要留下足够清楚的记录，方便回看它做了什么、为什么这么做、错在哪里。

Anthropic 在 [Measuring AI agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) 里有一个我觉得很重要的判断：有效监督不是把人塞进审批链，而是让人能真正监控和介入。经验用户有时会减少逐步批准，但这不等于完全放手。他们更像是在看仪表盘，在风险变大时接管。

OpenAI 的 [Harness engineering](https://openai.com/index/harness-engineering/) 说得更直白：Humans steer. Agents execute. 这句话比很多“数字员工”宣传都准确。人类不一定亲手做每个步骤，但仍然要设计环境、定义目标、建立反馈，并判断结果。

短期内我更看好分级自治。有些任务只需要观察和建议，比如整理会议纪要、找异常数据、生成候选方案。有些任务可以让 Agent 先执行，再由人验收，比如改一小段代码、草拟邮件、准备报告。有些任务可以自动执行，但要有预算和回滚，比如内部数据同步、低风险客服分流。再往上，涉及钱、法律责任、用户隐私、生产系统和公共安全的动作，就不该轻易交给无人值守的 Agent。

这不是保守。是现实。

## 我们应该做什么

如果要认真建设 Agent，我觉得方向反而很清楚。

先做边界清楚的小工作流。不要一开始就做“帮我经营整家公司”的万能 Agent。让它负责一个具体环节：收集信息、生成草稿、检查差异、跑测试、分拣工单、监控指标。任务越具体，反馈越清楚，Agent 越容易变好。

权限要小。能只读就别给写权限，能在沙箱里跑就别直接碰生产环境，能用临时凭证就别给长期 token。过去软件权限错了，是某个功能越权；Agent 权限错了，可能是一个会推理、会组合工具、会绕路的系统越权。

观测要早做。日志、轨迹、工具调用、输入输出、人工修改、失败原因，这些东西一开始看起来麻烦，但没有它们，Agent 产品最后会变成玄学。你不知道它为什么成功，也不知道它为什么失败。更糟的是，你以为它在变好，其实只是 demo 更顺了。

评测也要贴近场景。通用 benchmark 只能说明底模能力，不能说明你的 Agent 在真实业务里可靠。客服 Agent 要看误分率和升级率，代码 Agent 要看测试、diff 和 review，浏览器 Agent 要看任务完成率、误点击、敏感数据暴露和恢复能力。不同任务要有不同的验收口径。

最后，人类角色要重新设计。不要把人当成阻碍自动化的旧零件。人应该在目标、边界、异常和责任上出现，而不是被迫盯着每个低价值步骤。好的 Agent 系统会把人从重复执行里拉出来，但不会把判断和责任一起拿走。

## 我们不要做什么

我最不喜欢的一种做法，是把“完全自治率”当成先进程度。好像人介入越少，系统就越高级。这在低风险、强反馈的任务里可能成立，但一旦任务进入真实组织，未必如此。

很多 Agent 项目最后失败，不一定是模型太笨，常见问题反而更朴素：目标太大、权限太重、价值不清、成本失控。Gartner 预测到 2027 年底会有超过 40% 的 agentic AI 项目被取消，理由包括成本上升、业务价值不清和风险控制不足。这个数字不一定精确，但方向我相信。很多公司现在买的不是 Agent，是焦虑。

也不要把 demo 当生产。一个 Agent 能在录屏里订机票、改网页、跑代码，不代表它能在一万个真实用户那里稳定工作。生产系统最怕的不是一次失败，而是失败不可见、不可复现、不可追责。

更不要把 prompt 当治理。Prompt 可以约束行为，但它不是权限系统，不是审计系统，不是法律责任，也不是安全边界。让模型在 system prompt 里写“不要泄露隐私”，和真的限制它访问隐私数据，是两件事。

我也不建议急着把最强模型直接接到最高权限工具上。强模型更会做事，也更会把错误做成完整方案。它越像一个可靠员工，越需要真实的员工管理方式：职责、权限、考核、审计、离职交接，甚至事故复盘。

## AGI/ASI 讨论不能绕过可信

说到这里，就会回到 AGI/ASI。

很多关于 AGI 的讨论喜欢从能力外推：模型数学更强（在写到这一段时，GPT解决了 Erdos 平面距离猜想）、代码更强、工具使用更强、长任务更强，所以 AGI 越来越近。我不否认能力在推进，也不否认某些跳变可能发生。只是我越来越觉得，AGI 讨论如果只看能力，会漏掉最麻烦的部分。

一个系统足够通用，不等于它足够可信。一个系统足够聪明，也不等于它适合被授权长期行动。

OpenAI 的 [Superalignment](https://openai.com/index/introducing-superalignment/) 项目当年提出过一个很直白的问题：如果系统比人类更聪明，人类如何监督它？这个问题放到今天的 Agent 上，已经有了一个小号版本：如果 Agent 能比普通用户更懂浏览器、更懂代码、更懂金融产品、更懂公司流程，用户如何判断它的建议和行动是对的？

现在的答案还不够好。我们有评测，有安全策略，有人工审核，有红队测试，有模型监控，有权限系统。但这些东西加起来，还不是“我可以放心让一个通用智能体长期自主行动”的答案。

这也是我认为 AGI/ASI 仍然比较遥远的原因。模型会继续变强，但“强”距离“可以长期委托”还有很远。真正进入社会结构的智能，必须被制度化。它要能被授权，也能被撤权；能执行，也能解释；能学习，也能被审计；能带来收益，也能在出错时被停止。

## 结尾：未来是重新分工

我不觉得 Agent 的未来是彻底取代人类工作者。至少短期不是。

更可能发生的是，工作被重新切开。边界清楚、反馈明确、重复性高、数字化程度高的部分，会越来越多交给 Agent。人会继续留在目标定义、质量判断、跨人协调、伦理和责任这些位置上。很多岗位会变，某些岗位会消失，也会有新的岗位长出来。这个过程不会温柔，但它也不是一句“AI 取代一切”能概括的。

Agent 真正有价值的地方，是让人把一部分执行委托出去。可委托不是放弃控制。委托的前提是我知道你能做什么、不能做什么、做错了怎么办，以及我什么时候可以叫停。

我对 Agent 未来的判断很简单：我们会继续走向更强的自治，但主流形态会先是可治理的自治。人类不会一直亲手执行每一步，也不会很快从系统里消失。

未来的 AI 未必像一个完全独立的同事。它更可能像一层到处存在的行动能力，被嵌进浏览器、IDE、手机、公司后台、数据系统和个人工作流里。它会改变工作，也会改变人对自己能力边界的想象。

但在它足够可信之前，最好的问题仍然不是“能不能让它自己干完”，而是“我们应该怎样把一部分世界交给它”。

## 参考资料

- [从豆包手机风波看通用 AI Agent 的困局与出路](https://www.gm7.org/archives/23470)
- Anthropic, [Trustworthy agents in practice](https://www.anthropic.com/research/trustworthy-agents)
- Anthropic, [Measuring AI agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy)
- Anthropic, [Our framework for developing safe and trustworthy agents](https://www.anthropic.com/news/our-framework-for-developing-safe-and-trustworthy-agents)
- OpenAI, [Practices for Governing Agentic AI Systems](https://openai.com/index/practices-for-governing-agentic-ai-systems/)
- OpenAI, [Introducing Superalignment](https://openai.com/index/introducing-superalignment/)
- OpenAI, [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)
- METR, [Task-Completion Time Horizons of Frontier AI Models](https://metr.org/time-horizons/)
- Gartner, [Gartner Predicts Over 40% of Agentic AI Projects Will Be Canceled by End of 2027](https://www.gartner.com/en/newsroom/press-releases/2025-06-25-gartner-predicts-over-40-percent-of-agentic-ai-projects-will-be-canceled-by-end-of-2027)
- Gartner, [Gartner Says Applying Uniform Governance Across AI Agents Will Lead to Enterprise AI Agent Failure](https://www.gartner.com/en/newsroom/press-releases/2026-05-26-gartner-says-applying-uniform-governance-across-ai-agents-will-lead-to-enterprise-ai-agent-failure)
- Microsoft, [2026 Work Trend Index: Agents, Human Agency, and the Opportunity for Every Organization](https://www.microsoft.com/en-us/worklab/work-trend-index/agents-human-agency-and-the-opportunity-for-every-organization)
- NIST, [Announcing the AI Agent Standards Initiative for Interoperable and Secure AI Agents](https://www.nist.gov/news-events/news/2026/02/announcing-ai-agent-standards-initiative-interoperable-and-secure)
- OWASP, [AIVSS Crosswalk](https://aivss.owasp.org/aiuc-aivss-crosswalk)
- Yoshua Bengio, [Superintelligent Agents Pose Catastrophic Risks: Can Scientist AI Offer a Safer Path?](https://yoshuabengio.org/en/publication/superintelligent-agents-pose-catastrophic-risks-can-scientist-ai-offer-safer-path)
