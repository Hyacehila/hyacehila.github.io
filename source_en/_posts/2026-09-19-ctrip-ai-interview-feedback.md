---
title: "Don't Substitute Action for Thinking: Feedback from a Ctrip AI Interview"
title_zh: "不要用行动代替思考：携程 AI 面试反馈"
date: 2026-09-19 12:00:00 +0800
categories: ["Work & Society", "Career & Learning"]
tags: ["Interview", "Career", "Problem Solving", "Decision Making"]
author: Hyacehila
excerpt: "A Ctrip AI interview gave me an unexpected but useful piece of feedback: move from solving problems after they appear to modeling them before acting, and let hypotheses, validation, and contingency plans guide the work."
description: "A Ctrip AI interview gave me an unexpected but useful piece of feedback: move from solving problems after they appear to modeling them before acting, and let hypotheses, validation, and contingency plans guide the work."
excerpt_zh: "携程 AI 面试给了我一个有些意外、但很有价值的反馈：从遇到问题以后再想办法解决，转向在行动之前先建立问题模型，用假设、验证和预案来指导行动。"
mathjax: false
hidden: true
permalink: '/blog/2026/09/19/ctrip-ai-interview-feedback/'
lang: en
translation_key: 2026-09-19-ctrip-ai-interview-feedback
translation_status: machine
translation_source_hash: ec8f9a87c68a8db4451b484350cbc15828abb1189dd12ca8992a62fe73496414
---

<aside class="translation-notice" role="note">This English version was translated from the Chinese original. Some personal phrasing may require verification.</aside>

I've done quite a few AI interviews recently. Ctrip was the only one that gave me fairly clear feedback. Some of it also sounded close to ideas I associate with the company. What I want to write about here, though, is a question it made me reconsider: how have I been solving problems, and what could I do differently?

Looking back, I've solved plenty of problems in my previous work. When a metric moved unexpectedly, I tried to work through the possible causes. When collaboration across teams stalled, I talked to people and used quantitative data to help us make a judgment. When I encountered something unfamiliar, I gathered information, tried a few approaches, and gradually found patterns in the results. Most of these things worked out reasonably well, so I don't think those methods were fundamentally wrong.

But put them together and a pattern emerges. I often started organizing my thoughts and actions only after a problem had appeared. Why an approach was worth trying first, which conditions it depended on, and what to do if those conditions failed sometimes became clear only after I'd spent a while working on it.

The feedback made me want to move some of that thinking earlier. Before getting started, I could form a preliminary explanation, state a few hypotheses I can test, and consider where a mistake in my judgment would first become visible. I'll write the idea down here. Whether I can make good use of it in actual work is something I still need to try.

## I've gathered a lot of information. What comes next?

My usual approach has been to learn about a problem first. Read more, try a few possible solutions, and form a judgment once I've gathered enough information. In an unfamiliar field, some of that is necessary. It's hard to propose a useful hypothesis when I don't yet know what the problem involves.

Research can also keep going indefinitely. I finish one article, discover a few concepts I don't understand, and keep reading. I learn about one approach, then find several others that also seem worth investigating. Eventually I know more about the field, but the question I started with may still be unanswered. The information keeps accumulating, while I may have no idea how the next piece of it will affect my decision.

That's where I want to make a change. I can use what I already know to form a rough explanation, specify what I'm trying to solve and what I'm leaving aside, and then decide what evidence to look for. The explanation can be crude, and I may overturn it later. At least it gives me a reason for taking the next step.

Take an unexpected change in a metric. Before pulling every available dataset, I could think about three directions: the broader environment, competition, and internal changes. Has market demand fallen? Has a competitor launched a promotion or adjusted its product? Have we changed a version, a strategy, or a channel? The data itself also needs a place in this picture. A collection error or a changed metric definition can leave people analyzing a chart for quite some time.

These are only candidate explanations. I could call them H1, H2, H3, and H4, or draw a simple tree of possible causes. The format doesn't matter much. What matters to me is whether I can go on to say what I should observe if an explanation holds, and how I would revise my judgment if I don't see it.

If I suspect an internal version change, I can check whether the timing and scope of the fluctuation match that change. If I suspect falling overall demand, I need evidence about the market. These checks may not establish the cause, but they can help me decide where to look next. The hypotheses don't all deserve equal effort, either. I need to consider how much each one matters to the decision and what it would cost to test.

**Each round of research should ideally answer a specific question, and the evidence should be able to change what I do next.** That gives information gathering a direction.

There's a caveat I need to add for myself. Starting with a hypothesis can easily become starting with an answer and searching for material that supports it. A hypothesis is an explanation I'm using provisionally. I need to pay just as much attention to results that contradict it. If none of my hypotheses explain what's happening, I need to go back and rethink the problem. Drawing a diagram doesn't oblige the problem to fit inside it.

## Could the postmortem happen a little earlier?

Another familiar process has been to get a solution running, analyze problems as they appear, fix them, and eventually conduct a postmortem. Plenty of work can only be understood by doing it. When the cost of trial and error is low, a small experiment often reveals problems more readily than a discussion does.

But once a project is live, resources are committed, and several teams have begun working to the original plan, discovering that a condition the plan depends on doesn't hold is much harder to deal with. Some risks deserve a little thought before work begins.

A tabletop exercise sounds like a fairly grand undertaking. Here it could be quite simple. Suppose the plan fails: where would it most likely get stuck? Would we run out of resources, or does one step depend too heavily on a person, a team, or an external condition? If resources we assumed would be available aren't, which parts can continue and which need a different approach?

That leads to backup plans and stopping conditions. If A turns out to be wrong, what will I switch to? What signal should prompt an adjustment, and when should I stop altogether? I would like a preliminary answer before committing resources. After working on something for a long time, the time already spent can get mixed into the judgment about whether to abandon it.

The balance is difficult. I can't anticipate every risk, and writing an enormous contingency plan for a tiny experiment would be excessive. I want to focus first on decisions that, if wrong, would force many people to redo their work or would be expensive to reverse. Other uncertainties can be explored through small trials and observation during execution.

Postmortems still have a place. Their findings can also go somewhere else. A past mistake can become a condition to check before the next project begins, as well as a paragraph in the final review. That would give the experience a practical use when a similar problem comes around.

## Will we have to solve it all over again?

Even after moving analysis and risk planning earlier, one question remains. Once the current problem is solved, will the team still have to start from scratch the next time something similar happens?

Take the metric example again. Finding the cause of this decline can help the business recover. To handle the next one faster, though, we need to consider which metrics deserve continuous monitoring, what level of change should trigger an alert, what to check after the alert arrives, and who decides whether to intervene. We can also keep records of similar cases, helping the next person distinguish between changes that can be watched a little longer and those that need immediate attention.

These are concrete arrangements. Saying that we should build a “system for solving problems” leaves almost everything unspecified. We need to describe the checks, the basis for decisions, and the people responsible before the team can use it in its work.

The same applies to collaboration across teams. I can talk to people and get this particular piece of work moving. But if it stalls in the same place every time, it's worth looking at whether goals align, responsibilities are clear, information arrives in time, and someone has authority to make the necessary decision. Otherwise, however well this conversation goes, next time we'll have to find people and explain everything again.

I previously described this as a difference between executors and planners. Now I don't think I need to divide people so neatly. The same person working on the same task needs to resolve the immediate issue and can also ask which lessons are worth turning into practices the team can reuse. An isolated minor problem may not justify a whole process. If it keeps recurring, or is expensive to handle even once, doing a little more makes sense.

## A few questions to keep for myself

This is close to something I kept asking in my [AI Native Game post](/en/blog/2026/07/12/ai-native-game/). Before putting AI into a game, we need to understand why players would want it and what it can contribute to the gameplay. Building a feature establishes that we can build it. Whether players will enjoy the game more because of it is a separate question. Returning to this feedback, I also need to understand what I want to change before deciding which information to gather and which approaches to try.

For a sequence that's easy to remember, I'd keep these terms:

> Problem framing → Hypothesis → Validation → Decision → Execution → Risk control

In practice, that means stating the problem and its boundaries, proposing hypotheses, running the smallest useful check, and using the result to decide what to do. Risks need consideration before action and continued attention during execution. Sometimes I'll need to go back and revise a hypothesis. The words form a neat line on the page; actual work will probably move back and forth.

I'll keep a short checklist to consult when I encounter a complex problem:

1. What exactly am I trying to solve this time? What am I leaving aside?
2. Which variables mainly affect the outcome? What conditions does the plan depend on?
3. Given what I know, what explanations seem possible?
4. Which hypothesis deserves attention first? What evidence could support or overturn it fastest?
5. If my judgment is wrong, where would that show up first? Could I notice it in time?
6. What else could I do if it fails? What result would mean continue, adjust, or abandon?

The answers can be brief. They need to give the next action a basis. If I can't answer one of the questions yet, I can leave it as something to investigate next. Once the work is done, I can come back and compare: what did I miss, and which checks are worth keeping for next time?

## A final thought

“Don't substitute action for thinking, or the volume of information for judgment.” I still want to keep that sentence. It relates to how I've worked, and it reminds me that spending a long time on something doesn't automatically mean I understand it better.

My previous experience solving problems remains useful, and I'll keep acting and experimenting. The adjustment I want to make is to state my judgment before committing resources, design a way to test it, and think about how I would notice and handle being wrong. Afterwards, I can consider which lessons might save some unnecessary work next time.

Finishing this post only means I've thought the idea through once. When I encounter the next complex problem, I want to try writing down my current explanation and what result would make me change it. Then I'll start working with those two questions in mind and see how it differs from what I used to do.
