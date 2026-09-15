---
title: "Interview Preparation: Questions, Answers, and Questions to Ask"
title_zh: "面试准备：问题、回答与反问"
date: 2026-09-07 00:00:00 +0800
categories:
- Work & Society
- Career & Learning
tags:
- Interview
- Career
author: Hyacehila
excerpt: "An evolving collection of potential interview questions, prepared answers, and questions to ask interviewers."
description: "An evolving collection of potential interview questions, prepared answers, and questions to ask interviewers."
excerpt_zh: "整理面试中可能遇到的问题、提前准备的回答，以及向面试官反问的问题，随准备和面试经历持续更新。"
permalink: /blog/2026/09/07/about-interview/
lang: en
translation_key: 2026-09-07-about-interview
translation_status: machine
translation_source_hash: 123a967ede6e78ddb2f07a753dfaa6960def15a36343aa6095fc4daf19fdf34d
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

## Before We Begin: This Article Will Keep Evolving

This article is not a finished job-hunting guide. It is a work-in-progress collection of my thoughts on job hunting, interview preparation, and questions to ask interviewers. I will keep updating it as I apply for jobs, receive interview feedback, understand different roles, and reconsider my choices. For now, it is an evolving blog post rather than a final version. In fact, most articles on this blog are continually revised, so patches and occasional restructuring are fairly common.

The focus here is on finding work that suits me and giving it a try, rather than simply getting an offer. Not finding a job I desperately want is normal, and not necessarily a bad thing. Becoming too attached to a title or some other aspect of a job can make it easy to overlook the real costs and end up having an unpleasant time.

Job hunting is not just a one-way screening process or sending out a resume and waiting. With limited information, I need to keep asking: What do I want to do? What am I suited for? Is this opportunity worth pursuing? Will I be able to keep growing after joining the team?

This article will also collect questions I might ask or be asked, along with ideas I find useful. Some may not warrant a full blog post, but I still need somewhere to keep them, so I will put them here for now.

## Some General Advice

### Decide on a Direction Before Applying

First, we need an overall plan. Industry and academia often care about quite different things. Even within large companies, business-facing roles and research lab positions can involve very different work.

Once that direction is clear, it should guide the emphasis of the resume, how projects are presented, and the application strategy. Different paths call for different ways of explaining the same experience. Industry puts more weight on solving real problems, collaborating to deliver results, and bringing technology into business processes. An academic path places more emphasis on research questions, methodological innovation, publications, and long-term research potential.

A target role should not be judged by its title alone. Whether a job suits me usually depends on the industry, city, salary, room for growth, fit with my expertise, and whether the company culture matches the pace of work and amount of overtime I am comfortable with.

Many job titles sound similar while the actual work differs substantially. In algorithms, data, backend development, or agent development, some roles focus on research and exploration, others on business delivery, and others on engineering platforms and toolchains. I may need to work this out from the job description or ask around; at the very least, I should find out during the interview.

Planning is not a one-time decision. External opportunities, market demand, and personal circumstances change. A useful plan should leave room to adjust based on application results, interview feedback, and changing interests.

### How I Judge Whether an Opportunity Is Worth Pursuing

I consider several dimensions: income, direction, growth, fit, and team atmosphere. Salary matters, but it is not the only measure. If long-term prospects and skill development matter more, I need to look closely at the team's direction, the quality of its work, the opportunities to learn, and whether I will get to work on worthwhile problems.

Alignment with my intended direction is especially important. A role may pay well in the short term, yet have little to do with what I want to pursue over time. In that case, I need to consider whether it will lead me down a path I do not want to follow. Conversely, an opportunity that is not immediately optimal may still be worth considering if the team's problems are real and its technical work fits my long-term interests.

Team atmosphere matters too. Everyday collaboration, decision-making, attitudes toward technical debt, and room for growth and exploration all affect the experience of working there. Many of these things only become clearer when I ask questions during the interview. Those questions are part of deciding whether to take the job, rather than a polite formality at the end. I include some examples later in this article.

### Interview Preparation Goes Beyond Memorizing Answers

Interview preparation cannot be reduced to memorizing standard technical questions and answers. For technical roles, coding fundamentals and common questions certainly matter: they affect whether I pass the initial screening and technical interviews. But project experience also needs preparation, or the explanation can easily turn into a chronological list of things I did.

I try to organize each experience around a few questions: What problem needed solving? What made it difficult? What was my role? What methods did I use? What were the results? What were the value and limitations of the work? This makes it easier for the interviewer to understand the project and assess my actual contribution.

Research projects, collaborations with companies, papers addressing practical problems, competitions, and internships become useful interview material when they are organized around a clear sequence of questions. Otherwise, even a long list of experiences can amount to little more than names on a resume.

## Questions About My Background and Interests

### Some Thoughts on Embodied AI

My resume may seem to have nothing to do with embodied AI or robotics, but I am interested in the field. My focus is not on training VLA models. As with the problems I have been studying, I care more about applications and how to make them work in practice.

Putting embodied AI into practice may involve more than a powerful VLA model. The release of GPT 6 Astra may offer some new perspectives on generative AI.

If we want to draw a picture, we may not need a diffusion model: a text model could use tool calling to draw a pelican in SVG. If we want to build an excellent 3D model, a generative model such as Meshy AI or Hunyuan's 3D model might not work as well as letting AI gradually build it in Blender through tool calls and visual feedback. Generative language models now support omni-modal inputs, allowing them to move closer to human goals through repeated cycles of "generate—render/execute—inspect—revise."

We can call this Agentic Generation, another form of generation alongside end-to-end model generation. From this perspective, both the UI generation workflow I worked on at NetEase and the deployment of embodied AI can ultimately be placed under Agentic Generation. Traditional generative models, whether for images, video, or motion, can become tools within that system. Optimizing the agent system itself would then become a core task alongside fine-tuning Pi-0.5.

Compared with generation through an end-to-end model, Agentic Generation offers another major advantage: **the entire generation pipeline can be inspected and controlled.** Users need results that meet their requirements. Producing those results may involve choosing representations, calling tools, observing intermediate outputs, making local revisions, and validating the result. Agentic Generation can accept constraints expressed in language or agent logic code, route tasks to different capabilities, preserve intermediate states, and make local corrections. When the whole system can be traced and reproduced, debugging and changing it will always be faster than training a model.

Agentic Generation also explores trade-offs around questions such as: **Which decisions should remain inside the model, and which capabilities should be implemented through external tools, state management, and execution mechanisms? How should these boundaries change with the task?** Combining a lower-level VLA with a higher-level agent may be a more practical near-term approach, but where is the right balance?

For a robot operating in the physical world, many actions cannot be undone with Ctrl+Z. How should tools for embodied AI be designed, and what feedback should they give the "brain"? In [the discussion of forward knowledge injection and backward feedback in "What Problems Are We Really Solving When We Build AI Agents?"](/en/blog/2026/07/28/what-problems-are-we-really-solving-when-building-ai-agents/#feedback-design), I wrote: **Tools are never just a list of tools; they are the entire world an AI agent can observe and affect.** Tool calling has mainly focused on virtual environments. In the physical world, this view needs to translate into more concrete interface and feedback design.

For embodied AI, a tool should return more than "call successful." The brain also needs to know whether the action actually completed, what changed in the environment, and what remains possible after a failure. How should this information be obtained from sensors, VLAs, and lower-level control systems? At what granularity, and when, should it be passed to the higher-level agent? These are questions I want to investigate further, and I suspect no one has an answer to them yet.

### What Is the Biggest Challenge I Have Encountered?

I seem to have been asked this question in more than one interview. It is especially popular in the AI interviews that have become common recently: they like to throw a pile of fairly pointless questions at you, apparently simply because they want to ask them.

If I had to name the biggest challenge I have encountered, I do not think it would be one specific technical problem. It would be this: **deciding what to do next when I do not even know whether an answer exists.**

Much of my past work has not been the kind of engineering problem with a clearly defined specification. This is especially true of AI and agent projects. At the beginning, even the boundary of the problem is often unclear. Is the model capable enough? Can this technical direction work? What would count as an effective result? Does the problem itself even have a solution? None of these questions necessarily has an answer yet.

The approach I have gradually become used to is forming a belief from the information and experience I currently have: the technical direction I consider most plausible and most worth trying. I think having such beliefs is natural, and probably necessary, for a technically oriented developer. Mine come from the projects I have worked on before.

I then avoid spending too long trying to prove that my first judgment was correct. Instead, I build a minimal prototype around it as quickly as I can. The most important purpose of a prototype is not necessarily to become the finished product. It is to give me new information through real technical feedback. Every prototype and every failure changes how I understand the problem. I use that information to revise my belief, choose a more promising path, and begin another iteration.

Over time, I have come to think that the most important ability when facing an unknown problem may not be finding the right answer immediately. It is this: **can you form a good enough judgment, move quickly, and keep letting reality correct that judgment?**

This is also why I now rather enjoy dealing with problems whose answers are unclear. A problem with a known solution is mostly a matter of execution. When you do not know whether a solution exists, you have to keep observing, judging, and trying, then slowly find structure in the confusion. That process is challenging, but the challenge is also part of the fun.

So when someone suddenly gives me a completely unfamiliar problem, my first reaction is usually no longer:

“Do I know how to do this?”

It is:

“Given what I know now, which path do I believe in most? Can I build something small enough to see what reality tells me?”

I think this may be the most important habit I have developed for dealing with difficult problems over the years.

## Questions to Ask Interviewers: What I Want to Find Out

The main purpose of asking questions is to understand what the team actually does, what the role really requires, what I would be responsible for, and whether the job fits my own criteria.

The questions below lean toward AI agent engineering roles. They help assess the team's technical direction, approach to putting products into use, engineering maturity, and collaboration. Some are more general and can also serve as references. Another useful resource is [viraptor/reverse-interview](https://github.com/viraptor/reverse-interview).

### Agents and Engineering

**I understand that the company is investing in agents. Are the planned applications mainly intended for internal use, such as improving developer productivity, answering questions over internal knowledge bases, or automating operations? Or are there already clear use cases for commercial products aimed at external customers? What problems is the team working on, and what would my main responsibilities be? Could you give some concrete examples?**

Note 1: This is a fairly standard opening question for understanding a department's work. Almost every interview should include this question or something similar.

Note 2: ToB (company-facing) versus ToC is an important distinction for AI agents in my view. ToB applications can have more tolerance for errors and make it easier to test and iterate on new technologies, but may sit outside the company's core business. ToC applications require more attention to hallucinations, stability, and user experience. They can offer more opportunities to learn systems engineering and the design of safety boundaries, while making it harder to introduce new technologies.

Note 3: More detailed, technically informed follow-up questions may work in your favor. Offering two possible approaches and guiding the discussion can help. If every "A or B" question gets answered with "both," or the interviewer avoids answering, another option is to state your assumption explicitly and let them correct it.

**There are many different agent applications emerging across the industry. Is the team's focus on well-defined, structured tasks, such as automated processes executed through API calls, or on understanding and reasoning over unstructured knowledge, such as structure-aware processing of long documents and Agentic RAG?**

Note: The former puts more emphasis on automated workflows and requires a deep understanding of the business itself. The latter focuses more on agent development technology. Greater autonomy also means greater dependence on model capabilities and the developer's understanding of safety boundaries. This is a technical "A or B" question closely tied to agents.

**For the agent applications the team is pursuing, is the system mainly designed as a copilot that requires frequent human feedback and confirmation, or as a closed-loop workflow with a high degree of automation in a specific business domain?**

Note: Human involvement remains important. For a highly automated system, high-quality automated verification becomes central. If more human intervention is acceptable, expanding the system's functionality may be more valuable than pushing automation to its limits. This is another technical "A or B" question closely tied to agents.

**Given the needs of complex business scenarios, is the current focus on improving a single agent's ability to break down complex tasks, or has the team started exploring multi-agent collaboration in practical applications?**

Note: Multi-agent collaboration has gradually moved from technical reports into demos. Although reliability remains questionable, it is still an interesting frontier. It is closer to technical exploration than mature technology, however, and highly autonomous multi-agent systems may encounter problems similar to those in human collaboration. This is another technical "A or B" question closely tied to agents.

**How does the team evaluate its agents? Beyond conventional LLM benchmarks, are there business-specific measures of tool-call accuracy, execution traces, or automated evaluation frameworks?**

Note: Evaluation is unavoidable for agents. Without it, it is difficult to tell whether a system has improved or the demo simply looks smoother. If a company wants highly autonomous agents but does not maintain its own evaluation suite, it has not yet got on the right track.

**How does the team balance cost and latency? Does it mainly rely on established proprietary models, or does it route different requests to different model tiers? Are the trade-offs between cost, latency, and performance evaluated systematically?**

Note: These are unavoidable engineering questions. Moving from a technical demo to production requires careful consideration of cost, latency, performance, and safety boundaries. Production safety boundaries should not be casually traded away, while taking a little longer or spending a little more can sometimes be acceptable.

If a team has thought this through, it suggests that its application is already reasonably usable. Simply having an automatic router does not count, although having no routing at all is worse.

### Teamwork, Collaboration, and Personal Growth

**How much freedom does each developer have to make decisions? How is the team structured, how do people collaborate, and how is everyday work divided? How are disagreements usually handled?**

Note: This gives an overview of how the team works and opens the door to follow-up questions about team atmosphere and technical collaboration.

**What does a typical working day or week look like?**

Note: This is one possible follow-up.

**AI coding tools are now part of most programmers' workflows. How does the company view AI-assisted coding? Are employees allowed to use it extensively to speed up iteration? Does the team provide shared tools, accounts, or usage guidelines?**

Note: It really feels hard to survive as a programmer without AI now, though this comment does not apply to infrastructure experts or people writing kernels. Providing a Pro5x subscription is the minimum a company should meet.

**What do you like most about working here? Why did you choose this company, and why have you stayed?**

Note: This feels a bit like putting the interviewer on the spot. I copied it from the internet and take no responsibility for it.

**What are the team's working hours and overtime expectations? How does on-call work, and is overtime paid?**

Note: Work-life balance may deserve more thought for a full-time position. For an internship, I am less concerned. I already live the "9117" life at university; can a company really work me harder than the lab does? The Lei Jun Building has free air conditioning, free coffee, and showers. Give me a bed and I could live in the office.

**Can I contribute to open-source projects? Would I need approval?**

Note: The question speaks for itself.

**Does the company hold technical knowledge-sharing sessions? If so, how often?**

Note: The question speaks for itself.

**Are there company-wide learning resources, such as ebook subscriptions or online courses? Is there a budget for certifications or other learning expenses?**

Note: Spending the company's money on my own learning has the satisfying feel of getting a perk. It may be tiny compared with my salary, but somehow it still feels great.

**Is the company profitable? If not, when does it expect to become profitable? If it is, roughly what is its annual revenue? What are its plans for the future?**

Note: This is worth asking at startups and smaller companies that operate like startups. After all, a company that never makes money is slowly going out of business.

**What is the balance between remote work and working in the office?**

Note: Does remote work really exist in China?

This article will continue to be updated over time.
