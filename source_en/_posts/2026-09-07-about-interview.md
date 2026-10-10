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
translation_source_hash: 9b18ae2e79302a0a8417bdd2e6b716f9a0ce711d527f5a0b91b1759ea7a51c79
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

### What My Background Outside Game Client Development Brings to AI Tools for Game Production

I do not come from a traditional game client development background. My work has mainly involved statistics, machine learning, and AI agents in different fields. For building AI tools that improve game production, I think this can be an advantage: I find it easier to step outside established roles and tools, start with what the model can do and what the team needs to deliver, and reconsider how the work in between should be organized. Existing roles, tools, and handoffs are all worth another look. Some need to stay, but I want to understand what purpose each one serves.

Many parts of today's pipeline are arranged to make work easier for people: one role produces a document, another translates it into an engineering artifact, and someone else integrates and checks it. The information a model can read, the tools it can call, and the speed at which it receives feedback may allow those handoffs to work differently. We can give each role an assistant, or work backward from the final deliverable and ask which intermediate steps still need to exist. That second approach interests me more. Having fewer habits tied to an existing workflow makes it easier for me to ask these questions from outside and try ways of dividing work that suit AI.

I discussed a similar issue in ["Why Enterprise AI Gets Stuck in Pilots"](/en/blog/2026/07/22/enterprise-ai-from-delegation-to-absorption/). Even when employees can use AI, the team still needs to make a single generated output part of a repeatable delivery process: who provides the input, who checks and accepts the result, who takes over when something goes wrong, and how human corrections carry into the next run. I want to explore designing these working relationships together with the tools, so agents carry out routine tasks while people set goals, make key judgments, and handle exceptions. Existing team structures affect system design, and a new system can in turn change how people hand off work and collaborate. This can begin with one concrete task and evolve through actual use.

This view also comes from my experience building agents across domains. Security, public opinion analysis, and game UI involve very different business knowledge, yet I keep running into similar questions: what information is the model missing, which judgments can it make, and how should a tool's result be verified? In a new setting, I first clarify goals and constraints, then choose how to organize the system. Human roles are one reference point. At NetEase, I tried letting the model infer UI semantics and structure, passing the result through an intermediate representation for deterministic assembly, reusing existing controls, and checking the result through editor feedback. The path from a design mockup to a usable UI artifact showed me that representations and tool interfaces also deserve to be redesigned. They affect how easily a model can complete a task and whether errors can be corrected promptly. A game's project standards still need to be met, while the operations used to meet them remain open to exploration.

My statistics background makes me ask what a change actually improves. If one design mockup works, will a more complex interface work too? If we run the same task several times, will the results stay consistent? We need samples and comparisons to answer these questions. Time spent preparing materials, waiting for handoffs, checking results, and reworking them all needs to be counted. If the model generates quickly but someone downstream spends longer cleaning up, the process may not save work overall. I want this feedback to guide the next step: keep changes that help, find out where failures occur, and revise my judgment. Human corrections and exception records should also be retained. What we learn and verify can then feed into the knowledge, tools, and evaluations used for similar tasks.

In ["What We Talk About When We Talk About AI-Native Games"](/en/blog/2026/07/12/ai-native-game/), I kept asking what AI ultimately brings to players. I want to keep that question in mind when building tools for development: can the time saved let a team try one more approach, bring a design idea into the engine sooner, and get feedback? Model capabilities determine which new approaches we can try, while people still need to guide the direction, aesthetics, and experience of the game. I hope AI gives ideas that once stayed on paper because they cost too much to produce more chances to be built and tested. Coming from outside client development gives me a different place to examine the workflow. My experience with agents across domains and my training in statistics let me turn ideas into prototypes and check whether they help. That is the work I want to do in AI tools for game production.

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

The approach I have gradually become used to is forming a Belief from the information and experience I currently have: the technical direction I consider most plausible and most worth trying. I think having such a Belief is natural, and probably necessary, for a technically oriented developer. Mine comes from the projects I have worked on before.

I then avoid spending too long trying to prove that my first judgment was correct. Instead, I build a minimal prototype around it as quickly as I can. The most important purpose of a prototype is not necessarily to become the finished product. It is to give me new information through real technical feedback. Every prototype and every failure changes how I understand the problem. I use that information to revise my belief, choose a more promising path, and begin another iteration.

Over time, I have come to think that the most important ability when facing an unknown problem may not be finding the right answer immediately. It is this: **can you form a good enough judgment, move quickly, and keep letting reality correct that judgment?**

This is also why I now rather enjoy dealing with problems whose answers are unclear. A problem with a known solution is mostly a matter of execution. When you do not know whether a solution exists, you have to keep observing, judging, and trying, then slowly find structure in the confusion. That process is challenging, but the challenge is also part of the fun.

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

Note: Work-life balance may deserve more thought for a full-time position. For an internship, I am less concerned.

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

### HR Interviews and Salary Negotiation: Essential Questions to Confirm

At the HR interview and salary negotiation stage, judging whether a job suits me also means getting clear about some specific conditions: how compensation is structured, where I will work, how much time the job requires, and which benefits actually apply to me. This checklist can be worked through during HR interviews, compensation discussions, and before accepting an offer.

#### Compensation: Annual Package, Actual Payouts, and Negotiation

- [ ] **1. What makes up the annual compensation package?**

  Ask: How is compensation for this role structured? What is the gross fixed monthly salary, and how many months of salary are guaranteed each year? Are any additional salary payments conditional? How are quarterly performance bonuses, year-end bonuses, and allowances calculated, and are they already included in the quoted annual package?

  Record: List fixed salary, guaranteed bonuses, target performance pay, and bonuses that are not guaranteed separately. Check whether quarterly and year-end bonuses have been counted twice. Also ask whether the quote includes the employer's social insurance contributions, housing provident fund contributions, or other benefit costs.

- [ ] **2. How much compensation is variable, and how much is typically paid out?**

  Ask: Which bonuses in the annual package are guaranteed, and which depend on performance? How do individual, team, and company results affect quarterly and year-end bonuses? If this can be shared, what has the typical payout range been over the past year or two for someone in this role with a normal performance rating?

  Record: Assessment periods, the bonus associated with each rating, payment conditions, and whether there are circumstances in which no bonus is paid. Keep target bonuses separate from actual payouts; past results cannot be treated as a promise of future payments.

- [ ] **3. How do probation pay and first-year income differ?**

  Ask: How long is probation, and is the salary the same as after probation? For a new graduate, are quarterly bonuses, year-end bonuses, and additional salary payments prorated in the year of joining? Are there minimum service requirements or deadlines for completing probation?

  Record: Confirm how income is calculated for the calendar year of joining, the first twelve months after joining, and the first full calendar year. Note the performance assumptions behind each calculation.

- [ ] **4. When are salary and bonuses paid?**

  Ask: What is the monthly payday? When are quarterly bonuses, year-end bonuses, and additional salary payments normally paid? Are they paid in installments or deferred? Must I still be employed on the payment date, and how would leaving or changing roles affect settlement?

  Record: The work period covered by each payment, its payment date, and eligibility conditions. Money included in the annual package but not yet payable should not be treated as cash already available.

- [ ] **5. Are there signing bonuses, joining bonuses, or equity incentives?**

  Ask: Are there signing, joining, or retention bonuses, shares, or stock options? When are cash awards paid, and are they paid in installments? What are the conditions for equity grants, vesting, and realizing their value? Would leaving early require repayment of any award, and how is the amount calculated?

  Record: List one-time cash payments, awards with service requirements, and equity separately. For equity, record the number granted, the vesting schedule, and the conditions for realizing its value, then assess it separately.

- [ ] **6. What job level does the quote correspond to, and what can still be negotiated?**

  Ask: What job level and pay band does this quote correspond to? Which parts can still be adjusted: fixed monthly salary, bonuses, job level, or a joining bonus? Would increasing one part reduce another?

  Record: Each final agreed amount and its conditions. Compare fixed and variable compensation before and after the adjustment, and confirm exactly what has increased when the annual package is said to be higher.

#### Work Location: Address, On-site Assignments, and Transfers

- [ ] **7. Where exactly will I normally work?**

  Ask: Which district, office campus, and building is the workplace in? Can you provide the full address? Will the location be the same during training, probation, and after probation?

  Record: An address precise enough to check the commute. Compare commuting time and possible housing locations, and confirm whether the workplace has been finalized.

- [ ] **8. Does the role require business travel, on-site assignments, or work in other cities?**

  Ask: Does the role involve frequent visits to factories, client sites, or other cities? Are there regular on-site assignments? How long does each assignment typically last, how much of the year does it take up, and can I return on weekends?

  Record: Locations, frequency, duration, and whether these assignments are mandatory. For a group IT services or client delivery department in particular, clarify which team I would work with day to day and where the work takes place.

- [ ] **9. How are location changes and remote work arranged?**

  Ask: Could there be rotations, moves to another office campus, or transfers to another city? How are these changes discussed beforehand? If remote or hybrid work is available, does it apply to this role, what approval is needed, and is it available during probation?

  Record: Confirmed arrangements, the possible scope of changes, and how the employment contract describes the work location. Check both the remote work policy and what this team actually permits.

#### Working Hours: Daily Schedule, Overtime, and On-call Duties

- [ ] **10. What are the actual working hours, and are weekends normally free?**

  Ask: What are the official starting and finishing times, and when do team members usually arrive and leave? How long is the lunch break, and are both weekend days normally free? During releases, delivery deadlines, or busy periods, how does the schedule change, and for how long?

  Record: Official hours, the usual actual schedule, and arrangements during busy periods separately. Ask the future manager to explain the actual pace, ideally with an example from a recent project.

- [ ] **11. How frequent are on-call duties, overnight responses, and holiday shifts?**

  Ask: Is there a duty rotation or on-call schedule, and does it require being available online or attending in person? How often does each person take a turn, and what response is expected overnight or on holidays? When do new graduates start participating, and what handover and support are provided?

  Record: Frequency, response requirements, how much rest time the duties actually take up, and how rest is arranged afterward.

- [ ] **12. How are overtime and time off in lieu recorded and compensated?**

  Ask: Which working-hours system applies to this role? How is overtime on evenings, weekends, and public holidays recorded, and is advance approval required? When do overtime pay, duty allowances, and time off in lieu apply? How is time off requested, and is there a deadline for using it?

  Record: Recognition rules, compensation rates, approval procedures, and whether time off can actually be scheduled. Obtain the policy that applies to this role.

#### Job Responsibilities and Career Development

- [ ] **13. What technical direction does the actual work follow?**

  Ask: Does the role mainly involve developing large models and agent systems, integrating business applications, data governance, building IT systems, or operations support? What project would I join first, and how is the work divided between development, delivery, and support?

  Record: Specific projects, deliverables, and responsibility boundaries. Check whether the work builds the core technical skills I want to develop. Ask the future manager to clarify anything HR cannot explain.

- [ ] **14. Have the department, team, and role been finalized?**

  Ask: Which department and team within the group would I join, and who would I report to? Is the role assigned during recruitment or after initial training? Are there rotations, how is the final placement decided, and can I participate in the choice?

  Record: Confirmed arrangements and anything still awaiting allocation. Confirm when the placement will be finalized and whether the actual responsibilities match the job description and the technical interview discussion.

- [ ] **15. How are probation and ongoing performance assessed?**

  Ask: What am I expected to achieve in the first three to six months, and who evaluates successful completion of probation? How often is regular performance reviewed, what goals are used, and how is feedback provided? Who assesses performance during on-site assignments or work across teams?

  Record: Goals, assessors, review periods, and feedback processes. Connect probation requirements with the criteria used for later bonuses.

- [ ] **16. What opportunities are there for salary reviews, promotion, and development?**

  Ask: How many salary or promotion review rounds are usually held each year, and when are new graduates first eligible? What career paths are available for technical roles? Are mentors, training, or internal transfers provided, and what conditions apply to each?

  Record: Review policies, eligibility, and opportunities that are actually available. An annual review cycle does not itself constitute an agreed salary increase.

#### Employment Contract and Joining Arrangements

- [ ] **17. Which company will I sign with, and what form of employment is involved?**

  Ask: What is the full legal name of the company signing the employment contract, and how long is the contract? Will I sign directly with the employer, or does the arrangement involve labor dispatch or outsourcing? Which entities direct the work, pay the salary, and handle social insurance?

  Record: Each entity and its relationship to the others. Confirm how the recruiting brand, the department, and the actual contracting company relate to one another.

- [ ] **18. What conditions and deadlines apply to accepting the offer and joining?**

  Ask: What are the offer response deadline, the deadline for signing the tripartite graduate employment agreement, and the latest joining date? What graduation, degree, medical examination, or background check documents are required? How can changes to graduation or reporting dates be coordinated? Does prior internship experience affect probation or compensation?

  Record: Required documents, key dates, outstanding joining conditions, and the contact person.

- [ ] **19. Are there non-compete restrictions, service requirements, or repayment clauses?**

  Ask: Does the contract or any attachment contain a non-compete restriction after leaving, and how are its scope, duration, and compensation defined? Do training or joining bonuses carry minimum service periods or repayment obligations? How do confidentiality and intellectual property terms affect personal projects, open-source contributions, or outside work?

  Record: The specific clauses and their conditions. Obtain the relevant documents before signing, and check the applicable rules wherever anything remains unclear.

- [ ] **20. Where can the agreed conditions be confirmed in writing?**

  Ask: Which compensation details, responsibilities, location, joining date, and key benefits will appear in the offer, contract, or attachments? Can other policies be provided as documents or confirmed by email? If documents disagree, can the difference be explained and corrected first?

  Record: The final versions, the person confirming them, and their effective dates. List any questions that still await an answer.

#### Benefits: What Is Provided, Who Qualifies, and What It Costs

The same questions can guide the discussion of each benefit: what is provided, who can use it, when it starts, how much I must pay, how to claim it, and whether it is already included in the quoted annual package. Social insurance, the housing provident fund, and different types of leave should also be checked individually here.

- [ ] **21. Where is social insurance registered, and what contribution rules apply?**

  Ask: In which city will social insurance be registered, and how are the contribution base and the employee's share determined for each type of insurance? From which month is enrollment handled, and how do arrangements during and after probation connect? If the workplace, contracting company, and insurance registration are in different locations, who helps with medical care outside the registration city or transferring the insurance relationship?

  Record: The registration city, how the contribution base is calculated, employee deductions, and enrollment timing. Check the current local rules and the actual payroll deductions.

- [ ] **22. What are the housing provident fund contribution base, rates, and start date?**

  Ask: In which city will contributions be made, and what are the employee and employer contribution rates? How is the contribution base for a new employee determined, and when is it adjusted? From which month do contributions start, and has the employer's contribution already been included in the quoted annual package?

  Record: The city, contribution base, both rates, and the monthly amounts. List the employer's contribution separately, along with the employee's payroll deduction.

- [ ] **23. What supplementary benefits are available, such as additional insurance and health checks?**

  Ask: Beyond basic social insurance and the housing provident fund, are there supplementary insurance plans, enterprise annuities, or supplementary provident fund contributions? Must I contribute personally, and is participation voluntary? When do I become eligible, and are the employer's contributions subject to vesting or service requirements?

  Medical and commercial insurance: Does coverage include outpatient care, hospitalization, accidents, or other risks, and can it cover family members? Which hospitals and services qualify? Are there waiting periods, deductibles, reimbursement rates, or annual limits? Must I pay first, and how are claims submitted?

  Health checks and other health services: Who pays for the pre-employment medical examination, and is a designated provider required? Are annual health checks, psychological counseling, or other health services available, and what must I pay personally? How are appointments made, and do they require using my own leave?

  Record: Each benefit's name, employee and employer contributions, eligibility, and start date. For insurance, also record coverage and claims procedures; for health services, record the services, frequency, and costs. Where vesting or service requirements apply, confirm the conditions for receiving the benefits and how they are handled when leaving, and keep the relevant explanations.

- [ ] **24. What housing, accommodation, or rent benefits are available?**

  Ask: Are there dormitories, talent housing, or rent allowances? What are the room types, distance, waiting time, and maximum stay? Who pays rent, utilities, internet charges, or deposits? How much is the rent allowance, for how long is it paid, what documents are needed, and is it available during probation?

  Record: Actual housing costs, eligibility, duration, and start date. Check whether accommodation and rent allowances can be used together and whether the allowance is already included in the annual package.

- [ ] **25. What support is provided for commuting and transport after overtime?**

  Ask: What shuttle routes, stops, and schedules are available, and does the last shuttle cover the team's actual finishing time? Are there transport allowances, parking discounts, or reimbursement for late-night taxis? What time, spending, and approval conditions apply to reimbursement?

  Record: Arrangements that work for my commute, personal costs, and reimbursement limits. Compare shuttle schedules with actual working hours.

- [ ] **26. How much paid annual leave is available, and how is it actually scheduled?**

  Ask: How much statutory annual leave and additional company leave are provided, and how is prior service recognized? How is leave prorated in the joining year, and when can I start using it? How is leave requested during projects, what happens to leave that has not been scheduled, and can additional company leave be carried over?

  Record: Applicable leave days, calculation rules, request procedures, and whether the team can actually arrange time off.

- [ ] **27. What are the conditions for sick leave and other types of leave?**

  Ask: How are applicable types of leave, such as sick, marriage, parental, or caregiving leave, requested, and what documents are needed? How are salary, performance pay, and allowances calculated during leave? Does the company provide additional leave benefits, and what rules apply during probation?

  Record: Leave rules relevant to me and how pay is calculated. Distinguish local requirements from additional company benefits.

- [ ] **28. What other benefits and eligibility rules should I confirm?**

  Ask: Are there communications allowances, fitness benefits, holiday benefits, or other benefits? Do they vary by city, job level, contract type, performance, or completion of probation? Which apply only at headquarters or particular office campuses? Are they provided as cash, vouchers, or reimbursement, and must I apply for them?

  Record: The benefits that apply to me, payment frequency, application procedures, and personal costs. Confirm whether the amounts are already included in salary, the annual package, or another allowance.

Comparing these details can help us decide whether the offer meets our expectations and whether to compare it with other offers.

This article will continue to be updated over time.
