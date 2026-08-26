---
title: 'Werner Vogels'' Last Lesson: The Renaissance Developer in the AI Era'
title_zh: Werner Vogels 的最后一课：AI 时代的文艺复兴开发者
date: 2026-07-01 15:30:00 +0800
categories:
- Work & Society
- Builder & Product Thinking
tags:
- Software Engineering
author: Hyacehila
mathjax: false
excerpt: 'Werner Vogels'' final re:Invent keynote framed the AI-era developer as a Renaissance Developer: curious, systemic,
  communicative, responsible, and polymathic.'
description: 'Werner Vogels'' final re:Invent keynote framed the AI-era developer as a Renaissance Developer: curious, systemic,
  communicative, responsible, and polymathic.'
excerpt_zh: Werner Vogels 在 re:Invent 2025 的最后一场 keynote 里谈到文艺复兴开发者：保持好奇心、系统性思维、有效沟通、主人翁精神与博学。本文讨论 AI 时代开发者如何继续保留判断力和责任感。
permalink: /blog/2026/07/01/renaissance-developer-werner-vogels-reinvent-2025/
lang: en
translation_key: 2026-07-01-renaissance-developer-werner-vogels-reinvent-2025
translation_status: machine
translation_source_hash: f9050e98049256e74c333566fcf3fdc3b9bb3468e82e5962b0a465375c25c03a
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>Werner out.</p>
<p>This is not a big goodbye, more like Werner putting the microphone down and saying, "This is all I'm saying."</p>
<p>In 2025, AWS re:Invent, Amazon.com Vice President and Chief Technical Officer Dr. Werner Vogels did his last game on re:Invent stage. The AWS official recap summarized the speech as a sharing of "releasessight developer".</p>
<p>Werner did not follow AI hotspots to describe the hottest concepts of the time. He's been talking about a lot of things, and it's the old software project, and it's just going to be in the AI Coding scene.</p>
<p>Is AI gonna take away the developers' work, is Vibe Coding gonna turn the writing software into a natural language magic, is Spec-driven development not a new project paradigm, and can be discussed. But what needs to be understood, how risks are judged, how the consequences are to be held accountable, and the AI era just pushed these issues closer.</p>
<p>This talk is worth a note and save my current memory of fish.</p>
<h2>Why Renaissance Developer?</h2>
<p>Werner Vogels is a CTO of Amazon and a very visible person in AWS technology culture. Before joining Amazon, he did a distributed system study in Cornell; after joining Amazon, he studied on cloud computing, distribution systems, machines.</p>
<p>If werner is only from the product release, he is the leading technology spokesperson for the AWS history, and he issues many important products and ideas. He can also be seen as a representative of the quality of engineering: less obsessed with the eccentricity of technology, more often asking how the system will fail, how it will be simple, and what the customer will actually encounter.</p>
<p>The art rehab developer's statement has also become interesting here.</p>
<p>The Renaissance was not only the result of a sudden appearance of a tool, but also of printing, microscopes, visualization and new sea technologies. In that history, a group of people began to cross the disciplinary boundaries, bringing together science, art, engineering, commerce and society.</p>
<p>Today's developers are in a similar position. AI, cloud computing, smartness, bioengineering, space technology, energy systems are being linked to each other. Software engineering is no longer a screen application, and it increasingly moves into health, logistics, education, energy and public governance, thus posing new risks.</p>
<h2>Five qualities of Renaissance Developer</h2>
<h3>I. Stay curious: Don't jump over understanding.</h3>
<p>Werner puts curiosity in a very forward position. I like this order. Of course, good engineers are familiar with tools and frameworks, but what really opens the gap is often the willingness to stop shooting more often when it comes to phenomena: why is it doing so?</p>
<p>I'm here.<a href="/en/blog/2026/06/25/from-designer-to-agent-builder/">"From Designer to Age Builder: Look, do, want."</a>It says something like that. Seeing and doing it will give you a feeling of hand, and you'll sink it. Otherwise, it would be easy to look at more cases and demo more than anything else, but simply to move others' forms, and to see how to judge when new problems arise.</p>
<p>AI Coding makes this issue more obvious. You have to read documents, check source codes, log them, and narrow the recurrence range. Slow, annoying, but the system structure is a little bit brain-drained in the process. Now you can throw the blunders to the model, and make it quick to give a good look. Problem solved, but you don't really learn anything.</p>
<p>I certainly used AI, and I used it pretty much. It's just that sometimes when I'm done, I think, do I really understand something this time, or just get past the troublesome part more quickly.</p>
<p>AI can easily create a illusion, as if the answer were there, and understandings happen. Actually, no. Understanding often occurs in the course of ICP, questioning outputs, asking borders and re-programming itself.</p>
<p>Werner uses language to compare the process: mastering a language is not simply a reading of grammar books, but rather making mistakes in real conversation, being corrected, and repeating it. Software works are similar. A construction failure, a false assumption, a boundary exposed before going online is usually easier to remember than a well-generated code. Mistake is itself a process of learning and growth.</p>
<p>And keep wondering why the model is written, what it omits, whether it treats an old API as a new writing. Understands it's not by-product, you have to go get it yourself.</p>
<h3>II. Systemic thinking: looking at the scope of impact</h3>
<p>Werner explains the system thinking with the wolf tale of Yellowstone: fewer wolves, more moose, and more eco-efficient; and then, in the back, vegetation is over-eating, river banks are unprotected and river patterns change. A local change, which runs very far along the feedback loop.</p>
<p>It's too common in software.</p>
<p>A revision of a retry strategy may allow downstream services to be blown up in case of failure; a change in a cache TTL may change the system ' s consistency; and a change in a team boundary may change the problem from code to communication. Even a downgrading strategy that appears to be cost-saving could shift trouble to users in an accident.</p>
<p>The systemic thinking sounds big, falling into the day-to-day business, perhaps asking more questions: What feedback will this change have? Has the pressure been moved elsewhere? Did Gin touch it?</p>
<p>For AI Agent, systemic thinking is more important, and any Agent is a system. Individual model capabilities will certainly depend on the reliability of the system, but often depend on the availability of context, tool privileges, state management, validation mechanisms, failure recovery and human scrutiny. After the model is finished, it has to continue to see how the system deteriorates when it makes a mistake.</p>
<p>Werner actually goes on to say, "Design for Fairure" as it was long ago: Don't expect the components to be right forever, just assume they're bad, and then leave room for the system.</p>
<p>AI makes local production faster and local decision-making more numerous. Maintaining systemic thinking and control over the system should not deprive humanity of its decision-making and control over the system.</p>
<h3>III. Effective communication: specificizing needs</h3>
<p>In the AI programming era, communication capacity was amplified. Not only between people, but also between people and AI, and between people and people that AI brings.</p>
<p>Previously, developers used mainly programming language and machine communication. Programming language is strict and murky, but less ambiguous. Now, we are increasingly describing intent in natural languages, and then making models translate it into code. The threshold has fallen and the ambiguity has come together.</p>
<p>A single message for me can be used to make a system of notification that is responsive to many of the achievements: on-site mail, mail, text messages, Webbook, push, subscription preferences, de-graving, Zen, etc., re-test queues, frequency control, audit logs, data retention, permission boundaries. Short prompt will allow the model to decide a bunch of implied constraints for you first. You fix it later, you're probably fighting with its default choice.</p>
<p>And this is where Werner agreed when he talked about Spec-driving development and Kiro. Kiro's specs documents describe spec as a structured work that transforms high-level ideas into a traceable, reviewable, implementable development process using documents such as requirements, designs, tasks. It's kind of like scaffolding a natural language.</p>
<p>I don't think Spec will be the new paradigm for all software development. I'm here on that.<a href="/en/blog/2026/04/07/spec-is-not-the-new-paradigm/">Spec is not a new paradigm: Video Coding, SDD and AI-era software engineering shift</a>It's written in there. Rapid prototypes and feedback are still needed during the exploration period, and many constraints will only appear when the system runs. However, in high-risk, multi-person collaboration and long-term maintenance scenarios, the intent, boundaries, acceptance criteria and design trade-offs are clearly articulated and later returned to the unions.</p>
<p>The more specific the requirements, boundaries, acceptance criteria are, the smaller the scope for model speculation, because the less return work that communication produces, the more it is.</p>
<h3>IV. Ownership: Responsibility remains with developers</h3>
<p>My favorite sentence in this speech is, “Work is your work, not a tool”.</p>
<p>AI can generate codes, test, draft designs, or even explain a system behavior for you. But once things enter the production environment, responsibility does not fall on the model. The client would not be less vulnerable because it was written by AI, and the CRT would not end because the tool suggested it. The last person to come forward to explain, the person to build the system.</p>
<p>And this is the new version of the "You built it, you run it" in the AI era. In the past, it had emphasized that the engineering team could not simply throw the code at the service, but was responsible for the results of the operation. Now add the following sentence: The intent cannot be simply thrown at AI, then the output of AI can be thrown at the user. You need to understand, review, verify, and if necessary reverse it.</p>
<p>Werner mentioned in his speech the Amazonian Andon Cord practice, which gives the closest people the mechanism to expose the problem, to stop the system and to push for amendment. Here, I am concerned about the mechanism itself: the person who sees the problem can act, the system allows him to act and the organization respects the operation. It is not enough to say that everyone has responsibilities.</p>
<p>This is obvious in AI Coding. AI will generate codes faster than they understand, and it will also make more things that look like running. Code review, testing, observation, rollback mechanisms, segregation of access and data protection cannot be saved by using AI. Many times, the code that AI generates needs more of these things.</p>
<h3>V. BEING A BOOK OF LEARNING: COMBAT NEGOTIATION</h3>
<p>The last key word for Renaissance Developers is learning.</p>
<p>It's not about asking everyone to become da Vinci. To be realistic, it is more like a T-type capability: it is built deep enough in one field, while maintaining a basic understanding of the adjacent areas.</p>
<p>Software engineering is becoming less and less suitable to stare at only the small piece of itself. Database engineers will have different choices when designing indices if they understand the performance of the front end and the time that the user waits; back end engineers if they understand the distribution of products and client support, they will not only pursue structural beauty; and Agent developers if they understand the safety, interaction, assessment and organizational processes, they will not be able to directly equate models with systems.</p>
<p>I understand the learning, not much like gathering a lot of knowledge points. It is more like knowing what the adjacent field is concerned about and putting several perspectives together when it comes to problems. Difficult problems often arise at borders: between technology and operations, between models and tools, etc.</p>
<p>AI will repricing some single-point skills. Model codes for a framework may be written and may become less and less scarce. It is possible to break the ambiguity into verifiable pathways, understand the interaction between several systems, or take time to practice.</p>
<p>First, there's a professional chassis that can stand, then slowly expand the senses. And knowing more of the Allied constraints, many of the design decisions are less confident.</p>
<h2>The true look of Builders.</h2>
<p>Werner in this Keynote also talks about the Bulder in the real world.</p>
<p>These cases cut across agriculture, environmental protection, health care and energy: in the Amazon basin, companies support local communities through sustainable supply chains, leaving young people without having to leave their homes to earn their income; Ocean Cleanup uses drones, AI image analysis and GPS to track the path of plastic waste in rivers; the Rwandan health data system helps the Government to make public decisions based on indicators such as disease outbreaks and maternal and child health; and Koko Networks in Nairobi supports urban-level clean fuel distribution networks with cloud computing, allowing households to access cleaner energy at lower cost.</p>
<p>These stories are not much like the world's most common technology changers in the press conference. They are more specific: there are people who run a short distance, people who do not have to leave their homes, and people who can use cleaner and cheaper energy.</p>
<p>And that's where I think Werner is not quite like many technical preachers. He certainly cares about the principles of distributed systems, cloud computing, AI and architecture, but he talks about how often people end up: who uses them, who pays for them, who gets better because of the system.</p>
<p>I don't really want to explain the Builder spirit to be a faster ship. Werner is telling these examples that much of the work is done where the user is not seeing. The database is not broken, and users will not send thanks; the clean energy network is functioning steadily, and many people will only find cooking normal today; health data systems will identify risks in advance, and the best result may be that no disaster has occurred.</p>
<p>Such work is not easily written into the publication.</p>
<p>Jobs, evaluating John Scali, said he had an "emergence disease": 90 per cent of the job was done thinking that there was a wonderful idea. From great ideas to great products, there is a huge, dry work in between. The Builder job is also in this gap. AI changed the cost of some of the work, not to level the gap. Today's Builder doesn't have to get through all the trifles. AI assistants can already complete the template code, write first draft tests, flip the log, sort the documents, and run the batch. The time saved can be used to deal with things that are more difficult to automate: how does a user use it, where a seemingly partial change moves costs, and who can fix them when they go online.</p>
<p>And the focus of 90% of the work is not to screw every screw. It is in the details that one slowly finds out what was wrong with the original idea, what was worth it and what was unacceptable. AI can do a lot of work, and the judgment is still here. And this brings back the five features that are in front of us to the same scene: Curiosity is not enough for a running version; the system is thinking about keeping people informed about where the costs end; and communication ability determines whether AI has received enough clarity about its intentions. When mistakes occur, ownership does not take people back; it is also useful to understand what the product, the operation and the users are concerned about.</p>
<h2>Now Go Build</h2>
<p>Werner said "now Go Build" for years, and in 2025, this last Keynote, it sounded a little more like closing.</p>
<p>The United Nations World Population Production 2024 shows that the global population, which was estimated at 8.2 billion in 2024, is expected to continue to grow in the late part of the century and to peak at about 10.3 billion in the mid-2080s. Population projections will change as data update, but health, energy, education, infrastructure, climate adaptation and economic opportunities will face continuing pressures in the coming decades.</p>
<p>These pressures end up with specific technical issues: more reliable medical data systems, lower-cost distribution of clean energy, more accessible advanced computing capabilities, and more than AI applications that serve the economy.</p>
<p>AI will continue to grow stronger and develop tools will continue to change. The old principles that Werner left behind will probably continue to work.</p>
<p>Werner out.</p>
<h2>References</h2>
<ul>
<li>AWS Events, <a href="https://www.youtube.com/watch?v=3Y1G9najGiI">AWS re:Invent 2025 - Keynote with Dr. Werner Vogels</a></li>
<li>AWS News Blog, <a href="https://aws.amazon.com/blogs/aws/aws-weekly-roundup-aws-reinvent-keynote-recap-on-demand-videos-and-more-december-8-2025/">AWS Weekly Roundup: AWS re:Invent keynote recap, on-demand videos, and more</a></li>
<li>AWS, <a href="https://aws.amazon.com/events/reinvent/on-demand/">re:Invent 2025 on demand</a></li>
<li>Amazon Science, <a href="https://www.amazon.science/author/werner-vogels">Werner Vogels</a></li>
<li>Kiro Docs, <a href="https://kiro.dev/docs/specs/">Specs</a></li>
<li>United Nations Population Division, <a href="https://population.un.org/wpp/">World Population Prospects 2024</a></li>
<li>United Nations Population Division, <a href="https://population.un.org/dataportal/home">Data Portal</a></li>
<li>Classmethod, <a href="https://dev.classmethod.jp/articles/reinvent-2025-keynote-with-dr-werner-vogels-key005/">A Specializing Keynote with Dr. Werner Vogels</a></li>
</ul>
