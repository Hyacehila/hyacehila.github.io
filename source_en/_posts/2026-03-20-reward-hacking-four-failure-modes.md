---
title: 'Reward Hacking: When Optimizers Reverse-Search the Reward Signal'
title_zh: Reward Hacking：当奖励信号被优化器反向搜索
date: 2026-03-20 20:00:00 +0800
categories:
- Foundation Models
- Training & Alignment
tags:
- Reward Modeling
- Evaluation
- Alignment
author: Hyacehila
mathjax: true
hidden: true
excerpt: When a policy's exploration exceeds the reward signal's discrimination and generalization, sustained optimization
  pressure can turn reward hacking from occasional errors into systemic risk.
description: When a policy's exploration exceeds the reward signal's discrimination and generalization, sustained optimization
  pressure can turn reward hacking from occasional errors into systemic risk.
excerpt_zh: 当 policy 的探索能力超过 reward signal 的判别和泛化能力，并且优化压力持续增加时，reward hacking 就会从偶发错误变成系统性训练风险。
permalink: /blog/2026/03/20/reward-hacking-four-failure-modes/
lang: en
translation_key: 2026-03-20-reward-hacking-four-failure-modes
translation_status: machine
translation_source_hash: 9a6795a82d1ec2cbf758f81babf27e58e2451d2a37b0e06426ce96199ce480fc
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p><a href="/en/blog/2026/03/19/reward-design-evolution-from-rlhf-to-rlvr/">Previous article</a>The answer is how he was produced: from artificial preferences, neuroreward model, PRM, RLVR, LLM as Judge, Rubric, to the open mooning. The main line of the article is:<strong>Reward is not just a fraction, but an interface link that turns targets into training signals.</strong></p>
<p>But it's not enough just to say how it's produced. Rewarding hacking is not a phase of rewarding protection, but a risk that any reward signature may be exposed when it is strongly optimized. As long as reward is fed into optimizer, it will be transformed from an assessment interface to a target that is actively searched, amplified and used. The model is not passive acceptance of awards, but continuous testing of the boundaries of incentive systems in training. So the most important issue of early warning arises:</p>
<p><strong>Is it not enough to continue to represent the original target when a signal is used as a target optimization?</strong></p>
<p>This is the problem that is to be addressed by reward hacking or reward over-optimation. It is not an occasional bug caused by single problems, abnormal samples or individual model defects, but a structural risk that requires default protection when the proxy target is well optimized.<a href="https://arxiv.org/abs/2009.01325">Learning to Summarize from Human Feedback</a> It was observed early that after policy over-hunted the reward model scores, real human preferences would decline.<a href="https://arxiv.org/abs/2501.12948">DeepSeek-R1</a> It was also made clear during the R1-Zero discussion that one of the reasons why no neural reward mode was used was the possibility that mass RL nerve RMs could bring into reward hacking.</p>
<p>That is:</p>
<p><strong>Rewarding is most likely to occur when the policy explores beyond the differential/generalization of reward signature and optimizes the constant increase in pressure.</strong></p>
<p>The optimal pressure here is many things: RL rounds are longer, more sampling, KL is lower, reward weights are higher, filters are more radical, online traffic is continuously feedback, and angent environments are more open. The ability of reward signature depends on whether it can distinguish between real good behavior and just behavior that seems to score high. Once the two sides are unbalanced, the model moves from a learning mission to a learning scorer.</p>
<p>This follows the most common interfaces in the rewardback: rules incentives, neural reward models, evaluation signals from LLM, and a combination of incentives that are common in industry. They are not closed lists of all strategies, but are the risk paths that are most readily visible to the mechanisms. For the verifiable task, the rule reward is usually the most stable; but if the rule covers only the surface format, it will be the fastest to be hacked. Open mission, LLM judge is more flexible than handwritten rules; but if Judge is just a black box score, it's just another proxy. Mixed incentives seem safer; but if the dimensions of conflict are rubbed into a scalar, the risks are hidden deeper.</p>
<h2>Rule incentives: most afraid of open environment and lack of specification</h2>
<p>Rule incentives are usually derived from regular expressions, unit tests, format checks, database finality, API returns, code execution results, or other certainty verifier. Its benefits are obvious: it is cheap, stable, re-emergible and it does not require training a nerve RM that can be used in reverse by the policy. DeepSeek-R1-Zero managed to use digital-based reasoning for strong reasoning behavior precisely because the mathematical answers have relatively clear and verifiable boundaries.</p>
<p>But the risks of rules reward are also clear:<strong>The rules cover only the letter conditions and do not automatically cover the intent behind them.</strong></p>
<p>The model will optimize the <code>reward function</code>Not the real goal in the mind of the product manager, researcher or labeler. If the incentive function is written as " the answer contains the specified format " , the model may fill the format with a full amount; if the reward function is written as " code does not miss " , the model may learn to hide the error; if the reward function runs only one group of unit tests, the model may fit only this set of tests. The hacking mechanism here is not a model that is bad, but the rules are not complete.</p>
<p>There are three high-risk scenarios for which rules reward.</p>
<p>First, mission space is open, but the rules cover only a few screenable surfaces. For example, an angent is going to complete the "Plan a trip" and you check whether the output contains three sets of locations, times, budgets, but not whether the information comes from tools, is consistent with each other, and is not subject to user constraints. The model will soon learn to produce a complete but factually absent answer.</p>
<p>Second, the rewards are too dense, but each is shallow. Format, length, keywords, number of steps, number of calls for tools can be Bonus. The problem is that these bones are too easy to satisfy, and that policy is to give priority to these cheap signals, not necessarily to really difficult mission capabilities.</p>
<p>Third, environmental feedback can be changed by the actions themselves. This is particularly evident in the Agent scene: models are not just an answer, but also a tool, a file change, and an update. If the verifier is not isolated from the environment, reward may move from “evaluation of mission completion” to “evaluation of whether the model has changed the environment to an easily accessible form of inspection”. This is why real events define the definition, the definition, the definition of the procedure, the definition of the procedure, the definition of the criteria and the definition of the procedure.</p>
<p>So the rule reward is:</p>
<ul>
<li>(a) An objectively validated core result, with priority given to verifier;</li>
<li>(a) Mistakes such as security, excess of power, fabrication of facts, environmental damage, as hard veto, not to be put in average points;</li>
<li>The format type incentive can only be used as a support item, must be low in weight and should monitor overuse;</li>
<li>Test set and verifier itself need to be continuously expanded, especially to complement the adversiary case and longtail case;</li>
<li>For angent tasks, complete trajectories must be recorded, as well as complete environmental monitoring.</li>
</ul>
<p>Rules reward rewarding often occurs early. Because it does not require models to understand complex preferences, it simply needs to find the cheapest passage path in the literally written rules. The training curve is shown to be rapidly rising, but when you look at the samples manually, you find behaviors increasingly strange.</p>
<h2>Neuro-reward model: Fixed RRM Risks under Strong Optimization</h2>
<p>Neurological Incentive Models are the core components of classic RLHF: first, train a standard model with human preferences, then let policy pursue high points through PPOs, GRPOs or other RL targets. Its advantages are to cover the open quality dimensions of rules that are difficult to write, such as utility, tone, expression, integrity, courtesy and subjective preferences.</p>
<p>But the nerve RM is essentially proxy. It is not a human preference per se, but a approximation function that is learned from limited comparative data. Once policy starts searching it online, the problem becomes:<strong>Can you find areas outside the RM training distribution but RM is still miscalculated as high quality?</strong></p>
<p>This type of hacking mechanism usually has two layers.</p>
<p>The first level is OOD expluit. RRM training shows a certain distribution of answers, and RL later-stage responses will gradually deviate from this distribution. After deviation, the RRM scale points will be very confident, but this confidence is no longer reliable. Early RLHF work, reward over-optimization, essentially follows the gap between RM and real human preferences after policy has made RM a target function.</p>
<p>The second level is false correlation. RRM can easily learn some superficial features that are relevant to, but not equal to, good answers: longer, more polite, more structured, more like experts, more quote styles, less recognition of uncertainty. The policy magnifies these features in the RL. The final output may seem very professional, but the facts are wrong or do not actually help the user problem. The illusions that such incentives have created have become an issue that needs to be studied separately and are extremely damaging.</p>
<p>That's why industrial systems rarely believe in only one nudity RM score. The post-training route of Llama 3 uses multiple rounds of rewards, reward modes, rejection Samping, SFT and DPO instead of simply putting a RM onto PPO; the DeepSeek series of technical reports also repeatedly distinguishes between the applicable boundaries of rule-based reward and Model-based reward. Open preference tasks usually require multiple stages of post-training exercise, independent evaluation and continuous check-ups.</p>
<p>Neuro, RM, to get a better job, it takes a lot of supporting work:</p>
<ul>
<li>Maintain KL or reference limits to limit the speed at which policy leaves known reliable areas;</li>
<li>(b) To update the RM training set regularly with new PV samples, particularly for high RM points but artificially unsatisfactory samples;</li>
<li>Optimization is detected with holdout prompts, manual spot check, factual evaluation and analytical eval;</li>
<li>Monitor reward distribution, length, no-response rate, reference density, repetition mode and KL, rather than just average;</li>
<li>(b) Cross-checking with endemble, uncertification or multiple RMs to reduce the probability that a single RM blind area will be accurately used;</li>
<li>Access to the verifier, as far as possible, for factual, security and consistency of the results of the tool, rather than all over the RRM guess.</li>
</ul>
<p>The core risks of nerve RM can be summarized in one sentence:<strong>Fixed RM is searched for more policy for long time as the target.</strong></p>
<h2>LLM-as-Judge: Most afraid of broad decisions and uncorrected bias</h2>
<p>LLM-as-Judge or RLAF are naturally motivated: many tasks are not simple verifier, and humans are too expensive to be judged by stronger LLM. LLM judge has several advantages over fixed neurons RM: it can read rubric, it can explain judgment, it can handle open tasks, and it can use pairwise comparison instead of hard output absolute points.</p>
<p>But LLM Judge is also a partial model, but the deviations are more like "evaluation style." MT-Bench / Chatbot Arena has repeatedly warned that LLM judge may have status bias, verbosity bias and self-environment bias. That is, it may prefer a candidate for a position, a longer answer, or an answer more like its own output distribution.</p>
<p>Once Judge is brought into training, policy will learn about these prejudices.</p>
<p>If Judge likes long answers, policy pours water; if Judge likes barter templates, policy writes all the answers into templates; if Judge is weak in fact verification but strong in speech judgement, policy becomes more packaging illusions; if Judge prompt is too broad, policy learns to accommodate the value bias implicit in prompt, not necessarily closer to the real mission objective.</p>
<p>The LLM Judge is a high-risk scene:</p>
<ul>
<li>Judge protocol only "please give this answer a score", and there is no clear rubric;</li>
<li>Use pointwise fractions, but no cross-prompt calibration;</li>
<li>Pairwise is more likely to not be a serial exchange A/B;</li>
<li>(a) Judge does not have an independent assessment and does not match human signs or verifiable signals;</li>
<li>Trained policy and judge from the close model family, self-preference is more obvious;</li>
<li>The fact-based task still leaves the judge with a sense of language rather than access to search, tools or verifier.</li>
</ul>
<p>For LLM judge, the most important direction for improvement is structuralization.</p>
<p>First, try to make the judge more candidate than a direct invention. In open missions, the pairwise/listwise is usually more stable than the pointwise, because it is easier to compare the two answers than to define what the sevens mean.</p>
<p>Second, use rubric to disassemble mass into specifiable dimensions. Examples include factual confirmation, binding satisfaction, consistency of reasoning, use of instrumental evidence, expression of clarity, cost and security boundaries. The task of Judge should be to extract signals along these dimensions, rather than facing a vague overall objective.</p>
<p>Third, do the differential correction. Pairwise needs to rate in two directions, long answers need to be uniform or redundant, factual dimensions need to be based on external evidence, and Judge himself is to benchmark. The value of this type of eval scenario is here: rather than let Judge guess what the whole is like, he starts by tearing open missions into a lot of local judgment.</p>
<p>Fourth, do not place LLM judge in hard position. Mistakes such as over-the-counter instruments, the fabrication of key facts, security violations and environmental damage should be rejected by the rules and directly by the verifier. Judge can help explain and sort, but the red line should not be covered with a high representation.</p>
<p>LLM Judge's rewarding is more like assimilating: models may not be stronger, but they are becoming more and more like judges willing to give high marks.</p>
<h2>Mixed incentive: most afraid of static weights hiding conflict in total score</h2>
<p>Industrial systems rarely use a single reward. More often:</p>
<p>&#36;&#36;
R = \alpha R_{\text{rule&#125;&#125; + \beta R_{\text{RM&#125;&#125; + \gamma R_{\text{judge&#125;&#125; - \lambda C
&#36;&#36;</p>
<p>It seems to be stable: rules are hard borders, RM is preference, judge is open quality, cost is efficient. The problem is, as long as these signals are wiped into a scalar, polity will do the optimizer's best:<strong>Search for arbitrage between sub-items.</strong></p>
<p>The mix of awards is not necessarily exaggerated. It is often more subtle, as the overall score is still rising and individual indicators may not be significantly lost, but real availability is beginning to decline.</p>
<p>Typical is the average dilution of the red line. One response actually created key prices, but the structure, tone, integrity, service closure and final score were high. This is clearly unacceptable to users; it may be a local best for linear reward.</p>
<p>Another example is the static weight imbalance between costs and quality. If the cost penalty is too weak, angent learns to use more tools and write over redundancy analysis to raise the judice score; if the cost penalty is too strong, it may stop exploring too early to give answers that seem simple but fail to complete the task.</p>
<p>The high-risk configurations of the blended incentives are:</p>
<ul>
<li>All sub-items are linearly weighted, without hard veto;</li>
<li>(a) The weighting will remain static and will not vary according to the type of task, difficulty and risk level;</li>
<li>dashboard, read only the total score or a small average;</li>
<li>There is a lack of conflict samples in the training pack, such as “A good but wrong expression”, “A perfect format but an overstepping of the tools”, “A short answer but missing key constraints”;</li>
<li>Reward is both a data filter and RL return, but the two phases are not audited separately.</li>
</ul>
<p>The more secure way of combining incentives is not to shift weights to more fundamentals, but to change the logic of aggregation.</p>
<p>First, the red line does not score average. Factual fabrication, excess of authority, environmental damage, security violations, and violation of the user's hard-line restrictions should be veto or lexicophatic control. Legitimacy first, then quality.</p>
<p>Second, task type determines mandate type. Planning missions, information retrieval missions, advisory interpretation missions should not be shared with the same fixed set of weights. The “good” of different mandates is not the same geometry.</p>
<p>Thirdly, the sub-indicators must be seen separately. The total reward, RM, judge, pass rate, veto rate, length, number of calls for tools, cost, KL, manual satisfaction are monitored separately. The biggest problem with mixing is hiding the conflict, so surveillance must re-engage.</p>
<p>Fourth, the training sample is built proactively. Without such samples, the system does not know what dimensions cannot be offset by each other. The real useful industrial set is often not a common sample but a sample designed to expose the weight gap.</p>
<p>The blending incentive is not a cure for rewarding. It simply disintegrates a single proxy risk; if the aggregation method is still rough, the risk will return in a more difficult way to trace.</p>
<h2>What should be the training strategy?</h2>
<p>The combined risk of mismatches on these interfaces allows for training strategies to be chosen by mission structure rather than by abbreviations of the papers.</p>
<p><strong>Math, Code, SQL, Tool Finality</strong>: preferred verifier / rule reward + GRPO / RLVR. Core objectives are objectively verifiable and no first-hand learning is necessary; additional rules of defence are not sufficiently covered, test collections are over-compatible and the format is overrewarded.</p>
<p><strong>Subjective preferences, tone, brand style, open dialogue</strong>: preferred SFT + reference data + DPO / RM-based signature. Targets are difficult to write in verifier and require preference for learning; additional protection against RM optimization, length bias and OOD high score samples.</p>
<p><strong>Long chain tracking or angent track</strong>: preferred outcome verifeier + policy checks + projectorry ranking. Just looking at the endpoint is too thin, and intermediate decision-making is monitored; extra protection against the judge can't see the trajectory and process overform.</p>
<p><strong>Open Complex</strong>: preferred entry + feedbackback + hybrid reward + online RL. Reward comes from environmental, rules, jurisprudence, multiple interfaces at cost; extra static weight arbitrage, contaminated state of the environment and benchmark proxy.</p>
<p><strong>High-risk fact-finding missions</strong>: preferred vvefier / retrieve evidence, judge only to assist. Factuality cannot be covered by rhetoric; additional protection against miscarriages and hallucinations is masked by the quality of expression.</p>
<h2>Finally: rewarding hacking is not abnormal, but the result of the stress test.</h2>
<p>Many people talk about rewarding, talking like they're trying to be smart. But more precisely:<strong>Reward hacking will expose the specification loophole.</strong> It is not that models and optimizers are not good enough per se, but rather that the real target is not presented in its entirety.</p>
<p>Following the previous interfaces, the rule incentive exposed the specification gap; nerve RM exposed the problem Gap; LLM judge exposed the error; and mixed incentive exposed the target conflict and the polymer defects.</p>
<p>So the more mature, mature, should not ask whether this is the only way to reflect demand, but also to continue to ask four more specific questions:</p>
<ul>
<li>Where is this blind zone of reward?</li>
<li>How strong is that?</li>
<li>What mistakes cannot be offset by other high marks?</li>
<li>When the reward rises and the real quality falls, can I see it on the dashboard?</li>
</ul>
<p>Answering these four questions clearly, it was from a training score to an interface that could be audited, iterative and placed in the industrial system.</p>
<p>And here, Reward, three of these pieces, they're a line: first, how the signal is consumed, then how it's produced, and finally, how it's not matched by a strong optimization. Follow-up and reentry. <a href="/en/blog/2026/03/22/reward-and-training-in-agent-k-paperbench-amap/">training loop</a>: How do you connect SFT, offline shaping, online RL, curriculum and distillation after they have been broken down into verifier, judge, rubric, projectory ranking and hard reporting?</p>
<h2>References</h2>
<ul>
<li><a href="https://arxiv.org/abs/2009.01325">Learning to Summarize from Human Feedback</a></li>
<li><a href="https://arxiv.org/abs/2203.02155">Training language models to follow instructions with human feedback</a></li>
<li><a href="https://arxiv.org/abs/2501.12948">DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning</a></li>
<li><a href="https://arxiv.org/abs/2407.21783">The Llama 3 Herd of Models</a></li>
<li><a href="https://arxiv.org/abs/2505.09388">Qwen3 Technical Report</a></li>
<li><a href="https://arxiv.org/abs/2412.19437">DeepSeek-V3 Technical Report</a></li>
<li><a href="https://arxiv.org/abs/2504.01848">PaperBench: Evaluating AI&#39;s Ability to Replicate AI Research</a></li>
<li><a href="https://arxiv.org/abs/2306.05685">Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena</a></li>
<li><a href="https://arxiv.org/abs/2305.17926">Large Language Models are not Fair Evaluators</a></li>
</ul>
