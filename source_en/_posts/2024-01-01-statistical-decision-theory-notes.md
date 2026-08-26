---
title: 'Statistical Decision Theory: Risk, Uncertainty, and Multi-Objective Decisions'
title_zh: 统计决策：决策问题、风险型决策与不确定型决策
date: 2024-01-01 15:06:22 +0800
categories:
- Data Science
- Statistical Modeling & Inference
tags:
- Statistics
- Statistical Decision
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers decision problems, risky decisions, uncertain decisions, multi-objective decisions, and analytic hierarchy
  process.
description: Covers decision problems, risky decisions, uncertain decisions, multi-objective decisions, and analytic hierarchy
  process.
excerpt_zh: 整理决策问题、风险型决策、不确定型决策、多目标决策和层次分析法。
permalink: /blog/2024/01/01/statistical-decision-theory-notes/
lang: en
translation_key: 2024-01-01-statistical-decision-theory-notes
translation_status: machine
translation_source_hash: 1aa734e332a2d3e4e05075d9348621975d0cc78fa4e3d3c2e8fd3c1c42fa4f86
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>The questions in this article can also be addressed<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base: linear model, minimum 2x2 estimate and regression diagnosis</a>、<a href="/en/blog/2023/09/12/statistical-computing-notes/">Statistical calculations: random number generation, random variable simulation and Monte Carlo method</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>Decision-making is an important part of management activity, and in order to avoid decision-making errors, to ensure the science and effectiveness of decision-making, we need to look at the patterns of decision-making practice and develop a complete theoretical system.
With the advent of the big data age, data becomes more and more important in decision-making, leading to the theory of statistical decision-making.</p>
<h2>Summary of doctrine</h2>
<h3>Basic issues for statistical policymaking</h3>
<h4>Concept</h4>
<p>Decision-making: The process of making decisions about future action based on information and experience in order to achieve a certain objective
(a) Statistical decision-making: a particular form of decision-making;
Elements of statistical decision-making: decision-making subjects; decision-making objectives; decision-making target (what decision makers can influence); state of nature (external environment); options (it is important to compare good and bad options)</p>
<h4>Classification</h4>
<h5>Depending on the nature of decision-making,</h5>
<p>Deciding decision-making: a good understanding of the future situation to determine the most satisfactory options
Undetermined decision-making:
Competition type: Uncontrollable competitors in nature (presented in game theory)
Risk type: Probability of natural states can be predicted, uncertain but informative.
Uncertainty decision-making: the probability of a state cannot be predicted.</p>
<h5>By decision-making body</h5>
<p>Individual and group decision-making
As individual abilities are limited, we should use group decision-making as much as possible.</p>
<h5>By nature of decision-making</h5>
<p>Procedural decision-making: there are often precedents to follow and there are generally objective and correct answers
De-processed decision-making: lack of established decision-making tools</p>
<h5>Whether to use mathematical models</h5>
<p>Qualitative and quantitative decisions</p>
<h5>Target Number</h5>
<p>Single- and multi-purpose decision-making</p>
<h5>Competing.</h5>
<p>Independent and interactive decision-making
Interactive decision-making is also called game-making.</p>
<h5>Continuity</h5>
<p>Need for a series of decisions</p>
<h4>Wrong decision.</h4>
<p>With decisions, it's hard to avoid mistakes.
The need to minimize errors: to ensure a complete process of decision-making; to focus on feedback and corrections; to use experiments to validate decision-making.</p>
<h3>Development of policy-making theory</h3>
<p>Decision-making is an emerging theory that combines a wide range of research, such as preparation, mathematical modelling, statistics, computer science, and the theoretical system for managing decision-making.</p>
<h4>Theory system</h4>
<p>Classical decision-making: a very ideal policy-making theory
The decision-makers are required to be fully rational, and the natural situation is perfectly clear.
With the development of modern quantitative decision-making techniques, we know more about reality, so classical decision-making science is now gradually being applied widely.
And then decision-making evolved into decision-making organizationals and policy-making behaviors; the scholars in this area are more economists and management scientists, and there are other disciplines involved, not the focus of our statistical policy-making component.</p>
<h4>Development phase</h4>
<h5>Theory of spiritual expectations</h5>
<p>Presented an important concept in decision-making
I'd like to quantify the impact of the various decisions, which have a profound impact on decision-making.</p>
<h5>Expected utility theory</h5>
<p>Take on the utility proposed earlier
Logical and mathematical analysis of rational people ' s decision-making under the assumptions of justice
This is where microeconomics began, followed by macroeconomics, finance, econometrics.</p>
<h5>Bayesian decision-making theory</h5>
<p>Subjective probabilities are advanced for decision-making
Using the theory of decision-making to re-analyze the methodology of statistical analysis, the Bayesian theory of decision-making was created.
He's a huge achievement in Bayesian statistics.</p>
<h5>Policy analysis theory</h5>
<p>It expands the results of the previous Bayesian statistical decision-making and proposes an entirely new field of research on policy analysis.
Multi-purpose decision-making, multi-purpose decision-making, multi-target decision-making, multi-targeted decision-making are systematically studied by people after the policy analysis theory.</p>
<h5>Policy-making behaviour theory</h5>
<p>Some assumptions about existing policy-making theory have been questioned
The limited rational hypothesis is presented in the theory of decision-making behaviour and the theory of policy-making analysis parallels another theory.</p>
<h5>Decision support system</h5>
<p>It's not part of theoretical research.
I want to help people or replace humans in decision-making.
Of course, some theoretical research has been produced, but more about the use and reorganization of the results.</p>
<h4>Trends</h4>
<p>Moving from individual to group decision-making
Moving from single to multiple targets
Moving from qualitative to quantitative
Combined use of computer technology
<strong>Every trend meets the requirements.</strong></p>
<h2>Decision-making</h2>
<p>Decisive decision-making is the most basic theory of decision-making
We can say he's the domain of statistical decision-making because we really need to use information to make decisions.
We can also exclude it from statistical decision-making because all the information he uses is a priori.
There's no need to make any more distinctions on this point, and statistical decision-making is the sum of what we call this theoretical system.
<strong>Decision-making and operational preparation are very closely linked, and the methods are largely derived from operational preparation.</strong></p>
<h3>Basic issues for defined decision-making</h3>
<h4>Concept</h4>
<p>Called structured or standard decision-making
The characteristic is that the selection of the most satisfactory course of action can be made on the basis of fully defined circumstances and the outcome is entirely determined by the choice of decision makers
It's the most basic method of decision-making.
A lot of things in the whole mathematical modelling system are defined decisions.
He's a very difficult man, depending on the complexity of the model itself.</p>
<h4>Characteristics</h4>
<p>Easy to quantify
General decision-making targets are expressed in target functions.
It is possible to clearly calculate the programs and their target functions</p>
<h4>Classification</h4>
<p>Selecting decision-making only:
A limited number of programmes of action The data available need not be processed The best results can be obtained from the ICP (no need for research)
Model selection decision-making:
Build a mathematical model and choose the best option by analysing the model
The scope of definitive decision-making is very large.</p>
<h4>Steps</h4>
<p>Collecting existing information, building mathematical models, studying their optimal solution.</p>
<h3>Balance decision-making</h3>
<h4>Rationale</h4>
<p>The relationship between production, cost and profit in research projects
Determining the profit zone and the loss zone, the security margin, is a very basic, definitive decision.
Generally divided into linear and non-linear models</p>
<h4>Linear Balance</h4>
<p>Under a series of assumptions, we set&#36;TC&#36;Total cost &#36;Q&#36;It's production. &#36;TR&#36; It's total income. &#36;p&#36;It's the price. &#36;F&#36;It's a fixed cost. &#36;b&#36;It's the unit cost, and there is.
&#36;&#36;TR=pQ&#36;&#36;
&#36;&#36;TC=F+bQ&#36;&#36;
By lifting Max&#36;TR&#36; Planning is the best.&#36;Q&#36;
This is the most basic linear model of profit and loss.</p>
<h4>Non-linear balance of gains and losses</h4>
<p>This is our time.&#36;TR~TC&#36;Both.&#36;Q&#36;Non-linear function
We can still get the best of the best.&#36;TR&#36;
Even if this form becomes very complicated, we can try the worst. Break</p>
<h3>Linear planning decisions</h3>
<p>Under certain constraints.
Maximum (minimum) value for target function
When the binding and target function is linear, it's a matter of linear planning.
You can fully solve it using a simple and optimal method.
It's a very basic question of optimization.
<strong>As you can see, the whole definitive decision-making is basically a mathematical model based on the mechanisms of the subjects that are known to be studied, followed by some methods of resolution.</strong></p>
<h2>Risk-based decision-making rationale</h2>
<p>The natural state of risk-based decision-making is not the only one, and we know the probability of each of them; that's the biggest difference between him and the definitive decision-making.</p>
<h3>Basic issues</h3>
<h4>Concept</h4>
<p>Whatever decision-making options are taken, the natural state is not the only one that carries a certain risk and does not necessarily maximize the benefits; but we can judge the probability of each natural state based on past experience, and we can find the most effective solution through the ICP gain and loss.
We can think of risk-type decision-making as a classic use of a priori probabilities, but it's different from Bayesian decision-making due to lack of sample revision.</p>
<h4>Gain and Loss Matrix</h4>
<p>This is an important component of risk decision-making.</p>
<ul>
<li>Viable options</li>
<li>Natural state and probability of occurrence</li>
<li>Gains and losses (effects of a combination of different natural states and viable options)
We often use profit-and-loss matrices to describe it.
The normal vertical axis is a viable solution, the transect is a natural state and probability, and each position on the matrix is a combined gain and loss. Value</li>
</ul>
<h3>Guidelines for risk-based policymaking</h3>
<p>There's a profit and loss matrix, but we're still not sure that it's the best, and it's not the best.
The decision-making guidelines are intended to provide us with a viable formula for our final decision-making from the profit-and-loss matrix.</p>
<h4>Expectations guidelines</h4>
<p>Expectations to calculate gains and losses for each programme
We think he's the best.
&#36;&#36;E\left(d_{i}\right)=\sum L_{ij}p_{j}&#36;&#36;
Because the profit and loss matrix sometimes means the gain, and sometimes the loss.
Expectations generally apply to the apparent objective nature of probabilities and their stability.</p>
<h4>Waiting for probability criteria</h4>
<p>When the probability of a natural state is unpredictable, we choose to assume the probability of each natural state.
Then we'll use the expectation criterion to judge.
The probability of being applied to a natural state is not good and the probability is closer.</p>
<h4>Guidelines on maximum likelihood</h4>
<p>We're here to consider the greatest natural state of probability in a random experiment.
Select the best option based on the profit and loss of the maximum probability natural state.
This translates risk-based decision-making into defined decision-making
Applicable to situations where the probability of a natural state is significantly higher</p>
<h3>Decision Tree</h3>
<h4>Concept, drawing, application</h4>
<p>In essence, it's a way to make decisions, but it's more intuitive, and it's appropriate for discussion.</p>
<h4>Concept</h4>
<p>It's a picture of the decision-making process.
The decision tree algorithm is a typical way of classifying options and natural state and profit and loss into tree-shaped structures with all known probability.</p>
<h5>Draw</h5>
<p>Draw decision tree from left to right from decision point
The decision node can lead to the state node.
There are several decision nodes in the decision tree for multi-level decision-making.
The state nodes can lead to a probabilities branch, a probabilistic node, and the branch should spell out the probabilities. Value
Result node is the gain or loss under a programme or natural state Value</p>
<h5>Cut.</h5>
<p>Draws the decision tree from left to right, but applies it from right to left. Cut.
By calculating the expectations for each state node, select the optimal state node, then cut all the nodes except the best node, and move to the first decision section. Points
The final decision tree will leave a branch called the best option.
We have two classic cutting methods.</p>
<ul>
<li>If this node continues to divide in a manner that does not increase accuracy, treat it as a result node.</li>
<li>And if cutting leads to better accuracy, cut this branch.</li>
</ul>
<h4>Phase two decision tree</h4>
<p>There's no difference between a one-stage decision tree and a direct use of the profit-and-loss matrix.
The idea behind the second phase of the decision tree is to put the second decision behind the outcome point of the first decision.
Introduction of new outcome nodes again
The multilayered decision tree works in the same way as before.
Decision-making against expectations of multiple outcome nodes per layer</p>
<h3>Sensitivity analysis for risk-based decision-making</h3>
<p>Risk-type decision-making requires the use of a priori probabilities, but such a priori probabilities are uncertain, and we have to study the risks of such uncertainty</p>
<h4>Brief description of sensitivity analysis</h4>
<p>Our focus is on the probability of turning points.
And that's when the probability changes and the situation changes.
Because we're using expectations, the core is when the expectations of the program are equal.</p>
<h4>Two states, two operations.</h4>
<p>Two states of action are the most fundamental.
We only have two expectations.&#36;p&#36;
Solve the equation&#36;E(d_1)=E(d_2)&#36;You get a critical mass of expectations.</p>
<h4>Three states three.</h4>
<p>More questions for programmes and actions
We need to study more equations.
What would be the best outcome of a particular programme if the expectations were broken?</p>
<h3>Full information value</h3>
<p>We use it to assess the meaning of the information.
Everyone knows the importance of information for decision-making, but what is the value of this information?
All we have to do is compare the expectations of profit and loss with those of loss and gain with which there is no information.
Their margin is the value of complete information.</p>
<h2>Risk-based decision-making approach</h2>
<h3>Probability of use method</h3>
<p>Risks are assumed in any form of risk-based decision-making, each individual's capacity to take risks is not the same, gains and losses are not taken into account, and the effectiveness is introduced to compensate for the absence of subjective considerations</p>
<h4>The concept of effectiveness</h4>
<p>Effectiveness is the final comprehensive assessment of a given decision introduced after a combination of subjective factors and objective introductions; comparison with the gains and losses (a simple monetary measure) that we have used before; utility takes into account subjective factors and takes into account situations where the diversification of decision-making results cannot be measured in monetary terms
The introduction of effects will enable us to make better decisions.</p>
<h4>Function</h4>
<p>It's subjective, so it's a subjective process to determine it; it needs to be determined by asking questions.</p>
<ol>
<li>The two highest and lowest gains and losses were determined to be effective as 0 and 1</li>
<li>We can determine the effectiveness of any one of the defined gains and losses by having zero and one effect, but only by asking questions on a continuous basis, the effectiveness of determining that type of 01 portfolio is the same as that of the programme manager who identified the gains and losses, and the effect of a given point at this point can be determined by a linear combination.</li>
<li>Draw a cross-axis of gains and losses based on the utility of each defined gain and loss.
Using the effectiveness of the programme as a substitute for gain or loss for decision-making is our method of effectiveness decision-making.</li>
</ol>
<h3>Continuous variable risk decision-making</h3>
<p>The utility probability method is based on the decision tree, but the profit-loss matrix is not available in many decision-making situations because the variables are not only discrete but very numerous, and it is unrealistic to list all possible options, so we have introduced risk-based decisions for continuity variables.
Note that this is the way to simplify the selection of continuous variables; we need some special requirements to use it, and we need to list all the options (as many as possible).</p>
<h4>Marginal Analysis</h4>
<p>We use the marginal analysis method, which comes from economics, where the benefits of production tend to be the single-peak curve, and when the marginal gain is zero, the largest amount of production we need.
MP is marginal income. ML is marginal loss. Probability of successful sale is p.
&#36;&#36;MPp_0=ML(1-p_0)&#36;&#36;
From the equation.&#36;p_0&#36; It's the lowest sales chance we can afford.</p>
<h4>Regular decision-making methods</h4>
<p> Using the probability distribution of the state of nature to solve the problem, according to the central limit, many of the natural conditions can change to a normal distribution, thus simplifying the problem as follows: Example
The natural state (market demand) is a density function.&#36;f(x)&#36; The decision variable is the amount that needs to be produced.&#36;a&#36;  Marginal loss is&#36;b&#36;  The best way to calculate it is
 &#36;&#36;a\int_k^{+\infty}f(x)\mathrm{d}x=b\int_{-\infty}^kf(x)\mathrm{d}x,&#36;&#36;
 From the equation.&#36;k&#36; That's what we're supposed to produce.</p>
<h2>Uncertainty decision-making</h2>
<p>In uncertain decision-making, we don't know the probability of natural states happening, and at this point we can't come up with a fixed optimal scenario, and the decision-making guidelines are subjective.</p>
<h4>Optimistic guidelines</h4>
<p>We think it's the most likely that the maximum returns will occur.
So choose the option of maximum return
When we're confident, when we're lured by high returns, we use optimism.
The benefits of each option to take out the highest value for money are compared with the benefits of each option (in the case of loss, the reverse).
Final choice of the most profitable option</p>
<h4>Negative guidelines</h4>
<p>We think it's the best chance of a minimal gain.
We choose the least profitable option and the most beneficial option in the negative.
Often we use pessimism when we lack information.
The return of each option to take out the least-yielding seat is compared with the return of each option (in the case of loss, it should be reversed).
Final choice of the most profitable option</p>
<h4>Guidelines for optimistic factors</h4>
<p>The first two criteria for compromise&#36;\alpha&#36; It's the probability of the most optimistic scenario.
We select the linear combination of maximum and minimal returns and optimism for each option to determine the benefits for each option and ultimately the options for maximizing them.</p>
<h4>Regret values</h4>
<ol>
<li>Calculating the maximum yield for each natural state</li>
<li>The difference between the return on a given scheme and the maximum return on that natural state is our regret.</li>
<li>Give the maximum regret for every scheme </li>
<li>Select the least regretable option
This decision-making option is more appropriate for taking some risks, but the confidence that can be assumed is not entirely certain.</li>
</ol>
<h4>Waiting for probability criteria</h4>
<p>Consider all natural states equally probable.
At this point, the problem becomes common risk-based decision-making.</p>
<h2>Multi-purpose decision-making</h2>
<h3>Introduction to multi-purpose decision-making</h3>
<p>In most of the real world's problems, decision-making needs to take into account more than one goal, and we need to choose between multiple options, but we need to take into account a lot of factors.
Common multi-purpose decision-making lines include:</p>
<ul>
<li>Weighted multi-target to single-target issue</li>
<li>Serial sequence method.</li>
<li>A non-poor solution.</li>
<li>Multi-purpose planning.</li>
<li>Level analysis
<strong>We're just here to introduce some of the classics, more of them in other courses, and the issue of decision-making is always just a general outline, not an exhaustive one.</strong></li>
</ul>
<h3>Linear weighting method</h3>
<p>Linear weighting is the most commonly used multi-target decision-making method, with the idea of determining the weight of the various decision-making objectives, standardizing the objectives that require decision-making, transforming multiple objectives into individual goals and then making decisions
The core of linear weighting is how the weighted approach is determined, and we need to present it separately; here is how the decision indicators should be standardized.
The decision-making indicators are very varied, efficient (the bigger, the better), cost-based (the lower, the better), spatial (the closer a value, the better), and standardization is a mathematical shift that removes the quantum and volume of data.</p>
<ul>
<li>Benefits type:&#36;\frac{x_{ij}-x_{min&#125;&#125;{x_{max}-x_{min&#125;&#125;&#36;</li>
<li>Cost type:&#36;\frac{x_{max}-x_{ij&#125;&#125;{x_{max}-x_{min&#125;&#125;&#36;</li>
<li>Inter-sector type:&#36;1-\frac{x_{ij}-x_{j&#125;&#125;{max(x_{ij}-x_{j})}&#36;</li>
</ul>
<h3>Entropy Law</h3>
<p>The entropy law is an objective empowerment method that uses the entropy of information to calculate the right to entropy, depending on the degree of variation of the indicator, and then amends the right to indicators Heavy
The method of calculating the entropy of information, assuming that the system has a common m state, is the probability that each state will appear.&#36;p_i&#36; Can not open message
&#36;&#36;\mathrm{e}=-\sum_{i=1}^{m}p_i\cdot\mathrm{ln}p_i&#36;&#36;
Now we can calculate the weight of the information entropy. Rights</p>
<ul>
<li>Calculate first&#36;j&#36;Other Organiser&#36;i&#36;Share under programmes&#36;p_{ij}=\frac{x_{ij&#125;&#125;{\sum_{i=1}^{m}x_{ij&#125;&#125;&#36;</li>
<li>Calculate first&#36;j&#36;The entropy of an indicator &#36;e_{j}=-k=-\sum_{i=1}^{m}p_{ij}\cdot\mathrm{ln}p_{ij}&#36;</li>
<li>Calculate the entropy of each indicator &#36;\omega_{j}=\frac{1-e_{j&#125;&#125;{\sum_{j=1}^{n}\left(1-e_{j}\right)}&#36;
Entropy is the weight we use online weighting.</li>
</ul>
<h3>Subjective score</h3>
<p>Authority based on expert advice or subjective scoring by decision makers Heavy</p>
<h3>Level Analysis (AHP)</h3>
<h4>Rationale</h4>
<p>Analysis of the factors involved in complex issues and their interrelationships, delineating the issues to be studied into several levels, at each level judging relative importance by a guideline, at the lowest level calculated by comparing the weights given to the factors, from low to high levels of analysis, and finally calculating the weights for the overall objective to obtain the best options
The hierarchical approach is to straddle a huge problem, give relative importance by comparing it and then use it to determine the weight.</p>
<h4>Relative importance</h4>
<p>It's in the 9th. Out
<img src="/assets/images/probability-statistics-notes/statistical-decision-theory-notes-01.jpg" alt="Statistical policymaking"></p>
<h4>Model construction</h4>
<h5>Establishment of a hierarchy model</h5>
<p>In most of the AHP problems, we think of only one goal, and sometimes the selection of a target gives us the amount of empowerment that needs to be given.</p>
<h5>Two-to-two sub-targets. Construct the judgement matrix.</h5>
<p>The judgement matrix is at the heart of the hierarchical analysis.<em>I'm sorry.
of which&#36;m&#36;It's the number of sub-targets.</em>{ij}&#36; 是子目标&#36;i&#36; 和&#36;j&#36; relative weight</p>
<h5>Consistency of judgement matrices</h5>
<p>The consistency of the matrix is whether or not the decision maker has assured that if A is more important than B is more important than B is more important than C. </p>
<ul>
<li>Calculating Consistency Indicator CI &#36;C.I=\frac{\lambda_{\max}-m}{m-1}&#36; of which&#36;\lambda_{max}&#36; is the maximum feature value of the judgement matrix</li>
<li>Calculation of average random consistency indicator RI<img src="/assets/images/probability-statistics-notes/statistical-decision-theory-notes-02.jpg" alt="Statistical decision-making"></li>
<li>Calculate consistency ratio&#36;CR=\frac{CI}{RI}&#36; When &#36;CR&lt;When meeting the consistency criteria</li>
</ul>
<h5>Calculate weight</h5>
<p>We're using the semantic method to solve weights.&#36;W&#36; There is.
&#36;&#36;A\times W=\lambda_{max}\times W&#36;&#36;
The equation gets weight.
The characteristic vector clearly meets the norm.</p>
<h3>Dictionary style</h3>
<p>It's calculated according to the stratification sequence.
In most of the questions, once the main variable is required to be optimal, the plan is established.
In fact, at this point in time, this is the weighting method that gives the most important indicator a weight of 1 other weight of 0.</p>
<h4>TOPSIS</h4>
<p>It's in the field of research.</p>
<h3>ELECTRE</h3>
<h3>LINMAP</h3>
<p>Within the planning methodology</p>
<h3>Method of superiority and inferiority</h3>
<p>It's in the field of research.</p>
