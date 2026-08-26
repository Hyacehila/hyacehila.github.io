---
title: Questionnaire Reliability, Validity, and Structural Equation Modeling (SEM)
title_zh: 问卷的信效度分析与结构方程模型 (SEM)
date: 2026-01-08 12:00:00 +0800
categories:
- Data Science
- Statistical Modeling & Inference
tags:
- Linear Models
author: Hyacehila
mathjax: true
hidden: true
excerpt: In questionnaire research, reliability and validity are two key standards for measurement quality. This post introduces
  reliability analysis, validity analysis, confirmatory factor analysis, and SEM.
description: In questionnaire research, reliability and validity are two key standards for measurement quality. This post
  introduces reliability analysis, validity analysis, confirmatory factor analysis, and SEM.
excerpt_zh: 在问卷研究中，信度和效度是衡量问卷质量的两个标准。本文介绍信度分析、效度分析以及验证性因子分析（CFA）与结构方程模型（SEM）的基本概念与流程。
permalink: /blog/2026/01/08/questionnaire-sem/
lang: en
translation_key: 2026-01-08-questionnaire-sem
translation_status: machine
translation_source_hash: f2ec8f6c6ebf96bd865c90fa140571dfc40cd05504138ab8cadfb0afb544ced7
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>About Trust and Effectiveness</h2>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base: linear model, minimum 2x2 estimate and regression diagnosis</a>、<a href="/en/blog/2025/09/23/advanced-linear-regression-notes/">Linear regression step: proposed alignment, model selection and co-line Sex</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>In the questionnaire study,<strong>Trust</strong>and<strong>Effect</strong>These are two measures of the quality of the questionnaire. The analysis and conclusions based on the data collected from a questionnaire are unreliable if the letter and effectiveness of the questionnaire are not met.</p>
<ul>
<li><strong>Letter Analysis</strong> Concerned that “Does my questionnaire measure stable, consistent and reliable?” It cares.<strong>Measurement tool itself</strong>The question is not whether it measured the right thing.</li>
<li><strong>Effect Analysis</strong> Focusing on "Did my questionnaire measure what I wanted to measure?" It cares.<strong>Content and purpose of measurements</strong>Correct.</li>
</ul>
<p>Assuming that we are looking at “staff satisfaction”, this is a typical abstract concept that needs to be measured through a number of specific issues. We divide it into two dimensions:<strong>Pay satisfaction</strong>and<strong>Working environment satisfaction</strong>The following general questionnaire was developed.</p>
<p><strong>Part I: Remuneration satisfaction</strong></p>
<ul>
<li>&#36;X_1&#36;I think my pay matches my work. (1 - 2 - 3 - 4 - 5)</li>
<li>&#36;X_2&#36;I am satisfied with my level of remuneration compared to other colleagues in the company. (1 - 2 - 3 - 4 - 5)</li>
<li>&#36;X_3&#36;The company ' s welfare benefits (such as leave, subsidies, etc.) are attractive. (1 - 2 - 3 - 4 - 5)</li>
</ul>
<p><strong>Part II: Satisfactory working environment degrees</strong></p>
<ul>
<li>&#36;Y_1&#36;I have a good relationship with my colleagues and have a good cooperation. (1 - 2 - 3 - 4 - 5)</li>
<li>&#36;Y_2&#36;I work in a positive, positive atmosphere. (1 - 2 - 3 - 4 - 5)</li>
<li>&#36;Y_3&#36;. <strong>I often feel tremendous pressure from work.</strong> ( 1 - 2 - 3 - 4 - 5 ) <em>(Note: This is a reverse score.)</em></li>
</ul>
<h2>Letter Analysis</h2>
<p><strong>Letter Analysis</strong>Test the results of the questionnaire.<strong>Stability and consistency</strong>I don't know. This is the main concern.<strong>Internal consistency trust</strong>, i.e. the measurement of “consistent calibre” of multiple topics of the same concept. The usual method.<strong>Klonbach. &#36;\alpha&#36; Coefficient (Cronbach)&#39;s alpha)</strong>。</p>
<p>Because of the title &#36;Y_3&#36; It's a reverse score, while the rest is a positive description. In order to be consistent, we must &#36;Y_3&#36; . That is, in the 5-point scale, use <code>6 - 原始得分</code>, the 5-point system is maintained but converted to a positive measure.</p>
<p>The analysis of the level of confidence is directed at the “dimensional dimension” rather than the entire questionnaire. We need separate analyses of “pay satisfaction” and “work environment satisfaction”.</p>
<p>&#36;\alpha&#36; The coefficient measures a set of topics<strong>Internal coherence</strong>The extent. If these topics are indeed measuring the same concept, there should be a strong positive correlation between scores.&#36;\alpha&#36; The coefficient does not directly calculate the correlation between all topics, but is adopted<strong>Difference</strong>This larger indicator indirectly measures this relevance. The formula is as follows:</p>
<p>&#36;&#36;
\alpha = \frac{k}{k - 1} \left( 1 - \frac{\sum_{i=1}^{k} \sigma_{Y_i}^2}{\sigma_X^2} \right)
&#36;&#36;</p>
<p>of which &#36;k&#36; It's the number of topics.&#36;\sigma_{Y_i}^2&#36; I'm going to have to calculate the total difference of all the topics.&#36;\sigma_{X}^2&#36; Calculates the total division difference for the entire dimension. If there's a high correlation between the topics, the difference in the total score will be sustained by the “coordinate difference”, resulting in <code>(各题目方差之和 / 总分方差)</code> This is a small margin.&#36;\alpha&#36; Value is high. If the subject is not relevant, then it should be the same.&#36;\alpha&#36; Value is low. coefficient &#36;k / (k-1)&#36; It's an amendment to make sure that the subject is completely irrelevant.&#36;\alpha&#36; The theoretical value of the coefficient is 0.</p>
<p>Clonbach for each dimension &#36;\alpha&#36; A coefficient that measures the internal consistency of the questionnaire and supports the determination that the answer is stable.</p>
<h2>Effect Analysis</h2>
<p>Validity analysis to test the questionnaire<strong>Measure accurately</strong>The concept we want to measure. This is the main concern.<strong>Structure Effect</strong>is the theoretical structure of the questionnaire (two dimensions) consistent with the structure of the data actually collected. The usual method.<strong>Validity factor analysis (CFA)</strong>, to verify the consistency of data and theoretical assumptions.</p>
<p>The CFA is a common tool to test structural effectiveness. It's used to verify the model that we've designed. &#36;X_1, X_2, X_3&#36; It's a pay factor.&#36;Y_1, Y_2, Y_3&#36; “Environmental” factor) and the degree to which the data are proposed.</p>
<p>The first step is to develop theoretical assumptions that give us an understanding of the problem.<strong>Assumptions model</strong>e.g.:</p>
<ul>
<li>There are two potential factors (substantial variables): “salary satisfaction” and “work environment satisfaction”.</li>
<li>Title &#36;X_1, X_2, X_3&#36; It is an observation indicator of the “pay satisfaction” factor.</li>
<li>Title &#36;Y_1, Y_2, Y_3&#36; (reversed) is the observation indicator for the “work environment satisfaction” factor.</li>
<li>There may be a correlation between these two factors.</li>
</ul>
<p>The entire CFA workflow can be summarized as a comparison of the two matrixes:</p>
<ul>
<li><strong>Sample Arguments&#36;S&#36; - Sample Covariance Matrix)</strong>：<strong>Actual data collected</strong>The agreed matrix calculated reflects the true correlation between all the topics in the questionnaire.</li>
<li><strong>Model Implicit Difference Matrix ()&#36;\Sigma(\theta)&#36; - Model-implied Covariance Matrix)</strong>: Based on you<strong>Predefined theoretical model</strong>, by mathematical formula<strong>Figure</strong>A collusive matrix. It represents “if theoretical models are established, the data should be presented”.</li>
</ul>
<p>CFA By Adjusting All in Model<strong>Parameters</strong>The parameters to be adjusted to bring the two matrices closer together include:</p>
<ul>
<li><strong>Factor load</strong>: submersible variables (e.g. “pay satisfaction”) for observational variables (e.g. &#36;X_1&#36;The degree of impact.</li>
<li><strong>Because of the difference between the two</strong>Level of correlation between different submersible variables (e.g. “remuneration” and “environment”).</li>
<li><strong>Error</strong>: Part of each subject that cannot be explained by subvariant (i.e. measuring error).</li>
</ul>
<p>We need to map the theoretical structure in the CFA software, and then rely on software to optimize the parameters:</p>
<ul>
<li>Use<strong>Ellipse</strong>Organisation<strong>Subtract Variables</strong>。</li>
<li>Use<strong>Rectangle</strong>Organisation<strong>Observation variables</strong>。</li>
<li>Use<strong>One-way Arrow</strong>From submersible to its corresponding observation variable,<strong>Factor load</strong>。</li>
<li>Use<strong>Double Arrow</strong>Connect to different submersible variables, representing<strong>Because of the difference between the two</strong>。</li>
<li>Each observation variable will have a one-way arrow pointing at itself, representing<strong>Error</strong>。</li>
</ul>
<p>The CFA tests the structural effectiveness through three levels of evidence:</p>
<ul>
<li><strong>Card-check</strong>: test whether there are significant differences between matrices, but &#36;p &gt; 0.05 million can only be considered ready. Large samples are almost always significant, so only for reference purposes.</li>
<li><strong>RMSEA</strong>: The smaller the average error of the model in each degree of freedom, 0.08.</li>
<li><strong>SRMR</strong>: The average of the differences in the relevant coefficients, the smaller the better, 0.08.</li>
<li><strong>CFI / TLI</strong>: Compare your model with a baseline model that is "not relevant to all variables" and see how much better it is. The closer one, the better.&gt; 0.90 (Acceptable).</li>
</ul>
<p>Observation<strong>Standardised Factor Load</strong>Answer: "Is my subject measuring the subvariant I want?" Because the subload should be high enough and &#36;p&#36; Significant value. Usually. &gt; 0.5, ideal &gt; 0.7。</p>
<p>Research<strong>Differentiate Effect</strong>Answer: "Is something really different between the different submersible variables I've measured?" If the relevant coefficient is too high (if &gt; 0.85), indicates that the two factors may measure the same concept and are not sufficiently differentiated.</p>
<h2>Validity factor analysis (CFA) and structural equation model (SEM)</h2>
<p><strong>The CFA is the foundation and exception of the SEM, the extension and extension of the CFA</strong>I don't know. The former only includes<strong>Measurement model</strong>I don't know. It's about...<strong>Validation of measurement tools</strong>I don't know. The question it answered was, "Did my questionnaire title accurately measure the abstract concepts that I wanted to measure? "</p>
<p><strong>SEM (structure equation model)</strong> It's also included.<strong>Structure Model</strong>, to analyse causal relationships between submersible variables (abstract concepts). The answer to the question was: “Are my theoretical assumptions (e.g., that pay satisfaction will affect work environment satisfaction and, in turn, exit trends) supported by data?”</p>
<p>It's...<strong>Just add a causal path between submersible variables to the CFA.</strong>I don't know. When we use it, we mainly answer the following questions:</p>
<ul>
<li><strong>Path coefficient</strong>: the direct effect of one submersible variable on the size and direction of another submersible variable (similar to the regression factor), typically between -1 and +1 and provided &#36;p\text{-value}&#36;。</li>
<li><strong>Indirect effects/intermediaries Response</strong>: Variable A influences the extent of variable C through variable B.</li>
<li><strong>Model alignment</strong>: Does the entire theoretical model of causality that I designed fit the actual data?</li>
</ul>
<p>SEM is a return equation with some advantages over traditional multi-linear regression. Although the technical details have been better sealed, they can still be used in many academic studies. It includes the following advantages:</p>
<ol>
<li><strong>Process measurement errors</strong>This is the main advantage of SEM. SEM recognizes that there are errors in each observation variable and separates it from the model, thus estimating more purely the relationship between submersible variables.</li>
<li><strong>Addressing multiple causality at the same time</strong>: A regression analysis can only test one cause variable at a time. SEM can test a complex network of multiple variables and self-variant in a model.</li>
<li><strong>Estimated brokering and reconciliation effects</strong>: SEM is the one that tests the intermediate effect (A through B) and the regulatory effect (B by B)<strong>Best Tools</strong>I don't know. It can be clearly deciphered.<strong>Direct effects</strong>、<strong>Indirect effects</strong>and<strong>Total effect</strong>。</li>
<li><strong>Provide overall model alignment</strong>: A regression analysis can only tell you whether a single path is significant, but it is not possible to evaluate the good or bad of the whole “theory model”. SEM offers a series of composite indices (e.g., CFI, RMSEA) that allow you to judge the degree to which your entire theoretical blueprint matches reality.</li>
<li><strong>Comparative competition theory</strong>: You can propose two different theoretical models, and then use SEM data to determine which model can better interpret the data.</li>
</ol>
