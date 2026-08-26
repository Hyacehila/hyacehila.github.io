---
title: 'Shapley and SHAP: State-of-the-Art Tools for Model Interpretability'
title_zh: Shapley 与 SHAP——模型解释性的 SOTA 工具
date: 2026-02-28 00:00:00 +0800
categories:
- Machine Learning
- Classical Machine Learning
tags:
- Interpretability
- Evaluation
author: Hyacehila
mathjax: true
hidden: true
excerpt: An introduction to the theoretical basis of Shapley values and their machine-learning application in SHAP, including
  where they work and how to choose background data.
description: An introduction to the theoretical basis of Shapley values and their machine-learning application in SHAP, including
  where they work and how to choose background data.
excerpt_zh: 本文介绍 Shapley 值的理论基础及其在机器学习中的应用 SHAP，讨论它适合解释什么、容易被误读在哪里，以及工程使用时该怎样选择背景数据。
permalink: /blog/2026/02/28/shapley-and-shap/
lang: en
translation_key: 2026-02-28-shapley-and-shap
translation_status: machine
translation_source_hash: 5b77c54014ec4aff1602c9a5b1a076adf04cbf8e7163c7823c5660b0f020c442
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>This article is devoted to Shapley values and SHAP. If you need to understand the complete relationship between the interpretable model, the PDP, ALE, the replacement importance, LIME, the counterfact and impact function, you can<a href="/en/blog/2024/05/23/interpretable-machine-learning-notes/">Explanatory Machine Learning: Model Interpretation, SHAP and Anti-Factual Methods</a>Start.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2024/05/23/interpretable-machine-learning-notes/">Explanatory machine learning: model interpretation, SHAP and anti-fact methods</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h2>Shapley Value, fair distribution of game theory.</h2>
<p>Shapley values explain projections by assuming that each characteristic value of the example is the “player” in the game, where the projection target is total expenditure. Shapley values are a method of coalition games that will show us how to distribute total expenditures fairly among characteristics. We need to adopt a simple example of how to understand Shapley Valle from a theoretical point of view, so that it can be applied to the areas that are needed.</p>
<p>It is assumed that machine learning models have been trained to forecast apartment prices. For an apartment, the projected price is Euro300,000, which needs to be explained. The apartment is 50 square metres in size, located on the 2nd floor, near which there is a park, and where cats are prohibited from entering, the average forecast price for all apartments is Euro310,000. How much does each characteristic value contribute to the projection compared to the average projection?</p>
<p>We'll draw an analogy.</p>
<ul>
<li>Game is a predictive task for a single example of a data set</li>
<li>“Proceeds” is the actual projection for this example less the average projection for all examples Value</li>
<li>" Player" is the characteristic value of the example</li>
</ul>
<p>We want to use the player to explain the gain, that is, the characteristic value to explain the margin of 10,000.</p>
<h3>Mathual definition of Shapley values: cooperative game model</h3>
<p>In the game theory, a cooperative game is defined by two elements:&#36;(N, v)&#36;。</p>
<ol>
<li><p><strong>Players Gather (in %1)&#36;N&#36;)</strong>: All players involved in the game. In your case, players are the characteristics. To clarify, we assume that this apartment has only three features:</p>
<ul>
<li><p>Player 1: Near Park</p>
</li>
<li><p>Player 2: Ban Cats</p>
</li>
<li><p>Player 3:50 m2 (Area)</p>
<p>So, the total number of players &#36;M = 3&#36;Come on, men. &#36;N = {1, 2, 3}&#36;。</p>
</li>
</ul>
</li>
<li><p><strong>Feature Functions (&#36;v&#36;)</strong>: Also called Value Functions, which represent any player's subset (Union) &#36;S&#36;The proceeds that can be obtained.</p>
<ul>
<li>In machine learning,&#36;v(S)&#36; Defined as:<strong>In the subsets that only know &#36;S&#36; Model expectations for housing prices in the case of medium-specific values</strong>。</li>
<li>&#36;v(\emptyset)&#36;: The forecast value when there is no feature, i.e. the average price of all apartments (310,000 euros).</li>
<li>&#36;v(N)&#36;: Final projection when all characteristics are present (Euro300,000).</li>
</ul>
</li>
</ol>
<p><strong>Problem</strong>: We all have it. &#36;v(N) - v(\emptyset) = -10,000&#36; “Failment” (total gain) for the euro. This - 10,000 euros, how exactly, uncontroversially, should be allocated to players 1, 2 and 3?</p>
<p>Dr. Shapley's answer in 1953, which is the first. &#36;i&#36; Shapley value for a player (identity) &#36;\phi_i&#36; The precise formulas are as follows:</p>
<p>&#36;&#36;
\phi_i = \sum_{S \subseteq N \setminus {i&#125;&#125; \frac{|S|! (M - |S| - 1)!}{M!} (v(S \cup {i}) - v(S))
&#36;&#36;</p>
<ul>
<li><strong>&#36;S \subseteq N \setminus {i}&#36;</strong>: means all players are not included &#36;i&#36; The union (subset).</li>
<li><strong>&#36;v(S \cup {i}) - v(S)&#36;</strong>: This is the player &#36;i&#36; Join the Alliance &#36;S&#36; And then I brought it.<strong>Marginal Contribution</strong>。</li>
<li><strong>&#36;\frac{\mid S \mid ! (M - \mid S\mid - 1)!}{M!}&#36;</strong>: This is a<strong>Weight factor</strong>I'm sorry. It's the mathematical meaning of combining it: at all levels. &#36;M!&#36; A possible player in order, player &#36;i&#36; It's in the subset. &#36;S&#36; , the number of the rows added immediately after the player has joined.</li>
</ul>
<p>In summary:<strong>Shapley values are the average marginal contribution of the feature values in all possible alliances (Coalition).</strong></p>
<h3>Accurate calculation of the combination of house purchases</h3>
<p>Now, we're going to calculate the Shapley value of "No Cat" ( Player 2) &#36;\phi_2&#36;。</p>
<p>Known &#36;M = 3&#36;, we need to build up all coalitions that do not include Player 2 &#36;S&#36;These are:&#36;\emptyset&#36;(Blank collection)&#36;{1}&#36;、&#36;{3}&#36;、&#36;{1, 3}&#36;。</p>
<p>Assuming that we have obtained the following alliances by making the desired calculation of the data set: &#36;v(S)&#36; Projected value (in EUR):</p>
<ul>
<li>&#36;v(\emptyset) = 31&#36;</li>
<li>&#36;v({1}) = 32&#36; ♪ Only the park ♪</li>
<li>&#36;v({3}) = 30.5&#36; "Only 50 square meters."</li>
<li>&#36;v({1, 3}) = 31.5&#36; ♪ Knows there's a park and 50 square meters ♪</li>
<li>&#36;v({2}) = 30&#36; "Only a cat is forbidden."</li>
<li>&#36;v({1, 2}) = 31&#36; (Park + no cats)</li>
<li>&#36;v({2, 3}) = 29.5&#36; (Cats are forbidden + 50 m2)</li>
<li>&#36;v({1, 2, 3}) = 30&#36; (All features are assembled, i.e. final projection)</li>
</ul>
<p>According to the formula, we calculate the weighted sum of the four sub-items:</p>
<p><strong>1. When &#36;S = \emptyset&#36; (no feature exists, player 2 joins):</strong></p>
<ul>
<li>Weight:&#36;\frac{0! (3 - 0 - 1)!}{3!} = \frac{2}{6} = \frac{1}{3}&#36;</li>
<li>Marginal contributions:&#36;v({2}) - v(\emptyset) = 30 - 31 = -1&#36;</li>
<li>Weighted results:&#36;\frac{1}{3} \times (-1) = -0.333&#36;</li>
</ul>
<p><strong>2. When &#36;S = {1}&#36; (Park features already exist, player 2 joins):</strong></p>
<ul>
<li>Weight:&#36;\frac{1! (3 - 1 - 1)!}{3!} = \frac{1}{6}&#36;</li>
<li>Marginal contributions:&#36;v({1, 2}) - v({1}) = 31 - 32 = -1&#36;</li>
<li>Weighted results:&#36;\frac{1}{6} \times (-1) = -0.167&#36;</li>
</ul>
<p><strong>3. When &#36;S = {3}&#36; (50 m2 feature already exists, player 2 joins):</strong></p>
<ul>
<li>Weight:&#36;\frac{1! (3 - 1 - 1)!}{3!} = \frac{1}{6}&#36;</li>
<li>Marginal contributions:&#36;v({2, 3}) - v({3}) = 29.5 - 30.5 = -1&#36;</li>
<li>Weighted results:&#36;\frac{1}{6} \times (-1) = -0.167&#36;</li>
</ul>
<p><strong>4. When &#36;S = {1, 3}&#36; (Parks and 50 m2 features exist, player 2 joins):</strong></p>
<ul>
<li>Weight:&#36;\frac{2! (3 - 2 - 1)!}{3!} = \frac{2}{6} = \frac{1}{3}&#36;</li>
<li>Marginal contributions:&#36;v({1, 2, 3}) - v({1, 3}) = 30 - 31.5 = -1.5&#36;</li>
<li>Weighted results:&#36;\frac{1}{3} \times (-1.5) = -0.5&#36;</li>
</ul>
<p><strong>Final calculation &#36;\phi_2&#36;：</strong></p>
<p>&#36;&#36;
\phi_2 = (-0.333) + (-0.167) + (-0.167) + (-0.5) = -1.167 \text{ 万欧元}
&#36;&#36;</p>
<p><strong>Conclusions</strong>The feature, “No Cats”, strictly reduced the projected price of the house by 11,670 euros.</p>
<h3>Value of Shapley Valle</h3>
<p>You might ask: Why do you count with such weight? Just take the feature out and count the margin once. "that is, the idea of changing the importance of character, which is not perfect."</p>
<p>That is why Shapley's values are considered to be in the interpretable field.<strong>It's the only way to distribute the four following game justice simultaneously.</strong></p>
<ol>
<li><p><strong>Validity (Efficacy / Addability)</strong>：</p>
<p>The sum of the Shapley values for all characteristics shall be precise equal to the difference between the final projection and the underlying value. There is no redundancy, no omission.&#36;\sum_{i=1}^M \phi_i = v(N) - v(\emptyset)&#36;</p>
</li>
<li><p><strong>Symmetry</strong>：</p>
<p>If the two characteristics produce exactly the same marginal contribution in all possible alliances, then their Shapley value must be exactly equal. This ensures that algorithms are not biased against certain characteristics.</p>
</li>
<li><p><strong>Virtual (Dummy Axiom / zero player)</strong>：</p>
<p>If a characteristic (such as a worthless gift from a house purchase) joins any alliance and joins any alliance, and the margin contribution is never zero, then its Shapley value must be zero.</p>
</li>
<li><p><strong>Linear Group (Additivity)</strong>：</p>
<p>If we use Model A and Model B to forecast the house price separately, then add the results of both as the projection for Model C. Then a Shapley value for a feature in Model C is bound to be equal to the simple addition of Shapley values in A and B.</p>
</li>
</ol>
<p><strong>It's a problem.</strong>: In the face of machine learning models that include deep non-linear and interactive features, the traditional replacement feature importance (Permutation Importance) is difficult to address in terms of reliance and coupling between features. Shapley values give one by taking all possible "exit order" and taking expectations from all features<strong>A fairer way of attribution under the weight of the game theory.</strong>。</p>
<h3>Apply examples</h3>
<p><strong>Characteristic Value &#36;j&#36; The Shapley value is explained by the average projection of the data set. &#36;j&#36; The value of a characteristic contributes to the prediction of this particular example by: &#36;ϕ_j&#36;</strong>   Shapley values apply to both classification (output probability) and regression issues.</p>
<p>We use the risk-based data set for cervical cancer as an example.</p>
<p>The data on cervical cancer is concentrated on the Shapley value of one woman. The projection is 0.57, which is 0.54 more than average probability.
<img src="/assets/images/shap/%E5%8F%AF%E8%A7%A3%E9%87%8A%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-21.png" alt="">
And it's clear from this example that the impact of STDs is the greatest and the impact of increasing probability.</p>
<h3>Shapley can achieve precise explanations for the comparison.</h3>
<p>Shapley Value<strong>Allows contrasting interpretation</strong>I'm sorry. Without comparing the projections with the average projections of the entire data set, you can compare them with subsets or even single data points. This introduces characteristics that are often overlooked by beginners in practical applications:<strong>Relativity of Interpretation</strong>。</p>
<p>In human cognitive habits, when we ask "why," we actually ask, "Why is A?" <strong>Not</strong> B?”. It's called<strong>Comparative Interpretation</strong>I'm sorry. We need to return to the feature function mentioned in the previous statement. &#36;v(S)&#36; and base values &#36;v(\emptyset)&#36;。</p>
<p><strong>Change background distribution (Background Distribution)</strong></p>
<p>In the exact reasoning of the previous section, we set &#36;v(\emptyset) = 310,000&#36; Euros. How did you get &#36;310,000? It's a model in the context of our "no signature"<strong>Entire Data Set</strong>All apartments are forecasted for price.<strong>Average</strong>。</p>
<p>Calculates the union value with some features &#36;v(S)&#36; How do you deal with the mathematically when (for example, only “near the park”), those features of “not knowing” (for example, “area” and “does a cat have been allowed to be maintained” ?</p>
<p>The standard approach is to fill (marginalization) with the characterization values of other apartments in the entire data set, and then to expect. ** The actual meaning of this sentence is therefore:<strong>You can replace this "background data set to fill and calculate averages " , thereby changing the interpretation baseline (Baseline).</strong></p>
<p>Let's use the example of apartments to practice these three contrast levels:</p>
<p><strong>Compared with " Whole Data Set" (default standard scenario)</strong></p>
<ul>
<li><strong>Background data</strong>100,000 apartments in the city.</li>
<li><strong>Base Value &#36;v(\emptyset)&#36;</strong>Average city-wide housing cost: 310,000.</li>
<li><strong>You're going to explain the problem.</strong>"Why is this apartment 300 thousand?<strong>Average city-wide</strong>Ten thousand dollars?</li>
<li><strong>Meaning of the Shapley value</strong>The ban on cats (-11.67), near parks (+0.50) etc. is a contribution to the city-wide average.</li>
</ul>
<p><strong>Compare with "Specific Subsets" (comparable explanations - subsets)</strong></p>
<p>Suppose you want to explain the price to a client who's looking at the old city. The average price for the city doesn't mean anything to him.</p>
<ul>
<li><strong>Background data</strong>: You only enter 5,000 apartments in the Old City as background data sets into the model.</li>
<li><strong>Base Value &#36;v(\emptyset)&#36;</strong>Average housing price in the old city, assuming 350,000.</li>
<li><strong>You're going to explain the problem.</strong>"Why is this apartment 300 thousand?<strong>Average in the old city</strong>50 grand?</li>
<li><strong>Meaning of the Shapley value</strong>: The Shapley values recalculated at this time will undergo dramatic changes. Because old city probably doesn't have parks in general, and this is the positive contribution of the "near park" feature. &#36;\phi_1&#36; It'll be much bigger than before, and it fills the 50 grand gap between the average price of the old city.</li>
</ul>
<p><strong>Compare with " Single Data Point " (comparable explanation - individual)</strong></p>
<p>That's the most extreme comparison. The client pointed to another apartment across the street (sold at 320,000) and asked you, "Why do I have a 300 million apartment on the same floor as the two rooms, in the same area?"</p>
<ul>
<li><strong>Background data</strong>: Only the “320,000 apartments” across the street are used as the only background data.</li>
<li><strong>Base Value &#36;v(\emptyset)&#36;</strong>The estimated value of the contrast apartment (320,000).</li>
<li><strong>You're going to explain the problem.</strong>"Why is this apartment 300 thousand?<strong>The apartment across the street.</strong>Cheap, 20 grand?"</li>
<li><strong>Meaning of the Shapley value</strong>: in this case, all the same features (area, segment) will be given a Shapley value<strong>Directly to 0</strong>(Because they do not differ between the two samples, the marginal contribution is 0). The final difference will be 100%, perfectly distributed to those who are not.<strong>Different features</strong>(e.g., a cat is allowed across the street, and the cat is forbidden.</li>
</ul>
<p><strong>Note that a comparative interpretation does not mean that the model will be retrained. The model is still being trained throughout the data set; in pursuing the explanatoryity that Shapley brings, we change the baseline for background data, thereby changing the interpretation baseline.</strong></p>
<p>This comparison is useful in real business. The ability to customise benchmarks is also more difficult to achieve in a more stable way, such as LIME.</p>
<ul>
<li><strong>- We're gonna have to open the case.</strong>"Why is the credit rating of user A lower than that of user B 50? " (Pilot data points compared)</li>
<li><strong>Analysis of population disparities</strong>"Why is the probability of losing users this month 20% higher than the rate of active users last month?" (subset comparison)</li>
<li><strong>Model Debug</strong>"Why is this negative sample miscalculated as positive scored so much higher than the real negative sample group?" (Subset comparison)</li>
</ul>
<p>From the change of shaP calculation <code>background_data</code> Parameters (Python Extension) can transform model interpretation from general to precise impact. Shapley's excellent and interpretable skills are one of the best ways to do it.</p>
<h3>Advantages and disadvantages</h3>
<p>As can be seen from the above extrapolation, the complexity of calculating the Shapley value is that &#36;O(2^M)&#36;I'm sorry. If the model had 100 features, even using the world ' s top super-calculation would be the exact value of a prediction before the destruction of the universe. This is why the purely theoretical Shapley values “cannot be widely applied” in machine learning until the SHAP technology appears.</p>
<p>Shapley values may be misunderstood. The Shapley value of the feature value is not the difference in the projection after the feature has been removed from the model training. Shapley values are explained by the current set of feature values, which contribute to the difference between the actual and average projection values, which are estimated Shapley values.</p>
<p>The explanation created using the Shapley value method always uses all features. It's not appropriate for people who seek a thin explanation.</p>
<p>Shapley values return a simple value for each feature, but no predictive model like LIME</p>
<h2>SHAP - Faster, more usable</h2>
<p>From theoretically perfect Shapley values to engineering-based SHAP (SHAP) is a leap in the field of interpretativeity of machine learning. To understand SHAP, it cannot be seen as a Python library. And what is more important is:<strong>What were the flaws in the Shapley values that Lundberg and Lee had addressed when they introduced SHAP in 2017? What's the compromise?</strong></p>
<h3>Dimensions disaster and approximation calculations</h3>
<p>We have extrapolated in the previous section that calculating the exact Shapley values requires all possible combinations of features, the time complexity of which is, &#36;O(2^M)&#36;I'm sorry. For a model with 100 features, even a sample would take a terrible time and would be totally unrealistic.</p>
<p><strong>How did SHAP work out?</strong> SHAP did not invent new distribution theory, but cleverly converted the Shapley value into a single one.<strong>Addive Special Characterisation Method (Additive Special Organization Method)</strong>I'm sorry. It introduced a simplified local interpretation model. &#36;g&#36;：</p>
<p>&#36;&#36;
g(z&#39;) = \phi_0 + \sum_{j=1}^M \phi_j z&#39;_j
&#36;&#36;</p>
<p>of which &#36;z&#39; \in {0, 1}^M&#36; 表示特征是否存在的二进制向量（1 表示特征在联盟中，0 表示被隐藏），&#36;\phi j&#36; is the Shapley value we're counting.</p>
<p>To solve this equation, SHAP has proposed two main approximation algorithms:</p>
<ol>
<li><p><strong>KernelSHAP (memode unrelated)</strong>：</p>
<p>It's a...<strong>Weighted linear regression</strong>I'm sorry. It randomly extracts a portion of the union &#36;z&#39;And assign a weight to each union -<strong>Shapley Kernel</strong> &#36;\pi_{x&#39;}(z&#39;)&#36;：</p>
<p>&#36;&#36;
\pi_{x&#39;}(z&#39;) = \frac{M - 1}{\binom{M}{|z&#39;|} |z&#39;| (M - |z&#39;|)}
&#36;&#36;</p>
<p>Ration factor obtained by minimizing weighted square loss to match linear models &#36;\phi_j&#36; The nearest Shapley value. This directly integrates the LIME and Shapley values in mathematics.</p>
</li>
<li><p><strong>TreeSHAP (street model specific)</strong>：</p>
<p>Using the internal structure of the decision tree (separate paths and sample coverage of nodes), the complexity of calculation is reduced from index to multiple Level &#36;O(TLD^2)&#36;（&#36;T&#36; For the number of trees,&#36;L&#36; For leaves,&#36;D&#36; For depth. That's why SHAP would be so quick in XGboost or LightGBM.</p>
</li>
</ol>
<p>The problem seems to have been solved, but this approximation means we must have lost something.</p>
<h3>What exactly does the "missing" of the features mean? (The Missingness Problem)</h3>
<p>In game theory, the absence of a player is easy to understand (he does not participate in games). But in machine learning, the absence of characteristics is a very controversial mathematical issue. Once your model is trained, you must eat all the features, and most models cannot enter one directly into the nervous network. <code>NaN</code>。</p>
<p><strong>The choice and discussion of SHAP:</strong></p>
<p>To calculate the union value of the component features &#36;v(S)&#36;, we need to be able to read the features that are not in the alliance. &#36;\bar{S}&#36;It's a process. There are two distinct paths in the academic world:</p>
<ol>
<li><p><strong>Marginal Expectations (Marginal Exportation/ Intervention)</strong>：</p>
<p>Force break-up of the connection between the features, assuming known features &#36;X_S&#36; and unknown features &#36;X_{\bar{S&#125;&#125;&#36; Be independent. The practical operation is to randomly replace missing features with values in the Background Data.</p>
<p>&#36;&#36;
v(S) = E[f(X_S, X_{\bar{S&#125;&#125;) \mid X_S = x_S] \approx \int f(x_S, X_{\bar{S&#125;&#125;) dP(X_{\bar{S&#125;&#125;)
&#36;&#36;</p>
<p><em>Problem</em>: This can lead to serious<strong>Anti-fact samples (out-of-distribution, OOD)</strong>I'm sorry. For example, the Alliance retains “pregnancy = True” but has filled “gender = man” with background data, and the model is forced to predict a data point that is completely unrealistic. This is the area where the shap has been sorely ill.</p>
</li>
<li><p><strong>Conditional Expectations (Conditions Export / Observation)</strong>：</p>
<p>Considering the correlation between features, extrapolation of unknown features from a distribution that meets the conditions of known features.</p>
<p>&#36;&#36;
v(S) = E[f(X_S, X_{\bar{S&#125;&#125;) \mid X_S = x_S] = \int f(x_S, X_{\bar{S&#125;&#125;) dP(X_{\bar{S&#125;&#125; \mid X_S = x_S)
&#36;&#36;</p>
<p><em>Problem</em>: it contradicts the "virtual justice" of the Shapley values. If the model is not used at all for feature A, but the characteristic A is strongly associated with feature B, the condition distribution will wrongly attribute part of the credit for feature B to feature A.</p>
</li>
</ol>
<p><strong>Current status</strong>: Standard <code>shap</code> The library defaults to use the first type (marginal expectations/background data replacement), as it is easier to calculate and meets the instincts of causal intervention, but this raises the next fatal question.</p>
<h3>Characteristic Relevance Trap (The Correlation Trap)</h3>
<p>This is one of the most difficult research challenges in the AI field that can be explained at this time, and because the marginal expectation method produces unreliable samples, it leads to distortions in the final calculation of SAP Value, and only because of the relevance that will appear in high-dimensional statistics.</p>
<p>An extreme example is the assumption that “house area (m2)” and “house area (m2)” are thrown into the model at the same time. Because they're 100% relevant, the model actually needs only one of them to make perfect predictions.</p>
<ul>
<li>If TreeSHAP is used, it may randomly assign weights on both of these features (e.g. 50% on one side). This could have led to an otherwise critical “area” feature, which had been reduced in importance and fell sharply in the SHAP summary.</li>
<li>This would mislead the operatives and make them think that the area is not important.</li>
<li>The consolidation of highly relevant indicators was necessary, but could not fully address the problem.</li>
</ul>
<p><strong>The directions worth studying.</strong>: The academic community is now studying the imposition of the order of interpretation of characteristics by calculating “asymmetric Shapley values” or by combining “causal Graphics” to strip off mixed co-linear interference.</p>
<h3>Relevance vs. causality (Correlation vs. Causation)</h3>
<p>This is a common area of error in thinking when applying SHAP, and in causal and ex post facto interpretation.</p>
<p>SHAP explains ** "What patterns models learn."<strong>, not</strong>“What is the causal link in the real world”**.</p>
<p>If your model finds that the lighter is highly likely to cause lung cancer, SHAP will honestly tell you that the characteristic of the lighter is a very positive contribution to increasing the disease prediction.</p>
<p>But in operational applications, if you conclude that “the confiscation of people's lighters can reduce the incidence of lung cancer”, it is a big mistake. SHAP is only a faithful reflection of the bias of the model and its relevance in the data, and it cannot replace causal inference.</p>
<h3>Example:</h3>
<p>Below is an overview of the organization of SHAP syndication interpretation, feature importance, summary diagrams, reliance on maps and interactive values, which is based on the following: <em>Interpretable Machine Learning</em> .</p>
<p>We'll use the risk factors of cervical cancer as examples of a disaggregated data set.</p>
<p>Because SHAP calculates the Shapley value, the explanation is the same as in the Shapley Example section, although we have some interesting visualizations as follows:
<img src="/assets/images/shap/%E5%8F%AF%E8%A7%A3%E9%87%8A%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-22.png" alt="">
The arrow and length reflect the SHAP effect with the position of the median line. Reactions</p>
<ul>
<li>The first example, though STDs have increased the probability, was filled with a lot of retrofitting.</li>
<li>The second example is because Age and other factors increase probability, and eventually there's a lot of probability.
These are explanations of individual projections.</li>
</ul>
<p>Shapley values can be combined into global explanations. If we run SHAP for each instance, you will get the Shapley value matrix. This matrix has a row for each data instance and a column for each feature. We can interpret the entire model by analysing the Shapley values in this matrix.</p>
<h3>SHAP Characteristic Importance</h3>
<p>The idea behind the importance of the SHAP feature is straightforward: the larger Shapley absolute value is more important. Because of the global importance we need, the absolute Shapley value for each feature is averaged in the data:</p>
<p>&#36;&#36;
I_j=\sum_{i=1}^n|\phi_j^{(i)}|
&#36;&#36;</p>
<p>We can map the importance of the SHAP feature.
<img src="/assets/images/shap/%E5%8F%AF%E8%A7%A3%E9%87%8A%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-23.png" alt=""></p>
<p>SHAP Characteristic Importance is an alternative to the importance of changing features. There are significant differences between the two measures of importance: the importance of the replacement feature is based on the decline in model performance. SHAP is based on the size of the characteristic attribution.</p>
<h3>SHAP Summary Chart</h3>
<p>The feature importance map is useful but does not contain information other than materiality. To obtain more information, use summary charts</p>
<p>The summary chart combines the importance of the features with the effects of the features. Each point in the summary is a feature and an example of a Shapley value. The position on the y-axis is determined by the feature and the position on the x-axis by the Shapley value. These characteristics are sorted according to their importance.
<img src="/assets/images/shap/%E5%8F%AF%E8%A7%A3%E9%87%8A%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-24.png" alt=""></p>
<h3>SHAP dependency diagram</h3>
<p>In the summary, we first see the relationship between the value of the feature and the impact on the projections. But to know the exact form of this relationship, we have to look at the SAP dependency map.</p>
<p>SHAP Character Dependence is probably the simplest global interpretation: mathematically, he's just the following scattered point. Figure</p>
<p>&#36;&#36;
{(x_{j}^{(i)},\phi_{j}^{(i)})}_{i=1}^{n}
&#36;&#36;</p>
<p>As shown in the figure below:
<img src="/assets/images/shap/%E5%8F%AF%E8%A7%A3%E9%87%8A%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-25.png" alt="">
As can be seen, the increase in HCyears increases the SAP value; it increases the probability of disease significantly.</p>
<p>SHAP Dependency Charts are an alternative method that relies partly on the charts and cumulative local effect maps, not on average but on the assessment of the global effect. SHAP&#36;y&#36;The fragmentation of the axis tends to imply the existence of interaction.</p>
<h3>SHAP Interactive Value</h3>
<p>The interaction effect is the additional combination of characteristics after considering the individual characteristic effects. Shapley Interactive Index Definition in the Game:</p>
<p>&#36;&#36;
\phi_{i,j}=\sum\limits_{S\subseteq\setminus{i,j&#125;&#125;\frac{|S|!(M-|S|-2)!}{2(M-1)!}\delta_{ij}(S)
&#36;&#36;</p>
<p>The SHAP dependency figure that takes into account an interactive feature is
<img src="/assets/images/shap/%E5%8F%AF%E8%A7%A3%E9%87%8A%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-26.png" alt="">
High STDs lowers the last SHAP value</p>
<h2>Depth extension of SHAP: black box decomposition of the nervous network</h2>
<p>When processing table data, TreeshaP solves the problem of interpretation of tree models represented by LightGBM or CatBoost, by their time complexity at multiple levels. But in the real algorithm evolution, when we face deep learning structures that contain dense and thin features (e.g. DeepFM, Twin Towers or Large Language Models), Treeshap is powerless, and KernelSHAP ' s calculation costs are expected to trigger a significant explosion as the depth of the nervous network increases.</p>
<p>To address the attribution of the characteristics of the deep learning model, the academic community has extended two sets of specialized approximation frameworks:</p>
<h3>DeepSHAP: A trans-chain based on reverse transmission</h3>
<p>DeepSHAP is a suitable version of DeepLift algorithms. It's about:<strong>The contributions are distributed by using the Backpropagation mechanism of the nervous network, which bypasses violence.</strong></p>
<ul>
<li><strong>The rationale.</strong>: DeepSHAP requires a " Background input " and an " Actual input " . When the data is moving through the network, the margin of each neuroactive value is recorded (in the case of a single neuron)&#36;\Delta y&#36;) and the input margin (&#36;\Delta x&#36;）。</li>
<li><strong>Linear approximation</strong>: For non-linear activation functions (e. g. ReLU or Sigmoid), DeepSHAP calculates a 'multiplier 'by linear interpolation values &#36;m_{\Delta x \rightarrow \Delta y} = \frac{\Delta y}{\Delta x}&#36;。</li>
<li><strong>Chain Law Reverse</strong>: From the output layer onwards, the variation in the output projection is transmitted in reverse by the lateral transfer using the revised chain rule until the input layer. The gradient that eventually falls on each input feature accumulates is the approximation of the Shapley value of that feature.</li>
</ul>
<h3>Gradient SHAP: aspirationalization of the gradient</h3>
<p>GradientSHAP combines the theory of game between the Integrated Gradients and SHAP, especially for the continuous Embeding vector space.</p>
<ul>
<li><p><strong>Maths Essentials</strong>: It assumes continuous change in characteristics. Calculates the gradient of the model output relative to the input on these points by adding multiple random linear interpolation values to the Gaussian Noise between the background distribution and the current input.</p>
</li>
<li><p><strong>Formula Thinking</strong>: The feature's SHAP value is approximated as its fraction of the gradient above the plug-in path:</p>
<p>&#36;&#36;
\phi_j \approx (x_j - x&#39;_j) \int_0^1 \frac{\partial f(x&#39; + \alpha(x - x&#39;))}{\partial x_j} d\alpha
&#36;&#36;</p>
</li>
<li><p><strong>Apply scene</strong>And when you need to explain what the high-dimensional Embeding captured, or when smoothing the sharp gradients that are generated by models that are not linear, GradientSHAP is a very powerful theoretical weapon.</p>
</li>
</ul>
<h2>Project Guide for SHAP</h2>
<p>In real-life industrial applications, SHAP is by no means just a tool for generating reporting charts after model training has ended. It can be embedded in the pre-frontal feature engineering conduit as a data-processing tool.</p>
<h3>Compression of Background Data (K-Means approximation)</h3>
<p>The edge expectations in the formula require us to enter a background data set when calculating KernelSHAP or DeepSHAP.</p>
<ul>
<li><p><strong>A trap.</strong>: direct input of hundreds of thousands of lines of training as background data <code>shap.DeepExplainer</code>I'm sorry. This would allow the model to calculate expectations hundreds of thousands of forward reasoning for each sample, which would immediately cause the Out Of Memoory (OOM) collapse.</p>
</li>
<li><p><strong>Engineering</strong>: At the code level, the background data must be down-dipped. The standard practice is to use K-Means cluster, which consolidates large-scale training into dozens of representative physiocentrics.</p>
<p>This, while retaining the overall margin distribution of the data, reduces the complexity of the calculation by several orders of magnitude, which is the engineering compromise available under SHAP for big data. And this background data itself is at the heart of our study of the exact interpretation of comparison.</p>
</li>
</ul>
<h3>Insisting massive data cleansing</h3>
<p>A solid data base is the lifeline of subsequent model fine-tuning or enhanced learning. SHAP is an extremely efficient abnormality check.</p>
<ul>
<li><p><strong>Data Leakage Detection</strong>：</p>
<p>In the SHAP Summary (Summary Plot), if a characteristic that should be flat, its SHAP absolute value is a leading example of a cliff breaker and perfectly dominates the classification. This usually means that data leaks occur (e.g., inadvertently characterizing the “last login time” to predict whether there is a “loss”.</p>
</li>
<li><p><strong>Offset Points and Scanning of Dirty Data</strong>：</p>
<p>In the SHAP dependency diagram, the normal feature distribution creates a clear non-linear curve or step leap. If an isolated dispersing point is present at the extreme position of the X axis and its SHAP effect (Y axis) is completely contrary to business commons, this strongly suggests that the characteristics of the sample are dirty data at the pre-processing or fusion stage. It is important at this point to return immediately to the early large-scale data cleansing logic to add cut-off or filter rules.</p>
</li>
</ul>
<h3>Dealing with co-linear engineering compromises (feature grouping)</h3>
<p>To address the " character relevance trap " , asymmetrical Shapley values at the purely academic level are too costly to calculate when highly relevant features (such as multiple dimensions of user activity indicators) have to be fed to the model.</p>
<p><strong>Engineering</strong>: At the interpretation stage, not a single SHAP value for a single feature, but rather a " Feature Group " wrapping of the strong competitor linear feature at the code level. Only the contribution of this Group as a whole Shapley value value is calculated. This would prevent the misallocation of importance and ensure that the global added value of justice is not undermined.</p>
<h2>References</h2>
<ul>
<li>Christoph Molnar, <em>Interpretable Machine Learning: A Guide for Making Black Box Models Explainable</em>, 3rd ed., 2025. <a href="https://christophm.github.io/interpretable-ml-book/cite.html">Official Page</a></li>
<li>Christoph Molnar, “SHAP” chapter, <em>Interpretable Machine Learning</em>. <a href="https://christophm.github.io/interpretable-ml-book/shap.html">Chapter Link</a></li>
<li>Scott M. Lundberg and Su-In Lee, <em>A Unified Approach to Interpreting Model Predictions</em>, NeurIPS 2017. <a href="https://papers.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions">NeuIPS Page</a></li>
<li>Lloyd S. Shapley, <em>A Value for n-Person Games</em>, 1953.</li>
</ul>
