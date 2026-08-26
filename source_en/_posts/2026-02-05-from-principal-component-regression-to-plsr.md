---
title: From Principal Component Regression (PCR) to Partial Least Squares (PLS)
title_zh: 从主成分回归 (PCR) 到偏最小二乘 (PLS)
date: 2026-02-05 12:00:00 +0800
categories:
- Data Science
- Statistical Modeling & Inference
tags:
- Linear Models
- Dimensionality Reduction
author: Hyacehila
excerpt: PCR and PLSR are common dimensionality-reduction regression methods under multicollinearity. This post derives their
  logic and explains how PLSR brings in response relevance.
description: PCR and PLSR are common dimensionality-reduction regression methods under multicollinearity. This post derives
  their logic and explains how PLSR brings in response relevance.
excerpt_zh: 当数据存在多重共线性时，PCR 和 PLSR 都是常用的降维回归方法。本文详细推导了 PCR 与 PLSR 的数学原理，分析了 PCR “只看 X 不看 Y” 的潜在缺陷，并直观解释了 PLSR 如何通过引入因变量相关性来解决这一问题。
permalink: /blog/2026/02/05/from-principal-component-regression-to-plsr/
lang: en
translation_key: 2026-02-05-from-principal-component-regression-to-plsr
translation_status: machine
translation_source_hash: 25fc787bcaef1322bc2adcac5164e8f20b8ee37cad2ff40372901b1e6b4e7fc0
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<blockquote>
<p>The core views and parts of this paper are based on the post of the Fau Fumi:<a href="https://yihui.org/cn/2008/09/principle-component-regression-and-partial-least-square-regression/">Return of the main ingredient to the lowest 2x2</a>。</p>
</blockquote>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base: linear model, minimum 2x2 estimate and regression diagnosis</a>、<a href="/en/blog/2025/09/23/advanced-linear-regression-notes/">Linear regression step: proposed alignment, model selection and co-line Sex</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h1>Returns from main ingredient (PCR) to minus 2 (PLS)</h1>
<p>In Multilinear Regression, MLR, the minimum binary is the commonly used parameter estimation method. However, when data is centralized<strong>Multicollineability</strong>or the number of variables exceeds the sample (&#36;p) &gt; n&#36;）时，OLS 估计量会变得不稳定，甚至无法计算（因为设计矩阵 &#36;X^TX&#36; is irreversible or close to extraordinary.</p>
<p>To address this problem, the concept of Democratization is common. **Regression of main constituents (PCR)<strong>and</strong>The minimal 2x2 regression (PLSR)** is the two representatives of this approach. They all return by building a new " Potential Variables " instead of the original variables, but the logic of both building the potential variables is different and should therefore be used with caution.</p>
<h2>1. Primary Component Return (Prircipal Component Regresion, PCR)</h2>
<h3>1.1 The mathematical rationale</h3>
<p>The PCR can be summarized as follows:<strong>First, the PAA, then the OLS.</strong>。</p>
<p>Suppose we have a centralized variable matrix. &#36;X \in \mathbb{R}^{n \times p}&#36; and cause variables &#36;Y \in \mathbb{R}^{n \times 1}&#36;。</p>
<ol>
<li><p><strong>Spectrolysis</strong>: Yes. &#36;X&#36; Disaggregation of odd-value (SVD) or coordination matrix &#36;X^TX&#36; Conducting characterization.
&#36;&#36; X = U \Sigma V^T &#36;&#36;
of which &#36;V&#36; , the main ingredient.</p>
</li>
<li><p><strong>Construct the main ingredient</strong>: Scores&#36;Z&#36;。
&#36;&#36; Z = XV &#36;&#36;
Because the main ingredient is active, that is, &#36;Z^TZ&#36; It's an angle matrix, which eliminates the problem of multiple co-linearity.</p>
</li>
<li><p><strong>Cut and return</strong>: Usually we just take the front. &#36;k&#36; Main component of the maximum characteristic value (most variance of explanation) &#36;Z_k&#36;I'm sorry. Do it. &#36;Y&#36; About &#36;Z_k&#36; Minimum 2x2 returns:
&#36;&#36; \hat{Y} = Z_k \hat{\gamma} &#36;&#36;
&#36;&#36; \hat{\gamma} = (Z_k^T Z_k)^{-1} Z_k^T Y &#36;&#36;</p>
</li>
<li><p><strong>Revert Parameters</strong>: Map the regression coefficient back to the original space:
&#36;&#36; \hat{\beta}_{PCR} = V_k \hat{\gamma} &#36;&#36;</p>
</li>
</ol>
<h3>1.2 Methodological deficiencies</h3>
<p>PCR sounds tempting: it's eliminated the cosmopolitanity, it's preserved. &#36;X&#36; . But it has a logical fatal wound:</p>
<blockquote>
<p><strong>The process of extracting the main ingredient is based only on the variable &#36;X&#36; The co-ordinated structure, completely ignoring the variables. &#36;Y&#36;。</strong></p>
</blockquote>
<p>In the PCA, we select the main ingredient by "maximizing the variance". But...<strong>The difference doesn't mean we're going to be together. &#36;Y&#36; Relevant</strong>I'm sorry. There may be a situation where:&#36;X&#36; The direction is very small (and therefore discarded in PCR), but the direction contains &#36;Y&#36; Most of the information.</p>
<p>As Ali S. Hadi and Robert F. Ling (1998) are <em>The American Statistician</em> As noted above, if the main components of the explanation of the variable are not related to the response variable, the PCR may not even be as effective as the discarding of the variable. The article gives an example: the former p-1 PCs have nothing to do with variables, while the last PC explains all variations due to variables. The reason is that the PCA relies only on X ' s co-conforming structure, while ignoring Y ' s information.</p>
<h2>2. Minimum 2x2 regression (Partial List Review, PLSR)</h2>
<p>In order to solve the problem of PCR “Only X does not look Y”, the lowest 2-fold return (PLSR) was created. Its core idea is:<strong>In searching for potential variables, let it explain as much as possible &#36;X&#36; And the mutation of it makes it as explained as possible. &#36;Y&#36; ..the variation of</strong>。</p>
<h3>2.1 Optimizing objectives</h3>
<p>Suppose we're looking for a weight vector. &#36;w&#36;(Fulfilled) &#36;|w|=1&#36;) , construct potential variables &#36;t = Xw&#36;I'm sorry. The target function for PLSR is maximized &#36;t&#36; and &#36;Y&#36; Other Organiser</p>
<p>&#36;&#36; \max_{w} \text{Cov}(Xw, Y)^2 = \max_{w} \text{Var}(Xw) \cdot \text{Corr}(Xw, Y)^2 \cdot \text{Var}(Y) &#36;&#36;</p>
<p>PCR and PLSR:</p>
<ul>
<li><strong>PCR</strong>: Maximize &#36;\text{Var}(Xw)&#36;。</li>
<li><strong>PLSR</strong>: Maximize &#36;\text{Var}(Xw) \times \text{Corr}(Xw, Y)^2&#36;I'm sorry. (Note:&#36;\text{Var}(Y)&#36; (A constant)</li>
</ul>
<p>So this is more intuitive: the PLSR is trying to find a balance that it's found to contain both. &#36;X&#36; Main structure (large difference) and &#36;Y&#36; Highly relevant. PLSR brought <strong>The main change is that the extraction of the ingredient is based not only on variance but also on the correlation with the variable.</strong>。</p>
<h3>2.2 Summary of algorithms (NIPALS thought)</h3>
<p>PLSR solvers usually use NIPALS algorithms or SIMPLS algorithms. For the single variable &#36;Y&#36;The iterative process is broadly as follows:</p>
<ol>
<li>Calculate &#36;X&#36; and &#36;Y&#36; Accompany vector &#36;w = X^T Y&#36;。</li>
<li>Normalization &#36;w \leftarrow w / |w|&#36;。</li>
<li>Calculate score vector (Score Victor)&#36;t = Xw&#36;。</li>
<li>Calculate &#36;Y&#36; Yeah. &#36;t&#36; the load of the &#36;c = Y^T t / (t^T t)&#36;。</li>
<li>Calculate &#36;X&#36; Yeah. &#36;t&#36; the load of the &#36;p = X^T t / (t^T t)&#36;。</li>
<li><strong>Detachment</strong>: From &#36;X&#36; and &#36;Y&#36; minus the explanation of this ingredient:
&#36;&#36; X_{new} = X - t p^T &#36;&#36;
&#36;&#36; Y_{new} = Y - t c^T &#36;&#36;</li>
<li>The above steps are repeated using the disability matrix until sufficient ingredients are extracted.</li>
</ol>
<p>Ultimately, we have modeled in the form of:
&#36;&#36; X = TP^T + E &#36;&#36;
&#36;&#36; Y = TQ^T + F &#36;&#36;
(of which) &#36;Q&#36; Often directly associated with regression coefficients).</p>
<h3>2.3 Geometric interpretation</h3>
<ul>
<li><strong>OLS</strong> The search is for &#36;X&#36; Column Space Distance &#36;Y&#36; Recent projection.</li>
<li><strong>PCR</strong> First, find the largest square subspace in this space, then project it in the subspace.</li>
<li><strong>PLSR</strong> It's a rotational axis, which points to the longest-spreading of data, while the new axis is biased towards the same. &#36;Y&#36; The largest direction of the gradient.</li>
</ul>
<h2>3. Summary and comparison</h2>
<table>
<thead>
<tr>
<th align="left">Dimensions</th>
<th align="left">Return of main ingredient (PCR)</th>
<th align="left">Minimal 2x2 regression (PLSR)</th>
</tr>
</thead>
<tbody><tr>
<td align="left"><strong>Basis for extracting constituents</strong></td>
<td align="left">Only rely &#36;X&#36; Difference (&#36;\text{Var}(X)&#36;)</td>
<td align="left">Consider &#36;X&#36; Differences &#36;X,Y&#36; Relevance (&#36;\text{Cov}(X,Y)&#36;)</td>
</tr>
<tr>
<td align="left"><strong>Monitoring learning</strong></td>
<td align="left">No (no monitoring of first step)</td>
<td align="left">Yes (using target variable Y)</td>
</tr>
<tr>
<td align="left"><strong>Variable Selection</strong></td>
<td align="left">It's actually a hard cut.</td>
<td align="left">It's equivalent to a soft weight.</td>
</tr>
<tr>
<td align="left"><strong>Apply scene</strong></td>
<td align="left">The noise is mostly in the &#36;X&#36; and &#36;Y&#36; Association &#36;X&#36; ..when the main variation is</td>
<td align="left">The prediction is directed,&#36;X&#36; When multiple co-linears exist internally</td>
</tr>
</tbody></table>
<p>In most practical applications (especially chemical metrology, spectrometry) PLSRs tend to perform better or evenly than PCRs. By introducing information on variables, PLSRs often achieve the same prediction precision with fewer components, thus obtaining a simpler model (Parsimonious).</p>
<p>We suggest PLSR as a replacement for PCR for a straightforward reason: the former is statistically more superior, and it uses information from X and Y instead of using its own coordination structure only. PCR has the advantage of being able to deal with multiple co-linearity, but statistically, the statistical nature of the PCR is not reliable.</p>
<p>When actually used, PCRs are not always easily susceptible to the situations described above (the former p-1 ingredient is not related to Y, the last ingredient). In many cases (especially chemical data) the main ingredient can explain some of the built-in structures of the variables, which are often associated with the variables. In theory, however, PCR lacks the logic of return. PLSR retains the PCR's “goods” (the principle being similar) while taking account of the variables, which are more consistent with the purpose of return.</p>
