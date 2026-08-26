---
title: Benford's Law and Statistical Fraud Detection
title_zh: 数字不会撒谎，但撒谎的人会编数字：从本福特定律聊聊统计造假识别
date: 2026-02-04 23:00:00 +0800
categories:
- Data Science
- Statistical Thinking
tags:
- Statistical Thinking
author: Hyacehila
mathjax: true
excerpt: When data looks too perfect, it may have drifted away from reality. Starting from Benford's Law, this post reveals
  statistical fingerprints hidden in fabricated data.
description: When data looks too perfect, it may have drifted away from reality. Starting from Benford's Law, this post reveals
  statistical fingerprints hidden in fabricated data.
excerpt_zh: 数据过于平滑或数字分布异常时，可能值得进一步检查。本文介绍本福特定律、末位数字、方差、相关结构和门槛聚束等检测思路。
permalink: /blog/2026/02/04/benfords-law-and-statistical-fraud/
lang: en
translation_key: 2026-02-04-benfords-law-and-statistical-fraud
translation_status: machine
translation_source_hash: 184ec5cf02acf6f07ca8b32a9a158870c8ed79a967a216c87112800edcf3ae9e
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Introduction: Why is humans not good at forging random?</h2>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/01/10/statistics-and-truth/">Statistics and Truth: How to use the accident (Statistics and Truth)</a>、<a href="/en/blog/2026/01/14/anscombes-quartet/">Anscom Quartet: Visualized power and statistical illusion</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>If you write the next "random" number right now, like 20 times a coin, you might write something like that. <code>HTHHTHTH...</code> Such a sequence. You probably would have been unconsciously avoiding writing. <code>HHHHHH</code> This is a continuous combination, because the instinct tells you, "It doesn't look random."</p>
<p>But randomly, it doesn't matter if it looks random. In real random processes, it is not only possible but almost inevitable that a long series of repetitions will occur when the sample is sufficiently large.</p>
<p>In order to make the data natural, the fraudster tends to proactively avoid continuous repetition, extreme values and irregular fluctuations, making the results too consistent with the perception of “average” and “random”. These modifications may leave an anomaly in the distribution of numbers. Several common methods of inspection are described below.</p>
<h2>The Law of First Laws</h2>
<h3>Rationale: Why is it "1"?</h3>
<p>In our intuition, if a pile of data is randomly distributed, the first number is one to nine, the probability should be the same, each. &#36;\approx 11.1%&#36;。</p>
<p>But Simon Newcomb found an anti-intuitive phenomenon in a logarithmic table that was frequently viewed in the nineteenth century: numbers starting with 1 appear much more frequently than others. Later, the physicist Frank Benford had a more systematic validation of the law.</p>
<p><strong>Benefluents of the Order.</strong>It was noted that the first of many data sets (e.g. accounting, demographics, physical constants) that were naturally formed &#36;d&#36; (&#36;d \in {1, \dots, 9}&#36;) The probability of occurrence follows the logarithmic distribution:</p>
<p>&#36;&#36; P(d) = \log_{10} \left( 1 + \frac{1}{d} \right) &#36;&#36;</p>
<p>The corresponding probability is that:</p>
<ul>
<li><strong>1 Probability at beginning</strong>：&#36;\approx 30.1%&#36;</li>
<li><strong>2 Probability at the beginning</strong>：&#36;\approx 17.6%&#36;</li>
<li>...</li>
<li><strong>9 Probability at beginning</strong>: Only &#36;\approx 4.6%&#36;</li>
</ul>
<p>A visual explanation is derived from the growth process across the orders of magnitude. A country's population has to double from 1 million to 2 million, and from 9 million to 10 million, it needs to grow by about 11 per cent. In this category, values stay longer in the beginning of the first "1" period. However, not all data satisfy the specific law of Benfu, and the way in which the data are generated and the range of values to be taken is still to be checked before they are used.</p>
<h3>Application and case studies</h3>
<p>This pattern is often used for financial audits and election fraud detection. When the fabricator makes the data, it is often the first number that is evenly distributed for “scrutinizing”, resulting in a low frequency of 1 and a high frequency of 9.</p>
<p>Enron ' s financial fraud cases are often used to discuss such methods. An ex post analysis of each share of the proceeds and other financial data disclosed by it was performed with a Benfo-specific test and a deviation from the first numerical distribution and theoretical values was observed. Such deviations provide a trail for audits, but cannot be independently substantiated for falsification of data, and need to be investigated in conjunction with accounts, transactions and business processes.</p>
<p>Another example that is often discussed is that <strong>2009 Iranian presidential election</strong>I'm sorry. After the election was fraudulently challenged, the statisticians Walter R. Mebane, Jr. conducted a Benfu-specific second-order test of the votes obtained in the open area (2 BL test). The analysis found that there was an anomaly in the distribution of votes and the accumulation of final numbers in some constituencies. These results support further verification of electoral data, but the distribution of figures cannot in itself be a substitute for ballot auditing and evidence of the electoral process.</p>
<h3>Statistical test methods: calonian proposed eugenicity test</h3>
<p>It's not just the eye, but the use. <strong>Carpside Probability Test (Ch-Square Goodness of Fit Test)</strong>。</p>
<ul>
<li><strong>zero scenario (&#36;H_0&#36;)</strong>: The first digital distribution of data is consistent with the Benefu-specific law.</li>
<li><strong>Alternative scenario (&#36;H_1&#36;)</strong>: The first numerical distribution of data does not conform to the Benfu law.</li>
</ul>
<p>Calculate statistics &#36;\chi^2&#36;：</p>
<p>&#36;&#36; \chi^2 = \sum_{i=1}^{9} \frac{(O_i - E_i)^2}{E_i} &#36;&#36;</p>
<p>of which &#36;O_i&#36; It's the frequency observed.&#36;E_i&#36; The expected frequency is calculated on the basis of the Benefig-specific law. Calculate &#36;p&#36; After value if &#36;p &lt; 0.05 (or more stringent threshold) we have reason to reject the zero assumption and suspect that the data is abnormal.</p>
<h2>Late Number Analysis (Last Digit Analysis)</h2>
<h3>Principle: Human perception of random intuitive deviation</h3>
<p>If the first figure is the pattern of “natural growth”, the last figure is the psychology of “man-made intervention”.</p>
<p>In the measurement or counting data, the last number (0-9) should normally be<strong>Uniform Distribution</strong> And the probability of each number is about 10 percent.</p>
<p>Two mistakes the fraudster makes:</p>
<ol>
<li><strong>Avoidance of duplication</strong>: In human subconscious, <code>88</code>、<code>99</code>、<code>11</code> Such figures are too false and therefore deliberately avoid them when they are made up. Even in multiple numbers, the adjacent figures are deliberately different.</li>
<li><strong>It's a good idea.</strong>: For the sake of economy or psychological comfort, fake data <code>0</code> and <code>5</code> The frequency of occurrence is often abnormally high (heavy effect).</li>
</ol>
<h3>Application and case studies</h3>
<p>Here.<strong>Supermarket sales</strong>or<strong>Height records.</strong>is more common. If the percentage of data ending in 0 or 5 (e.g. 170 cm, 175 cm) is abnormally high in a height record, the record should first be checked for clean-up, estimation or manual entry. The distribution of tails alone does not determine whether the data are false.</p>
<p>In the analysis of the elections in Iran, statisticians also checked the last two figures of the total number of votes cast. The frequency of some of the numerical combinations is higher than random expectations, which may be related to manual filling or other data generation mechanisms, and needs to be continued in conjunction with the electoral process.</p>
<h3>Statistical testing methods: evenness tests</h3>
<p>This step can also be used for a calibration, but for a balanced distribution.</p>
<ul>
<li><strong>zero scenario (&#36;H_0&#36;)</strong>: equal probability of the last number (0-9) (i.e. &#36;P = 0.1&#36;）。</li>
<li><strong>Statistics</strong>: Same calculation &#36;\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}&#36;But here. &#36;E_i&#36; It's all the total sample. &#36;1/10&#36;。</li>
</ul>
<p>It's also possible to introduce <strong>Runs Test</strong> To check the randomity of the numerical sequence and determine whether there is an anomaly in the serial relevance that is caused by “intentionally avoiding duplication”.</p>
<h2>Too Good To Be True</h2>
<h3>Rationale: Differences and fluctuations</h3>
<p>Observation data such as stock prices, temperature and experimental measurements are usually randomly volatile. The size of the fluctuations depends on the object, the method of measurement and the sampling process, and the data cannot be judged only on the basis of the smoothness of the curve.</p>
<p>The forger may interpret “good data” as “pretty data”, erase fluctuations in fabrication or embellishment, smooth the curve too much or over-compatible the theory of the variable relationship. The lack of the desired variance (Lack of Variance) is therefore a signal worth checking, but the determination of “how much difference” must rely on specific models and cross-reference data.</p>
<h3>Application and case studies</h3>
<p><strong>Madoff Ponzi Scheme.</strong> It is a frequent case. Bernard Madoff reported that investment returns remained exceptionally stable over the long term and that there was little significant rebound when markets fluctuated. The Quantification Analyst Harry Markopolos therefore questioned whether his strategy could be realized. The volatility analysis did not single out the fraud, but it pointed to an inexplicable inconsistency between the public performance and the stated trading strategy.</p>
<p>Macroeconomic data can also be cross-checked through relevant indicators. For example, the “gang index” is based on the idea of looking at GDP together with indicators such as industrial electricity, rail freight and bank balance for medium- and long-term loans. If reported GDP growth is long-term deviation from the relevant physical indicators, this<strong>Relevance Break</strong>This is worth further explanation. However, the relationship between indicators is influenced by industry structures and statistical calibres, and deviations cannot themselves directly prove that data are false.</p>
<h3>Statistical testing methods: differential tests and relevance</h3>
<p>We're looking at data for such fraud.<strong>Volatility</strong>and<strong>Multi-dimensional related structures</strong>。</p>
<p>First of all, we can use it. <strong>F-test (F-test)</strong> Comparison of target data (%2)&#36;S_1^2&#36;) with baseline data ( )&#36;S_2^2&#36;) The difference. By Calculator &#36;&#36; F = \frac{S_1^2}{S_2^2} &#36;&#36;If the F value is significantly less than 1, indicate that the target data are subject to a lower rate of volatility than the benchmark. This is an unusual signal to be investigated and whether it is artificially smoothed and judged in conjunction with data sources.</p>
<p>And then...<strong>Structure Consistency and Disability Analysis (Structural Consortium) &amp; Residual Analysis)</strong>I'm sorry. For time series such as GDP and electricity use, it is possible to use <strong>Cointegration Analysis</strong> Distinguishing between “false return” and long-term association. Common metaphors are drunks and dogs he's holding: both can move randomly, but distance does not increase indefinitely. In time series, although the two variables are not stable, a linear combination of them may be stable.</p>
<p>In the test, you can. <strong>Cannot initialise Evolution's mail component.</strong> Check for long-term balanced relationships. If the two historically harmonized indicators begin to deviate, the potential for a widening of the gap and its unstable appearance suggests a change in relationship. Changes may come from statistical calibres, economic structures, external shocks or human intervention, and therefore structural mutations and background information are also needed to determine the causes.</p>
<h2>Thrust at the threshold (Bunching / Threshold Effects)</h2>
<h3>Principle: Avoiding harm by profit</h3>
<p>When there is some sort of appraisal indicator, tax threshold or academic publication standard (e.g. &#36;p) &lt; At 0.05.00, data tend to be distorted near the threshold. It's called <strong>Clock effect (Bunching)</strong>。</p>
<h3>Application and case studies</h3>
<p>This can be seen in academic publications and tax returns. When analysing the distribution of P values for published papers, researchers have observed that values are concentrated in rapid declines before the significant threshold of 0.05. This may be related to selective reports or P-hattering. Similarly, if significant amounts of declared revenue are concentrated below the tax threshold, possible reasons such as disclosure rules, integrity behaviour and tax avoidance incentives need to be examined.</p>
<h3>Statistical testing methods: Visualization and McRay density tests (McCray Density Test)</h3>
<p>The threshold effect can be identified first.<strong>Visualise</strong>Start. A fine histogram shows whether the data are at a certain threshold (e.g., &#36;p=0.05&#36; The government has also been able to provide information on the situation in the country, and has been able to provide information on the situation. Such graphics are a clue to further testing, not a direct proof of man-made manipulation.</p>
<p>We use it for statistical validation. <strong>MacCray Density Test (McCray Density Test)</strong>I'm sorry. This is a test commonly used in the Breakpoint Return Design (RDD) in the sense that the probability density function of the variable is continuous at the breakpoint. If the left density of the threshold is significantly higher than the right and the difference is statistically significant (even if random fluctuations are taken into account), we have reason to reject the continuity assumption and to assume that there is artificial manipulation.</p>
<h2>Conclusion: anomalies are only the starting point of the investigation</h2>
<p>Benfoux-specific laws do not apply to all data, such as heights with limited range of values, lottery numbers and fixed-priced goods. The end figures, differences, related structures and thresholds are also subject to their respective conditions.</p>
<p>The effect of these methods is to identify anomalies that need to be explained and to help auditors decide what to examine next. They cannot be characterized separately from the data generation process, business context and other evidence.</p>
<p>When looking at too smooth a growth curve or an abnormally even digital distribution, one more step can be asked: can the existing data generation process explain this shape?</p>
