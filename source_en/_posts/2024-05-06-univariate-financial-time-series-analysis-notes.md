---
title: 'Univariate Financial Time Series Analysis: Asset Returns, ARCH/GARCH Effects, and Volatility Modeling'
title_zh: 金融时间序列分析：ARCH/GARCH 效应与波动率建模
date: 2024-05-06 23:07:47 +0800
categories:
- Data Science
- Time Series & Spatial Data
tags:
- Statistics
- Time Series
- Financial Time Series
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers asset returns, ARCH/GARCH effects, volatility modeling, and related model diagnostics.
description: Covers asset returns, ARCH/GARCH effects, volatility modeling, and related model diagnostics.
excerpt_zh: 整理资产收益率、ARCH/GARCH 效应、波动率建模和相关模型检验方法。
permalink: /blog/2024/05/06/univariate-financial-time-series-analysis-notes/
lang: en
translation_key: 2024-05-06-univariate-financial-time-series-analysis-notes
translation_status: machine
translation_source_hash: 4729ea9b1aa6bbae9d4b471c653e34594e5c748f9cfd303dd562279f4ff47e9f
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Financial data and their characteristics</h2>
<p>Financial time series analysis will be original.<a href="/en/blog/2024/01/30/linear-time-series-analysis-notes/">Linear Time Series Analysis</a>We're going to complement the financial time series background by expanding the way we learn from the original time series; and<a href="/en/blog/2025/09/27/multivariate-financial-time-series-analysis-notes/">Financial time series analysis (multiple)</a>Problems in introducing multiple time series analysis</p>
<h3>Asset return</h3>
<p>Let's count.&#36;P_t&#36;For the net value of assets at a given time.</p>
<h4>Simple rate of return</h4>
<p>Single period gross return:
&#36;&#36;1+R_t=\frac{P_t}{P_{t-1&#125;&#125;&#36;&#36;
Simple net return for a single period, simple rate of return:
&#36;&#36;R_t=\frac{P_t}{P_{t-1&#125;&#125;-1=\frac{P_t-P_{t-1&#125;&#125;{P_{t-1&#125;&#125;&#36;&#36;
&#36;k&#36;Simple gross rate of return for the period:
&#36;&#36;1+R_t[k]=\frac{P_t}{P_{t-k&#125;&#125;=\prod_{j=0}^{k-1}(1+R_{t-j}).&#36;&#36;
&#36;k&#36;Net return for the period:
&#36;&#36;R_t[k]=\frac{P_t}{P_{t-k&#125;&#125;-1=\frac{P_t-P_{t-k&#125;&#125;{P_{t-k&#125;&#125;&#36;&#36;</p>
<h4>Continuous compound rate of return</h4>
<p>The initial value of an asset is&#36;C&#36;, nominal annual interest rate is&#36;r&#36;, but split in one year&#36;m&#36;Second-rate, theoretically every time.&#36;C\frac rm&#36;, the final net asset value should read&#36;C+C\frac{r}{m}\times m=C(1+r);&#36; </p>
<p>However, because of the advance payment of interest, the advance payment of interest is added to the value of the account and, starting with the second payment, the interest paid exceeds&#36;C\frac rm&#36;That's why the net value after a year is higher.&#36;C(1+r)&#36;I don't know. Net value after one year
&#36;&#36;C\Big(1+\frac rm\Big)^m&#36;&#36;
When?&#36;m\to\infty&#36;, from limit&#36;\lim_x\to+\infty(1+\frac1x)^x=e&#36;You know,
&#36;&#36;\lim_{m\to\infty}C\Big(1+\frac{r}{m}\Big)^m=\lim_{m\to\infty}C\Big[\Big(1+\frac{r}{m}\Big)^{\frac{m}{r&#125;&#125;\Big]^r=Ce^r.&#36;&#36;
And then...&#36;r&#36;It's called continuous compound interest, which corresponds to a certain unit of time (generally year)</p>
<p>&#36;R=e^r-1&#36;It's a lump sum.&#36;r&#36;Corresponding actual interest rates,&#36;r&#36;and&#36;R&#36;The relationship is
&#36;&#36;R=e^r-1,\quad r=\ln(1+R)&#36;&#36;</p>
<h4>Assets portfolio rate of return</h4>
<p>Existing&#36;N&#36;item of assets&#36;t-1&#36;The current net value is
&#36;&#36;A_{p,t-1}=\sum_{j=1}^NA_{i,t-1}=A_{p,t-1}\sum_{j=1}^Nw_i&#36;&#36;
of which&#36;w_i=A_{i,t-1}/A_{p,t-1}&#36;No. No.&#36;i&#36;Weights of assets</p>
<p>And...
&#36; \begin{aligned}A p,t}&amp;=\sum_{j=1}^NA_{i,t-1}(1+R_{i,t})=A_{p,t-1}\sum_{j=1}^nw_i(1+R_{i,t})\&amp;=A_{p,t-1}\left(1+\sum_{j=1}^nw_iR_{i,t}\right)\end{aligned}&#36;&#36;
所以资产组合的简单收益率为
&#36;&#36;I'm sorry.
Attention.&#36;w_i&#36;No. No.&#36;t-1&#36;The weight of the moment. If we continue to calculate&#36;R_{p,t+1}&#36;, weight should be used&#36;t&#36;The weight of the moment.
Of course, if the proportion of assets does not change much, it will not change.&#36;{w_i}&#36;The approximation is possible.</p>
<p>There is no such simple formula for logarithmic rates of return. It looks like it's been there for hours.
&#36;&#36;r_{p,t}\approx\sum_{j=1}^nw_ir_{i,t}&#36;&#36;</p>
<h4>Payments of dividends and rates of return</h4>
<p>Right price.&#36;P_{t-1}&#36;A certain asset, if in&#36;t-1&#36;Present.&#36;t&#36;Pay each unit between them.&#36;D_t&#36;The dividend, then.&#36;t&#36;At the moment, the proceeds are
&#36;P_t-P_{t-1}+D_t&#36;, so then the rate of return should be calculated as
&#36;&#36;R_t=\frac{P_t-P_{t-1}+D_t}{P_{t-1&#125;&#125;,\quad r_t=\ln(P_t+D_t)-\ln P_{t-1}&#36;&#36;</p>
<h4>Excess rate of return</h4>
<p>&#36;&#36;Z_t=R_t-R_{0t},\quad z_t=r_t-r_{0t}&#36;&#36;
of which&#36;R_{0t}&#36;and&#36;r_{0t}&#36;It is the rate of return on a reference asset, such as the rate of return on United States short-term Treasury debt.</p>
<p>The excess rate of return is negative, the real rate of return is positive, and is still generally considered to be a deficit. Loss</p>
<h3>Bond rate of return</h3>
<h4>Bond type</h4>
<p>Investors buy bonds at market prices and collect cash at paper prices on maturity dates. The purchase price is lower than the nominal price. That's what makes it work.</p>
<p>Some of the bonds are also subject to periodic interest payments, calculated on the basis of book interest rates (coupon payment) and nominal amounts, for example, a nominal value of &#36;100 and a nominal interest rate of 6 per cent, or, if interest is distributed semi-annually, each interest rate&#36;100\times0.06/2=3&#36;Won. Some bonds are not intermediate-interested, and such bonds are called zero-interest bonds.</p>
<h4>Current rate of return</h4>
<p>The current rate of return calculates only the apparent annual gain, regardless of the time cost of the funds.</p>
<p>&#36;&#36;\text{当期收益率}=\frac{\text{每年派息额&#125;&#125;{\text{买入价格&#125;&#125;\times100%&#36;&#36;</p>
<h4>Due rate of return</h4>
<p>In the case of zero interest bonds, there is no interest income during the period in which they are held. If the purchase price is&#36;P&#36;, par value is&#36;F&#36;, hold&#36;k&#36;Rate of return due
&#36;&#36;\left(\dfrac{F}{P}\right)^{1/k}-1&#36;&#36;
This is called the mature rate of return (Yield To Maturity, YTM).</p>
<p>If there was interest on the holding period, the calculation would be more complicated, and each of the stages would be calculated separately to give an accurate result, but the simple rate of return would have to be considered directly.</p>
<h3>logarithmic rate of return</h3>
<p>A logarithmic return (Logarithmic Return) is a method commonly used in financial sciences to calculate the return on assets. It measures the percentage change in asset prices by taking natural logarithms. A logarithmic rate of return is usually used to calculate the return on financial assets such as equities, bonds and foreign exchange.
Define
&#36;&#36;r_t=\ln\left(\frac{P_t}{P_{t-1&#125;&#125;\right)&#36;&#36;
We use a logarithmic rate of return because...</p>
<ul>
<li><strong>Time added</strong>: The logarithmic rate of return can be added in time, i.e. the sum of the logarithmic rate of return for multiple consecutive periods equals the total logarithmic rate of return for those periods. This makes the logarithmic rate of return very appropriate for the calculation of long-term investment returns</li>
<li>The logarithmic rate of return reflects a continuous composite increase in asset prices, which is consistent with the compounding effect of financial assets</li>
<li>The logarithmic rate of return is not sensitive to extreme price changes, which help to reduce the impact of anomalies on the calculation of the rate of return.</li>
<li>We generally assume that<strong>logarithmic positive distribution</strong> But it has a thick tail.</li>
</ul>
<h3>Assets volatility</h3>
<p>The most important concern in the financial data, apart from asset prices and rates of return, is asset volatility. Asset volatility measures the risk of an asset, and volatility is a key factor in the pricing of options and asset allocation. Volatility rates play an important role in calculating VaR (risk values) in risk management. Some volatility indices have become financial instruments themselves.</p>
<h4>Characteristics of volatility</h4>
<p>Volatility refers to the degree of volatility in asset prices, similar to the notion of a standard deviation of random variables in probabilities. <strong>Volatility cannot be directly observed.</strong>, some of the characteristics of the volatility can be seen in the rate of return on assets such as</p>
<ul>
<li>Volatility concentration</li>
<li>Volatility rates change over time, with no jumpovers in the volatility rate</li>
<li>Volatility tends to change within a fixed range, which means that dynamic volatility is stable.</li>
<li>In the case of large increases in asset prices and sharp declines, volatility is not the same as in the case of large declines, which are generally higher, which is called leverage effect.
These properties are important for the presentation and improvement of the Volatility Models, and many of the new Volatility Models are based on the diagnosis that the original models do not reflect a particular pattern above.</li>
</ul>
<p>There are three sources of data for stocks.</p>
<ul>
<li>(a) The daily rate of return for each transaction date;</li>
<li>IBM options data</li>
<li>(a) Disaggregated data on transactions and quotations in the disk;
Three different fluctuations can be calculated:</li>
<li><strong>Standard deviation (or differential condition) as daily rate of return</strong> The model in this chapter is a definition of such volatility.</li>
<li>Implicit Volatility Rates: The resulting Volatility Rates are referred to as Implicit Volatility Rates, based on theoretical formulas such as BS formulas, which are derived from stock prices and options price data. Implied fluctuations tend to be larger than those obtained by modelling the daily rate of return. The CBOE VIX index is the implied volatility rate.</li>
<li>Real rate of fluctuations: the difference in the estimated rate of return for one day (variation rate square) using all yield data within one day, e.g., rate of return every five minutes. Realized volatility, RV</li>
</ul>
<p>Similar to interest rates, the range of time in which fluctuations are measured is usually one year, and the rate in which fluctuations are generally annualized. If you have a day rate, you can multiply it and convert it.&#36;\sqrt{252}&#36;Volatility of adultisation</p>
<h4>Structure of the volatility model</h4>
<p>We start with the one-dollar volatility model.<strong>The one-dollar volatility model is an attempt to portray rates of return as irrelevant or low-level, but not independent.</strong> <em>We find, for example, that a sequence has passed the white noise test, and the ACF reflects little relevance, but if it's absolute value (or square calculation), it can't pass the white noise test and the ACF has its own relevance.</em></p>
<p>Use&#36;F_{t-1}&#36;Other Organiser&#36;t-1&#36;Information on current rates of return, in particular the linear combination of these rates of return, to be examined&#36;r_t&#36;Yes.&#36;F_{t-1}&#36;average of conditions and difference of conditions:
&#36;&#36;\mu_t=E(r_t|F_{t-1}),\quad\sigma_t^2=\mathrm{Var}(r_t|F_{t-1})=E[(r_t-\mu_t)^2|F_{t-1}]&#36;&#36;
The experience gained through the analysis of the examples,&#36;{r_t}&#36;It's usually easier, like a smooth ARMA.&#36;(p,q)&#36;Sequence.</p>
<p>General logarithmic rate of return&#36;{r_t}&#36;It's an ARMA.&#36;(p,q)&#36;Models:
&#36;&#36;r_t=\phi_0+\sum_{j=1}^p\phi_jr_{t-j}+a_t+\sum_{j=1}^q\theta_ja_{t-j}&#36;&#36;
of which&#36;{a_t}&#36;For the unrelated white noise column,
&#36;&#36;\mu_t=E(r_t|F_{t-1})=\phi_0+\sum_{j=1}^p\phi_jr_{t-j}+\sum_{j=1}^q\theta_ja_{t-j}=r_t-a_t&#36;&#36;
Here we're assuming white noise.&#36;a_t=r_t-E(r_t|F_{t-1})&#36;Call this white noise column&#36;{a_t}&#36;To keep it steady.&#36;{r_t}&#36;New or disturbing column</p>
<p>&#36;r_t&#36;Decompose to (transfer)
&#36;&#36;r_t=\mu_t+a_t.&#36;&#36;
Models can be created if other explanatory variables (outside variables) are available&#36;r_t=\mu_t+a_t&#36;, where
&#36;&#36;\mu_t=\phi_0+\sum_{i=1}^k\beta_ix_{i,t-1}+\sum_{j=1}^p\phi_jy_{t-j}+\sum_{j=1}^q\theta_ja_{t-j}&#36;&#36;
of which&#36;x_{i,t-1}&#36;No. No.&#36;i&#36;An explanation variable is in&#36;t-1&#36;The value of the moment,&#36;y_{t-j}&#36;It's after removing the explanation variable.&#36;r_{t-j}&#36;value. (This is where the time series error was introduced in the regression.)<a href="/en/blog/2024/01/30/linear-time-series-analysis-notes/">Linear Time Series Analysis</a> Section on regression models with time series errors)</p>
<p>&#36;\mu_t&#36;Obeyable ARMA.&#36;(p,q)&#36;The steps are related to the frequency of sampling of the data, and the daily frequency data of the stock index tend to be less relevant and the monthly data may not have any significant correlation.</p>
<p>** Now we have a structure, a whole mind test: the original basic time series can be broken down into the sum of the average and new interest columns, which can be estimated using regression models with time series errors, and the new interest lines are irrelevant.</p>
<p><strong>We used to model time series.<a href="/en/blog/2024/01/30/linear-time-series-analysis-notes/">Linear Time Series Analysis</a>It's all a simulation of the mean bar. The new interest is considered a random error.</strong> </p>
<p>There's a combination of the structures ahead.
&#36;&#36;\sigma_t^2=\mathrm{Var}(r_t|F_{t-1})=\mathrm{Var}(a_t|F_{t-1})=E(a_t^2|F_{t-1}).&#36;&#36;
<strong>Here.&#36;\sigma_t&#36;It's the rate of volatility. It's the rate of return. Bad</strong>I don't know. If you assume the white noise in the model&#36;{a_t}&#36;is a separate sequence, then&#36;\sigma_t^2\equiv\sigma^2&#36; There is no possibility of modelling fluctuations. Here's the assumption.&#36;{a_t}&#36;Zero-average irrelevant smooth column, satisfactory&#36;E(a_t|F_{t-1})=0&#36;But not a separate sequence.</p>
<p>The problem with this chapter is yes.&#36;\sigma_t^2&#36;Modelling, this model is called the Conditional Differential Model. There are two types of condition variant models:</p>
<ul>
<li>Draw with a certain function&#36;\sigma_t^2&#36;The ARCH and GARCH models fall into this category;</li>
<li>Synchronising folder&#36;\sigma_t^2&#36;The random fluctuations (SV) model falls into this category.</li>
</ul>
<p>&#36;\mu_t&#36; The model is called&#36;r_t&#36;The average equation.&#36;\sigma_t^2&#36;The model is called&#36;r_t&#36;. The conditional variance model is right.&#36;r_t&#36;Average&#36;\mu_t&#36;On the basis of modelling, a model describing the difference in terms of return on assets over time was added. It's more precise.&#36;r_1,r_2,\ldots,r_t&#36;Back&#36;r_t+1&#36;(c) The distribution of the conditions subject to them.</p>
<h4>Steps to establish a volatility model</h4>
<p>Four steps are required to establish a volatility model for the asset yield sequence:</p>
<ol>
<li>The equation of the mean value item is created by testing the self-relevance of the sequence, and appropriate explanatory variables can be introduced if necessary;</li>
<li>It's a white noise test of the balance of the mean equation. <strong>After adoption, check the disability for ARCH effects Response</strong>；</li>
<li><strong>If ARCH results are significant, specify a volatility model</strong>, joint estimates of the mean equations and fluctuations;</li>
<li>Validation of the model obtained, improvement if needed.</li>
</ol>
<h2>ARCH Model</h2>
<h3>ARCH Effects Test</h3>
<p>Naturally, we know that we want to know if we need an ARCH model.</p>
<p>In order to test the ARCH effects, the average model is to be established.&#36;\mu_t&#36;, calculate the disability&#36;a_t=r_t-\mu_t&#36;I don't know. Use the square of the residual sequence&#36;{a_t^2}&#36;Conduct ARCH effects tests.</p>
<p>There are two tests. One is right.&#36;{a_t^2}&#36;The Ljug-Box white noise test did not have ARCH effects when the tests were not significant and had ARCH effects when the tests were significant.</p>
<p>Another test was proposed by R.F. Engle. Consider the following minimum two-fold problem:
&#36;&#36;a_t^2=\alpha_0+\alpha_1a_{t-1}^2+\cdots+\alpha_ma_{t-m}^2+e_t,:t=m+1,\ldots,T&#36;&#36;
of which&#36;T&#36;For the sample volume,&#36;m&#36;Is the right number of AR steps,&#36;e_t&#36;To return disability. zero
&#36;&#36;H_0::\alpha_1=\cdots=\alpha_m=0&#36;&#36;
Reject&#36;H_{0}&#36;There's an ARCH effect. It's called Engle's LaGrandian Sniper Test.
That's the OLS equation. <a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a> "Specificity of the regression equation" section</p>
<h3>ARCH Formula</h3>
<p>The ARCH model is the most basic model for modelling volatility.</p>
<ul>
<li>Disturbing sequence of asset yields&#36;a_t=r_t-E(r_t|F_{t-1})&#36;It is irrelevant, but not independent.</li>
<li>&#36;a_t&#36;The lack of independence, described as Var&#36;( r_t| F_{t- 1}) =&#36;Var&#36;( a_t| F_{t- 1})&#36; Yes.&#36;a_t^2&#36;is the linear combination of the values that are lagging.
of which&#36;F_t=\sigma({r_t,r_{t-1},\ldots})&#36;</li>
</ul>
<p>Specifically, ARCH()&#36;m)&#36;Model as
&#36; \begin{aligned}&amp;a_{t}=\sigma_{t}\varepsilon_{t},\&amp;== sync, corrected by elderman == @elder man &#36;
of which&#36;{\varepsilon_t}&#36;is the independent and distributed white noise of the unit difference of zero mean, &#36;\alpha 0&gt; 0&#36;,&#36;\alpha _j\geq 0, j= 1, 2, \ldots , m&#36;,另外&#36;{\alpha_j}&#36;还需要满足一些条件使得Var&#36;( a_t)&#36;有限，类似于AR&#36;(p) Characteristic root conditions for the smoothness of the &#36;-series.
In essence, this is one.&#36;AR(p)&#36; Sequence to estimate the difference
<strong>The first line reflects randomity and the second line reflects the implied certainty of volatility.</strong> ^03b43c</p>
<p>On the right side of the swing equation, only the deadline appears.&#36;t-1&#36;The moment. &#36;a_{t-1},\ldots,a_{t-m}&#36;So the ARCH model is a certain volatility model, which means&#36;\sigma_{t}^2&#36;About&#36;F_{t-1}&#36;It's measurable.&#36;t-1&#36;Time to determine the difference.&#36;\sigma_t^2&#36;value.</p>
<p>&#36;\varepsilon_{t}&#36;The distribution is often based on standard normal distribution, standardized t-distribution, broad error distribution (Generalized Error Distribution), and, in some cases, partial distribution.</p>
<p>If&#36;\varepsilon_t\sim N(0,1)&#36;Remember,&#36;\mu_t=E(r_t|F_{t-1})&#36;Think about it.&#36;p=1&#36;scenario,
&#36;&#36;r_t|F_{t-1}\sim\mathrm{N}(\mu_t,\sigma_t^2)=\mathrm{N}(\mu_t,\alpha_0+\alpha_1a_{t-1}^2).&#36;&#36;
So in ARCH model,&#36;\varepsilon_{t}&#36;The distribution is referred to as the "conditional distribution" where&#36;F_{t-1}&#36;Conditions&#36;r_t&#36;the type of condition distribution.</p>
<p>Because...&#36;a_t=r_t-E(r_t|F_{t-1})&#36;So...&#36;Ea_t= 0&#36;, &#36;E( a_t| F_{t- 1}) = 0&#36;I don't know. by&#36;{\varepsilon_t}&#36;Independent knowledge &#36;\varepsilon_t&#36;and&#36;F_{t-1}&#36;Independence, and thus&#36;\sigma_t^2&#36;Independent, then.
&#36; \begin{aligned}\mathrm{Var=&amp;E[(r_{t}-E(r_{t}|F_{t-1}))^{2}|F_{t-1}]=E(a_{t}^{2}|F_{t-1})\=&amp;E(\varepsilon_{t}^{2}\sigma_{t}^{2}|F_{t-1})=\sigma_{t}^{2}E(\varepsilon_{t}^{2}|F_{t-1})=\sigma_{t}^{2}\=&amp;Other Organiser &#36;
It's in the previous section.&#36;r_t&#36;condition equation.</p>
<p>Because of the coefficient.&#36;\alpha_j&#36;It's all non-negative, so it's historical.&#36;a_{t-j}^2&#36;Bigger means&#36;a_t&#36;The conditions vary considerably, so that, in the framework of the ARCH model, large disturbances tend to be followed by larger disturbances.</p>
<p>The word "prefer" here does not mean that there will be a major disturbance, because&#36;a_{t-j}^2&#36;It's a big difference.&#36;\sigma_t^2&#36;It's bigger, and the difference is just bigger.&#36;a_t&#36;It's more likely than it's bound to be.&#36;a_t&#36;I don't know. This explains the concentration of fluctuations in the rate of return on assets.</p>
<p><strong>Now we have multiple sequences and research subjects.</strong></p>
<ul>
<li><strong>We study the original yield sequence.&#36;r_t&#36; And his difference.&#36;\sigma^2_t&#36;</strong> </li>
<li><strong>We study.&#36;r_t&#36;Volatility sequence after trend&#36;a_t&#36;  He needs to use random fluctuations.&#36;\epsilon_t&#36;and the difference in the yield sequence &#36;\sigma^2_t&#36;</strong> </li>
<li><strong>The difference in the yield sequence.&#36;\sigma^2_t&#36;Use&#36;a_t^2&#36;Linear delayed combinations</strong></li>
</ul>
<p>Note: Some authors use&#36;h_t=\sigma_t^2&#36;As a condition difference mark, this is a disturbance.&#36;a_t=\varepsilon_t\sqrt{h_t}&#36;。</p>
<h3>Nature of ARCH model</h3>
<p>We have the simplest ARCH model.
&#36;&#36;a_t=\sigma_t\varepsilon_t,:\sigma_t^2=\alpha_0+\alpha_1a_{t-1}^2.&#36;&#36;
Of which &#36;\alpha 0&gt;0,0&lt;\alpha_1&lt;1&#36;。
&#36;\alpha_1&gt;0&#36;是因为如果等于零就不能算ARCH(1),&#36;\alpha_1&lt;1&#36;是为了&#36;a Limited variance of &#36;US.</p>
<h4>New Zero Nature</h4>
<p>You!&#36;F_t= \sigma ( { r_t, r_{t- 1}, \ldots } )&#36; Press&#36;a_t&#36;Definitions,
&#36;&#36;a_t=r_t-E(r_t|F_{t-1}),&#36;&#36;
&#36;{a_t}&#36;Called&#36;{r_t}&#36;New interest sequences.</p>
<p>He's satisfied.
&#36;begin{aligned}E(a)=&amp;E[r_t-E(r_t|F_{t-1})|F_{t-1}]\=&amp;E(r_t|F_{t-1})-E(r_t|F_{t-1})=0,\end{aligned}&#36;&#36;</p>
<p>Actually...
&#36;&#36;E(a_t)=E[E(a_t|F_{t-1})]=0.&#36;&#36;
There's a difference.
&#36; \begin{aligned}
&amp;&amp;\mathrm{Var}(a_{t})=&amp; E(a_t^2)=E[E(a_t^2|F_{t-1})]  \
&amp;&amp;&amp;E[E(\sigma_t^2\varepsilon_t^2|F_{t-1})]=E[\sigma_t^2E(\varepsilon_t^2|F_{t-1})] \
&amp;&amp;&amp;E[\sigma_{t}^{2}E(\varepsilon_{t}^{2})]=E(\sigma_{t}^{2}) \
&amp;&amp;&amp;= \alpha_0+\alpha_1E(a_{t-1}^2).  \
&amp;\text {because }oporatorname{Var}(a t)\text{is constant,}
&amp;&amp;&amp;\mathrm{Var}(a_{t})=E(a_{t}^{2})=\frac{\alpha_{0&#125;&#125;{1-\alpha_{1&#125;&#125;. \
&amp;\text{request}0&lt;\alpha_{1}&lt;1。
\end{aligned}&#36;&#36;</p>
<h4>AR Model</h4>
<p>So-called ARCH effects.&#36;{r_t}&#36;It's a white noise, but...&#36;{r_t^2}&#36;It is clearly relevant. For example, in the preceding section,
If&#36;r_t= a_t&#36;You can prove it.&#36;r_t^2&#36;Obey an AR(1) model.</p>
<h4>Nature of generic ARCH models</h4>
<p>It's similar in nature to the simple examples we've discussed before.</p>
<h3>The advantages and disadvantages of the ARCH model</h3>
<p>Advantages:</p>
<ul>
<li>Can generate a concentration of volatility (widespread features of financial time series)</li>
<li>Disturbing&#36;a_t&#36;A thick tail distribution (which is widely present in the financial time series)</li>
</ul>
<p>Disadvantages</p>
<ul>
<li>Because, hypothetically,&#36;a_{t-j}&#36;Pass.&#36;a_{t-j}^2&#36;Impact volatility&#36;\sigma_t&#36;Thus, positive and negative disturbances have the same effect on the volatility rate, but positive and negative disturbances in the actual asset return have different effects on the volatility rate, with larger negative disturbances being greater than those caused by positive disturbances.</li>
<li>ARCH models have stricter constraints on model parameters, even if ARCH (1) is necessary to calculate peaks&#36;\alpha_1\in(0,\frac{\sqrt{3&#125;&#125;3)&#36;High class.&#36;ARCH(m)&#36;The constraints are more complex. It's a limit to the fact that the ARCH model with the new Gaussian breath expresses a thick tail by super-peak.</li>
<li>The change in the variance can only be described, but the reasons for the change cannot be explained.</li>
<li>The model's fluctuations are projected to be high.</li>
<li>It could be bigger.&#36;m&#36;。</li>
</ul>
<h3>ARCH Modelling Method</h3>
<h4>Level</h4>
<p>After the ARCH effects test is clear, this is the section on ARCH effects.&#36;{a_t^2}&#36; The PCAF is here to determine.</p>
<p>First, the model is...
&#36;&#36;\sigma_t^2=\alpha_0+\alpha_1a_{t-1}^2+\cdots+\alpha_ma_{t-m}^2&#36;&#36;
Because...&#36;E(a_t^2|F_{t-1})=\sigma_t^2&#36;So I think it's close.
&#36;&#36;a_t^2\approx\alpha_0+\alpha_1a_{t-1}^2+\cdots+\alpha_ma_{t-m}^2&#36;&#36;
It'll work.&#36;{a_t^2}&#36;The finality of the sequence PACF to estimate the ARCH steps.&#36;m&#36;。</p>
<p>If the order of magnitude shows that an AR model is required, then perhaps the ARCH model is not suitable for this sequence.</p>
<h4>Model estimation</h4>
<p>In fact, the ARCH model estimates should be divided into two steps.
See the “relevant paragraphs” section of this paper.
The apparent function of the model and the assumed&#36;\epsilon_t&#36;. The distribution is related, and there are a number of apparent functions.</p>
<h5>Normal distribution</h5>
<p>When assumed&#36;\varepsilon_t&#36;The condition logarithmic function is
&#36;&#36;\ell(a_{m+1},\ldots,a_T|\boldsymbol{\alpha})=-\frac{1}{2}\sum_{t=m+1}^T\left[\ln\sigma_t^2+\frac{a_t^2}{\sigma_t^2}\right]+\text{常数项}&#36;&#36;</p>
<h5>t Distribution</h5>
<p>Because of the thick end of the yield distribution, some applications assume that the standard t distribution is subject to a priori design t-distribution freedom.
&#36; \begin{aligned}
&amp;\ell(a_{m+1},\ldots,a_T|v,\boldsymbol{\alpha},a_1,\ldots,a_m) \
\text{=}&amp; (T-m)\left[\ln\Gamma\left(\frac{v+1}{2}\right)-\ln\Gamma\left(\frac{v}{2}\right)-\frac{1}{2}\ln((v-2)\pi)\right]  \
&amp;+\ell(a_{m+1},\ldots,a_T|\boldsymbol{\alpha},a_1,\ldots,a_m).
\end{aligned}&#36;&#36;</p>
<h5>Offset t distribution</h5>
<p>The distribution of return on assets is often biased in addition to the thick end. The t-distribution can be modified to become a standardized single-peak density. There are a number of ways to make such a change, using here the Fernández and Steel approach. This method can introduce bias in any continuous single-peak one dollar distribution of symmetric zero. Will (t)&#36;v&#36;) distribution after bias is
&#36;US&#36;\left.g (\varepsilon|v,\xi)=\left{begin{array}ll}\frac{2}{\xi+\frac{1}f(c 2\varepsilon t+c 1)\mid v,&amp;\varepsilon_t&lt;-\frac{c_1}{c_2},\\frac{2c_2}{\xi+\frac{1}{\xi&#125;&#125;f((c_2\varepsilon_t+c_1)/\xi\mid v),&amp;\varepsilon_t\geq-\frac{c_1}{c_2}.\end{array}\right.\right.&#36;&#36;</p>
<h5>Broad error distribution assumptions</h5>
<p>&#36;\varepsilon_t&#36;The other desirable distribution is the broad error distribution (GED), the density is
&#36;f (x|v) = \frac{v }{\lambda2^1\frac1}v} \gamma(\)}e\frac{w\lambda},: x\in(-\info,\info):&lt;V\leq\info)
of which&#36;v=2&#36;standard normal distribution, &#36;0&lt; v&lt; At &#36;2.00 is the thick end distribution.
&#36;&#36;\lambda=\left[2^{-\frac2v}\frac{\Gamma(\frac1v)}{\Gamma(\frac3v)}\right]^{\frac12}.&#36;&#36;</p>
<h4>Model Validation</h4>
<p>For a built ARCH model, a standardized residual can be calculated
&#36;&#36;\tilde{a}_t=\frac{a_t}{\sigma_t},&#36;&#36;
of which&#36;a_t&#36;This is the difference in the equation.&#36;\sigma_t&#36;is the value proposed for the swing equation.&#36;{\tilde{a}_t}&#36;It should be shown in a zero-average, unit-standard difference independent and distributed sequence.</p>
<ul>
<li>Yeah.&#36;{\tilde{a}_t}&#36;Ljung-Box white noise test allows the adequacy of the mean equation.</li>
<li>Yeah.&#36;{\tilde{a}_t^2}&#36;Ljung-Box white noise test allows the adequacy of the swing equation.</li>
<li>&#36;{\tilde{a}_t}&#36;QQQ diagrams can be used to compare&#36;\varepsilon_t&#36;to test the validity of model assumptions.</li>
</ul>
<h4>Projections</h4>
<p>ARCH models project similar to AR models. From Predicted Origin&#36;h&#36;Let's go, yeah.&#36;\sigma_t^2&#36;Sequences are projected ahead, that is, projections.&#36;\sigma_{h+1}^2&#36;Yes.
&#36;&#36;\sigma_h^2(1)=\sigma_{h+1}^2=\alpha_0+\alpha_1a_h^2+\cdots+\alpha_ma_{h+1-m}^2.&#36;&#36;
When we do the first two steps,&#36;a_{h+1}&#36;Unknown, yes. &#36;E(a_{h+1}^2|F_h)=\sigma_h^2(1)&#36;♪ And so ♪
&#36;&#36;\sigma_h^2(2)=\alpha_0+\alpha_1\sigma_h^2(1)+\alpha_2a_h^2+\cdots+\alpha_ma_{h+2-m}^2.&#36;&#36;
In general,&#36;\sigma_h^2(\ell)&#36;You can scroll it.
&#36;&#36;\sigma_h^2(\ell)=\alpha_0+\sum_{j=1}^m\alpha_j\sigma_h^2(\ell-j).&#36;&#36;</p>
<h2>GARCH Model</h2>
<p>The ARCH model is used to describe the effect of volatility, but the actual modelling may require higher steps, considering model changes similar to those that have been extended from AR to ARMA.</p>
<h3>GARCH model equation</h3>
<p>(Bollerslev 1986) An important extension model of the ARCH model, known as the GARCH model, was presented. For a logarithmic yield sequence&#36;r_t&#36;You...&#36;a_t=r_t-\mu_t=r_t-E(r_t|F_{t-1})&#36;For its new interest sequence,&#36;{a_t}&#36;Obey garch.&#36;(m,s)&#36;model, if&#36;a_t&#36;Satisfied
&#36;&#36;a_t=\sigma_t\varepsilon_t,\quad\sigma_t^2=\alpha_0+\sum_{i=1}^m\alpha_ia_{t-i}^2+\sum_{j=1}^s\beta_j\sigma_{t-j}^2&#36;&#36;
of which&#36;{\varepsilon_t}&#36;Independent and distributed white noise column for difference in zero mean units, &#36;\alpha  0&gt; 0, \alpha <em>i\geq 0, \beta <em>j\geq 0&#36;,
&#36;0&lt;\sum</em>{i=1}^m\alpha_i+\sum</em>{j=1}^s\beta_j&lt;1&#36;,这最后一个条件用来保证满足模型的&#36;a_t&#36;的无条件方差有限且不变，而条件方差&#36;\sigma_t^2&#36;可以随时间&#36;t&#36; and change.</p>
<h3>GARCH Model Nature</h3>
<p>It's the easiest down there.&#36;GARCH(1,1)&#36;Study the nature of the GARCH model for example. You!&#36;F_{t-1}&#36;Other Organiser&#36;t-1&#36;The moment.&#36;a_{t-i}&#36;and&#36;\sigma_{t-j}&#36;Information included. Model as
&#36; \begin{aligned}&amp;a_{t}=\sigma_{t}\varepsilon_{t},\quad\varepsilon_{t}\mathrm{<del>i.i.d.</del>WN}(0,1),\&amp;\sigma_{t}^{2}=\alpha_{0}+\alpha_{1}a_{t-1}^{2}+\beta_{1}\sigma_{t-1}^{2}.\end{aligned}&#36;&#36;
为了计算无条件均值&#36;Ea_t&#36;,先计算条件期望
&#36;&#36;E(a_t|F_{t-1})=E(\sigma_t\varepsilon_t|F_{t-1})=\sigma_tE(\varepsilon_t|F_{t-1})=0.&#36;&#36;
这里用了&#36;\sigma_t\in F_{t-1}&#36;而&#36;\varepsilon_t&#36;与&#36;F_{t-1}&#36;独立。于是
&#36;&#36;Ea t=E(a t|F )=&#36;0.00
That's the new interest in the GARCH model.&#36;a_t&#36;Unconditional expectations are zero.</p>
<p>To calculate.&#36;a_t&#36;the unconditional difference. Modelled (18.1)&#36;{a_t}&#36;There's a smooth and smooth sequence, then.
&#36; \begin{aligned}\mathrm{Var}(a t}=&amp;E(a_{t}^{2})=E[E(a_{t}^{2}|F_{t-1})]=E[E(\sigma_{t}^{2}\varepsilon_{t}^{2}|F_{t-1})]\=&amp;E[\sigma_{t}^{2}E(\varepsilon_{t}^{2}|F_{t-1})]=E[\sigma_{t}^{2}E(\varepsilon_{t}^{2})]\=&amp;E[\sigma_{t}^{2}]=E[\alpha_{0}+\alpha_{1}\alpha_{t-1}^{2}+\beta_{1}\sigma_{t-1}^{2}]\=&amp;\alpha_{0}+\alpha_{1}E(a_{t-1}^{2})+\beta_{1}E[E(a_{t-1}^{2}|F_{t-2})]\=&amp;\alpha_{0}+(\alpha_{1}+\beta_{1})E(a_{t-1}^{2}).\end{aligned}&#36;&#36;
令&#36;Ea_t^2=Ea_{t-1}^2&#36;,解得
&#36;&#36;\mathrm{Var}(a_t)=Ea_t^2=\frac{\alpha_0}{1-\alpha_1-\beta_1}.&#36;&#36;
<strong>There is no difference between these studies and the section on "The Nature of ARCH Models" here.</strong></p>
<p>In summary:
First, like the ARCH model,&#36;a_t&#36;There's volatility. One big one.&#36;a_{t-1}&#36;or&#36;\sigma_{t-1}&#36;This has led to a large difference in conditions after 1 step, which tends to have a larger logarithmic rate of return.</p>
<p>Second, when&#36;\varepsilon_t&#36;For standard normal distribution, under the following conditions:&#36;a_t&#36;There are unconditional four-steps:
&#36;1-2\alpha 1^2^1\1^beta 1^2&gt;0.&#36;&#36;
这时超额峰度为
&#36;&#36;\frac{Ea_t^4}{(Ea_t^2)^2}-3=\frac{2\left[1-(\alpha_1+\beta_1)^2+\alpha_1^2\right]}{1-(\alpha_1+\beta_1)^2-2\alpha_1^2}&gt;&#36;0.00
That's...&#36;a_t&#36;Distribution of thick tails. However, even with the use of condition t distributions for modelling of actual data, this may not be sufficient for a robust tailing of the data.</p>
<p>Third, the GARCH model gives a simpler model of volatility.</p>
<p>Fourth, because&#36;\sigma_t^2&#36;Yeah.&#36;a_{t-i}&#36;♪ Reliance through&#36;a_{t-i}^2&#36;, so a positive disturbance&#36;a_{t-i}&#36;And a negative value.&#36;a_{t-i}&#36;As long as absolute values are equal, the effect on subsequent fluctuations is equal and does not reflect leverage.</p>
<h3>Projections</h3>
<p>Volatility rates can be projected using methods similar to ARMA projections. Still by&#36;GARCH(1,1)&#36;For example, there's a step ahead.
&#36;&#36;\sigma_{h+1}^2=\alpha_0+\alpha_1a_h^2+\beta_1\sigma_h^2\in F_h.&#36;&#36;
Organisation&#36;a_{h+1}&#36; The two-step projection is here.
&#36; \begin{aligned}
== sync, corrected by elderman ==&amp; \alpha_0+\alpha_1a_{h+1}^2+\beta_1\sigma_{h+1}^2  \
\text{=}&amp; \alpha_0+\alpha_1\sigma_{h+1}^2\varepsilon_{h+1}^2+\beta_1\sigma_{h+1}^2  \
=&amp; \alpha_0+(\alpha_1\varepsilon_{h+1}^2+\beta_1)\sigma_{h+1}^2
\end{aligned}&#36;&#36;
借此实现循环 进行超前多步预测有
&#36;&#36;== sync, corrected by elderman == @elder man
That's what we're predicting.&#36;a_{t}&#36;Unconditional difference</p>
<h3>Model estimation</h3>
<p>The modelling steps of the ARCH model also apply to the GARCH model. There's not much research on the ranking methods of the GARCH model, and it's basically enough to use methods like the ARAMA model.&#36;GARCH(2,2)&#36; It's acceptable to use error at this time.</p>
<p>The model's test method is fully referenced in this section on Model Validation, which examines the model's estimated effects by examining standardized residuals.</p>
<h3>Two-step estimation method</h3>
<p>In the traditional GARCH modeling, the estimation method we use is to estimate the entire model directly in the section of this "ARCH Model Modeling" section. </p>
<p>In many studies, we have proposed the following two-step estimation method for the GARCH model:</p>
<ul>
<li>Ignores the ARCH effect, and establishes the mean equation for the yield sequence using linear time-series modelling methods (e.g. maximum seemingly estimated). Discrepancies&#36;a_{t}&#36;Organisation</li>
<li>Will&#36;{a_t^2}&#36;As a series of observations, the parameters can be estimated using the maximum approximation method. Use&#36;\hat{\phi}_i&#36;and&#36;\hat{\theta}_i&#36;The coefficient estimates for AR and MA components are indicated, respectively. GARCH model parameters are estimated&#36;\hat{\beta}_i=-\hat{\theta}_i,\hat{\alpha}_i=\hat{\phi}_i+\hat{\theta}_i&#36;。<strong>This method of estimation is only a approximation and does not have theoretical results to justify it, but some experience suggests that such estimates tend to give a good approximation, especially in the case of large and medium samples.</strong></li>
</ul>
<h3>IGARCH</h3>
<p>We're considering extending to the ARIMA model.</p>
<p>GARCH model can be written in (18.2) about&#36;a_t^2&#36;THE ARMA FORM,&#36;\eta_t=a_t^2-\sigma_t^2&#36;It's the disturbance of the model. It's the noise:
&#36;&#36;a_t^2=\alpha_0+\sum_{i=1}^{\max(m,s)}(\alpha_i+\beta_i)a_{t-i}^2+\eta_t-\sum_{j=1}^s\beta_j\eta_{t-j},&#36;&#36;
If the AR component of this model has unit root (with signature root equal to 1), the corresponding model no longer meets the conditions of the GARCH model, known as the IGARCH model, or the unit root GARCH model. A disturbance similar to the ARIMA model, the IGarch model.&#36;\eta_t=a_t^2-\sigma_t^2&#36;Yeah.&#36;a_t^2&#36;The impact is durable and undiminished.</p>
<p>IGARCH (1,1) Model
&#36;&#36;a_t=\sigma_t\varepsilon_t,\quad\sigma_t^2=\alpha_0+\alpha_1a_{t-1}^2+(1-\alpha_1)\sigma_{t-1}^2.&#36;&#36;</p>
<h3>GARCH-M model</h3>
<p>The average of the return on some financial assets is affected by their volatility and is referred to as a risk premium. The GARCH-M model is used to describe such phenomena, and M indicates that the condition average is dependent on the GARCH model. A simple GARCH-M (1,1) model is
&#36; \begin{aligned}&amp;r_{t}=\mu+c\sigma_{t}^{2}+a_{t},\quad a_{t}=\sigma_{t}\varepsilon_{t},\&amp;== sync, corrected by elderman == @elder man &#36;
The average rate of return in the model is &#36;E(r_t|F_{t-1})=\mu+c\sigma_t^2&#36; The condition difference is required&#36;\sigma_t^2=\operatorname{Var}(r_t|F_{t-1})&#36;Description. Parameters&#36;c&#36;Called risk premium parameters, if&#36;c&#36;The positive rate of return is linked to the volatility rate.</p>
<p>There are other risk premium models in the literature, such as
&#36; \begin{aligned}&amp;r_t=\mu+c\sigma_t+a_t\&amp;r t=mu=sigma t^2+a t\end{aligned}
Rate of return&#36;r_t&#36;It's not about the column, it's about the sequence, it's about the correlation.&#36;\sigma_{t}^{2}&#36;is the serial relevance. The existence of a risk premium is one of the reasons for the serial relevance of the equity rate of return.</p>
<h2>Expand Garch</h2>
<p>This chapter is about some of the targeted improvements to the GARCH model that have not been detailed for the time being.</p>
<h3>EGARCH</h3>
<p>The index GARCH (EGARCH) model (Nelson 1991) allows positive and negative asset yields to have asymmetric effects on volatility. Consider the following variations:
&#36;&#36;g(\varepsilon_t)=\alpha\varepsilon_t+\gamma\left[|\varepsilon_t|-E|\varepsilon_t|\right],&#36;&#36;
of which&#36;\alpha&#36;and&#36;\gamma&#36;is the real constant.&#36;{\varepsilon_t}&#36;and&#36;{|\varepsilon_t|-E|\varepsilon_t|}&#36;They are both zero-average independent and distributed white noise, distributed in a continuum. Visibility.
&#36;Eg(\varepsilon_t)=0&#36;I don't know.
Visible from Bottom&#36;g(\varepsilon_t)&#36;The distribution is asymmetric:
&#36;g (\varepsilon t) =\left(begin{matrix}(alpha+\gamma)\varepsilon t-\varepsilon t|,&amp;\varepsilon_t\geq0,\(\alpha-\gamma)\varepsilon_t-\gamma E|\varepsilon_t|,&amp;\varepsilon_t&lt;0.\end{matrix}\right.&#36;&#36;
当&#36;\varepsilon_t\sim&#36;N(0,1)时&#36;E|\varepsilon_t|=\sqrt{\frac2\pi}&#36;。 对式(17.6)中的标准t分布，有
&#36;&#36;E|\varepsilon_t|=\frac{2\sqrt{v-2}\Gamma(\frac{v+1}2)}{(v-1)\Gamma(\frac v2)\sqrt{\pi&#125;&#125;.&#36;&#36;</p>
<p>EGARCH&#36;(m,s)&#36;The model can be written in the form of a lag counter.
&#36;a t=t\varepsilon t,\sigma t^2=alpha 0&#39;+\frac{1+\alpha_2B+\cdots+\alpha_mB^{m-1&#125;&#125;{1-\beta_1B-\cdots-\beta_sB^s}g(\varepsilon_{t-1})&#36;&#36;</p>
<p>&#36;\alpha_0^{\prime}&#36;of which&#36;B&#36;It's a lag counter. Multiple.&#36;1+\alpha_2z+\cdots+\alpha_mz^{m-1}&#36;and &#36;1-\beta_1z-\cdots-\beta_mz^m&#36;It's all there.</p>
<p>in the outer circle without a public factor in two polygons.</p>
<p>Note that this model is equivalent to GARCH.</p>
<p>Remember&#36;\xi_t=\ln\sigma_t^2&#36;, and (19.3)&#36;\xi_{t}&#36;A smooth line of ARMA.&#36;(s,m-1)&#36;Sequence, independent of zero mean and distributed white noise&#36;g(\varepsilon_{t-1})&#36;For a new break; but,&#36;\ln\sigma_t^2&#36;Pass.&#36;\varepsilon_{t-j}=a_{t-j}/\sigma_{t-j}&#36;Yeah.&#36;{a_t}&#36;Sequence depends. The original Garch model.&#36;\sigma_t^2&#36;The equation is directly dependent on&#36;a_t-j^2&#36;It's...&#36;\pm a_{t-j}&#36;Yeah.&#36;\sigma_t^2&#36;The impact is the same.</p>
<p>Visibility.&#36;E\ln\sigma_t^2=\alpha_0^{\prime}&#36;。</p>
<p>More generally, it can be ordered.&#36;g(\cdot)&#36;Medium&#36;\gamma&#36;With ageing&#36;j&#36;Changed, the model becomes:
&#36; \begin{aligned}a&amp;\sigma_{t}\varepsilon_{t},\\ln\sigma_{t}^{2}=&amp;\alpha_{0}+\sum_{j=1}^{m}\left[\alpha_{j}\varepsilon_{t-j}+\gamma_{j}(|\varepsilon_{t-j}|-E|\varepsilon_{t-j}|)\right]+\sum_{i=1}^{s}\beta_{i}\ln\sigma_{t-i}^{2}.\end{aligned}&#36;&#36;
(19.4)</p>
<p>In (19.4),&#36;\alpha_j&#36;represents the different effect of a positive or negative disturbance of the logarithmic rate of return on the volatility rate if&#36;\alpha_j=0&#36;, positive-negative disturbance against volatility</p>
<p>The impact is the same.</p>
<p>The difference between EGARCH and GARCH models is also:</p>
<ol>
<li>The use of a logarithmic logarithmic of condition deviations, since the logarithmic values are positive and negative, removes the requirement that the GARCH model be non-negative. 2. &#36;g(\varepsilon_{t-j})=g(a_{t-j}/\sigma_{t-j})&#36;It's used to match fluctuations.&#36;a_{t-j}&#36;Dependency relations and&#36;a_{t-j}&#36;It's a positive or negative number. It's a good description.
The impact of negative rates of return on fluctuations, i.e. the leverage effect.</li>
</ol>
<h3>GJR-GARCH</h3>
<p>The GJR-GARCH model is another model of volatility that reflects leverage, see (Glosten, Jaganathan, and Runkle)
1993 and (Zakoian 1994) Or Tgarch.&#36;m,s)&#36;Model as
&#36;&#36;\sigma_t^2=\alpha_0+\sum_{i=1}^m(\alpha_i+\gamma_iN_{t-i})a_{t-i}^2+\sum_{j=1}^s\beta_j\sigma_{t-j}^2&#36;&#36;
(19.9)</p>
<p>of which&#36;N_{t-i}&#36;It means &#36;a t-i}&lt;An indicative function of &#36;0.00, or
&#36;N t-i}\left{array}ll1,&amp;a_{t-i}&lt;0\0,&amp;a_{t-i}\geq0\end{array}\right.&#36;&#36;
&#36;\alpha_i,\gamma_i,\beta_j&#36;is a non-negative parameter that meets parameters similar to those of the GARCH model.
Nice.&#36;a_{t-i}&#36;Yeah.&#36;\sigma_t^2&#36;The impact is:&#36;\alpha_ia_{t-i}^2&#36;Negative.&#36;a_{t-i}&#36;Yeah.&#36;\sigma_t^2&#36;The impact is:&#36;(\alpha_i+\gamma_i)a_{t-i}^2&#36;, when&gt;0&#36;时负的&#36;a  t-i&#36; has greater impact.
Model zero.&#36;a_{t-i}&#36;, other real values may be used as the threshold. See chapter 4 (Tsay 2010) for door-limit models.</p>
<h3>ARARK Model</h3>
<p>Ding, Granger.and Engle presented the Asymmetric Power ARCH model (APARCH model) in the form of
&#36; \begin{aligned}&amp;r_{t}=\mu_{t}+a_{t},\quad a_{t}=\sigma_{t}\varepsilon_{t},\quad\varepsilon_{t}\sim D(0,1)\&amp;== sync, corrected by elderman == @elder man &#36;
of which&#36;\mu_t&#36;is the condition average,&#36;D(0,1)&#36;This is the difference in the distribution of a zero-average unit.&#36;\delta&#36;Positive, factor&#36;\omega,\alpha_i,\gamma_i,\beta_j&#36;Meeting certain positive conditions makes the volatility positive.</p>
<p>The simplest APARCH(1,1) model is most commonly used. This model contains many other models.
When?&#36;\delta=2&#36;and&#36;\gamma_j=0&#36;It's a regular Garch model.
When?&#36;\delta=2&#36;Time or TGARCH model (in slightly different forms).
When?&#36;\delta=1&#36;Direct use of fluctuations in the hourly rate equation&#36;\sigma_t&#36;And a new break.&#36;a_t&#36;Not square. The transformation of the aluminum in APARCH is intended to improve the degree of integration, but it's not.&#36;\delta&#36;No good explanation.</p>
<h3>Normal random fluctuations SV model</h3>
<p>In front of the swing equation&#36;\sigma_t^2=\operatorname{Var}(a_t|F_{t-1})&#36;It's all been...&#36;\sigma_{t-1},\ldots&#36;and &#36;a_{t-1},\ldots&#36;Totally. The other way is to assume.&#36;\sigma_t^2&#36;The model itself is new, and it is called the Standard Volatility, SV model. Model
&#36;&#36;a_t=\sigma_t\varepsilon_t,\quad(1-\alpha_1B-\cdots-\alpha_mB^m)\ln\sigma_t^2=\alpha_0+v_t.&#36;&#36;
of which&#36;\sigma_t^2&#36;The logarithm is taken in order to remove the limitation that the coefficient must not be negative.&#36;{\varepsilon_t}&#36;It's the same as standard normal distribution.&#36;{v_t}&#36;Independence and N&#36;(0,\sigma_v^2)&#36;Distribution, &#36;{\varepsilon_t}&#36;and&#36;{v_t}&#36;Independence from each other.&#36;\alpha_i&#36;It's a constant. Characteristic Multiform&#36;1-\alpha_1z-\cdots-\alpha_mz^m&#36;The roots are all outside the circle. Remember&#36;\xi_t=\ln\sigma_t^2&#36; , then&#36;{\xi_t}&#36;It's a tight and steady AR.&#36;(m)&#36;Sequence.</p>
<p>Add&#36;v_t&#36;After the new interest rate, the rate of return&#36;r_t&#36;A new moment.&#36;a_t&#36;That's it.&#36;\varepsilon_t&#36;and&#36;v_t&#36;Two new ones, which increase the freedom of the model, but make it from&#36;r_t&#36;Data estimation model parameters have become more difficult and are to be calculated using Kalman filters or random simulation methods.</p>
<p>The SV model often improves on alignment, but it's good and bad outside of the fluctuations.</p>
<h3>Long Memory Random Volatility Model</h3>
<p>An empirical analysis of the rate of return on assets found that the rate of return was not per se long-term memory, but its square or absolute-value series ACF tended to decline slowly. In front of the GARCH model&#36;\sigma_{t-1}^2&#36;The coefficient is very close to one and suggests a long memory.</p>
<p>A simple LMSV model can be written.
&#36;&#36;a_t=\sigma_t\varepsilon_t,\quad\sigma_t=\sigma e^{\frac{1}{2}u_t},\quad(1-B)^du_t=\eta_t.&#36;&#36;
Of which \sigma &gt; 0&#36;, &#36;{ \varepsilon <em>{t&#125;&#125;&#36;和&#36;{\eta</em>{t&#125;&#125;&#36;是两个相互独立的独立同分布高斯白噪声列，&#36;\varepsilon_t\sim&#36;N&#36;(0,1),\eta_{t}\sim&#36;N&#36;(0,\sigma_{\eta}^{2})&#36;,&#36;0&lt;d&lt;0.5&#36;。长记忆来源于分数差分&#36;(1-B)^d&#36;,这使得&#36;u t}&#36;US&#36;ACF is reduced by negative velocity instead of negative velocity.</p>
<p>Yes, for LMSV.
&#36; \begin{aligned}&amp;\ln(\sigma_t^2\varepsilon_t^2)=\ln\sigma^2+u_t+\ln\varepsilon_t^2\=&amp;(\ln\sigma^2+E\ln\varepsilon_t^2)+u_t+(\ln\varepsilon_t^2-E\ln\varepsilon_t^2)\=&amp;\mu+u t+e.\end{aligned}
of which&#36;u_t&#36;It's a long memory smooth Gaussy time series.&#36;e_t&#36;It's an independent and distributed white noise column of non-Gorse.</p>
<p>LMSV estimates are complex, fractional parameters&#36;d&#36;It can be estimated using either the maximum apparent method or the regression method.</p>
