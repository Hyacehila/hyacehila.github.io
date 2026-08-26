---
title: 'Financial Stochastic Analysis: Derivatives, Binomial Pricing, and No-Arbitrage Theory'
title_zh: 金融随机分析：金融衍生品、二叉树定价与无套利理论
date: 2024-10-17 15:19:25 +0800
categories:
- Data Science
- Time Series & Spatial Data
tags:
- Statistics
- Stochastic Processes
- Financial Mathematics
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers derivatives, binomial pricing, no-arbitrage theory, risk-neutral measures, martingales, and continuous-time
  financial models.
description: Covers derivatives, binomial pricing, no-arbitrage theory, risk-neutral measures, martingales, and continuous-time
  financial models.
excerpt_zh: 整理金融衍生品、二叉树定价、无套利理论、风险中性测度、鞅和连续时间金融模型。
permalink: /blog/2024/10/17/financial-stochastic-analysis-notes/
lang: en
translation_key: 2024-10-17-financial-stochastic-analysis-notes
translation_status: machine
translation_source_hash: 6900957e90efe909a54663f7cf7b6d15a4fd506c17c92a4bd928bc91959de71e
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Introduction and Introduction</h2>
<h3>Introduction</h3>
<p>The mathematical theory of financial science is studied by financial random analysis, and we are concerned with financial derivatives developed by financial institutions, i.e. investment products derived and combined from primary financial products outside the stock and spot markets. The random analysis of finance is intended to provide a reasonable price for these financial derivatives, which would guarantee fair profits to both financial institutions and firms that purchase them on the market.</p>
<p>The entire financial random analysis is based on the Master of CMU ' s teaching materials, which are divided into two volumes, each of which examines the pricing theory of discrete time and the random analysis theory of continuous time. We have chosen to learn about it, focusing only on the basic theories and methods of financial mathematics.</p>
<h3>Some classic financial derivatives.</h3>
<p>The variety of financial derivatives themselves is so great that we need to introduce some of the common ones to keep moving forward.</p>
<h4>Pasal and Eurosalt</h4>
<p>The two most common derivatives are: <strong>Options and Futures</strong> We only consider the former in the underlying pricing model. He was a financial derivative designed on the basis of stock. All options are a right, not a commodity that actually exists.</p>
<p>First, let's introduce the basics.<strong>The European Call scenario.</strong> And it's also American, and it's different options like that, and we'll talk about it later.</p>
<p>The European-based rise-up option is defined as a contract that is purchased by the A-B, and that is agreed to be available at a future moment.&#36;T&#36;Prices determined&#36;K&#36;Purchase of a certain amount of stocks from B.</p>
<p>The corresponding definition of the European fall option is that the party in the A direction has purchased a contract that the party in the A can at a future moment.&#36;T&#36;Prices determined&#36;K&#36;A certain amount of stocks sold to B.</p>
<p>Let's assume the future.&#36;T&#36;The price of the stock is&#36;S_T&#36; So the natural benefits of giving two options are as follows:</p>
<ul>
<li>The European way of looking at options. &#36;W = max(S_{T}-K,0)&#36; </li>
<li>European-like drop options &#36;W = max(K-S_{T},0)&#36;</li>
</ul>
<p>Of course, we can give you some other options, like...</p>
<ul>
<li>binary options, with the right to increase and to decline</li>
<li>Increase barrier prices on the basis of initial European options, and when the barrier price is touched, the barrier is effective or invalid</li>
</ul>
<p>Indeed, many of the classic portfolios need only basic European options to be combined, so our main presentation is only European-style options.</p>
<h4>The role of the right to option</h4>
<p>As the most fundamental financial derivatives, options can make a significant difference, as follows:</p>
<h5>Leverage</h5>
<p>Assuming we have an initial fund of &#36;100, and we think that stock prices will rise and European options will increase by &#36;2.</p>
<p>If equity rises to &#36;110, investment stocks can yield 10 per cent of the return, while purchasing 50 European-style options can yield &#36;500, 50 times the original yield, which is the leverage of options.</p>
<p>Of course, if stock prices fall, we will lose 100 per cent if all options purchased are sold into waste paper without the value of the right to do business.</p>
<h5>Hedge!</h5>
<p>We still think that stock prices will rise, but we choose to buy a European-style downside option while buying shares. And then our gain becomes
&#36;&#36;W = S_{T} + (K-S_{T})^{+}&#36;&#36;
Which means we'll have a floor of loss, no matter how much the stock price falls. &#36;K&#36;  That's the risk of a hedge.</p>
<h5>Arbitrage</h5>
<p>The arbitrage refers to the fact that we have identified the defects in options pricing, and that we can obtain profits without risk, by purchasing options and shares in proportion, regardless of how the stock prices change.</p>
<p>Research on arbitrage and risk-free arbitrage is an act of mutual aggression between financial institutions and investors in the market.</p>
<h4>Some European-style options derivatives</h4>
<p>All we need is simple European options and stocks, so we can build a mix of things.</p>
<h5>Equities + drop options</h5>
<p>Proceeds are satisfied
&#36;&#36;W = S_{T} + (K-S_{T})^{+} = max(S_T,K)&#36;&#36;
Combination of hedge risk</p>
<h5>Pre-subscription (to buy shares while selling an increase)</h5>
<p>Proceeds are satisfied
&#36;&#36;W = S_{T} - (S_{T}- K)^{+} = min(S_T,K)&#36;&#36;</p>
<p>We're looking at the profits from selling options, and we think stocks will increase prices, but they won't rise to the sales margin.</p>
<h5>Cross-optional rights</h5>
<p>Buy.&#36;K&#36;The price of Europe is rising and&#36;K&#36;The price falls in Euros, the returns are satisfied.
&#36;&#36;W = (S_{T}- K)^{+} + (K-S_T)^{+}&#36;&#36;
We want to see the sharp price fluctuations in equities to benefit from those we think are volatile but not in the direction of them.</p>
<h5>Wide-range options</h5>
<p>Buy.&#36;K_1&#36;The price of Europe is rising and&#36;K_2&#36;The price falls in Euros, the returns are satisfied.
&#36;&#36;W = (S_{T}- K_1)^{+} + (K_2-S_T)^{+}&#36;&#36;
Yes.&#36;S_T&#36;in&#36;K_1&#36;Present.&#36;K_2&#36;There is no return between, which is a more radical option than a cross-cutting option.</p>
<h5>Butterfly price differential options</h5>
<p>Buy &#36;K s&lt;K_{3}&#36;两个价格的看涨期权 卖出两份 &#36;K 2(2)=frac{K 1}K &#125;&#125;&#36;(2}Rooms are satisfied
&#36;&#36;W = (S_{T}- K_1)^{+} + (S_{T}- K_{3})^{+} - 2 (S_{T}- K_2)^{+}&#36;&#36;
A risk-push strategy, only when stock prices are right.&#36;K_2&#36;And then we can maximize our gains, compared to the sharp swings in cross-optional options, which are the options for predicting stock price stability, using two different prices to see the hedges.</p>
<h5>Invert risk options</h5>
<p>Buy&#36;K_2&#36;Look up, sell it.&#36;K_1&#36;Seen down the karma&lt;K &#36;2
&#36;&#36;W = (S_{T}-K_{2})^{+} - (K_{1} - S_{T})^{+}&#36;&#36;
Purely aggressive behavior, while profiting from selling the proceeds of the right to decline and from the right to increase prices, balanced the loss of some of the right to increase. Loss</p>
<h3>Continuous compound interest bonds</h3>
<p>Many assets are valued in relation to interest rates, for which we are generally called fixed-income assets.</p>
<p>The classic fixed-income asset is the Zero Debt Bill Bond. He will pay you the specified cash (known as the nominal) during the specified period (due date). The price of the Zero bond must be below nominal value before maturity, provided that the interest rate is greater than zero.</p>
<p>I'd like to set the moment.&#36;t&#36; Due date&#36;T&#36; The interest-free bond price at maturity is&#36;1&#36; The interest rate for each cycle is&#36;r&#36;  So you can give us the price of the Zero Debt Bill at the moment.&#36;x&#36; Satisfied
&#36;&#36; x e^{r(T-t)} = 1&#36;&#36;
That's right.
&#36;&#36;B_{t}(T)= e^{-r(T-t)}&#36;&#36;</p>
<p>And why? &#36;e&#36; That's because normal, continuous compounding is enough.
&#36;&#36;(1 + \frac{r}{n})^{n}&#36;&#36;
♪ With &#36;n \to \infty&#36; That's the maximum interest rate for the high frequency. &#36;e^{rt}&#36; That means a straight compound.</p>
<h2>No arbitrage pricing model for the diagonal tree</h2>
<h3>Single-time dident tree model</h3>
<h4>Definition of a single-time dident tree pricing model</h4>
<p>The Didentary Asset Pricing Model will provide a knowledge base for our understanding of arbitrage pricing theory, considering the single-time two-knot pricing model, the more widely used multi-temporal two-knot pricing model and the related calculation of the model. It is worth noting that: <strong>Study of the problem of the diagonal pricing model for discrete time</strong> Not applicable for consecutive periods.</p>
<p>The single-time dident tree model has two moments, zero and one, and at zero, we hold a stock at zero.&#36;S_0&#36; He's defaulting as positive. At the first minute, stocks have two possible prices. &#36;S_1(H)&#36; and &#36;S_1(T)&#36;  of which&#36;H,T&#36; The government has been able to provide a good opportunity to the people of the country to help them to get rid of the problem. So the price of the stock at the moment is dependent on a coin-droping experiment. We assume that the probability of a positive outcome of this experiment is...&#36;p&#36; The odds of the opposite are... &#36;q = 1-p&#36; </p>
<p>So we can define two new positives.
&#36;&#36;u=\frac{S_1(H)}{S_0},\quad d=\frac{S_1(T)}{S_0}&#36;&#36;
We promise you &#36;d. &lt; u&#36; 如果违背在调换硬币正反面的定义 我们称&#36;u&#36; 是上升因子 &#36;d&#36; 是下降因子 仅凭直觉，我们可以说&#36;u &gt;1, d&lt;1&#36;  通常我们假定&#36;d = \frac}</p>
<p>To make models more marketable, we introduced interest rates. &#36;r&#36;  It means "0" at a time. &#36;1&#36;The dollar is at the moment of the currency market becoming&#36;1+r&#36; General market &#36;r \ge 0&#36; Our reasoning only requires    &#36;r \ge -1&#36;</p>
<p>The essence of an effective market is that:<strong>If a deal can be made without effort, there's a risk, or we call it a arbitrage opportunity.</strong> So the so-called arbitrage is zero probability of loss, and the positive probability of earning money is that our model does not allow arbitrage to exist, and in real markets, as long as arbitrage opportunities exist, arbitrage transactions take place that make the opportunity disappear.</p>
<p>In the single-time dident tree model, to avoid arbitrage, we ask that
&#36;0&lt;d&lt;1+r&lt;I'm sorry, but I'm sorry.
We've assumed that stock prices are constant, so there's definitely a dollard.&gt;0&#36;。</p>
<p>The other two variants were derived from the absence of arbitrage, as explained below: If &#36;d\geqslant1+r&#36;And investors can start with zero wealth, at zero in the moment, borrowing from banks to buy shares. Even in the worst cases, the time 1 stock is worth enough to pay off bank loans, and the probability of remaining surplus is positive, which offers an opportunity for arbitrage.</p>
<p>On the other hand, if &#36;u\leqslant1+r&#36;, investors can sell short shares and will earn money for their investments. Even in the best case of equities, equity 1 would not exceed the value of investments in the currency market at any time, and stock 1 would be at a time when the probability of a strictly lower value than the value of investments in the currency market was positive, providing an opportunity for arbitrage.</p>
<p>Real stock changes are much more complicated than the dident tree model, but...</p>
<ul>
<li>The dident tree model clearly explains arbitrage and risk neutrality pricing</li>
<li>The multi-time double-knot tree model can effectively address the problem of continuous time.</li>
<li>He'll draw some important mathematical probabilities from behind.</li>
</ul>
<h4>Consider the European-looking options pricing.</h4>
<p>Consider a European-style view of the increase in options, which gives the holder a time to finalize the price &#36;K&#36;(c) The right to purchase a stock (but not an obligation). Assumptions &#36;{S} 1(T)&lt;K&lt;{S}<em>1(H)&#36;,如果时刻 1 的股票价格低于敲定价格 &#36;K&#36;,则期权无价值；如果时刻 1 的股票价格高于敲定价格&#36;K&#36;,则期权被实施，由此获利为 &#36;S</em>{1}(H)-K&#36;。因此，期权在时刻1 的价值为&#36;The question of pricing options ^&#36; is what value options are in a time of zero before the results of known moment 1.</p>
<p>Here's an example of the arbitrage pricing method for options.</p>
<p>Consider a single-time model, set &#36;S(0)=4,u=2,d=\frac{1}{2}&#36;, &#36;r=\frac{1}{4}&#36;, and &#36;S_1(H)=8&#36; and &#36;S_1(T)=2&#36;I'm sorry. The price of the European-based increase in options &#36;K=5&#36;.</p>
<p>And then we'll assume that our initial wealth is... &#36;X_0=1.20&#36;, buy in at the time of 0 &#36;\Delta_0=\frac12&#36;Shares. Because at 0, each stock price is four, we must use the initial wealth. &#36;X_0=1.20&#36;And borrow an extra 0.80. So we're in cash. &#36;-0.8&#36; At the moment, we have a cash position.&#36;(1+r)(X_0-\Delta_0S_0)=-1&#36;(i.e. our debt in the currency market is 1.</p>
<p>On the other hand, in time one, we will have value for...&#36;\frac12S_1(H)=4&#36; Or...&#36;\frac12S_1(T)=1&#36; The stock. Specifically, if the result of the coins being thrown is positive, at the time of the first, the portfolio value of our assets (stock and currency market accounts) will be:
&#36;&#36;X_1(H)=\frac12S_1(H)+(1+r)(X_0-\Delta_0S_0)=3&#36;&#36;
If it's the other way around,
&#36;&#36;X_1(T)=\frac12S_1(T)+(1+r)(X_0-\Delta_0S_0)=0&#36;&#36;</p>
<p>In any case, at the moment, the portfolio is equal to the value of the option, i.e.&#36;(S_1(H)-5)^+=3&#36; or
&#36;(S_1(T)-5)^+=0&#36;I'm sorry. We copied options through transactions in the stock and currency markets.</p>
<p>The initial wealth 1.2 required to replicate the above-mentioned portfolio is the zero-reciprocal price of the option at any time. If the option price is above 1.2, the person who sells the option can replicate the portfolio and put the excess money into the currency market, achieve a risk-free arbitrage and do not need to invest any principal.</p>
<p>If the option price is below 1.2, the investor should buy the option and set up a reverse investment strategy: half the stock, with two dollars to buy the option (less than 1.2 prices) and the money market. (c) Achieving risk-free arbitrage without the principal.</p>
<p>If the option price is 1.2, then neither the buying nor selling options have arbitrage, which is the principle of arbitrage.</p>
<p>Our discussion here is based on the following assumptions:</p>
<ul>
<li>Stock can be broken down into unlimited numbers and allowed to be empty.</li>
<li>Investments at the same and non-negative rates as borrowing</li>
<li>No transaction costs and borrowing risk</li>
<li>There are only two possible prices for the next stock.</li>
<li>No arbitrage principle (prove demand)</li>
</ul>
<p>They're almost in the real world.</p>
<p>In fact, based on the principle of SAT, it is easy to calculate the price of LTAG once the LTAG has been completed. Formulae
&#36;&#36;C+K=P+S_T&#36;&#36;
of which&#36;C&#36;It's about raising the option price. &#36;K&#36;It's a right price.&#36;P&#36;It's a drop in the options price. &#36;S&#36;It's the current price of the stock.</p>
<p>That's two in the building.&#36;T&#36;The same assets that are the same at the moment.</p>
<ul>
<li>Watching the increase in options+&#36;e^{-rT}K&#36;Cash (cash discounts were made)</li>
<li>Watching the options plus a share of the stock
Formulas are based on the principle of arbitrage.</li>
</ul>
<h4>Derivative certificates and their pricing model</h4>
<p>In the general single-time model, we define derivative securities as a security, and if the result of a coin-throwing is positive, the payment at the moment 1 is&#36;V_1(H);&#36;If the result of throwing a coin is the back, the payment at the moment is&#36;V_1(T)&#36;。</p>
<p>The European-based increase option is a special derivative.&#36;(S_{1}-K)^{+}&#36;I'm not sure. The other is the European-style drop option, which is paid at the moment of the event (&#36;K-S_1)^+&#36;, of which &#36;K&#36; For constant. The third is a forward contract, which is the value of the derivative at the moment.&#36;S_1-K&#36;</p>
<p>We're still using a copy strategy to determine the price of the derivatives.</p>
<p>Suppose our initial wealth is...&#36;X_0&#36;Buy at 0-O. &#36;\Delta_0&#36;Shares, cash position.&#36;X_0-\Delta_0S_0&#36;
So the combination value of the moment 1 is
&#36;&#36;X_1=\Delta_0S_1+(1+r)(X_0-\Delta_0S_0)=(1+r)X_0+\Delta_0[S_1-(1+r)S_0]&#36;&#36;</p>
<p>We want to choose.&#36;X_0&#36;and&#36;\Delta_0&#36;♪ Make&#36;X_1(H)=V_1(H)&#36;and&#36;X_1(T)=V_1(T)[&#36;Here.&#36;V_{1}(H)&#36;and&#36;V_1(T)&#36;For known values, the option is agreed, depending on the result of a coin-throwing; at 0, we know.&#36;V_1(H)&#36;and&#36;V_1(T)&#36;"the probability of the eventuality, but not knowing which of these two possibilities will become a reality."</p>
<p>Therefore, for reproduction purposes, it is necessary to:
&#36;&#36;X_0+\Delta_0\left(\frac1{1+r}S_1(H)-S_0\right)=\frac1{1+r}V_1(H)&#36;&#36;
&#36;&#36;X_0+\Delta_0\left(\frac1{1+r}\mathrm{S}_1(T)-\mathrm{S}_0\right)=\frac1{1+r}V_1(T)&#36;&#36;
It's a binary equation. <strong>Delta Plot</strong>
&#36;&#36;\Delta_0=\frac{V_1(H)-V_1(T)}{S_1(H)-S_1(T)}&#36;&#36;
&#36;X_0&#36; You can give it to me.
&#36;&#36;X_0=\frac1{1+r}[\tilde{p}V_1(H)+\tilde{q}V_1(T)]&#36;&#36;</p>
<p>The formula that is calculated allows the reproduction of the derivatives presented at the beginning of this section, and the price of the derivatives should be the funds required to replicate the certificates.&#36;X_0&#36; We call this formula the same.<strong>Risk neutral pricing formula</strong>。</p>
<p>The probability solves the result:
&#36;&#36;\tilde{p}=\frac{1+r-d}{u-d},\quad\tilde{q}=\frac{u-1-r}{u-d}&#36;&#36;
<strong>They're risk neutral probabilities, not real probabilities.</strong>It's just a tool to help us solve the equation.</p>
<h3>Multi-temporal dident tree model</h3>
<p>We now extend the study of the previous section to a multi-temporal period. Imagine throwing coins over and over again, if the result is positive, the stock price multiplied by the increase factor. &#36;u;&#36;As long as the throw is the back, the stock price multiplied by the drop factor.&#36;d&#36;I'm sorry. There are constant interest rates on the market in addition to stocks &#36;r&#36; - Money market assets. The only assumption for these parameters is that there are no arbitrage conditions
&#36;0&lt;d&lt;1+r&lt;u&#36;&#36;</p>
<p>With the definition of a multistage model, we can easily give the two-stage model's stock price.
&#36;&#36;S_2(HH)=uS_1(H)=u^2S_0,\quad S_2(HT)=dS_1(H)=duS_0,&#36;&#36;
&#36;&#36;S_2(TH)=uS_1(T)=udS_0,\quad S_2(TT)=dS_1(T)=d^2S_0&#36;&#36;</p>
<p>So we can easily give more stages of the model.</p>
<p>We're still studying the European way of looking at the future. He's at the moment. &#36;T&#36; Finalize and finalize the price for &#36;K&#36; The pricing of rights during this period needs to take into account the results of several previous coins.</p>
<p>Theoretically (<strong>Copying in multi-time dident tree model</strong>Consider a N-time two-knot asset pricing model, with &#36;0&lt;d&lt;1+r&lt;u&#36; and;
&#36;&#36;\tilde{p}=\frac{1+r-d}{u-d},\quad\tilde{q}=\frac{u-1-r}{u-d}&#36;&#36;
Set&#36;V_{N}&#36; A random variable (payment of derivative security at time N), depending on the first N-thrower
Coin Process&#36;\omega_1\omega_2{\cdots}\omega_N&#36;I'm sorry. by
&#36;&#36;V_n(\omega_1\omega_2\cdots\omega_n)=\frac{1}{1+r}[\widetilde{p}V_{n+1}(\omega_1\omega_2\cdots\omega_nH)+\widetilde{q}V_{n+1}(\omega_1\omega_2\cdots\omega_nT)]&#36;&#36;
It's retroverted by time. &#36;V_{n-1}...V_0&#36;  They're the price of the derivative.</p>
<p>The formula that you're going to use to show the number of shares you're going to buy and sell for a simulation of options.
&#36;&#36;\Delta_n(\omega_1\cdots\omega_n)=\frac{V_{n+1}(\omega_1\cdots\omega_nH)-V_{n+1}(\omega_1\cdots\omega_nT)}{S_{n+1}(\omega_1\cdots\omega_nH)-S_{n+1}(\omega_1\cdots\omega_nT)}&#36;&#36;</p>
<p><strong>The theorem also applies to path-dependent options, which can be priced only by introspection as required by the inference.</strong></p>
<h3>Model calculations</h3>
<h4>Think about a options option.</h4>
<p>Example &#36;S_0=4,u=2,d=\frac12&#36;, also assuming interest rate &#36;r=\frac{1}{4}&#36;, and &#36;\widetilde p=\tilde{q}=\frac12&#36;I'm sorry. Considering such a right of retroactivity, its payment at the moment 3 is:
&#36;&#36;V_3=\max_{0\leqslant n\leqslant3}S_n-S_3&#36;&#36;
And we can do the following pricing.</p>
<p>&#36;V_3&#36;There's a price.
&#36;&#36;00\
&amp;V_{3}(HHH)=&amp;&amp; S_3(HHH)-S_3(HHH)=32-32=0 \
&amp;V_{3}(HHT)=&amp;&amp; S_{2}(HH)-S_{3}(HHT)=16-8=8 \
&amp;V_{3}(HTH)=&amp;&amp; S_1\left(H\right)-S_3\left(HTH\right)=8-8=0 \
&amp;V_{3}(HTT)=&amp;&amp; S_1(H)-S_3(HTT)=8-2=6 \
&amp;V_{3}(THH)=&amp;&amp; S_3(THH)-S_3(THH)=8-8=0 \
&amp;V_{3}(THT)=&amp;&amp; S_{2}(TH)-S_{3}(THT)=4-2=2 \
&amp;V_{3}(TTH)=&amp;&amp; S_{0}-S_{3}\left(TTH\right)=4-2=2 \
&amp;V_{3}(TTT)=&amp;&amp; S_{0}-S_{3}\left(TTT\right)=4-0.50=3.50
\end{aligned}&#36;&#36;
反推&#36;V_2&#36;定价有
&#36;&#36;\begin{gathered}
V_{2}(HH)= \frac{4}{5}\biggl[\frac{1}{2}V_{3}(HHH)+\frac{1}{2}V_{3}(HHT)\biggr]=3.20 \text{.} \
V_{2}(HT)= \frac{4}{5}\biggl[\frac{1}{2}V_{3}(HTH)+\frac{1}{2}V_{3}(HTT)\biggr]=2.40 \
V_{2}(TH)= \frac{4}{5}\bigg[\frac{1}{2}V_{3}(THH)+\frac{1}{2}V_{3}(THT)\bigg]=0.8 \text{-} \
V_{2}(TT)= \frac{4}{5}\bigg[\frac{1}{2}V_{3}(TTH)+\frac{1}{2}V_{3}(TTT)\bigg]=2.20
\end{gathered}&#36;&#36;
反推&#36;V_1&#36;有
&#36;&#36;\begin{gathered}
V_{1}(H)= \frac{4}{5}\bigg[\frac{1}{2}V_{2}(HH)+\frac{1}{2}V_{2}(HT)\bigg]=2.24 \
V_{1}(T)= \frac{4}{5}\bigg[\frac{1}{2}V_{2}(TH)+\frac{1}{2}V_{2}(TT)\bigg]=1.20 \
\end{gathered}&#36;&#36;
得到期权的初始定价为
&#36;&#36;V_{0}= \frac{4}{5}\bigg[\frac{1}{2}V_{1}(H)+\frac{1}{2}V_{1}(T)\bigg]=1.376 &#36;&#36;</p>
<p>Of course, we can replicate the first-stage portfolio by buying the following stock.
&#36;&#36;\Delta_0=\frac{V_1(H)-V_1(T)}{S_1(H)-S_1(T)}=\frac{2.24-1.20}{8-2}=0.1733&#36;&#36;</p>
<h4>Simplifying the workings</h4>
<p>If you use the initial&#36;V(TT..HH)&#36;Think of space, and the results of the diagonal tree model will be there.&#36;2^n&#36;We calculated in hindsight that the index swelled was too terrible for a model that was too many stages.</p>
<p>But if only stock prices and corresponding derivatives payments were considered, the problem would be much simpler, with the three-stage model having four prices, actually for the didentary tree model.&#36;n&#36;The stage will be... &#36;n+1&#36;Middle stock prices. We can put the original.&#36;V(TT..HH)&#36;Change to&#36;v(s)&#36;of which&#36;s&#36;The result of the mapping is still the price of the derivative paper.</p>
<p>So for phase three,
&#36;&#36;V_{2}\left(\omega_{1} \omega_{2}\right)=\frac{2}{5}\left[V_{3}\left(\omega_{1} \omega_{2} H\right)+V_{3}\left(\omega_{1} \omega_{2} T\right)\right]&#36;&#36;
Change to
&#36;&#36;v_{2}(s)=\frac{2}{5}\left[v_{3}(2 s)+v_{3}\left(\frac{1}{2} s\right)\right]&#36;&#36;</p>
<p>In fact, the option is at any moment.&#36;n&#36;The price is only for the stock price.&#36;S_n&#36;It's not about throwing a coin. We can use it.&#36;v(S_n)&#36;Replace the original.&#36;V_(TT...HH)&#36;To calculate without prejudice to the results.</p>
<p>Similar approaches are still available in path-dependent options such as retrospect options, but cannot be simply applied directly. We need to add the character of the options to the formula that was based on price.</p>
<h2>Math Basis of Random Analysis</h2>
<h3>Dispersed time</h3>
<h4>Equities price discount process</h4>
<p>We've given you the average discount value of the current price of the stock for the future.
&#36;&#36;S_n(\omega_1\cdots\omega_n)=\frac1{1+r}[\widetilde{p}S_{n+1}(\omega_1\cdots\omega_nH)+\widetilde{q}S_{n+1}(\omega_1\cdots\omega_nT)]&#36;&#36;
And then with the desired symbol, it's reworded to
&#36;S n=\frac1{1+r}\matbb{E}<em>n[S</em>{n+1}]&#36;&#36;
两边除以&#36;(1+r)^n&#36;有
&#36;&#36;\frac{S_{n&#125;&#125;{(1+r)^{n&#125;&#125;=\widetilde{\mathbb{E&#125;&#125;<em>{n}\left[\frac{S</em>You know, I'm not sure if you're gonna be able to get a good job.
Economics means that for stocks that do not pay dividends, based on&#36;n&#36;Time message is correct.&#36;n+1&#36;The best estimate of the discount price of the stock at the moment is&#36;n&#36;Discount price at any given moment.</p>
<h4>Adaptation and Demolition</h4>
<p>But we're not here to discuss economics. In fact, such a process is a process.</p>
<p>Definition: For random variable sequences&#36;{M_{i&#125;&#125;,i = 0,1,2,...&#36;  Every one.&#36;M_n&#36;Just rely on the front&#36;n&#36;The result of the coins thrown out, we call this random process.<strong>Adapt to random processes</strong></p>
<p>Definition: If an adaptation to a random process&#36;M_n&#36;Satisfied
&#36;&#36;M_{n}= E_n[M_{n+1}]&#36;&#36;
It's called a process.<strong>Which means there's no premium on stocks.</strong></p>
<p>corresponding, if satisfied&#36;M_{n}\le E_n[M_{n+1}]&#36; And this sequence, which is an increasing trend, is called<strong>Downside</strong>;if satisfied&#36;M_{n}\ge E_n[M_{n+1}]&#36; And this is a sequence that's decreasing.<strong>Go to the top.</strong></p>
<p>The definition of Zen is one step ahead, but we can actually prove that
&#36;&#36;M_{n}= E_n[M_{m}]&#36;&#36;
of which&#36;m \ge n&#36; </p>
<p>And we know that the direct expectations of the Zhuang are constant.
&#36;&#36;M_{0}= E[M_{n}]&#36;&#36;</p>
<p>There is no upward or downward trend, but there are equities, so we generally think that equities are a top (if there is a downward trend) or a bottom (if there is an incremental trend)</p>
<p>But if we move from real probability to risk neutrality, the corresponding stock discount price is a bump. In essence, it is also because of the difference between risk neutrality and real probability,<strong>Risk neutrality probability&#36;u,d,r&#36;Calculated, is the probability of ensuring the principle of risk neutrality</strong></p>
<h4>Wealth process and derivatives discount prices</h4>
<p>We're continuing to return to the wealth process that we're looking at in the price-fixing model of the fork tree.
&#36;&#36;X_{n+1}=\Delta_nS_{n+1}+(1+r)(X_n-\Delta_nS_n)&#36;&#36;
He's also an adaptation process. We'll come to conclusions about the discounting process.&#36;\frac{X_{n+1&#125;&#125;{(1+r)^{n+1&#125;&#125;&#36;  It's a risk neutral probability measure, which is...
&#36;&#36;\frac{(1+r)^<em>n\bigg[\frac{X</em>{n+1&#125;&#125;{(1+r)^{n+1&#125;&#125;\bigg],\quad n=0,1,\cdots,N-1&#36;&#36;
并且可以自然的给出推论
&#36;&#36;\tilde{\mathbb{E&#125;&#125;\frac{X_n}{(1+r)^n}=X_0,\quad n=0,1,\cdotp\cdotp\cdotp,N&#36;&#36;</p>
<p>In fact, we can continue to infer that the discount price of the derivative paper is also a quail at a risk-neutral probability measure, i.e., a cylindrical.
&#36;&#36;\frac{V n}(+r)^<em>n\bigg[\frac{V</em>{n+1&#125;&#125;{(1+r)^{n+1&#125;&#125;\bigg],\quad n=0,1,\cdots,N-1&#36;&#36;</p>
<h3>Consequence time</h3>
<p>This should be the content of a random process, but it is not described and is used in random analysis and is therefore supplemented.</p>
<h4>Information and Streams</h4>
<p>Definitions: Establishment&#36;T&#36;is a constant number and &#36; {\mathscr{F}<em>t}</em>{t\in[0,T]}&#36;是一族&#36;\sigma&#36;-代数，若对&#36;\forall0\leqslant s\leqslant t\leqslant T&#36;,都有&#36;\mathscr{F}_s\subset\mathscr{F}_t&#36;,则称该族&#36;\sigma-&#36;代数 &#36;\left{\mathscr{F}<em>t\right}</em>{t\in[0,T]}&#36;is a stream/filtration</p>
<p><strong>That means that the flow/filtration is one family without loss.&#36;\sigma&#36;Algebra</strong></p>
<p>Most of the time, we'll take it.&#36;\mathscr{F}_{0} = { \phi, \Omega}&#36;  It's just plain.&#36;\sigma&#36;Algebra</p>
<p>Definitions: We call&#36;\left(\Omega,\mathcal{F},\mathbb{F}=\left{\mathcal{F}_{t};t\geqslant0\right},\mathbb{P}\right)&#36;A filter probability space, a filtered probability space.</p>
<p>Definitions: set (\Omega,\mathscr{F}, {mathscr{F}<em>t}</em>{t\in[0,T]},\mathbb{P})&#36; 是一过滤概率空间，并且&#36;{X_t}_{t\in[0,T]}&#36;是其上的一族随机变量。若对&#36;\forall t\in[0,T]&#36;,都有&#36;X t\mathscr{F} t&#36;, which is called random variables about whether the current is adapted.</p>
<h4>The Ma's process.</h4>
<p>Definitions: set (\Omega,\mathscr{F}, {mathscr{F}<em>t}</em>{t\in[0,T]},\mathbb{P})&#36;是一过滤概率空间，并且&#36;X t&#36; is the random process on it if</p>
<ul>
<li>&#36;X&#36;It's adapted.</li>
<li>For Any &#36;s\leq t&#36;Both.&#36;\mathbf{E}[X_t|\mathscr{F}_s]\geq(\leq)X_s&#36;I'm not sure.
And then, "Could"&#36;X&#36;Is lower (up) tungsten. If the random process &#36;X&#36; It's both a low and a high, but it's called &#36;X&#36; It's a twilight.</li>
</ul>
<p>Definitions: set (\Omega,\mathscr{F}, {mathscr{F}<em>t}</em>{t\in[0,T]},\mathbb{P})&#36;是一过滤概率空间，并且&#36;X=x t} t\in[0,T}&#36;isa random process on it. If</p>
<ul>
<li>&#36;X&#36;It's adapted.&#36;\forall t\in[0,T]&#36;Yes.&#36;X_t\in\mathscr{F}_t)&#36; 。</li>
<li>For Boundary Functions&#36;f&#36;, Existing Functions&#36;g&#36;♪ And make ♪&#36;\mathbf{E}[f(X_t)|\mathscr{F}_s]=g(X_s)&#36; Of which &#36;S&lt;I'm not gonna get you out of here.
And then, "Could"&#36;X&#36;It's the Marge Process. <strong>It's a much broader place than the Mall chain, but the basics are the same.</strong></li>
</ul>
<p>Definitions: For a random process defining real values in probability space&#36;{X(t),t\geq0}&#36;The secondary variation is also a random process and is recorded as &#36; [X]<em>t&#36;, defined as:
&#36;&#36;[X]<em>t=\lim</em>{|P|\to0}\sum</em>Other Organiser
I'm not sure.&#36;P&#36;Remove the area&#36;0,t&#36;All the divisions,&#36;|P|&#36;Equals&#36;P&#36;The maximum length of the sub-zone is defined by probability condensation.</p>
<p>We use symbols, normally.&#36;Q_{\Pi }&#36; Representation &#36;\Pi&#36; It's...&#36;\sum_{i=0}^{n-1}(X(t_{i+1})-X(t_i))^2&#36; That's right.
&#36;&#36;Q_{\Pi }=\sum_{i=0}^{n-1}(X(t_{i+1})-X(t_i))^2&#36;&#36;
The secondary variation is ultimately evidenced by the extraction of the limit and the request and exchange order.</p>
<h4>Brown Movement and its Nature</h4>
<h5>Definition of the Brown Movement</h5>
<p>In random processes, the Brown movement is defined as follows: if the random process W=&#36;{\mathbf{W}_{t},\mathbf{t}\geq0}&#36;Satisfy: </p>
<ul>
<li>&#36;\quad W_0=0&#36; </li>
<li>&#36;\quad W={W_t,t\geq0}&#36; It's a smooth, independent incremental process.</li>
<li><strong>For any of the flaileq s&lt;t&#36;,有&#36;W_t-W_s\sim N(0,(t-s))&#36;</strong>
The random process W is standard Brown movement.</li>
</ul>
<p>Remove the first 0-point, which is called Brown Movement.</p>
<p>At this point, we amend article 2 to read:<strong>&#36;W_t&#36;Independent incremental, for any&#36;0\le s \le t&#36;，&#36;W_t-W_s&#36;and&#36;\mathscr{F}_s&#36;Independence</strong></p>
<p>The modified nature is actually the same as the original, but it is more conducive to the discussion later on, and it is not proven here.</p>
<h5>Mean of Brown motion and related functions</h5>
<p>See<a href="/en/blog/2023/03/18/stochastic-process-basics-notes/">Random Process Basis</a> The Brown Movement/ Digital Characteristics section.</p>
<h5>Brown's a tweezer.</h5>
<p>Adaptation is not required, but only proof. Sex
&#36;&#36;00\
\mathbb{E} [W t\mathscr{F} s]&amp; =\mathbb{E}[W_t-W_s+W_s|\mathscr{F}_s] \
&amp;=\mathbb{E}[W_t-W_s|\mathscr{F}_s]+\mathbb{E}[W_s|\mathscr{F}<em>s] \
&amp;=\mathbb{E}[W_t-W_s]+W_s \
&amp;=W</em>{s}
\end{aligned}&#36;&#36;</p>
<h5>Brown movement is a Marge process.</h5>
<p>Adaptability does not need to be proven, but only by demonstrating marzipanity, i.e., by having a boundary function&#36;f&#36;, Existing Functions&#36;g&#36;♪ And make ♪&#36;\mathbf{E}[f(X_t)|\mathscr{F}_s]=g(X_s)&#36; </p>
<p>According to the Brown movement, there is a basic construct that has been established.
&#36;&#36;\mathbb{E}[f(W_t)|\mathscr{F}_s]=\mathbb{E}[f(W_t-W_s+W_s)|\mathscr{F}_s]&#36;&#36;
Let the expectations be ours.&#36;g(X_s)&#36;  Because...&#36;W_t-W_s\bot\mathscr{F}_s&#36;And w s\matscr{F}<em>I'm not gonna be late, s.
&#36;g(x)=\mathbb{E}[f(W t-W s+x)]=int</em>^infty}f(y+x)e^frac{y^2(t-s)}dy&#36;
I'm just a man of my own.&#36;g(x)&#36; So Browning is a Marion process.</p>
<h5>The Brown movement's second-time variation.</h5>
<p>Brown movement is on&#36;[0,T]&#36;The second difference in the top is...&#36;T&#36; We'll start to prove it.
&#36;&#36;Q_\Pi=\sum_{j=0}^{n-1}(W_{t_{j+1&#125;&#125;-W_{t_{j&#125;&#125;)^2&#36;&#36;
According to the definition of Brown Movement, we know that. &#36;W_{t_{j+1&#125;&#125;-W_{t_{j&#125;&#125;&#36; It's a zero-normal distribution.</p>
<p>Easy to calculate expectations and differences
&#36;&#36;\mathbb{E}[(W_{t_{j+1&#125;&#125; -W_{t_{j&#125;&#125;)^2]=\mathrm{Var}[W_{t_{j+1&#125;&#125;-W_{t_{j&#125;&#125;]=t_{j+1}-t_j&#36;&#36;</p>
<p>&#36;&#36;\begin{aligned}&amp;\mathrm{Var}[(W_{t_{j+1&#125;&#125;-W_{t_{j&#125;&#125;)^2]=\mathbf{E}[\left|(W_{t_{j+1&#125;&#125; -W_{t_{j&#125;&#125;)^2-(t_{j+1}-t_j)\right|^2]\&amp;=\mathbb{E}[(W_{t_{j+1&#125;&#125;-W_{t_{j&#125;&#125;)^4]-2(t_{j+1}-t_j)\mathbb{E}[(W_{t_{j+1&#125;&#125;-W_{t_{j&#125;&#125;)^2]+(t_{j+1}-t_j)^2\&amp;=3(t_{j+1}-t_j)^2-2(t_{j+1}-t_j)^2+(t_{j+1}-t_j)^2=2(t_{j+1}-t_j)^2\end{aligned}&#36;&#36;</p>
<p>We have, in sum,
&#36;&#36;\begin{gathered}
\mathbf{E}[\left(Q_\Pi-T\right)^2]=\mathbf{E}\left[\left(\sum_{j=0}^{n-1}\left(W_{t_{j+1&#125;&#125; -W_{t_{j&#125;&#125;\right)^2-T\right)^2\right] \
=\sum_{j=0}^{n-1}\mathbb{E}\left[\left|\left(W_{t_{j+1&#125;&#125;-W_{t_{j&#125;&#125;\right)^2-\left(t_{j+1}-t_j\right)\right|^2\right] \
=\sum_{j=0}^{n-1}\mathrm{Var}[\left(W_{t_{j+1&#125;&#125;-W_{t_{j&#125;&#125;\right)^2]=\sum_{j=0}^{n-1}2(t_{j+1}-t_j)^2 \
\leq\sum_{j=0}^{n-1}2\parallel\Pi\parallel(t_{j+1}-t_j)=2\parallel\Pi\parallel T\to0.\quad(\parallel\Pi\parallel\to0)
\end{gathered}&#36;&#36;</p>
<p>We usually write down the second variation of Brown's movement as
&#36;&#36;dW(t)dW(t)=dt&#36;&#36;</p>
<h2>Random analytical basis</h2>
<h3>Riemann - Steeljes points</h3>
<p>The points of Riemann-Stiltjes are the extension of Riemann's points, allowing us to match the original.&#36;x&#36;Convert the score to a logarithm&#36;g(x)&#36;The points, that's very helpful to us in understanding random points.</p>
<p>The basic form of the Riemann - Steeljes points is as follows:
&#36;&#36;\int_a^bf\left(x\right)dg\left(x\right)&#36;&#36;
It's natural, if...&#36;g(x)&#36;It's possible to direct the way to Riemann's points, but using basic partitioning and intellectual research is the essence of this section.</p>
<p>♪ Split up for sum
&#36; \begin{aligned}S pi(f)&amp;=\sum_{k=1}^nf(c_k)[g\left(x_{k+1}\right)-g\left(x_k\right)]\&amp;c k\in[x k,x k+1]\end{aligned} &#36;
If a value exists&#36;I&#36;, makes for any \\varepsilon&gt;0&#36;,都存在一个&#36;\delta&gt;0&#36;,使得对于任意满足&#36;|P|&lt;\delta&#36;的分割&#36;P&#36;,都有&#36;|I-S(f,g,P)|&lt;\varepsilon&#36;,则称&#36;I&#36;为&#36;f&#36;关于&#36;g&#36;在&#36;a,b on Lehman-Stiljes points and stated:
&#36;&#36;\int_a^bf\left(x\right)dg\left(x\right)&#36;&#36;</p>
<p>This is the basic definition of RS points, which is essentially a variation of the idea of division and of thought.</p>
<h3>Definition of Ito points</h3>
<p>For Positive Numbers&#36;T&#36; Let's study the size of the area.
&#36;&#36;\int_0^T\Delta(t)dW(t)&#36;&#36;
The basic elements of this are the Brown Movement.&#36;W(t)&#36;  Corresponding international currents&#36;\mathscr{F}_t&#36;  Accumulated Functions&#36;\Delta(t)&#36;What is needed is to adapt to random processes. The biggest problem we've ever had is that we've been through. <strong>The Brown movement itself cannot be about time differentials.</strong></p>
<p>So we want to start with the simplest random process, the simplest process.
&#36;&#36;&#36;1eta t=sum k\pepratorname=<em>{[t_k,t</em>{k\operatorname{+}1})}(t\operatorname{)}&#36;&#36;
这意味着，如果积分在某个局部进行就是常数被积，我们可以给出下面的推论
&#36;&#36;\begin{aligned}&amp;I\left(t\right)=\Delta{<em>0}(W_t-W</em>{t_0})=\Delta_0W_t,\quad0\leq t\leq t_1,\&amp;I\left(t\right)=\Delta_0W_{t_1}+\Delta_{t_1}(W_t-W_{t_1}),\quad t_1\leq t\leq t_2,\&amp;I\left(t\right)=\Delta_0W_{t_1}+\Delta_{t_1}(W_{t_2}-W_{t_1})+\Delta_{t_2}(W_t-W_{t_2}),\quad t_2\leq t\leq t_3.\end{aligned}&#36;&#36;
一般的，对于某个&#36;k&#36; 使得 &#36;t_{k}\le t \le t_{k+1}&#36;
&#36;&#36;I\left(t\right)=\sum_{j=0}^{k-1}\Delta_{t_j}\left(W_{t_{j+1&#125;&#125;-W_{t_j}\right)+\Delta_{t_k}(W_t-W_{t_k}).&#36;&#36;</p>
<p>For the general&#36;\Delta(t)&#36; We can choose a series of simple processes.&#36;\Delta_n(t)&#36; And the limit of the random fraction of this simple process is the Ito fraction of the original function.</p>
<p>It's also a random process, Ito points.&#36;t&#36;It's his time, it's the ceiling of the points.</p>
<h3>Nature of Ito points</h3>
<h4>Six properties of Ito 's points</h4>
<p>Set&#36;T&#36;It's positive.&#36;\Delta(t)&#36; is an adaptation random process that meets the above-mentioned characteristics, and Ito points&#36;\int_0^T\Delta(t)dW(t)&#36;
The following characteristics are present:</p>
<ul>
<li><strong>Continuity</strong>: as the ceiling of the points&#36;t&#36;function,&#36;I(t)&#36;. The path of the</li>
<li><strong>Adaptability</strong>: For each&#36;t,I(t)&#36;Yes &#36;\mathcal{F}(t)&#36;- It's measurable.</li>
<li><strong>Linear</strong>：&#36;\int_0^t(\alpha\Delta_s+\beta\Gamma_s)dW_s=\alpha\int_0^t\Delta_sdW_s+\beta\int_0^t\Gamma_sdW_s.&#36;</li>
<li><strong>- It's a twilight.</strong>: &#36; (I\left)}t\in[0,T]}text{about the world stream&#125;&#125;{\matscr{F}<em>t}</em>{\t\in[0,T] \text{is鞅</li>
<li><strong>Ito equidistance</strong>：&#36;\mathbb{E}[I^2(t)]=\mathbb{E}\bigg[\int_0^t\Delta_u^2du\bigg]&#36;</li>
<li><strong>Secondary variation</strong>：&#36;dI\left(t\right)\cdot dI\left(t\right)=\Delta_t^2dt&#36;</li>
</ul>
<p>We're doing a separate validation study of the gillness, Ito equidistance, secondary variation, the core point being in a combination of definitions.
&#36;&#36;I\left(t\right)=\sum_{j=0}^{k-1}\Delta_{t_j}\left(W_{t_{j+1&#125;&#125;-W_{t_j}\right)+\Delta_{t_k}(W_t-W_{t_k}).&#36;&#36;
Expand definition-based certification. As for the remaining three, it is very natural that no additional proof is required.</p>
<h4>It's a tweezer.</h4>
<p>By definition, we need to prove it.
&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;<em>=I(s).&#36;
If&#36;s,t&#36;The same division (simplified random process) has &#36;t k\leq s\leq t&lt;t</em>So we're using our bad thinking about Brown's movement.
&#36;&#36;I\left(t\right)-I\left(s\right)=\Delta_{t_k}\left(W_t-W_s\right).&#36;&#36;
Or use the same method of consolidation as the previous study of the Brown movement.
&#36;&#36;\begin{aligned}\matbb{E}[I(t)-I(s)\mascr{F}<em>s]&amp;=\mathbb{E}[\Delta</em>{t_k}(W_t-W_s)|\mathscr{F}<em>s]\&amp;=\Delta</em>{t_k}\mathbb{E}[(W_t-W_s)|\mathscr{F}<em>s]=\Delta</em>♪ The world is so full of shit ♪
So there's a cynicism.</p>
<p>If&#36;s&#36;and&#36;t&#36;Not in the same division, that is, there is.&#36;t_\ell&#36;and&#36;t_k&#36;, meet &#36;t ell&lt;t_k&#36;,且&#36;s\in[t_\ell,t_{\ell+1})&#36;和&#36;t\in[t_k,t_{k+1})&#36;,则对&#36;I(t)&#36;-distorted
&#36;&#36;00begin{aligned}I\left(t\right)&amp;=\sum_{j=0}^{\ell-1}\Delta_{t_j}\left(W_{t_{j+1&#125;&#125;-W_{t_j}\right)+\Delta_{t_\ell}\left(W_{t_{\ell+1&#125;&#125;-W_{t_\ell}\right)\&amp;+\sum_{j=\ell+1}^{k-1}\Delta_{t_j}\left(W_{t_{j+1&#125;&#125;-W_{t_j}\right)+\Delta_{t_k}\left(W_t-W_{t_k}\right)\end{aligned}&#36;&#36;
继续对此拆分
&#36;&#36;\begin{aligned}I\left(t\right)&amp;=\sum_{j=0}^{\ell-1}\Delta_{t_j}\left(W_{t_{j+1&#125;&#125;-W_{t_j}\right)+\Delta_{t_\ell}\left(W_s-W_{t_\ell}\right)+\Delta_{t_\ell}\left(W_{t_{\ell+1&#125;&#125;-W_s\right)\&amp;+\sum_{j=\ell+1}^{k-1}\Delta_{t_j}\left(W_{t_{j+1&#125;&#125;-W_{t_j}\right)+\Delta_{t_k}\left(W_t-W_{t_k}\right)\end{aligned}&#36;&#36;
继续做差有
&#36;&#36;I\left(t\right)-I\left(s\right)=\Delta {t m}\lt( }t }-W s\right)+sum j=m+ll+k\;\Delta t t j}\left(t j+1}m}-W t j}+right)+Delta t k}\left(W t-W k}&#36;&#36;&#36;&#36;&#36;&#36;
Or is the condition of use expected to prove zero.</p>
<h4>Ito equidistance</h4>
<p>We want proof.
&#36;&#36;\mathbb{E}[I^2(t)]=\mathbb{E}\bigg[\int_0^t\Delta_u^2du\bigg]&#36;&#36;
Rewrite the points in front&#36;D_j=W_{t_{j+1&#125;&#125;-W_{t_j}&#36; and &#36;D_k=W_t-W_{t_k}&#36; And there is.&#36;I\left(t\right)=\sum_{j=0}^{k-1}\Delta_{t_{j&#125;&#125;D_{j}&#36; And then, the original expectation could be reworded to
&#36;&#36;00begin{aligned}\matbb{E}[I^2(t)]&amp;=\mathbb{E}\left[\left(\sum_{j=0}^{k-1}\Delta_{t_j}D_j\right)^2\right]\&amp;=\sum_{j=0}^{k-1}\mathbb{E}\left[\Delta_{t_j}^2D_j^2\right]+2\sum_{0\leq i&lt;j\leq k}\mathbb{E}\left[\Delta_{t_i}D_i\Delta_{t_j}D_j\right]\end{aligned}&#36;&#36;
由于独立性和布朗运动性质，交叉项期望一定为0 则
&#36;&#36;\begin{aligned}\mathbb{E}[I^2(t)]&amp;=\sum_{j=0}^{k-1}\mathbb{E}\left[\Delta_{t_j}^2D_j^2\right]=\sum_{j=0}^{k-1}\mathbb{E}\left[\Delta_{t_j}^2\right]\mathbb{E}\left[D_j^2\right]\&amp;=\sum_{j=0}^{k-1}\mathbb{E}[\Delta_{t_j}^2(t_{j+1}-t_j)]+\mathbb{E}[\Delta_{t_k}^2(t-t_k)]\&amp;=\mathbb{E}\bigg[\int_0^t\Delta_u^2du\bigg].\end{aligned}&#36;&#36;</p>
<h4>Secondary variation</h4>
<p>I want proof.
 &#36;&#36;dI\left(t\right)\cdot dI\left(t\right)=\Delta_t^2dt&#36;&#36;
Formally: &#36;I(t)=\int_{0}^{t} \Delta_{s} d W_{s}&#36; ♪ And so ♪  &#36;d I(t)=\Delta_{t} d W_{t}&#36; .</p>
<p>And so...  &#36;&#36;d I(t) \cdot d I(t)=\Delta_{t} d W_{t} \cdot \Delta_{t}  W_{t}=\Delta_{t}^{2} d W_{t} \cdot d W_{t}=\Delta_{t}^{2} d t&#36;&#36;
- That's it.</p>
<h3>Ito process and Ito formula</h3>
<h4>Ito Process</h4>
<p>The Ito process is a continuous random process that can be described by means of random differential equations. Specifically, an Ito process can be described as:
&#36;&#36;dX_t=\Theta_tdt+\Delta_tdW_t&#36;&#36;
And there's a credit form for it.
&#36;&#36;X_t=X_0+\int_0^t\Theta_sds+\int_0^t\Delta_sdW_s.&#36;&#36;
of which&#36;W_t&#36;It's Brown. The rest is not considered.&#36;dt&#36;It's a drift item.&#36;dW_t&#36;Formerly the diffusion item, referred to as drift and diffusion coefficients, respectively.</p>
<p>The secondary variation is:
&#36;&#36;dX_tdX_t=\Delta_t^2dt&#36;&#36;</p>
<h4>Ito Formula&#39;s Formula）</h4>
<p>The Ito formula is the core tool in random calculus that allows us to calculate the differentials of a random process function, set&#36;X_t&#36;It's an Ito process.&#36;f(x)&#36;It's a second-order micro-function, then.&#36;f(X_t)&#36;It's also an Ito process. Think about the binary.&#36;f(T,X_T)&#36; </p>
<p>So we can give Ito the formula that
&#36;&#36;00\
&amp;\begin{aligned}&amp;f\left(T,X_T\right)=f\left(0,X_0\right)+\int_0^Tf_t(t,X_t)dt+\int_0^Tf_x(x,X_t)dX_t\end{aligned} \
&amp;+\frac12\int_0^Tf_{xx}(x,X_t)dX_tdX_t \
&amp;=f\left(0,X_0\right)+\int_0^Tf_t(t,X_t)dt+\int_0^Tf_x(x,X_t)\Delta_tdW \
&amp;+ \Tf x,X t\tta tdt+\frac12\int 0 ^Tf xx}(t,X t)\Delta t^2dt.
I'm sorry, I'm sorry.
This is actually the form of the Newton-L formula in random analysis, where we seek to deflect and then to add a second-stage guide and secondary variation to the end, then to use the secondary variation nature of the Ito process and to define it in a simplified way.</p>
<p>There are some sort of scoring forms that give Ito formulas.
&#36;&#36;00\
df\left(t,X t}\right)&amp; =f_t(t,X_t)dt+f_x(x,X_t)dX_t+\frac12f_{xx}(x,X_t)dX_tdX_t \
&amp;=f_t(t,X_t)dt+f_x(x,X_t)\Delta_tdW_t+f_x(x,X_t)\Theta_tdt \
&amp;+\frac12f_{xx}(t,X_t)\Delta_t^2dt.
\end{aligned}&#36;&#36;</p>
<p>Formally, the Ito formula is the traditional full-microform formula that adds a secondary variation.</p>
<h4>One example</h4>
<p>We have.
&#36;&#36;X_t=\int_0^t\sigma_sdW_s+\int_0^t\biggl(\mu_s-\frac12\sigma_s^2\biggr)ds&#36;&#36;
And...&#36;S_t=e^{X_t}&#36; Please.&#36;S_t&#36;Satisfactory Equation</p>
<p>I'm not sure if I'm going to be able to get a job.&#36;X_t&#36;It's the Ito process. We use the Ito formula and we use the usual calculus format.
&#36; \begin{aligned}dS t&amp;=df\left(X_t\right)=f&#39;(X_t)dX_t+\frac12f&#39;&#39;(X_t)dX_tdX_t\&amp;=\left(\mu_t-\frac12\sigma_t^2\right)S_tdt+\sigma_tS_tdW_t+\frac12\sigma_t^2S_tdt\&amp;=\mu_tS_tdt+\sigma_tS_tdW_t.\end{aligned}&#36;&#36;</p>
<h3>Black-Scholes-Merton formula</h3>
<h4>BS formula definition</h4>
<p>After the previous discussion, we'll consider the end of this chapter.<strong>Black-Scholes-Merton equation</strong> This is an equation that is very important in the financial field.</p>
<p>Equities prices&#36;S_t&#36; It's geometry Brown.
&#36;&#36;dS_t\boldsymbol{=}\mu_tS_tdt\boldsymbol{+}\sigma_tS_tdW_t.&#36;&#36;
The investment strategy is set to&#36;\Delta_t&#36;  In combination with the "Single Time Two-Fork Model" section, there are wealth processes that
&#36;&#36;X_t=\Delta_tS_t+(X_t-\Delta_tS_t)&#36;&#36;
Keep the previous portion as an equity account and the following as a bank account&#36;D_t&#36; ; for the former, the scoring is only right.&#36;S_t&#36;Micro-diphorization, for the latter &#36;dD_{t}= rD_{t}&#36; This is the continuous process of compounding. And the wealth process is nuanced.
&#36;&#36;dX_t\boldsymbol{=}\Delta_tdS_t\boldsymbol{+}r\left(X_t\boldsymbol{-}\Delta_tS_t\right)dt&#36;&#36;
Research on the process of disbursing wealth
&#36;&#36;e^{-rt}X_t&#36;&#36;
The European way of looking at the increase of options is satisfied.
&#36;&#36;C_t(K,T;S)=c\left(t,S_t\right),\text{ 满足 }C_T(K,T;S)=(S_T-K)^+&#36;&#36;
I'm copying it on the principle of arbitrage.
&#36;&#36;e^{-rt}X_t=e^{-rt}c\left(t,S_t\right)&#36;&#36;
That's right.
&#36;&#36;d\left[e^{-rt}X_t\right]=d\left[e^{-rt}c\left(t,S_t\right)\right]&#36;&#36;
Now we just have to split the two sides of the equation according to the Ito formula and get the BS equation.
&#36;&#36;00\&amp;c_t(t,x)+rxc_x(t,x)+\frac12\sigma^2x^2c_{xx}(t,x)=rc\left(t,x\right)\&amp;c\left(T,x\right)=(x-K)^+\end{aligned}&#36;&#36;</p>
<h4>BS formulae calculation and nature</h4>
<p>All calculations are based on Ito formulas, and the proof here is actually just some calculations.</p>
<p><strong>Consider discounting the discounting of the equity.</strong> That's...&#36;d(e^{- rt}S(t))&#36; He is not necessary, but he can help us understand some of these calculations.</p>
<p>Preset formula
&#36;&#36;00\
d (e^-rt}S(t))&amp; =df(t,S(t)) \
&amp;=f_{t}(t,S(t))dt+f_{x}(t,S(t))dS(t)+\frac{1}{2}f_{x}(t,S(t))dS(t)dS(t)
\end{aligned}&#36;&#36;
代入计算，结合前一节的定义
&#36;&#36;\begin{aligned}&amp;=-re^{-rt}S(t)dt+e^{-rt}dS(t)\&amp;=(\mu-r)e^{-rt}S(t)dt+\sigma e^{-rt}S(t)dW(t)\end{aligned}&#36;&#36;</p>
<p><strong>Consider discounting the process of wealth.</strong>
&#36;&#36;\begin{aligned}
d(e^{-rt}X(t))&amp; =df(t,X(t)) \
&amp;=f_{t}(t,X(t))dt+f_{x}(t,X(t))dX(t)+\frac{1}{2}f_{xx}(t,X(t))dX(t)dX(t) \
&amp;=-re^{-rt}X(t)dt+e^{-rt}dX(t) \
&amp;={\Delta(t)}({\mu}-r)e^{-rt}S(t)dt+\Delta(t)\sigma e^{-rt}S(t)dW(t) \
&amp;&amp; \
&amp;=Delta(t)d (e^-rt}S(t))
I'm sorry, I'm sorry.
Changes in discounting process prices are fully determined by changes in discounting stock prices</p>
<p><strong>Consider discount options.</strong>
&#36;&#36;\begin{aligned}&amp;=e^{-rt}\left[-rc(t,S(t))+c_{t}(t,S(t))+\mu S(t)c_{x}(t,S(t))+\frac{1}{2}\sigma^{2}S^{2}(t)c_{xx}(t,S(t))\right]dt\&amp;+e^{-rt}\sigma S(t)c_{x}(t,S(t))dW(t)&amp;\end{aligned}&#36;&#36;</p>
<p>Re-incorporate.
&#36;&#36;d\left[e^{-rt}X_t\right]=d\left[e^{-rt}c\left(t,S_t\right)\right]&#36;&#36;
Based on&#36;dW_t&#36;Equal access
&#36;&#36;\Delta(t)=c_{x}(t,S(t))&#36;&#36;
It's called delta.</p>
<p>Reorder.&#36;dt&#36;The coefficient equals the conclusions given in the previous section.
&#36;&#36;00\&amp;c_t(t,x)+rxc_x(t,x)+\frac12\sigma^2x^2c_{xx}(t,x)=rc\left(t,x\right)\&amp;c\left(T,x\right)=(x-K)^+\end{aligned}&#36;&#36;</p>
<h4>Three Symbols</h4>
<p>We put the function&#36;c(t,x)&#36;The guides on the variables have some special names to use in some real financial scenes.</p>
<p><strong>delta</strong>： &#36;c_x(t,x)&#36;</p>
<p><strong>theta</strong>：&#36;c_t(x,t)&#36;</p>
<p><strong>gamma</strong>：&#36;c_{xx}(x,t)&#36;</p>
<p>The name is seen in other financial books.</p>
<h2>Status price</h2>
<p>In this section, “The Two-Fork Tree No-Girl Pricing Model”, we look at both the probability of being neutral and the probability of being real. The latter is real and is the probability in the real world. The former are fictional and are structured according to indicators of changes in assets such as interest rates. But it's good for many of the results.</p>
<p>This is essentially two probability measurements in the same probability space, and here we have more discussion of the problem in finance, which is rooted in the fact that the financial sector is a very important source of probability.<a href="/en/blog/2024/10/09/advanced-probability-notes/">High probability theory</a> The "Radon-Nikodym Theory" section</p>
<h3>Meteration Change</h3>
<p>Consider general limited sample space &#36;\Omega&#36; Two probability measurements above&#36;\mathbb{P}&#36; and&#36;\widetilde{\mathbb{P&#125;&#125;&#36;I'm sorry. Assumptions&#36;\mathbb{P}&#36; and&#36;\widetilde{\mathbb{P&#125;&#125;&#36; Yeah. &#36;\Omega&#36;
Each of these elements gives a positive probability, so we can write the following:
&#36;&#36;Z(\omega)=\frac{\widetilde{\mathbb{P&#125;&#125;(\omega)}{\mathbb{P}(\omega)}&#36;&#36;
<strong>&#36;Z&#36;It's a random variable.</strong>, because it relies on random tests&#36;\omega&#36;。&#36;Z&#36;It's called&#36;\hat{\mathbb{P&#125;&#125;&#36;About&#36;\mathbb{P}&#36;The number of the Radon-Nikodim guide. This definition and<a href="/en/blog/2024/10/09/advanced-probability-notes/">High probability theory</a> The “Radon-Nikodym Theorem” section is the same and a more narrow expression.</p>
<p>Although in limited sample space&#36;\Omega&#36;In the case of a commercial rather than a guided number. Random Variables&#36;Z&#36;With three important characteristics, we express them as the indemnity below.</p>
<p>Theorem: Set&#36;\mathbb{P}&#36; and&#36;\tilde{P}&#36; It's limited sample space. &#36;\Omega&#36; Two probability measurements, whatever.
&#36;omega\omega, \\mathbb{P}\left(\omega\right)&gt;0,\tilde{\mathbb{P&#125;&#125;\left(\omega\right)&gt;0&#36;,定义随机变量&#36;Z&#36; as above. We have:</p>
<ul>
<li>&#36;\mathbb{P}(Z&gt;0)=1;&#36;</li>
<li>&#36;EZ=1&#36;,</li>
<li>For Any Random Variables&#36;Y&#36;, by:&#36;\tilde{\mathbb{E&#125;&#125;Y=\mathbb{E}\begin{bmatrix}ZY\end{bmatrix}&#36;</li>
</ul>
<p>Thinking about our problem? <strong>The pricing formula, which uses a risk neutral probability measure to calculate the full course, does not take into account the probability of a real event, which must be problematic.</strong> To solve this problem, we need to use the RN guide between two measurements to help us weight, which is what the section is about.</p>
<p>Definitions: consideration &#36;N&#36; Time-scale dident tree model with real probability measure &#36;P&#36;, the risk neutral probability measure is&#36;\hat{\mathbb{P&#125;&#125;&#36;。&#36;Z&#36; Organisation&#36;\hat{\mathbb{P&#125;&#125;&#36;About &#36;P&#36; The Laton-Nicodim guide, which is:
&#36;&#36;Z(\omega_1\cdots\omega_N)=\frac{\tilde{\mathbb{P&#125;&#125;(\omega_1\cdots\omega_N)}{\mathbb{P}(\omega_1\cdots\omega_N)}=\left(\frac{\tilde{p&#125;&#125;p\right)^{#H(\omega_1\cdots\omega_N)}\left(\frac{\tilde{q&#125;&#125;q\right)^{#T(\omega_1\cdots\omega_N)}&#36;&#36;
Sequence&#36;\omega_1\cdots\omega_N&#36;, and then the number of times the back appears. Definitions<strong>State price density random variable</strong>Is:
&#36;&#36;\zeta(\omega)=\frac{Z(\omega)}{(1+r)^N}&#36;&#36;
And called&#36;\zeta(\omega)\mathbb{P}(\omega)&#36;Correspond to&#36;\omega&#36;It's...<strong>Status price</strong>。</p>
<p>Recalling the time frame for the risk neutral pricing formula in chapter II &#36;N&#36; Pays as&#36;V_{\mathrm{N&#125;&#125;&#36; Any derivative security at the time 0 is priced&#36;V_0=\widetilde{\mathbb{E&#125;&#125;\frac{V_N}{(1+r)^N}&#36;I'm sorry. Using state price density, it can simply be rewritten as:
&#36;&#36;V_0=\mathbb{E}\begin{bmatrix}\zeta V_N\end{bmatrix}=\sum_{\omega\in\Omega}V_N(\omega)\zeta(\omega)\mathbb{P}(\omega)&#36;&#36;</p>
<h3>RN Lead Process</h3>
<p>In the last section, we considered it.&#36;N&#36;The Laton Nicodim guide for the probability neutrality measure of risk in the time-frame diagonal tree model for real probability measurements. This random variable&#36;Z&#36;It depends on the model. &#36;N&#36; The result of the coins thrown.</p>
<p>To get the corresponding random variable that depends on less coin-throwing, we can base our time on &#36;n.&lt;\mathbb{N}&#36;的信息，对&#36;Z&#36;进行估计（就是条件期望）。这种估计在其他场合也会出现，因此以下我们给出一个一般的结论，其中并不要求&#36;Z&#36; is the Latung-Nicodim guide.</p>
<p>Theorem: Set &#36;Z&#36; Yes. &#36;N&#36; A random variable in the time-frame dident tree model, defined as:
&#36;&#36;Z_n=\mathbb{E}_nZ,\quad n=0,1,\cdotp\cdotp\cdotp,N&#36;&#36;
then&#36;Z_n,n=0,1,\cdotp\cdotp\cdotp,N&#36;At real probability.&#36;\mathbb{P}&#36;The next is a twig.</p>
<p>Still stick to the original symbol, define the RN guide process
&#36;Z n=\mathbb{E}<em>[Z], \\quad =0,1, \\cdotp\cdotp, &#36;
Naturally, there's &#36;Z.</em>{N}=z,z_0=1&#36;</p>
<p>So the derivative paper in front of you is...&#36;n&#36;Fix Price to
&#36;&#36;V_n=\tilde{\mathbb{E&#125;&#125;_n\frac{V_N}{(1+r)^{N-n&#125;&#125;=\frac{(1+r)^n}{Z_n}\mathbb{E}_n\frac{Z_NV_N}{(1+r)^N}=\frac1{\zeta_n}\mathbb{E}_n\begin{bmatrix}\zeta_NV_N\end{bmatrix}&#36;&#36;
Where the state density price is the process
&#36;&#36;\zeta_n=\frac{Z_n}{(1+r)^n},\quad n=0,1,\cdotp\cdotp\cdotp,N&#36;&#36;</p>
<h2>Change of unit of measure</h2>
<p>The price of assets is generally measured in the currency of a country, for the purpose of facilitating financial investment or simplifying models, and we often need to convert units of account, which leads to problems that deserve to be discussed.</p>
<p>In fact, when the unit changes,<strong>Risk neutral probability measurements must be changed, otherwise the avalanche of a large number of processes will be lost.</strong>  RN Conduit Formula
&#36;&#36;\frac{D(t)N(t)}{N(0)}=\widetilde{\mathbb{E&#125;&#125;\left[\frac{D(T)N(T)}{N(0)}|\mathcal{F}(t)\right],0\leqslant t\leqslant T&#36;&#36;</p>
<p>of which&#36;D(t)&#36;It's a discount process.  &#36;N(t)&#36;It's the price process of an asset.</p>
<h2>Assets dependent on interest rates</h2>
<p>In this chapter, we look at assets whose value depends on interest rates, which we call fixed-income assets, the most typical of which is a zero-interest bond.</p>
<p>We define the Zero Debt Roll.<strong>Rate of return</strong>Yes
&#36;&#36;\text{零息债券价格}=\text{面值}\times e^{-\text{收益单}\times\text{存续期&#125;&#125;&#36;&#36;
For each due date, we have a zero-interest bond. Draws a curve of the yield of the Zero Debt Bills at different maturity dates, which we call<strong>yield curve</strong></p>
<p>This means that a large number of tradable assets exist, and it is our intention to avoid arbitrage. Models for studying this type of problem are called<strong>Term Structure Model</strong></p>
<p>The classic model of interest rate dependency includes <strong>Interest rate fork tree model</strong> <strong>Forward contracts and futures model</strong> <strong>Multifactor simulation yield model</strong> <strong>HJM model</strong> <strong>Forward LIMBOR model</strong> The introduction is omitted here.</p>
