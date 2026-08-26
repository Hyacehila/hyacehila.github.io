---
title: 'Elementary Probability: Random Events, Probability Models, and Random Variables'
title_zh: 初等概率论：随机事件、概率模型与随机变量
date: 2023-03-18 21:28:02 +0800
categories:
- Data Science
- Probability & Statistical Foundations
tags:
- Statistics
- Probability
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers random events, probability models, random variables, common distributions, numerical characteristics, laws
  of large numbers, and central limit theorems.
description: Covers random events, probability models, random variables, common distributions, numerical characteristics,
  laws of large numbers, and central limit theorems.
excerpt_zh: 整理随机事件、概率模型、随机变量、常见分布、数字特征、大数定律和中心极限定理。
permalink: /blog/2023/03/18/elementary-probability-notes/
lang: en
translation_key: 2023-03-18-elementary-probability-notes
translation_status: machine
translation_source_hash: f987da624f92da7ba3c8982712ab1e2067936cac6e24686f02d318ebe6fef992
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Random event and probability</h2>
<h3>Random phenomena and statistical patterns</h3>
<p>Starting with the probabilistic theory, we started looking at random questions that were different from the mathematical branches of his previous studies. And of course, we're here to use a lot of analytical tools and meta-mathematics to study random issues.</p>
<h4>Random phenomena</h4>
<p>(a) The inevitable event must occur, and it is unlikely that it will never occur under certain conditions; among them, the event may or may not occur at random when the basic conditions remain unchanged;</p>
<h4>Frequency stability</h4>
<p>In large-scale experiments, sometimes frequency.
&#36;&#36;F_{N}(A)=n/N&#36;&#36;
That means random events aren't random, he has one. <strong>Statistical regularity</strong> And we have the concept of probability. <em>A irrelevant experiment. The inherent value of the system.</em></p>
<h4>Frequency and probability</h4>
<p>They have some nature and connections.</p>
<ul>
<li>Non-negative &#36;F_{N}(A)\ge 0&#36;</li>
<li>Inevitability and impossibility &#36;F_{N}(A)=0 不可能事件，F_{N}(A)=1必然事件&#36;</li>
<li>Addability of frequency &#36;F_{N}(A+B) = F_{N}(A)+F_{N}(B) 对于不同时发生的AB成立&#36;</li>
<li>The nature of a large number of theorem guarantees &#36;N\longrightarrow \infty~F_{N}(A)\longrightarrow P_{N}(A)&#36;
Probability is the measure of the probability of an event.
The frequency is the result of the experiment.</li>
</ul>
<h3>Sample space and events</h3>
<p>Here we continue to explain some of the basic concepts of probabilities.</p>
<h4>Sample space</h4>
<p>Experiments. Trail is a necessary study of random phenomena.
Sample point is a possible result of the experiment
Sample space is the entire sample point
The source of the earthquake. &#36;(x,y,z)&#36; Sample space, 3D area <em>It doesn't matter the number of sites in the sample space.</em>
<em>It's very clear that studying what kind of sample space depends on our problem, and there's some important sample space in probabilistic theory, but when it comes to practical problems, it's mathematical modelling.</em></p>
<h4>Events</h4>
<p>Event event is the assembly of some sample points, and at this point, for each event, we can judge whether a sample point is the event. Medium
Obviously. &#36;\Omega 整个样本空间 是一个必然事件；\emptyset 是一个不可能事件&#36;</p>
<h4>Operation of events</h4>
<p>How to calculate an event using the probability of a simple event to study the probability of a complex event is a question of probabilities.
&#36;A\subset B&#36; All sample points for A are in B.
&#36;A\subsetB<del>and</del>B\subset A&#36; 2 events equivalent
&#36;\bar{A}&#36;  The opposing event of the A, all of the sample points not in A.
&#36;A\cup B~~A\cap B&#36; The combination of events means that both occur simultaneously and that one of them is more than the other.
&#36;A\cap B=AB=\emptyset&#36; AB can't happen at the same time.&#36;A\cup B=A+B&#36;
&#36;A-B&#36; A, but not B.
Demorgen Law&#36;&#36;(\bigcup_{i=1}^n A_i)^c = \bigcap_{i=1}^n A_i^c~~~(\bigcap_{i=1}^n A_i)^c = \bigcup_{i=1}^n A_i^c&#36;&#36;
Operation laws
&#36;&#36;A\cup B=B\cup A~~~AB=BA~~~(A\cup B)\cup C=A\cup (B\cup C)~~~(A\cup B)\cap C=AC\cup BC&#36;&#36;
It's obvious that this assembly theory is very close, and we'll give him a more detailed explanation in future studies, which is closely linked to the higher form of probabilities, the measurement theory.
Definitions:&#36;P(w_{1})+P(w_{2})+...+P(w_{n})=1&#36;
Definition: the probability of the event is the probability of each sample point
Limited sample space: a limited number of sample sites
Dispersed sample space: sample space in which the number of sample points can be quantified</p>
<h3>Classical profile</h3>
<p>Let's start with this kind of simpler probabilistic problem.</p>
<ul>
<li>The results of the experiment are limited and different. Hand it over.</li>
<li>Probability of all events
These random phenomena are called classical generals, which are exactly the same as the classical generals we studied in high school.
<em>The classical definition of probability, the definition of probabilistic theory.&#36;P(A)=m/n其中m是利场合数，n是样本点总数&#36;</em>
In fact, the classical model is because he does use it very widely, and we usually use it to study it, but its application goes far beyond the touch of the ball.</li>
</ul>
<h4>Basic combination analysis formula</h4>
<h5>Core principles</h5>
<p>Multiplication doctrine: serial &#36;n=n_{1}*n_{2}&#36;
Added: Parallel &#36;n=n_{1}+n_{2}&#36;</p>
<h5>Arranged questions</h5>
<p>Of the n elements, the r elements are selected for ranking, considering not only the elements to be taken, but also the order to be taken.</p>
<ul>
<li>There's a playback. &#36;n^{r}&#36;</li>
<li>No Releases &#36;A_{n}^{r}=n(n-1)(n-2)...(n-r+1)&#36;
It's quite understandable, when... &#36;n=r&#36;When it's full, it's full.&#36;n!&#36;</li>
</ul>
<h5>Cluster issues</h5>
<p>Of the n elements, the r elements are selected for grouping, the elements to be considered and the order of removal not considered</p>
<ul>
<li>R for n elements &#36;C_{n}^{m}=A_{n}^{r}/r!&#36;   It's also shown in the following form, so it's not an integer, it's also called a binary coefficient.
&#36;&#36;\left(\begin{array}{l}
n \
v
\end{array}\right)&#36;&#36;</li>
<li>The n elements are divided into k groups, the number of groups has been given, the number divided by &#36;n!/r_{1}!r_{2}!...r_{k}!&#36;
He's also known as the multiple coefficient.&#36;C_{n}^{m}&#36;Accumulated</li>
</ul>
<p>In fact, most of the grouping problems are deformations that are the most fundamental.  </p>
<h5>Some supporting formulae and definitions</h5>
<p>&#36;&#36;0!=1 &#36;&#36;
&#36;&#36;n&gt;~left (\begin{array}l)
I'm sorry.
k
=0 &#36;&#36;
&#36;&#36;\left(\begin{array}{l}
n \
k
\end{array}\right)=\left(\begin{array}{l}
n \
n-k
\end{array}\right)&#36;&#36;</p>
<h5>Extension to non-integer form</h5>
<p>[Mathematic Analysis 1 Limits and Continuity Theory]</p>
<h4>A classic example of classical generalization.</h4>
<p><a href="/en/blog/2024/09/24/probability-and-statistics-exercises-notes/">Examples of this section</a> "The Classic Example of the Classic Outline" section</p>
<h4>Two distributions and hypergeometric distributions</h4>
<p>&#36;a件次品，b件合格品，抽n件，研究k件不合格的概率&#36;
We consider it easier to consider differences between products;</p>
<p><strong>In case of re-entry.</strong>
General &#36;(a+b)!&#36;A favourable occasion for &#36;C n^k}<em>a^{k}<em>I'm sorry.
Probability is &#36;P=C n^k}</em>(a/(a+b))^{k}</em>(b)(a+b))^n-k}&#36;
He's a dichotomy extension, all of which is called the dichotomy.</p>
<p><strong>Without putting it back</strong>
&#36;&#36;总数\left(\begin{array}{l}
a+b \
n
\end{array}\right) 合适的样本点数目\left(\begin{array}{l}
a \
k
\end{array}\right)*\left(\begin{array}{l}
b \
n-k
\end{array}\right)&#36;&#36;
That's the way it's distributed. </p>
<p><strong>In fact, when the total number is very large but the number of samples is low, the results of the two profiles are close.</strong></p>
<h4>An example of mathematical statistics</h4>
<p><a href="/en/blog/2024/09/24/probability-and-statistics-exercises-notes/">Examples of this section</a> "Mathematical examples of probabilities" section</p>
<h4>Some nature</h4>
<p>&#36;&#36;P(A\cup B)=P(A)+P(B)-P(AB)&#36;&#36;
&#36;&#36;P(A)=1-P(\bar{A})&#36;&#36;
&#36;&#36;AB=\emptyset ~~~~P(A_{1}+A_{2}+...+A_{n})=P(A_{1})+P(A_{2})+...+P(A_{n})&#36;&#36;</p>
<h3>Geometry</h3>
<p>In the geometry section, we have identified the apparent limitation of the limited number of classical overview sample points and have decided to use geometry to address this situation;
<strong>The core of geometry is the scale.</strong></p>
<p><strong>Geometry tells us that the probability is zero and that the event will not take place at an equal price.</strong></p>
<h4>Simple geometry example</h4>
<p>Place a circle equal to the diameter in the square; the probability of a random drop point within the square, which falls within the circle, is the ratio of the size of the two.&#36;\frac{\pi}{4}&#36;  </p>
<h4>Buffoon.</h4>
<p>On the plane.&#36;a&#36;A parallel line, one for a long one.&#36;l&#36;The probability that needles, needles and parallel lines will intersect
Obviously, if it's used&#36;x&#36;The distance between the midpoint of the needle and the parallel line,&#36;\theta&#36;Angular angle, and our final intersection is &#36;x &lt;l/2*sin(\theta)&#36;,此时的&#36;P=\frac{2\pia}
If we do computer simulations, we can calculate the probability.&#36;\pi&#36;The value, then developed into a random simulation of statistics</p>
<h4>The Monte Carlo method of calculating points</h4>
<p>The ordinary Lehman is essentially a measure, and we can study him in size, while space can be obtained through random simulation experiments, with a plumb injection, which is the method of calculating points in Monte Carlo;
Although the method is very inefficient, the error is stable and is still widely applied.
We don't discuss this method in detail in our probabilistic research, but rather teach it where it's practice-oriented.</p>
<h4>Bertrand.</h4>
<p>The core of geometry is the construction of a measure of probability; this gives rise to some questions and subsequent mathematicians' thinking; Bertrandicism is due to a contradictory geometry problem that arises from a probability calibration, which we will study more in the high probability theory. Him.</p>
<h3>Probability of conditions</h3>
<p>The probability theory of conditions, which is studied in the case of event B, is recorded as the probability of event A occurring at this time.&#36;P(A|B)&#36;He's also an important part of probabilistic theory. We're here to study it.
Definitions: Under a given probability space &#36;P(A|B)=\frac {P(AB)}{P(B)}&#36; That's the probability formula.
<em>Obviously, both classical and geometric can easily be used.</em></p>
<h4>Nature of probability of conditions</h4>
<p><strong>Inferences</strong> ：<em>Looks like it's just a migration, but it's important, called a multiplication formula.</em>
&#36;&#36;P(A|B)*P(B)=P(AB)&#36;&#36;
&#36;&#36;非负性：P(A|B)\ge 0&#36;&#36;
&#36;&#36;规范性：P(\Omega|B)= 0 &#36;&#36;
&#36;&#36;可列可加性：AB=\emptyset ~~~~P(A_{1}+A_{2}+...+A_{n}|B)=P(A_{1}|B)+P(A_{2}|B)+...+P(A_{n}|B)&#36;&#36;
<strong>Inferences</strong>: Multiplication formula for promotion
&#36;&#36;P(A_{1}A_{2}...A_{n})=P(A_{1})*P(A_{2}|A_{1})*P(A_{3}|A_{1}A_{2})...P(A_{n}|A_{1}...A_{n-1})&#36;&#36;</p>
<h4>Full Probability Formula</h4>
<p>(a) The full probability formula is the probability of studying another thing by using the probability of something, which is closely linked to the probability of conditions;
&#36;&#36;P(B)=P(AB)+P(\bar{A} B)=P(A)P(B|A)+P(\bar{A})P(B|\bar{A})&#36;&#36;
<em>For some of the problems that are very difficult to address, there's an experiment that a previous experiment can refer to, and the full probability formula is a good choice, which is that his core should be, step by step.</em>
<em>The full probability formula is freely extended to more dollars.</em>
&#36;&#36;P(A)=\sum\limits P(AB_{i})=\sum P(B_{i})P(A|B{i})&#36;&#36;</p>
<h4>Bayes Formula</h4>
<p>The Bayes formula is about studying the connection between the two events, on which Bayes' decision-making and distinction are based.
If&#36;B&#36;Only incompatible.&#36;A_{i}&#36; It happens at the same time. &#36;B=\sum P(BA_{i})&#36;<br>&#36;&#36;P(A_{i}B)=P(A_{i}|B)*P(B)=P(A_{i})*P(B|A_{i})&#36;&#36;
And so...
&#36;&#36;P(A_{i}|B)=P(A_{i})*P(B|A_{i})/P(B)&#36;&#36;
So...
&#36;&#36;P(A_{i}|B)=\frac{P(A_{i})*P(B|A_{i})}{\sum\limits P(A_{i} )~P(B|A_{i})}&#36;&#36;
That's the Bayes formula. &#36;B&#36; In the event &#36;A_{i}&#36; Probability of events;
&#36;P(A_{i})&#36; It's a summary of past experience, which we call probabilistic data, which he learned before the experiment.
&#36;P(A_{i}|B)&#36; It's what we'd like to study, commonly known as the probability of a posteriori.
The Bayes formula is widely used for disease diagnosis, and at this point B means indicator A means disease; the probability of disease occurring under one indicator can be calculated using the Bayes formula, and the degree of confidence obtained in the calculation can be used by the doctor for reference purposes.</p>
<h3>Independence of events</h3>
<p>Just as we're looking at probability, we're looking at two things here.</p>
<h4>Independence of the two incidents</h4>
<p>Definitions:&#36;P(AB)=P(A)P(B)&#36; Combining the multiplication formula, which means the condition of probability is no longer working.
Inference: Independent events satisfied &#36;P(A|B)=P(A)&#36;
Inference: &#36;A<del>B independent, rule A</del>\bar{B&#125;&#125;,{\bar{A}<del>\bar{B&#125;&#125;,{A</del>I don't know.</p>
<h4>Independence of multiple events</h4>
<p>You have to meet the following conditions simultaneously.
&#36;P(AB)=P(A)P(B)<del>P(AC)=P(A)P(C)</del>P(BC)=P(C)P(B)~~P(ABC)=P(A)P(B)P(C)&#36;&#36;</p>
<p>Independence is a matter of high standards, and Bienstein's example shows this.</p>
<p>It's very natural that there's no need to give a clear description of the forms of independence of multiple events.</p>
<h4>Independence of the experiment</h4>
<p>From events to experiments, experiments are the core of probabilistic science.</p>
<p>If there's an experiment, and they have separate sample space, the total sample space is the calcium of the sample space.</p>
<p>At this point,&#36;P(A^{1}A^{2}...A^{n})=P(A^{1})P(A^{2})...P(A^{n})&#36;
We call these experiments independent.</p>
<p>The independence of the experiment explains what we've learned, but we don't know.
Repeated independent experiments will be a very common concept to learn from behind.</p>
<h3>Bernuli.</h3>
<p>In some cases, we focus only on two types of results, qualified and unqualified, which is the subject of the Benoligue study;
Consider repeating.&#36;n&#36;The next Benuli experiment required</p>
<ul>
<li>Up to two results per experiment.&#36;A~\bar{A}&#36; Probability and 1</li>
<li>&#36;P(A)&#36; Stabilization</li>
<li>The experiment is independent of each other.</li>
<li>Conduct&#36;n&#36;Minor experiments
Total number of points for such experiments&#36;2^{n}&#36;In fact, as the number of experiments draws closer, the number of sample points becomes the first infinity.
This is a very broad profile, such as the oversale of airline tickets, genetic problems, etc., which is consistent with the repeated 01 results. That's why we're here for a more detailed analysis.</li>
</ul>
<h4>Benuli distribution</h4>
<p>Just one experiment. The distribution column is very easy to write.</p>
<h4>Two distributions</h4>
<p>We have described the two distributions earlier. In fact, in the case of the extraction and return of the defective items, it's a one-one-time test, and that's a Benuli experiment, so we don't want to go on here too much; as for the probability of the two distributions, it's a very good thing.
&#36;&#36;b(k;n,p)=C_{n}^{k}p^{k}q^{n-k}&#36;&#36;</p>
<h4>Geometric distribution</h4>
<p>This is not a supergeometric distribution. The so-called geometric distribution is a study. The first successful experiment is in the first place.&#36;k&#36;Probability of repeats
And we can easily calculate the geometric distribution column.
&#36;&#36;g(k;p)=q^{k-1}p&#36;&#36;
<strong>Geometrically unrememberable</strong>
It's a very important feature of geometry; it means &#36;P (X)&gt;(m+n)|X&gt;m)=P(X&gt;That's it.
He means that no matter how many experiments have been done, it doesn't affect the probability behind it.</p>
<h4>Pascal distribution/negative distribution</h4>
<p>We're trying to expand the geometrical distribution of boundaries.&#36;k&#36;It's been a success.&#36;r&#36;It's not really hard to calculate.
&#36;&#36;f(k;r,p)=C_{r-1}^{k-1}p^{k-1}q^{k-r}p=C_{r-1}^{k-1}p^{k}q^{k-r}&#36;&#36;</p>
<h3>Two distributions and porcelain distribution</h3>
<p>We're here to study a few examples and a new distribution, which is closely linked to the two previous studies.</p>
<h4>A little more.</h4>
<p>&#36;&#36;\begin{aligned}
&amp;\frac{b(k;n,p)&#125;&#125;b(k+i,r,b)}=1+\frac{(n+1)p-k}{kq}\
I'm sorry.
That tells us.&#36;k=(n+1)p=m&#36;, at which time the probability of the two distributions is taken to the maximum;
Of course, because of the integer limit, we can only go to a close value.
This is what we call the center of two distributions.
Unexplained conclusions:&#36;P(m)=(2\pi npq)^{-\frac{1}{2&#125;&#125;&#36;</p>
<h4>A simple example.</h4>
<p>Assuming a total of 200 machine beds are powered at 1 kw each, with a 60 per cent probability of opening each other independently, the workshop is expected to be stable above 99 per 1,000.
It's obviously a probabilistic problem.
Calculating \\sum\limitesb(k,200,0.6)&gt;0.999&#36; 研究找到的&#36;That's all we need.
Now the problem is this huge distribution is hard to calculate.</p>
<h4>Two distributions approaching.</h4>
<p>In a lot of Benuli experiments, if&#36;n&#36;Large&#36;p&#36;They're small.&#36;\lambda&#36;It's more moderate in size and, in this case, Persson has found a more easy form to calculate.
&#36;&#36;当np接近\lambda，则在n\to \infty 时有
b(k;n,p)=p(k,\lambda)=\frac{\lambda^{k&#125;&#125;{k!}e^{-\lambda}&#36;&#36;
It's called theorem.</p>
<p>In fact, the porcelain distribution has found many uses.</p>
<ul>
<li>Web access (this is all the counting process in unit time)</li>
<li>Thermal electronic launch and microbiological distribution</li>
<li>Composition of other random phenomena
This has become an important stand-alone distribution rather than a double distribution calculation;</li>
</ul>
<h2>Random variables and their distribution</h2>
<p>By now we have concluded a small phase of our research, and from now on, the question of the probabilities of quantitatively uniform research is a very important one, which is random variables, and we will study in detail the dimensions of this chapter.</p>
<h3>Random variables and their distribution</h3>
<p>We call random variables as variables that represent random phenomena, how they describe the results of random phenomena, and how to further study practical issues using random variables, which is clear from our chapter;</p>
<h4>Concept of random variables</h4>
<p>Many of the sample points are expressed in a number and because of the randomity of the sample points, they are a random variable;</p>
<p><strong>Definitions</strong>: define real-value functions in sample space&#36;X=X(w)&#36; Called a random variable, usually expressed in capital letters and taken values in lowercase letters;</p>
<p>If the value of a random variable is limited or columnable, it is referred to as an discrete random variable or, if the value is full of space on a number of axes, as a continuous random variable;</p>
<h4>Distribution function for random variables</h4>
<p>In order to master the statistical regularity of a random variable, we need to know the probability of his going to values, and the probability of random variables is clearly cumulative, so we just need to know.&#36;F(x)=P{X\le x}&#36; That's it.&#36;F(x)&#36;   It's a definition.&#36;(-\infty,\infty)&#36; functions;</p>
<p><strong>Definitions</strong>: setting X is a random variable for any actual number&#36;x&#36; Claims&#36;F(x)=P{X\le x}&#36; It's random.&#36;X&#36;The distribution function called&#36;X&#36;Obey.&#36;F(x)&#36;</p>
<p>Random variables, whether continuous or discrete, have distributed functions</p>
<p>Theorem: Any distribution function&#36;F(X)&#36; Both have the following characteristics:</p>
<ul>
<li>&#36;F(X)&#36;It's a monotony.</li>
<li>&#36;F(X)&#36;The range of values is 0 to 1 in the closed range, and the limits of both ends are 0 and 1.</li>
<li>&#36;F(X)&#36;It's a right continuous function. &#36;\lim_{x \to x_{0}+0}F(x)=F(x_{0})&#36;</li>
</ul>
<p>The function that satisfies these three characteristics must be a distributed function</p>
<h4>Distribution column of discrete random variables</h4>
<p>For discrete random variables, we often use the following distribution column to express it accurately: It's...</p>
<p>Definitions: Establishment &#36;X&#36; is a discrete random variable, if&#36;X&#36; All possible values are  &#36;x_{1}, x_{2}, \cdots ,  x_{n}&#36; , or&#36;X&#36;  Remove  &#36;x_{i}&#36; Probability&#36;p_{i}=p\left(x_{i}\right)=P\left(X=x_{i}\right), i=1,2, \cdots, n&#36;Yes &#36;X&#36; the probability distribution column or abbreviation column, as  &#36;X \sim\left{p_{i}\right}&#36;</p>
<p>The distribution column can also be expressed as follows:
&#36;
\begin{array}ccccc}
X  &amp;  x_{1}  &amp;  x_{2}  &amp;  \ldots  &amp;  x_{n}  &amp;  \cdots  \
\hline P  &amp;  p\left(x_{1}\right)  &amp;  p\left(x_{2}\right)  &amp;  \cdots  &amp;  p\left(x_{n}\right)  &amp;  \cdots
\end{array}&#36;&#36;
或者
&#36;&#36;
\left(\begin{array}{ccccc}
x_{1} &amp; x_{2} &amp; \cdots &amp; x_{n} &amp; \cdots \
p\left(x_{1}\right) &amp; p\left(x_{2}\right) &amp; \cdots &amp; p\left(x_{n}\right) &amp; \cdots
\end{array}\right)&#36;&#36;</p>
<p>The distribution column clearly has two basic properties.</p>
<ul>
<li>Non-negative&#36;p(x_{i})\le 0&#36;</li>
<li>Regularity&#36;\sum p(x_{i})=1&#36;</li>
</ul>
<h4>Probability density function for continuous random variables</h4>
<p>The distribution column will certainly not be able to study a range of random variables, but imagine that the distribution column is intended to describe the probability of a single point and corresponds to the continuous, distributed function.&#36;F(X)&#36;And that's how it works, so...</p>
<p><strong>Definitions</strong>: set random variables&#36;X&#36;Distribution Functions&#36;F(X)&#36; There is a non-negative buildup function&#36;p(x)&#36; Satisfied&#36;\int_{-\infty}^{x} p(t)dt=F(X)&#36; Claims&#36;p(t)&#36;Yes&#36;X&#36;Probability density function</p>
<p>Probability density functions have two basic properties</p>
<ul>
<li>Non-negative&#36;p(x)\ge 0&#36;</li>
<li>Regularity&#36;\int_{-\infty}^{\infty}  p(x_{i})dx=1&#36;
In order to calculate probability from a density function, we have to look at it from the point of view of points that he cannot avoid.&#36;N-L&#36;The formula.</li>
</ul>
<p><strong>If the density is&#36;0&#36;Use&#36;N-L&#36;The formula must be taken out of this compartment when it takes down points.</strong> Because I'm here.&#36;0&#36;It must mean the existence of a non-continuous point.&#36;N-L&#36;I'm sure there's something wrong with the score.</p>
<h4>Comparison of two random variables</h4>
<p>Probability density functions and distribution columns are basically close, but there are still some differences, as we simply describe here.</p>
<ul>
<li>The distribution function of the discrete random variable is right continuous, but the distribution function of the continuous random variable is completely continuous</li>
<li>The discrete random variable has a zero probability at some points, but the probability of any single point of a continuous random variable is zero.<strong>Our probability is zero.</strong></li>
<li>Because the single point probability of a continuous random variable is zero, you're out. The removal of a few points should not affect; otherwise, the discrete random variable must be measured against each point to ensure that the end result is correct</li>
</ul>
<h3>A mathematical expectation for random variables</h3>
<p>Studying the characteristics of random variables, describing the characteristics of random variables as a whole in simple quantities is what we've always wanted to do, and that's the numerical characteristics of random variables, and we present the characteristics of one of the most important random variables, mathematical expectations;</p>
<h4>The concept of mathematical expectations</h4>
<p>We would like to know where this random variable is going in its entirety, and that's what mathematical expectations or averages want to answer.</p>
<p>arithmetical average: directly calculated average
Weighted average: taking into account the impact on overall trends of different numbers, depending on the frequency of occurrence
It is clear that in random variables, the use of probability as a power value is a very natural thing;</p>
<h4>Definition of mathematical expectations</h4>
<p>For discrete random variables, we call&#36;E(X)=\sum p(x_{i})x_{i}&#36; is the mathematical expectation of discrete random variables, if they are constricted;</p>
<p>It's because it's not the only way to avoid the expectations of non-inclusion, which we mention in the infinity of mathematical analysis; for random variables of a limited number, the expectations must be there; in fact, the expectations of Cauchy distribution are not there, and we make this additional requirement reasonable.</p>
<p>For continuous random variables, we call&#36;E(X)=\int_{-\infty}^{\infty}  x_{i}p(x_{i})dx&#36; It's the mathematical expectation of a continuous random variable.</p>
<h4>Nature of mathematical expectations</h4>
<h5>The mathematical expectations of the functions of random variables</h5>
<p>We know the mathematical expectations of random variables.&#36;E(X)&#36; The only thing that is identified in the distribution, and it is very clear that the random variable function is also a random variable, and in order to solve the mathematical expectations of the random variable function, we need to do the following research.
In the most basic context, we can study this further by calculating the distribution column of the new random variable, the probability density function, and in order to facilitate the next calculation, we give the following theory.
We have a distribution column for discrete random variables.&#36;p(x_{i})&#36;and Functions&#36;g(X)&#36;
Mathistic Expectations&#36;E{g(X)}=\sum g(x_{i})p(x_{i})&#36;
For continuous random variables, we have a probability density function.&#36;p(x)&#36;and Functions&#36;g(X)&#36;
Mathistic Expectations&#36;E{g(X)}=\int g(x)p(x)dx&#36;
It's natural for theorem to consider it directly intuitively.</p>
<h5>Several export properties</h5>
<ul>
<li>&#36;E(x)=c&#36;</li>
<li>&#36;E(aX)=aE(X)&#36;</li>
<li>&#36;E(g(x_{1})+g(x_{2}))=E(g(x_{1}))+E(g(x_{2}))&#36;</li>
<li>&#36;X,Y&#36;On my own.&#36;E(XY)=E(X)E(Y)&#36; </li>
<li>&#36;E(X+Y)=E(X)+E(Y)&#36;</li>
</ul>
<h3>The difference of the random variable</h3>
<p>The variance of the random variable is presented in order to study the size of the fluctuations of the random variable;</p>
<h4>Definition of variance and standard deviation</h4>
<p>Of course, it's impossible for any variable to happen to be the average, and there's bound to be a deviation; and...&#36;X-E(X)&#36;It's also a random variable that we choose to erase positive and negative effects and not to use obnoxious absolute values.&#36;(X-E(X))^{2}&#36; As an image of fluctuations,&#36;E((X-E(X))^{2})&#36; It'll reflect the overall fluctuations.
<strong>Definitions</strong> If Random Variables&#36;X&#36;The mathematical expectation exists, but it's called&#36;E((X-E(X))^{2})&#36;It's a random variable method.&#36;Var(X)&#36;
For differential calculations, it's easy to see what he's expected to be a random variable function, and it's very easy to calculate.
The standard difference is the amount derived from the equation, his schematics are the same as the original random variable, which is what he meant to exist.
When mathematical expectations exist, differences don't necessarily exist, but vice versa.</p>
<h4>Nature of the difference</h4>
<ul>
<li>&#36;Var(X)=E(X^{2})-(E(X))^{2}&#36;It's more practical to calculate the difference.</li>
<li>&#36;Var(c)=0&#36;</li>
<li>&#36;Var(aX+b)=a^{2}Var(X)&#36;</li>
<li>&#36;\mathrm{Var}(X)=E(X(X-1))-\mu_X(\mu_X-1)&#36;</li>
<li>&#36;X,Y&#36;On my own. &#36;Var(X+-Y)=Var(X)+Var(Y)&#36;</li>
</ul>
<h4>Chebby Scheffer's all the same.</h4>
<p>A constant of random variables when expectations and differences exist&#36;\varepsilon&#36;  Yes.
&#36;&#36;P\left(\left|\xi- E(\varepsilon)\right|\geqslant\varepsilon\right)\leqslant\frac{D\left(\xi\right)}{\varepsilon^{2&#125;&#125;&#36;&#36;
The core of the Chebbyschev heterogeneity is to give a high probability of deviation. </p>
<p><strong>Theorem</strong>: Random variable&#36;X&#36;Difference&#36;Var(X)=0&#36;Meaning&#36;X&#36;Almost everywhere equals a constant.&#36;c&#36;</p>
<h3>Frequent discrete distribution</h3>
<h4>Two distributions</h4>
<p>&#36;&#36;b(k;n,p)=C_{n}^{k}p^{k}q^{n-k}&#36;&#36;
It's the probability of two distributions, and it's born to study sampling and put back, and we add here about averages and differences.</p>
<ul>
<li>&#36;E(X)=np&#36;</li>
<li>&#36;Var(X)=np(1-p)&#36;</li>
</ul>
<h4>Benuli distribution</h4>
<p>He's the two distributions of degradation.</p>
<h4>Porsche distribution</h4>
<p>&#36;p(k,\lambda)=\frac{\lambda^{k&#125;&#125;{k!}e^{-\lambda}&#36;
The porcelain distribution is exported by the approximation of two distributions.</p>
<ul>
<li>&#36;E(X)=\lambda&#36;</li>
<li>&#36;Var(X)=\lambda&#36;
It's a very amazing feature.</li>
</ul>
<h4>Supergeometric distribution</h4>
<p>It was also exported in two distribution experiments.
&#36;&#36;总数\left(\begin{array}{l}
a+b \
n
\end{array}\right) 合适的样本点数目\left(\begin{array}{l}
a \
k
\end{array}\right)*\left(\begin{array}{l}
b \
n-k
\end{array}\right)&#36;&#36;
The supergeometric distribution is complicated, and we don't study his averages and differences here.</p>
<h4>Geometric distribution</h4>
<p>Studying another problem with sampling
&#36;&#36;g(k;p)=q^{k-1}p&#36;&#36;</p>
<ul>
<li>&#36;E(X)=\frac{1}{p}&#36;</li>
<li>&#36;Var(X)=\frac{1-p}{p^{2&#125;&#125;&#36;</li>
</ul>
<h4>Pascal distribution</h4>
<p>One extension of geometry is called negative dichotomy.
&#36;&#36;f(k;r,p)=C_{r-1}^{k-1}p^{k-1}q^{k-r}p=C_{r-1}^{k-1}p^{k}q^{k-r}&#36;&#36;</p>
<ul>
<li>&#36;E(X)=\frac{r}{p}&#36;</li>
<li>&#36;Var(X)=\frac{r(1-p)}{p^{2&#125;&#125;&#36;
The probability is derived from geometry.</li>
</ul>
<h3>Regular and continuous distribution</h3>
<p>The continuous distribution of density functions and distribution functions can be exported from one another, but people give more attention to probabilities density functions, which we will also reflect in our subsequent narratives.</p>
<h4>Normal distribution</h4>
<p>This is the most important continuous distribution in probabilistic and mathematical statistics; we repeat it countless times in a lot of subsequent studies;</p>
<h5>Density and distribution functions for normal distribution</h5>
<p>If the density function of random variable X is&#36;p(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu )^{2&#125;&#125;{2\sigma^{2&#125;&#125;}&#36;  Name&#36;X&#36;Subject to normal distribution &#36;X\sim N(\mu,\sigma)&#36; His distribution functions are usually expressed directly in the form of points.&#36;\mu&#36;Center called Location Parameters for Normal Distribution &#36;\sigma&#36;It's called a scale parameter that determines the degree of fragmentation.</p>
<h5>Standard normal distribution</h5>
<p>The normal distribution of position parameters 0 and scale parameters 1 is called the standard normal distribution, i.e.&#36;N(0,1)&#36; His density function is... &#36;\varphi (x)=\frac{1}{\sqrt{2\pi&#125;&#125;e^-{\frac{\mu^{2&#125;&#125;{2&#125;&#125;&#36;<br>Since the standard normal distribution does not contain any parameters, we can give them directly. &#36;\Phi(X)&#36;He'll be used extensively to calculate probabilistic values.</p>
<ul>
<li>&#36;\Phi(-u)=1-\Phi(u)&#36;</li>
<li>&#36;P(X&gt;u)=1-\Phi(u)&#36;</li>
<li>&#36;P(a&lt;X&lt;b)=\Phi(b)-\Phi(a)&#36;</li>
<li>&#36;P(|X|&lt;c) =2\Phi(c)-1 US&#36;
These formulas are easy to predict.</li>
</ul>
<h5>Standardization of normal distribution</h5>
<p><strong>Theorem</strong> : If&#36;X\sim N(\mu,\sigma)&#36;  then&#36;U=\frac{X-\mu}{\sigma}\sim N(0,1)&#36;
Based on this theory, we know that when calculating the probability of a non-standard normal distribution, we can choose to translate it directly into a standard form and then proceed with the next calculation.
&#36;P(X)&lt;(c) = (\frac{c-mu}{\sigma}
This is the common way to calculate the value of any normal distribution.
Functions&#36;\Phi(x)&#36;The value can be obtained from a checklist.</p>
<h5>Expectations and differences in normal distribution</h5>
<p>In fact, we studied in high school for normal distribution.</p>
<ul>
<li>&#36;E(X)=\mu&#36;</li>
<li>&#36;Var(X)=\sigma^{2}&#36;</li>
</ul>
<h5>Normal distribution&#36;3\sigma&#36; Principles</h5>
<p>There are two important explanations for this theory. </p>
<ul>
<li>If there's a serious deviation,&#36;3\sigma&#36; Principle Distribution is not a normal distribution</li>
<li>If there is a serious deviation from production&#36;3\sigma&#36;   It means production is uncontrolled.</li>
</ul>
<h4>even distribution</h4>
<p>We say the distribution of the distribution function meets the following conditions:&#36;U(a,b)&#36;
&#36;&#36;p(x)=\left{\begin{array}{ll}
\frac{1}{b-a}, &amp; a&lt;x&lt;b, \
0, &amp; \text {Other.}
{\bord0\shad0\alphaH3D}right.</p>
<ul>
<li>&#36;E(X)=\frac{a+b}{2}&#36;</li>
<li>&#36;Var(X)=\frac{(b-a)^{2&#125;&#125;{12}&#36;</li>
</ul>
<h4>Index distribution</h4>
<p>We call the distribution of distribution functions that meet the following conditions as an index distribution.&#36;EXP(\lambda)&#36;
&#36;&#36;p(x)=\left{\begin{array}{ll}
\lambda e^{-\lambda x}, &amp; x\ge0, \
0, &amp; \text {Other.}
{\bord0\shad0\alphaH3D}right.
He's used to describe life expectancy. </p>
<ul>
<li>&#36;E(X)=\frac{1}{\lambda}&#36;</li>
<li>&#36;Var(X)=\frac{1}{\lambda^{2&#125;&#125;&#36;
The index distribution is immutable, as is geometry.</li>
</ul>
<h4>Gamma distribution</h4>
<p>Give Gamma function again
&#36;&#36;\Gamma(\alpha)=\int_0^\infty t^{\alpha-1}e^{-t}dt&#36;&#36;
Nature gives a probability density function for Gamma distribution
=x(x)=\left{erray}\frac{\a^e\beta}&amp;x&gt;0\0&amp;\text{otherwise}\end{array}\right.&#36;&#36;
&#36;E[X]=\frac\alpha\beta,\quad\text{,}Var(X)=\frac\alpha{\beta^2}&#36;
It's natural.&#36;\alpha = 1&#36; He's the probability density function for index distribution.
If&#36;\alpha=\frac{n}{2};\beta=\frac{1}{2}&#36; It's freedom.&#36;n&#36;the distribution of the card
Gamma distribution is derived from the index distribution; he is the sum of several independent and distributed index distribution variables
Trying to prove that this theory can use the theory of rectangular parent function (Moment Generating Fund)</p>
<h4>Against Gamma Distribution</h4>
<p>It's also called inv Gamma distribution.
&#36;&#36;\left.\left{array}\frac{\alpha}{\alpha} (xalpha})1}exp0\ (\frac{\beta}x0\),\geq0\,0\&lt;0\end{array}\right.\right.&#36;&#36;
计算他的期望和方差有
&#36;&#36;\begin{aligned}
&amp;E(x)=\frac{\lambda^{\alpha&#125;&#125;{\Gamma(\alpha)}\int_{0}^{+\infty}x^{-\alpha}e^{-\frac{\lambda}{x&#125;&#125;dx=\frac{\Gamma(\alpha-1)}{\Gamma(\alpha)}\lambda=\frac{\beta}{\alpha-1} \
&amp;E(x^{2})=\frac{\lambda^{\alpha&#125;&#125;{\Gamma(\alpha)}\int_{0}^{+\infty}x^{-\alpha+1}e^{-\frac{\lambda}{x&#125;&#125;dx=\frac{\Gamma(\alpha-2)}{\Gamma(\alpha)}\lambda^{2}=\frac{\beta^{2&#125;&#125;{(\alpha-1)(\alpha-2)} \
&amp;== sync, corrected by elderman == @elder man
I'm sorry.
It's more common in Bayesian statistics.</p>
<h4>Beta distribution</h4>
<p>Beta distribution is also a very common continuum.
&#36;&#36;f(x;\alpha,\beta)=\frac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1}&#36;&#36;
Note that the parameter range here is &#36;\alpha~beta&gt;&#36; 0 million
Of which&#36;B(\alpha,\beta)&#36; is the value of the Beta function presented in the mathematical analysis.
&#36;&#36;B(p,q)=\int_{0}^{1}x^{p-1}(1-x)^{q-1}dx&#36;&#36;
Their expectations and equations are different.
&#36;&#36;E(X)=\frac{\alpha}{\alpha+\beta}&#36;&#36;
&#36;&#36;Var(X)=\frac{\alpha\beta}{(\alpha+\beta+1)(\alpha+\beta)^2}&#36;&#36;</p>
<h3>Distribution of random variable functions</h3>
<p>From one distribution, look at the other with one function.&#36;g(X)&#36; It's very central to probabilistic and mathematical statistics.</p>
<h4>Distribution of discrete random variable functions</h4>
<p>In fact, the function of a discrete random variable must be an discrete random variable, so the problem is very simple.
Because of its limited nature, we can calculate which function values are mapped for which new function and add the same probability to it.</p>
<h4>Distribution of functions of continuous random variables</h4>
<h5>Turned out.</h5>
<p>If this function leads to a new discrete distribution, we can only calculate it by the way we calculate it, or by the time we calculate the probability of each discrete point.</p>
<h5>Strictly monotonous.&#36;g(x)&#36;</h5>
<p>We can give a very effective theory to deal with this kind of problem.
<strong>Theorem</strong> Set&#36;X&#36;It's a continuous random variable.&#36;p(x)&#36; &#36;Y=g(X)&#36; It's another continuous random variable if&#36;g(x)&#36;It's a strictly monotonous function.&#36;h(y)&#36;Other Organiser&#36;Y&#36;The density function meets
&#36;p(y)=\left(begin{array}ll}
P x[h(y)]&#39;}(y)|, &amp; a&lt;y&lt;b, \
0, &amp; \text {Other.}
{\bord0\shad0\alphaH3D}right.
of which&#36;a=min{g(-\infty),g(\infty)} b=max{g(-\infty),g(\infty)}&#36;
This is a very important theorem in which the non-zero definition of the new density function should be replaced by the definitional domain of the original density function to the transformational function.</p>
<h5>Common theorem and nature derived</h5>
<p>Set Random Variables&#36;X&#36;Subject to normal distribution&#36;N(\mu,\sigma^2)&#36; There is.&#36;Y=aX+b\sim N(a\mu+b,a^2\sigma^2)&#36;
It's a very basic nature of normal distribution.
Set Random Variables&#36;X&#36;Subject to normal distribution&#36;N(\mu,\sigma^2)&#36;  There is.&#36;Y=e^X&#36; The probability density function is
&#36;&#36;\left.P (y)=\left{begin{aligned}&amp;\frac{1}{\sqrt{2\pi}y\sigma}\exp\left{-\frac{\left(\ln y-\mu\right)^{2&#125;&#125;{2\sigma^{2&#125;&#125;\right},y&gt;0,\&amp;I'm sorry.
What we call a logarithmic normal distribution is a very common distribution.
Set Random Variables&#36;X&#36;Obey Gamma distribution.&#36;Ga(\alpha,\beta)&#36; There is.&#36;Y=kX\sim Ga(\alpha,\frac{\beta}{k})&#36; </p>
<h5>Other&#36;g(x)&#36;Situation</h5>
<p>The basics are that we can use definitions for research.
For research.&#36;Y=g(X)&#36;The distribution function we have.&#36;F_{Y}(y)=P(g(X)\le y)&#36;  Turned into about&#36;x&#36;By&#36;y&#36;Probability forms of restriction
Different studies are conducted according to different characteristics</p>
<h3>Other features of distribution</h3>
<h4>Rectangular</h4>
<p>&#36;k&#36;Paradox:&#36;\mu_k=E(X^{k})&#36; </p>
<h4>Central Rectangular</h4>
<p>&#36;k&#36;Center Rectangular:&#36;v_k=E(X-E(X))^{k}&#36; </p>
<h4>Variable factor</h4>
<p>The amount given below is ununited.
Defines the variable factor as
&#36;&#36;V=\frac{\sigma}{\mu}&#36;&#36;</p>
<h4>Bits</h4>
<p>The following conditions are met:&#36;x_p&#36;Called&#36;p&#36;It's called the lower.&#36;p&#36;It's not very useful for us to circumvent the concept of the upper fraction.
&#36;&#36;F\left(x_{p}\right)=\int_{-\infty}^{x_{p&#125;&#125;p\left(x\right)dx=p&#36;&#36;
&#36;p=0.5&#36;in which case the fraction is called the median</p>
<h4>Offset coefficient</h4>
<p>If Random Variables&#36;X&#36;The first three steps exist.
&#36;&#36;\beta_{S}=\frac{\nu_{3&#125;&#125;{\nu_{2}^{3/2&#125;&#125;=\frac{E\left(X-E\left(X\right)\right)^{3&#125;&#125;{\left[\mathrm{Var}\left(X\right)\right]^{3/2&#125;&#125;&#36;&#36;
For the margin
He reflects the degree to which the distribution deviates from symmetry.</p>
<h4>Peak coefficient</h4>
<p>If Random Variables&#36;X&#36;The first four steps exist.
&#36;&#36;\beta_{k}=\frac{\nu_{4&#125;&#125;{\nu_{2}^{2&#125;&#125;-3=\frac{E\left(X-E\left(X\right)\right)^{4&#125;&#125;{\left[Var\left(X\right)\right]^{2&#125;&#125;-3&#36;&#36;
is the peak coefficient
He reflects the steepness of the peak or the thickness of the tail.</p>
<h2>Random vector and its distribution</h2>
<p>In some random phenomena, it's not enough for sample points to use only one random variable to describe it, which leads to the concept of random vectors, and we can't consider these random variables separately.</p>
<h3>Random vector and joint distribution</h3>
<h4>Multi-dimensional Random Variables</h4>
<p><strong>Definitions</strong> : If&#36;X_{i}(w)&#36; It's defined in the same sample space.&#36;n&#36;A random variable is called&#36;X(w)=(X_{1}(w),X_{2}(w),X_{3}(w)....)&#36; It's one.&#36;n&#36;V's random vector </p>
<p>The core must be in the same sample space.</p>
<h4>Joint Distribution Functions</h4>
<p>Definitions: Yes&#36;n&#36;Could not close temporary folder: %s&lt;x_{1&#125;&#125;,{X_{2}&lt;x_{2&#125;&#125;,{X_{3}&lt;x_{3&#125;&#125;...&#36;同时发生的概率&#36;F(x_{1},x_{2}...x_{n})=P(X_{1}&lt;x_{1}...X_{n}&lt;x_{n})&#36; 就是&#36;Distribution function of the n&#36;D random variable
In our later study, we're focusing on the combined distribution of two dollars, and the natural extension of more dimensions.
Basic properties of multiple joint distribution functions</p>
<ul>
<li>Monophonic &#36;F(x,y)&#36; Both variables are single.</li>
<li>There's a boundary. &#36;F(x,y)&#36; The value range is 0-1 and as long as one is close and negative, it's zero.</li>
<li>Right continuity, right continuity for individual variables.</li>
<li>Non-negative &#36;P(a)&lt;X&lt;b,c&lt;Y&lt;d) = F(b,d)-F(a,d)-F(b,c)+F(a,c)\ge0
I can also prove that all functions that satisfy this nature are distributed functions.</li>
</ul>
<h4>Joint Distribution Columns</h4>
<p>For the discrete binary distribution, we can use the matrix to describe the probability of the event. Icon
<strong>Joint distribution display matrix. We'll talk about it later.</strong>
Nature of the joint distribution column</p>
<ul>
<li>Non-negative&#36;p(x_{i})\ge 0&#36;</li>
<li>Regularity&#36;\sum \sum p_{ij}=1&#36;
The core of the joint distribution is research probabilities.</li>
</ul>
<h4>Joint density function</h4>
<p>Now handle continuous multiple distribution functions
Definitions: If existing&#36;p(x,y)&#36; Distribution function for random variables of the binary&#36;F(x,y)&#36;Satisfied&#36;&#36;F(x,y)=\iint\limits_{-\infty}^{x,y}p(x,y)dxdy&#36;&#36;It's calculated using a cumulative fraction.
Name&#36;p(x,y)&#36;It's a joint density function, and the dilution of the distribution function must be a density function.</p>
<ul>
<li>Non-negative&#36;p(x,y)\ge 0&#36;</li>
<li>Regularity&#36;\iint\limits_{-\infty}^{\infty}p(x,y)dxdy=1&#36;
In the case of multiples, let's stress that when we use points to calculate probability,
<strong>The range of points is the intersection between the range required by the title and the non-zero zone, which is then calculated as a cumulative fraction.</strong>
As for the issue of cumulative fractions, it's clear from the mathematical analysis that there's no very difficult cumulative calculation here.</li>
</ul>
<h4>Some common multi-dimensional random variables</h4>
<h5>Multiple distributions</h5>
<p>In high school, there's a certain amount of research, which means we have more than one option.
&#36;&#36;P=C_{n}^{k_{1&#125;&#125;C_{n}^{k_{2&#125;&#125;...C_{n}^{k_{n&#125;&#125;p_{1}^{k_{1&#125;&#125;p_{2}^{k_{2&#125;&#125;...p_{n}^{k_{n&#125;&#125;&#36;&#36;
Multiple distributions are a discrete distribution.</p>
<h5>Multi-dimensional hypergeometric distribution</h5>
<p>Or don't put it back for sampling.
&#36;&#36;P(X_{1}=n_{1},X_{2}=n_{2},\cdots,X,=n_{r})=\frac{\binom{N_{1&#125;&#125;{n_{1&#125;&#125;\binom{N_{2&#125;&#125;{n_{2&#125;&#125;\cdots\binom{N_{r&#125;&#125;{n_{r&#125;&#125;}{\binom{N}{n&#125;&#125;&#36;&#36;</p>
<h5>Multi-dimensional even distribution</h5>
<p>&#36;&#36;p(x)=\left{\begin{array}{ll}
\frac{1}{S}, &amp; x\in S, \
0, &amp; \text {Other.}
{\bord0\shad0\alphaH3D}right.</p>
<h5>Binary Normal Distribution</h5>
<p>The core of the normal distribution is five.&#36;(X,Y)\sim N(\mu_{1},\mu_{2},\sigma_{1},\sigma_{2},p)&#36; The joint density function is
&#36;&#36;f(x_1,x_2)=\frac1{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2&#125;&#125;e^{-\frac1{2(1-\rho^2)}\left(\frac{(x_1-\mu_1)^2}{\sigma_1^2}-2\rho\cdot\frac{x_1-\mu_1}{\sigma_1}\cdot\frac{x_2-\mu_2}{\sigma_2}+\frac{(x_2-\mu_2)^2}{\sigma_2^2}\right)}&#36;&#36;
They have edge density functions.
We saw it in the joint density function.&#36;p&#36;He's the relevant coefficient.</p>
<h3>Marginal distribution and independence</h3>
<p>There's a lot of diversity in distribution that deserves our study.</p>
<ul>
<li>Density function for a single variable - marginal density function</li>
<li>Level of correlation between the two volumes - relevant coefficient</li>
<li>When a measure is given, another distribution - the condition distribution
We'll be working on it later, and we'll be working on it.</li>
</ul>
<h4>Marginal Distribution Functions</h4>
<p>It's not a difficult problem.
And so... &#36;F(x,y)&#36; About&#36;x&#36;and&#36;y&#36;The marginal distribution is as follows:
&#36;\lim_{y \to \infty} F(x,y)&#36;
&#36;\lim_{x \to \infty} F(x,y)&#36;
Marginal distribution is the distribution of a fraction or parts of a vector
It lacks an image of the relationship between weight and weight.</p>
<h4>Marginal Distribution Bar</h4>
<p>For discrete scenes, the marginal distribution column needs to add up each row and each column. </p>
<h4>Marginal density function</h4>
<p>We can give this formula. He understands it very well.
&#36;p_{X}(x)=\int\limits_{-\infty }^{\infty } p(x,y)dy&#36;
&#36;p_{Y}(y)=\int\limits_{-\infty }^{\infty } p(x,y)dx&#36;
The formula itself is very well understood, or is it an old problem? </p>
<h4>Random variable independence</h4>
<p>Sometimes the weights between multiple random vectors interact with each other, but sometimes they're independent of each other.</p>
<p>Definitions: If  &#36;F(x_{1},x_{2}...,x_{n})=F(x_{1})F(x_{2})...F(x_{n})&#36;  Call these random variables independent of each other.</p>
<p>For separate random variables:</p>
<ul>
<li>Disconnection can determine the probability of a big event by building up the probability of every small event.</li>
<li>The continuous combined probability density is the accumulation of marginal probability density.</li>
</ul>
<p><strong>In order to determine whether the random variable is independent, the core is to determine whether the combined probability density is the amount of the marginal probability density.</strong></p>
<h3>Functions of random vectors</h3>
<h4>Classical partition scene</h4>
<p>It's always our choice.&#36;Y&#36;Once the value is taken out, it's enough to calculate the composition and the final probability.
We'll give a simple example of how his mind is repeated in the back.
Prove the additionality of the porcelain distribution, which is &#36;X\simP (\lambda )<del>Y\sim P(\lambda_{2})</del>\\text{requirements}&#36; &#36;Z=X+Y\sim P(\lambda 1+\lambda 2) &#36;
It's easy to know. &#36;Z&#36; Can go to all non-negative integers is a discrete distribution and&#36;Z=k&#36; Yes.
&#36;&#36;\left{X=i,Y=k-i\right}&#36;&#36;
It's an incompatible set of times, and given the independence,
&#36;&#36;P\left(Z=k\right)=\sum_{i=0}^{k}P\left(X=i\right)P\left(Y=k-i\right).&#36;&#36;
We call it<strong>Dispersive volume formula</strong> So we can calculate.
&#36;&#36;X+Y\sim P(\lambda_1+\lambda_2)&#36;&#36;
The volume is the sum of two random variables.</p>
<h4>Maximum distribution</h4>
<p>Use definition studies to remove the minimum value mark by the nature of probability as follows:
Set&#36;X_1,X_2,...X_n&#36; It's independent.&#36;n&#36;A random variable.&#36;max{X_1,X_2,...X_n};min{X_1,X_2,...X_n}&#36; Distribution
&#36; \begin{aligned}F y(y)&amp;=P(\max{X_1,X_2,\cdots,X_n}\leqslant y)=P(X_1\leqslant y,X_2\leqslant y,\cdots,X_n\leqslant y)\&amp;=P(X_1\leqslant y)P(X_2\leqslant y)\cdots P(X_n\leqslant y)=\prod_{i=1}^nF_i(y).\end{aligned}&#36;&#36;
&#36;&#36;\begin{aligned}
F_{z}\left(z\right)&amp; =P\left(\min\left{X_{1},X_{2},\cdots,X_{n}\right}\leq z\right)  \
&amp;=1-P\left(\min\left{X_{1},X_{2},\cdots,X_{n}\right}&gt;z\right) \
&amp;=1-P\left(X_{1}&gt;z,X_{2}&gt;z,\cdots,X_{n}&gt;z\right) \
&amp;=1-P\left(X_{1}&gt;z\right)P\left(X_{2}&gt;z\right)\cdots P\left(X_{n}&gt;z\right) \
&amp;== sync, corrected by elderman == @elder man
I'm sorry.
It's a very common treatment, and we'll meet somewhere else.</p>
<h4>Continuous volume formula</h4>
<p>For continuous&#36;Z=X+Y&#36;  Two random variables are irrelevant.
Theoretically:&#36;p_{Z}(z)=\int\limits_{-\infty }^{\infty} P_{X}(z-y)P_{Y}(y)dy=\int\limits_{-\infty }^{\infty} P_{X}(x)P_{Y}(z-x)dx&#36;
He's still easy to remember and understand.
<strong>The core point is that we have to make sure that neither of the two probability densities is zero, so we need to process an iniquities between the blocks.</strong>
The volume formula can also be used for two random variables that are not independent.</p>
<h4>Variable variant</h4>
<p>We studied the function of random variables when they were random. Down
Set 2D random variable&#36;(X,Y)&#36; The joint density function is&#36;p(x,y)&#36; If there is.
&#36;&#36;\begin{cases}u=g_{1}\left(x,y\right)\v=g_{2}\left(x,y\right)\end{cases}&#36;&#36;
There's a continuous deviation and there's a single inverse.
&#36;&#36;\left.\left{\begin{matrix}x=x\left(u,v\right)\y=y\left(u,v\right)\end{matrix}\right.\right.&#36;&#36;
Transforming the Yacima one.
&#36;&#36;\left.\left.J=\frac(x, y\right)\partial\left(u,v\right)}\left\begin{matrix}\frac(partialx}{\partialu}&amp;\frac{\partial x}{\partial v}\\frac{\partial y}{\partial u}&amp;\frac{\partial y}{\partial v}\end{matrix}\right.\right|=\left(\frac{\partial\left(u,v\right)}{\partial\left(x,y\right)}\right)^{-1}=\left(\left|\begin{matrix}\frac{\partial u}{\partial x}&amp;\frac{\partial u}{\partial y}\\frac{\partial v}{\partial x}&amp;\frac{\partial v}{\partial y}\end{matrix}\right.\right|\right|^{-1}\neq0&#36;&#36;
如果
&#36;&#36;\left.\left{\begin{matrix}U=g_{1}\left(X,Y\right)\V=g_{2}\left(X,Y\right)\end{matrix}\right.\right.&#36;&#36;
则&#36;(U,,V)&#36; 的联合密度为
&#36;&#36;p\left(u,v\right)=p\left(x\left(u,v\right),y\left(u,v\right)\right)|J|&#36;&#36;</p>
<h4>Stock and commerce</h4>
<p>Set Random Variables&#36;X,Y&#36;Independent&#36;p_X(x),p_Y(y)&#36;
&#36;U=XY&#36;The density function is
&#36;&#36;P_{U}\left(u\right)=\int_{-\infty}^{\infty}p_{X}\left(\frac{u}{v}\right)p_{Y}\left(v\right)\frac{1}{\left|v\right|}dv.&#36;&#36;
&#36;U=\frac{X}{Y}&#36; The density function is
&#36;&#36;P_{U}\left(u\right)=\int_{-\infty}^{\infty}p_{X}\left(uv\right)p_{Y}\left(v\right)\mid v\mid dv.&#36;&#36;
The formula is the same as usual.</p>
<h3>Multi-dimensional digital characteristics</h3>
<h4>Multi-dimensional random vector expectations</h4>
<p>The formula that we want to be able to give the desired function of a 2-D random variable, which is the new feature of a multi-dimensional situation, is that only the function of a multi-dimensional random variable is a new random variable that can study expectations.
Set 2D random variable&#36;(X,Y)&#36; The distribution is expressed as a joint distribution column&#36;P(X=x,Y=y)&#36; The joint density function is&#36;p(x,y)&#36; then&#36;Z=g(X,Y)&#36; The mathematical expectation is...
&#36;&#36;\left.E\left(Z\right)=\left{\begin{matrix}\sum_{i}\sum_{j}g\left(x_{i},y_{j}\right)P\left(X=x_{i},Y=y_{j}\right)\\\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}g\left(x,y\right)p\left(x,y\right)\mathrm{d}x\mathrm{d}y,\end{matrix}\right.\right.&#36;&#36;</p>
<h4>Agreement</h4>
<p>It's called the center of the equation.
If for a 2D random variable&#36;(X,Y)&#36; Yes.&#36;E\left[\left(X-E\left(X\right)\right)\left(Y-E\left(Y\right)\right)\right]&#36; Existence
It's called an agreement.
&#36;&#36;\mathrm{Cov}\left(X,Y\right)=E\left[\left(X-E\left(X\right)\right)\left(Y-E\left(Y\right)\right)\right]&#36;&#36;
Especially. &#36;Cov\left(X,X\right)=Var\left(X\right)&#36;
The difference is a mathematical expectation of two differential multipliers, so the difference is positive and negative, too.
When the balance is positive, it's called positive correlation.
When the alignment is 0, either two random variables are irrelevant or they have non-linear relationships.</p>
<p>Here are some of the usual characteristics.</p>
<ul>
<li>&#36;Cov(X,Y)=E(XY)-E(X)E(Y)&#36;   </li>
<li>It's not like you're gonna be able to be independent.</li>
<li>&#36;Var(X+Y)=Var(X)+Var(Y)+2Cov(X,Y)&#36;</li>
<li>&#36;Cov(X,Y)=Cov(Y,X)&#36;</li>
<li>&#36;Cov(X,c)=0&#36; </li>
<li>&#36;Cov(aX,bY)=abCov(X,Y)&#36;</li>
<li>&#36;Cov(X+Y,Z)=Cov(X,Z)+Cov(Z,Y)&#36;</li>
</ul>
<h4>Related coefficient</h4>
<p>If for a 2D random variable&#36;(X,Y)&#36; &#36;Var(X),Var(Y)&gt;&#36;0.00
&#36;&#36;Corr\left(X,Y\right)=\frac{Cov\left(X,Y\right)}{\sqrt{Var\left(X\right)}\sqrt{Var\left(Y\right)&#125;&#125;=\frac{Cov\left(X,Y\right)}{\sigma_{X}\sigma_{Y&#125;&#125;&#36;&#36;
A linear correlation between two random variables is illustrated.
There's another explanation for the coefficient.<strong>Coordinated random variables Bad</strong></p>
<p>Or give me something.</p>
<ul>
<li>&#36;-1\leqslant Corr\left(X,Y\right)\leqslant1,或\left|Corr\left(X,Y\right)\right|\leqslant1&#36;</li>
<li>&#36;|Corr(X,Y)|=1&#36; . . . . . .&#36;X,Y&#36;It's almost linear, which is...&#36;P\left(Y=aX+b\right)=1&#36;
The relevant coefficient is&#36;[-1,1]&#36;Here's some evidence of the value taken between.
= \ \ \ \ \ = = = = = = = = = = = = = = = = } } } } } } } } } } } } } } } } } = = = \ \ \ \ \ \ \ = <em>Y})=2+2p</em>{xy}&#36;&#36;
&#36;&#36;0\leq Var(\frac x{\sigma _X}-\frac y{\sigma _Y})=\frac{\mathrm{Var}\left[X\right]}{\sigma _X^{2&#125;&#125;+\frac{\mathrm{Var}\left[Y\right]}{\sigma _Y^{2&#125;&#125;-2Cov(\frac x{\sigma _X},\frac y{\sigma <em>Y})=2-2p</em>&#36;
A combination of two variants would prove the original proposition.</li>
</ul>
<h4>Mathematical Expectations Matrix and Convergence Matrix</h4>
<p>We use the matrix.&#36;n&#36;The mathematical expectations of the wi-random vectors, the difference between the parties; the difference between the parties, of course, leads to a correlation coefficient.
Mathematical Expectations Matrix
&#36;&#36;E\left(X\right)=\left(E\left(X_{1}\right),E\left(X_{2}\right),\cdots,E\left(X_{n}\right)\right)^{\prime}&#36;&#36;
Coordinated Matrix (a non-negative matrix)
&#36; = \begin{matrix}\pectorname{Var} (X 1)&amp;\operatorname{Cov}(X_1,X_2)&amp;\cdots&amp;\operatorname{Cov}(X_1,X_n)\\operatorname{Cov}(X_2,X_1)&amp;\operatorname{Var}(X_2)&amp;\cdots&amp;\operatorname{Cov}(X_2,X_n)\\vdots&amp;\vdots&amp;&amp;\vdots\\operatorname{Cov}(X_n,X_1)&amp;\operatorname{Cov}(X_n,X_2)&amp;\cdots&amp;\operatorname{Var}(X_n)\end{pmatrix}&#36;&#36;</p>
<h3>Distribution of conditions and expectations</h3>
<p>The theory of the overall distribution of conditions is largely consistent with the form of the theory of the distribution of conditions that we have studied in chapter I.
It must be a one dollar distribution of the initial distribution of the binary.</p>
<h4>Dispersed condition distribution</h4>
<p>Or is the probability of simultaneous occurrence divided by the probability of a condition occurring at a time when the probability of a condition occurring is a marginal distribution?
First give a distribution column of 2D discrete random variables
&#36;&#36;p_{ij}=P\left(X=x_{i},Y=y_{j}\right)&#36;&#36;
So the condition distribution is
&#36;&#36;P_{ij}=P\left(X=x_{i}\mid Y=y_{j}\right)=\frac{P\left(X=x_{i},Y=y_{j}\right)}{P\left(Y=y_{j}\right)}=\frac{P_{ij&#125;&#125;{P_{.j&#125;&#125;&#36;&#36;</p>
<h4>Continuous conditions distribution</h4>
<p>Or is the probability of simultaneous occurrence divided by the probability of a condition occurring at a time when the probability of a condition occurring is a marginal distribution?
&#36;&#36;P\left(X\leq x\mid Y=y\right)=\int_{-\infty}^{x}\frac{p\left(u,y\right)}{p_{r}\left(y\right)}du&#36;&#36;</p>
<h4>It's a mathematical expectation.</h4>
<p>The mathematical expectation of condition is the mathematical expectation of a certain distribution.
&#36;&#36;\left.E\left(X\mid Y=y\right)=\left{\begin{matrix}\sum_{i}x_{i}P\left(X=x_{i}\mid Y=y\right)\\int_{-a}^{\infty}xp\left(x\mid y\right)dx,\end{matrix}\right.\right.&#36;&#36;
The mathematical expectation is the expectation, but it's the same.&#36;y&#36;It's a function, so we often give another pattern. &#36;E(X|Y)&#36; </p>
<p><strong>Full expectation formula</strong>
&#36;&#36;E\left(X\right)=E\left(E\left(X|Y\right)\right)&#36;&#36;</p>
<h2>Big-digit theorem and center-critical.</h2>
<h3>Condensity</h3>
<h4>Concealed by probability</h4>
<p>We've come up with the theory long ago that probability is the stable value of frequency.</p>
<ul>
<li>Frequency&#36;v&#36;Right chance.&#36;p&#36;Absolute deviation&#36;|p-v|&#36;Approaching stabilization value</li>
<li>Because of randomity, we can't rule out the possibility of big deviations, but the probability of big deviations getting smaller.</li>
</ul>
<p>Now we have a general definition.
Definitions: Establishment&#36;{X_{n&#125;&#125;&#36;It's a random variable sequence. &#36;X&#36;It's a random variable.&#36;\varepsilon&#36; Yes.
&#36;lim n\info}P (Xn-X|)&lt;\varepsilon) = &#36;1
It's called a random variable with probability.&#36;X_{n}\longrightarrow X(P)&#36; </p>
<h4>Weaknesses by distribution</h4>
<p>The distribution function is also an important image of the probabilities problem.</p>
<p>We've been able to tell you about the downs and downs of the function in the mathematical analysis, but it's too strong a condition to make the distribution function no longer conform to the distribution function, so let's put it down.</p>
<p>Definition: Set a distribution function column&#36;{F_{n}(x)}&#36; What if...&#36;F(x)&#36;Any point of continuity.
&#36;&#36;\lim_{n \to \infty} F_{n}(x)=F(x)&#36;&#36;
is the distribution function column&#36;{F_{n}(x)}&#36;Weak harvest&#36;F(x)&#36; Or a random series of variables.&#36;{X_{n&#125;&#125;&#36;Condense X by distribution
Recorded&#36;X_{n}\longrightarrow X(L)&#36;  Or... &#36;F_{n}(x)\longrightarrow F(x)(W)&#36;  </p>
<p>Theorem: Probability-based roll-out by distribution
Theorem: Consistency on the basis of distribution is a sine qua non of probabilization (at the same time)</p>
<h4>Almost everywhere. Probability one.</h4>
<p>It's almost everywhere.
&#36;&#36;P(\lim_{n\to\infty}X_n=X)=1&#36;&#36;
In general, the difference between probability and probability is just changing the position of the limit sign.</p>
<p>In fact, almost everywhere, it's much more mathematically than probabilistic, and he's a version of the probabilistic theory, similar to the concept of a dot-compression function in mathematical analysis.</p>
<p>It's almost everywhere that the random variable sequence is condensed and random at every point. Probability only requires that probability be calculated first, that the probability be limited to one, not every point.</p>
<h3>Large-digit laws</h3>
<p>In practice, it is recognized that the arithmetical averages of a large number of measurements are also stable.</p>
<h4>Benuli's law of big numbers.</h4>
<p>He describes the connection between frequency and probability as a retort to what's ahead.
<strong>Theoretically:</strong> Set&#36;S_{n}&#36;Yes.&#36;n&#36;The event at the Hobernuli experiment.&#36;A&#36;Number of incidents&#36;p&#36;For each experiment&#36;A&#36;There's a chance that there's a chance that there's any varepsilon. &gt;Yes.
&#36;lim n\info} (\frac{S}n}-p|&lt;\varepsilon) = &#36;1
Benuli's law of big numbers gives us the theoretical basis for using frequency to determine probability.</p>
<h4>Chelby Scheffer's Law of Major Numbers.</h4>
<p>Theorem: Set&#36;{X_{n&#125;&#125;&#36; It's a series of random variables that are not relevant.&#36;X_{i}&#36; There's a difference and there's a common upper boundary for any \\varepsilon&gt;0&#36;
&#36;&#36;\lim_{&lt;\to\infty}P\left(\left|\frac{1}{n}\sum_{i=1}^{n}X_{i}-\frac{1}{n}\sum_{i=1}^{n}E\left(X_{i}\right)\right|&lt;\varepsilon\right=&#36;1.
We've weakened the distribution requirement.</p>
<h4>Markov's Law of Numeracy.</h4>
<p>Set&#36;{X_{n&#125;&#125;&#36; Yeah, random variable sequences.&#36;\frac{1}{n^{2&#125;&#125;D(\sum\limits X{i})\longrightarrow0&#36; For any \\varepsilon&gt;0&#36;
&#36;&#36;\lim_{&lt;\to\infty}P\left(\left|\frac{1}{n}\sum_{i=1}^{n}X_{i}-\frac{1}{n}\sum_{i=1}^{n}E\left(X_{i}\right)\right|&lt;\varepsilon\right=&#36;1.
Markov's Law of the Magnificent is a condition for another big law to repeat.</p>
<h4>The Law of Sinchin.</h4>
<p>Theorem: Set&#36;{X_{n&#125;&#125;&#36; It's a series of random variables that are irrelevant and subject to the same distribution and have mathematical expectations.&#36;\mu&#36;For any \\varepsilon&gt;It's all there.
&#36;&#36;\lim_{n \to \infty}P(|\frac{1}{n}\sum\limits X_{i}-\mu|\ge\varepsilon)=0 &#36;&#36;
The Sinchin Big Number Theorem is the basis of the theory of subsequent rectification. </p>
<h3>It's very restrictive.</h3>
<p>Some kind of deviation is the sum of small errors caused by a large number of small and incidental factors, and the small errors caused by all these different factors are independent of each other, and each of them has little effect on the sum. </p>
<p>It's very difficult to calculate using a volume formula at this point, but when we're drawing the graphics, we find that there's a lot more to be done.&#36;n&#36;The increase and the closeness of the function to the normal distribution, which is the central limit.</p>
<h4>The Lindbergh-Levi Center is extremely restrictive.</h4>
<p>Theorem: setting random variables&#36;X1, X2,…, Xn&#36;Independently, subject to the same distribution, with the same limited mathematical expectations and differences if Remember
&#36;Y <em>}=\frac{X_{1}+X_{2}+\cdots+X_{n}-n\mu}{\sigma\sqrt{n&#125;&#125;&#36;&#36;
有
&#36;&#36;\lim_{x\to\infty}P\left(Y_{n}^{</em>== sync, corrected by elderman == @elder man
The sum of all random variables that are independent and distributed can be approximated by normal distribution.</p>
<p><strong>Under certain conditions, the sum distribution of a large number of stand-alone random variables is normal.</strong></p>
<h4>The Mofo-La Plas Center is extremely restrictive.</h4>
<p>He's the narrow form of the first theorem in two distributions, considering the two distributions as multiple Bernouli distributions and</p>
<p>Theorem: Set&#36;Y_{n}&#36;, subject to two distributions and with mathematical expectations and differences, random variables &#36;\bar{Y}=\frac{ Y-np}{\sqrt{np(1-p)&#125;&#125;&#36; The probability density function is&#36;\frac{1}{\sqrt{2\pi&#125;&#125;e^{\frac{-t^{2&#125;&#125;{2&#125;&#125;&#36; </p>
<p>This theorem means that the normal distribution is the limit of the two distributions.</p>
<h3>The law of power and power and their strength.</h3>
<p>Large-digit law is an important theory in modern probabilistic theory and an important bridge between probabilistic and statistical theory.</p>
<p>Most of the theories in mathematics are named after theorem, which in turn reflects their results through rigorous extrapolation. The law is usually used to describe patterns in nature and is based on observations. The close connection and importance of his application can be seen in the introduction of this law in mathematics.</p>
<h4>Basic law of big numbers.</h4>
<p>Large-digit laws describe a phenomenon for a random series of variables&#36;\left { X_n\right }&#36; It's... Front&#36;n&#36;Item average
&#36;&#36;A_n=\frac{X_1+X_2+\cdots+X_n}n&#36;&#36;</p>
<p>For meeting certain conditions&#36;{X_n}&#36;, when the number of random variables&#36;n&#36;Very large, their averages are highly likely to be valued.&#36;\mu&#36;</p>
<p>&#36;&#36;A_{n}\to\mu &#36;&#36;</p>
<p>This is a value.&#36;\mu&#36;Usually.&#36;X_i&#36;Mathematical expectations. Depending on the mode of consolidation, the law of large numbers is divided into strong and weak law of large numbers.</p>
<p><strong>In essence, we're still looking at the stability of average results in a lot of random phenomena.</strong></p>
<h4>The law of powerful numbers.</h4>
<p>Its mathematical form is
&#36;&#36;P\left{\lim_{n\to\infty}A_n-\mu=0\right}=1&#36;&#36;</p>
<p>Meaning<strong>When the length of the series of random variables is infinite, their averages necessarily tend to be constant.</strong>I don't know. We call this random variable sequence a strong number law.</p>
<h4>The law of weakness.</h4>
<p>The mathematical form is
&#36;&#36;\forall\varepsilon&gt;0:\lim_{n\to\infty}P{|A_n-\mu|&lt;\varepsilon}=1&#36;&#36;</p>
<p>Meaning<strong>When the sequence of random variables is of infinity length, the probability of their average approaching the fixed value is close to 1.</strong>I don't know. Calls this random series of variables consistent with the law of a weak large number.</p>
<h4>Distinction and linkage</h4>
<p>A strong law of numbers is easy to understand, similar to the contraction of columns. And the law of the big and the weak is relatively difficult to understand and at first glance seems to be no different from the law of the strong. Actually, it's about right.<strong>Limits</strong>Understand.</p>
<p>The powerful law of numbers is an act of constriction, or he means almost everywhere. And the law of weakness requires only one probability of conceiving.<strong>Consistency by probability</strong>I don't know. That means that if a random series of variables meets the law of strong numbers, then he must also meet the law of weak large numbers.</p>
<p>In particular, if we look back on the section of this paper, "The Lindbergh-Levy Center is extremely restrictive", we can conclude that the form of enrichment in this section is just as distributive.</p>
<ul>
<li>The law of powerful numbers: almost everywhere.</li>
<li>Weaknesses Law: Concealed by probability</li>
<li>Centers are extremely restrictive: decrease by distribution</li>
</ul>
<p><strong>Most of the random variable sequences in practical application are also subject to the law of strong numbers, so all the later references to the law of large numbers refer to the law of strong numbers.</strong></p>
<p>Caution: Largest laws are by their very nature established by the innumerable number of tests, and we cannot carry out infinity large experiments, so in a limited number of experiments, any major deviations are not in conflict with the large number laws themselves. The law of big numbers will not affect the independence of the experiment.</p>
