---
title: 'Zipf''s Law: From the Voynich Manuscript to Alien Civilizations'
title_zh: 聊聊齐普夫定律：从伏尼契手稿到外星文明
date: 2026-02-18 20:00:00 +0800
categories:
- Data Science
- Statistical Thinking
tags:
- Statistical Thinking
author: Hyacehila
mathjax: true
excerpt: Is the Voynich manuscript random scribbling or a lost language? What would alien signals look like? Both questions
  point to the same statistical law.
description: Is the Voynich manuscript random scribbling or a lost language? What would alien signals look like? Both questions
  point to the same statistical law.
excerpt_zh: 伏尼契手稿是胡乱涂鸦还是失落的语言？外星人信号长什么样？这一切都指向同一个统计学定律。
permalink: /blog/2026/02/18/zipfs-law-and-power-law/
lang: en
translation_key: 2026-02-18-zipfs-law-and-power-law
translation_status: machine
translation_source_hash: ccb1ea20dfa45ab6f988df3b01e8eaef300d2cef20358b831d347ec2b38e357b
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Quote: Is that a lie?</h2>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/01/10/statistics-and-truth/">Statistics and Truth: How to use the accident (Statistics and Truth)</a>、<a href="/en/blog/2026/01/14/anscombes-quartet/">Anscom Quartet: Visualized power and statistical illusion</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>In 1912, the Polish bookkeeper Wilfrid Voynich bought a mysterious manuscript in Italy. The book is full of strange plant illustrations, astrometric charts and naked bathing women, and, more importantly, it is written in a text that has never been seen on Earth.</p>
<p>That's famous. <strong>Voynich Manuscript</strong>。</p>
<p>For a century, top coders, including experts in the Enigma code, have tried to untangle it, but have found nothing. Some start to wonder: Is this a medieval charade that some medieval liar drew to cheat money? It doesn't mean anything?</p>
<p>Until statisticians get involved. Instead of trying to read every word, they counted the frequency of the appearance of these symbols. They found that these seemingly obscurantized symbols, which are distributed in a word-frequency manner that is consistent with the statistical patterns common in human languages, are not only the most important but also the most important ones.<strong>Zipf Law&#39;s Law)</strong>I'm sorry. This means that whatever it is written, it is likely to be a real, logical language, not random graffiti.</p>
<p>Today, we come to talk about this truly odious law that dominates the universe from the human language to the size of the city, and possibly even the signals of alien civilization.</p>
<h2>Simple math, amazing universality.</h2>
<p>In 1949, George Kingsley Zipf, a linguist, discovered an amazing phenomenon:</p>
<p>In any English book, if you have all words in order of frequency of occurrence, then:</p>
<ul>
<li>Number 1 word (usually) &quot;the&quot;The frequency of occurrence is about the word of the second ranking (in the second place).&quot;of&quot;It's a "sweet" <strong>Two times.</strong>。</li>
<li>About the third word in the ranking (in the first place)&quot;and&quot;It's a "sweet" <strong>Three times.</strong>。</li>
<li>...</li>
<li>About the top of the ranking. &#36;n&#36; The word. <strong>&#36;n&#36;Double</strong>。</li>
</ul>
<p>In mathematical terms, it's:</p>
<p>&#36;&#36; f(r) \propto \frac{1}{r^\alpha} &#36;&#36;</p>
<p>of which &#36;f(r)&#36; It's a frequency.&#36;r&#36; The blog is a blog that has been published by the blog of the blog @Rank.&#36;\alpha&#36; Usually close to 1.</p>
<h3>Visual magic.</h3>
<p>If you draw this distribution on the normal axis, it's a steep drop. &quot;L&quot; . But if you take both the cross-axis (rank) and the vertical axis (frequency) log-Log Plot, the miracle happens:<strong>It became a straight, straight, last line, leaning down.</strong>- The slope is -1.</p>
<p><strong>How is the world even even a logarithmic?</strong></p>
<h2>Interesting application</h2>
<p>The law of Chippov is not just a toy of linguists, but it can also work in many unexpected areas.</p>
<h3>Looking for aliens.</h3>
<p>How do we know if the incoming cosmic radio waves are signals of alien civilization or the noise from a dead neutron star?</p>
<p>The SETI scientist, Zipulance Doyle, has proposed a method based on the law of Zipov:</p>
<ul>
<li><strong>Pure random noise</strong>: If we sort the pattern of the signal by frequency, it is usually flat (Flat) or it shows an exponential decline, and the slope is flat.</li>
<li><strong>Extremely simple signal.</strong>(e.g. pulsed): Only repeats the same frequency, lacking information entropy.    </li>
<li><strong>Complex language</strong>: Must be between “total random” and “total repetition”. The logarithmic frequency slope of human languages is just about -1.</li>
</ul>
<p>If we capture a waveform in the universe, which is perfectly distributed at a frequency that is in line with the law of Zipov, it is likely that it carries information of some kind of intellectual exchange.</p>
<h3>Music's "good" password</h3>
<p>Why is Mozart's music so good and cats' piano sound so bad?</p>
<p>It was found that the frequency of the music leap in the beautiful melody was also consistent with the Chippov law. <strong>&#36;1/f&#36; Noise</strong> or <strong>Pink Noise</strong>）。</p>
<ul>
<li><strong>White Noise</strong>: Totally random, too noisy, like a TV snow spot.</li>
<li><strong>Brown Noise</strong>: Randomly swimming, too dreary and solphisticated.</li>
<li><strong>1/f Noise</strong>: right in between. It's got enough.<strong>Predictability</strong>"and enough to make you familiar and comfortable."<strong>Accident</strong>(to surprise you) That's the mathematical essence of art beauty.</li>
</ul>
<h3>The Internet's “long tail effect”</h3>
<p>Why would Amazon challenge Wal-Mart? The law of Zipov gives an angle.</p>
<p>In traditional physical bookstores, because of limited shelf space, the merchants are able to sell the most popular “hot books” (head) ranking. The millions of hard-core books that are ranked behind are too low to be worth the price.</p>
<p>But the law of Zipov tells us that although each element of the long tail is low, it's very long.&#36;r&#36; The blogger says that the government is not a party to the law.<strong>The area of the long tail is huge.</strong>。</p>
<p>The Amazon ate this part of the long-term market, using virtual shelves that are almost unlimited. This is the long-tail theory that Chris Anderson has put forward — a business model that has moved from being dominated by a few hot commodities to being supported by an endless array of small commodities.</p>
<h2>The truth against intuition: is it a miracle or a coincidence?</h2>
<p>Since the law of Zipov is so universal, is it hiding some kind of uniform mechanism behind it?</p>
<p>The answer may be disappointing or more interesting.</p>
<h3>Monkey type paradox.</h3>
<p>The early linguists believe that the law of Zipov represents a human communication strategy: In order to improve efficiency, we prefer to use short words to convey the meaning of high frequency (H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-<strong>Prince of Last Effort</strong>）。</p>
<p>But the mathematician Benoit Mandelbrot poured a cold water bowl. He testified:<strong>If you let a monkey knock on a typewriter randomly, the frequency distribution of random words can also be consistent with the law of Chippov!</strong></p>
<p>This suggests that the law of Zipov may not need "smart." It may be just a statistical necessity that is naturally emerging in a combination system, as is the case with normal distribution, as a sort of “default setting” in nature.</p>
<h3>The rich get richer.</h3>
<p>Another interpretation is much more cruel. Herbert Simon has submitted <strong>Preferential Connection</strong> The mechanism, which we used to say in sociology. <strong>Matthew effect</strong>：</p>
<blockquote>
<p>"Anything, and he shall be given more than that."</p>
</blockquote>
<ul>
<li>The more a word goes, the easier you'll use it again, even if you don't have a brain.</li>
<li>The larger the city, the easier it is to attract new immigrants.</li>
<li>The more a website is linked, the more easily it is linked to new pages.</li>
</ul>
<p>Such mechanisms magnify inequalities. In the Zipov distribution, the first element is a huge share of resources. That explains why 1% of people have huge wealth, why Internet traffic is concentrated in the hands of several giants, Google and Netflix.</p>
<h3>80/20: Another face of the law of Tsipf.</h3>
<p>You must have heard of inequality. <strong>Paretto Law (Pareto Prince)</strong>It's called. <strong>80/20 Code</strong>80% of the wealth is in the hands of 20% of the people.</p>
<p><strong>The Paretto Law and the Tsipov Law describe the same structure, but they see the world differently.</strong>。</p>
<ul>
<li><strong>Zipov's law.</strong>Answer: Number one. &#36;r&#36; How much money is there?<strong>Individual</strong>）</li>
<li><strong>The Paretto Law.</strong>The answer is: how many of these people have more wealth than they have. &#36;x&#36;♪ I'm not sure ♪ "Concerned by the<strong>Cumulative total</strong>）</li>
</ul>
<p>It can be shown mathematically that if a system is in line with the law of Zipov, then it is also in line with the Paretto distribution. They're like two sides of a coin, one with a long tail, telling us how long it is; the other with a head, telling us how many heads there are.</p>
<p>When we say "the first 20 percent of the vocabulary covers 80 percent of the daily conversation," we're describing the Zipov phenomenon in the Paretto language.</p>
<h3>King's effects and defined traps</h3>
<p>The law is beautiful, but it often fails. The most famous is... <strong>The King's Effect</strong>: The first data point (the King) is often unruly, either out of scale or smaller than expected.</p>
<p>Many times, it's because the border is painted wrong.
If you count the population of Shanghai City District, you may find it inconsistent with the law of Zipov. But if you count the population of Shanghai Metropolitan Circles (which includes the surrounding regions of Kunshan, Suzhou and so on, which are closely linked to their economies), the pattern is amazing.</p>
<p>This reminds us:<strong>The law of Zipov describes the natural boundaries of the organic system, not the administrative boundaries that are divided by humans.</strong></p>
<h2>Summary</h2>
<p>The law of Zipov is like a fingerprint from a complex system. It appears in books, cities, wealth, and possibly even in radio waves of distant galaxies.</p>
<p>It may represent the efficient (saving) of systems, the cruel Matteo effect (inequity) or simply the random result of maximization of entropy (mix).</p>
<p>But anyway, next time you see that kind of thing,<strong>A very small number of them have a very large share, and the vast majority of them have a very small share.</strong>And when it happened, it was probably the Chippew Law that worked.&#36;\frac{1}{r^\alpha}&#36; And it's the frequency of the universe's breathing.</p>
