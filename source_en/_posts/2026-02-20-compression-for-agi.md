---
title: 'Compression for AGI: Compression as Intelligence'
title_zh: Compression for AGI：压缩即智能
date: 2026-02-20 20:00:00 +0800
categories:
- Foundation Models
- Model Mechanics
tags:
- Pre-Training
- Model Mechanics
- Paper Notes
author: Hyacehila
mathjax: true
excerpt: 'A summary of Jack Rae''s Compression for AGI: foundation-model training as lossless compression of useful information,
  and why lower loss can imply stronger generalization.'
description: 'A summary of Jack Rae''s Compression for AGI: foundation-model training as lossless compression of useful information,
  and why lower loss can imply stronger generalization.'
excerpt_zh: 整理 Jack Rae 在《Compression for AGI》中的观点：基础模型训练可以理解为对有效信息的无损压缩；压缩率越高（loss 越低），模型越可能呈现更强的泛化行为。
permalink: /blog/2026/02/20/compression-for-agi/
lang: en
translation_key: 2026-02-20-compression-for-agi
translation_status: machine
translation_source_hash: b87ae43fa425d772623c2b6ebc78202728a53bf373221494ad51dc374b82a0a1
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p><strong>Theme: Compression is intelligence: Why is ChatGPT intelligent?</strong> This paper is based on the theme-sharing of OpenAI researcher Jack Rae, "Community for AGI." The main lines of discussion were:<strong>Basic model training can be understood as compressing effective information as much as possible without loss.</strong></p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/01/26/neural-scaling-laws/">Neal Scaling Laws: From Kaplan to Chinchilla</a>、<a href="/en/blog/2026/02/22/loss-landscape-of-llms/">What's a big model of Los Landscape?</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h2>About General Artificial Intelligence</h2>
<p>Before explaining why “compression” is a path to universal intelligence, we briefly look back at a classic thought experiment: the Chinese room (John Seal, 1980).</p>
<blockquote>
<p>A man who knows nothing about Chinese and speaks only English is locked in a closed room with only one small window. There was a manual with a Chinese and English translation, with sufficient paper and pencils. Paper pieces written in Chinese were sent to the room through a small window. The room will be accessible to those who can translate Chinese into English and then to those who can translate English back into Chinese. Although he was totally unChinese, the outsiders would think he could speak Chinese fluently.</p>
</blockquote>
<p>Such a “big manual” clearly corresponds to a low level of intelligence: once it is encountered with input that is not covered, it cannot be responded to.</p>
<p>If we can extract grammar and rules from a large amount of data, the manual can be streamlined; at the same time, the system is more intelligent (more broadly oriented).</p>
<p>The thicker the manual, the weaker the intelligence; the thinr the manual, the stronger the intelligence. Just like companies employ one person: the stronger the ability, the less you need to explain; the weaker the capacity, the more you need to explain.</p>
<p>The example above explains in a visual way why “compression is intelligence”: to get a smaller description (the shortest “handbook”) the system is closer to intelligence.</p>
<h2>Generate models and compression</h2>
<p>Set to give &#36;D&#36;We can use the generation model. &#36;f&#36; Compress it:</p>
<p>&#36;&#36;
\lvert D \rvert = -\log P_f(D) + \lvert f \rvert
&#36;&#36;</p>
<p>of which &#36;\lvert D \rvert&#36; The size of the data set, which is no-loss compression, is equal to the sum of the losses projected for the next token, plus the minimum description of the estimated function (here) &#36;\lvert f \rvert&#36; is an abstract item that describes the minimum length/coding cost and is not equivalent to the parameter amount. At this point, the process of compressing data is the process of training to generate models.</p>
<p>A more compressed expression is available:</p>
<p>&#36;&#36;
r_n = 1 - \frac{S_1}{S_0}</p>
<blockquote>
<p>1 - \frac{\lvert f_1 \rvert + n + \sum_{t=1}^{n} -\log P(x_{t+1} \mid x_{1:t}, f_1)}{\lvert f_0 \rvert + n \log m}
&#36;&#36;</p>
</blockquote>
<p>This explains why larger models often show greater generalization: larger models, usually means lower ross, and therefore higher compression rates; in the context of “compression is intelligence”, this should be a shorter and more effective description.</p>
<p><strong>Next Token Prediction</strong> Although it may seem simple, it can be justified by the compression theory: this is one of the reasons why teams like OpenAI have long insisted on Next Token Protection. In relative terms, the BERT “predict intermediate” is often difficult to align directly to a strong generating capacity in terms of the end application effect.</p>
<h2>Limitations and summary</h2>
<p>It is not realistic to compress everything: for example, the cost of pixel-level image modelling is enormous. In reality, it is often necessary to identify information fragments that want to be preserved and modelled before finding a way to filter out unwanted irrelevant calculations and information clips, thereby reducing the subset of data being processed before it is compressed without loss.</p>
<p>From the perspective of “compressed intelligence”, the current single-state model aims to continue to improve the effective information compression capability, while the multi-modular model is tasked with finding a compressed modelling method for complex modulate information. BPE (Byte-Pair Encoding) can process the word-formulation of the text, and the method based on statistical frequency is not only efficient, but also ultimately significant, with the greater disadvantage that it may be unfriendly to small languages; however, the sound and video-like mosaics still need more appropriate dissemination and expression.</p>
<p>Many of the data in reality may not be directly observed and cannot simply be expected to be achieved by compressing “all observations”. A more conservative understanding is that compression provides a path to explain the capability of the underlying model to be generalized, but it also needs to be discussed with data selection, modelling, interactive environments, etc.</p>
