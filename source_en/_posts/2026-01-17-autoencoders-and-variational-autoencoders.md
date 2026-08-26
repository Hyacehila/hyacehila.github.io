---
title: 'Autoencoders and Variational Autoencoders: Reparameterization, KL Divergence, and ELBO'
title_zh: 自编码器与变分自编码器：重参数化、KL 散度与 ELBO
date: 2026-01-17 23:58:14 +0800
categories:
- Machine Learning
- Deep Learning
tags:
- Machine Learning
- Autoencoders
- Variational Autoencoders
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers AEs, DAEs, SAEs, VAEs, reparameterization, KL divergence, and ELBO.
description: Covers AEs, DAEs, SAEs, VAEs, reparameterization, KL divergence, and ELBO.
excerpt_zh: 整理 AE、DAE、SAE、VAE、重参数化技巧、KL 散度和 ELBO 等内容。
permalink: /blog/2026/01/17/autoencoders-and-variational-autoencoders/
lang: en
translation_key: 2026-01-17-autoencoders-and-variational-autoencoders
translation_status: machine
translation_source_hash: 27cdf55e634195def721bc3ace9898708ad1f7ff0abe137a2a42ed5793dfa75c
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<blockquote>
<p>From basic to front, step by step, master the encoder architecture in in-depth learning</p>
</blockquote>
<p>The questions in this article can also be addressed<a href="/en/blog/2024/09/02/deep-learning-basics/">In-depth learning base: neuronet, optimization approach and integration</a>、<a href="/en/blog/2024/11/13/deep-learning-network-architectures/">In-depth learning network architecture: CNN, RNN and Seq2Seq</a>How the concept of a relatively close read together is developed in different contexts.</p>
<hr>
<h2>Custom Encoder Base</h2>
<h3>What's a self-encoder?</h3>
<p><strong>Core definitions</strong></p>
<p>The cipher is a kind of...<strong>Self-supervised learning</strong>The nervous network structure. Its core objective is not to predict labels, but to predict labels.<strong>Compression of learning input data</strong>, that is, to re-enter the output of the nervous network as far as possible.</p>
<p><strong>Intuitive understanding.</strong></p>
<p>You can imagine the encoder as&quot;Compression-Depressure&quot;Process:</p>
<ul>
<li><strong>Encoder</strong>: conversion of high-resolution images to a short code vector (condensed)</li>
<li><strong>Codec</strong>: restore original images from code vector (discharge)</li>
</ul>
<p><strong>Core thinking</strong></p>
<p>If a network can successfully recover raw data from a compressed code, the compressed code must contain the most central and important features of the data, leaving out noise and redundancy information.</p>
<h3>Core structure</h3>
<p>The self-codifier consists of three parts:</p>
<pre><code class="language-mermaid">graph LR
    A[输入 x] --&gt; B[编码器]
    B --&gt; C[隐表征 h/z&lt;br/&gt;瓶颈层（潜在空间）]
    C --&gt; D[解码器]
    D --&gt; E[重构输出 x̂]
</code></pre>
<p><strong>Mathistic Symbols Definition</strong></p>
<ul>
<li>&#36;\mathbf{x}&#36;: Enter data, dimensions are &#36;d&#36;（&#36;\mathbf{x} \in \mathbb{R}^d&#36;）</li>
<li>&#36;\mathbf{h}&#36; or &#36;\mathbf{z}&#36;: hidden surfaces (potential variables), dimensions are &#36;p&#36;(usually &#36;p) &lt; d&#36;）</li>
<li>&#36;\hat{\mathbf{x&#125;&#125;&#36;: Reconstruct output, dimensions and &#36;\mathbf{x}&#36; Same</li>
<li>&#36;f_\theta&#36;: Encoding Functions, with parameters &#36;\theta&#36;</li>
<li>&#36;g_\phi&#36;: decode function, parameter is &#36;\phi&#36;</li>
</ul>
<h3>Forward dissemination and training</h3>
<p><strong>Encoding Phase</strong>
&#36;&#36;\mathbf{h} = f_\theta(\mathbf{x}) = \sigma(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e)&#36;&#36;</p>
<p><strong>Decoding Phase</strong>
&#36;&#36;\hat{\mathbf{x&#125;&#125; = g_\phi(\mathbf{h}) = \sigma&#39;(\mathbf{W}_d \mathbf{h} + \mathbf{b}_d)&#36;&#36;</p>
<p><strong>Training objectives</strong>
&#36;&#36;\min_{\theta,\phi} \mathbb{E}<em>{\mathbf{x} \sim \mathcal{D&#125;&#125; [\mathcal{L}(\mathbf{x}, g</em>\phi(f_\theta(\mathbf{x})))]&#36;&#36;</p>
<h3>Loss Functions</h3>
<p><strong>Average error (MSE)</strong>
For numerical data (e.g. image pixel values):
&#36; \mathcal{L}<em>{\text{MSE&#125;&#125;(\mathbf{x}, \hat{\mathbf{x&#125;&#125;) = \frac{1}{N} \sum</em>{i=1}^{N} \lVert \mathbf{x}_i - \hat{\mathbf{x&#125;&#125;_i \rVert_2^2&#36;&#36;</p>
<p><strong>Diutsil Interpolation (BCE)</strong>
For binary data or to be consolidated &#36;[0,1]&#36; Data:
&#36; \mathcal{L}<em>{\text{BCE&#125;&#125;(\mathbf{x}, \hat{\mathbf{x&#125;&#125;) = - \frac{1}{N} \sum</em>{i=1}^{N} \sum_{j=1}^{d} \left[ x_{i,j} \log(\hat{x}<em>{i,j}) + (1 - x</em>{i,j}) \log(1 - \hat{x}_{i,j}) \right]&#36;&#36;</p>
<h3>Why do you need it?&quot;Bottlenecks&quot;</h3>
<p><strong>Problem</strong></p>
<p>If the depth of the hidden layer &#36;p \geq d&#36; And without any other constraints, the network can learn to be constant equivalent (in the form of a map)&#36;\mathbf{h}=\mathbf{x}&#36;It doesn't make any sense.</p>
<p><strong>Solutions</strong></p>
<p><strong>Undeveloped self-encoder</strong>Force &#36;p &lt; d. Forcing the web to learn the most prominent features of the data, similar to the main components of the data captured by the PCA.</p>
<blockquote>
<p><strong>Expert perspective</strong>: If the active function is linear and the loss function is MSE, the full-fledged self-encoder is not equal to <strong>Main Component Analysis (PCA)</strong>I'm sorry. But because the self-encoder uses non-linear activation functions (e. g. ReLU), it learns more powerful than the PCA<strong>Non-linear flow</strong>。</p>
</blockquote>
<h3>Noise-deductor (DAE)</h3>
<p><strong>Core thinking</strong></p>
<p>Add noise to the input data and force the network to learn the roulette character.</p>
<p><strong>Mathistically expressed</strong>
&#36;&#36;00\
\text{noise input:} &amp;\mathbf}=mathbf{x}+ \varepsilon, \varepsilon\m\mmmatbl{, \mathbf^2\mathbf}
\text{training target:} &amp;\mathbf}
\\text{loss function:} &amp;\quad \mathcal{L} = \lVert \mathbf{x} - g_\phi(f_\theta(\tilde{\mathbf{x&#125;&#125;)) \rVert_2^2
\end{aligned}&#36;&#36;</p>
<p><strong>Meaning</strong>: Forced the network to learn the character of a stick, rather than simply copying, and to extrapolate complete information from damaged data.</p>
<h3>Rare Encoder (SAE)</h3>
<p><strong>Core thinking</strong></p>
<p>Allow hidden dimensions &#36;p &gt; d&#36;, but add in loss function<strong>Screeching constraints</strong>。</p>
<p><strong>Loss Functions</strong>
&#36;&#36;\mathcal{L}<em>{\text{SAE&#125;&#125; = \mathcal{L}</em>{\text{reconstruction&#125;&#125; + \lambda \sum_{i} |h_i|&#36;&#36;</p>
<p>Activate near target thinness using KL-dispersible containment &#36;\rho&#36;：
&#36;&#36;\mathcal{L}<em>{\text{SAE&#125;&#125; = \mathcal{L}</em>{\text{reconstruction&#125;&#125; + \beta \cdot D_{\text{KL&#125;&#125;(\rho | \hat{\rho})&#36;&#36;</p>
<p><strong>Meaning</strong>: Limiting the same time to only a very small number of neurons activated, simulated the way the bioneurosystem works.</p>
<h3>From AE to VAE: The qualitative leap</h3>
<p>Although DAE and SAE have improved standards to some extent, they still cannot be solved.<strong>Generate new data</strong>This is the core issue. It's coming. <strong>VAE</strong> The birth.</p>
<hr>
<h2>VAE</h2>
<p>VAE is one of the cornerstones of the deep generation model, which radically changes the generation capacity of the self-codifier by introducing probability distribution.</p>
<h3>Why do you need VAE?</h3>
<p><strong>Review of the standard self-encoder</strong></p>
<ul>
<li>Map input to fixed vectors → Potential space is not continuous</li>
<li>Only&quot;Compression&quot;No, I can't.&quot;Generate&quot;</li>
</ul>
<p><strong>VAE Core Insight</strong></p>
<p>VAE does not map input into one&quot;Points&quot;, it's a map.<strong>Probability distribution</strong>(usually in the Gaussian distribution):</p>
<ul>
<li>I'm not saying.&quot;This is the coordinates. &#36;(3, 2)&#36;&quot;</li>
<li>It's about...&quot;This is the average figure. &#36;(3, 2)&#36; A range around. Internal&quot;</li>
</ul>
<p><strong>Key strengths</strong></p>
<ul>
<li>Introduce probability distribution and regularization.</li>
<li>Yes.&quot;1&quot;and&quot;7&quot;The results of the decoded data are smooth transitions.</li>
<li><strong>You can generate new data</strong>！</li>
</ul>
<h3>Core structure</h3>
<p>VAE contains three key components: probability encoder, sample layer, code decoder.</p>
<h4>Encoder</h4>
<p>Input &#36;\mathbf{x}&#36;, neural network output distribution parameters:</p>
<p>&#36;&#36;\begin{aligned}
\boldsymbol{\mu} &amp;= f_\mu(\mathbf{x}) \
\log\boldsymbol{\sigma}^2 &amp;= f_\sigma(\mathbf{x})
\end{aligned}&#36;&#36;</p>
<ul>
<li><strong>Mean vector</strong> &#36;\boldsymbol{\mu}&#36;: Central location of distribution</li>
<li><strong>logarithmic vector</strong> &#36;\log\boldsymbol{\sigma}^2&#36;: Dispersion of distribution</li>
</ul>
<h4>Reparatification technique — <strong>Core point.</strong></h4>
<p>From Distribution &#36;\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)&#36; Medium Sample &#36;\mathbf{z}&#36; Pass it to the decodor.</p>
<p><strong>Problem</strong>: Direct sampling is random,<strong>Not transposable</strong>. Reverse transmission cannot pass the gradient by random nodes.</p>
<p><strong>Skills</strong>: Will randomize&quot;Strip&quot;Come out.
&#36;&#36;\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})&#36;&#36;</p>
<p><strong>Meaning</strong>: Now &#36;\boldsymbol{\mu}&#36; and &#36;\boldsymbol{\sigma}&#36; Just a certainty parameter calculation, gradients are passed back to the encoder without hindrance, randomity is entered entirely &#36;\boldsymbol{\varepsilon}&#36; Provision.</p>
<h4>Codec</h4>
<p>Enter sampler &#36;\mathbf{z}&#36;, output re-constructing &#36;\hat{\mathbf{x&#125;&#125;&#36;：
&#36;&#36;\hat{\mathbf{x&#125;&#125; = g_\phi(\mathbf{z})&#36;&#36;</p>
<h3>Maths: Deductive extrapolation</h3>
<h4>Target: Maximum logarithmic</h4>
<p>The goal is to make the model generate real data. &#36;\mathbf{x}&#36; Probability &#36;P(\mathbf{x})&#36; Maximize:
&#36;&#36;P(\mathbf{x}) = \int P(\mathbf{x}|\mathbf{z})P(\mathbf{z}) d\mathbf{z}&#36;&#36;</p>
<p>Because this is an incalculable fraction of a complex neural network, we cannot directly optimize it.</p>
<h4>Delineation of the lower boundary (ELBO)</h4>
<p>Include approximation distribution &#36;q_\phi(\mathbf{z}|\mathbf{x})&#36;(coding) Approaching real back check. &#36;p(\mathbf{z}|\mathbf{x})&#36;。</p>
<p>Mathically extrapolated (instinct), obtained &#36;\log P(\mathbf{x})&#36; Bottom boundary:
&#36;&#36;\text{ELBO} \underbrace{E}<em>{\mathbf{z} \sim q}[\log p(\mathbf{x}|\mathbf{z})]}</em>- \underbrace{D {text{KL} (\matbf{x}p}mathbf{z&#125;&#125;  text{regulated entry}</p>
<p><strong>Loss Functions</strong>
&#36;&#36;\mathcal{L}<em>{\text{VAE&#125;&#125; = -\text{ELBO} = \mathcal{L}</em>{\text{reconstruction&#125;&#125; + D_{\text{KL&#125;&#125;(q(\mathbf{z}|\mathbf{x}) | p(\mathbf{z}))&#36;&#36;</p>
<h4>Intuitive interpretation of the loss function</h4>
<p>The VAE Loss is shared by two units.&quot;Fight.&quot;The power of the world:</p>
<p><strong>Reconstructing loss</strong></p>
<ul>
<li><strong>Role</strong>: Make decoded images as original as possible</li>
<li><strong>Preferative</strong>: If you are not bound, you will make a difference &#36;\boldsymbol{\sigma} \to \mathbf{0}&#36;It's degenerated to normal AE.</li>
</ul>
<p><strong>KL Scatter</strong></p>
<ul>
<li><strong>Role</strong>: Forced encoder output distribution nearing standard normal distribution &#36;\mathcal{N}(\mathbf{0}, \mathbf{I})&#36;</li>
<li><strong>Preferative</strong>: If this is only optimized, the encoder will ignore the input and always output the standard noise distribution</li>
</ul>
<p><strong>Balance</strong>VAE strikes a balance between the two, allowing the code to contain original information and allowing potential space to fit the normal distribution shape, ensuring continuity and generating capacity.</p>
<h4>Why VAE generated blurry images</h4>
<p>That's VAE's most famous weakness.<strong>Reason</strong>VAE uses the Gaussian distribution assumptions and MSE losses. MSE tends to be all possible pixels Remove&quot;Average&quot;... leading to the loss of edge details, similar to over-wrought skin.</p>
<p>The normal AE/VAE is usually trained by MSE Los (the pixel-scale average error). MSE has a big problem, which is not sensitive to high frequency textures or which tends to produce “mixed averages”. But in mathematics, it keeps a lot of pixel-level redundancy that people don't care about.</p>
<p>The VQ-GAN or fine-tuned VAE used by LDM introduced Perceptual Los and Patchgan Discriminator. This change forces the compression model to focus on preserving the semantic structure and texture of the picture and its spatial relationship, while ignoring the meaningless random pixel-level noise.</p>
<h3>VAE 's strengths and limitations</h3>
<p><strong>Advantages</strong></p>
<ul>
<li>Training is stable, unlike the way that gan is prone to a pattern collapse.</li>
<li>There are clear probabilistic models and mathematical interpretations</li>
<li>Potential space smooth, consistent, suitable for insertion and exploration</li>
<li>It's a valid reasoning.</li>
</ul>
<p><strong>Limits</strong></p>
<ul>
<li>Generating images tends to be vague (compared to PAN)</li>
<li>The potential dimensions of space need experience to choose</li>
<li>Limited ability to express certain complex data distributions</li>
</ul>
<h3>Beta-VAE: Characteristical detoxification</h3>
<p>Standard VAE, Potential Vector &#36;\mathbf{z}&#36; The dimensions are usually...<strong>- A entanglement.</strong>: Change a value may affect multiple properties (e.g. colour, size, angle) at the same time.</p>
<p><strong>Solutions: adjust KL weights</strong></p>
<p>Modify loss function to weight KL dispersive item &#36;\beta&#36;(usually &#36;beta) &gt; 1&#36;）：
&#36;&#36;\mathcal{L}<em>{\beta\text{-VAE&#125;&#125; = \mathcal{L}</em>{\text{reconstruction&#125;&#125; + \beta \cdot D_{\text{KL&#125;&#125;(q_\phi(\mathbf{z}|\mathbf{x}) | p(\mathbf{z}))&#36;&#36;</p>
<p><strong>Rationale</strong></p>
<p>By increasing &#36;\beta&#36;, forced potential distribution &#36;q_\phi(\mathbf{z}|\mathbf{x})&#36; Strict adherence to standard normal distribution (dependency of dimensions). This kind of constraint forces the model to look for data. Medium<strong>Best effective, most independent</strong>Factor:</p>
<ul>
<li>A dimension only controls colour</li>
<li>The other dimension only controls shape.</li>
<li>The third dimension is only an angle.</li>
</ul>
<p>It's called<strong>DistantReduction</strong>。</p>
<p><strong>Cost and trade-offs</strong></p>
<p>&#36;\beta&#36; Bigger:</p>
<ul>
<li>Potential space is more entanglement and more interpretable.</li>
<li>Reconstructing images is usually blurry (reconstructing error weights is relatively small)</li>
</ul>
<p>A balance needs to be found between the level of de-tachment and the quality of the re-construction.</p>
<p><strong>Theoretically.</strong></p>
<p>&#36;\beta&#36;The-VAE is a significant contribution to the theory of DeepMind, which:</p>
<ul>
<li>It reveals the relationship between potential spatial structures and the solution to the problem.</li>
<li>New ideas for explaining AI</li>
<li>Applications in areas such as intensive learning, robotic control, etc.</li>
</ul>
<hr>
<h2>Revolutionary breakthrough - VQ-VAE</h2>
<p>This is the VAE family.<strong>Most revolutionary</strong>One of the members, proposed by Google DeepMind, broke&quot;Potential space must be continuously distributed by Goss.&quot;The dogma.</p>
<h3>- It hurts.</h3>
<p>Standard VAE assumes that potential variables are continuous, which leads to:</p>
<ul>
<li>The resulting image edge is blurred</li>
<li>The language, the logical concept of human beings is often<strong>Disperse</strong>(e.g.&quot;Cat.&quot;、&quot;Dog.&quot;(Classification concept)</li>
<li><strong>Could not connect to Transformer and Language Model</strong></li>
</ul>
<h3>Solutions: Codebook mechanism</h3>
<p>VQ-VAE Introduction<strong>Codebook</strong> The concept.</p>
<h4>Forward transmission process</h4>
<p><strong>Encoding</strong>
Encoder Output Continuous Vector &#36;\mathbf{z}_e(\mathbf{x})&#36;</p>
<p><strong>Quantification of vectors</strong>
Finds the nearest vector in the code book:
&#36; \mathbf{z} (\mathbf{x}) =\text{Codebook}[k^]<em>], \quad k^</em> = \arg\min_k \lVert \mathbf{z}_e(\mathbf{x}) - \mathbf{e}_k \rVert_2&#36;&#36;</p>
<p>of which &#36;\mathbf{e}_k&#36; It's the first in the code book. &#36;k&#36; A code vector.</p>
<p><strong>Decoding</strong>
The decodor receives a quantified vector &#36;\mathbf{z}_q(\mathbf{x})&#36;, output re-constructing &#36;\hat{\mathbf{x&#125;&#125;&#36;</p>
<h4>Direct to estimate</h4>
<p>Because&quot;Checklist&quot;and&quot;Quick Search&quot;is not a guideable operation, VQ-VAE is used <strong>Straight-Through Estimator</strong>：</p>
<ul>
<li><strong>Forward transmission</strong>: Using a quantified vector &#36;\mathbf{z}_q&#36;</li>
<li><strong>Reverse transmission</strong>: Copy gradient directly to encoder output &#36;\mathbf{z}_e&#36;</li>
</ul>
<p>&#36;&#36;\nabla_{\mathbf{z}<em>e} \mathcal{L} = \nabla</em>{\mathbf{z}_q} \mathcal{L}&#36;&#36;</p>
<h4>Loss Functions</h4>
<p>&#36;&#36;\mathcal{L}=underplace(lVert\mathbf{x} - \\mathbf}\rVert 2^} underbrace{lVert\sec&#125;&#125;(mathbf}) - \mathbf{e}<em>k \rVert_2^2}</em>+ \lderbrace \lVert\mathbf{z} e (mathbf{x} -\text{stop}<em>k) \rVert_2^2}</em>- Next{encoding committee}</p>
<h3>Why, VQ-VAE is revolutionary.</h3>
<h4>It's very clear.</h4>
<p>Forced use of discrete, high-quality character codes, leaving vague intermediates behind.</p>
<h4>Connect Transformer</h4>
<p>This is the most critical innovation. Because potential space becomes discrete (like word Token), images can be transformed into a series of Token sequences, using GPT/Transformer directly to process images!</p>
<h4>Practical application</h4>
<ul>
<li>OpenAI <strong>DALL-E 1</strong> VQ-VAE based variant</li>
<li>Audio Generation Model <strong>MusicLM</strong>、<strong>AudioLM</strong></li>
<li>Multi-modular Models (e.g., the visual capability component of GPT-4o)</li>
</ul>
<h3>Why VQ-VAE Important</h3>
<p>VQ-VAE Yes<strong>Bridges to visual and language</strong>：</p>
<ul>
<li>Disperse continuous images into Token</li>
<li>Make possible a unified language-visual architecture</li>
<li>Laying the foundations for a large multi-modular model</li>
</ul>
<hr>
<h2>Modern Extension - MAE</h2>
<p>Not even in the name.&quot;Variational&quot;But MaE is Autoencoder's idea of <strong>Transformer Times</strong>The continuation.</p>
<h3>Background: From BERT to visual</h3>
<p>BERT, kill the Quartet in the NLP field.&quot;Full Fill&quot;Thought. Mae moved this idea to computer vision.</p>
<h3>Core approaches</h3>
<p><strong>Image Segment</strong>
Cutting pictures into small pieces, such as &#36;16 \times 16&#36; Pixels.</p>
<p><strong>Random mask</strong>
<strong>Throw away 75% of the pieces at random.</strong>(Masking) - Note the high rate!</p>
<p><strong>Encoding</strong>
Only the remaining 25 percent is fed to the encoder.</p>
<p><strong>Decoding</strong>
The decoding device is responsible for completing the 75% of the pieces that were thrown away.</p>
<h3>Mathistically expressed</h3>
<p><strong>Mask Policy</strong>
&#36;&#36;\mathbf{M} \in {0, 1}^{N \times N}, \quad \sum_{i,j} M_{i,j} \approx 0.25 \times N \times N&#36;&#36;</p>
<p><strong>Restructure the target.</strong>
&#36;&#36;\mathcal{L}<em>{\text{MAE&#125;&#125; = \frac{1}{|\mathcal{U}|} \sum</em>{i \in \mathcal{U&#125;&#125; \lVert \mathbf{x}_i - \hat{\mathbf{x&#125;&#125;_i \rVert_2^2&#36;&#36;</p>
<p>of which &#36;\mathcal{U}&#36; is the index of the masked block.</p>
<h3>Why is MaE working?</h3>
<p><strong>Force semantic learning</strong>
If you don't understand,&quot;Dog.&quot;The semantics of the dog cannot be filled with the veiled head.</p>
<p><strong>Efficient pre-training</strong></p>
<ul>
<li>Only 25% of data processed, calculated efficiency High</li>
<li>The masked mission forces models to learn global dependency.</li>
</ul>
<h3>Meaning and impact</h3>
<p>MAE proves Autoencoder structure is in progress<strong>Self-supervised learning</strong>The Great Tool:</p>
<ul>
<li>Without labels, the model can understand the semantics of the image.</li>
<li>Now, many high-performance visual models are trained in this way.</li>
<li>To lay the foundation for the success of Vision Transformer's computer vision.</li>
</ul>
<hr>
<h2>Integrated application scenario</h2>
<p>The self-codifier and its variants are very widely applied in practice. The following are the main areas of application.</p>
<h3>Decline and Visualization</h3>
<p>Similar to t-SNE or PCA, compress high-dimensional data to 2D or 3D for visualization or reduce the amount of calculation as a pre-processed step.</p>
<h3>Unusual detection</h3>
<p><strong>Core logic</strong>: With a large amount&quot;Normal Data&quot;Training AE. When Input&quot;Unusual Data&quot;This is when the re-engineering error increases significantly.</p>
<p><strong>Decision Formula</strong>
&#36;&#36;\text{Anomaly}(\mathbf{x}) = \mathbb{I}[\mathcal{L}(\mathbf{x}, \hat{\mathbf{x&#125;&#125;) &gt; \tau]&#36;&#36;</p>
<p><strong>Apply</strong>: Credit card fraud detection, early warning of industrial equipment failure</p>
<h3>Image to Noise and Fix</h3>
<p>Use the denocator idea:</p>
<ul>
<li>Remove the noise from the old picture.</li>
<li>Complete the masked part of the image</li>
</ul>
<h3>Feature extraction and pre-training</h3>
<p>When label data are scarce, a large amount of unlabelled data is used to train the self-codifier. And then keep it.<strong>Encoder</strong>Partially, access to classification layers fine-tuned.</p>
<p>This method is widely applied in models such as BERT.</p>
<h3>Generate new data (VAE)</h3>
<p>After training, the encoder is discarded. Directly from &#36;\mathcal{N}(\mathbf{0}, \mathbf{I})&#36; Sample Random Vector &#36;\mathbf{z}&#36;, feed the decoding device to create an absence of a human face or scene. Mainstream generation models have largely ceased using VAE as the generation structure, but instead used it for compression.</p>
<h3>Potential space plug-in (VAE)</h3>
<p>Take two figures A and B, coded separately &#36;\mathbf{z}_A&#36; and &#36;\mathbf{z}_B&#36;。</p>
<p>Calculates the intermediate vector:
&#36;&#36;\mathbf{z}_{\text{mid&#125;&#125; = \alpha \mathbf{z}_A + (1-\alpha)\mathbf{z}_B, \quad \alpha \in [0,1]&#36;&#36;</p>
<p>Decoding &#36;\mathbf{z}_{\text{mid&#125;&#125;&#36;, you can see that Figure A smoothly becomes Figure B.</p>
<h3>The blog is a blog for the Global Voices community.&#36;\beta&#36;-VAE）</h3>
<p>Jean. &#36;\mathbf{z}&#36; Each dimension controls the independent feature.</p>
<p>For example:</p>
<ul>
<li>&#36;z_1&#36; Control hair</li>
<li>&#36;z_2&#36; Controlling the colour of skin</li>
<li>&#36;z_3&#36; Control angle</li>
</ul>
<p>Adjustment &#36;z_1&#36; , only the color changes, the rest remains unchanged.</p>
<h3>Characteristic decoupling</h3>
<p>Determines what dimensions of the code represent what information.</p>
<p>For example, in a 100-dimensional vector, the first 50-dimensional represents the sentence content and the second 50-dimensional represents the talking person character.</p>
<h3>Disperse the hidden signs.</h3>
<p>Forced encoded as a single heat vector (only 1 dimensional and the rest 0), which allows for unsupervised classification.</p>
<p>For example, handwritten digital recognition (0-9) and training self-coding machines to force 10-dimensional code to be a single heat vector. This 10 unique thermal codes may correspond to a single number each, leading to a complete unsupervised classification learning.</p>
<h3>Data compression</h3>
<p>Encoder output is a low-dimensional vector and can be considered as a direct compression result:</p>
<ul>
<li>Encoder Executes Compression</li>
<li>Decoding Decoder Executing Decompression</li>
<li>It's a decompression.</li>
</ul>
<h3>Stable Diffusion - Most important application</h3>
<p><strong>This is the most important application at this time.</strong>I'm sorry. Stable Diffusion, actually, is called&quot;Latent Diffusion Model&quot;：</p>
<ul>
<li>Not just processing large pictures in pixel space.</li>
<li>First <strong>VAE</strong> Compress pictures to potential space</li>
<li>In this small space, it is spreading.</li>
<li>Last use. <strong>VAE Decoding</strong>Revert to "Big Chart"</li>
<li>The self-coding is already the basis of the current mainstream generation model.</li>
</ul>
<h3>Multi-modular Model (VQ-VAE)</h3>
<ul>
<li><strong>DALL-E</strong>:VQ-VAE discrete expression +GPT</li>
<li><strong>MusicLM</strong>: Dispersion of audio + language model</li>
<li><strong>GPT-4o</strong>: Visual ability is based in part on similar discrete expressions</li>
</ul>
<hr>
