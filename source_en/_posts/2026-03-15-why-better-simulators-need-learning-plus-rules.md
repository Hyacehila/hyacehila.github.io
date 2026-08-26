---
title: 'Why Better Simulators Often Combine Learning and Rules: From PDEs and Ray Tracing to DLSS'
title_zh: 为什么更好的 Simulator 往往是 Learning + Rules：从 PDE、光线追踪到 DLSS
date: 2026-03-15 21:00:00 +0800
categories:
- Data Science
- Time Series & Spatial Data
tags:
- Survey
author: Hyacehila
mathjax: true
excerpt: 'Better simulators often come from a clear division of labor: encode conservation, geometry, causality, boundaries,
  and rendering structure, then let learning cover expensive or fuzzy parts.'
description: 'Better simulators often come from a clear division of labor: encode conservation, geometry, causality, boundaries,
  and rendering structure, then let learning cover expensive or fuzzy parts.'
excerpt_zh: 更好的 simulator 往往来自明确分工：把守恒、几何、因果、边界条件与渲染/求解结构当作归纳偏置，再让 learning 去补昂贵、模糊或难解析的部分。
permalink: /blog/2026/03/15/why-better-simulators-need-learning-plus-rules/
lang: en
translation_key: 2026-03-15-why-better-simulators-need-learning-plus-rules
translation_status: machine
translation_source_hash: de289e2136769ae19bc6831cb629df36d1da74c78f38faa2495acb1b7f751002
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>tldr: Better simulator usually retains the known world structure and allows the slowness of the resolution, incomplete modelling or the difficulty of expressing it directly.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/03/18/stochastic-process-basics-notes/">Random process basis: random process definition, digital characteristics and smooth process</a>、<a href="/en/blog/2024/01/01/statistical-forecasting-notes/">Statistical projections: qualitative projections, quantitative projections and extrapolations of trends</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h2>Before we start: Pure rules and pure learning are a false issue</h2>
<p>Many discussions put rules and learning on the opposite sides of zero and one side: the one hand, the other, the rule of resolution, the numerical solution and the artificial modelling, and the other, data, the neural network and end-to-end learning. The powerful systems in reality usually do not have such a neat border, and they combine the parts that each side is good at.</p>
<p>Simulator can be summarized as either an interpretable state shift in the restricted state space or the generation of observations according to the state. Fluid simulations, material simulations, light tracking and visual reconstruction are actually questions of the same kind: which states are legal, how systems can evolve, and how observations can be generated from these states.</p>
<p>Take this frame apart and look at it:</p>
<ul>
<li>Fluid simulation: state is the speed and pressure field, binding from the Navier-Stokes equation and unpressible condition, state shift is time step, and observations are measurable flow speed, pressure or resistance.</li>
<li>Material simulation: state is stress-resilient and positional shift, binding from the relationship to power balance, state transfer is evolution on the load path, and observations are shape-forming, cracking or fracture.</li>
<li>Light tracking: state is the distribution of light in the scene, binding from the rendering equation and geometric visibility, "evolution" is the transmission of light in reflection, refraction, dispersion, and observations are the final pixel values.</li>
<li>Visual Rebuilding (NeRF, 3DGS): The state is the bottom 3D scenario, which is derived from the geometry of camera models and multiple views, and observations of images from various perspectives.</li>
</ul>
<p>This perspective will bring many approaches to a more specific design issue:<strong>Which constraints should be written into the system and which parts could be left for data learning.</strong> The more binding the space in the state, the more efficient the search is, the less efficient the modelling elasticity; the more the data are given, the more flexible the model, the more likely it is to be illegal decomposition or OOD failure. The design of the system is essentially to allocate this bound budget.</p>
<p>The real world is not without structures. Constantity, continuity, boundary conditions, camera projection, shielding, reflection, refraction, localism and time consistency are all hard constraints. Since they already exist, it is generally more cost-effective to write them directly as summary bias (bias).</p>
<p>I don't agree that "rule-based will be completely replaced by learning." Closer to engineering practice is described as the rule system is evolving into a hybrid system with structured casings and learning modules. The rules build the skeleton, leaving the expensive, vague or hard-to-man modelling parts.</p>
<p>This explains the changes in recent years: the classical computation problem has not been replaced by the whole of the datadiving method. The mainstream approach in the past has preferred rules to cover the whole system; now, local tasks such as closure, sorrogate, counter-issue, reconstruction, noise reduction, ultra-resolution and near-suggestion can be given to learners. The space of status, geometry, constant restraint, boundary conditions and feasible decomposition are also of greater importance, limiting the scope of learning devices to work.</p>
<p>In the light of this division of labour, we will discuss three issues later:</p>
<ul>
<li>Traditional simulator, why natural is rule-heavy? Because the ruler defines the learning space: constantness, geometry, boundary conditions, visibility, sampling structure, camera models and grid-bending to determine what answers are acceptable.</li>
<li>What calculations did you actually take? It is more suited to expensive mapping, local patterns and sensory reconstruction that are difficult to write. The idea is to learn how to map the parameters to solve; the DLSS is how to recreate a stable image with thin, low-resolution or noise signals.</li>
<li>Why do you always make a stronger system? The model can concentrate on the remaining parts of the structure, after a summary of the structure is biased, while balancing efficiency, breadth and consistency.</li>
</ul>
<p>Note: YC first published RFS, 2024 Summer, which refers to this issue.</p>
<h2>What did the rules system solve first?</h2>
<p>First, scientific calculations. The classic PDE solver starts from the control equation and then processes dispersion, grid and numerical stability; its goal is never to give a direct answer based on similar data. The most abstract writing is usually like this:</p>
<p>&#36;&#36;
\partial_t u + \mathcal{N}[u] = 0
&#36;&#36;</p>
<p>of which &#36;u&#36; Is state,&#36;\mathcal{N}[u]&#36; It's a calculator. Practical solvency also addresses fragmentation, grids, numerical stability, border conditions, time propulsion and error control. Classic Solver has answered several key questions in his design: What meets the physical norms, what updates will spread and which borders cannot be destroyed.</p>
<p>The value of such systems is not only precision, but also clarity as to which operations cannot be performed. They may be slow, not friendly to complex counter-issues and costly in multiple inquiries, but they can be powerful in limiting the space for legal status.</p>
<p>The same goes for graphics. Early <a href="https://cseweb.ucsd.edu/~viscomp/classes/cse274/wi26/readings/whitted.pdf">Whitted 1980</a> The retrospective light tracking has essentially made the geometric-optical coding of reflection, refraction, shadows and so forth a clear set of rules. Here we are. <a href="https://www.cs.rpi.edu/~cutler/classes/advancedgraphics/S13/papers/kajiya_rendering_equation_86.pdf">Kajiya 1986</a>The rendering equations are used to write "the radiation brightness of a point in the scene in a direction" as a form of fraction:</p>
<p>&#36;&#36;
L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{\Omega} f_r(x, \omega_i, \omega_o) L_i(x, \omega_i) (n \cdot \omega_i) , d\omega_i
&#36;&#36;</p>
<p>This formula does not suddenly make the rendering simple, but it turns the "where the image comes from" into a physical problem of self-conformity. The decades since then, path trafficking, migration Samling, MIS and denoising have largely evolved around this issue.</p>
<p>The common denominator of PDE Solver and Ray Tracer is not all hand-held, but all of the key structure graphics are written into the system:</p>
<ul>
<li>(a) Status space is defined in a visible manner;</li>
<li>Legal updating is regulated by rules;</li>
<li>There is a theory of error analysis and stability;</li>
<li>Each module knows what it looks like.</li>
</ul>
<p>And that's why classic simulator, slow, rarely gives strange output. They are expensive, rough, difficult, but not very good at exporting completely illegal answers. For many scientific and engineering missions, this characteristic is in itself an asset.</p>
<table>
<thead>
<tr>
<th>Route</th>
<th>- Where's the first check?</th>
<th>What's best at?</th>
<th>Typical shortboard</th>
<th>Representative scene</th>
</tr>
</thead>
<tbody><tr>
<td>Pure rules</td>
<td>Equation, Geometry, Parsing Approximation, Numerical Format</td>
<td>Explanatory, stable, clear-cut.</td>
<td>Slow, inexorable, difficult to cover complex perceptions</td>
<td>CDF, FEM, track.</td>
</tr>
<tr>
<td>♪ Clearing</td>
<td>Data distribution, parameter alignment, end-to-end target</td>
<td>Fast, flexible, reversible and re-structuring</td>
<td>OOD Weakness, Unguaranteed Legitimacy</td>
<td>Image Rebuild, Approximate Subroogate</td>
</tr>
<tr>
<td>hybrid</td>
<td>Rules to structure, leaving error or map</td>
<td>Balancing efficiency, inclusiveness and coherence</td>
<td>System design is more complex</td>
<td>learned simulators、NeRF、DLSS</td>
</tr>
</tbody></table>
<h2>What part of the Learning is taking over?</h2>
<p>In practice, for almost 10 years, the most frequent tasks that Learning takes over are the local tasks of physical processes that are costly but stable.</p>
<p>The first category is surrogate/ emulator. Original Solver is expensive, and scans of the same equation, boundary conditions or parameters are repeatedly searched. The learning device here is a near-approximate algorithm: given parameters, geometry or opening value, and quickly returns to approximate solvency. <a href="https://arxiv.org/abs/1910.03193">DeepONet</a>、<a href="https://arxiv.org/abs/2010.08895">FNO</a> and <a href="https://arxiv.org/abs/2010.03409">MeshGraphNets</a> All of them belong to this route.</p>
<p>The second category is crease/unresolved scale modeling. Small scale effects are not easily visible modelling in many real systems, such as flow flow patterns, sub-grid parameterization, complex material response, and parmeterration in Earth systems. Here, learning is about a closed item that is not perfect or expensive. You didn't abandon the equation; you replaced the most difficult local module in the equation.</p>
<p>The third category is invert problem. Forward simulations often know how to do it, but in turn it is difficult to “recover from observation to state, geometry, materials, parameters”. In this direction, learning tends to be more advantageous than pure optimization, as it is naturally suitable for moving from observation to potential variable space. Yeah. <a href="https://arxiv.org/abs/2111.12503">NVDiffrec</a> This inversion system is the question of turning the "from image to geometry, materials, light" into a question of fine-tuning and learning.</p>
<p>The fourth category is reconstruction/denoising/sub-resolution. This is particularly evident in real-time graphics. Path tracking can give you high-quality signals, but sampling budgets are never enough, so images can be noiseful, resolution low, time-scattered. At this point, learning is not about taking over the light itself, but about how to recover more stable images from thin, noise-intensive, incomplete signals. DLS with Ray Reconstruation is exactly the paradigm.</p>
<p>Thus, learning is often used in four types of work: expensive but repeated queries, hard-to-write closed items, reverse mapping from observations to hidden variables, and reconstruction from incomplete signals to high-quality results.</p>
<p>The development of the Data Driven methodology has not weakened the role of the rules. It is more like a reminder that we have to get the problem right before we decide which parts are worth learning.</p>
<h2>How do you get into the learning system?</h2>
<p>“Rules as a general bias” can easily be described as an abstract slogan. Dismantling it into four layers of expression, objective, structure and reasoning would make it easier to fit into specific design.</p>
<h3>1. Express layer bias: determine what the status space looks like first</h3>
<p>The difference is very wide in the direct output of the model pixels, the direct output grid nodes, and the direct output hidden fields. It means that it is more likely that models will learn something and much harder to learn.</p>
<ul>
<li>In the PDE, grids, cloud, spectrum, function space are all expressed differently.</li>
<li>In graphics, the same is true of mesh, radiance field, and Gaussian budgets.</li>
<li>In the time-series system, the visible last state is also completely different from the pure operation model.</li>
</ul>
<p>A lot of progress has been made at the representation level. <a href="https://research.nvidia.com/publication/2022-07_instant-neural-graphics-primitives-multiresolution-hash-encoding">Instant-NGP</a> The key is multi-resolution Hash encoding, not a larger network;<a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/">3D Gaussian Splatting</a> The video is a very popular video of the event, and it is a very interesting one to recast it as a highly reductive Gaussian project. For neurographs and AA4S, the expression of pairs is often more important than simply deepening the network.<strong>The importance of sign-learning remains, or the most important issue in in-depth learning.</strong></p>
<h3>2. Target layer bias: miscalculation of what is lost</h3>
<p>The most intuitive point of PINNs is that it writes about the loss function of the PDE reidoual. You don't just monitor data points, but rather tell the network in a visible way: these connections, these boundaries, these persistences, can't go against them.</p>
<p><a href="https://www.nature.com/articles/s42254-021-00314-5">Karniadakis et al.</a> Describes the philsics-informed learning as the integration of noisy data with philical law. For system design, the focus is on the definition of "loss " , which determines which space the model will optimize; it is not a subsidiary that is added at the end of the training, but is very important surrogate.</p>
<h3>3. Architecture level Bias: Writing interactive modes into the network</h3>
<p><a href="https://arxiv.org/abs/1910.03193">DeepONet</a> Learn from the brancy/trunk structure, oprator;<a href="https://arxiv.org/abs/2010.08895">FNO</a> (b) Move the nucleus to Fourier space;<a href="https://arxiv.org/abs/2010.03409">MeshGraphNets</a> The local interaction and the pamphleting of mesh is directly embedded in the map network;<a href="https://jmlr.org/beta/papers/v24/23-0064.html">Geo-FNO</a> A visible treatment of geometry is not just a rule grid for FFT.</p>
<p>The common denominator of these approaches is that they do not pretend that the world is a table of any kind. They plug locality, pounce, frequency-area structure, function-to-function mapping into the architecture.</p>
<h3>4. Debrillator level bias: embed the learning device in a slver or rendering loop</h3>
<p>The strongest hybrid system often allows the learning module to work directly in the original solver ring. For example, a micro-rendering device is placed in an optimised loop; many scientific MLs emulator, conditioner or preconditioner are embedded in the original numerical process.</p>
<p>This layer is particularly like the engineering wisdom of reality: preserving the reliable parts of the old system and replacing the most expensive section with the most expensive one.</p>
<pre><code class="language-mermaid">graph LR
  A[现实世界规则&lt;br/&gt;守恒 几何 边界条件 可见性 采样] --&gt; B[归纳偏置&lt;br/&gt;表示 损失 架构 推理环]
  B --&gt; C[学习模块&lt;br/&gt;surrogate closure inverse mapping denoiser super-resolution]
  C --&gt; D[混合 simulator / model]
  A --&gt; D
</code></pre>
<p>I'll take this as the core of the whole text, international model: the rule is not the opposite of the rule, but the compression a priori of the rule.</p>
<h2>Case 1: scientificly calculated</h2>
<p>Scientific course learning best illustrates how the learning + rules are found in a specific system.</p>
<h3>Starting with PINNs: Write PDF directly into the training target</h3>
<p><a href="https://maziarraissi.github.io/PINNs/">PINNs</a> The classic is a close-up of the Internet. &#36;u(t, x)&#36;, then automatically microstructed the PDE reidoual, and added the primary value, boundary and equation to the loss. It's very attractive:</p>
<ul>
<li>There is no need to rely entirely on large-scale labelling data;</li>
<li>- Yes, yes, yes.</li>
<li>(a) Direct access to the control equation;</li>
<li>A priori could be translated into effective monitoring in a data-scarce landscape.</li>
</ul>
<p>PINNs also states that adding philsics to the list does not automatically make the problem simpler. <a href="https://www.nature.com/articles/s42254-021-00314-5">Nature Reviews Physics 2021 General</a> Their capabilities and constraints are clearly summarized: such approaches have great potential in terms of forward / inversion issues, but scalability, rutility and standardization are still central challenges. Physical a priori is important, as is the way of training.</p>
<p>In my opinion, PINNs are the old kind of way to put rules straight into learning. It shows the entire field that the neural network can use not just i.i.d. samples, but also problem structures; but it's far from over.</p>
<h3>From individual to solver: DeepNet and FNO</h3>
<p>And the next crucial step is to learn from one of the pDEs to one of the pDEs.</p>
<p><a href="https://arxiv.org/abs/1910.03193">DeepONet</a> The meaning is here. It writes questions as a map of functions to functions, encodes functions with a blancy net, and encodes the output position with a trunk net. It's not just a static approximator, but an imprato:</p>
<p>&#36;&#36;
\mathcal{G}: a(x) \mapsto u(x)
&#36;&#36;</p>
<p>This step is like upgrading simulator from a single solver to a reusable mapr. You will not solve each parameter example in an inverted manner, but rather train an approximation that can be used repeatedly.</p>
<p><a href="https://arxiv.org/abs/2010.08895">FNO</a> Go further, turn the Kernel parameter into Fourier space. The first results have shown that on PDs like Burgers, Darcy, Navier-Stokes, it can be much faster than traditional solvers and, in some settings, demonstrates the ability of zero-shot super-resolument. For missions that require repeated queries about the same type of an optator, learning to start is often more cost-effective than solving the value from the beginning.</p>
<h3>From the rule grid to complex scaleup: MeshGraphNets and Geo-FNO</h3>
<p>However, the real question will soon be with you: the rule grid is not the whole world. Engineering issues often involve complex boundaries, irregular grids, deformations and multiscale coupling.</p>
<p><a href="https://arxiv.org/abs/2010.03409">MeshGraphNets</a> Very representative. It does message passing directly on mesh graph and incorporates information into forwarding memory. Dispersion structures themselves are physical bias: models do not have to level everything to rules, but can use the system's original pounces.</p>
<p><a href="https://jmlr.org/beta/papers/v24/23-0064.html">Geo-FNO</a> Another limitation was addressed. The classic FNO relies on FFT, which is more suitable for the rule grid and rectangular domain; Geo-FNOs apply FNOs in subspace by placing any geometry into the space. When the problem is not fit into the original structure, it is common to rewrite the bias, not to give up the bias.</p>
<h3>Direction for nearly a year: greater emphasis on validation and more on system collaboration</h3>
<p>Three changes have been particularly noteworthy in the past year.</p>
<p>First, the field started to discuss benchmark and OOD more seriously. <a href="https://www.nature.com/articles/s44172-025-00513-3">2025 benchmark on complex geometrical flow forecasting</a> It was noted that traditional simulation was accurate but expensive and that SciML was used to pursue faster and more scalable programmes; the gap between methods quickly became apparent when geometry became complex, distribution was skewed or precision requirements were improved. The question has shifted from “can you learn” to “what boundaries do you learn to be reliable in sorrogate”.</p>
<p>Second, the study re-enacts the solver back in the ring. <a href="https://openreview.net/forum?id=DPzQ5n3mNm">SC-FNO</a> The 2025 work is not just about integrating itself, but also about sensibility, inversion issues and distributive legal solutions. Numerical structure is being re-introduced.</p>
<p>Finally, emulator started to be treated as a software component, not as a demo in the paper. <a href="https://www.nature.com/articles/s43247-026-03238-z">2026 climate emulator perceptive</a> It is proposed that simulator and emulator should co-design, benchmark should have machine-learning-ready, emulator should also be deployed and analysed as reliable software components. This provides clearer criteria for the engineering of the Learned simulator.</p>
<table>
<thead>
<tr>
<th>Methodology</th>
<th>Learn from what?</th>
<th>Where do you get in, Rules?</th>
<th>Advantages</th>
<th>Typical problem.</th>
<th>Representative sources</th>
</tr>
</thead>
<tbody><tr>
<td>PINNs</td>
<td>Individual PDF solves or reverses problem parameters</td>
<td>PDE resaidual, border conditions, constant entry</td>
<td>Small sample, counter-problem friendly.</td>
<td>Training in pathological and mesometric conditions</td>
<td><a href="https://maziarraissi.github.io/PINNs/">PINNs</a>, <a href="https://www.nature.com/articles/s42254-021-00314-5">Overview</a></td>
</tr>
<tr>
<td>DeepONet / FNO</td>
<td>Parameters to solve</td>
<td>lanch-trunk structure, frequency volume</td>
<td>Multiple queries fast, learningable functions to function map</td>
<td>OOD and complex geometry constraints</td>
<td><a href="https://arxiv.org/abs/1910.03193">DeepONet</a>, <a href="https://arxiv.org/abs/2010.08895">FNO</a></td>
</tr>
<tr>
<td>MeshGraphNets</td>
<td>Grid Dynamics Rollout</td>
<td>Mesh topology, local interactions, daaptity</td>
<td>It's for complex pouncers and shapes.</td>
<td>Long-term stability, and hierarchy is difficult.</td>
<td><a href="https://arxiv.org/abs/2010.03409">MeshGraphNets</a></td>
</tr>
<tr>
<td>recent physics-informed operator variants</td>
<td>Continue sensitivity, geometry and Solver structure on an optator learning</td>
<td>differentiable solvers、geometry-aware mapping、benchmark co-design</td>
<td>Closer to real engineering.</td>
<td>Complex system, high validation requirements</td>
<td><a href="https://jmlr.org/beta/papers/v24/23-0064.html">Geo-FNO</a>, <a href="https://openreview.net/forum?id=DPzQ5n3mNm">SC-FNO</a>, <a href="https://www.nature.com/articles/s44172-025-00513-3">2025 benchmark</a></td>
</tr>
</tbody></table>
<p>The main line of Scientific ML can be summarized as: Find the most valuable calculation in the PDF family and let the web learn this part.</p>
<h2>Case 2: graphics capable of giving, NeRF and 3DGS</h2>
<p>Scientific calculations show that the rule of rendering is still fully preserved by many of the seemingly Data-driven methods.</p>
<h3>Differentiable rendering: first with replaying rings, then with learning.</h3>
<p><a href="https://arxiv.org/abs/2111.12503">differentiable rendering / inverse rendering</a> The idea was simple: I had visual observations, I wanted to restore geometry, materials, light, so I built a retrograde rendering that would rediscover the gradient and rediscover the difference through the retrofitting process to the hidden variable.</p>
<p>The basis of this process is not learning, but rather the following visible structure:</p>
<ul>
<li>Camera model;</li>
<li>Visibility and projection;</li>
<li>Photo and material parameteration;</li>
<li>Micro-Rasterization or Monte Carlo obtaining;</li>
<li>Geometrics are expressed with mesh extraction.</li>
</ul>
<p><a href="https://arxiv.org/abs/2111.12503">NVDiffrec</a> And NVIDIA about <a href="https://developer.nvidia.com/blog/differentiable-slang-example-applications/">Differentiable Slang / nvdiffrec</a> The materials are typical: learning and optimization restores the sape, material, lighting, but only if the replay loop itself is highly structured. Learning to solve with a rendering equation.</p>
<h3>NeRF: Looks like a nervous field, actually standing on a traditional shoulder. Go, go, go!</h3>
<p><a href="https://arxiv.org/abs/2003.08934">NeRF</a> It's often used as a direct study of the 3D world by the neural network, but it binds the expression of the nerve closely to the classic volume environment.</p>
<p>NeRF enters a 3D position and perspective direction, output density and radiation colour; images are generated by the cumulative sampling, fraction and transceiving rate along the camera Rays, a clear set of retrofitting processes. The division of labour is as follows:</p>
<ul>
<li>(a) The camera posture is known or is estimated;</li>
<li>Radio sampling is a geometric process of writing a death;</li>
<li>Colour synthesis compliance render-up;</li>
<li>Optimizing targets relies on multiple-view geometry.</li>
</ul>
<p>NeRF expresses the scene as neurosis while retaining the structure of the rendering process, which is typical of the hybrid.</p>
<h3>Instant-NGP: Structurally expressed speed</h3>
<p>NVIDIA's <a href="https://research.nvidia.com/publication/2022-07_instant-neural-graphics-primitives-multiresolution-hash-encoding">Instant-NGP</a> It's not just speed that matters. It shows that upgrading is not necessarily based on larger models, and stronger bias can change cost structures.</p>
<p>It's multi-resolution Hash table to store a resourceable web, with a small network. This design significantly reduces the cost of training and reasoning, and compresses the training time of high-quality neural grampics from time to day to seconds or minutes. When spatial structures enter the code, the network does not need to learn the geometry organization from scratch.</p>
<h3>3D Gaussian Spratting: Select a scene that can be directly retrofitted</h3>
<p><a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/">3D Gaussian Splatting</a> The project page gives a very direct result: methods that combine 3D Gaussians, intelleaved optification/device control and visualization-aware environment, achieve high quality novel-view sensesis at 10080p.</p>
<p>3DGS provides more than "fast and good." When the problem has a clear geometric and rendering structure, the shift of the network from a universal accesser to a part of a powerful structured representation is often more consistent with the calculation path.</p>
<p>From NeRF to Instant-NGP, and then to 3 DGS, the rules remain on the scene. The network only learns about appearance, sustainability, local detail and more difficult to write.</p>
<h2>Case 3: Ray Tracing, Denoiser and DLSS in Real Time Graphics</h2>
<p>DLSS in real time graphics places this division in a more visible engineering landscape.</p>
<h3>Why does light follow nature?</h3>
<p>Light pursuit or path tracking follows the mechanism for the generation of geometry, shadows, reflection, reflection and global light. The costs are also straightforward: when the budget for sampling is insufficient, the results will be loud. Real-time retrofitting of the budget is limited and simply increases the sample is not a sustainable solution.</p>
<p>The engineering treatment is usually:</p>
<ul>
<li>(a) The use of Ray traffic to generate a substrate of physical restraint;</li>
<li>(b) Use denoiser/reconversion to convert limited samples into visible images;</li>
<li>Use super-resolution and frame promotion in exchange for real-time sex.</li>
</ul>
<p>This is a very clear hybridback: the bottom structure is still rule-based, exceptingly familiar.</p>
<h3>DLSS 3.5: Replace a set of denoisers with a unified searchr</h3>
<p><a href="https://www.nvidia.com/en-us/geforce/news/gfecnt/20238/nvidia-dlss-3-5-ray-reconstruction/">NVIDIA official description of DLS 3.5 Ray Regulation</a> It is noted that Ray Reconstruation replaces multiple hand-tuned images by single non-urgical network to improve the quality of images.</p>
<p>DLSS 3.5 focuses not on AI being more photo-stalking than physics, but on being able to replace a single set of dedicated denoisers relying on manual transfer with a single line of guidance when there are Ray-traced signature noises due to budget constraints.</p>
<p>It's like the science calculation's reading pattern: the bottom rule is still, the learning machine takes over the most expensive, hard to adjust manually, and most dependent on the private project.</p>
<h3>DLS4 and 4.5: The learning module still relies on rendering lines</h3>
<p>The official NVIDIA data as of March 15, 2026, is a source of information about the situation in the country.<a href="https://www.nvidia.com/en-us/geforce/technologies/dlss/">DLS4 Technology Page</a> The DLS Super Resolute, Ray Reconstruction and DLAA use transformer AI Moders;<a href="https://www.nvidia.com/en-us/geforce/news/dlss-4-5-super-resolution-available-now/">DLSS 4.5 Proclamation of 14 January 2026</a> Note that its SuperResolument has been upgraded to 2nd promotion transfer mode.</p>
<p>Industrial systems are consistent with the conclusion of the academic system that complex reconstruction tasks can be replaced by hand-crafted heuristics on a continuous basis. They still depend on the existing lines of rendering. No bottom G-buffer, Ray-traded samples, time history and rendering constraints, and no effort in neuromodules.</p>
<p>DLSS is a good teaching case for this division of labour:</p>
<ul>
<li>In the real world, learning modules are often replacing heuristics, not physics;</li>
<li>(a) A landing AI rendering, usually relying on highly structured inputs;</li>
<li>The strongest system is from "signal promotion by rules + signature reconversion by learning".</li>
</ul>
<table>
<thead>
<tr>
<th>System</th>
<th>What do you offer?</th>
<th>What's in charge of?</th>
<th>Why is hybrid stronger?</th>
<th>Representative sources</th>
</tr>
</thead>
<tbody><tr>
<td>Ray tracing / path tracing</td>
<td>Visibility, reflectivity, sampling and light transmission</td>
<td>Usually not learning core communication, only talking down or rebuilding.</td>
<td>The bottom signal is credible, the upper level is more efficient.</td>
<td><a href="https://cseweb.ucsd.edu/~viscomp/classes/cse274/wi26/readings/whitted.pdf">Whitted 1980</a>, <a href="https://www.cs.rpi.edu/~cutler/classes/advancedgraphics/S13/papers/kajiya_rendering_equation_86.pdf">Kajiya 1986</a></td>
</tr>
<tr>
<td>NeRF</td>
<td>Camera Model, Ray Sampling, volume Rendering</td>
<td>Density / Radice Field</td>
<td>Learning scenes are meant, but not the rendering structure</td>
<td><a href="https://arxiv.org/abs/2003.08934">NeRF</a></td>
</tr>
<tr>
<td>3DGS</td>
<td>visibility-aware rendering、Gaussian splat compositing</td>
<td>scenes represent and optimize</td>
<td>Replace the big box with a more replicative representation.</td>
<td><a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/">3D Gaussian Splatting</a></td>
</tr>
<tr>
<td>DLSS-RR</td>
<td>Rendering pipe lines, Ray-traced Buffers, time sequence structure</td>
<td>denoising、super-resolution、frame reconstruction</td>
<td>Replace hand-tuned heuristics with read reconstructor</td>
<td><a href="https://www.nvidia.com/en-us/geforce/news/gfecnt/20238/nvidia-dlss-3-5-ray-reconstruction/">DLSS 3.5 RR</a>, <a href="https://www.nvidia.com/en-us/geforce/technologies/dlss/">DLSS 4</a></td>
</tr>
</tbody></table>
<h2>To simulator to divide the learning and the rules boundary</h2>
<p>When the system is designed, the question becomes more specific: which parts are worth learning and which must be kept in the rules?</p>
<h3>More suitable for the section that gives</h3>
<ul>
<li>Inverse problem: recovery of hidden variables, parameters, materials, geometry, state from observation.</li>
<li>Surogate / emulator: Learning from the same question is cost-effective when you have to ask many times.</li>
<li>closure / unsolved physics: sub-grids, lessons, complex material responses, difficult-to-dissemble interactions.</li>
<li>Perception driven reconstruction: noise reduction, overscoring, frame-up, missing information completion.</li>
<li>The elements of multimodular and statistical patterns are very strong: for example, complex visual appearances, real texture, noise models.</li>
</ul>
<h3>More than that, write in the rule section.</h3>
<ul>
<li>Definition of legal space: what is constant, viable, stable, impermeable and non-negative density.</li>
<li>Geometry and enlargement: grid proximity, visibility, boundary conditions, object exposure.</li>
<li>Basic generation mechanisms: how light spreads, how fluids are constant, how borders come into effect.</li>
<li>Assessment interface: What is a mistake, what is a philificial Valid, what is a visual match.</li>
<li>(c) High-risk hard-barrel: sections dealing with safety, scientific findings, engineering certification.</li>
</ul>
<h3>The type of hybrid that deserves priority</h3>
<p>If a default starting point is needed, three structures can be given priority:</p>
<ol>
<li>Solver outside, learning inside: learning to be learning machine.</li>
<li>Reader outside, Solver inside: Let the learner predict parameters, starting values, boundaries or proposal, then hand over to Solver for correction.</li>
<li>Stractturing + learner: Bottoms generate bound signals with rules, Uppers with learning.</li>
</ol>
<p>These three structures cover most of the success stories in this paper. The most important design is to draw the boundaries of the modules: it should be learningable, worthwhile and not undermine the legitimacy of the system.</p>
<h2>Concluding remarks</h2>
<p>Better simulator/mode would write the structures that have been defined - constant, geometric, boundary conditions, visibility, sampling and puffing - as beas, and allow the learning to deal with the remaining parts that are expensive, vague or difficult to model.</p>
<p>This is also my judgment about the world model. A stronger system should not simply be expected to follow the next token or frame; it also needs to evolve in a reality state. DRSS, the NeRF and the neural operator have demonstrated a feasible division of labour: rules create constraints, learning to deal with reconstruction, approximation or local errors.</p>
<p>Translating the core rules of the real world into a general bias is a practical way to build more reliable AI. For a specific system, the sequence is also clear: first, to determine which complex remaining items cannot be compromised and then to determine which complex items are worth leaving to the learning board.</p>
<h2>References</h2>
<ul>
<li>J. Turner Whitted, <a href="https://cseweb.ucsd.edu/~viscomp/classes/cse274/wi26/readings/whitted.pdf">An Improved Illumination Model for Shaded Display</a>, 1980.</li>
<li>James T. Kajiya, <a href="https://www.cs.rpi.edu/~cutler/classes/advancedgraphics/S13/papers/kajiya_rendering_equation_86.pdf">The Rendering Equation</a>, SIGGRAPH 1986.</li>
<li>Maziar Raissi et al., <a href="https://maziarraissi.github.io/PINNs/">Physics Informed Deep Learning / PINNs project page</a>.</li>
<li>George Em Karniadakis et al., <a href="https://www.nature.com/articles/s42254-021-00314-5">Physics-informed machine learning</a>, Nature Reviews Physics, 2021.</li>
<li>Lu Lu et al., <a href="https://arxiv.org/abs/1910.03193">DeepONet</a>, 2019/2020.</li>
<li>Zongyi Li et al., <a href="https://arxiv.org/abs/2010.08895">Fourier Neural Operator for Parametric Partial Differential Equations</a>, 2020.</li>
<li>Tobias Pfaff et al., <a href="https://arxiv.org/abs/2010.03409">Learning Mesh-Based Simulation with Graph Networks</a>, 2020/ICLR 2021.</li>
<li>Zongyi Li et al., <a href="https://jmlr.org/beta/papers/v24/23-0064.html">Fourier Neural Operator with Learned Deformations for PDEs on General Geometries</a>, JMLR 2023.</li>
<li>Huayu Deng et al., <a href="https://openreview.net/forum?id=DPzQ5n3mNm">Sensitivity-Constrained Fourier Neural Operators for Forward and Inverse Problems in Parametric Differential Equations</a>, ICLR 2025.</li>
<li>A. Radha et al., <a href="https://www.nature.com/articles/s44172-025-00513-3">Benchmarking scientific machine-learning approaches for flow prediction around complex geometries</a>, Communications Engineering, 2025.</li>
<li>A. Mankin et al., <a href="https://www.nature.com/articles/s43247-026-03238-z">Rewiring climate modeling with machine learning emulators</a>, Communications Earth &amp; Environment, 2026.</li>
<li>Ben Mildenhall et al., <a href="https://arxiv.org/abs/2003.08934">NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis</a>, 2020.</li>
<li>Thomas Muller et al., <a href="https://research.nvidia.com/publication/2022-07_instant-neural-graphics-primitives-multiresolution-hash-encoding">Instant Neural Graphics Primitives with a Multiresolution Hash Encoding</a>, SIGGRAPH 2022.</li>
<li>Jon Hasselgren et al., <a href="https://arxiv.org/abs/2111.12503">Extracting Triangular 3D Models, Materials, and Lighting From Images</a>, CVPR 2022.</li>
<li>NVIDIA Developer Blog, <a href="https://developer.nvidia.com/blog/differentiable-slang-example-applications/">Differentiable Slang: Example Applications</a>, 2023.</li>
<li>Bernhard Kerbl et al., <a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/">3D Gaussian Splatting for Real-Time Radiance Field Rendering</a>, SIGGRAPH 2023.</li>
<li>NVIDIA, <a href="https://www.nvidia.com/en-us/geforce/news/gfecnt/20238/nvidia-dlss-3-5-ray-reconstruction/">NVIDIA DLSS 3.5 Ray Reconstruction</a>, 2023.</li>
<li>NVIDIA, <a href="https://www.nvidia.com/en-us/geforce/technologies/dlss/">DLSS 4 Technology</a>, accessed 2026-03-15.</li>
<li>NVIDIA, <a href="https://www.nvidia.com/en-us/geforce/news/dlss-4-5-super-resolution-available-now/">NVIDIA DLSS 4.5 Super Resolution Available Now</a>, 2026-01-14.</li>
</ul>
