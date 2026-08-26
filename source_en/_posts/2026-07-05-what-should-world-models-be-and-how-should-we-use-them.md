---
title: What Should World Models Be, and How Should We Use Them?
title_zh: 世界模型应该是什么样子，又该如何被利用？
date: 2026-07-05 00:00:00 +0800
categories:
- Foundation Models
- Model Mechanics
tags:
- World Models
- Multimodality
- Model Mechanics
author: Hyacehila
mathjax: true
excerpt: World models matter less as beautiful video generators than as internal simulators for agents. This post compares
  the main routes, explains why JEPA is appealing, and revisits Critique of World Model and PAN.
description: World models matter less as beautiful video generators than as internal simulators for agents. This post compares
  the main routes, explains why JEPA is appealing, and revisits Critique of World Model and PAN.
excerpt_zh: 世界模型最有价值的部分，是让智能体在行动前模拟未来。本文从几条路线的差异讲到 JEPA，再回到 Critique of World Model 与 PAN。
permalink: /blog/2026/07/05/what-should-world-models-be-and-how-should-we-use-them/
lang: en
translation_key: 2026-07-05-what-should-world-models-be-and-how-should-we-use-them
translation_status: machine
translation_source_hash: 45ea743478ad8448ae06036b870977cb16bdef812b903dedc0f7e240cc489224
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Why do you have to do a world model?</h2>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/01/20/from-llm-to-vlm-visual-understanding/">From LLM to VLM, how language models achieve visual understanding</a>、<a href="/en/blog/2026/01/26/neural-scaling-laws/">Neal Scaling Laws: From Kaplan to Chinchilla</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>LLM success, to a large extent, comes from a simple and strong goal: predict the next word. But when intellectual decency is true, it is not enough to predict the next word. It needs to know what happens if it goes left, if it tilts the glass, if it comforts a person who is crying, if the other person stops crying, continues to collapse, or if it is understood as disturbing.</p>
<p>The world model is not a larger visual speech machine, nor is it just a stronger video generator. It's more like a test site in the brain of a smart body: a given state of current and a possible action, modeling the next state. In a more formal way, it's from &#36;s&#36; and &#36;a&#36; Let's go, estimate.&#39; \sim p(s&#39; |, a) &#36;. The problem has changed from identifying the world to predicting it before action is taken.</p>
<p><a href="https://arxiv.org/abs/2507.05169">Critique of World Model</a> This is the core reference. The author focuses on the potential response and action: the world model simulates all actionable possibilities, allowing intelligents to choose the next step. It first serves decision-making and is followed by visualism.</p>
<p>It's critical to intelligence. Agent without a world model tends to be tested in the real environment, or rely on a language model to give sound advice (a language model brings a price of course, which is at the heart of the current language model intelligence). With the world model, angent could run several branches inside: Will this step hit the barrier, will the subsequent gains be higher and there are no safer alternatives. Searches in AlphaGo, trajectories in autopilots, and action planning in robots can be seen as a partial realization of this approach under different constraints.</p>
<p>So when I was talking about the world model, I was more concerned about the future that was being simulated, and whether it was useful for action. The precision of the picture is certainly valuable, but it should not be overridden. The water cups, balls, vehicles, web pages, teammates' emotions, long-term strategies, all of which are completely different, are all asking the same thing: whether models can connect the current state, candidate actions and consequences.</p>
<h2>Landscape: Several roads outside JEPA</h2>
<p>The world model has suddenly warmed up in the past few years, and a problem has arisen: a lot of things are called world model, but they're not the same thing they're trying to solve. Considering that I prefer Lecun's Jepa, let's take a quick look at a few routes that are not Jepa-centric.</p>
<h3>Game and Interactive World Model</h3>
<p>The game route is represented by Google DeepMind <a href="https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/">Genie 2</a>Microsoft's. <a href="https://www.nature.com/articles/s41586-025-08600-3">WHAM / Muse</a>, and Decart and Etched <a href="https://oasis-model.github.io/">Oasis</a>I'm sorry. The common denominator of these systems is to place the world model in an interactive environment: after input action, the model continues to generate the next image. Geneie 2 can generate a 3D funable environment from a single tip, while Oasis shows a real-time generating environment like Minecraft.</p>
<p>The benefits of this route are clear. The game has a natural state, action, feedback and can easily be seen whether the world is evolving with action. The problem is that the environment for games is often heavily restricted by rules, perspectives and action interfaces. They are well suited to train and assess certain types of smart body, but they do not directly indicate that models understand the open world. A model that allows keyboard input in a Minecraft-style environment is a long way from handling kitchens, streets, offices and social scenes. This type of dedicated model is valuable and, although the game is a virtual environment, it may be sufficient to train many special-purpose smart bodies.</p>
<h3>3D scene and space intelligence</h3>
<p>The second is 3D with space intelligence. This article is based on World Labs. <a href="https://www.worldlabs.ai/blog/marble-world-model">Marble</a> For example, it generates a 3D world from text, images or videos that can be viewed and edited. The intuition behind this direction is strong, and the real world is above all spatial. The object has location, scale, shielding, depth and accessibility; many reasonings become drifting if the model does not have stable spatial representation.</p>
<p>I agree with the 3D that is important, especially robotics, ARs, games, simulations. But if the understanding of the world is completely equivalent to the reconstruction of a three-dimensional space, the problem becomes narrower. Space structures are just one layer of the world. The consequences of the action also relate to physics, intent, mission objectives, social relations and time scales. The more realistic judgement is that 3D world generation becomes an important component of the world model, but it is unlikely that it will be able to take on the full body of intelligence reasoning alone. Of course, the 3D scenario is very valuable, and when we explore a space intelligence later, a stable 3D environment that can be the basis for everything.</p>
<h3>Physical AI and Autopilot World Model</h3>
<p>The third route is directed towards the Physical AI, especially autopilot and robotics. NVIDIA's <a href="https://arxiv.org/abs/2501.03575">Cosmos</a> Position the world foundation model as a base platform for robotics and auto-driving; for Wayve <a href="https://arxiv.org/abs/2503.20523">GAIA-2</a> It is more clearly oriented towards auto-driving, generating multi-perspective, controlable driving videos and controlling the scenes with road syntax, vehicle dynamics, weather and angent configuration.</p>
<p>Such models can be seen as reality-world-style games that interact with the world, but the difference is that they deal with physical patterns. The mission boundary is clearer: how the car moves, where the pedestrians are, how the weather affects the sensors, and whether a rare scene is worth joining the training. Their problems also stem from such clear borders. Models often bind specific sensors, tasks and control interfaces in depth, and many structures are reworked when extended to family robots, web sites, or strategic planning. They are good world models, but not necessarily the final form of a universal world model.</p>
<h3>Generic video generation model</h3>
<p>Article IV is the most special model route for the world: video generation. OpenAI in Sora Technical Report <a href="https://openai.com/index/video-generation-models-as-world-simulators/">Video generation models as world simulators</a> It is clearly suggested that the extended video generation model may be a route to the universal simulation of the physical world. Google DeepMind <a href="https://deepmind.google/models/veo/">Veo</a> The series is also increasing the control, quality and consistency of video generation.</p>
<p>The advantages of video generation are straightforward: output is visible and progress is easily felt. Its weaknesses can be equally evident from the perspective of world models. Most video generation models remain prompt-to-video, which generate a fixed trajectory that does not allow angent to insert action in the middle, compare multiple consequences, and do not necessarily have a clear state and action expression. They can learn a part of the world's law, but without interactive, branchable and evaluable structures, they are more like the world's image model, one step closer to the action world model.</p>
<p>These routes are valuable. The game route emphasizes interaction, the 3D route emphasizes space, the Physical AI route emphasizes controlled physical scenes, and the video-generation route emphasizes visual dynamics. Under the word world model, there is a set of options for how to simulate the future.</p>
<p>There are also digital people in the video generation sector, but it's not the same idea. Universal video generation emphasizes the importance of universal and inclusive capabilities to cover different scenarios, such as films, short plays and news stories, and therefore often relies on large-scale end-to-end training and next frame generation. Digital people are more like a special scene, focusing on the human expression, the demography and the synchronization of images and voices. There are also Vido S1 products that generate video links to Interaction Model, emphasizing that the model is based on immediate feedback from people. The technical realization behind it is not clear, either as a separate International Model and digital layer, or as an independent digital person for end-to-end training. For this paper, what really matters is not the appearance of digitals, but the underlying Interaction Model: how the system changes the face, tone and next move in real time, based on the human response. Real-time voice interaction, TTS and ASR are also important components of this branch, but this is not being developed.</p>
<p>JEPA cuts the problem to the other level: understanding the world may not have to start with creating the whole world.</p>
<h2>JEPA: I see better abstract prediction routes</h2>
<p>- LeCun. -Yeah. <a href="https://openreview.net/pdf?id=BZ5a1r-kVsf">A Path Towards Autonomous Machine Intelligence</a> The central judgement is that smart systems require learning, forecasting and planning at multiple abstract levels. And then he proposed JEPA, Joint Employment Building Capacity. It is very simple to start from the point of departure: to bypass the raw data itself and to predict the signs behind the original data.</p>
<p>This is very close to human instincts. We predict that when the ball falls off the ground, the inner brain does not render every frame of the ball by pixel, nor does it restore the texture of the background wall. We have a few useful variables: balls, hands, gravity, supporting relationships, time. JEPA wants models to learn such predictions in abstract space, with each pixel, each paragraph of the text, and every unmanageable tiny end of the branch, to be placed in a secondary position.</p>
<p>Typical JEPA structure can be simplified into three parts: context encoder encodes the visible part into context, target encoder encoder encodes the target area into target indicator, predictor predicts target representation according to context. Compared signs during training, pixels and token remain behind. One advantage of this is that models can ignore low-level noise and focus on semantic structures, object relationships and dynamic changes.</p>
<p>I-JEPA is the starting point of this route on the image.<a href="https://arxiv.org/abs/2301.08243">I-JEPA</a> The display of the target block from the context block of a chart is not dependent on manually designed data enhancement, nor is the model required to complete the pixels. It wants to learn high-level semantics more than a re-establishment-like approach; missing texture is just a jamming item.</p>
<p>V-JEPA pushes this idea to the video.<a href="https://arxiv.org/abs/2404.08471">V-JEPA</a> Training in video displays only for the use of text, negative samples, reconstruction or pre-trained image encoders. Present. <a href="https://arxiv.org/abs/2506.09985">V-JEPA 2</a>The route is another step towards physical action: first, accompaniment-free pre-training with large-scale video, then action-conditions with small robotic trajectory data, and then video representation into prediction and planning.</p>
<p>There is a similar expansion in the audio direction.<a href="https://arxiv.org/pdf/2311.15830">A-JEPA</a> Move I-JEPA ideas onto the audio spectrum, predicting potential manifestations of the area that is covered, avoiding the reconstruction of original waveforms or spectrum details.</p>
<p>Visual language direction. <a href="https://arxiv.org/abs/2512.10942">VL-JEPA</a> The new version of the VLM is closer to the traditional VLM embedded form. Traditional VLM mostly follows the interface of the Visual Encoder + Aligning + LLM decoder: LaVA uses MLP projector to map visual patch features to spaces where language models can receive them, and InstractBLIP uses Q-Former to extract visual information from images and questions, and then toggle answers to the language model. This route works, and VQA, Caption, multi-modular dialogue all runs on it. But it also pushes many light tasks towards full language generation. It's just to judge if there's any anomaly in the picture, to retrieve a video, to answer a short classification question, and to start a big decoder with text.</p>
<p>VL-JEPA is mainly adjusting this interface. It can be broken down into X-Encoder, Y-Encoder, Predator and Y-Decode four pieces: X-Encoder is responsible for the input of images or videos, Y-Encoder encodes the target text into a continuous semantic inlay, Predictor predicts the target's embedding based on visual indications and queries, and Y-Decoder only appears when it needs a readable answer. The model approaches the answer in semantic space before deciding whether to turn the answer into a token. This sequence is noteworthy. It moves the most expensive, language-resisting generation interface in VLM back to the embedded layer, where understanding, matching, monitoring and light-weight questions and answers are first left.</p>
<p>This also has the effect of embedding the traditional text: synonyms are naturally close. The light is out and the room is darkened at a distance from the token layer, and the semantic inlay can point to the same event. This is more useful for a world model than repeating a fixed text. The intelligent concerned the changing state of the scene, the heightened risk and the next step in avoiding a particular region; the final phrase, often the issue of reprocessing, was the final one.</p>
<p>The paper also reported that selective decodering can achieve a maximum of approximately 2.85 x decodering, while maintaining similar performance. This result is specific to the tasks and settings in the paper, but it points to one direction: Visual language models do not always have to turn understanding into full text. VL-JEPA still faces the question of detail authenticity, location of appearances and complex reasoning, but it re-aligns the question of when language is needed on the table.</p>
<p>That's why I'm biased against Jepa. It has moved the understanding of the world away from the path of reconstruction. Video generation and 3D generation are certainly useful, but they can also easily spend a lot of computing on details that are not helpful for decision-making. JEPA asks: If intelligence really needs actionable information about the next step, why not just learn about it? The best way to achieve this is to try and try not to be a more appropriate option than to recaptulate the human information architecture.</p>
<p>The difficulties of JPA are also clear: whether the display space is stable, whether the mission is really preserved, how the action conditions are added, how long the error is handled, and how abstract representations are captured by real action. None of these issues has been resolved. But as a model of the world, I think it has a clue to bet on: don't let the world model be held hostage by the appearance of humans.</p>
<h3>Course comparison</h3>
<p>By putting several routes together, one can see real differences: at what level the predictions occur.</p>
<table>
<thead>
<tr>
<th align="left">Route</th>
<th align="left">Representation</th>
<th align="left">Forecast object</th>
<th align="left">Advantages</th>
<th align="left">Main issues</th>
</tr>
</thead>
<tbody><tr>
<td align="left">Game/Interactive World Model</td>
<td align="left">Genie 2, Muse, Oasis</td>
<td align="left">Interactive Game Status and Images</td>
<td align="left">Action input. Easy to evaluate interactive.</td>
<td align="left">The scene and the motion space are narrow.</td>
</tr>
<tr>
<td align="left">3D/Space intelligence</td>
<td align="left">World Labs Marble</td>
<td align="left">Space structure and world viewability</td>
<td align="left">Space is so consistent that it suits simulation and editing</td>
<td align="left">It's not the same as a complete operation theory.</td>
</tr>
<tr>
<td align="left">Physical AI</td>
<td align="left">Cosmos, GAIA-2</td>
<td align="left">Physical/drive/robots scene</td>
<td align="left">The mission is clear. The project is very valuable.</td>
<td align="left">Area-based binding</td>
</tr>
<tr>
<td align="left">Generic video generation</td>
<td align="left">Sora, Veo</td>
<td align="left">Video frame or submersible variable</td>
<td align="left">High-quality vision and wide coverage</td>
<td align="left">Mostly fixed tracks, lack of actionable branches</td>
</tr>
<tr>
<td align="left">JEPA</td>
<td align="left">I-JEPA, V-JEPA, V-JEPA 2, VL-JEPA</td>
<td align="left">Abstract representation</td>
<td align="left">Avoid rebuilding with no detail.</td>
<td align="left">The surface and long-term control are still difficult.</td>
</tr>
</tbody></table>
<h3>What's the JEPA for?</h3>
<p>The Jepa application should not be written as a list of all-powerful things, but only predict that abstract representation actually means a great application limitation. It should be used at this time in several categories that do fit its strengths.</p>
<p>The first is real-time perception and interaction. Smart glasses, surveillance, vehicle-mounted systems, robots do not often need to produce text statements for each frame, and they need to know if semantics have changed. The selective decode of VL-JEPA is appropriate for this idea: monitor and predict in embedded space, and then output language when reporting, explaining or interacting is required.</p>
<p>The second category is end-side and edge deployment. Continuous embedding predictions are usually lighter than full token generation, and if the task is only classification, retrieval, abnormality detection or light VQA, it is not necessary to start a large-language decoder each time. The benefits here are mainly faster, more economical and more stable, with chatting coming behind.</p>
<p>The third category is smart. V-JEPA 2 This type of video representation model, if combined with a small amount of action data, has the opportunity to act as a compression of world dynamics in robotic planning. It does not need to generate a good video in its brain, but it is worth it to determine which action is more likely to get the cup picked up and the object pushed to the target position.</p>
<p>The fourth category is content understanding and retrieval. The embedded space is naturally suitable for similarity calculations and can more easily crush synonyms in the nearest area. VL-JEPA's symmetrical projection of text makes this part of the training target. This is more important than a beautiful result for searching, auditing, weighting and open terminology classifications.</p>
<h2>Critique of World Model: Arguments and PAN</h2>
<p><a href="https://arxiv.org/abs/2507.05169">Critique of World Model</a> The Jepa faction isn't exactly buying. It acknowledged that many of the current world models were over-discussed around video generation, but also considered that there was a set of assumptions in the LeCun/JEPA route that deserved scrutiny. I read it, more like: If the world model ultimately serves the universal intelligence, is it too narrow to do continuous representational predictions of only fixed dimensions?</p>
<p>The first difference is the data. LeCun often emphasizes sensory data, especially video and action experience, as the real world has much more information than the text. The author of Critique responded that the amount of data does not equal the density of information. There is a lot of redundant pixels in the video, and language is the result of long human experience of compression, which contains causality, social rules, counterfact, plans, value judgements that are not easily visible from the eyes. If the world model is focused on the sensory stream, it is easy to learn what the world looks like, but it leaves out why people act in the world. The introduction of language signs in VL-JEPA is also a measure of this, and we cannot abandon language.</p>
<p>The second difference is the manifestation. JEPA tends to embed continuously, as continuous space is suitable for gradient optimization and can also carry fine sensory differences. Critique's author emphasizes the value of discrete token: language, symbols, concepts, combustible memory structures, all of which are much needed for long-term reasoning. It is difficult to give an answer to this question now. Continuous manifestations are suitable for low-level perception, discrete manifestations are suitable for stabilization concepts and long-range reasoning. The real promising world model probably needs a mixed representation.</p>
<p>The third difference is structure. JEPA antipathy directly generates raw observations because pixel reconstruction introduces a large amount of unpredictable detail. Critique authors say that the complete elimination of the generation decoder poses a problem of the site: the model predicts very closely in the space, not as much as it predicts what makes sense in real observation space. In other words, the next representation projection cannot be a complete substitute for the next observational constraint. This does not overturn JEPA, but reminds us that the Latin space cannot speak for itself, and that it must be constantly calibrated by reality. Retaining the calibration capability that generates observations is also valuable for training in embedding signs.</p>
<p>The fourth difference is the goal of training. JEPA uses the space-space object to try to circumvent the complexity of the raw data space. Critique author is concerned that the risk of collapse and unrecognized risk requires a lot of additional positives to maintain the quality of the representation. They favour the generation loss that anchors the observation data, as it at least makes the internal state of the model responsible to the outside world. I'd like to retain a little doubt here: the generation loss is really more relevant, but it may also pull back the details back. Perhaps the better way to set different intensity constraints on the abstract.</p>
<p>The fifth difference is the way in which it is used. The common idea in the LeCun system is to put the world model in the MPC, to let angent roll several steps in the reasoning and choose the least costly action. Critique authors believe that the MPC is suitable for short-sighted control, but universal intelligence also needs to learn from simulation experience, use the world model as a training ground, and internalize strategies through RL or other learning signals. And perhaps, eventually, there is still a need for mixed training; but for me, the embedded signs generated by Jepa are already useful enough at this stage.</p>
<p>This group of critics finally led to PAN.<a href="https://arxiv.org/abs/2511.09057">PAN: A World Model for General, Interactable, and Long-Horizon World Simulation</a> Specific architectures are presented: controlling world evolution with language actions, maintaining abstract and long-range knowledge with LLM-style slots, and re-establishing the state as an observable future piece with video Diffusion decoder.</p>
<p>The features of PAN can be summarized in several articles. It takes multimodular experience, using continuous and discrete manifestations; it places self-regression generation in a stratification structure, with high-level dynamics and low-level visual details separately processed; it uses observational data to anchor internal state; and it uses world models as an agrent learning and error simulation. PAN and Critique are compatible. The author criticized JEPA for its excessive reliance on perception, continuous symptoms, latent loss and short-sighted MPC, and then offered an almost reverse solution: Texts are important, token is important, and the generation of places is important, and world models should be involved in training intelligent bodies. It is closer to completing the Jepa, focusing on the few dimensions that are missing.</p>
<p>I'm still more inclined to the basic instinct of JPA: smarts need to predict useful signs, recapitulating the world's appearances behind. But Critique and PAN are timely. If the abstract expression of JEPA is unplaceable, unmoveable, unaccumulated over a long distance, it will stop at a beautiful loss. The world model finally goes back to angent: can it make the system better choose actions, less truly wrongly try and manage more firmly the scenes that it has never seen.</p>
<h2>Concluding remarks</h2>
<p>The most valuable part of the world model is giving intelligent people an imagination to try before they can move. Video, 3D, games, Physical AI are filling this puzzle, and JEPA reminds us that sometimes the most predictable future lies in abstract signs, and pixels are just one layer of it.</p>
<p>I think it's more important that Jepa because it liberated the world from the image of reconstruction. But Critique and PAN are also timely: the world model cannot communicate in the space alone, but is also subject to real observation calibration, is called by the action interface, and finally helps angent learn better strategies.</p>
<p>A mature world model probably does not belong to just one route. It will have Jepa abstract predictions and a PAN-style location; it will use compression experience in text, as well as physical experience in video and interactive interactions. Finally, we have to go back to the problems that intelligent bodies encounter every day: what happens to the world if I do that?</p>
<h2>References</h2>
<ul>
<li>Eric Xing, Mingkai Deng, Jinyu Hou, Zhiting Hu, <a href="https://arxiv.org/abs/2507.05169">Critique of World Model</a>, arXiv:2507.05169, 2025.</li>
<li>Yann LeCun, <a href="https://openreview.net/pdf?id=BZ5a1r-kVsf">A Path Towards Autonomous Machine Intelligence</a>, 2022.</li>
<li>David Ha, Jürgen Schmidhuber, <a href="https://arxiv.org/abs/1803.10122">World Models</a>, arXiv:1803.10122, 2018.</li>
<li>Google DeepMind, <a href="https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/">Genie 2: A large-scale foundation world model</a>, 2024.</li>
<li>Anssi Kanervisto et al., <a href="https://www.nature.com/articles/s41586-025-08600-3">World and Human Action Models towards gameplay ideation</a>, Nature, 2025.</li>
<li>Decart and Etched, <a href="https://oasis-model.github.io/">Oasis: A Universe in a Transformer</a>, 2024.</li>
<li>World Labs, <a href="https://www.worldlabs.ai/blog/marble-world-model">Marble: A Multimodal World Model</a>, 2025.</li>
<li>NVIDIA, <a href="https://arxiv.org/abs/2501.03575">Cosmos World Foundation Model Platform for Physical AI</a>, arXiv:2501.03575, 2025.</li>
<li>Wayve, <a href="https://arxiv.org/abs/2503.20523">GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving</a>, arXiv:2503.20523, 2025.</li>
<li>OpenAI, <a href="https://openai.com/index/video-generation-models-as-world-simulators/">Video generation models as world simulators</a>, 2024.</li>
<li>Google DeepMind, <a href="https://deepmind.google/models/veo/">Veo</a>.</li>
<li>Mahmoud Assran et al., <a href="https://arxiv.org/abs/2301.08243">Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture</a>, arXiv:2301.08243, 2023.</li>
<li>Adrien Bardes et al., <a href="https://arxiv.org/abs/2404.08471">Revisiting Feature Prediction for Learning Visual Representations from Video</a>, arXiv:2404.08471, 2024.</li>
<li>Mido Assran et al., <a href="https://arxiv.org/abs/2506.09985">V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning</a>, arXiv:2506.09985, 2025.</li>
<li>Zhengcong Fei, Mingyuan Fan, Junshi Huang, <a href="https://arxiv.org/abs/2311.15830">A-JEPA: Joint-Embedding Predictive Architecture Can Listen</a>, arXiv:2311.15830, 2023.</li>
<li>Delong Chen et al., <a href="https://arxiv.org/abs/2512.10942">VL-JEPA: Joint Embedding Predictive Architecture for Vision-language</a>, arXiv:2512.10942, 2025.</li>
<li>PAN Team, <a href="https://arxiv.org/abs/2511.09057">PAN: A World Model for General, Interactable, and Long-Horizon World Simulation</a>, arXiv:2511.09057, 2025.</li>
</ul>
