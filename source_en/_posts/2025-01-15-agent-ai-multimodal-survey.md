---
title: 'Multimodal Agent AI Survey: Paradigms, Learning Mechanisms, and Applications'
title_zh: 多模态 Agent AI 综述：范式、学习机制与应用
date: 2025-01-15 20:00:00 +0800
categories:
- Agent Systems
- Agent Architecture
tags:
- Multimodality
- Embodied AI
- Survey
author: Hyacehila
mathjax: true
hidden: true
excerpt: Based on a 2024 survey paper, covers Agent AI paradigm design, Agent Transformer architecture, learning mechanisms
  (RL/IL/in-context), and cross-domain applications.
description: Based on a 2024 survey paper, covers Agent AI paradigm design, Agent Transformer architecture, learning mechanisms
  (RL/IL/in-context), and cross-domain applications.
excerpt_zh: 基于 2024 年综述论文，整理 Agent AI 的范式设计、Agent Transformer 架构、学习机制（RL/IL/上下文学习）以及跨领域应用。
permalink: /blog/2025/01/15/agent-ai-multimodal-survey/
lang: en
translation_key: 2025-01-15-agent-ai-multimodal-survey
translation_status: machine
translation_source_hash: b91306dbe26c7b359d0f9ab6b8a4c4712e66609f9bef387e3a8741bb502dc01e
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>This is a synopsis article on multimodular AI smarts published in January 2024. The article discusses how a multimodular intelligence system is embedded in the physical and virtual environment, with the core orientation:<strong>Smart, multimodular, environment embedded</strong>。</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/03/21/agent-memory-panorama/">From memory formation to memory governance: the panorama of Agent Memoory</a>、<a href="/en/blog/2026/06/07/agent-runtime-teardown/">Agent Memoory and Runte technology inventory: hang memory, running time research and capacity-building in a framework</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h2>Background</h2>
<p>The Large Foundation Models (LLLM and VLM) can deal with complex tasks previously considered to be limited to human experts or field-specific algorithms. These tasks include mathematical reasoning, professional legal and medical issues, and the generation of complex plans for robotics and game AI. This is the core foundation for building Agent.</p>
<p>Embodied AI was able to implement complex mission planning and reasoning using LLM's WWW-scale knowledge and the emerging zero sample. It divides natural language commands into a series of subtasks, expressed in natural language or Python code, and then performs these subtasks using low-level controllers.</p>
<p>AI smarts not only learn at the training stage, but also from real-time interaction with users. Learning methods include user initiatives to correct their mistakes, and tacit learning by observing user interactions. Through continuous interaction, intelligent bodies can enhance their capabilities. Most LLM will not update the knowledge base and internal parameters after the last training, but will only learn through context learning.</p>
<h2>Agent AI Integration</h2>
<h3>Why AI Agents</h3>
<p>LLM based and VLM based<strong>Basic model</strong>There is still limited performance in the area of artificial intelligence. They are particularly weak in understanding, generating, editing and interacting with environments or scenes that are unknown.</p>
<p>Through an integrated intelligence body artificial intelligence framework, large base models are able to understand user input in greater depth and develop complex and self-adapted human interaction systems, i.e. large action models. The visual language model of integrated intelligence opens up new possibilities for the development of universal physical systems (such as planning, problem resolution and learning) in complex environments.</p>
<p>AI smart body systems typically have the following capabilities:</p>
<ul>
<li>Projection modelling: for example, prediction of text continuity, answer to questions, next steps for robots</li>
<li>Decision-making: In some applications, intelligent bodies can make decisions based on their reasoning, such as referral systems</li>
<li>Addressing ambiguity: handling fuzzy input based on context and training extrapolation of most likely interpretation
<strong>Note that their capabilities are very limited and limited by training data and training strategies.</strong></li>
</ul>
<h3>Downsides and Advantages</h3>
<p>The introduction of LLM into AI Agent has had a remarkable effect, leading to excellent research and high current research heat. However, generating AI has inherent flaws, which are also brought into the current smart system. At the same time, some ideas for improvement were brought in.</p>
<p><strong>Hallucination</strong> Generating artificial intelligence creates hallucinations, produces meaningless content or content that does not match the source material. The hallucinations can be divided into two categories: internal and external. The underlying illusion finger contradicts the source material, and the text generated by the underlying illusion finger contains additional information not originally included in the source material.</p>
<p>Some promising ways to reduce hallucinogenicity in language generation include the use of search-enhanced generation. However, in the context of a multimodular intelligence system, visual language models have proved to be hallucinogenic, as they rely entirely on the pre-trained knowledge base and may not be able to understand accurately the dynamics of the world state in which they are deployed.</p>
<p><strong>Biases and Inclusivity</strong> Prejudice is inevitable and stems from the data used in training, the limitations of the language model itself and the inadequacy of training strategies. While researchers have adopted many methods to control it, readers still need to be aware of possible bias in the response and interpret it with critical thinking.</p>
<p><strong>Data Privacy and Usage</strong> This is an issue that cannot be ignored in the age of the Internet. In the design of an intelligent body, it is desirable to protect the privacy of the user through anonymity.</p>
<p><strong>Imitation Learning and Generalization</strong> Broadening is achieved by empowering intelligent bodies by imitating learning rather than slowly updating enhanced learning with parameters.</p>
<p><strong>Inference Augmentation</strong> The ability of AI intelligents to reason has been enhanced through various methods, thereby enhancing the ability of the most important intelligents to make decisions. Common methods include:<strong>More extensive context data, enhanced reasoning LLM algorithms, manual feedback (HFRL), real-time feedback integration, field fine-tuning models</strong>。</p>
<h2>Agent AI Paradigm</h2>
<p>As stated in the title, Agent ' s research paradigm (Paradigm) is discussed here, with no details of actual realization, but only an overview of functions. It needs to achieve the following objectives:</p>
<ol>
<li>Use of existing pre-training models and pre-training strategies to effectively provide intelligent bodies with an effective understanding of important modulations, such as text and visual input.</li>
<li>Long-term mission planning capacity</li>
<li>Memory frame allowing learning to be coded and retrieved later</li>
<li>Use of environmental feedback to effectively train intelligent bodies to learn what to do</li>
</ol>
<p>The following is an overview of the overall framework:</p>
<p><img src="/images/agent-ai-paradigm-overview.png" alt="Agent AI Paradigm Overview"></p>
<h3>Agent and Agent Transformer</h3>
<p>We can use LLM or VLM models to guide the components of Agent. As shown in the figure above, this approach has proved to be more effective in terms of both world knowledge and reasoning planning.</p>
<p>You can also use a separate intelligent Transformer model that uses visual and language tags as input. In addition to vision and language, we have added a third generic type of input, which is presented as an intelligent marker.</p>
<p>Smart labels are used to preserve space for specific subspaces in model input and output space, specifically for intelligent body behaviour. This label is similar to the action step used in the RL, which facilitates customisation of intelligent body behaviour that is not easily described in the language, while also facilitating interaction between intelligent bodies and the environment.</p>
<p>We can use the new Agent model based on LLM and VLM and use data generated by large base models to train Agent Transformer to implement specific objectives. In doing so, the Agent Transformer model was trained to specialize and customize specific tasks and areas. This approach enables you to take advantage of the characteristics and knowledge of pre-existing basic models.</p>
<p>In order to train Agent Transformer, the target and action space of Agent need to be specified in the context of each given environment. This includes the identification of specific tasks or actions to be performed by the agent and the allocation of the only markings. Continuous monitoring of model performance and feedback collection for targeted improvements.</p>
<h3>Agent AI Learning</h3>
<h4>Strategy and Mechanism</h4>
<p><strong>Reinforcement Learning (RL)</strong> The use of enhanced learning (RL) to train interactive intelligence with intelligent behaviour has a rich history. However, the design of incentive functions and the collection of appropriate data are difficult issues. RL is more difficult to absorb in long-term decision-making because of the need to explore too much space. Using LLM to handle long process decisions and using RL policy for low-level controls is a possible solution.</p>
<p><strong>Imitation Learning (IL)</strong> Attempts to use expert data to imitate experienced Agent or expert behaviour. Behavioural cloning is the main framework for imitating learning, but not the mainstream.</p>
<p><strong>Traditional RGB</strong> The use of image input to learn the behaviour of intelligent bodies has attracted attention for many years. The inherent challenge of using RGB input is the dimension disaster. There is already a lot of research on how to change the way AI processes images and make them handle more types of data, such as the 3D model.</p>
<p><strong>In-context Learning</strong> Context learning has proved to be an effective solution to NLP tasks after the emergence of large language models such as GPT-3. Context learning can further improve the performance of smarts in the environment by incorporating environment-specific feedback when taking specific actions in the environment.</p>
<p><strong>Optimization in the Agent System</strong> Time optimizes attention to how intelligent bodies carry out their tasks over time. REACT is one way to address efficient mission planning through an interactive combination of environmental factors.</p>
<h4>Agent Modules</h4>
<p>Cf. Modules.</p>
<h4>Agentic Foundation Models</h4>
<p>The application of pre-training basic models is widely applicable in a variety of cases and offers significant advantages. The integration of these models allows for the development of customized solutions for applications, thus avoiding the need for a large number of tagging data sets for each specific mission.</p>
<p>The core of the use of Foundation Models is the use of its already trained mode (usually text and visual). Use LLM capabilities to translate its output into actions that need to be performed through the word engineering and through external tools.</p>
<h3>Agent AI Categorization</h3>
<p><strong>Generalist Agent Areas</strong> Computer-based actions and General Intelligence (GAs) apply to many missions. Recent developments in large base models and interactive AI domains have brought new functions to GAs. However, in order for GA to be truly useful to users, it must be easily interactive and broadly applicable to a wide range of contexts and patterns.</p>
<p><strong>Embodied Agents</strong> The goal of artificial intelligence is to create intelligent bodies, such as robots, capable of learning to creatively address challenging tasks that require interaction with the environment. Whether it's an entityless intelligence in a game or a robot in the real world, they belong to Embodied Ages.</p>
<p><strong>Generative Agents</strong> Recent developments in large-scale generated AI models have the potential to significantly reduce the high cost and time required for current interactive content. This will benefit both large play studios and small independent studios and individuals to create quality experiences beyond their current abilities. Generative Actors are not limited to the production of 3D content, but also include the Agentization of all processes throughout the game making field, i.e. the provision of efficient interfaces between large AI models (including GPT series models and diffuse image models) and rendering engines.</p>
<p><strong>Knowledge and Logical Inference Agents</strong> The extrapolation and application of knowledge is a key feature of human awareness. The inference of knowledge ensures that AI ' s responses and actions are consistent with known facts and logical principles, which are key mechanisms for maintaining trust and reliability in the AI system. Direct use of LLM is equivalent to direct access to the knowledge contained in the model, but the LLM illusion suggests that this is not reliable and needs to be ensured by using separate intelligent body structures.</p>
<p><strong>LLMs and VLMs Agent</strong> Use LLM and VLM directly to achieve intelligence. All the AIAgents in front are basically based on this. It is the basis for the construction of AI Agent and is listed separately because such realization is often simpler and LLM capacity is generally used directly to plan implementation.</p>
<h2>Agent AI Application Tasks</h2>
<h3>Agents for Gaming</h3>
<p><strong>NPC Behavior</strong> In the modern game system, the pre-defined foothold of the behaviour of the non-player role (NPC) is mainly prepared by the developers. These scripts cover reactions and interactions based on the behaviour of various triggers or players in the game environment. However, this scriptization often leads to predictability or duplication of NPC behaviour, which hinders the expected immersion experience in a dynamic game environment. Thus, the use of LLM to give NPC autonomy and adaptability to behaviour, to make interaction more detailed and attractive, is gradually attracting the interest of researchers.</p>
<p><strong>Human-NPC Interaction</strong> The interaction between human players and NPC is a key aspect of the game experience. The traditional interactive paradigm is mostly one-dimensional, NPC reacts to player input in a preset way. This limitation limits the more organic and rich potential for interaction, similar to people-to-people interaction in the virtual realm. The emergence of LLM and VLM technologies offers hope of changing this paradigm.</p>
<p><strong>Agent-based Analysis of Gaming</strong> A new AI system is needed to analyse player behaviour and provide appropriate support where necessary. Smart interactive systems can change the way players interact with the game system. NPC interactions with players are no longer limited to a limited set of rules designed by game developers. Its core is the use of LLM to analyse text data in the game, and VLM to analyse images and video data in the game.</p>
<p><strong>Scene Synthesis for Gaming</strong> Modern games usually have a broad open world environment. The manual design of these landscapes can be time-consuming and resource-intensive. Automated terrain generation (usually using programmable or AI-driven technologies) can generate complex, realistic landscapes, light and atmospheric effects with less manual effort.</p>
<h3>Robotics</h3>
<p>Robots are representative actors requiring effective interaction in the environment, mainly involving the following technologies.</p>
<p><strong>Visual Motor Control</strong> Visual motion control refers to the integration of visual perception and movement to effectively perform tasks in robotic systems. This combination is essential because it enables robots to interpret visual data from the environment and adjust their movements accordingly to interact accurately with the environment.</p>
<p><strong>Language Conditioned Manipulation</strong> Language manipulation refers to the ability of robotic systems to interpret and perform their tasks in accordance with language directives. This aspect is particularly important for creating visual and user-friendly robotic interfaces. Through natural language commands, users can assign targets and tasks to robots like people-to-people communication, thus lowering the threshold for operating robotic systems.</p>
<p><strong>Skill Optimization</strong> Recent studies have highlighted the effectiveness of LLM in robotic mission planning. However, the optimal implementation of mandates, particularly those involving physical interaction (e.g. capture), requires a deeper environmental understanding that goes beyond a simple interpretation of human directives. The capture of these minor indirect clues in the scene and their effective transformation into robotic skills remains a major challenge.</p>
<h3>Healthcare</h3>
<p>In the area of health care, LLM and VLM can serve as diagnostic intelligence, patient care assistants and even treatment aids, but they also pose unique challenges and responsibilities. AI has great potential to improve patient care and save lives, but there is also a dangerous risk of endangering millions of people around the globe through misuse or hasty deployment.</p>
<p><strong>Diagnostic Agents</strong> The use of LLM as a medical chat robot for patient diagnosis is of great interest. This is due to the high demand for medical specialists and the potential of LLM to assist in the diagnosis and diagnosis.</p>
<p><strong>Knowledge Retrieval Agents</strong> In the medical field, model hallucinations are particularly dangerous and serious errors can even lead to serious injury or death. Combining the use of Agent with the medical researcher Agent has the potential to significantly reduce hallucinations and at the same time improve the quality and accuracy of the response of diagnostic dialogue agents.</p>
<p><strong>Telemedicine and Remote Monitoring</strong> Agent-based AI also has great potential in the field of telemedicine and telemonitoring. It can improve access to health care, improve communication between health-care providers and patients, and increase the efficiency and cost of frequent patient interaction.</p>
<p><strong>Image understanding and Video understanding</strong> Understanding common medical examination images such as X-rays, CTs, MRIs and ultrasound video is important for creating a true medical treatment, Agent.</p>
<h3>Multimodal Agents</h3>
<p>Interactive multimodular intelligence comprises four main pillars: interaction, voice, vision and language. The integration of visual and linguistic understanding is essential for the development of complex multimodular AI agents. This includes tasks such as image descriptions, visual questions and answers, video language generation and video understanding.</p>
<p><strong>Image-Language Understanding and Generation</strong> Image-linguistic understanding is a task involving the linguistic interpretation of visual content in given images and the generation of relevant language descriptions. Multimodular intelligence bodies should be able to identify objects in images, understand their spatial relationship, generate accurate descriptive sentences on scenes and use reasoning to address knowledge-intensive visual reasoning. This requires not only the ability to identify objects, but also a deep understanding of space relations, visual semantics and the ability to map these visual elements into language structures and integrate them with world knowledge.</p>
<p><strong>Video-language generation</strong> The task of video language generation (or video storytelling) is to generate a consistent set of sentences for the video stream.</p>
<p><strong>Video Understanding</strong> Video understanding extends the image understanding to dynamic visual content. This involves the interpretation and reasoning of the frame sequence in the video, usually combined with accompanying audio or text information.</p>
<p><strong>Knowledge-Intensive Agent</strong> Knowledge-based visual questions and answers and visual-linguistic search missions are challenging in multimodular machine learning and require external knowledge beyond image content. KAT is the representative.</p>
<h3>Agent for NLP</h3>
<p>Identifying mission directives and taking action has been a fundamental challenge for decades in the field of interactive artificial intelligence and natural language processing. We have identified three specific directions (and others) for improving language-based intelligence:</p>
<p><strong>Tool use and querying from knowledge bases</strong> This direction emphasizes the importance of integrating external knowledge bases, web searches or other useful tools into the reasoning of artificial intelligence. Technologies such as Function Calling, RAG, MCP, etc.</p>
<p><strong>Improved agent reasoning and planning</strong> Increased intelligence reasoning and planning capabilities are essential for effective human collaboration. This involves developing models that understand complex instructions, extrapolate user intentions and predict potential future scenarios. React and all kinds of code intelligences.</p>
<p><strong>Incorporating system and human feedback</strong> AI intelligence bodies can usually operate in two main environments: those that provide clear signals of action effects (systemic feedback) and those that work with humans who can provide oral comments (human feedback). This direction emphasizes the need for self-adaptation learning mechanisms to enable agents to refine their strategies and correct mistakes. This is an important direction for the moment.</p>
<p>In addition to this, enhancing the command and follow-up capabilities of intelligent bodies is an area of great concern. At present, it is more at the centre of fine-tuning of research, rather than intelligent research.</p>
