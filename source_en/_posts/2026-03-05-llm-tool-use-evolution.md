---
title: 'The Evolution of LLM Tool Use: From Toolformer to ToolLLM'
title_zh: LLM 工具使用的技术演进：从 Toolformer 到 ToolLLM
date: 2026-03-05 13:20:00 +0800
categories:
- Agent Systems
- Agent Infrastructure
tags:
- Tool Use
- Retrieval
- Survey
author: Hyacehila
hidden: true
excerpt: A survey of LLM tool-use research, from Toolformer's self-supervision and Gorilla's fine-tuning plus retrieval to
  Tulip Agent and ToolLLM.
description: A survey of LLM tool-use research, from Toolformer's self-supervision and Gorilla's fine-tuning plus retrieval
  to Tulip Agent and ToolLLM.
excerpt_zh: 梳理 LLM Tool Use 领域的研究脉络：Toolformer 的自监督学习、Gorilla 的微调+检索、Tulip Agent 的递归分解、ToolLLM 的大规模框架，以及这条路线如何转向协议、运行时和工程实践。
permalink: /blog/2026/03/05/llm-tool-use-evolution/
lang: en
translation_key: 2026-03-05-llm-tool-use-evolution
translation_status: machine
translation_source_hash: 2808ad3ccc29a69e266e223e4cf0d9c573ece5a6356d32722521b3de868c3d97
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Why do you care about the tools?</h2>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/02/16/mcp-model-context-protocol/">MCP (Model Context Protocol)</a>、<a href="/en/blog/2026/03/01/structured-output-and-constrained-decoding/">Make it work, big model structured output and restricted decoding techniques</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>The Big Language model is very strong in a number of classic NLP fields, but still unstable in the context of arithmetical questions and factual answers -- - The model is not able to update internal parameters in a timely manner and there are hallucinating problems. LLM is allowed to use external tools that allow it to access real-time, accurate knowledge banks and perform computing tasks.</p>
<p>About Tool Use,<strong>Perhaps the MCP protocol is of greatest interest to the engineering community</strong>I'm sorry. MCP provides a standardized tool definition and description communication protocol that allows models to identify and access tools in a uniform manner. For a detailed description of MCP, reference can be made to previous articles of this blog. The paper on Tool Use is more in order to understand the development context and the evolution of the technical approach.</p>
<h2>Toolformer: Self-supervised learning tool call</h2>
<blockquote>
<p>Schick et al., <em>Toolformer: Language Models Can Teach Themselves to Use Tools</em>, NeurIPS 2023.</p>
</blockquote>
<p>Toolformer considers fine-tuning the original model to enhance its ability on Tool Use while learning the tools given. Its micro-modification dataset is generated using self-monitoring and is used once in a single decode.</p>
<ul>
<li><strong>On the fine-tuning of self-supervised data sets</strong>: Use context learning to make models volunteer for tools. When the model uses the tool, the value of the tool is measured by comparing the tool/decode loss without the tool, and the value results are then organized into micro-alteration data Set</li>
<li><strong>About tools calling in decoding</strong>: Execute a general decode, request to call API when the decode has been requested to use the token symbol, return the result to the decoding sequence and continue the decoder process</li>
</ul>
<p><strong>Toolformer is an early attempt to introduce external tools, but reliance on decoding modifications is limited because of the lack of the ability to combine the reasoning of the current model. Also, because of the one-way decoded strategy, Toolformer cannot use tools in a chained manner to meet the needs of multi-smart body tools. The most worthwhile part of this is probably a self-monitoring learning strategy.</strong></p>
<h2>Gorilla: The combination of fine-tuning and retrieval</h2>
<blockquote>
<p>Patil et al., <em>Gorilla: Large Language Model Connected with Massive APIs</em>, 2023.</p>
</blockquote>
<p>Most of the previous work on integrating tools into LLM takes into account a small, well-documented set of APIs that can easily be injected into the hint. However, support for a super-large, overlapping API warehouse requires new technologies to address it.</p>
<p>This paper considers ways to circumvent the introduction of the phrase by using Self-Instract Fine-Tuning (a fine-tuned parameter for the original model) and Retrieval (a search to obtain contextal tips, similar to MCP, but not all of which are injected into Prompt) to enhance the correct callability of the model to the large API library.</p>
<p>Combining directly Fine-Tune and Retrieva is the main improvement of this paper, namely, <strong>Retrieval-Aware Training</strong>I'm sorry. Experimental certificate:</p>
<ul>
<li>When there is a good searcher, Retriev-Aware Training is better than a simple fine-tuning.</li>
<li>Retrieval-Aware Training adapts to fast changes in API documents</li>
<li>When the call constraints (which require balancing multiple needs to determine which API to choose) are involved, the performance of all models has decreased significantly</li>
</ul>
<p><strong>Consider using fine-tuning and simple retrieval to achieve a tool call. Since multi-wheel reasoning and call is not involved, the relevant API that is retrieved from the command is referred to the model as its context.</strong></p>
<h2>Tulip Agent: Remittance task decompose and semantic search</h2>
<blockquote>
<p>Ruis et al., <em>Tulip Agent: Enabling LLM-Based Agents to Solve Tasks Using Large Tool Libraries</em>, 2024.</p>
</blockquote>
<p>Tulip Agent does not encode all available tools into the system alert (which will take advantage of the context window of the model) nor does it embed the entire tip to retrieve the tool. It will then translate tasks into multiple subtasks, and then allow each subtask to perform a semantic-level vector database search, match the appropriate tool and allow dynamic management tools.</p>
<p>Compared to the technology before:</p>
<ol>
<li><strong>Dropping all tool descriptions into LLM as a hint</strong>To avoid the problem of the long context</li>
<li>Discard the one-time embedded tool to find the hint instead of the first yes<strong>The mission plans, embeds, retrieves and continues to reason.</strong></li>
<li>Use<strong>Vector Database, Embeding and RAG-like Technologies</strong>Conducting tool retrieval</li>
<li>Dynamic management of allowed tools</li>
</ol>
<p><strong>In this technique, the searcher is activated to retrieve the API after each of the steps planned for the reasoning, and then the form below provides the model with reference to which API should be called.</strong></p>
<h2>ToolLLM: A framework for the use of common tools</h2>
<blockquote>
<p>Qin et al., <em>ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs</em>, ICLR 2024.</p>
</blockquote>
<p>ToolLLM introduced a framework for the use of generic tools covering data construction, model training and evaluation, and the engineering design was more complete. As an open source project, it received considerable attention in the Tool Use direction and was one of the representative projects in the direction of the LLM Acts tool.</p>
<p>The closed-source model already has a strong tool mobilization capacity, but existing research in open-source communities is inadequate:</p>
<ol>
<li>API limited in number, possibly too small in coverage and insufficient in diversity</li>
<li>Limit to single-tool calls, often assuming that the user gives the desired API set manually</li>
<li>Insufficient planning and reasoning, including CTT reasoning or REACH reasoning and action</li>
</ol>
<h3>Core Component</h3>
<p><strong>API Collection</strong>: 16,464 RET APIs were collected from the RapidAPI platform, covering 49 different categories and containing detailed documents for LLM learning.</p>
<p><strong>Command Generation</strong>: Sample from the whole API collection, prompting ChatGPT to generate diversified commands, involving single and multi-tool scenarios (Self-Instract).</p>
<p><strong>Path to Solutions</strong>: Each solution path may contain multi-wheel model reasoning and real-time API calls. To this end, the Decision Tree (DFSDT), based on the Depth Priority Search, was developed to enhance the planning and reasoning capabilities of LLM.</p>
<p><strong>Evaluation (ToolEval)</strong>: The AutoEval was developed to assess the use of LLM tools.</p>
<p><strong>ToolLAMA</strong>: command-generated model by fine-tuning LLAMA on ToolBech.</p>
<h3>Key findings</h3>
<ul>
<li>ToolLaMA demonstrates the ability to process single tools and multi-tool commands</li>
<li>ToolLAMA has demonstrated a strong panorama capability for an API that has not been seen, and only an API document can be used to adapt effectively to a new API</li>
<li>DFSDT expanded the search space by considering multiple reasoning tracks, achieving better performance than React --<strong>It's a study of the reasoning strategy.</strong></li>
</ul>
<p><strong>In this study, searches are conducted only once for user commands, and the search is linked to API, which gives models in multiple rounds of reasoning as context references to facilitate their thinking on reasoning strategies.</strong></p>
<p><strong>The DFSDTT given here is a decision-making reasoning strategy that is closely linked to the React. This paper is concerned not only with the utility caller performance but also with the process of extrapolating questions with the tool caller.</strong></p>
<h2>Concluding remarks</h2>
<p>Recalling the research context of Tool Use, from the self-supervised fine-tuning of Toolformer, the Retrieving Enhancement Training of Gorilla, the Retrievation Retrieval of Tulip Agent, to the large-scale system engineering of ToolLLM, the study line has been structured around a question:<strong>How to get the model to find the right one in the big tool and call it right.</strong></p>
<p>But,<strong>With the upgrading of basic modelling capacity and the emergence of agreements such as MCP (Model Context Protocol), the marginal benefits of Tool Calling as an independent training orientation are declining.</strong> The current front-line model already has a strong functional call capability and many scenes no longer require additional fine-tuning or complex search tubes for the “church” model tool. MCP deals with another layer of questions: how the tool is described, discovered, called and how to create a stable interface between the client and Server.</p>
<p>Many of the problems previously covered by the Tool Use study, such as the expansion of context, tool retrieval, multiple rotations, are now being more oriented towards protocols, running time and Agent projects.<strong>The focus of follow-up attention may be on how to move beyond training a model that is more tool-friendly and how to use tools, competencies, context and implementation processes as a stabilization system.</strong> This document is not updated.</p>
