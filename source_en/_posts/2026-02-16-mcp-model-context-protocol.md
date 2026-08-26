---
title: MCP (Model Context Protocol)
title_zh: MCP (Model Context Protocol)
date: 2026-02-16 11:30:00 +0800
categories:
- Agent Systems
- Agent Infrastructure
tags:
- MCP
- Protocols
- Tool Use
- Context Engineering
- Tutorial
author: Hyacehila
excerpt: An introduction to MCP's Host, Client, and Server layers, stdio and Streamable HTTP transports, and the workflow
  for building and debugging MCP servers and clients.
description: An introduction to MCP's Host, Client, and Server layers, stdio and Streamable HTTP transports, and the workflow
  for building and debugging MCP servers and clients.
excerpt_zh: 介绍 MCP 的 Host、Client、Server 分层，stdio 与 Streamable HTTP 传输，以及用 Python SDK 编写和调试 MCP Server/Client 的基本流程。
permalink: /blog/2026/02/16/mcp-model-context-protocol/
lang: en
translation_key: 2026-02-16-mcp-model-context-protocol
translation_status: machine
translation_source_hash: 118085f877726ee8660a87f5f0cf49a7c6f11ed68bf4c7cc891e09e4659fcd65
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>MCP originated on 25 November 2024  <a href="https://www.anthropic.com/news/model-context-protocol">Anthropic published article</a> </p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/03/10/from-mcp-to-agent-skills/">From MCP to Argentina Skills: Why does Agent need a new context work protocol?</a>、<a href="/en/blog/2026/05/17/agent-resource-collection/">Agent Extra Resource Collection: Skills, MCP Server, Plugins and Practical Tools</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>This paper refers to one of the most important articles in the world. <a href="https://zhuanlan.zhihu.com/p/29001189476">- It's a piece.</a>, part of the introduction and code is derived from this text.</p>
<h2>MCP Introduction</h2>
<p>MCP (Model Context Protocol, Model Context Protocol) defines the way in which context information is exchanged between the application and the AI model. It provides a protocol interface for tools, data sources and alert templates to enable developers to<strong>Connect data sources, tools and functions to AI models in a consistent manner</strong>。</p>
<p>MCP aims to reduce duplication of tools when accessing. Developer does not have to rewrite a login and callback set for each client; the same Server can be found and called by different host applications as long as the client supports MCP.</p>
<p>MCP represents two Agent approaches: the former relies on developers to adapt and harmonize such interfaces; the latter allows LLM to simulate human operations by visual recognition of what is near human sight.</p>
<p>Let's see why MCP is here. The early developers mainly rely on prompt to stuff the scene information into models, but handwritten prompt alone is difficult to maintain after the growing number of tools, files, databases and business situations.</p>
<p>Before the MCP was born, we usually manually paste the scene information to complement the prompt. The problem has changed, and it has become more and more.<strong>Manual</strong>It's getting harder and harder to put information in a stable format.</p>
<p>Many LLM platforms were introduced to address manual prompt limitations <code>function call</code> function. Models can be used to access data or perform operations by predefined functions, where needed, with increased automation and performance.</p>
<p><strong>Function call platform is highly dependent</strong>, the difference between the different LLM platform 's funcing call API is greater, and the adaptation cost increases; when switching to the platform, most of the codes are often rewritten.</p>
<p><strong>Data and tools are objective in themselves.</strong>We want to connect them to models that are smoother and more uniform. Anthropic designed MCP based on this, making it easier for LLM to access data or to call tools. The advantages of MCP include:</p>
<ul>
<li><strong>Ecology</strong> - MCP offers a lot of ready plugs, your AI can use directly.</li>
<li><strong>Uniformity</strong> - No restriction on a specific AI model, any model that supports MCP can be switched.</li>
<li><strong>Data security</strong> - You left your sensitive data on your computer, not all of it. Because we can design our own interfaces to determine which data to transfer.</li>
</ul>
<p>Detailed methods for MCP use, reference <a href="https://modelcontextprotocol.io/introduction">Document</a>  The relevant SDKs were presented with some examples.</p>
<p><strong>Whether it is FUNKING or MCP, the model's own Toolcall capabilities still stem from the original JSON Schema Output; we simply add functions to it, and do not change the model itself.</strong></p>
<h2>MCP Architecture</h2>
<h3>Basic components</h3>
<p>MCP consists of three components: Host, Clint and Server. They can be understood in a practical context:</p>
<p>Assuming you are asking with Claude Desktop (Host):&quot;What documents are on my desktop?&quot;</p>
<ol>
<li><strong>Host</strong>Claude Desktop, as Host, receives your questions and interacts with Claude.</li>
<li><strong>Client</strong>: When Claude Model decides to access your file system, the embedded MCP Clinic will be activated. This Clinent is responsible for establishing connections with the appropriate MCP Server.</li>
<li><strong>Server</strong>: In this example, the file system MCP Server will be called. It is responsible for executing the actual file scanning operation, accessing your desktop directory and returning the list of documents found.</li>
</ol>
<p>Of which Host handles semantic and interactive needs, Clit interacts with Server as an intermediary, Server accesss the database to obtain data and returns to Host to generate answers.</p>
<p>Process: Your question →Claude Desktop (Host)→Claude Model →Claude requires file information →MCP Clinic to connect to the →MCP Server → to execute the operation →Claude returns the result →to generate answers → on Claude Desktop.</p>
<p>This architecture allows LLM to access tools and data sources in different settings; developers need only develop corresponding MCP Servers, without concern for the details of the Host and Clarents' realizations.<strong>MCP Server</strong> : A program that provides context information to MCP clients can run on remote hosting servers or locally.</p>
<h3>Transfer level: stdio, Streamable HTTP, and why much of the information is still written on SSE</h3>
<p>The understanding of the basic components also requires a clear understanding of how they communicate and what changes have occurred in the communication mechanisms. The standard transmission in the current code is mainly <code>stdio</code> and <code>Streamable HTTP</code>I'm sorry. And... <code>SSE</code> The following is a historical legacy of the official document: Many of the online materials refer to the old version of the information and are not included in the MCP follow-up update. See the quotations in this section.</p>
<table>
<thead>
<tr>
<th>Dimensions</th>
<th><code>stdio</code></th>
<th><code>Streamable HTTP</code></th>
</tr>
</thead>
<tbody><tr>
<td>Start with</td>
<td>Client Start Server Subprocess</td>
<td>Server independently run and expose HTTP endpoint</td>
</tr>
<tr>
<td>Deployment location</td>
<td>Usually on the plane.</td>
<td>It's available on the machine and more commonly in remote services.</td>
</tr>
<tr>
<td>Message Channel</td>
<td>JSON-RPC, GO! <code>stdin/stdout</code>♪ Log goes ♪ <code>stderr</code></td>
<td>JSON-RPC go HTTP <code>POST/GET</code>, if necessary <code>SSE</code> Fluid Return</td>
</tr>
<tr>
<td>Typical scene</td>
<td>Local IDE, desktop end tool, file system and script tool</td>
<td>Cloud tool, team-sharing service, connector to certify</td>
</tr>
<tr>
<td>Certification and security</td>
<td>Dependence on in-house privileges, start-up commands, environment variables and client configuration</td>
<td>Dependence on HTTP assurance, authorization process and network boundaries</td>
</tr>
<tr>
<td>Remote and Multi-Custal</td>
<td>Not suitable for direct use as remote shared service</td>
<td>More suitable for accessing multiple clients as remote service</td>
</tr>
</tbody></table>
<p><code>stdio</code> Understandable as the most localized connection: Clit started MCP Server subprocesses like a father process and then wrote JSON-RPC requests into Server <code>stdin</code>From Server <code>stdout</code> Read JSON-RPC response. Because... <code>stdout</code> It's already been negotiated, so the debugging log should have been written. <code>stderr</code>Otherwise, it is easy to contaminate protocol data. (Do not write in a locally run MCP Server and Clean code)<code>print</code>Or other uses.<code>stdio</code>function)</p>
<p><code>Streamable HTTP</code> More like our familiar remote service: MCP Server run independently, Clean connects a MCP endpoint through HTTP <code>POST/GET</code> Send and receive protocol messages. Servers are available when needed <code>SSE</code> Keep pushing messages, so... <code>SSE</code> Not today with <code>stdio</code> , which is a fluid mechanism in HTTP transfers.</p>
<blockquote>
<p><strong>Version Description</strong>The blogger says:<a href="https://modelcontextprotocol.io/specification/2024-11-05/basic/transports">Version 2024-11-05</a>The remote transmission in it is called <code>HTTP with SSE</code>From <a href="https://modelcontextprotocol.io/specification/2025-03-26/changelog">Version 2025-03-26</a>Started by <code>Streamable HTTP</code> replace;update <a href="https://modelcontextprotocol.io/specification/2025-11-25/basic/transports">2025-11-25 Transmission Code</a>And by <code>stdio</code> and <code>Streamable HTTP</code> As standard transmission. So read it in the old article. <code>SSE/HTTP+SSE</code> It is not necessarily wrong, it is just the version that is earlier.</p>
</blockquote>
<h3>Communications before the start of the mission</h3>
<p>MCP Start with life cycle management, client sending <code>initialize</code> Request for a link and a consultative support function. After initialization is successful, the client sends a notice indicating that it is ready. In the initialization process, the MCP client manager that AI applies will establish connections to the configured server and store its functionality for subsequent use. The application uses this information to determine which servers provide specific types of functionality (tools, resources, tips) and whether they support real-time updates.</p>
<p>When a connection is created, the client sends it <code>tools/list</code> Request, get the Server exposure tool list. Responding <code>tools</code> arrays containing each tool <code>name</code>、<code>description</code>、<code>inputSchema</code> - What?<strong>The array structure allows a Server to open multiple tools at the same time and allows the client to display and call one by one.</strong></p>
<p>Each of the response 's target objects contains several key fields:</p>
<ul>
<li><strong><code>name</code></strong> : The only identifier in the server namespace.</li>
<li><strong><code>title</code></strong> : user-friendly display name of the tool that the client can show to the user</li>
<li><strong><code>description</code></strong> : detail the functionality of the tool and when it will be used.</li>
<li><strong><code>inputSchema</code></strong> : a JSON Schema to define the expected input parameters, to support type validation and to provide clear documentation of the required and optional parameters.</li>
</ul>
<p>Host or Clear consolidates the tools connected to MCP Server into a tool registration form, and then gives the tool description to the model for reference. The model determines whether to call on the basis of user requests and tool descriptions; the true execution is still sent back to the client for the parameters.</p>
<h3>How do models determine the choice of tools?</h3>
<p>The basic structure should be as</p>
<ol>
<li>Client (Host) sends your questions to Claude.</li>
<li>Claude analyses the tools available and decides which one (or more) to use.</li>
<li>Client executes the selected tool through MCP Server.</li>
<li>The results of the tool are returned to Claude.</li>
<li>Claude constructs the final prompt with the results and generates a response in the natural language.</li>
<li>Response to final display to user!</li>
</ol>
<p>This call can be made in two steps:</p>
<ol>
<li>The LLM (Claude) determines which MCP Servers use.</li>
<li>Implement corresponding MCP Server and reprocess the results.</li>
</ol>
<p>The overall logical reference figure, which is shown in the watermark</p>
<p><img src="https://pica.zhimg.com/v2-9d3681630ed930a8dc74d3b452c0cc94_1440w.jpg" alt="alt text"></p>
<h4>Tool Selection</h4>
<p>First step, first step.<strong>How does the model determine which tools should be used?</strong> </p>
<p>Read the code and find out that the model is used to determine which tools are currently available through the Prompt. We're through.<strong>Passing the specific use description of the tool to the model in text</strong>, to provide models with an understanding of the tools and the real-time selections.</p>
<p>That's...</p>
<pre><code class="language-python">.. # 省略了无关的代码
 async def start(self):
     # 初始化所有的 mcp server
     for server in self.servers:
         await server.initialize()
 ​
     # 获取所有的 tools 命名为 all_tools
     all_tools = []
     for server in self.servers:
         tools = await server.list_tools()
         all_tools.extend(tools)
 ​
     # 将所有的 tools 的功能描述格式化成字符串供 LLM 使用
     # tool.format_for_llm() 我放到了这段代码最后，方便阅读。
     tools_description = &quot;\n&quot;.join(
         [tool.format_for_llm() for tool in all_tools]
     )
 ​
     # 这里就不简化了，以供参考，实际上就是基于 prompt 和当前所有工具的信息
     # 询问 LLM（Claude） 应该使用哪些工具。
     system_message = (
         &quot;You are a helpful assistant with access to these tools:\n\n&quot;
         f&quot;{tools_description}\n&quot;
         &quot;Choose the appropriate tool based on the user&#39;s question. &quot;
         &quot;If no tool is needed, reply directly.\n\n&quot;
         &quot;IMPORTANT: When you need to use a tool, you must ONLY respond with &quot;
         &quot;the exact JSON object format below, nothing else:\n&quot;
         &quot;{\n&quot;
         &#39;    &quot;tool&quot;: &quot;tool-name&quot;,\n&#39;
         &#39;    &quot;arguments&quot;: {\n&#39;
         &#39;        &quot;argument-name&quot;: &quot;value&quot;\n&#39;
         &quot;    }\n&quot;
         &quot;}\n\n&quot;
         &quot;After receiving a tool&#39;s response:\n&quot;
         &quot;1. Transform the raw data into a natural, conversational response\n&quot;
         &quot;2. Keep responses concise but informative\n&quot;
         &quot;3. Focus on the most relevant information\n&quot;
         &quot;4. Use appropriate context from the user&#39;s question\n&quot;
         &quot;5. Avoid simply repeating the raw data\n\n&quot;
         &quot;Please use only the tools that are explicitly defined above.&quot;
     )
     messages = [{&quot;role&quot;: &quot;system&quot;, &quot;content&quot;: system_message}]
 ​
     while True:
         # Final... 假设这里已经处理了用户消息输入.
         messages.append({&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: user_input})
 ​
         # 将 system_message 和用户消息输入一起发送给 LLM
         llm_response = self.llm_client.get_response(messages)
 ​
     ... # 后面和确定使用哪些工具无关

 ​
 class Tool:
     &quot;&quot;&quot;Represents a tool with its properties and formatting.&quot;&quot;&quot;
 ​
     def __init__(
         self, name: str, description: str, input_schema: dict[str, Any]
     ) -&gt; None:
         self.name: str = name
         self.description: str = description
         self.input_schema: dict[str, Any] = input_schema
 ​
     # 把工具的名字 / 工具的用途（description）和工具所需要的参数（args_desc）转化为文本
     def format_for_llm(self) -&gt; str:
         &quot;&quot;&quot;Format tool information for LLM.
 ​
         Returns:
             A formatted string describing the tool.
         &quot;&quot;&quot;
         args_desc = []
         if &quot;properties&quot; in self.input_schema:
             for param_name, param_info in self.input_schema[&quot;properties&quot;].items():
                 arg_desc = (
                     f&quot;- {param_name}: {param_info.get(&#39;description&#39;, &#39;No description&#39;)}&quot;
                 )
                 if param_name in self.input_schema.get(&quot;required&quot;, []):
                     arg_desc += &quot; (required)&quot;
                 args_desc.append(arg_desc)
 ​
         return f&quot;&quot;&quot;
 Tool: {self.name}
 Description: {self.description}
 Arguments:
 {chr(10).join(args_desc)}
 &quot;&quot;&quot;
</code></pre>
<p><strong>Models determine which tools to use by providing structured descriptions of all tools and example for the few-shot</strong>。</p>
<h4>Tool implementation and structural feedback</h4>
<p>The tool is more straightforward in implementing this step. Take the last step, we send the system program with the user message and then receive the model response. After the model analyses the user request, it is decided whether the tool needs to be called:</p>
<ul>
<li><strong>When no tools are needed</strong>: The model directly generates natural language responses.</li>
<li><strong>When tools are needed</strong>: Model output structured JSON format tool call request.</li>
</ul>
<p>The response contains a structured JSON format tool to call, and the client will execute the corresponding tool according to this json code. If the model is implemented tool call, the result of the tool implementation <code>result</code> will be joined with system program and user message<strong>Resend</strong>To the model, request the model to generate the final response. If the json code is in trouble or the model is hallucinating, we'll skip the invalid call request.</p>
<pre><code class="language-python">... # 省略无关的代码
 async def start(self):
     ... # 上面已经介绍过了，模型如何选择工具
 ​
     while True:
         # 假设这里已经处理了用户消息输入.
         messages.append({&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: user_input})
 ​
         # 获取 LLM 的输出
         llm_response = self.llm_client.get_response(messages)
 ​
         # 处理 LLM 的输出（如果有 tool call 则执行对应的工具）
         result = await self.process_llm_response(llm_response)
 ​
         # 如果 result 与 llm_response 不同，说明执行了 tool call （有额外信息了）
         # 则将 tool call 的结果重新发送给 LLM 进行处理。
         if result != llm_response:
             messages.append({&quot;role&quot;: &quot;assistant&quot;, &quot;content&quot;: llm_response})
             messages.append({&quot;role&quot;: &quot;system&quot;, &quot;content&quot;: result})
 ​
             final_response = self.llm_client.get_response(messages)
             logging.info(&quot;\nFinal response: %s&quot;, final_response)
             messages.append(
                 {&quot;role&quot;: &quot;assistant&quot;, &quot;content&quot;: final_response}
             )
         # 否则代表没有执行 tool call，则直接将 LLM 的输出返回给用户。
         else:
             messages.append({&quot;role&quot;: &quot;assistant&quot;, &quot;content&quot;: llm_response})
</code></pre>
<p>Accordingly:</p>
<ul>
<li>Tool documents directly affect the quality of the model selection tool. Name, docstring and parameter description are written to avoid writing only one general verb.</li>
<li>MCP tool selection continues to rely on models to understand text descriptions. The more the model understands the boundaries of tasks and tools, the more stable the call effect.</li>
</ul>
<p><strong>The blogger adds:<code>@mcp.tool()</code> It's a direct function name and <code>docstring</code> It's a tool. <code>name</code> and <code>description</code>。</strong> Parameters and return value information will also be derived from type labels,<code>docstring</code> or SDK for the interpretation of a function signature. So when writing MCP Tool, the function name and description text are not an annotated text, they are the interface documents that the model sees.</p>
<h2>MCP Servers</h2>
<p>MCP servers are procedures to make specific features available to AI applications through standardized protocol interfaces. This is also the layer that developers need to reach. The server provides functionality through three basic components:</p>
<ul>
<li><strong>Tools</strong> LLM can call these functions on its own initiative and decide when to use them upon request. Tools can write to a database, call an external API, modify files or trigger other logic.</li>
<li><strong>Resources</strong> Passive data sources Provide context information</li>
<li><strong>Prompts Hint</strong> Pre-engineered command templates that show the model how to use specific tools and resources.</li>
</ul>
<p>Tools are fixed-form interfaces that LLM can access. MCP to validate with JSON Schema. Each tool performs a single operation and has clearly defined inputs and outputs. The tool may require prior user clearance, which helps ensure that users maintain control over the operation of the model. Legal<strong>Protocol operations</strong> Including<code>tools/list</code>  and <code>tools/call</code> Returns the performance results of the description arrays and tools for the tool, respectively.</p>
<p>The tools are controlled by models, which can be automatically detected and called upon by artificial intelligence models. However, MCP also maintains manual monitoring through a variety of mechanisms, including the opening and closing of user control models, pre-sets and the execution of approvals for each tool.</p>
<p>Resources provide structured information access, AI <strong>Application</strong>This information is read and then presented to the model as context. It differs from the tool in that resources are primarily responsible for providing context and not for enabling the model to implement the action. Resource support<strong>Direct resources</strong>URI, which points to fixed data; also supports<strong>Resource Templates</strong>, which is the dynamic with parameters. Relevant <strong>Protocol operations</strong> Including <code>resources/list</code>、<code>resources/templates/list</code>、<code>resources/read</code>、<code>resources/subscribe</code>I'm sorry. Resource discovery and access is application-driven and the interface format is determined by the specific client.<strong>Resources is called by Application rather than directly by the model.</strong></p>
<p>The hint provides a reusable template. They allow MCP server authors to provide parameterized indicators for field tasks or to demonstrate how best to use the MCP server. Legal<strong>Protocol operations</strong> Including<code>prompts/list</code>  <code>prompts/get</code> The hint is controlled by the user and requires a visible call.<strong>Prompts are called by users rather than by models</strong></p>
<p>Only the tool for LLM is the Tools, which are used by the other parts of the program, and which are not the focus of consideration, and we will focus only on the Tools for building the MCP Servers themselves in our subsequent presentations and allow LLM to use the Tools. Actually... <strong>For most AI developers, we just need to care about the realization of Server.</strong> </p>
<p><strong>MCP Servers is responsible for harmonizing exposure tools and other elements for Agent dynamic detection and call without having to coding the tool sheet (need to work with Clint). This is the MCP's main value. Local <code>stdio</code> Server often runs in an independent subprocess, remote <code>Streamable HTTP</code> In the mode, Server is more like a service.</strong></p>
<h2>Connecting Clinent to Servers</h2>
<p>As can be seen from the previous steps, MCP runs on the general level of Host, Clean, Server. Users use Claude Desktop, Claude Code, VS Code, such as Host/Application; they usually have a Client that supports MCP; Server runs locally or in clouds.<strong>If only using MCP, it would not affect understanding if Host and Client were to be seen as a whole for the time being.</strong></p>
<p>When only MCP Server can be done, you can not care how it works within the Clinent. MCP serves to fix the interaction between Server and LLM into a set of protocols. The developers are primarily responsible for achieving Server and deploying it to the cloud or handing it over to the user for local installation. To allow the model to identify MCP Server, only to configure it in the client that supports MCP: Local <code>stdio</code> Server usually configures start-up commands, remote <code>Streamable HTTP</code> Server usually configures URL/endpoint and authentication information.</p>
<p>This one down here, JSON, is local. <code>stdio</code> Typical configuration for server:</p>
<pre><code class="language-json">{
  &quot;mcpServers&quot;: {
    &quot;filesystem&quot;: {
      &quot;command&quot;: &quot;npx&quot;,
      &quot;args&quot;: [
        &quot;-y&quot;,
        &quot;@modelcontextprotocol/server-filesystem&quot;,
        &quot;/Users/username/Desktop&quot;,
        &quot;/Users/username/Downloads&quot;
      ]
    }
  }
}
</code></pre>
<p>of which <code>filesystem</code> is the name displayed by the server in Claude Desktop or other client.<code>command</code> It's the enforceable program that Client is about to start.<code>args</code> Writes automatically installed parameters in lycée, server package name, and directory that allows Server access. Rebooting the Clit will be possible with modifications to the configuration, and the tool will be used in the follow-up of the model.</p>
<p><strong>For Local <code>stdio</code> Server, MCP Server is actually configured to fine-tune and split manual commands, and there is no difference in the nature of manual MCP Server running commands, which are pulled by Client and then passed <code>stdin/stdout</code> Exchange protocol messages.</strong></p>
<p>Now we have basically all the clients supporting us to connect to remote tools. Compared to local tools, remote MCP server usually passes <code>Streamable HTTP</code> Exposure of a URL/endpoint, Clit or Custom Contractors to act as a bridge between Claude and a remote MCP server. In order to ensure that access is legal, most remote Servers require authentication or authorization, which is determined by the corresponding Clit product and remote service, and is sufficient to act as a reminder. When the connection is successful, the remote server resource and tip message will appear in your Claude dialogue.</p>
<p><strong>Different ways of deploying MCP Servers are different, with all Servers of some platforms being controlled in one JSON, and some of the Clients achieve more independent configuration logic, distinguishing local <code>stdio</code> Servers and Remote <code>Streamable HTTP</code> Servers</strong></p>
<h2>MCP Server based on Python SDK</h2>
<p> <strong>For most AI developers, we just need to care about the realization of Server.</strong> This requires some understanding of MCP Server's working principles to ensure the expansiveness of codes.</p>
<p><strong>MCP is valuable in removing the capability upgrade from the Agent main code.</strong> Traditional Function Calling often requires changes to the clit code or hint; MCP Server can rediscover the tool by configuration access. The cost is that the tool search and the exposure to the capacity will be an additional layer of abstraction. Complex projects still require a budget for their own handling tools for naming, authorizing, searching quality and context.</p>
<p>Below is an example of a MCP Server with Python SDK.<code>mcp</code> FastMCP in the package will help us with the details of the protocol. This package also contains terminal tools and CIline associated codes, but is not used temporarily when writing Server.</p>
<pre><code class="language-python"># 导入开发MCP以及工具本身需要的Packages
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# 创建 MCP Server 也就是一个mcp对象，此时他还是空的，并给了这个Server一个名字
mcp = FastMCP(&quot;桌面 TXT 文件统计器&quot;)

#使用@mcp.tool() (装饰器)修饰了一个普通的Python函数，这样就从python函数到了一个MCP tool
#Python装饰器是一个非常强大的工具，不过我们再这里不再强调他
#为函数增加了输出类型提示int，这可以被后面的MCP SDK解析
#使用了文档字符串 撰写了doc 这个doc也会被MCP SDK解析 位于模块、类、方法或函数的第一个这样的注释为doc
#代码内部就是普通函数逻辑，很简单
@mcp.tool()
def count_desktop_txt_files() -&gt; int:
    &quot;&quot;&quot;Count the number of .txt files on the desktop.&quot;&quot;&quot;
    # Get the desktop path
    username = os.getenv(&quot;USER&quot;) or os.getenv(&quot;USERNAME&quot;)
    desktop_path = Path(f&quot;/Users/{username}/Desktop&quot;)

    # Count .txt files
    txt_files = list(desktop_path.glob(&quot;*.txt&quot;))
    return len(txt_files)

#装饰了另一个tool，一个Server里面可以拥有多个Tool很合理
@mcp.tool()
def list_desktop_txt_files() -&gt; str:
    &quot;&quot;&quot;Get a list of all .txt filenames on the desktop.&quot;&quot;&quot;
    # Get the desktop path
    username = os.getenv(&quot;USER&quot;) or os.getenv(&quot;USERNAME&quot;)
    desktop_path = Path(f&quot;/Users/{username}/Desktop&quot;)

    # Get all .txt files
    txt_files = list(desktop_path.glob(&quot;*.txt&quot;))

    # Return the filenames
    if not txt_files:
        return &quot;No .txt files found on desktop.&quot;

    # Format the list of filenames
    file_list = &quot;\n&quot;.join([f&quot;- {file.name}&quot; for file in txt_files])
    return f&quot;Found {len(txt_files)} .txt files on desktop:\n{file_list}&quot;
#mcp.run(): 这是服务器启动指令，在本地 stdio 示例中启动后会等待来自标准流的协议请求
if __name__ == &quot;__main__&quot;:
    # Initialize and run the server
    mcp.run()
</code></pre>
<h2>MCP Clinic based on Python SDK</h2>
<h3>What's Client doing?</h3>
<p>Look at the Clit's realization. For most tool developers, the Clit is provided by such host products as Claude Desktop, Cursor, Claude Code; all you need to know is how to fit MCP Server instead of making it happen.</p>
<p>But if MCP is to be embedded in self-study, then it is necessary to know what the Client in SDK has done.</p>
<p>A distinction must be made between Host and Clinent when studying the framework. Host is the layer that carries the Agent logic: maintaining dialogue, deciding when to call the tool, processing the loop and terminating conditions. MCP Clinic is an internal protocol adapter, which connects Server, column tools, adjusts and reads resources.<strong>Core Agent is on the Host level; the Clint level is responsible only for tool discovery and protocol communication.</strong></p>
<p><strong>Client does only two things: access to tools, resources and alerts exposed by Server; and call or read from the Host/Agent decision-making implementation tool.</strong> The Client code in SDK is intended to encapsulate these protocol actions. The multiple tool call cycle does not belong to the MCP Clinic itself, and the number of calls and when they will stop is still determined by the Host Layer code.</p>
<p>Host + Clent + Server combines to achieve one thing:<strong>Agent is responsible for decision-making and MCP is responsible for providing the capability interface.</strong></p>
<p>So,<strong>MCP Clinic is not Agent; it is Agent's protocol/ bus layer using MCP capabilities.</strong> MCP does not itself do reasoning, planning, memory or circulation control. It allows Host/Agent to discover tools through a standard interface, capture context, call external capabilities and decorate tools and data from the Host code.</p>
<p>If you re-use a high-level product like Claude Code SDK, you're connected to an Agent Runt that already contains the Host decision logic, not the nudity MCP Clinic. Naked Client only handles protocols; high-level SDKs may contain react loops, tool selection and the termination logic of the task.</p>
<h3>A simple example of Client and Host.</h3>
<p>A simple example of what was achieved by Client and Host is presented below, which is used to refer to the achievements of the Client and its use by Host.</p>
<p>This example is local. <code>stdio</code> Transport, so Clinent will start the Python subprocess and create it through standard streams and it will be created <code>ClientSession</code>I'm sorry. If it's a remote <code>Streamable HTTP</code>, Host and Clinic still have similar duties, but the bottom connector will be from <code>stdin/stdout</code> Replace with HTTP endpoint.</p>
<pre><code class="language-python">import asyncio
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client, get_default_environment

def build_env() -&gt; dict:
    &quot;&quot;&quot;
    构建传递给 MCP Server 的环境变量。

    MCP Server 运行在一个独立的子进程中，因此需要显式传递必要的环境变量。
    本函数首先获取 SDK 提供的默认安全环境配置，然后将我们需要增强的
    数据路径变量 &#39;ENHANCED_DATA_PATH&#39; 注入其中。
    &quot;&quot;&quot;
    # 1. 获取默认环境：包含 PATH 等基础变量，确保 Python 能正常运行
    env = get_default_environment()

    # 2. 注入自定义变量：让 Server 能够通过环境变量获取配置信息
    enhanced = os.environ.get(&quot;ENHANCED_DATA_PATH&quot;)
    if enhanced:
        env[&quot;ENHANCED_DATA_PATH&quot;] = enhanced
    return env

def server_params(server_py: str = &quot;utils/mcp_server.py&quot;) -&gt; StdioServerParameters:
    &quot;&quot;&quot;
    构造 Server 的启动参数 (StdioServerParameters)。

    这里指定了如何启动 MCP Server：
    - command: 使用 &quot;python&quot; 命令
    - args: 传递脚本路径作为参数
    - env: 使用 build_env() 构建的环境变量
    &quot;&quot;&quot;
    # 使用绝对路径，避免因 cwd (当前工作目录) 不同导致找不到文件
    server_py = os.path.abspath(server_py)
    return StdioServerParameters(command=&quot;python&quot;, args=[server_py], env=build_env())

def parse_result(result):
    &quot;&quot;&quot;
    解析 MCP Protocol 的返回结果 `CallToolResult`。

    MCP 的返回结果结构可能包含 TextContent, ImageContent 或 EmbeddedResource。
    本函数的目的是将其简化为 Host 易处理的字典或数据结构。
    &quot;&quot;&quot;
    # 结果的主要内容都在 content 列表字段中
    content = getattr(result, &quot;content&quot;, None)
    if content:
        # 策略 1: 优先提取结构化数据 (EmbeddedResource 或类似 data 字段)
        for item in content:
            data = getattr(item, &quot;data&quot;, None)
            if data is not None:
                return data

        # 策略 2: 提取文本内容，并尝试解析为 JSON
        for item in content:
            text = getattr(item, &quot;text&quot;, None)
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except Exception:
                    # 如果不是 JSON，则直接返回原始文本
                    return {&quot;raw_text&quot;: text}

    # 兜底：如果无法解析，返回原始对象的字典包装
    return {&quot;result&quot;: result}

async def list_tools(server_py: str = &quot;utils/mcp_server.py&quot;):
    &quot;&quot;&quot;
    Client 核心功能：列出 Server 提供的所有工具。

    步骤：
    1. stdio_client: 启动子进程，建立 stdio 管道。
    2. ClientSession: 在管道上建立 MCP 协议会话。
    3. initialize: 执行握手协议。
    4. list_tools: 发送 tools/list 请求。
    &quot;&quot;&quot;
    async with stdio_client(server_params(server_py)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            # 提取关键元数据 (name, description, schema) 返回给 Host 用于决策
            return [
                {&quot;name&quot;: t.name, &quot;description&quot;: t.description, &quot;input_schema&quot;: t.inputSchema}
                for t in resp.tools
            ]

async def call_tool(tool_name: str, arguments: dict | None = None, server_py: str = &quot;utils/mcp_server.py&quot;):
    &quot;&quot;&quot;
    Client 核心功能：调用指定工具。
    &quot;&quot;&quot;
    arguments = arguments or {}
    async with stdio_client(server_params(server_py)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return parse_result(result)

# --- Host Agent 使用示例 (Simulated) ---

async def host_agent_demo():
    &quot;&quot;&quot;
    模拟 Host (Agent) 使用 Client 进行工具发现和调用的过程。
    这里的 Host 扮演“决策者”的角色，而 Client 扮演“执行者”。
    &quot;&quot;&quot;
    print(&quot;=== Host Agent Started ===&quot;)

    # 1. 发现能力 (Tool Discovery)
    # Host 询问 Client：目前有哪些工具可用？
    print(&quot;\n[Host] Discovering tools...&quot;)
    # 假设 utils/mcp_server.py 是我们编写好的 Server 脚本
    tools = await list_tools(&quot;utils/mcp_server.py&quot;)

    print(f&quot;[Host] Found {len(tools)} tools:&quot;)
    for t in tools:
        print(f&quot;  - Name: {t[&#39;name&#39;]}&quot;)
        print(f&quot;    Desc: {t[&#39;description&#39;]}&quot;)

    if not tools:
        print(&quot;[Host] No tools found. Exiting.&quot;)
        return

    # 2. 模拟 LLM 决策过程
    # 假设 LLM 根据 Prompt 和工具描述，决定调用 &#39;list_desktop_txt_files&#39;
    # 注意：这里的逻辑通常由 LLM 完成
    target_tool = tools[0][&quot;name&quot;]  # 简单起见，直接取第一个
    print(f&quot;\n[Host] DECISION: I will use the tool &#39;{target_tool}&#39; to gather information.&quot;)

    # 构造通过 prompt 分析出的参数 (此处为硬编码示例)
    args = {}

    # 3. 执行工具 (Tool Execution)
    print(f&quot;[Host] Requesting Client to execute &#39;{target_tool}&#39;...&quot;)
    result = await call_tool(target_tool, args, &quot;utils/mcp_server.py&quot;)

    # 4. 获取结果
    print(f&quot;[Host] Execution Result received:&quot;)
    # 结果可能是列表、字典或文本，这里做简单的打印
    print(result)

    print(&quot;\n=== Host Agent Finished ===&quot;)
</code></pre>
<h3>A complete local stdio collaborative process</h3>
<p>Once. <code>call_tool()</code> The end-to-end collaborative process (Agent ↔ MCP Clent ↔ MCP Server). It's still local. <code>stdio</code> transport。</p>
<ol>
<li><p><strong>Agent Decision-making</strong></p>
<ul>
<li><code>DecisionToolsNode</code> Select the tool to execute. <code>ExecuteToolsNode</code></li>
</ul>
</li>
<li><p><strong>Host Call MCP Clinic</strong></p>
<ul>
<li><code>ExecuteToolsNode</code> Call:<ul>
<li><code>call_tool(&quot;utils/mcp_server&quot;, tool_name, {})</code></li>
</ul>
</li>
</ul>
</li>
<li><p><strong>MCP Clint Start Server Subprocess</strong></p>
<ul>
<li>Parameters for the start of the construction process:<ul>
<li><code>StdioServerParameters(command=&quot;python&quot;, args=[server_script_path], env=...)</code></li>
</ul>
</li>
</ul>
</li>
<li><p><strong>MCP SDK Creates local stdio channel (IPC)</strong></p>
<ul>
<li><code>stdio_client(server_params)</code> Create stdin/stdout channel for main and subprocesses</li>
</ul>
</li>
<li><p><strong>MCP SDK Launch RPC Call</strong></p>
<ul>
<li><code>session.initialize()</code> Shake hands, please.</li>
<li><code>session.call_tool(tool_name, arguments)</code> Launch Tool Call</li>
</ul>
</li>
<li><p><strong>MCP Server Implementation Tool</strong></p>
<ul>
<li>Server-end corresponding <code>@mcp.tool()</code> Function triggers</li>
<li>Call after internal reading/loading of data <code>analysis_tools</code> Complete statistical or chart generation</li>
</ul>
</li>
<li><p><strong>Return to Clinent</strong></p>
<ul>
<li>Server returns results via MCP protocol</li>
<li>Clinent parsing results and returning to <code>ExecuteToolsNode</code></li>
</ul>
</li>
<li><p><strong>Write share status (shared)</strong></p>
<ul>
<li><code>ExecuteToolsNode</code> Writes products such as charts/tables to:<ul>
<li><code>shared[&quot;stage2_results&quot;]</code></li>
</ul>
</li>
</ul>
</li>
</ol>
<h2>MCP Inspector</h2>
<h3>About Inspector</h3>
<p>MCP Inspector is a visual interactive tool for testing and debugging MCP Server. You can imagine it as a "web-based version of Claude Desktop" or "API debugging tool" (like Postman), which is used to check if your MCP Server works properly.</p>
<p>Inspector is one<strong>Standard MCP Clit Achieved</strong>I'm sorry. It's not responsible for talking to users, it's responsible for making protocol requests, showing responses and helping you see what Server has exposed.</p>
<ul>
<li><strong>It simulates client behavior.</strong>: it sends standard JPON-RPC requests, following the MCP Clint to Server path.</li>
<li><strong>It checks the agreement.</strong>: If your Server can display Schema, call tools, read resources in an Inspector, then the protocol layer is much less problematic when moving to Claude Desktop, Cursor or other MCP Clinic.</li>
</ul>
<p>You don't need to install it around the world, just run it using npx, as follows:</p>
<pre><code class="language-bash">npx @modelcontextprotocol/inspector &lt;你的启动命令&gt;
</code></pre>
<p>For MCP Server developed with Python</p>
<pre><code class="language-bash">npx @modelcontextprotocol/inspector uv run main.py
# 或者
npx @modelcontextprotocol/inspector python main.py
</code></pre>
<p>For the MCP Server developed by Node</p>
<pre><code class="language-bash">npx @modelcontextprotocol/inspector node build/index.js
</code></pre>
<p>If you need an environment variable setting</p>
<pre><code class="language-bash"># 在命令前加 env 变量，或者直接在 npx 后接命令
KEY=value npx @modelcontextprotocol/inspector python main.py
</code></pre>
<p>When running successfully, the terminal will display a local address (usually) <code>http://localhost:5173</code>) The browser will automatically open this page. That's MCP Inspector. <strong>When using MCP Inspector commands, strict attention is required to the catalogue, and only if the startup command is itself enforceable can MCP Inspector be used to correct the analysis, otherwise the connection cannot be made</strong></p>
<p>In the visual interface created by MCP Inspector, we can copy the order given to Server by the front UI to copy the order issued by Clint, and UI records the order to initiate the order on the left side, and Hestory records the order given by Clint to Server and the response of Server. Servers do not provide information for logs. The main interface is the MCP-related feature that we introduced in MCP Server, at which point LLM is not needed to call Tool, users to use Prompt or App to use Resource, and all operations to simplify the use of Server for our test interface.</p>
<p>When the Inspector reviews correctly, you can safely add configurations to Claude Desktop profile or host the Server platform online.</p>
<h3>From Inspector to Client</h3>
<p>Although Inspector and Claude Desktop/Code are both MCP Client, they're both MCP Client.<strong>Run Logic</strong>and<strong>Configure</strong>There are essential differences:</p>
<ul>
<li><strong>Inspector</strong>: Yes<strong>In a moment.</strong>、<strong>Command line</strong>I'm sorry. You tell it "go to the line now," and it runs and closes the web site process.</li>
<li><strong>Claude Client</strong>: Yes<strong>Lasting</strong>、<strong>Profile</strong>I'm sorry. You need to write running instructions into a JSON file, and Claude will read them and run them quietly in the back.</li>
</ul>
<p>In Inspirator, you usually finish all the content in one line:</p>
<pre><code class="language-bash"># 示例：一个需要 API Key 的 Python Server
MY_API_KEY=12345 npx @modelcontextprotocol/inspector python main.py --verbose
</code></pre>
<p>In the claude desktop config.json file, the above line of commands must be broken down into the following structures:</p>
<pre><code class="language-json">{
  &quot;mcpServers&quot;: {
    &quot;my-server-name&quot;: {
      &quot;command&quot;: &quot;python&quot;,
      &quot;args&quot;: [
        &quot;main.py&quot;,
        &quot;--verbose&quot;
      ],
      &quot;env&quot;: {
        &quot;MY_API_KEY&quot;: &quot;12345&quot;
      }
    }
  }
}
</code></pre>
<ol>
<li><strong>Comand (Major Command) I'm not sure.</strong>：<ul>
<li><strong>Inspector</strong>: python or node or uv.</li>
<li><strong>Claude</strong>: Must be in JSON &quot;command&quot; field.</li>
<li><strong>Attention.</strong>: Must be the name or absolute path of the enforceable procedure. If you're in Institute with npx, in JSON, usually. &quot;npx&quot;(In the case of Python, it is necessary to write in Comand the absolute path of the specific enforceable procedure).</li>
</ul>
</li>
<li><strong>Args (list of parameters)</strong>：<ul>
<li><strong>Inspector</strong>: A string separated by spaces, such as main.py-verbose.</li>
<li><strong>Claude</strong>: Must be<strong>String array</strong> [&quot;main.py&quot;, &quot;--verbose&quot;]。</li>
<li><strong>Significant differences</strong>: Can't put &quot;python main.py&quot; Writes in a string! File names and parameters must be removed.</li>
</ul>
</li>
<li><strong>Env (Environmental Variables)</strong>：<ol>
<li><strong>Inspector</strong>: Writes in front of the command, like KEY=value.</li>
<li><strong>Claude</strong>: Must be written in &quot;env&quot; Object inside. Claude. <strong>No, I won't.</strong>Automatically inherits the environment variables in your terminal, so all the required Key must be defined here in a visible way.</li>
</ol>
</li>
</ol>
<p><strong>The different platforms may have different versions of the specific JSON configuration. For Local <code>stdio</code> Server, the core remains split start-up commands, parameters and environmental variables; for remote <code>Streamable HTTP</code> Server, the focus of the configuration will be endpoint, authentication and authorization.</strong></p>
<h3>Inspect and MCP Server Code</h3>
<p>We can explain how Inspector works by simply telling us how he understands how MCP Clinic works, from Clit to Host, to further encapsulating and hiding their communication processes.</p>
<p>This section shows locals. <code>stdio</code> Server's debugging. At this point, <strong>Inspector Proxy</strong> You start your code (subprocess) like a "father process" and then communicate and interact with it through standard input/output. If a remote MCP service, Clit or Inspect connects to HTTP endpoint, instead of pulling up local subprocesses.</p>
<ul>
<li><strong>Writing (stdin)</strong>: Inspector sends the JPON-RPC request (e.g. "please list all tools") to your code.</li>
<li><strong>Read (stdout)</strong>: Your code prints the processing results (JSON format) to the console, and Inspect intercepts these outputs and shows them on the web page.</li>
</ul>
<p>Because stdout is used to transmit protocol data,<strong>Absolutely not.</strong>Use print(Python) or console.log(Node) in your code to print debug messages! This will destroy the JSON format, leading to Inspector's misreporting.<strong>Debug information should be printed in stderr, as shown below</strong></p>
<pre><code class="language-python">import sys
# 正确的调试方式：写入 stderr
print(&quot;Debug: Function called with a=10&quot;, file=sys.stderr)

# 或者使用 logging 模块（配置为写 stderr）
logger.info(&quot;Processing request...&quot;)
</code></pre>
<h4>Map of the Tools</h4>
<p>Suppose we have a simple Python Code that defines Server and Tool. </p>
<pre><code class="language-python">@mcp.tool()
async def calculate_sum(a: int, b: int) -&gt; int:
    &quot;&quot;&quot;Add two numbers.&quot;&quot;&quot;
    return a + b
</code></pre>
<p><strong>ListTools</strong>: Sends tils/list requests after Inspector starts, and then the MCP SDK based on Pydantic or type tips automatically generates the following JSON Schema:</p>
<pre><code class="language-json">{
  &quot;name&quot;: &quot;calculate_sum&quot;,
  &quot;description&quot;: &quot;Add two numbers.&quot;,
  &quot;inputSchema&quot;: {
    &quot;type&quot;: &quot;object&quot;,
    &quot;properties&quot;: {
      &quot;a&quot;: { &quot;type&quot;: &quot;integer&quot; },
      &quot;b&quot;: { &quot;type&quot;: &quot;integer&quot; }
    },
    &quot;required&quot;: [&quot;a&quot;, &quot;b&quot;]
  }
}
</code></pre>
<p>Inspect read this Schema:</p>
<ul>
<li>See name -&gt; Show in left list &quot;calculate_sum&quot;。</li>
<li>Seeing properties--&gt; Generate two input boxes on the right, with labels for each &quot;a&quot; and &quot;b&quot;, the type is limited to numbers.</li>
</ul>
<p>Click Run Tool</p>
<ol>
<li>Inspect sent request from   Other Organiser&quot;name&quot;: &quot;calculate_sum&quot;, &quot;arguments&quot;: {&quot;a&quot;: 10, &quot;b&quot;: 20&#125;&#125;。</li>
<li>Your calculate sum function is called.</li>
<li>Returns value 30 by stdout returned, inspector displayed in &quot;Result&quot; Area.</li>
</ol>
<h4>Map of Resources</h4>
<p><strong>Python code:</strong></p>
<pre><code class="language-python">@mcp.resource(&quot;file://logs/{name}&quot;)
def read_log(name: str) -&gt; str:
    return f&quot;Log content for {name}...&quot;
</code></pre>
<p><strong>Process for Inspira:</strong></p>
<ol>
<li><strong>ListResources</strong>: Sending resources/list.</li>
<li><strong>UI Display</strong>: Inspector lists all available resources in the Resourcees panel URI templates (e.g. file://logs/{name}).</li>
<li><strong>Interactive</strong>: Click on the resource in the list, Inspector will try to read (send results/read) and display the returned text or binary content in the preview window.</li>
</ol>
<h4>Map of Prompts (Phrases)</h4>
<p><strong>Python code:</strong></p>
<pre><code class="language-python">@mcp.prompt()
def review_code(code: str) -&gt; list[Message]:
    return [UserMessage(content=f&quot;Review this code: {code}&quot;)]
</code></pre>
<p><strong>Process for Inspira:</strong></p>
<ol>
<li><strong>ListPrompts</strong>: Inspector gets the list of hints.</li>
<li><strong>Parameter Fill</strong>: Inspect recognizes the review code required parameter code and generates a text box on UI for you to enter the code clip.</li>
<li><strong>Preview</strong>: After clicking on run, Inspector will not execute any AI calls, but will show<strong>The end-generated Prompt structure</strong>I'm sorry. That makes you check if your template logic is correct.</li>
</ol>
<h2>Concluding remarks</h2>
<p>MCP is a valuable engineering interface: it places tool discovery, tool call, resource access and reminder templates in the same set of protocols, and provides a clearer boundary realization for Server developers. It addresses standard interfaces, however, and does not automatically address tool design, competency governance, context organization and product experience.</p>
<p>That's why MCP and Agent Skills would be there together. MCP is more appropriate for external capabilities that are stable, reusable and require clear lines of authority; and Skills is better suited to hand over team processes, scripts and project knowledge to Agent at low cost. For developers, the key is not to bet on which to replace the other, but to see whether current capabilities are more like “service interfaces” or more like “readable work packages”.</p>
