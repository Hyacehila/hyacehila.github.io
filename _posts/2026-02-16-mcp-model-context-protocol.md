---
layout: blog-post
title: MCP (Model Context Protocol)
date: 2026-02-16 11:30:00 +0800
categories: [智能体系统]
tags: [MCP, Agents, Context Engineering]
excerpt: 本文详细介绍了MCP（Model Context Protocol）的架构与实现，包括Host、Client、Server的核心概念与Python SDK实战，并深入探讨了从MCP到Agent Skills的技术演进与未来思考。
---

# 关于MCP

MCP 起源于 2024 年 11 月 25 日  [Anthropic 发布的文章](https://www.anthropic.com/news/model-context-protocol) 

本文参考了一篇 [知乎文章](https://zhuanlan.zhihu.com/p/29001189476)，很多介绍与代码都源自此文。

## MCP 简介
MCP （Model Context Protocol，模型上下文协议）定义了应用程序和 AI 模型之间交换上下文信息的方式。这使得开发者能够**以一致的方式将各种数据源、工具和功能连接到 AI 模型**（一个中间协议层）

MCP 的目标是创建一个通用标准，使 AI 应用程序的开发和集成变得更加简单和统一。我们无须再为每一个模型单独设计Agent的代码，使用MCP构建的Agent可以不加修改的适配到所有支持MCP的LLM中。Anthropic 旨在实现 LLM Tool Call 的标准（LLM调用外部应用程序接口的标准）。

MCP与视觉派是两个不同的Agent思路，前者希望开发者提供足够的接口适配并统一这一类别接口，后者希望让LLM视觉识别看到和人类一样的内容并模拟人类进行操作。

现在我们来聊一聊为什么会有MCP，我们为什么需要他。MCP 的出现是 prompt engineering 发展的产物。更结构化的上下文信息对模型的 performance 提升是显著的。在构造 prompt 的时候添加场景信息也可以让模型更加容易理解真实场景中的问题。

在 MCP 诞生之前，我们采用手动的粘贴来将场景信息补充给prompt。随着我们要解决的问题越来越复杂，**手工**把信息引入到 prompt 中会变得越来越困难。并且很难保证足够的格式化。

为了克服手工 prompt 的局限性，许多 LLM 平台引入了 `function call` 功能。这一机制允许模型在需要时调用预定义的函数来获取数据或执行操作，显著提升了自动化水平与性能表现。

**function call 平台依赖性强**，不同 LLM 平台的 function call API 实现差异较大。增加了适配成本。切换平台就需要重写几乎全部的代码。

**数据与工具本身是客观存在的**，只不过我们希望将数据连接到模型的这个环节可以更智能更统一。Anthropic 基于此设计了 MCP，让 LLM 能轻松的获取数据或者调用工具。更具体的说 MCP 的优势在于：
- **生态** - MCP 提供很多现成的插件，你的 AI 可以直接使用。
- **统一性** - 不限制于特定的 AI 模型，任何支持 MCP 的模型都可以切换。
- **数据安全** - 你的敏感数据留在自己的电脑上，不必全部上传。（因为我们可以自行设计接口确定传输哪些数据）

关于MCP的详细使用方法，参考 [文档](https://modelcontextprotocol.io/introduction)  介绍了相关 SDK 与一些例子。

**无论是Function Calling还是MCP，模型本身的Toolcalling能力仍旧源自最初的JSON Schema Output，我们只是在其基础上附加功能，而不是模型本身的巨大改变。**

## MCP Architecture 
### 基本组件
MCP 由三个核心组件构成：Host、Client 和 Server。让我们通过一个实际场景来理解这些组件如何协同工作：

假设你正在使用 Claude Desktop (Host) 询问："我桌面上有哪些文档？"

1. **Host**：Claude Desktop 作为 Host，负责接收你的提问并与 Claude 模型交互。
2. **Client**：当 Claude 模型决定需要访问你的文件系统时，Host 中内置的 MCP Client 会被激活。这个 Client 负责与适当的 MCP Server 建立连接。
3. **Server**：在这个例子中，文件系统 MCP Server 会被调用。它负责执行实际的文件扫描操作，访问你的桌面目录，并返回找到的文档列表。

其中 Host 用于处理核心的语义与交互需求。Client 作为中介与 Server 交互，Server 用于访问各个数据库获得数据，返回给 Host 来生成回答。

流程：你的问题 → Claude Desktop(Host) → Claude 模型 → 需要文件信息 → MCP Client 连接 → 文件系统 MCP Server → 执行操作 → 返回结果 → Claude 生成回答 → 显示在 Claude Desktop 上。

这种架构设计使得LLM可以在不同场景下灵活调用各种工具和数据源，而开发者只需专注于开发对应的 MCP Server，无需关心 Host 和 Client 的实现细节。**MCP Server** ：一个为 MCP 客户端提供上下文信息的程序，可以运行在远程的托管服务器或者本地。



### 传输层：stdio、Streamable HTTP，以及为什么很多资料还在写 SSE

了解了基本组件，这里需要简单介绍一下这些组件之间的通信时如何实现的，以及通信机制本身发生了什么变化。当前规范中的标准传输主要是 `stdio` 和 `Streamable HTTP`。至于 `SSE` 的说法则是官方文档修改后留下的历史遗留问题，很多网上的资料引用的是旧版本的原始信息，没有考虑MCP本身的更新。见本节的引用部分。

| 维度 | `stdio` | `Streamable HTTP` |
| --- | --- | --- |
| 启动方式 | Client 启动 Server 子进程 | Server 独立运行并暴露 HTTP endpoint |
| 部署位置 | 通常在本机 | 可以在本机，也更常见于远程服务 |
| 消息通道 | JSON-RPC 走 `stdin/stdout`，日志走 `stderr` | JSON-RPC 走 HTTP `POST/GET`，必要时可用 `SSE` 流式返回 |
| 典型场景 | 本地 IDE、桌面端工具、文件系统和脚本工具 | 云端工具、团队共享服务、需要认证的 connector |
| 认证与安全 | 依赖本机权限、启动命令、环境变量和客户端配置 | 依赖 HTTP 鉴权、授权流程和网络边界 |
| 远程与多客户端 | 不适合直接作为远程共享服务 | 更适合作为远程服务被多个客户端访问 |

`stdio` 可以理解成最本地化的一种连接方式：Client 像父进程一样启动 MCP Server 子进程，然后把 JSON-RPC 请求写进 Server 的 `stdin`，再从 Server 的 `stdout` 读取 JSON-RPC 响应。因为 `stdout` 已经被协议占用，所以调试日志应该写到 `stderr`，否则很容易把协议数据污染掉。（不要在一个本地运行的MCP Server以及Client代码中随便写`print`或者其他利用`stdio`的函数）

`Streamable HTTP` 则更像我们熟悉的远程服务：MCP Server 独立运行，Client 连接一个 MCP endpoint，通过 HTTP `POST/GET` 发送和接收协议消息。服务器可以在需要时使用 `SSE` 持续推送消息，所以 `SSE` 在今天不是与 `stdio` 并列的主名称，而是 HTTP 传输里的一个流式机制。

> **版本说明**：根据官方规范，[2024-11-05 版本](https://modelcontextprotocol.io/specification/2024-11-05/basic/transports)里的远程传输叫 `HTTP with SSE`，从 [2025-03-26 版本](https://modelcontextprotocol.io/specification/2025-03-26/changelog)开始被 `Streamable HTTP` 取代；最新的 [2025-11-25 传输规范](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)则以 `stdio` 和 `Streamable HTTP` 作为标准传输。因此读到旧文章里写 `SSE/HTTP+SSE` 并不一定是错，只是版本口径更早。



### 在任务开始之前的通信

MCP 从生命周期管理开始，客户端发送 `initialize` 请求以建立连接并协商支持的功能。初始化成功后，客户端会发送通知表明其已准备就绪。初始化过程中，AI 应用的 MCP 客户端管理器会建立与已配置服务器的连接，并存储其功能以供后续使用。应用会利用这些信息来确定哪些服务器可以提供特定类型的功能（工具、资源、提示），以及它们是否支持实时更新。



连接建立后，客户端可以通过发送 `tools/list` 请求来发现可用工具。此请求是 MCP 工具发现机制的基础——它允许客户端在尝试使用工具之前了解服务器上有哪些工具可用。响应中包含一个 `tools` 数组，其中提供了每个可用工具的完整元数据。**这种基于数组的结构允许服务器同时公开多个工具，同时保持不同功能之间的清晰界 限。** 



响应中的每个工具对象都包含几个关键字段：

* **`name`** ：工具在服务器命名空间中的唯一标识符。
* **`title`** ：客户端可以向用户显示的该工具的易于理解的显示名称
* **`description`** ：详细说明该工具的功能以及何时使用该工具。
* **`inputSchema`** ：一个 JSON Schema，用于定义预期的输入参数，支持类型验证，并提供关于必需参数和可选参数的清晰文档。



人工智能应用程序从所有已连接的 MCP 服务器获取可用工具，并将它们整合到一个统一的工具注册表中，供语言模型访问。这使得语言学习模型 (LLM) 能够理解它可以执行哪些操作，并在对话过程中自动生成相应的工具调用。在发现可用工具后，客户端可以使用适当的参数调用它们。


### 模型是如何确定工具的选用的？
基本的结构应该表示为
1. 客户端（Host）将你的问题发送给 Claude。
2. Claude 分析可用的工具，并决定使用哪一个（或多个）。
3. 客户端通过 MCP Server 执行所选的工具。
4. 工具的执行结果被送回给 Claude。
5. Claude 结合执行结果构造最终的 prompt 并生成自然语言的回应。
6. 回应最终展示给用户！

这个调用过程可以分为两个步骤：
1. 由 LLM（Claude）确定使用哪些 MCP Server。
2. 执行对应的 MCP Server 并对执行结果进行重新处理。

整体逻辑参考下图，图源见水印

![alt text](https://pica.zhimg.com/v2-9d3681630ed930a8dc74d3b452c0cc94_1440w.jpg)
#### 工具选择
先理解第一步**模型如何确定该使用哪些工具？** 

通过阅读代码，可以发现模型是通过 prompt 来确定当前有哪些工具。我们通过**将工具的具体使用描述以文本的形式传递给模型**，供模型了解有哪些工具以及结合实时情况进行选择。

即
```python
.. # 省略了无关的代码
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
     tools_description = "\n".join(
         [tool.format_for_llm() for tool in all_tools]
     )
 ​
     # 这里就不简化了，以供参考，实际上就是基于 prompt 和当前所有工具的信息
     # 询问 LLM（Claude） 应该使用哪些工具。
     system_message = (
         "You are a helpful assistant with access to these tools:\n\n"
         f"{tools_description}\n"
         "Choose the appropriate tool based on the user's question. "
         "If no tool is needed, reply directly.\n\n"
         "IMPORTANT: When you need to use a tool, you must ONLY respond with "
         "the exact JSON object format below, nothing else:\n"
         "{\n"
         '    "tool": "tool-name",\n'
         '    "arguments": {\n'
         '        "argument-name": "value"\n'
         "    }\n"
         "}\n\n"
         "After receiving a tool's response:\n"
         "1. Transform the raw data into a natural, conversational response\n"
         "2. Keep responses concise but informative\n"
         "3. Focus on the most relevant information\n"
         "4. Use appropriate context from the user's question\n"
         "5. Avoid simply repeating the raw data\n\n"
         "Please use only the tools that are explicitly defined above."
     )
     messages = [{"role": "system", "content": system_message}]
 ​
     while True:
         # Final... 假设这里已经处理了用户消息输入.
         messages.append({"role": "user", "content": user_input})
 ​
         # 将 system_message 和用户消息输入一起发送给 LLM
         llm_response = self.llm_client.get_response(messages)
 ​
     ... # 后面和确定使用哪些工具无关
     
 ​
 class Tool:
     """Represents a tool with its properties and formatting."""
 ​
     def __init__(
         self, name: str, description: str, input_schema: dict[str, Any]
     ) -> None:
         self.name: str = name
         self.description: str = description
         self.input_schema: dict[str, Any] = input_schema
 ​
     # 把工具的名字 / 工具的用途（description）和工具所需要的参数（args_desc）转化为文本
     def format_for_llm(self) -> str:
         """Format tool information for LLM.
 ​
         Returns:
             A formatted string describing the tool.
         """
         args_desc = []
         if "properties" in self.input_schema:
             for param_name, param_info in self.input_schema["properties"].items():
                 arg_desc = (
                     f"- {param_name}: {param_info.get('description', 'No description')}"
                 )
                 if param_name in self.input_schema.get("required", []):
                     arg_desc += " (required)"
                 args_desc.append(arg_desc)
 ​
         return f"""
 Tool: {self.name}
 Description: {self.description}
 Arguments:
 {chr(10).join(args_desc)}
 """
```

**模型是通过提供所有工具的结构化描述和 few-shot 的 example 来确定该使用哪些工具**。

#### 工具执行与结构反馈
其实工具的执行就比较简单和直接了。承接上一步，我们把 system prompt（指令与工具调用描述）和用户消息一起发送给模型，然后接收模型的回复。当模型分析用户请求后，它会决定是否需要调用工具：
- **无需工具时**：模型直接生成自然语言回复。
- **需要工具时**：模型输出结构化 JSON 格式的工具调用请求。

果回复中包含结构化 JSON 格式的工具调用请求，则客户端会根据这个 json 代码执行对应的工具。如果模型执行了 tool call，则工具执行的结果 `result` 会和 system prompt 和用户消息一起**重新发送**给模型，请求模型生成最终回复。如果 json 代码存在问题或者模型产生了幻觉，我们会 skip 掉无效的调用请求。

```python
... # 省略无关的代码
 async def start(self):
     ... # 上面已经介绍过了，模型如何选择工具
 ​
     while True:
         # 假设这里已经处理了用户消息输入.
         messages.append({"role": "user", "content": user_input})
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
             messages.append({"role": "assistant", "content": llm_response})
             messages.append({"role": "system", "content": result})
 ​
             final_response = self.llm_client.get_response(messages)
             logging.info("\nFinal response: %s", final_response)
             messages.append(
                 {"role": "assistant", "content": final_response}
             )
         # 否则代表没有执行 tool call，则直接将 LLM 的输出返回给用户。
         else:
             messages.append({"role": "assistant", "content": llm_response})  
```

据此：
- 工具文档至关重要 - 模型通过工具描述文本来理解和选择工具，因此精心编写工具的名称、docstring 和参数说明至关重要。
- 由于 MCP 的选择是基于 prompt 的，所以任何模型其实都适配 MCP，只要你能提供对应的工具描述。



**大部分情况下，当使用装饰器 `@mcp.tool()` 来装饰函数时，对应的 `name` 和 `description` 等其实直接源自用户定义函数的函数名以及函数的 `docstring`**，关于参数和返回值的信息一方面可以通过 `docstring` 获取，另一方面可以解析我们在撰写Tool时候的内容。

## MCP Servers

MCP 服务器是通过标准化协议接口向 AI 应用程序公开特定功能的程序。这也是开发者需要接触到的层。服务器通过三个基本组成部分提供功能：

* **Tools 工具** LLM 可以主动调用这些函数，并根据用户请求决定何时使用这些函数。工具可以写入数据库、调用外部 API、修改文件或触发其他逻辑。
* **Resources 资源** 被动数据源 提供上下文信息
* **Prompts 提示** 预先构建的指令模板，告诉模型如何使用特定的工具和资源。



工具是固定模式的接口，LLM可以调用这些接口。MCP 使用 JSON Schema 进行验证。每个工具都执行单一操作，并具有明确定义的输入和输出。工具可能需要在执行前获得用户许可，这有助于确保用户对模型执行的操作保持控制。合法的**Protocol operations** 包括`tools/list`  和 `tools/call` 分别返回工具的描述数组和工具的执行结果。



工具由模型控制，这意味着人工智能模型可以自动发现并调用它们。然而，MCP 通过多种机制强调人工监督。包括用户控制模型的开启和关闭，预先设置和每次批准的工具执行审批。



资源提供结构化的信息访问，人工智能**应用程序**可以检索这些信息并将其作为上下文提供给模型。他也是一种工具，但是这种工具的作用就是提供一些上下文信息。 他同时支持**直接资源** - 指向特定数据的固定 URI 以及 **资源模板** ——带有参数的动态 URI。相关的**Protocol operations** 包括 `resources/list`,`resources/templates/list` ,`resources/read` ,`resources/subscribe`  资源由应用程序驱动.应用程序可以自由选择任何符合自身需求的界面模式来实现资源发现功能。**Resources由Application调用而非模型调用**



提示符提供可重用的模板。它们允许 MCP 服务器作者为领域任务提供参数化提示符，或展示如何最佳地使用 MCP 服务器。合法的**Protocol operations** 包括`prompts/list`  `prompts/get` 提示由用户控制，需要显式调用。**Prompts由用户调用而非模型调用**



整个Servers中，只有tools是提供给LLM使用的工具，Resources和Prompts均用于程序的其他部分，也并非在此需要考虑的重点，我们在后面的介绍中也只会侧重关于自行构建MCP Servers的Tools，并让LLM去使用这些Tools。事实上 **对绝大部分 AI 开发者来说，我们只需要关心 Server 的实现。** 

**MCP Servers 层的作用是实现工具和其他内容的统一暴露，供 Agent 动态发现与调用，而无需在主进程硬编码工具表（需要和Client配合），这就是MCP的核心价值。本地 `stdio` 模式下 Server 常以独立子进程运行，远程 `Streamable HTTP` 模式下 Server 则更像一个独立服务。**

## 将Client连接到Servers

根据前面的叙述我们已经可以确定，整个MCP协议分为了三层，用户直接对话的LLM作为Host，一般会和Host一起存在，在我们安装如Claude Code/ Vscode 等Application的时候将包含支持MCP协议的Client。 而Servers则托管在云端或者本地。**无需关注Host和Client的分离结构，将他们看作一个整体不影响我们理解整个MCP协议**



我们无需关注Client层是如何实现的，MCP协议的作用就是通过这个中介层，将Servers与LLMs的交互逻辑隐藏并固定。开发者只需要关注按照开发规范完成Servers的内容设计，然后托管到云端或者允许用户安装到本地。需要让模型能够识别到MCP Servers，我们只需要在支持MCP的IDE里面进行配置即可：本地 `stdio` server 通常配置启动命令，远程 `Streamable HTTP` server 则通常配置 URL/endpoint 和认证信息。

下面这个 JSON 是本地 `stdio` server 的典型配置方式：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/username/Desktop",
        "/Users/username/Downloads"
      ]
    }
  }
}
```

其中 `filesystem` 是服务器在 Claude Desktop（或者其他Client） 中显示的名称。 `command` 是 Client 要启动的可执行程序。参数中分别是，自动安装 server package，对应 server package 的名称，允许Servers访问的目录。按修改配置文件后重新启动Client，就可以在对应的位置看到服务器的信息，并且模型后面就可以使用服务器中的工具了。

**对于本地 `stdio` server，MCP Server 的配置实际上就是将手动执行的命令细化并且拆分，和手动启用 MCP Server 的运行命令没有本质差别，都是由 Client 拉起进程，然后通过 `stdin/stdout` 交换协议消息。**

现在基本所有客户端都支持我们连接远程工具。相比于本地工具，远程 MCP server 通常通过 `Streamable HTTP` 暴露一个 URL/endpoint，Client 或 Custom Connectors 充当 Claude 与远程 MCP 服务器之间的桥梁。为了保证访问合法，大部分远程 Servers 都需要身份验证或授权，其具体方式由对应 Client 产品和远程服务决定，按照提示操作即可。连接成功后，远程服务器的资源和提示信息将出现在你的 Claude 对话中。



**不同Client配置MCP Servers的方式并不相同，有的平台的所有Servers都在一个JSON中进行控制，部分Client则实现了更加独立的配置逻辑，区分了本地 `stdio` Servers 与远程 `Streamable HTTP` Servers**

## 基于Python SDK 实现MCP Server

 **对绝大部分 AI 开发者来说，我们只需要关心 Server 的实现。** 这需要我们一定程度上理解MCP Server的工作原理以保证代码的可扩展性。

**MCP 允许在不修改代理代码的情况下进行功能更新** 这可能是是他最大的作用，在使用传统Function Calling的模式下，我们需要求修改client处的代码，如果这个client在本地实现，则需要进行提示词修正。如果在云端实现，我们则需要对应的开发者权限，但是MCP Server只需要简单的配置就可以将对应的提示词插入的Client中，这是他的价值所在。 **对于复杂的开发，现在的MCP抽象层逻辑可能会成为提升工具检索性能的绊脚石**

下面的代码示例将作为我们开发MCP Server的例子，为了让MCP的Dev更加容易，`mcp` python package被引入，其中的FastMCP将帮助我们轻松的开发MCP Server. 这个package中还包含终端工具，Client的相关内容，他们在我们开发Server的时候用不到。



```python
# 导入开发MCP以及工具本身需要的Packages
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# 创建 MCP Server 也就是一个mcp对象，此时他还是空的，并给了这个Server一个名字
mcp = FastMCP("桌面 TXT 文件统计器")

#使用@mcp.tool() (装饰器)修饰了一个普通的Python函数，这样就从python函数到了一个MCP tool
#Python装饰器是一个非常强大的工具，不过我们再这里不再强调他
#为函数增加了输出类型提示int，这可以被后面的MCP SDK解析
#使用了文档字符串 撰写了doc 这个doc也会被MCP SDK解析 位于模块、类、方法或函数的第一个这样的注释为doc
#代码内部就是普通函数逻辑，很简单
@mcp.tool()
def count_desktop_txt_files() -> int:
    """Count the number of .txt files on the desktop."""
    # Get the desktop path
    username = os.getenv("USER") or os.getenv("USERNAME")
    desktop_path = Path(f"/Users/{username}/Desktop")

    # Count .txt files
    txt_files = list(desktop_path.glob("*.txt"))
    return len(txt_files)

#装饰了另一个tool，一个Server里面可以拥有多个Tool很合理
@mcp.tool()
def list_desktop_txt_files() -> str:
    """Get a list of all .txt filenames on the desktop."""
    # Get the desktop path
    username = os.getenv("USER") or os.getenv("USERNAME")
    desktop_path = Path(f"/Users/{username}/Desktop")

    # Get all .txt files
    txt_files = list(desktop_path.glob("*.txt"))

    # Return the filenames
    if not txt_files:
        return "No .txt files found on desktop."

    # Format the list of filenames
    file_list = "\n".join([f"- {file.name}" for file in txt_files])
    return f"Found {len(txt_files)} .txt files on desktop:\n{file_list}"
#mcp.run(): 这是服务器启动指令，在本地 stdio 示例中启动后会等待来自标准流的协议请求
if __name__ == "__main__":
    # Initialize and run the server
    mcp.run()
```




##  基于Python SDK 实现 MCP Client

### Client 做什么
下面我们来研究实现Client , 对于大部分工具的开发者来说 , Client 是交给 Agent Dev 来实现的 , 因此用户只需要关系如何将MCP Server 连接到 Client [[MCP（Model Context Protocol）#将Client连接到Servers]] 而不用关心 Client 如何实现.

但如果我们希望实现一个 Client 的高层封装 , 将 MCP 嵌入到一个 Agent Framework（或者自己实现的Agent） 里面 , 就需要了解对应的 SDK 中 Client 是如何实现的了. 

在这里我们首先可以开始考虑区分Client层和Host层，之前我们一直把他们看在一起，IDE，Claude等成品的工具同时负责了Client层和Host层，但是当我们自己尝试开发的时候，**核心Agent是Host层，Client层是嵌入在Agent代码中的用于工具发现和执行的代码**，Client本身不事先任何Agent逻辑而是作为Agent的组成部分。**Client层 是 Host 里的一层工具调用适配器，负责连接 MCP Server、列工具、调工具。**

**Client只干两件事，获取封装在Server的资源，整理为文本传输给Host层的LLM Agent；根据Agent的决策，实现工具调用和资源使用。** SDK内部Client相关的代码就是为了一键完成这些工作。这里我们就能理解多次工具调用具体是什么了，MCP Client只是工具调用的工具，具体怎么调用，调用多少工具都是Host层的代码决定的。

Host + Client + Server 实现一件事 **Agent 只负责做决策，MCP 负责提供能力。**

因此，**MCP Client不是 Agent；它是给 Agent 用的协议/总线层。** MCP（Model Context Protocol）本身不做推理、规划、记忆、循环控制这些Agent的本体能力，他们协同起来为了实现Host/Agent 通过一个标准接口去发现/获取上下文/调用外部能力，把工具与数据从 Host 代码里解耦出来。

一个 Client 就是一个自主决策的 Agent , 符合 React 的基本结构 , 存在内部自己的循环和终止循环功能.  如果我们有条件的话可以直接使用那些开发的极为完善的现成的 Client , 比如直接利用 Claude Code 的自主决策能力,将其的功能嵌入到我们的 Framework 中。这就是Claude Code SDK。

### Client与Host的简单例子

下面是一个Client与Host的简单例子，用于参考Client的实现与Host对其的使用。

这个例子使用的是本地 `stdio` transport，所以 Client 会启动 Python 子进程，并通过标准流和它建立 `ClientSession`。如果换成远程 `Streamable HTTP`，Host 和 Client 的职责仍然类似，但底层连接对象会从 `stdin/stdout` 换成 HTTP endpoint。

```python
import asyncio
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client, get_default_environment

def build_env() -> dict:
    """
    构建传递给 MCP Server 的环境变量。
    
    MCP Server 运行在一个独立的子进程中，因此需要显式传递必要的环境变量。
    本函数首先获取 SDK 提供的默认安全环境配置，然后将我们需要增强的
    数据路径变量 'ENHANCED_DATA_PATH' 注入其中。
    """
    # 1. 获取默认环境：包含 PATH 等基础变量，确保 Python 能正常运行
    env = get_default_environment()
    
    # 2. 注入自定义变量：让 Server 能够通过环境变量获取配置信息
    enhanced = os.environ.get("ENHANCED_DATA_PATH")
    if enhanced:
        env["ENHANCED_DATA_PATH"] = enhanced
    return env

def server_params(server_py: str = "utils/mcp_server.py") -> StdioServerParameters:
    """
    构造 Server 的启动参数 (StdioServerParameters)。
    
    这里指定了如何启动 MCP Server：
    - command: 使用 "python" 命令
    - args: 传递脚本路径作为参数
    - env: 使用 build_env() 构建的环境变量
    """
    # 使用绝对路径，避免因 cwd (当前工作目录) 不同导致找不到文件
    server_py = os.path.abspath(server_py)
    return StdioServerParameters(command="python", args=[server_py], env=build_env())

def parse_result(result):
    """
    解析 MCP Protocol 的返回结果 `CallToolResult`。
    
    MCP 的返回结果结构可能包含 TextContent, ImageContent 或 EmbeddedResource。
    本函数的目的是将其简化为 Host 易处理的字典或数据结构。
    """
    # 结果的主要内容都在 content 列表字段中
    content = getattr(result, "content", None)
    if content:
        # 策略 1: 优先提取结构化数据 (EmbeddedResource 或类似 data 字段)
        for item in content:
            data = getattr(item, "data", None)
            if data is not None:
                return data
        
        # 策略 2: 提取文本内容，并尝试解析为 JSON
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except Exception:
                    # 如果不是 JSON，则直接返回原始文本
                    return {"raw_text": text}
    
    # 兜底：如果无法解析，返回原始对象的字典包装
    return {"result": result}

async def list_tools(server_py: str = "utils/mcp_server.py"):
    """
    Client 核心功能：列出 Server 提供的所有工具。
    
    步骤：
    1. stdio_client: 启动子进程，建立 stdio 管道。
    2. ClientSession: 在管道上建立 MCP 协议会话。
    3. initialize: 执行握手协议。
    4. list_tools: 发送 tools/list 请求。
    """
    async with stdio_client(server_params(server_py)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            # 提取关键元数据 (name, description, schema) 返回给 Host 用于决策
            return [
                {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
                for t in resp.tools
            ]

async def call_tool(tool_name: str, arguments: dict | None = None, server_py: str = "utils/mcp_server.py"):
    """
    Client 核心功能：调用指定工具。
    """
    arguments = arguments or {}
    async with stdio_client(server_params(server_py)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return parse_result(result)

# --- Host Agent 使用示例 (Simulated) ---

async def host_agent_demo():
    """
    模拟 Host (Agent) 使用 Client 进行工具发现和调用的过程。
    这里的 Host 扮演“决策者”的角色，而 Client 扮演“执行者”。
    """
    print("=== Host Agent Started ===")
    
    # 1. 发现能力 (Tool Discovery)
    # Host 询问 Client：目前有哪些工具可用？
    print("\n[Host] Discovering tools...")
    # 假设 utils/mcp_server.py 是我们编写好的 Server 脚本
    tools = await list_tools("utils/mcp_server.py")
    
    print(f"[Host] Found {len(tools)} tools:")
    for t in tools:
        print(f"  - Name: {t['name']}") 
        print(f"    Desc: {t['description']}")
    
    if not tools:
        print("[Host] No tools found. Exiting.")
        return

    # 2. 模拟 LLM 决策过程
    # 假设 LLM 根据 Prompt 和工具描述，决定调用 'list_desktop_txt_files'
    # 注意：这里的逻辑通常由 LLM 完成
    target_tool = tools[0]["name"]  # 简单起见，直接取第一个
    print(f"\n[Host] DECISION: I will use the tool '{target_tool}' to gather information.")
    
    # 构造通过 prompt 分析出的参数 (此处为硬编码示例)
    args = {} 
    
    # 3. 执行工具 (Tool Execution)
    print(f"[Host] Requesting Client to execute '{target_tool}'...")
    result = await call_tool(target_tool, args, "utils/mcp_server.py")
    
    # 4. 获取结果
    print(f"[Host] Execution Result received:")
    # 结果可能是列表、字典或文本，这里做简单的打印
    print(result)
    
    print("\n=== Host Agent Finished ===")

```

### 一次完整的本地 stdio 协作流程
一次 `call_tool()` 的端到端协作流程（Agent ↔ MCP Client ↔ MCP Server）。这里讨论的仍然是本地 `stdio` transport。

1. **Agent 决策**
   - `DecisionToolsNode` 选择要执行的工具 → 交给 `ExecuteToolsNode`

2. **Host 调用 MCP Client**
   - `ExecuteToolsNode` 调用：
     - `call_tool("utils/mcp_server", tool_name, {})`

3. **MCP Client 启动 Server 子进程**
   - 构造子进程启动参数：
     - `StdioServerParameters(command="python", args=[server_script_path], env=...)`

4. **MCP SDK 建立本地 stdio 通道（IPC）**
   - `stdio_client(server_params)` 创建主进程与子进程的 stdin/stdout 通信通道

5. **MCP SDK 发起 RPC 调用**
   - `session.initialize()` 完成握手
   - `session.call_tool(tool_name, arguments)` 发起工具调用

6. **MCP Server 执行工具**
   - Server 端对应的 `@mcp.tool()` 函数被触发
   - 内部读取/加载数据后调用 `analysis_tools` 完成统计或图表生成

7. **结果返回给 Client**
   - Server 将结果通过 MCP 协议返回
   - Client 解析结果并返回给 `ExecuteToolsNode`

8. **写回共享状态（shared）**
   - `ExecuteToolsNode` 将图表/表格等产物写入：
     - `shared["stage2_results"]`





## MCP Inspector

### About Inspector

MCP Inspector 是一个用于测试和调试 MCP Server 的可视化交互工具。你可以把它想象成是一个“网页版的 Claude Desktop”或者“API 调试工具（类似 Postman）”，专门用来检查你写的 MCP Server 是否工作正常。



Inspector 本质上就是一个**标准的 MCP Client 实现**，只不过它的目的不是为了“对话”，而是为了“透视”。

- **它模拟 Claude**：它发送和 Claude 一模一样的 JSON-RPC 请求。
- **它不仅是测试工具**：它是协议一致性的校验器。如果你的 Server 在 Inspector 里能完美运行（Schema 显示正确、工具调用成功、资源读取无误），那么理论上它在 Claude Desktop、Cursor 或任何其他 MCP Client 中都能运行。



你不需要全局安装它，直接使用 npx 运行即可,如下

```bash
npx @modelcontextprotocol/inspector <你的启动命令>
```



对于使用Python开发的MCP Server

```bash
npx @modelcontextprotocol/inspector uv run main.py
# 或者
npx @modelcontextprotocol/inspector python main.py
```



对于Node开发的MCP Server

```bash
npx @modelcontextprotocol/inspector node build/index.js
```



如果需要环境变量设置

```bash
# 在命令前加 env 变量，或者直接在 npx 后接命令
KEY=value npx @modelcontextprotocol/inspector python main.py
```



运行成功后，终端会显示一个本地网址（通常是 http://localhost:5173），浏览器会自动打开这个页面。这就是MCP Inspector。 **在使用MCP Inspector命令的时候，需要严格注意目录的问题，只有启动命令本身可以执行，才能够使用MCP Inspector正确解析，否则则无法连接**



在MCP Inspector创建的可视化界面中，我们可以通过前端UI模仿Client对Server发出命令，并且UI的左侧记录了启动相关命令，History记录了Client对Server发出的命令和Server的回复。Server Notifications则记录日志信息。主界面则是我们在MCP Server中介绍到的MCP相关功能，此时不需要LLM来调用Tool，用户来使用Prompt或者App使用Resource，全部的操作简化为了我们的测试界面的点击来调用Server的功能。



当Inspector审查正确后，就可以放心的去 Claude Desktop 配置文件里添加配置，或者托管到在线的Server平台了。

### From Inspector to Client

虽然 Inspector 和 Claude Desktop/Code 都是 MCP Client，但它们的**运行逻辑**和**配置方式**有本质的区别：

- **Inspector**：是**瞬时的**、**命令行的**。你告诉它“现在立刻运行这行代码”，它就跑起来，关闭网页进程就结束。
- **Claude Client**：是**持久的**、**配置文件的**。你需要把运行指令写进一个 JSON 文件里，Claude 启动时会去读取并后台静默运行。



在 Inspector 中，你通常是一行写完所有内容：

```bash
# 示例：一个需要 API Key 的 Python Server
MY_API_KEY=12345 npx @modelcontextprotocol/inspector python main.py --verbose
```



在 claude_desktop_config.json 文件中，上面的一行命令必须拆解为以下结构：

```json
{
  "mcpServers": {
    "my-server-name": {
      "command": "python",
      "args": [
        "main.py",
        "--verbose"
      ],
      "env": {
        "MY_API_KEY": "12345"
      }
    }
  }
}
```

1. **Command (主命令)**：
   - **Inspector**: python 或 node 或 uv。
   - **Claude**: 必须是 JSON 中的 "command" 字段。
   - **注意**：必须是可执行程序的名称或绝对路径。如果你在 Inspector 用 npx，在 JSON 里 command 通常也是 "npx"（如果是 Python，则需要在Command写明白具体的可执行程序的绝对路径）。
2. **Args (参数列表)**：
   * **Inspector**: 用空格分隔的字符串，如 main.py --verbose。
   * **Claude**: 必须是**字符串数组** ["main.py", "--verbose"]。
   * **重要差异**：不能把 "python main.py" 写在一个字符串里！必须把文件名和参数拆开。
3. **Env (环境变量)**：
   1. **Inspector**: 写在命令最前面，如 KEY=value。
   2. **Claude**: 必须写在 "env" 对象里。Claude **不会**自动继承你终端里的环境变量，所以所有需要的 Key 必须在这里显式定义。



**不同的平台对于具体JSON配置文件的写法可能是不同的。对于本地 `stdio` Server，核心仍然是拆分启动命令、参数和环境变量；对于远程 `Streamable HTTP` Server，配置重点会变成 endpoint、认证和授权。**

### Inspector 与 MCP Server Code

我们用简单的例子就可以解释明白 Inspector 是如何工作的，这样我们就可以从中理解MCP Client的工作方式，从Client 到 Host 就是将他们的通信流程进一步的封装和隐藏。



这一节演示的是本地 `stdio` Server 的调试方式。此时 **Inspector Proxy** 会像一个“父进程”一样，启动你的代码（子进程），然后通过标准输入/输出通信并交互。如果是远程 MCP service，Client 或 Inspector 连接的是 HTTP endpoint，而不是拉起本地子进程。

- **写入 (stdin)**: Inspector 把 JSON-RPC 请求（比如“请列出所有工具”）发送给你的代码。
- **读取 (stdout)**: 你的代码把处理结果（JSON 格式）打印到控制台，Inspector 截获这些输出并显示在网页上。



因为 stdout 被用来传输协议数据，**绝对不要**在你的代码里使用 print() (Python) 或 console.log() (Node) 来打印调试信息！这会破坏 JSON 格式，导致 Inspector 报错。**调试信息请打印到 stderr，如下所示**

```python
import sys
# 正确的调试方式：写入 stderr
print("Debug: Function called with a=10", file=sys.stderr) 

# 或者使用 logging 模块（配置为写 stderr）
logger.info("Processing request...")
```



#### Tools (工具) 的映射

假设我们有很简单的Python Code定义了Server 和 Tool 

```python
@mcp.tool()
async def calculate_sum(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```



**ListTools**: Inspector 启动后，发送 tools/list 请求 ，然后 MCP 的SDK基于Pydantic 或类型提示，自动生成了如下 JSON Schema：

```json
{
  "name": "calculate_sum",
  "description": "Add two numbers.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "a": { "type": "integer" },
      "b": { "type": "integer" }
    },
    "required": ["a", "b"]
  }
}
```



Inspector 读取这个 Schema：

- 看到 name -> 在左侧列表显示 "calculate_sum"。
- 看到 properties -> 在右侧生成两个输入框，标签分别为 "a" 和 "b"，类型限制为数字。



点击Run Tool后

1. Inspector 发送 tools/call 请求：{"name": "calculate_sum", "arguments": {"a": 10, "b": 20}}。
2. 你的 calculate_sum 函数被调用。
3. 返回值 30 通过 stdout 发回，Inspector 显示在 "Result" 区域。



#### Resources (资源) 的映射

**Python 代码：**

```python
@mcp.resource("file://logs/{name}")
def read_log(name: str) -> str:
    return f"Log content for {name}..."
```



**Inspector 的处理过程：**

1. **ListResources**: 发送 resources/list。
2. **UI 显示**: Inspector 在 Resources 面板列出所有可用的资源 URI 模板（如 file://logs/{name}）。
3. **交互**: 你点击列表中的资源，Inspector 会尝试读取（发送 resources/read），并在预览窗口展示返回的文本或二进制内容。



####  Prompts (提示词) 的映射

**Python 代码：**

```python
@mcp.prompt()
def review_code(code: str) -> list[Message]:
    return [UserMessage(content=f"Review this code: {code}")]
```



**Inspector 的处理过程：**

1. **ListPrompts**: Inspector 获取提示词列表。
2. **参数填充**: Inspector 识别出 review_code 需要参数 code，并在 UI 上生成一个文本框供你输入代码片段。
3. **预览**: 点击运行后，Inspector 不会执行任何 AI 调用，而是展示**最终生成的 Prompt 结构**。这让你检查你的模板逻辑是否正确。

## 结语

MCP 是一个很好的工程实践，他也有着非常完善的文档，开发约束以及用户指南。但是基于JSON Schema的Tool Calling终究是一个妥协形态，因为模型不够强大，不能为了新需求直接去开发一个完美工具，因此我们选择了预先测试并封装。

MCP 不是用户友好的工具，它太复杂了，充满了开发者的精力，对于一个新手来说去理解他的机制并去开发不是一个容易的事情。而AI Agent本身就在飞速发展，就在此时形成一个完善而固定的工程实践不是一个好事，这也是为什么Agent Skills最终在用户侧的热度明显更高，哪怕Skills的工作MCP基本都能做到（可能需要进行一些小修小补），Skills作为完全不完善的框架依旧快要将MCP取代了。
