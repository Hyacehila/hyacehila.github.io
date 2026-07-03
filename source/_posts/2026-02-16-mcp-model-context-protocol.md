---
title: MCP (Model Context Protocol)
title_en: "MCP (Model Context Protocol)"
date: 2026-02-16 11:30:00 +0800
categories: ["AI & Agents", "Agent Infrastructure"]
tags: ["MCP", "Protocols", "Tool Use", "Context Engineering", "Tutorial"]
author: Hyacehila
excerpt: 介绍 MCP 的 Host、Client、Server 分层，stdio 与 Streamable HTTP 传输，以及用 Python SDK 编写和调试 MCP Server/Client 的基本流程。
excerpt_en: "An introduction to MCP's Host, Client, and Server layers, stdio and Streamable HTTP transports, and the workflow for building and debugging MCP servers and clients."
permalink: '/blog/2026/02/16/mcp-model-context-protocol/'
---

MCP 起源于 2024 年 11 月 25 日  [Anthropic 发布的文章](https://www.anthropic.com/news/model-context-protocol) 

本文参考了一篇 [知乎文章](https://zhuanlan.zhihu.com/p/29001189476)，部分介绍与代码源自此文。

## MCP 简介
MCP （Model Context Protocol，模型上下文协议）定义了应用程序和 AI 模型之间交换上下文信息的方式。它给工具、数据源和提示模板提供了一层协议接口，让开发者能够**以一致的方式将各种数据源、工具和功能连接到 AI 模型**。

MCP 的目标是减少工具接入时的重复适配。开发者不必为每个客户端重新写一套工具注册和调用逻辑；只要客户端支持 MCP，同一个 Server 就可以被不同宿主应用发现和调用。

MCP 与视觉派代表两种 Agent 思路：前者依赖开发者提供接口适配，并统一这类接口；后者让 LLM 通过视觉识别看到接近人类看到的内容，再模拟人类操作。

接下来先看 MCP 为什么会出现。早期开发者主要靠 prompt 把场景信息塞给模型，但工具、文件、数据库和业务状态越来越多之后，单靠手写 prompt 很难维护。

在 MCP 诞生之前，我们通常手动粘贴场景信息来补充 prompt。问题变多、变细之后，**手工**把信息引入 prompt 会越来越难，也很难保持稳定的格式。

为了解决手工 prompt 的局限，许多 LLM 平台引入了 `function call` 功能。模型可以在需要时调用预定义函数来获取数据或执行操作，自动化程度和性能表现都有提升。

**function call 平台依赖性强**，不同 LLM 平台的 function call API 实现差异较大，适配成本随之增加；切换平台时，往往要重写大部分代码。

**数据与工具本身是客观存在的**，我们希望把它们连接到模型时更顺畅、更统一。Anthropic 基于这一点设计了 MCP，让 LLM 更容易获取数据或调用工具。MCP 的优势包括：
- **生态** - MCP 提供不少现成插件，你的 AI 可以直接使用。
- **统一性** - 不限制于特定的 AI 模型，任何支持 MCP 的模型都可以切换。
- **数据安全** - 你的敏感数据留在自己的电脑上，不必全部上传。（因为我们可以自行设计接口确定传输哪些数据）

关于MCP的详细使用方法，参考 [文档](https://modelcontextprotocol.io/introduction)  介绍了相关 SDK 与一些例子。

**无论是 Function Calling 还是 MCP，模型本身的 Toolcalling 能力仍旧源自最初的 JSON Schema Output；我们只是在其基础上附加功能，并没有改变模型本身。**

## MCP Architecture 
### 基本组件
MCP 由三个组件构成：Host、Client 和 Server。可以通过一个实际场景理解它们如何协同工作：

假设你正在使用 Claude Desktop (Host) 询问："我桌面上有哪些文档？"

1. **Host**：Claude Desktop 作为 Host，负责接收你的提问并与 Claude 模型交互。
2. **Client**：当 Claude 模型决定需要访问你的文件系统时，Host 中内置的 MCP Client 会被激活。这个 Client 负责与适当的 MCP Server 建立连接。
3. **Server**：在这个例子中，文件系统 MCP Server 会被调用。它负责执行实际的文件扫描操作，访问你的桌面目录，并返回找到的文档列表。

其中 Host 处理语义与交互需求，Client 作为中介与 Server 交互，Server 访问数据库取得数据，再返回给 Host 生成回答。

流程：你的问题 → Claude Desktop(Host) → Claude 模型 → 需要文件信息 → MCP Client 连接 → 文件系统 MCP Server → 执行操作 → 返回结果 → Claude 生成回答 → 显示在 Claude Desktop 上。

这种架构让 LLM 可以在不同场景下调用各种工具和数据源；开发者只需开发对应的 MCP Server，不必关心 Host 和 Client 的实现细节。**MCP Server** ：一个为 MCP 客户端提供上下文信息的程序，可以运行在远程的托管服务器或者本地。



### 传输层：stdio、Streamable HTTP，以及为什么很多资料还在写 SSE

了解基本组件之后，还需要看清这些组件如何通信，以及通信机制发生了哪些变化。当前规范中的标准传输主要是 `stdio` 和 `Streamable HTTP`。至于 `SSE` 的说法，是官方文档修改后留下的历史遗留问题：不少网上资料引用的是旧版本信息，没有纳入 MCP 后续更新。见本节的引用部分。

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



连接建立后，客户端会发送 `tools/list` 请求，获取 Server 暴露的工具列表。响应中的 `tools` 数组包含每个工具的 `name`、`description`、`inputSchema` 等信息。**数组结构让一个 Server 可以同时公开多个工具，也让客户端能逐个展示和调用。**



响应中的每个工具对象都包含几个关键字段：

* **`name`** ：工具在服务器命名空间中的唯一标识符。
* **`title`** ：客户端可以向用户显示的该工具的易于理解的显示名称
* **`description`** ：详细说明该工具的功能以及何时使用该工具。
* **`inputSchema`** ：一个 JSON Schema，用于定义预期的输入参数，支持类型验证，并提供关于必需参数和可选参数的清晰文档。



Host 或 Client 会把已连接 MCP Server 的工具汇总成工具注册表，再把工具描述交给模型参考。模型根据用户请求和工具描述决定是否调用；真正的执行仍由客户端把参数发回对应 Server。


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
工具执行这一步比较直接。承接上一步，我们把 system prompt（指令与工具调用描述）和用户消息一起发送给模型，然后接收模型回复。模型分析用户请求后，会决定是否需要调用工具：
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
- 工具文档会直接影响模型选工具的质量。名称、docstring 和参数说明要写得具体，避免只写一个笼统动词。
- MCP 的工具选择依然依赖模型理解文本描述。模型越能理解任务和工具边界，调用效果越稳定。



**大部分情况下，`@mcp.tool()` 会直接把函数名和 `docstring` 转成工具的 `name` 与 `description`。** 参数和返回值信息也会来自类型标注、`docstring` 或 SDK 对函数签名的解析。所以写 MCP Tool 时，函数名和说明文字不是注释，它们就是模型看到的接口文档。

## MCP Servers

MCP 服务器是通过标准化协议接口向 AI 应用程序公开特定功能的程序。这也是开发者需要接触到的层。服务器通过三个基本组成部分提供功能：

* **Tools 工具** LLM 可以主动调用这些函数，并根据用户请求决定何时使用这些函数。工具可以写入数据库、调用外部 API、修改文件或触发其他逻辑。
* **Resources 资源** 被动数据源 提供上下文信息
* **Prompts 提示** 预先构建的指令模板，告诉模型如何使用特定的工具和资源。



工具是固定模式的接口，LLM可以调用这些接口。MCP 使用 JSON Schema 进行验证。每个工具都执行单一操作，并具有明确定义的输入和输出。工具可能需要在执行前获得用户许可，这有助于确保用户对模型执行的操作保持控制。合法的**Protocol operations** 包括`tools/list`  和 `tools/call` 分别返回工具的描述数组和工具的执行结果。



工具由模型控制，人工智能模型可以自动发现并调用它们。不过，MCP 也通过多种机制保留人工监督，包括用户控制模型的开启和关闭、预先设置，以及每次工具执行审批。



资源提供结构化的信息访问，AI **应用程序**可以读取这些信息，再把它们作为上下文交给模型。它和工具的区别在于：资源主要负责提供上下文，不负责让模型主动执行动作。资源支持**直接资源**，也就是指向固定数据的 URI；也支持**资源模板**，也就是带参数的动态 URI。相关的 **Protocol operations** 包括 `resources/list`、`resources/templates/list`、`resources/read`、`resources/subscribe`。资源发现和读取由应用程序驱动，界面形式由具体客户端决定。**Resources 由 Application 调用，而非模型直接调用。**



提示符提供可重用的模板。它们允许 MCP 服务器作者为领域任务提供参数化提示符，或展示如何最佳地使用 MCP 服务器。合法的**Protocol operations** 包括`prompts/list`  `prompts/get` 提示由用户控制，需要显式调用。**Prompts由用户调用而非模型调用**



整个Servers中，只有tools是提供给LLM使用的工具，Resources和Prompts均用于程序的其他部分，也并非在此需要考虑的重点，我们在后面的介绍中也只会侧重关于自行构建MCP Servers的Tools，并让LLM去使用这些Tools。事实上 **对绝大部分 AI 开发者来说，我们只需要关心 Server 的实现。** 

**MCP Servers 层负责统一暴露工具和其他内容，供 Agent 动态发现与调用，而无需在主进程硬编码工具表（需要和 Client 配合）。这是 MCP 的主要价值。本地 `stdio` 模式下 Server 常以独立子进程运行，远程 `Streamable HTTP` 模式下 Server 则更像一个独立服务。**

## 将Client连接到Servers

前面已经能看出，MCP 运行时大致分为 Host、Client、Server 三层。用户直接使用的是 Claude Desktop、Claude Code、VS Code 这类 Host/Application；其中通常内置支持 MCP 的 Client；Server 则运行在本地或云端。**如果只是使用 MCP，把 Host 和 Client 暂时看成一个整体，不会影响理解。**



只做 MCP Server 时，可以先不关心 Client 内部怎么实现。MCP 的作用是把 Server 与 LLM 之间的交互固定成一套协议。开发者主要负责实现 Server，并把它部署到云端或交给用户在本地安装。要让模型识别 MCP Server，只需要在支持 MCP 的客户端里配置它：本地 `stdio` server 通常配置启动命令，远程 `Streamable HTTP` server 通常配置 URL/endpoint 和认证信息。

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

其中 `filesystem` 是服务器在 Claude Desktop 或其他 Client 中显示的名称。`command` 是 Client 要启动的可执行程序。`args` 里依次写入自动安装参数、server package 名称，以及允许 Server 访问的目录。修改配置后重启 Client，就能看到这个 Server，模型后续也可以使用其中的工具。

**对于本地 `stdio` server，MCP Server 的配置实际上就是将手动执行的命令细化并且拆分，和手动启用 MCP Server 的运行命令没有本质差别，都是由 Client 拉起进程，然后通过 `stdin/stdout` 交换协议消息。**

现在基本所有客户端都支持我们连接远程工具。相比于本地工具，远程 MCP server 通常通过 `Streamable HTTP` 暴露一个 URL/endpoint，Client 或 Custom Connectors 充当 Claude 与远程 MCP 服务器之间的桥梁。为了保证访问合法，大部分远程 Servers 都需要身份验证或授权，其具体方式由对应 Client 产品和远程服务决定，按照提示操作即可。连接成功后，远程服务器的资源和提示信息将出现在你的 Claude 对话中。



**不同Client配置MCP Servers的方式并不相同，有的平台的所有Servers都在一个JSON中进行控制，部分Client则实现了更加独立的配置逻辑，区分了本地 `stdio` Servers 与远程 `Streamable HTTP` Servers**

## 基于Python SDK 实现MCP Server

 **对绝大部分 AI 开发者来说，我们只需要关心 Server 的实现。** 这需要我们一定程度上理解MCP Server的工作原理以保证代码的可扩展性。

**MCP 的价值在于把能力更新从 Agent 主代码里拆出来。** 传统 Function Calling 往往需要改 client 代码或提示词；MCP Server 可以通过配置接入，让支持 MCP 的客户端重新发现工具。代价是工具检索和能力暴露会多一层抽象。复杂项目仍然需要自己处理工具命名、权限、检索质量和上下文预算。

下面用 Python SDK 写一个 MCP Server 示例。`mcp` package 里的 FastMCP 会帮我们处理协议细节。这个 package 也包含终端工具和 Client 相关代码，但写 Server 时暂时用不到。



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
下面看 Client 的实现。对大部分工具开发者来说，Client 由 Claude Desktop、Cursor、Claude Code 这类宿主产品提供；你只需要知道如何把 MCP Server 配上去，而不用自己实现 Client。

但如果要把 MCP 嵌进自研 Agent Framework，就需要了解 SDK 里的 Client 做了哪些工作。

自研框架时必须区分 Host 和 Client。Host 是承载 Agent 逻辑的那层：维护对话、决定何时调用工具、处理循环和终止条件。MCP Client 是 Host 内部的协议适配器，负责连接 Server、列工具、调工具和读取资源。**核心 Agent 在 Host 层；Client 层只负责工具发现和协议通信。**

**Client 只做两类事：获取 Server 暴露的工具、资源和提示信息；根据 Host/Agent 的决策执行工具调用或资源读取。** SDK 里的 Client 代码就是为了封装这些协议动作。多次工具调用的循环不属于 MCP Client 本身，具体调用多少次、何时停止，仍由 Host 层代码决定。

Host + Client + Server 合起来实现一件事：**Agent 负责做决策，MCP 负责提供能力接口。**

因此，**MCP Client 不是 Agent；它是 Agent 使用 MCP 能力的协议/总线层。** MCP 本身不做推理、规划、记忆或循环控制。它让 Host/Agent 通过标准接口发现工具、获取上下文、调用外部能力，并把工具与数据从 Host 代码里解耦出来。

如果直接复用 Claude Code SDK 这类高层产品能力，你接入的是一个已经包含 Host 决策逻辑的 Agent Runtime，而不是裸 MCP Client。裸 Client 只处理协议；高层 SDK 才可能包含 ReAct 循环、工具选择和任务终止逻辑。

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



Inspector 是一个**标准的 MCP Client 实现**。它不负责和用户对话，只负责发协议请求、展示响应，并帮助你看清 Server 暴露了什么。

- **它模拟客户端行为**：它发送标准 JSON-RPC 请求，走的就是 MCP Client 调 Server 的路径。
- **它校验协议一致性**：如果你的 Server 在 Inspector 里能正常显示 Schema、调用工具、读取资源，那么迁移到 Claude Desktop、Cursor 或其他 MCP Client 时，协议层问题会少很多。



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



运行成功后，终端会显示一个本地网址（通常是 `http://localhost:5173`），浏览器会自动打开这个页面。这就是MCP Inspector。 **在使用MCP Inspector命令的时候，需要严格注意目录的问题，只有启动命令本身可以执行，才能够使用MCP Inspector正确解析，否则则无法连接**



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

MCP 是一套有价值的工程接口：它把工具发现、工具调用、资源读取和提示模板放进同一套协议里，也给 Server 开发者提供了比较明确的实现边界。但它解决的是标准接口问题，不自动解决工具设计、权限治理、上下文组织和产品体验问题。

这也是为什么 MCP 和 Agent Skills 会同时存在。MCP 更适合稳定、可复用、需要明确权限边界的外部能力；Skills 更适合把团队流程、脚本和项目知识低成本地交给 Agent。对开发者来说，关键不是押注哪一个取代另一个，而是看当前能力更像“服务接口”，还是更像“可读的工作包”。
