---
title: "从 ComfyUI 到 LibTV：视频生成时代，工作流编排应该长出什么能力？"
title_en: "From ComfyUI to LibTV: What Workflow Orchestration Needs in the Video Generation Era"
date: 2026-06-18 15:00:00 +0800
categories: [Agent 基础设施]
tags: [ComfyUI, AI Video, Workflow]
author: Hyacehila
excerpt: "ComfyUI 早已从生图 UI 长成开放的生成式 workflow/runtime。视频生成把问题推高了一层：长视频要靠分段生成、素材继承、批量抽卡、质量核验和长任务编排一起撑起来。"
excerpt_en: "ComfyUI has outgrown the label of an image-generation UI. It is closer to an open generative workflow/runtime, and video generation now exposes what that runtime still needs: segments, inherited assets, candidate sampling, evaluation, and long-running orchestration."
permalink: '/blog/2026/06/18/comfyui-video-workflow-orchestration/'
---

过去两年，图片生成里最常见的动作，大概就是抽卡。

原因倒也朴素：生成模型本来就带有试错味道。换一个 seed，换一点提示词，换一个 LoRA，换一张参考图，甚至什么都不换，结果就可能从普通变成可用。Stable Diffusion、SDXL、Flux 这些模型让很多人习惯了「批量出图、挑一张、再局部重绘、再放大」的流程。ComfyUI 能在这个阶段变得重要，也和这件事有关。它把生成过程拆成节点和连线，把原本一次次点按钮的事情，变成了可以保存、修改和复用的工作流。

现在，视频生成也开始进入这个阶段。

只是视频麻烦得多。图片抽卡失败，最多是这一张不行；视频失败，可能是角色第一秒还对，第三秒脸就漂了。也可能是上一段和下一段接不上，镜头运动不错但产品变形了。当前不少视频模型仍然以短片段、单镜头、文生视频或图生视频为基本单位。比如 Stable Video Diffusion 发布时提供的是 14 帧和 25 帧的图生视频模型，更强的闭源模型如Seedance 2.0 和 HappyHorse 也只最多支持 15 秒的生成能力。而人们所需要的绝对不止 15 秒。

这篇不是 ComfyUI 教程。我也不是 ComfyUI 的专业用户。更准确地说，我想借这篇文章从 workflow 的角度重新理解它：ComfyUI 现在到底是什么？为什么把它叫成本地生图软件已经有点别扭？当视频生成逐渐变成常用能力时，它需不需要像 LibTV 这类产品一样，补上一些面向长视频的编排能力？

我的判断先放在这里：ComfyUI 已经从生图 UI 变成了开放的生成式 workflow/runtime；视频生成正在把问题推向更高一层。它未来值得补一些镜头、分段、批量抽卡、核验和长任务管理能力。但它不必变成 LibTV。ComfyUI 最有意思的地方，还是开放、可控、本地化、模型生态和节点可组合。

## ComfyUI 现在是什么

如果只从界面看，ComfyUI 很容易被理解成一个把 Stable Diffusion 画成流程图的工具。这个说法曾经不算错。

[官方文档](https://docs.comfy.org/)现在把 ComfyUI 描述为 node-based interface and inference engine for generative AI。它处理的基本对象是 workflow：一组由 nodes 和 links 连接起来的计算图。节点可以加载模型、编码提示词、采样、解码、保存结果，也可以做 ControlNet、IP-Adapter、LoRA、放大、视频合成、音频处理、3D 或 agent 相关任务。也就是说，ComfyUI 关心的是这次生成由哪些步骤组成，不只盯着最后那张图。

传统 WebUI 更像参数面板。用户填 prompt、选模型、调步数、点生成。ComfyUI 则把这些步骤摊开，让模型、参数、输入、输出都成为显式节点。一个 KSampler 节点连接了 prompt embedding、latent、模型、采样器和 seed；一个 VAE Decode 节点把 latent 还原成图像；一个 Save Image 节点负责落盘。看起来复杂，但它让用户能看见生成过程。

如果是第一次理解 ComfyUI，我会先抓几个概念，不急着去背节点名。

先看 node graph。节点图的用处不在炫技，它只是把生成过程摆到明面上。图像生成里，一张图背后有模型加载、文本编码、latent 初始化、采样、解码、后处理等步骤。视频生成里，这条链更长：可能还要处理参考图、首尾帧、运动模块、补帧、放大、拼接。节点图把这些步骤拆开，用户就可以替换其中一段，而不用整套流程从头来。

再看 workflow。workflow 是一套可以保存和复现的生成配置。除了 prompt，它还记录节点结构、模型选择、参数连接和处理顺序。很多时候真正影响结果的，是模型、参考图、ControlNet 强度、LoRA 权重、denoise、采样器和后处理一起形成的状态。

下面是一套基础的节点组合与他们的一般编排顺序：

```text
Load Checkpoint：请模型上场
CLIP Text Encode：翻译提示词
Empty Latent Image：准备空画布
KSampler：真正生成
VAE Decode：把 latent 变成图片
Save Image：保存结果
Load Image：导入参考图
VAE Encode：把图片压回 latent
Load LoRA：加载风格/角色补丁
ControlNet：控制结构
Upscale：放大增强
Video Combine：把帧合成视频
```

本地模型生态也很重要。ComfyUI 的能力来自本地开源模型和社区开源节点。这些信息可以通过 Load Checkpoint、Load LoRA、VAE、ControlNet 等节点进入工作流。随着社区扩展，Flux、Wan、AnimateDiff、Stable Video Diffusion、HunyuanVideo 等图像和视频路线都能以某种形式接进来。这是它和纯云端 WebUI 生成产品不一样的地方。用户可以在自己的机器上，利用整个社区提供的信息，将自己的生成任务组织成一个工作流。

最后是可编程。[ComfyUI Server API](https://docs.comfy.org/development/comfyui-server/api-examples) 支持把 workflow 作为任务提交，通过队列、history 和 WebSocket 监听执行状态。外部脚本或 agent 可以修改 seed、prompt、参考图和参数，批量跑结果，再取回输出。到这里，ComfyUI 已经不只是一个交互界面，也可以被当作生成后端来调度。

现在的 ComfyUI 理解为三个东西叠在一起：可视化生成图谱、本地/远程推理运行时、可被程序调用的 workflow 系统。它最早因生图而流行，但现在讨论它，只看图片已经不够了。

在本章结束之前还需要聊一下关于生图 Pipeline 的补充问题，前面的角度基本还是在本地模型拆解的角度展开的，现在大家更多的依赖外部 API 使用Partner Nodes 进行图片生成，那实现一个基础的生图 Pipeline 需要先用快速模式区分风格进行抽卡，在4-10张多样性图片中挑选以后，开始使用更高质量的模式进行多轮次的迭代微调以获得更好的生成效果。系统需要自动化的评估（一般是通过外部的 LLM Rubric 作为评估来源），减少人工的提示撰写工作，让系统自我优化，只在必要的时候使用人。

## ComfyUI 正在往哪里走

如果只看社区里那些复杂节点图，ComfyUI 似乎永远属于技术用户。但官方近年的动作，明显在把它往外推。节点图的控制力还在，复杂 workflow 也开始被包装给普通用户、agent 和云端系统调用。

一条线是 API 化。官方文档里有 local API examples 和 workflow API format：用户可以把工作流导出成 API 格式，通过 `/prompt` 提交到队列，通过 `/queue` 查看队列，通过 `/history` 取结果，通过 WebSocket 监听执行状态。这意味着 ComfyUI 很适合放进更大的系统里。一个外部脚本可以读取 100 个 prompt，给每个 prompt 分配 20 个 seed，批量提交任务，再把输出图送去 CLIP、OCR 或 VLM 打分。

另一条线是 Partner Nodes。[Partner Nodes](https://docs.comfy.org/tutorials/partner-nodes/overview) 可以把外部 API 服务、闭源模型或第三方托管模型接进 ComfyUI workflow。这个方向很现实，因为生产流程很少只靠一个模型。本地模型可能负责低成本抽卡，闭源模型负责高质量图生视频，VLM 负责审核，OCR 负责检查文字，人脸模型负责一致性。ComfyUI 要成为 workflow/runtime，就需要让这些能力待在同一张图里。

还有 App Mode、Agent 和 Cloud。官方博客 [From Workflow to App](https://blog.comfy.org/p/from-workflow-to-app-introducing) 提到，App Mode 可以把复杂工作流包装成更像应用的界面；[Agent Tools](https://docs.comfy.org/agent-tools) 让 agent 调用 ComfyUI 生成 image、video、audio、3D；[Comfy Cloud](https://blog.comfy.org/p/comfy-cloud-is-now-in-public-beta) 则指向云端运行 workflow、部署为 API、并行运行多个 workflow、团队协作等场景。

这些东西放在一起看，ComfyUI 已经不像一个单点工具了。它更像一套生成式媒体工作流平台：底层是开放节点图，中间有 API、队列、云端和模型连接，上层再把 workflow 包装成 app 或 agent tool。

也正因为这样，长视频编排才会变成一个绕不开的问题。如果 ComfyUI 只是一次运行一个图的工具，长视频当然离它很远。但如果它已经是 workflow/runtime，那么分段、继承、筛选、重试和拼接迟早会来到它面前。

## 视频不是一次生成能解决的

长视频的问题，不在于把 prompt 写长，也不在于把视频模型的 duration 参数拉满。

在目前的模型条件下，很多视频生产更像一个多阶段流水线。先生成角色或产品参考图，完成整体的脚本和镜头初步设计。再利用生图模型抽关键帧，把关键帧变成短视频，拼接检查一致性。如果某一段坏了，还要回到对应镜头重新抽卡，不用把整条视频全部推倒重来。

图片生成常常只需要回答这一张够不够好。视频生成要多想很多：角色有没有漂，场景有没有换，上一段尾帧和下一段首帧能不能接上。镜头结构、画面运动、文字和产品细节都可能出问题。还有成本。视频重抽一次，比图片肉疼得多。

同时受限于生成模型的能力限制与成本约束，长视频（大于15s）生成的基本动作是分段。分段之后，真正麻烦的问题出现了：段与段之间如何共享信息？

如果第一段确认了角色形象，第二段应该继承这个角色参考。如果第一段的尾帧很好，第二段可以用它作为首帧或参考图。如果第三个镜头是产品特写，它应该继承产品图、品牌色、材质和构图要求。如果某个镜头的候选有 20 个，系统应该能记录哪个被选中，为什么被选中，下一阶段使用的是哪一版。

这时，workflow 本身还不够。一个 ComfyUI workflow 可以很好地描述怎么生成一个镜头，但很难天然描述这部视频有 24 个镜头，每个镜头有 10 个候选，前 6 个镜头共享同一个角色参考，第 7 个镜头使用第 6 个镜头的尾帧，10 个镜头之间共享某种环境，但人物可能需要进行着装上的改变。问题已经越过单个节点图，到了项目级、镜头级、任务级编排。

生成长视频从的问题目前已经从生成问题变为了系统问题。模型能力当然重要，但素材结构、执行顺序、筛选机制和失败重试同样重要。没有这些外部结构，模型再强，也很容易停留在生成了一个不错的短片段，难以去生成一个可以交付的产品（如短剧）。

## 为什么拿 LibTV 做参照

我把 LibTV 拿出来，不是为了说它替代 ComfyUI，也不是给它写评测。它在这里更像一个参照物：当产品直接面向视频创作时，它会自然长出哪些上层结构？

LibTV 更像一个 AI 视频创作工作台。它绕开采样器、VAE、denoise 这些底层参数，把界面放在剧本、角色、参考图、视频片段、会话和结果下载上。官方 [libtv-skills](https://github.com/libtv-labs/libtv-skills) 仓库里提供的脚本能力包括创建会话、发送创作指令、查询会话进展、上传文件、下载结果等；[LibTV CLI](https://www.liblib.tv/zh/cli) 页面也强调可以在 Claude Code 、Codex 等 agent 工具中调用 LibTV 完成图片、视频和角色生成。

这和 ComfyUI 的抽象层级不同。ComfyUI 的节点更像推理算子：加载哪个模型，使用哪个条件，如何采样，如何解码，如何把结果传给下一个节点。LibTV 的对象更像创作资产：这个项目里有哪些角色，哪个镜头要生成什么，参考图是什么，当前会话进展到哪里，结果如何下载，素材如何继续被使用。他们的不同源于生成模型的阶段，前者来自开源生图模型的巅峰期，后者则是利用闭源生成模型进行长视频生成的工具。

LibTV 给 ComfyUI 的启发，不是把节点全部藏起来，这会丢掉他原本的优势。恰恰相反，ComfyUI 不应该丢掉节点图。真正值得借鉴的是：视频生产需要比单个 workflow 更高的组织单位。图片时代，一个 workflow 生成一批图已经很有用；视频时代，用户需要围绕 scene、shot、asset、candidate、version 来工作。

## ComfyUI 应该补什么

如果从这个角度看，我不会先给 ComfyUI 补一个一键出片按钮。更该补的是一层中层编排能力。它夹在底层生成模型和上层 App/Agent 之间，管住视频生产里的分段、继承、筛选和长任务。

我会先补 Shot/Scene 层。ComfyUI 可以继续保留 workflow 作为底层执行图，但在 workflow 之上增加镜头和场景概念。一个视频项目可以包含多个 scene，每个 scene 包含多个 shot，每个 shot 有时长、描述、参考图、首帧、尾帧、使用的 workflow、候选结果和当前状态。用户管理的就会是一条视频结构，而不是一堆散落的输出文件。

然后是批量抽卡层。一个镜头不应该只生成一次，而应该能生成 N 个候选。系统需要知道这些候选来自同一个 shot，参数差异是什么，成本是多少，哪些被淘汰，哪一个进入下一阶段。这个能力不一定复杂，但能少掉很多手动复制 workflow、改 seed、找文件的烦躁。

核验层也需要补。核验不必一开始就全自动，也不必假装 VLM 能判断一切。更现实的做法是把人工评分和模型评分放在一起。比如 VLM 检查是否符合镜头描述，CLIP 检查图文相关性，OCR 检查画面文字，人脸相似度检查角色一致性，NSFW 模型做安全过滤，最后仍然可以由人打勾或打叉。它的目的不是替代审美，而是把筛选结果结构化，让结果能回到下一轮生成。

资产继承层可能更基础。长视频不能每次从零开始，已经确认的东西要能带下去。角色参考、产品参考、场景参考、风格参考、上一镜头尾帧，都应该可以作为资产在不同 workflow 之间传递。这里的资产不只是文件路径，还应该包含用途：这是角色脸部参考，这是服装参考，这是场景深度图，这是上一段可续写尾帧。

长任务管理也绕不开。视频生成会失败，会排队，会中断，会需要重试。ComfyUI 已经有队列和 history，但长视频需要镜头级、项目级的任务视角。用户想知道的不是某个节点是否执行过，而是「第 12 个镜头还有 3 个候选没跑完」「第 5 个镜头通过了核验但还没放大」「第 8 个镜头失败两次，应该换模型」。

最后可以有一个轻量 timeline。ComfyUI 不需要变成 Premiere，也不需要做完整剪辑软件。但如果它要支持视频生产，至少可以提供一个按镜头排列、预览、替换候选、导出粗剪的轻量时间线。用户应该能从结构上看到整条视频，而不是在文件夹里挨个打开 mp4。

这些能力加起来，并不是把 ComfyUI 改造成 LibTV。它们更像是给现有节点图补一点视频时代需要的生产语义。解决生成哪些、按什么顺序、如何继承、如何筛选、如何继续。

## 不必变成 LibTV

我觉得更合理的方向，也许是三层结构。

底层仍然是 node graph。专业用户在这里控制模型、参数、参考图、采样、放大和后处理。

中层是 orchestration。它负责镜头、场景、批量候选、核验、资产继承、长任务和版本。或许他可以是一个视频生成的专门层，生图和视频生成需要作为两个独立人物拆分开，保留一个资源层用于将他们形成中介，这里仅仅是随便聊聊想法，具体怎么做谁知道呢？

上层是 App Mode、Agent 或 Studio UI。普通用户不需要看到所有节点，只需要上传素材、填写需求、选择候选、确认结果。复杂 workflow 可以被封装成应用，也可以被 agent 调用。

未来真正有用的系统，可能会同时吸收两边的经验：底层像 ComfyUI 一样开放可控，上层像 LibTV 一样围绕视频项目组织。ComfyUI 不再是一个本地生图软件，视频生成的兴起会逼这类 runtime 面对更麻烦的问题：如何把一次次短片段生成，组织成一个可以被创作、被筛选、被继承、被重试、被交付的长视频生产流程。

也许未来的视频模型会强到一次生成完整短片。但在那之前，工作流编排仍然绕不开。哪怕最终的生成模型已经强大到可以生成几分钟甚至几个小时的短片，但真正的有价值的长视频生产，还需要一层能理解镜头和项目的结构。而不是提出一句 prompt ，剩下的全部交给模型去猜。

## 参考资料

- [ComfyUI 官方文档](https://docs.comfy.org/)
- [ComfyUI Workflow 核心概念](https://docs.comfy.org/development/core-concepts/workflow)
- [ComfyUI Server API 示例](https://docs.comfy.org/development/comfyui-server/api-examples)
- [ComfyUI Partner Nodes 概览](https://docs.comfy.org/tutorials/partner-nodes/overview)
- [From Workflow to App: Introducing App Mode, App Builder, and ComfyHub](https://blog.comfy.org/p/from-workflow-to-app-introducing)
- [The next chapter for ComfyUI](https://blog.comfyui.ca/comfyui/update/2024/06/18/Next-Chapter.html)
- [Comfy Cloud is now in public beta](https://blog.comfy.org/p/comfy-cloud-is-now-in-public-beta)
- [ComfyUI Agent Tools](https://docs.comfy.org/agent-tools)
- [Stable Video Diffusion 发布介绍](https://stability.ai/news-updates/stable-video-diffusion-open-ai-video-model)
- [Wan2.1 GitHub 仓库](https://github.com/Wan-Video/Wan2.1)
- [LibTV CLI 页面](https://www.liblib.tv/zh/cli)
- [libtv-skills GitHub 仓库](https://github.com/libtv-labs/libtv-skills)
