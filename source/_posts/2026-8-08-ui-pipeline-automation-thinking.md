---
title: "正在取代前端工程师的 AI，做不出游戏里最简单的一块面板"
title_en: "The AI Replacing Front-End Engineers Can't Build the Simplest Panel in a Game"
date: 2026-08-08 20:00:00 +0800
categories: ["Creative Media & Games", "Game AI & Production"]
tags: ["Game AI", "AI Agent", "Workflow"]
author: Hyacehila
excerpt: "同一个模型，写网页界面几分钟就能跑起来，但哪怕有视觉稿的参考，也依旧难以生成可以直接使用的游戏 UI 工程文件。UI 工作流应该如何引入 AI Agent？为什么大多数团队不仅没能简化流程，反而给管线上的其他岗位带来了新的工作？UI 工作流的 AI 化又应该向什么方向、走什么技术路线发展？本文从一些简单的例子出发，聊聊 UI 工作流引入 AI Agent 的核心问题和瓶颈在哪里，以及我们未来应该做些什么。"
excerpt_en: "The same model spins up a web interface in minutes, yet even with a design mockup in hand, it still struggles to produce a game UI file you can actually use. How should AI agents enter the UI workflow? Why have most teams not only failed to simplify the pipeline, but ended up creating new work for everyone else on it? And in what direction, along which technical path, should AI-assisted UI workflows develop? Starting from a few simple examples, this post looks at where the core problems and bottlenecks really are, and what we should be doing next."
mathjax: false
hidden: true
permalink: '/blog/2026/08/08/ui-pipeline-automation-thinking/'
---

## 序

AI 一句话生成一个网页已经是日常。同一批模型，给它一张完整的视觉稿，让它产出游戏里的一块面板——一张背景、几个按钮、一列奖励格，我做了两个月的时间，但很难说取得了多么完美的效果，离网页还差的很远。

为什么？

游戏 UI 更复杂吗？从上限看反而是 Web 更强。Coherent Gameface 就是拿 HTML/CSS/JS 写游戏 UI 的中间件，Minecraft、Civilization VII、Alan Wake 2 都在用它；格式层面同样没什么门槛，Unity 新做的 UXML 和 USS 写起来就是标签加样式表；Cocos 和我们的 XGUI，界面文件本身就是 JSON 或各种变体。Unreal 的 UMG 确实是里面最复杂的，但它也暴露了大量标准接口，够我们程序化地把一个界面拼出来。

游戏 UI 需要艺术设计吗？如果起点是策划案，那确实是问题。但真实分工里，视觉稿这一步 GUI 设计师早就做完了。剩下的是把它翻译成引擎里的 UI 工程文件，不需要美感，也不需要创造力，需要的是细心和大量重复。而这恰恰是 AI 本该比任何人都强的地方。

AI 依旧做不到。无论是商业引擎自带的 AI 工具，还是各个团队在做的内部方案，没有任何一个能说自己彻底解决了这个问题，然后推广出去让所有人一起用。

在正文展开之前，这里会先留下两条会贯穿全文的 Take Home Message：
* 网易可能是最适合做 UI 工作流 AI 化研究的公司，同时也是最不适合真正做成它的公司。 一家公司里同时有 Unity、Unreal 和多个自研引擎，市面上几乎所有的游戏 UI 方案你都能找到对应的团队。这意味着共性问题会暴露得更快。但反过来，技术路线越多，适配工作就越没有尽头，corner case 也越难穷举。
* 动手做一个 AI Agent 之前，最该问清楚的是目标。 是给现在的人配一个 Copilot 来提效，还是让 Agent 接管某个岗位或某道流程的全部工作？这两个目标听起来只差一步，实际可能是完全不同的技术路线和完全不同的成本。

## 项目介绍

先说我做了什么、为什么这么做，顺便借这个机会聊聊 UI Agent 本身。

### UI Agent 尝试解决的问题

游戏 UI 的生产链条并不短。策划提出需求，UX 给出交互方案，GUI 产出视觉设计，UIP 在引擎里把工程搭起来，VX 补上特效与动画，最后由程序把工程和游戏逻辑接到一起。这是一条跨越多个岗位、多个部门的协作链，而 UI Agent 想做的就是把它压缩。

最理想的形态可以对照今天 Coding Agent 在 Web 上的表现：给一段文字需求，直接看到能跑的产物。

但这条链上并非每一环都该交给 AI。审美和对设计意图的判断离不开人，我们真正想让 AI 接手的是重复劳动，而不是艺术创造。所以更现实的边界是从 UIP 这一环开始。往前接住已经定稿的视觉设计，往后一直做到代码能上线。

当我们开发一个系统时，最应该先定义清楚的是输入和输出。这里的输入是 Figma 或 PSD 的设计稿，加上少量文字信息；输出是完整的 UI 工程文件和配套的程序代码。

```
▎ 一张横向三栏的信息对照图，主题是"设计稿里有什么、UI 工程文件要什么"。
▎
▎ 整体是从左到右的流向。左侧一个大方框标题为"设计稿（静态，给人看）"，右侧一个大方框标题为"UI 工程文件（可交互，给引擎读）"。两个方框之间不是简单的箭头，而是一道由窄变宽的楔形缺口，缺口上标注"语义鸿沟"。
▎
▎ 中间区域被横向分成三条水平带，每条带有自己的标签，从上到下依次是：
▎
▎ 第一条带（标签"设计稿里有"，用实心、饱和的颜色，视觉上最"实"）：像素、位置与尺寸、图层名、不透明度、文字内容。这一条带从左侧方框一直贯通到右侧，表示这些信息可以直接搬过去。
▎
▎ 第二条带（标签"部分有，且不可靠"，用半透明或虚线填充，视觉上"半实半虚"）：层级结构、分组意图。这一条带从左侧出发，到中间楔形处变细、变淡，但仍然连到右侧。
▎
▎ 第三条带（标签"完全没有"，用空心虚线框，视觉上最"虚"，颜色最浅）：控件类型、状态（常态/悬停/按下/禁用）、数据绑定、交互逻辑、动画时序、多分辨率自适应规则。这一条带在左侧方框内是空的，只在右侧方框里有内容，中间用断开的虚线表示信息在这里断掉了。
▎
▎ 第三条带是整张图的重点，可以用一个醒目的标注框指向它，写"这一段信息，设计稿里根本没有"。
▎
▎ 风格：干净的技术示意图，扁平化，无阴影无渐变，浅色背景。三条带用同一色系的三个明度（深/中/浅）区分，不要用红绿等语义色。中英文标签并排（如"控件类型 control-type"）。整体宽高比约 16:9。
```

把输入和输出摆在一起，有件事立刻就清楚了：两侧的信息并不对称。设计稿描述的是"它长什么样"，而 UI 工程文件要回答的是"它怎么运作"。哪些像素合起来是一个按钮，按下去会变成什么样，这一列奖励格的数据从哪里来，换一个分辨率之后谁拉伸、谁钉住。控件类型对人来说是自然的，但设计稿不会告诉你这是 Tab 还是滚动容器。这些信息，设计稿里根本没有。

这解释了为什么在生成式模型出现之前，所有从设计稿到工程文件的尝试都失败了。确定性脚本的输出只能是输入的函数。输入里没有的信息，它变不出来。哪怕反过来要求设计师在图层名里加各种约定字段，能补上的也只是零碎几项，还要给设计师增加很多工作量。

生成式语言模型改变的正是这一点。它带来了第二个信息源：大规模预训练沉淀下来的先验。它能像一个有经验的人那样，看到一排等距排列、结构相同的元素就知道那是个列表，看到右上角的小叉就知道那多半是关闭按钮。设计稿里没写的东西，它可以猜，而且猜得比任何规则都准。

到这一步，这件事才第一次在原理上可解：在不增加设计师工作量的前提下，减少后面那一段所需的人力。



中间件 + JSON 这个思路本身不对，依靠固定化接口和兼容二进制文件可能是对的，直接 Web 化靠大规模训练可能也是对的。这三个不同的文件格式为我们带来的问题值得进一步思考和讨论。

让 AI 模拟人类分工一定是不对的，整个 UI 管线涉及很多人但是最终目标一致，就像现在没有人写个 Web 需要一个 Agent Team 一样。策划，GUI，UX，UIP，VX，程序的共同任务都是做出好用的 UI，和 Web 没有本质区别，那么按照职能将这个任务拆分成 Team 这件事情是让不全能的人类分工取代了全能的 AI 协作体系，你没有相信模型本身的能力。























## Stage2 一个什么样的结构用来解决这个问题

当问题变为 semantic gap，当前后端的格式允许变换（我们已经解释了无论是否考虑前后端情况，我们的核心工作都是一致的；很显然，允许前后端格式转变会为这个系统带来更大的想象力），我们不得不引入一个 **IR（中间表示，Intermediate Representation）**。中间组件在这里的作用包括：

* **引入通用中间层**，让系统能够更好地一次开发、适用于变换前后端文件类型的多种任务。N 个前端 × M 个后端，如果两两直连就是 N×M 的复杂度；IR 把三方解耦成 N+M。同时通过多阶段的生成，适当解耦结构、降低单步任务的难度，便于生成式语言模型解决任务（和 CoT 一样，更多的中间件能够进一步降低单步难度，但是中间件本身也会带来新的出错可能，我们需要平衡收益与风险）。
* **提供新的观测层**。中间件会和日志一起，便于我们在系统出错溯源的时候定位任务的来源、思考优化角度；而一个端到端系统优化起来会更困难。
* **允许 Human in the loop**。一个容易观测和介入的 IR 允许我们更好地利用人工的能力。我们并不指望这套系统能够在任何场景下完全替代人工，那么就要提供一个简单的入口，允许人工快速介入、避免系统跑偏。

综上，我们可以把系统抽象为下面的编译器架构，和很多利用 LLM 动态编译的系统完全一样：

```
设计意图 + docs ──(LLM 前端 frontend)──▶ 小而可审计的 IR ──(确定性后端 backend)──▶ 庞大的 .uiprefab JSON
                    短小语义判断           （人/机都能读）        确定性 lowering，无 LLM
```

- **LLM 前端**负责跨越语义鸿沟里需要「理解」的部分（哪些图层组成一个按钮、哪个是重复列表项、哪些是互斥状态），产出一份**小而可审计的 IR**——**绝不**直接产出庞大 JSON。
- **确定性后端**把 IR **lower** 成精确的 `.uiprefab`（布局计算、类型检查、fuid 分配），这一步**没有模型**、完全可复现。
  > 注：「没有模型」指**算浮点 + 写字节**这层——布局计算（anchor 预设解析 + PSD 左上原点→引擎锚点的 Y-flip）是一个编译器 pass；校验是类型检查；fuid 分配是符号分配。后端有两条并存路径——单体 `emit()` 批量发射 / 可拼接工具 `tools_api` 增量拼装；走工具路径时模型仍在场，但只做**编排**（选工具/顺序/高层参数），不碰字节（详见 Stage3.3 与 docs/tool-design-spec.md）。
- 为什么这么分？因为「需要判断的语义」适合交给 LLM，「精确、易错、必须可复现的格式细节」适合交给确定性代码。

**把 LLM 的输出面积压到最小，把所有冗长与精确推进确定性代码里。绝不让 LLM 直接写庞大 JSON。**

它在我们继承的真实系统里有证据：旧的 `validate_output.py` 里**塞满了 JSON 修复逻辑**（剥 markdown 围栏、中文弯引号→直角引号、去尾逗号、逐字符转义字符串内引号），外加多编码兜底。这套补丁的存在本身，就证明了「让 LLM 直接吐又长又精确的 JSON 不可靠」。我们要做的，是从架构上消灭这种不可靠，而不是不断给它打补丁。

**我们怎么形成 IR——复用 prefab-studio 的思考逻辑，换掉 IR 的形式**

我们没有全新的关于研究原始数据的想法。**研究原始数据、把图层翻译成语义的那套思考逻辑，整体复用 prefab-studio Skill**；新版本项目的重心，落在**换一种 IR 形式**和**从 IR 到 JSON 的方式**上。下面先把"IR 是怎么被一步步形成的"讲清楚，再说明哪些是复用、哪些是新的。

#### IR 的形成是一条"先确定性预处理、再 LLM 语义判断"的管线

无论 legacy 还是 design2ui，把设计稿变成 IR 都不是模型一口吐出来的，而是一条**确定性步骤打底、LLM 只在两处做语义判断**的管线（prefab-studio 的真实分阶段，详见 skills/prefab-studio/references/pipeline-stages.md）：

```
PSD ─① 预处理 ─② 预聚类 ─③ 元素裁决(LLM) ─④ 结构组装(LLM) ─▶ IR ─(后端)─▶ .uiprefab
      确定性      确定性     视觉+语义判断     视觉+语义判断
   拆图层/合成图  几何重复簇   这是不是控件？     谁是谁的父节点？
   /智能对象     /背景自判    什么 engine_type？  什么 anchor 预设？
```

- **① 预处理（确定性）**：解析 PSD，产 `layers_enriched.json`（每层 bbox / 文字属性 / blend / opacity）、`composite.png` 合成图、`layer_tree.json`、`exclusive_groups.json`、智能对象目录。文字层直接带可读内容——**不需要 OCR**。
- **② 预聚类（确定性）**：几何上检测重复簇与大块背景，产 `proto_groups.json`；背景/大块**自动裁决**（auto_resolve），LLM 只需处理"拿不准"的组。这一步把"哪些图层长得像、可能是重复列表项"先用几何算出来，降低后面 LLM 的判断量。
- **③ 元素裁决（LLM，视觉 + 语义判断）**：对 auto_resolve 为空的组，看小合成图 + 预聚类结果，做 keep/split/merge，并定 `engine_type`（这是背景图？可点击按钮？可编辑文字？重复卡片？）。
- **④ 结构组装（LLM，视觉 + 语义判断）**：把裁决出的元素组织成层级树，为每个节点推断 `anchor_preset`（居中？拉伸？贴边？）。
- **后端（确定性）**：把 IR lower 成 `.uiprefab`（布局浮点 + Y-flip + fuid + 样板 + 装配）。

**③④ 这两处 LLM 语义判断，就是 IR 形成的实质**——它们正是 Stage1 所说"需要理解、需要正确猜测"的部分。①②⑤全是确定性代码。

#### 复用 prefab-studio 的四个想法

新项目原封不动继承 prefab-studio 在"研究原始数据"上沉淀的四块思考逻辑：

1. **图层模型**：图层树/组/blend/opacity/文本层/智能对象的解读方式，"组 + 命名 = 一个控件的边界""文字层免 OCR""mod_ 前缀 = 复用锚点"等先验（对应 ① 预处理与领域知识，详见 Stage3.2）。
2. **元素裁决（element-resolver）**：判断"一组图层是不是一个控件、应落哪个 engine_type"的规则——含**图片滥用三道闸门**（含文字？可点击？重复簇？）（对应 ③，详见 Stage3.2）。
3. **结构组装（structure-assembler）**：把元素拼成层级树、推断 anchor 预设的**六条合并规则**（真背景才 stretch-full、中心美术 center、内容簇包裹、标题簇、成对边缘、模态遮罩）（对应 ④，详见 Stage3.2）。
4. **复用识别**：从几何重复簇（`proto_groups` 的 `repeat_candidates`）识别"单原型多引用"，判 `XControl + 子 prefab`、同簇共享 `prototype_id`（横跨 ②③④，详见 Stage3.2 与 Stage3.4 检索优先）。

也就是说，**"怎么从图层看懂这是什么 UI"这件事我们不重造轮子**；prefab-studio 在 14000+ 语料上打磨出的这套规则，连同它的确定性后端知识（坐标/fuid/样板/装配），都被新项目继承。

#### 新的是"IR 的形式"和"从 IR 到 JSON"

重心变化集中在两点，恰好就是 Stage3 的前两问：

- **IR 形式**：把 legacy 的**双 JSON IR**（`elements_identified.json` + `structure_llm.json`）换成**单一 `ir.md`**（Markdown 标题树，标题深度=节点嵌套）。同时**砍掉 legacy 一个不透明的 `post_process` 阶段**——legacy 会在 LLM 写完结构树之后偷偷改写它（注入适配脚手架、reparent、删背景、塌缩层级、覆写 anchor），导致"作者写的树 ≠ 最终构造的树"，才不得不维护一份投影快照。design2ui 用 **WYSIWYG 字面契约**取代它：构造树 ≡ ir.md 树，任何改树形的操作必须显式写进 IR（详见 Stage3.1）。
- **从 IR 到 JSON**：legacy 是"LLM 产 JSON → `emit.py` 单体编译"；design2ui 是"Agent 读 `ir.md` → 调可拼接工具**逐节点直造**，无中间 JSON"（详见 Stage3.3）。

综上，这个逻辑中的三个核心问题——**用什么格式的 IR、如何从图层模型重建鸿沟、编译生成 or 迭代修改**——会在 Stage3 被详细介绍。前两问对应上面"新的 IR 形式"与"复用的图层思考逻辑"，第三问对应"从 IR 到 JSON 的方式"。

---

## Stage3 一些问题

### 3.1 使用什么样格式的中间件——我们如何思考和理解 IR

这里要对比原本的 JSON IR 和新的 `.md` IR，介绍我们从现有 uiprefab 文件和 JSON IR 的信息中挖掘出来的东西，以及为什么我们选择了这个新的 IR、它的优势是什么、还有什么不足。

#### legacy 的双 JSON IR（prefab-studio 旧路径）

旧 prefab-studio skill 的机制是 **emit-from-JSON**：先让 LLM 产**两个** JSON IR——`elements_identified.json`（元素清单）+ `structure_llm.json`（结构树，用 `element_ref` 引用元素），再由 `assemble.py` → `compute_layout.py` → `emit.py` 从 JSON **单体编译** `.uiprefab`，中间还落 `ui_skeleton.json` / `node_properties.json` 等产物，校验要过 `validate_elements` + `validate_structure_llm` **双闸门**。

这套机制有几个固有问题：① LLM 仍要背负"写两份精确 JSON"的成本（前述 `validate_output.py` 的修复补丁正是为它存在的）；② 双闸门、双产物、`register_element`/`element_ref` 的间接引用，增加了出错面与外挂修复成本；③ "确定性 JSON → 确定性脚本 → 确定性 JSON"这一架构本身的意义并不明确（详见 3.3 的批判）。

#### 新的 `.md` IR：标题树即节点树

design2ui 用**单一 `ir.md`**（一份人机皆可读的 Markdown 文档）取代上面的双 JSON。核心语法是：**每个节点（含叶子）是一个 Markdown 标题，标题深度直接表达 uiprefab 的节点嵌套深度**——父节点深度 N，子节点深度 N+1。这与真实 `.uiprefab` 的 `child_list` 容器嵌套一一对应。

```markdown
# ir · <界面名>
meta: <pc|mobile> <W>×<H> · root_adaptation=<fullscreen|modal-center> · ir_version=1

## panel_root · XWidget · stretch-full
semantic: <description>

### <node_id> · <engine_type> · <anchor_preset> [· region=[x,y,w,h]]
<key>: <value>     # 叶子视觉属性直接内联：text / font / color / sprite / opacity / reuse / prototype ...
```

- 标题格式：`<node_id> · <engine_type> · <anchor_preset> [· region=[x,y,w,h]]`。
- H6 是天然深度上限（Markdown 标题到 `######` 为止），容器嵌套通常 3–5 层；逻辑树超过 H6 是**应当拆子 prefab 的信号**，恰好对齐工具的嵌套上限 `max_depth=5`。
- 校验从 legacy 的双闸门收敛为**单一 `validate_ir`**，含 8 条规则：标题语法可解析、`node_id` 命名 + 全局唯一、`engine_type` 落在 **46 类白名单**内、`anchor_preset` 落在 **16 个预设**内、叶子 vs 容器约束、`reuse`/`prototype` 互斥、嵌套深度 ≤ 5、`text`/`sprite` 类型不交叉。

#### 进 IR 的 vs 工具保证的——边界（值级降解）

| 作者（LLM/人）写进 IR | 工具 "by construction" 保证（**不进 IR**）|
|---|---|
| 画布 meta（平台/尺寸/root_adaptation） | `fuid` 分配（单调计数器）、`max_fuid` |
| 节点层级（标题嵌套） | 每节点 `__editor` 块、transform、~15 个 flags（zorder/raycast/touch_enable…）|
| 每节点 node_id · engine_type · anchor_preset · region | 布局浮点：`anchor_min/max`、`offset_min/max`、`position`、`size`（由 anchor_preset + region 经 **Y-flip** 算出）|
| 叶子视觉：text/font/color/align、sprite/nine_patch、opacity | 类型 lowering：native vs 模板 component vs XControl 的落地差异、`template_type_name`、`component_list` 装配 |
| 复用意图：reuse / prototype；组件/状态意图 | 值转换：`color` RGB→Color32 整数、`font_key`→字体表、`.studio/` 路径 re-root |

以上工具侧全是**值级降解**：只填充每个节点的引擎细节，**不改变节点身份或树形**。

#### WYSIWYG 字面契约

这带来一个关键不变量：**工具构造出的 `.uiprefab` 节点树 ≡ `ir.md` 的标题树**——节点身份与父子关系一一对应。`ir.md` 既是唯一作者源，**又是字面产物结构**：你写什么节点、什么嵌套，就构造出什么；**没有任何东西背着 Agent 增删或重组拓扑**，因此不存在「投影/realized 快照」概念，第 0 层人审可以直接审 `ir.md`。

#### 为什么选这个 IR——优势

1. **最小输出面积**：`ir.md` 只表达语义判断（类型、层级、布局意图、文字内容、复用关系）；所有冗长 lowering 细节由确定性工具承担。
2. **人机皆可读**：Markdown 标题树清晰、token 效率高，能表达中间态，降低阅读和修改 IR 的成本（直接服务 Human-in-the-loop）。
3. **单一闸门可验证**：一份 `ir.md` → 一道 `validate_ir`（不再有 elements/structure 双闸门）。
4. **增量可构建**：支持从零构建、局部添加、单节点编辑三种操作模式（详见 3.3）。

#### 诚实的不足

- **最大的不足是这套 IR 缺少人工核验**：它是 AI 根据以前的 JSON IR 以及 uiprefab 文件组织出来的，目前尚未经过领域专家 / 真实美术-引擎工作流的核验确认。spec 自身已诚实标注"凡涉及真实字段以代码为准"。
- **布局意图层仍 free-form**：现管线能产出"嵌套分组正确的 panel 树"，但 layout-intent 层仍以绝对定位为主——实测叶子节点中 free-form 绝对定位占比 **82%–96%**（pailian 94% / fuwuqi 95% / huiyuan 96% / chengpin 94% / baoxiang 82%）。即"分组对了，布局意图（线性/网格/自适应）还没真正抽象出来"。

### 3.2 从 PSD/Figma 的图层模型中获取信息、重建鸿沟

无论是 PSD 还是 Figma 都包含一些图层信息。如果提供更加严格的图层约定范式、减少缺少实际含义的图层，就可以减少 prefab-studio Skill 里使用的一些启发式方法（包括子 uiprefab 文件识别、图层划分等）。这里先介绍目前是怎么做图层合并和子 prefab 识别的。

#### 图层模型携带什么

PSD 是一棵**图层树**（root → 组 → 图层），从下往上叠加合成。CS 类比：一组带 z-order 的 DOM 节点。关键信息：

- **组（group）**：极其关键——组 + 命名往往就是"一个控件"的边界。
- **blend mode / opacity / mask**：类比 CSS `mix-blend-mode` / 透明度 / alpha 裁剪。
- **文本图层（text layer）**：矢量文字层，**直接携带可读的内容/字体/字号/颜色**——文字不需要 OCR，这是 PSD 相对截图的核心优势。
- **智能对象（Smart Object）**：引用/嵌入层，常是同源复用的信号。
- **关键事实**：PSD 格式里**根本没有"状态（state）"这个原生概念**。语义靠图层命名 + 人的理解编码。

#### 命名约定：mod_ 前缀 = 复用锚点

真实 PSD 里会看到诸如 `mod_xxx` 这样的图层名前缀。这类命名约定正是设计师在"设计稿层"里手工编码语义、试图给"运行时层"留线索的痕迹（思想源头可追溯到 Adobe Generator 2013 按图层名后缀自动导出切图）。`mod_` 前缀具体用作**复用锚点**的识别信号。prefab-studio 的命名分层贯穿 element_id（S2，全局唯一语义业务名）→ node_id/display_name（S3，PascalCase 编辑器显示名）→ engine_type（X 前缀，落白名单）。

#### 当前的图层合并启发式（structure-assembler 六规则）

- **规则 1 真背景才 stretch-full**：一个被标记为背景的 `XImage`，**只有当其源 bbox 与画布近似重合**（原点 ≈ (0,0) 且宽高 ≈ 画布 W/H，各 ±5% 容差）才用 `stretch-full`；装饰光效/特效层而非真背景 → `center` + 实际尺寸。
- **规则 2 中心美术用 center**：大型中央立绘/主视觉 → `center`，不因贴边就误用边缘预设。
- **规则 3 内容簇包裹**：非背景内容收进**单个 center 锚定、画布尺寸的容器**，与背景同级。
- **规则 4 标题簇**：标题图 + 副标题 caption 收进单一容器，"标题相邻 caption 要全部收进簇"。
- **规则 5 成对边缘控件**：镜像预设（`left-center` ↔ `right-center`）只给真正贴住视口边缘的控件。
- **规则 6 模态遮罩**：全屏压暗遮罩 → 最底层子节点，`stretch-full`。
- **冗余去重**：多个真背景层 IoU ≥ 0.9、同为 normal blend、同不透明 → 删冗余；叠色层（screen/add、半透明）保留。

#### 子 prefab / 复用簇识别

- **CLI 确定性信号**：`proto_groups.json` 的 `repeat_candidates`（几何检测出的重复簇：相似尺寸 + 行/列等距），形如 `{cluster_id, axis, count, member_group_ids, bbox_each}`。
- **落地**：簇成员各判 `engine_type = XControl` + sub_prefab；**同簇所有实例共享同一 `prototype_id`**（单原型多引用），各自保留独立 element / layer_ids / bbox。下游生成**一个**子 prefab 文件 + **多个** XControl 引用。子 prefab 的 `data_list:[]` 由策划后配、pipeline 不填——空列表是正确行为。
- **已废弃**：`repeat_group` / `prototype_children` / `instances`，v9 校验器直接拒绝；多实例 = 多个平级 XControl 共享 prototype_id。

#### 图片滥用三道闸门

成熟工程里 **XImage 只配三种角色**：真背景/底图/纹理、纯装饰/光效/立绘、确实无可点击语义且无可编辑文字的静态图标。判为 XImage 前必须过三道闸门：

1. **它含可编辑文字吗？** → 有则必须 XLabel/XText 或 XButtonLabel，绝不烤进 XImage。
2. **它可点击吗？**（按钮底/胶囊/暖黄确认色/底部操作区）→ 是则 XButtonLabel/XButtonImage/XButton。
3. **它是重复簇成员吗？** → 是则 XControl + sub_prefab。

"拿不准 → XImage"是被明确否定的错误默认。配套有**发射前自检回路**：扫 `conformance_hints` 的 `baked_text`（非文本宿主吞了可编辑文字层）和 `repeat_candidates_uncontrolled`（重复簇未走 XControl），必须清零或逐条解释。

#### 锚点模型（anchor / pivot / offset）——本节最难的 lowering

锚点模型是已知难解的一块：引擎的 RectTransform 用 `anchor_min/max` + `offset_min/max` + `pivot` 描述一个节点相对父节点的定位与拉伸，而 PSD 只给左上原点的绝对 bbox。从 bbox → 锚点预设的映射，既要做 **Y-flip**（PSD 左上原点 → 引擎锚点系），又要在"绝对定位 / 居中 / 拉伸 / 贴边 / 镜像成对"之间做语义选择。当前用 16 个具名 anchor_preset 收敛这个空间，但选错预设会在不同分辨率下表现错误。**这是需要参考外部文档信息进一步研究的难点**（anchor 预设表当前在 4 处校验层有分歧，S6a 里程碑就是为它立单一权威源）。

#### 更多图层结构能带来什么——目前缺验证

目前对图层结构的利用还有限。我们**猜测**：更严格的图层约定范式、更多有语义的图层结构信息，可以减少这里的启发式，带来更好的生成效果和更低的反思花费——但**这一点目前还缺乏验证**。一个直接佐证是：现有 5 个 benchmark case 实测 `exclusive_groups`（互斥/状态组）**全为空**，因为这些 PSD 没把替代态作者化为隐藏重叠图层。也就是说，"系统已能检测互斥组、只差发射规则"，但**输入信号本身不就绪**——这恰恰说明"更结构化的图层输入"的价值需要先有数据才能验证。

### 3.3 编译生成 or 迭代修改——受限于目标件的设计哲学

在原本设计中，我们采用的是编译生成的架构：一个确定性的脚本文件，根据一个 LLM 生成的 JSON 文件，编译出 uiprefab 文件。但是——为什么是"确定性的 JSON 通过确定性的脚本生成确定性的 JSON"？这样的架构意义并不明确。**AI 还是要承担写 JSON 的成本，反而是多层更容易出错和外挂修复。**

与此同时，并不是所有的 UI 组件都用 JSON。将系统的核心改为二进制文件（如 Unreal 的 `.uasset`）并暴露一套接口，是部分厂商的选型。我们之前的目标件是 JSON，采用编译生成很自然；但**如果系统只有二进制底层文件和接口，这个"让 AI 生成长文本再编译"的思路就没有这么灵了**——你没法让 AI 去吐一段会被编译成二进制的文本。

结论：**一套足够 robust 的 IR，搭配被 Agent 调用的确定性 tools 合集**，能够完全规避掉"编译架构让 AI 生成长 JSON"的不稳定问题以及相关的修复成本。而且**迭代修改的 tools 更加便于我们考虑复用**。

具体到 design2ui 的工具设计（详见 docs/tool-design-spec.md）：

- **六类工具（A–F）**：A 场景初始化（工作区、ir.md 骨架）、B 节点构建（标题树 + 叶子视觉属性内联）、C 组件（布局/适配意图）、D 状态机（多态意图）、E 子 prefab（独立 ir.md）、F 验证与构造执行（`validate_ir` / `build_uiprefab` / MCP 预览）。
- **增量构建三模式**：从零构建（init → add_node×N + set_props → validate → build）、增量添加（load → 在已有父节点下加新节点 → 全量重建）、局部编辑（load → 改单节点属性 → 重建）。
- **reuse 作为普通工具调用**：`set_reuse(node_id, visual_path)` 走 `reuse_existing` 通道、`auto_load_prefab=false`，让"复用一个检索到的成熟组件"成为一次普通的工具调用——这把 Stage3.4 的"检索优先"直接接进了构造流程。
- **不变量**：Agent 绝不用常规编辑命令文本编辑 `.uiprefab` JSON 字节（严令禁止），只调 `tools_api` 的确定性函数；模型驱动的是"编排"（选工具/顺序/高层参数），不是"写字节"。

设计目的是在"直接手改 JSON"与"黑箱单体编译"之间取平衡——既灵活（可对个别节点局部增量编辑），又安全（不退化成手写易错字节），**显著降低反馈审阅后反复修改的成本**。两条后端路径并存：legacy 的 emit-from-JSON（`emit.py` 单体编译）只属于过去；design2ui 的工具承袭其确定性**知识**（坐标/fuid/样板/装配），但重新封装为"读 ir.md、逐节点直造 uiprefab"的可拼接函数，**无 JSON 中间件**。

### 3.4 更多的信息从哪里来？——为什么 AI 能写好 HTML+CSS+JS 却写不好更简单的 UIprefab

我们遇到的最大的问题就是：我们想利用生成式语言模型的 zero-shot 泛化能力去解决 UIP 的问题，奈何模型压根没有在相关数据上训练过，所有的知识都没有被注入到模型参数内部以实现 zero-shot 泛化。

对比就很刺眼：AI 能写好复杂的 CSS / HTML / JS 动画，甚至能看图反推 CSS 设计和 JS 控件——因为市面上最不缺的就是相关语料。但 AI 难以理解 uiprefab 的设计逻辑，哪怕它是相对更好理解的 JSON。**核心差别不在格式难度，而在训练语料的有无。**

这导致我们相当于要把全部的知识**外挂**给模型，然后这个任务本身涉及大量多模态，外挂多模态内容难得要死、效果还不一定好。我们还没有现成的知识库能用，需要半路出家研究怎么构建外挂的知识库，然而我自己都不知道知识库里应该有点什么。

#### 检索优先（retrieval-first）是当前的主答案

UI 开发同学的硬反馈是：让 AI **直接生成** `.uiprefab` 几乎不可用——(a) 格式特殊、超长、精确（真实样本 max_fuid 上千，每节点带 `__editor`/transform/RectTransform 锚点/component_list）；(b) UI 任务重度依赖视觉反馈，zero-shot 生成缺少"对照已知好样本"的锚定。

但我们手里有别人没有的东西：**14,225 个现有 `.uiprefab` 语料** + 大量成熟、已上线、被反复打磨的 UI 组件。人类美术拼 UI 的真实做法本就是"**找一个相似的再改**"。检索优先就是把这个工作流变成系统的第一步：

| 维度 | 从零生成 | 检索复用 |
|---|---|---|
| 鲁棒性 | 低——超长 JSON 易崩 | 高——基于已上线成熟样本改 |
| 视觉保真 | 模型"想象" | 从成熟组件继承 |
| 可解释 | 黑箱 | "复用了 X、改了 Y"，可审计 |
| 起步成本 | 高——要先做出可靠生成 | 低——包装 AI Sim + 语料即可 |

可量化的天然真值：语料里有 **38,629 条 `XControl→visual_path` 引用**——这些是真实 prefab 里 XControl 指向真实子 prefab 的确定性事实，去名（防泄漏）后即是现成的检索召回评测集（**recall@k / MRR**），不依赖任何缺失的真值配对、今天就能跑。配合公司内部的多模态检索工具 **AI Sim（以图搜图）**做 few-shot 视觉锚定，作为可插拔的检索后端。

#### 知识库：诚实的现状

知识库该装什么，目前有两条路、尚无定论：① **第一性原理的语义知识库**（显式写"布局范式""按钮状态""文字流"——作者成本高、维护脆）；② **基于 AI Sim 的 few-shot 利用**（给相似 prefab 让模型 pattern-match——降低作者成本、但依赖鲁棒检索）。当前方向是检索优先 MVP → 先量 recall@k 验证 thesis → 随范式涌现逐步加语义层。诚实标注：**复用 E2E 命中率当前 = 0**（通道已实现但 0 验证，S3 待点亮，依赖 P1 的 common_library 件库）。

#### 三个信息缺口最大的具体实例（折入难点）

这三个都是"图里猜不出来、信息论上就缺"的典型，需要参考外部文档信息专门研究：

- **字体走 `font_key` 间接层**：字体不是直接写字号字形，而是经 `font_key` 间接引用字体表（如标题 f1 / 正文 b2 / 四态 normal=f1·press=f2·disable=f3·hover=ff4）。**图是猜不出字体结构的**——这层间接需要额外的适配工作和外挂的字体注册表（已 vendored 738 行字体表）。
- **`anim_track`（动画轨道）**：所需的信息远比静态的 PSD/Figma 更多——静态稿**没有时间维度**，动画从静态图**信息论上推不出**，缺口比生成静态界面更大。真实 emitter **当前并不生成** `anim_track`（只在根写一个空 `anim_timeline`，代码为准）。策略是**检索-重定向**而非生成。
- **`UICustomStateComponent`（多状态组件）**：状态切换的理解需要新方法。真实 prefab 里多状态是**双编码**（根节点 `UICustomStateComponent` 聚合 + 每节点 `state_track`），每条变化是四元组 `[节点name, 变化类型(Static/Template), 属性名, 属性值]`，`custom_state_lst` 的顺序是权威的。信号来源是 **PSD 多图层 SPEC 命名 / Figma 状态组 / PSD 互斥组**——但如前述，现有 case 互斥组信号全空，这是 R9 的硬前置数据缺口。

### 3.5 在做什么、做到什么、能做什么——prefab-studio → Codex 迁移与迭代优化

我们希望在这里回应：原本的 prefab-studio Skill 向 Codex 的迁移，以及我们所进行的迭代优化——我们做到了什么、它们对于新 Skill 开发的启示。改进的具体内容位于桌面的 `prefab-studio-优化汇报.md`，此处只做汇报级摘要。

#### 工程化重构：v8 → 交付版

把 v8 一条**强依赖网易内网 trunk + CodeMaker 多 Agent 宿主**、靠人工检查点保质量的流水线，重写成一个**自包含、离线可跑、确定性兜底合法性**的交付级 Skill：

- **运行依赖**：4 个后端包 vendored 进 `scripts/lib/`，瘦入口经 `_common.bootstrap()` 注入路径 + 强制 UTF-8 → headless 离线可跑、不碰 trunk，根治 Windows GBK 打印崩溃。
- **宿主架构**：从派发 6+ 个钉死三供应商的 LLM 子 Agent，改为 Codex 主线程就地读 `references/` 决策，只在"元素裁决 / 结构组装"两处需视觉判断。
- **合法性保证**：沿用并就地硬化确定性 emitter；LLM **绝不写 .uiprefab JSON**，只产小 IR，合法性由代码兜底。主路径已离线实测端到端跑通多个 PSD，**首次发射即合法**。

#### 三层裁判系统（防泄漏 / 防刷分 / 防过拟合）

1. **确定性 scorer**（`score_conformance.py`）：对语料分布一致性打 **8 维分**（D1 类型分布 / D2 可编辑文字保真 / D3 模板利用 / D4 visual_path 命中 / D5 文字角色 / D6 锚点 / D7 复用 / D8 结构卫生，D2 权重最高 0.20）。**防刷分关键**：archetype 由 **PSD 侧信号**判定（与被判 engine_type 无关），杜绝"把屏过度图片化以逃进宽松档"的循环。
2. **确定性自检中间件**（`resolve_save` 内）：`baked_text` + `repeat_candidates_uncontrolled` 写进 `conformance_hints`，把图片滥用/复用不足从主观判断变成可核验闭环。
3. **LLM 对抗 critic**（`build_judge_bundle.py`）：只看 IR + 发射骨架 + scorer findings + 同档语料范例（**不看像素**），默认倾向证伪。**作者 agent 与裁判 agent 分离**，orchestrator 亲自 gate 每处修改。

#### 真实的多轮迭代（有产物留痕，不是一次性改完）

- **拍脸 PSD 10 轮硬化**：质量轨迹 **42 → 68 → 68 → 82 → 82 → 78 → 86 → 82 → 82 → 88**，round3 起 0 critical，round10 `is_good_enough=TRUE`。沉淀出 7 个确定性硬修复（bbox 并集、资源相对路径、root_adaptation 通道、blend/opacity 信号、覆盖率闸门后去重、锚点继承门、opacity 保真链）。
- **四屏一致性优化**：聚合 **82.7 → 99.8**（huiyuan 72.6→100、fuwuqi 76.7→100、chengpin 87.6→99、baoxiang 93.9→100）。**判别力保留的硬证据**：修完误伤后，原始过度图片化基线仍被正确低分（huiyuan-OLD 仍 72.6），即 scorer 没被调宽到失去判别力。对抗 critic 两轮共抓 **2 个真 bug + 5 个系统性误伤**。

> 口径校正：本次扫描的是 **14,156 个**成熟 `.uiprefab`（0 解析错误）烤成 `corpus_priors.json`；14,225 是更广义的语料规模口径。emitter **不是净新增**——v8 已有（~1118 行），本次是移植 + 硬化。

#### 启示

毫无疑问，迭代优化的成果证明了 **benchmark 的价值**，以及**现有 LLM 有能力在适当反馈下进行自我改进**。三层防泄漏防火墙（盲生成 / 对抗 critic / orchestrator 亲自 gate）这套方法论，可直接套到 design2ui 的 benchmark 回归守护，无需重设计——这正是 Stage4 的起点。

---

## Stage4 Benchmark 和 feedback 反馈系统应该如何构造

我们希望在这里介绍 feedback 的设计，以及基于 feedback 设计提出的 benchmark 应该如何评测、benchmark 评测系统对于这个问题的意义是什么。

视觉反馈应当被纳入 feedback 回路。我们所需要做的事情其实只有两件：**前馈的知识库**（让 AI 能泛化理解 UIprefab 的生成逻辑与我们的生成工具，这是相比于其他问题的核心——见 Stage3.4）；以及**完整的 feedback**，允许 AI 在生成过程中获得不同程度、不同层级的反馈。

### 4.1 feedback 设计——四层金字塔（自上而下、最便宜优先）

核心原则有两条（详见 docs/design/feedback.md）：

> **"视觉无法直接校验节点结构。"** 一张渲染图是 UI 树坍缩后的二维投影——两份结构完全不同的 prefab（按钮挂面板下 vs 挂根节点下）可以渲染出几乎一样的像素。渲染**丢掉了我们最关心的信息**（层级、分组、复用、语义角色）。

> **"反馈必须分层；每一类错误都在它「最便宜」的层，被对应的手段抓住。"** 所以**不要把一切押在视觉 diff**。

两个维度排布反馈：**成本**（能不渲染就不渲染、能在生成前就不等生成后、能用确定性数学就不叫模型）+ **结构敏感度**（越早的层越能直接看见结构，越晚的视觉层只看见像素）。

```
第0层 生成前人审 ir.md        免费、生成前、最靠前          抓语义错（漏整块/装饰当按钮/分组错）
第1层 确定性自动反馈           无模型/无渲染/纯读 JSON   ←重心  抓合法性/引用断裂/字段损坏/布局降解链断
第2层 LLM-as-Judge           读 IR/node_properties/预览  语义主战场（分组/控件降级/复用/chrome 归因/锚点意图）
第3层 VLM 视觉裁判             需渲染、用途收窄            封顶 minor、不得 BLOCK，纯视觉问题兜底
```

- **第1层是投入重心**：合法性七查（白名单 / element_ref 解析 / 命名 / JSON 修复 / 100% 图层覆盖 / 资源完整 / 经验规则）、几何 sanity（零尺寸 / 出画布 / 子越父）、**布局自一致门**（PSD源bbox → 生成节点 computed bbox 恒≈1.0，验证 lowering 数学链未损坏、**非保真度量**）、round-trip diff（载入→重存→diff 抓 emitter 字段损坏）。
- **第3层用途明确收窄**：禁止依赖文字内容/材质/九宫格/blend 的评估（真渲染前标 N/A），只做粗布局/zorder/明显错位兜底，封顶 minor、绝不 BLOCK，绝不替代 P0 编辑器 MCP 最终验收。

#### 错误分诊（每类错误 → 最便宜的拦截层）

| 错误类型 | 最佳拦截层 | 成本/时机 |
|---|---|---|
| 语义错（装饰当按钮、漏整块） | 第0层（人审 + LLM critic） | 最低、生成前 |
| 引用断裂/资源缺失 | 第1层（合法性 + 引用存在性） | 极低、不渲染 |
| Builder 字段损坏 | 第1层（round-trip diff） | 低、重存+diff |
| 布局降解链损坏（region 丢/anchor 改写） | 第1层（自一致门 + 回归报警） | 低、不渲染、确定性 |
| 布局/位置偏移（对设计） | 基准（GT 几何比对，挂 P0） | 低、不渲染、需真同源 gt |
| 分组不合理/控件降级/复用决策 | 第2层（LLM-as-Judge） | 中、不渲染、需模型 |
| 嵌套/层级（半视觉） | 第3层（线框叠加 + VLM） | 中、一次渲染 + VLM |
| 切图/颜色/视觉保真 | 第3层（VLM 定向提问） | 高、需渲染 |

#### 三级反馈的实现状态（用户要研究的三类）

1. **结构反馈（确定性分析函数）**：✅ 已在现系统（`corpus_priors.json` + `score_conformance.py` + 自检中间件）。
2. **经验反馈（LLM 外挂经验 / 语料）**：🎯 规划中（第2层 LLM-as-Judge，对应 R4–R5，尚未落 core）。
3. **视觉反馈（编辑器截图）**：🎯 阻塞于 **P1 数据缺口：nxgui-editor 截图 MCP**（当前 MCP 无截图能力）。

### 4.2 benchmark 评测体系——它对这个问题的意义

> 基准是**反馈的更进一步**。反馈是回路内对**单份产物**判 pass/block；基准是**离线、跨版本、对照真实**地度量系统进步——是指导后续迭代研发的核心、横向"系统是否在变好"的指标。

要回答"整个系统是不是在变好"，需要三个性质：**跨版本可比**（同 scorer 同 case 集，今天 vs 上周直接可比）、**对照真品**（不只问"合法吗"，而问"离人做的真品有多近"——这才是"好"的标准）、**多层**（确定性防回退 + 模型判语义视觉，单一指标不够）。

学术接地（Design2Code 实证）直接塑形了指标重心：真实痛点是**元素召回 + 布局正确性**，不是颜色/文本；视觉 revision prompting 增益有限；**VLM-as-judge 增长最快**。

#### 指标栈

```
确定性回归底座      复用反馈第1层、与 GT 无关        今天可跑，防回退
确定性 GT 几何比对   生成 ↔ 真同源 gt 的 bbox-IoU    挂 P0，真保真主指标
★ LLM-as-Judge     真实 ↔ 生成 .uiprefab 逻辑比对   核心，今天可跑（有真品即可）
★ VLM-as-Judge     真实 ↔ 生成 编辑器预览视觉比对    核心，待截图 MCP（P1）
检索质量评测        recall@k / MRR，38629 golden    今天可跑，验证检索优先 thesis
```

读法：确定性底座防回退、模型裁判量保真度上界、检索层验证 thesis。**LLM-as-Judge 与 VLM-as-Judge 是核心**（与反馈第2、3层共用同一套手）。

#### 数据缺口

- **真同源 PSD↔uiprefab gt = 0（P0）**：trunk 仅 30 个编辑器图标 PSD、不含生产件。pailian 的占位 gt 与其 PSD **非同源**（公告弹窗 vs 拍脸 splash），只能冒烟、**不能算 bbox-IoU**。这挂住了 GT 几何比对（§指标栈第2行）和严格同源的 VLM 视觉比对，但**不挡**确定性回归底座、检索 recall、以及"有真品即可跑"的 LLM 逻辑比对。
- **5 个 benchmark case**（均有 2026-06 同源 run 可作回归素材）：baoxiang（2560×1440 PC，list-reward，seed）/ chengpin（2340×1080 mobile，modal）/ fuwuqi（2560×1440，settings）/ huiyuan（2540×1440，list-reward）/ pailian（1920×1080 PC，splash）——覆盖 panel/splash/settings/list-reward/modal 五种版式。

### 4.3 视觉反馈纳入回路（第3层的具体做法）

视觉反馈不是"渲染图丢给 VLM 做朴素像素 diff"，而是：

- **线框叠加（wireframe overlay）**：把每个节点画成带 label 的彩色 bbox → 结构可视化 → 喂 VLM，让"嵌套/分组合理性"变成半视觉问题。
- **结构化图像指标**：灰度 SSIM、edge-map 比较、VLM-as-judge 定向提问（"标题位置一致吗？有元素缺失吗？有重叠/裁切吗？"），而非朴素像素 diff。
- **数据前置**：依赖 nxgui-editor 截图 MCP（P1 缺口）；离线 composite 预览有系统性失真（九宫格/blend/字形），case metadata 须记 `视觉信号来源=composite|editor_render`，回归只允许同源比较。
- **底线**：第3层封顶 minor、不得 BLOCK，**绝不替代 P0 编辑器 MCP 最终验收**。

### 4.4 服务封装化

服务封装化的目标是**减少 UIP 使用工具的难度、优化迭代反馈通道**：

- 将能力变成 Skill **放到 UIP 的电脑上去跑，不如放到一台锁定环境的机器上跑**——锁定环境天然规避了 v8 那种"强依赖个人机器 trunk + 环境变量 + GBK 编码"的脆弱性（这正是工程化重构已经验证过的方向）。
- 预览需要**编辑器环境**，并要把产物放到对应的文件夹、处理好路径——这部分对应 P0 编辑器 MCP 门禁与第3层视觉反馈的落地，是封装时必须一起解决的环境依赖。