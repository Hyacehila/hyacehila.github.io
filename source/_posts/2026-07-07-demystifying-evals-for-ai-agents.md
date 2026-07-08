---
title: "拆解 AI Agent 的 Eval — Anthropic"
title_en: "Demystifying evals for AI agents — Anthropic"
date: 2026-07-07 23:30:00 +0800
categories: ["Agent Systems", "Agent Training"]
tags: ["Agent Evaluation", "Evals", "AI Engineering", "LLM"]
author: Hyacehila
excerpt: "让 agent 变得有用的那些能力，也让它们变得难以 eval。跨部署行之有效的策略会组合多种技术，以匹配它们所衡量系统的复杂度。"
excerpt_en: "The capabilities that make agents useful also make them difficult to evaluate. The strategies that work across deployments combine techniques to match the complexity of the systems they measure."
mathjax: false
---

> 本文转载翻译自 Anthropic Engineering Blog 于 2026 年 1 月 9 日发布的 [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)。这篇文章在今年 1 月发布时我便收藏了，直到 7 月真正动手构建一套 agent eval 系统时才仔细通读全文，读完后深感质量极高——从 eval 的基础概念、不同 agent 类型的 eval 策略，到从零搭建 eval suite 的实操路线图，再到 eval 与其他质量手段（production monitoring、A/B testing 等）的配合关系，体系完整且处处有实战洞见。这里留个记录方便自己后面随时查验。正文只进行了翻译没有进行删改，技术术语不作翻译保证可读性，图片直接引用了原文 CDN 地址，链接均保持原样。

## Introduction

良好的 eval 能帮助团队更有信心地交付 AI agent。没有 eval，团队很容易陷入被动循环——问题只在生产环境中才被发现，修复一个故障又引发另一个。Eval 让问题和行为变化在影响用户之前就暴露出来，而且它们的价值会在 agent 的整个生命周期中不断累积。

正如我们在 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 中所描述的，agent 会跨越多个 turn 运作：调用 tool、修改 state、根据中间结果进行调整。正是这些让 AI agent 变得有用的能力——自主性、智能和灵活性——也让它们变得更难 eval。

通过我们内部的工作以及与处于 agent 开发前沿的客户的合作，我们学会了如何为 agent 设计更严谨、更有用的 eval。以下是在各种 agent 架构和真实部署场景中被验证有效的做法。

## The structure of an evaluation

一个 evaluation（"eval"）是对 AI 系统的一次测试：给 AI 一个输入，然后对其输出应用 grading 逻辑来衡量是否成功。在本文中，我们聚焦于可以在开发过程中无需真实用户即可运行的自动化 eval。

Single-turn eval 非常直接：一个 prompt、一个 response、以及 grading 逻辑。对于早期的 LLM，single-turn、非 agent 的 eval 是主要的评估方法。随着 AI 能力的进步，multi-turn eval 变得越来越普遍。

![simple eval vs multi-turn eval 对比图](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2Fbd42e7b2f3e9bb5218142796d3ede4816588dec0-4584x2834.png&w=3840&q=75)

在一个简单 eval 中，agent 处理一个 prompt，grader 检查输出是否符合预期。而在更复杂的 multi-turn eval 中，一个 coding agent 会接收 tool、一个 task（比如构建一个 MCP server）和一个环境，执行 "agent loop"（tool call 和 reasoning），并在环境中更新实现。Grading 随后使用 unit test 来验证 MCP server 是否正常工作。

Agent eval 则更加复杂。Agent 会在多个 turn 中使用 tool，修改环境中的 state 并不断调整——这意味着错误可能传播和累积。前沿模型还可能找到超越静态 eval 限制的创造性解决方案。例如，Opus 4.5 在解决一个 τ2-bench 的机票预订问题时，发现了 policy 中的一个漏洞。按照 eval 的书面标准它"失败"了，但实际上它为用户找到了一个更好的解决方案。

在构建 agent eval 时，我们使用以下定义：

- 一个 **task**（也称为 problem 或 test case）是一个具有明确定义的输入和成功标准的单次测试。
- 对每个 task 的每次尝试称为一次 **trial**。由于 model 的输出在不同运行之间会变化，我们会运行多次 trial 以产生更一致的结果。
- 一个 **grader** 是对 agent 表现的某些方面进行评分的逻辑。一个 task 可以有多个 grader，每个 grader 包含多个 assertion（有时称为 check）。
- 一个 **transcript**（也称为 trace 或 trajectory）是一次 trial 的完整记录，包括输出、tool call、reasoning、中间结果以及所有其他交互。对于 Anthropic API 而言，这就是 eval 运行结束时的完整 messages 数组——包含所有对 API 的调用和所有返回的 response。
- **Outcome** 是 trial 结束时环境中的最终 state。一个机票预订 agent 可能在 transcript 末尾说"您的航班已预订"，但 outcome 是环境的 SQL 数据库中是否真的存在该预订记录。
- 一个 **evaluation harness** 是端到端运行 eval 的基础设施。它提供指令和 tool，并发运行 task，记录所有步骤，对输出进行 grading，并汇总结果。
- 一个 **agent harness**（或 scaffold）是使 model 能够作为 agent 运行的系统：它处理输入，编排 tool call，并返回结果。当我们 eval "一个 agent"时，我们 eval 的是 harness 和 model 共同工作的效果。例如，Claude Code 是一个灵活的 agent harness，我们通过 Agent SDK 使用其核心原语来构建我们的 long-running agent harness。
- 一个 **evaluation suite** 是一组旨在衡量特定能力或行为的 task。Suite 中的 task 通常共享一个广泛的目标。例如，一个 customer support eval suite 可能测试退款、取消和升级场景。

![agent eval 组件图](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F0205b36f9639fc27f2f6566f73cb56b06f59d555-4584x2580.png&w=3840&q=75)

## Why build evaluations?

当团队刚开始构建 agent 时，通过人工测试、dogfooding 和直觉的组合，他们可以推进得相当远。更严格的 eval 甚至可能被视为拖慢交付速度的额外开销。但在早期原型阶段之后，一旦 agent 上线并开始 scale，没有 eval 的开发方式就会开始崩溃。

临界点通常出现在：用户反馈 agent 在改动后"变差了"，而团队"在盲飞"——除了猜测和检查之外，没有任何方法来验证。缺少 eval 时，debug 是被动的：等待投诉、手动复现、修复 bug，然后寄希望于没有其他东西回退。团队无法区分真正的 regression 和噪声，无法在发布前用数百个场景自动测试变更，也无法衡量改进。

我们见过这个演进过程反复上演。例如，Claude Code 起步时基于 Anthropic 员工和外部用户的反馈进行快速迭代。后来，我们加入了 eval——最初是针对 conciseness 和 file edit 等窄领域，然后是针对 over-engineering 等更复杂的行为。这些 eval 帮助识别问题、指导改进，并聚焦 research-product 协作。结合 production monitoring、A/B test、用户研究等手段，eval 为持续改进 Claude Code 提供了信号。

编写 eval 在 agent 生命周期的任何阶段都是有用的。早期，eval 迫使产品团队明确 agent 的成功意味着什么；后期，eval 帮助维持一致的质量标准。

Descript 的 agent 帮助用户编辑视频，因此他们围绕成功的编辑工作流的三个维度构建 eval：别搞砸、做我要求的事、把它做好。他们从人工 grading 演进到由产品团队定义标准并定期进行人工校准的 LLM grader，现在定期运行两套独立的 suite：一套用于质量 benchmark，一套用于 regression test。Bolt AI 团队则是在已经有了一个广泛使用的 agent 之后才开始构建 eval。在 3 个月内，他们构建了一个 eval 系统：运行他们的 agent 并用 static analysis 对输出进行 grading，使用 browser agent 来测试应用，并使用 LLM judge 来处理 instruction following 等行为。

有些团队在开发之初就创建 eval；另一些则在达到一定规模、eval 成为改进 agent 的瓶颈时才加入。Eval 在 agent 开发初期特别有用，可以用来显式编码预期行为。两个工程师阅读同一份初始 spec，可能会对 AI 应如何处理边缘情况得出不同的理解。一个 eval suite 可以解决这种歧义。无论何时创建，eval 都能加速开发。

Eval 还决定了你能多快采用新的 model。当更强大的 model 发布时，没有 eval 的团队面临数周的测试，而有 eval 的竞争对手可以快速确定 model 的优势，调整 prompt，并在几天内完成升级。

一旦 eval 就位，你就免费获得了 baseline 和 regression test：latency、token 使用量、每个 task 的成本和错误率都可以在一个静态的 task 集合上进行追踪。Eval 还可以成为 product 和 research 团队之间最高带宽的沟通渠道，定义出 researcher 可以优化的指标。显然，eval 的好处远不止追踪 regression 和改进。它们的价值会不断累积，这一点很容易被忽略，因为成本是前期可见的，而收益是后期才显现的。

## How to evaluate AI agents

我们看到当前大规模部署的 agent 主要有几种类型：coding agent、research agent、computer use agent 和 conversational agent。每种类型可能部署在各种各样的行业中，但它们可以使用相似的技术进行 eval。你不需要从零开始发明一套 eval。以下各节描述了针对几种 agent 类型的成熟技术。请以这些方法为基础，然后扩展到你的领域。

### Types of graders for agents

Agent eval 通常组合三种类型的 grader：code-based、model-based 和 human。每个 grader 评估 transcript 或 outcome 的某一部分。有效的 eval 设计的一个关键要素是选择合适的 grader。

#### Code-based graders

| 方法 | 优势 | 劣势 |
|---|---|---|
| String match check（精确、正则、模糊等）<br>Binary test（fail-to-pass、pass-to-pass）<br>Static analysis（lint、类型、安全）<br>Outcome 验证<br>Tool call 验证（使用了哪些 tool、参数）<br>Transcript 分析（turn 数、token 使用） | 快<br>便宜<br>客观<br>可复现<br>易于 debug<br>可验证特定条件 | 对不符合精确匹配模式的有效变体很脆弱<br>缺乏 nuance<br>对某些更主观的 task 能力有限 |

#### Model-based graders

| 方法 | 优势 | 劣势 |
|---|---|---|
| Rubric-based 评分<br>自然语言 assertion<br>Pairwise comparison<br>Reference-based evaluation<br>Multi-judge consensus | 灵活<br>可 scale<br>捕捉 nuance<br>处理开放式 task<br>处理自由格式输出 | 非确定性<br>比 code 更昂贵<br>需要与 human grader 校准以保持准确性 |

#### Human graders

| 方法 | 优势 | 劣势 |
|---|---|---|
| SME 审查<br>Crowdsourced 判断<br>Spot-check 抽样<br>A/B testing<br>Inter-annotator agreement | 黄金标准质量<br>匹配专家用户判断<br>用于校准 model-based grader | 昂贵<br>慢<br>通常需要大规模获取人类专家 |

对于每个 task，评分可以是加权（组合 grader 分数必须达到阈值）、二元（所有 grader 必须通过）或混合方式。

### Capability vs. regression evals

Capability eval（或"quality" eval）问的是："这个 agent 能做好什么？"它们应该从一个较低的 pass rate 开始，瞄准 agent 难以处理的 task，给团队一个可以攀登的山坡。

Regression eval 问的是："agent 是否仍然能处理所有它以前能处理的 task？"它们应该具有接近 100% 的 pass rate。它们防止回退，得分的下降表明某些东西出问题了，需要修复。当团队在 capability eval 上爬坡时，同时运行 regression eval 也很重要，以确保改动不会在其他地方引发问题。

当 agent 上线并优化后，pass rate 很高的 capability eval 可以"毕业"成为 regression suite，持续运行以捕捉任何漂移。曾经衡量"我们能不能做到这个？"的 task，现在衡量的是"我们是否还能可靠地做到这个？"

### 补：关于 Agent 信任与安全评估

信任与安全评估对于进入生产环境的智能体很重要，绝大多数评估都侧重于研究能力，这也是开发者在开发的过程中比较关注的点。信任与安全评估可以作为一种同时包含结果与过程评估的特殊评估角度。

这有助于评估智能体在非理想条件下的可靠性和适应性。这样做是为了避免智能体与系统之间出现不良交互。实际上，当智能体被投入实际应用后，它们可能会面临各种意想不到的考验。因此，确保智能体能够妥善应对这些情况就显得非常重要。

我们关注在恶劣条件下的可靠性。重点关注稳定性（错误处理能力）、安全性（抵御指令注入的能力）以及公平性（减少偏见），以及任何在上线后可能遇到的安全问题。


### Evaluating coding agents

Coding agent 像人类开发者一样编写、测试和调试代码，浏览 codebase 并运行命令。对现代 coding agent 有效的 eval 通常依赖于明确指定的 task、稳定的测试环境以及为生成的代码准备的充分测试。

Deterministic grader 对 coding agent 来说是很自然的，因为软件通常比较直接：代码能跑吗？测试通过了吗？两个广泛使用的 coding agent benchmark——[SWE-bench Verified](https://www.swebench.com/SWE-bench/) 和 [Terminal-Bench](https://www.tbench.ai/)——就遵循这种方法。SWE-bench Verified 给 agent 提供流行 Python 仓库的 GitHub issue，并通过运行 test suite 对解决方案进行 grading；只有当解决方案修复了失败的测试而不破坏现有测试时才算通过。LLM 在这个 eval 上的成绩在短短一年内从 40% 进步到超过 80%。Terminal-Bench 则走了不同的路线：它测试端到端的技术 task，比如从源码构建 Linux 内核或训练一个 ML 模型。

一旦你有一套 pass-or-fail 的测试来验证 coding task 的关键 outcome，通常还需要对 transcript 进行 grading。例如，基于启发式的 code quality 规则可以超越通过测试这一维度来评估生成的代码，而带有明确 rubric 的 model-based grader 可以评估 agent 如何调用 tool 或与用户交互等行为。

**示例：一个 coding agent 的理论 eval**

考虑一个 coding task，agent 必须修复一个 authentication bypass 漏洞。如下面的示意性 YAML 文件所示，可以使用 grader 和 tracked metric 的组合来评估这个 agent。

```yaml
task:
  id: "fix-auth-bypass_1"
  desc: "Fix authentication bypass when password field is empty and ..."
  graders:
    - type: deterministic_tests
      required: [test_empty_pw_rejected.py, test_null_pw_rejected.py]
    - type: llm_rubric
      rubric: prompts/code_quality.md
    - type: static_analysis
      commands: [ruff, mypy, bandit]
    - type: state_check
      expect:
        security_logs: {event_type: "auth_blocked"}
    - type: tool_calls
      required:
        - {tool: read_file, params: {path: "src/auth/*"}}
        - {tool: edit_file}
        - {tool: run_tests}
  tracked_metrics:
    - type: transcript
      metrics:
        - n_turns
        - n_toolcalls
        - n_total_tokens
    - type: latency
      metrics:
        - time_to_first_token
        - output_tokens_per_sec
        - time_to_last_token
```

请注意，这个示例展示了各种可用 grader 的全貌以供说明。在实践中，coding eval 通常依赖 unit test 来进行正确性验证，并依赖 LLM rubric 来评估整体代码质量，额外的 grader 和 metric 仅按需添加。

### Evaluating conversational agents

Conversational agent 在支持、销售或辅导等领域与用户交互。与传统的 chatbot 不同，它们会维护 state、使用 tool，并在对话中途采取行动。虽然 coding agent 和 research agent 也可能涉及与用户的多轮交互，但 conversational agent 面临一个独特的挑战：**交互本身的质量也是你要 eval 的一部分**。对 conversational agent 有效的 eval 通常依赖于可验证的最终 state outcome，以及能够同时捕捉 task 完成度和交互质量的 rubric。与大多数其他 eval 不同，它们通常需要第二个 LLM 来模拟用户。我们在 [alignment auditing agents](https://alignment.anthropic.com/2025/automated-auditing/) 中使用这种方法，通过扩展的对抗性对话来对 model 进行压力测试。

Conversational agent 的成功可以是多维的：工单解决了吗（state check），它在不超过 10 个 turn 内完成了吗（transcript 约束），语气合适吗（LLM rubric）？两个体现多维度的 benchmark 是 [τ-Bench](https://arxiv.org/abs/2406.12045) 及其继任者 [τ2-Bench](https://arxiv.org/abs/2506.07982)。它们模拟零售支持和机票预订等领域的 multi-turn 交互，其中一个 model 扮演用户角色，agent 则在真实场景中导航。

**示例：一个 conversational agent 的理论 eval**

考虑一个 support task，agent 必须为一位愤怒的客户处理退款。

```yaml
graders:
  - type: llm_rubric
    rubric: prompts/support_quality.md
    assertions:
      - "Agent showed empathy for customer's frustration"
      - "Resolution was clearly explained"
      - "Agent's response grounded in fetch_policy tool results"
  - type: state_check
    expect:
      tickets: {status: resolved}
      refunds: {status: processed}
  - type: tool_calls
    required:
      - {tool: verify_identity}
      - {tool: process_refund, params: {amount: "<=100"}}
      - {tool: send_confirmation}
  - type: transcript
    max_turns: 10
tracked_metrics:
  - type: transcript
    metrics:
      - n_turns
      - n_toolcalls
      - n_total_tokens
  - type: latency
    metrics:
      - time_to_first_token
      - output_tokens_per_sec
      - time_to_last_token
```

与 coding agent 示例一样，这个 task 展示了多种 grader 类型以供说明。在实践中，conversational agent eval 通常使用 model-based grader 来同时评估沟通质量和目标完成度，因为许多 task，比如回答一个问题，可能有多个正确的解决方案。

### Evaluating research agents

Research agent 收集、综合和分析信息，然后产生诸如答案或报告之类的输出。与 coding agent 可以用 unit test 提供二元 pass/fail 信号不同，research 的质量只能相对于 task 来判断。什么是"全面"、"来源可靠"甚至"正确"，取决于上下文：一份市场扫描、一份收购尽职调查报告和一份科学报告各自需要不同的标准。

Research eval 面临独特的挑战：专家可能对一份综合报告是否全面存在分歧，ground truth 会随着参考内容的不断变化而漂移，更长、更开放式的输出为错误留下了更多空间。例如，[BrowseComp](http://arxiv.org/abs/2504.12516) 这个 benchmark 测试 AI agent 是否能在开放网络的"大海捞针"中找到答案——这些问题被设计为容易验证但难以解决。

构建 research agent eval 的一个策略是组合多种 grader 类型。Groundedness check 验证声明是否有检索到的来源支撑，coverage check 定义一个好的答案必须包含的关键事实，source quality check 确认所引用的来源是权威的，而不仅仅是检索到的第一条结果。对于有客观正确答案的 task（"X 公司第三季度的收入是多少？"），exact match 是可行的。一个 LLM 可以标记缺乏支撑的声明和覆盖范围的缺口，但同时也验证开放式综合报告的连贯性和完整性。

鉴于 research 质量的主观性，基于 LLM 的 rubric 应经常与专家人工判断进行校准，以有效地对这类 agent 进行 grading。

### Computer use agents

Computer use agent 通过与人类相同的界面与软件交互——截图、鼠标点击、键盘输入和滚动——而不是通过 API 或代码执行。它们可以使用任何具有图形用户界面（GUI）的应用程序，从设计工具到遗留企业软件。Eval 需要在真实或沙盒环境中运行 agent，让它使用软件应用程序，并检查它是否达到了预期的 outcome。例如，[WebArena](https://arxiv.org/abs/2307.13854) 测试基于浏览器的 task，使用 URL 和页面 state check 来验证 agent 是否正确导航，同时对于修改数据的 task 进行后端 state 验证（确认订单确实被下了，而不仅是确认页面出现了）。[OSWorld](https://os-world.github.io/) 将这一点扩展到完整的操作系统控制，使用在 task 完成后检查各种产物的 eval 脚本：文件系统 state、应用程序配置、数据库内容和 UI 元素属性。

Browser use agent 需要在 token 效率和 latency 之间取得平衡。DOM-based 交互执行快速但消耗大量 token，而 screenshot-based 交互较慢但 token 效率更高。例如，让 Claude 总结 Wikipedia 时，从 DOM 中提取文本更高效；而在 Amazon 上找一个新的笔记本电脑包时，截图更高效（因为提取整个 DOM 非常消耗 token）。在我们的 Claude for Chrome 产品中，我们开发了 eval 来检查 agent 是否为每种上下文选择了正确的 tool。这使我们能更快、更准确地完成基于浏览器的 task。

### How to think about non-determinism in evaluations for agents

无论 agent 类型如何，agent 的行为在不同 run 之间会变化，这使得 eval 结果比乍看起来更难解读。每个 task 有自己的成功率——可能在某个 task 上是 90%，另一个是 50%——而一次 eval 运行中通过的 task 可能在下次运行中失败。有时，我们想要衡量的是 agent 在某个 task 上成功的频率（trial 的占比）。

两个指标有助于捕捉这种 nuance：

**pass@k** 衡量 agent 在 k 次尝试中至少得到一个正确解的概率。随着 k 增加，pass@k 得分上升：更多的"射门机会"意味着至少一次成功的概率更高。50% 的 pass@1 分数意味着 model 在 eval 中一半的 task 上第一次尝试就能成功。在 coding 中，我们通常最关心 agent 在第一次尝试时就找到解决方案——pass@1。在其他情况下，提出多个解决方案是可以接受的，只要有一个有效即可。

**pass^k** 衡量所有 k 次 trial 都成功的概率。随着 k 增加，pass^k 下降，因为要求更多 trial 的一致性是一个更难达成的标准。如果你的 agent 每次 trial 的成功率是 75%，而你运行 3 次 trial，那么通过所有三次的概率是 (0.75)³ ≈ 42%。这个指标对面向客户的 agent 特别重要，因为用户期望每次都是可靠的行为。

![pass@k 和 pass^k 的 divergence 图](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F3ddac5be07a0773922ec9df06afec55922f8194a-4584x2580.png&w=3840&q=75)

pass@k 和 pass^k 随 trial 数量增加而分道扬镳。在 k=1 时，它们是相同的（都等于每次 trial 的成功率）。到 k=10 时，它们讲述了相反的故事：pass@k 趋近 100%，而 pass^k 下降到 0%。

两个指标都有用，使用哪一个取决于产品需求：对于一次成功就足够的 tool，用 pass@k；对于一致性至关重要的 agent，用 pass^k。

## Going from zero to one: a roadmap to great evals for agents

本节给出了我们从实践中总结的、经过实战检验的建议，帮助你从没有 eval 走向拥有可信任的 eval。把这看作 eval-driven agent development 的路线图：早早定义成功、清晰地衡量它、并持续迭代。

### Collect tasks for the initial eval dataset

**Step 0. Start early**

我们看到很多团队推迟构建 eval，因为他们认为需要数百个 task。实际上，20-50 个从真实失败中提取的简单 task 就是一个很好的起点。毕竟，在 agent 开发的早期，每次对系统的改动通常都有明显、可察觉的影响，这种大的 effect size 意味着小样本量就够了。更成熟的 agent 可能需要更大、更困难的 eval 来检测更小的效果，但一开始最好采用 80/20 的方法。你等待的时间越长，构建 eval 就越困难。早期，产品需求会自然地转化为 test case。等太久，你就得从一个线上系统中逆向工程出成功标准。

**Step 1. Start with what you already test manually**

从你在开发过程中已经进行的那些人工检查开始——你在每次发布前验证的行为，以及最终用户尝试的常见 task。如果你已经在生产环境中了，看看你的 bug tracker 和 support queue。将用户报告的失败转化为 test case 可以确保你的 suite 反映真实使用情况；按用户影响优先级排序，可以帮助你将精力投入到最值得的地方。

**Step 2: Write unambiguous tasks with reference solutions**

把 task 质量做对，比看起来要难得多。一个好的 task 是这样的：两个领域专家会独立地得出相同的 pass/fail 结论。他们自己能通过这个 task 吗？如果不能，这个 task 就需要改进。Task spec 中的歧义会变成指标中的噪声。同样的原则也适用于 model-based grader 的标准：模糊的 rubric 会产生不一致的判断。

每个 task 应该能被一个正确遵循指令的 agent 通过。这一点可能很微妙。例如，在审计 Terminal-Bench 时发现，如果某个 task 要求 agent 写一个脚本但没有指定文件路径，而测试假定脚本在某个特定文件路径，那 agent 可能不是它自己的错就失败了。Grader 检查的每一样东西都应该从 task 描述中清晰可见；agent 不应该因为模糊的 spec 而失败。对于前沿 model，在许多 trial 中 0% 的 pass rate（即 0% pass@100）最常意味着是一个有问题的 task，而不是一个无能的 agent——这是你应该重新检查 task spec 和 grader 的信号。对于每个 task，创建一个 reference solution 是有用的：一个已知能通过所有 grader 的工作输出。这证明了 task 是可解的，并验证 grader 配置正确。

**Step 3: Build balanced problem sets**

同时测试某种行为应该发生和不应该发生的情况。片面的 eval 会导致片面的优化。例如，如果你只测试 agent 是否在应该搜索的时候搜索，你可能最终得到一个几乎对什么都搜索的 agent。尽量避免 [class-imbalanced eval](https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets)。我们在为 Claude.ai 构建 web search 的 eval 时亲身体验了这一点。挑战在于：既要防止 model 在不该搜索时搜索，又要保留其在适当情况下做广泛研究的能力。团队构建了覆盖两个方向的 eval：model 应该搜索的查询（如查找天气），以及应该从已有知识回答的查询（如"谁创立了苹果公司？"）。在 undertriggering（该搜时不搜）和 overtriggering（不该搜时搜了）之间找到恰当的平衡非常困难，需要经过多轮的 prompt 和 eval 修改。随着更多示例问题的出现，我们继续向 eval 添加内容以提高覆盖率。

### Design the eval harness and graders

**Step 4: Build a robust eval harness with a stable environment**

至关重要的是，eval 中的 agent 行为方式与生产环境中使用的 agent 大致相同，且环境本身不应引入额外的噪声。每次 trial 应该是"隔离的"——从一个干净的环境开始。运行之间不必要的共享 state（残留文件、缓存数据、资源耗尽）可能导致由于基础设施的不稳定性而非 agent 表现所致的相关失败。共享 state 也可能人为抬高表现。例如，在一些内部 eval 中，我们观察到 Claude 通过检查之前 trial 的 git history 在某些 task 上获得了不公平的优势。如果多个独立的 trial 由于环境中相同的限制（如有限的 CPU 内存）而失败，这些 trial 就不是独立的，因为它们受同一因素影响，eval 结果就变得不可靠，无法衡量 agent 的表现。

**Step 5: Design graders thoughtfully**

如上所述，优秀的 eval 设计涉及为 agent 和 task 选择最佳的 grader。我们建议在可能的情况下选择 deterministic grader，在必要或需要额外灵活性时使用 LLM grader，并审慎地使用 human grader 进行额外的验证。

有一种常见的直觉是检查 agent 是否按照非常具体的步骤——比如以正确的顺序进行一串 tool call——来执行。我们发现这种方法过于僵化，导致测试过于脆弱，因为 agent 经常会找到 eval 设计者没有预料到的有效方法。为了不无谓地惩罚创造性，通常更好的做法是对 agent 产出的东西进行 grading，而不是对它走的路径进行 grading。

对于有多个组件的 task，要引入 partial credit。一个正确识别问题并验证了客户身份，但未能处理退款的 support agent，比一个立刻失败的 agent 有意义的更好。在结果中体现这种成功的连续性很重要。

Model grading 通常需要仔细迭代来验证准确性。LLM-as-judge 应该与 human expert 紧密校准，以建立对 human grading 和 model grading 之间没有显著分歧的信心。为了避免幻觉，给 LLM 一个"出路"，比如提供一条指示，当它没有足够信息时返回 "Unknown"。另外有帮助的是：创建清晰、结构化的 rubric 来对 task 的每个维度进行 grading，然后用独立的 LLM-as-judge 分别对每个维度进行 grading，而不是用一个来 grading 所有维度。一旦系统稳健，只需偶尔使用 human review 就足够了。

一些 eval 存在微妙的 failure mode，即使在 agent 表现良好的情况下也会导致低分——agent 由于 grading bug、agent harness 约束或歧义而无法解决 task。即使是老练的团队也可能错过这些问题。例如，Opus 4.5 在 CORE-Bench 上最初得分是 42%，直到一位 Anthropic researcher 发现了多个问题：僵化的 grading 将 "96.12" 判定为错误而期望 "96.124991……"，模糊的 task spec，以及无法精确复现的随机 task。在修复 bug 并使用更少约束的 scaffold 后，Opus 4.5 的分数跃升至 [95%](https://x.com/sayashk/status/1996334941832089732)。类似地，METR 在其 time horizon benchmark 中发现了几个配置错误的 task：它们要求 agent 优化到某个声明的分数阈值，但 grading 却要求超过该阈值。这惩罚了像 Claude 这样遵循指令的 model，而忽略声明目标的 model 反而得到了更好的分数。仔细双重检查 task 和 grader 可以避免这些问题。

让你的 grader 对绕过或 hack 具有抵抗力。Agent 不应该能够轻易地"作弊"通过 eval。Task 和 grader 应该被设计为使通过 eval 真正需要解决问题，而不是利用意想不到的漏洞。

### Maintain and use the eval long-term

**Step 6: Check the transcripts**

除非你阅读了大量 trial 的 transcript 和 grade，否则你不会知道你的 grader 是否工作良好。在 Anthropic，我们投资构建了查看 eval transcript 的工具，并且我们定期花时间阅读它们。当一个 task 失败时，transcript 告诉你 agent 是真的犯了错误，还是你的 grader 拒绝了一个有效的解决方案。它还经常揭示出 agent 和 eval 行为的关键细节。

失败应该看起来是公平的：agent 哪里做错了、为什么错，应该是清楚的。当分数不上升时，我们需要有信心是 agent 表现的原因，而不是 eval 的原因。阅读 transcript 是你验证 eval 确实在衡量真正重要的东西的方式，也是 agent 开发中的一项关键技能。

**Step 7: Monitor for capability eval saturation**

一个 100% 的 eval 可以追踪 regression，但无法提供改进的信号。Eval saturation 发生在 agent 通过了所有可解的 task，没有留下改进空间的时候。例如，SWE-bench Verified 的分数今年年初从 30% 起步，而前沿 model 现在正接近饱和，超过 80%。随着 eval 接近饱和，进展也会放缓，因为只剩下最困难的 task。这可能使结果具有欺骗性，因为巨大的能力提升表现为分数的微小增加。例如，code review 创业公司 Qodo 最初对 Opus 4.5 没什么印象，因为他们的 one-shot coding eval 没有捕捉到在更长、更复杂的 task 上的提升。作为回应，他们开发了一个新的 agentic eval 框架，为进展提供了更清晰的图景。

作为原则，在有深入挖掘 eval 细节并阅读一些 transcript 之前，我们不会将 eval 分数视为表面值。如果 grading 不公平、task 模糊、有效的解决方案被惩罚，或者 harness 限制了 model，那么 eval 应该被修订。

**Step 8: Keep evaluation suites healthy long-term through open contribution and maintenance**

一个 eval suite 是一个活的工件，需要持续的关注和明确的 ownership 才能保持有用。

在 Anthropic，我们试验了各种 eval 维护方法。最有效的方式是建立专门的 eval 团队来拥有核心基础设施，而领域专家和产品团队贡献大部分 eval task 并自行运行 eval。

对于 AI 产品团队来说，拥有和迭代 eval 应该像维护 unit test 一样常规。团队可能在"在早期测试中有效"但未能满足未明确说明的期望的 AI 功能上浪费数周——而一个设计良好的 eval 本可以早早揭示这些期望。定义 eval task 是压力测试产品需求是否足够具体到可以开始构建的最佳方式之一。

我们推荐实践 eval-driven development：在 agent 能够满足预期的能力之前就构建 eval 来定义这些能力，然后迭代直到 agent 表现良好。在内部，我们经常构建今天"够好"的功能，但这些功能实际上是对 model 几个月后能力的押注。以低 pass rate 起步的 capability eval 让这一点变得可见。当新 model 发布时，运行 suite 能快速揭示哪些押注得到了回报。

最接近产品需求和用户的人是定义成功的最佳人选。以当前的 model 能力，产品经理、客户成功经理或销售人员可以使用 Claude Code 以 PR 的形式贡献一个 eval task——让他们做吧！或者，更好的是，主动赋能他们。

![创建有效 eval 的过程图](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F0db40cc0e14402222a179fc6297b9c8818e97c8a-4584x2580.png&w=3840&q=75)

## How evals fit with other methods for a holistic understanding of agents

自动化 eval 可以对 agent 运行数千个 task，而无需部署到生产环境或影响真实用户。但这只是理解 agent 表现的众多方法之一。完整的图景还包括 production monitoring、user feedback、A/B testing、manual transcript review 和 systematic human evaluation。

| 方法 | 优势 | 劣势 |
|---|---|---|
| **Automated evals**：无需真实用户即可编程运行测试 | 更快的迭代<br>完全可复现<br>不影响用户<br>可以在每次 commit 上运行<br>无需生产部署即可大规模测试场景 | 需要更多的前期投资来构建<br>随着产品和 model 演变需要持续维护以避免漂移<br>如果不匹配真实使用模式，可能产生虚假的信心 |
| **Production monitoring**：追踪线上系统中的指标和错误 | 揭示真实用户的大规模行为<br>捕捉 synthetic eval 遗漏的问题<br>提供 agent 实际表现的 ground truth | 被动的；问题在你知道之前就已经到达用户<br>信号可能有噪声<br>需要 instrumentation 投资<br>缺乏用于 grading 的 ground truth |
| **A/B testing**：用真实用户流量比较变体 | 衡量真实的用户 outcome（留存、task 完成率）<br>控制混杂因素<br>可 scale 且系统化 | 慢；需要数天或数周才能达到显著性，且需要足够的流量<br>只测试你部署的变更<br>在不仔细审查 transcript 的情况下，对指标变化的底层"为什么"信号较少 |
| **User feedback**：显式信号，如 thumbs-down 或 bug report | 暴露你没有预料到的问题<br>带有真实人类用户的真实示例<br>反馈通常与产品目标相关 | 稀疏且自选择<br>偏向严重问题<br>用户很少解释为什么某些东西失败了<br>非自动化<br>主要依赖用户来发现问题可能产生负面的用户影响 |
| **Manual transcript review**：人类阅读 agent 对话记录 | 建立对 failure mode 的直觉<br>捕捉自动化检查遗漏的细微质量问题<br>帮助校准"好"的标准并把握细节 | 时间密集<br>无法 scale<br>覆盖率不一致<br>审查者疲劳或不同的审查者可能影响信号质量<br>通常只给出定性信号，而非清晰的定量 grading |
| **Systematic human studies**：由受过训练的评估者对 agent 输出进行结构化 grading | 来自多个人类评估者的黄金标准质量判断<br>处理主观或模糊的 task<br>为改进 model-based grader 提供信号 | 相对昂贵且周转慢<br>难以频繁运行<br>Inter-rater 分歧需要调和<br>复杂领域（法律、金融、医疗）需要人类专家来进行研究 |

这些方法对应 agent 开发的不同阶段。Automated eval 在发布前和 CI/CD 中特别有用，作为抵御质量问题的第一道防线，在每次 agent 变更和 model 升级时运行。Production monitoring 在发布后启动，检测分布漂移和未预料到的真实世界失败。A/B testing 在你拥有足够流量后验证重大变更。User feedback 和 transcript review 是填补空白的持续性实践：持续分类反馈，每周抽样阅读 transcript，并根据需要深入挖掘。将 systematic human studies 留给校准 LLM grader 或评估主观输出——在这些场景中，人类共识作为参考标准。

![Swiss Cheese Model 图](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2Fb77b8dbb7c2e57f063fbc8a087a853d5809b74b0-4584x2580.png&w=3840&q=75)

就像安全工程中的 [Swiss Cheese Model](https://en.wikipedia.org/wiki/Swiss_cheese_model)，没有单层 eval 能捕捉每个问题。当组合多种方法时，穿过一层的失败会被另一层捕捉。

最有效的团队将这些方法组合使用：automated eval 用于快速迭代，production monitoring 用于 ground truth，periodic human review 用于校准。

## Conclusion

没有 eval 的团队会陷入被动的循环——修复一个失败，又创造另一个失败，无法区分真正的 regression 和噪声。早早投入的团队则发现相反的情况：随着失败变成 test case，test case 防止 regression，指标取代猜测，开发速度加快。Eval 给整个团队一个明确的山坡去攀登，将"agent 感觉变差了"转化为可操作的东西。价值会不断累积，但前提是你把 eval 当作核心组件，而不是事后补救。

模式因 agent 类型而异，但本文描述的基本原则是恒定的。尽早开始，不要等待完美的 suite。从你看到的失败中获取真实的 task。定义无歧义、稳健的成功标准。精心设计 grader 并组合多种类型。确保问题对 model 来说足够难。迭代 eval 以提高信噪比。阅读 transcript！

AI agent eval 仍然是一个新兴的、快速演进的领域。随着 agent 承担更长的 task、在 multi-agent 系统中协作、以及处理越来越主观的工作，我们将需要调整我们的技术。随着我们学到更多，我们会继续分享最佳实践。

## Acknowledgements

Written by Mikaela Grace, Jeremy Hadfield, Rodrigo Olivares, and Jiri De Jonghe. We're also grateful to David Hershey, Gian Segato, Mike Merrill, Alex Shaw, Nicholas Carlini, Ethan Dixon, Pedram Navid, Jake Eaton, Alyssa Baum, Lina Tawfik, Karen Zhou, Alexander Bricken, Sam Kennedy, Robert Ying, and others for their contributions. Special thanks to the customers and partners we have learned from through collaborating on evals, including iGent, Cognition, Bolt, Sierra, Vals.ai, Macroscope, PromptLayer, Stripe, Shopify, the Terminal Bench team, and more. This work reflects the collective efforts of several teams who helped develop the practice of evaluations at Anthropic.

## Appendix: Eval frameworks

有几个开源和商业 framework 可以帮助团队在不从零构建基础设施的情况下实施 agent eval。正确的选择取决于你的 agent 类型、现有技术栈，以及你是需要 offline evaluation、production observability，还是两者都需要。

**[Harbor](https://harborframework.com/)** 专为在容器化环境中运行 agent 而设计，提供跨云提供商大规模运行 trial 的基础设施，以及定义 task 和 grader 的标准化格式。流行的 benchmark 如 Terminal-Bench 2.0 通过 Harbor registry 发布，使得运行既有 benchmark 和自定义 eval suite 变得容易。

**[Braintrust](https://www.braintrust.dev/)** 是一个将 offline evaluation 与 production observability 和 experiment tracking 结合起来的平台——对于需要在开发过程中迭代，同时监控生产环境质量的团队很有用。其 `autoevals` 库包含了用于 factuality、relevance 和其他常见维度的预构建 scorer。

**[LangSmith](https://docs.langchain.com/langsmith/evaluation)** 提供 tracing、offline 和 online evaluation，以及 dataset management，与 LangChain 生态紧密集成。**[Langfuse](https://langfuse.com/)** 作为一个自托管的开源替代方案提供了类似的能力，适合有数据驻留需求的团队。

**[Arize](https://arize.com/)** 提供 Phoenix——一个用于 LLM tracing、debugging 和 offline/online evaluation 的开源平台，以及 AX——一个为 scale、optimization 和 monitoring 扩展 Phoenix 的 SaaS 产品。

许多团队组合使用多个 tool，自己构建 eval framework，或者仅仅使用简单的 eval 脚本作为起点。我们发现，虽然 framework 可以成为加速进展和标准化的宝贵方式，但它们的好坏取决于你通过它们运行的 eval task。通常最好的做法是快速选择一个适合你工作流程的 framework，然后将精力投入到 eval 本身——迭代高质量的 test case 和 grader。

## 在正文结束之后

