---
layout: blog-post
title: "科学研究 Skills 盘点：哪些技能库正在把通用 Agent 变成领域研究助手"
date: 2026-03-16 13:40:00 +0800
categories: [智能体系统]
tags: [Agent, Skills, Scientific-Research, Bioinformatics, AI4S]
author: Hyacehila
excerpt: "从 AI Research Skills Library 出发，盘点 Agent Skills、OpenClaw 与 Hugging Face 生态里值得关注的科研 Skills 仓库，并按学科门类与科研阶段做一张更完整的地图。"
featured: false
math: false
---

# 科学研究 Skills 盘点：哪些技能库正在把通用 Agent 变成领域研究助手

我原本想写的，只是 Orchestra Research 的 [AI Research `Skills` Library](https://github.com/Orchestra-Research/AI-Research-SKILLs)：它怎样把 `autoresearch`、微调、评测、部署这些能力打包进 `SKILL.md`，让 agent 更像一个研究助理。

但沿着这条线往外看，很快会发现一件更重要的事：**真正值得注意的，不是某一个 skills 仓库，而是“Skills 正在变成通用 Agent 的领域能力层”。**

它们做的事情并不是训练一个新模型，而是把一整套可以重复使用的领域知识封装起来：

- 什么时候该调用什么工具；
- 常见数据格式和目录结构是什么；
- 文献、数据库、实验和写作怎样串起来；
- 遇到报错、版本差异、评测口径冲突时，先看什么、再查什么；
- 怎样让流程可复现、可回放、可交给别人继续做。

对科研尤其如此。很多时间并不花在“提出想法”本身，而是花在：

- 找文献、筛基线、接数据库；
- 处理训练、仿真、排队、显存、日志、结果回放；
- 把零散实验整理成结论，再整理成论文、报告或 SOP；
- 把一个人的经验，变成团队里任何 agent 都能复用的能力。

所以这篇不再只是一个单仓库入门，而是一篇更完整的盘点：**截至 2026 年 3 月 18 日，今天有哪些正式的 Skills 仓库，正在把通用 Agent 往 AI 研究、生物医学、临床、材料仿真、AI4S 这些具体科研方向推。**

---

## 1. 我这次到底在盘点什么？

这篇只纳入三类东西：

1. **正式 Skills 仓库**：仓库里明确以 `SKILL.md` 为核心单元组织能力。
2. **正式的技能分发/注册/桥接生态**：它们本身不一定生产科研内容，但定义了 Skills 怎样被安装、发布、发现或转接。
3. **明确面向科研、科学计算、医学、生物、材料、ML 研究的技能包**。

我**不**把下面这些作为正文主角：

- 纯 prompt 集；
- 纯 MCP 工具目录；
- 只是“某个 agent 产品很强”，但没有公开 skills 结构的项目；
- 没有明确科研/学术场景定位的通用 productivity skill 包。

证据口径也尽量保守：我主要根据各仓库的官方 README、官方技能目录与仓库结构来写；动态数字一律理解成“**截至 2026-03-18 的观察结果**”，而不是长期不变的事实。

---

## 2. 为什么 Skills 特别适合科研？

因为科研场景里，最稀缺的往往不是一句 prompt，而是**带上下文的、可执行的、带排障经验的领域流程**。

从结构上看，不管是 Agent Skills、Claude Skills 还是 OpenClaw skills，它们本质上都在做同一件事：

- 用 `SKILL.md` 提供可被 agent 触发的行动手册；
- 用 `references/`、`scripts/`、`assets/` 把深资料、工具脚本、模板补齐；
- 把“该怎么做”变成可重复调用的能力单元，而不是每次临时搜索。

这对科研特别重要，因为科研里的能力天然就适合被拆成模块：

- 文献与数据库检索；
- 数据清洗与格式转换；
- 实验或仿真的流程编排；
- 模型训练、调参、评测；
- 写作、投稿、复现、合规。

换句话说，**Skills 更像科研 agent 的中间层基础设施**：模型还是那个模型，但它接上了某个学科的数据库、术语、工具链、模板和 workflow，于是它不再只是“会说话的通用 agent”，而开始像一个懂特定领域的研究助手。

---

## 3. 来源地图：科研 Skills 主要从哪几类生态里长出来？

今天这类仓库，大致已经分成四层：

1. **规范/分发生态**：定义 Skills 的开放格式，或者负责安装与分发。
2. **通用科研技能库**：跨学科，或者至少覆盖 AI/ML 研究全流程。
3. **垂直学科技能库**：面向生物医学、材料、地学、临床等某个特定领域。
4. **聚合/桥接层**：把多个技能源打包到某个 agent 生态里，或把 skills 转成别的接口。

下面这张表，是我觉得现在最值得记住的一张“来源总表”。

## 4. 仓库来源总表

| 仓库 / 生态 | 类型 | 维护方 / 来源 | 主要学科 | 主要覆盖阶段 | 我怎么看 |
| --- | --- | --- | --- | --- | --- |
| [Agent Skills](https://github.com/agentskills/agentskills) | 规范/分发生态 | `agentskills` / 官方规范站点 | 多学科基础设施 | 发现、复用、跨 agent 兼容 | 它不是科研内容库，但给“技能可移植”这件事提供了开放格式。 |
| [vercel-labs/skills](https://github.com/vercel-labs/skills) | 规范/分发生态 | Vercel Labs | 多学科基础设施 | 安装、分享、分发 | 更像 Skills 生态里的 `npm`/CLI 层，而不是领域知识层。 |
| [openclaw/clawhub](https://github.com/openclaw/clawhub) | 规范/分发生态 | OpenClaw 相关社区 | 多学科基础设施 | 注册、搜索、发布 | Claw 系生态里的公共技能目录，面向 `SKILL.md` 注册与发现。 |
| [Orchestra-Research/AI-Research-SKILLs](https://github.com/Orchestra-Research/AI-Research-SKILLs) | 通用科研技能库 | Orchestra Research | AI/ML 研究 | 选题、训练、评测、写作、部署 | 目前最像“AI 研究技能树”的原生仓库。 |
| [K-Dense-AI/claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills) | 通用科研技能库 | K-Dense | 生物、医学、化学、材料、地学、金融 | 检索、数据、分析、写作、部分建模 | 跨学科范围最广，像一个科学数据库与工作流技能总库。 |
| [huggingface/skills](https://github.com/huggingface/skills) | 通用科研技能库 | Hugging Face | AI/ML 研究 | 数据集、训练、评测、论文发布 | 更偏平台工作流，把 Hub 与研究步骤打通。 |
| [ClawBio/ClawBio](https://github.com/ClawBio/ClawBio) | 垂直学科技能库 | ClawBio | 生物信息学、基因组学、单细胞、药物基因组学 | 数据获取、分析管线、复现、文献综合 | 我见到的最像“bioinformatics-native” 的原生技能库。 |
| [FreedomIntelligence/OpenClaw-Medical-Skills](https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills) | 聚合型垂直技能包 | Freedom Intelligence / OpenClaw 生态 | 临床、基因组学、药物发现、医学设备 | 检索、临床文书、omics 管线、合规 | 更像一个医疗超大集合包，价值在聚合，不完全在原创。 |
| [HeshamFS/materials-simulation-skills](https://github.com/HeshamFS/materials-simulation-skills) | 垂直学科技能库 | Hesham F. S. | 材料科学、数值模拟、HPC | 建模、仿真、验证、作业提交 | 这是我最看重的 AI4S / 传统科学计算例子。 |
| [biocontext-ai/skill-to-mcp](https://github.com/biocontext-ai/skill-to-mcp) | 聚合/桥接层 | BioContextAI | 生物医学社区优先，但本身通用 | 技能桥接、系统集成 | 它提醒我们：Skills 和 MCP 不是替代关系，而是可以互转。 |

如果用一句话概括这张表，就是：

> **`agentskills` / `skills.sh` / `ClawHub` 负责让技能“能流通”，`AI-Research-SKILLs` / `Claude Scientific Skills` / `ClawBio` / `materials-simulation-skills` 负责让技能“有内容”，`OpenClaw-Medical-Skills` / `skill-to-mcp` 负责把内容重新打包或桥接到具体生态里。**

---

## 5. 代表仓库逐一看：谁在给 Agent 加什么“科研脑”？

### 5.1 `AI-Research-SKILLs`：最像“AI 研究总控台”的技能树

这是 Orchestra Research 维护的开源仓库。它的 README 把自己定义成：让 agent 能覆盖 **from idea to paper** 的完整 AI 研究生命周期；截至我这次检索，README 的口径是 **86 个 skills、22 个类别**。

它最关键的地方，不只是数量，而是层次很清楚：

- **研究编排层**：`autoresearch`、`research-ideation`、`ml-paper-writing`
- **研究工程层**：`axolotl`、`peft`、`trl`、`lm-evaluation-harness`、`vllm`、`sglang`、`mlflow`、`weights-and-biases`

我仍然觉得它最值得先看的 skill 是 `autoresearch`。原因不是它最“技术”，而是它最清楚地表达了一个研究 agent 应该怎么推进项目：

- 内循环：选假设 -> 做实验 -> 量化结果 -> 记录；
- 外循环：把实验整理成理解，再决定下一步加深、扩展还是转向。

这也是为什么它依然是我心里“研究流程层”的代表仓库。它解决的不是某一个工具怎么配，而是**怎样把文献、实验、评测和写作连成研究闭环**。

如果你做的是：

- LLM / foundation model 研究；
- 微调、后训练、评测、推理部署；
- 想把“想法 -> 实验 -> 论文”尽量交给 agent 推进，

那它依然是最该先看的起点。

### 5.2 `claude-scientific-skills`：跨学科范围最广的科研技能总库

如果说 `AI-Research-SKILLs` 更像“AI 研究工程树”，那 K-Dense 的 [Claude Scientific Skills](https://github.com/K-Dense-AI/claude-scientific-skills) 更像“**科学数据库 + 学科 workflow + 写作工具**”的超大合集。

截至这次检索，README 的口径是：

- **170+ ready-to-use scientific and research skills**
- **250+ databases**
- 面向 Cursor、Claude Code、Codex 等支持 Agent Skills 标准的 agent

它最有意思的，不是某一个爆款 skill，而是**覆盖面**。README 明确列出了：

- 生物信息学与基因组学；
- 化学信息学与药物发现；
- 蛋白质组学与质谱；
- 临床研究与精准医疗；
- 医疗 AI 与 clinical ML；
- 材料科学与化学；
- 物理与天文学；
- 地理空间科学与遥感。

如果只挑几个代表技能，我会记这些名字：

- `anndata`、`biopython`、`cellxgene-census`
- `chembl-database`
- `clinical-decision-support`
- `scientific-writing`、`peer-review`、`venue-templates`
- `pymatgen`
- `geomaster`

也就是说，它做的不是“替你成为某个单一领域专家”，而是给 agent 一套**跨学科检索、分析、写作与数据库接入能力**。如果你的研究不是纯 LLM 工程，而是会横跨生物、化学、临床、材料、地学这些不同子领域，那这个仓库的意义甚至可能比 `AI-Research-SKILLs` 更大。

### 5.3 `huggingface/skills`：把 AI/ML 研究流程直接接到 Hub

Hugging Face 的 [skills](https://github.com/huggingface/skills) 不是“科学学科最全”的那种仓库，但它非常重要，因为它说明了一件事：

> 平台型公司也开始把数据集创建、训练、评测、发布论文这些事情，直接封装成公开 Skills。

README 对它的定义很直接：**Skills are definitions for AI/ML tasks like dataset creation, model training, and evaluation.**

截至这次查看，仓库里的主技能包包括：

- `hugging-face-datasets`
- `hugging-face-dataset-viewer`
- `hugging-face-model-trainer`
- `hugging-face-vision-trainer`
- `hugging-face-evaluation`
- `hugging-face-paper-publisher`
- `hugging-face-trackio`

它的独特价值不在“学科知识特别深”，而在于**把研究流程和 Hugging Face 的平台基础设施绑在了一起**：

- 数据集可以直接进入 Hub 视图和版本管理；
- 训练与评测可以沿着平台规范走；
- 论文、模型、数据、工具的发布链条是连着的。

所以它更像一个**AI/ML 平台工作流技能库**。如果你的研究工作已经大量依赖 Hugging Face 生态，这类 skill 包的实际价值会很高。

### 5.4 `ClawBio`：最像“生物信息学原生技能库”的例子

[ClawBio](https://github.com/ClawBio/ClawBio) 的 README 直接把自己叫做：

> **The first bioinformatics-native AI agent skill library.**

这句话我觉得非常关键。很多所谓的“科研 agent”其实还是通用 agent 套一个医学 prompt；但 ClawBio 走的是完全不同的方向：它把生物信息学里已经非常明确的工具链、数据类型和分析步骤，做成了真正可运行的技能。

它当前最值得记住的代表 skills，我会选这几个：

- `gwas-lookup`
- `gwas-prs`
- `rnaseq-de`
- `scrna-embedding`
- `scrna-orchestrator`
- `vcf-annotator`
- `galaxy-bridge`
- `lit-synthesizer`
- `repro-enforcer`

这串名字本身就说明问题了：它不是一个“我也能看论文”的医学聊天机器人，而是试图覆盖：

- GWAS 与 PRS；
- RNA-seq / scRNA-seq；
- 变异注释；
- Galaxy 工具桥接；
- 文献综合；
- 复现打包。

这意味着它已经非常接近真实生信 workflow。尤其 `galaxy-bridge` 这类 skill，直接把 8,000+ Galaxy 工具的搜索、推荐和工作流连接起来；`repro-enforcer` 这种 skill，则把 Conda、容器、Nextflow 一类可复现打包思路带进 agent 行为里。

如果你关心的是“通用 agent 怎样真的进入生信分析，而不是只会解释概念”，那 `ClawBio` 是现在必须单独记住的一条线。

### 5.5 `OpenClaw-Medical-Skills`：医疗垂直领域里的超大聚合包

[OpenClaw-Medical-Skills](https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills) 的 README 口径非常激进：它把自己称为 **The largest open-source medical AI skill library for OpenClaw**，并给出了 **869 curated skills** 的数字。

但我觉得它更值得注意的地方，不是“869” 本身，而是它在 README 里写得很清楚：

- 它覆盖 clinical、genomics、drug discovery、bioinformatics、medical devices；
- 它明确说自己聚合了 **12+ open-source skill repositories**；
- 分类表里能看到 `ClawBio Pipelines`、`BioOS Extended Suite`、`Data Science & Tools` 这样的来源痕迹。

所以，对它最准确的理解不是“又一个原生技能库”，而是：

> **它是一个面向医学研究与临床工作流的聚合型垂直技能包。**

如果只看技能名，你能很快感觉到它的覆盖范围：

- `clinical-decision-support`
- `clinical-reports`
- `autonomous-oncology-agent`
- `variant-interpretation-acmg`
- `drugbank-database`
- `pubmed-search`
- `bio-workflows-scrnaseq-pipeline`
- `bio-workflows-gwas-pipeline`
- `cellagent-annotation`

它的现实价值非常高，因为它把几个本来分散的世界接到了一起：

- 文献与数据库检索；
- 临床文书与决策支持；
- omics 流程；
- 药物与蛋白设计；
- 医疗法规与合规知识。

但也正因为这样，我会把它放在“**聚合型垂直产品包**”而不是“最重要的原生技能库”这一栏。它的价值更多在**把医学领域能力一口气装给 OpenClaw agent**，而不是像 `ClawBio` 那样从零定义一个很鲜明的原生学科路线。

### 5.6 `materials-simulation-skills`：AI4S / 传统科学计算最值得关注的例子

如果说前面的例子大多还停留在 AI/ML、生物医学这两条主线，那么 [materials-simulation-skills](https://github.com/HeshamFS/materials-simulation-skills) 则很像我一直在等的那类仓库：

> **它证明 Skills 完全可以服务于传统科学计算、数值模拟和 AI4S。**

这个仓库的 README 很直白：它是面向 **computational materials science and numerical simulation workflows** 的 Agent Skills。

它的结构也很漂亮，直接分成四组：

- `core-numerical`
- `simulation-workflow`
- `hpc-deployment`
- `ontology`

再往下看，几乎每个名字都很“科研现场”：

- `numerical-stability`
- `mesh-generation`
- `time-stepping`
- `convergence-study`
- `simulation-orchestrator`
- `simulation-validator`
- `performance-profiling`
- `slurm-job-script-generator`
- `ontology-validator`

这类 skill 和 LLM 微调、RLHF 几乎没有关系，但和很多 AI4S 团队、传统计算科学团队的日常高度相关：

- 网格怎么生成；
- 数值稳定性怎样检查；
- 收敛性怎么做；
- Slurm 作业脚本怎么配；
- 仿真结果怎么后处理、怎么验证。

所以如果你问我：**Skills 能不能用于传统机器学习科研，或者 AI4S 的特定领域研究？**

这个仓库本身就是一个很好的回答：**可以，而且这可能正是下一波真正有价值的方向。**

### 5.7 规范、分发与桥接层：为什么它们虽然不“学科”，但仍然重要

除了上面的六个内容库，我还会特别记住四个基础设施层项目：

- [Agent Skills](https://github.com/agentskills/agentskills)：把 Skills 定义成开放格式，核心主张是 **write once, use everywhere**。
- [vercel-labs/skills](https://github.com/vercel-labs/skills)：用 `npx skills` 做安装、分享和 source format 管理，更像 Skills 的 CLI 分发层。
- [ClawHub](https://github.com/openclaw/clawhub)：Claw 系生态里的公共技能注册表，强调发布、版本化、搜索。
- [skill-to-mcp](https://github.com/biocontext-ai/skill-to-mcp)：把 skills 暴露成 MCP 资源，说明 skill 和 MCP 可以桥接而不是互斥。

这些项目的重要性在于：**如果没有它们，领域技能就很难形成真正的生态。**

换句话说，科研 Skills 的竞争并不只发生在“谁写了更懂生信的 skill”，还发生在：

- 技能能不能被不同 agent 复用；
- 技能怎样被安装、更新和注册；
- 技能是否能接到别的协议与平台里。

---

## 6. 按学科门类看：今天哪些 Skills 真正在长出领域能力？

这张表更适合回答一个更接近研究者视角的问题：**如果我来自某个学科，我该先看哪些技能库？**

## 7. 按学科分类表

| 学科门类 | 更值得关注的仓库 | 代表 skills / 能力 | 这些 skills 主要解决什么问题 |
| --- | --- | --- | --- |
| AI/ML 研究 | [AI-Research-SKILLs](https://github.com/Orchestra-Research/AI-Research-SKILLs), [huggingface/skills](https://github.com/huggingface/skills) | `autoresearch`, `research-ideation`, `axolotl`, `trl`, `lm-evaluation-harness`, `hugging-face-model-trainer`, `hugging-face-evaluation` | 选题、训练、评测、部署、论文写作与平台发布。 |
| 生物医学与 omics | [ClawBio](https://github.com/ClawBio/ClawBio), [claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills), [OpenClaw-Medical-Skills](https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills) | `rnaseq-de`, `scrna-embedding`, `vcf-annotator`, `anndata`, `cellxgene-census`, `bio-workflows-scrnaseq-pipeline` | 单细胞、变异、RNA-seq、GWAS、生物数据库与复现 workflow。 |
| 临床研究 / 医学文书 / 公共卫生 | [claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills), [OpenClaw-Medical-Skills](https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills) | `clinical-decision-support`, `clinical-reports`, `clinicaltrials-database`, `pubmed-search`, `autonomous-oncology-agent` | 临床检索、文书生成、治疗支持、试验筛选、病例与知识库连接。 |
| 药物发现 / 化学信息学 | [claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills), [OpenClaw-Medical-Skills](https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills) | `chembl-database`, `bindingdb-database`, `drugbank-database`, `bindcraft`, `alphafold`, `bio-admet-prediction` | 药物库检索、蛋白结构、ADMET、配体-靶点、蛋白设计。 |
| 材料科学 / 数值模拟 / HPC | [materials-simulation-skills](https://github.com/HeshamFS/materials-simulation-skills), [claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills) | `numerical-stability`, `convergence-study`, `simulation-orchestrator`, `slurm-job-script-generator`, `pymatgen` | 数值方法、材料仿真、HPC 提交、后处理、领域本体校验。 |
| 多学科科研基础设施 | [Agent Skills](https://github.com/agentskills/agentskills), [vercel-labs/skills](https://github.com/vercel-labs/skills), [ClawHub](https://github.com/openclaw/clawhub), [skill-to-mcp](https://github.com/biocontext-ai/skill-to-mcp) | 开放格式、安装分发、技能注册、MCP 桥接 | 让技能可发现、可迁移、可接入不同 agent 与不同协议。 |

如果只看这张表，有两个判断我觉得已经很清楚：

- **AI/ML 与生物医学，是今天技能库最成熟的两条主线。**
- **材料/数值模拟/AI4S 已经出现了很有代表性的起点，但规模远没有医学和 AI 研究那么大。**

---

## 8. 按科研阶段看：Skills 到底覆盖了研究流程的哪几段？

光按学科看还不够，因为很多 skill 的价值不是体现在“它属于哪门学科”，而是体现在“它卡住了科研流程的哪一段”。

## 9. 按科研阶段分类表

| 科研阶段 | 代表仓库 / skills | 典型问题 | 现在谁最强 |
| --- | --- | --- | --- |
| 选题与 ideation | `AI-Research-SKILLs` 的 `research-ideation`、`autoresearch` | 研究问题怎样从主题变成可检验假设？ | `AI-Research-SKILLs` 最系统。 |
| 文献检索与综述 | `pubmed-search`, `arxiv-database`, `biorxiv-database`, `lit-synthesizer`, `scientific-writing` | 文献怎样被筛、记、汇总成 narrative？ | `claude-scientific-skills` 和医学生态更强。 |
| 数据获取与整理 | `hugging-face-datasets`, `hugging-face-dataset-viewer`, `cellxgene-census`, `drugbank-database`, `ukb-navigator` | 数据从哪里来、怎么读、怎么转格式？ | HF 与 K-Dense 的数据入口更成熟。 |
| 实验 / 流程编排 | `autoresearch`, `bio-orchestrator`, `scrna-orchestrator`, `simulation-orchestrator`, `galaxy-bridge` | 多步骤 workflow 怎样让 agent 稳定跑？ | `AI-Research-SKILLs`、`ClawBio`、材料库各有代表。 |
| 训练 / 仿真 / 建模 | `axolotl`, `peft`, `trl`, `hugging-face-model-trainer`, `hugging-face-vision-trainer`, `numerical-stability`, `mesh-generation` | 模型训练、数值求解、仿真与调参如何标准化？ | AI 研究仓库和材料仿真仓库分工明显。 |
| 评测 / 验证 / 统计分析 | `lm-evaluation-harness`, `hugging-face-evaluation`, `equity-scorer`, `convergence-study`, `simulation-validator` | 怎么确认结果可比、稳定、可信？ | AI 评测和数值验证各自已经长出专门 skill。 |
| 写作 / 发表 / 汇报 | `ml-paper-writing`, `scientific-writing`, `peer-review`, `venue-templates`, `hugging-face-paper-publisher`, `clinical-reports` | 怎样把结果变成论文、报告、文书？ | K-Dense 与 Orchestra/HF 各有强项。 |
| 复现 / 部署 / 合规 / 可观测性 | `mlflow`, `weights-and-biases`, `vllm`, `repro-enforcer`, `slurm-job-script-generator`, 医疗法规类 skills, `skill-to-mcp` | 怎样把实验回放、交接、部署、纳入合规工作流？ | 这里最分散，但也最接近真实生产。 |

这张表其实说明了另一个现实：

> **今天最成熟的技能库，已经不只是“帮你搜文献”，而是在往数据、流程、评测、写作、部署这些完整研究链条上长。**

---

## 10. 一个关键案例：OpenClaw 医疗生态，最像“通用 Agent + 垂直 skills 包”的现实路径

如果只让我选一个最值得单独拿出来看的案例，我会选 OpenClaw 这一条医学线。原因很简单：它把“通用 agent 如何被加装成领域 agent”这件事，展示得最完整。

这条线至少包含四层：

1. **OpenClaw / ClawHub 这样的底座与注册层**：负责 skills 的发现、发布与使用。
2. **`ClawBio` 这样的原生垂直内容层**：把生物信息学 workflow 真的写成 skills。
3. **`OpenClaw-Medical-Skills` 这样的聚合层**：把临床、omics、药物发现、合规等能力一口气打包给医疗 agent。
4. **`skill-to-mcp` 这样的桥接层**：把 skills 再暴露到 MCP 世界里，扩大复用范围。

如果把这条线放回你最关心的问题——“怎样给通用 Agent 加上特定领域能力”——它给出的答案几乎就是一套模板：

- 先有一个通用 agent；
- 再有一个开放的 skill 格式或技能目录；
- 然后出现原生垂直内容库；
- 最后再由聚合包把数据库、流程、文书、合规、工具链一起打包。

社区里有时会出现 `MedClaw`、`Medge Claw` 一类叫法，但**至少在我这轮能核对到的官方仓库里，真正成体系公开存在的，是 `OpenClaw-Medical-Skills` 与 `ClawBio` 这条生态线，而不是一个完全统一命名的单一总仓库。**

这件事本身也很有代表性：**领域 agent 的能力，越来越不像一个“大模型名字”，而更像一组可以安装、替换、组合、发布的 skills。**

---

## 11. 这对传统机器学习科研和 AI4S 意味着什么？

我前面已经单独问过一个问题：这类 skills 能不能用于传统机器学习科研，或者 AI4S 的特定领域研究？

现在答案其实更清楚了：**能，但要区分“流程层”与“学科层”。**

### 11.1 已经明显成立的部分

- `AI-Research-SKILLs` 的研究流程层：选题、实验记录、写作、评测、部署；
- `claude-scientific-skills` 的跨学科数据库与写作层；
- `materials-simulation-skills` 的数值方法、仿真验证、HPC 层；
- `skill-to-mcp` 这种桥接层，让已有 skills 能接到别的系统里。

这些都不依赖“你必须做 LLM”。它们解决的是：

- 文献怎么进来；
- 数据怎么整理；
- 实验或仿真怎么编排；
- 验证怎么做；
- 结果怎么写出来；
- 复现怎么交给别人。

### 11.2 还不够成熟的部分

但如果你做的是更窄、更硬核的 AI4S 子领域，比如：

- CFD / PDE surrogate；
- 量子化学；
- 计算天体物理；
- 地球系统建模；
- 实验室自动化与仪器控制；

那么今天公开可见的原生 skills 库还远远不够多。很多方向还停留在“有工具、有库、有 workflow，但还没有人把它们认真封装成 Skills”。

所以我反而觉得，接下来最值得期待的不是又多一个通用 agent，而是这些领域开始长出自己的 skills：

- 仿真器接入与模板；
- Slurm / Ray / 集群作业封装；
- 学科本体、单位制、约束检查；
- 领域评测指标；
- 论文写作与 supplementary 模板；
- 面向实验室或机构的合规 SOP。

换句话说：

> **传统机器学习科研和 AI4S 不缺模型，也不缺工具，真正缺的是“把学科工作流封装给通用 agent”的那层 Skills。**

---

## 12. 我现在的结论：哪些仓库最值得盯，哪些只是生态底座？

如果你的目标是“给通用 Agent 加特定领域能力”，我现在的判断大概是这样：

### 更值得长期关注的“原生内容库”

- [AI-Research-SKILLs](https://github.com/Orchestra-Research/AI-Research-SKILLs)：最完整的 AI 研究流程与工程技能树；
- [claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills)：跨学科科研能力面最广；
- [ClawBio](https://github.com/ClawBio/ClawBio)：最鲜明的生物信息学原生路线；
- [materials-simulation-skills](https://github.com/HeshamFS/materials-simulation-skills)：最值得继续关注的 AI4S / 数值模拟路线；
- [huggingface/skills](https://github.com/huggingface/skills)：AI/ML 平台工作流最强的官方示范。

### 更像“聚合 / 包装 / 分发层”的项目

- [OpenClaw-Medical-Skills](https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills)：医疗超大集合包，价值很高，但更偏聚合；
- [Agent Skills](https://github.com/agentskills/agentskills)：开放格式；
- [vercel-labs/skills](https://github.com/vercel-labs/skills)：安装与分发工具；
- [ClawHub](https://github.com/openclaw/clawhub)：技能注册与公共目录；
- [skill-to-mcp](https://github.com/biocontext-ai/skill-to-mcp)：跨协议桥接。

如果只用一句话收尾，我会这样说：

> **科研 Skills 真正有意思的地方，不是“给 agent 多一段提示词”，而是把某个学科的数据库、工具链、评测方法、复现模板与写作方式，打包成可组合、可分发、可迁移的能力层。**

而从今天能看到的版图看，这件事在 AI/ML 和生物医学里已经开始成形，在材料科学与 AI4S 里刚刚起步，但方向非常明确。

---

## 13. 入口与参考

- [Agent Skills 官方规范仓库](https://github.com/agentskills/agentskills)
- [skills.sh / vercel-labs/skills](https://github.com/vercel-labs/skills)
- [ClawHub](https://github.com/openclaw/clawhub)
- [AI Research `Skills` Library](https://github.com/Orchestra-Research/AI-Research-SKILLs)
- [Claude Scientific Skills](https://github.com/K-Dense-AI/claude-scientific-skills)
- [Hugging Face Skills](https://github.com/huggingface/skills)
- [ClawBio](https://github.com/ClawBio/ClawBio)
- [OpenClaw Medical Skills](https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills)
- [Materials Simulation Skills](https://github.com/HeshamFS/materials-simulation-skills)
- [Skill-to-MCP](https://github.com/biocontext-ai/skill-to-mcp)

---

如果你愿意，我下一步还可以继续把这篇文章再往前推两步：

1. 补一张“**仓库 -> 代表 skill -> 对应数据库 / 工具链**” 的更细颗粒度附表；
2. 进一步筛一轮“**最值得真的装进 Codex / Claude Code 的科研 skills 小集合**”，按 AI 研究、生信、AI4S 三条路线各给一套最小可用组合。
