---
title: MineCLIP、视觉信号与奖励函数
title_en: "MineCLIP, Visual Signals, and Reward Design"
date: 2026-03-23 21:00:00 +0800
categories: ["Agent Systems", "Agent Training"]
tags: ["Vision-Language Models", "Reinforcement Learning", "Reward Modeling"]
author: Hyacehila
excerpt: Minecraft agent 的目标往往很清楚，麻烦在于中间步骤没有奖励。MineCLIP 把视频片段和任务文本的相似度变成 dense reward，Plan4MC 则展示了这种视觉奖励在技能训练里怎样被使用。
excerpt_en: "A short appendix on MineCLIP as a vision-language reward model for Minecraft skill learning, and how Plan4MC uses this idea in its reward stack."
mathjax: true
hidden: true
permalink: '/blog/2026/03/23/mineclip-visual-reward-appendix/'
---

Minecraft Agent 的最终目标往往很清楚：milk a cow、craft an iron pickaxe、dig a hole。麻烦在中间过程。任务完成以前，环境很难给出有价值的反馈信号。RL 知道终点，却不知道刚才那一步有没有更接近终点。

把长程任务拆成技能以后，问题会小一点，但不会消失。`find a cow`、`harvest milk_bucket`、`place crafting_table` 比完整任务短，仍然需要反馈。给每个技能手写 dense reward，很快会变成一堆脆弱规则。MineCLIP 的入口就在这里：用玩家视频和字幕训练视觉-语言模型，让它判断最近这段画面像不像这个技能描述。

[MineDojo](https://arxiv.org/abs/2206.08853) 提出的 MineCLIP 从这个缝隙切入。它用 YouTube 视频片段和时间对齐字幕训练 video-language contrastive model。形式上它很像 CLIP，只是图像端换成短视频。

给定视频窗口 $V_t$ 和任务文本 $G$，视频编码器 $\phi_V$ 输出 $v_t=\phi_V(V_t)$，文本编码器 $\phi_G$ 输出 $g=\phi_G(G)$。reward head 做归一化余弦相似度，再乘一个可学习温度：

$$
s(V_t,G)=\exp(\alpha)\left\langle
\frac{\phi_V(V_t)}{\|\phi_V(V_t)\|},
\frac{\phi_G(G)}{\|\phi_G(G)\|}
\right\rangle .
$$

训练时，batch 里每个视频片段都有对应字幕。正样本是同一个视频-文本对，负样本则来自 batch 内其他文本或视频。一个常见写法是对 video-to-text 方向做 InfoNCE：

$$
\mathcal{L}_{v\rightarrow g}
=-\frac{1}{B}\sum_{i=1}^{B}
\log
\frac{\exp(s(V_i,G_i))}
{\sum_{j=1}^{B}\exp(s(V_i,G_j))}.
$$

训练完以后，$s(V,G)$ 不只服务检索，也能当成一个软判断：这段观察是否符合目标描述？

进入 RL 时，智能体每一步拿最近的 16 帧组成 $V_t$。如果候选任务文本集合是 $\mathcal{G}=\{G,G_1^-,\ldots,G_{N_T-1}^-\}$，MineCLIP 先把目标文本的相似度转成 softmax 概率：

$$
P_{G,t}=
\frac{\exp(s(V_t,G))}
{\sum_{G'\in\mathcal{G}}\exp(s(V_t,G'))}.
$$

论文附录讨论了两种转成标量 reward 的方式。第一种是 direct reward：

$$
r_t=\max\left(P_{G,t}-\frac{1}{N_T},0\right).
$$

$1/N_T$ 是随机猜中的基线。低于基线的视觉匹配不奖励，免得把模型自己也不确定的分数送进优化器。第二种是 delta reward：

$$
r_t=P_{G,t}-P_{G,t-1}.
$$

这个写法更像进度奖励。它不奖励站着一直看同一个东西，而奖励视觉上朝目标更接近的变化。direct 对会移动的动物任务比较有效；静态目标上，单纯看概率可能让 agent 学会盯着目标，却忘了继续交互。

变化其实很简单：传统稀疏奖励只在任务完成时给 $1$ 或 $100$；MineCLIP 给的是每个时间步的视觉语言相似度。它缓解了探索问题，也省掉了不少手写 dense shaping。但它仍然是 proxy。视觉上像shear a sheep，不等于羊毛真的进了背包。

[Plan4MC](https://arxiv.org/abs/2303.16563) 很好的结合了传统RL 与 LLM Agent。它没有直接用 MineCLIP 当奖励，而是先把 Minecraft 技能拆成三类：Finding-skills 找东西，Manipulation-skills 挖、杀、放置、采集，Crafting-skills 合成。高层用 LLM 生成 skill graph，再用图搜索排技能序列；只有底层技能仍然靠 RL 学。

训练 Manipulation-skills 时，Plan4MC 用 MineCLIP 给 intrinsic reward。做法和 MineDojo 很接近：取过去 16 帧，与当前 skill prompt 以及 31 个负样本文本一起打分，得到目标 prompt 的 softmax 概率 $p$，然后给：

$$
r_{\mathrm{CLIP}}=\max\left(p-\frac{1}{32},0\right).
$$

Plan4MC 也把边界写得很实在。MineCLIP reward 对一些视觉可描述的技能有用，却不够覆盖所有行为。combat 还要加 distance 和 attack reward，挖 log / cobblestone 要加距离奖励，挖 iron ore / diamond 要加 depth reward。VLM 提供的是一些信号，但不是完整的 reward stack。

视觉 reward 也容易被钻空子。优化器会寻找让模型觉得像的状态，而不一定寻找让环境真的完成的状态。agent 可能把视角对准某个实体，制造高相似度画面，却没有做正确交互。视觉信号密集，但看不到库存、配方、工具耐久和长期因果链。

MineCLIP 是一个局部答案：它解决技能训练里的信号密度问题，不解决完整 agent 的规划、状态验证和长期 credit assignment。Plan4MC 正好说明了这一点。预训练视觉-语言模型可以把看起来像不像目标变成 reward，但可用的 agent 还要做的更多，reward 不是一切。

[CLIP4MC](https://arxiv.org/abs/2303.10571) 也是类似的研究，整体逻辑相似，可以作为参考。这里就不展开了。

参考资料：

- [MineDojo](https://arxiv.org/abs/2206.08853)
- [MineCLIP GitHub](https://github.com/MineDojo/MineCLIP)
- [Plan4MC](https://arxiv.org/abs/2303.16563)
- [CLIP4MC](https://arxiv.org/abs/2303.10571)
