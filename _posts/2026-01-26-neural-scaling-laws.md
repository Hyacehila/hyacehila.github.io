---
layout: blog-post
title:  About Scaling Laws ?
date: 2026-01-26 21:00:00 +0800
categories: [机器学习]
tags: [Scaling Laws, LLM, TreeModel]  
author: Hyacehila
math: true
excerpt: 本文从 Kaplan 定律到 Chinchilla 修正，探讨关于ScalingLaw的研究，并且简单讨论一下TreeModel与ScalingLaw的关系。
---

# About Scaling Laws ?

> 我并不研究Deep Learning Theory，对相关的内容也并不熟悉；以下的内容仅仅是一个理论领域的小白简单学习后的总结。随便写着玩玩。

## Neural Scaling Laws 的从0到1

神经缩放定律（Neural Scaling Laws） 是大规模深度学习（特别是大语言模型 LLM）领域的基石理论。它揭示了**模型性能与计算资源、数据量和参数量之间的幂律（Power Law）关系。**

这意味着，只要我们成倍地增加算力、数据或参数，模型的Loss（损失函数值）就会以可预测的速度下降。这让炼丹从一门玄学变成了有可能不是很可靠但是有用的理论的工程科学（P.s. 新时代材料大炼丹，对实验人员除了头发以外的地方都极其友好）。

Neural Scaling Laws 最基本的公式形式通常为：
$$ L(x) \propto x^{-\alpha} $$
其中 $L$ 是测试集上的 Loss，$x$ 是规模变量（如参数量 $N$、数据量 $D$ 或 算力 $C$），$\alpha$ 是缩放指数。
**核心结论：** 模型的性能主要取决于规模（Scale），而与具体的模型架构（如层数、宽度比例）关系不大（只要架构不是太离谱）。

### 奠基之作：OpenAI 的 Kaplan 定律——乘法形式与参数优先

Kaplan 团队假设测试集 Loss ($L$) 与参数量 ($N$) 和数据集大小 ($D$) 遵循独立的幂律，且二者相互耦合。 原始文章为 *Scaling Laws for Neural Language Models* (Kaplan et al., OpenAI) 

这是大模型时代的"摩尔定律"。它证明了 Transformer 模型的性能与 $N$（参数量）、$D$（数据集大小）、$C$（计算量）之间存在严格的幂律关系。

其单变量形式为：
$$ L(N) \approx \left( \frac{N_c}{N} \right)^{\alpha_N}, \quad L(D) \approx \left( \frac{D_c}{D} \right)^{\alpha_D} $$
其中 $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$，使用实验进行拟合。

为了让Scaling Laws的经验公式能够进一步指导下一阶段的架构设计，Kaplan 提出了耦合的**联合缩放公式**来描述 $N$ 和 $D$ 同时受限时的情况：
$$ L(N, D) = \left[ \left( \frac{N_c}{N} \right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D} \right]^{\alpha_D} $$
 **数学含义：** 这是一个类似调和平均的形式。由于 $\alpha_N < \alpha_D$，这意味着随着算力增加，参数 $N$ 的边际收益递减速度比数据 $D$ 慢。

 **推论：** 为了让 Loss 最小化，**参数量 $N$ 的增长速度应该快于数据量 $D$**（即 $N \propto C^{0.73}, D \propto C^{0.27}$）。因此他们建议在增加算力时，应该优先把模型做大，而不是无限制增加数据。

这直接导致了早期大模型（如 GPT-3, PaLM, MT-NLG）疯狂堆参数（175B, 540B），但训练数据相对较少（通常只训练 1 个 Epoch）。

### 修正时刻：DeepMind 的 Chinchilla 定律——加法形式与等比缩放

Chinchilla 团队在 *Training Compute-Optimal Large Language Models* (Hoffmann et al., DeepMind) 指出了 Kaplan 方法在拟合超参数时的漏洞，并提出了一个更符合直觉的**加法模型**，包含不可约误差。

**核心公式：**
$$ L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} $$
*   $E$: 不可约损失（Irreducible Loss），即自然语言本身的熵（贝叶斯误差），即便模型完美也无法消除。
*   $\frac{A}{N^\alpha}$: **模型近似误差**，源于模型容量不足。
*   $\frac{B}{D^\beta}$: **数据估计误差**，源于有限样本带来的方差。


在总算力 $C \approx 6ND$ 固定的约束下，利用拉格朗日乘数法求极值，DeepMind 发现 $\alpha \approx 0.5, \beta \approx 0.5$。 由于 $\alpha \approx \beta$，这意味着 $N$ 和 $D$ 对 Loss 的贡献是**对称的**。 在固定总算力下最优策略是 **$N$ 和 $D$ 应该等比例增长**（$N \propto C^{0.5}, D \propto C^{0.5}$）。这就是著名的"Chinchilla Scaling Laws"。

Chinchilla Scaling Laws  推翻了 Kaplan 的部分结论。DeepMind 发现，之前的模型（如 GPT-3）都**严重训练不足（Undertrained）**。对于当前的 Transformer，**每个参数大约需要训练 20 个 Token**（即 Token数 / 参数量 $\approx 20$）。 这极大的改变了行业的方向大家不再盲目追求万亿参数，而是追求相对小的模型+极大量的数据。


### 涌现能力：Scaling Laws 的"不连续性" (The Discontinuity)


在 Kaplan 和 Chinchilla 的定律中，Loss 的下降是丝般顺滑的。但在现实应用中，工程师们发现了一个令人费解的现象：**Loss 的降低并不意味着模型学会了做题。**

模型在某些任务（如三位数加法、复杂推理）上的表现会经历一个 **相变（Phase Transition）**：从"完全随机"突然跳变到"接近人类"。 *Emergent Abilities of Large Language Models* (Wei et al., Google, 2022) 一定程度上回复了这个问题——为什么 Loss 是平滑的，结果却是陡峭的？

我们可以用一个简单的概率模型来推导这种非线性。

假设一个逻辑推理任务需要连续 $L$ 个步骤（Steps）全部正确才能得分（Exact Match）。
令 $p$ 为模型预测**单个 Token（或单步推理）** 的正确率。根据 Scaling Laws，随着规模 $N$ 增大，单步正确率 $p$ 是平滑上升的（例如 $p \propto 1 - N^{-\alpha}$）。

但整个任务成功的概率 $P(\text{Task})$ 是所有步骤概率的乘积：
$$ P(\text{Task}) = p^L $$

这里引入了幂次放大的非线性。假设任务需要 $L=5$ 步：
**小模型** ($p=0.5$): $P(\text{Task}) = 0.5^5 \approx 0.03$ (接近 0，表现为"不会")

**中模型** ($p=0.8$): $P(\text{Task}) = 0.8^5 \approx 0.32$ (依然不及格)

**临界点** ($p \to 0.95$): $P(\text{Task}) \approx 0.77$ (**涌现！能力突然爆发**)

**结论：** 涌现并非魔法，而是**微观概率的乘积效应在宏观指标上的投影**。这解释了为什么我们需要把模型做到极大——因为只有当单步准确率 $p$ 极度接近 1 时，长链条推理的成功率 $p^L$ 才有意义。

从Metric的角度看，Stanford 的 Schaeffer (2023) 指出，如果我们把评价指标从"全对才得分"换成平滑的"Token 编辑距离"，涌现曲线就会变回平滑曲线。这告诉我们：**能力一直在积累，只是由于评估标准的严苛，导致我们在临界点前看不见它。**

### 数据重复与质量：Data-Constrained Scaling 


Scaling Laws对数据的要求太高了，互联网的高质量文本快被用光了（Data Wall）。我们面临一个新问题：**如果数据不够，能不能把旧数据重复学几遍？**

*Scaling Data-Constrained Language Models* (Muennighoff et al.) 修正了 Chinchilla 公式，引入了重复训练次数（Epochs, 记为 $R$）作为变量。他发现，数据的**有效性**随着重复次数衰减。

我们可以建立一个简化的概念公式：
$$ D_{eff} \approx D_{unique} \cdot (1 + \lambda \cdot \log R) $$
或者更直观的经验结论：
**$R \le 4$ (4个 Epoch以内):** 数据收益几乎不衰减。模型能从重复数据中榨取剩余价值。

**$R > 40$:** 收益几乎归零，甚至因为过拟合（Overfitting）导致 Loss 反升。

在数据枯竭时代，我们最多只能把手中的高质量数据重复训练 4 遍。再之后，就必须寻找新的出路（如合成数据）。既然 $D$（数量）受限，唯一的出路就是提升数据质量系数。微软 Phi 系列证明了：
$$ L \propto \frac{1}{(Q \cdot D)^\beta} $$
如果数据质量 $Q$ 足够高（如教科书级别的合成数据），极小的 $D$ 也能达到极低的 Loss。这打破了盲目 Scaling 迷信。

### 混合专家模型：MoE Scaling 

为什么 GPT-4、Gemini 1.5、Mixtral 和 DeepSeek 都转向了 MoE（Mixture of Experts）架构？因为 Scaling Law 在 MoE 上展现了**某种程度上堪称作弊的效率**。

传统的稠密（Dense）模型，参数量 $N$ 直接决定了计算量 $C$（FLOPs）。模型越大，跑得越慢。
MoE 打破了这种绑定，它引入了两个维度的 $N$：
*   $N_{total}$：总参数量（决定了模型的"知识容量"和记忆力）。
*   $N_{active}$：活跃参数量（决定了推理时的计算成本/速度）。

*Unified Scaling Laws for Routed Language Models* (DeepMind, Google)发现，MoE 的 Loss 下降遵循包含两个项的幂律：
$$ L(N_{total}, N_{active}) \approx \frac{A}{(N_{active})^\alpha} + \frac{B}{(N_{total})^\beta} $$

这里有一个关键的**不对称性**：
**推理成本**主要由 $N_{active}$ 决定。

**模型性能**却能同时享受 $N_{total}$ 带来的红利（虽然边际收益比 Dense 低，但在大规模下非常可观）。

这使得 MoE 能够在 **Pareto Frontier（帕累托前沿）** 上击败 Dense 模型——**在同等推理算力下，MoE 总是更智能。**

### 结语
神经缩放定律将 AI 从"炼丹术"升格为一门**可预测的工程科学**。这条探索之路始于 Kaplan 对参数规模的信仰，经 Chinchilla 修正为算力与数据的精妙平衡；我们在 Scaling 中见证了能力的突然"涌现"，也在数据枯竭（Data Wall）的压力下转向了"质量优于数量"的策略与更高效的 MoE 架构。如今，随着 Training Scaling 逐步逼近边际，我们正站在新的岔路口：一方面利用合成数据延续 Scaling 的奇迹，另一方面开启 Inference Time Scaling（推理时扩张）的新维度。

## Bias-Variance No Trade-off：Neural Scaling Laws and Statistical Learning Theory

神经缩放定律（Neural Scaling Laws）之所以震撼学界，不仅是因为它指导了工程实践，更因为它在数学表现上**严重挑战了传统统计学习理论（Statistical Learning Theory, SLT）的直觉**。

传统的 SLT 告诉我们"模型太大会过拟合"，做一个更好的模型的核心是要控制模型的复杂度，而 Scaling Laws 告诉我们"越大越好"。这种矛盾的核心在于**偏差-方差权衡（Bias-Variance Trade-off）在深度学习时代的失效与重构**。

### 与传统 SLT 的碰撞：从一致收敛到张量悖论

在经典 SLT（以 Vapnik-Chervonenkis 理论为代表）中，我们试图寻找一个"最坏情况"的保证。泛化误差（Risk）通常被分解为：
$$ \text{Risk} = \text{Bias}^2 + \text{Variance} + \text{Noise} $$

经典的 **VC 泛化界（VC Generalization Bound）** 告诉我们，对于模型复杂度为 $h$（近似于参数量 $N$）的假设空间，泛化误差 $R(f)$ 与训练误差 $\hat{R}(f)$ 之间存在如下关系：
$$ R(f) \leq \hat{R}(f) + \underbrace{\sqrt{\frac{h (\log(2n/h) + 1) + \log(1/\delta)}{n}}}_{\text{Complexity Penalty}} $$

**经典理论的预言：** 随着参数量 $N$ 增加，训练误差 $\hat{R}$ 会下降到 0，但在 $N > n$ 时，复杂度惩罚项（包含 $\sqrt{h/n}$）会趋向无穷大。这直接导致了著名的 **U 型曲线** 推论：总误差必定先降后升。

然而，2017 年 Zhang et al. 在 *Understanding deep learning requires rethinking generalization* 中提出的悖论彻底击碎了这一界限。他们发现：**即使把训练集的标签 $y$ 全部随机打乱（纯噪声），深度网络依然能达到 $\hat{R}(f)=0$**。这证明了模型的 VC 维大到足以记住纯噪声，但在真实数据上，它却神奇地选择了"泛化"而非"记忆"，这说明基于"一致收敛"的上界在深度学习中是失效的。

### 逃离诅咒：良性过拟合 (Benign Overfitting)

为了解释 Scaling Law（即 $N \to \infty$ 时 Loss 单调下降），学界提出了 **良性过拟合** 理论（Bartlett et al., 2020）。这一理论的核心在于重新理解 **方差（Variance）** 在高维空间的行为。

当参数量 $d$ 远大于样本量 $n$ 时，SGD（随机梯度下降）会隐式地收敛到 **$\ell_2$ 范数最小（Minimum Norm）** 的解：
$$ \hat{\theta} = \arg\min_{\theta} \|\theta\|_2 \quad \text{s.t.} \quad X\theta = y $$

在这种设置下，Risk 的分解发生了变化：

- **Bias 单调下降：** 随着模型变大，子空间覆盖能力变强，模型能更好地逼近真实函数。

- **Variance 消失（而非爆炸）：** 这是最关键的部分。Bartlett 证明，只要数据协方差矩阵满足特定的光谱衰减，**噪声能量会被"涂抹"分散到无数个多余的维度上**。

### 完整的数学图像：双重下降 (Double Descent)

将上述理论结合，我们得到了 Scaling Laws 完整的 **双重下降** 图像：

- **欠参数区间 ($N < n$)：** 受制于经典 VC 维上界，呈现 U 型曲线。Bias 下降，Variance 上升。

- **临界区间 ($N \approx n$)：** $XX^T$ 的最小特征值接近 0，逆矩阵范数极大，导致 **Risk 爆炸**。这是传统统计学最恐惧的区域。

- **良性过拟合区间 ($N \gg n$)：** 进入 Scaling Law 领域。Bias 继续下降，而由于维度 $d$ 极大，逆矩阵变得良态，且噪声被分散到高维零空间中，**Variance 不仅没炸，反而趋近于 0**。

Scaling Laws 证明了：在深度神经网络 + SGD 的组合下，我们正处于双重下降曲线的右侧。增加 $N$ 能够同时降低 Bias（更好的逼近）和 Variance（更好的噪声平滑），从而使 Loss 呈现单调的幂律下降。

### 理论重构：Scaling Laws 对数学直觉的修正

Scaling Laws 证明了：在深度神经网络 + SGD 的组合下，**过参数化带来的方差收益（平滑与集成）超过了其带来的方差风险**。

我们不妨对比一下新旧两个时代的数学观：

| 特性 | 传统统计学习 (SLT) | 神经缩放定律 (Scaling Laws) | 数学本质区别 |
| :--- | :--- | :--- | :--- |
| **曲线形态** | U 型曲线 (U-shape) | 幂律下降 (Power Law) | 单调性 vs 非单调性 |
| **主要矛盾** | 偏差 vs 方差 | 计算资源分配 (Allocative Efficiency) | 优化问题 vs 统计推断问题 |
| **过参数化** | 危险 (过拟合) | 必须 (涌现能力的基础) | 良性过拟合 (Benign Overfitting) |
| **误差界限** | 依赖 $\sqrt{N/D}$ | 依赖 $N^{-\alpha} + D^{-\beta}$ | 这里的 $N$ 是作为分母出现的 (收益) |

Chinchilla 定律的公式 $L = E + A/N^\alpha + B/D^\beta$ 实际上重写了泛化误差界：它不再包含导致爆炸的 $N/D$ 项，而是将 $N$ 和 $D$ 视为两个独立的、对降低 Loss 有正向贡献的变量。这是对传统统计学习理论在深度学习语境下的一次重大修正。

## Universality：Scaling Laws的架构无关性与计算缩放定律

这两部分内容触及了神经缩放定律（Scaling Laws）从"纯数学理论"转向"工程实战"与"历史演变"的核心。

如果说"平滑性"是 Scaling Law 存在的**数学地基**，那么"计算缩放"就是指导怎么盖楼的**施工图纸**，而"架构无关性"则解释了为什么我们最终选择了 Transformer 这种**建筑材料**。

### 计算缩放 (Compute Scaling)：资源分配的优化问题

在实际的大模型训练中，我们关心的不仅仅是"参数 $N$"或"数据 $D$"本身，而是 **"我有 1 亿美元的算力预算（FLOPs），我该怎么花？"**   这就是 Scaling Law 作为"资源分配优化问题"的本质。

训练一个 Transformer 模型的总计算量 $C$（Floating-point operations）可以近似为：
$$ C \approx 6 \cdot N \cdot D $$
*   $N$: 模型参数量。
*   $D$: 训练数据的 Token 数量。
*   $6$: 经验系数（前向传播约 $2N$，反向传播约 $4N$）。

优化问题的几何解释：想象一个二维坐标系：
*   X 轴：计算量 $C$（对数坐标）。
*   Y 轴：Loss（对数坐标）。

如果我们固定模型大小 $N$（比如 10B 参数），不断增加数据 $D$，我们会得到一条曲线。随着 $D$ 增加，Loss 先下降后趋于平缓（受限于模型容量）。如果我们画出 1B、10B、100B 不同参数模型的曲线，它们会像一簇下垂的线条。

**Scaling Law 曲线（Compute Frontier）是这一簇曲线的下包络线（Lower Convex Hull）。**

这意味着在任意给定的算力预算 $C$ 下，**必然存在**一个最优的 $(N^*, D^*)$ 组合，使得 Loss 达到全局最小。

偏离这个最优组合会有相应的代价。**过大模型（Over-sized）** 指的是如果你用 huge 的模型但数据很少（如 Kaplan 早期建议），你在浪费算力做矩阵乘法，而没有给模型足够的信息（Undertrained）。相反，**过小模型（Under-sized）** 指的是如果你用 tiny 的模型跑了无限的数据（如 LLaMA 之前的很多小模型），模型容量满了，学不动了，算力被浪费在重复的数据读取上。

在深度学习早期，瓶颈往往是**算法**（不知道怎么训练深层网络）或**数据**（没有 ImageNet）。
但在 Scaling Law 时代：
*   **算法已知：** Transformer + SGD。
*   **数据充足：** CommonCrawl 包含了整个互联网。
*   **唯一限制是 $C$：** 每一单位的 Loss 下降，都需要指数级增加的 FLOPs。

**Chinchilla 定律的本质就是解这个优化方程：**
$$ \min_{N,D} L(N,D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} \quad \text{s.t.} \quad C = 6ND $$
解得最优分配是 $N \propto C^{0.5}, D \propto C^{0.5}$。这意味着算力不仅是瓶颈，更是决定模型设计（该做多大）的唯一自变量。


### 架构无关性 (Universality)：为什么最后赢的是 Transformer？

#### 架构无关性 (Universality)
Scaling Law 最令人震惊的发现之一是：**它似乎是一种"物理规律"，不以架构意志为转移。**

OpenAI 和 DeepMind 的研究表明，不仅仅是 Transformer，包括：
*   **LSTM / RNN**（循环神经网络）
*   **CNN**（卷积神经网络）
*   **Linear Transformers**
*   甚至一些纯全连接网络

它们的 Loss 随规模增长都遵循 $L \propto C^{-\alpha}$ 的幂律形式。这意味着，如果你给 LSTM 无限的算力和参数，它也能变得非常智能。**并没有哪个架构拥有"独家"的缩放特权。**

既然都遵循定律，为什么 LSTM 死了？ 这涉及到一个关键区分：**理论性能上限 vs. 工程实现效率**。

虽然它们都遵循幂律公式 $L = a C^{-k}$，但其中的系数 $a$（截距）和 $k$（下降斜率）略有不同，更重要的是**实际训练的可行性**。

#### 效率即一切：串行瓶颈与梯度流

**串行 vs. 并行 (The Wall-Clock Time Wall)**

这是 Transformer 胜出的决定性因素。**LSTM/RNN** 的数学形式 $h_t = f(h_{t-1}, x_t)$ 决定了它必须串行计算：要计算第 1000 个 token 的状态，必须先算完前 999 个。这种 $O(T)$ 的复杂度无法并行，导致即使它理论上能 Scaling，实际上训练一个 175B 的模型可能需要 **100 年**。

相反，**Transformer** 的 Attention 机制 $Softmax(QK^T)V$ 允许一次性并行计算所有 token 之间的关系。虽然复杂度是 $O(T^2)$，但它可以完美吃满数万张 H100 GPU 的算力，在 **几个月** 内走完 LSTM 需要几百年走的 Scaling 曲线。

**梯度流与长距离依赖**

除了计算效率，梯度流也是关键。在 **LSTM** 中，梯度需要通过时间步反向传播（BPTT），在处理极长序列（如一本书）时容易丢失早期信息，导致 Scaling 曲线比 Transformer 扁平。而在 **Transformer** 中，任意两个 token 之间的距离都是 1（通过 Attention 连接），极其通畅的梯度流使其在 Scaling 时能更有效地利用数据。

#### 硬件彩票 (The Hardware Lottery)
Google 研究员 Sara Hooker 提出了 **"硬件彩票"** 理论：
Scaling Law 并不是说 Transformer 是数学上完美的架构，而是说 **Transformer 是最适合当前 GPU 硬件（矩阵乘法加速器）的架构**。

如果我们使用的是一种某种类脑芯片（擅长串行、稀疏计算），也许 Scaling Law 的赢家就是 LSTM 或 Spiking Neural Networks。

但现实是，我们拥有的是擅长稠密矩阵乘法的 GPU。Transformer 的架构特性（高度并行、稠密计算）让它在这场 Scaling 的赛跑中，**单位时间内下降 Loss 的速度（Loss per GPU-hour）** 远超其他架构。

### 总结：Scaling Law 的残酷真相

将这两点结合起来，我们可以得出一个关于现代 AI 发展的残酷总结：

**架构不再是核心护城河：** 只要架构能支持梯度下降且具备一定的表达能力，它就能 Scale。Transformer 的胜利主要是**工程效率**的胜利（它能把 Scaling Law 跑得最快）。

**算力决定智力：** 在最优化的资源分配策略（Chinchilla）下，模型的智能水平（Loss）几乎完全由投入的算力预算（$C$）决定。

**游戏规则：** 深度学习变成了一个**将电力和芯片转化为智能的工业过程**。谁能更高效地（并行度更高、硬件利用率更高）沿着 Scaling Law 曲线跑，谁就是赢家。

这就是为什么现在的 AI 实验室更像是一个**系统工程公司**，而不是传统的算法实验室。核心竞争力从"设计精巧的网络结构"变成了"构建能稳定运行万卡集群的基础设施"。

## 反面教材：GBDT 的"硬切割"与失效的 Scaling

这是一个极其敏锐且具有实战意义的问题。我们发现横扫 AI 界的 Scaling Laws 在经典的梯度提升树（GBDT，如 XGBoost、LightGBM）面前失效了。这恰恰帮助我们厘清了 Scaling 的本质。

简短的回答是：**GBDT 依然受困于经典统计学习理论的 U 型曲线，而深度神经网络（DNN）通过"良性过拟合"逃逸了。**

为了理解 Scaling Law 为什么只眷顾深度学习，我们需要深入到函数逼近论（Function Approximation）的底层。

### U 型曲线 vs 单调下降

如果你在 XGBoost 中不断增加树的数量（`n_estimators`）或深度，你会观察到教科书般的**偏差-方差权衡（Bias-Variance Trade-off）**：
*   **初期：** 随着树的增加，模型捕捉到了数据特征，Bias 下降，测试集 Loss 降低。
*   **后期：** 一旦越过临界点，模型开始强行拟合噪声。虽然训练集 Loss 趋近于 0，但测试集 Loss 迅速反弹。
*   **结局：** 这就是经典的 **U 型曲线**。GBDT 极其依赖 Early Stopping 来防止崩盘。

相比之下，LLM 处于"双重下降"的现代区间。只要数据量足够（符合 Chinchilla 比例），增加参数量几乎总是能带来 Loss 的持续下降。这种"堆参数"的红利是深度学习独有的。

### 分段常数 vs 连续流形：归纳偏置的战争

为什么？核心在于二者数学上的 **归纳偏置（Inductive Bias）** —— 即模型预设的"世界观"完全不同。

#### GBDT：空间的硬切割 (Space Partitioning)
树模型本质上是将高维空间切割成无数个互不相关的**超立方体（Hyper-rectangles）**。
$$ f(x) = \sum c_m \cdot \mathbb{I}(x \in R_m) $$
它假设世界是由无数个互不相关的平坦的切块组成的，然后在每个块里给一个常数预测值（分段常数函数），在区域的边界上预测值是跳跃（Step Function）的。
*   **死记硬背：** 增加参数（加深树）等于把空间切得更碎。当树越来越深，空间被切成粉末，每个叶子节点可能只包含 1 个样本。
*   **非平滑性：** 对于未见过的区域（Gap），模型只能输出僵硬的常数，无法根据梯度进行插值。这种**不连续、非平滑**的逼近方式，导致过参数化直接引发方差爆炸。

#### DNN：流形的软逼近 (Manifold Approximation)
神经网络通过层层矩阵乘法和激活函数，构建的是一个**高度平滑的连续流形**。
$$ f(x) = \phi_L(\dots \phi_1(W_1 x)) $$
它假设世界是连续的、可微的。
*   **良性过拟合：** 虽然大模型也有能力"记住"噪声，但在 SGD（随机梯度下降）的驱动下，模型倾向于收敛到 **范数最小（Minimum Norm）** 的解。
*   **隐式正则化：** 数学直觉告诉我们，在无数个能拟合数据的解中，SGD 自动选择了**曲率最小、最平滑**的那一个。增加参数，实际上是给了模型更多的自由度去画出更精细且平滑的曲线，而不是制造锯齿。

### 结论：平滑性是 Scaling 的入场券

平滑性是 Scaling Law 能否生效的关键。在高维空间中，数据是极度稀疏的。模型必须根据训练点去猜测（插值）空白区域的值。如果模型是"跳跃"的（如 GBDT），那么在空白区域的预测就会非常不准确（高方差）。如果模型是"平滑"的（如 DNN），它就能利用已有的数据点进行平滑插值，从而在空白区域保持较低的误差。

我们可以用一个直观的类比来总结：
*   **GBDT 的 Scaling** 就像是在堆积木。为了拟合一条曲线，你用越来越小的积木（方块）去堆。积木越小，边缘的锯齿就越明显，对位置的偏差就越敏感（过拟合）。
*   **DNN 的 Scaling** 就像是拉一根橡皮筋。为了拟合数据点，你增加橡皮筋的弹性（自由度）。在 SGD 的张力下，橡皮筋不仅穿过了数据点，还在空白处保持了平滑的过渡。

**Scaling Law 本质上是平滑性（Smoothness）的特权。** GBDT 缺乏全局平滑性，过参数化导致的是空间碎片化；而 DNN 具备全局流形结构，将过参数化转化为了对真实函数的**高精度平滑插值**。

这就是为什么 GBDT 依然是表格数据的王者（低维、不连续），但在通往 AGI 的道路上，二者的命运截然不同：

**GBDT**： 树模型擅长处理低维、稠密、表格型数据。但在极高维空间（如文本、图像的像素级特征）中，要覆盖整个空间所需的树的深度和数量呈指数级爆炸。此时增加规模并不能带来泛化能力的提升，只能带来过拟合。

**DNN**： 神经网络（特别是 Transformer）擅长通过 Embedding 将高维稀疏数据压缩到低维流形中，其特征组合能力随着深度增加呈指数级增强，且能保持泛化性。

只有深度神经网络能打破维度诅咒，踏上通往 AGI 的 Scaling 之路。

## 树模型的 Scaling：非参数化与参数化的本质区别

**既然树模型可以通过增加树的数量（`n_estimators`）或深度无限增加复杂度，这难道不也是一种 Scaling 吗？**

答案是：**是的，这是一种 Scaling，但它属于"非参数化 Scaling"的范畴，与深度学习的"参数化 Scaling"在动力学上有着本质的区别。**

我们可以尝试建立这种联系，但必须直面它们在**数学机理**上的根本分歧。以下是基于之前讨论的深入发散：

### 重新审视：树模型真的只有 U 型曲线吗？（连接 Double Descent）

我们在之前的讨论中提到树模型通常遵循 U 型曲线，但这其实是一个"有条件的真理"。最新的研究（如 Mikhail Belkin 团队的工作）表明，**在特定条件下，树模型也能展现出 Scaling Law 式的单调下降或"双重下降"现象。**

#### 随机森林 (Random Forest) 的"良性过拟合"
随机森林是 Scaling Law 最好的"传统盟友"。

**数学形式：** RF 是 Bagging（Bootstrap Aggregating）。
$$ f_{RF}(x) = \frac{1}{M} \sum_{m=1}^M T_m(x) $$

**Scaling 行为：** 随着树的数量 $M \to \infty$，RF 的测试误差通常**单调下降**并收敛到一个常数（不可约误差 + 模型偏差）。它几乎不会因为树太多而过拟合（U 型右侧不翘起）。

**与 Deep Learning 的联系：** 这非常像深度学习中的**宽度扩展（Width Scaling）**。超宽的神经网络本质上类似于无数个子网络的集成（Ensemble）。

**差异：** RF 的 Loss 下降速度通常遵循统计学的 $1/\sqrt{M}$ 或 $1/M$ 律，收敛极快但上限（Bias）很难突破。而 Deep Learning 的 Scaling Law 往往跨越更多的数量级。

#### GBDT 的"双重下降"可能性
对于 Boosting（如 XGBoost），传统观点认为必过拟合。但近期实验发现，如果**完全不剪枝（No Pruning）**且学习率极低（Shrinkage $\to 0$），Boosting 也能观察到类似双重下降的现象。

**解释：** 当树极其深（过参数化）时，每棵树都过拟合了部分残差。但如果学习率足够低，这种过拟合是"缓慢"发生的，且后续的树会对前面的噪声进行"微调"。这在某种程度上模拟了 SGD 的迭代过程。

### 核心发散：两种截然不同的"Scaling 范式"

虽然都能 Scaling，但树模型和神经网络是在攀登两座不同的山峰。我们需要理解二者 Scaling 效率差异的几何核心。

#### 树模型的 Scaling (Tiling / Partitioning)：

**操作：** 树模型通过不断切分空间（Axis-aligned splits）来逼近目标函数。

**复杂度：** 它是**局部（Local）**的。为了逼近一个高维球体，树模型需要切出成千上万个微小的正方体（Hyper-cubes）来把球体"拼"出来。

**Scaling 困境：** **维度灾难（Curse of Dimensionality）**。每增加一个特征维度，为了维持同样的逼近精度，所需的树的数量（参数量）需要呈指数级增长 ($2^D$)。

**结论：** 树模型的 Scaling 效率在低维（表格）数据极高，但在高维（图像/文本）数据极低。

#### 神经网络的 Scaling (Composition / Folding)：

**操作：** 神经网络通过线性变换（旋转/拉伸）和非线性激活（折叠）来扭曲空间。

**复杂度：** 它是**全局（Global）** 且 **组合（Compositional）** 的。深层网络不需要把空间切碎，它只需要学会一个函数去学习数据的流形结构。

**Scaling 优势：** 这种 **组合性（Compositionality）** 使得神经网络可以用线性增长的参数量，去表达指数级复杂的函数（这一点被 Telgarsky 等人在 2016 年理论证明）。

**结论：** 树模型的 Scaling 是"加法式"的（逼近效率随维度恶化），神经网络的 Scaling 是"乘法/复合式"的（逼近效率随深度增强）。

#### 记忆 vs. 理解 (Memorization vs. Interpolation)

回到我们之前讨论的**平滑性**。

**树的扩展：** 当我们把 GBDT 扩展到极致（每个叶子只有一个样本），它实际上变成了一个 **k-Nearest Neighbor (kNN)** 或者查表机。它完美记住了训练数据。

**测试时：** 对于新样本，它只是去查"哪一片叶子离我最近"。这种插值是**非平滑的阶梯状**。

**网络的扩展：** 当网络参数无限大，它学到的是穿过所有点的**最平滑流形**。

**测试时：** 对于新样本，它是在流形上游走。

**结论：** Scaling Law 的神奇之处不在于"变大"，而在于"变大后还能保持空间上的平滑预测结构"。树模型变大后（如果不强行加正则），空间本质上是变得**破碎**。

### 如果让树模型适配 Scaling Law？

如果我们非要让树模型拥有像 Transformer 那样的 Scaling 能力，我们需要怎么做？这其实指出了当前 AI 研究的一些融合方向。

#### 软树 / 神经树 (Differentiable / Soft Trees)
如果我们把树模型中"硬"的 `if x > 0.5 then left else right` 变成"软"的 `Sigmoid(x - 0.5)`（即以此概率向左走），那么：
*   树就变成了可微分的。
*   树就变成了神经网络的一种特殊形式（稀疏连接的全连接层）。
*   **结果：** 这种"神经树"就可以使用 SGD 训练，就可以享受 Scaling Law！
*   **启示：** 这再次证明了，Scaling Law 的核心可能不在于架构的名字（树 vs 网），而在于**可微分性（Differentiability）**带来的**梯度优化**和**连续流形表示**。

#### 表征学习 (Representation Learning) 的缺失
Scaling Law 在 LLM 上的成功，很大程度源于模型随着规模增加，学会了更好的 **Embedding（特征表示）**。
*   **树模型：** 通常直接在原始特征上切分。它不创造新特征，只组合旧特征。
*   **改造：** 如果我们在 GBDT 前面加一个 Transformer 做特征提取，后面接 GBDT 做分类头。那么这个整体系统是符合 Scaling Law 的。但此时，功劳归于 Transformer（学到了好的表示），而不是 GBDT。
