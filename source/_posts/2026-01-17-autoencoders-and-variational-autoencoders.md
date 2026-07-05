---
title: "自编码器与变分自编码器：重参数化、KL 散度与 ELBO"
title_en: "Autoencoders and Variational Autoencoders: Reparameterization, KL Divergence, and ELBO"
date: 2026-01-17 23:58:14 +0800
categories: ["Data Science & Statistics", "Deep Learning"]
tags: ["Learning Notes", "Machine Learning", "Deep Learning", "Autoencoders", "Variational Autoencoders"]
author: Hyacehila
excerpt: "整理 AE、DAE、SAE、VAE、重参数化技巧、KL 散度和 ELBO 等内容。"
excerpt_en: "Covers AEs, DAEs, SAEs, VAEs, reparameterization, KL divergence, and ELBO."
mathjax: true
hidden: true
permalink: '/blog/2026/01/17/autoencoders-and-variational-autoencoders/'
---
> 从基础到前沿，循序渐进掌握深度学习中的编码器架构

---

## 自编码器基础

### 什么是自编码器

**核心定义**

自编码器是一种**自监督学习**的神经网络架构。它的核心目标不是预测标签，而是**学习输入数据的压缩表征**，即让神经网络的输出尽可能复现输入。

**直观理解**

可以将自编码器想象为"压缩-解压"过程：
- **编码器**：将高清图片转换为简短的编码向量（压缩）
- **解码器**：从编码向量恢复原始图片（解压）

**核心思想**

如果一个网络能够成功从压缩的编码中恢复原始数据，那么这个压缩编码一定包含了数据中最核心、最重要的特征，而丢弃了噪声和冗余信息。

### 核心架构

自编码器由三部分组成：

```mermaid
graph LR
    A[输入 x] --> B[编码器]
    B --> C[隐表征 h/z<br/>瓶颈层（潜在空间）]
    C --> D[解码器]
    D --> E[重构输出 x̂]
```

**数学符号定义**

- $\mathbf{x}$：输入数据，维度为 $d$（$\mathbf{x} \in \mathbb{R}^d$）
- $\mathbf{h}$ 或 $\mathbf{z}$：隐层表征（潜在变量），维度为 $p$（通常 $p < d$）
- $\hat{\mathbf{x}}$：重构输出，维度与 $\mathbf{x}$ 相同
- $f_\theta$：编码函数，参数为 $\theta$
- $g_\phi$：解码函数，参数为 $\phi$

### 前向传播与训练

**编码阶段**
$$\mathbf{h} = f_\theta(\mathbf{x}) = \sigma(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e)$$

**解码阶段**
$$\hat{\mathbf{x}} = g_\phi(\mathbf{h}) = \sigma'(\mathbf{W}_d \mathbf{h} + \mathbf{b}_d)$$

**训练目标**
$$\min_{\theta,\phi} \mathbb{E}_{\mathbf{x} \sim \mathcal{D}} [\mathcal{L}(\mathbf{x}, g_\phi(f_\theta(\mathbf{x})))]$$

### 损失函数

**均方误差（MSE）**
适用于数值型数据（如图像像素值）：
$$\mathcal{L}_{\text{MSE}}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{N} \sum_{i=1}^{N} \lVert \mathbf{x}_i - \hat{\mathbf{x}}_i \rVert_2^2$$

**二元交叉熵（BCE）**
适用于二值型数据或归一化到 $[0,1]$ 的数据：
$$\mathcal{L}_{\text{BCE}}(\mathbf{x}, \hat{\mathbf{x}}) = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{d} \left[ x_{i,j} \log(\hat{x}_{i,j}) + (1 - x_{i,j}) \log(1 - \hat{x}_{i,j}) \right]$$

### 为什么需要"瓶颈"

**问题**

如果隐层维度 $p \geq d$ 且无其他约束，网络可以直接学习恒等映射（$\mathbf{h}=\mathbf{x}$），这没有任何意义。

**解决方案**

**欠完备自编码器**强制 $p < d$，迫使网络学习数据中最显著的特征，类似于 PCA 捕捉数据的主成分。

> **专家视角**：若激活函数为线性且损失函数为 MSE，欠完备自编码器等价于 **主成分分析（PCA）**。但由于自编码器使用非线性激活函数（如 ReLU），它能学习到比 PCA 更强大的**非线性流形**。


### 降噪自编码器（DAE）

**核心思想**

在输入数据中加入噪声，强迫网络学习鲁棒特征。

**数学表述**
$$\begin{aligned}
\text{含噪输入：} &\quad \tilde{\mathbf{x}} = \mathbf{x} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}) \\
\text{训练目标：} &\quad \mathbf{x} \\
\text{损失函数：} &\quad \mathcal{L} = \lVert \mathbf{x} - g_\phi(f_\theta(\tilde{\mathbf{x}})) \rVert_2^2
\end{aligned}$$

**意义**：迫使网络学习鲁棒特征，而非简单复制，学会从损坏数据推断完整信息。

### 稀疏自编码器（SAE）

**核心思想**

允许隐层维度 $p > d$，但在损失函数中加入**稀疏性约束**。

**损失函数**
$$\mathcal{L}_{\text{SAE}} = \mathcal{L}_{\text{reconstruction}} + \lambda \sum_{i} |h_i|$$

或使用 KL 散度约束隐层激活接近目标稀疏度 $\rho$：
$$\mathcal{L}_{\text{SAE}} = \mathcal{L}_{\text{reconstruction}} + \beta \cdot D_{\text{KL}}(\rho \| \hat{\rho})$$

**意义**：限制同一时间只有极少数神经元被激活，模拟生物神经系统的工作方式。

### 从 AE 到 VAE：质的飞跃

虽然 DAE 和 SAE 在一定程度上改善了标准 AE，但它们仍然无法解决**生成新数据**这一核心问题。这催生了 **变分自编码器（VAE）** 的诞生。

---

## 变分自编码器（VAE）

VAE 是深度生成模型的基石之一，它通过引入概率分布，彻底改变了自编码器的生成能力。

### 为什么需要 VAE

**标准自编码器的问题回顾**

- 将输入映射为固定向量 → 潜在空间不连续
- 只能"压缩"，不能"生成"

**VAE 的核心洞察**

VAE 不将输入映射为一个"点"，而是映射成一个**概率分布**（通常为高斯分布）：
- 不说"这张图是坐标 $(3, 2)$"
- 而是说"这张图大概率在 $(3, 2)$ 附近的一个范围内"

**关键优势**

- 引入概率分布和正则化 → 潜在空间变得平滑连续
- 在"1"和"7"的分布之间采样 → 解码结果呈现平滑过渡
- **可以生成新数据**！

### 核心架构

VAE 包含三个关键部分：概率编码器、采样层、解码器。

#### 编码器

输入 $\mathbf{x}$，神经网络输出分布参数：

$$\begin{aligned}
\boldsymbol{\mu} &= f_\mu(\mathbf{x}) \\
\log\boldsymbol{\sigma}^2 &= f_\sigma(\mathbf{x})
\end{aligned}$$

- **均值向量** $\boldsymbol{\mu}$：分布的中心位置
- **对数方差向量** $\log\boldsymbol{\sigma}^2$：分布的离散程度

#### 重参数化技巧 —— **核心考点**

从分布 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ 中采样 $\mathbf{z}$ 传给解码器。

**问题**：直接采样是随机过程，**不可导**！反向传播无法通过随机节点传递梯度。

**技巧**：将随机性"剥离"出来
$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**意义**：现在 $\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma}$ 只是确定性参数计算，梯度可畅通无阻地传回编码器，随机性完全由输入 $\boldsymbol{\varepsilon}$ 提供。

#### 解码器

输入采样的 $\mathbf{z}$，输出重构 $\hat{\mathbf{x}}$：
$$\hat{\mathbf{x}} = g_\phi(\mathbf{z})$$

### 数学原理：变分推断

#### 目标：最大化对数似然

目标是让模型生成真实数据 $\mathbf{x}$ 的概率 $P(\mathbf{x})$ 最大化：
$$P(\mathbf{x}) = \int P(\mathbf{x}|\mathbf{z})P(\mathbf{z}) d\mathbf{z}$$

由于这个积分在复杂神经网络中不可计算，我们无法直接优化。

#### 变分下界（ELBO）

引入近似分布 $q_\phi(\mathbf{z}|\mathbf{x})$（编码器）逼近真实后验 $p(\mathbf{z}|\mathbf{x})$。

经过数学推导（使用琴生不等式），得到 $\log P(\mathbf{x})$ 的下界：
$$\text{ELBO} = \underbrace{\mathbb{E}_{\mathbf{z} \sim q}[\log p(\mathbf{x}|\mathbf{z})]}_{\text{重构项}} - \underbrace{D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{正则化项}}$$

**损失函数**
$$\mathcal{L}_{\text{VAE}} = -\text{ELBO} = \mathcal{L}_{\text{reconstruction}} + D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

#### 损失函数的直观解释

VAE 的 Loss 由两股相互"对抗"的力量组成：

**重构损失**
- **作用**：让解码图像尽可能像原图
- **倾向**：若无约束，会让方差 $\boldsymbol{\sigma} \to \mathbf{0}$，退化成普通 AE

**KL 散度**
- **作用**：强迫编码器输出分布接近标准正态分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$
- **倾向**：若仅优化此项，编码器会忽略输入，始终输出标准噪声分布

**平衡**：VAE 在两者间寻找平衡，既让编码包含原始信息，又让潜在空间符合正态分布形状，保证连续性和生成能力。


#### 为什么 VAE 生成的图像模糊

这是 VAE 最著名的缺点。**原因**：VAE 使用高斯分布假设和 MSE 损失。MSE 倾向于对所有可能像素值取"平均"，导致边缘细节丢失，类似过度磨皮。

普通的 AE/VAE 通常用 MSE Loss（像素级均方误差）来训练。MSE 有个很大的毛病，它对高频纹理不敏感，或者说它倾向于生成“模糊的平均值”。但在数学上它又为了降低这个误差，保留了大量人眼根本不在乎的像素级冗余信息。

LDM 用的 VQ-GAN 或者微调过的 VAE，引入了 Perceptual Loss（感知损失） 和 PatchGAN Discriminator（判别器损失）。这个改动迫使压缩模型专注于保留图片的语义结构和纹理及其空间关系，而忽略掉那些无意义的像素级随机噪声。


### VAE 的优势与局限

**优势**
- 训练稳定，不像 GAN 容易出现模式崩溃
- 具有明确的概率模型和数学解释
- 潜在空间平滑连续，适合插值和探索
- 可以进行有效的推理

**局限**
- 生成图像倾向于模糊（相比 GAN）
- 潜在空间的维度选择需要经验
- 对于某些复杂数据分布，表达能力有限

### β-VAE：特征解纠缠

标准 VAE 中，潜在向量 $\mathbf{z}$ 的各维度通常是**纠缠**的：改变一个数值可能同时影响多个属性（如颜色、大小、角度）。

**解决方案：调整 KL 权重**

修改损失函数，给 KL 散度项加权重系数 $\beta$（通常 $\beta > 1$）：
$$\mathcal{L}_{\beta\text{-VAE}} = \mathcal{L}_{\text{reconstruction}} + \beta \cdot D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

**原理**

通过增大 $\beta$，强迫潜在分布 $q_\phi(\mathbf{z}|\mathbf{x})$ 严格服从标准正态分布（各维度相互独立）。这种强约束迫使模型寻找数据中**最有效、最独立**的因子：
- 一个维度只控制颜色
- 另一个维度只控制形状
- 第三个维度只控制角度

这称为**解纠缠表征（Disentangled Representation）**。

**代价与权衡**

$\beta$ 越大：
- 潜在空间更加解纠缠，可解释性更强
- 重构图像通常越模糊（重构误差权重相对变小）

需要在解纠缠程度和重构质量之间寻找平衡。

**理论意义**

$\beta$-VAE 是 DeepMind 在理论上的重要贡献，它：
- 揭示了潜在空间结构与解纠缠的关系
- 为可解释 AI 提供了新思路
- 在强化学习、机器人控制等领域有应用

---

## 革命性突破 - VQ-VAE

这是 VAE 家族中**最革命性**的一员，由 Google DeepMind 提出，打破了"潜在空间必须是连续高斯分布"的教条。

### 核心痛点

标准 VAE 假设潜在变量是连续的，这导致：
- 生成的图像边缘模糊
- 人类的语言、逻辑概念往往是**离散**的（如"猫"、"狗"是分类概念）
- **无法连接到 Transformer 和语言模型**

### 解决方案：码本机制

VQ-VAE 引入**码本（Codebook）** 的概念。

#### 前向传播过程

**编码**
编码器输出连续向量 $\mathbf{z}_e(\mathbf{x})$

**向量量化**
在码本中寻找最近的码向量：
$$\mathbf{z}_q(\mathbf{x}) = \text{Codebook}[k^*], \quad k^* = \arg\min_k \lVert \mathbf{z}_e(\mathbf{x}) - \mathbf{e}_k \rVert_2$$

其中 $\mathbf{e}_k$ 是码本中的第 $k$ 个码向量。

**解码**
解码器接收量化后的向量 $\mathbf{z}_q(\mathbf{x})$，输出重构 $\hat{\mathbf{x}}$

#### 直通估计器

由于"查表"和"最近邻搜索"是不可导操作，VQ-VAE 使用 **Straight-Through Estimator**：

- **前向传播**：使用量化后的向量 $\mathbf{z}_q$
- **反向传播**：直接将梯度复制给编码器输出 $\mathbf{z}_e$

$$\nabla_{\mathbf{z}_e} \mathcal{L} = \nabla_{\mathbf{z}_q} \mathcal{L}$$

#### 损失函数

$$\mathcal{L} = \underbrace{\lVert \mathbf{x} - \hat{\mathbf{x}} \rVert_2^2}_{\text{重构损失}} + \underbrace{\lVert \text{stop}(\mathbf{z}_e(\mathbf{x})) - \mathbf{e}_k \rVert_2^2}_{\text{码本更新}} + \underbrace{\beta \lVert \mathbf{z}_e(\mathbf{x}) - \text{stop}(\mathbf{e}_k) \rVert_2^2}_{\text{编码器 commitment}}$$

### 为什么 VQ-VAE 是革命性的

#### 清晰度极高
强制使用离散的、高质量的特征码，抛弃模糊的中间态。

#### 连接 Transformer
这是最关键的创新。因为潜在空间变成离散的（类似单词 Token），可以将图像变成一串 Token 序列，直接用 GPT/Transformer 处理图像！

#### 实际应用
- OpenAI 的 **DALL-E 1** 基于 VQ-VAE 变体
- 音频生成模型 **MusicLM**、**AudioLM**
- 多模态模型（如 GPT-4o 的视觉能力部分）

### 为什么 VQ-VAE 重要

VQ-VAE 是**连接视觉与语言的桥梁**：
- 将连续图像离散化为 Token
- 使得统一的语言-视觉架构成为可能
- 为多模态大模型奠定基础

---

## 现代拓展 - MAE

虽然名字里没有"Variational"，但 MAE（Masked Autoencoder）是 Autoencoder 思想在 **Transformer 时代**的延续。

### 背景：从 BERT 到视觉

BERT 在 NLP 领域大杀四方，用的是"完形填空"思想。MAE 将这个思想搬到计算机视觉。

### 核心做法

**图像分块**
将图片切成许多小块（Patches），如 $16 \times 16$ 像素。

**随机掩码**
**随机扔掉 75% 的块**（Masking）—— 注意这个比例很高！

**编码**
只把剩下的 25% 喂给编码器（Vision Transformer, ViT）。

**解码**
解码器负责补全那扔掉的 75% 的块。

### 数学表述

**掩码策略**
$$\mathbf{M} \in \{0, 1\}^{N \times N}, \quad \sum_{i,j} M_{i,j} \approx 0.25 \times N \times N$$

**重构目标**
$$\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{U}|} \sum_{i \in \mathcal{U}} \lVert \mathbf{x}_i - \hat{\mathbf{x}}_i \rVert_2^2$$

其中 $\mathcal{U}$ 是被掩码的块的索引集合。

### 为什么 MAE 有效

**强制学习语义**
如果不理解"狗"的语义，是无法补全被遮挡的狗头部的。

**高效预训练**
- 只处理 25% 的数据，计算效率高
- 掩码任务迫使模型学习全局依赖关系

### 意义与影响

MAE 证明 Autoencoder 架构是进行**自监督学习**的绝佳工具：
- 不需要标签，就能让模型理解图像语义
- 现在许多高性能视觉模型都用这种方式预训练
- 为 Vision Transformer 在计算机视觉的成功奠定基础

---

## 综合应用场景

自编码器及其变体在实际应用中非常广泛。以下是主要应用领域。

### 降维与可视化

类似 t-SNE 或 PCA，将高维数据压缩到 2D 或 3D 进行可视化，或作为预处理步骤减少计算量。

### 异常检测

**核心逻辑**：用大量"正常数据"训练 AE。当输入"异常数据"时，重构误差会显著增大。

**判定公式**
$$\text{Anomaly}(\mathbf{x}) = \mathbb{I}[\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) > \tau]$$

**应用**：信用卡欺诈检测、工业设备故障预警

### 图像去噪与修复

使用去噪自编码器思想：
- 去除老照片的噪点
- 补全图像中被遮挡部分

### 特征提取与预训练

在标签数据稀缺时，先用大量无标签数据训练自编码器。然后保留**编码器**部分，接入分类层进行微调。

这种方法在 BERT 等模型中广泛应用。

### 生成新数据（VAE）

训练完成后，丢弃编码器。直接从 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 采样随机向量 $\mathbf{z}$，喂给解码器即可生成不存在的人脸或场景。主流的生成模型已经基本不再使用VAE作为生成结构了，而是将其用于压缩。

### 潜在空间插值（VAE）

取两张图 A 和 B，分别编码得到 $\mathbf{z}_A$ 和 $\mathbf{z}_B$。

计算中间向量：
$$\mathbf{z}_{\text{mid}} = \alpha \mathbf{z}_A + (1-\alpha)\mathbf{z}_B, \quad \alpha \in [0,1]$$

解码 $\mathbf{z}_{\text{mid}}$，会看到图 A 平滑渐变为图 B。

### 解纠缠表征学习（$\beta$-VAE）

让 $\mathbf{z}$ 的每个维度控制独立特征。

例如：
- $z_1$ 控制发色
- $z_2$ 控制肤色
- $z_3$ 控制角度

调整 $z_1$ 时，只有发色变化，其他不变。

### 特征解耦

确定编码的哪些维度代表哪些信息。

例如：100 维向量中，前 50 维代表句子内容，后 50 维代表说话人特征。

### 离散隐表征

强迫编码为独热向量（只有一维为 1，其余为 0），可实现无监督分类。

例如：手写数字识别（0-9），训练自编码器强迫 10 维编码为独热向量。这 10 种独热编码可能各自对应一个数字，实现完全无监督的分类学习。

### 数据压缩

编码器输出是低维向量，可直接视为压缩结果：
- 编码器执行压缩
- 解码器执行解压缩
- 这是有损压缩（会失真）

### Stable Diffusion —— 最重要的应用

**这是目前最重要的应用**。Stable Diffusion 实际上称为"Latent Diffusion Model"：

- 不直接在像素空间处理巨大图片
- 先用 **VAE** 将图片压缩到潜在空间
- 在这个小空间中进行扩散生成
- 最后用 **VAE 解码器**还原为大图
- 自编码器已经是现在主流生成模型的基础

### 多模态模型（VQ-VAE）

- **DALL-E**：VQ-VAE 的离散表示 + GPT
- **MusicLM**：音频的离散化 + 语言模型
- **GPT-4o**：视觉能力部分基于类似的离散表示

---
