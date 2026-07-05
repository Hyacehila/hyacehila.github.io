---
title: "机器学习炼丹技术备忘"
title_en: "Machine Learning Training Tricks Memo"
date: 2025-11-25 12:33:02 +0800
categories: ["Data Science & Statistics", "Applied Machine Learning & AutoML"]
tags: ["Learning Notes", "Machine Learning", "Model Training", "Research Methods"]
author: Hyacehila
excerpt: "一篇机器学习训练和实验技巧备忘，整理算力、超参数、小修小改、增量设计、测试方法和常见炼丹套路。"
excerpt_en: "A memo on machine learning training and experiment tricks, covering compute, hyperparameters, small architecture changes, incremental design, evaluation methods, and common training tactics."
mathjax: false
hidden: true
permalink: '/blog/2025/11/25/machine-learning-training-tricks/'
---
## 算力碾压
1.1 改大 batchsize，假装迭代次数对齐
	
1.2 多训 epoch，但是不明说，把训练长度换成以迭代次数报告，反之亦然，反正不能让人一眼看出来不对齐
	
1.3 epoch 数不变，但是一个样本用好几回，从而偷偷多过数据
	
1.4 把模型里下采样次数减小，模型计算量大了好几倍，但是只和别人比参数量
	
1.5 不在意计算量和参数量的领域狂堆算力
	
1.6 把算力很大的组件描述一笔带过，效率分析也只分析其它组件
	
1.7 用重参数化把模型搞的很大，训练很慢但是反正比推理开销
	
1.8 EMA / 多模型融合涨点，有条件还能自蒸馏
	
1.9 选个超级小的训练集，这样只要专心解决过拟合
## 超参数
2.1 通过把 cosine 学习率变化调成固定学习率，或者反过来，来得到想要的实验结果（cosine 降低学习率的最后那一部分一般会让模型性能快速上涨，提前下降学习率就会显得训练高效）
	
2.2 稍微调大一点学习率，把 baseline 的学习率调小
	
2.3 把各种超参数都隐藏在代码里面成为 magic number
	
2.4 挑随机种子
## 小修小改
3.1 把模型的 relu 都换成 swish 或者 leaky relu / prelu
	
3.2 偷偷到处加 SE layer，反正基本上会涨点；加便宜的 attention 连接
	
3.3 把诸如 pooling, resize 不带参数的组件都换成带可学参数的，多学一点是一点
	
3.4 模组之间乱拉跳边，多 concat 一些特征反正不亏
	
3.5 在没 BN 的地方加 BN，在有 BN 的地方把 BN 去掉，还有 GN / IN / LN / WN 等等可以换
	
3.6 针对训练集和测试集的差异对训练集增广，改训练集分布
## 增量设计
4.1 加奇奇怪怪的 GAN Loss，一致性 Loss，反正有没有用很难说还能贴很多公式
	
4.2 把别人在论文里一句话带过的技术详细展开，加上一些魔法公式变换凑半页论文
	
4.3 要设计组件 x 加到模型上时，造一个可学习的 beta 参数，初始值为 0，改成把 beta * x 加到模型上，最差情况 beta=0 保持不变
	
4.4 扩展上一条，设计一堆组件，以可学参数的方式加起来
	
4.5 继续扩展，加一个 NAS 进去
	
4.6 从别的模型拿一些预训练参数，这样模型起点变高，上限也会变高因为相当于加数据和标注
	
4.7 搞一些非常复杂的课程学习，花式蒸馏（特征层，特征层的特征，跨模态蒸），别人做不 work 就说需要调参
	
4.8 不管有没有用，套上强化学习框架，让模型更多拥有自主能力
## 测试方法
5.1 测十个指标，报告有进步的三个
	
5.2 做十个数据集的实验，把没效果的五个扔掉
	
5.3 故意让测试方法和别人的训练场景不对齐，做低 baseline，比如把 RGB 通道搞反让别人挂掉
	
5.4 发明新的创新评价指标；魔改指标，比如 Y 通道测 PSNR，但是和别人 RGB 测的一起比
	
5.5 找 trivial 但是别人没考虑的场景，做出极其大的提升
	
5.6 用大模型比别人小模型，不报告别人的大模型；用针对某种指标训练的模型比别人没训的
	
5.7 在不同的硬件上测速，放在一起报告
	
5.8 最近语言大模型的，偷偷在测试 prompt 里加提示，few-shot 和 zero-shot 比
	
5.9 变相在测试集过拟合，比如泄露数据，泄露随机种子；把测试样本放到上游预训练里
	
5.10 测试数据集加真实场景，OOD 样本，baseline 掉点很多，这时候加点增广或者 dropout 把点补回来，但是把涨点贡献算到其它地方
	
5.11 私有测试集，人工评判，改进要多显著都能做出来
	
5.12 客观比不过比主观，主观比不过 cherry pick
## 终极方法
6.1 抄一个别人的方法，但是把名字换一遍
	
6.2 报高性能，问开源就是只有 README
	
6.3 直接开始写论文，不用做实验，反正恰好比 sota 高那么一点点 

1. self-gating基本加上都涨点
	
变体有context gating和SE模块等
		核心思想都是用自己gate自己
	
基本形式是 y = sigmoid(wx)x
	
2. 各种重建，先把输入corrupt一下，然后用autoencoder重建一下，基本都能让feature更robust，何凯明的MAE也是如此。
	
3. 各种dropout，是个地方都可以试着加点dropout，embedding可以加dropout，attention可以加，ffn可以加，mlp可以加，输入上也可以直接加，相当于某种corrupt
4. mixup，也是个神级idea，输入上a类+b类混合一下，然后label也变成a+b混合，基本也是无脑增强，必定涨点
	
5. 对比学习大神器，核心就看如何构造正样本和负样本。有个惊艳的idea，同一个输入foward两次，因为dropout不同，就可以当正样本，也是无脑涨点
