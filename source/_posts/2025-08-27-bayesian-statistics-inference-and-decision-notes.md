---
title: "贝叶斯统计：推断与决策学习笔记"
title_en: "Bayesian Statistics: Inference and Decision Notes"
date: 2025-08-27 00:18:46 +0800
categories: ["Data Science & Statistics", "Probability & Statistical Foundations"]
tags: ["Learning Notes", "Statistics", "Bayesian Statistics", "Statistical Decision"]
author: Hyacehila
excerpt: "一篇贝叶斯统计推断与决策学习笔记，整理条件方法、贝叶斯估计、假设检验、多假设问题和决策理论。"
excerpt_en: "A study note on Bayesian inference and decision, covering conditional methods, Bayesian estimation, hypothesis testing, multiple hypotheses, and decision theory."
mathjax: true
hidden: true
permalink: '/blog/2025/08/27/bayesian-statistics-inference-and-decision-notes/'
---
## 贝叶斯统计推断
这一章的开篇将会复习我们在前面接触的一些内容；然后开展对于统计推断的研究
### 条件方法
后验分布是综合先验分布 总体分布 样本分布三种信息于一身的分布

我们关于参数估计和假设检验等多种统计推断问题都是从从后验分布提取信息（**一切统计推断必须从后验分布出发**） 比经典统计学提取信息容易的多

**贝叶斯学派条件方法的思想是  ：只考虑已出现的数据（样本观察值）而认为未出现的数据与推断无关**

经典统计学往往认为参数的估计应该无偏 也就是
$$E[\hat{\theta}(x)]=\int_x\hat{\theta}(x)p(x\mid\theta)dx=\theta $$
其中的平均是对样本空间中所有可能出现的样本而求的，可实际中样本空间中绝大多数样本尚为出现过，因此持有者条件观点的贝叶斯学派不考虑无偏性 这点是可以被理解的

### 似然原理
似然原理可以帮助我们更好的理解贝叶斯统计的思想 同时理解整个概率统计体系

补充一点：似然 (likehood) 与概率 (probability) 在英语语境中是可以互换的。但是在统计学中，二者有截然不同

概率描述了已知参数时的随机变量的输出结果；似然则用来描述已知随机变量输出结果时，**未知参数的可能取值**
#### 似然函数
若设$\mathbf{x}=(x_1,...,x_n)$是来自密度函数$\mathrm{p( x|\theta) }$的一个样本，则其乘积：
$$
p(\mathbf{x}\mid\theta)=\prod_{i=1}^np(x_i\mid\theta)
$$
有两个解释：
* 当$\theta$给定时，$p(\mathbf{x}|\theta)$是样本x的联合密度函数；
* 当样本x的观察值给定时，$p(\mathbf{x}|\theta)$是未知参数θ的函数 称为似然函数 记作$L(\theta)$

#### 似然原理
* 有了观察值$x$之后，在做关于$\theta$的推断和决策时，所有与试验有关的$\theta$信息均被包含在似然函数$L(\theta)$之中
* 如果有两个似然函数是成比例的，比例常数与$θ$无关，则他们关于$\theta$含有相同的信息

#### 似然原理的例子
##### 介绍
问题的描述：设$\theta$为向上抛一枚硬币时出现正面的概率，现要检验如下二个假设：
$${H_0: ~\theta= 1/2, ~H_1: ~\theta> 1/2}$$
为此做了一系列相互独立的抛此硬币的试验，结果出现9次正面和3次反面。怎样作出合理的判断

这个问题的重要问题是对**一系列相互独立的实验**没有给出足够的信息 他可能有两个情形

我们事先决定进行12次实验 也就是服从二项分布 给出对应的似然函数有$$L_1(\theta)=P_1(X=x\mid\theta)=\begin{pmatrix}n\\x\end{pmatrix}\theta^x\left(1-\theta\right)^{n-x}=220\theta^9\left(1-\theta\right)^3$$
我们希望在三次失败后终止实验 也就是负二项分布 给出对应的似然函数有
$$L_2\left(\theta\right)=P_2\left(X=x\mid\theta\right)=\binom{k+x-1}{x}\theta^x\left(1-\theta\right)^{n-x}=55\theta^9\left(1-\theta\right)^3$$

**似然原理告诉我们这种情况下含有的样本信息是一样的，这符合我们前面的猜测，毕竟只有实验方法的差别**

##### 经典统计学的假设检验
使用经典统计学的假设检验分别处理两个问题 选取$0.05$作为显著性水平 

* 使用二项分布模型 不拒绝$H_0$
* 使用负二项分布模型 拒绝$H_0$

这点和似然原理矛盾了
##### 贝叶斯统计的假设检验
明显的简单对复杂 使用无信息先验分布有
$$\pi(\theta)=\pi_{0}I_{\{0.5\}}(\theta)+\pi_{1}g_{1}(\theta)$$
其中$\pi_0=\pi_1=1/2,\mathrm{~g_1(\theta)=U(0.5,1)}$
计算贝叶斯因子有
$$B_i^{\pi}\left(x=9\right)=\frac{\alpha_0\pi_1}{\alpha_1\pi_0}=\frac{P_i\left(X=9\mid\theta=1/2\right)}{m_i\left(x=9\right)}$$
其中分子
$$P_i\left(X=9\mid\theta=1/2\right)=k_i\theta^9\left(1-\theta\right)^3=0.000244k_i$$
分母
$$\begin{aligned}
m_{i}(x=9)&=\int_{1/2}^{1}P_{i}\left(X=9\mid\theta=1/2\right)g_{1}(\theta)d\theta   \\
&=\int_{1/2}^1k_i\theta^9\left(1-\theta\right)^3\cdot2d\theta  \\
&=2k_i\int_{1/2}^1(\theta^9-3\theta^{10}+3\theta^{11}-\theta^{12})d\theta  \\
&=0.000666k_i
\end{aligned}$$
因此两种情况下的贝叶斯因子实际上相同
$$B_i^\pi(x=9)=\frac{\alpha_0\pi_1}{\alpha_1\pi_0}=\frac{P_i(X=9\mid\theta=1/2)}{m_i\left(x=9\right)}=0.3664$$
贝叶斯因子拒绝$H_0$ 我们选择接受$H_0$
##### 回应矛盾
贝叶斯统计学派是支持似然原理的，因此他们认为经典统计学中给出的假设检验结果是错误的；

对于经典统计学派：事实上，很多的统计方法并不满足似然原理，他们在使用极大似然估计的时候支持似然原理 但是在找到MLE后就不承认似然原理了

有的统计学家认为需要知道$f(x|\theta)$是非常合理的要求，这些的差距就是会导致最后统计推断的结果不同；他们要求**实验设计的方法（思想）已知**

### 贝叶斯点估计
#### 贝叶斯估计的定义

贝叶斯估计一般有三种
* 后验众数估计$\hat{\theta}_{MD}$
* 后验中位数估计$\hat{\theta}_{_{Me}}$ 
* 后验期望估计$\hat{\theta}_E$

他们都可以用来估计参数的值  怎么评估哪个更好后面再考虑

* 后验众数估计$\hat{\theta}_{MD}$的计算方法是：使用数学分析中的知识对后验分布密度函数进行极大化（比如对数后求偏导，找偏导为0的点）
* 后验期望估计$\hat{\theta}_E$ 的计算方法是：使用概率论中的技巧计算后验分布的期望
* 后验中位数估计$\hat{\theta}_{_{Me}}$ 因为不容易计算 所有使用的比较少

#### 几个例子
为估计不合格率 $\theta$, 今从一批产品中随机抽取$n$ 件，其中不合格品数$X$服从$B(n,\theta)$, 一般选取$Be(\alpha,\beta)$ 为$\theta$的先验分布，设$\alpha$,β已知，求$\theta$ 的Bayes估计
根据共轭先验分布的知识 后验分布为
$$Be(\alpha+x,\beta+n-x)$$

则有
$$\hat{\theta}_{MD}=\frac{\alpha+x-1}{\alpha+\beta+n-2},\quad\hat{\theta}_{E}=\frac{\alpha+x}{\alpha+\beta+n}$$

能看出，如果我们选取贝叶斯假设作为先验分布 也就是$\alpha=\beta=1$
$$\hat{\theta}_{_{MD}}=\frac{x}{n},\quad\hat{\theta}_{_E}=\frac{x+1}{n+2}$$
能看出 后验众数估计就是极大似然估计

一些经典统计学中的估计方法 就是贝叶斯估计在某种情况下的特例，这个例子就证明了这一点

同时 这个后验期望估计被后验众数估计更合理一些：当$x$全为0时 后验期望估计考虑到了抽样增加置信增加的情况 后验众数估计则不然

设$x$是来自如下指数分布的**一个观察值**
$$
p(x|\theta)=e^{-(x-\theta)},\quad x\geq\theta 
$$
又取柯西分布作为θ的先验分布，即：
$$\pi(\theta)=\frac{1}{\pi(1+\theta^{2})},\:-\infty<\theta<\infty$$
求θ的最大后验估计$\hat{\theta}_{MD}$

容易计算后验分布为
$$\pi(\theta|x)=\frac{e^{-(x-\theta)}}{m(x)(1+\theta^2)\pi},\theta\le x$$
**分析后验众数估计的时候，分母的边缘密度不是很重要，因为里面不含有$\theta$**

对数后求偏导令结果等于0解得
$$\theta=1$$
这是明显不合理的情况 我们的后验估计一般都会含有抽样给出的结果
否则就是无论抽样情况如何 我们的估计永远是这个值？  这不合理 

直接对后验密度求导不再取对数有
$$\frac{d}{d\theta}\pi(\theta|x)=\frac{e^{-x}}{m(x)\pi}\biggl[\frac{e^\theta}{1+\theta^2}-\frac{2\theta e^\theta}{\left(1+\theta^2\right)^2}\biggr]=\frac{e^{-x}e^\theta\left(\theta-1\right)^2}{m(x)(1+\theta^2)^2\pi}\ge0$$
也就是$\theta$单调增 又因为$\theta\le x$ 因此
$$\hat{\theta}_{MD}=x$$

#### 贝叶斯点估计的精度
在数理统计中 我们使用均方误差作为衡量估计误差的标准 在贝叶斯估计中也一样 使用后验均方差来考虑估计的误差

设参数$\theta$的后验分布为$\pi(\theta|x)$,贝叶斯估计为$\hat{\theta}$, 则$(\theta-\hat{\theta})^2$的后验期望
$$
PMSE(\hat{\theta}\Big|x)=E^{\theta|x}(\theta-\hat{\theta})^{2}
$$
称为$\hat{\theta}$的后验均方差，而其平方根称为后验标准均方差

对PMSE解释有
* $E^{\theta|x}$ 表示用条件分布 $\pi(\theta|x)$求期望
* 当$\hat{\theta}=\hat{\theta}_{E}=\boldsymbol{E}(\theta|x)$时，则$PMSE(\hat{\theta}_{E}\Big|x)=E^{\theta|x}(\theta-\hat{\theta}_{E})^{2}=Var(\theta\big|x)$ 也就是是后验分布的方差
* 后验均方误差和后验方差有关系为
$$\begin{aligned}
PMSE(\hat{\theta}|x)& =E^{\theta|x}(\theta-\hat{\theta})^{2}  \\
&=E^{\theta|x}[(\theta-\hat{\theta}_{E})+(\hat{\theta}_{E}-\hat{\theta}_{})]^{2} \\
&=Var(\theta|x)+(\hat{\theta}_{E}-\hat{\theta})^{2} \\
&\geq Var(\theta|x)
\end{aligned}$$

也就是**后验期望估计是PMSE最小的估计，因此也是我们最常用的估计方法**

能看出 贝叶斯估计精度的计算和评价相较于经典统计学精简了很多 只有一个PMSE了

一个离散型的例子
设一批产品的不合格率为 $\theta$ ,检查是一个一个进行，直到发现第一个不合格品为止，若$X$为发现第一个不合格品时已检查的产品数，则$X$服从几何分布，其分布列为：
$$
P(X=x|\theta)=\theta(1-\theta)^{x-1},x=1,2,\cdots 
$$
设 $\theta$ 的先验分布为 $P(\theta=\frac{t}{4})=\frac{1}{3},i=1,2,3$ , 如今只获得一个$x$
样本观察值$x=3$, 求$\theta$ 的最大后验估计、后验期望估计，并计算它们的误差
离散型的单个样本的后验前面练习比较少 这里给出例子好好理解并且练习一下 后验分布列为
$$P(\theta=i/4|X=3)=\frac{P(X=3,\theta=i/4)}{P(X=3)}=\frac{4i}{5}(1-\frac{i}{4})^{2},i=1,2,3$$
因此 后验众数估计$\hat{\theta}_{MD}=1/4$
后验期望估计 $\hat{\theta}_{E}=E(\theta|X=3)=17/40$
使用二阶原点矩和一阶原点矩的平方计算后验方差 
$$\begin{aligned}
Var(\theta|x)& =E(\theta^{2}\Big|x)-E^{2}(\theta\Big|x)  \\
&=17/80-(17/40)^2=51/1600
\end{aligned}$$
计算后验均方误差PMSE
$$\begin{aligned}
PMSE(\hat{\theta}|x)& =Var(\theta\big|x)+(\hat{\theta}_{MD}-\hat{\theta}_{E})^{2}  \\
&=51/1600+(1/4-17/40)^{2}=\frac{1}{16}
\end{aligned}$$
### 区间估计
#### 可信区间
贝叶斯区间估计和核心是构造可信区间 也就是得到两个统计量 $\hat{\theta}_L=\hat{\theta}_L(x)$ 与$\hat{\theta}_U=\hat{\theta}_U(x)$ 使得
$$
P(\hat{\theta}_L\leq\theta\leq\hat{\theta}_U\mid x)\geq1-\alpha 
$$
这里的可信水平和可信区间与经典统计中的置信水平与置信区间虽是同类的概念，但两者还是有本质的差别

* 可信区间是针对随机变量$\theta$研究的 置信区间是针对确定的数$\theta$研究，多次使用这个置信区间有若干次能覆盖$\theta$ 的频率解释对使用一两次来说没有意义，实际上在实际研究中 置信区间往往被当作可信区间使用
* 在经典统计学的区间估计中，构造枢轴量是不容易的，但是可信区间不需要这样的构造，只使用后验分布更好计算
#### 等尾的可信区间
和数理统计中等尾的区间估计思路是一样的 事实上，由于贝叶斯的区间估计只使用后验密度就可以完成 其难度事实上比数理统计上难度更低

在已知后验分布的情况下 只需要查表就可以得到各处的概率 比如
$$\theta_L=\theta_{0.01},\theta_R=\theta_{0.91};\\
\\\theta_L=\theta_{0.05},\theta_R=\theta_{0.95}$$
都是百分之九十的可信区间 我们应该选择哪一个？

在这一小节 我们要求使用等尾的分布 也就是
$$\theta_L=\theta_{\frac{\alpha}{2}},\theta_R=\theta_{1-\frac{\alpha}{2}}$$
我们只需要给出下限和上限的值就可以了

分布比较简单的时候可以通过查表得到我们需要的结果；当分布比较复杂不好直接研究的时候，计算机技术可以帮助我们进行运算
#### 最大后验密度（HPD）可信区间
##### 定义
等尾区间是最好的可信区间吗？ 事实上我们在数理统计学中已经介绍过了，最好的置信区间应该有着最短的区间长度，之所以我们使用等尾是因为在分布函数对称的情况下有着最短的区间长度并且足够简单 

这里我们来介绍如何去寻找最短的可信区间（置信区间同理） 也就是HPD可信区间；

定义：设参数$\theta$的后验密度为$\pi(\theta|x)$, 对给定的概率1-$\alpha(0{<}\alpha{<}1)$, 若在直线上存在这样一个子集$C$，满足下列二个条件：
* $\mathrm{P(C|x)=1-\alpha}$
* $\text{对任给}\theta_1{\in}\mathbb{C}\text{和 }\theta_2\notin C\text{,总有}\pi(\theta_1|\mathbf{x}){\geq}\pi(\theta_2|\mathbf{x})$
则称C是$\theta$的可信水平为(1-α)的最大后验密度可信集， 简称(1-α)HPD可信集，如果C是一个区间，则C又称为$1-\alpha$ HPD可信区间

简单理解一下这个HPD可信区间 非常好理解，就是去寻找那些密度函数的纵坐标更大的区间，这样能实现可信区间长度的最小化 如下所示
![贝叶斯统计 2](/assets/images/probability-statistics-notes/bayesian-statistics-inference-and-decision-notes-01.png)
##### 一些解释
一些关于HPD可信区间的基本解释
* 离散型随机变量的后验很难计算HPD可信集，一般不研究
* 单峰后验密度的HPD可信区间总是存在
* 多峰后验密度往往会得到多个互不连接的区间组成HPD可信集
关于多峰后验密度
* 多峰后验密度的出现，常常是由于先验信息与抽样信息不一致引起的，研究这种抵触对贝叶斯统计来说很重要
* 共轭先验分布大多是单峰的，这必定导致后验分布也是单峰的，他可能会掩盖原本应该产生的多峰后验这种抵触，所以要慎重使用共轭先验
使用HPD可信区间的时候应该注意
*  HPD可信区间不是严格意义上的可信区间
* 单峰、对称时，HPD可信区间也是等尾可信区间；
* 单峰、非对称时，借助计算机数值方法求解；
* 多峰时，建议放弃HPD准则，采用相连接的等尾可信区间估计
##### 计算机数值求解单峰非对称HPD可信区间
事实上 这个计算机迭代的过程非常好理解
* 给定初始的$k$值
* 计算$\pi(\theta|x)=k$ 得到$\theta_1,\theta_2$
* 计算$\theta_1,\theta_2$上$\pi(\theta|x)$的积分 
* 如果大于需要的置信度 增大$k$ 否则减小$k$ 从第二步继续迭代
理解这个思路就足够了
##### 大样本方法
在大样本的情况下 使用近似HPD可信区间

在适当条件下可证明： 当 $n$ 充分大时，$\pi_n(\theta|x)$近似服从 $N(\mu^\pi(x),V^\pi(x))$ , 此处$\mu^\pi(x)$和$V^\pi(x)$分别为后验均值和后验方差

也就是后验目前近似于正态分布 他是一个对称的单峰分布，HPD可信区间和等尾可信区间一致 容易给出$\theta$ 的可信水平近似为 1-$\alpha$ 的HPD区间为
$$(\mu^\pi(x)-u_{\alpha/2}\sqrt{V^\pi(x)},\mu^\pi(x)+u_{\alpha/2}\sqrt{V^\pi(x)})$$

**大样本不是不用计算后验分布**
举一个比较有意思的大样本HPD例子
某币每周火灾事件数 X 服从泊松分布 $P(\theta)$,关于 θ 的先验分布一无所知，认为无信息先验$\pi(\theta)=\theta^{-1}I_{(0,\infty)}(\theta)$ 是合适的. 设 5 周中火灾事故总次数为 3, 求泊松分布均值 $\theta$ 的可信水平为 90%的 HPD 可信区间，使用大样本的方法；
我们应该去计算后验分布 但是5周是是五次观测 我们只有观测的总次数为3 怎么计算后验分布呢？
泊松分布的核为
$$\theta^ke^{-\theta}$$
我们不妨这5周的观测时 1 1 1 0 0 （其他的设法并不影响后验分布的计算结果）
所以样本联合的核为
$$\theta^3e^{-5\theta}$$
所以后验分布的核为（先验里有示性变量，很好理解）
$$\theta^2e^{-5\theta}$$
这是Gamma分布的核 计算得后验分布的均值为 $\frac{3}{5}$ 方差为$\frac{3}{25}$
根据大样本方法有 研究$N(\frac{3}{5},\frac{3}{25})$的百分之九十 HPD 可信区间 由于正态分布是对称的 所以研究等尾可信区间就可以了
### 假设检验
假设检验也是整个经典统计学中研究的一大类问题，在完成估计以后，检验我们的估计是否合理是非常自然的思想，想较于经典统计学中需要构造假设检验统计量，贝叶斯统计在这方面又展现了他的优势；
#### 一般的假设检验方法
##### 经典统计中的假设检验
1. 建立原假设$H_0$和备择假设$H_1$
2. 选择检验统计量$T=T(x)$ 当原假设$H_0$为真的时候其分布已知
3. 对给定的显著性$\alpha$ 确定拒绝域 保证犯第一类错误（拒真）的概率小于$\alpha$ 
4. 样本观察值$x$落入拒绝域的时候 拒绝原假设$H_0$ 否则保留原假设

和枢轴量的构造一样，经典统计学中检验统计量的确定是比较困难的
##### 贝叶斯统计中的假设检验
1. 获得后验概率$\pi(\theta|x)$ 后 分别计算假设$H_0$ $H_1$ 的后验概率 $\alpha_i=P(\theta_i|x)$ 
2. 当后验概率比（机会比）$\frac{\alpha_0}{\alpha_{1}}>1$ 时不拒绝$H_0$  $\frac{\alpha_0}{\alpha_{1}}<1$  时不拒绝$H_{1}$ 接近$1$的时候不做判断 无法给出结论

##### 两个学派的假设检验思想比较
容易看出
* 贝叶斯假设检验更易理解、更简单
* 贝叶斯假设检验无需选择检验统计量，确定抽样分布
* 无需事先给出显著性水平，确定其拒绝域
* 易推广到多重假设检验的场合 还是去寻找最大后验概率的假设（理解思路就可以）

事实上 贝叶斯统计的假设检验也是用和经典统计学一样的小概率原理，只是不需要使用反证法了
##### 贝叶斯假设检验的详细解释
我们这里介绍如何进行贝叶斯假设检验

后验概率密度计算：略
假设$H_0$  假设$H_1$ $H=H_{0}\cup H_1$ 为总空间 所有的假设都意味着$\theta\in H$ 
1. 计算假设$H_0$  的后验概率 $P(H_0|x)=\int_{H_0}^{}\pi(\theta|x)d\theta\triangleq\alpha_0$ 
2. 计算假设$H_1$  的后验概率 $P(H_1|x)=\int_{H_1}^{}\pi(\theta|x)d\theta\triangleq\alpha_1$ 
3. 计算后验概率比$\frac{\alpha_0}{\alpha_{1}}$

* $\frac{\alpha_0}{\alpha_{1}}>1$ 时不拒绝$H_0$  也就是接受$H_0$
* $\frac{\alpha_0}{\alpha_{1}}<1$  时不拒绝$H_{1}$ 也就是接受$H_1$
* $\frac{\alpha_0}{\alpha_{1}}\approx 1$ 不做判断 无法给出结论 需要进一步补充抽样信息或者修正先验信息

贝叶斯统计中假设检验问题被转化为积分问题 这里定义什么是简单假设与复杂假设
* 简单假设：指此时我们的假设为$\theta=x$ 
* 复杂假设：假设对应的参数取值为一个区间

#### 贝叶斯因子与假设检验
贝叶斯因子可以帮助我们更好的理解贝叶斯假设检验问题

设两个假设$\Theta_{0}$与$\Theta_{1}$的先验概率分别为$\pi_0$与$\pi_1$, 后验概率分别为$\alpha_0$与$\alpha_1$, 则称
$$B^\pi(x)=\frac{\text{后验机会比}}{\text{先验机会比}} = \frac { \alpha _ 0 / \alpha _ 1 }{ \pi _ 0 / \pi _ 1 }=\frac{\alpha_0\pi_1}{\alpha_1\pi_0}$$
为贝叶斯因子（Bayes factor）

能看出
* 贝叶斯因子同时依赖于数据$x$和先验分布$\pi(\theta)$
* 两种机会比做除法会减弱先验分布的影响 突出数据的影响（我们在后面的描述中会继续介绍贝叶斯因子）
* 贝叶斯因子反映数据$x$支持原假设$H_0$的程度（和各种机会比一样 以1为分界线）

#### 简单假设对简单假设
我们现在在几个不同假设的情况下研究一下贝叶斯因子  同时也继续强化我们给出的最核心的假设检验方法：后验概率比

首先研究简单假设对简单假设
假设情形为
$$H_{0}:\Theta_{0}=\{\theta_{0}\}\leftrightarrow H_{1}:\Theta_{1}=\{\theta_{1}\}.$$
对应的后验概率有
$$\begin{aligned}\alpha_0&=P(\Theta_0|\boldsymbol{x})=\frac{f(\boldsymbol{x}|\theta_0)\pi_0}{f(x|\theta_0)\pi_0+f(\boldsymbol{x}|\theta_1)\pi_1},\\\alpha_1&=P(\Theta_1|\boldsymbol{x})=\frac{f(\boldsymbol{x}|\theta_1)\pi_1}{f(\boldsymbol{x}|\theta_0)\pi_0+f(\boldsymbol{x}|\theta_1)\pi_1},\end{aligned}$$
*就是用定义进行了计算 此时因为对应了离散概率空间 所以这样写才方便计算 使用密度函数的形式只会出现零概率的情况*

故后验机会比
$$\frac{\alpha_0}{\alpha_1}=\frac{\pi_0f(\boldsymbol{x}|\theta_0)}{\pi_1f(\boldsymbol{x}|\theta_1)}$$
计算贝叶斯因子
$$B^\pi(\boldsymbol{x})=\frac{\alpha_0/\alpha_1}{\pi_0/\pi_1}=\frac{f(\boldsymbol{x}|\theta_0)}{f(\boldsymbol{x}|\theta_1)}.$$
想要拒绝原假设 也就是要求 $\frac{\alpha_0}{\alpha_{1}}<1$ 也就是$$\frac{f(\boldsymbol{x}|\theta_1)}{f(\boldsymbol{x}|\theta_0)}>\frac{\pi_0}{\pi_1}.$$

直观理解就是密度函数值的比要大于临界值 这和N-P引理的基本结果类似
* 从此可以看出 $B^\pi(\boldsymbol{x})$ 应该被看作数据的机会比 他完全不依赖于先验分布 只依赖于样本的情况
* 因此我们将贝叶斯因子$B^\pi(\boldsymbol{x})$ 看作数据$x$对假设$H_0$的支持程度

#### 复杂假设对复杂假设
##### 计算贝叶斯因子
考虑以下的假设检验问题
$$H_0:\theta\in\Theta_0\leftrightarrow H_1:\theta\in\Theta_1,$$
此时我们可以用下面的形式改写先验密度函数*改写的目的是方便后面的计算和表示*
$$\left.\pi(\theta)=\left\{\begin{array}{ll}\pi_0g_0(\theta),&\theta\in\Theta_0,\\\pi_1g_1(\theta),&\theta\in\Theta_1,\end{array}\right.\right.$$

在这种记号下改写后验概率比（还是约掉了分母）
$$\frac{\alpha_0}{\alpha_1}=\frac{\int_{\Theta_0}f(\boldsymbol{x}|\theta)\pi_0g_0(\theta)\mathrm{d}\theta}{\int_{\Theta_1}f(\boldsymbol{x}|\theta)\pi_1g_1(\theta)\mathrm{d}\theta},$$
*使用积分的形式是因为复杂对复杂的密度函数不需要处理单点 连续区间积分更合适*
给出贝叶斯因子
$$B^\pi(\boldsymbol{x})=\frac{\alpha_0/\alpha_1}{\pi_0/\pi_1}=\frac{\int_{\Theta_0}f(\boldsymbol{x}|\theta)g_0(\theta)\mathrm{d}\theta}{\int_{\Theta_1}f(\boldsymbol{x}|\theta)g_1(\theta)\mathrm{d}\theta}=\frac{m_0(\boldsymbol{x})}{m_1(\boldsymbol{x})}.$$
也就是边缘分布变形的比

##### 解释贝叶斯因子
* 此时的贝叶斯因子还是和先验分布的情况有关 贝叶斯因子并不是似然比，但是可以看作是似然比的加权形式 部分消除了先验分布的影响 强调样本
* 若设$\hat{\theta}_0$ 与$\hat{\theta}_1$ 分别是$\theta$在$\Theta_{0}$与$\Theta_{1}$上的极大似然估计(MLE), 那么经典统计中所使用的似然比统计量是贝叶斯因子${B}^\pi(\mathbf{x})$的特殊情况
* 贝叶斯因子对样本信息变化的反应是灵敏的，而对先验信息变化的反应是迟钝的（这点是针对复杂对复杂这样解释的，简单对简单假设的贝叶斯因子和先验完全无关）
#### 简单假设对复杂假设
考虑以下的假设检验问题
$${H}_0：\theta=\theta_{0}~~{ H}_1:\theta\neq\theta_0$$
这种情况是最复杂的了

如果我们直接使用连续的密度函数那么单点的先验概率一定是零 后面也就没办法计算了
因此我们需要通过像复杂对复杂的方式补充参数 改写密度函数来解决问题 

变形有
$$\pi(\theta)=\pi_0I_{\theta_0}(\theta)+\pi_1g_1(\theta)$$
其中的$I$是补充进来的示性函数 只有$\theta=\theta_0$的时候取为1  $\pi_0+\pi_1=1$ 我们可以认为此时的先验密度时由离散和连续两部分组成的 因此有
$$\left.\pi(\theta)=\left\{\begin{array}{ll}\pi_0,&\theta=\theta_0,\\\pi_1g_1(\theta),&\theta\neq\theta_0,\end{array}\right.\right.$$
计算边缘密度有
$$m(\boldsymbol{x})=\int_\Theta f(\boldsymbol{x}|\theta)\pi(\theta)\mathrm{d}\theta=\pi_0f(\boldsymbol{x}|\theta_0)+\pi_1m_1(\boldsymbol{x}),$$
其中$m_1(x)$为$$m_1(\boldsymbol{x})=\int_{\theta\neq\theta_0}f(\boldsymbol{x}|\theta)g_1(\theta)\mathrm{d}\theta.$$
分别计算两个假设下的后验密度有
$$\alpha_0=\pi(\Theta_0|\boldsymbol{x})=\frac{\pi_0f(\boldsymbol{x}|\theta_0)}{m(\boldsymbol{x})},\quad\alpha_1=\pi(\Theta_1|\boldsymbol{x})=\frac{\pi_1m_1(\boldsymbol{x})}{m(\boldsymbol{x})}.$$
因此可以计算后验机会比
$$\frac{\alpha_0}{\alpha_1}=\frac{\pi_0f(\boldsymbol{x}|\theta_0)}{\pi_1m_1(\boldsymbol{x})}.$$
计算贝叶斯因子有
$$B^\pi(\boldsymbol{x})=\frac{\alpha_0/\alpha_1}{\pi_0/\pi_1}=\frac{f(\boldsymbol{x}|\theta_0)}{m_1(\boldsymbol{x})}.$$

能看出 贝叶斯因子的表示形式时更加简单的 而且没有我们为了辅助研究补充进来的两个参数 因此在实际的研究中我们往往是先计算贝叶斯因子的，使用贝叶斯因子再去计算后验概率是一个很简单的方程问题
#### 一个例子
这个例子并不是讲解关于计算的知识 我们要解释计算的结果

设从正态总体$\mathbb{N}(0,1)$中随机抽取一个容量为$10$的样本x，算得样本均值 $\overline{x}=1.5$, 试对如下两个假设进行检验：
$$
{H}_0{}\text{ θ≤1, H}_1{}\text{ θ>1}
$$
$\text{取}\theta\text{的共轭先验分布为N}(0.5,2)\text{}$

明显是复杂对复杂 带入公式计算得到后验机会比
$$\begin{aligned}\alpha_0=&\mathrm{P(\theta\leq1|x)=0.0708}\\\alpha_1=&\mathrm{P(\theta>1|x)=1-\alpha_0=0.9292}\end{aligned}$$
后验机会比支持假设$H_1$

计算先验机会比得到
$$\pi=0.6368,\quad\pi_1–0.3632$$
先验机会比支持$H_0$

在计算贝叶斯因子
$$B^\pi(x)=0.0434$$
也就是贝叶斯因子支持假设$H_1$

能看出我们的贝叶斯因子和先验的判断是矛盾的 这也回应了我们给出的结论 **贝叶斯因子更多的考虑样本信息 事实上这句话对于任何形式的贝叶斯假设检验问题都是成立的** 

### 预测推断
我们是在对随机变量未来的观察值作出统计推断，在数理统计中并没有与之对应的一个章节
#### 简单介绍
我们需要做的是根据随机变量已知的情况估测随机变量未来观察值 基本上会分为以下的情况
* 无观测信息，参数$θ$未知，预测$X$（$X$含有$\theta$参数）
* 有观测信息，参数$θ$未知，预测$X$（$X$含有$\theta$参数）
* 有观测信息，参数$θ$未知，预测$Z$（$Z$含有$\theta$参数）
#### 在无观测数据情形下的预测
虽然此时我们没有任何观测信息 但是有样本分布（含有参数） 和 参数的先验分布 非常自然的给出边缘分布
$$m(x)=\int_{\Theta}p(x|\theta)\pi(\theta)d\theta $$
作为我们的预测分布 此时称为先验预测分布

预测方法：
使用预测分布的期望值、中位数或众数作为预测值（正如我们在贝叶斯点估计中的操作一样）

使用某个置信度计算预测分布的置信区间（正如我们在贝叶斯区间估计中干的一样）
#### 有X的观测数据时预测X
计算后验密度$\pi(\theta|{x})$  使用后验密度计算我们的预测分布
$$m(x\mid\mathbf{x})=\int_{\Theta}p(x\mid\theta)\pi(\theta\mid\mathbf{x})d\theta $$
称为后验预测分布
预测方法不变
#### 有X的观测数据时预测Z
计算后验密度$\pi(\theta|{x})$  使用后验密度计算我们的预测分布
$$m(z\mid\mathbf{x})=\int_{\Theta}g(z\mid\theta)\pi(\theta\mid\mathbf{x})d\theta $$称为后验预测分布
预测方法不变

### 贝叶斯假设检验与模型选择
#### 多假设情形的贝叶斯假设检验
前面的假设检验局限于原假设和备择假设之间的联系 [贝叶斯统计1(贝叶斯统计与后验分布)](/blog/2023/10/28/bayesian-statistics-posterior-distributions-notes/) 的“贝叶斯假设检验与模型选择”一节  但是 贝叶斯假设检验的一个重要优点就在于非常容易推广到多假设情形；

我们只需要计算多个假设之间的后验概率比或者贝叶斯因子，根据其大小决定我们是否接受原假设就可以了；

至于贝叶斯因子的大小与模型支持分子上假设的关系 Jeffeys 给出了一些建议

| 贝叶斯因子      | 解释           |
| ---------- | ------------ |
| $B<1$      | 否定分子上的假设     |
| $1<B<3$    | 支持分子上的假设证据不足 |
| $3<B<10$   | 较强的支持        |
| $10<B<30$  | 强烈的支持        |
| $30<B<100$ | 非常强烈的支持      |
| $100<B$    | 肯定支持         |
#### 贝叶斯模型评价
##### 贝叶斯模型评价的重要性
贝叶斯统计推断与决策都依赖于后验分布进行；因此导致推断的结果研究依赖于后验分布的质量；因此评价我们的贝叶斯模型就非常重要了；常用的贝叶斯模型评价方法不仅包括从经典统计学中引入的AIC BIC准则 还有BPIC
##### AIC与BIC
他们都是基于极大似然原理 也就是MLE估计的思想[数理统计](/blog/2023/03/18/mathematical-statistics-notes/) 的“极大似然原理”一节
构建的

AIC准则的形式为 
$$AIC=-2\ln f\left(x_{n}|\widehat{\theta}_{MLE}\right)+2p,$$
其中$\widehat{\theta}_{MLE}是\theta 的最大似然估计\left(MLE\right)$ $p$是估计参数的维数

BIC准则的形式为 
$$BIC=-2\ln f\left(x_{n}|\widehat{\theta}_{MLE}\right)+p\ln n$$
我们的目的都是极小化二者
##### 贝叶斯预测信息准则 BPIC
考虑下列两个假设：(a) 参数模型 $f(x|\theta)$ 包含了真实的模型 $g(x)=f(x;\theta_0)$ $\theta_0\in\Theta$,且指定的模型并不远离真实模型；(b) 对数先验的阶为 $\ln\pi(\theta)=O_p(1).$ 在上述两个假定和某些正则条件下，Ando (2007) 提出贝叶斯预测信息准则 (the Bayesian predictive information criterion, BPIC)
$$
BPIC=-2\int_{\Theta}\ln f\left(x_{n}|\theta\right)\pi\left(\theta|x_{n}\right)d\theta+2p,
$$

由于对数似然的后验均值一般没有解析表达 所以我们一般使用MC方法逼近
$$\int_{\Theta}\ln f\left(x_{n}|\theta\right)\pi\left(\theta|x_{n}\right)d\theta\approx\frac{1}{L}\sum_{j=1}^{L}\ln f\left(x_{n}|\theta^{\left(j\right)}\right),$$
**这个方法适用于先验较弱的情形**
##### 偏差信息准则 DIC
令$D\left(\theta\right)=-2\ln f\left(x_{n}|\theta\right)$,它是常用的模型偏差的一种度量. Spiegelhalter 等 (2002) 指出对数似然的后验期望 $\bar{D}=E[D(\theta)|x_n]$, 可以作为模型拟合程度的一个贝叶斯度量. 一个模型拟合数据的程度越高，$\bar{D}$ 越小. 下面定义有效参数个数来刻画模型的复杂程度：
$$
p_{D}=\overline{D}-D(\overline{\theta}_{n})=2\ln f(\boldsymbol{x}_{n}|\overline{\boldsymbol{\theta}}_{n})-2\int_{\Theta}\ln f(\boldsymbol{x}_{n}|\boldsymbol{\theta})\pi(\boldsymbol{\theta}|\boldsymbol{x}_{n})\mathrm{d}\boldsymbol{\theta},
$$

其中$\vec{\theta}_n$ 为后验均值. Spiegelhalter 等 (2002) 定义偏差信息准则 (Deviance information criterion, DIC) 为

$$
DIC=\overline{D}+p_{D}=-2\int_{\Theta}\ln f\left(x_{n}|\theta\right)\pi\left(\theta|x_{n}\right)\mathrm{d}\theta+p_{D},\left(4.6.13\right)
$$
其中第一项 $\tilde{D}$ 可解释为模型拟合程度的一个度量，越小越好；第二项 $p_D$ 被认为是模型复杂性的一种度量，上述定义的 $DIC$ 可以改写为 $DIC=D\left(\overline{\theta}_{n}\right)+$ $2p_{D}=-2\ln f\left(\boldsymbol{x}_{n}|\overline{\boldsymbol{\theta}}_{n}\right)+2p_{D}$,其中$\overline{\boldsymbol{\theta}}_{n}$ 为后验均值，从形式上看，它与 AIC 很相似，因此可以认为 $DIC$ 是 $AIC=D(\widehat{\theta}_{MLE})+2p$ 的一个推广，此处 $\widehat{\theta}_{MLE}$ 为$\theta$ 的最大似然估计，对非分层模型而言，当$n$ 充分大时有 $p\approx p_D,\widehat{\theta}_{MLE}\approx\overline{\theta}_n$, 从而 DIC\approx AIC.

**DIC 可以通过 MCMC 方法 容易计算其结果.因此DIC 准则被用于各种贝叶斯模型选择问题**


## 统计决策
### 引言
在统计决策三要素：样本空间与分布族 行动空间 损失函数 之外，贝叶斯统计决策在统计决策的基础上引入了第四个要素 先验分布函数$F(\theta)$ 

贝叶斯统计决策在统计贝叶斯推断的基础上引入了第四个要素 损失函数$L$

在这里我们作出一个约定：
如果作出决策的数据没有收到随机性的影响 就称为普通的决策问题（所有的确定型决策都属于此类）
相反的 如果收到了随机性的影响 就称为统计决策问题
### 后验风险最小原则
后验风险最小原则对统计决策的意义就和后验概率针对于统计推断的意义一样 是贝叶斯统计决策的灵魂
#### 后验风险定义
我们把损失函数对后验分布的期望称为后验风险函数
$$R(\delta(x)|x)=E^{\boldsymbol{\theta}|x}[L(\theta,\delta(x))]$$
$$\left.=\left\{\begin{array}{l}\int_\Theta L(\theta,\delta(x))\pi(\theta|x)\mathrm{d}\theta,\\\sum_iL(\theta_i,\delta(x))\pi(\theta_i|x),\end{array}\right.\right.$$
他和贝叶斯期望损失一脉相承 一个使用先验概率一个使用后验概率

如果存在决策函数使得后验风险最小 我们称之为后验风险最小准则下的最优贝叶斯决策函数
#### 后验风险和贝叶斯风险的关系
在贝叶斯统计推断中我们就知道
$$f(x,\theta)=f(x|\theta)\pi(\theta)=\pi(\theta|x)m(x)$$
**这就是后验分布计算公式移项**

用它来给贝叶斯风险做变形有
$$R_{\pi}(\delta(x))=E^{\theta}\bigl[R(\theta,\delta(x))\bigr]=E^{X}\bigl[R(\delta(x)|x)\bigr]$$
也就是贝叶斯风险 这一用于计算贝叶斯解的核心有两个等价表达式

**一个是：先计算风险函数 再对它用先验概率密度$\pi(\theta)$求均值**
**另一个是：先计算后验风险 再对他用边缘分布$m(x)$求均值** 

证明如下
$$\begin{aligned}
R_{\pi}(\delta)& =E^{\theta}[R(\theta,\delta(x))]  \\
&=\int_{\Theta}R(\theta,\delta(x))\pi(\theta)d\theta  \\
&=\int_{\Theta}\int_{\chi}L(\theta,\delta(x))f(x\mid\theta)\pi(\theta)dxd\theta  \\
&=\int_{\chi}\biggl[\int_{\Theta}L(\theta,\delta(x))\pi(\theta\mid x)d\theta\biggr]m(x)d\mathbf{x} \\
&=E^{\mathrm{x}}\Big[R(\delta(x)|x)\Big],
\end{aligned}$$
#### 后验风险最小原则
我们将证明 ：**后验风险最小原则下的决策函数就是贝叶斯解**
定理：设存在非随机化决策函数 $\delta_{\pi}(x)$, 满足条件
$$
R(\delta_{\pi}(x)|x)=\operatorname*{inf}_{\delta}R(\delta(x)|x)=\operatorname*{inf}_{\delta}\int_{\Theta}L(\theta,\delta(x))\pi(\mathrm{d}\theta|x),
$$
 则 $\delta_\pi(x)$ 为先验分布 $\pi(\theta)$ 下的贝叶斯解 $\pi(\mathrm{d}\theta|x)=\pi(\theta|x)\mathrm{d}\theta.$
 
 如果$\pi(\theta)$是广义先验分布 我们按照上面做法得到的是广义贝叶斯解 除此以外定理表述不发生变化
 **理解前面这么多繁杂的概念就能理解我们的操作了**
 
 证明如下： 后验风险最小有
$$\begin{aligned}R(\delta(x)|x)&=\int_\Theta L(\theta,\delta(x))\pi(\mathrm{d}\theta|x)\\&\geqslant\int_\Theta L(\theta,\delta_\pi)\pi(\mathrm{d}\theta|x)=R(\delta_\pi(x)|x),\end{aligned}$$
两边同时对边缘分布$m(x)$做积分有
$$\begin{aligned}
R_{\pi}(\delta(x))& =\int_{\mathcal{X}}R(\delta(x)\mid x)m(x)\mathrm{d}x  \\
&\geqslant\int_{\mathcal{X}}R(\delta_\pi(x)|x)m(x)\mathrm{d}x=R_\pi(\delta_\pi(x)).
\end{aligned}$$
证明了 后验风险最小也是贝叶斯风险最小 是贝叶斯解

#### 一个简单的例子
设$\theta$的先验分布为 $\pi(\theta_1)=0.6~\pi(\theta_2)=0.4$ 设随机变量$X$取0 1两个值 记$p(i|\theta_j)=P(X=i|\theta=\theta_j)$  有$X$的概率分布为
$$p(1|\theta_1)=0.1,\quad p(1|\theta_2)=0.2,\quad p(0|\theta_1)=0.9,\quad p(0|\theta_2)=0.8.$$
计算后验概率 如果我们知道损失函数为$L$ 计算后验风险

离散型的后验概率总是会比较绕 不像连续型一样做的那么多 但是本质上还是代入公式进行运算
$$\pi(\theta_i|x)=\frac{f(x|\theta_i)\pi(\theta_i)}{\sum_if(x|\theta_i)\pi(\theta_i)}\quad(i=1,2,\cdots).$$
根据题意 我们要计算两种样本下的后验概率 一个是$X=0$ 另一个是$X=1$
**当我们计算边缘密度的时候，不要代入具体的$\theta$，因为要取遍所有**

后验风险就是损失函数对后验密度积分 无论是损益矩阵还是连续的函数都一样
### 一般损失函数下的贝叶斯估计
我们使用决策的方法考虑统计推断中的贝叶斯点估计问题

其中行动空间为所有的点估计量 最后得到的贝叶斯解就是推断问题中想要求的参数估计量
#### 平方损失函数下的贝叶斯估计
如果使用损失函数为
$$L(\theta,\delta)=(\delta-\theta)^{2}$$
那么我们知道$\theta$的贝叶斯估计就是后验均值 也就是
$$\delta_{_B}(x)=E(\theta|x)$$
证明如下：
$$\begin{gathered}
R(a|\boldsymbol{x}) =E[(\theta-a)^2|x]=\int_\Theta(\theta-a)^2\pi(\theta|\boldsymbol{x})\mathrm{d}\theta  \\
=\int_{\Theta}(\theta^{2}-2a\theta+a^{2})\pi(\theta|\boldsymbol{x})\mathrm{d}\theta. 
\end{gathered}$$
我们需要找到合适的$a$ 让后验风险最小 对$a$求偏导有
$$\frac{\mathrm{d}R(a|\boldsymbol{x})}{\mathrm{d}a}=-2\int_{\Theta}\theta\pi(\theta|\boldsymbol{x})\mathrm{d}\theta+2a=0.$$
因此$a$ 等于后验均值的时候实现极小化 

**直接在$\Theta$上对先验密度函数，后验密度函数上的积分都是1，这是密度函数的性质决定的**

加权平方损失函数的形式
如果使用损失函数为
$$L(\theta,\delta)=w(\theta)(\delta-\theta)^{2}$$
那么我们知道$\theta$的贝叶斯估计就是
$$\delta_{_B}(x)=\frac{E[w(\theta)\theta|x]}{E[w(\theta)|x]}$$
他的计算式为 也就是对后验密度求期望
$$\frac{\int_a\theta w(\theta)\pi(\theta\mid x)\mathrm{d}\theta}{\int_aw(\theta)\pi(\theta\mid x)\mathrm{d}\theta}$$
这个期望的计算还是比较复杂的 一般是构造新的分布积分为1实现
证明如下
$$\begin{aligned}
R(a|\boldsymbol{x})& =E\bigl[w(\theta)(\theta-a)^2|\boldsymbol{x}\bigr]  \\
&=\int_{\boldsymbol{\Theta}}\left[\theta^2w(\theta)-2a\theta w(\theta)+a^2w(\theta)\right]\pi(\theta|\boldsymbol{x})\mathrm{d}\theta.
\end{aligned}$$
求偏导有
$$\frac{\mathrm{d}}{\mathrm{d}a}[R(a|\boldsymbol{x})]=-2\int_\Theta\theta w(\theta)\pi(\theta|\boldsymbol{x})\mathrm{d}\theta+2a\int_\Theta w(\theta)\pi(\theta|\boldsymbol{x})\mathrm{d}\theta=0$$
解方程就可以得到上面的结论了

当参数向量是多元的 $\theta^{\prime}=(\theta_1,\cdotp\cdotp\cdotp,\theta_k)$ 对于多元二次损失函数
$$L(\theta,\delta)=(\delta-\theta)^{\prime}Q(\delta-\theta)$$
贝叶斯估计为后验均值估计
$$\delta_B(x)=E(\theta\mid x)=\begin{pmatrix}E(\theta_1\mid x)\\\vdots\\E(\theta_k\mid x)\end{pmatrix}$$
#### 线性损失函数下的贝叶斯估计
我们取损失函数为线性函数
$$L(\theta,\delta)=\begin{cases}k_0(\theta-\delta),\delta\leq\theta\\k_1(\delta-\theta),\delta>\theta&\end{cases}$$
他的贝叶斯估计是后验分布$\pi(\theta|x)$ 的$\frac{k_0}{k_{0}+k_{1}}$分位数

特殊的 如果使用损失函数为（也就是绝对值损失函数）
$${L}(\theta,\delta)=|\theta\text{-}\delta|$$
贝叶斯估计是后验中位数

### 假设检验和有限行动问题
在估计问题中 行动往往是无穷多可选的 但是很多统计决策的问题只能在有限个行动中选择 比如假设检验问题 对于这类问题贝叶斯统计决策是很好处理的

行动空间为 $A=\{a_{1},a_{2},...,a_{r}\},$  损失为 $L(\theta,a_i)$ ，想要找到最优的行动就要让后验期望损失 $E^{\theta|\mathbf{x}}[L(\theta,a_i)]$ 达到最小

下面研究两种问题 两行动（假设检验） 多行动（分类）问题
#### 假设检验问题
考虑以下假设检验问题
$$H_0:\theta\in\Theta_0\leftrightarrow H_1:\theta\in\Theta_1\quad(\Theta_0\cup\Theta_1=\Theta).$$
使用行动$a_0$表示接受原假设 行动$a_1$ 表示否定原假设

选用$0-k_i$损失函数如下
$$\left.L(\theta,a_0)=\left\{\begin{array}{ll}0,&\theta\in\Theta_0,\\k_0,&\theta\in\Theta_1,\end{array}\right.\right.$$
$$\left.L(\theta,a_1)=\left\{\begin{array}{ll}k_1,&\theta\in\Theta_0,\\0,&\theta\in\Theta_1.\end{array}\right.\right.$$
当然是行动空间和参数空间的函数

后验风险为
$$\begin{gathered}
R(a_0\mid x)=E^{\theta\mid x}[L(a_0,\theta)]=\int_{\Theta_1}k_0\pi(\theta\mid x)d\theta=k_0P(\Theta_1\mid x) \\
R(a_1\mid x)=E^{\theta\mid x}[L(a_1,\theta)]=\int_{\Theta_0}k_1\pi(\theta\mid x)d\theta=k_1P(\Theta_0\mid x) 
\end{gathered}$$
按照后验风险准则 比较后验风险的大小确定最优行动就可以了

拒绝原假设则有
$$k_0P\left(\Theta_1|x\right)\geqslant k_1P\left(\Theta_0|x\right),$$
等价于
$$P\left(\Theta_{1}|x\right)\geqslant\frac{k_{1}}{k_{0}+k_{1}}.$$
这与经典统计中贝叶斯假设检验的拒绝域
$$D=\left\{X=\left(X_{1},X_{2},\cdots,X_{n}\right):P\left(\Theta_{1}|X=x\right)\geqslant\frac{k_{1}}{k_{0}+k_{1}}\right\},$$
*这个形式我们在似然比检验中见到过*

#### 多行动问题
对于多个行动的问题 我们处理问题的思路是没有发生变化的，针对每个行动 我们独立的给出损失函数的表达形式（比较自然）

**这种损失函数要根据题意相对合理的选择**

然后计算后验风险 比较后验风险的大小来作出最后的决策 思想和我们在前面直接研究假设检验的时候是一样的
[贝叶斯统计1(贝叶斯统计与后验分布)](/blog/2023/10/28/bayesian-statistics-posterior-distributions-notes/) 的“贝叶斯假设检验与模型选择”一节
#### 统计决策中的区间估计
考虑应用统计决策的方法考虑可信区间或可信集的问题

此时的行动空间是所有的可能的区间构成的集合$C(x)=[d_{1}(x),d_{2}(x)]$
损失函数在习惯上取为
$$L(\theta,C(x))=m_1[d_2(x)-d_1(x)]+m_2[1-I_{C(x)}(\theta)]$$
其中的$m_1~m_2$是提前给定的常数
前半部分衡量了区间长度引起的损失 长度越大损失越大
后半部分表示了当$\theta$偏离出区间带来的损失 

比较多个区间的后验风险 找后验风险最小的那一个

### Minimax准则
一致最优的决策函数可能不存在，或者说往往不存在；那么我们就需要一个新的准则，直接从风险函数的角度来考虑哪个决策函数最优
**当我们对先验分布没有把握的时候，才需要这样干，其他情况下研究后验风险（贝叶斯准则）是更好的选择**

考虑风险函数$R(\theta,\delta)$  [贝叶斯统计1(贝叶斯统计与后验分布)](/blog/2023/10/28/bayesian-statistics-posterior-distributions-notes/) 的“风险函数和一致最优决策函数”一节 令
$$M(\delta)=\sup_{\theta\in\Theta}R(\theta,\delta).$$
能看出 我们研究在某种决策的情况下的最多风险 在决策时 我们选择最大风险最小的决策
这个决策准则被我们称为**Minimax准则** 

**Minimax准则是一种不要求得到许多，但是希望失去的不要很多的思想** 

具体计算Minimax解是比较苦难的 我们给出一个用来验证Minimax解的定理
设 $\widehat{g}_k=\widehat{g}_k(\boldsymbol{x})$ 为在先验分布 $\pi_k(\theta)$ 下 $g(\theta)$ 的一列贝叶斯估 计，$k=1,2,\cdots;$ 假定 $\widehat{g}_k$ 的贝叶斯风险为 $r_k,k=1,2,\cdots$, 且有$\lim_{k\to\infty}r_{k}=r<\infty$,  
设$\widehat{g}^{*}=\widehat{g}^{*}(x)$ 为 $g(\theta)$ 的一个估计量，满足条件
$$M\left(\widehat{g}^{*}\right)\leq r$$
则$\widehat{g}^*$为此决策问题的 Minimax 估计

## 贝叶斯统计计算
### 引言
在贝叶斯统计方法中经常要计算后验分布的期望、方差、分位数或众数等数字特征；
比如常用的后验均值，它是在平方损失下的贝叶斯估计，此估计跟的精度是通过后验方差来度量的.后验众数、后验中位数以及后验分位数也常常被用来作为员叶斯估计或建立贝叶斯可信区间等；
如果先验分布不是其轭先验分布（这在许多问题里经常遇到），那么后验分布往往不再是标准的分布.因此，需要计算的后验分布数字特征往往没有显式表达，这就需要一些特殊的计算方法.

比如
$$\pi(\theta|x)\propto\exp\{-(\theta-x)^{2}/(2\sigma^{2})\}[\tau^{2}+(\theta-\mu)^{2}]^{-1}.$$
他的后验期望与方差都是复杂的没有显式解的积分
当然我们还可以用一些数值积分方法来求解

考虑另一个问题
先验分布用对数联合分布给出
$$\nu=\left(\ln\theta_{1},\ln\theta_{2},\cdots,\ln\theta_{k}\right)^{T}\sim N\left(\mu1_{k},\tau^{2}\left\{\left(1-\rho\right)I_{k}+\rho J_{k}\right\}\right)$$
因此可以给出后验分布为
$$\begin{aligned}&\pi\left(\nu|x\right)\propto f\left(x|\nu\right)\pi\left(\nu\right)\propto g\left(\nu|x\right)\\&=\exp\left\{-\sum_{i=1}^{k}\left(e^{\nu_{i}}-\nu_{i}x_{i}\right)-\frac{1}{2\tau^{2}}\left(\nu-\mu1_{k}\right)^{\mathrm{T}}\left[\left(1-\rho\right)I_{k}+\rho J_{k}\right]^{-1}\left(\nu-\mu1_{k}\right)\right\}.\end{aligned}$$
他的期望是两个$k$重积分的比值；数值积分方法对高维积分束手无策（与高维灾难有关） 现在我们需要一些新的处理手法来解决这个问题了；其中MCMC算法是最核心的手段

### EM算法
EM算法是一类非常重要的统计方面的算法 他用于解决两类的非常重要的统计计算问题 一类是极大似然估计问题 另一类是Bayes统计中后验众数的估计问题 事实上前者是后者的一种特殊情况 我们着重从Bayes统计的角度研究EM算法
EM算法属于一种扩充算法（数据添加算法） 由于直接对后验众数计算（极大化）非常的困难 而是对原有的数据进行一些扩充，添加一些潜在数据（latent data），从而简单的实现一系列极大化或者模拟
这些潜在数据可以是缺损的数据或者未知的参数 

**后验众数的估计是我们EM算法最主要用途**
#### 算法过程
我们还是以后验分布为例 对于难以计算的$p(\theta|Y)$ 我们扩充进去一个变量$Z$ 得到简单的${p(\theta|Z,Y)}$ 从而简化计算流程 最后再对添加的$Z$进行改进 最后实现我们的后验众数的计算 

EM算法是一种迭代算法 分为E步（期望步）和M步（极大步） 其中$p(\theta|Y)$ 表示需要研究的后验分布 称 $p(\theta|Z,Y)$为添加后验分布 称$p(Z|\theta,Y)$ 为扩充变量的条件密度函数 我们的目标是研究后验分布的众数
记$\theta^{i}$是第$i+1$次迭代开始的时候后验众数的估计值 则我们的下一步迭代过程是
E步 期望化
对${p(\theta|Z,Y)}$ 或者$log({p(\theta|Z,Y)})$  关于$Z$的条件分布求期望 把$Z$ 积分掉
$$\begin{aligned}
Q(\theta|\theta^{(i)},Y)& \hat{=}E_{Z}\big[\log p(\theta|Z,Y)|\theta^{(i)},Y\big]  \\
&=\int\big[\log p(\theta|Y,Z)\big]p(Z|\theta^{(i)},Y)dZ.
\end{aligned}$$
**这里是对扩充进来的变量求期望，此时我们使用了一个参数正在迭代过程中的估计数目**
M步 极大化
把$Q(\theta|\theta^{(i)},Y)$ 作极大化 得到点$\theta^{i+1}$ 使得
$$Q(\theta^{(i+1)}|\theta^{(i)},Y)=\max Q(\theta|\theta^{(i)},Y).$$
反复重复这个EM过程直到我们一直在迭代的$\theta$序列收敛
**极大化，是通过寻找合适的$\theta$ 实现$Q$的极大**
#### 理论证明
定理：
如果$f(x)$是凸函数 那么对于随机变量X 一定有$$E[f(X)]\leq f[E(X)]$$
这称为Jensen不等式

定理：
EM算法每次迭代都能提升后验密度函数的值
$${p(\theta^{(i+1)}|Y)\geq p(\theta^{(i)}|Y)}$$
定理：
如果EM算法迭代中$\theta$序列满足
* $\left.\frac{\partial Q(\theta|\theta^{(i)},Y)}{\partial\theta}\right|_{\theta=\theta^{(i+1)}}=0;$
* $\text{令}p(Z|\theta^{(i)},Y)\text{充分光滑,}\theta^{(i)}\text{ 收敛到某些值}\theta^{(*)}.$
则
$$\frac{\partial\log p(\theta|Y)}{\partial\theta}|_{\theta=\theta^*}=0.$$
也就是EM算法的迭代一定能收敛到一个稳定点 但不一定是一个最大值点 如果想保证最大 需要选取多个初值进行多次模拟确定稳定性 

#### EM算法的例子
##### 缺失数据的极大似然估计
对于总体$X\sim N(\mu,\sigma^{2})$ $X_{1},X_{2},X_{3}$ 是来自总体的样本 $X_{2}$缺失 使用极大似然估计确定总体分布的参数
对于这种类型的问题 EM算法可以通过补充缺失数据来处理

扩充$X_{2}$ 得到完整的似然函数并取对数
$$\log p(\theta\mid X_1,X_2,X_3)=-3\ln\sigma-\frac{\sum_{i=1}^3(X_i-\mu)^2}{2\sigma^2}.$$
执行E步 也就是针对$X_{2}$ 求期望
非常明显的对$X_{2}$求期望 不含$X_{2}$ 的都可以看作常数 所以实际上我们只需要计算很小的一部分
$$E_{X_{2}}[(X_{2}-\mu)^{2}\mid\theta^{(i)},X_{1},X_{3}]=(\mu_{i}-\mu)^{2}+\sigma_{i}^{2}$$
对于$X_{2}$的条件期望 我们有上一次迭代给出的参数估计值$\theta^i$ 那么容易知道$X_{2}$服从正态分布 最后所求的是正态分布的平方的期望 也就是一个二阶原点矩的问题  很容易计算
那么可以得到E步的最后结果
$$\begin{aligned}
Q(\theta\mid\theta^{(i)},X_{1},X_{3})& \left.\hat{=}\left.E_{X_{2}}\right[\log p(\theta\mid X_{1},X_{2},X_{3})\mid\theta^{(i)},X_{1},X_{3}\right]  \\
&=-3\mathrm{~ln}\sigma-\frac{(X_1-\mu)^2+(X_3-\mu)^2+(\mu_i-\mu)^2+\sigma_i^2}{2\sigma^2}.
\end{aligned}$$
M步
找到合适的$\theta$取值 让Q极大 我们只需要研究对$\theta$的偏导数
$$\left.\left[\begin{aligned}&\frac{\partial Q}{\partial\mu}=\frac{(X_1-\mu)+(X_3-\mu)+(\mu_i-\mu)}{\sigma^2}=0,\\&\frac{\partial Q}{\partial\sigma}=\frac{-3}{\sigma}+\frac{(X_1-\mu)^2+(X_3-\mu)^2+(\mu_i-\mu)^2+\sigma_i^2}{\sigma^3}=0.\end{aligned}\right.\right.$$
求解方程就能得到下一步迭代的结果
要注意 我们是在从$\theta^i$向$\theta^{i+1}$迭代 其中$\theta^i$是在研究E步的时候不得不给出的 
通过反复迭代很快就能得到收敛的序列 这就是EM算法的意义

##### 研究后验分布众数
实际上 极大似然就是在研究似然函数后研究分布众数 缺项的极大似然就涉及了后验分布的问题 缺项的极大似然就是研究后验分布众数的一个特殊情况

假设试验一共有四种可能的结果 发生的概率分别是 $\frac{1}{2}+\frac{\theta}{4},\frac{1}{4}(1-\theta),\frac14(1-\theta),\frac\theta4,$  其中$\theta$在$(0,1)$取值 一共进行了197次试验 四种结果的发生次数观测如下$Y=(y_{1},y_{2},y_{3},y_{4})=(125,18,20,34).$

现在我们来研究一下$\theta$的分布情况 （把原始的量看作随机变量，这是Bayes统计的思想） 假设原本的$\pi(\theta)$ 先验分布是平坦分布 研究后验分布有
$$\begin{aligned}
p\left(\theta\mid Y\right)& \propto\pi(\theta)p(Y\mid\theta)  \\
&=\left(\frac{1}{2}+\frac{1}{4}\right)^{y_{1}}\left[\frac{1}{4}(1-\theta)\right]^{y_{2}}\left[\frac{1}{4}(1-\theta)\right]^{y_{3}}\left(\frac{1}{4}\theta\right)^{y_{4}} \\
&\infty\left(2+\theta\right)y_1(1-\theta)^{y_2+y_3}\theta^{y_4}.
\end{aligned}$$
这个后验分布众数可不好研究 因此我们假定第一种结果可以分成两部分 概率分别为$\frac{1}{2}$和$\frac{\theta}{4}$ 用$Z$和$y_{1}-Z$ 表示试验结果落入其中的次数（Z是我们补充的隐藏数据）那么添加后验分布为
$$\begin{aligned}
p(\theta\mid Y,Z)& \propto\pi(\theta)p(Y,Z\mid\theta)  \\
&=\left(\frac{1}{2}\right)^{z}\left(\frac{\theta}{4}\right)^{y_{1}-z}\left[\frac{1}{4}(1-\theta)\right]^{y_{2}}\left[\frac{1}{4}(1-\theta)\right]^{y_{3}}\left(\frac{1}{4}\theta\right)^{y_{4}} \\
&\infty(\theta)^{y_1-Z+y_4}(1-\theta)^{y_2+y_3}.
\end{aligned}$$
对于这样的添加后验分布 求众数明显就简单了 所以我们使用EM算法继续计算
E步 对添加后验分布的添加量的对数求期望 和添加量Z无关的算常数
$$\begin{aligned}Q(\theta\mid\theta^{(i)},Y)&=E^Z[(y_1-Z+y_4)\mathrm{log}\theta+(y_2+y_3)\mathrm{log}(1-\theta)\mid\theta^{(i)},Y]\\&=[y_1-E^Z(Z\mid\theta^{(i)},Y)+y_4]\mathrm{log}\theta+(y_2+y_3)\mathrm{log}(1-\theta).\end{aligned}$$
而$Z$的条件分布很明显是一个二项分布 $Z\sim b\left(y_1,\frac2{\theta^{(i)}+2}\right)$
因此$$E^{Z}\left(Z\mid\theta^{(i)},Y\right)=\frac{2y_{1}}{\theta^{(i)}+2}.$$
现在我们就能得到极大化函数Q
M步 对极大化函数Q关于$\theta$ 进行极大 还是使用求偏导的手段 然后执行迭代就可以了

##### 混合分布问题
对于分布函数$$f_{X}\left(x\right)=\sum_{j=1}^{K}p_{j}f_{X_{j}}\left(x\right).$$
我们有$\sum\limits p_{j}=1,p_j>0$ $f_{X}(x)$是总体密度函数 $f_{X_{j}}(x)$是子总体密度函数
对于这样的混合分布的问题 求解参数估计也是非常重要的 
但是使用矩估计方法会非常的复杂 Pearson 经过尝试已经证明了一点 
在后续的研究中 人们发现使用EM算法可以非常轻松的估计混合分布的参数 这有力的助推了混合分布模型的研究 

### Monte Carlo积分方法
贝叶斯统计中非常多目标都是一个积分（比如后验期望，方差，众数） 我们先把问题转化为积分 然后再使用MC方法求解
**MC方法本质上就是一种数值积分方法，他对我们刚才提到的高维灾难无效，目前也有很多更优秀的数值积分手段了，这里介绍MC方法只是为了引出后面的MCMC算法**
#### 理论基础
**伯努利大数定律**
$$\underset{n\to\infty}{\operatorname*{lim}}P\left\{\left.\frac{\mu_{n}}{n}-p\right|<\varepsilon\right\}=1.$$
伯努利大数定律告诉我们频率是依概率收敛于概率 这意味着积分问题在转化为面积或者体积这类测度计算问题后 直接转变为了比例的计算问题 可以使用随机投点的方法解决
**辛钦大数定律**
$$\lim_{n\to\infty}P\left\{\left|\frac{1}{n}\sum_{i=1}^{n}X_{i}-\mu\right|<\varepsilon\right\}=1.$$
辛钦大数定律告诉了我们随机变量的数学期望可以使用样本均值来进行近似的统计规律性
选取合适的密度函数把积分问题转化为期望的计算问题 然后依靠抽样解决
#### 随机投点法
计算
$$\theta=\int_{a}^{b}f\left(x\right)\mathrm{d}x.$$
转化为了计算曲线$f(x)$下区域面积的问题
向$D=[a,b]\times [0,M]$中 中随机投点 如果落在曲线$f(x)$的下方 则单独标出
最后 统计随机投点落在曲线$f(x)$的下方的概率$P$
根据下面公式得到$\theta$的估计值
$$P=P\langle Z_i\in\Omega\rangle=\frac{S(\Omega)}{S(D)}=\frac{\theta}{M(b-a)}$$
伯努利大数定律保证了我们的估计精度随着投点的增多可以无限增高
#### 平均值法
平均值法提高了积分估算的效率
平均值方法一方面基于辛钦大数定律 另一方面使用随机变量函数的数学期望进行了重要的变形
当$X$服从概率密度函数为$g(x)$的分布时
$$\theta=\int_{a}^{b}f\left(x\right)\mathrm{d}x=\int_{a}^{b}\frac{f\left(x\right)}{g\left(x\right)}g\left(x\right)\mathrm{d}x=E\Bigl[\frac{f\left(X\right)}{g\left(X\right)}\Bigr].$$
为了简化问题 我们将引入的辅助分布$X$取为$X\sim U(a,b)$  
所以
$$\theta=(b-a)E[f(X)].$$
现在我们把问题转化为了数学期望的计算问题，可以非常容易的使用统计模拟的方法生成对应的随机数然后进行期望的计算

对于无穷曲线上的问题 可以使用积分变换来实现转化为有限区间上的问题 
和数学分析中研究广义积分一章中思想一样
#### 高维形式
高维投点法的核心思路并没有发生任何改变 其核心公式变化为
$$P=P\langle Z_{i}\in\Omega\rangle=\frac{V(\Omega)}{V(D)}=\frac{\theta}{MV(C)}=\frac{\theta}{M\prod_{j=1}^{d}\left(b_{j}-a_{j}\right)}$$
高维的平均值法的核心公式变化为
$$\widetilde{\theta}=\prod_{j=1}^{d}\left(b_{j}-a_{j}\right)\frac{1}{n}\sum_{i=1}^{n}f(x_{i}).$$

### 马尔可夫链蒙特卡洛（MCMC）方法
前面的MC方法一方面无法处理高维灾难问题，另一方面对后验分布形式不确定的贝叶斯统计问题束手无策；但是前者在很多问题中都很常见，后者在分层贝叶斯中更是普遍现象 这就是我们引入**马尔可夫链蒙特卡洛（MCMC）方法**的价值

#### 马尔可夫链大数定律
定理：假设 $\{X_n,n\geqslant0\}$ 为一具有可数状态空间 $S$ 的马氏链，其转移概率矩阵为 $P$. 进一步假设它是不可约的且有平稳分布$\pi=\{\pi_i:i\in S\}$,则对任何有界函数 $h:S\to\mathbf{R}$ 以及初值 $X_{0}$ 的任意初始分布有

$$
\frac{1}{n}\sum_{i=0}^{n-1}h(X_{i})\to\sum_{j}h(j)\pi_{j},\quad n\to\infty 
$$
依概率成立.当状态空间为不可数，马氏链 $\{X_n,n\geqslant0\}$ 为不可约且有平稳分布$\pi$时，也有
$$
\frac{1}{n}\sum_{i=0}^{n-1}h(X_{i})\to\int_{S}h(x)\mathrm{d}\pi(x),\quad n\to\infty.
$$

这个定理结论是非常有用的. 比如给定集合 $S$ 上的概率分布 π,以及 $s$ 上的实函数 $h(\theta)$,假设我们要计算积分 $\mu= \int _Sh( \theta) d\pi( \theta|x)$, 当从后验分布 $\pi(\theta|x)$ 中难以直接抽样时，
则可以构造一个马氏链，使得其状态空间为 $S$ 且其平稳分布 $\pi$ 就是目标后验分布 $\pi(\cdot|x)$,从一初值 $\theta_{0}$ 出发，将此链运行一段时间，比如 $0,1,2,\cdots,n-1$,生成随机数 (样本) $\theta_0,\theta_1,\cdots,\theta_{n-1}$, 则由前面的定理知
$$\overline{\mu}_{n}=\frac{1}{n}\sum_{j=0}^{n-1}h\left(\theta_{j}\right)$$
为所要求积分 $\mu$ 的一个相合估计，这种求积分的计算方法称为 MCMC 方法

**MCMC会给我们提供一系列样本，他就是从目标后验中的抽样，是否需要构造函数计算其他积分取决于我们的需求**

#### 一些会用到的术语
##### 初值 initial value
用来初始化一个马尔可夫链；如果初值原理后验密度较高的区域 且算法迭代次数不够多；那么我们最后的积分结果可能出错

为了避免初值的影响 我们建议
* 放弃一些刚开始迭代的样本
* 从多个初值出发进行比较

可以考虑根据先验的期望或者众数作为初值 如果先验信息足够

##### 预烧期 burn in period
我们前面提到了要放弃一些刚开始迭代的样本 在进入平稳状态后再记录 这个被去除的迭代部分就叫做预烧期 链运行的时间足够长的情况下 去除预烧期不会再理论上影响我们的结果

##### 抽样步长 sampling lag
马氏链产生的样本不可能完全独立 但是我们需要独立的样本；可以通过观看ACF图来找到合适的间隔 保证样本近似独立

##### 迭代保持数 number of iterations retained
总迭代次数和预烧期迭代次数的差

##### 算法收敛性
马尔可夫链是否达到了平稳状态 平稳状态后的抽样可以近似为后验中的抽样
##### 蒙特卡洛误差 error
报告我们的随机模拟是否逼近了平稳分布

#### 收敛性诊断
没有一个确定的指标可以帮助我们研究MCMC方法的收敛性 只能综合下面的方法
* MC误差较小意味着收敛
* 样本路径图在一个区域内 没有确定的趋势
* 累积均值稳定
* ACF图
* 一些诊断方法 如 Gelman-Rubin诊断

### Metropolis - Hasting 算法
从一般性后验分布中进行抽样需要使用MCMC算法 他的核心在于建立一个马氏链 满足一系列预设的条件；因此其中最为核心的是如何在各个状态中进行转移的规则

Metropolis - Hasting 算法（MH算法）是其中最为经典的一种算法

### Gibbs算法
推广Metropolis - Hasting 算法到高维抽样的情况
