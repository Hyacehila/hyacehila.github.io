---
title: "Logistic 回归学习笔记"
title_en: "Logistic Regression Notes"
date: 2024-03-13 13:23:55 +0800
categories: ["Data Science & Statistics", "Statistical Modeling & Inference"]
tags: ["Learning Notes", "Statistics", "Regression", "Logistic Regression"]
author: Hyacehila
excerpt: "一篇 Logistic 回归学习笔记，整理二分类因变量、线性概率模型、Logit 模型、参数估计、模型解释和相关扩展。"
excerpt_en: "A study note on logistic regression, covering binary response variables, linear probability models, logit models, parameter estimation, interpretation, and related extensions."
mathjax: true
hidden: true
permalink: '/blog/2024/03/13/logistic-regression-notes/'
---
## 二分类因变量和Logistic回归
### 引言
线性回归模型在现代统计分析中可以算是最流行的方法之一了，他的广泛应用我们在[线性回归基础](/blog/2023/09/04/linear-regression-basics-notes/)中进行了不少的介绍；但是在非常多的情况下 线性回归会受到限制 一种非常常见的情况就是 **因变量是分类变量** 

在处理这种情况的时候 最为常用的方法是 对数线性模型（log linear model） 我们在这里介绍一种特殊的对数线性模型 Logistic模型 为了方便理解 后面将不会从对数线性模型的理论出发 而是类比我们学习过的线性回归模型来描述Logistic回归模型

### 线性概率模型
在传统的回归分析中 我们没有限制自变量的类型 也就是无论是连续型变量 还是 二分类的变量（取 0 1） 都是可以被我们接受的 但是我们要求因变量一定是连续的（我们也允许近似连续的情形，诸如年龄） 
如果因变量只能取分类值 现在我们用最为普遍的 0 1 分类 看看用最小二乘法（OLS）处理会发生什么 不妨设最为简单的模型形式为
$$y_{i}=\alpha+\beta x_{i}+e_{i},$$
其中$y_i$是二分类变量 只能取 0 和 1
在给定$x$的情况下 计算$y$的期望有
$$\begin{gathered}
E\left(y_{i}|x_{i}\right) =E\left(\alpha+\beta x_{i}+e_{i}\right) \\
=\alpha+\beta x_{i}. 
\end{gathered}$$
此时有
$$E\left(\left.y_{i}\right|x_{i}\right)=P\left(\left.y_{i}=1\right|x_{i}\right)$$
这意味着  $y_i$的期望值就是第$i$个家庭 回归计算出的事情发生的概率 他还是一个线性的概率模型 因此：二分类因变量的线性回归模型也叫作线性概率模型（linear probability model） 
对应的可以计算事情不发生的概率为
$$\begin{aligned}&P\left(\left.y_i=0\right|x_i\right)=1-\left(\alpha+\beta x_i\right)=1-\alpha-\beta x_i.\end{aligned}$$
据此 我们可以计算残差有
$$\begin{aligned}&e_{i}=y_{i}-\alpha-\beta x_{i}\end{aligned}$$
计算残差的方差有
$$\begin{aligned}
&=\left(\alpha+\beta x_{i}\right)\left(1-\alpha-\beta x_{i}\right) \\
&=P\left(y_{i}=1|x_{i}\right)\times P\left(y_{i}=0|x_{i}\right)
\end{aligned}$$
这意味着 **残差的方差依赖于因变量的值的变动而变动 不同观测值会带来不同的方差 这在统计学中被称为不满足方差齐性 （方差非齐）**

正式因为因变量的值的特殊性质  LPM的预测还是有很多的问题
* 方差非齐意味着参数估计的方差是有偏的，并且任何的假设检验都是无效的
* 预测的概率很可能超过$[0,1]$区间 违背常理
* 线性的函数关系不能拟合这种模型的真实形式

综上所述，我们需要取寻找一种新的模型来研究这种二分类变量 或者推广一步 分类变量的回归问题

### Logistic回归模型
非常现实的 我们需要寻找一种合适的函数作为我们的预测模型 最常用的是 Logistics分布 （还有一种可以替换的是 probit 模型，我们会在比较靠后的地方进行介绍）下面来介绍一下我们选择Logistics分布的依据

假设存在一个连续型的变量描述了事件发生的可能性 他可以自由变动 当它的大小达到0的时候 事件发生了 于是$y_i^*>0$ $y_i=1$ 其他情况下 $y_i=0$ 我们只能观测到$y_i$  假设一个线性模型
$$y_{i}^{*}=\alpha+\beta x_{i}+\varepsilon_{i}.$$
研究条件概率有
$$\begin{aligned}
P\left(y_{i}=1|x_{i}\right)& =P\left[\left(\alpha+\beta x_{i}+\varepsilon_{i}\right)>0\right]  \\
&=P\left[\varepsilon_{i}>\left(-\alpha-\beta x_{i}\right)\right].
\end{aligned}$$
通常我们假设误差项$\varepsilon_{i}$ 服从logistic分布或者标准正态分布（分别对应logistics模型和probit模型） 因此我们可以改写出累积分布函数的形式
$$\begin{aligned}
P\left(y_{i}=1|x_{i}\right)& =P\left[\varepsilon_{i}\leq\left(\alpha+\beta x_{i}\right)\right]  \\
&=F\left(\alpha+\beta x_{i}\right),
\end{aligned}$$
我们后面主要研究logistics模型 对于probit模型只在最后进行简单的介绍
那么现在假设误差项$\varepsilon_{i}$ 服从logistic分布有
$$\begin{aligned}
P\left(y_{i}=1\left|x_{i}\right)\right.& =P\left[\varepsilon_{i}\leqslant\left(\alpha+\beta x_{i}\right)\right]  \\
&=\frac{1}{1+e^{-\varepsilon_{i}}}.
\end{aligned}$$
他的取值范围在0-1之间 这个性质保证了我们后面的模型不会生成大于0或者小于0的概率 

现在我们开始研究具体的模型该如何设计 改写前面的式子有
$$P(y_{i}=1\left|x_{i}\right)=\frac{1}{1+e^{-\left(\alpha+\beta x_{i}\right)}} $$
这里 我们定义 $\alpha+\beta x_{i}$ 是一系列影响事件发生概率的因素的线性函数（想要多元化只需要简单的替换）
再将事件发生的概率进行替换就有logistics回归模型
$$\begin{gathered}
p_{i} =\frac{1}{1+e^{-\left(\alpha+\beta x_{i}\right)}} \\
=\frac{e^{\alpha}+\beta x_{i}}{1+e^{\alpha+\beta x_{i}}}, 
\end{gathered}$$
通过和1作差就能得到事件不发生的概率 两者作比有
$$\frac{p_{i}}{1-p_{i}}=e^{(\alpha+\beta x_{i})}.$$
这个值被我们称为发生比（odds）
对发生比取对数就有
$$\ln\left(\frac{p_{i}}{1-p_{i}}\right)=\alpha+\beta x_{i}.$$
这意味着 odds 的对数和原本的模型是线性关系了；我们可以用原本的那一套解释系数的含义 也不必担心模型拟合的过大导致超过概率的值域

多自变量的模型自然变换为
$$p_{i}=\frac{e^{a+\sum_{i=1}^{k}\beta_{k}x_{ki}}}{1+e^{a}+\sum_{i=1}^{k}\beta_{k}x_{ki}}.$$
$$\ln\left(\frac{p_{i}}{1-p_{i}}\right)=a+\sum_{k=1}^{k}\beta_{k}x_{ki}.$$

**我们介绍的因变量为分类变量的回归模型是经典的非线性回归模型，虽然介绍了有一个线性化的过程，但是那并不是因变量**
**当我们拥有了样本的取值情况，并且知道此时事件是否发生，就可以研究特定情况下发生的概率了**

## Logistic回归模型的参数估计
非常自然的思想 在介绍完模型的形式后 我们需要开始研究如何根据已知的信息取计算模型中参数的取值情况了
### 极大似然估计（MLE）
线性回归模型中我们介绍的最主要的参数估计的方法是最小二乘法（OLS）
但是在Logistic回归模型中它效果并不好 这里我们要使用的最多的估计方法是极大似然估计；在线性回归模型中，极大似然估计和最小二乘法会给出相同的结果 但是极大似然估计还可以适用于非线性模型的估计

极大似然估计的思想是建立一个似然函数 选取合适的参数让似然函数取到最大值 这样得到的参数值就是我们想要的估计值[数理统计](/blog/2023/03/18/mathematical-statistics-notes/) 的“极大似然估计”一节
设$p_i$是给定$x_i$的情况下观测到 1 的概率 他的取值为前面给出的
$$\begin{gathered}
p_{i} =\frac{1}{1+e^{-\left(\alpha+\beta x_{i}\right)}} \\
=\frac{e^{\alpha}+\beta x_{i}}{1+e^{\alpha+\beta x_{i}}}, 
\end{gathered}$$
那么 得到一个观测值的概率就是
$$P\left(y_{i}\right)=p_{i}^{y_{i}}\left(1-p_{i}\right)^{1-y_{i}},$$
多个观测值的情况就是
$$L\left(\theta\right)=\prod_{i=1}^{n}p_{i}^{y_{i}}\left(1-p_{i}\right)^{\left(1-y_{i}\right)}.$$
对我们计算得到的似然函数进行极大化（手算的时候类似于MLE处，软件有自己的迭代计算方式）就可以得到Logistic回归模型的参数估计

### 假设条件
正如OLS要求Gauss-Markov假设 Logistics模型也有自己的假设要求，他们和OLS的假设要求并不相同
* 数据来自随机样本
* 因变量被假设为自变量的非线性函数形式（需要内在的联系）
* 对复共线性敏感
* 因变量只能是二分类变量（其他类型我们后面介绍）
* 不要自变量求正态性 可以是连续的 离散的 或者是虚拟变量
### 样本规模与MLE的性质
在满足上述假设的情况下 Logistic回归模型的极大似然估计有
* 一致性
* 渐进有效性
* 渐进正态性
他们都属于大样本性质；
这意味着在小样本的情况下 Logistic回归模型的统计性质究竟如何我们无法确定 我们应该在条件允许的情况下适当扩大样本；目前学界对样本的数量有一定的研究但是未能形成确定的结果 一般认为
* 样本数在100以上 效果就能维持的较好
* 在样本数量大于500的时候 就属于大样本的范畴了
* 更多的参数依赖更多的观测来估计 每个参数应当搭配10个以上的样本
* 存在复共线性等性质的时候 要适当扩大样本
* 当模型中存在更多分类的时候 应当适当扩大样本
### 分组数据与Logistic回归模型
这是一种特殊的Logistics回归模型 以至于他和线性回归模型有一点接近 叙述如下；

我们考虑研究大学升学情况和学生性别 重点高中就读情况 学生成绩分级水平三个分类变量的关系 其中大学升学情况是作为回归因变量的二分类变量；
从这个角度来看 这就是一个普通的Logistics回归模型；

现在 我们将角度从个体变化为某个整体；统计某性别某成绩某高中的所有学生 计算升学比例得到$f_i$ 计算某个组的回归模型用于预测
此时 自变量不变 因变量变为来 一个$[0,1]$ 之间的量 如果直接用连续型使用最小二乘法 难免会出现问题 所以我们再此变形有
$$\begin{aligned}\ln\left(\frac{f_j}{1-f_j}\right)=a+\beta_1\text{GENDER}_j+\beta_2\text{KEYSCH}_j+\beta_3\text{GRADE2}_j+\varepsilon_j\end{aligned}$$
现在直接使用最小二乘法就可以估计回归系数；这本质上还是一种非线性回归Logistics回归模型的一种

由于直接使用OLS进行拟合会出现异方差性 我们需要使用残差项的方差进行加权最小二乘估计 其中权数是误差项标准误的倒数 也就是
$${S_{j}}^{2}=\mathrm{Var}\left(\frac{\varepsilon_{j}}{p_{j}\left(1-p_{j}\right)}\right)=\frac{1}{n_{j}f_{j}\left(1-f_{j}\right)}$$
$$\begin{aligned}\left(\frac{1}{S_{j}}\right)\ln\left(\frac{f_{j}}{1-f_{j}}\right)&=\left(\frac{1}{S_{j}}\right)\alpha^{*}+\beta_{1}^{*}\mathrm{GENDER}_{j}\left(\frac{1}{S_{j}}\right)+\beta_{2}^{*}\mathrm{KEYSCH}_{j}\left(\frac{1}{S_{j}}\right)\\&+\beta_{3}^{*}\mathrm{GRADE}2_{j}\left(\frac{1}{S_{j}}\right)+u_{j}.\end{aligned}$$
软件可以帮助我们完成这些繁杂的计算
## 回归模型的评价
拟合优度检验是为了检验模型拟合值和真实观测值是否有足够小的差距；我们有不止一个指标可以帮助我们研究拟合的优度如何 将会逐个进行介绍
### Pearson $\chi^2$ 拟合优度
首先我们需要介绍 covariate pattern 的概念 它翻译成汉语是协变量类型；
协变量的定义为：对因变量有影响的变量 并且不受试验者的控制 协变量可能被纳入的回归模型 也可能并没有；
covariate pattern 指的就是的协变量组合数目（只有在全分类因变量的时候才好用）
我们可以分别给出观测值和预测值在不同的covariate pattern 中的取值情况 最终呈现的效果就是一个高维度的分类表；
我们在[数理统计](/blog/2023/03/18/mathematical-statistics-notes/) 的“Peason卡方拟合优度检验的思想”一节 中就介绍过类似的情况
这个检验可以帮助我们比较预测是否很好的拟合了观测 当$\chi^2$较大的时候 就可以认为拟合效果并不好 这就是Pearson $\chi^2$拟合优度
### 偏差 Deviance
我们使用似然来描述观测值和预测值的比较 似然意味着在一定的参数估计条件下 产生观测结果的概率；
我们用$L_s$表示模型的最大似然值，它和样本规模是有关的 因此我们需要给出一个基准 也就是饱和模型（完美预测的模型）的似然 为 $L_f$  通过比较两个似然就可以判断模型的拟合效果了 于是给出$D$统计量
$$D=-2\ln\left(\frac{\hat{L}_{s}}{\hat{L}_{f}}\right)=-2\left(\ln\hat{L}_{s}-\ln\hat{L}_{f}\right).$$
它在样本容量足够大的时候近似服从$\chi^2$分布 它描述了我们的模型和完美模型差距的偏差 也就是Deviance
当它足够小的时候 就意味着模型的拟合效果足够好
**使用MLE拟合模型的时候，偏差和卡方拟合优度一般有着接近的取值**

我们对使用 偏差 Deviance Pearson $\chi^2$ 拟合优度 作为衡量模型拟合情况的指标有下面的要求
* 每一个协变类型有10以上的观测数目
从这里我们容易看出 **偏差 Deviance Pearson $\chi^2$ 拟合优度都不适用于具有连续性自变量的Logistics回归模型的拟合优度检测**
### Hosmer-Lemeshow 拟合优度
这是一种类似于Pearson $\chi^2$ 拟合优度的检测方式 但是它通过人为的分组手段规避了有的协变类型观测数量太小的问题 公式如下
$$HL=\sum_{g=1}^{G}\frac{\left(y_{g}-n_{g}\widehat{p}_{g}\right)}{n_{g}\widehat{p}_{g}\left(1-\widehat{p}_{g}\right)},$$
还是和Pearson $\chi^2$ 拟合优度一样 越小意味着拟合优度越好；
**如果和标准的卡方分布检验出不显著（$p>0.05$）的差异就意味着很好的拟合**
### 信息测量指标
正如我们在[线性回归基础](/blog/2023/09/04/linear-regression-basics-notes/) 的“$AIC$准则”一节提到的 我们可以基于两种信息量给出模型的AIC （Akaike信息量准则）BIC（贝叶斯信息量准则）
两者都是越小越好
**信息量判断模型拟合优度是现代数理统计应用越来越广泛的准则**
### 类$R^2$
在线性回归中 $R^2$ 是使用的最为广泛的用于衡量模型效果的指标；由于Logit模型的因变量是分类 我们原本的$R^2$统计量不能继续被使用；但是我们还是可以建立类似的统计量 表示为
$$LRI=\left(\frac{-2L\hat{L}_{0}-\left(-2L\hat{L}_{s}\right)}{-2L\hat{L}_{0}}\right).$$
其中 $-LL_0$ 可以类比为总平方和 $-LL_s$ 类比为偏差平方和 
和$R^2$类似 LRI的取值范围也是0到1 并且越接近1意味着效果越好

数学家总是想建立可以统一两者的指标 不过由于logit模型的特殊性 还没有表达形式能同时囊括二者 但是我们依旧定义了logit模型的$R^2$ 类似于上面的形式 以后我们就用它来作为评估logit模型拟合效果的量 它是最为常用的拟合优度统计量
## 回归系数的研究
当模型有着很好的拟合效果的时候 研究模型的系数就有了意义 我们要研究模型系数的显著性  研究他的显著性检验和区间估计 等等问题
由于logic模型的特殊性
### 发生比和发生比率（Odds and Odds Ratio）
发生比研究的是事件发生频数的比
$$\mathrm{odds}=(\text{事件发生频数})/(\text{事件不发生频数}).$$
也可以同时除以事件总数
$$odds_{k}=\left[p_{k}/\left(1-p_{k}\right)\right]$$
我们也可以立即为 发生比是一个事件发生的概率和不发生的概率的比

由于他的比值构成 因此值域的上限没有边界 当比值大于1的时候容易发生 反之则不容易发生

比较发生比的方式应该是除法 用两个发生比做比得到发生比率（Odds Ratio） OR就描述了不同群体同一个事件发生比的关系 它在后面的logit回归中会非常的常用  尤其是在解释logit模型中 自变量对因变量的影响方面；
* 如果他的OR大于1 意味着自变量对事件有着正的作用 
* 适用于多分类变量和多元模型
**后面我们主要就使用OR来评估logit模型的系数**
### 按照OR解释logit模型系数
#### logit模型的发生比率
我们在线性回归模型中 模型的系数是比较好解释的 正系数意味着正作用 越大意味着作用越强 系数的差值直接体现了作用的大小（标准化后）

但是在logit模型中  我们涉及的是非线性模型 系数直接的作用体现在对数化的单位上 很难给出估计的结果 所以我们需要使用发生比率来研究

在前面的例子中
$$\begin{aligned}\ln\left(\frac{f_j}{1-f_j}\right)=a+\beta_1\text{GENDER}_j+\beta_2\text{KEYSCH}_j+\beta_3\text{GRADE2}_j+\varepsilon_j\end{aligned}$$
我们的$\alpha$ 体现了基准发生比 也就是参数都取0的情况下的发生比的对数

理解发生比比理解对数发生比容易的多 所以进行对数改写
$$\begin{aligned}\mathrm{odds}=\frac{p}{1-p}&=\exp\left(\alpha+\beta_1\text{GENDER}+\beta_2\text{KEYSCH}+\beta_3\textbf{MEANGR}\right)\\&=e^a\times e^{\beta_1\text{GENDER}} \times e ^ { \beta _ 2\text{KEYSCH}} \times e ^ { \beta _ 3\textbf{MEANGR}} . \end{aligned}$$
此时 我们能看到 当系数是正的时候 他就是发生比有着正作用；发生比的变化用$e^{\beta}$ 刻画 实际上他就是发生比率
#### 连续自变量的发生比率
当$x_k$增加一个单位的时候 发生比变化的倍数为$e^{\beta_k}$  
同理 发生比的变化百分点为$e^{\beta_k}-1$   
我们所叙述的系数 都是控制其他变量不变的时候的影响系数 因此 我们称$e^{\beta_k}$ 
为调整发生比率 （adjusted odds ratio） 记为 AOR

在很多情况下 我们不希望研究连续变量变化一个单位的影响 更加关于变化多个单位 我们不妨认为它从$a$变化到$b$ 此时的AOR为
$$e^{\beta_k(b-a)}$$
此时我们的AOR是一个只和变化差值有关的量 也就是我们的建模其实这里含有线性 在很多的情况中 这里不应该是一个线性的关系 我们在后面再考虑这种非线性的建模
#### 二分类自变量的发生比率
在二分类问题中 我们只存在两种变化 从0到1 或者从1到0 
非常容易的可以计算得到
$$AOR=e^\beta$$
**AOR大于1和小于1都是等价的**
#### 多分类自变量的发生比率
根据我们在回归分析中的基础手法 需要建立虚拟变量[广义线性回归](/blog/2024/05/24/generalized-linear-regression-notes/) 的“分类自变量与虚拟变量问题”一节  代表类型的归属性质 原则上 如果我们的分类变量有$m$个类别 就需要$m$个变量来描述他的所属于 为了选取参照 那就是$m-1$个虚拟变量 此时模型变形为
$$\ln\left(\frac{p}{1-p}\right)=\alpha+\beta_{1}\text{GENDER}+\beta_{2}\text{SCH}1+\beta_{3}\text{SCH2}+\beta_{4}\textbf{MEANGR}.$$
此时我们的原本变量SCH有三个类别 我们随便选择其中的一个作为了参照类 所以建立的方程有两个虚拟变量
此时 我们的 $AOR=e^\beta$ 是类别从参照变成1 or 2 的发生比率
**目前的统计学软件都有从多分类变量自动生成虚拟变量的功能，并且会自动为我们计算相关的系数与AOR值，同时进行显著性检验**

**AOR值只能体现0-1的发生比率问题，别的问题我们以后会再其他地方聊到**

### logit模型的标准化
在回归分析中 我们就介绍了经过标准化后的系数有了相互比较的意义
但是对于分类变量 标准化就毫无意义了 我们在线性回归中直接不需要对分类自变量进行额外的处理；
但是对于logit模型而言 他的因变量分类 但是是非线性的 它实际上也有标准化的说法 此时我们对应的各个自变量的系数需要因此进行额外的处理  我们给出计算公式为
$$\beta^{*}=\frac{\widehat{\beta}s_{x}}{\sqrt{s_{logit}^{2}/R^{2}}}=\frac{\widehat{\beta}s_{x}R}{s_{logit}}$$
也就是先进行logit回归 再对系数进行标准化
**标准化后的系数就可以进行互相的比较，思路和线性回归相同**
### 回归系数的显著性检验
回归系数的显著性水平 也就是我们一般使用的$p$值 和线性回归中一样 它体现的是假设检验中的第一类错误；当$p$值小于$0.05$等值的时候 我们判断足够显著 也就是拒绝原假设 认为系数有价值
#### Wald 检验
对于规模很大的样本 检验总体系数是否为0可以考虑$Z$统计量
$$Z=\hat{\beta}_{k}/SE_{\beta_{k}}.$$
使用双侧的$t$检验
在大多数统计软件中侧重于使用Wald检验 也就是
$$\dot{W}=\left(\hat{\beta}_{k}/\mathrm{SE}_{\beta_{k}}\right)^{2}.$$
服从$\chi^2$分布
**当回归系数的绝对值很大的时候，Wald统计值会变得很小，此时不适用于此检验方式**
#### 似然比检验
统计学中给出过关于似然的证明；两个模型之间的对数似然值的-2倍服从卡方分布 想要进行LR检验 只需要研究似然值就可以了
使用软件计算得到的似然比直接服从卡方分布 检验方式和Wald统计量没有区别
#### 检验系数子集
有时候 我们不仅仅想研究某个系数的显著性水平 而是想知道一些系数整体是否是显著的 这点和多元回归中的$F$检验思想一样；当然 如果涉及虚拟变量的问题 系数子集的显著性水平就更重要了
非常明显的 LR检验只需要比较模型变化的似然值，非常适合用来检验系数子集  除了自由度变化以外 没有什么区别
### 预测概率
logit 模型当然能够给出预测结果 代入我们的变量情况进入计算出来的logit回归公式中  再进行相关的非线性变化就可以计算出 因变量发生的概率
$$p_{i}=\frac{e^{a+\sum_{i=1}^{k}\beta_{k}x_{ki}}}{1+e^{a}+\sum_{i=1}^{k}\beta_{k}x_{ki}}.$$
我们可以对计算出的概率进行各种各样的研究 比如把多个概率作差
后面我们会再介绍到预测结果的置信区间
### 回归参数的置信区间
显著性检验能够告诉我们一个系数是否是显著的
预测概率能给我们一个事件的具体预测概率值
但是参数估计不可能是非常精确的 我们要考虑我们估计的不确定性的度量 也就是置信区间的问题
#### 回归系数的置信区间
对于选定的$\alpha$ 置信区间为 其中SE是对应系数的标准误
$$\hat{\beta}_{k}\pm Z_{\alpha/2}\times SE_{\beta_{k}}$$
#### 发生比率的置信区间
大部分的研究者都不怎么关注回归系数的置信区间
和我们在[线性回归基础](/blog/2023/09/04/linear-regression-basics-notes/) 的“区间预测”一节中介绍的一样 我们更关注前面给出的预测概率本文“预测概率”一节的置信区间 首先我们就要研究发生比率的置信区间

在调整系数（从0-1）之后 我们能给出两个预测的发生比率 对应的置信区间体现为（从系数的置信区间到发生比的置信区间）
$$(e^{0.509},e^{1.223})\longrightarrow(1.664,3.397).$$
#### 事件概率的置信区间
发生比率的置信区间还是不如事件概率的置信区间；后者才是真正的体现了我们的最终预测目标的置信区间
他的形式为
$$(e^{\mathrm{logit}\left(y\right)-1.96\sqrt{\mathrm{Var}\left[\mathrm{logit}\left(y\right)\right]}},e^{\mathrm{logit}\left(y\right)+1.96\sqrt{\mathrm{Var}\left[\mathrm{logit}\left(y\right)\right]}}).$$
其中logit(y)的方差比较难计算 不过不需要手工实现了
## 回归诊断
### 变量选择
我们的任务是识别可以对预测起到很好效果的变量 并把他们纳入我们的模型
#### 筛选自变量
我们首先考虑哪些变量值得纳入
在线性回归中 我们把具有相关性的量都纳入我们的模型；但是在logit回归中没有相关性这个指标了[描述性统计与可视化](/blog/2023/11/05/descriptive-statistics-and-visualization-notes/) 的“相关分析”一节没有介绍衡量定性量和定量的量之间的相关性的方法 
**我们应该做一元的logit回归 观察其是否有着显著的效果**

只有检验显著的量才应该被纳入回归模型
#### 逐步回归
正如我们在[线性回归基础](/blog/2023/09/04/linear-regression-basics-notes/) 的“逐步回归”一节中介绍过逐步回归的思想 这里只需要把我们用来比较模型效果的方法换成本文“回归模型的评价”一节中介绍的
向前法和向后法的效果依旧不如混合进行的逐步回归

**删除有意义的变量和保留没意义的变量都会对模型产生负面的效果，这点和线性回归中是一样的，这也是我们进行变量选择的问题**

### 非线性
在我们目前使用的logit回归模型中
$$\ln\left(\frac{p_{i}}{1-p_{i}}\right)=a+\sum_{k=1}^{k}\beta_{k}x_{ki}.$$
也就是右侧的函数是一个经典的线性回归模型中的右半边
正如非线性回归存在的可能一样 我们有时候也需要把右侧修改成非线性的形式 在这里我们不进行过多的介绍 理解线性部分已经基本足够了
### 交互作用
和我们使用的线性回归模型一样
logit模型也可能存在自变量交互作用的情况 这种情况需要特殊的处理
在这里我们不进行过多的介绍 理解无交互部分已经基本足够了
### 过离散
过离散是一种特殊的导致拟合优度下降的情况；又名过二项变异
它只会在以下模型缺陷的时候产生 导致的后果是模型拟合效果不好 残差的方差远大于残差的均值 
**处理过离散是没有意义的，我们需要在发现问题之后去考虑下面的问题**
* 部分协变类型中观察过少（建议合并）
* 部分重要的因变量或者交叉项没有被纳入模型
* 非线性但是没有考虑
* 存在奇异值
* 数据预处理中的变换不到位
### 空单元
指部分协变类型中观测数为0
一般是因为变量太多导致的 可以是因为虚拟变量的建立
如果出现大量空单元则需要合并 负责会对拟合效果产生影响
**最明显的现象是部分系数过大 并且标准误很大**
### 完全分离
一个变量的变动直接对我们的预测结果起了决定性作用
**最明显的现象是部分系数过大 并且标准误很大**
原因是小样本但是大量参数
### 复共线性
复共线型是线性回归中经典的问题 我们用了很大的篇幅来刻画[线性回归基础](/blog/2023/09/04/linear-regression-basics-notes/) 的“回归参数的估计2（复共线性下的估计）”一节
在logit回归中 这也是一个很重要的问题 
复共线性在logit回归中也体现为较大的标准误
和一系列损失回归方法不同 我们在logit回归中缺少用来处理复共线性的方法 只能尝试着去解决
* 使用类似PC回归等降维回归方法
* 尝试删除导致复共线性的变量 要小心不能删除那些真正对因变量有作用的量
### 奇异值
我们在线性回归中研究了两个重要的回归诊断方法 他们分别是
* 影响分析
* 残差分析
他们都和那些极端特殊的值密切相关 这里我们称其为奇异值
**这里的统计量和线性回归中是有变化的**
#### logit模型中的残差
下面是我们使用的比较频繁的残差
##### 非标准化残差
预测概率和真实情况的差
$$y-\hat{P}\left(y=1\right)$$
##### Pearson残差（标准化残差）
标准化调整后的残差
$$z=\frac{y-\hat{P}\left(y=1\right)}{\sqrt{P\left(y=1\right)\left(1-P\left(y=1\right)\right)}},$$
##### Logit残差
$$L=\frac{y-\widehat{P}\left(y=1\right)}{\widehat{P}\left(y=1\right)\left(1-\widehat{P}\left(y=1\right)\right)}.$$
##### Deviance 残差
$$d=\pm\sqrt{-2\left[y\ln\left(P\right)+\left(1-y\right)\ln\left(1-P\right)\right].}$$
##### 学生化残差
学生化的logit残差其实和残差已经没什么联系了
它是模型变动后 拟合偏差的变化的情况 用来衡量某个参数是否重要
#### 影响分析
##### 杠杆统计量
$$H=X\left(X^{\prime}X\right)^{-1}X^{\prime},$$
$H$矩阵对角线的元素
##### Cook 统计量
是标准化残差和杠杆统计量的结合指标 体现了这个参数对模型的影响
$$Cook'sD_{i}=\left(Z_{i}^{2}\times h_{i}\right)/\left(1-h_{i}\right)^{2},$$
#### 如何检验奇异值
我们有下面的基本思路
* 某个观测有着过大非系统性的残差
* 过大的杠杆度
* 过大的Cook统计量
## Logistic回归模型的扩展
### Probit 模型
#### Probit 模型形式
我们在全文的开头就介绍过 除了Logistics 函数 我们还有别的选择
事件的概率可以表述为
$$\begin{aligned}
\text{P}& =P\left(y=1|x\right)  \\
&=F\left(\alpha+\beta x\right) \\
&=\int_{-\infty}^{\alpha+\beta x}f\left(z\right)dz,
\end{aligned}$$
其中$F,f(z)$ 分别是正态概率密度的CDF和PDF
变形可以得到
$$F^{-1}(p)=\alpha+\beta x,$$
这就是我们的probit模型
里面参数的估计可以使用MLE来实现
#### 模型的解释
由于因变量被变化为了正态CDF的逆的作用，这种解释明显是不直观的，在logit模型中我们可以引入odds来帮助我们计算，但是probit模型就没有这么好的性质了
##### 预测概率
只要直接代入自变量的值 然后对CDF求逆 就可以计算得到具体的概率值
##### 对概率的作用
最直观的处理方法就是：先计算概率值的变化情况，然后计算变化的百分比，就可以确定一个自变量对因变量的影响作用

#### 分组数据的Probit模型
正如我们在本文“分组数据与Logistic回归模型”一节中介绍的一样 我们需要
* 先将频率使用CDF计算为对应的回归因变量
* 使用OLS进行拟合 
* 考虑加权的OLS 权值为残差方差平方根的倒数
#### Logit模型和Probit模型的比较
在二分类的情况下 Logit模型和Probit模型有着极度接近的CDF曲线
这意味着他们的回归结果是基本一样的

**但是我们对Logit模型的拟合优度 显著性检验等研究都无法类推，这意味着Probit模型的可解释性较差，我们使用的实际上是比较少的**
### 次序因变量的Logit回归
回归因变量是序次反应变量的情况是非常常见的
在部分学者提出的观点中：只要次序因变量的数目大于5 就可以视为连续型变量处理，虽然偶尔会遇到一些问题；
为了处理次序数目较小 或者 我们不太想进行转化连续操作的情况 我们引入了次序因变量的Logit模型
#### 累积Logit回归模型（Cumulative LRM）
模型的基本形式如下
$$y^{*}=\alpha+\sum_{k=1}^{K}\beta_{k}x_{k}+\varepsilon.$$
当实际观测变量有$J$个类别的时候 我们给$y^*$ 设定$J-1$个未知的门槛 当他们达到门槛后 就意味着自动进入下一个级别 这些门槛被记为$\mu_j$ 

此时我们可以给出CDF形式如下
$$\begin{aligned}
P\left(y\leqslant j|x\right)& =P\left(y^{*}\leqslant\mu_{j}\right)  \\
&=P\left[\left(\alpha+\sum_{k=1}^{K}\beta_{k}x_{k}+\varepsilon\right)\leqslant\mu_{j}\right] \\
&=P\left[\varepsilon\leqslant\mu_{j}-\left(\alpha+\sum_{k=1}^{K}\beta_{k}x_{k}\right)\right] \\
& =F\left[\left.\mu_{j}-\left(\alpha+\sum_{k=1}^{K}\beta_{k}x_{k}\right)\right].\right.
\end{aligned}$$
根据它就可以计算出累计概率 然后推出各个类别的概率
$$P\left(y\leqslant j\mid x\right)=P\left(y^{*}\leqslant\mu_{j}\mid x\right)=\frac{\mathrm{e}^{\left[\mu_{j}-\left(a+\sum_{k=1}^{K}\beta_{k}x_{k}\right)\right]}}{1+\mathrm{e}^{\left[\mu_{j}-\left(a+\sum_{k=1}^{K}\beta_{k}x_{k}\right)\right]}}.$$
现在的统计学软件已经可以很轻松的帮助我们计算出需要的结果了 需要注意的是
**Pearson拟合优度和Devance都不适用连续自变量情形，这点在CLRM也一样**
### 多分类Logit 模型
在无次序多分类的情况下适用于这个模型
我们基本不使用多元的probit模型 因为多元正态分布不好计算
多项式Logit模型也有一个非常明显的缺点 
**它要求对任意两个类别做选择的时候，要假设该选择和其他类型无关，也就是独立于无关类型**
这意味着 如果可供选择的量和其他量存在替代关系 就要预先进行处理 最常用的处理方式是**合并替代物**
模型的形式为
$$\ln\left[\frac{P\left(y=j|x\right)}{P\left(y=J|x\right)}\right]=\alpha_{j}+\sum_{k=1}^{K}\beta_{jk}x_{k}.$$
这意味着 多项式Logit模型本质上 **是在构建多个不重复类别Logit模型**形成的
多项式模型中本质上存在$J-1$个普通的Logit模型
这意味着 **每一个自变量都有着一套系数和一整套完整的分析**
第$J$个类别被选为了参照
想要计算某个类别的概率可以使用下面的公式
$$P\left(y=j\mid x\right)=\frac{e^{\alpha_{j}+\sum_{k=1}^{K}\beta_{jk}x_{k}}}{1+\sum_{j=1}^{J-1}\mathrm{e}^{a_{j}+\sum_{k=1}^{K}\beta_{jk}x_{k}}}.$$
具体的计算过程还是由软件进行计算的
