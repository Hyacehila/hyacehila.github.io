---
title: 'Deep Learning Basics: Neural Networks, Optimization, and Normalization'
title_zh: 深度学习基础：神经网络、优化方法与归一化
date: 2024-09-02 22:28:33 +0800
categories:
- Machine Learning
- Deep Learning
tags:
- Machine Learning
- Neural Networks
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers neural networks, loss functions, optimization, learning rates, classification, batch normalization, and residual
  networks.
description: Covers neural networks, loss functions, optimization, learning rates, classification, batch normalization, and
  residual networks.
excerpt_zh: 整理神经网络、损失函数、优化方法、学习率、分类、批量归一化、残差网络等基础概念。
permalink: /blog/2024/09/02/deep-learning-basics/
lang: en
translation_key: 2024-09-02-deep-learning-basics
translation_status: machine
translation_source_hash: c8d8cf28093c8eb9bfa3bd1a61c2106674f894f75df82c28d306eb539fc11f73
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Before we start.</h2>
<p>There is a difference between deep learning in the way we think about things and traditional machine learning, and here we need to come to an understanding of how deep learning thinks about our problems.</p>
<h3>A case and our thoughts on him.</h3>
<h4>A basic process</h4>
<p>The number of video hits is projected as an example of the operation of machine learning.</p>
<p>Our goal is to find a function whose input is information from the back and the output is the total number of views that this channel will have the next day.</p>
<p>Step 1, we need to find a model, which is essentially a function that contains unknown parameters, such as the most basic linear function.
&#36;&#36;y=b+wx_1&#36;&#36;</p>
<p>Step 2 is the definition of loss (loss), which is also a function. This function<strong>Input is the parameter in the model</strong>, the output represents whether the model given by a group of parameters is sufficient to fit our reality.</p>
<p>In general, loss should be defined as a difference between the projected and the real value, such as MSE MAE, if, of course,&#36;y,\hat{y}&#36; They're all probabilities. We also choose cross entropy.</p>
<p>Next step into machine learning is step 3: solve a perfect problem. Finds a value for an unknown parameter to see which value to use to minimize the value of lost L. The most rapid method of decreasing gradients is that of optimizing the introduction.</p>
<p><strong>The loss function appears to be a function consisting of predictions and real values, which are in fact influenced by model parameters and training samples, so he is a function of model parameters. So there's room for the gradient.</strong></p>
<h4>Think about Sigmod approaching.</h4>
<p>Of course, purely linear models are approaching, which may not be good in a complex real world, and the usual method is to use them.<strong>A set of Sigmod functions and any linear segment functions to approach</strong>, and segment linear functions can almost approach any function at this time<strong>Sigmoid's number is also a hyperparameter</strong></p>
<p>The Sigmod function is
&#36;&#36;y=c\frac1{1+e^{-(b+wx_1)&#125;&#125;&#36;&#36;
of which&#36;c&#36; &#36;b~w&#36; Unknown parameter</p>
<p>And we can put it in brief.
&#36;&#36;y=c\sigma(b+wx_1)&#36;&#36;
of which&#36;\sigma&#36; For the Sigmoid function</p>
<p>Specially, we can get the Sigmoid function to have more variables, or characteristics of the field of deep learning, to get the Sigmoid function below
&#36;&#36;y=c\sigma(b+w_1x_1+w_2x_2+w_3x_3+...)&#36;&#36;</p>
<p>Usually, we use it. &#36;\theta&#36; It refers to all the parameters that need to be optimized, in which the number of Sigmoids is used as an excess rather than an optimisation, at which point our loss function is recorded as &#36;L(\theta)&#36; It's a multi-purpose function, which can still be used to solve the optimal problem by decreasing the gradient.</p>
<h4>A little detail.</h4>
<p>There's a problem with the details, when the gradient actually goes down. The N data is randomly divided into one batch (batch). Each batch contains B-data (known as "batch size".</p>
<p>All of the data would have been included as a loss, and only the data contained in one batch would have been included as a loss. We'll update the parameters once every Batch.</p>
<p>All batches were examined once, called a round (epoch), and each update parameter was called an update.</p>
<h4>ReLU</h4>
<p>In fact, we could have done more deformation of the model, and we were thinking of using the Sigmod function to get closer to the real situation, and in fact he could have been seen as the sum of two modified linear units.
&#36;&#36;c*\max(0,b+wx_1)&#36;&#36;</p>
<p>In machine learning, Sigmoid or ReLU is called the Activation function. There are, of course, other commonly activated functions, but Sigmoid and ReLU are the most commonly activated functions.</p>
<h4>In-depth learning</h4>
<p>Just like we already are. <a href="/en/blog/2024/03/28/machine-learning-introduction-supervised-learning/">Machine Learning Introduction and Monitoring of the Neuronet in Learning</a>It's the same structure that we usually call Sigmoid or ReLU.<strong>Neuron</strong>The network structure of many neurons is called<strong>Neural network</strong> Each row is called one.<strong>Hidden player</strong> A lot of hidden layers are "deep."<strong>In-depth learning</strong></p>
<h3>Practice methodology</h3>
<p>This is the problem we encounter when we really deal with a problem and some ways to deal with it.</p>
<h4>Model deviation</h4>
<p><strong>The problems represented by the real world are extremely complex, the model ' s structure is too simple and too flexible, leading to a proposed comparison of effects Bad</strong></p>
<p>Solutions: Increased complexity and flexibility of models, selection of more complex traditional models or deeper learning models, introduction of more features</p>
<h4>Optimization issues</h4>
<p><strong>Gradient drops are common, but there are many problems that may be stuck to the lowest local value, and the resulting under-optimization will lead to poor alignment.</strong></p>
<p>Model deviations and under-optimization can lead to poor alignment, and it often takes a lot of experience to distinguish the difference between the two, and if small models are to achieve better results than complex models, it is likely that the optimization has not been achieved. Need to improve optimization methods</p>
<h4>Compromise</h4>
<p><strong>Good alignment and poor generalization often mean overcompatibility, often because models are too flexible and training data are not sufficiently covered</strong></p>
<p>I've dealt with the following lines.</p>
<ul>
<li>Increase training data, or rely on<strong>Data enhancement</strong>To get more training data.</li>
<li>Limit the flexibility of models, with the focus on reducing parameters that need to be optimized</li>
<li>Reduced features</li>
<li>Early Stopping, Regularization Dropout</li>
</ul>
<h4>Do not match</h4>
<p><strong>The mismatch is due to a serious error in prediction due to different patterns of distribution of training and testing data in nature</strong></p>
<p>Normally, this can be overcome by gathering more data, but mismatches mean that the training set is not distributed in the same way as the test set.</p>
<p>Whether or not a mismatch is encountered depends on the researcher ' s understanding of the data itself and the way the training set is produced and the test set is produced to determine whether there is a mismatch.</p>
<h2>In-depth learning base</h2>
<p>This chapter introduces the common concept of in-depth learning, which will be the foundation for the various in-depth studies that follow.</p>
<h3>Generation of neuronet models and in-depth learning</h3>
<p><a href="/en/blog/2024/03/28/machine-learning-introduction-supervised-learning/">Machine Learning Introduction and Monitoring of the Neuronet in Learning</a></p>
<h3>The world's smallest and very small.</h3>
<p>The question of global and local minimum values is a very common part of the issue of optimization;<strong>All algorithms based on gradients can't achieve the lowest level of direct search.</strong></p>
<p>And searching on the basis of a drop in gradient is normal for most neural networks, so we've studied some remedial strategies.</p>
<ul>
<li>Here.<strong>Multiple sets of different parameter values to initialize multiple neural networks</strong>, with the least error resolution (i.e. the most common method to optimize) is used as the final parameter when trained in standard methods.</li>
<li>Use '<strong>Simulate a retreat.</strong>&quot; The simulation of a retreat at every step accepts the result that is worse than the current one, which is not the optimal solution, and helps us to get out of the region.</li>
<li>Use<strong>Random gradient down</strong> It increases the randomity factor for the gradient drop, even if it's in the smallest part.</li>
<li><strong>Genetic algorithm</strong></li>
</ul>
<h3>Small local value and saddle point</h3>
<p>The question of local miniscule values and saddlepoints responds to a common question that we have raised in the previous “practical approach”:<strong>As the parameters continue to be updated, the loss of training will not decline, but we are still not satisfied with this loss.</strong></p>
<h4>Critical points and their types</h4>
<p>In the case of previous optimizations, our basic guess is to optimize the moment when the parameters are subdivided into zero losses, at a time when the algorithm based on the decline in gradients cannot continue to optimize the parameters to reduce losses, and training is over.</p>
<p>For a gradient of zero, the most common position is a local minimum and a local maximum, but due to our downward orientation, general in-depth learning is reduced to a local minimum.</p>
<p>In fact, the loss is not just zero in a local very small gradient, but other points that might make the gradient zero, for example.<strong>Saddle point</strong>I don't know. A classic example is the centre of the saddle face.</p>
<p>When the saddle point is constricted, the gradient drop algorithm does not help us to continue to optimize the loss, but there is clearly a lower point of loss around it, which can make the loss even lower, and we have no good way of reducing it when we get to a local small point.</p>
<h4>Methodology for determining the type of threshold</h4>
<p>It is necessary to know the shape of the loss function to determine whether a critical point is a local very small value or a saddle point. But how do you know the shape of the loss function? The network itself is complex, and the loss function calculated with a complex network is clearly complex.</p>
<p>But we can think of doing something about the loss function with a local, approximate loss function for Taylor, in the parameter group.&#36;\theta^{\prime}&#36; There's a spread around.
&#36;&#36;L(\boldsymbol{\theta}) \approx L\left(\boldsymbol{\theta}^{\prime}\right)+\left(\boldsymbol{\theta}-\boldsymbol{\theta}^{\prime}\right)^{\mathrm{T&#125;&#125; \boldsymbol{g}+\frac{1}{2}\left(\boldsymbol{\theta}-\boldsymbol{\theta}^{\prime}\right)^{\mathrm{T&#125;&#125; \boldsymbol{H}\left(\boldsymbol{\theta}-\boldsymbol{\theta}^{\prime}\right) .&#36;&#36;
of which&#36;g&#36;and&#36;H&#36;Gradients and Hessian matrix store first and second steps differentials, respectively</p>
<p>At all points with a gradient of 0, the approximate result becomes
&#36;&#36;L(\boldsymbol{\theta}) \approx L\left(\boldsymbol{\theta}^{\prime}\right)+\frac{1}{2}\left(\boldsymbol{\theta}-\boldsymbol{\theta}^{\prime}\right)^{\mathrm{T&#125;&#125; \boldsymbol{H}\left(\boldsymbol{\theta}-\boldsymbol{\theta}^{\prime}\right) .&#36;&#36;
We can build on the rest.&#36;\left(\boldsymbol{\theta}-\boldsymbol{\theta}^{\prime}\right)^{\mathrm{T&#125;&#125; \boldsymbol{H}\left(\boldsymbol{\theta}-\boldsymbol{\theta}^{\prime}\right)&#36; To judge.&#36;\theta^{\prime}&#36;The shape of the nearby error surface (error surface) determines whether this is a local small value, a local great value or a saddle point for the loss function.</p>
<p>We'll use it.&#36;v&#36;Replace&#36;\theta - \theta^{\prime}&#36; Here's the conclusion.</p>
<ul>
<li>If for all &#36;v&#36; Both &#36;v^Hv &gt; It's a small local value.</li>
<li>If for all &#36;v&#36; Both &#36;v^Hv &lt; It's local.</li>
<li>If it's now and then, &gt; 0&#36;   时而 &#36;v^{T}Hv &lt; It's a saddle.</li>
</ul>
<p>In fact, this conclusion can be streamlined directly.&#36;H&#36;. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .</p>
<ul>
<li>&#36;H&#36;Qualitative values are positive</li>
<li>&#36;H&#36;The feature value is all negative</li>
<li>&#36;H&#36;Characteristic values are positive and negative</li>
</ul>
<h4>The way out of the saddle.</h4>
<p>Let's start with a simple situation. We need an optimised function that's very simple and can calculate a clear matrix directly.&#36;H&#36; So all we need is the characterization of the plan matrix to determine whether it's at the saddle point. At this time &#36;H&#36; It also indicates the direction of the parameters that can be updated, the matrix.&#36;H&#36;direction of negative characteristic vectors</p>
<p>From this point of view, the saddle dots don't seem so scary. But actually, we're hardly going to really figure out the Heisen Matrix, because the Heisen Matrix needs to count a double calibration, it's very large, and it's going to have to find its characterization values and its characterization vectors, so there's hardly any way to escape the saddle spots. There are other ways to get out of the saddle spots that are much smaller than the Heisen Matrix.</p>
<p>Now we have a rule based on experience:<strong>Small local values are less common when training a large neural network of parameters. Most of the time, we train to a very small gradient, and the parameters are no longer updated, and often just meet the saddles.</strong></p>
<p>In other words, the minimum values often don't exist. We don't have to run away from the saddles.</p>
<h3>Batch and momentum</h3>
<p>As we said earlier, the calculation of losses in practice does not always use all the data.
To achieve this, the bulk idea referred to above is still followed: the data are randomly divided into one watch, each of which calculates the loss and updates the parameters once; all of which are read once, called an epoch.</p>
<p>In fact, before each round begins, we are redrawn, that is to say, the volume of each round is different. To enhance the effectiveness of the training to avoid possible oversizing and accelerated training using parallel calculations</p>
<h4>Impact of batch size on gradient reduction</h4>
<p>Let's look at the two most extreme situations.</p>
<ul>
<li>No batch size (batch size) is the size of the training data, and this method of updating parameters using full load data is the Batch Gradient Decline (BGD). At this point, the model had to read all 20 training data in order to calculate the loss and gradient and update the parameters once.</li>
<li>Batch size equals 1 and the method used at this time is the random gradient reduction method (Stochastic Gra-dient Decent, SSD), also known as the incremental gradient reduction method. The loss calculated with a single data is relatively more noiseful, so the direction of its renewal is generally convoluted.</li>
</ul>
<p>The drop in the volume gradient is more stable and accurate than the decrease in the random gradient. However, random noise has been introduced on the gradient of the drop in the random gradient, which makes it easier to escape the local minimum than the drop in the volume gradient in the non-comb optimization problem.</p>
<p>While we have just said that a volume gradient decline would require a larger iterative calculation, it takes into account the accelerated calculations currently provided by GPU. Large quantities do not necessarily take longer than small quantities.</p>
<p>Actually... <strong>When the bat size is smaller, it takes more time to run through a round.</strong>I don't know. Assuming that the training data are only 60000, mass size 1 and 60000 updates are required to "run" a round; if the mass size is equal to 1,000, 60 updates to "run" a round, the time to calculate the gradient is almost equal. However, the time gap between 60,000 updates and 60 updates is very large.</p>
<p>In fact, small batches also help with testing. It is assumed that some methods (e.g. large bulk learning rates) can train large volumes as well as small volumes. The results of the experiment show that<strong>Small quantities will be better at testing.</strong>  The current explanation is that the randomity of small quantities has helped us to jump out of some of the lost canyons that have been dramatically transformed into a lost basin.</p>
<h4>Dynamic Method</h4>
<p>Momentum method is another way to counter a saddle point or a local minimum.</p>
<p>Our thinking is that, in the traditional gradient reduction method, we decide on the direction of the next optimisation according to the gradient of the current point, whereas the kinetic method does not move the parameters only in the reverse direction of the gradient, but in the direction of the reverse direction of the gradient plus the direction of the previous step. We believe that the inertia of this movement allows us to take off the saddlepoints and the minimal local effects and enter the lost basin.</p>
<p>We'll use it.&#36;m&#36;Other Organiser &#36;g&#36;is the gradient that gives you the drop structure of the kinetic method, where &#36;\mu&#36; It's learning.&#36;\lambda&#36; is the weight parameter for the previous direction
&#36; \begin{array}
\bardsymbol{<em>{0}=0 \
\boldsymbol{m}</em>{1}=-\eta \boldsymbol{g}<em>{0} \
\boldsymbol{m}</em>{2}=-\lambda \eta \boldsymbol{g}<em>{0}-\eta \boldsymbol{g}</em>{1} \
...
\end{array}&#36;&#36;</p>
<h3>Learning rate for adaptation</h3>
<h4>On adaptive learning rates</h4>
<p>The threshold is not necessarily the biggest obstacle to training a network.</p>
<p>Sometimes we find that the gradient of the loss function (measured in its model) is still large, but the loss does not continue to decline, and we are not stuck at some point with a gradient of zero, but we cannot continue to update the parameters to reduce the loss. It's often because we're stuck at some point because of the high rate of learning. It's a parameter that continues to circulate at both ends of a valley, but cannot reach the bottom.</p>
<p>This is not in line with the general training situation, as the parameters of the in-depth learning model are so many that they tend to be reduced when the gradient is still large, and we do not need a small gradient to stop training.</p>
<p>We can try to lower the learning rate, which determines the pace at which the parameters are updated, and the learning rate is too high to slip slowly into the valley. However, it is not always valleys, and too little learning is usually not allowed to move on.</p>
<p>In conclusion, in the gradient decline, all parameters are subject to the same learning rate, which is clearly not sufficient and should be customized for each parameter, i.e. introduced<strong>Adaptive learning rate (adaptative learning rate)</strong> The method gives each parameter a different learning rate.</p>
<p>If the gradient is very small and flat in one direction, we would like the learning rate to increase; if it is very steep and steep in one direction, we would like the learning rate to be smaller.</p>
<h4>AdaGrad</h4>
<p>AdaGrad (Adaptive Gradient) is a typical self-adaptation learning rate method that allows for automatic adjustment of the learning rate to the size of the gradient. When AdaGrad can achieve a higher gradient, the learning rate decreases, and the learning rate increases when the gradient is smaller.</p>
<p>This is our traditional way of updating our parameters.
&#36;&#36;\bardsymbol(theta}<em>{t+1}^{i}\leftarrow\boldsymbol{\theta}</em>{t}^{i}-\eta\boldsymbol{g}_{t}^{i}&#36;&#36;</p>
<p>Now there's a learning rate that customizes with parameters, i.e., the original learning rate.&#36;\eta&#36;♪ Turn into ♪&#36;\frac\eta{\sigma_t^i}&#36;
&#36;&#36;\theta_{t+1}^i\leftarrow\theta_t^i-\frac\eta{\sigma_t^i}\boldsymbol{g}_t^i&#36;&#36;
This learning rate goes with the parameters.&#36;i&#36;Number of times&#36;t&#36;Now our learning rate becomes parameter-related.</p>
<p>A common type associated with parameters is the mean root of the gradient. The process of updating the parameters is:
&#36;&#36;\theta_1^i\leftarrow\theta_0^i-\frac\eta{\sigma_0^i}\boldsymbol{g}_0^i&#36;&#36;
of which&#36;\boldsymbol{\theta}_0^i&#36;is the initialization parameter. And...&#36;\sigma_0^i&#36;..the calculation process is
&#36;&#36;\sigma_0^i=\sqrt{\left(\boldsymbol{g}_0^i\right)^2}=\begin{vmatrix}\boldsymbol{g}_0^i\end{vmatrix}&#36;&#36;
of which&#36;g_0^i&#36;It's a gradient. Will&#36;\sigma_0^i&#36;the value of which can be replaced by an updated formula&#36;\frac{g_0^i}{\sigma_0^i}&#36;+1 or 1. First time updating parameters, from&#36;\boldsymbol{\theta}_0^i&#36;Update to&#36;\boldsymbol{\theta}_1^i&#36;Sometimes, you add&#36;\eta&#36;Or cut it off.&#36;\eta&#36;,<strong>It's not about the size of the gradient. This is the first step.</strong></p>
<p>The second update of parameters is as follows:
&#36;&#36;\bardsymbol(theta}<em>{2}^{i}\leftarrow\boldsymbol{\theta}</em>{1}^{i}-\frac{\eta}{\sigma_{1}^{i&#125;&#125;\boldsymbol{g}_{1}^{i}&#36;&#36;
其中
&#36;&#36;\sigma 1^sqrt{frac)^left[\left(\boldsymbol{g}^i\right}^left(\boldsymbol{g}^i\right)^ &#36;
The average root of the gradient is used to limit learning rates&#36;\mu&#36; That's how it came to be.</p>
<p>The same thing can be repeated.
&#36;&#36;\boldsymbol\theta}i\leftarrow\boldsymbol{t^i-\frac\eta}\boldsymbol{g}<em>t^i\quad\sigma_t^i=\sqrt{\frac1{t+1}\sum</em>{i=0}^t\left(\boldsymbol{g}_t^i\right)^2}&#36;&#36;</p>
<p>This algorithm works:<strong>Adapting to lower learning rates when the gradient is larger and increasing learning rates when the gradient is smaller ensures that the overall optimization is limited in length</strong></p>
<h4>RMSProp</h4>
<p>The learning rate required by the same parameter will change over time. RMSProp (Root Mean Squared Protection) was designed to address this issue. He made the gradients from time to time very important.</p>
<p>RMSprop does not have a dissertation, and Geoffrey Hinton has an in-depth study on Coursera, where he talks about RMSprop, which, if quoted, needs to refer to the link to the corresponding video.</p>
<p>RMSprop's first step is the same as Adagrad's, which is
&#36;&#36;\sigma_0^i=\sqrt{\left(\boldsymbol{g}_0^i\right)^2}=\begin{vmatrix}\boldsymbol{g}_0^i\end{vmatrix}&#36;&#36;</p>
<p>The second update process is as follows:
&#36;&#36;\boldsymbol{\theta}_2^i\leftarrow\boldsymbol{\theta}_1^i-\frac\eta{\sigma_1^i}\boldsymbol{g}_1^i\quad\sigma_1^i=\sqrt{\alpha\left(\sigma_0^i\right)^2+\left(1-\alpha\right)\left(\boldsymbol{g}_1^i\right)^2}&#36;&#36;
of which&#36;\alpha&#36; is a hyperparameter between 0-1. In RMSprop, you can adjust yourself to the importance of the new calculation gradient.&#36;\alpha&#36; The smaller the newly calculated gradient, the more important.</p>
<p>The next steps are:
&#36;&#36;\boldsymbol(t+mathbf) loftarrow\boldsymbol(t\frac\eta}\boldsymbol{g}<em>t^i\quad\sigma_t^i=\sqrt{\alpha\left(\sigma</em>{t-1}^i\right)^2+\left(1-\alpha\right)\left(\boldsymbol{g}_t^i\right)^2}&#36;&#36;</p>
<p><strong>RMSprop can quickly "stamp the brakes." Adjusting the learning rate as soon as possible based on the latest gradient</strong>Whether it's bigger or abbreviated</p>
<h4>Adam</h4>
<p>The most commonly used optimizer strategy or optimizer (optimizer) is Adam Adam, who can be considered RMSprop plus momentum, using kinetics to update the direction of the parameters and be able to adapt to the learning rate.</p>
<p>PyTorch has already written Adam Optimizer, which contains some super-parameters that need to be determined by humans, but it's always enough with the PyTorch pre-set parameters.</p>
<h3>Movement of learning rates</h3>
<p>None of the study rate adjustment algorithms we have in front of us avoid a problem, and the learning rates in all directions are affected by several gradients. Although the RMSProb algorithm reinforces the current gradient, it still takes time to adjust the learning rate to the right level, and these calculations and the time spent on computing resources are actually wasted.</p>
<p>Pass.<strong>Learning rate scheduling</strong> It can be solved. Previous study rate adjustment&#36;\eta&#36;It's a fixed value, and the learning rate is moving. Medium&#36;\eta&#36;It's about time, as shown below.
&#36;&#36;\boldsymbol{\theta}_{t+1}^i\leftarrow\boldsymbol{\theta}_t^i-\frac{\eta_t}{\sigma_t^i}\boldsymbol{g}_t^i&#36;&#36;
The most common strategy in the program is learning rate decline, also known as<strong>Learning rate retreating</strong>I don't know. As the parameters update, let&#36;\eta&#36;It's getting smaller and smaller, and it helps us get to target accuracy faster in the finely optimized part of the back.</p>
<p>In addition to the decline in the learning rate, there is another classic way of moving the learning rate - preheating. The method of preheating is to make learning grow bigger and smaller, and how much, how much, how much, how much, how much, how much, how much, how much. Preheat is used in training for the disability network, Bert and Transformer.</p>
<p>The core of preheat
 <strong>When we use Adam, RMSprop or AdaGrad, we need to calculate&#36;\sigma&#36;I don't know. And...&#36;\sigma&#36;It's a statistical result. From&#36;\sigma&#36;Know the steepness of a particular direction. The results of the statistics require enough data to be accurate, starting with the results&#36;\sigma&#36;It's not accurate. At first, the low learning rate was used to explore and gather information on the surface of errors.&#36;\sigma&#36;Statistics, etc.&#36;\sigma&#36;When statistics are more accurate, learning rates slowly climb.</strong></p>
<p>Preheat can be consulted in Adam's advanced version - - R Adam.</p>
<h3>Optimization of summary</h3>
<p>After the previous subsections, we have finally become quite clear in introducing the issue of optimization, and we have evolved from the most primitive gradient to this version.
&#36;&#36;\bardsymbol(theta}<em>{t+1}^{i}\leftarrow\boldsymbol{\theta}</em>{t}^{i}-\frac{\eta_{t&#125;&#125;{\sigma_{t}^{i&#125;&#125;\boldsymbol{m}_{t}^{i}&#36;&#36;</p>
<p>There's momentum in this version.&#36;m_{t}^{i}&#36; The previous section, “Bulk and kinetic”, does not update the parameters in the direction of the gradient calculated at a given time, but uses all past calculations of the gradient as a weighted sum as the direction of the update.</p>
<p>We then use the self-adaptation learning rate methodology to help us optimize and add to the learning rate schedule.</p>
<p>This is the complete version of the current optimization, which, in addition to Adam, has various variations. But they're all fixing the drive in different ways, adapting to the learning rate, and doing the learning rate movement.</p>
<p><strong>In particular, both the kinetic method and the self-adaptation learning rate take into account past gradients, but the kinetic method also takes into account their direction, while the self-adaptation learning rate focuses more on its size.</strong></p>
<h3>Batch Harmonization</h3>
<h4>Normalization.</h4>
<p>If the error surface is rough, it is harder to train. Can you just change the surface of the error and make it better to train by "shaping the mountains"? Batch Normalization BN is one of the ideas of “plaining the mountains”.</p>
<p>Let us now reflect on a matter of the nature of the error that is difficult to train. The rugged error surface is essentially a small parameter disturbance leading to a large error, and we deal with this problem in traditional statistics --<strong>Normalization</strong>  Largely variable-differentiated regression factors are generated in online regression models. That's what we're talking about.  <strong>Characterization</strong></p>
<h4>Consider in-depth learning</h4>
<p>Since the Depth Learning Model is a layered web structure, his integration naturally raises a number of other issues.</p>
<p>Although we've got the initial data on training.&#36;x&#36;It's been consolidated, but it's going through a network.&#36;W_1&#36;Later, we get it.&#36;z&#36;No regularization, which leads to training next level&#36;W_2&#36;There are some difficulties.</p>
<p>So we're right.&#36;z&#36;Re-incorporate (the choice of the form of integration depends on the activation function), where the common habit is to put integration before the activation function, and then transmit the result of the activation function to the bottom of the network.</p>
<p>Special: The training is conducted on a catch, so the integration is also done on a catch-wide basis, which means we need a slightly larger watch size to ensure the approximation of distribution.</p>
<p>In the case of batch integration, the following operations are often carried out:
&#36;&#36;\hat{\boldsymbol{z&#125;&#125;^i=\gamma\odot\tilde{\boldsymbol{z&#125;&#125;^i+\beta &#36;&#36;
Of which,&#36;\odot&#36;represents the multiplication of elements by element.&#36;\beta,\boldsymbol{\gamma}&#36; It can be conceived as a network parameter that needs to be learned again.</p>
<p>Why?&#36;\boldsymbol{\beta}&#36;Call.&#36;\boldsymbol{\gamma}&#36;And?</p>
<p>If you do this, you'll have to do it.&#36;\tilde{\boldsymbol{z&#125;&#125;&#36;The average must be zero, and if the average is zero, this will limit the network, which may have a negative impact, so we need to put&#36;\beta,\boldsymbol{\gamma}&#36;Add it back so that the output average of the network's hidden layer is not zero. Let the Internet learn.&#36;\boldsymbol{\beta},\boldsymbol{\gamma}&#36;To adjust the distribution of the output, to adjust it.&#36;\hat{\boldsymbol{z&#125;&#125;&#36;the distribution.</p>
<p>Batch integration is designed to make each dimension the same, if&#36;\gamma&#36;Call.&#36;\beta&#36;It's not like it's all the same.</p>
<p>It's possible, but actually in training,&#36;\boldsymbol{\gamma}&#36;The initial values are set at one, so...&#36;\boldsymbol{\gamma}&#36;All values are 1 vectors.&#36;\beta&#36;is all zero vectors, or zero vectors. So the distribution of each dimension of the network is closer at the beginning of the training, and perhaps it's been trained long enough to find a better surface of error, to go to a better place, and then...&#36;\gamma,\beta&#36;I put it in slowly, so I added it.&#36;\gamma,\beta&#36;Batch aggregation is often helpful for training.</p>
<h4>Batch Harmonization at Test</h4>
<p>These are all parts of the training, and the tests are sometimes called extrapolations. What's wrong with mass integration when it's tested? At the time of testing, we will get all the test data at once, and it is not appropriate to continue with the batch.</p>
<p>In fact, mass integration does not require any special treatment at the time of testing, and PyTorch has already done it. When you're training, if you're doing batch integration, it's done every batch.&#36;\mu,\sigma&#36; , which is used to calculate the moving average. Assuming there's now a lot of calculations.&#36;\boldsymbol{\mu}^1,\boldsymbol{\mu}^2,\boldsymbol{\mu}^3,\cdots\cdots,\boldsymbol{\mu}^t&#36;, the moving average is calculated
&#36;&#36;\bar{\boldsymbol{\mu&#125;&#125;\leftarrow p\bar{\boldsymbol{\mu&#125;&#125;+(1-p)\boldsymbol{\mu}^t&#36;&#36;
Of which,&#36;\bar{\boldsymbol{\mu&#125;&#125;&#36;Yes.&#36;\boldsymbol{\mu}&#36;It's an average.&#36;p&#36;It's a factor. It's also a constant. It's also a super-parameter and one that needs to be adjusted.</p>
<p>In PyTorch,&#36;p&#36;Set 0.1. Calculating an average slide to update&#36;\mu&#36;average. At the end of the test, you don't have to count what's inside.&#36;\mu&#36;Call.&#36;\sigma&#36;Got it. Because at the time of the test, there was no batch in the real application, so you could just take it.&#36;\bar{\mu}&#36;Call.&#36;\bar{\boldsymbol{\sigma&#125;&#125;&#36; Which means...&#36;\boldsymbol{\mu},\boldsymbol{\sigma}&#36;During training, the average move received replaced the original&#36;\boldsymbol{\mu}&#36;Call.&#36;\boldsymbol{\sigma}&#36;, and that's how batch integration works when it's tested.</p>
<h4>The role of bulk consolidation</h4>
<p>There's a paper to support the core of the presentation. <strong>Batch consolidation can change the surface of the error and make it more smoother.</strong> So we can choose a higher rate of learning, thereby increasing the efficiency of training.</p>
<p>While the literature considers the final training to be in the same location, the increased efficiency of training can save resources and time for training, and therefore there is merit in bulk consolidation.</p>
<p>Of course, mass subordination is not the only way to normalize, and there are many ways to regularize them, which are the result of dealing with the appearance of error, although there are significant differences in specific thinking, most of which are by chance found by the author of the article.</p>
<h2>Neural network, multilayer sensor MLP, deep learning.</h2>
<h3>Neutron model</h3>
<p><strong>Neural networks are a broad, parallel network of simple adaptive units whose organization can simulate the interaction of bioneurological systems with real world objects.</strong>This is our definition of neural network.</p>
<p>In fact, the neural network is not a branch of machine learning, it is much earlier in biology than machine learning; the neural network that we are introducing here is the product of the interlocking of machine learning and neural network learning, and the basic theory of deep learning before it comes into being.</p>
<p>The most basic component of the nervous network is the neuron model; it accepts data from multiple connected neurons and gives output; The basic neurons model is the MP neurons model. The neurons receive input signals from several other neurons, which are transmitted through the connection of weights, the total input values received by the neurons are compared to the threshold values of the neurons, and are then processed through the Activation function to produce neurons output.</p>
<p>Connecting many of these neurons to a certain level of structure gives them a neural network, which in mathematics is a mathematical model with very many parameters, receiving input, giving output.</p>
<h3>Sensor and multilayer network</h3>
<p>Perceptron consists of two layers of neurons, which receive external input signals and pass them to the output layer, which is a neuron. Select the appropriate active function.</p>
<p>It should be noted that the sensor is only active with an output-level neurons, i.e. with only one layer of functional neurons, with very limited learning skills, which in fact proves mathematically that the sensor cannot handle any non-linear problem.</p>
<p>To solve the problem of non-linear subdivisions, consideration needs to be given to using multi-layer functional neurons to add an intermediate layer to the input neurons and output neurons, and it also has an active function to find the right weights in training.</p>
<p>Each of the most classic multi-layer nervous network structures is fully interconnected with the lower neurons, and there is no inter-layer connection between the neurons and no inter-layer connection.<strong>Multi-layer front feed neural network.</strong>&quot; (multi-layer feedforward neuralnetworks, MLP) is also one of our most basic nervous network structures.</p>
<h3>Error/dissemination algorithm (BP)</h3>
<p>Multilayered networks are much better at learning than single-layer sensory machines, and training them is a problem.</p>
<p>In reality, most of the neuronets are trained by BP algorithms, which can be used not only for the multilayer frontal feed neural network that we've introduced, but also for a lot of neural network training, but the BP neural network usually refers specifically to the multilayeral feeder neural network (MLP) that is trained by BP algorithms.</p>
<p>Now, let's talk about the BP algorithm, the training data set.&#36;D&#36; Each sample includes&#36;d&#36;Inputs and&#36;l&#36;So, we built a possession.&#36;d&#36; It's an input neurons. &#36;l&#36;An output neuron.&#36;q&#36;The multilayer front feed network structure of a hidden neurons Activate all functions selected as Sigmod functions</p>
<p>For any training example, the average error caused by the network is
&#36;&#36;E_k=\frac{1}{2}\sum_{j=1}^{l}(\hat{y}_j^k-y_j^k)^2.&#36;&#36;
We need to learn the parameters of input to the hidden layer.&#36;d\times q&#36; A weight to hide the layer to the output layer.&#36;q\times l&#36; Right &#36;q+l&#36;The threshold of a neuron, it's obviously not realistic to optimise so many parameters at a time, and BP is an iterative learning algorithm that updates the parameters in a broad sense learning rule for each round of it.
&#36;&#36;v\leftarrow v+\Delta v.&#36;&#36;</p>
<p>BP algorithm is based on the Gradient Decline policy, adjusting parameters in the negative gradient direction of the target.
gave error&#36;E_k&#36; Learning rate&#36;\eta&#36; Yes.
&#36;&#36;\Delta w_{hj}=-\eta\frac{\partial E_{k&#125;&#125;{\partial w_{hj&#125;&#125;&#36;&#36;
Depending on the link of influence, we can give you
That's a good idea.<em>{j}^{k&#125;&#125;\cdot\frac{\partial\hat{y}</em>{j}^{k&#125;&#125;{\partial\beta_{j&#125;&#125;\cdot\frac{\partial\beta_{j&#125;&#125;{\partial w_{hj&#125;&#125;&#36;&#36;
因此 有
&#36;&#36;\frac{\partial\beta_{j&#125;&#125;{\partial w_{hj&#125;&#125;=b_{h}.&#36;&#36;
&#36;&#36;\begin{aligned}
g_{j}&amp; =-\frac{\partial E_{k&#125;&#125;{\partial\hat{y}<em>{j}^{k&#125;&#125;\cdot\frac{\partial\hat{y}</em>{j}^{k&#125;&#125;{\partial\beta_{j&#125;&#125;  \
&amp;=-(\hat{y}<em>{j}^{k}-y</em>{j}^{k})f^{\prime}(\beta_{j}-\theta_{j}) \
&amp;=\hat{y}<em>{j}^{k}(1-\hat{y}</em>{j}^{k})(y_{j}^{k}-\hat{y}<em>{j}^{k}).
\end{aligned}&#36;&#36;
我们就能给出BP算法权重的更新公式为
&#36;&#36;\Delta w</em>== sync, corrected by elderman ==
Similarly, we can give updated formulas for other parameters;</p>
<p>The learning rate controls the pace at which algorithms are updated, too much can produce oscillations, too much can lead to too slow a contraction, and the choice of the appropriate learning rate is an issue worth considering in training.</p>
<p>The algorithm that follows is based on a model of minimization of MSEs, and if we read all the training data at once, we can optimize the entire MSE, but this slows down the pace of training, especially when the training data sets are too large, and in fact we usually use batch training for NN-based models.</p>
<p>There's a mathematical theory: a multi-layer frontal feed network can approach a continuous function with any degree of precision. However, how to set the number of hidden neurons remains an open question and is usually applied by a “test-in” method. Special-by-error) adjustments</p>
<p>Because of its powerful representational ability, the BP nervous network is often colluded; either we divide the training and test sets and terminate the training without continuing to reduce the concentration error, or we add a regular item to the target function to punish the network complexity.
&#36;&#36;E=\lambda\frac1m\sum_{k=1}^{m}E_{k}+(1-\lambda)\sum_{i}w_{i}^{2},&#36;&#36;</p>
<h3>In-depth learning</h3>
<p>In theory, the more complex models the more parameters, the greater the capacity, which means that it can perform more complex learning tasks. In general, however, complex models are inefficiently trained, prone to oversatisfaction and therefore difficult to favour.</p>
<p>And with cloud computing, big data age,<strong>A significant increase in computing capacity mitigates training inefficiencies and a significant increase in training data reduces the risk of overloading</strong>, therefore, with "Deep Learning"&quot;(Deep learning) The complex model represented is beginning to get attention.</p>
<p>The typical deep learning model is the deep neural network; In terms of increasing the complexity of models,<strong>The increase in the number of hidden spheres is clearly more effective than the increase in the number of hidden neurons.</strong>, because increasing the number of hidden layers increases not only the number of neurons with active functions, but also the number of layers that activate the embedded functions.</p>
<p>We'll use two subsections to briefly discuss two small issues in the field of in-depth learning, which will be the basis for our detailed presentation on in-depth learning.</p>
<h3>FFN before full connection.</h3>
<p>The whole connection means that the parameters are densely connected, and the so-called FeedForward network, FFN, consists of an active layer in the middle of two linear transformers.</p>
<p>FeedForward extracts deeper features from linear transformations and non-linear activation functions, which first map data into space at high latitudes and then into space at low latitudes; and by introducing non-linear transformations to activate functions, enhance model alignment to complex models.</p>
<p>While the structure and multilayer sensor MLP are the same logic of formation as the entire pre-connected neural feed network FPN, the FPN places greater emphasis on Module in a large network, which is Module for enhancing the ability to develop the most complex models, and MLP emphasizes that we use a separate network.</p>
<p>In the Transformer model, the FNN layer occupies the vast majority of parameters. Instead, the attention-level QKVO matrix does not occupy that much of the parameters, although they are at the heart of the attention mechanism.</p>
<h3>Disabled connection</h3>
<p>ResNet is a landmark model in the history of in-depth learning, and the models of in-depth learning prior to ResNet are generally between 20 and 30 floors, but after the emergence of ResNet, the number of in-depth learning models is raised to over 100 floors, even to 1,000 floors.</p>
<p><strong>ResNet primarily addresses degradation of deep networks</strong>It's not to solve the problem of alignment, it's not to solve the problem of the disappearance of gradients and the explosion of gradients, which can already be solved by bulk consolidation, but it's not as good as the subsurface network when your network level is further deepened, which is called degradation.</p>
<p>The key features of in-depth learning, compared to traditional machine learning, are deeper network layers, non-linear conversion (activated), automatic characterization extraction and characterization. Non-linear conversion is a key objective, which maps data to high latitudes to facilitate better completion of “data disaggregation”. As the network deepens, it is introduced.<code>激活函数</code>And more and more, the data are being mapped into more discrete spaces, and it is difficult to get the data back to where they came from.</p>
<p>The core idea of the disability network is that each additional layer should more easily include the original function as one of its elements. If we can map the new layer with constant training, then there will be no degradation, while the new layer can bring new identification possibilities, and thus better results.</p>
<p>Disable connections are very simple in terms of implementation.&#36;x&#36;As input, after multiple layers of full connection and activation (e.g. through a FPN), output&#36;F(x)&#36;So the reverse transmission process behind us is to match this function.&#36;F&#36;.</p>
<p>After using the residual connection, our output became&#36;F(x)+x&#36; In order to ensure the dimensions of the residual connection, we need to introduce some layers of a 1 width. In debris (one of the smallest Modules used for disability connections), input can be disseminated more rapidly forward via cross-layer data lines. It brings about better performance.</p>
<h2>Questions about deep learning Essence</h2>
<p>In-depth learning of DLs is parametrical, that is, Over-Parametrization, but still successful, contrary to some theoretical assumptions about the original machine, and in the absence of a new theoretical explanation.</p>
<p>The optimization of in-depth learning is an obvious non-optimization problem, but he uses a simple optimisation method (Adam, SSD, etc., using gradients) that tends to yield good results in practice.</p>
<p>This is done with the current initial technology of randomization (avoiding local non-ejectability), parametrical neural network over-theatre (super-high space avoids local excellence), gradients disappear (reLU, Adam, Resnet, Batch Normal, Layer Normal).</p>
<h2>Categorization and cross-radon losses</h2>
<p>Classification and return are the two most common types of issues for in-depth learning. There has been only sporadic cross-plymerization, starting with negative logarithmic losses, which have been associated with subclassifications, multiclassifications and the realization of works.</p>
<h3>Start with negative logarithmic losses</h3>
<p>Instead of looking at the model structure for the time being, it would be a black box. Assuming that the model has given the right type of probability. &#36;p&#36;, zero of which&lt;p&lt;1&#36;。我们希望 &#36;p. The larger the loss, the smaller the loss; the most direct option is negative logarithmic loss:</p>
<p>&#36;&#36;L=-\log p&#36;&#36;</p>
<p>The code is simple:</p>
<pre><code class="language-python">import math

loss = -math.log(p)
</code></pre>
<h3>Start Category 2</h3>
<p>The issue is now classified as a category II. Labels only 0 and 1, use &#36;y&#36; Show real labels, use &#36;p&#36; It means the model thinks &#36;y=1&#36; The probability, then. &#36;y=0&#36; The probability is... &#36;1-p&#36;I don't know. Write two scenarios into the same pattern:</p>
<p>&#36;&#36;L=-[y\log p+(1-y)\log(1-p)]&#36;&#36;</p>
<p>This is the source of binary Cross Entropy, BCE.</p>
<p>Training usually takes place in a mini-batch. Calculates the loss for each sample, and then fixes or averages the values for the frame; the common default settings are averages. To look straight at the formula, write it in a loop:</p>
<pre><code class="language-python">import math

def binary_cross_entropy(ys, ps):
    total_loss = 0.0

    for y, p in zip(ys, ps):
        loss = -(y * math.log(p)
                 + (1 - y) * math.log(1 - p))

        total_loss += loss

    return total_loss / len(ys)
</code></pre>
<p>The actual project will be quantified using a NumPy or a stretch frame without hand Write <code>for</code> Loop. The visualization up there is assuming everything. &#36;p&#36; It's all strictly located. &#36;(0,1)&#36; I'll meet you there. &#36;\log 0&#36;。</p>
<p>The binary label can be considered as subject to Bernoulli distribution. The negative logarithmic is as follows:</p>
<p>&#36;&#36;\mathrm{NLL}=-[y\log p+(1-y)\log(1-p)]&#36;&#36;</p>
<p>So, minimizing BCE is the equivalent of minimizing negative logarithm, which is maximizing. This idea remains valid when it is later extended to multi-classification and soft target.</p>
<h3>From output logit to probability</h3>
<p>The previous section always assumed that the model had given probabilities. &#36;p&#36;, but the last linear layer of the neural network usually produces a real number, i.e. logit, which is not limited in scope. In the second category, it can be written as follows:</p>
<p>&#36;&#36;z=w^{T}x+b&#36;&#36;</p>
<p>Sigmoid can map any actual number. &#36;(0,1)&#36;, so this output can be interpreted as probability:</p>
<p>&#36;&#36;p=\sigma(z)=\frac{1}{1+e^{-z&#125;&#125;&#36;&#36;</p>
<pre><code class="language-python">import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))
</code></pre>
<p>A single sample of BCE can be written in visual terms:</p>
<pre><code class="language-python">import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def binary_cross_entropy_from_logit(z, y):
    p = sigmoid(z)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))
</code></pre>
<p>This code corresponds to a mathematical definition, but it is not suitable for direct training: extreme logit will allow &#36;p&#36; Round to 0 or 1, then trigger &#36;\log 0&#36;I don't know. In Sigmoid, logit is also a probability logarithmic:</p>
<p>&#36;&#36;z=\log\frac{p}{1-p}&#36;&#36;</p>
<h3>BCEWithLogits</h3>
<p>Extreme logit gives rise to numerical instability that has nothing to do with the mathematical definition, but that actually affects the calculation. Combining Sigmoid with the BCE, you can bypass the problem of probability before logarithm.</p>
<p>Start with the following formula:</p>
<p>&#36;&#36;L=-[y\log\sigma(z)+(1-y)\log(1-\sigma(z))]&#36;&#36;</p>
<p>Take Sigmoid and simplify it.</p>
<p>&#36;&#36;L=y\log(1+e^{-z})+(1-y)\log(1+e^z)&#36;&#36;</p>
<p>Further organized as:</p>
<p>&#36;&#36;L=\log(1+e^z)-yz&#36;&#36;</p>
<p>The BCEWithLogits format can eventually be changed to a numerically stable form:</p>
<p>&#36;&#36;L=\max(z,0)-zy+\log(1+e^{-|z|})&#36;&#36;</p>
<p>The code is as follows:</p>
<pre><code class="language-python">import math

def bce_with_logits(z, y):
    return (
        max(z, 0)
        - z * y
        + math.log1p(math.exp(-abs(z)))
    )
</code></pre>
<p>That's why PyTorch is used. <code>torch.nn.BCEWithLogitsLoss</code> , instead of first doing Sigmoid. Multi-Category Use <code>torch.nn.CrossEntropyLoss</code> It also receives logits directly, but its internal combinations are LogSoftmax and NLLLOSs, not mixed.</p>
<p>For a single sample requesting a gradient, the result happens to be:</p>
<p>&#36;&#36;\frac{\partial L}{\partial z}=\frac{1}{1+e^{-z&#125;&#125;-y=p-y&#36;&#36;</p>
<p>The code is direct:</p>
<pre><code class="language-python">def grad_bce_logit(z, y):
    p = sigmoid(z)
    return p - y
</code></pre>
<p>If the damage is averaged, the number of samples will also need to be divided. Compare Sigmoid to MSE, this gradient will not add up Let's go. &#36;p(1-p)&#36;As a result, it is usually easier to optimise in the Sigmoid saturation area; this does not mean that the entire network does not have gradients that disappear.</p>
<h3>Softmax and multiple categories</h3>
<p>There's only one logit that fits the second category. For multiple categories, the last linear layer will output a vector with dimensions equal to the number of categories. Take three categories, for example:</p>
<p>&#36;&#36;[x_1,x_2,x_3]&#36;&#36;</p>
<p>If Sigmoid is done separately for each value, the probability of each category is independent of each other and does not have to be 1 in sum, which is more appropriate for multi-label classification. Multiple classifications require competition between categories, and Softmax translates the whole group of logits into a combined probability distribution of 1:</p>
<p>&#36;&#36;p_i=\frac{\exp(x_i)}{\sum_j\exp(x_j)}&#36;&#36;</p>
<p>At this probability distribution, the multi-classical cross-tape is:</p>
<p>&#36;&#36;L=-\sum_i y_i\log p_i&#36;&#36;</p>
<p>If the target is one-hot vector, only the one that corresponds to the correct category will be retained. Look at a visual realization:</p>
<pre><code class="language-python">import math

def softmax(logits):
    exp_values = [math.exp(z) for z in logits]
    total = sum(exp_values)

    return [v / total for v in exp_values]

# target 只需要接受一个目标类别，不需要全部 label
def cross_entropy(logits, target):
    probs = softmax(logits)
    return -math.log(probs[target])
</code></pre>
<p>This version is close to the formula, but the larger logit will spill over the index. When calculating Softmax, you can subtract the maximum value first; when calculating crossbow, you can use LogSumExp:</p>
<pre><code class="language-python">import numpy as np

def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)

def cross_entropy(logits, target):
    logits = np.asarray(logits, dtype=float)
    m = np.max(logits)
    logsumexp = m + np.log(np.sum(np.exp(logits - m)))
    return logsumexp - logits[target]
</code></pre>
<p>Two categories Softmax can also be transformed into two loguit differences Sigmoid. For example, the second probability is that &#36;\sigma(x_2-x_1)&#36;So the two are very close in mathematics and engineering.</p>
<p>Crossing entropy. &#36;k&#36; The gradient of a logit is also simple:</p>
<p>&#36;&#36;\frac{\partial L}{\partial x_k}=p_k-y_k&#36;&#36;</p>
<h3>Other isolated topics</h3>
<p>Multi-label classification usually calculates BCE for each label separately, rather than using Softmax. Each label is highly probable because it is not mutually exclusive and does not need to add up to equal 1.</p>
<p>LLM's next token projection is essentially to calculate multi-classic cross-paramerics on the vocabulary. Training usually calculates loss only for valid target token and takes averages on all valid token of a bat; other options can be used for different frames.</p>
<p>Padding should not be involved in loss calculations, usually by <code>ignore_index</code> Or los mask excludes.</p>
<p>Crossing entropy is also acceptable, soft target is retained at this time &#36;-\sum_i y_i\log p_i&#36; All items in it. As long as the target is the combined probability distribution of 1 the gradient remains &#36;p-y&#36;I don't know. Knowledge distillation and label smoothing are used in this form, in which the distillation of knowledge can be used to transmit more information on the distribution of categories given by the teacher model.</p>
<p>Temperature parameters will replace logit in Softmax with &#36;x_i/T&#36;I don't know. The following is given for a value stabilization:</p>
<pre><code class="language-python">import math

def softmax(logits, temperature=1.0):
    if temperature &lt;= 0:
        raise ValueError(&quot;temperature must be positive&quot;)

    scaled = [z / temperature for z in logits]
    maximum = max(scaled)
    exp_values = [math.exp(z - maximum) for z in scaled]
    total = sum(exp_values)
    return [v / total for v in exp_values]
</code></pre>
<p>When &#36;T&gt;1&#36; 时，概率分布会更平滑；当 &#36;0&lt;T&lt;The distribution will be sharper at &#36;1.00.</p>
