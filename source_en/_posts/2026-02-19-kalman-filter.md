---
title: 'The Kalman Filter Family: KF, EKF, UKF, and EnKF'
title_zh: 卡尔曼滤波家族：KF、EKF、UKF 与 EnKF
date: 2026-02-19 20:00:00 +0800
categories:
- Data Science
- Time Series & Spatial Data
tags:
- Time Series
author: Hyacehila
mathjax: true
hidden: true
excerpt: State estimation is a foundation of modern technology. This post introduces Kalman filters, extended Kalman filters,
  unscented Kalman filters, and ensemble Kalman filters.
description: State estimation is a foundation of modern technology. This post introduces Kalman filters, extended Kalman filters,
  unscented Kalman filters, and ensemble Kalman filters.
excerpt_zh: 本文介绍卡尔曼滤波（KF）、扩展卡尔曼滤波（EKF）、无迹卡尔曼滤波（UKF）和集合卡尔曼滤波（EnKF）的基本假设、更新过程与适用场景。
permalink: /blog/2026/02/19/kalman-filter/
lang: en
translation_key: 2026-02-19-kalman-filter
translation_status: machine
translation_source_hash: 65781b8a5527c34bbca303f778fea71ef8696160415f377eede39fd4bb01b3f1
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Kalman Filter</h2>
<p>The questions in this article can also be addressed<a href="/en/blog/2024/01/30/linear-time-series-analysis-notes/">Linear time series analysis: smooth sequence, ARMA and ARIMA</a>、<a href="/en/blog/2024/05/06/univariate-financial-time-series-analysis-notes/">Financial time series analysis: ARCCH/GARCH effect and volatility modelling</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h3>Why do you need Kalman filters?</h3>
<p>State Estimates are fundamental issues in robotics, autopilot and aerospace. Engineering systems need to combine physical models with noise-bearing sensor readings to estimate states that cannot be observed directly and accurately. Kalman Filter provided a progressive calculation method for the linear status spatial model.</p>
<p>Assuming that we estimate the location and speed of the vehicles, two sources of information can be used, but they all have errors:</p>
<ol>
<li>Your speed sheet: shows 100 km/h, but it may actually be between 90 and 110 km/h.</li>
<li>Your GPS: Shows position, but the signal is highly drifting, possibly wandering within 50 meters.</li>
</ol>
<p>The speed tables and GPS measurements are not identical. Kalman filters first push the position and speed of the previous moment to the current moment using a motion model, then revise the predictions with observations such as GPS. This process can be understood as assigning weights to uncertainties, but not just as a simple average of two readings.</p>
<p>If the sensor noise is smaller, more reference observations are made when updated; if the model predicts less uncertainty, the level of update is reduced. The two are being given a weight of ** Kalman Gain, a group of people who are not in the country. &#36;K&#36;** Decision.</p>
<p>When assumptions such as online, Goss noise are established, the Kalman filter can be estimated at a minimum average error, while maintaining the uncertainty of the estimate.</p>
<ul>
<li>Status &#36;x&#36;: Values to be sought (e.g. location, speed).</li>
<li>Arguments &#36;P&#36;: describe the difference between the difference in the state estimate and the amount of the agreement between the variables.&#36;P&#36; The smaller the model, the more concentrated the current estimate is, but it is not directly equivalent to the "Error range" &#36;\pm 5&#36; Me.</li>
</ul>
<h3>Create two equations (world description)</h3>
<p>First, you describe the system in mathematical language. Standard Kalman filtering scenario shift and observation relationship is<strong>Linear</strong>and assigns the difference between the process noise and the measurement of the noise:</p>
<p><strong>State equation (physical model):</strong></p>
<p>&#36;&#36;
x_k = Fx_{k-1} + Bu_k + w_k
&#36;&#36;</p>
<ul>
<li>&#36;x_k&#36;: Current status.</li>
<li>&#36;F&#36;: Status transfer matrix (e. g. next-second position = last-second position + speed) &#36;\times&#36; Time.</li>
<li>&#36;u_k&#36;: Control volume (e.g., stepping on the gas).</li>
<li>&#36;w_k&#36;：<strong>Process Noise</strong>(Process Noise), subject to Goss distribution. &#36;N(0, Q)&#36;I'm sorry. Here. &#36;Q&#36; It represents the degree of spectrometry of physical models.</li>
</ul>
<p><strong>Observation equation (sensor model):</strong></p>
<p>&#36;&#36;
z_k = Hx_k + v_k
&#36;&#36;</p>
<ul>
<li>&#36;z_k&#36;Sensor reading.</li>
<li>&#36;H&#36;: Observation matrix (magnify the state to read, for example, the state is [location, velocity], the sensor only measures [location],&#36;H&#36; Yeah. &#36;[1, 0]&#36;）。</li>
<li>&#36;v_k&#36;：<strong>Measuring Noise</strong>(Measurement Noise) &#36;N(0, R)&#36;I'm sorry. Here. &#36;R&#36; (c) Representing the sensor ' s insinuity.</li>
</ul>
<h3>Core algorithm</h3>
<p><strong>Step A: Forecast (Time Update) - Before looking at sensors, estimate where.</strong></p>
<ol>
<li><p><strong>Debit status:</strong></p>
<p>&#36;&#36;
\hat{x}<em>k^- = F\hat{x}</em>{k-1} + Bu_k
&#36;&#36;</p>
</li>
<li><p><strong>Deduce uncertainty:</strong></p>
<p>&#36;&#36;
P_k^- = FP_{k-1}F^T + Q
&#36;&#36;</p>
</li>
</ol>
<p><strong>Step B: Update (Measurement Update) - See Sensors &#36;z_k&#36; After that, the projections are revised with observations.</strong></p>
<ol start="3">
<li><p><strong>Calman Gain Calculated &#36;K&#36;：</strong></p>
<p>&#36;&#36;
K_k = \frac{P_k^- H^T}{H P_k^- H^T + R}
&#36;&#36;</p>
<ul>
<li>This is the most critical formula in the Kalman filter.</li>
<li>Intuitive understanding:&#36;K \approx \frac{\text{误差&#125;&#125;{\text{误差} + R}&#36;。</li>
<li>If the sensor noises &#36;R \to 0&#36;, and &#36;K \to 1&#36; (fully believed sensors)</li>
<li>If you're expecting a mistake, &#36;P \to 0&#36;, and &#36;K \to 0&#36; (Full letter forecast)</li>
</ul>
</li>
<li><p><strong>Status of amendment (final result):</strong></p>
<p>&#36;&#36;
\hat{x}_k = \hat{x}_k^- + K_k(z_k - H\hat{x}_k^-)
&#36;&#36;</p>
<ul>
<li>Final estimate = projection + &#36;K \times&#36; (Measurements - predicted measurements).</li>
<li>The part in brackets is called<strong>Innovation</strong></li>
</ul>
</li>
<li><p><strong>Amending uncertainty:</strong></p>
<p>&#36;&#36;
P_k = (I - K_kH)P_k^-
&#36;&#36;</p>
</li>
</ol>
<h3>What do you get when you're supposed to be?</h3>
<p>The filters keep up to date with state and uncertainty following the "Prognation-Observation-Amended" cycle:</p>
<ol>
<li><p><strong>Integration models and observations:</strong>
Final estimate when models, noise assumptions and parameters are set reasonably &#36;\hat{x}_k&#36; It is usually more stable than readings using only physical projections or single sensors.</p>
</li>
<li><p><strong>Depress the part measuring noise:</strong>
If the GPS signal suddenly changes, the Kalman filter will be caused by &#36;R&#36;(sensor noise) Large or &#36;P&#36;(Prodict error) is smaller and chooses not to fully believe in the jump, thus drawing a smooth track.</p>
</li>
<li><p><strong>Tracking uncertainty:</strong>
Coordinated matrix under observation and stability of parameters, etc. &#36;P_k&#36; There is a potential for gradual stabilization. The deviations may also be distorted or dispersed when models are not matched or noise settings are not set.</p>
</li>
</ol>
<p>Intuitively, physical models provide continuous predictions, and sensors are responsible for correcting prediction deviations, with relative weights of both being subject to uncertainty.</p>
<h3>Engineering practice</h3>
<p>In the code, the matrix. &#36;F&#36; and &#36;H&#36; It is usually determined by the laws of physics and is fixed. But... &#36;Q&#36; and &#36;R&#36; It's the parameters you need to "tweak."</p>
<p>&#36;R&#36;（<strong>Measuring noise association differences</strong>: can be estimated on sensor specifications, static measurements or repeated experiments.</p>
<p>&#36;Q&#36;（<strong>Process Noise Argument</strong>): Disturbing without coverage in the description status model. The greater the physical or external interference that is missing from the model, the greater the need for it to be set up and repeatedly checked through the residual and physical tracks.</p>
<p>&#36;x_0&#36;（<strong>Initial Status</strong>: First measurement values can be used or a priori tested. The initialization deviations are modified depending on the system ' s detectability and the quality of subsequent observations.</p>
<p>&#36;P_0&#36;（<strong>Initial difference</strong>: If yes &#36;x_0&#36; Lack of assurance allows for a larger initial co-conformation to allow filters to refer more to observations in early updates. Values still need to match the status profile and actual error range.</p>
<h3>Summary</h3>
<p>The regression form of the Kalman filter is suitable for real-time state estimates, but the standard version relies on linear models and noise assumptions. When the real system is clearly non-linear, variants such as extended Kalman filters are required.</p>
<p>The real world is often non-linear (e.g. robots do not go straight, but turn). And then we need to introduce it.<strong>Expand EKF</strong>。</p>
<h2>Progress: Expand the Kalman filter (EKF)</h2>
<p>EKF is the non-linear state estimation method commonly used in the Karman filter family, and is widely used for robotics, navigation and sensor integration.</p>
<h3>Why EkF?</h3>
<p>Standard KF requirement state transfer and observation models are linear. The actual system often includes non-linear relationships, such as the angles in robotic motion. &#36;\sin/\cos&#36;and the square root of radar range &#36;\sqrt{}&#36;。</p>
<p>After a change in the distribution of Goss through non-linear functions, the strict Goss shape is usually not maintained. Directly applies the standard KF linear dissemination formula, which introduces an indescribable approximation error.</p>
<p>EKF performs a linear first-orderization of non-linear functions near the current estimate point. Similar to the use of a plane near the Earth ' s surface for a sufficient time range, it requires only local approximation to describe changes in the immediate state.</p>
<h3>Mathematical mechanisms (linearization and the Yacca matrix)</h3>
<p>In standard KF, we assume &#36;x_k = Fx_{k-1}&#36;I'm sorry. However, in the EKF, state transfer and observation became non-linear functions:</p>
<p>&#36;&#36;
\begin{aligned}
x_k &amp;= f(x_{k-1}, u_k) + w_k \
z_k &amp;= h(x_k) + v_k
\end{aligned}
&#36;&#36;</p>
<ul>
<li><strong>When calculating status (average)</strong>: We can immediately add the estimate of the previous step to the non-linear function &#36;f(\cdot)&#36;It's okay.</li>
<li><strong>Count Uncertainty (Assisting P)</strong>: The ACSM cannot " substitute " directly for non-linear functions. We can't just count. &#36;P_k = f(P_{k-1})&#36;。</li>
</ul>
<p>To update the differences. &#36;P&#36;We have to find one.<strong>Linear Matrix</strong>Approximately represents the "distortion level" of the non-linear function at the current point. This matrix is...<strong>Jacobian Matrix</strong>。</p>
<p>EKF will set a fixed matrix in the standard KF &#36;F&#36; and &#36;H&#36; Replace the acoustical matrix that changes with the changing state of the state:</p>
<ul>
<li>&#36;F_k = \frac{\partial f}{\partial x} \mid_{\hat{x}_{k-1&#125;&#125;&#36; (local slope of the status shift function)</li>
<li>&#36;H_k = \frac{\partial h}{\partial x} \mid_{\hat{x}_k^-}&#36; (local slope of observation function)</li>
</ul>
<p>The EKF projection process can be written as follows:</p>
<ol>
<li><strong>Status prediction (retain non-linear functions):</strong>
&#36;&#36; \hat{x}<em>k^- = f(\hat{x}</em>{k-1}, u_k) &#36;&#36;</li>
<li><strong>Accompanying predictions (with the Yacca matrix, approximation):</strong>
&#36;&#36; P_k^- = F_k P_{k-1} F_k^T + Q &#36;&#36;</li>
</ol>
<p>It's like: I'm going to go around the curve, but I'm going to go around the cut-off line to estimate my margin of error.</p>
<h3>Engineering practices and challenges</h3>
<p>When's the EKF?</p>
<ul>
<li>Non-linear, with significant local linear error: UKF or particle filters (PF) can be considered.</li>
<li>When the Yacima matrix is difficult to extrapolate or maintain: UKF, automatic calibration or numerical method may be considered.</li>
<li>When the state distribution is clearly multiple peaks: particle filters are usually more appropriate.</li>
</ul>
<p>The UKF avoids the visible and altruistic matrix and reduces the costs of extrapolation and maintenance in some non-linear systems, but the amount calculated, the parameter settings and the numerical stability still need to be assessed separately.</p>
<h2>Step 2: Unscrew Kalman Filter (UKF)</h2>
<p>When the Arby matrix is difficult to extrapolate or local linear precision is insufficient, the unscrutinized Kalman filter (UKF) can be considered.</p>
<h3>Intuitive Understanding (The Intuition)</h3>
<p>Remember the problem we had in the EkF? ♪ When one ♪<strong>Goss distribution</strong>Through one.<strong>Non-linear functions</strong>At times, it often emerges in shape that is no longer a standard elliptical, but that may become a curved “banana”.</p>
<p><strong>EKF approach:</strong> Use a first-order barometer near the average, which is almost non-linear, and therefore accuracy depends on linear effects near the current point.</p>
<p><strong>UKF approach:</strong> Julier and Uhlmann suggest that instead of linearizing non-linear functions directly, a group of determinative sampling points is chosen to approximate post-functional changes in the distribution of the state.</p>
<p>UKF Select a group from the current status distribution <strong>Sigma dots</strong>, and then reestimate the average and the co-conforming difference by the changed point.</p>
<h3>Mathematical: Unhappinessed Transform</h3>
<p>This process is called <strong>Unscented Transform, UT</strong>I'm sorry. It uses a distribution of determinative samples that is similar to non-linear variations.</p>
<p>But unlike Monte Carlo, the UKF uses one of the most random samples.<strong>Specimen Sample (Deterministic Sampling)</strong>。</p>
<p><strong>The following concrete steps are taken:</strong></p>
<h4>1. Select Sigma Points (Sigma Points Semenation)</h4>
<p>Assuming your state vector. &#36;x&#36; Yes. &#36;n&#36; V. The UKF will be symmetrical around the average Choose &#36;2n+1&#36; Point.</p>
<ul>
<li><p><strong>Centre:</strong> &#36;\mathcal{X}_0 = \mu&#36; (current average)</p>
</li>
<li><p><strong>Around:</strong> &#36;\mathcal{X}_i = \mu \pm (\sqrt{(n+\lambda)P})_i&#36;</p>
<ul>
<li>Here. &#36;\sqrt{P}&#36; It is the square root of the co-ordinated matrix (usually obtained through Cholesky decomposition).</li>
<li>&#36;\lambda&#36; is the scaling parameters, how far are the controls on these points from the centre?</li>
</ul>
</li>
</ul>
<h4>2. Non-linear transmission</h4>
<p>This step does not require a dramatic extrapolation of the Arbya Matrix, which directly points Sigma &#36;\mathcal{X}_i&#36; Substitute non-linear physical equations &#36;f(\cdot)&#36;：</p>
<p>&#36;&#36; \mathcal{Y}_i = f(\mathcal{X}_i) &#36;&#36;</p>
<ul>
<li><strong>Advantages:</strong> No one-step linearization of the visible. However, when functions are inconsistent, jump or value instability, the approximation of UKF may still be invalid and subject to experimental examination.</li>
</ul>
<h4>Reorganization distribution</h4>
<p>Now we got a set of converted points. &#36;\mathcal{Y}_i&#36;I'm sorry. How do you get back to Goss distribution?
<strong>Weighted average!</strong></p>
<ul>
<li><strong>New average:</strong> &#36;\hat{y} = \sum_{i=0}^{2n} W_i^{(m)} \mathcal{Y}_i&#36;</li>
<li><strong>New Association difference:</strong> &#36;P_y = \sum_{i=0}^{2n} W_i^{(c)} (\mathcal{Y}_i - \hat{y})(\mathcal{Y}_i - \hat{y})^T&#36;</li>
</ul>
<p>Here. &#36;W_i&#36; The fixed weights are calculated on the basis of the distance.</p>
<h4>4. Complete UKF closed loops</h4>
<p>With the predicted averages and the difference, UKF still uses the Kalman update:
&#36;&#36; K = P_{xy} P_{yy}^{-1} &#36;&#36;
&#36;&#36; \hat{x} = \hat{x}^- + K(z - \hat{z}) &#36;&#36;</p>
<p>The point is, the difference between the two is that &#36;P_{xy}&#36; It's also calculated by Sigma's weighting, which completely avoids the Yacca Matrix. &#36;H&#36; .</p>
<h3>Summary</h3>
<p>UKF does not need to calculate the autonometric matrix, which is easier to achieve than EKF in some non-linear issues and may be more similar to it. It is not a generic replacement for EKF: state dimensions, calculation of budgets, model structure and realization will affect choices.</p>
<p>KF, EKF and UKF all need to maintain their differences and perform matrix operations. When the state dimension is large, storage and costing increases rapidly and numerical stability is more likely to be encountered. EnKF is a sample collection of near-coherent differences that is suitable for handling certain high-dimensional systems.</p>
<h2>Step 3: Gather the Karman filter (EnKF)</h2>
<p>EnKF is mainly oriented towards situations with large state dimensions and direct maintenance of complete matrix costs, such as reaching dimensions &#36;10^6&#36; .</p>
<p>Standard KF requires storage and calculation of one &#36;n \times n&#36; . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . &#36;P&#36;。</p>
<ul>
<li>If &#36;n = 10^6&#36;,A matrix &#36;P&#36; Yes. &#36;10^{12}&#36; An element.</li>
<li>We need to save this matrix. <strong>8TB RAM</strong>。</li>
<li>Direct storage and operation of such dense matrices are not usually feasible and therefore require the use of structural, approximation or sample methods.</li>
</ul>
<p>EnKF Unobvious maintenance complete &#36;P&#36;, instead of a statistically approximate distribution and coordination difference of a group of samples.</p>
<p>EnKF no longer maintains that huge thing. &#36;P&#36; The matrix, but the one that's kept. <strong>"Convergence" (Ensemble)</strong>I'm sorry. Imagine, you would have drawn a perfect ellipse. Now you just have to put a little bit of paper on it. <strong>&#36;N&#36; A little bit.</strong>Like what? &#36;N=50&#36; or &#36;100&#36;I'm not sure. Just this. &#36;N&#36; The distribution of a dot is similar to that ellipse, so we can use it. &#36;N&#36; One point for the Goss distribution.</p>
<h3>There's only two steps of the cycle.</h3>
<p><strong>Step A: Forecast - Parallel Run Model</strong></p>
<p>This step is the simplest, and the strongest place for EnkF. There is no need for an acoustical matrix, no need for linearity. You just have to take this. &#36;N&#36; A sample. Each one of them is thrown into you.<strong>Non-linear physical model</strong> &#36;f(\cdot)&#36; Run a little bit.</p>
<ul>
<li><strong>Key points:</strong> I'll add a separate one for each sample.<strong>Random Process Noise</strong> &#36;\mathcal{w}_i&#36;I'm sorry. This is to prevent all samples from ending up at the same point (loss of diversity).</li>
<li>This step can be paralleled on GPU, very fast.</li>
</ul>
<p><strong>Step B: Analysis/update (Analysis) - Update the collection with sample statistics</strong></p>
<p>The forecast is made. &#36;N&#36; After a sample, the next step is to use observations. &#36;z&#36; Updates the whole set.</p>
<p>EnKF Use<strong>Sample statistics</strong>Replace the full matrix of the agreement &#36;P&#36;I'm sorry. The updating process consists of the following three steps:</p>
<ol>
<li><p><strong>Calman Gain Calculated &#36;K&#36;：</strong>
The standard formula is:&#36;K = P H^T (H P H^T + R)^{-1}&#36;I'm sorry.
But in EnkF, we...<strong>It doesn't exist. &#36;P&#36;</strong>I'm sorry. We'll calculate it directly from a sample. &#36;PH^T&#36; and &#36;HPH^T&#36; These two:</p>
<ul>
<li>&#36;PH^T \approx&#36; <strong>Status set</strong>and<strong>Forecast observation cluster</strong>The difference between the union.</li>
<li>&#36;HPH^T \approx&#36; <strong>Forecast observation cluster</strong>The self-contribution difference.</li>
<li>So we're avoiding it. &#36;10^6 \times 10^6&#36; The storage and calculation of large matrices involves only small arrays (usually the number of observations is much smaller than the state dimension).</li>
</ul>
</li>
<li><p><strong>Update each sample (disturbation observation):</strong>
To maintain statistical correctness, we cannot use only the same observations. Value &#36;z&#36; Go update all the samples. We need to generate. &#36;N&#36; individual<strong>Noise-bearing observations</strong>：</p>
<p>&#36;&#36; z_i = z + v_i, \quad v_i \sim N(0, R) &#36;&#36;</p>
<p>And then, for every sample, &#36;x_i&#36; Do Kalman update alone:</p>
<p>&#36;&#36; x_i^{new} = x_i + K (z_i - H x_i) &#36;&#36;</p>
</li>
<li><p><strong>Aggregation:</strong>
Update &#36;N&#36; Different. &#36;x_i^{new}&#36;I'm sorry. EnKF retains the whole set, instead of choosing a sample from it as an output.</p>
<ul>
<li>If you need a specific...<strong>Estimated value</strong>, the average of the pools is calculated:&#36;\hat{x} = \frac{1}{N} \sum x_i^{new}&#36;。</li>
<li>If you need to know,<strong>Uncertainty</strong>, calculate the difference of the pool:&#36;P = \text{Cov}(x^{new})&#36;。</li>
<li>This new collection will be used directly as input into the next step of the projection, and will be repeated.</li>
</ul>
</li>
</ol>
<h3>Engineering issues</h3>
<p>In high-dimensional applications such as WRF weather patterns and ocean circulation models, limited aggregations can lead to significant sampling errors. The actual system usually also needs to address two issues:</p>
<h4>Hypocrisy (Spurious Correlations)</h4>
<ul>
<li><strong>Reason:</strong> The key assumption for EnKF is to use &#36;N&#36; Samples (e.g. 50) to measure estimates &#36;10^6&#36; The coordination difference for the dimension. Statistics tell us that when the sample is too small, the relevant coefficients are calculated to be huge.<strong>Sample error</strong>。</li>
<li><strong>Performance:</strong> If we calculate the matrix of differences, we find some absurdity:<strong>The temperature in Brazil is actually 0.9 for Texas wind speed.</strong>I'm sorry. It's not technically reasonable, more like mathematical coincidence.</li>
<li><strong>Consequences:</strong> When the temperature in Brazil is observed, the filter may modify the wind speed of Texas to the wrong co-convene, so that the update can be transmitted to areas without physical connection.</li>
<li><strong>Solving: Localization</strong><ul>
<li><strong>Rationale:</strong> Introduction of physical commons — there is no correlation between too far-off states.</li>
<li><strong>Operation:</strong> Calman gain is being calculated. &#36;K&#36; , the calculated matrix of the agreement &#36;PH^T&#36; Light one.<strong>Distance weight matrix</strong>I'm sorry. The closer the weight is closer to 1 and the distance exceeds a certain radius (e.g. 500 km) and is directly set to 0. That cut off those remote and false links.</li>
</ul>
</li>
</ul>
<h4>Filter Diversion</h4>
<ul>
<li><strong>Reason:</strong> Models are always imperfect, and ours. &#36;N&#36; The individual samples are often drawn from the same observations during the iterative process, resulting in their increasing similarity (the differences are smaller and smaller).</li>
<li><strong>Performance:</strong><ol>
<li>The difference of the collection &#36;P&#36; Quick approaching 0.</li>
<li>Based on Formula &#36;K = PH^T(\dots)^{-1}&#36;Kalman gain. &#36;K&#36; It's close to zero.</li>
<li><strong>Results:</strong> The filter underestimates its own uncertainty, and the Kalman gain is close to zero, and new observations are difficult to correct the deviating state estimates.</li>
</ol>
</li>
<li><strong>Solve: Covariance Inflation</strong><ul>
<li><strong>Rationale:</strong> Since the difference is always small, it's artificial.<strong>Increase</strong>One, maintain sensitivity to new data.</li>
<li><strong>Operation:</strong> Every time we're done with the prediction, we'll take all the samples. &#36;x_i&#36; Deviation from average &#36;\bar{x}&#36; Part times one coefficient &#36; \lambda &gt; &#36;1 (e.g. 1.01):
&#36;&#36; x_i^{new} = \bar{x} + \lambda (x_i - \bar{x}) &#36;&#36;</li>
<li>It's like injecting a little uncertainty into the system, preventing it from being too early to be blindly confident.</li>
</ul>
</li>
</ul>
<h3>Summary</h3>
<p>The main advantage of EnKF is that it directly runs non-linear models, does not require the visible maintenance of the complete matrix of differences and that the group members can calculate in parallel. A small sample can make the estimation of a million-dimensional state possible, but the approximate quality depends on aggregate size, localization, complication variation and specific physical models.</p>
<p>The approach described above has one thing in common:<strong>Assuming a single-peak distribution (however, it ends with a mean and a variance).</strong></p>
<p>If the distribution of the state is clearly high, reliance on a single-peaker approach may not be able to retain different models. I'll think about it. <strong>Particle Filter</strong>, the distribution of the Beyers filtering in a form similar to the usual type of a gravitational particle.</p>
<h2>Appendix: Horizontal comparison of mainstream estimation algorithms</h2>
<table>
<thead>
<tr>
<th align="left">Methodology</th>
<th align="left">State dimensional (n)</th>
<th align="left">Distribution assumptions</th>
<th align="left">Why did you do that?</th>
<th align="left">Disadvantages</th>
</tr>
</thead>
<tbody><tr>
<td align="left"><strong>KF / EKF / UKF</strong></td>
<td align="left">Dimensions &#36;n&#36; The matrix size is determined. &#36;(n \times n)&#36;</td>
<td align="left">Gaussian</td>
<td align="left">If you save the average and the difference, calculate quickly, not only inverted, but also solves.</td>
<td align="left">It is impossible to deal with “multi-peak” situations (e.g., not knowing which one of the two similar rooms is in).</td>
</tr>
<tr>
<td align="left"><strong>EnKF (collecting Kalman)</strong></td>
<td align="left">Dimensions &#36;n&#36; It's big. Only a few samples. &#36;N&#36;</td>
<td align="left">It's like Goss.</td>
<td align="left">To solve it. &#36;n&#36; Too big to leave the matrix unattended.</td>
<td align="left">The distribution of the Nang Figos remains difficult to deal with.</td>
</tr>
<tr>
<td align="left"><strong>Particle Filter (PF)</strong></td>
<td align="left">Dimensions &#36;n&#36; Not too big.</td>
<td align="left">Any distribution (compared with particles)</td>
<td align="left">To address the complex situations of “Multi Peaks” and “Figos”.</td>
<td align="left">The size of the calculations is large and the dimensions are high enough for particles to be used (the shortage of particles).</td>
</tr>
</tbody></table>
