---
title: "TouchDesigner 点云与 3D Gaussian Splatting"
title_en: "TouchDesigner Point Clouds and 3D Gaussian Splatting"
date: 2026-04-21 09:00:00 +0800
categories: ["Essays"]
tags: [Multimodality, Methodology]
author: Hyacehila
excerpt: TouchDesigner 处理点云的魅力，在于它能把扫描到的现实、生成式动画和实时系统揉成同一种视觉语言。
excerpt_en: "The appeal of point clouds in TouchDesigner is that they can turn scanned reality, generative animation, and realtime systems into one visual language."
permalink: '/blog/2026/04/21/touchdesigner-point-clouds-and-3d-gaussian-splatting/'
---

点云是个很好玩的艺术形式。它既像现实世界被扫描之后留下的残影，也像被重新抽象过的粒子雕塑：远看像雾，近看却是密密麻麻、各自占据空间的一批点。它不是传统意义上完整、光滑、封闭的模型，所以总带着一点不稳定的感觉；但也正因为这种不稳定，它特别适合被拿来做视觉表达，尤其适合那些想同时保留“真实感”和“数字感”的迷幻作品。

如果说 Blender 或 Houdini 更像是在精细地构造一个三维对象，那 TouchDesigner 给我的感觉更像是在组织一套实时视觉系统。它关心的不只是一个模型最后长什么样，还包括这个模型如何被驱动、如何被扭曲、如何跟声音、传感器、时间、天气甚至现场空间一起工作。点云刚好很适合这种思路：它不是死的表面，而是一堆可以被实时操纵的数据。

## TouchDesigner 到底是什么

从 Derivative 官方产品页的定义看，TouchDesigner 是一个 **visual development platform**：它是实时的、节点式的、偏视觉开发的工具，适合做互动媒体系统、建筑投影、现场音乐视觉以及快速原型。它把视觉系统的信号流、图像流和几何流都摆在你面前，让你能一边搭系统一边看结果。

这也是它特别适合点云的原因。Derivative 的 `Point Clouds` 文档里直接提到，TouchDesigner 里点云往往最有效的处理方式是放在 GPU 纹理里：一个像素就可以对应一个点的位置或属性。`Point File In TOP` 和 `Point File Select TOP` 还能直接导入常见点云文件，把颜色、位置和其他属性拆成可继续处理的数据层。在 TouchDesigner 里，点云不是一块静态模型，而是一团还能继续被变形、混合、映射和驱动的数据。

## 点云在 TouchDesigner 里到底能做什么

这一节先不讲节点，也不讲流程（我也不熟）。点云在 TouchDesigner 里最吸引人的地方，是它可以把模型变成一种更像雾、星群、建筑残影和激光雕塑的东西。它不一定要完整，也不一定要稳定；它可以闪烁、散开、重组、漂移，甚至像一团被声音和空间推着走的发光物质。

### 1. 巨量空间：城市变成发光颗粒

Derivative 官方案例 **Point Cloud Mastery from Think and Sense** 很适合放在这里当第一眼效果：现实街区变成一片会发光的空间尘埃，既像城市扫描，也像一场巨大的数字星图。

![Think and Sense 官方案例：巨幕上的城市点云像一片被点亮的空间尘埃。](https://derivative.ca/sites/default/files/styles/og_image/public/field/image/DSC01837.jpg)
*巨幕上的城市点云像一片被点亮的空间尘埃。*

![Think and Sense 官方案例：远看是建筑，近看是悬浮的点，画面天然带着一种“现实正在被拆解”的感觉。](https://derivative.ca/sites/default/files/styles/content_colorbox/public/field/body-images/DSC01752.jpg)
*远看是建筑，近看是悬浮的点，画面天然带着一种“现实正在被拆解”的感觉。*

### 2. 解体与重组：边界开始融化

点云最适合做的炫技之一，就是让画面处在“将要成形”和“正在消散”之间。传统模型的边界通常是硬的，点云的边界则天然可以变成密度、噪声和光斑。

![Think and Sense 官方案例：建筑边界被打散后，空间更像一层发光的空气。](https://derivative.ca/sites/default/files/styles/content_colorbox/public/field/body-images/DSC01695.jpg)
*建筑边界被打散后，空间更像一层发光的空气。*

![Think and Sense 官方案例：点云的层次、亮度和远近关系会把画面推向更强的舞台感。](https://derivative.ca/sites/default/files/styles/content_colorbox/public/field/body-images/DSC01764.jpg)
*点云的层次、亮度和远近关系会把画面推向更强的舞台感。*

### 3. 过渡与爆发：画面像在呼吸

TouchDesigner 的长处，是把这类画面从静态渲染图变成一段会持续变化的现场视觉。点可以从稀疏变密集，从轮廓变成云，从空间残影变成抽象光团。

![Think and Sense 官方案例：点云过渡时，画面像在从一个空间滑入另一个空间。](https://derivative.ca/sites/default/files/styles/content_colorbox/public/field/body-images/rearity4.jpg)
*点云过渡时，画面像在从一个空间滑入另一个空间。*

![Think and Sense 官方案例：同一套空间素材可以被推成更强烈的光流和运动感。](https://derivative.ca/sites/default/files/styles/content_colorbox/public/field/body-images/rearity2.jpg)
*同一套空间素材可以被推成更强烈的光流和运动感。*

### 4. 抽象雕塑：把点当成视觉能量

当然，点云也不必总是来自现实扫描。它可以直接变成抽象视觉：像星云、像沙暴、像一个正在被噪声塑形的数字雕塑。这里吸引人的不是“它代表什么物体”，而是它在屏幕上形成的密度、速度和光感。

![Derivative 官方效果图：生成式点云更像一团正在凝结的数字星云。](https://derivative.ca/sites/default/files/styles/og_image/public/field/image/Generative%20Point%20Clouds%20in%20TouchDesigner.jpg)
*生成式点云更像一团正在凝结的数字星云。*

它能把现实世界变成一种更炫、更不稳定的视觉状态：一秒钟像城市，一秒钟像雾，一秒钟又像正在爆开的星群。

## 这些点云素材从哪里来

如果真的要做这类项目，素材来源大概有三条主路：自己建模、摄影测量、3D Gaussian Splatting。

| 路线 | 门槛 | 真实感 | 后续可编辑性 | 更适合什么项目 |
| --- | --- | --- | --- | --- |
| 自己建模 | 最高，需要明确建模能力 | 可控，但真实世界细节要靠手工补 | 最强，结构最清晰 | 需要强控制、强设计感、后续还要继续制作动画和资产管理的项目 |
| 摄影测量 | 中等，需要拍摄规范和重建流程 | 很强，尤其适合建筑、场景、雕塑 | 中等，常常要清理、减面、转格式 | 想快速把现实空间带进系统，但仍希望保留传统 3D 管线接口的项目 |
| 3DGS | 最低之一，采集门槛已经显著下降 | 很强，尤其擅长保留光照、反射和空间氛围 | 偏弱，不是为精细编辑而生 | 氛围捕捉、空间记忆、快速采样现实世界、装置视觉原料 |

自己建模当然最稳。你知道每个面、每个层级、每个 UV 在哪里，后期想怎么改都方便。但它的问题也很明显：慢，而且对现实世界的偶然细节不够友好。不少 TouchDesigner 点云作品打动人的地方，恰恰不是几何多么标准，而是现实场景里那些杂乱但真实的空间信息。

摄影测量是一条更折中的路。Think and Sense 的案例就是典型做法：先拍无人机影像，再用摄影测量软件恢复出点云，最后送到 TouchDesigner。它的好处是，你依然在比较熟悉的三维场景重建逻辑里工作，最后输出的东西也比较容易进入既有图形流程，只是成本依旧不低。

而 3D Gaussian Splatting 更像是最近几年突然打开的另一扇门。**现实世界不再必须先被网格化之后才能进入实时视觉系统。** 并且 3D Gaussian Splatting 天生就很接近点云。

![Inria 官方 3D Gaussian Splatting 项目页示例图：这种表示方式更擅长保留真实场景中的体积感、光照感和空间连续性。](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/content/images/comparisons/ours_bicycle.png)
*Inria 官方 3D Gaussian Splatting 项目页示例图：这种表示方式更擅长保留真实场景中的体积感、光照感和空间连续性。*

更现实的一点是，3DGS 已经不是只能在研究论文里看的东西了。以 Scaniverse 为代表的工具，已经把 Gaussian Splatting 放进移动端采集工作流里。先去学一整套重型三维软件，再想办法把现实世界搬进去，不再是唯一入口。你可以先把一个空间捕捉下来，再决定它最后是做档案、做浏览，还是进入 TouchDesigner 变成现场视觉素材。

## 为什么我觉得 3DGS 特别有潜力

3DGS 和 TouchDesigner 放在一起，最有趣的地方不是更真实，而是现实空间突然变成了一种可以被实时操纵的材料。

你可以扫一个房间、一条街、一个展厅，得到一团带着光照、反射、空气感和深度的高斯点。它不需要先变成干净的模型，也不必像传统资产那样被完全整理好。它可以先作为真实世界的碎片进入 TouchDesigner，然后被拉伸、打散、旋转、闪烁、随音乐抖动，或者在观众靠近时重新聚成一个空间。

这就很好玩：3DGS 给了你现实的质感（虽然没那么精确），TouchDesigner 给了你实时表演的能力。前者负责把世界采下来，后者负责把世界重新点亮、拆开、变形。最后得到的东西不只是一个三维扫描，而是一种介于记忆、空间和舞台视觉之间的发光材料。点云的变换反而把 3DGS 的短板收住了：不精确，但足够好玩。

![Inria 官方项目页示例图：3DGS 把日常场景变成可观看、可游走、也可继续改写的空间材料。](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/content/images/comparisons/ours_truck.png)
*3DGS 把日常场景变成可观看、可游走、也可继续改写的空间材料。*

3DGS 可能做不到更像真的模型，但是**采集现实、打碎现实、实时表演现实。** 这正是它和 TouchDesigner 放在一起有意思的地方。

## 官方案例与视频

- [Point Cloud Mastery from Think and Sense（Derivative 官方案例）](https://derivative.ca/community-post/point-cloud-mastery-think-and-sense/63851)
- [Think and Sense 官方视频](https://youtu.be/vG8scFzoGCA)
- [Generative Point Clouds in TouchDesigner（Derivative 官方教程）](https://derivative.ca/community-post/tutorial/generative-point-clouds-touchdesigner/67562)
- [Generative Point Clouds in TouchDesigner 视频](https://www.youtube.com/watch?v=__dHYGe9bQs)
- [Audio Reactive 3D Point Clouds in TouchDesigner 视频](https://www.youtube.com/watch?v=rlptcQpTMuo)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering（Inria 官方项目页，含视频）](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

## 参考资料

- Derivative, [Created with TouchDesigner](https://derivative.ca/product)
- Derivative Docs, [TouchDesigner Main Page](https://docs.derivative.ca/)
- Derivative Docs, [Point Clouds](https://docs.derivative.ca/Point_Clouds)
- Derivative Docs, [POP](https://docs.derivative.ca/POP)
- Derivative, [Point Cloud Mastery from Think and Sense](https://derivative.ca/community-post/point-cloud-mastery-think-and-sense/63851)
- Derivative, [Generative Point Clouds in TouchDesigner](https://derivative.ca/community-post/tutorial/generative-point-clouds-touchdesigner/67562)
- Derivative, [Audio Reactive 3D Point Clouds in TouchDesigner](https://derivative.ca/community-post/tutorial/audio-reactive-3d-point-clouds-touchdesigner/68021)
- Inria, [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- Scaniverse, [Home](https://scaniverse.com/)
- Scaniverse, [How to use Scaniverse 3D Scanner for iOS and Android](https://scaniverse.com/support)
