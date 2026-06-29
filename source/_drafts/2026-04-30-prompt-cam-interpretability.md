---
title: "从可解释性看 Prompt-CAM：让 ViT 看见细粒度特征"
date: 2026-04-30 21:30:00 +0800
categories: [深度学习]
tags: [Interpretability, Computer Vision, Vision Transformer, Prompt-CAM, FGVC]
author: Hyacehila
excerpt: "Prompt-CAM 把类别提示、分类依据和注意力热图绑定在同一套 ViT 结构中，为细粒度视觉分类提供了一种轻量而直接的可解释性路径。"
---

# 从可解释性看 Prompt-CAM：让 ViT 看见细粒度特征

在计算机视觉中，Interpretability 研究关注的不只是模型“预测了什么”，更关注模型“凭什么做出预测”。尤其是在医学影像、生态物种识别、工业质检等高风险或高精度场景中，一个分类结果如果无法说明依据，就很难被研究者、医生或业务系统真正信任。随着 Vision Transformer 成为视觉大模型的重要架构，如何解释 ViT 的内部决策，也成为可解释性研究中的关键问题。

[Prompt-CAM](https://openaccess.thecvf.com/content/CVPR2025/html/Chowdhury_Prompt-CAM_Making_Vision_Transformers_Interpretable_for_Fine-Grained_Analysis_CVPR_2025_paper.html)，即 Prompt Class Attention Map，来自 CVPR 2025 论文《Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis》。它面向预训练 ViT，重点解决细粒度视觉分类 FGVC 中的解释问题，例如区分外观相近的鸟类、鱼类、昆虫、真菌、汽车或食物类别。此类任务的难点在于，真正决定类别的往往不是整体轮廓，而是局部 trait：一小块羽毛颜色、斑纹形状、鳍部结构，或车辆某个细节部件。

## 传统 CAM 方法的局限

Grad-CAM 等 Class Activation Mapping 方法长期被用来生成热力图，帮助研究者观察模型关注图像的哪些区域。但在预训练 ViT 上，这类方法常出现热图粗糙的问题：它们可以高亮整只鸟、整辆车或整个目标物体，却未必能定位真正区分类别的细微特征。对于普通分类任务，这种粗粒度解释或许已经够用；但对于 FGVC，它会掩盖最重要的信息，因为细粒度识别本质上依赖“相似类别之间的差异”。

Prompt-CAM 的出发点正是这一痛点：如果 ViT 本身已经能提取局部、判别性的 patch 表征，那么可解释性方法不应只在模型预测后做事后分析，而应把解释机制嵌入分类过程本身。

## Prompt-CAM 的方法逻辑

Prompt-CAM 借鉴了 Visual Prompt Tuning 的思想，但目标不是单纯提升分类性能，而是让 prompt 同时承担类别预测和特征定位的角色。具体来说，假设数据集有 C 个类别，Prompt-CAM 会为每个类别学习一个 Class-Specific Prompt，并将这些 prompts 与图像 patch tokens 一起送入冻结的预训练 ViT。

训练时，模型不更新 ViT backbone，而是主要学习这些类别提示及预测头。每个 class prompt 的输出用于对应类别的分类判断。为了正确分类，真实类别对应的 prompt 会被训练目标推动去关注那些能区分该类别与其他类别的图像 patches，也就是具有判别力的 traits。推理时，研究者提取真实类别 prompt 的 multi-head attention maps，就可以得到更清晰的 Trait Localization 结果。

这个设计的关键之处在于，解释不再完全依赖梯度反传后的事后可视化，而是来自分类机制内部的注意力交互。换言之，Prompt-CAM 将“类别提示”“分类依据”和“注意力热图”绑定到同一套结构中，使解释结果更贴近模型完成细粒度判断时实际使用的局部证据。

## 学术贡献与工程价值

Prompt-CAM 的价值首先在于简洁。[论文](https://arxiv.org/abs/2501.09333)将其描述为接近 free lunch 的方法，因为它基本只需要修改 Visual Prompt Tuning 的 prediction head，不要求重新设计复杂的解释网络，也不需要全面微调大型 ViT。对于计算资源有限的实验室，这一点非常重要：冻结 backbone、学习少量 prompts，通常比全参数微调更经济，也更容易复现实验。

其次，它与预训练视觉模型具有较好的兼容性。对于 DINO、DINOv2、BioCLIP 等具备强局部表征能力的 ViT，Prompt-CAM 可以作为一种轻量解释层，帮助研究者观察模型是否真正利用了目标 trait，而不是背景、拍摄角度或数据集偏差。论文实验也覆盖了多个细粒度领域，说明该思路并非只适用于单一数据集。

当然，Prompt-CAM 不应被理解为解决所有 ViT 可解释性问题的最终方案。注意力图本身仍需谨慎解读，解释质量也会受到预训练模型、数据分布和类别定义方式的影响。但它提供了一个很有启发性的方向：可解释性不一定只能在模型之后附加，也可以通过任务结构设计被自然引入模型内部。

对科研实践而言，Prompt-CAM 很适合作为细粒度图像分析、医学影像诊断、生态物种识别和多模态视觉对齐中的可解释性 baseline。进一步看，Class-Specific Prompts 的思想也可以扩展为一种条件探针：让模型在特定类别、属性或语义条件下暴露其关注证据，从而为视觉大模型和多模态大模型的解释研究提供更轻量、可控的实验路径。
