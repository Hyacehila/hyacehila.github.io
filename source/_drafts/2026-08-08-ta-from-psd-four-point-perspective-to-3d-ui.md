# 从 PSD 四点透视到经典 3D UI：TA 需要研究的技术路线

技术美术（Technical Artist，TA）不是“会一点代码的美术”，也不是“帮美术导资源的程序”；它的职责是在视觉目标、内容生产流程和运行时限制之间建立可复用的技术方案。TA 会把设计师在 Photoshop、Figma 或 DCC 软件里做出的效果，拆成引擎可渲染、可交互、可维护且性能可接受的资产与规则。本文讨论的“四角透视卡片”最接近 **UI TA、渲染/材质 TA、工具与管线 TA** 的交叉工作：前者负责 UI 的可编辑性和还原度，后者负责 3D/材质表达与 PSD 导入工具。角色绑定 TA、纯特效 TA 或环境 TA 不一定会直接实现它；如果页面是静态宣传图、没有运行时交互，普通 UI 美术把效果烘焙为图片即可，未必需要专门投入 TA 工作。

## 问题从哪里来

在 `打卡活动2.psb` 中，中间悬挂的活动卡片是智能对象。卡片 `1`、`3`、`4` 的关键数据并不是一个 `rotation` 数值，而是最终四个角点：`PlLd.transform` / `SoLd.nonAffineTransform`。其中上下边的倾角不同，说明它们不是“旋转后的矩形”，而是一个投影到任意四边形上的平面。该文件的 warp 参数为零，因此这里也不是网格弯曲；更准确的名称是 **四点透视（projective transform / homography）** 或 Photoshop 的自由变换/透视扭曲结果。

这解释了一个常见误区：从顶边 `atan2` 算出一个角度，最多只能得到“顶边朝向”，不能称作 PSD 原始旋转。只有当四边形满足平行边、近似正交等条件时，才可能退化为普通的 `position + scale + rotationZ`。若只支持平面位置、缩放和单一旋转，经典 UI 编辑器无法精确复现这三张卡片。

## 把 2D 与 3D 放进同一套模型

| 变换 | 屏幕结果 | 是否足以表达样本卡片 |
| --- | --- | --- |
| SRT | 矩形仍为矩形，只能平移、缩放、平面旋转 | 否 |
| Affine | 可加入 shear，矩形变平行四边形 | 否，仍要求对边平行 |
| Projective / Homography | 矩形可变为任意凸四边形，允许消失点 | 是 |
| Mesh warp | 多顶点、可弯曲和局部变形 | 当前样本不需要 |

四点投影并不只是“用 2D 假装 3D”。一张真实 3D 平面经过透视相机投影，在固定视角下就是一个二维单应变换；因此“3D quad + 透视相机”和“2D 四点映射”可以得到相同的画面。区别是前者保留了相机、深度、旋转轴、遮挡和动画能力，后者只保存最后的屏幕结果。PSD 往往属于视觉结果优先：它不保证保存了可唯一还原的相机参数，所以不能从四角反推出唯一的 `rotationX/Y/Z`、距离和 FOV。若引擎相机参数已知，可以用四角最小化拟合求一个可用的 3D 姿态；若参数未知，应把拟合结果称为“引擎内近似解”，而不是 PSD 的原始姿态。

## 不同平台、软件与团队习惯

- **Photoshop / 设计工作流**：设计师倾向先追求单帧视觉效果，智能对象、自由变换、透视和合成图层很高效。对 TA 而言，PSD 是“设计意图和视觉证据”，不是天然的运行时场景描述；导入时必须判断哪些内容保留为节点，哪些内容应烘焙。
- **Unity**：屏幕空间 Canvas 适合常规 UI；World Space Canvas 放进 3D 场景后，才能受透视相机、深度和旋转影响。要验证的不是 Inspector 是否存在 Rotation，而是是否有 `rotationX/Y`、透视相机和正确的 UI 渲染路径。
- **Unreal**：普通 UMG 的 Render Transform 包含平移、缩放、shear 和 angle，属于二维 affine 思路；Widget Component 则能把 Widget 放进世界空间。前者适合业务 UI，后者更适合悬浮卡、屏幕、展台等 2.5D 表现。
- **Cocos / 自研经典 3D UI**：重点核验 Canvas/RenderRoot 与 Camera 的关系、相机是正交还是透视、UI 是否能渲染到纹理、节点是否支持完整三轴旋转。仅有 `position + scale + rotation` 这一组字段还不够，必须确认 rotation 的轴和最终投影。
- **网页**：CSS `perspective`、`rotateX/Y/Z` 适合带语义和动态文字的 3D 卡片；`matrix3d()` 可表达更一般的矩阵；需要逐像素贴合 PSD 四角、遮挡或复杂网格时，通常应使用 WebGL/WebGPU，将内容渲染为纹理再贴到 quad/mesh 上。

## 后续调研与实验清单

1. **先做能力矩阵，而不是先写导入器。** 记录编辑器是否支持 `rotationX/Y/Z`、`positionZ`、透视相机、RenderTexture、自定义材质、四点网格、裁切与命中测试。将“字段存在”和“渲染管线真正支持”分开验证。
2. **构建最小测试场景。** 用一张 `402 × 732` 的卡片，分别测试 SRT、shear、3D 平面加透视相机和四点 mesh；与 PSB 的四个角点比对，记录最大像素误差。这会直接回答引擎能否拟合卡片 `1/3/4`。
3. **定义中间数据，而非只导出 rotation。** 建议输出 `transform_kind`（`srt`/`affine`/`projective`/`mesh`）、`source_size`、`corners`、`fit_residual` 和 `fallback`。`projective` 在当前引擎不可表达时，应明确降级为 `bake`，而不是伪造一个旋转角。
4. **比较三条落地路线。** 静态卡片用已投影的透明 PNG；需要运动或视差的卡片用 3D quad + 姿态拟合；既要动态子 UI 又要精确四点效果时，用“子 UI → RenderTexture → projective material”。最后一种最灵活，但会增加渲染、分辨率、层级与交互成本。
5. **把质量与生产成本一起验收。** 除视觉误差外，还要测试多分辨率、字体和动态文案、点击区域、遮挡排序、批次/Draw Call、纹理清晰度、资产更新成本，以及设计师改稿后能否稳定复现。TA 的价值不在一次性把图做像，而在于把这种决策变成团队可重复使用的规则和工具。

## 当前判断

这个题目值得作为 UI/渲染方向 TA 的研究项目，但研究目标不应是“从 PSD 读出旋转”，而应是建立一套 **设计稿变换分类、引擎能力匹配、四角拟合、可靠降级和视觉验收** 的流程。若最终发现当前编辑器只支持二维 SRT，那么烘焙不是失败，而是经过技术判断后的正确生产策略；如果编辑器有完整三维平面和透视相机，则可以把四角数据作为拟合目标，而非把 PSD 当成唯一的三维真相。

## 参考资料

- [Unity：Technical Artists 的工具与工作流概览](https://unity.com/blog/engine-platform/complete-overview-of-unity-toolsets-workflows-for-technical-artists)
- [Adobe Photoshop：定义平面以调整透视](https://helpx.adobe.com/photoshop/desktop/repair-retouch/clean-restore-images/define-planes-to-adjust-perspective.html)
- [OpenCV：Homography 教程](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)
- [Unity Manual：Canvas](https://docs.unity3d.com/2022.3/Documentation/Manual/UICanvas.html)
- [Unreal Engine：Widget Components](https://dev.epicgames.com/documentation/en-us/unreal-engine/widget-components-in-unreal-engine)
- [Cocos Creator：Canvas](https://docs.cocos.com/creator/3.8/manual/en/ui-system/components/editor/canvas.html)
- [Cocos Creator：Camera](https://docs.cocos.com/creator/3.8/manual/en/renderer/camera.html)
- [W3C：CSS Transforms Level 2](https://www.w3.org/TR/css-transforms-2/)
- [MDN：`matrix3d()`](https://developer.mozilla.org/en-US/docs/Web/CSS/transform-function/matrix3d)
- [MDN：WebGL shader 教程](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Tutorial/Using_shaders_to_apply_color_in_the_webgl_canvas)
