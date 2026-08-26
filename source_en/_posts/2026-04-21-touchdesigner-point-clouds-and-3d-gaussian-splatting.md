---
title: TouchDesigner Point Clouds and 3D Gaussian Splatting
title_zh: TouchDesigner 点云与 3D Gaussian Splatting
date: 2026-04-21 09:00:00 +0800
categories:
- Creative Media & Games
- Generative Media Tools
tags:
- Generative Media
author: Hyacehila
hidden: true
excerpt: The appeal of point clouds in TouchDesigner is that they can turn scanned reality, generative animation, and realtime
  systems into one visual language.
description: The appeal of point clouds in TouchDesigner is that they can turn scanned reality, generative animation, and
  realtime systems into one visual language.
excerpt_zh: TouchDesigner 处理点云的魅力，在于它能把扫描到的现实、生成式动画和实时系统揉成同一种视觉语言。
permalink: /blog/2026/04/21/touchdesigner-point-clouds-and-3d-gaussian-splatting/
lang: en
translation_key: 2026-04-21-touchdesigner-point-clouds-and-3d-gaussian-splatting
translation_status: machine
translation_source_hash: 3c3ada5a8c67754928636a5e8a13ba46940e2b82edbdffc973185fc22d28c1c4
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>A cloud is a very funny art form. It is like a re-opposing particle sculpture after the real world was scanned: far as it looks like fog, near as a moist, each occupying a few spots in space. It is not a model of integrity, smoothness, closure in the traditional sense, and therefore always carries a sense of instability; but it is also because of this instability that it is particularly suited to visual expression, especially to the illusions that want to retain both “real sense” and “digital sense”.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/06/18/comfyui-video-workflow-orchestration/">From ComfyUI to LibTV: What capacity should workstream programming grow in the age of video generation?</a>、<a href="/en/blog/2026/05/05/ai-agent-game-industry-pipeline/">How the game industry is introduced AI Agent</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>If Blender or Houdini are more like a three-dimensional object in fine-tunnel construction, then TouchDesigner makes me feel more like organizing a real-time visual system. It is concerned not only with what a model will look like in the end, but also with how it will be driven, distorted, and work with sound, sensors, time, weather and even in the field. A cloud fits the line: it's not the surface of death, it's a bunch of data that can be manipulated in real time.</p>
<h2>Touch Designer, what is it?</h2>
<p>From the definition of the official product page of Derivative, TouchDesigner is a <strong>visual development platform</strong>: It is a real-time, nodal, visual development tool suitable for interactive media systems, architectural projection, live music visualization and fast prototypes. It places the signal flow, the image stream and the geometric stream of the visual system in front of you, so that you can look at the results while you're in the system.</p>
<p>That's why it's particularly good for clouds. It's a Derivative. <code>Point Clouds</code> Directly mentioned in the document, the cloud in the TouchDesigner is often most effectively handled by placement in the GPU texture: a pixel can correspond to the location or properties of a point.<code>Point File In TOP</code> and <code>Point File Select TOP</code> It also directly imports common cloud files, tearing colours, positions and other properties into data layers that can continue to be processed. In TouchDesigner, the cloud is not a static model, but a cluster of data that can continue to be deformed, mixed, map and driven.</p>
<h2>What can I do in Touch Designer?</h2>
<p>This section does not begin with no nodes or process (and I am not familiar with it). The most attractive place for a cloud in Touch Designer is where it can turn models into something more like fog, constellations, building debris and laser sculptures. It is not necessarily complete, nor is it stable; it can blink, disperse, recombine, drift, even like a luminous mass driven by sound and space.</p>
<h3>1. Large space: cities become light particles</h3>
<p>Official case of Derivative <strong>Point Cloud Mastery from Think and Sense</strong> It's a good place to put it at first sight: real neighborhoods become a luminous space dust, like city scans and a huge digital map.</p>
<p><img src="https://derivative.ca/sites/default/files/styles/og_image/public/field/image/DSC01837.jpg" alt="The city of the curtain is a lit space dust.">
<em>The city on the screen is a cloud of lighted space.</em></p>
<p><img src="https://derivative.ca/sites/default/files/styles/content_colorbox/public/field/body-images/DSC01752.jpg" alt="The official case of Think and Sense: far from being a building, near being a floating spot, the image is naturally carrying a sense of “realism being dismantled”.">
<em>Far from being a building, near being a floating spot, the picture is naturally carrying a sense of “realism is being dismantled”.</em></p>
<h3>Dissolution and reorganization: the beginning of the melting of the boundary</h3>
<p>One of the best shows of the cloud is to put the picture between "to be shaped" and "to be disintegrated." The boundaries of traditional models are usually hard, while the boundaries of the clouds can naturally become density, noise and light spots.</p>
<p><img src="https://derivative.ca/sites/default/files/styles/content_colorbox/public/field/body-images/DSC01695.jpg" alt="Think and Sense official case: space is more like a light air when the construction boundary is broken up.">
<em>When the construction boundary is dispersed, space is more like a layer of light air.</em></p>
<p><img src="https://derivative.ca/sites/default/files/styles/content_colorbox/public/field/body-images/DSC01764.jpg" alt="The following is a case of the following: The level of the cloud, the brightness and the distance that it is associated with, the image is moving towards a stronger stage.">
<em>The level, brightness and proximity of the cloud will push the picture to a stronger stage.</em></p>
<h3>Transition and outbreak: images breathing</h3>
<p>The advantage of TouchDesigner is to move from static rendering to a live vision that will change continuously. The dots can be compact, dense, from contours to clouds, from space debris to abstract rays.</p>
<p><img src="https://derivative.ca/sites/default/files/styles/content_colorbox/public/field/body-images/rearity4.jpg" alt="Think and Sense official case: during the cloud transition, the image is sliding from one space to another.">
<em>When the clouds are passing, the images are sliding from one space to another.</em></p>
<p><img src="https://derivative.ca/sites/default/files/styles/content_colorbox/public/field/body-images/rearity2.jpg" alt="The same set of spatial materials can be pushed into a stronger sense of light and motion.">
<em>The same set of spatial materials can be pushed into a stronger sense of light and motion.</em></p>
<h3>4. Abstract sculpture: point as visual energy</h3>
<p>Of course, a cloud does not always come from a reality scan. It can be directly abstracted: like a nebula, like a sandstorm, like a digital sculpture that is being shaped by noise. It's not “what it represents”, it's the density, speed and light that it creates on the screen.</p>
<p><img src="https://derivative.ca/sites/default/files/styles/og_image/public/field/image/Generative%20Point%20Clouds%20in%20TouchDesigner.jpg" alt="Official ID: Generating a dot cloud is more like a cluster of digital nebulas that are condensing.">
<em>Generating dot clouds is more like a cluster of digital nebulas that are condensing.</em></p>
<p>It can turn the real world into a more flairful and unstable visual state: one second like a city, one second like a fog, one second like a group of stars that are bursting.</p>
<h2>Where did these clouds come from?</h2>
<p>If such projects are really going to be carried out, there are probably three main sources of material: self-modelling, photogrammetry, 3D Gaussian Splatting.</p>
<table>
<thead>
<tr>
<th>Route</th>
<th>Threshold</th>
<th>Realism.</th>
<th>Subsequent redactionability</th>
<th>What's the best for you?</th>
</tr>
</thead>
<tbody><tr>
<td>Model yourself.</td>
<td>Top, need to have a clear modelling capability</td>
<td>Controllable, but the real world is a manual detail.</td>
<td>The strongest, the clearest structure.</td>
<td>Projects that require strong control, a sense of design and follow-up to the production of animation and asset management</td>
</tr>
<tr>
<td>Photogrammetry</td>
<td>Medium, need to film the norms and reconstruction process</td>
<td>It's strong, especially for architecture, scenery, sculpture.</td>
<td>Medium, often clean, surface, change format</td>
<td>I want to bring real space into the system quickly, but still want to keep the traditional 3D pipe interfaces.</td>
</tr>
<tr>
<td>3DGS</td>
<td>One of the lowest, the collection threshold has dropped significantly.</td>
<td>Very strong, especially for the preservation of light, reflection and space. Round</td>
<td>Weakness, not for fine editing.</td>
<td>Atmospheric capture, spatial memory, rapid sampling of the real world, installation of visual materials</td>
</tr>
</tbody></table>
<p>The best way to model yourself is, of course. You know where every face, every level, every UV is, how it's convenient to change at a later stage. But its problems are also obvious: slow, and the occasional details of the real world are not very friendly. A lot of TouchDesigner's cloud-drived places are not the standard of geometry, but the tumultuous, but real space information in reality.</p>
<p>Photogrammetry is a more convenient route. The case of Think and Sense is typical: first, a drone image, then a photogrammetry software to restore a cloud and then to TouchDesigner. The advantage is that you still work in a more familiar three-dimensional re-engineering logic, and that what is exported last is easier to access to the existing graphic processes, but still at a cost.</p>
<p>And 3D Gaussian Spratting is more like another door that has been opened suddenly in recent years.<strong>The real world no longer has to be gridd before it can enter the real-time visual system.</strong> And 3D Gaussian Spratting is born close to the clouds.</p>
<p><img src="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/content/images/comparisons/ours_bicycle.png" alt="Inria Official 3D Gaussi Splatting Project page example: This expression is more suited to preserving volume, light and spatial continuity in the real scene.">
<em>Inria Official 3D Gaussi Splatting Project page example: This expression is more suited to preserving volume, light and spatial continuity in the real scene.</em></p>
<p>And more realistically, the 3DGS are not just about reading in research papers. The tool, represented by Scanivers, has put Gaussian Splatting in the mobile collection stream. Learn a full set of heavy 3D software before trying to move the real world into it, which is no longer the only entry point. You can capture a space and decide whether it ends up as a file, a browse, or a TouchDesigner, into a visual material on the ground.</p>
<h2>Why do I think 3DGS has a special potential?</h2>
<p>And the most interesting thing about 3DGS and TouchDesigner is not that it's more real, but that the real-world space suddenly becomes a material that can be manipulated in real time.</p>
<p>You can sweep a room, a street, a gallery, and get a big-ass spot with light, reflection, air sense and depth. It does not need to be transformed into a clean model, nor needs to be completely organized as traditional assets. It can enter the TouchDesigner as a piece of the real world, then be stretched, dispersed, rotated, scintillated, vibrated with music, or re-assembled into a space as the audience approaches.</p>
<p>It's funny: 3 DGS gives you a sense of reality (although not so precise), TouchDesigner gives you real-time performance skills. The former is responsible for taking the world down, while the latter is responsible for re-emerging, breaking and deforming the world. The last thing that comes out is not just a 3D scan, but a light material between memory, space and stage vision. The change in the cloud has instead captured the 3DGS shortboard: inaccurate, but fun enough.</p>
<p><img src="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/content/images/comparisons/ours_truck.png" alt="Inria, an example of the official project page: 3 DGS transforms the daily scene into a visual, walkable or rewriting space material.">
<em>3DGS transforms daily scenes into space materials that are visible, can be travelled and can be continued.</em></p>
<p>3DGS may not be able to be more like a real model, but...<strong>The government is not only collecting reality, breaking reality, but also performing reality in real time.</strong> That's where it's interesting to put with Touch Designer.</p>
<h2>Official case and video</h2>
<ul>
<li><a href="https://derivative.ca/community-post/point-cloud-mastery-think-and-sense/63851">Point Cloud Mastery from Think and Sense (Derivative Official Case)</a></li>
<li><a href="https://youtu.be/vG8scFzoGCA">Think and Sense Official Video</a></li>
<li><a href="https://derivative.ca/community-post/tutorial/generative-point-clouds-touchdesigner/67562">General Point Clubs in TouchDesigner (Derivative Official Curriculum)</a></li>
<li><a href="https://www.youtube.com/watch?v=__dHYGe9bQs">General Point Clubs in TouchDesigner</a></li>
<li><a href="https://www.youtube.com/watch?v=rlptcQpTMuo">Audio Reactive 3D Point Clubs in TouchDesigner</a></li>
<li><a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/">3D Gaussian Splatting for Real-Time Radice Field Rindering (Inria Official Project Page, with Video)</a></li>
</ul>
<h2>References</h2>
<ul>
<li>Derivative, <a href="https://derivative.ca/product">Created with TouchDesigner</a></li>
<li>Derivative Docs, <a href="https://docs.derivative.ca/">TouchDesigner Main Page</a></li>
<li>Derivative Docs, <a href="https://docs.derivative.ca/Point_Clouds">Point Clouds</a></li>
<li>Derivative Docs, <a href="https://docs.derivative.ca/POP">POP</a></li>
<li>Derivative, <a href="https://derivative.ca/community-post/point-cloud-mastery-think-and-sense/63851">Point Cloud Mastery from Think and Sense</a></li>
<li>Derivative, <a href="https://derivative.ca/community-post/tutorial/generative-point-clouds-touchdesigner/67562">Generative Point Clouds in TouchDesigner</a></li>
<li>Derivative, <a href="https://derivative.ca/community-post/tutorial/audio-reactive-3d-point-clouds-touchdesigner/68021">Audio Reactive 3D Point Clouds in TouchDesigner</a></li>
<li>Inria, <a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/">3D Gaussian Splatting for Real-Time Radiance Field Rendering</a></li>
<li>Scaniverse, <a href="https://scaniverse.com/">Home</a></li>
<li>Scaniverse, <a href="https://scaniverse.com/support">How to use Scaniverse 3D Scanner for iOS and Android</a></li>
</ul>
