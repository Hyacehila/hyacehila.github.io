---
title: How Game NPCs Became Agents Before Generative AI
title_zh: 生成式 AI 之前，游戏 NPC 是怎样成为 Agent 的
date: 2026-06-20 12:00:00 +0800
categories:
- Creative Media & Games
- Game AI & Production
tags:
- Survey
author: Hyacehila
excerpt: Before LLM agents became popular, game NPCs had already formed a practical agent architecture around perception,
  state, decision making, and action.
description: Before LLM agents became popular, game NPCs had already formed a practical agent architecture around perception,
  state, decision making, and action.
excerpt_zh: 在 LLM Agent 流行之前，游戏 NPC 早已围绕感知、状态、决策和行动形成了一套传统 Agent 工程。本文从小白视角介绍 FSM、行为树、Utility、GOAP、HTN 等 NPC 决策技术。
permalink: /blog/2026/06/20/game-npc-agents-before-llm/
lang: en
translation_key: 2026-06-20-game-npc-agents-before-llm
translation_status: machine
translation_source_hash: b2c7660a2229b7c1a9cc19790b272e5e269e86f2b2f2ef2d80e06fef01be4a8e
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>When it comes to NPC intelligence, the first reaction of many people has become a digital partner who takes over the task of the LLM, role-playing, long-term memory. This is in line with the recent trend of the epidemic, and many NPC people are doing this now. But moving a little bit further, the game industry is early on doing a classic Agent system: NPC is watching the world, keeping it in place, choosing the next step, and changing the world through movement, assault, trading, dialogue, escape or collaboration. LLM's only been out for a few years, and the game AI has been doing at least a dozen years.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/05/05/ai-agent-game-industry-pipeline/">How the game industry is introduced AI Agent</a>、<a href="/en/blog/2026/07/12/ai-native-game/">When we talk about AI Native Game, what are we talking about?</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>This traditional game AI usually does not call Agent. It could be called enemy AI, combat AI, teammates AI, monster behavior, Boss logic, NPC Brain, or simply behavioral tree. But in terms of system structure, it fits the basic definition of Agent: it is placed in an environment with observable information, objectives or preferences, action that can be implemented and changes the next behavior as a result of environmental feedback.</p>
<p>This article does not address the new issue of "Let NPC access big model chats" but starts with the game AI before the generation model appears. Viewing NPC as a tradition, Agent, to see how it is broken down into perception, state, decision-making, behavior and action; then explaining scripts, limited-state machines, behavioral trees, utility AI, GOAP and HTN in a way that is readable in little white. And finally, LLM Agent, and look at the old and new ones, and where they are, and where they are different. Looking back on those things, we are looking to better understand what we should do now.</p>
<h2>Think of NPC as a loop.</h2>
<p>The NPC in the game is not a static setup, nor is it an animated skin. It's in the game cycle. The engine of the game is moving at a time when the players, monsters, props, sound, bullets, mission status in the scene are changing. The NPC's intelligence is not calculated at once, but is updated over and over and over and over again on each frame.</p>
<p>A simplest NPC Agent cycle, probably:</p>
<pre><code class="language-mermaid">flowchart LR
  A[&quot;感知&lt;br/&gt;看到玩家、听到声音、读取任务状态&quot;] --&gt; B[&quot;世界状态 / 黑板&lt;br/&gt;敌人位置、血量、警戒值、目标&quot;]
  B --&gt; C[&quot;决策&lt;br/&gt;选择巡逻、追击、攻击、撤退、求援&quot;]
  C --&gt; D[&quot;行为&lt;br/&gt;执行一段可持续的动作流程&quot;]
  D --&gt; E[&quot;动作&lt;br/&gt;移动、瞄准、播放动画、发射技能&quot;]
  E --&gt; F[&quot;环境反馈&lt;br/&gt;玩家逃走、受到伤害、目标死亡&quot;]
  F --&gt; A
</code></pre>
<p>There are several keywords in this picture.</p>
<p>The first is perception. NPC should not know everything about the universe. A guard may only see the players in the view cone, hear the footsteps nearby, and remember the direction that was just attacked. Even if the programmer can get all the objects in his or her memory, the design will deliberately limit the NPC ' s sources of information. The source determines what he can do, and what NPC can do is closely linked to the player experience.</p>
<p>Second is the state. NPC needs to organize the perceived information into forms that it can use. The player is not in sight, last seen where the player is, whether the current blood levels are dangerous, whether the teammates are seeking help, whether the mission objectives have been completed, and this information is written into a local state. Some projects call it a blackboard, meaning that many modules can read and write temporary information on it.</p>
<p>The third is decision-making. The decision-making level answers what the time is. If the player attacks in a close proximity, if the player disappears, if the blood is too low, then the teammate goes. This sounds like a few if-else, but when real projects are complex, decision-making becomes a specialized problem.</p>
<p>Fourth is behavior. Behaviour is not an instant button, but a continuous process. Chase requires road search, diversion, shelter, distance; attack requires targeting, waiting to shake ahead, playing animation, settling injuries, and cooling. The decision-making level, which says only “I want to attack”, is responsible for turning this intention into a set of enforceable steps.</p>
<p>The fifth is action. The action is the first level of the closest engine, involving animation, physics, road search, skills, sound, special effects, network synchronization. Action is the foundation of everything, and a decision seems smart, but if animated, road-scrambling, and skills are not swayed, players still see a stupid NPC.</p>
<p>So the core of the traditional NPC AI is to put an entity in a controlled closed ring that is constantly moving. It must be able to run, be debugged, be modified, be stable on different machines and be reasonable to the player.</p>
<h2>Spectrometry of decision-making techniques</h2>
<p>The requirements for NPCs vary greatly from game to game. A 2D platform jumper in a game of patrol freaks, which may only require movement around and attack after encountering a player. A guard in a sneaking game needs to patrol, listen, search, call for support and return to his post after losing his target. An open world RPG companion needs to follow players, fight, treat teammates, avoid blocking, respond to the drama. An AI in a strategy game, which also takes into account resources, technology, military, map control and long-term planning.</p>
<p>So the game AI doesn't have a universal technology. More commonly, technology spectrum systems are more direct and manageable as they move to the left; more abstract and more complex objectives are expressed to the right, but the cost of achieving and debugging is higher.</p>
<pre><code class="language-mermaid">flowchart TB
  A[&quot;脚本 / 规则&lt;br/&gt;直接写死条件和动作&quot;] --&gt; B[&quot;FSM&lt;br/&gt;把行为拆成有限状态&quot;]
  B --&gt; C[&quot;Behavior Tree&lt;br/&gt;把决策组织成层级任务&quot;]
  C --&gt; D[&quot;Utility AI&lt;br/&gt;为候选动作打分&quot;]
  D --&gt; E[&quot;GOAP&lt;br/&gt;根据目标自动规划动作序列&quot;]
  E --&gt; F[&quot;HTN&lt;br/&gt;把高层任务分解成低层步骤&quot;]
  G[&quot;RL / ANN / GA / SA / MCTS&lt;br/&gt;学习、搜索、优化、模拟&quot;] -. &quot;常作为补充能力&quot; .-&gt; C
  G -. &quot;也可辅助&quot; .-&gt; D
  G -. &quot;也可辅助&quot; .-&gt; E
</code></pre>
<p>And then you press this line. Each approach begins with a description of what it solves and then a fictional game scene shows how it works.</p>
<h2>Perceptions and blackboards: the world before decision-making</h2>
<p>Before the specific decision algorithm, there is a layer that is often overlooked: how does the NPC know the world? Many newcomers understand AI NPC as a smart selection function, but in real games, what selection functions eat is often more important than selection functions themselves.</p>
<p>Seeing the senses first. Sensory is not simply a reading of player coordinates. A more human guard needs vision, hearing, touch and notification of events. The vision takes into account distance, angle, shield, light and player postures; the hearing takes into account sound intensity, transmission distance, materials and environmental noise; the notification of the event may come from a team member ' s call, alarm, mission script or the body the player has just triggered. The designer will deliberately make the information incomplete, as it is not complete that will give rise to suspense. When the player was hiding behind the box, the guards should not know the player ' s location directly, but only that there was a sound there or that the player was seen on the corner the last time.</p>
<p>And then look at the memory. NPC should not normally empty every frame of perception. It needs to remember the last time it saw the player, the source of the sound that was just heard, the fact that a door had been opened, and the enemy that had just attacked itself. This memory can be short, lasting only a few seconds; it can be longer, for example, when people in a simulation game remember a role and their own relationship. Memory keeps NPC from acting like a memory loss machine, and makes players understand why it continues to search, why it suspects a certain area, why it runs in one direction.</p>
<p>Blackboards are a common way of concentrating these messages. The blackboard is accessible through behavioral tree nodes, Utility ratings, GOAP actions, and road search modules. For example, there may be blackboards:</p>
<ul>
<li><code>enemy_visible</code>: Do you see the enemy at this time?</li>
<li><code>last_enemy_position</code>: Last time I saw the enemy's location.</li>
<li><code>health_ratio</code>Current blood ratio.</li>
<li><code>cover_position</code>: Recommended bunker location.</li>
<li><code>current_goal</code>: Current high-level targets.</li>
<li><code>suspicion_level</code>: Level of vigilance or suspicion.</li>
</ul>
<p>The advantage of blackboard is to decorate. The sensory system is responsible for what I see, the decision system is responsible for reading what I believe now, and the action system is responsible for where I'm going. But blackboards can also become waste dumps. If any module is written in a random way, the variable is not named, the life cycle is not clear, and the person who covers it is not clear, the problem is very difficult to find. Therefore, mature items usually define the type of blackboard variable, default value, expiry time, write permissions and debug displays, and cannot allow every developer to write the blackboard.</p>
<p>And there is another key point: the NPC's state of the world is not the same as the state of the real world. The real world may be playing behind the door, but the NPC blackboard only records the player's last appearance at the door. This difference is the space for the game AI's performance. The good NPC is not knowing the truth, but responding rationally based on limited information. Scramble games, horror games, tactical shootings depend on this. Players observe the information boundaries of the NPC and use them to develop strategies.</p>
<p>So, traditional Agent's intelligence is not just at the decision-making level. Sensation gives it input, blackboard gives it memory, and decision algorithms are based on this information. A very common FSM can also be credible if it is accompanied by reasonable vision, hearing, memory and search behaviour; a complex planning device, if it is a fraudulent global message, can destroy the experience. Whether it is to develop traditional decision-making systems or to develop LLM-based NPC, controlling the perception and memory of information is a matter for consideration, but it may have different angles.</p>
<h2>Script: The most intuitive NPC smart</h2>
<p>The earliest and most understandable game of AI is script. Scripts mean simple: they do specific actions under certain conditions. The player enters the room and the enemy brushes it out; the player approaches the businessman and the businessman turns around and talks; Boss has less than half blood to go to phase II; the player has the key and the door guard is free.</p>
<p>The advantages of the script are clarity, cost-effectiveness and stability. It is well suited for drama events, one-time performances, teaching posts, organ triggers and simple monster logic. The plan seeks to have a definite effect, and the script is often the most direct.</p>
<p>The problem is that with more scripts, the relationship between conduct becomes difficult to maintain. For example, a guard who was to patrol and see a player chasing, heard a voice and investigated, fled with low blood levels and suffered a breakdown in morale after the captain died. If all logic is written in if-else, it quickly becomes a patchwork of conditions for each other to cover.</p>
<p>A simple script may be this long:</p>
<pre><code class="language-text">每次更新守卫（随游戏时间流逝决策）:
  如果守卫死亡:
    播放死亡动画
    停止更新

  如果看见玩家:
    朝玩家移动
    如果距离足够近:
      攻击玩家
  否则如果听到可疑声音:
    移动到声音来源
  否则:
    沿巡逻路线移动
</code></pre>
<p>This is a good way to go, but it hides “what is being done” in the order of conditions. It may be reasonable to assume that the guards are investigating the sound and that the next frame suddenly sees the player and the script will immediately reach the chase. But what if the guards are playing an unbroken open-door animation? If it just lost its player, how many seconds should it search instead of immediately returning to patrol? Scripts themselves do not give a clear position for the continuity of these states, and different states should bring different decision logics, whereas scripts can only place all of this in if-else, which does not deal with complex logic but only with a certainty effect.</p>
<p>So, the limited-state machine came up.</p>
<h2>FSM: Disassembly NPC into several specified states</h2>
<p>FSM is Finite State Machine, a limited-state machine. Its core idea is that a NPC is one of the only limited situations at any given moment, each responsible for one category of conduct, and the situation is changed by conditions.</p>
<p>Or the example of the guards. We can give it five states:</p>
<ul>
<li><code>Patrol</code>: Patrols.</li>
<li><code>Investigate</code>: Investigations.</li>
<li><code>Chase</code>: Hunting the player.</li>
<li><code>Attack</code>: Attacking players.</li>
<li><code>Flee</code>- Run away or ask for help.</li>
</ul>
<p>So, what the NPC is doing right now is clear. Each state has its own entry logic, updated logic and exit logic, and complements behavior and action. Like getting in. <code>Attack</code> The draw-on animation is played, the attack distance and cooling is checked for updates and the impact of the attack is stopped when exit is completed.</p>
<p>The code can be written as follows:</p>
<pre><code class="language-text">状态 Patrol:
  沿路线移动
  如果听到声音: 切换到 Investigate
  如果看见玩家: 切换到 Chase
  如果血量过低: 切换到 Flee

状态 Chase:
  向玩家最后位置移动
  如果进入攻击距离: 切换到 Attack
  如果丢失玩家超过 5 秒: 切换到 Investigate
  如果血量过低: 切换到 Flee

状态 Attack:
  面向玩家并攻击
  如果玩家离开攻击距离: 切换到 Chase
  如果血量过低: 切换到 Flee
</code></pre>
<p>The benefits of FSM are simple, clear and low-cost. It is well suited to deal with small-scale behaviour: a limited number of states, clear conditions for conversion and a low demand for subsequent incrementality. For example, simple soldiers, organs, tools, page processes, or less complicated dialogue and delivery logic can be done using FSM. It can be explained, debugged and easily created; it is usually run only by checking the current state and a small amount of conversion conditions, and it is less expensive to perform.</p>
<p>It's typically a state explosion. If we add conditions such as “toxication” “freezing” “scoffed” “caring” “protecting VIP” “enhanced night alert”, the original conditions would be combined with those conditions. You might start writing. <code>ChaseWhilePoisoned</code>、<code>AttackWhileProtectingVIP</code>、<code>FleeWithFlag</code>I'm sorry. The situation is growing, the rules of transition are becoming increasingly confusing and no one can change it. That is, the FSM, like scripts, is suitable for small issues with clear boundaries; once the problem itself continues to become complex, the number and state of the state will increase dramatically, and the maintenance costs will soon exceed the simplicity it brought about at the outset.</p>
<p>Another problem is the lack of expression at the hierarchical level. For example, fighting itself can include pursuit, assault, evasion, changing bullets, sheltering; non-combat can also include patrols, chatting, viewing, repairing equipment. It is difficult for normal FSM to express this embedded structure naturally, but only to level the hierarchy to more states and more conversions. While the FSM can continue to develop the layers, many games will choose another form that is more appropriate for hierarchical organization: the behavioral tree.</p>
<h2>Behavior Tree: Organizing decision-making into a mission tree</h2>
<p>Behavior Tree, behavioral tree, often used in game AI. It's tearing the NPC's behavior into small nodes, and then it's combined with tree structures. Returns one of three states after each node is run:</p>
<ul>
<li><code>Success</code>: Success.</li>
<li><code>Failure</code>: Failed.</li>
<li><code>Running</code>: Still in implementation.</li>
</ul>
<p>Two types of combination nodes are common in behavioral trees.<code>Selector</code> Like "Find a plan that can be done from left to right" and stop trying the latter nodes as long as a subnode is successful or running.<code>Sequence</code> Like "Six steps in sequence" the whole sequence fails as long as a subpoint fails.</p>
<p>A guard behavior tree can understand this:</p>
<pre><code class="language-text">根节点 Selector:
  Sequence: 处理危险
    条件: 血量过低?
    动作: 寻找掩体
    动作: 呼叫支援

  Sequence: 战斗
    条件: 看见玩家?
    动作: 移动到攻击距离
    动作: 攻击玩家

  Sequence: 调查
    条件: 听到声音?
    动作: 移动到声音来源
    动作: 搜索附近区域

  动作: 巡逻
</code></pre>
<p>The tree means that if it is in danger, it is first treated as a priority; otherwise, if it is capable of fighting, it is fought; otherwise, if it is heard in a suspicious voice, it is investigated; and no patrol is conducted. It is clearer than a big body of if-else, because each act is broken into nodes and combinations are written in the trees.</p>
<p>The behavioral tree is particularly suited for several reasons.</p>
<p>First, it can express priorities. Putting emergency behaviour on the left side of the tree and ordinary behaviour on the right can create a natural logic of capture. Running out of low blood levels is a priority over patrolling, and players are entering the range of attacks more than talking.</p>
<p>Second, it's suitable for reuse.<code>移动到目标</code>、<code>检查距离</code>、<code>播放动画</code>、<code>等待冷却</code> It can all be a node, re-used between different enemies.</p>
<p>Third, it fits into the tooling. The behavioral tree can be a visual editor, allowing planning and programming to be adjusted together. The planning does not necessarily involve writing codes, but also understanding that “the enemy will first determine the blood mass, then the vision, and then the attack or patrol”.</p>
<p>Fourthly, it is naturally supportive of continuing behaviour. An action node can be returned <code>Running</code>That means the act is not complete. For example, moving to the target point takes a frame to complete, and the behavior tree next frame continues to tick this node.</p>
<p>The behavioral tree is not silver bullets. The larger the tree, the higher the cost of reading. If priority is expressed in a nodal order, the question may arise at a later stage why this behaviour is always taking control. Many complex behaviours also require sharing blackboard variables, which are written too freely and become another kind of invisible coupling.</p>
<p>The most appropriate problem for behavioural trees is to break complex behaviour down into hierarchical tasks. But it's not very good at answering another question: If there are many options, each is not absolutely right, but there are different levels of good and bad.</p>
<p>That's the Utility AI issue.</p>
<h2>Utility AI: Not if it's worth it, but if it's worth it.</h2>
<p>Utility AI can be translated into Utility AI. It is a human choice: it is not a simple judgement that can be made, but a point of each candidate to choose the most valuable act of the time. If you are of traditional statistical/ML origin, the utility of a similar SHAP can be understood.</p>
<p>For example, a teammate NPC might have these options:</p>
<ul>
<li>Attacking the enemy recently.</li>
<li>Therapeutic player.</li>
<li>Treat yourself.</li>
<li>Get behind the bunker.</li>
<li>Pick up the ammunition around.</li>
<li>The teammate of the resurrection fall.</li>
</ul>
<p>These acts are usually possible, but the degree of importance will change with the scene. The player's blood count is only 10%, and the treatment player's score is high; the NPC is dying, and the hideout or self-help score is high; the enemy is bleeding and close, and the attack score may be higher; the ammunition is running out and the collection score is up.</p>
<p>A small Utility AI can say:</p>
<pre><code class="language-text">候选行为:
  治疗玩家:
    分数 = 玩家受伤程度 * 距离可达性 * 治疗技能可用

  攻击敌人:
    分数 = 敌人威胁度 * 命中概率 * 武器弹药充足度

  躲避:
    分数 = 自己受伤程度 * 附近掩体质量

每次决策:
  计算所有候选行为分数
  选择最高分行为
  如果最高分比当前行为高出足够多:
    切换行为
</code></pre>
<p>Here's a little bit of detail: usually it doesn't change just a little bit higher. Otherwise, the NPC would be swinging between “attack” and “treatment”. Mechanisms such as delay, cooling, minimum execution time, switching costs are commonly used in engineering to make behaviour more stable.</p>
<p>The advantage of Utility AI is that it is appropriate to weigh multiple factors. It doesn't require you to draw all the state in advance, nor does it fix priorities in tree structures like the behavioral tree. It turns blood, distance, enemy threats, resources, mission targets, teammates, and then compares.</p>
<p>The difficulty is here: how do you design the score? A behavioral contour is too steep, NPC will be executed prematurely; the peace, NPC seems to be non-responsive. Multiple fractions multiplied or added? Should certain conditions be directly nil? A seemingly reasonable formula may be acting strangely in the real Guerrée.</p>
<p>So Utility AI relies heavily on debugging tools. Designers need to see the score for each current behaviour and how much each factor contributes. Otherwise, the player says, "Why didn't the teammates save me?" The planning and the procedure are guessed.</p>
<p>Utility AI is well placed to make immediate choices, but sometimes the NPC needs more than "what next to do," but "what to do with the next set of actions in order to achieve the goal." For example, there is no ammunition for attempting to attack a player; there is no key for trying to open the door; there is no medicine for trying to make it, but there is a lack of material. This is when planning issues are introduced.</p>
<h2>GOAP: Allowing NPC to target the generation of action plans</h2>
<p>GOAP is the Goal-Oriented Action Plan, a goal-oriented action plan. It's not about writing a fixed process, it's about breaking the world into a set of facts, breaking into a set of actions that NPC can do, and then putting it in the planners to spell out a course of action.</p>
<p>If you're going to use GOAP in the project, you're usually going to do three things.</p>
<p>First, define the state of the world. The planners don't understand the world of continuous, chaotic games, so you have to translate it into a discrete reality. Like what? <code>有钥匙 = true</code>、<code>仓库门打开 = false</code>、<code>知道玩家位置 = true</code>、<code>玩家被抓住 = false</code>I'm sorry. These facts can be derived from sensor systems, mission systems, backpack systems, level scripts, or blackboards.</p>
<p>Secondly, define the objectives. The goal is also a set of facts that we hope to achieve. For example, the goal of the guards is not to capture the player in a natural language, but to capture the player. <code>玩家被抓住 = true</code>I'm sorry. The treatment team could be targeting. <code>玩家安全 = true</code>The merchants may be targeting <code>完成交易 = true</code>The residents may be targeting <code>自己吃饱 = true</code>。</p>
<p>Third, define the action. Each action requires at least three components: precondition, execution effect, cost. Preconditions indicate when the action is available; performance effects indicate how the world will change when the action is successful; costs indicate how expensive, slow or dangerous the move is.</p>
<p>Take a guard from a fictional submersible game. The current state is that the guards do not have the keys and the warehouse door is locked, but it suspects the player is hiding in the warehouse. We can define these actions:</p>
<pre><code class="language-text">动作 拿钥匙:
  前置: 知道钥匙位置 = true
  效果: 有钥匙 = true
  成本: 1

动作 走到仓库门口:
  前置: 知道仓库位置 = true
  效果: 在仓库门口 = true
  成本: 2

动作 开仓库门:
  前置: 有钥匙 = true, 在仓库门口 = true
  效果: 仓库门打开 = true
  成本: 1

动作 进入仓库:
  前置: 仓库门打开 = true
  效果: 在仓库内 = true
  成本: 1

动作 抓住玩家:
  前置: 在仓库内 = true, 看见玩家 = true
  效果: 玩家被抓住 = true
  成本: 1
</code></pre>
<p>The developers did so without handwritten “take the key, then go to the warehouse door, open the door and then enter the warehouse”. You just tell the system what the world is about, what the goals are, how the actions change the facts.</p>
<p>What's the GOAP system doing behind it? It's going to start from the current state of the world, trying to connect the actions, see which sequences of actions make the target a reality. The process is essentially a search: the current state is a node, the execution of an action is given a new state, and the planners are expanding these states until a path to the target is found. This can be achieved by A*, Dijkstra, or other search strategies; there are also often many limitations in the game, such as how many steps to search, how often to re-programme, and which NPCs allow for complex planning.</p>
<p>Using the example just given, the planner might have received such a plan:</p>
<pre><code class="language-text">目标: 玩家被抓住 = true

当前状态:
  有钥匙 = false
  知道钥匙位置 = true
  知道仓库位置 = true
  仓库门打开 = false

规划结果:
  拿钥匙
  走到仓库门口
  开仓库门
  进入仓库
  抓住玩家
</code></pre>
<p>The implementation system will then be implemented in each action plan. Here, let's be aware: planning and implementation are not the same thing. The planner is only in abstraction and is actually being carried out by calling on the search, animation, interaction, combat, physics and mission systems. For example, walking to the warehouse door was an exercise in the plan, which could be carried out using a navigational grid, avoiding obstacles, handling disruptions by players, and playing open-door animations.</p>
<p>The world may also change in implementation. Players may have fled, keys may have been taken, doors may have been blown up. If the prefix of an action is no longer met or if the action fails, NPC cannot continue to do as it used to. It is common to discontinue the current plan, write back the new state of the world to the blackboard and reprogram it. So it seems that the NPC is really adjusting to the situation, not implementing an outdated list.</p>
<p>So the glamour of GOAP is the sense of emergence. Instead of writing dead processes for each situation, the designer provides action blocks and status rules, and the system forms its own route. If there is no key, go get the key, if the door is already open, skip the door and if the enemy is too strong, find the weapon first. NPC looks like it's working out.</p>
<p>The cost is clear. The state of the world must be fragmented, action defined with reliability, search space controlled and failure to implement can be rolled back and reprogrammed. GOAP is often suited to NPCs that need a sense of purpose: tactical enemies, sub-guards, team members, residents of the simulated world. It makes the role look more proactive, but only if you are willing to seriously model the state, the movement and the debugging tool.</p>
<h2>HTN: Disaggregating complex tasks into implementable steps</h2>
<p>HTN is Hierarchical Task Network, a hierarchical task network. It's also a planning exercise, but the thinking is not the same as the GOAP. GOAP is more like "I know the target state, please search for a sequence of actions" and HTN is more like "I know to do a high-level task, please break it down to a smaller task, as a rule, until it becomes a direct action."</p>
<p>If you want to use HTN, the most important task is not to define a large number of preconditions and effects, but to define how tasks are broken down.</p>
<p>There are usually two types of tasks in HTN. The first category consists of complex tasks, such as “helping players open through their strongholds” “to organize an attack” “to complete a daily patrol”. Complex tasks cannot be carried out directly, and they must continue to be dismantled. The second category is atomic missions, such as “Moving to bunkers” “Face-off” “three seconds” “therapeutic player”. The atomic mission is specific enough to be carried out by the action system in the engine.</p>
<p>Developers also define methods for complex tasks. The methodology could be understood as “how, under certain conditions, this task should be dismantled”. There are several ways to use the same complex task, and one of them is to be selected according to the current state of the world.</p>
<p>For example, the top-level task of a fellow NPC is to help players cross the stronghold. It can be broken in two ways:</p>
<pre><code class="language-text">复合任务 帮助玩家通过据点:
  方法 A: 潜行通过
    条件: 玩家未被发现
    子任务:
      标记敌人
      关闭探照灯
      跟随玩家潜入

  方法 B: 正面交火
    条件: 玩家已被发现
    子任务:
      找掩体
      压制敌人
      治疗玩家
      跟随玩家推进
</code></pre>
<p>The “enemy repression” itself is not an engine operation, but it can continue to be dismantled:</p>
<pre><code class="language-text">复合任务 压制敌人:
  方法 默认压制:
    条件: 有可见敌人
    子任务:
      选择高威胁目标
      移动到可射击位置
      瞄准目标
      连续射击三秒
      评估是否换位
</code></pre>
<p>What would HTN do with the back? It would start with a high-level mission, examining current methods for replacing the task with its sub-mission; continue to break down if there were a complex task in the sub-mission; and until the whole of the mission tree was broken into a series of atomic tasks. The last system does not get an abstract objective, but an enforceable list.</p>
<pre><code class="language-text">输入任务:
  帮助玩家通过据点

当前状态:
  玩家已被发现 = true
  有可见敌人 = true

HTN 分解过程:
  帮助玩家通过据点
  -&gt; 找掩体 -&gt; 压制敌人 -&gt; 治疗玩家 -&gt; 跟随玩家推进
  -&gt; 找掩体 -&gt; 选择高威胁目标 -&gt; 移动到可射击位置 -&gt; 瞄准目标 -&gt; 连续射击三秒 -&gt; 评估是否换位 -&gt; 治疗玩家 -&gt; 跟随玩家推进
</code></pre>
<p>If the conditions of a method are not met, the HTN planner will try another way of doing the same job. For example, players are not found and pass through; players are found and there is a direct exchange of fire. If all methods are not available, the task fails to break down, and the upper level of the task can be replaced by another option, or the implementation system can be discontinued and the top level selected.</p>
<p>HTN and behavioral trees are a little bit like these because they're good at expressing the hierarchy. But the two are different. The behavioral tree usually runs repeatedly at the tick while judging to implement; HTN is more like Mr. Man's mission plan and then hands it over to the implementer. That is, the behavior tree is "real time control structure" and HTN is "mission decomposition and plan generation". Of course, it can be used in the actual project: the outer behavioral tree determines the current large pattern, and then enters a pattern and generates a task plan using HTN.</p>
<p>HTN has the advantage of being very close to the designer's thinking. Many games are based on the decomposition of tasks: cooking, patrolling, trading, cleaning rooms, organizing attacks, rescue teammates. The designer can write the top-level processes clearly, while allowing the system to choose different dismantlings according to the conditions. It also has easier access to narrative and mission systems, as drama missions, scheduling systems and camp operations are inherently hierarchical.</p>
<p>It costs more than a previous modelling. You have to define complex tasks, atomic tasks, methods, conditions, sub-task sequences, and deal with failure retreats. HTN is too dead to be a more complex script; it is too abstract to predict the results of the operation. It is suitable for medium-sized and large-scale behaviour systems and not necessarily for a little monster who only controls around.</p>
<p>One sentence can distinguish between GOAP and HTN: GOAP, which allows the system to search in action space for "how to turn the world into a target state" and HTN, which allows the system to break down "the big things I'm going to do" into a series of small things according to the rules you have written. The former places greater emphasis on state and action effects, while the latter places greater emphasis on the decomposition of knowledge provided by mission structure and designers.</p>
<h2>Where do you put the learning and searching methods?</h2>
<p>With regard to game AI, many people naturally think about intensive learning, neural networks, genetic algorithms, simulation of the fire-release, and the Monte Carlo tree search. These are, of course, AI, but their place in the traditional NPC is often misunderstood. They do not always directly control the running of the NPC, and more often, the ability to support a particular link.</p>
<p>Enhanced learning is appropriate for intelligent people to pass the mislearning strategy. It is valuable in chess, simulation of combat, automatic testing, balanced exploration, robotic control, etc. But the NPC in the business game is not necessarily right for a black box strategy to control. The reasons are practical: high training costs, behaviour difficulties, difficulties in debugging, and the risk of uncontrolled online performance. Players want NPC not only to be strong enough, but also to be fair, characterful and make reasonable mistakes.</p>
<p>Neural networks can be used for sensory, predictive, animation control, strategic approximation, difficulty regulation, etc. For example, predicts the next direction of the player, determines whether a position is easily discovered or compresses complex rating functions into a model. But if the nerve network directly determines all the NPC behavior, it is difficult for designers to control experience precisely. And of course we can say that the LLM is a neural network, which is of course not very meaningful.</p>
<p>Genetic algorithms and simulations of the fire retreat are more like tools for optimization. They can help search for a combination of parameters, such as enemy recharging frequency, skill cooling, team formation, race routes, and gate layout ratings. They are not usually the answer to how each frame of the monster works, but help developers find better configurations.</p>
<p>The Monte Carlo tree search is often used in chess, card, strategy and local tactical simulations. It is adapted to a well-defined and rapidly evolving environment through a large number of simulations to assess candidate actions. The problem is that real-time action games are usually in a large state space and are costly to simulate each time, so MCTS is often used for local decision-making, offline analysis or certain specific play methods.</p>
<p>These types of methods and the front FSM, behavioral trees, Utility, GOAP, HTN do not conflict. The more common structures are mixed: The top layer is still explained by behavioral tree or HTN control, with local Utility ratings, with some parameters generated by learning methods, and with some candidate routes assessed by search algorithms. The goal of the game AI is not to show off, but to serve experience.</p>
<h2>Why is tradition more controllable than NPC?</h2>
<p>If you look at it from the paper or Demo, the more automatic, the smarter, the more likely it will appear, the more advanced it seems. But the game development has a very realistic set of constraints.</p>
<p>The first is real-time sex. NPC decision-making cannot hold up the game. There may be dozens of, and hundreds of, entities in an open world that are updating at the same time. Every NPC is doing a complicated plan, and the frame rate will drop. It is common practice in engineering to reduce the frequency of updates, to tiered calculations, to simplify simulations at a distance, to spread expensive decisions over multiple frames.</p>
<p>The second is debugability. The player said the enemy was unreasonable and the developer had to be able to put it back on the scene, see what it saw, what it saw in the blackboard, what the behavioral tree went to, what the Utility score was, why the GOAP program failed. Without visuality, AI becomes an unreserved mystery box.</p>
<p>The third is designability. The NPC is not the smarter, the better, the better. The monster in the horror game may need to feel oppressive, but not the best in every case; the enemy of the new village needs to let the player learn the system, not to knock the player down; and the teammate NPC needs help, but cannot take away the hero moment for the player. AI's smart is bound by the experience target.</p>
<p>The fourth is certainty. Many games need to be replayed, synchronized, video-recorded, and online. NPC behaviour is problematic for debugging and network synchronization if it is full of uncontrolled randomity. Even when random, remnant random seeds and clear probabilities are often used.</p>
<p>Fifth is production collaboration. The NPC behavior is not done by writing alone. Plan the parameters, art to animate, level to be set up to patrol, audio to sound, QA to repeat the problem. Behavior Tree Editor, Blackboard Viewer, Debug Line Box, Log, Hot Update Configuration, these tools are part of the AI game.</p>
<p>So the core value of traditional game AI is not to make NPC a real person, but to make it act like a credible actor under a limited rule. That credibility is crucial. Players do not demand that every enemy have a free will, but that it be made sense in the rules of the game.</p>
<h2>A complete NPC. How can we combine these technologies?</h2>
<p>To tie up the concept, imagine a imaginary action of a teammate in RPG. Its duties are to follow players, to fight, to treat, to alert to danger and to perform special actions at the scene.</p>
<p>The outermost layer can be prioritized by behavioral tree organization:</p>
<ul>
<li>If the plot is forced to control, perform the act of the plot.</li>
<li>If you're dying, you're going to have to hide and save yourself.</li>
<li>If the player falls, try to rescue.</li>
<li>If you're fighting, go into the subtree.</li>
<li>If there is no fight, follow the player or stand by.</li>
</ul>
<p>Within the subtree of combat, the choice is to use Utility AI:</p>
<ul>
<li>The score of the therapeutic player depends on the player ' s blood, distance, treatment cooling and current hazard.</li>
<li>The score of the enemy attack depends on the enemy ' s threat, the probability of impact, distance and ammunition.</li>
<li>The score of flight depends on the amount of blood, enemy fire and bunker quality.</li>
</ul>
<p>If you have a complex target, you can use GOAP or HTN:</p>
<ul>
<li>The player is trapped behind the door, and the NPC needs to find the console, unlock it, and go back to the player.</li>
<li>The team is ready to break into the room, and the NPC needs to throw smoke bombs, find shelter, suppress the enemy and then follow the player.</li>
</ul>
<p>Bottom-level actions are given to road-seeking, animation, skill-based systems, collisions and audio systems. AI decision says "Moving to bunkers" and action layers need to find specific paths, process jamming, and play roll or crouch animation. The decision says “therapeutic player”, and the skills system checks distance, direction, cooling, resources, and synchronizing with the network.</p>
<p>That's why the game AI is often a system project, not a single algorithm. The FSM, the behavioral tree, Utility, GOAP, HTN are just tools at the decision-making level. The real NPC Agent also needs a budget for perception, memory, action, debugging, content tools and performance.</p>
<h2>When LLM intervenes in the game NPC: What hasn't changed, what has changed</h2>
<p>Generating AI let's re-think NPC: They can say more naturally, remember the player's experience, explain their motivations, and even organize a response based on the player's expression. But linking LLM to the game does not mean that the traditional NPC architecture can be replaced as a whole. A talking role still has to act in the game world; as long as it acts, it cannot bypass the closed circle of perception, state, decision-making, action and feedback.</p>
<p>What has not changed is the bottom engineering relationship. LLM can be involved in understanding what players say, or can produce a character-appropriate line, but it cannot simply allow characters to cross walls, skip animation, ignore skills cooling, or offer job awards in a vacuum. The ultimate ability of NPC to move depends on the path and physics; whether to attack depends on the skill system and the rules of combat; whether to do the job depends on the mission system; whether to say something, depends on the world view, the security of content and the rules of localization.</p>
<p>So, LLM is not usually a brain that replaces behavioral trees, but is attached to semantic or high-level intent layers. It helps NPC understand the players ' natural language, generates more responsive responses, explains the objectives of the current mission, collates long-term interactive memories, or proposes several high-level candidates. When the game really is going to run, the traditional system still needs to translate these semantic results into the action of being bound: going to where, looking to whom, triggering which animation, calling which skill, updating which task. Language models can be used to think, but the action interface must be wrapped in the rules of the game.</p>
<p>The real change is the semantic interface between NPC and the player. Traditional NPCs often respond only to fixed options: click on dialogue, take over, deliver goods, trigger combat. When I got into LLM, the player might have said, "I'm lost, take me somewhere safe," "Why don't you believe that businessman" and "I don't want to fight now, do I have any other way?" Models can express these openings as a more structured intent, such as seeking help, asking for directions, questioning the story, requesting alternative routes, and leaving them to mission, navigation or decision-making systems to judge whether they can be done.</p>
<p>But the LLM also brings new risks. It can create hallucinations, promise incentives that do not exist, create assumptions that are not in the worldview, misinterpret players ' intentions, or give unfair messages to a player in a multiplayer game. It also has delays and costs, which are not suitable for each frame to participate in real-time decision-making. The closer the player experience, the more border is needed: what can be generated, what must be read from the mission database, what actions require the rule system to identify, what answers must be safely filtered and what memories can be preserved over time.</p>
<p>So, the more conservative understanding is that traditional games AI are responsible for making NPC act credible in a world of rules, and LLM is responsible for making NPC better understand and express itself. The two are not simply old or new relationships, but rather relationships of collaboration at the upper and lower levels. Traditional systems provide boundaries, status, movement and validation, and LLM offers language, interpretation, intention and candidate options.</p>
<h2>How should White understand it?</h2>
<p>If you first touch AI, you can leave the algorithms out of your hands. The better way to get into the door is to ask five questions.</p>
<p>First, what does NPC need to know? This is a resonance with perception and the state of the world. It can see the player, hear the sound, know where the teammates are, remember what just happened.</p>
<p>Second, what can NPC do? This corresponds to the action space. It can move, attack, hide, speak, trade, open doors, call for help, or only patrol to the left.</p>
<p>Third, why did NPC choose this action? This corresponds to decision-making. Simple behavior is enough with scripts or FSM; hierarchy is with behavioral trees; multifactorial trade-offs with Utility; target-driven processes are with GOAP or HTN.</p>
<p>And fourth, how does the player feel about it? It's a performance layer. Animation, sound, turn speed, shake before attack, lines, hints UI, influence the player 's judgement of AI. Sometimes NPC makes sound decisions, but it's too sudden and players still feel it's cheating.</p>
<p>Fifth, if you access LLM, which floor is it responsible for? This is the easiest thing to confuse when we discuss LLM NPC today. LLM understands what the player says, it can produce a more natural answer, it can organize vague requests into high-level intentions, and it can help the role to explain its behaviour. But it should not decide whether the role is rewarded, whether the player can be found, whether the skills can be used, whether the situation can be changed. These should still be confirmed by the rules of the game, mission systems, combat systems and traditional AI structures.</p>
<p>So for example, the player said to his teammate NPC, "Look for a safe route for me." LLM can interpret this phrase as “a player wants to avoid fighting and reach a target point”, or it can generate a response that is consistent with the role set, such as “I will see if there is a way around the patrol”. But the real safe route is not made up of models. The route is inaccessible, and questions are asked about navigation systems; which areas are dangerous depending on the enemy perception, patrol route and blackboard status; whether the route should be bypassed by Utility or GOAP to assess the cost; and how the NPC eventually moves, and is left to the search, animation and shielding systems.</p>
<p>And in this framework, many of the AI phenomena become easier to understand. The enemy “suddenly knows where you are”, which may be a perceived border that is not designed; the teammates “do not save you”, which may be Utility scores or behavioral priorities are problematic; the monster “silent” may be a behavioral tree at a point that has been Running; Boss “silent in the transition phase”, which may be a lack of transition for FSM switching; and the simulator “self-acting too mechanical a day” may require HTN or a more abundant agenda. If NPC, after access to LLM, starts to make up the job incentives, create the non-existent location, the problem is not necessarily “memorning in a way that is not smart enough”, but semantic production is not constrained by the mission database and rules system.</p>
<p>Scripts make behavior controllable, FSM makes behavioral trees make hierarchyable, Utility makes multi-factor selection more flexible, GOAP allows NPC to generate plans around targets, HTN allows complex tasks to be decomposed. Learning and search methods are like enhanced modules in toolboxes that help to train, optimize, simulate or solve local decision-making problems. These traditional technologies are not old parts that LLM NPC has come to pass, but the foundations that make the players truly capable of moving in the game.</p>
<p>LLM allows NPC to speak, explain, connect to players' intentions, but the credible game Agent still relies on perception boundaries, state management, action constraints, debugging tools and design targets. It is not so much a good line for the character to say, but rather to keep, stabilize and control the game. It is already in the Agent domain as long as it can observe the environment, maintain the state, choose action and receive feedback. Whether semantic layers are connected to the language model is the next option based on the traditional AI base.</p>
<h2>References</h2>
<ul>
<li>Ian Millington, <em>Artificial Intelligence for Games</em>. <a href="https://www.taylorfrancis.com/books/mono/10.1201/9781315375229/artificial-intelligence-games">Taylor &amp; Francis Page</a></li>
<li>Steve Rabin, editor,<em>Game AI Pro</em> Series. The official network provides chapter indexing and publicly downloading instructions:<a href="https://www.gameaipro.com/">Game AI Pro</a></li>
<li>Alex J. Champandard, Philip Dunstan, <em>The Behavior Tree Starter Kit</em>. <a href="https://www.gameaipro.com/GameAIPro/GameAIPro_Chapter06_The_Behavior_Tree_Starter_Kit.pdf">Game AI Pro Chapter 6</a></li>
<li>Kevin Dill, Dave Mark, <em>An Introduction to Utility Theory</em>. <a href="https://www.gameaipro.com/GameAIPro/GameAIPro_Chapter09_An_Introduction_to_Utility_Theory.pdf">Game AI Pro Chapter 9</a></li>
<li>Jeff Orkin, <em>Three States and a Plan: The AI of F.E.A.R.</em> <a href="https://www.gamedevs.org/uploads/three-states-plan-ai-of-fear.pdf">GDC Information PDF</a></li>
</ul>
