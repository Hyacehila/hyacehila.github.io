---
title: 'Video Game Genre Classification: From Genre Trees to Gameplay Tags'
title_zh: 电子游戏类型分类：从树状分类到玩法标签
date: 2026-06-26 16:10:00 +0800
categories:
- Creative Media & Games
- Game Design
tags:
- Game Design
- Survey
author: Hyacehila
hidden: true
excerpt: Covers video game genre classification through genre trees, gameplay tags, player skills, platform constraints, and
  hybrid genres.
description: Covers video game genre classification through genre trees, gameplay tags, player skills, platform constraints,
  and hybrid genres.
excerpt_zh: 电子游戏类型不是一棵严丝合缝的树，而是一套帮助我们描述玩法、玩家技能、平台和体验预期的语言。
permalink: /blog/2026/06/26/game-classification/
lang: en
translation_key: 2026-06-26-game-classification
translation_status: machine
translation_source_hash: 61bc1b23224d518ef0ae6791eeec412b05262c15e25d94e9f254e5bbe7b919b1
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>When I first learned the type of game, I was thinking of a very simple watch: ACT, RPG, AVG, SIM, SLG. Like naming a folder. The action is in action, role playing is in RPG, strategy is in strategy, it looks like it's a little bit less. That's what I've been doing for years, but it's stuck.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/06/20/game-npc-agents-before-llm/">How the NPC became Agent before the generation AI</a>、<a href="/en/blog/2025/01/15/agent-ai-multimodal-survey/">Multi-modular Agent AI Overview: Model, Learning Mechanisms and Applications</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>The legend of Zelda: The Wildness is a movement risk, of course. But it's open to the world, to explore, to solve puzzles, and even a little bit of the sandbox system. The God of the Oracle is either an open world ARPG or a role-forming and card-screeching service game. Kings Glory is a Moba, but mobile input, short-chance, social relations are important. The Ark of Tomorrow is more troublesome: towers, tactics, role collection, jigsaw puzzles, long run, less one missing.</p>
<p>Game type is not a one-time-corrected folder. It's more like a shorthand for chatting. Saying “Roguelike”, others think of random, dead, repeated attempts; saying “MOBA”, others think of heroes, soldiers, team members; saying “recreation”, others assume that it is easy to handle, less stressful, and suitable for short-term opening.</p>
<p>These terms are not necessarily precise, but they are useful. The problem is that they are labeled and can easily be lazy. Is a RPG talking about role growth or is it about the drama? Is an open world about big maps or is it about players who really organize their goals? Is a "scratch card made" a play, or is it a long-term operation and business structure?</p>
<p>So what this article does is simple: untrace these types of names. What are the traditional broad categories, what are the types of integration and what are actually describing platforms, emotions, social ways or business models? It's not a dictionary. It's more like a learning note. When we say what kind of game we will be, we will know at least which layer we're focusing on.</p>
<h2>Why is it so hard to separate the game?</h2>
<p>Traditional classifications are certainly valuable. The action game can be divided into shooting, combat, platform action; the RPG can be divided into JRPG, CRG, ARPG, MMORPG; and the strategy can be divided into RTS, SLG, 4X, Tower Defence. It is clear, it is good for entry and it is easy to remember.</p>
<p>The problem is that modern games are often not grown from a branch.</p>
<p>The mechanism is borrowed from each other. The first person to use FPS, known as the Perspective and Mobile Operation, could be used to shoot or solve puzzles, was Portal. RPG role growth can serve round-based combat or instant action, and there is ARPG. The map control and resource competition for strategic games can be combined with a single hero, and there's a MOBA.</p>
<p>The platform will also change the type. Keyboard mouse is suitable for complex shortcuts and information density, so PC is long-term suitable for strategy, simulation, management and hard-core shooting. The handle is suitable for movement of the role, action fighting and speed. Touch screens are suitable for clicking, dragging, short cycle and debris time. The senses, VRs, steering wheel, flying poles, rhythm controllers can then push some experiences into new directions.</p>
<p>Player skills are not the same. Action games look at reaction and timing, strategy games at planning and resources, puzzle games at understanding, simulation games at understanding the system, RPG at character formation and long-term growth. Many games deliberately fold these skills together.</p>
<p>And one thing is often overlooked: the type grows out of the player community. Roguelike from Rogue, Soulslike from the Spirit of Darkness series, Metroidvania from the traditions of Galactica Soldiers and Devil City. MOBA and RTS are deeply related to the custom map, and the modern form of Battle Royale is also related to Mod, military simulation, survival play. The type is not invented only from the design document, but it is also played by players and then named by the market and community.</p>
<p>So instead of asking “what type of game is it really”, let me ask first, “Why should I classify it”. When recommending players, labels are closely experienced; when designing analyses, they are broken down into mechanisms and player skills; when looking at historical evolution, they are based on iconic works and communities; and when market positioning is done, they are based on platforms, business models and audience expectations.</p>
<h2>Use MDA to break it down first.</h2>
<p>MDA breaks the game into three layers: Mechanics, Dynamics, Aesthetics. In short, Mechanics is the rules and the system, Dynamics is the process that happens after the player interacts with the system, Aesthetics is the last thing that a player feels. Because many of the types actually mix these three layers.</p>
<table>
<thead>
<tr>
<th>Type</th>
<th>Mechanics: What's in the system?</th>
<th>Dynamics: What do players do?</th>
<th>Aesthetics: What does it feel for a player?</th>
</tr>
</thead>
<tbody><tr>
<td>Sports</td>
<td>Realistic motion rules, scores, body or sensory input</td>
<td>Practice, confrontation, the rhythm of the game, replaying the real game.</td>
<td>Competition, body in, and entry. Sensor.</td>
</tr>
<tr>
<td>Simulation</td>
<td>Variables, resources, production chains, physical or economic rules</td>
<td>Access, observation of feedback, optimization of processes</td>
<td>Managing satisfaction of complex systems</td>
</tr>
<tr>
<td>Actions</td>
<td>Move, attack, determine, avoid, cool.</td>
<td>Response, re-entry, departure, risk control.</td>
<td>Nervous, happy, controlled.</td>
</tr>
<tr>
<td>Adventure.</td>
<td>The scene, the object, the conversation, the play point.</td>
<td>Explore, reason, choose the path, advance the story.</td>
<td>Curiosity, doubt, story-in-story.</td>
</tr>
<tr>
<td>RPG</td>
<td>Properties, levels, equipment, skills, choices</td>
<td>Build roles, advance stories, take consequences.</td>
<td>Growing up, playing, feeling like you're a person.</td>
</tr>
<tr>
<td>Policy</td>
<td>Units, resources, maps, technology, poor information</td>
<td>Planning, trade-offs, restraint, movement control</td>
<td>You're the best. You're the best.</td>
</tr>
<tr>
<td>Roguelike</td>
<td>Process generation, random objects, death penalty</td>
<td>Mistakes, memories, adjoining structures.</td>
<td>Surprise, defeat, another round.</td>
</tr>
<tr>
<td>Solve the puzzle.</td>
<td>Organisms, space relations, physical rules, clues.</td>
<td>Tests, reasoning, awareness, route planning</td>
<td>I'm trying to get to the point.</td>
</tr>
<tr>
<td>Leisure</td>
<td>Simple input, short cycle, low penalty, light socialization</td>
<td>Walk in, repeat thrust, play time with debris</td>
<td>Relax, be with you, be low on the load.</td>
</tr>
</tbody></table>
<p>In Starbucks, Mechanics is planting, collecting, producing, socializing, time and physical strength; Dynamics is a daily gamer who decides whether to water, mine, fish or give gifts; Aesthetics is typical of relaxing, accumulating and accompanying. It can be called a simulation, of course, or a farm life simulation, or even close to leisure games under some play. Different labels just capture different layers.</p>
<p>Portal is the same. The first person, who is a member of the group, is a member of the group, and he is a member of the group. Dynamics is not fighting, but using portals to understand space relations. Aesthetics is not a sense of enemy killing, but a sense of understanding. Just that it's a FPS that's misleading, just that it's a puzzle and it's missing the operating framework that it borrowed.</p>
<p>So when I say the type later, I try to see in the order of MDA: what is the common mechanism for this type, what kind of game players will develop, and what is the final sense of being human.</p>
<h2>Basic categories: first look at what players are doing.</h2>
<h3>Sport and Race</h3>
<p>Sport games transform the rules of real sport, points and physical skills into an operational digital system. FIFA, NBA 2K, Madden NFL relies on real leagues, player lists and fan culture; Wii Sports turns body moves like swinging, throwing, waving, etc. into family entertainment.</p>
<p>Races often come independently from sport. Streets compete for speed, drift, props and irritation, such as the Marriotkadin; and vehicular calibration, track memory, brake point and driving technique, such as Gran Turismo, Forza Motorsport. The most special product of the game should be Forza Horizon, which weakens sports competition, opens the world to the players and gives them a view and experience, and the whole play experience becomes more leisure. It's also a race, and the players can be completely different.</p>
<p>The border is not too difficult: are players competing in a set of rules? Is input in simulated body or delivery techniques? If so, it is close to sports or racing.</p>
<h3>Simulation</h3>
<p>Simulations are not a complete reproduction of reality, but a condensed version of the key relationships in the real system. A good simulation may not be the most real, but it is important that the player understand and manipulate a system.</p>
<p>The City: Skyline makes transport, population, taxation and public services variable; the Star of the Hill Cruiser, the Twin Point Hospital, turns business, operating lines and customer satisfaction into management issues; and the Animal Forest Friends, the Star-Lown Grain Language, makes life rhythm, collection, decoration and socialization more moderate.</p>
<p>The other end is hard core simulation. The Microsoft Flight Simulation, Kerbal Space Program, retains additional flight or orbital mechanics constraints. The difficulty is not to bring in a head of reality, but to make complex systems learning. Too real, too abstract, and loses the taste of simulation.</p>
<p>The home-grown, run, and living-type games are often near simulations. The Chinese Parents, The Daughters of Volcanic Volcanics, simulated growth paths and resource allocation; many restaurants, farms, city run hand-to-hand tours, made the production chain, waiting time and order systems mobile-end cycles.</p>
<h3>Actions</h3>
<p>The action game is based on instant operation. Move, hide, attack, jump, aim, block, and move, and players have to react in a constantly changing situation.</p>
<p>Platform action is a classic form. Super Mario looks at the leap, drop and junction rhythms; Celeste makes leap and sprint as a delicate challenge; and the Cosmic Robots place greater emphasis on space interaction and the criticalism. Fight games such as " Street King " and " Iron Box " , which constrict action to short periods of confrontation, input precision, distance and psychological games.</p>
<p>Shooting is the big part of the action game. FPS emphasizes targeting, walking and visual control in the first person's view, such as Doom, " Counter-Strike " , " Apex Legends " ; TPS can see the body of the role and is better suited to the bunker, roll, near-field combat and role movements, such as the War Machine, " The Jedi for Life " . But the shooting framework does not necessarily serve to kill, and Portal uses FPS to solve a space puzzle.</p>
<p>Action games are often misunderstood as violent, but violence is not the essence. It is really concerned with real-time input and instant feedback. The player is slow, the experience changes.</p>
<h3>Adventure.</h3>
<p>The core of adventure games is not the story of adventure, but the player advances the unknown by observing the scene, understanding the situation, using objects, dialogue or choice.</p>
<p>Traditional point-and-click adventure, such as Grim Fandango, the "Secret of Monkey Island", allows players to find clues, combine items, and engage in character dialogue. Modern adventure games have wider boundaries. Life is Scrange, Fairwatch, which is biased in narrative exploration and role relationships; Detroit: The Mutant, which is the Walking Dead, which is the interactive movie and branch selection; and Japanese ADV, visual novels and reasoning writings that risk pushing text, role and choice to the front, such as the Reversive Judge, Bullets Debut.</p>
<p>Action risk is a more mainstream form of integration. The legend of Zelda, the reflection of the tomb, the Mystery Sea, combines exploration, space understanding, puzzles and instantaneous action. It's not very important to argue whether it's action or adventure. More important is to see how players are allocating their attention: to operate, to read space and stories.</p>
<p>The problem of adventure is also obvious. For the first time, fixed puzzles and linear stories were strong, and the second time, perhaps, was less fresh. Branches, hidden clues, multiple demarches and environmental narratives are all supplementing the slab.</p>
<h3>RPG</h3>
<p>The RPG is not about hierarchy, but rather about players playing one or more roles, which form growth through attributes, equipment, skills, teams, choices and stories. It's about three questions that are most attractive: who am I? How am I growing? Will my choice change the world?</p>
<p>JRPG usually emphasizes fixed roles, clear narrative lines, team growth and narrative. The final fantasy, the warrior fight the dragon, the goddess's acclaim, is a typical example. The turn system is common, but not absolute. More importantly, the player goes into a heroic story that others have written, but I can be involved in experience.</p>
<p>The new policy is to create a new system of regulation, to create roles, to organize teams and to select the consequences. The Gate of Bird 3 Radiation 1/2 The Spirit of the Exotic Towns has a tradition of table and tour. Players do not just read the lead story, but write their own story through role creation, dialogue choices, mission paths and camp relationships. The Open World of Bethesda-style RPGs, like the Upper Roll 5 "Radio 4" and free exploration and player self-discovery in a larger space.</p>
<p>ARPG puts RPG growth, equipment and construction into immediate action combat. The Darkness of the Darkness of the Sunshine, the Monster Hunter, the Elden Ring of the Law, the Holy Spirit can be understood from different angles as ARPG. MMORPG, for its part, puts RPG in a long-term online society, where the World of Monsters, End of Fantasia XIV, Dream West Travel, emphasizes occupation, team, union, trade, copy and version operation.</p>
<p>SRPG or tactical RPG combines role growth with chess strategy. The Flaming Flaming Seal, the Royal Order, the Dream War, combines role growth, position, terrain, resources and permanent death. A unit is not just a pawn, it's a role.</p>
<h3>Policy</h3>
<p>The core of the strategy game is to win with your head. It requires that players understand rules, resources, terrain, units, timing, information and long-term plans. Traditional chess can be seen as its spiritual source: the winner and the loser are not reaction, but judgement.</p>
<p>Turn-based strategies give players more time to think. The Civilization Series represents 4X: Exploration, Expansion, Development, Conquerion. Players need to make long-term choices between science and technology, diplomacy, military, cities and resources. The Frozen Flaming Seal places greater emphasis on position, hit rate, bunker, skills and role risk at the tactical level.</p>
<p>RTS put together strategy and time pressure. Star Trek, The Age of the Empire, Warcraft III, requires players to deal with both resource collection, science and technology trees, build armies, detection, expansion, micro-practition and multi-line fighting. It is not a “action-plus strategy”, but a compression of macro-planning and micro-implementation into the real-time rhythm.</p>
<p>The big strategy and war simulation is another path. The Hearts of Iron look at national industry, supplies, fronts and politics; Total War often combines turn-made large map management and real-time battlefield command. Light tactics, such as the Worlds, are intuitive in terms of terrain damage, projector trajectories and round decisions.</p>
<p>The SLG in the domestic language is sometimes confusing. It may refer to traditional tactical chess or to long-standing war/coalition games at the mobile end. The latter often include both construction, breeding, coalition diplomacy, PVP, time acceleration and payment resources. Just one SLG, which is often unclear.</p>
<h3>Solve the puzzle.</h3>
<p>So the puzzle game allows the player to understand the rules and then use that understanding to solve the problems that are being designed. The good puzzle is not just hard, but it's a moment when the player suddenly realizes that the system worked this way.</p>
<p>The Russian Blocks compress space filling, velocity pressure and long-term risk to the extreme. Lemmings allows players to distribute skills to small people in real-time environments, addressing pathways, sacrifices and overall goals. Portal borrows the FPS operating framework to turn perspective, space, portal and physical relations into puzzles. The Angry Birds, on the other hand, makes the touch screen, parabolic lines and physical damage a leisure puzzle.</p>
<p>The puzzle is often mixed with adventures, terror, platform moves, physical simulations. The Memorial Valley relies on visual illusions and spatial structures; Baba Is You turns the rule itself into a removable object; and the Secret Room Escape and reasoning games rely more on the thread organization.</p>
<p>The pit of the puzzle is a solution. If the designer accepts only one answer without giving enough feedback, the player is not understanding the system, but guessing what the author thinks in his mind. The early pixel hunt in point-and-click early was the problem: players didn't get it, they just didn't know which pixel to use.</p>
<h3>Leisure</h3>
<p>Leisure games are not low quality games. It is a design goal: to reduce learning costs, failure pressures and single inputs, allowing players to enter in light, debris or social scenes.</p>
<p>The three-point, click, synthesis, light-run, placement, light-handed puzzles are often part of the leisure. The Candy Crush, Best Friends, Township, emphasizes short-term level, intuitive input, instant feedback and long-term light targets. Cricker or the game where you put it down, the fun comes from digital growth, unlocking the rhythm and "the system itself is pushing."</p>
<p>Leisure can also be a play-like state. Minecraft can be played as a red stone project by hard core players, or by leisure players as a builder, a farm and accompaniment. Starbucks can be counted as efficient players or used as a loop of relaxation after work.</p>
<p>Mobile leisure is also often tied together with commercial design. The waiting times, physical strength, accelerators, production slots, daily mission and activity calendars all change the rhythm. If the design is overly dependent on waiting and paying, the player is not allowed to play when he has time to play. That's weird.</p>
<h2>Integration type and modern labels</h2>
<h3>Music, terror, sandboxes, open world.</h3>
<p>These are not much like basic categories, more like the experience labels that are used in modern games. They often cross the broad categories of action, adventure, simulation, strategy.</p>
<p>The core of music games is rhythm, hearing, timing and feedback. The rhythm of the sky, the DJMAX, the DJMAX, the Muse Dash, the Phigros, all require players to convert the music structure into a body input. It can be a street machine, a mainframe, a mobile end or a dedicated controller. It is often singled out because it is too tightly bound to the aesthetic.</p>
<p>Terror is more like an emotional label than a single mechanism. Bio-Crisis' early survival terror, emphasizing scarcity of resources and space oppression; Silent Hills' psychological terror; Escapes' emphasis on inability to escape and pursue; and Chinese-style terror games may be more dependent on folklore, taboos and narratives. Terror can be combined with adventure, movement, puzzle resolution, walk simulation, shooting, so continue to ask: What does it do to scare the player?</p>
<p>Sandboxes emphasize openness, creativity and self-expression by players. The Minecraft, Telaliah, Roblox, explains that sandboxes are not just large maps, but whether players can change, combine, build, destroy or customize the world. The open world emphasizes space openness and sandboxes emphasize the operationalization of systems. The two are often reunified, but not the same.</p>
<p>Open world itself is more like a space structure label. It is about the ability of players to move freely, choose targets and route in a continuous space. The GTA, the Zelda Legend: The Land of the Wild, the Elden Fir Ring, the God of the Oracle, and Forza Horizon can all be understood in this light. Open worlds do not guarantee sandboxes or free narratives; some simply spread their tasks on big maps, while others really make exploration, encounters and system interaction the main pleasures.</p>
<h3>ARPG、A-AVG、SRPG</h3>
<p>The most common type of integration is from cross-cutting basic categories.</p>
<p>ARPG is the combination of action and RPG. Darkness destroys the cycle of equipment, words and brushes; Monster Hunters is biased in his action understanding, weapons proficiency and material growth; the Gods are biased in their role teams, elemental reactions, open world exploration and service-type renewal; and the Elden Ring is biased in its action, construction, map exploration and high penalty challenges.</p>
<p>A-AVG or action takes the risk of combining instant action with adventure exploration. The legend of Zelda, the reflection of the tomb, the Mystery Sea, the little nightmare, is all around this. They are not pure actions, because players also have to read space, find paths, understand scenes; they are not pure adventures, because of the importance of operation and timing.</p>
<p>SRPG combines strategy with RPG. The Flame Sketch, the Triangular Strategy, Dream Simulations, allows role growth, occupation, equipment and drama to be chosen into a chess situation. Players win this game while they train this team.</p>
<p>This kind of integration can actually be broken down by one sentence: looking for the main axis, then the support axis. The main axis answers what players do most of the time, and the support axis answers what system to make it deeper.</p>
<h3>MOBA</h3>
<p>MOBA is a good case of a type evolution. It inherited the RTS maps, the military lines, the resources, the vision, the team strategy and unit skills, but it narrowed the focus of player operations from an entire army to a hero. The small soldiers are moving along the routes, and the players influence the war by growing heroes, releasing skills, equipping choices, map resources and teaming.</p>
<p>DotA has a deep relationship with RTS ' s custom map culture, and later works like The Heroes Alliance Dota 2 " Glory of the King " have made this game product, competitively and platform. It's hard to fit into traditional tree categories because it's both action-oriented, tactical, RPG growth, team communication and competitive pressure.</p>
<p>The Kings Glory also indicates the type of platform that will be rewritten. The complexity of the operation is reduced by the mobile end by the use of virtual rollers, automatic road search, fast application and shorter match-up debris scenarios. It's not simply narrowing PC MOBA, but reorganizing input and rhythm.</p>
<h3>Battle Royale</h3>
<p>The first of these is the great map of the country, the large number of players born, random materials, high death penalties, shrinking safe areas and the final survivors. It can be combined with the shooting in PUBG, Apex Legends, Fort Night, or with parties, movements, close combat or other games.</p>
<p>The most critical design for this type is not "many people hit each other," but the indent. The opening of large maps is prone to loss of rhythm after a reduction in numbers and the players may have been hiding. The shrinking of the security zone kept the space under pressure, forcing the encounter to continue and allowing a game to move from collection, transfer and ambush to final confrontation.</p>
<p>PUBG is a writer of military shooting and survival pressures, Apex Legends adds hero skills and high mobility, and Fortress Night adds attributes to construction, activity and social platforms. They share the Battle Royale structure, but the player skills and aesthetics differ.</p>
<h3>Roguelike / Roguelite</h3>
<p>Roguelike from Rogue in 1980. Traditional Roguelike keywords include process generation, permanent death, turn-back, grid movement, resource identification, dungeon exploration and difficulty. Players die once and for all, not just by painting numbers, but by learning monsters, objects, risks and world rules.</p>
<p>In modern terms, the Roguelite is more common. It retains the cycle of change and death along with the machine cards, but does not necessarily adhere to all the rules of the traditional Roguelike and often joins out-of-house growth. Hades is action, Rogerite, the kill tower is card-built, FTL is spaceship management and random events, Isaac's combination is double-strangled and random prop, and The Snow is a combination of national wind, current formation and repeated challenges.</p>
<p>The core of Roguelike is not “random maps”, but uncertainty, high penalties, short to medium cycles, and the player's understanding of the system through failure. Only random rewards or random copies are not necessarily called Roguelike.</p>
<h3>Soulslike</h3>
<p>Soulslike from The Spirit of the Devil, The Spirit of the Dark, and later from FromSoftware. It usually includes high-punishment death, precision manoeuvres, manual management, campfires or checkpoints, enemy configuration learning, fragmentation narratives and interconnected junction structures.</p>
<p>But Soulslike is not the "hard ARPG." It is difficult to understand only the high value of the enemy, the vulnerability of the player to death, but without learning about enemy action, risk returns, the route of the level and a re-understanding after death. With the entry of Elden Ring into the open world, borders have been widened: players can go around, explore, grow and come back to the challenge.</p>
<p>Domestic players sometimes use “suffering” as a label. That is an image, but not enough. Soulslike really built where death, learning, control and the world's mystery coexist.</p>
<h3>Metroidvania</h3>
<p>Metroidvania comes from the tradition of the Galactic Warriors and the Devil's City. At its core is a continuous, interconnected world in which players unlock new paths to old regions by acquiring new capabilities. It does not advance at a level like a linear platform, nor does it move around the world at the beginning, but rather uses its capabilities, maps, doors and routes to organize its exploration.</p>
<p>The " Sky Knight " O'Day, " The Scares: The Night Ceremony " can be understood from this tradition. Its pleasure is to look back and see the old world change. Not the map has changed, but the player's abilities and understanding have changed. A platform that could not jump, a door that could not be opened, a route that could not be understood, suddenly joined up after gaining the ability to jump, sprint, hook or deform.</p>
<h3>Survival, Crafting and Sandbox</h3>
<p>Survival games emphasize resource scarcity, environmental threats, physical state, construction and long-term risks. Don&#39;t Starve, Ark: Evolution of Survival, Forest Famine, requires players to deal with hunger, temperature, monsters, materials and bases. Survival is not just low blood, but the world will keep consuming players.</p>
<p>Crafting and construction are often tied to survival and can also be expressed in sandboxes. Minecraft can be both alive and creative; Telaria can be both exploratory and built, and action risk and equipment growth. The key depends on where the main pressure of the player comes from: to live or to express itself.</p>
<h3>Tower Defense, 4X and Card Construction</h3>
<p>Towers are a clear branch of strategy. The enemy is moving along the route, and the player is defending through resources, towers, units, barriers, skills and terrain. The botanical zombies, the Ark of Tomorrow, Kingdom Rush, are in different forms of tower defence. The interesting thing about the Ark is that it places towers, role formation, jigsaw puzzles, cards and long-term operations in the same framework.</p>
<p>4X is exploration, expansion, exploitation, conquest. Civilization, Stars, Space Ends, allows players to operate a nation, civilization or interstellar empire over a long period of time. It and RTS are both strategic, but at a different pace: RTS is concerned with real-time operations and local combat, and 4X is more concerned with macro-direction, technology, diplomacy, resources and long-term plans.</p>
<p>Card construction compresses the policy into the card group, probability, resources and round selection. The Story of the Flying Stone, King of the Game: Masters duel, Quint Cards, RPGIET, the Sword of the Killing, the Full Night of the Moon, and the Shaun Yeung Ying, the Phrase of the Shadow, with roles, IP and operational content. The interesting thing about card games is that players not only choose current actions, but also design what they will draw in the future.</p>
<h3>Party, social reasoning, placement and smoking cards.</h3>
<p>Some modern labels are describing not just mechanisms but social scenes or operating structures.</p>
<p>Party games emphasize multiple screens, low thresholds, short play and ridiculous scenes. The Mario party, the kitchen bullshit, the Sugarman, is all for friends' parties or live. Social reasoning such as Along Us, " Duck Kill " , " Wolf Kill " , is not operational, but rather conceals identity, language persuasion, information asymmetry and group psychology.</p>
<p>The placement game emphasizes low-operational, long-term growth and offline gains. Cookie Cricker, Traveling Frog, and the large number of RPGs, have allowed continued feedback from players with low input. They are often growing and numerically, collecting, fighting automatically and forming a combination.</p>
<p>The smoking card is not a pure play-type, but it's important today. It describes role acquisition, resource planning, long-term updating and business models. The God of the Nativity, Crash: Star Dome, the Sunny Man, and the Ark of Tomorrow have different levels of pull cards that form the structure. When analysing such games, only ARPG, turn RPG or tower guard are left out of a significant part of the player ' s experience: role pool, version rhythm, bottoming, resource allocation, team matching and content operation.</p>
<p>This is why modern games are increasingly classified like labels. One label is not enough, just fold it up.</p>
<h2>How does a type usually grow?</h2>
<p>When you look at these types together, you find a few common paths.</p>
<p>Some types are introduced by technology and equipment. The street pole, handle, mouse, touch screen, body feeling, VR, steering wheel can all change the experience. Many games are difficult to establish without suitable input; with new input, old ones can also change faces.</p>
<p>Some types grow out of communities and Mod. The MOBA, Battle Royale, many survival games and UGC platforms all indicate that players create new rules in existing games. Developers sometimes do not invent games in a vacuum, but rather identify what players have played, sort, balance and product.</p>
<p>Some types are named by iconic works. Roguelike, Soulslike, Metroidvania are not full theories before games, but have a strong experience, and then the player needs a word to describe something like it.</p>
<p>Business models can also change types. MMORPG, mobile leisure, pumping cards, season competitions, Battle Pass, not just a fee, can affect the content rhythm, player goals and system design.</p>
<p>Finally, there is a transfer of player skills. FPS players take sight and walk to tactical shooting; RPG players take the construction consciousness to the brush and to the draw cards; and strategists take the resource planning to towers, cards and 4X. The type is integrated, and the player is moving himself.</p>
<h2>When I analyze a game, I'll ask.</h2>
<p>It might be useful to see a game later on, along the following lines.</p>
<ol>
<li><p>What's a player doing most of the time? Which is the highest percentage of combat, exploration, operation, construction, puzzle resolution, collection, socialization, reading?</p>
</li>
<li><p>Where does failure come from? Slow response, misture, inadequate resources, poorly constructed, ununderstanded leads, or failed social judgment?</p>
</li>
<li><p>What's the core cycle? How does a player start, get feedback, grow and come back?</p>
</li>
<li><p>Where does the player grow? Is it role growth, player technological growth, knowledge growth, collection growth, social relationships growth, or aesthetic growth?</p>
</li>
<li><p>What has changed the platform and the input? Does the mouse, handle, touch screen, body feeling, VR determine its rhythm and complexity?</p>
</li>
<li><p>What are the main labels and the by-labels? Main label describes the core experience, sublabel describes the support system. For example, Open World ARPG+ Smuggle Cards for + Service Content is clearer than RPG.</p>
</li>
<li><p>Who's this label working for? The labels that are needed for the recommendation of players, for the design dismantling, for market positioning, for academic research may be different.</p>
</li>
</ol>
<p>Looking back at the first example in this way, the Gods don't need to choose between ARPG and the open world. Its primary experience could be open world action RPG, supported systems that include role collection, elemental reactions, team formation, exploration of puzzles and service-type updates. The Ark of Tomorrow is not just a tower guard, but a combination of towers and barriers, tactical puzzles, role formation, card collection and long-term operation. The King's Glory is not just a mobile Dota, but a re-organized MoBA for touchscreens, short-runs, social communication and mobile competitions.</p>
<p>So the game classification does not end up with a name tag that will never change. It's more like a set of words describing the experience. Tree classification helps us capture history and boundaries, label classification helps us face the hybrid reality of modern games, and a multidimensional framework reminds us that the type of name is backed by mechanisms, skills, platforms and player perceptions.</p>
<p>A game can be a lot of different types at the same time. It's not a classification failure. It's the way video games are. It is enough to say a type name, to know what it is and to know what it is not.</p>
<h2>References</h2>
<ul>
<li>PLOS ONE, <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0299819">The tangled ways to classify games</a></li>
<li>Robin Hunicke, Marc LeBlanc, Robert Zubek, <a href="https://users.cs.northwestern.edu/~hunicke/MDA.pdf">MDA: A Formal Approach to Game Design and Game Research</a></li>
<li>Steamworks Documentation, <a href="https://partner.steamgames.com/doc/store/tags">Tags and Genres</a></li>
<li>Game Developer, <a href="https://www.gamedeveloper.com/design/postmortem-i-defense-of-the-ancients-i-">Postmortem: Defense of the Ancients</a></li>
<li>RogueBasin, <a href="https://www.roguebasin.com/index.php/Berlin_Interpretation">Berlin Interpretation</a></li>
<li>ESPN, <a href="https://www.espn.com/gaming/story/_/id/29364632/how-made-how-brendan-greene-pubg-revolutionized-gaming">How Brendan Greene&#39;s PUBG revolutionized gaming</a></li>
</ul>
