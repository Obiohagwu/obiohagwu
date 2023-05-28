---
title: Building out a mini-compute cluster
date: 2023-04-25
---

Ok, to start off, I'm not entirely sure that making htis public is a good idea goven the precariousness that GPUs might inhabit in the near future as a result of the porliferation of really advanced autonomous systems.

But whatever, lol, I'm still going to document to publicly document the build, as it might be helpful to others.

Lets start with the most important component, the GPUs:
So I decided to go with the RTX 3090s as the base GPUs. My primary heuristic for this selection was available flops per dollar. Comparing the 3090 to other GPUs with comparable VRAM (24GB), the 3090 is a superior option to both the A6000/A5000 and the current 4090.


Depending on your needs, you'd want either a server set-up or a workstation setup. Usually, servers are set-ups are ideal for >=4GPUs. 



Workstations are ideal for a smaller number of computers with less heating requirements. For such systems, an AMD threadripper CPU is good. As opposed to the EPYC line with higher bandwith interconnects and higher core/thread count, the threadripper is generally lower and this doesn't necessarily negate performance for most users, but in our case it might. Althoug, if your'e only runing on <3GPUs you don't need much core/thread count. 

The threadripper series also has lower RAM capacity at 128GB, as opposed to EPYC 7002/7001 series (7502 in my case).

Given that I am planning to build-up a server cluster, I will have to go with a server CPU, and accompanying motherboard. Given that I chose the 7502 cpu, I will have to go with a rack/server cpu. The best I have seen from multiple sources is the ASRock rack ROMED8-2T board.



Given that I am planning to build-up a server cluster, I will have to go with a server CPU, and accompanying motherboard. Given that I chose the 7502 cpu, I will have to go with a rack/server cpu. The best I have seen from multiple sources is the ASRock rack ROMED8-2T board.



**Breakdown**
- 4x Nvidia RTX 3090 
- Motherboard: ASRock rack ROMED8-2T
- CPU: AMD EPYC 7502 (node unlocked version)
- Power Supply Unit (PSU): 1600W Corsair
- RAM/Memory: 128GB Samsun RAM sticks