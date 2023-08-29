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




Given that I am planning to build-up a server cluster, I will have to go with a server CPU, and accompanying motherboard. Given that I chose the 7502 cpu, I will have to go with a rack/server cpu. The best I have seen from multiple sources is the ASRock rack ROMED8-2T board. Kudos to jbetker for being the primary inspiration behind the build. His blog is generally informative for anyone interested in deep learning engineering research.
Anyways, given that I plan to get the server to at 4GPUs eventually, (8-peak), a server sertup that provided a motherboard with multiple (>6 x16 PCIe ports). ROMED8-2T supports 7 16x PCIe. I also needed a mother



**Breakdown**
- 4x Nvidia RTX 3090 
- Motherboard: Supermicro H11SSL-i [decicded on the Supermicro board)]
- CPU: AMD EPYC 7401 2.0GHz 24 cores, 48 threads on 7nm (node unlocked version)
- Power Supply Unit (PSU): 1600W Corsair
- RAM/Memory: micron 2133p DDR4 128GB + 1TB Samsung Evo Internal SSD


**Motherboard**
So, I decided on the Supermicro H11SSL-1 motherboard for a wide range of reasons. First off, the price point of these from 2nd hand sellers from deprecated Shenzhen datacenters is quite literally unbeatable.
I was able to pick this up, along with the AMD EPYC CPU for $ 320 USD.

Here is an accompanying photo: 

![Supermicro board](https://user-images.githubusercontent.com/73560826/11c2a15b-8bc2-4bb1-9850-8624311b6217.png)


This motherboard comes with: 
-  3 PCI-E 3.0 x16
- 3 PCI-E 3.0 x8
- M.2 interface: 1 PCI-E 3.0 x4 for an internal SSD (Samsung Evo 870)
- up to 1TB Registered ECC DDR4 2666MHz SDRAM in 8 DIMMs
- Up to 5 USB 3.0 ports (2 rear + 2 via header + 1 Type A)
- Up to 4 USB 2.0 ports (2 rear + 2 via header)







