---
title: Building out a mini-compute cluster
date: 2023-04-25
---

Ok, to start off, I'm not entirely sure that making htis public is a good idea goven the precariousness that GPUs might inhabit in the near future as a result of the porliferation of really advanced autonomous systems.

But whatever, lol, I'm still going to document to publicly document the build, as it might be helpful to others.

Lets start with the most important component, the GPUs:
So I decided to go with the RTX 3090s as the base GPUs. My primary heuristic for this selection was available flops per dollar. Comaparing the 3090 to other GPUs woth coparable VRAM (24GB), the 3090 is a superior option to both the A6000/A5000 and the current 4090.
