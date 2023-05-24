---
title: Consumer llms are coming (here?)
date: 2023-04-24
---

Consumer llms, which I define as LLMS that are at least as good as GPT3.5 (specify which), and able to run inference, finetune, and hopefully (more superflous goal for now) train, on consumer-grade hardware (which I define as non-cluster [<200GB of VRAM, and <# of flops used to finetune or train(if necessary [seeming less obviously required]) >1trillion token and/or 65B-120Bparam models]).

What is prompting this sudden realization (had thought about it before but assumed atleast a bit away) was the flurry of newer - more effective n-bit weight quantization techniques. Without going too deep into the specifics of n-bit weight quantization, I feel most of you here will do well with the preliminary analogy of quasi-losless compression. Yeah, operate with that anology for a sec so we can finish the post.

It started more recently around februaury of 2023, when @ggreganov realeased his github repo of the n-bit quanitized version of the leaked LLaMa weights from Meta AI research (Thanks again to meta for seeming to be the most opensource so far). The crux of the quantized weights was itself releasd