---
title: Consumer llms are coming (here?)
date: 2023-04-24
---

Consumer llms, which I define as LLMS that are at least as good as GPT3.5 (specify which), and able to run inference, finetune, and hopefully (more superflous goal for now) train, on consumer-grade hardware (which I define as non-cluster [<200GB of VRAM, and <# of flops used to finetune or train(if necessary [seeming less obviously required]) >1trillion token and/or 65B-120Bparam models]).

What is prompting this sudden realization (had thought about it before but assumed atleast a bit away) was the flurry of newer - more effective n-bit weight quantization techniques. Without going too deep into the specifics of n-bit weight quantization, I feel most of you here will do well with the preliminary analogy of quasi-losless compression. Yeah, operate with that anology for a sec so we can finish the post.

It started more recently around februaury of 2023, when @ggreganov realeased his github repo of the n-bit quanitized version of the leaked LLaMa weights from Meta AI research (Thanks again to meta for seeming to be the most opensource so far). The crux of the quantized weights was to reduce the bits per weight in effect making models lighter. A common downside of convetional qunatization techniques has been the loss in precision that usually accompanies it when going from say FP16 to int4 or somethiing like that. The ideal scenario would be that we can trim/prune the bits per weight, while still holding on to the underlying precision level of unquantized models. 

As of may 23rd 2023, Tim dettemers, Artidoro Pagnoni and Ari Hotlzman co-authored a paper called "QLoRa: Efficient Finetuning of Quantized LLMs". I belive this paper wa a divergence point in what i thought would be possible with LLMs in the near future. It's one thing to be able to quantize a model,. it's another thing to be ab le to efficently finetune said quantized model, while still preserving the precision. QLoRA promises efficient finetuning on a single 48GB GPU, while preserving full 16-bit precision level and finetuning task performance. 

