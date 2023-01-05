---
title: Notes on speech from signal processing perspective
date: 2022-12-28
---

The ideal multimodal agent would integrate an audio processing system. I decided to share some notes on speech,starting from a signal procesing perspective. 


- Talk about how to sample a wave form, period, sampling frequency, and FFTs etc. 

Ah, yes. Information! Don't we all love information? from the tiniest level of organization (neucleotide sequence in DNA) to the high bitrate transfer of bitstreams between different memory compartments in a classica Von Neumman architected computer. Information is akin to manna, analogized over a metallo-silicon substrate.
Today thiough, i'd like to dive into a specific form of information, sound. Sound is a pretty weird one because it's not at all a tangible physical quantity, but it is vvery much ubiquitious in our physical system. Starting from the definitation of waves, and fields, we can define "sound" as a the phenomenon produced by pertubations of the air, ie vibrations propagated through the air, like any other wave over it's medium. 

To approach this from a more information theoretic pov, 










- Move on to short time fourier transforms, and how that works, and why to use it in place of canonical FFTs.



- Move on to mel spectrograms, and how they can be decoded.




- Do a thorough account of Vocoders (nerual vocoders). Talk about different backbones and their advantages and drawbacks

#### *References*
- *[DSPGAN: a GAN-based universal vocoder for high-fidelity TTS by time-frequency domain supervision from DSP](https://arxiv.org/abs/2211.01087)*
- *[SpecGrad: Diffusion Probabilistic Model based Neural Vocoder with Adaptive Noise Spectral Shaping](https://arxiv.org/abs/2203.16749)*
- *[TorToiSe Architectural Design Doc](https://nonint.com/2022/04/25/tortoise-architectural-design-doc/)*
- *[CMU - Notes on Short-Time Fourier Transforms](https://course.ece.cmu.edu/~ece491/lectures/L25/STFT_Notes_ADSP.pdf)*
