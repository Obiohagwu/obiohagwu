---
title: Notes on speech from signal processing perspective
date: 2022-12-28
---

The ideal multimodal agent would integrate an audio processing system. I decided to share some notes on speech,starting from a signal procesing perspective. 

**NOTE: This is a very high-level overview. Will use minial math notation.**

- Talk about how to sample a wave form, period, sampling frequency, and FFTs etc. 

Ah, yes. Information! Don't we all love information? from the tiniest level of organization (neucleotide sequence in DNA) to the high bitrate transfer of bitstreams between different memory compartments in a classical Von Neumman architected computer. Information is akin to manna, analogized over a metallo-silicon substrate.
Today though, i'd like to dive into a specific form of information, sound. Sound is a pretty weird one because it's not at all a tangible physical quantity, but it is very much ubiquitious in our physical system. Starting from the definitation of waves, and fields, we can define "sound" as a the phenomenon produced by pertubations of the air, ie vibrations propagated through the air, like any other wave over it's medium. 

To approach this from a more information theoretic persperctive, we call upon the inital information propagation pipeline as denoted by Shannon:

- Let the *source* S, be the set containing the list of states that one can conceive as letters of a primitive alphabet.
- Let the *encoder* E, be the encoding function that transforms the written or spoken message into bits of information to be transmitted.
- Let the *channel* CH, be the medium by which said encoded signal passes.
- Let *receiver* R, be the decoding function that decompresses the encoded signal to be received by the destination.
- Let the destination D, be the entity receiving said message.

<p align="center">
    <img width="467" alt="Screen Shot 2022-08-08 at 10 12 11 AM" src="https://user-images.githubusercontent.com/73560826/194781153-bc4237f3-39af-459b-8887-86a4a6bccc98.png">
</p>

> Figure 1. A Canonical information propagation system as denoted by Shannon & Weaver in "The mathematical theory of communication"

To return to more applicable means. In the feild of signal processing, primarily, digital signal processing, a common and somewhat indispensable method is that of signal transformation between different domains (ie, frequency, time, amplitude, etc). Generally speaking, input signals of any form are usually exist in a continious form, meaning that in order to use such an input signal at any capacity, we would have to initally convert it into a dicsrete repesenatation. This is where sampling a continous signal somes into play!

We all know what sampling is. Sweet sweet sampling. I'm sorry, I just really love any sort of sparsification or compression type derivative. In this case, given the set of all points in the continous input signal, we discriminately select samples of the set by a sampling criterion denoted by 1/Fs (period), where Fs is the sampling frequency.
More formally: 

let the raw input signal be defined as X(t), where t is the time variable. Using the sampling criterion, we collect n samples from X(t) at a rate of Fs per second. The output of this gives us our number of sampled points (which, again is quite heavily dependent on the sampling frequency used. Will talk more about this as we proceed).
Then we proceed to normalize our samped points in time with Ts = 1/Fs to obtain the discretized version X[n], were n is the number of samples, of the inital continous raw input. Finally, with this discretized version of the inital input, we can work our digital magic on it!

Ok, now we can get inrto the meat of things. We've now seen how to convert a raw continous input signal into a discretized version for better computation over a digital substrate, now we look at the algorithms that actually allow us to process X[n]. In the space of dicrete signal processing - or more specifically, Fourier analysis on discrete-time signals, we have these fourier transforms called DFT (Discrete Fourier Transform), and DTFT (Discrete Time Fourier Transform).

Canonically, the DFTF of a raw input signal undergoes two intermediary transformation before DFT being computed. 

We denote the DTFT of an input sequence, X[n] to be:

<p align="center">
    <img alt="Screen Shot 2022-08-08 at 10 12 11 AM" src="https://user-images.githubusercontent.com/73560826/211385211-24230aff-d956-41f2-bac1-fedabaaeec62.svg">
</p>

where ![CodeCogsEqn (26)](https://user-images.githubusercontent.com/73560826/211385745-54f9bd27-b836-40b4-ae15-da86e3ad14b2.svg) denotes the raw input signal as a continous function of omega. So as we saw above, we have to sample points on ![CodeCogsEqn (26)](https://user-images.githubusercontent.com/73560826/211385745-54f9bd27-b836-40b4-ae15-da86e3ad14b2.svg) based on the sampling criteria wee defined above, 1/Fs. Recall that we do such a samplig to discretice inital signal. Note that sampling of raw signal occurs in frequency domain.









- Move on to short time fourier transforms, and how that works, and why to use it in place of canonical FFTs.



- Move on to mel spectrograms, and how they can be decoded.




- Do a thorough account of Vocoders (nerual vocoders). Talk about different backbones and their advantages and drawbacks

#### *References*
- *[DSPGAN: a GAN-based universal vocoder for high-fidelity TTS by time-frequency domain supervision from DSP](https://arxiv.org/abs/2211.01087)*
- *[SpecGrad: Diffusion Probabilistic Model based Neural Vocoder with Adaptive Noise Spectral Shaping](https://arxiv.org/abs/2203.16749)*
- *[TorToiSe Architectural Design Doc](https://nonint.com/2022/04/25/tortoise-architectural-design-doc/)*
- *[CMU - Notes on Short-Time Fourier Transforms](https://course.ece.cmu.edu/~ece491/lectures/L25/STFT_Notes_ADSP.pdf)*
