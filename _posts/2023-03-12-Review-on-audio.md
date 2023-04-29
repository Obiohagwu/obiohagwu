---
title: Review on audio generative systems
date: 2023-02-11
---

## Introduction

The past decade has seen immense gains, as well as dramatic breakthroughs in the space of arbitrary audio synthesis
conditioned on text-based inputs. More specifically, text-tospeech systems, as well as text-to-music have seen commendable improvements, primarily powered by algorithmic
advances in the space of neural networks and end-to-end
probabilistic modeling. In this paper, Advances in Audio
Generation: From Sequence Models to DDPMs, we provide
a survey of the underlying mechanisms of such systems, as
well as the likely negative impacts of unabated use of these
models, and solutions to minimize such instances. Generally
speaking, TTS systems, as of recent times, operate under two
fairly distinct paradigms in terms of how they are architected;
namely, the cascaded text-to-speech (TTS) systems [Shen et
al., 2018, Ren et al., 2019, Li et al., 2019], which primarily
leverages pipelines having acoustic models and a vocoder
using a mel spectrogram as an intermediate representation
to be learned, that would map to an ideal waveform corresponding to the desired output. We also now have text-tospeech (TTS) systems that are based on sequence modeling
of discretized audio codec codes based on phenome and
acoustic code prompts that map onto the target content and
the input speakers’ voice [Brown et al., 2020]. The emergence
of Denoising Diffusion probabilistic models (DDPMs) has
evoked what some might call a renaissance in the space of
generative machine learning. Initially, the positive effects of
DDPMs have been thought to be constrained to the realm
of generative sequence modeling and image generation, but
more recent works like Audio-diffusion have shown great improvements in the output fidelity of their DDPM UNET
based upsampler and downsampler block [4]. In subsequent
sections of this paper, we investigate, on a deeper - more
technical plane, how these paradigms of text-to-speech (TTS)
systems function, as well as Audio-diffusion, and how best to
leverage their fairly distinct architectures in order to provide
failsafes and contingencies in order to minimize the likelihood
of such systems being used for malfeasant acts.

## System Overview

TortoiseTTS, from an architectural perspective, can be conceived as five separately trained neural networks that have been
pipe-lined together to achieve high-fidelity audio few-shot
outputs conditioned on textual input. It includes, as its subnetworks an autoregressive decoder based on GPT(Generative
Pre-Trained Transformer), CLVP and CVVP(contrastive voicevoice pre-training) embedding models, a DDPM-based decoder, and, finally, a Univnet-based neural vocoder. These
sub-networks in conjunction allow for relatively high-fidelity
speech audio outputs based on few-shot inputs.
In multiple ways, the open-source text-to-speech system,
AudioDiffusion is similar in architectural design to TortoiseTTS, the primary caveat though is in the diffusionbased upsampler applied to the waveform, as well as the
UNet-based vocoder for decoding mel-spectrograms into highfidelity music output [4].
Now, we arrive at the final sort of text-to-speech architecture
; the Large language model(LLM) based systems. These are
the latest architectural paradigm being explored as viable
methods for high-fidelity speech production given minimal
input examples (few/zero-shot systems). The main caveat with
such Language-model based text-to-to speech systems is in
the intermediate representation used to train the sub-neural
networks involved; namely, the use of an audio codec as that
intermediate representation [Wang et al., 2020], as opposed to
a mel-spectrogram as is the case in cascaded text-to-speechsystems, such as TortoiseTTS.

### Cascaded text-to-speech(TTS) systems for zero-shot TTS

TortoiseTTS has as one of its underlying network modules,
a decoder-only generative pre-trained transformer (GPT) network. This submodule is used primarily as an autoregressive
decoder and is the crux of much of the functions of the
system. The system is based on a generative model that uses
autoregressive decoding to produce highly-compressed audio
data from text inputs and reference clips. The components
of the Tortoise system work together to generate naturalsounding, high-fidelity speech outputs that most closely match
the input text and reference audio it’s being conditioned on.
The autoregressive decoder is the core component of the
Tortoise system. It takes in text inputs and reference clips and
generates latents and corresponding token codes that represent
highly-compressed audio data. The latents are then used by
the diffusion decoder to produce a MEL spectrogram that
represents the speech output.
To generate natural-sounding speech, the Tortoise system
uses a nucleus sampling decoding strategy. This means that the
system generates multiple ”candidate” latents for each input
text and reference clip. The system then uses the CLVP and
CVVP models to select the best candidate.
The CLVP model produces a similarity score between the
input text and each candidate code sequence. The CVVP
model produces a similarity score between the reference
clips and each candidate. The two similarity scores are then
combined with a weighting provided by the Tortoise user. This
allows the system to choose the candidate with the highest total
similarity to proceed to the next step.
Once the candidate has been selected, the diffusion decoder
consumes the autoregressive latents and the reference clips to
produce a MEL spectrogram representing the speech output.
Finally, a Univnet vocoder is used to transform the MEL
spectrogram into actual waveform data that can be played back
as speech.
Overall, the Tortoise system is a highly sophisticated text-tospeech system that uses advanced machine learning techniques
to produce natural-sounding, high-fidelity speech output that
would most closely match the input text and reference audio. The system’s autoregressive decoding, nucleus sampling,
CLVP, CVVP, and diffusion decoder components all work
together seamlessly as subnetworks or cascaded networks
to create a highly effective and efficient speech synthesis
pipeline.

### Cascaded text-to-music(TTM) systems for zero-shot TTA

This section focuses on the underlying mechanism of Audio
Diffusion based systems. Text-to-audio (TTA) systems are a
technology that has recently gained attention due to their seemingly remarkable ability to synthesize high-fidelity general
audio output based on text descriptions as input. Although,
it must be noted that, prior studies in the space of TTA
have reported high-frequency instances of limited generation
quality despite the high computational costs required. This
is where the novel text-to-audio (TTA) system named AudioLDM comes in.
AudioLDM is a more recent - novel text-to-audio (TTA)
system built on a latent space, which means that it learns
continuous audio representations from contrastive languageaudio pretraining (CLAP) latents. By using pre-trained CLAP 
** IMGS**
models, we can train LDMs (latent discriminative models)
with audio embedding and text embedding as a condition
during sampling. This allows AudioLDM to learn the latent
representations of audio signals and their compositions without
modeling explicitly the cross-modal relationship, which makes
it advantageous in both generation quality and computational
efficiency.
In addition, AudioLDM is trained on AudioCaps with a
single GPU, yet it achieves state-of-the-art TTA performance
measured by both objective and subjective metrics such as the
Frechet distance. What’s more, AudioLDM is the first TTA
system that enables various text-guided audio manipulations,
such as style transfer, in a zero-shot fashion.
Overall, the proposed AudioLDM system presents a significant improvement over previous TTA systems, as it achieves
both high-fidelity generation quality and computational efficiency. Furthermore, its ability to perform text-guided audio
manipulations in a zero-shot fashion makes it a highly versatile
tool that could be used in various applications, such as virtual
assistants, audiobook production, and speech therapy, among
others.

### LLM based systems

For this section on language model-based zero-shot text-tospeech system, we will use as our case of study, VALL-E, a
zero-shot high-fidelity text-to-speech system that is predicated
on using audio codec tokens for intermediate representations,
as opposed to mel-spectrograms in cascaded architectures.
We can properly conceive VALL-E, then as a neural codec
language model such that the neural network tokenizes the
input speech and proceeds to use intermediary networks to
use those output tokens to build waveforms that correspond
to the voice of the speaker, including keeping the speaker’s
timbre and emotional tone.