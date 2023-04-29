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