# I Spent $150 on GPUs to Find Out if SSMs Can Make Music

Ok, so I don't really have a lab. Also no funding(quite the contrary, very broke haha), or research advisor (undergrad), or like any institutional affiliation. What I have though, is a runpod account, a soundcloud library full of high-tempo grungy groovy house music, and a question that nobody seemed to have answered yet.

Namely; Can a hybrid Mamba-Attention model generate music as well as a Transformer?

Generally speaking, it seems like most in the music gen space use Transformers. MusicGen, MusicLM, JEN-1 etc, the entire paradigm is built on attention. And for good reason too. Like it seems obvious that self-attention allows for phenomenomal long range cohenrence as well as holding fine detail structure. such is the nature of any autoregressive mechanism with a comparative codebook. It maintiains local context very well. But it's also expensive. Like quadratically expensive, in fact, which means that generating longer pieces of music gets painful pretty fast.

Then along comes Mamba, a state space model that processes sequences in linear time. It's faster, leaner, and it's been tearing through NLP benchmarks. Although there is a small caveat, and the main reason we are trying this jamba style hybrid approach over 9-codebooks. A paper called SMDIM found that when you swap out attention for Mamba entirely, you lose local harmonic detail. The chords get muddy and the melodies tend to blur. The thing that makes music sound like *music* starts to slip away.

So I had an idea. What if you don't go all-in on Mamba? What if you keep just enough attention to handle the fine-grained harmonic stuff, and let Mamba do what it's thing with tracking long-range global structure? A hybrid. And I wanted to find out if it actually works.

---

## What Even Is Mamba or SSM?

If you already know what state space models are, skip ahead. If not, here's the short version.

**Transformers** work by having every token in a sequence attend to every other token. When you're generating the 500th token, it looks back at all 499 previous tokens to decide what comes next. This is a pretty obviously powerful method. The model can directly reference any point in the past. But the cost scales as O(n^2) with sequence length. Double the sequence, quadruple the compute.

**Mamba** (and state space models more broadly) takes a fundamentally different approach. Instead of looking at the entire history every time, it processes tokens one by one and maintains a compressed "state" that summarizes everything it's seen so far. This state gets updated with each new token. The cost? O(n). Linear. Double the sequence, double the compute. That's it.

Here's an analogy:

The Transformer approach is like re-reading the entire manuscript every time you write a new sentence. Sure, you'll catch every callback and thematic thread, but it gets pretty slow by chapter 40.

The Mamba approach is like keeping a running set of notes containing character arcs, plot threads, the current emotional tone, and consulting those notes when you write the next sentence. It is definitely faster, but you might miss a subtle detail from page 12 that your notes didn't capture.

The trade-off is real. For music, that "subtle detail from page 12" might be the key of the song, or a chord voicing that should resolve a certain way. That's exactly what the SMDIM paper warned about.

I also like to think of it as mamba being a more locally lossy compressor than transformers, specifically in representation deecoding space if that makes sense.

---

## The Jamba Insight

This is where things get interesting.

AI21 Labs published a model called Jamba. It's a large language model that mixes Mamba layers with Transformer attention layers. Their key finding was striking: you don't need *all* attention or *all* Mamba. In their experiments, even a single attention layer out of every eight was enough to restore the in-context learning capabilities that pure Mamba lost.

One out of eight. That's it.

For music generation, this finding is tantalizing. Think about what different parts of music require:

- **Harmonics and chords** are local patterns. A chord is a few notes sounding together *right now*. Attention, which excels at modeling precise relationships between nearby elements, seems tailor-made for this.
- **Song structure** is a long-range pattern. The fact that a chorus should return after the verse, or that tension built 30 seconds ago should resolve now is usually where Mamba's efficient long-range memory could shine.

What if you could get the best of both? Mamba for the big picture, a sprinkle of attention for the harmonic glue?

That's the hypothesis I'm trying to test here.

---

## My Setup

Let me walk through what the actual setup i have.

### The Codec

Raw audio is a terrible thing to model directly. At like CD quality, you're looking at 44,100 samples per second. Nobody models that autoregressively. Like any sort of compute, you have to discretize based on an ideal sample rate. Instead, you use a neural audio codec to compress the audio into discrete tokens, just like how language models work over word tokens.

I used **DAC** (Descript Audio Codec), which converts 44.1kHz audio into 9 parallel streams of discrete tokens at about 86 tokens per second. Each stream is called a "codebook," and each token in a codebook is one of 1,024 possible codes. The first codebook captures the broad strokes (overall pitch, loudness), and each subsequent codebook adds finer detail (timbre, texture, noise).

To handle all 9 codebooks being predicted simultaneously, I used a MusicGen-style **delay pattern** where each codebook is offset by one timestep from the previous one, so the model can predict all 9 in parallel rather than sequentially. This is a well-established trick from Meta's MusicGen.

### The Dataset

About 108 hours of electronic music total:
- **25 hours** from my own SoundCloud collection. Groovy house, mostly. The kind of stuff you'd want playing at a rooftop bar at sunset.
- **83 hours** from the Free Music Archive, filtered for electronic genres.

108 hours is small by industry standards. MusicGen trained on 20,000 hours. But for a controlled experiment comparing architectures, it's enough. Both models see the exact same data.

### The Models

Here's the head-to-head:

| | **Transformer** | **Hybrid 1:3** |
|---|---|---|
| Params | 162M | 116M |
| Layers | 20 | 20 (15 Mamba + 5 Attention) |
| Architecture | Pure causal attention | Interleaved: 3 Mamba, 1 Attention, repeat |
| d_model | 768 | 768 |
| Heads | 12 | 12 (in attention layers) |

Both models share the same embedding and output architecture, per-codebook embeddings summed into a shared representation, with per-codebook output heads. Same optimizer (AdamW), same learning rate schedule (cosine with warmup), same batch size, same everything. The *only* difference is what happens in those 20 layers.

The hybrid interleaves the layers in a repeating pattern: three Mamba blocks, then one attention block, three Mamba blocks, one attention block, and so on. Out of 20 layers, 15 are Mamba and 5 are attention. The Mamba blocks use the original Mamba-1 architecture with selective state spaces -- d_state of 64, a local convolution width of 4, and an expansion factor of 2.

### The Compute

Everything ran on a single A100 80GB GPU rented through RunPod. My total cost: roughly $150. quite a lot of money for me tbh.

---

## The Gap I'm Filling

Let me be clear about what's been done before and what hasn't.

Lee et al. (2026) published SSM-TTM, testing Mamba-2 for music generation. Solid work, but they used cross-attention for text conditioning and only modeled 4 codebooks. Their architecture is different from what I'm doing here.

Nobody, as far as I can tell, has tested **Mamba-1 in a Jamba-style interleaved hybrid for music generation**. And nobody has done this with **all 9 DAC codebooks**, which means full audio quality without dropping the fine detail that codebooks 5-9 capture.

That's the gap. It is narrow and also quite specific, and it's exactly the kind of thing a solo researcher with a clear question can address.

---

## Results

*[RESULTS PENDING -- training is in progress]*

This section will be updated once both models have finished training and evaluation is complete. Here's what I'll be reporting:

### Training Curves
- Loss over time for both models
- Whether the hybrid converges faster, slower, or comparably
- Any signs of instability in the hybrid training

### Generated Audio Quality
- Side-by-side audio samples (link to listening page)
- Fr&eacute;chet Audio Distance (FAD) scores using VGGish embeddings which the standard automated metric for generative audio quality
- 256 generated samples per model, 10 seconds each

### Per-Aspect Breakdown
- **Rhythm**: Does the groove hold? Does the kick pattern stay consistent?
- **Harmony**: Do chord progressions make sense? Do notes clash?
- **Structure**: Is there any sense of sections, builds, drops?
- **Texture**: How does the timbre quality compare? (This is where codebooks 5-9 matter)

### The Efficiency Angle
The hybrid model has 28% fewer parameters than the Transformer (116M vs 162M). If it even *matches* the Transformer's quality, that's still a win in my opinion. we're getting comparable results with a cheaper model. If it *beats* the Transformer... lmfao.

---

## Honest Limitations

I believe in being upfront about what this experiment can and can't tell us. Here's what keeps me up at night:

**The parameter mismatch.** The Transformer has 162M parameters. The Hybrid has 116M. That's not an apples-to-apples comparison. Mamba layers are inherently more parameter-efficient than attention layers (no QKV projections), so matching the layer count gives you fewer total params. A truly fair comparison would train a 116M Transformer alongside the 116M Hybrid, or scale the Hybrid up to 162M. That's on the to-do list.

**Single seed.** I ran each model once with seed 42. Best practice is at least 3 runs with different seeds to account for training variance. But 3 runs means 3x the cost, and I'm already stretching my budget.

**Small dataset.** 108 hours sounds like a lot until you learn that MusicGen trained on 20,000 hours. My models might be data-limited in ways that mask architectural differences.

**No human evaluation.** FAD scores tell you something, but they're not the whole story. A proper evaluation would have human listeners rate the samples for quality, musicality, and preference. I'm one person. I can't run a statistically valid listening study by myself.

**I'm one person with $150, not a research lab with $150K.** This is an exploration, not a definitive answer. I'm trying to plant a flag that says "this direction is worth investigating further," not claiming I've solved music generation.

---

## What I Learned

*[PENDING -- personal reflections after seeing results]*

This section will cover:
- Technical lessons from implementing Mamba-1 for audio tokens
- What surprised me about the training dynamics
- What I'd do differently if I had more compute
- The experience of doing ML research as a solo practitioner

---

## What's Next

Regardless of how the results shake out, there's a clear roadmap for follow-up work:

1. **Parameter-matched comparison.** Train a Transformer with the same 116M parameter budget as the Hybrid. This isolates the architecture question from the scale question.

2. **Pure Mamba baseline.** I have the code for this already. Adding a pure Mamba model would let me quantify exactly how much the attention layers contribute. Is the hybrid better than pure Mamba? By how much?

3. **Text-conditioned generation.** Right now, both models do unconditional generation. They just produce music with no text prompt. Adding text conditioning (like "upbeat house track at 124 BPM") would make this more practically useful and open up comparison with systems like MusicGen.

4. **More data.** 108 hours is a proof of concept. Scaling to 500+ hours and including more diverse genres would test whether these architectural findings generalize.

5. **Workshop paper.** If the results are interesting, and I think they will be, regardless of which direction they go -- I'll write this up for a workshop at ICML or ISMIR.

---

## Closing Thoughts

Here's what I want you to take away from this, even before the results are in.

It's good to have money.

The entire infrastructure for independent ML research exists right now. Just too poor at the moment.

I'll update this post when the results are in. In the meantime, the code is available in the repo, and the technical report goes into significantly more detail on the architecture and methodology.

*[Link to code repo -- coming soon]*
*[Link to technical report -- coming soon]*

---

*If you have thoughts, questions, or suggestions for follow-up experiments, I'd genuinely love to hear them. The best part of doing research in public is that someone smarter than you usually shows up with a better idea.*
