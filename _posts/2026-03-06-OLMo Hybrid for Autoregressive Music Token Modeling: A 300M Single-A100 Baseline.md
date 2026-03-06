Something really cool happened yesterday.

It's really convenient that while i was working on alternate architecures for music generation that are subquadratic, olmo hybrid just gets dropped.

The funny weird cool thing is we seem to have had similar (maybe obvious?) intuitions about the required architectural tweaks haha, although they are much more comptetent and serious about their research and implementation.

But yeah, their olmo7bhybrid model is not necessarily for music, but the subquadratic scaling that the architecure the model uses is really quite promising as a decoder for my DAC tokens.

I'm really excited to break ground on this.

Here is a more thorough view of my intentions:

Long-context music generation lives in an awkward middle ground for sequence models.

On one end, full attention is dependable and expressive, but expensive. On the other, purely recurrent models are efficient, but they can blur local detail or struggle to preserve the exact interactions that matter for music. For autoregressive music token modeling, that tradeoff matters a lot: music depends on both fine local structure and longer-range organization.

This post attempts to describes a new baseline I added to my existing music language modeling codebase: an **OLMo-hybrid-style decoder for DAC-token music generation**. The model will use a repeating **3:1 layer schedule**, three DeltaNet-style recurrent blocks followed by one full attention block—and combines that with **RoPE**, **per-head QK RMS normalization**, **SwiGLU feed-forwards**, **grouped-query attention**, and **PyTorch scaled dot-product attention** so it can take advantage of FlashAttention on A100 hardware.

The instantiated model lands at **294.3M parameters** and is designed as a **single-A100 baseline** for unconditional music modeling on **44.1 kHz DAC tokens**.

This is not a results post, lol. I'm posting this moreso as a methdology: the goal is to define the architecture, training configuration, and evaluation plan for a new baseline in the repository.

## Why this architecture may be really useful for music

Autoregressive music token modeling needs to preserve local acoustic structure while also carrying information across long spans.

With DAC tokens, a 24-second clip turns into roughly **2,000 autoregressive steps** once you apply a MusicGen-style delay pattern over **9 RVQ codebooks**. That is long enough for quadratic attention to become costly, but not so long that abandoning exact attention entirely feels justified (PAIN!).

That makes hybrid architectures a lot more appealing from an efficiency point of view as we've already covered in prior experiments.

Recent OLMo Hybrid models suggest a useful compromise: let recurrent sequence layers do most of the work, but insert periodic full-attention layers so the model can refresh exact token-to-token interactions. For music, that seems especially natural. Musical sequences contain local events—chord color, rhythmic edges, timbral transitions—but also slower-moving structure such as motif reuse, phrasing, and section-level form.

The goal here is simple: define a practical **300M-class hybrid baseline** that can be trained on a single A100 and compared against the transformer, Mamba, hybrid, and memory-caching baselines already present in the codebase.

## Tokenization

Audio is represented using **Descript Audio Codec (DAC)** at **44.1 kHz**.

The codec emits **9 RVQ codebooks** at roughly **86 frames per second**, with a **codebook size of 1024**. As in MusicGen, the model uses a **delay pattern** that offsets codebook (k) by (k) steps, allowing all codebooks to be predicted in parallel within each autoregressive step.

The final vocabulary size is **1027**, after adding **pad**, **BOS**, and **EOS** tokens. During training, sequences are truncated to a maximum length of **2048 timesteps**.

## The model

The architecture added to the repository is called `olmo_hybrid`.

At a high level, it is an OLMo-hybrid-style decoder with the following design:

* a repeating **3:1 block schedule**, with three recurrent DeltaNet-style blocks followed by one full attention block
* **RoPE** in attention layers
* **per-head RMS normalization** on queries and keys before attention
* **SwiGLU** feed-forward layers
* **grouped-query attention (GQA)** with separate query-head and KV-head counts
* **PyTorch scaled dot-product attention**, enabling the FlashAttention fast path on A100s when masks are full-length
* **no learned positional embeddings** on this path

Each DeltaNet-style recurrent block uses a standard pre-norm residual structure:

1. RMS normalization on the input
2. a recurrent mixer with learned forget and update gates
3. a residual connection
4. a second RMS normalization
5. a SwiGLU feed-forward layer
6. a second residual connection

The attention blocks mirror that same structure, replacing the recurrent mixer with RoPE-based attention.

## Current implementation status

One important detail: the attention path is aligned with the public OLMo recipe in terms of architecture and kernel choice, but the recurrent path is **not** a verbatim import of fused OLMo-core kernels, just for my usecase at this point at least.

Instead, it uses a **native PyTorch DeltaNet-style approximation** implemented inside the local codebase.

That is on purpose. The immediate goal is to test the architecture at music scale first, before investing in lower-level kernel work.

## The 300M baseline configuration

The repository should now have a preset called `music_olmo_hybrid_300m_a100`.

Here is the model configuration:

* **Model dimension:** 1024
* **Layers:** 22
* **Attention heads:** 16
* **KV heads:** 4
* **Feed-forward width:** 2816
* **Dropout:** 0.1
* **Max sequence length:** 2048
* **Attention period:** 4
* **Total parameters:** 294,263,072

With an attention period of 4, the 22-layer model contains **17 recurrent blocks** and **5 full-attention blocks**. The grouped-query setup reduces KV projection cost while preserving full query resolution.

## Training plan

The default preset is designed for **mixed-precision training on a single A100 GPU**.

Initial training configuration:

* **Dataset:** FMA-Large DAC tokens
* **Optimizer:** AdamW
* **Learning rate:** (2 \times 10^{-4})
* **Warmup:** 2000 steps
* **Max steps:** 200,000
* **Per-device batch size:** 4
* **Gradient accumulation:** 8
* **Effective batch size:** 32 sequences
* **Precision:** bfloat16 autocast on CUDA
* **Evaluation interval:** 2000 steps
* **Checkpoint interval:** 5000 steps

For a **40GB A100**, the repository should also now expose `--grad_accum_steps`, which makes it easier to shift toward smaller microbatches without changing the effective batch size too aggressively.

## Why this is a useful baseline and the ACTUAL crux

This model is meant to answer a fairly narrow but important question:

> Can an OLMo-hybrid-style decoder preserve more musical structure than the current transformer and Mamba-family baselines at roughly the same parameter scale?

I think this is a worthwhile music baseline for three reasons.

First, it avoids a false choice between exact attention and recurrent state. Music benefits from both.

Second, it is practical. A **294M model that trains on one A100** is much easier to iterate on than a multi-billion-parameter reproduction of the full public OLMo stack.

Third, it is easy to ablate. The **3:1 schedule**, **RoPE setup**, **GQA configuration**, and **feed-forward width** can all be varied cleanly in follow-up experiments.

## Evaluation plan

The evaluation setup right now below.

Primary metrics include:

* validation loss and perplexity
* codebook-wise token accuracy
* long-range coherence across sliding windows
* Fréchet Audio Distance on decoded generations... maybe?
* listening-based inspection of rhythmic stability, harmonic continuity, and phrase development

The comparison set should include:

* the existing transformer baseline
* pure Mamba baselines
* earlier hybrid models already in the repo
* memory-caching variants where relevant

## Limitations

This is more so an implementation and experiment-design note, not a completed empirical claim. No training results are reported here.

There are also two technical limitations worth being explicit about.

First, the recurrent mixer is a **native PyTorch approximation**, not the fused **OLMo-core Gated DeltaNet** implementation (I will definitely use this in subsequent runs).

Second, FlashAttention acceleration only applies to the **attention blocks**. The recurrent blocks remain sequential by design.

If this baseline performs well, the natural next step is to move the recurrent side closer to the public runtime stack.

## Thoughts?

The repository now contains an OLMo-hybrid-style music model that is large enough to be meaningful, but still small enough to train on a single A100. That makes it a strong next baseline.

The contribution here is seemingly straightforward: we have a concrete architecture, decent preset, and also a decent clear evaluation plan. Whether it actually beats the existing baselines is now an experimental question rather than an implementation gap.
