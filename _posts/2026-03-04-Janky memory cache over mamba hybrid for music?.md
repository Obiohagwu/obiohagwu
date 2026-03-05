<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>

<style>
html, body {
  overflow-x: hidden;
  max-width: 100%;
}
table {
  display: block;
  overflow-x: auto;
  white-space: nowrap;
  max-width: 100%;
  font-size: 0.9em;
}
.post-content {
  overflow-x: hidden;
}
</style>

So I've been attending house type like dance events more frequently. Not to make this too romantic, but i really enjoy music, and dancing, and like, house really is the perfect manifestation of that haha.

Anyway, I decided to try a janky re-implementation of this new memory-caching fix over rnn or just linear attention type operations. 

**Now to be more specific:**
State space models in general are a cool more efficent sequence modeler. They are pretty good at global coherence and understanding and unlike self-attention with its quadratic burden, State space methods like mamba are subquadratic. Linear by input sequence etc. 
Mamba processes sequences by maintaining a sort compressed hidden, or for analogy sake, latent state. It holds a fixed-size summary of everything it's seen, which im sure you might already be seeing why this is an efficency boost, but might degrade over large enough corpi, or atleast at some point be unable to easily decode local detail. 
More specifically, that state gets updated with every new token, and old information gets gradually overwritten. For a 30-second music clip tokenized at 86 frames/second across 9 codebooks, that's over 23,000 tokens. A lot of information to compress into one fixed-size state vector.

So, imagine a scenario where the model could sort of *remember* what happened earlier? Not in the vague, compressed-state-vector sense. Actually remember it. Store specific checkpoints from earlier in the sequence and pull them back when they're relevant?

Yeah, see below...

---

## The Memory Caching Paper

Some days ago actually, Behrouz et al. published "Memory Caching: RNNs with Growing Memory" (arXiv:2602.24281). The idea is pretty sweet:

1. **Segment the sequence.** Divide the input into fixed-size segments of S tokens.
2. **Cache boundary states.** At the end of each segment, save the model's hidden state.
3. **Retrieve when needed.** For each new token, let the model query the cache -- "which of my past states is most relevant right now?" -- and inject that information via a learned gating mechanism.

The retrieval uses what they call **Gated Residual Memory (GRM)**. For each position, the model computes a softmax distribution over all cached segments plus the current segment. The gates decide: how much should I trust my current computation vs. what I cached from 5 segments ago?

It's a way of giving RNNs an explicit, growing memory bank without abandoning the efficiency of recurrence. Think of it as Mamba's usual compressed state, supplemented by like a series of snapshots from earlier in the sequence.

And it seems they only tested it on linear attention and Titans and havent tried State space models. I mean, it is a pretty recent paper though lol.

I did want to know if it could work for music generation though. A bit of a redundant side project but i was just bored and have extra runpod creds.

---

## Mamba's Hidden state seems difficult to penetrate

Here's the thing about Mamba that makes this non-trivial.

The Memory Caching paper assumes you can grab the model's hidden state at any point during the forward pass. For linear attention that's fairly straightforward. The recurrent state is an explicit matrix $S_t \in \mathbb{R}^{d_k \times d_v}$ that you compute and can cache directly.

Mamba doesn't work that way. Its fused CUDA kernel runs the entire selective scan inside GPU registers. The hidden state (a (d_inner, d_state) tensor per layer) is never materialized in accessible memory during training. You can't just reach in and grab it.

So I decided to not rewrite the kernel and just cache a proxy of the hidden state (i know i know , janky, sloppy... haha but this really is purely experimental)

---

### The implementations and the primary

I built two models to create a controlled comparison:

### MC-Linear-Attention (the "faithful" version)

Linear attention *does* expose its state matrices. So I built a version of MC that caches the actual recurrent state S_t at segment boundaries to match exactly what the paper as best i could. When the model wants to retrieve cached context, it does `phi(q_t) @ S_i`, querying the cached state matrix with the current token's feature-mapped key. This is as close to the paper's formulation as you can get.

### MC-Mamba (the "proxy" version)

For Mamba, I cache the **output activations** at segment boundaries instead of hidden states. After each Mamba block processes the full sequence (preserving the fast CUDA kernel), I extract the output vector at positions S-1, 2S-1, 3S-1, etc. These d_model-dimensional vectors serve as compressed summaries of what the model computed at each segment boundary.

The GRM gating mechanism is identical in both: softmax over segment means from the input space, with a single learnable W_u projection per layer. The only difference is *what* gets cached and *how* it gets retrieved.

If MC works on linear attention but fails on Mamba, the proxy approach is the bottleneck, not the mechanism itself. That's the controlled comparison.

---

### Train Setup

Both models were trained on music tokenized with **DAC** (Descript Audio Codec) at 44.1kHz -- 9 codebooks at ~86 tokens/second. The dataset is **FMA-Large**, a collection of 106,574 freely licensed tracks spanning a wide range of genres. Training ran on A100 80GB GPUs rented through RunPod.

The task is unconditional music generation (again, seriously gpu constrained even though im doing this): predict the next audio token across all 9 codebooks, autoregressively.

---

### Experiment 1: The 48M Model (Proof of Life)

Before going big, I ran a small-scale test to see if the MC mechanism could learn at all.

| | MC-LA 48M |
|---|---|
| Architecture | MC-Linear-Attention |
| Parameters | 48.6M (2.5% MC overhead) |
| Dataset | 25k tracks (FMA subset) |
| Segment size | 256 tokens |
| Peak LR | 1e-4 |
| Batch size | 32 |
| Steps | ~24,700 |

This was an early version of the code. Not optimized, some rough edges. I'm telling you this upfront because what happened next is quite important and matters more than the absolute numbers.

### It learned, haha wtf.

The val loss dropped steadily across the entire run:

| Step | Val Loss | Perplexity |
|------|----------|------------|
| 2,000 | 6.854 | 947.5 |
| 4,000 | 6.746 | 850.6 |
| 8,000 | 6.363 | 579.8 |
| 12,000 | 6.139 | 463.6 |
| 16,000 | 6.046 | 422.3 |
| 20,000 | 5.997 | 402.1 |
| 24,000 | 5.969 | 391.3 |

No instability. No divergence. Just a clean, steady decline over 24,000 steps.

But the really interesting signal was the **GRM entropy**.

GRM entropy measures how spread out the attention weights are across cached segments. High entropy means the model is attending uniformly and looking at everything equally, which is basically the same as looking at nothing in particular. Low entropy means the model is being *selective* and it has learned to focus on specific cached segments when they're relevant.

The 48M model started with GRM entropy around 1.26 (roughly uniform over cached segments). By step 24,000 it had dropped to **0.22**.

The gating mechanism was learning. The model wasn't just blindly averaging cached states. It seemed to have been developing preferences for specifically *which* cached segment to retrieve at each position. That's seems to align with the behavior the MC paper predicts: learned, position-dependent retrieval over a growing memory bank.

Was the audio actually good though? To be honest, with 48M parameters on 25k tracks, "good" is a stretch. But there actually surprisngly was a *semblance of something*. Like rhythmic patterns that held, frequency content that wasn't just noise. Enough to warrant scaling up.

---

### Experiment 2: Scaling to 95M (Where Things Went Wrong)

Encouraged by the 48M results, I scaled up. Bigger model, bigger data, improved codebase.

| | MC-LA 95M |
|---|---|
| Architecture | MC-Linear-Attention |
| Parameters | 95.1M (3.1% MC overhead) |
| Dataset | 106k tracks (full FMA-Large) |
| Segment size | 128 tokens |
| Peak LR | 3e-4 |
| Batch size | 64-72 |
| MC start layer | 10 (only top half gets MC) |

I made several changes from the 48M run: halved the segment size (128 vs 256), tripled the learning rate (3e-4 vs 1e-4), only applied MC to the top 10 layers (leaving layers 0-9 as plain linear attention), and switched to the full 106k-track dataset.

I ran it twice. Both times, the same thing happened.

### Gradient explosions

The gradient norms tell the story:

```
step   2900 | grad_norm 18.98
step   3100 | grad_norm  5.55
step   3500 | grad_norm  3.95
step   3800 | grad_norm  2.82
step   3900 | grad_norm  3.83
step   4500 | grad_norm  5.10
step   4600 | grad_norm  5.41
step   5000 | grad_norm 13.76
step   5100 | grad_norm 15.64
step   5300 | grad_norm 14.42
step   5500 | grad_norm  8.41
```

Between the spikes, gradient norms sat around 0.15-0.20 which was pretty ok. But every few hundred steps, something in the MC pathway would fire and blow the norm up by two orders of magnitude. These seem mechanistic and not necessarily random, again it could just be a weird artifact of early training dynamics, idk.

### The gate never opened

The model logs a `gate` value representing the learned bias on the GRM gate. It initialized at -2.0, which corresponds to sigmoid(-2.0) = 0.12, which is a very conservative starting point that barely lets cached context through.

Over 5,500 steps, the gate moved from -2.00 to -1.91.

That's nothing. The gate was supposed to learn when to open and trust the cached memory. Instead, it stayed almost exactly where it started. The MC mechanism was architecturally present but functionally dormant.

### GRM entropy stayed high

Remember how the 48M model's entropy dropped from 1.26 to 0.22? The 95M model's GRM entropy sat at **~2.4 the entire run**. That's near-uniform attention -- the model never learned to selectively retrieve from its cache. It was looking at everything equally, which is the same as ignoring the cache entirely.

### The loss seems fine (which made it more annoying)

The *loss* was actually quite reasonable.

```
step   2000 | EVAL: val_loss=6.6908
step   4000 | EVAL: val_loss=6.0978
```

The underlying linear attention was doing all the work while the cache sat there with its gate barely open and likely contributing to gradient noise.

---

### What to do?

The 48M model does seem to work though.

**What I suspect:**

The core issue is a **cold-start problem**. The MC mechanism needs to learn useful retrieval patterns, but it starts nearly disabled (gate at 0.12) while the base model races ahead. By the time the gate bias might have wandered open, the base linear attention has already found a loss landscape that doesn't need the cache. The MC gradients become noise and they push the gate around randomly without a clear signal, maybe occasionally causing a spike hard enough to disrupt training.

The 48M model avoided this maybe because: (a) the lower learning rate gave the GRM more time to adapt gradually, (b) the smaller model capacity meant the base attention couldn't "solve" the task on its own as easily, creating demand for the cached context, and (c) the larger segment size (256) meant fewer cache entries, making the softmax distribution easier to sharpen.

---

## Limitation :)

To be clear this was more of a learning thourgh experiment thing, and i'd definitely take it further if I had more GPU time.

**For 48m:** It's a smaller model, smaller dataset, lower learning rate, and an older version of the code. The declining GRM entropy is a genuinely positive signal and the model by all metric, learned selective retrieval. But I can't rule out that the entropy drop is an artifact of the training dynamics rather than evidence that the cache is meaningfully improving generation quality (compute and data to rule out)

**I never ran a 48M model *without* MC for comparison to get a decent baseline:** The 48M result stands alone. I can't tell you how much of its final loss is attributable to MC vs. how much the base linear attention would have achieved on its own. That's a gap I need to fill.

**The 95M runs were cut short.** 5,500 steps isnt usually enough to declare failure in evry circumstance, as i said earlier. It's definitely possible the grad spikes would have settled and the gate would have eventually opened. But the trajectory wasn't promising, and I didn't want to burn GPU hours watching it not work.

**Three variables changed:** between the 48M and 95M runs: model size, dataset size, and learning rate. Any one of these could be the culprit. Proper experimental design would change one at a time. I didn't, because GPU hours cost money and a bit of impatience.

---

## Learnings?

**1. The mechanism works (when it works)**

The 48M experiment is proof that GRM-based memory caching *can* learn selective retrieval over cached segment states in a music generation context. The entropy dropping from 1.26 to 0.22 is not nothing.  The model seemingly learns to look at specific past segments when generating new tokens. That's the whole promise of MC.

**2. Initialization and learning rate matter more than I expected**

The gate bias at -2.0 combined with LR 3e-4 seems to be a bad combination. The MC paper doesn't discuss initialization sensitivity because they're working with linear attention models where the state matrices are already part of the standard forward pass. When you're bolting MC onto an architecture as an *addition*, the optimization dynamics change. The MC pathway needs to be initialized assertively enough to receive gradient signal, but not so aggressively that it destabilizes the base model.

**3. The "proxy vs. faithful" comparison is still pending**

I haven't actually trained MC-Mamba (the output-activation proxy version) yet. The experiments above are all MC-Linear-Attention. The original question I did have was can you cache Mamba's *output activations* and get useful retrieval?. I think thats remains a bit open. The 48M result suggests the GRM mechanism itself works for music, so the next step is testing whether the proxy representation (boundary output vectors) is informative enough to retrieve from.

**4. Scale-up isn't free**

Also learned the hard way that a mechanism that works at small scale can fail at large scale for reasons that have nothing to do with the mechanism's fundamental validity. Initialization, learning rate schedules, interaction with other training hyperparameters etc etc can all obviously change the outcome. The 48M model's success didn't predict the 95M model's failure.

---

## Maybe next steps

Some ideas will work on later.


---

### Outlook

I started this project because I thought recurrent models needed better long-range memory for music generation, and MC seemed like an elegant way to provide it. The results so far are... humbling.

The 48M experiment proved the concept can work, like a pretty good concept level proof. Although I messed up with trying to scale prematurely. The gate staying shut, the entropy staying flat, the grad norms spiking are all symptoms of general issues in my models optimization landscape. So the whitepill there is it might not necessarily be anything wrong with my architecture "design".

What I have right now is a mechanism that learned selective memory retrieval on a small model, and a larger model where that same mechanism refused to engage because of mostly dumb mistakes on my part.


---

*All code is available in the [mc-mamba repository](https://github.com/Obiohagwu/mc-la-mamba). Training logs for all runs are included in `pod_runs/`.*

*If you've dealt with similar cold-start problems when adding auxiliary mechanisms to neural networks, I'd genuinely love to hear how you solved it. Sometimes the difference between "doesn't work" and "works beautifully" is one initialization trick.*
