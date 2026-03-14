---
title: "Completed 12k Run: Early Speech Emergence in a 21.9M OLMo-Hybrid Speech LM"
---

This is the follow-up to my earlier `1800`-step pilot. I resumed the same `21.9M` OLMo-hybrid speech codec LM on A100 and let it run all the way to `12000` steps.

The short version is that the run finished cleanly, validation kept improving the whole time, and the final samples are noticeably stronger than the earlier `15%`-budget checkpoint.

Final result:

- best checkpoint: `step 12000`
- EMA validation loss: `3.8207`
- perplexity: `45.63`
- dataset: `LJ Speech`
- tokenizer: `EnCodec 24 kHz`, `8` codebooks
- hardware: `A100-SXM4-80GB`

Useful links:

- [full technical report (PDF)](/images/olmo-hybrid-speech/final_report.pdf)
- [code](https://github.com/Obiohagwu/olmo-hybrid-speech)
- [full eval history CSV](/images/olmo-hybrid-speech/a100_eval_history_full.csv)

---

### Setup

This is still an unconditional codec language model, not a text-conditioned TTS system yet.

| Item | Value |
|---|---|
| Model | OLMo-hybrid speech LM |
| Parameters | `21.9M` |
| Backbone | `8` layers = `6` Gated DeltaNet blocks + `2` attention blocks |
| Width | `d_model=384`, `d_ff=1024` |
| Attention | `6` heads, `2` KV heads |
| Hybrid schedule | attention every `4th` block, final block forced to attention |
| Data | `LJ Speech` |
| Tokenizer | `EnCodec 24 kHz`, `8` codebooks, vocab `1027` |
| Chunking | `8s` chunks |
| Split | `12,624` train / `666` val |
| Context | `1024` delayed steps |
| Hardware | `A100-SXM4-80GB` |
| Runtime | `bf16`, fused AdamW, CUDA SDPA flash path on, fused FLA GDN path off |
| Stable batch | true batch `24`, grad accum `1` |
| Throughput | about `18k tok/s` |

One important systems caveat: this successful run still used the plain PyTorch recurrent fallback for the Gated DeltaNet blocks. The intended fused FLA recurrent kernel was unstable on the available pod stack, so this is a clean modeling result more than a clean kernel-stack result.

---

### Validation Progression

The important part is that the run never really rolled over. It just kept getting better more slowly.

| Step | Train Loss | EMA Val Loss | PPL |
|---|---:|---:|---:|
| 200 | 5.0049 | 6.7878 | 886.99 |
| 1000 | 4.1367 | 4.8510 | 127.87 |
| 1800 | 4.2224 | 4.2847 | 72.58 |
| 3200 | 4.2821 | 4.0500 | 57.40 |
| 5200 | 3.7050 | 3.9194 | 50.37 |
| 7400 | 3.8102 | 3.8551 | 47.23 |
| 10000 | 3.8445 | 3.8267 | 45.91 |
| 12000 | 3.7043 | **3.8207** | **45.63** |

So the original `1800`-step pilot was real, but it was not the end of the useful training regime.

---

### Sample Progression

What I cared about most was whether this architecture could stay in a clearly speech-like regime and get cleaner with more training. It did.

Earlier pilot checkpoint: `step 1800`

<audio controls preload="none" src="/images/olmo-hybrid-speech/step1800_sample1.wav"></audio>

Mid-run checkpoint: `step 7400`

<audio controls preload="none" src="/images/olmo-hybrid-speech/step7400_sample0.wav"></audio>

Final-best checkpoint: `step 12000`

Sample 1

<audio controls preload="none" src="/images/olmo-hybrid-speech/step12000_sample0.wav"></audio>

Sample 2

<audio controls preload="none" src="/images/olmo-hybrid-speech/step12000_sample1.wav"></audio>

Sample 3

<audio controls preload="none" src="/images/olmo-hybrid-speech/step12000_sample2.wav"></audio>

These are still babbly and not semantically grounded. But they are much less in the "barely holding together" regime than the early pilot, and that matters.

---

### What I Think This Says

The narrow claim I am comfortable making is:

> a small OLMo-hybrid / Gated-DeltaNet-style speech codec LM can learn enough local speech structure on a clean single-speaker corpus to produce stable, clearly voice-like audio, and it keeps improving well past the first emergence point.

What I am **not** claiming:

- that this is a finished TTS system
- that it beats a matched transformer baseline
- that the current no-FLA run reflects the architecture's ideal efficiency
- that these samples are semantically meaningful speech

This is still an architecture viability result. But it is a real one.

---

### Next Step

This is enough to justify moving into text conditioning.

The question is no longer "can this thing speak at all?" The next question is whether I can inject text cleanly enough to get controllable reading rather than only speech-like babble.

So the plan from here is:

- keep the current audio decoder
- add a small text encoder
- add cross-attention from selected decoder blocks into text states
- start with sentence-level LJ Speech transcripts before worrying about anything longer-form
