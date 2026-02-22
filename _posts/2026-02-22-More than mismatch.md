
*mongoobi, Feb 2026*
---

Follow-up to [Part I](https://www.lesswrong.com/posts/your-sae-looks-solved-your-model-disagrees), where I trained TopK SAEs on Pythia-70M and 410M across layer depth and sparsity budget \(k\), and found a proxy gap: \(R^2\) and CE-patching could disagree about which SAE was “better.”

So, Part I primarily relied on 10M training tokens per SAE.  
The obvious next move was to rerun at higher budget and see what survives.

I did that at 50M and 100M tokens, with seed repeats at anchor settings.

---

## What changed after training longer?

Deltas are always \(410M - 70M\).  
Mismatch means \( \Delta_{R^2}>0 \) while \( \Delta_{CE}<0 \).

Mid-layer, \(k=8\):

| tokens | seeds | \(\Delta_{CE}\) | 95% CI | \(\Delta_{R^2}\) | mismatch rate |
|---:|---:|---:|---:|---:|---:|
| 10M | 4 | -0.099 | [-0.117, -0.082] | +0.150 | 100% |
| 50M | 1 | +0.018 | [0.018, 0.018] | +0.108 | 0% |
| 100M | 3 | +0.009 | [+0.004, +0.013] | +0.096 | 0% |

Mid-layer, \(k=16\):

| tokens | seeds | \(\Delta_{CE}\) | 95% CI | \(\Delta_{R^2}\) | mismatch rate |
|---:|---:|---:|---:|---:|---:|
| 10M | 4 | -0.028 | [-0.064, +0.009] | +0.111 | 50% |
| 50M | 1 | +0.018 | [0.018, 0.018] | +0.072 | 0% |
| 100M | 3 | +0.020 | [+0.016, +0.024] | +0.067 | 0% |

At 10M, the mismatch was real and repeatable.  
By 50M+, sign disagreement mostly disappears in this regime.

That’s, I guess the initial update.

The other update is the one I actually care more about: the **magnitude gap** seems to stay pretty large.

At 100M:
- \(k=8\): \( \Delta_{R^2}=0.096 \) vs \( \Delta_{CE}=0.009 \) (\(\sim 10.7\times\))
- \(k=16\): \( \Delta_{R^2}=0.067 \) vs \( \Delta_{CE}=0.020 \) (\(\sim 3.3\times\))

So the core problem is a bit sharper; more clearly, it seems to be the case that even when proxies agree on direction, they can disagree hard on effect size.

---

## Why this is a stronger result than Part I?

Part I posed a wide question about the ability of proxies to reverse ranking.
Part II lands on the question I trust more: can proxies miscalibrate magnitude in a scale-dependent way?

That second question is obviously way less ambitious or flashy, but seems more practically useful maybe.

If one metric says “big win” and another says “near tie,” you get different decisions on:
- where to spend compute,
- which model-\(k\) settings to prioritize,
- and what to claim about scaling behavior.

I’m not asking \(R^2\) and CE to be numerically identical. They are different objects.  
I am asking whether one of them gives a systematically inflated reading for the decisions we actually make.

---

## Funny convergence with e2e SAE work

While running this, I read [Braun et al. (2024)](https://arxiv.org/abs/2405.12241), who train SAEs end-to-end with a KL objective over outputs. Their motivation is close to this one: local reconstruction can miss functional importance.

That lines up cleanly:
- their work upgrades the training objective,
- this work tries to upgrade the evaluation layer for existing SAEs.

Most people still operate on pre-trained or local SAEs. A post-hoc diagnostic for “proxy confidence is inflated here” remains valuable even if end-to-end retraining is better in principle.

---

## The next measurement: sensitivity-weighted distortion

The mechanism test I’m running now is:

\[
\text{SWD} = \mathbb{E}\left[(g^\top \delta)^2\right]
\]

with \(g=\nabla_a L\) at the hookpoint and \(\delta=\hat a-a\).

Intuition: reconstruction error matters most along directions the model is sensitive to.  
SWD measures that directly.

I’m building a proxy leaderboard (\(R^2\), cosine, \(1/\text{NMSE}\), \(1/\text{SWD}\), \(1/|g^\top \delta|\)) against \(CE_{rec}\), with Pearson/Spearman and bootstrap intervals.

Success condition is simple:
1. SWD beats \(R^2\) predictively.
2. That lift survives resampling.

If it does, the geometry story earns real weight.  
If it doesn’t, I’ll treat the remaining gap as a different failure mode and update accordingly.

---

## Status

- Low-budget mismatch exists and replicates.
- Higher token budget resolves most sign disagreement in the tested mid-layer low-\(k\) slice.
- Magnitude disagreement remains large at 100M.
- Mechanism test is active.

This is no longer “did I find a weird corner case.”  
It’s becoming “where do our favorite SAE proxies stay calibrated, and where do they drift.”

That map is what I want to publish.

---

## Current limits at time of publish:

- Two model sizes so far (70M, 410M).
- One SAE family.
- 50M rows are single-seed.
- 100M repeats are still small-\(n\).
- CIs here are rough small-sample bands, not final inferential stats.

---

## Rep

- `info-geo/outputs/phase2_repeat_analysis.md`
- `info-geo/outputs/phase2_repeat_analysis.csv`
- `info-geo/outputs/proxy_gap_lowk_10m_100m_report.md`
- `info-geo/run_expb.py`
- `info-geo/full_paper_outline.md`

---

## References

- Braun et al. (2024), *Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning*. [arXiv:2405.12241](https://arxiv.org/abs/2405.12241)
- Gao et al. (2024), *Scaling and Evaluating Sparse Autoencoders*. arXiv:2406.04093
- Bricken et al. (2023), *Towards Monosemanticity*
- Elhage et al. (2022), *Toy Models of Superposition*. arXiv:2209.10652
- Finzi et al. (2026), *From Entropy to Epiplexity*. arXiv:2601.03220
