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

*mongoobi, Feb 2026*

---

Continuation of [Part I](/2026/02/09/your-sae-looks-solved-your-model-disagrees.html). The short version of that post: in mid-layer Pythia residual streams at fixed sparsity budget $k$, $R^2$ can go up with model scale while patched CE loss gets worse. I called it the proxy gap.

This post is about what happens when you train longer, what that rules out, and where the explanation might live.

---

## The sign mismatch was real but temporary

Part I's headline was a sign flip: $R^2$ says the 410M SAE is better, CE says the 70M SAE is better. That was on a 10M-token training budget.

I extended to 50M and 100M tokens with seed repeats. Mid-layer, $k=8$:

| tokens | seeds | $\Delta_{CE}$ | 95% CI | $\Delta_{R^2}$ | mismatch rate |
|---:|---:|---:|---:|---:|---:|
| 10M | 4 | -0.099 | [-0.117, -0.082] | +0.150 | 100% |
| 50M | 1 | +0.018 | — | +0.108 | 0% |
| 100M | 3 | +0.009 | [+0.004, +0.013] | +0.096 | 0% |

Sign mismatch: gone by 50M. The larger model just needed more tokens to converge. Fair criticism of Part I, and I'm reporting it.

But the magnitudes at 100M are still off by roughly 10x. $R^2$ says the 410M SAE is substantially better ($\Delta = 0.096$). CE says they're basically tied ($\Delta = 0.009$). At $k=16$ the ratio is about 3x. This isn't noise — it's a systematic, scale-dependent distortion in the metric everyone defaults to.

So: **H0 (pure optimization artifact) explains the sign flip but not the magnitude gap.** Which means something else is going on, and I want to know what.

---

## Three spaces and the mismatch between them

Here's the mental model I'm working with. There are three different spaces your SAE reconstruction lives in, and they don't agree about what "close" means.

**Activation space.** This is where SAEs optimize. The error is Euclidean: $\|h - \hat{h}\|^2$. The metric that normalizes this is $R^2$. This space is shaped by the covariance of activations — high-variance directions dominate.

**Probability-sensitive space.** This is what downstream computation actually cares about. The natural local metric here is the pullback of the output Fisher through the downstream Jacobian:

$$G_L = J_L^\top F_{out} J_L$$

where $F_{out} = \text{diag}(p) - pp^\top$ for the output distribution. A perturbation that's small in activation space can be large in this space if it points along a direction the model is sensitive to.

**Attention-plan space.** More speculative, and I'm not touching this in the first pass. But recent work frames attention weights as entropy-regularized transport plans (Litman, 2025). If you buy that framing, then patching activations changes the transport plan, and the cost of that change isn't Euclidean either. This is a second-wave thing.

The proxy gap is, in this framing, a gap between space 1 and space 2. $R^2$ measures fidelity in activation space. CE measures fidelity in probability-sensitive space. When the geometry of those two spaces diverges — which it does more at larger scale, especially in mid-layer low-PR regimes — the metrics disagree.

This isn't a new idea in the abstract. "Reconstruction isn't behavior" is something people say. What I'm trying to do is make it *testable* and *localized*: where exactly does it happen, how bad is it, and can you build a cheap proxy that tracks the right space?

---

## SWD: a first-pass bridge metric

Before trying to estimate the full pullback Fisher (expensive, probably unstable at this scale), there's a simpler object. Take the CE gradient at the hookpoint:

$$g_L = \nabla_{h_L} \mathcal{L}_{CE}$$

and define sensitivity-weighted distortion:

$$\text{SWD}_L = \mathbb{E}\left[(g_L^\top \delta h_L)^2\right]$$

where $\delta h_L = \hat{h}_L - h_L$ is the reconstruction error.

$R^2$ weights error by the covariance structure of activations. SWD weights error by the sensitivity structure of the loss. If those two weightings diverge, $R^2$ is lying to you, and SWD should catch it.

The concrete test: across a grid of (model, $k$, seed) conditions, does $1/\text{SWD}$ correlate with $CE_{rec}$ better than $R^2$? I'm building a proxy leaderboard — $R^2$, cosine sim, $1/\text{NMSE}$, $1/\text{SWD}$, $1/|g^\top\delta|$ — ranked by Pearson and Spearman with bootstrap CIs.

This is running on my M4 Mini right now. Slowly. Results when they're done.

---

## The hypothesis stack

I want to be precise about what claims are on the table and what gates them.

**H0 (optimization-only).** The cross-scale CE gap shrinks to zero as SAE training budget increases. *Status: partially supported.* Sign mismatch gone. Magnitude gap not gone.

**H1 (residual intrinsic component).** After convergence, a nonzero magnitude gap remains. *Status: supported at 100M tokens, but I'd like higher token budgets and more seeds to be confident.*

**H2 (geometry-aware proxies explain residual).** SWD or similar sensitivity-weighted metrics predict CE better than $R^2$. *Status: running.*

**H3 (task-relevant dimensional mismatch).** MI-derived task-relevant dimensionality diverges from geometric PR in exactly the regimes where proxy failure is worst. *Status: planned, contingent on H2.*

The rule I set before writing any code: don't interpret H2/H3 until H0 vs H1 is adjudicated. Phase 2 data adjudicates H0 vs H1 in favor of H1 (magnitude gap persists). So H2 is now live.

---

## Where this is going

The decision tree, stated plainly:

If SWD outpredicts $R^2$ for CE → the sensitivity geometry story has teeth, and I have a cheap diagnostic. Write it up as "here's the problem (Phase 1-2), here's why (SWD), here's a fix (report SWD alongside $R^2$)."

If SWD doesn't beat $R^2$ → the magnitude gap is real but the explanation isn't sensitivity geometry. Maybe it's purely an SST normalization artifact. Maybe the gap is in higher-order terms that a linear sensitivity proxy can't capture. Either way, the empirical characterization still stands as a contribution, and I'd pivot to the anisotropy/deflated-PR story or to pullback Fisher approximations as a second wave.

Second-wave experiments, if the first wave works:

- **Pullback Fisher approximation.** Approximate $G_L$ via subsampled Jacobians and diagonal $F_{out}$, compute Fisher-Euclidean deviation, test whether it adds predictive value beyond SWD.
- **MI critic transfer.** Run separable vs hybrid MI estimators on $(h_L, \text{logits})$, compare inferred task-relevant $k^*$ profiles with proxy gap magnitude.

I'm not pre-committing to these. They're on the shelf if the simple version works and I want to push the mechanism story deeper.

---

## Rate-distortion-geometry

One framing I keep coming back to: an SAE is a lossy code. The sparsity budget $k$ determines the rate. What changes is which distortion measure you evaluate against.

In Euclidean distortion, the SAE looks increasingly good with scale (high $R^2$). In sensitivity-weighted distortion, the picture may be different. The rate-distortion curve *depends on the distortion geometry*, and if you're measuring in the wrong geometry, your curve is wrong.

This connects, loosely, to the bounded-observer information frameworks (Finzi et al., 2026, on epiplexity). The common thread: the same representation can contain structure that's "there" in an information-theoretic sense but not extractable by a given tool class under a budget. The proxy gap is a concrete instance of this: $R^2$ says the structure is there, CE says your SAE didn't extract it in a way that matters.

I'm not claiming a formal reduction. I'm using it as a lens.

---

## Concurrent work worth reading

[Braun et al. (2024)](https://arxiv.org/abs/2405.12241) trained SAEs to minimize KL divergence instead of reconstruction error ("end-to-end SAEs") and showed a Pareto improvement: more CE explained with fewer features. Their argument is basically "reconstruction objectives learn dataset structure, not computational structure."

Same core problem, different angle. They built a better training objective. I'm trying to build a better evaluation diagnostic. The e2e paper is evidence for the premise — if a separate group independently builds an entire training pipeline to work around this problem, the problem is real. But most people in the interpretability community are using existing SAEs (SAELens, Anthropic's published dictionaries), not retraining from scratch. A cheap post-hoc metric that flags "your $R^2$ is misleading here" is useful even in that world.

---

## Limitations

Everything from Part I still applies (two model sizes, one decoder class, not ablation-normalized, late-layer confound). New:

- Seed counts are low. 3-4 at anchor conditions. Not enough for robust seed-level CIs.
- The 50M row has one seed per $k$. Single data point.
- Exp-B (SWD vs classical proxies) hasn't finished. I'm reporting setup and rationale, not results.
- I haven't computed deflated PR yet. The canyon story is suggestive but might not survive deflation.

---

## Repro

- Phase 2 data: `info-geo/outputs/phase2_repeat_analysis.md`
- Token sweep: `info-geo/outputs/proxy_gap_lowk_10m_100m_report.md`
- Exp-B script: `info-geo/run_expb.py` (standalone, runs on MPS/CUDA/CPU)
- Paper outline: `info-geo/full_paper_outline.md`

---

## References

1. Litman, E. (2025). *Scaled-Dot-Product Attention as One-Sided Entropic Optimal Transport*. arXiv:2508.08369.
2. Park, K., et al. (2026). *The Information Geometry of Softmax: Probing and Steering*. arXiv:2602.15293.
3. Gulati, P., et al. (2026). *Mutual Information and Task-Relevant Latent Dimensionality*. arXiv:2602.08105.
4. Braun et al. (2024). *Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning*. arXiv:2405.12241.
5. Gao, L., et al. (2024). *Scaling and Evaluating Sparse Autoencoders*. arXiv:2406.04093.
6. Bricken et al. (2023). *Towards Monosemanticity*. Anthropic.
7. Finzi et al. (2026). *From Entropy to Epiplexity*. arXiv:2601.03220.
