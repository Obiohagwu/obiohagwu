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

Continuation of [Part I](/2026/02/09/your-sae-looks-solved-your-model-disagrees.html).

To summarize it though, in mid-layer Pythia residual streams at fixed sparsity budget $k$, $R^2$ can go up with model scale while patched CE loss gets worse. I called it the proxy gap.

This post is about what happens when you train longer, what that rules out, and where the explanation might live. I found some more seemingly interesting stuff. But unbeknownst to me, naively; this seems to be a fairly active research region. I'll expand more below/


---

## The sign mismatch was real but temporary

The more salient portion of the previous post as the apparent signflip in certain regimes. More specifically, $R^2$ says the 410M SAE is better, CE says the 70M SAE is better. That was on a 10M-token training budget.


I extended to 50M and 100M tokens with seed repeats. Mid-layer, $k=8$:

| tokens | seeds | $\Delta_{CE}$ | 95% CI | $\Delta_{R^2}$ | mismatch rate |
|---:|---:|---:|---:|---:|---:|
| 10M | 4 | -0.099 | [-0.117, -0.082] | +0.150 | 100% |
| 50M | 1 | +0.018 | — | +0.108 | 0% |
| 100M | 3 | +0.009 | [+0.004, +0.013] | +0.096 | 0% |

Sign mismatch: gone by 50M. The larger model just needed more tokens to converge. Fair criticism of Part I, and I'm reporting it.

Although some anomolies seem to persist. 10x magnitude distortion still? The magnitudes at 100M are still off by roughly 10x. $R^2$ says the 410M SAE is substantially better ($\Delta = 0.096$). CE says they're basically tied ($\Delta = 0.009$). At $k=16$ the ratio is about 3x. I may be missing something but this seems like a plausible issue. A sort of systematic, scale-dependent distortion in the metric everyone defaults to.


So, **H0 (pure optimization artifact) explains the sign flip but not the magnitude gap.** Which means something else may be going on. We should investigate, haha.

---

## Three spaces and the mismatch between them

Ok, so here's my running mental model right now. There are three different spaces SAE reconstruction generally lives in or occupies, and they usually don't necessarily agree about what "close" means.

**Activation space.** This is where SAEs optimize. The error is Euclidean: $\|h - \hat{h}\|^2$. The metric that normalizes this is $R^2$. This space is shaped by the covariance of activations, usually leading to the situation where high-variance directions dominate.

**Probability-sensitive space.** Ok so this is i'd say the real meat, it seems to be what downstream computation actually cares about. The natural local metric here is the pullback of the output Fisher through the downstream Jacobian:

$$G_L = J_L^\top F_{out} J_L$$

where $F_{out} = \text{diag}(p) - pp^\top$ for the output distribution. A perturbation that's small in activation space can be large in this space if it points along a direction the model is sensitive to.

**Attention-plan space.** Although a bit more speculative, and I'll likely not go too deep on this due to my lack of sufficent background atm, but really interesting recent work frames attention weights as entropy-regularized transport plans (Litman, 2025). If you buy that framing, then patching activations changes the transport plan, and the cost of that change isn't Euclidean either. This is a second-wave thing. I will definitely dive alot deeper into this very soon.

Ok, so we may say the proxy gap is, in this framing, a gap between space 1 and space 2. $R^2$ measures fidelity in activation space. CE measures fidelity in probability-sensitive space. When the geometry of those two spaces diverges (which it does seem to do more at larger scales, especially in mid-layer low-PR regimes), the metrics disagree.

This isn't a new idea in the abstract. "Reconstruction isn't behavior" is something people say and maybe usually know intuitively. What I'm trying to do is make it *testable* and *localized*: where exactly does it happen, how bad is it, and can you build a cheap proxy that tracks the right space?

---

## SWD: a first-pass bridge metric

Before trying to estimate the full pullback Fisher (expensive, probably unstable at this scale), there's a generally much simpler object. We can take the CE gradient at the hookpoint:

$$g_L = \nabla_{h_L} \mathcal{L}_{CE}$$

and define sensitivity-weighted distortion:

$$\text{SWD}_L = \mathbb{E}\left[(g_L^\top \delta h_L)^2\right]$$

where $\delta h_L = \hat{h}_L - h_L$ is the reconstruction error.

$R^2$ weights error by the covariance structure of activations. SWD weights error by the sensitivity structure of the loss. If those two weightings diverge, $R^2$ is lying to you, and SWD should catch it.

The concrete test: across a grid of (model, $k$, seed) conditions, does $1/\text{SWD}$ correlate with $CE_{rec}$ better than $R^2$? I'm building a proxy leaderboard for $R^2$, cosine sim, $1/\text{NMSE}$, $1/\text{SWD}$, $1/|g^\top\delta|$ — ranked by Pearson and Spearman with bootstrap CIs.

This is running on my local mac right now. Actually quite slowly. I'll append the results with accompanying edits when the run in done in 28hrs.

---

## The hypothesis stack right now

I want to be precise about what claims are on the table and what gates them.

**H0 (optimization-only).** The cross-scale CE gap shrinks to zero as SAE training budget increases. *Status: partially supported.* Sign mismatch gone. Magnitude gap not gone.

**H1 (residual intrinsic component).** After convergence, a nonzero magnitude gap remains. *Status: supported at 100M tokens, but I'd ideally much prefer higher token budgets and more seeds to increase confidence.*

**H2 (geometry-aware proxies explain residual).** SWD or similar sensitivity-weighted metrics predict CE better than $R^2$. *Status: running.*

**H3 (task-relevant dimensional mismatch).** MI-derived task-relevant dimensionality diverges from geometric PR in exactly the regimes where proxy failure is worst. *Status: planned, contingent on H2.*

The rule I set before writing any code: don't interpret H2/H3 until H0 vs H1 is adjudicated, lol. Phase 2 data adjudicates H0 vs H1 in favor of H1 (magnitude gap persists). So H2 is now live.

---

## Where this is going

The decision tree, stated plainly:

If SWD outpredicts $R^2$ for CE implies the sensitivity geometry story has teeth, and I have a cheap diagnostic. Proceed maybe?

If SWD doesn't beat $R^2$ implies the magnitude gap is real but the explanation isn't sensitivity geometry. Maybe it's purely an SST normalization artifact? Maybe the gap is in higher-order terms that a linear sensitivity proxy can't capture, sigh. Either way, the empirical characterization still stands as a fair contribution really, and I'll probably branch to the anisotropy/deflated-PR story or to pullback Fisher approximations as a second wave.

Second-wave experiments, if the first wave works:

- **Pullback Fisher approximation.** Approximate $G_L$ via subsampled Jacobians and diagonal $F_{out}$, compute Fisher-Euclidean deviation, test whether it adds predictive value beyond SWD.
- **MI critic transfer.** Run separable vs hybrid MI estimators on $(h_L, \text{logits})$, compare inferred task-relevant $k^*$ profiles with proxy gap magnitude.

I'm not pre-committing to these. They're on the shelf if the simple version works and I want to push the mechanism story deeper.

---

## Rate-distortion-geometry

One framing I seem to keep coming back to is conceiving an SAE primarily as a lossy code. The sparsity budget $k$ determines the rate. What changes is which distortion measure you'd want to evaluate against.

It's so weird that in euclidean distortion space, the SAE looks increasingly good with scale (high $R^2$). In sensitivity-weighted distortion, the picture may be different. The rate-distortion curve *depends on the distortion geometry*, and if you're measuring in the wrong geometry, your curve is wrong.

This connects, loosely, to the bounded-observer information frameworks (Finzi et al., 2026, on epiplexity). The common thread: the same representation can contain structure that's "there" in an information-theoretic sense but not extractable by a given tool class under a certain budgets. The proxy gap is a concrete instance of this: $R^2$ says the structure is there, CE says your SAE didn't extract it in a way that matters.

Although, to be clear I'm not claiming a formal reduction at all. I'm moreso trying to use it as a lens.

---

## Concurrent work worth reading if you're really interested:

[Braun et al. (2024)](https://arxiv.org/abs/2405.12241) trained SAEs to minimize KL divergence instead of reconstruction error ("end-to-end SAEs") and showed a Pareto improvement: more CE explained with fewer features. Their argument is basically "reconstruction objectives learn dataset structure, not computational structure."

Same core problem, different angle. They built a better training objective. I'm trying to build a better evaluation diagnostic. The e2e paper is evidence for the assumption that a separate group independently builds an entire training pipeline to work around this problem, the problem is real. But most people in the interpretability community are using existing SAEs (SAELens, Anthropic's published dictionaries), not retraining from scratch. A cheap post-hoc metric that flags "your $R^2$ is misleading here" is useful even in that world.

---

## Some more limitations?

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
