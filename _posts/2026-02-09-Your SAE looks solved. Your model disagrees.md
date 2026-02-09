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

## A Budgeted Pythia Sweep Showing a Depth-Localized Proxy Gap (TopK SAEs)

*mongoobi, Feb 2026*

---

This is a research log with a small argument:

1. If you're using SAEs as interpretability tools or safety monitors, **reconstruction fidelity alone is not a reliable acceptance test** — behavioral metrics like patched CE loss or ablation-normalized CE recovery should be primary.
2. In one common regime (mid-layer residual stream), $$R^2$$ can be **systematically inflated across scale at fixed sparsity**, due to a concrete mechanism (activation variance scaling), while behavioral preservation gets worse.

This is not a scaling-law fit and not evidence for a hard interpretability ceiling.

---

## Brief Definitions

**Sparse autoencoder (SAE).** A learned lossy codec for activations: encode dense activations into a sparse vector of "features," then decode back to the activation space.

**TopK SAE.** An SAE where each token activates exactly `k` nonzero features (hard sparsity). In this post, the decoder is linear.

**Hookpoint.** Where I patch the activation: `blocks.{L}.hook_resid_post` (residual stream post-block).

**Reconstruction fidelity ($$R^2$$).** Variance explained in activation space:

$$
R^2 := 1 - \frac{\mathrm{SSE}}{\mathrm{SST}}
$$

with mean-centered totals:

$$
\mathrm{SSE} = \sum \lVert a - \hat{a} \rVert^2, \quad \mathrm{SST} = \sum \lVert a - \mathbb{E}[a] \rVert^2.
$$

**Patched loss score ($$CE_{rec}$$).** For each eval batch, let $$L_{orig}$$ be original next-token cross-entropy and $$L_{recon}$$ be CE after replacing the hookpoint activation with the SAE reconstruction. I report:

$$
CE_{rec} := 1 - \frac{L_{recon} - L_{orig}}{L_{orig}} = 2 - \frac{L_{recon}}{L_{orig}}.
$$

Notes (to be specific):

- This is **not** ablation-normalized. It is **not** the same as "CE loss recovered" in Bricken et al. (2023) or SAE Lens's `ce_loss_score`, which normalize against a zero-ablation baseline. Do not compare magnitudes across papers without converting.
- Because this metric normalizes by each model's own $$L_{orig}$$, cross-model comparisons can be affected by baseline-loss differences. I mainly use it here as a within-depth behavior preservation score and for sign patterns across $$k$$.
- $$CE_{rec}=1$$ means perfect preservation ($$L_{recon}=L_{orig}$$).
- $$CE_{rec}=0$$ means loss doubled ($$L_{recon}=2 L_{orig}$$).
- It can be negative.

**Per-token MSE (`mse_mean`).** Mean squared reconstruction error, averaged over model dimensions:

$$
\mathrm{MSE} := \mathbb{E}\left[\frac{1}{d_{model}}\lVert a - \hat{a} \rVert^2\right].
$$

**Cosine similarity (`cosine sim`).** Mean tokenwise cosine between original and reconstructed activation vectors:

$$
\cos(a,\hat{a}) := \frac{a^\top \hat{a}}{\lVert a\rVert\,\lVert \hat{a}\rVert}, \quad \text{report } \mathbb{E}[\cos(a,\hat{a})].
$$

**Relative error norm (`relative error norm`).** Mean tokenwise relative L2 error:

$$
\mathrm{RelErr} := \mathbb{E}\left[\frac{\lVert a - \hat{a}\rVert}{\lVert a\rVert + \varepsilon}\right], \quad \varepsilon=10^{-10}.
$$

**Alive fraction (`alive %`).** Fraction of SAE features that fire at least once on the eval set (for TopK, "fires" means encoder output $$\neq 0$$ on some token):

$$
\mathrm{alive} := \frac{\#\{i : \exists\,t \text{ s.t. } z_{t,i}\neq 0\}}{d_{sae}}.
$$

**Participation ratio (PR).** A geometry diagnostic for effective dimensionality, computed from eigenvalues $$\lambda_i$$ of the mean-centered covariance:

$$
PR := \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}.
$$

**Rate proxy (bits/token).** An explicit coding-budget proxy for TopK codes: entropy-coded active indices plus fixed-point values (8 bits/value in this sweep). This is not mutual information; it's just a concrete budget.

---

## Experiment Setup (still quite crude atm)

I trained a grid of TopK SAEs and evaluated them two ways: (1) patching reconstructions back into the model and measuring CE change, and (2) direct reconstruction metrics in activation space.

**Main sweep (Fast2):** two models, three depths, six sparsity budgets = 36 SAEs.

Models:

- `pythia-70m` (`d_model=512`, `n_layers=6`)
- `pythia-410m` (`d_model=1024`, `n_layers=24`)

Hookpoints (matched by rough relative depth; late is confounded). Due to intense compute constraints, I've been limited to only 2 models at the moment. Will follow up.

| depth regime | 70M layer | 410M layer | caveat |
|---|---:|---:|---|
| early | L1 | L4 | roughly pre-"canyon" |
| mid | L3 | L12 | inside the "canyon" regime |
| late | L5 | L20 | 70M late is final block; 410M late is not |

**Supplementary sweep (Fast3):** three models at mid depth only, $$k \in \{8, 16, 32, 64\}$$ = 12 SAEs. Adds Pythia-160M (`d_model=768`) as a bridge point.

SAE class and budgets (both sweeps):

- Decoder class: single-layer TopK SAE with **linear decoder**.
- Expansion: $$d_{sae} = 32 \cdot d_{model}$$.
- Training budget: 10,000,000 tokens per SAE (budgeted pilot).
- Dataset: streaming `NeelNanda/pile-small-tokenized-2b`.
- Eval: 200 held-out sequences of length 256 (≈51k tokens) from the same stream (separated by skipping ahead in the stream), with 95% CIs from a bootstrap over sequences (5,000 resamples).

---

## The Number That Started This

Mid-layer, $$k=8$$ (Fast2):

| model | layer | $$R^2$$ | $$CE_{rec}$$ [95% CI] | implied $$L_{recon}/L_{orig}$$ |
|---|---:|---:|---:|---:|
| 70M | 3 | 0.807 | 0.340 [0.169, 0.469] | 1.660 |
| 410M | 12 | 0.961 | 0.235 [0.050, 0.382] | 1.765 |

So the larger model looks "nearly solved" by variance explained, but patched loss is still much worse than baseline. A blunt translation of $$CE_{rec}=0.235$$ is: "patching reconstruction increases loss by about 76.5%."

Stats caution: the CIs overlap at $$k=8$$, so do not treat this single row as decisive. The stronger evidence is the **consistent sign across the full $$k$$ sweep** (in this 10M-token/SAE budgeted regime) below.

---

## Result 1: The Proxy Gap Is Depth-Localized

Across depths, the relationship between reconstruction and patched loss behaves differently.

The most direct evidence is the mid-layer delta table. At every $$k$$ in this **10M-token/SAE budgeted sweep** (Fast2), scaling from 70M to 410M **increases** $$R^2$$ while **decreasing** $$CE_{rec}$$:

| $$k$$ | $$\Delta R^2$$ (410M minus 70M) | $$\Delta CE_{rec}$$ (410M minus 70M) |
|---:|---:|---:|
| 8 | +0.154 | -0.106 |
| 16 | +0.112 | -0.068 |
| 32 | +0.084 | -0.017 |
| 64 | +0.057 | -0.027 |
| 128 | +0.036 | -0.032 |
| 256 | +0.017 | -0.037 |

All six rows show the same sign pattern: $$\Delta R^2 > 0$$, $$\Delta CE_{rec} < 0$$.

The gap is largest at low $$k$$ and shrinks substantially by $$k \ge 32$$, where the absolute CE delta drops below 0.04. This is consistent with the first ~8-16 features being where variance-capture and loss-sensitivity diverge most: at low budgets, the SAE prioritizes high-variance directions (inflating $$R^2$$) while missing loss-sensitive structure. At higher budgets, there's enough capacity to cover both.

For reference, correlations between $$R^2$$ and $$CE_{rec}$$ from the full Fast2 grid:

| depth | corr pooling $$k$$ and scale | corr within fixed $$k$$ (scale-only) |
|---|---:|---:|
| early | +0.907 | +0.353 |
| mid | +0.474 | **-0.943** |
| late | +0.702 | +0.976 (confounded) |

**Important caveat:** `corr within fixed $$k$$` is computed by subtracting the mean within each $$k$$ (i.e. "demeaning by $$k$$") and then correlating the residuals across all points in that depth ($$n=12$$ here = 2 models $$\times$$ 6 $$k$$ values). With only two model sizes, treat it mainly as a **sign diagnostic** for how scaling moves $$R^2$$ vs $$CE_{rec}$$ at fixed $$k$$, not as a stable statistic. The delta table above is the real evidence.

---

## Why This Can Happen (no alien neuralese required)

The mid-layer sign flip (higher $$R^2$$ but worse $$CE_{rec}$$ at fixed $$k$$) is explainable with two mundane facts, plus one important confound.

### 1) $$R^2$$ is variance-normalized, and variance scale changes with model size

Mid-layer, $$k=8$$:

| model | mse mean | relative error norm | $$R^2$$ | SSE/token | SST/token |
|---|---:|---:|---:|---:|---:|
| 70M | 0.0854 | 0.509 | 0.807 | 43.7 | 226.7 |
| 410M | 0.1575 | 0.576 | 0.961 | 161.3 | 4131.1 |

Here SSE/token = mse_mean times d_model and SST/token is backed out from $$R^2$$ via $$R^2 = 1 - \mathrm{SSE}/\mathrm{SST}$$. The 410M mid layer has about 18x larger mean-centered variance scale (SST/token), so variance explained can look great even when absolute errors are not small.

**Prediction:** metrics that do *not* divide by SST (e.g. cosine similarity, relative error norm, raw MSE) should not show the same "looks solved" inflation. And they don't. At mid-layer $$k=8$$ (Fast3 three-model check):

| model | $$R^2$$ | cosine sim | relative error norm | $$CE_{rec}$$ |
|---|---:|---:|---:|---:|
| 70M | 0.810 | 0.856 | 0.506 | 0.372 |
| 160M | 0.907 | 0.869 | 0.485 | 0.331 |
| 410M | 0.961 | **0.809** | **0.574** | 0.269 |

$$R^2$$ improves monotonically with scale. Cosine similarity and relative error norm both show the 410M reconstruction is **worse**, consistent with $$CE_{rec}$$. The proxy gap here is specific to variance-normalized reconstruction metrics like $$R^2$$.

### 2) Loss sensitivity weights directions differently than covariance geometry

Let $$\hat{a} = a + \delta$$ be the reconstruction. For downstream loss $$L(a)$$, a first-order Taylor approximation gives:

$$
L(\hat{a}) - L(a) \approx g^\top \delta, \quad g := \nabla_a L.
$$

Reconstruction metrics weight directions by the activation distribution (covariance). Loss change weights directions by sensitivity $$g$$ (and, beyond first order, curvature). If sensitivity mass lives in comparatively low-variance directions, you can have high $$R^2$$ and still hurt loss.

This is the "proxy gap" mechanism in one sentence: **MSE/variance and loss sensitivity are different measures on activation space.**

### Confound: fixed SAE training budget undertrains larger SAEs

In these sweeps, every SAE gets the same 10M-token training budget, while $$d_{sae} \propto d_{model}$$. Empirically, alive fraction often falls with model size (especially early/mid at low $$k$$) under this fixed budget. That is consistent with larger SAEs being relatively more undertrained, which can degrade behavior preservation and direction-sensitive reconstruction metrics even when $$R^2$$ looks strong. (See Limitations and the 100M-token check below.)

---

## Result 2: Three-Model Check (Fast3, Mid-Layer Only)

A separate mid-only sweep adds Pythia-160M as a third point:

| model | $$d_{model}$$ | $$k$$ | $$R^2$$ | $$CE_{rec}$$ [95% CI] | cosine | alive % |
|---|---:|---:|---:|---:|---:|---:|
| 70M | 512 | 8 | 0.810 | 0.372 [0.232, 0.478] | 0.856 | 15.2% |
| 160M | 768 | 8 | 0.907 | 0.331 [0.122, 0.477] | 0.869 | 12.9% |
| 410M | 1024 | 8 | 0.961 | 0.269 [0.094, 0.404] | 0.809 | 11.9% |
| 70M | 512 | 16 | 0.855 | 0.580 [0.439, 0.671] | 0.893 | 37.1% |
| 160M | 768 | 16 | 0.924 | 0.538 [0.369, 0.653] | 0.908 | 28.5% |
| 410M | 1024 | 16 | 0.968 | 0.530 [0.394, 0.634] | 0.883 | 25.4% |
| 70M | 512 | 32 | 0.891 | 0.729 [0.622, 0.796] | 0.920 | 63.2% |
| 160M | 768 | 32 | 0.938 | 0.696 [0.569, 0.780] | 0.932 | 51.1% |
| 410M | 1024 | 32 | 0.974 | 0.723 [0.618, 0.801] | 0.919 | 51.4% |
| 70M | 512 | 64 | 0.922 | 0.836 [0.758, 0.882] | 0.944 | 83.8% |
| 160M | 768 | 64 | 0.951 | 0.809 [0.712, 0.871] | 0.934 | 71.2% |
| 410M | 1024 | 64 | 0.979 | 0.814 [0.730, 0.875] | 0.901 | 65.0% |

Three observations:

1. **At $$k=8$$ and $$k=16$$, the pattern holds across three models:** $$R^2$$ increases monotonically with scale while $$CE_{rec}$$ decreases monotonically.

2. **At $$k \ge 32$$, monotonicity breaks for $$CE_{rec}$$.** At $$k=32$$: 410M ($$CE_{rec}=0.723$$) recovers more behavior than 160M (0.696). At $$k=64$$: 410M (0.814) slightly outperforms 160M (0.809). The proxy gap is a **low-$$k$$ phenomenon**.

3. **All CIs overlap at every $$k$$.** At $$k=8$$ (the largest gap): 70M CI = [0.232, 0.478], 410M CI = [0.094, 0.404]. Per-batch variability is large — CE standard deviations range from 0.36 to 0.93 across conditions. Cohen's $$d$$ for the 70M-vs-410M difference at $$k=8$$ is approximately 0.14. The consistent sign across $$k$$ values is more informative than any individual comparison.

---

## Result 3: There Is a Measurable $$k^*$$ Tax (Early/Mid)

To make scaling concrete, define $$k^*(t)$$ as the minimum $$k$$ needed to reach $$CE_{rec} \ge t$$ (linear interpolation on the Fast2 $$k$$ grid; single-seed, no uncertainty estimate on the interpolation itself).

| depth | target $$t$$ | $$k^*$$(70M) | $$k^*$$(410M) | ratio |
|---|---:|---:|---:|---:|
| early | 0.85 | 40.6 | 55.0 | 1.36 |
| early | 0.90 | 62.0 | 106.9 | 1.72 |
| mid | 0.85 | 71.2 | 98.0 | 1.38 |
| mid | 0.90 | 116.5 | 187.6 | 1.61 |

This is not a blow-up (at these scales), but it is also not zero. Under a fixed decoder class and a fixed tool-training budget, larger scale can require more active features to preserve loss. However, see the alive-fraction caveat in Limitations — this may partly reflect undertraining rather than intrinsic difficulty.

---

## Geometric Context: A Raw PR "Canyon"

Recall: this is diagnostic, not the primary claim.

I computed raw PR of `hook_resid_post` across layers.

Selected points:

| model | layer | raw PR | PR/$$d_{model}$$ |
|---|---:|---:|---:|
| 70M | 1 | 51.914 | 0.101 |
| 70M | 3 | 3.961 | 0.008 |
| 410M | 4 | 111.967 | 0.109 |
| 410M | 5 | 2.045 | 0.002 |
| 410M | 12 | 1.177 | 0.001 |

So both models show a sharp drop into a long low-PR band, with 410M exhibiting an especially extreme canyon ($$PR \approx 1$$ for many layers).

Two important caveats:

- Raw PR can be dominated by a leading direction. Standard practice is to "deflate" (remove) the top eigenvector and recompute PR. When you do that, the absolute PR values change a lot, but the "expand then compress" depth profile often remains. That's a known phenomenon, and I'm not claiming novelty on it.
- In this post, PR is a **warning label**: "this is where variance-normalized reconstruction proxies are likely to lie." The mid-layer proxy gap above happens inside this low-PR regime.

I haven't included deflated PR plots yet because my current RunPod environment was unstable (numpy/datasets stack broke). I'll add deflated PR once it's cleanly reproducible.

---

## Information-Theoretic Framing

If you squint, an SAE is a lossy code. Then "interpretability under budgets" starts looking like **rate-distortion**:

- Rate: bits/token of your sparse code (I log a crude proxy).
- Distortion: either reconstruction distortion ($$1-R^2$$) or behavior distortion ($$1-CE_{rec}$$).

The practical point is not that "mutual information is falling" (I did not measure MI). The point is that **you can put a real budget axis on the x-axis** instead of just reporting $$k$$ or $$R^2$$.

This also connects (loosely) to recent "bounded observer" information frameworks, such as Finzi et al. (2026) on epiplexity. The common theme is: the same object can contain structure that is "there," but not extractable by a weaker observer/tool class under a budget. In this sweep, a fixed SAE class looks better by $$R^2$$ but worse by patched loss in mid layers as scale increases, which is at least qualitatively consistent with "extractable structure for this tool class" degrading even when "variance structure" looks easy.

I am not claiming a formal reduction from SAE failure modes to epiplexity or time-bounded entropy. I'm using it as a framing: interpretability tools are observers with constraints, and proxy metrics can hide when you're falling out of the extractable regime.

---

## Relevance?

Given that PR/anisotropy phenomena are known, any novel claim here is fairly narrow, but still we see:

- A depth-localized, reproducible proxy gap where $$R^2$$ can improve with scale at fixed $$k$$ while patched loss gets worse — **strongest at low $$k$$, closing by $$k \ge 32$$**.
- A concrete mechanism (SST inflation in anisotropic layers) that **predicts which reconstruction metrics are affected**: $$R^2$$ yes, cosine/relative error norm no — but the deeper point is that reconstruction metrics in general can diverge from behavioral ones.
- An operational "interpretability budget" object ($$k^*$$ for loss targets, plus rate proxies) that moves with depth and scale.
- A concrete diagnostic (raw PR canyon) that flags where reconstruction-only evaluation is especially untrustworthy.

If you already believe "reconstruction isn't behavior," you may find this useful as an attempt to make that belief operational, with a knob you can sweep and a failure mode you can reproduce.

**Practical methodology tweak:** if you're evaluating SAE quality, behavioral metrics like patched CE loss or ablation-normalized CE recovery (e.g. SAE Lens's `ce_loss_score`) should be the acceptance test — not reconstruction fidelity alone. Among reconstruction metrics, cosine similarity or relative error norm are more robust than $$R^2$$ in mid-layer residual streams, but they still measure reconstruction, not behavior.

---

## Limitations (Things A Reviewer Should Hit Me For)

- **Two model sizes in the main sweep** is not a scaling law fit. Fast3 adds a third mid-layer point but does not span depths.
- **One SAE training seed per condition.** Bootstrap CIs reflect eval-batch variability, not training variability. All CIs overlap at every $$k$$.
- **Fixed tool-training budget (10M tokens/SAE).** These are budgeted curves, not best-achievable. Alive fraction **often decreases with model size** in early/mid at low $$k$$ under this budget (with exceptions at higher $$k$$ / different depths), consistent with larger SAEs being relatively more undertrained under the same token budget. The observed $$CE_{rec}$$ gap may partly reflect differential undertraining rather than intrinsic representation difficulty. See the 100M-token check below.
- **Late depth is confounded** (final block vs non-final), and late 70M runs show very low alive fractions consistent with undertraining.
- **My $$CE_{rec}$$ definition is nonstandard.** It is not the ablation-normalized "CE loss recovered" used in Bricken et al. (2023), Gao et al. (2024), and SAE Lens. Do not compare magnitudes without converting. See Definitions section.
- **The $$R^2$$ inflation is specific to variance-normalized metrics.** Cosine similarity and relative error norm agree with $$CE_{rec}$$ that larger-model reconstructions are worse at fixed $$k$$. But more broadly, reconstruction metrics and behavioral metrics can diverge — behavioral acceptance tests (patched CE, ablation-normalized CE recovery) are the more direct measure of what matters.

## Update: A 100M-Token Mid-Layer $$k=32$$ Check (Token-Budget Sensitivity)

To probe the "fixed token budget" confound, I retrained the mid-layer $$k=32$$ TopK SAE for both models with a 10x larger training budget (100M tokens/SAE; SAE Lens v6.36.0, context size 256). Compared to the budgeted Fast2 runs:

| model | train tokens | $$k$$ | $$CE_{rec}$$ | implied $$L_{recon}/L_{orig}$$ | alive % |
|---|---:|---:|---:|---:|---:|
| 70M (Fast2) | 10M | 32 | 0.731 | 1.269 | 61.7% |
| 410M (Fast2) | 10M | 32 | 0.714 | 1.286 | 50.1% |
| 70M (100M) | 100M | 32 | 0.898 | 1.102 | 94.7% |
| 410M (100M) | 100M | 32 | 0.920 | 1.080 | 94.1% |

This suggests the mid-layer $$k=32$$ "scale hurts $$CE_{rec}$$" effect in the 10M-token sweep is at least partly an undertraining artifact: alive fraction jumps, behavior preservation improves sharply, and the scale ordering flips. This does **not** test the low-$$k$$ proxy gap (where the original effect is strongest).

---

## Next Steps (High Leverage)

1. **Bridge model at all depths:** add Pythia-160M at early/mid/late to get three-point depth-resolved comparisons.
2. **Token-budget sensitivity:** extend the 100M-token check to low $$k$$ (8, 16) and to early/late depths (optionally include 50M as a midpoint) to separate undertraining from intrinsic difficulty and equalize alive fractions.
3. **Deflated PR and anisotropy controls:** mean subtraction, top-eigen removal, to tighten the geometry story.
4. **Sensitivity-weighted distortions:** Fisher/Hessian approximations to predict loss impact better than $$R^2$$.
5. **Legibility evaluation:** SAEBench, MDL-style probing, to connect fidelity to human-usable features.

---

## Repro (Minimal)

Fast2 results: `interpretability/workspace/results/k_scaling_early-mid-late_fast2/`

Fast3 results: `interpretability/workspace/results/k_scaling_mid_fast3/`

100M-token mid-layer $$k=32$$ check: `/Users/oboh/Downloads/experiment_results.tar.gz` (metrics in `results/*.json`).

Regenerate the tables:

```bash
python3 interpretability/analyze_k_scaling_results.py \
  interpretability/workspace/results/k_scaling_early-mid-late_fast2 \
  --include-legibility \
  --markdown-out interpretability/workspace/results/k_scaling_early-mid-late_fast2/writeup_ready_tables.md
```

---

## References (Non-Exhaustive)

- Bricken et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Anthropic.
- Cunningham, Ewart, Riggs, Huben and Sharkey (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models*. ICLR 2024.
- Gao et al. (2024). *Scaling and Evaluating Sparse Autoencoders*. arXiv:2406.04093.
- Elhage et al. (2022). *Toy Models of Superposition*. arXiv:2209.10652.
- Finzi et al. (2026). *From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence*. arXiv:2601.03220.
