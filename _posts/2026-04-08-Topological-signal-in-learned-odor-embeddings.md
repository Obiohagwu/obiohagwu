
Got nerdsniped recently.

Q:
> if a learned odor embedding appears to show nontrivial topology, does that signal survive metric choice, subsampling, null models, checkpoint variation, and comparison to ordinary chemical baselines?

That is a much more defensible question than "what is the topology of odor space?"

Useful links:

- [paper (PDF)]({{ '/images/odor-topology/robust-but-not-unique-odor-topology.pdf' | relative_url }})
- [code and analysis repo](https://github.com/Obiohagwu/odor-topology)

---

### What I Actually Tested

The representation under audit was OpenPOM, the open implementation of the Principal Odor Map line of work.

The comparison stack was:

- two primary OpenPOM checkpoints for matched direct comparisons
- all `10` released OpenPOM ensemble checkpoints for stability checks
- a curated `4,983`-molecule GoodScents/Leffingwell table
- a broader `5,862`-molecule GS/LF table
- a `1,600`-molecule non-overlap subset
- Morgan bit fingerprints
- Morgan count fingerprints
- a small RDKit physicochemical descriptor baseline

The main target was persistent `H1`, not "all topology." The whole point was to see whether any apparent loop-like structure was actually robust enough to survive the usual failure modes.

---

### The Main Result

The shortest honest summary is:

- POM really does show reproducible `H1` signal
- that signal survives repeated subsampling, matched nulls, dataset changes, and checkpoint variation
- but it is **not unique to POM**

That last point matters. The strongest Morgan bit fingerprint baselines were at least as strong and often stronger than POM on the raw robustness ratios used here.

So this is **not** a "we found hidden loops in odor space and POM wins" paper.

It is a much narrower and, I think, more useful result:

> topological signal in learned odor representations is real enough to survive serious scrutiny, but current evidence does not justify treating that topology as uniquely favorable to POM.

---

### Why I Still Think The Result Is Interesting

The part I find most interesting is not the trivial leaderboard question of which space gets the biggest persistence ratio.

The interesting part is the compression mismatch.

POM here is a `256`-dimensional dense learned embedding. The strongest chemical baselines are `2048`-bit sparse combinatorial encodings. Those are not the same kind of objects.

So we have:
- sparse fingerprints preserve lots of raw combinatorial chemical detail, which can naturally produce strong persistent structure
- POM is much more compressed, so the fact that it still retains robust `H1` at all is nontrivial
- but that still does **not** prove that POM preserves *better* odor-relevant structure

That distinction is basically the spine of the paper now.

If I push this line further, the next version would likely compare POM against **matched-bottleneck chemical baselines** rather than only against much larger sparse spaces. PCA-compressed fingerprints, random projections, or autoencoder-compressed baselines would make the fairness question much cleaner.

---

### Utility: Beyond descriptive topology?

I also did a fairly narrow first utility pass, because I did not want the whole project to stop at "look, persistence diagrams."

The utility question was:

> do topology-derived local features explain neighborhood-level odor-label structure better than local geometry alone?

The answer was cautiously yes, mostly in a limited and target-dependent way.

Local topology sometimes improved prediction of neighborhood ambiguity / entropy style targets beyond local geometry. But again, the gains were not uniquely favorable to POM. Some ordinary chemistry baselines seemed to also benefit.

So the utility result is quite modest:

- topology can add explanatory value
- the gains are conditional and not currently enough to claim that topology is a uniquely powerful odor-modeling advantage
---

### Clear ?'s

I want to be very explicit about the non-claims here.

I am **not** claiming:

- that this reveals the true topology of human odor perception
- that persistent `H1` implies clean interpretable circular perceptual dimensions
- that current learned odor embeddings uniquely beat strong chemical baselines
- that topology has already been shown to confer large practical gains for odor design

The paper is strongest as a representation audit.

That is still worth something. In a lot of modern embedding work, people jump from "the space predicts well" to "the geometry must mean something deep." This project is an attempt to put that move under pressure in a domain where the representations are interesting, but the claims can get fuzzy very quickly.

---

### Why?
I think there is real value in results that sharpen what a representation does and does not justify.

The cleanest positive claim from this line is mostly:

- "compressed learned odor spaces can preserve real topological structure, but robust topology is not automatically a learned-embedding-only phenomenon"

That is a stricter and less glamorous claim, but it is the one I can currently defend.

If you want the compact artifact version, the paper is here again:

- [Topological signal in learned odor embeddings (PDF)]({{ '/images/odor-topology/robust-but-not-unique-odor-topology.pdf' | relative_url }})
