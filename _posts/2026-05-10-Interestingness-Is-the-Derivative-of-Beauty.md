---
title: "Interestingness Is the Derivative of Beauty"
date: 2026-05-10
kind: essay
---

Schmidhuber has a style I really like: take a thing people usually protect as
vibes, then ask what quantity would have to change inside an observer for the
vibe to happen.

In [Driven by Compression Progress](https://arxiv.org/abs/0812.4360), the
central move is not merely "beauty is compression." That would be too static.
The sharper claim is that many of the lively things - curiosity, fun,
interestingness, jokes, aesthetic reward, scientific delight - are closer to
**first differences** of compression.

Beauty is a state. Interestingness is a slope.

If some data stream \(x\) takes \(C_t(x)\) bits for my current internal
compressor to encode, then one simple proxy for subjective beauty is:

$$
B_t(x) = -C_t(x)
$$

The object feels more beautiful, simpler, or more lawful when it admits a
shorter description relative to the machinery I already have. But the object
feels **interesting** when the description length is actively falling:

$$
I_t(x) = B_{t+1}(x) - B_t(x)
$$

or equivalently:

$$
I_t(x) = C_t(x) - C_{t+1}(x)
$$

The sign matters. Positive interestingness means I can now describe the same
thing with fewer bits than before. Something in me got better at seeing it.

That is the derivative trick.

---

### The derivative dictionary

This way of thinking turns a bunch of fuzzy nouns into a small computational
grammar:

| Phenomenon | Static version | Derivative version |
| --- | --- | --- |
| Beauty | low description length | high compressibility |
| Interestingness | novelty | improvement in compression |
| Boredom | familiar order | low or zero compression progress |
| Noise | high Shannon surprise | no learnable compression progress |
| Humor | incongruity | sudden recompression |
| Science | facts | laws that shorten the code for facts |
| Art | pattern | a curriculum for perceptual model updates |
| Creativity | making artifacts | producing data that creates compression progress |

The important part is that the reward does not live only in the object. It
lives in the relation between object and observer over time.

A proof can be beautiful because it compresses a family of facts into one
argument. But the pleasure of the proof is often the moment when the argument
lands and many separate things become one thing. A joke works similarly. The
setup is locally awkward, ambiguous, or undercompressed. The punchline changes
the codebook. Suddenly the setup is cheaper to encode.

The feeling is the update.

---

### Why random noise is not interesting

This is where Schmidhuber's formulation is cleaner than a raw-surprise account.

Random noise has high Shannon information. It is hard to predict. It keeps
surprising you. But it does not give you a useful learning curve. You do not get
better at compressing it, except in the trivial sense of learning "this source
is noise."

So the four regimes look like this:

| Regime | Compressible now? | Becoming more compressible? | Felt quality |
| --- | ---: | ---: | --- |
| Solved pattern | yes | no | pleasant, maybe boring |
| Random noise | no | no | irritating, blank, meaningless |
| Learnable pattern | not yet | yes | interesting |
| Great compression event | suddenly yes | strongly yes | insight, joke, elegance |

This is the anti-noisy-TV move. A system that rewards prediction error can get
stuck watching inherently unpredictable input. A system that rewards **reducible
error** asks a better question: did my model improve?

That distinction shows up all over later curiosity work. Oudeyer, Kaplan, and
Hafner's survey of [intrinsic motivation
systems](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/neuro.12.006.2007/full)
emphasizes learning progress: agents should seek situations that are neither
already mastered nor hopelessly unpredictable. Modern RL papers such as
[Pathak et al.](https://proceedings.mlr.press/v70/pathak17a.html) and
[Houthooft et al.](https://arxiv.org/abs/1605.09674) use different machinery,
but they live in the same neighborhood: build an intrinsic reward out of model
change, prediction improvement, or information gain rather than external reward
alone.

Schmidhuber's version feels especially clean because he phrases the whole thing
in compression language. The agent is not rewarded for surprise. It is rewarded
for becoming a better compressor of its own history.

---

### The "same data" clause is doing real work

There is a subtle technical point in the paper that makes the idea much less
hand-wavy.

You should not reward the agent merely because the new data is easier than the
old data. You reward it when a **new compressor** encodes relevant data better
than the **old compressor**. In the idealized version, both compressors are
evaluated on the same observation history.

Very roughly:

$$
r_{\text{int}} = C_{\text{old}}(h) - C_{\text{new}}(h)
$$

where \(h\) is the agent's history. If the new compressor saves bits on the
same history, the agent has made progress. This makes the reward about learning,
not about accidentally walking into an easier room.

That is why the first-derivative framing matters. The object is not assigned a
single eternal aesthetic score. The agent gets paid for an update to its own
model.

The world says: here is a stream.

The mind says: I can now encode more of it.

The reward says: go find more streams like that.

---

### Older aesthetics were often too static

One reason the paper feels satisfying is that it upgrades older mathematical
aesthetics from object-level scores into observer-level dynamics.

Birkhoff's old aesthetic measure, often summarized as order divided by
complexity, has the right smell: beauty has something to do with the ratio
between structure and cost. But it treats beauty more like a property of the
artifact.

Schmidhuber makes it subjective and temporal. The same artifact can be boring to
one observer, overwhelming to another, and fascinating to a third because their
compressors sit at different points on the learning curve.

This also connects naturally to the Wundt/Berlyne intuition that pleasure often
peaks at intermediate novelty or complexity. Too simple is boring. Too chaotic
is noise. The sweet spot is not "medium complexity" in the abstract. The sweet
spot is where your current model can make progress.

That is more precise:

$$
\text{aesthetic reward} \approx \frac{d}{dt}\text{subjective simplicity}
$$

Not simplicity alone. The acquisition of simplicity.

---

### Jokes, proofs, music, and science

The derivative story gives a common shape to things that normally seem far
apart.

**A joke** creates a delayed codebook switch. The setup invites one model. The
punchline installs another. The whole sequence is suddenly shorter under the new
model, and the compression delta arrives as laughter.

**A proof** compresses cases into mechanism. A bad proof can verify a result
without reducing much cognitive load. A beautiful proof makes the theorem feel
inevitable because the final representation is shorter than the pile of facts
you started with.

**Music** often plays with partial predictability. Total repetition stops
paying rent. Total randomness never becomes yours. A good motif gives your
predictive machinery enough regularity to lock onto, then enough variation to
keep producing small updates.

**Science** is the large-scale version. Theories are compressors. Newton
compresses a field of falling and orbiting things. Maxwell compresses
electricity, magnetism, and light. Evolution compresses biological mess into a
generative principle. The scientist's reward is not just "the world is lawful."
It is the event of lawfulness becoming available.

This is what I like about the paper: it does not drain mystery out of the
examples. It gives the mystery handles.

---

### Creativity as compressor hacking

Schmidhuber's [broader formal theory of creativity, fun, and intrinsic
motivation](https://people.idsia.ch/~juergen/creativity.html) pushes the same
idea further: a creative agent searches for artifacts, actions, experiments, or
data streams that trigger future compression progress.

That makes creativity less like an ornamental human faculty and more like a
control problem:

1. I have a compressor.
2. I can act to generate or select data.
3. Some data will improve the compressor.
4. I should seek the data with high expected improvement.

In this view, a painter, mathematician, musician, scientist, and child at play
are all doing versions of the same thing. They construct curricula for their own
model updates.

The artist is not merely making a pretty object. The artist is shaping an
experience whose perceptual description length changes as you move through it.

The scientist is not merely collecting facts. The scientist is arranging contact
with reality so that the mind is forced into a shorter theory.

The child is not merely messing around. The child is sampling the edge between
mastery and confusion.

This also explains why "taste" is not just preference. Taste is a detector for
promising gradients in model space.

---

### Relation to information gain

There is a close cousin of compression progress: Bayesian information gain.

Instead of asking "how many bits did my compressor save?", an agent can ask "how
much did this observation change my posterior over models?" That is the flavor
behind Bayesian surprise work by Itti and Baldi and exploration methods such as
VIME. The common theme is that the agent is rewarded for epistemic movement, not
for external payoff.

The difference is mostly what object receives the derivative:

| Family | Reward roughly tracks |
| --- | --- |
| Prediction error | how wrong the model is |
| Learning progress | how much less wrong the model becomes |
| Compression progress | how many fewer bits the history costs |
| Bayesian surprise | how much the belief state changes |
| Information gain | expected reduction in uncertainty |

Schmidhuber's formulation is especially aesthetically pleasing because it
collapses learning, attention, art, humor, and science into the same variable:
improvement in the code.

When the code improves, the world gets shorter.

---

### Why this feels like a deep move

The general pattern is:

$$
\text{static quality} \rightarrow \text{change in static quality}
$$

That move turns nouns into training signals.

Beauty becomes the current compressibility of experience. Interestingness
becomes the derivative of beauty. Curiosity becomes action selection under
expected future interestingness. Creativity becomes the production of artifacts
or situations that cause compression progress in oneself or others.

You can keep going:

$$
\text{fun} \approx \Delta\text{competence}
$$

$$
\text{insight} \approx \Delta\text{description length}
$$

$$
\text{taste} \approx \nabla\text{future compression progress}
$$

The paper's charm is that it makes this reduction without making the world feel
smaller. "Beauty is compression" can sound flat. "Interestingness is the first
derivative of subjective beauty" sounds alive because minds are not static
measuring devices. They are learning systems with histories.

The same object can be dead, noisy, beautiful, or electric depending on where it
sits relative to your current compressor.

That is the good part.

The magic is not in compressibility alone. It is in becoming able to compress.

---

### Notes

- Schmidhuber, [Driven by Compression Progress: A Simple Principle Explains
  Essential Aspects of Subjective Beauty, Novelty, Surprise,
  Interestingness, Attention, Curiosity, Creativity, Art, Science, Music,
  Jokes](https://arxiv.org/abs/0812.4360), 2008/2009.
- Schmidhuber, [Formal Theory of Creativity, Fun, and Intrinsic
  Motivation](https://people.idsia.ch/~juergen/creativity.html).
- Oudeyer, Kaplan, and Hafner, [Intrinsic Motivation Systems for Autonomous
  Mental Development](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/neuro.12.006.2007/full), 2007.
- Itti and Baldi, [Bayesian Surprise Attracts Human
  Attention](https://papers.nips.cc/paper/2822-bayesian-surprise-attracts-human-attention), 2005.
- Pathak et al., [Curiosity-driven Exploration by Self-supervised
  Prediction](https://proceedings.mlr.press/v70/pathak17a.html), ICML 2017.
- Houthooft et al., [VIME: Variational Information Maximizing
  Exploration](https://arxiv.org/abs/1605.09674), 2016.
- Birkhoff, [Aesthetic Measure](https://openlibrary.org/books/OL6287206M/Aesthetic_measure), 1933.
