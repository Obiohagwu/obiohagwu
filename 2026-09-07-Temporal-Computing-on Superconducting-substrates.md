---
title: "Temporal computing on superconducting substrates?"
date: 2026-09-07
kind: research
---
Ok, I've been meaning to write this one up for abouth a month now. I started playing with superconducting optoelectronic networks as a way to understand Shainline's non-vonneumann approach to ASI, and ended up spending quite some time on a simulator and some inverse-design methods. Got a little lost in the abstractions along the way though and ended up probably wasting a lot of time, haha. Still fun.

So, more concretely, the state we're simulating is current stored in a superconducting loop. Incoming optical events contribute flux, the Josephson response changes the current, and the current decays on a timescale set by the loop. Connect several of these branches and their responses depend on when inputs arrive relative to one another.

What interests me is whether we can work backwards. Say I want a circuit to respond most strongly when B follows A by a particular interval. Can I calculate the memories and couplings it needs, then check whether the nonlinear simulation actually does that? Same question for an object moving in a particular direction at a particular speed.

### Working Backwards From Timing

There are two primary classes so far.

For order, the reduced construction uses memories that decay at different rates. Their difference has a preferred interval. We can solve for the memory constants from a requested interval and contrast, rather than searching blindly over parameters.

For motion, events arrive at different positions at different times. Choose the path delays so that a particular moving input brings those events together. Change the delay gradient and you change the preferred direction and speed. Delay-based motion detection is established prior work; the point here is making the mapping explicit and checking the resulting circuit under a fixed simulation protocol.

Then we arrive at another question, specifically asking if several settings can produce roughly the same response. Which one should we choose? I've been testing selection under parameter perturbations and constrained optimization that tries to reduce a design objective without losing the response we asked for.

### Did It Work?

| Test | Result |
|---|---|
| Signed motion, six held-out targets | All six preferred the requested direction/speed on the tested grid. Halving the timestep kept those preferences. |
| Order, five held-out targets | All five preferred intervals were within 11.4% of the requested value. Three failed the original timestep-stability check on the peak location. |
| Repeated constrained optimization, three fresh targets | Four accepted steps per target reduced the sum of memory time constants by 5.7-8.7%, while keeping the measured response features within the preset limits. |

That last objective is a proxy for a design resource. It isn't a measured energy saving. And the order result is approximate: the reduced equation has an exact inverse, but its nonlinear embedding does not inherit that guarantee.

The peak-location failure is interesting to me. The response trajectories can be quite close while the location of their maximum moves around. If the optimizer is chasing that maximum, a small numerical change can look like a meaningful design improvement. That is why the safeguards and timestep checks matter here.

### Giving It Something To Look At

The later bitmap experiment puts more of the computation together. Moving shapes arrive as pixel inputs. Simulated spatial arbors produce events, and those events feed motion and relative-timing circuits. The intermediate object events come from the simulated arbors, not a software classifier handing the circuit the answer.

In September I tested a fixed design across 5x5, 9x9, and 17x17 site grids. Those are grids of receptive-field sites, not the pixel dimensions of the images. The pilot used three crossing relationships: triangle first, simultaneous, and diamond first, with one direction and speed pairing.

| Input condition | Correct final crossings |
|---|---:|
| Clean | 9/9 |
| Moderate corruption | 6/9 |
| Strong corruption | 0/9 |

Moderate corruption meant 10% foreground dropout and 1% background clutter. Strong corruption used 25% and 3%. This was a small paired pilot, not a general vision benchmark. The misses produced no final decision rather than the wrong final class. Nine blank/clutter controls produced no final decisions either.

<img width="1920" height="1280" alt="image" src="https://github.com/user-attachments/assets/b093b951-c6a4-4fd9-b3e2-93dbc83f8829" />

The apparent improvement with grid size in the middle column isn't enough to establish a scaling benefit. The paths, durations, and noise draws differ across grids.

The more useful detail was inside the failed runs. Under moderate corruption, both objects' direction and speed labels survived in all nine cases. The relative-timing result survived in only six. The timing circuit depended on events at the crossing center, and sometimes one of those events disappeared even though there was still enough evidence elsewhere to recognize the motion.

I checked two of those failures by restoring the center patch from the clean input. Both recovered. Restoring an equally sized off-center patch recovered neither. This used clean pixels as a diagnostic intervention, so it is not a proposed recognition method. It does help locate the weakness in those two cases. At stronger noise, motion also failed, so fixing timing alone would not solve everything.

