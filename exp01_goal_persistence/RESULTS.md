# LACUNA V2 — First Ablation Results (CPU build, July 4, 2026)

## What was run

Four arms, identical task and training budget, trained/evaluated entirely on CPU
(numpy + autograd; ~1M-param 2-layer transformers; K=4 goals, 8-way answers,
chance = 12.5%). Task: hold an assigned goal across L tokens of filler and apply
it to a query. Three rounds were needed to get a clean experiment:

1. **Round 1 flaw (caught & fixed):** query position was confounded with goal
   distance → retrained with randomized window sizes.
2. **Round 2 finding:** with *clean* filler, every arm — including plain v1 —
   scores 100% out to distance 256 (4x its training range). A lone goal token is
   trivially findable by content lookup. **No drift, nothing to fix.** Per the
   program's discipline rule, recorded as a negative result and escalated.
3. **Round 3 (the real test):** filler now contains plausible-but-wrong goal
   tokens ("distractors," ~15%). The model must keep the *originally assigned*
   goal, not just *a* goal. This is the miniature of goal drift.

## Round 3 results (results/np_sweep.csv, np_sweep.png)

| arm | accuracy (all lengths 0–256) |
|---|---|
| v1 (standard) | ~21–27% (barely above 12.5% chance) |
| v1 + prompt reminder | ~21–28% (no better than v1) |
| v1 + external monitor (oracle goal, MLP head) | 100% |
| **v2 (anchored goal channel)** | **100%, flat** |

Training-side observation, arguably the strongest result: **v1 plateaued at
loss ~1.4 and never learned the task in ~1,750 steps; v2 reached ~0.01 in 645
steps** — the anchor made goal-binding *learnable* where the standard
architecture failed to learn it at all under distractor pressure.

## Honest reading (per the discipline rule)

* **The surface fix fails.** Re-inserting the true goal into the prompt every 16
  tokens did not help at any length. A prompt reminder cannot fix a binding the
  architecture never learned. This is the "prompt fix hits a wall" evidence the
  program calls for.
* **The wrapper "succeeds" — but only by cheating in an instructive way.** The
  external monitor scores 100% because it (a) is handed the true goal from an
  external store and (b) recomputes the answer itself in its own MLP, bypassing
  the model's computation. An external goal store + external computation IS an
  anchored channel — just relocated outside the model, with the wrapper forced
  to duplicate the model's job. At toy scale duplication is trivial; at real
  scale it is not. Next experiment (`task=hard`, deeper computation) is designed
  to expose exactly this.
* **The structural fix works where surface fails.** v2 differs from v1 by one
  component (a per-goal register added into the residual stream at every layer)
  and holds 100% flat to 4x its trained context range.
* **Deviation from the pre-registered expectation, stated plainly:** we expected
  v1 to degrade *gradually with length*. Instead, with distractors, v1 fails at
  *all* lengths because it never acquires the goal-binding at all. The result is
  a *learnability* advantage for the anchored channel, not an in-context decay
  curve. This is stronger in one way (bigger gap) and weaker in another (it
  doesn't yet demonstrate length-dependent drift in a model that HAS the
  binding). Both framings belong in any writeup.

## Caveats

Single seed, one task family, tiny models, one distractor rate. The v1 plateau
could in principle be an optimization artifact of this scale — needs seed
replication and a longer-budget v1 run before any strong claim. The GPU harness
(run.py, PyTorch) is built for exactly that on the Legion.

## Next steps

1. Replicate across 3–5 seeds (GPU harness, minutes per run on the 3070 Ti).
2. `task=hard`: goal gates intermediate computation → tests whether the wrapper
   wall appears where predicted.
3. Give v1 a curriculum (start with no distractors, ramp up) — the strongest
   possible surface-side baseline before claiming a structural gap.
4. Then primitive B (friction signal).
