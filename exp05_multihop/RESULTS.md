# EXP05 — Goal-Gated Multi-Hop Retrieval: RESULTS

Run 2026-07-06, session 5. Spec: EXPERIMENT_SPEC.md (written before any
training; baselines computed and MC-verified before thresholds). Seeds 1–3,
all arms, matched 1750-step cap. This file is written once and never edited
after the fact (Corrections section only).

## Headline

OUTCOME: H-NEITHER-LEARNS, all three seeds, unambiguous. No model — v1, v2,
v1c (curriculum), v1+reminder, or monitor — learned hop2 at this scale and
budget. Every cell of every arm of every seed, both sweeps (15% and 0%
distractors), sits in the chance band: min 0.073, max 0.219 (chance 0.125,
n=192/cell). The pre-registered wall question is INCONCLUSIVE: a wall cannot
be shown where the structural arm also fails.

THE ON-RECORD PROGRAM PREDICTION (H-wall: v2 succeeds, surface arms fail)
FAILED on its v2 half. Reported straight. This is the second program
prediction to fail (exp03's oracle-monitor wall prediction was the first).

## Scope consequence (the real finding)

The seed-invariant exactness claim is now explicitly SCOPED: the anchor buys
seed-invariant, curriculum-free, length-invariant exactness on tasks whose
goal-conditioned rule the backbone can represent and learn (shift, hard —
non-compositional (g,q)-functions). It does NOT extend to compositional
retrieval at this scale: the anchor guarantees the goal is never lost, but
hop2's bottleneck was never goal retention — it was building two dependent
context lookups, and Primitive A contributes nothing to retrieval formation.
Anchoring is orthogonal to retrieval capacity.

## Numbers

Final losses at cap (all arms, all seeds): 1.89–2.05, i.e. pinned between
the table-ignorant floor ln 7 ≈ 1.946 and chance ln 8 ≈ 2.079, never
approaching the goal-ignorant floor ln 4 ≈ 1.386. The extension contingency
(loss < 1.2 at cap) fired for NO arm in NO seed. Early-stop fired nowhere.

Accuracy summary (results/summary.csv; per-cell CSVs + per-goal breakdowns in
results/): every arm × seed has mean accuracy 0.127–0.149 across the 18-cell
grid (9 lengths × 2 sweeps). Loss trajectories: slow drift 2.08 → ~1.95 over
1750 steps in every arm — models learned q-exclusion (answer ≠ query) and
nothing else. The monitor's best final loss (1.89, seed 1) is marginally
below ln 7, consistent with weak feature scraps, nowhere near usable.

Per-goal cells show no structure: no goal class above 0.25 at L ≥ 16
anywhere; no below-chance wrong-table/one-hop signature anywhere (the models
did not learn a wrong retrieval — they learned no retrieval). No
diagnostic-sweep divergence (0% ≈ 15% everywhere): no fragile circuit formed,
in any arm, including v1c after its clean curriculum window.

## Pre-registered commitments, honored

* v2 < 0.75 at every L ≥ 16, every seed → H-neither-learns territory per
  spec: exp05 is inconclusive on the wall; exactness claim scoped to
  non-compositional tasks. So reported.
* v2 = 1.000-every-cell prediction: BREACHED comprehensively (max v2 cell
  0.214). Reported.
* Kill rule: did not fire — no surface arm reached 0.75 anywhere (max
  surface cell 0.219, v1+reminder seed 2 diag L=16).
* mon_c contingency: did not trigger (v1c max cell 0.182).
* v1 basin sub-prediction (table-ignorant, loss → ~1.95, not goal-ignorant
  1.386): CORRECT in all seeds — the one part of the program prediction
  that held.
* Every arm ran to the exact 1750 cap; no reruns, no post-hoc eval choices;
  eval used goal-balanced cells (48/goal), disclosed in code comments.

## Reading

1. hop2 (two dependent lookups among 32 in-context pairs, 4 tables, under
   left-padding and distractors) exceeds the learnable capacity of this
   config — 3 layers, D=48, H=2, ~1750 steps, batch ~9–32 — for BOTH
   architectures. The task gradient gives no foothold: unlike shift/hard,
   partial circuits (1 hop, wrong table) earn ZERO accuracy by construction,
   so there is no incremental path the optimizer can climb at this scale.
   The design choice that made the baselines sharp (distinct candidates,
   0-scoring partial strategies) may also have removed the learning
   curriculum the task itself would otherwise provide. Noted for redesign.
2. Nothing here revives learnability claims for the anchor, and nothing here
   damages the surviving claim — it simply bounds its scope: exactness where
   the backbone can learn the rule at all (9/9 seeds, exp01/02/03 stand).
3. The wrapper-wall question remains OPEN and moves to scale: exp04's GPU
   run (Legion 3070 Ti, ~50x params, longer budgets) should carry hop2 —
   if hop2 becomes learnable at scale, the wall test becomes live again
   (H-wall vs H-rescue discriminate exactly as specced here).

## Next (for the roadmap)

* exp04 scale check should now carry three payloads: (a) multi-seed
  reliability quantification of the exactness claim, (b) hop2 at scale —
  wall test live if learnable, (c) optionally a hop2 curriculum variant
  (e.g., hop-count or table-count ramp) as the pre-registered learnability
  contingency, per the post-exp02b standing rule.
* Any hop2 redesign for CPU scale should reconsider the zero-partial-credit
  property: a task variant where 1-hop earns partial accuracy (e.g.,
  non-distinct candidates, or 1-hop queries mixed in) provides the gradient
  foothold hop2 lacked, at the cost of muddier baselines. Trade-off is now
  documented, decision deferred to the next spec.

## Corrections

* 2026-07-06 (same session, no data changed): the "Next" section above was
  written under the then-current roadmap (exp04 GPU scale check next). Later
  the same session, Creighton revised the program roadmap: ALL toy-scale
  work first (Primitive B, Primitive C, toy A+B+C), then ONE consolidated
  large-scale campaign, which absorbs everything the "Next" section assigns
  to exp04 (reliability quantification, hop2-at-scale wall test, curriculum
  contingency). The technical content of "Next" stands; only the ordering
  and destination changed. See NEXT_SESSION.md and program_log.md.
