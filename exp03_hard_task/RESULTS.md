# EXP03 — Hard Task: RESULTS

Date: 2026-07-05, session 3. Written after all training and eval, against the pre-registered spec (EXPERIMENT_SPEC.md, including Amendment 1 made before any training). Raw per-seed data: results/np_sweep_hard_s{1,2,3}.csv; aggregate: results/summary.csv, results/np_sweep_hard_replication.png.

## Headline

The learnability result from exp01/exp02 DOES NOT transfer to task=hard — it reverses. v1 (standard transformer), which never converged on `shift` in six seeds, learned substantial goal-binding on `hard` in all three seeds (accuracy 0.68–0.97 across all cells, far above the 0.469 goal-ignorant ceiling). The pre-registered falsifier condition (v1 converges) fired and is reported as such.

What survives — and sharpens — is the exactness/invariance result: v2 (anchor) scored 1.000 in all 27 cells (3 seeds × 9 lengths), while every baseline is imperfect and length-sensitive. And per the program's discipline rule, exp03 is NOT a clean structural-win cell: the prompt fix fails, but the add-on monitor does not fail by the pre-registered bar.

## Scorecard vs pre-registered predictions

1. v2 ≥ 0.90 everywhere — **PASS, exceeded**: 1.000 in 27/27 cells, to 4× trained length, every seed.
2. (As amended before training) v1 and reminder ≤ 0.55 everywhere — **FAIL in all three seeds**. v1 range 0.68–0.97; reminder 0.75–0.96. Both materially exceed the 0.469 goal-ignorant ceiling and the 0.75 goal-binding bar in most cells: v1 genuinely learned the goal on this task. v1 losses confirm (s1: 0.016 — full convergence; s2: ~0.26; s3: ~0.55 vs marginal-optimal ≈ 1.08).
3. Monitor hits its wall (< 0.90 at ≥ 1 length) — **FAIL (26/27 cells ≥ 0.90)**. Monitor sits at 0.90–0.98 (no longer pinned at ~1.00 as on shift). Single breach cell, disclosed: seed 2, L=224, 0.896 — within noise of the 0.90 line (n=192), not a wall. The Amendment-1/interpretation-commitment reading applies: `hard` is still a pure function of (goal, query), so an oracle-goal MLP wrapper can largely memorize the 32-entry table; "goal gates intermediate computation" constrains the sequence model, not an output wrapper. The wall claim remains untested and needs a task where the answer is not computable from (g, q) alone — goal-gated multi-hop retrieval over the context is the queued exp05 candidate.
4. v2 converges in < half of v1's budget — **PASS**: 513/522/543 steps vs 1750 (same 507–547 band as on shift; the anchor's convergence speed is task-insensitive so far).

## The interesting inversion (post-hoc reading, labeled as such)

Why did v1 learn `hard` but not `shift`? The loss geometry differs. On `shift` the goal-ignorant optimum is uniform over 4 answers (cross-entropy ln 4 ≈ 1.386) — exactly where v1 plateaued in exp01/exp02 (1.36–1.46): the marginal strategy is a strong local basin and the marginal payoff for attending to the distant goal token is symmetric and small. On `hard` the four goal-rules collide unevenly (goal-ignorant optimum ≈ 1.08, accuracy ceiling 15/32 ≈ 0.469), and the multiplicative rule makes errors from ignoring g much more costly per example — the gradient toward goal-attention is steeper. Observed v1 trajectories dropped through the 1.08 marginal floor by ~800 steps in every seed. Harder task, better learning signal. This is a genuinely useful finding for the robustness framing: task difficulty and binding learnability are not monotonically related, so the anchor's value cannot be argued from learnability alone — it must rest on exactness and length-invariance, which is what v2 uniquely delivers here.

## What v2 uniquely delivers on hard

* Exactness: 1.000 in every cell vs best-baseline 0.984 (monitor, one cell) and typical baseline 0.79–0.96.
* Length-invariance: v1 mean drifts 0.87 → 0.79 from L=0 to L=256 (seed 3: 0.875 → 0.682); v2 is flat at 1.000 with zero variance across seeds.
* Convergence: ~3.3× fewer steps than v1's budget, unchanged from shift.

## Disclosures

* Pre-registered threshold breaches: prediction 2 failed in all seeds (reported above); prediction 3 failed (single sub-0.90 monitor cell disclosed, not counted as a wall).
* mon seed 1 stopped at the step cap without reaching the loss criterion (1868 steps, loss 0.045; criterion < 0.03). Its eval (0.91–0.96) may modestly understate a fully-converged monitor. mon s2/s3 converged (0.009/0.021).
* mon seed 3 training: one burst accidentally launched two processes on the same checkpoint; both trained the same weights independently and the surviving save is a valid checkpoint (restarts re-seed the data stream by design), but that burst's data stream is not exactly reproducible from the seed alone.
* v1 seed 1 trained to 1758 steps, seed 2 to 1739, seed 3 to 1745 (budget "~1750").
* Cosmetic code bug, not fixed post-hoc: eval's final console message prints the unpatched filename ("np_sweep_s{N}.csv"); the files actually written are correctly task-tagged (np_sweep_hard_s{N}.csv). Output files verified present and consistent.
* Eval reminder arm at L=256 is NaN by construction (reminder tokens push sequence past T=288), as in exp01/exp02.

## Corrections (2026-07-05, same-session audit pass; no data changed)

* The marginal-optimal (goal-ignorant) cross-entropy on `hard` is **1.0397 ≈ 1.04**, not "≈ 1.08" as stated in three places below/above (scorecard item 2, the inversion section twice). The 1.08 figure came from a hand-calculation that missed q=4 collapsing to only two possible answers. All dependent claims survive: v1 trajectories still fell through the true 1.04 floor by ~800 steps in every seed, and the accuracy ceiling 15/32 ≈ 0.469 was computed correctly and is unaffected.
* Scorecard item 2: the reminder arm's range is **0.74–0.96** (min cell 0.7448, s3 L=128), not "0.75–0.96".
* Both errors found by recomputing every prose claim from the raw CSVs and analytic formulas; per program rules the original text is left in place and corrected here.

## Consequences for the program

1. The claim "the anchor makes goal-binding learnable where v1 cannot learn it" is now scoped to shift-class tasks; on hard-class tasks the claim is "the anchor makes goal-binding exact and length-invariant where v1 is approximate and degrading."
2. Discipline rule: hard is not a both-baselines-fail cell (monitor ≥ 0.90). The structural-necessity argument still rests on shift (exp01/exp02, where reminder failed and only the oracle-cheating monitor matched v2) and awaits a wrapper-inaccessible task.
3. exp05 candidate (queued): goal-gated multi-hop retrieval — answer requires goal-dependent intermediate reads of the context, so it is not a function of (g, q) and an output-only wrapper cannot table-memorize it.
4. Curriculum-v1 control (queued before any strong structural claim, per exp02 plan) — now doubly motivated: if curriculum also unsticks v1 on shift, the learnability gap narrows further and exactness/invariance carries the whole claim.
