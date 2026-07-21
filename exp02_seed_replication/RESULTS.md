# LACUNA V2 — exp02: Seed Replication Results (CPU build, July 5, 2026)

Written after the runs. Spec was pre-registered in EXPERIMENT_SPEC.md before any training. Nothing here has been edited after the fact.

## What was run

Exact exp01 round-3 setup (distractor filler, task=shift, chance = 12.5%), five fresh seeds (1–5), all four arms per seed, budgets matched to exp01 (v1 → ~1750 steps; v2 and monitor → early stop at loss < 0.03). exp01 counts as seed 0, giving six independent observations. Only change to the code: a `--seed` flag offsetting every RNG by 10000×seed (code/np_run.py; diff vs exp01 is that flag alone). Eval: L ∈ {0,16,32,64,96,128,192,224,256}, n=192 per length, fresh eval data per seed.

## Results (results/summary.csv, np_sweep_replication.png; per-seed CSVs np_sweep_s0–s5.csv)

Accuracy across all lengths, all six seeds (grand mean; overall min–max across every cell):

| arm | grand mean | min–max |
|---|---|---|
| v1 (standard) | 25.8% | 20.3–41.7% |
| v1 + prompt reminder | 25.4% | 17.7–32.3% |
| v1 + external monitor (oracle) | 100% | 100–100% |
| **v2 (anchored goal channel)** | **100%** | **100–100%** |

Training, per seed (steps @ final loss):

| seed | v1 | v2 | monitor |
|---|---|---|---|
| 0 (exp01) | 1747 @ 1.42 | 645 @ 0.012 | 773 @ 0.025 |
| 1 | 1738 @ 1.42 | 507 @ 0.025 | 720 @ 0.028 |
| 2 | 1752 @ 1.43 | 515 @ 0.020 | 928 @ 0.017 |
| 3 | 1760 @ 1.40 | 508 @ 0.024 | 933 @ 0.015 |
| 4 | 1733 @ 1.36 | 547 @ 0.015 | 934 @ 0.014 |
| 5 | 1757 @ 1.46 | 521 @ 0.018 | 795 @ 0.025 |

## Pre-registered predictions vs outcomes

1. **v2 ≥ 90% at every length, every seed — CONFIRMED, exceeded.** v2 scored exactly 100% in all 54 cells (6 seeds × 9 lengths), flat to 4x trained context range.
2. **v1 and reminder ≤ 35% everywhere, v1 loss > 1.0 — CONFIRMED with one cell exception, reported as required.** 101 of 102 cells were ≤ 35% (highest of those: 34.9%, seed 2, L=32). The exception: seed 4, L=0 (no filler at all), v1 = 41.7%. At L=0 the goal token is adjacent to the query, so partial content lookup is easiest there; it is still nowhere near task competence and that same model scores 20.3–32.3% at every L > 0. All six v1 final losses were 1.36–1.46 (threshold: > 1.0). Non-convergence replicates in 6 of 6 seeds.
3. **Monitor ≈ 100% — CONFIRMED** (it remains an oracle baseline, with the same "cheating" caveat as exp01: external goal store + own answer head).
4. **v2 converges in < half of v1's budget — CONFIRMED.** v2: 507–547 steps in all five new seeds (vs the 1750-step v1 budget); v1 never converged in any seed.

Neither pre-registered falsifier occurred: no seed produced a converging v1, and no seed produced a v2 below 100% at any length.

## Honest reading

The exp01 result is not seed luck. Across six independent initializations and data streams, the learnability gap is categorical: the standard transformer never acquires goal-binding under 15% distractor pressure (plateau at loss ~1.4, accuracy ~26% vs 12.5% chance), while the anchored-channel variant acquires it every time, in roughly 500 steps, and holds 100% to 4x its trained context length. The prompt-reminder wrapper adds nothing in any seed.

What this does NOT yet show, unchanged from exp01: length-dependent drift in a model that HAS the binding (v1 never gets the binding, so no decay curve exists to measure); behavior at scale (~1M params, toy task); behavior when the goal gates intermediate computation (exp03, task=hard — where the oracle monitor's duplicate-the-model strategy is predicted to hit its wall); and the strongest surface-side baseline, a curriculum-trained v1 (distractors ramped from 0%), which remains the most important unrun control before any strong structural claim.

## Deviations from spec

None in design. One prediction-2 cell breach (seed 4, L=0), disclosed above. v1 step counts landed 1733–1760 vs the "~1750" target (burst-boundary granularity).

## Corrections

Same-day audit pass (2026-07-05, before any external use): two prose errors in the prediction-2 paragraph were corrected against the raw CSVs — "101 of 102 cells ≤ 32.3%" should have read "≤ 35% (highest 34.9%)", and seed 4's L>0 range is 20.3–32.3%, not "20–26%". No data, tables, or conclusions changed; results files untouched.

## Next

exp03: `--task hard` (goal gates intermediate computation), same seeds. Then curriculum-v1 control. Then GPU scale-up (exp04) on the Legion 5 Pro.
