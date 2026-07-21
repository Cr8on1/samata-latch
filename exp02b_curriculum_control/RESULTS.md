# EXP02b — Curriculum-v1 Control on SHIFT — RESULTS

Run 2026-07-06, session 4. Spec: EXPERIMENT_SPEC.md (registered before any
training this session). Written once, not edited after the fact.

## Headline

THE KILL RULE FIRED. Seed 2's curriculum-trained v1 (v1c) reached 0.917–0.979
at every eval length — including ≥ 0.917 across all trained lengths — at the
matched 1750-step budget. Per the pre-registered kill rule (any seed ≥ 0.75
across all trained lengths), **the learnability claim on shift is dead as
stated.** The exp01/02 reading "the anchor made the task learnable; v1 cannot
learn it" does not survive a distractor curriculum. Reported straight.

The failure is stochastic, not uniform: seeds 1 and 3 stayed in the marginal
basin for the full budget (final losses 1.49 / 1.42; accuracies 0.21–0.43,
never reaching the 0.75 binding bar anywhere). 1 of 3 seeds escaped. So
H-basin and H-distractor both lose as clean stories: the basin is escapable
under curriculum, but escape at this scale is seed-contingent.

## Per-seed results (15% eval distractors, n=192/length, L ∈ {0..256})

| seed | steps | final loss | acc range | trained-L min | binding (≥0.75, L≥16) |
|---|---|---|---|---|---|
| 1 | 1782 | 1.492 | 0.208–0.432 | 0.208 | NO |
| 2 (cap) | 1761 | 0.218 | 0.917–0.979 | 0.917 | YES — kill rule |
| 3 | 1730 | 1.421 | 0.224–0.401 | 0.302 | NO |
| 2 (ext 2443) | 2443 | 0.028 | 0.990–1.000 | 0.990 | YES |

Loss-curve reading (results/loss_curves.npz, plot in
results/np_sweep_curr_all.png): seed 2 escaped the basin at step ~310, when
the ramp was only at d ≈ 4.6% — the clean-phase window did its work early.
Seeds 1 and 3 never escaped (min logged losses 1.217 / 1.214, i.e. they
touched the basin floor ln4 ≈ 1.386 region and dipped only marginally below
before reverting; seed 1 had a transient dip to ~1.05 near step 930 that did
not survive the full-pressure phase).

## Pre-registered prediction outcomes

1. Prediction 1 (H-basin: loss in [1.2,1.6], acc ≤ 0.35 at every L ≥ 16,
   every seed) — **FAILED in seed 2** (massively: 0.917+ everywhere), and
   breached in single cells elsewhere, all reported: seed 1 L=16 (0.422) and
   L=64 (0.411); seed 3 L=16 (0.401). Seeds 1 and 3 otherwise consistent
   with the basin account.
2. Prediction 2 (elevated L=0 via last-token reading) — mild at best: seed 1
   L=0 = 0.432, seed 3 L=0 = 0.318; neither shows the clean 1.0-at-L=0
   signature. Seed 2's L=0 is part of across-the-board binding, not the
   last-token pattern.
3. Prediction 3 (no seed reaches the binding criterion) — **FAILED** (seed 2).

## Extension (pre-registered contingency, disclosed)

Seed 2's final cap loss (0.218) was < 1.2, triggering the registered
extension. Training stopped at step 2443, before the 3500 max, on the
program's standing sustained-convergence criterion (loss < 0.03 sustained;
0.014/0.011/0.028 over the final bursts) — disclosed here as an
interpretation of the contingency's stopping rule, decided when the criterion
was met, not after seeing eval numbers. Extended eval: 0.990–1.000 at every
length, flat to 4× the trained range. Cap-1750 CSV for seed 2 was regenerated
from the preserved cap checkpoint after the extension overwrote it; the
regenerated sweep reproduced the original printed cap numbers exactly (same
eval RNG stream) — noted for the record.

## Diagnostic sweep (0% eval distractors)

Seeds 1/3 score higher clean than under distractors (seed 3: up to 0.578 clean
vs 0.302 trained-min at 15%; seed 1: 0.458 vs 0.208) — the pre-registered
"circuit-formed-but-distractor-broken" third reading is present in weak form
in both non-escaping seeds. Whatever partial structure they built is above
the 0.25 ceiling clean, and distractors break it.

## What this does to the program (straight)

* **Dead:** "v1 cannot learn shift; the anchor makes it learnable." A
  data-side curriculum unsticks v1 without touching architecture. Combined
  with exp03 (v1 learns hard unaided), no task in the current family
  supports an architectural learnability claim.
* **Weakened:** the exactness gap. On this seed, extended v1c reaches
  0.990–1.000 flat to L=256 — near-v2-exact on shift. Exactness +
  length-invariance survives untouched only on hard (exp03: v1 drifts
  0.87→0.79 with L; v2 flat 1.000), where it was rescoped anyway.
* **Alive, and now the sharpest toy-scale signal: reliability.** v2 converged
  and hit 1.000 in every cell of every seed it has ever been run in (9/9
  seeds across exp01/02/03, 507–547 steps, no curriculum, no tuning). v1
  needs either a steep task gradient (hard) or a curriculum, and with the
  curriculum still fails 2 of 3 seeds at matched budget. The candidate claim
  going forward: the anchor buys seed-invariant, curriculum-free,
  length-invariant exactness — a variance/robustness claim, not a
  possibility claim. It needs more seeds to be quantified properly.
* exp05 (multi-hop, non-(g,q)-function task) is now carrying more weight: it
  is the remaining place where an architectural wall — rather than a
  reliability gap — could still show.

## Outputs

code/ (diff vs exp03: `_drate` float distractor rate in data.py; `train_curr`,
`eval_curr`, v1c tag in np_run.py; nothing else), checkpoints/seed1,
checkpoints/seed2_cap1750, checkpoints/seed2_ext2443, checkpoints/seed3,
results/np_sweep_curr_s{1,2,3}.csv (+ _diag), results/summary.csv,
results/loss_curves.npz, results/np_sweep_curr_all.png,
results_ext/np_sweep_curr{,_diag}_s2_ext2443.csv.
