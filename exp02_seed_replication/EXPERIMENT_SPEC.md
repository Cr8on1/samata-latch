# EXP02 — Seed Replication (pre-registered spec)

Written BEFORE any exp02 training run. Date: 2026-07-05. CPU build (numpy/autograd), same sandbox class as exp01.

## Question

Is the exp01 result (v1 fails to learn goal-binding under distractor pressure; v2 anchor learns it and holds 100% to 4x trained context) reproducible across independent random seeds, or was it luck of one initialization/data stream?

## Design

Identical to exp01 round 3 in every respect except the random seeds. Same code (copied from exp01/code, one change: `--seed` parameter offsets every RNG by 10000×seed and gives each seed its own checkpoint dir and results file). Same config: K=4, V_ans=8 (chance 12.5%), V_filler=32, 2 layers, H=2, D=48, T=288, train_max_len=64, task=shift, distractors=15%.

Arms (all four, as in exp01): v1 (standard), v1+prompt-reminder (every 16 tokens), v1+external monitor (oracle goal + own MLP head — known-cheating baseline, kept for continuity), v2 (anchored goal channel).

Seeds: 1, 2, 3, 4, 5 (exp01 = seed 0, counted as a sixth observation where applicable). Minimum acceptable if compute runs short: seeds 1–3.

Training budget per seed, matched to exp01: v1 trained to ~1750 steps (exp01: 1747) regardless of loss; v2 and monitor trained until loss < 0.03 sustained or the same ~1750-step cap, whichever first (exp01: v2 645 steps, mon 773). Eval sweep at L ∈ {0,16,32,64,96,128,192,224,256}, n=192 per length, fresh eval data per seed (eval RNG also seed-offset).

## Pre-registered predictions

1. v2 ≥ 0.90 accuracy at every length, every seed (exp01: 1.00 flat).
2. v1 and v1+reminder ≤ 0.35 at every length, every seed, and v1 final loss > 1.0 (non-convergence replicates).
3. Monitor ≈ 1.00 (it's an oracle; anything else means a bug).
4. v2 convergence in < half of v1's step budget in every seed.

Falsifiers, stated in advance: any seed where v1 converges (loss < 0.5) and holds accuracy comparable to v2 kills the learnability claim as stated and demotes exp01 to an optimization artifact. Any seed where v2 fails to converge or drops materially below 1.00 at any length weakens the anchor claim and must be reported as such, not rerun until it passes.

## Outputs

results/np_sweep_s{N}.csv per seed, results/summary.csv (mean ± min–max per arm per length across seeds), results/np_sweep_replication.png, RESULTS.md written after — never edited after the fact.
