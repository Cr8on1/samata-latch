# EXP03 — Hard Task (pre-registered spec)

Written BEFORE any exp03 training run. Date: 2026-07-05, session 3. CPU build (numpy/autograd), same sandbox class as exp01/exp02.

## Question

Does the exp01/exp02 pattern hold when the goal gates intermediate computation rather than only the final lookup? Task `hard`: a = ((q + g) * (g + 1)) mod V_ans, replacing `shift`: a = (q + g) mod V_ans. Specifically: does the anchored goal channel (v2) still learn and hold the binding, and does the external-monitor baseline — which so far matches v2 by recomputing the answer itself from oracle goal + frozen v1 features — finally separate from v2?

## Design

Identical to exp02 in every respect except the task rule. Same code (copied from exp02/code; one change: a `--task` CLI flag that sets CFG["task"], plus task-tagged checkpoint dirs and results filenames so shift and hard artifacts can never collide). Same config: K=4, V_ans=8 (chance 12.5%), V_filler=32, 2 layers, H=2, D=48, T=288, train_max_len=64, distractors=15%.

Arms (all four): v1 (standard), v1+prompt-reminder (every 16 tokens, eval-time prompt change only), v1+external monitor (oracle goal + MLP head on frozen v1 query features — known-cheating baseline), v2 (anchored goal channel).

Seeds: 1, 2, 3 (minimum per program plan; extend to 4–5 only if compute allows, decided before looking at any eval numbers). Same seed-offset RNG scheme as exp02.

Training budget per seed, matched to exp02: v1 trained to ~1750 steps regardless of loss; v2 and monitor trained until loss < 0.03 sustained or the ~1750-step cap, whichever first. Eval sweep at L ∈ {0,16,32,64,96,128,192,224,256}, n=192 per length, fresh eval data per seed.

## Pre-registered predictions

1. v2 ≥ 0.90 accuracy at every length, every seed. Convergence may be slower than shift (the head must implement a 2-step function of g), but within the 1750-step cap.
2. v1 ≤ 0.35 at every length, every seed, final loss > 1.0; reminder arm no better. (hard is a superset of the difficulty that already defeated v1.)
3. Monitor — program's on-record prediction (Creighton): this is where the output-only wrapper hits its wall — monitor < 0.90 at one or more lengths, because the goal now gates intermediate computation it cannot reach.
4. v2 convergence in < half of v1's step budget in every seed.

## Interpretation commitments (stated in advance)

Flagged dissent, on record before running: the monitor receives the oracle goal AND v1 query-position features that plausibly still encode q through the residual stream; since `hard` remains a pure function of (g, q) — a 32-entry table at K=4, V_ans=8 — a 2-hidden-layer MLP may simply memorize it. Therefore:

* Monitor < 0.90 somewhere → prediction 3 confirmed; the wall claim gains its first positive evidence.
* Monitor ≈ 1.00 flat → prediction 3 FAILS and is reported as such. Reading: `hard` does not yet operationalize the wall, because gating intermediate computation constrains the sequence model, not an oracle-goal output wrapper. The wall claim then requires a task where the answer is not a function of (goal, query token) alone — e.g., goal-gated multi-hop retrieval over the context — to be queued as exp05 candidate. This outcome would NOT count against v2.
* Any seed where v1 converges (loss < 0.5) and matches v2 kills the learnability claim as stated.
* Any v2 cell < 0.90 is reported, not rerun until it passes.

## Amendment 1 — 2026-07-05, before any training run

While smoke-testing data generation (no model trained yet), we computed the analytic goal-ignorant ceiling for `hard`: a predictor that ignores g and outputs argmax_a P(a|q) achieves 15/32 ≈ 0.469, because the four goal rules collide on answers for 7 of 8 operands (on `shift` this ceiling is exactly 0.25, which is why exp01/exp02's v1 plateaued at 22–27%). The original prediction-2 threshold (v1 ≤ 0.35) is therefore wrong for this task: v1 could breach it by learning a pure function of q, which is not goal-binding.

Amended prediction 2: v1 and v1+reminder ≤ 0.55 at every length, every seed (goal-ignorant ceiling 0.469 + sampling margin), and materially below v2. Goal-binding is credited to an arm only if it materially exceeds 0.469 — operationally ≥ 0.75. v1 landing near 0.469 (rather than near 0.125) is reported as "learned the q-marginal, not the goal" — a distinct and interesting failure mode vs shift. v1 final-loss prediction adjusted: loss may fall below 1.0 by fitting the marginal (entropy of the marginal-optimal predictor is lower than uniform); the non-convergence criterion for v1 is therefore accuracy-based, not loss-based, in this experiment.

## Outputs

code/ (diff vs exp02 = --task flag + artifact naming only), checkpoints/seedN/, results/np_sweep_hard_s{N}.csv per seed, results/summary.csv, RESULTS.md written after — never edited after the fact.
