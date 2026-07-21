# EXP02b — Curriculum-v1 Control on SHIFT (pre-registered spec)

Written BEFORE any training run. Date: 2026-07-06, session 4. CPU build
(numpy/autograd), same sandbox class as exp01–03. Roadmap step 1 (agreed end of
session 3): the strongest surface-side attack on the exp01/02 learnability
claim, doubly urgent after the exp03 inversion.

## Question

exp01/02 found v1 never learns goal-binding on `shift` (plateau at the
goal-ignorant optimum, 6/6 seeds), and credited the anchor with making the task
learnable. But all v1 training ran with 15% goal-like distractors from step 0.
If a distractor curriculum — 0% ramped to 15% during training — lets v1 learn
shift, then the learnability gap was an artifact of training-data pressure, not
architecture, and the learnability claim on shift dies. The exp03 mechanism
note makes this a real discrimination between two hypotheses:

* H-basin: the marginal basin (goal-ignorant optimum) traps v1 regardless of
  distractors — the basin exists even in clean data. Curriculum does NOT
  unstick v1; the learnability claim survives its strongest surface attack.
* H-distractor: goal-like distractors block goal-circuit formation early in
  training; a clean early phase lets the circuit form, after which ramped
  distractors don't destroy it. Curriculum unsticks v1; learnability dies as a
  claim and exactness + length-invariance (exp03 rescope) carries the program.

## Analytic baselines (computed BEFORE thresholds — exp03 Amendment-1 lesson)

Config K=4, V_ans=8: chance = 0.125. Goal-ignorant ceiling on shift = exactly
0.25 (for each q the 4 goals give 4 distinct equiprobable answers); goal-
ignorant loss floor = ln 4 ≈ 1.3863 (v1's observed exp01/02 plateau, 1.36–1.46).
Third signature, computed 2026-07-06 pre-run: a model that binds to the MOST
RECENT goal-like token (rather than the position-1 assignment) scores
P = 0.85^L + (1 − 0.85^L)/4 under 15% eval distractors: 1.000 at L=0, 0.306 at
L=16, 0.254 at L=32, → 0.25. High-L=0-only accuracy is therefore last-token
reading, not goal-binding (retrospectively consistent with exp02 seed-4 L=0
41.7% breach).

## Design

Single new arm: v1c — architecture, config, optimizer, data stream identical to
exp01/02 v1 (K=4, V_ans=8, V_filler=32, 2 layers, H=2, D=48, T=288,
train_max_len=64, batch scheme unchanged, task=shift) EXCEPT the training
distractor rate follows a curriculum:

    d(step) = 0.15 * min(1, step / 1000)      (linear 0% → 15%, hold after 1000)

Rationale for the 1000-step knee: v2 converges in ~510 steps and v1's plateau
is visible by ~500 in every prior seed, so 0–1000 gives the goal circuit a
low-pressure formation window ≈ 2× v2's full convergence budget, then ≥ 750
steps at full 15% pressure before the cap.

Seeds 1, 2, 3 — same seed-offset RNG scheme as exp02/03. No retraining of v2,
reminder, or monitor arms: comparison values come from exp01/02 published
results (v2 = 100% all cells on shift; v1 plateau 22–27%).

Training budget: 1750 steps (matches exp01/02 v1 exactly). Pre-registered
contingency, adversarial-generous to v1c: if at the 1750 cap final loss < 1.2
(escaped the basin but not converged), extend that seed to 3500 steps max,
disclosed in RESULTS. If loss ≥ 1.2 at cap (still in basin), stop — matched
budget already answers the question.

Eval: primary sweep identical to exp01/02 — L ∈ {0,16,32,64,96,128,192,224,256},
n=192 per length, 15% eval distractors (comparability). Secondary diagnostic
sweep at 0% distractors, same lengths and n (did a goal circuit form at all,
even one that distractors then break?). Loss curve logged with curriculum
milestones (d = 0%, 5%, 10%, 15% crossings).

## Pre-registered predictions

1. On-record program prediction: H-basin. The marginal basin exists in clean
   data too, so the clean phase does not remove the trap. v1c final loss lands
   within [1.2, 1.6] (basin) and accuracy ≤ 0.35 at every L ≥ 16, every seed.
2. L=0 cells may exceed 0.35 via last-token/position reading (signature above);
   that is reported as such, not as goal-binding.
3. No seed reaches the goal-binding criterion (below).

## Interpretation commitments (stated in advance)

* Goal-binding is credited to v1c only if accuracy ≥ 0.75 at some L ≥ 16
  (materially above the 0.25 ceiling and outside the last-token signature
  regime; same 0.75 operational bar as exp03 Amendment 1).
* KILL RULE: if ANY seed shows v1c ≥ 0.75 across all trained lengths
  (L ≤ 64), the learnability claim on shift is dead as stated, reported
  straight, and exactness + length-invariance carries the program (exp03
  rescope becomes the program-wide anchor claim). Same single-seed severity
  as exp03's kill rule.
* Partial unsticking (≥ 0.75 at some L ≥ 16 but not across the trained range,
  or in some seeds only) = learnability claim DOWNGRADED to "v1 goal-binding
  on shift is curriculum-fragile"; reported with per-cell detail.
* If v1c plateaus at the basin (pred 1 holds): the learnability claim survives
  its strongest surface-side attack and gains the H-basin mechanism as
  positive evidence. It remains scoped to shift-class (flat-gradient) tasks
  per exp03.
* Diagnostic-sweep divergence (0% ≫ 15% accuracy at the same L) is reported as
  circuit-formed-but-distractor-broken — evidence for a THIRD reading
  (curriculum forms a fragile circuit) and pre-registered here so it can't be
  spun post hoc.
* Every threshold breach reported, single cells included. No rerun-until-pass.

## Outputs

code/ (diff vs exp03 = curriculum rate parameter + v1c tag + diagnostic eval;
no other changes), checkpoints/seedN/, results/np_sweep_curr_s{N}.csv +
np_sweep_curr_diag_s{N}.csv per seed, results/summary.csv, RESULTS.md written
after — never edited after the fact.
