# EXP07 — Results (Witness Record: Post-Hoc Verifiability of the Binding)

Written 2026-07-07 (session 9), after all three seeds + the s1 contingency
arms + hop2 battery, per the pre-registered spec (EXPERIMENT_SPEC.md,
2026-07-07, incl. Amendments 1–2, both pre-results). Results are not edited
after this point; any amendment goes in Corrections.

## Headline

**KILL RULE 1 FIRES. Primitive C as an emergent, from-scratch primitive is
dead at this scale.** vW fails base competence in all three seeds after the
extension arm (3500 steps) AND the curriculum arm (vWc, s1, 3500 steps):
best clean cell at L≥16 is 0.521 (s1), 0.490 (s2), 0.474 (s3) against the
0.75 bar. The pre-registered tax finding stands: the binding does not
compress to a K-way commitment channel from scratch — a 4-dim gradient
bottleneck cannot train this backbone on this task (stable ~2-slot local
optimum, loss plateau ≈1.07 in every seed, above the goal-ignorant plateau
but flat for thousands of steps).

**The retrofit path (Amendment 2) fails on audit, in a specific and
replicated way.** vWr passes base competence on frozen competent features
in all three seeds (best cells 0.83 / 0.82 / 0.94) but fails the PER-CELL
AUDIT PASS in **72/72 injected cells** (V 0.41–0.68 everywhere, vs the 0.90
bar; best-π witness-goal match never exceeds 0.51 in any clean cell). The
gradient repurposes the K-way record as an answer-hint channel, not goal
semantics — the CBM-leakage precedent (RELATED_WORK.md) reproduced in a
6-line architecture. Label-free + causal + trainable was not enough to make
the record MEAN anything.

**Surviving claim (constructive corollary, seed-1 shape confirmed 3/3):
the latch is the only faithful witness, and it is free.** vB's symbolic
latch readout scores V=1.000 / OC=0.000 / W_g=1.000 in all 72 injected
cells and every clean cell of all three seeds — a perfect, label-free,
per-example-verifiable audit record, at zero training cost beyond exp06's
B, by construction (the latch IS the executed binding). The probe
(incumbent) is confirmed seed-variant on the new statistics: audit-pass
0/24 (s1), 1/24 (s2), 24/24 (s3). exp08's C should be a latch-derived
record, not an emergent or retrofit head.

## Training (matched budgets; early stop <0.03 ×3 checks; Gumbel-ST τ=1.0, λ_lb=0.1)

| seed | vW | vWo | vWr (frozen v1) | contingencies |
|---|---|---|---|---|
| 1 | 3500*, loss ≈1.10 | 1750, .31 | 1750, .40 | vWc 3500*, mean20 ≈1.12; vWr_v1_cap1750 1750, .45 |
| 2 | 3500*, ≈1.08 | 1750, ≈.58 | 1750, .42 | — |
| 3 | 3500*, ≈1.07 | 1750, ≈.70 | 1750, .15 | vW_degraded snapshotted (step 2030, loss .60) |

\* extension gate (loss <1.2 at cap-1750) fired for vW in every seed and for
vWc-s1; cap-1750 checkpoints snapshotted before extension (vW_cap1750,
vWc_cap1750). No vW seed escaped the plateau by 3500. Seed 3's vW dipped to
loss 0.60 at step 2030 — triggering the vW_degraded snapshot that was
infeasible-as-specified in s1/s2 (disclosed) — then fell back to the
plateau: the dip was an excursion, not convergence.

vWc (s1): curriculum did not rescue the bottleneck (clean 0.396–0.510 at
L≥16, indistinguishable from vW). Contrast exp02b, where the same recipe
unstuck v1-on-shift: the curriculum fixes a data-schedule problem, not a
gradient-channel problem. Per the handoff plan, vWc ran on s1 only (the
seed that triggered the contingency); kill rule 1's adjudication rests on
s1's complete arm set, with s2/s3 vW failures as replication.

## Base competence (clean 15% cells, bar = ≥0.75 at some L≥16)

- vW: 0.40–0.52 (s1), 0.41–0.49 (s2), 0.35–0.47 (s3). **FAIL ×3.**
- vWc (s1): 0.40–0.51. **FAIL.**
- vWo: PASS s1 (0.71–0.79) and s2 (0.68–0.82); **FAIL s3** (max 0.693 at
  L≥16 with 15% distractors; passes the 0% diag sweep at 0.76–0.87 —
  the failure is distractor-driven). Kill rule 2's premise (vWo passes
  everywhere every seed) is dead on competence alone.
- vWr: PASS ×3 (0.79–0.84 s1, 0.75–0.82 s2, 0.88–0.94 s3, on own answers).
- v1: 0.78–0.84 / 0.75–0.84 / 0.95–1.00 (exp06 checkpoints, replicated).
- vB: 1.000 every cell, every seed, both distractor rates.

## π dictionaries (fit once per arm on clean L=16, frozen; disclosure)

π ≠ identity in 11 of 15 fitted arms (and the four identities are floor
artifacts: match 0.25–0.31 ≈ the 0.25 chance floor). Best-π match never
exceeds 0.46 (vWr s2) anywhere in the experiment, hard task or hop2.
H-π-identity is WRONG in the only sense that matters: weight tying pinned
nothing; worse, there is no goal semantics for a dictionary to recover —
every witness head in every arm is either at floor (vW, vWc) or carrying
answer-hint structure (vWr, vWo). The π disclosure requirement is moot in
the worst way: the audit does not need a learned dictionary, it needs a
record that means something, and no trained arm produced one.

## Injection grid (24 cells/arm/seed; bars V≥0.90 ∧ OC≤0.10 per cell)

Audit-pass cells (V≥0.90 ∧ OC≤0.10):

| readout | s1 | s2 | s3 |
|---|---|---|---|
| vW witness | 0/24 (V .31–.44) | 0/24 (V .39–.52) | 0/24 (V .33–.44) |
| vWc witness (s1) | 0/24 (V .35–.57) | — | — |
| vWo witness | 0/24 (V .54–.66) | 0/24 (V .35–.59) | 0/24 (V .21–.38) |
| vWr witness | 0/24 (V .55–.68) | 0/24 (V .51–.68) | 0/24 (V .41–.57) |
| probe_v1 (incumbent) | 0/24 (V .78–.88) | 1/24 (V .80–.90) | **24/24 (V .95–1.00)** |
| **vB latch** | **24/24 (V=1.000, OC=0.000)** | **24/24** | **24/24** |

- vWr behavior columns: R 0.76–0.83 / 0.67–0.78 / 0.84–0.94 — first-binder
  tendencies on competent features (echoes exp06's v1), while its witness
  fails audit in the same cells: executing the right binding and RECORDING
  it are dissociated. That dissociation is the experiment's sharpest
  positive datum for the causal-bottleneck THESIS even as both C arms fail
  its TEST: without an architecturally pinned record, the trained channel
  optimizes answers, not testimony (CBM leakage, on schedule).
- vWo laziness (H-lazy-open): clean W_g^π aggregate 0.38 (s1), 0.31 (s2),
  0.26 (s3) — under the 0.40 signature line in every seed. The bypass
  drains the slot, as predicted; but since vW never became competent, this
  is attribution for the bottleneck's necessity-for-the-record, not
  sufficiency-for-competence.
- mon (oracle): R 0.92–1.00 across seeds, reported, non-claim-bearing.
- Kill rule 3 (probe passes everything, all seeds): does NOT fire (0/24,
  1/24 s1/s2). The probe's audit-validity failure is real — but the
  advantage accrues to the latch, not to C.

## Degradation battery (audit honesty; pre-registered bars: OC(witness)≤0.15
## every cell; OC(probe on v1_cap1750)≥0.35 in ≥half)

| arm (backbone) | cells | V | OC | verdict |
|---|---|---|---|---|
| probe_v1_cap1750 (s1 degraded v1) | 7 | .46–.63 | **.135–.260** | probe half FAILS (no cell ≥.35) |
| vWr_v1_cap1750 (s1, same features) | 7 | .55–.64 | .109–**.198** | witness half FAILS (3/7 cells >.15) |
| vW_degraded (s3, loss-matched .60) | 7 | .19–.30 | **.172–.224** | witness half FAILS (7/7 cells >.15) |

**The degradation-honesty claim is dead as pre-registered — both halves
fail.** The probe on a confused backbone overclaims mildly (OC .14–.26),
nowhere near the .35+ an assignment-tracker should show at these confusion
levels (the s1 probe is too weak even to track assignments confidently —
its clean W_g was only .62–.78). The degraded witnesses do refuse to
certify (V .19–.64, far below .90 — the failure is loud in the V statistic,
visible to any per-example verifier), but their OC exceeds .15 in 10 of 14
cells. Descriptive contrast survives; the pre-registered quantitative claim
does not. Reported straight.

## hop2 battery (standing rule; prediction: NO RESCUE, high confidence)

vW on exp05's hop2 config (n_layer=3, T=384), seeds 1–3, cap-1750 (loss
plateau ≈1.95–1.98 > 1.2; extension gate never fires): **no rescue, 3/3.**
Accuracy 0.08–0.19 across all 27 cells (chance 0.125, goal-ignorant ceiling
0.25); witness W_g at the 0.25 floor in every seed (best-π match .25–.32);
V 0.10–0.23 ≈ the random-witness floor. No cell near 0.75 → the vWo-on-hop2
contingency does not fire. A witness records a binding; it cannot create
retrieval. H-hop2 CONFIRMED — the only fully correct on-record prediction.
Witness V/OC on hop2 computed per example against the two-hop candidates
(the answer is not a function of (g,q); see np_run_hop2W.py header).

## Hypothesis scorecard (6 on record here; 5 of 6 wrong/partial — spec's
## running tally of 5 wrong/partial extends to 10)

- H-tax: PARTIAL-WRONG. Direction right (bottleneck impedes), magnitude
  wrong: not "slower convergence" (600–1200 predicted) but a stable
  non-convergent local optimum in 3/3 seeds; extension fired 3/3, not ≤1.
- H-first-witness: NOT EVALUABLE as stated (vW never competent); its
  retrofit analog is WRONG (vWr first-binds behaviorally but Rw^π ≤ .52).
- H-π-identity: WRONG (see π section).
- H-lazy-open: RIGHT on the audit half (W_g^π < .40 every seed); the
  "passes competence" premise failed in s3.
- H-silent-probe: WRONG on both quantitative halves (see degradation).
- H-hop2 (no rescue): RIGHT, 3/3.

## Kill-rule adjudication (single-seed severity, as specified)

1. **FIRES** (vW fails competence after extension + vWc, s1; replicated
   s2/s3). From-scratch C is dead pending redesign; the tax is the finding.
2. Does not fire (premises fail: vWo fails audit everywhere and competence
   in s3) — but its pre-committed downgrade wording is moot since vW has
   no numbers to be matched against.
3. Does not fire (probe fails s1/s2 on V/OC). C's audit-validity advantage
   over the probe is real but inherited by the latch, not earned by vW/vWr.
4. Breach accounting: every vW/vWc/vWo/vWr cell breaches the audit bars —
   reported in full above; nothing survives to scope.

Amendment 2's pre-committed re-scope ("witness as retrofit primitive")
CANNOT be claimed: vWr passes competence but fails the vWr success
criterion (audit pass) in 72/72 injected cells. The witness is not a
retrofit primitive at this scale either. What IS on the table, for exp08,
is the latch-derived record: B already emits the faithful witness as a
side effect of binding first.

## Operational disclosures (session 9)

- Eval-plumbing patch to np_run.py, statistics untouched: `--only` arm
  filter for eval_inject (used to append vWc rows to the s1 grid without
  duplicating existing arms; vWc traces stored under prefixed keys
  lw_vWc_*), `--arms` filter + vWr_v1_cap1750 arm block for eval_degrade.
- s1 clean-sweep CSV: snapshot-renamed to np_sweep_hard_s1_session8_preVWc.csv
  (spec: snapshot-and-rename before re-evals that change rows); full sweep
  re-run with vWc columns. Replication check: all 27 shared columns × 9
  rows byte-identical to session 8 (same eval rng) — zero diffs.
- hop2 π-s3 wrinkle: fit_pi ran at step 1745, cap reached at 1750; refit
  on the final checkpoint produced the identical permutation (1,3,0,2), so
  the three already-written rows stand.
- np_run_hop2W.py smoke-tested on seed 9 (33 steps + eval mechanics),
  purged before real runs. data_hop2.py = exp05 file verbatim; candidates
  reconstructed at eval time in the runner.
- All checkpoints, CSVs, traces, π files byte-verified (sha256) to OneDrive.

## Files

results/: np_sweep_hard_s{1,2,3}.csv (+_diag, + s1 session-8 snapshot),
np_inject_s{1,2,3}.csv, np_degrade_s{1,3}.csv, pi_s{1,2,3}.json,
witness_traces_s{1,2,3}.npz, np_sweep_hop2W_s{1,2,3}.csv,
pi_hop2W_s{1,2,3}.json. checkpoints/: np_ckpt_hard_s{1,2,3}/ (vW, vW_cap1750,
vWo, vWr, s1: vWc, vWc_cap1750, vWr_v1_cap1750; s3: vW_degraded; + exp06
copies), np_ckpt_hop2W_s{1,2,3}/. code/: np_model.py, np_run.py (patched,
disclosed), np_run_hop2W.py, data.py, data_hop2.py, baselines_C.py.

## Corrections

* 2026-07-07 (same session, post-write spot-check pass — all 31 quantitative
  claims re-verified programmatically against the raw CSVs; one range error
  found): vWo-s3 0% diag sweep range at L≥16 is 0.714–0.854 (bar met at
  L∈{16,32,64,96,128,256}), not "0.76–0.87" as written in Base competence.
  Verdict unchanged: passes diag, fails 15% — distractor-driven.
