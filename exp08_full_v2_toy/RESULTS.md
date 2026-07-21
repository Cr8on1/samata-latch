# EXP08 — Results (The Full Toy Stack: A + B(latch) + C(latch readout))

Written 2026-07-07 (session 10), after all three seeds (hard task + hop2
battery + degradation battery), per the pre-registered spec
(EXPERIMENT_SPEC.md, 2026-07-07, no amendments). Results are not edited
after this point; any amendment goes in Corrections.

## Headline

**STACK SUCCESS: the pre-registered joint criterion holds in all three
seeds with zero breaches anywhere.** vS holds all four surviving program
properties SIMULTANEOUSLY, per cell: (a) clean exactness 1.000 in every
cell (9 lengths × 2 distractor rates × 3 seeds, n=192, zero errors);
(b) override exactness R=1.000/C=0.000 in all 72 injected cells;
(c) convergence 440/400/400 steps — 8.0×/4.4×/8.8× vs the same-seed v1
bar-clearing budgets (3500/1750/3500), both efficiency bars met;
(d) faithful witness V=1.000/OC=0.000 (W_g≡1 analytic) in every clean and
injected cell. No kill rule fires. No contingency fires (no extension, no
curriculum arm, hard task).

**The minimality result is the news: per-layer re-injection is dead weight
at this scale.** vSo (register injected ONCE at the embedding, never
renewed) matches vS on every bar in every cell of every seed — clean 1.000
everywhere including L=256 at 15% distractors, 72/72 injected cells exact,
convergence 420/360/410. H-anchor-earns is WRONG, and the pre-committed
wording fires: A's surviving contribution inside the stack is the injection
SITE (pre-attention residual stream), not the repetition. The anchor joins
the gate (exp06) in the dead-weight column. The minimal stack going into
the scale campaign is: WRITE-ONCE LATCH → REGISTER ADDED ONCE AT INPUT →
SYMBOLIC READOUT. One K×D table, one indexing rule, zero parameters of
audit machinery.

## Training (matched budgets; early stop <0.03 ×3 checks; cap 1750 never binding)

| seed | vS | vSo | vS_degraded snapshot | v1 (on record, exp06) | ratio vS |
|---|---|---|---|---|---|
| 1 | **440**, loss .011 | 420, .027 | step 100, loss .588 | 3500* | 7.95× |
| 2 | **400**, .029 | 360, .026 | step 120, .683 | 1750 | 4.38× |
| 3 | **400**, .010 | 410, .024 | step 80, .548 | 3500* | 8.75× |

Efficiency bars: convergence ≤500 ✓ every seed (both arms); ratio ≥3.5× ✓
every seed. The 320–450 exp06 vB/vBl band replicates (vS 400–440).
Degraded-band snapshot ([0.45,0.70] first hit) fired in all three seeds —
no infeasibility disclosure needed.

## Clean sweeps (bar: accuracy = 1.000 EVERY cell; n=192)

vS and vSo: **1.000 in every cell** — L ∈ {0,16,32,64,96,128,192,224,256},
both distractor rates, all seeds (108 cells total across arms/rates/seeds).
Audit columns on the same cells: W_g=1.000, V=1.000, OC=0.000 everywhere.
Zero breaches. The A-arc exactness signature holds inside the stack, and
holds WITHOUT re-injection (vSo) out to 4× the training length.

## Injection grid (24 cells/arm/seed; L ∈ {32,64,128,192} × φ ∈ {0.25,0.75,end} × m ∈ {1,4})

| arm | s1 | s2 | s3 | breaches |
|---|---|---|---|---|
| vS | 24/24 exact | 24/24 | 24/24 | 0 |
| vSo | 24/24 exact | 24/24 | 24/24 | 0 |

Exact = R=1.000, C=0.000, V=1.000, OC=0.000, Rw=1.000, Cw=0.000 per cell.
Comparison rows (cited from exp06/exp07 CSVs, no re-eval): vB 72/72 exact
(behavior, exp06; latch audit exact, exp07) — vS−vB gap 0.000, H-gap-zero
RIGHT, gate removal changes nothing; v1 61/72 resistance with R misses to
0.698 (exp06); probe_v1 audit-pass 0/24, 1/24, 24/24 (exp07, seed-variant).
Per-goal breakdowns written (np_inject_pergoal_s{1,2,3}.csv), no per-goal
asymmetry (all cells exact).

## Degradation battery (pre-registered boundary of the faithful-witness claim)

Identity check: OC = 1 − V holds to machine precision in all 21 cells
(the latch always asserts ŵ=g; V = R analytically on these cells — verified
empirically, R column equals V column exactly). Snapshots landed at loss
.55–.68 (accuracy .64–.89, milder degradation than exp07's v1_cap1750):

* OC range .109–.365; **OC > 0.15 in 20/21 cells** — the latch witness
  FAILS the exp07 honesty bar under degradation, as pre-registered
  (H-degraded-overclaim RIGHT in direction and mechanism). One s1 cell
  (clean L=64, V=.891) sits at OC=.109, below the bar — magnitude tracks
  1−a exactly, so mild degradation produces mild overclaim; reported
  straight, no rescue wording claimed.
* V is loud in every cell (.64–.89 vs the 1.000 competent signature) and
  per-example checkable without labels — the scoped surviving half.
* SCOPE ON RECORD: the faithful-witness property is a property of the
  COMPETENT stack. The latch is an assignment record; its faithfulness
  comes from the register causally driving behavior, not from any
  execution-sensitivity of the record itself.

## hop2 battery (standing rule; prediction on record: NO RESCUE, HIGH)

vS on exp05's hop2 config (n_layer=3, T=384), seeds 1–3, cap-1750 (loss
1.89–2.05 > 1.2; extension gate never fires): **no rescue, 3/3.** Accuracy
0.083–0.188 across all 27 cells (chance 0.125, goal-ignorant ceiling 0.25);
no cell ≥0.75 → the vSo-on-hop2 attribution contingency does not fire.
H-hop2 RIGHT — fourth consecutive architecture (v2, vB, vW, vS) to confirm
that anchoring/friction/witnessing are orthogonal to retrieval. Witness
columns: W_g=1.000 in every cell while V sits at 0.083–0.188 and OC at
.81–.92 — the record says "bound g" while retrieval fails, loud in V,
exactly the pre-registered pattern. First latch-only hop2 run; scoping
unchanged: hop2 remains the scale campaign's open question.

## Hypothesis scorecard (6 on record; 4 right, 2 wrong — program tally of
## wrong/partial predictions grows 10 → 12)

- H-stack-holds: **RIGHT**, 3/3, zero breaches (marked HIGH/low-risk in
  the spec; the consolidation is the value, not the surprise).
- H-gap-zero: **RIGHT** (gap 0.000, cited-rows comparison).
- H-anchor-earns: **WRONG** — vSo breaches nothing at any L in any seed.
  The honest alternative pre-worded in the spec is the finding: per-layer
  anchoring is dead weight at this scale. (LOW-MODERATE confidence,
  disclosed as genuinely uncertain; the wrong direction is the more
  consequential result for the scale campaign.)
- H-degraded-overclaim: **RIGHT** (OC=1−V exact; 20/21 cells over the
  honesty bar; magnitude nuance reported above).
- H-hop2-no-rescue: **RIGHT**, 3/3.
- H-byte-replication: **WRONG, by mechanism** — resume RNG streams are
  seeded OFF+1000+step+ao, so the data order after the first 34s burst
  depends on where that burst stopped; exp06 vBl-s1 finished at 350 steps
  vs vS-s1's 440 (0/31 arrays equal, sha256 differ). Convergence bands and
  all evaluated properties replicate; bit-level replication of burst-timed
  training does not. Disclosed as an infrastructure fact worth knowing
  before the scale campaign.

## Kill-rule adjudication (single-seed severity, as specified)

1. Competence/tax: does not fire (vS clean 1.000 everywhere, converged
   400–440; no extension, no curriculum arm).
2. Capitulation: does not fire (C=0.000 in all 72 vS cells; max C anywhere
   in the experiment = 0.000).
3. Joint-holding failure: does not fire (all four bars met, every seed).
4. Efficiency: does not fire (both clauses met, every seed).
Breach accounting: zero cells breached any bar in any arm on the hard task.

MINIMALITY ADJUDICATION (pre-committed wording, first branch): vSo matches
vS on (a), (b), (d) in every cell of every seed and converges ≤500 →
per-layer re-injection is DEAD WEIGHT at this scale; minimal stack =
latch-once + readout; A's surviving contribution inside the stack is the
injection site, not the repetition. Same epistemic status as exp06's gate
verdict: distinguished by guarantees claimed elsewhere, not by numbers here.

## Claim (program style, exactly as pre-registered)

One small architecture — a write-once latch keyed by the first assignment
marker, its register added once at the input stream, read back symbolically
at apos+1 — buys, simultaneously and in every seed: seed-invariant clean
exactness at every tested length (dilution), seed-invariant exactness under
format-identical override pressure (injection), 4.4–8.8× less training than
the standard baseline needs to clear the competence bar, and a free,
label-free, per-example-verifiable record of the executed binding. Scoped:
at toy scale, on a (g,q)-function task; the witness's faithfulness is a
property of the competent stack (degradation battery); none of it touches
retrieval (hop2). This is the PoC input to the consolidated scale campaign.

## Operational disclosures (session 10)

- Eval batching relaxed vs playbook (lens ≥128 solo was calibrated on
  8-arm exp06 evals; exp08 evals carry 2 arms and ran 5–36s/call). No eval
  timed out; all CSV row counts verified programmatically (9/9 per sweep,
  24/arm per injection grid, 192 per-goal rows, 7 per degradation battery,
  9 per hop2 sweep).
- One vSo training burst (seed 1, burst 2) failed to launch (shell cwd
  slip, command error output only); no state was touched, the arm resumed
  cleanly next call. No RNG consequence beyond the resume-boundary
  mechanism already disclosed under H-byte-replication.
- Degradation eval uses a fresh eval stream (OFF+4242, in code); hop2S
  arm offsets {vS:8} disclosed in np_run_hop2S.py. vS shares exp06 vBl's
  ARM_OFF=3 by design (spec's determinism check).
- Smoke seed 9: both runners, mechanics only, purged including CSVs before
  any real run.
- All checkpoints and CSVs byte-verified (sha256) to OneDrive at two sync
  points (post-hard-task, post-hop2); all OK.

## Files

results/: np_sweep_hard_s{1,2,3}.csv (+_diag), np_inject_s{1,2,3}.csv,
np_inject_pergoal_s{1,2,3}.csv, np_degrade_s{1,2,3}.csv,
np_sweep_hop2S_s{1,2,3}.csv. checkpoints/: np_ckpt_hard_s{1,2,3}/ (vS,
vSo, vS_degraded), np_ckpt_hop2S_s{1,2,3}/vS.npz. code/: np_model.py,
np_run.py, np_run_hop2S.py (+ data.py, data_hop2.py copies — exp06/exp05
files verbatim; requirements.txt). Comparison rows cited in place from
exp06_friction_signal/results/ and exp07_witness_audit/results/ (no copies,
no re-evals).

## Corrections

(none)
