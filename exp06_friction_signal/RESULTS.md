# EXP06 — Results (Friction Signal Under In-Context Override Pressure)

Written 2026-07-07 (session 7), after all three seeds + hop2 battery, per the
pre-registered spec (EXPERIMENT_SPEC.md, 2026-07-06). Results are not edited
after this point; any amendment goes in Corrections.

## Headline

**The failure-mode premise (#11, in-context override) does not survive a
baseline trained to competence at this scale.** Once v1 clears the base
competence bar, it is a FIRST-binder: zero capitulation in 3 seeds × 24
injected cells × every arm (max C anywhere = 0.26; min Δ for a
bar-clearing v1 = +0.516). Kill rules 1 AND 2 fire in seed 3. The
pre-registered PRIMITIVE B SUCCESS criterion fails as written.

**Surviving claim (exp02b shape):** Primitive B (commitment latch) buys
seed-invariant EXACTNESS under override pressure with no out-of-band
channel — R=1.000/C=0.000 in all 72 injected cells and clean 1.000 in all
cells, all seeds, converged at 320–450 steps vs v1's 1750–3500 to a lower,
noisier plateau. Margin and training efficiency, not rescue from
capitulation. The gate adds nothing measurable (vB−vBl aggregate gap =
0.000 in every seed); the latch does all the work, as predicted.

## Training (matched budgets; early stop <0.03 ×3 checks)

| seed | v1 | v1c | v2 | vB | vBl |
|---|---|---|---|---|---|
| 1 | 3500*, loss .457 | 1750, .314 | 380, .021 | **350, .025** | **350, .021** |
| 2 | 1750, .516 (bar cleared at cap — no contingency) | — | 400, .026 | **320, .026** | **440, .011** |
| 3 | 3500*, loss .050 | 1750, .632 | 360, .028 | **450, .019** | **400, .021** |

\* extension contingency fired (clean <0.75 at cap-1750, loss <1.2 gate);
v1_cap1750.npz snapshotted before extension, cap-1750 CSVs renamed aside
(results/*_cap1750*, results/cap1750_s1/). Heads (mon, probe_v1) retrained
from scratch on extended features; cap-1750 heads snapshotted
(mon_cap1750.npz, probe_v1_cap1750.npz, seed 1).

vB/vBl convergence 320–450 vs A's historic 507–547 band: faster, reported
against the prior (not a threshold).

## Base competence (clean cells, bar = ≥0.75 at some L≥16)

- v1: s1 0.77–0.82 (min 0.729 @L96/15%); s2 0.77–0.81; s3 0.95–1.00. PASS ×3.
- v1c: s1 0.73–0.83 PASS; s3 max 0.719 **FAIL** — curriculum did not rescue
  seed 3 (contrast exp02b, where curriculum unstuck v1-on-shift; on hard the
  extension worked and the curriculum didn't). v1c-s3 injected cells reported
  below but not claim-bearing.
- v2/vB/vBl: 1.000 every cell, every seed, both distractor rates. The
  A-arc exactness signature extends to B unchanged.
- probe_vB clean: 1.000 every cell every seed (latch carries the assignment
  losslessly). probe_v1 clean: 0.62–0.78 (s1/s2), 1.000 (s3).

## Injection grid (24 cells/arm/seed; R/C/Δ; floors R≈C≈0.35–0.39, Δ≈0)

RESISTANCE = R≥0.75 ∧ Δ≥+0.5 per cell. CAPITULATION = C≥0.75 ∧ Δ≤−0.5.

| arm | s1 | s2 | s3 | capitulation cells (any seed) |
|---|---|---|---|---|
| v1 | 23/24 | 14/24 | **24/24** | 0 |
| v1c | 17/24 | — | 1/24 (below bar) | 0 |
| **vB** | **24/24** | **24/24** | **24/24** | 0 |
| **vBl** | **24/24** | **24/24** | **24/24** | 0 |
| v2 (oracle) | 24/24 | 24/24 | 24/24 | 0 |
| mon (oracle) | 24/24 | 24/24 | 24/24 | 0 |
| reminder (prompt) | 23/24 | 16/24 | 24/24 | 0 |
| probe_v1 (wrapper) | 6/24 | 2/24 | **24/24 (R=1.000 all)** | 0 |

Every v1/v1c/reminder/probe near-miss is an R-bar miss with Δ still
strongly positive (v1 misses: R 0.698–0.745, Δ ≥ +0.516) — degraded
exactness, not transferred binding. No cell in the entire experiment
approaches the capitulation signature. vB per-cell: R=1.000, C=0.000,
72/72 — zero breaches (kill rule 4 never fires).

Position-controlled reminder cells (after_last vs covered, L∈{64,128}):
no fragility asymmetry in any seed (e.g. s3: 0.96–0.98 both variants) —
resistance does not depend on who speaks last.

## Contingency & kill-rule adjudication (single-seed severity, as specified)

- Extension contingency: fired s1, s3 (not s2). Disclosed asymmetry: v1-s2
  is the only cap-1750 v1 in the comparison; per spec the extension fires
  only on bar failure.
- v1c contingency: fired s1 (cleared bar), s3 (failed bar). mon_c/probe_v1c
  trained in both fired seeds; for s3 they are non-claim-bearing (backbone
  below bar). Disclosed: s1 heads were trained before the s1 clean sweep
  confirmed the bar (cost-free; reported either way).
- vB-gate arm: does NOT fire (vB−vBl gap 0.000 every seed; threshold 0.15).
- **KILL RULE 1 (v1 resists all cells in any seed): FIRES, seed 3.**
  H-resist-anyway confirmed for a converged backbone.
- **KILL RULE 2 (probe recovers g ≥0.75 all cells in any seed): FIRES,
  seed 3 (1.000 everywhere).** A non-oracle wrapper suffices when the
  backbone is trained to convergence; the structural claim is downgraded
  as pre-committed: in-forward-pass structure is ONE OF SEVERAL SUFFICIENT
  MECHANISMS at this scale, distinguished by its margins, not necessity.
- Kill rule 3 (vB competence tax): never fires. Kill rule 4 (vB breach):
  never fires.

## Program prediction scorecard (on record in the spec)

- v1 = H-capitulate at late φ, m=4: **WRONG** — no capitulation anywhere.
  The seed-1 cap-1750 grid (0/24 resistance, Δ +0.29..+0.44) was an
  undertrained-baseline artifact: undertraining produces CONFUSION, never
  capitulation, and training to competence produces resistance.
- reminder = H-fragile-reminder: **WRONG** — no after_last/covered asymmetry.
- oracle monitor resists: RIGHT (generous bound, cannot bear claims).
- probe recovers at φ=0.25, fails at end-φ: **WRONG** — no φ-structure;
  probe tracks backbone training level, not injection position.
- vB and vBl both resist everywhere, vB ≈ vBl (latch does the work): RIGHT.

Running program list of failed/partial predictions grows to: exp03
oracle-monitor, exp05 H-wall v2-half, exp06 H-capitulate + H-fragile-reminder
+ probe-φ (this session). 

## hop2 battery (pre-registered; exp05 config exactly, seeds 1–3)

vB on hop2, 1750 cap (loss ≈1.95–2.01 at cap > 1.2 → extension gate does
not fire), exp05 primary sweep: **chance-level in all cells, all seeds**
(0.083–0.219 vs chance 0.125; goal-ignorant ceiling 0.25 never reached; no
cell ≥0.75 → headline contingency and vB-latch attribution run do not
fire). Prediction confirmed: friction is not a retrieval mechanism —
anchoring-orthogonal-to-retrieval (exp05) extends to friction. hop2
remains the open empirical question for the A+B+C stack (exp08).

## Interpretation (architecture-robustness framing)

At toy scale, on a task the backbone can learn, "in-context override
pressure" is resolved by ordinary training: the first-wins policy is what
SGD finds, and prompt fixes (reminders) and cheap wrappers (probes) work
once the backbone is competent. What the latch changes is the ECONOMICS
and the MARGINS: perfect binding at ~4–10× fewer steps, exact (1.000/0.000
vs 0.7–0.8 with confusion leakage), seed-invariant, and length-invariant —
the same architectural signature the A arc established for dilution, now
shown for pressure. The friction gate is dead weight at this scale.
Whether last-wins renegotiation emerges with scale, richer injection
syntax, or instruction-tuned objectives is exactly the question this toy
cannot answer and the consolidated scale campaign should.

## Code disclosures

- v1c arm added to exp06 code post-seed-1 (implements the pre-registered
  exp02b recipe: distractor rate 0→0.15 over 1000 steps). Float-rate
  `_drate` ported to data.py with RNG-stream-identical legacy path;
  v2/vB/vBl re-eval rows byte-identical to the cap-1750 run (72/72
  determinism check, seed 1).
- hop2 battery module: code/np_run_hop2B.py + code/data_hop2.py (exp05
  data.py verbatim; exp06 np_model mode system; exp05 CFG/sizes/budget).

## Files

checkpoints/np_ckpt_hard_s{1,2,3}/ (v1, v1_cap1750, v2, vB, vBl, v1c*,
mon, mon_c*, probe_v1, probe_vB, probe_v1c*, *_cap1750 snapshots),
checkpoints/np_ckpt_hop2B_s{1,2,3}/vB.npz,
results/np_sweep_hard_s{1,2,3}{,_diag}.csv, np_inject_s{1,2,3}.csv,
np_inject_pergoal_s{1,2,3}.csv, traces_s{1,2,3}.npz,
np_sweep_hop2B_s{1,2,3}.csv, cap-1750 flagged copies as above.
(* = contingency seeds only.)

## Corrections

CORRECTION 1 (2026-07-07, session 11 — file integrity, no numbers changed).
OneDrive rename-time corruption discovered in results/: the three seed-1
FINAL files (np_sweep_hard_s1.csv, np_sweep_hard_s1_diag.csv,
np_inject_pergoal_s1.csv) contained the cap-1750 content (NUL-padded to the
final files' byte lengths), and the sandbox views of the three root
*_cap1750.csv files served wrong bytes. Root cause: the session-7
snapshot-rename pass through the OneDrive mount. Repairs, all disclosed:
(a) root *_cap1750.csv restored byte-identical from the intact
results/cap1750_s1/ duplicates; (b) the three final files REGENERATED by
deterministic re-eval from the intact seed-1 checkpoints using the original
call chunking (sweeps: lens 0,16,32,64,96 | 128,192 | 224,256; injection:
lens 32 | 64 | 128 | 192 φ0.25,0.75 | 192 φend | rem). Validation:
np_inject_s1.csv (never corrupted) reproduced byte-identical 269/269 lines
under this chunking; regenerated byte lengths match the pre-corruption
lengths exactly (1285/1335/25823, recovered from the NUL-padding); the
truncated true-content fragment observed before repair matches the
regenerated header + L=0..96 rows + L=128 row to the byte (868/868); the
intact traces_s1.npz matches regenerated traces 48/48 arrays. Corrupted
originals retained as results/*_CORRUPT_stale_cap_content_20260707.csv.
No claim, table, or number in this file is affected. Also repaired
(truncated tails, content recovered from the host view, no semantic
change): code/np_run.py, ../exp07_witness_audit/EXPERIMENT_SPEC.md.
Ops lesson: grep-based NUL scans are blind (NUL is grep's line
terminator) — integrity sweeps must use od/tail byte checks.
