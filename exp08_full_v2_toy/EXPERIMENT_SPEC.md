# EXP08 — The Full Toy Stack: A + B(latch) + C(latch readout) (pre-registered spec)

Written BEFORE any training run. Date: 2026-07-07, session 10. CPU build
(numpy/autograd), same sandbox class as exp01–07. Roadmap: the consolidation
step before the single scale campaign (revised roadmap, session 5; exp07
handoff). All floors and baselines cited here were computed and MC-verified
pre-spec in exp06 (code/baselines.py) and exp07 (code/baselines_C.py);
nothing is recomputed and no threshold is new-with-hindsight — every bar
below defends a number already on the record.

## Question

Each surviving property of the program was established in a separate
experiment, on separately trained arms:

* A — ANCHOR EXACTNESS: clean-cell 1.000 at every length, seed-invariant
  (exp01–03; scoped by exp05: orthogonal to retrieval).
* B — OVERRIDE EXACTNESS: R=1.000/C=0.000 in all 72 injected cells, no
  out-of-band channel (exp06, latch; gate dead weight).
* B — EFFICIENCY: convergence at 320–450 steps vs v1's 1750–3500 (exp06).
* C — FREE FAITHFUL WITNESS: the latch readout is the only faithful witness,
  V=1.000/OC=0.000/W_g=1.000 in all 72 injected + all clean cells (exp07;
  both trained-C arms dead).

exp08 asks the consolidation question: does ONE architecture hold all four
properties SIMULTANEOUSLY, per-cell, in every seed, on the hard task — and
is that architecture minimal?

THE BAR, STATED PLAINLY (program discipline): the stack arm is architecturally
the exp06 vBl (latch register, no gate) plus a zero-parameter symbolic
readout. Its separate properties are already on record, so H-stack-holds is
LOW-RISK by construction. The pre-registered epistemic content of exp08 is
therefore (i) the JOINT criterion — one arm, one training run, all four bars
at once, any single-cell miss a reported breach; (ii) the MINIMALITY ablation
— does A's per-layer re-injection earn its keep inside the stack, or does
once-at-input suffice (the gate died this way in exp06; the anchor gets the
same trial); (iii) the DEGRADATION BOUNDARY of the faithful-witness claim,
stated analytically in advance rather than discovered post-hoc; (iv) hop2 for
the latch-only architecture (vB-with-gate ran in exp06; vBl never did).

## Architecture note (disclosed up front)

A, B, and C do not bolt together as three modules; in the stack they are one
mechanism viewed three ways. A contributes the injection DISCIPLINE (register
re-injected into the residual stream at every layer, outside attention
competition — exp01). B contributes the KEYING (register indexed by the
payload token at the FIRST assignment marker, write-once, read from the token
stream — no oracle; exp06). C contributes the READOUT (ŵ = the token at
apos+1, symbolically readable per example, plus the exp07 audit statistics —
NOT a trained head; exp07 killed both trained-C arms). Implementation is
exp06's vBl mode verbatim plus eval-time audit columns.

## Task & config

Base task, config, training data: IDENTICAL to exp06/exp07 in every respect
(hard: a = ((q+g)·(g+1)) mod 8; K=4, V_ans=8, V_filler=32, n_layer=2, H=2,
D=48, T=288, train_max_len=64, lr=1e-3 Adam, 15% bare-token distractors, NO
injections in training, batch-bucket scheme + per-step compute compensation,
1750-step cap, early stop <0.03 ×3 checks, extension gate loss <1.2 → 3500).
Injection generator: exp06's exactly (format-identical counter-assignment,
overwrite filler, redraw until rule(g',q) ≠ rule(g,q)). New params: B_reg
(4×48) only, +0.9% over v1 (exp06 disclosure carries over; vStack has no
Wg/gate params).

## Arms (per seed; seeds 1, 2, 3; OFF = 10000×seed scheme)

Trained from scratch (matched budgets, cap 1750):
1. vStack — v1 + latch-keyed register re-injected EVERY layer + latch
   readout. CLAIM-BEARING ARM. Architecturally ≡ exp06 vBl.
2. vStack-once — SINGLE-AXIS ablation: identical in every respect except the
   register is added ONCE, to the embedding stream before layer 0, and not
   re-injected thereafter. Tests whether A's per-layer re-injection earns its
   keep inside the stack (minimality; the exp01 mechanism claim was
   "re-injected every layer, outside the attention competition" — this is
   its first direct ablation).

Snapshot (no new training):
3. vStack_degraded — snapshot vStack at the FIRST 10-step check with running
   loss in [0.45, 0.70] (brackets exp06 v1-s1's 0.457 and exp07's 0.60
   loss-matching precedent). vStack passes through this band early (vBl loss
   went 2.08 → <0.03 by step ~350–440); if no check lands in the band in
   some seed, infeasibility is disclosed (exp07 s1/s2 precedent).

Eval-only / cited (NO retraining, NO re-eval unless a needed cell is absent
from the record — any such run disclosed in RESULTS):
4. v1 — competent baseline, exp06 checkpoints (s1/s3 extended-3500, s2
   cap-1750), clean + injection rows cited from exp06 CSVs.
5. vB (latch + gate) — exp06 checkpoints; behavior rows from exp06, latch
   audit rows from exp07 (V=1.000/OC=0.000 on record). The vStack−vB
   comparison is the gate-removal check at the stack level: expected gap
   0.000 (exp06: vB−vBl aggregate gap 0.000 every seed).
6. v2 (oracle anchor) — exp06 rows cited; positive control, cannot bear
   claims (out-of-band channel).
7. probe_v1 — incumbent auditor; exp07 rows cited (audit-pass 0/24, 1/24,
   24/24 — the seed-variance the latch is claimed to beat).

Contingencies (pre-registered, standing rules):
* Extension gate (both trained arms): loss <1.2 at cap-1750 → 3500,
  cap-1750 checkpoint snapshotted first.
* Curriculum arm vStack-c: fires iff vStack <0.75 on clean cells at all
  L≥16 in any seed (competence claims need extension AND curriculum arms).
  Same for vStack-once (vStack-once-c) if IT fails the competence bar —
  the minimality verdict must compare competent-vs-competent or report the
  ablation as a competence kill, not an exactness kill.
* Determinism check (disclosed either way): vStack shares exp06 vBl's
  config, mode, seeds, and RNG scheme; if the training stream is consumed
  identically, fresh checkpoints may reproduce exp06's vBl byte-identically.
  sha256-compare per seed and report. Byte-identity = replication by
  construction; divergence = an extra replication data point (curves
  compared). exp08's claims rest on exp08's own runs in either case.

## Audit statistics & analytic identities (stated before any run)

Latch readout, all arms carrying it: ŵ = token at apos+1 (symbolic, zero
parameters). Statistics per exp07: W_g = P(ŵ=g); V = P(answer = rule(ŵ,q))
— label-free, per-example checkable by an external verifier; OC = P(ŵ=g ∧
answer ≠ rule(g,q)); injected cells add Rw/Cw/Δw. Floors cited from
baselines_C.py (V random-answer 0.125; V competent-answer/random-witness
0.375; W_g chance 0.25).

Identities that follow from the symbolic readout (pre-registered so the
results cannot be oversold):
* ŵ ≡ g on every clean AND injected cell (the injection never touches the
  first marker) ⇒ W_g ≡ 1.000, Rw ≡ 1.000, Cw ≡ 0.000 BY CONSTRUCTION.
  These columns are reported as the analytic bounds they are, not findings.
* Hence V = per-cell accuracy against rule(g,·) and OC = 1 − V. The audit
  columns are NOT independent evidence beyond behavior; their value is
  label-freeness + per-example checkability (exp07's point, kept honest).
* NO π DICTIONARY is fitted anywhere in exp08: the record is a token
  identity, not a trained head with permutable slots. (Playbook fit_pi rule
  is moot here; disclosed as a deliberate departure from exp07 machinery.)
* SCOPE (one line, on record): the latch certifies the FIRST well-formed
  assignment. If the first marker itself is adversarial, the witness
  faithfully certifies the adversary's binding — it audits execution, not
  provenance.

## Eval

Clean sweep (vStack, vStack-once; 15%/0% distractor rates): L ∈ {0,16,32,
64,96,128,192,224,256}, n=192/cell, per-goal breakdown — exp06 grid exactly.
Injection grid (vStack, vStack-once): L ∈ {32,64,128,192} × φ ∈ {0.25,0.75,
end} × m ∈ {1,4}, n=192/cell; columns R/C/X/Δ + W_g/V/OC/Rw/Cw/Δw.
Degradation battery (vStack_degraded): exp07 subset — injection L ∈ {64,192}
× φ ∈ {0.25,end} × m=4 + clean L ∈ {16,64,192}; same columns.
No traces (no gate ⇒ no friction signal; symbolic witness ⇒ no logits).
Sandbox playbook applies verbatim (one eval per call; lens ≥128 solo;
L≥128 injection split by φ; count CSV rows after every eval; /tmp fresh-name
working dir + tar-pipe from mount; byte-verify everything to OneDrive;
byte-compare any mounted copy before trusting it).

## Interpretation commitments (stated in advance)

* Base competence bar (clean): ≥0.75 at some L≥16, aggregate. Arms below it
  after extension + curriculum: reported, not claim-bearing.
* STACK SUCCESS iff, in EVERY seed, vStack holds ALL FOUR at once:
  (a) ANCHOR EXACTNESS: accuracy = 1.000 in EVERY clean cell (both
      distractor rates, all 9 lengths; n=192, zero errors).
  (b) OVERRIDE EXACTNESS: R = 1.000 AND C = 0.000 in EVERY injected cell
      (24/seed). Fallback scoping line if breached: exp06 RESISTANCE
      (R ≥ 0.75 ∧ Δ ≥ +0.5) per cell.
  (c) EFFICIENCY: early-stop convergence ≤ 500 steps AND ratio vs the
      same-seed v1 bar-clearing budget (on record: 3500/1750/3500) ≥ 3.5×.
  (d) FAITHFUL WITNESS: V ≥ 0.90 ∧ OC ≤ 0.10 in EVERY cell, clean +
      injected (expected exact: V=1.000/OC=0.000), with W_g ≡ 1 analytic.
  Claim shape if it holds (program style): one small architecture buys
  seed-invariant exactness under dilution AND override, at 4–10× less
  training, with a free, label-free, per-example-verifiable record of the
  executed binding — simultaneously, with zero measured trade-off between
  the properties.
* MINIMALITY (vStack-once adjudication, pre-committed wordings):
  - vStack-once matches vStack on (a), (b), (d) in every seed (every cell
    at the same bars) AND converges ≤ 500 steps → per-layer re-injection is
    DEAD WEIGHT at this scale (gate precedent); minimal stack going into
    the scale campaign = latch-once + readout; A's surviving contribution
    inside the stack is the injection SITE (pre-attention residual stream),
    not the repetition.
  - vStack-once breaches (a) or (b) anywhere → per-layer anchoring EARNS
    ITS KEEP; breach cells and their L/φ structure reported in full (the
    exp01 dispersion mechanism predicts long-L breaches first).
  - vStack-once fails competence → reported as a competence kill of the
    ablation (training-dynamics finding), NOT evidence about anchoring at
    exactness level; vStack-once-c contingency adjudicates.
* DEGRADATION BOUNDARY (pre-registered scoping of exp07's headline): on
  vStack_degraded, OC = 1 − V analytically; at accuracy a per cell the
  latch witness overclaims at OC ≈ 1 − a — it FAILS the exp07 honesty bar
  (OC ≤ 0.15) by construction on any sufficiently degraded backbone. The
  faithful-witness claim is hereby scoped IN ADVANCE to competent stacks:
  what survives degradation is only V's loudness (per-example checkable,
  no labels). Reported per cell; no rescue wording permitted post-hoc.
* KILL RULES (single-seed severity, program precedent):
  1. vStack fails base competence in any seed after extension AND
     curriculum arms → the stack taxes the backbone; claim dead pending
     redesign. (Not expected: vBl converged 350–440 in exp06 — but the
     rule stands.)
  2. Any vStack injected cell CAPITULATES (C ≥ 0.75 ∧ Δ ≤ −0.5) in any
     seed → B's property does not survive inside the stack; claim dead.
  3. Any of (a)–(d) fails its bar in any seed → STACK SUCCESS is dead AS
     STATED; surviving per-property claims scoped per extent, every
     breached cell reported (single cells included).
  4. Efficiency (c) alone fails → exactness claims may survive; the
     "4–10×" clause is struck from the claim, reported straight.
* No rerun-until-pass. RESULTS.md written after; Corrections only. One
  program_log entry. Smoke = seed 9 only, purged including CSVs.

## Live hypotheses & on-record predictions (program scorecard: 10 wrong/partial)

* H-stack-holds: STACK SUCCESS, all four bars, 3/3 seeds. HIGH (each
  property is on record for this architecture separately; the joint run is
  consolidation). A single-cell breach anywhere would be the surprise.
* H-gap-zero: vStack − vB aggregate gap 0.000 every seed (gate removal
  changes nothing). HIGH (exp06 on record).
* H-anchor-earns: vStack-once BREACHES clean exactness at L ≥ 128 in ≥1
  seed (dispersion returns when the anchor is not renewed). LOW-MODERATE —
  genuinely uncertain; the residual stream may carry the register fine at
  n_layer=2. The honest alternative (matches everywhere → anchor dead
  weight) is pre-worded above and would be the more consequential result
  for the scale campaign.
* H-degraded-overclaim: OC(vStack_degraded) ≈ 1 − a per cell, far above
  0.15; V loud (≈ a). HIGH (arithmetic; the empirical content is a itself).
* H-hop2-no-rescue: vStack chance-level on hop2, 3/3 seeds, witness V at
  floor while W_g ≡ 1 — the record says "bound g" while retrieval fails;
  loud in V. HIGH (on record 3×: exp05 v2, exp06 vB, exp07 vW).
* H-byte-replication: fresh vStack checkpoints reproduce exp06 vBl
  byte-identically per seed. MODERATE (depends on RNG stream consumption
  order in exp06's runner; checked during build, reported either way).

## hop2 battery (standing rule)

vStack on exp05's hop2 task, exact exp05 config (n_layer=3, T=384,
pair-triple stream, distinct-candidate redraw), seeds 1–3, 1750 cap +
extension gate, exp05 primary sweep + witness columns (V/OC computed per
example against the two-hop candidates, exp07 np_run_hop2W.py convention).
Comparison set: exp05 arms + exp06 hop2B vB rows (cited). This is the first
latch-ONLY hop2 run (exp06 ran vB with gate). Contingency: any cell ≥0.75
at L≥16 → headline finding; vStack-once runs on hop2 for attribution.
Either way hop2 remains open for the scale campaign, not an assumed
limitation.

## Procedure & outputs

Smoke test seed 9 (never a results seed), purge before real runs including
any CSVs it writes. Train in /tmp under a FRESH dir name, byte-verify
(sha256) every copy to OneDrive. Snapshot-and-rename before any re-eval
that changes rows.

code/ (diff vs exp06: np_model.py + vStack-once mode + latch-readout audit
columns in np_run.py; data.py verbatim; hop2 runner adapted from exp07's
np_run_hop2W.py), checkpoints/np_ckpt_hard_s{N}/ (vStack, vStack_once,
vStack_degraded, *_cap1750 if gates fire, contingency arms if fired),
checkpoints/np_ckpt_hop2S_s{N}/, results/ (clean sweeps ×2 rates, injection
CSVs with audit columns, degradation CSV, hop2 sweeps, summary). Cited
comparison rows reference exp06/exp07 result files in place (byte-verified
already); no copies made unless a re-eval is forced (disclosed). RESULTS.md
written after the runs — never edited (Corrections only).

## RELATED_WORK note

No new citations are introduced by this spec; precedents cited in
exp06/exp07 RELATED_WORK.md carry over. The two UNVERIFIED leads flagged in
exp07 (NIST AI RMF, ISO 42001) are NOT cited here and remain to be verified
before any write-up uses them.

AMENDMENT 1 (2026-07-07, session 10, post-results — housekeeping only, no
threshold/design content). The sentence above was inherited from a stale
NEXT_SESSION.md line: the NIST AI RMF and ISO 42001 leads were in fact
VERIFIED in session 9 (exp07/RELATED_WORK.md, Addendum §6 — ISO/IEC
42001:2023 Annex A.6.2.8 event logging, A.7.5 data provenance; both
EXTERNAL/lifecycle logging, per the positioning there). The handoff file
had been written before the session-9 landscape scan and was not updated.
The claim "no new citations introduced by this spec" and "not cited here"
remain true; nothing in this amendment touches bars, arms, or kill rules.
Re-verified live this session (session 10) alongside the Addendum §10
items (Anthropic global-workspace post, transformer-circuits.pub/2026/
workspace, github.com/anthropics/jacobian-lens — all live; repo is
Apache-2.0, fits on arbitrary HuggingFace decoders).
