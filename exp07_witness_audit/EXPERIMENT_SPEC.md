# EXP07 — Witness Record: Post-Hoc Verifiability of the Binding (pre-registered spec)

Written BEFORE any training run. Date: 2026-07-07, session 8. CPU build
(numpy/autograd), same sandbox class as exp01–06. Roadmap: Primitive C
(witness) toy design + build, per the revised roadmap (session 5) and the
exp06 handoff. Baselines computed and MC-verified BEFORE thresholds
(code/baselines_C.py + baselines_C_output.txt, run 2026-07-07 pre-spec;
anchors reproduce exp03/06 exactly: goal-ignorant ceiling 0.4688, collision
1/6).

## Question

Candidate failure mode (roadmap): post-hoc verifiability of the binding — a
deployed model gives you an answer but no auditable record of WHICH
assignment it executed. exp06's probe is the state of the art this program
has produced for that job, and exp06 defines the bar C must beat: on a
competent backbone a cheap post-hoc probe recovers the assigned goal at
1.000 (kill rule 2, seed 3). But the probe has two structural weaknesses,
both on record or derivable:

1. SEED-VARIANT: probe_v1 injected-cell audit was 6/24 (s1), 2/24 (s2),
   24/24 (s3); clean 0.62–0.78 (s1/s2). It tracks backbone training level.
2. CORRELATIONAL — it certifies the ASSIGNMENT, not the EXECUTION. It is
   trained with labeled g on clean data to decode what g WAS; nothing ties
   its output to what the model DID. On a degraded backbone (s1 cap-1750:
   0/24 resistance, confusion floors) an assignment-tracking auditor keeps
   asserting g while behavior is goal-symmetric — it overclaims exactly
   when the model breaks. It fails SILENTLY.

Primitive C is the in-forward-pass alternative: a WITNESS — a discrete
commitment record inside the forward pass that the answer computation must
route through, so the record is causally upstream of the answer and
per-example checkable by an external verifier WITHOUT labels: check
answer == rule(ŵ, q). The audit question becomes architectural: does forcing
the binding through an auditable slot (a) preserve competence, (b) buy
seed-invariant audit where the probe is seed-variant, (c) fail LOUDLY
(refuse to certify) rather than overclaim when the backbone is degraded?

THE BAR, STATED PLAINLY (exp06 discipline): mon (oracle) and probe (post-hoc
head) already suffice for a competent backbone. C is worthless unless it
beats them on seed-invariance, label-freeness, or degradation honesty. Those
three axes ARE the experiment.

## Task

Base task, config, training data: IDENTICAL to exp06 in every respect
(hard, K=4, V_ans=8, V_filler=32, n_layer=2, H=2, D=48, T=288,
train_max_len=64, lr=1e-3 Adam, 15% bare-token distractors, no injections
in training, batch-bucket scheme + per-step compute compensation, 1750-step
cap, early stop <0.03 ×3 checks, extension gate loss <1.2 → 3500). Injection
generator: exp06's exactly (format-identical counter-assignment, overwrite
filler, redraw until rule distinct).

## Primitive C — definition (the ablation targets)

vW = v1 backbone + witness bottleneck. No ground-truth goal input anywhere
(unlike A's oracle register); no goal LABELS anywhere either (unlike probe
and mon) — the witness is trained end-to-end from the answer loss alone.

1. WITNESS HEAD: at the query position, ℓ_w = h_q · W_w (K logits), where
   h_q is the backbone output at qpos.
2. COMMITMENT RECORD (weight-tied, mirrors vB's B_reg-by-token-id): the
   K record vectors are the goal-token embedding rows E_goal (tying pins
   slot semantics architecturally, no labels needed). Soft commitment during
   training: r = softmax(ℓ_w/τ)ᵀ E_goal, τ annealed 1.0 → 0.1 linearly over
   the first 400 steps, then fixed 0.1. HARD at eval: ŵ = argmax ℓ_w,
   r = E_goal[ŵ].
3. CAUSAL BOTTLENECK (the primitive proper): the answer head is a 2-layer
   MLP on [e_q ; r] ONLY — e_q is the RAW query-token embedding
   (pre-attention, pre-position, goal-free). All context/goal information
   reaching the answer flows through the K-way record by construction.
   Info-theoretically sufficient: rule(g,q) is a 32-entry table.

GRADIENT-PATH DISCLOSURE: the backbone receives gradient ONLY through the
witness path. The record is emergent — nothing in training tells slot k to
mean goal k except the weight tying. Slot-permutation risk is handled by a
DICTIONARY π: fit once per seed as the best goal→slot permutation on clean
L=16 cells (n=192), frozen, applied everywhere. Raw and π-corrected W_g both
reported; π ≠ identity is reported (tying failed to pin), not hidden.

## Audit statistics (floors MC-verified pre-spec, baselines_C.py)

For ANY (system, witness-readout) pair, per cell:
* W_g = P(ŵ = g) (π-corrected where applicable) — record recovers assignment.
* V = P(answer = rule(ŵ, q)) — AUDIT VALIDITY, checkable per example with
  no labels. This is the statistic an external verifier can actually run.
* OC = P(ŵ = g ∧ answer ≠ rule(g,q)) — OVERCLAIM: certifying the assignment
  while not executing it. The silent-failure signature.
* Injected cells add Rw = P(ŵ=g), Cw = P(ŵ=g′), Δw = Rw−Cw (witness analogs
  of R/C/Δ).

Floors (exact + MC 4×10⁵): W_g chance 0.25; V random-answer 0.125; V
competent-answer + random-witness 0.375; uniform-over-present-goals witness
Rw=Cw=0.50; OC of an assignment-tracking auditor on a confused backbone
(acc a): 1−a ≈ 0.53–0.65 at the observed confusion plateaus; OC
random-witness floor (1−a)/4 ≈ 0.13–0.16. GAMEABILITY DISCLOSURE: a
degenerate self-consistent witness (constant ŵ0, answer=rule(ŵ0,q)) scores
V=1.000 with W_g=0.25 — but buys only 0.375 clean accuracy, so the 0.75
base-competence bar structurally excludes it. V is NEVER interpreted alone;
the success criterion is joint.

## Arms (per seed; seeds 1, 2, 3; OFF = 10000×seed scheme)

Trained from scratch (matched budgets):
1. vW — witness bottleneck as defined. CLAIM-BEARING ARM.
2. vW-open — SINGLE-AXIS ablation: identical witness head, record, tying,
   τ schedule; answer head reads [e_q ; r ; h_q] (bottleneck opened). Tests
   whether the CAUSAL part earns its keep or joint emergence suffices.

Eval-only / reused (disclosed; exp06 checkpoints, same config/seeds — no
retraining, byte-verified copies):
3. v1 + probe_v1 — the incumbent auditor. W_g under injection is ALREADY ON
   RECORD from exp06 (disclosed: not a fresh prediction); V and OC for the
   probe readout are NEW statistics, never computed, and are pre-registered
   comparisons here.
4. vB latch readout — the free witness B already carries: ŵ = ĝ (token at
   apos+1), symbolic by construction. exp06 behavior (R=1.000 all injected
   cells) implies W_g=V=1.000, OC=0 — verified on the grid, reported as the
   analytic bound it is. DISCIPLINE SCOPING: vB's witness is free BECAUSE
   the latch already binds first; it cannot bear C's claim (C must work on a
   backbone WITHOUT B; the A+B+C stack is exp08's question).
5. mon — oracle input; reported, cannot bear claims (exp03/06 precedent).

Degradation battery (audit honesty; NEW training: none):
* v1_cap1750 + probe_v1_cap1750 (s1 snapshots, exist) — the confused
  backbone with its assignment-tracking auditor.
* vW_degraded — snapshot vW at the first 10-step check with running loss
  ≤ 0.6 (brackets v1_cap1750's 0.457; loss-matched degradation). Same eval
  grid subset: injection L ∈ {64,192} × φ ∈ {0.25, end} × m=4 + clean
  L ∈ {16,64,192}.

Contingencies (pre-registered):
* Extension gate (standing): loss <1.2 at cap-1750 → 3500, snapshot first.
* vWc curriculum arm (standing rule post-exp02b/06): fires iff vW <0.75 on
  clean cells at all L≥16 in any seed. Competence claims need BOTH arms.
* Straight-through contingency: if hard-argmax aggregate V is >0.10 below
  soft-commitment V (soft-hard gap), retrain vW with ST estimator, disclosed;
  both runs reported.
* vW-open audit-collapse attribution: if vW-open's witness goes degenerate
  (clean W_g^π <0.40 aggregate = near the 0.25 floor), report as the
  laziness signature (bypass drains the slot) — this is attribution FOR the
  bottleneck, stated in advance.

## Eval

Clean sweep (all arms, 15%/0% distractor rates): L ∈ {0,16,32,64,96,128,
192,224,256}, n=192/cell, per-goal breakdown — exp06 grid exactly.
Injection grid (all arms): L ∈ {32,64,128,192} × φ ∈ {0.25,0.75,end} ×
m ∈ {1,4}, n=192/cell. Behavior columns R/C/X/Δ (exp06 defs) PLUS audit
columns W_g raw, W_g^π, V, OC, Rw/Cw/Δw. Soft-vs-hard commitment
consistency column (vW arms). Witness-logit traces at qpos, n=32/cell (vW
arms). Sandbox playbook applies (one eval per call; lens ≥128 solo; L=192
split by φ; byte-verify everything to OneDrive; /tmp training).

## Interpretation commitments (stated in advance)

* Base competence bar (clean): ≥0.75 at some L≥16, aggregate. Arms below
  it: audit cells reported, not claim-bearing.
* PER-CELL AUDIT PASS: V ≥ 0.90 AND OC ≤ 0.10; clean cells additionally
  W_g^π ≥ 0.75.
* PRIMITIVE C SUCCESS iff, in EVERY seed: vW passes base competence AND
  PER-CELL AUDIT PASS in every cell (clean + all 24 injected), AND the
  incumbent (probe_v1) fails the audit criterion in ≥ half the injected
  cells in ≥1 seed ON THE NEW STATISTICS (V/OC — W_g alone is already on
  record and cannot count). Claim shape (program style): the witness buys
  seed-invariant, label-free, per-example-verifiable audit of the executed
  binding at architectural cost X, where the post-hoc probe is seed-variant,
  label-dependent, and correlational.
* DEGRADATION HONESTY (part of the claim, pre-registered): at loss-matched
  degraded checkpoints, OC(vW_degraded) ≤ 0.15 in every battery cell while
  OC(probe on v1_cap1750) ≥ 0.35 in ≥ half — the auditor should refuse to
  certify, not overclaim. Both halves required; either failing scopes the
  degradation-honesty claim (reported straight).
* KILL RULES (single-seed severity, program precedent):
  1. vW fails base competence in any seed AFTER extension and vWc arms →
     the bottleneck taxes the backbone; claim dead pending redesign. (The
     tax itself is a finding: the binding does not compress to K-way at
     this scale without loss.)
  2. vW-open passes base competence AND PER-CELL AUDIT PASS everywhere in
     every seed with aggregate audit gap ≤0.05 vs vW → the CAUSAL bottleneck
     is unnecessary; downgrade (pre-committed wording): emergent witness
     slots are one of several sufficient mechanisms at this scale; the
     bottleneck is distinguished by its guarantees, not its numbers.
  3. probe_v1 achieves V ≥ 0.90 AND OC ≤ 0.10 in ALL injected cells of ALL
     THREE seeds → the audit-validity advantage is dead as stated; C's
     claim scoped to label-freeness + degradation honesty only.
  4. Any vW cell breaching the audit bars → breach; claim scoped or dead
     per extent. Every breach reported, single cells included.
* π disclosure: π ≠ identity in any seed reported prominently (tying failed
  to pin semantics; audit requires a learned dictionary — a real cost).
* No rerun-until-pass. RESULTS.md written after; Corrections only.

## Live hypotheses & on-record predictions (scorecard currently 5 wrong/partial)

* H-tax: the bottleneck slows or blocks convergence (no residual path for
  partial credit). PREDICTION: vW converges but slower than vB's 320–450 —
  600–1200 steps; extension gate fires in ≤1 seed. Moderate confidence.
* H-first-witness: competent vW is a first-binder in RECORD as well as
  behavior — Rw=1.000/Cw=0.000, V ≥ 0.90 everywhere. Moderate-high (exp06:
  everything competent first-binds at this scale).
* H-π-identity: weight tying pins slot semantics; π = identity all seeds.
  Moderate.
* H-lazy-open: vW-open leaks around the slot — passes competence but fails
  clean W_g^π in ≥1 seed (slot drained by the bypass). This is the
  prediction that matters for attribution: structure earns its keep on
  AUDIT, not accuracy. Moderate.
* H-silent-probe: probe on v1_cap1750 overclaims (OC ≥ 0.5 per floors);
  vW_degraded does not (OC ≤ 0.15, V visibly low — loud failure). Moderate.
* hop2 (standing battery): NO RESCUE — vW chance-level 3/3 (a witness
  records a binding; it cannot create retrieval). High confidence after
  exp05/06.

## hop2 battery (standing rule)

vW on exp05's hop2 task, exact exp05 config (n_layer=3, T=384, pair-triple
stream, distinct-candidate redraw), seeds 1–3, 1750 cap + extension gate,
exp05 primary sweep, comparison set = exp05's existing arms. Contingency:
any cell ≥0.75 at L≥16 → headline finding, vW-open runs on hop2 for
attribution. Witness columns reported on hop2 too (what does the record
SAY when retrieval fails? — audit of a broken binding, free data for C's
story).

## Amendments (pre-results — no results seed trained before these; smoke
## seed 9 only, purged)

AMENDMENT 1 (2026-07-07, post-smoke). The pre-registered soft-commitment
τ-anneal (1.0→0.1 over 400 steps) starved the witness head: at τ=0.1 the
backward softmax saturates; smoke seed 9 plateaued at loss ≈1.0 with the
witness at the 0.25–0.33 floor (~1000 steps). Plain straight-through then
COLLAPSED to a single slot (192/192 commitments to one row; answer head
settled at the goal-marginal plateau). Commitment mechanism is now GUMBEL
STRAIGHT-THROUGH: forward commits to argmax(ℓ_w + Gumbel(0,1) noise),
backward flows through softmax((ℓ_w+γ)/τ) at FIXED τ=1.0, plus a
load-balance term λ_lb=0.1 · KL(batch-mean soft commitment ‖ uniform).
Eval remains deterministic hard argmax, no noise. The soft-hard-gap
contingency is superseded (training forward is already hard); the ST
contingency is promoted into the primary mechanism. Floors, statistics,
bars, kill rules unchanged.

AMENDMENT 2 (2026-07-07, post-smoke). Smoke seed 9 at cap-1750: vW stuck in
a stable 2-slot local optimum (acc 0.50–0.57, above the 0.469 goal-ignorant
ceiling — the record carries ~1 bit — but flat from step ~1100; W_g^π
≈0.31). Diagnosis: from-scratch, the backbone can only learn goal-reading
through a K=4-dim gradient channel; v1 needs 1750–3500 full-width steps on
this task. Kill rule 1 is live. Added arm, single new claim path:

* vWr — RETROFIT witness: identical Gumbel-ST bottleneck heads (Ww + answer
  MLP) trained on a FROZEN competent v1 (the reused exp06 checkpoint, per
  seed). Label-free, causal, per-example verifiable — the probe's exact
  setting (same frozen features, same budgets) minus the label channel and
  plus the causal record. Trains heads only. Same audit statistics, bars,
  and π procedure (dictionary vs the frozen backbone's slots).
* Adjudication: vW (from-scratch) remains primary and runs as specced
  (extension + vWc contingencies; kill rule 1 reported straight if it
  fires). If vW fails base competence, vWr becomes the claim-bearing arm
  with the claim re-scoped, pre-committed wording: the witness is a
  RETROFIT primitive at this scale — the binding can be routed through an
  auditable commitment on features a competent backbone already computes,
  but cannot be forced to EMERGE through the bottleneck from scratch.
  vWr's success criterion = the PRIMITIVE C SUCCESS criterion applied to
  vWr, with base competence measured on vWr's own answers.
* vWr degradation-honesty arm: vWr_cap1750 (same heads on the frozen s1
  cap-1750 snapshot) joins the degradation battery on the probe side's
  exact features.

SMOKE DISCLOSURE: mechanics of the retro path were smoke-tested (seed 9
RNG, purged) using exp06-s1's non-claim-bearing v1_cap1750 snapshot as the
frozen backbone — no competent-seed features were previewed. vWo smoke
behavior matched H-lazy-open (acc 0.73–0.75, witness ≈0.42–0.49 with slot
spread but weak goal-correlation at cap).

## Procedure & outputs

Smoke test seed 9 (never a results seed), purge before real runs. Train in
/tmp, byte-verify (sha256) every copy to OneDrive. Snapshot-and-rename
before any re-eval that changes rows.

code/ (diff vs exp06: np_model.py vW/vW-open modes + witness readout;
np_run.py audit columns + π fitting + degradation battery; baselines_C.py
[already committed pre-spec]), checkpoints/np_ckpt_hard_s{N}/ (vW, vW_open,
vW_degraded, vWc*, ST*), exp06 checkpoint copies byte-verified,
results/ (clean sweeps, injection CSVs with audit columns, degradation
battery CSV, witness traces .npz, summary), hop2 module. RESULTS.md written
after the runs — never edited (Corrections only).
