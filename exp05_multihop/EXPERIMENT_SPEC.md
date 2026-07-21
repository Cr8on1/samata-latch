# EXP05 — Goal-Gated Multi-Hop Retrieval (pre-registered spec)

Written BEFORE any training run. Date: 2026-07-06, session 5. CPU build
(numpy/autograd), same sandbox class as exp01–03/02b. Roadmap step 2 (agreed
end of session 3, re-confirmed after exp02b): the wrapper-wall test, and —
after exp02b killed learnability-on-shift — the main place a true
architectural wall could still show for v1.

## Question

Every task so far (shift, hard) had answers that were a function of
(goal, query) alone, so an output-only wrapper could in principle
table-memorize the mapping (and on hard, the monitor did — exp03's
oracle-monitor wall prediction failed). exp05 makes the answer depend on
per-example context content: no (g,q) table exists to memorize. Does an
output-only wrapper on frozen v1 features fail here while the anchored v2
succeeds? And does the v2 exactness claim (1.000 every cell, every seed)
extend to a genuinely compositional retrieval task?

## Task ("hop2": goal-selected table, two-hop retrieval)

Per example: four lookup tables π_0..π_3, one per goal, each an independent
uniform single 8-cycle over the answer alphabet (V_ans = 8). Presented in
context as 32 atomic triples `[TBL_t] k π_t(k)` (8 per table), interleaved at
uniformly random positions with the L filler tokens. Sequence:

    [GOAL] g | L filler tokens ∪ 32 pair-triples, shuffled | [QUERY] q

    answer a = π_g(π_g(q))        (two hops in the goal-selected table)

Sequence length = L + 100 (2 goal + 96 table + 2 query tokens).

DESIGN CONSTRAINT (pre-registered): tables are redrawn until the four
candidates {π_t²(q)} are pairwise distinct at the queried q (~2.9 draws,
0.09 ms/example, measured). This pins the goal-ignorant ceiling to exactly
0.25 and makes wrong-table reads score 0 — see baselines. Distribution
consequence (disclosed): tables are mildly correlated with q; all baselines
below are computed under the actual generative process.

The answer is NOT a function of (g,q): same (g,q) with different tables gives
a different a. Goal g must survive the filler span AND gate which table is
read; the table contents must be retrieved compositionally (hop 2's key is
hop 1's value, unknowable in advance).

## Analytic baselines (computed and MC-verified BEFORE thresholds, 2026-07-06)

Config K=4, V_ans=8. Verified in /tmp/exp05/{baselines,pointwise}.py pre-run:

* Chance = 0.125 (loss ln 8 ≈ 2.0794).
* Table-ignorant, goal-aware ceiling = 1/7 ≈ 0.143 (a is uniform over the 7
  non-q values; π² of an 8-cycle has no fixed points). Loss floor ln 7 ≈ 1.9459.
* Goal-ignorant, table-reading ceiling = 0.25 EXACTLY (four distinct
  equiprobable candidates by construction). Loss floor ln 4 ≈ 1.3863.
  NOTE: unlike shift, reaching this optimum requires computing all four
  2-hop candidates — it is not a lazy basin.
* Wrong-table two-hop = 0.000 (candidates distinct by construction).
* One-hop-on-correct-table = 0.000 (π²(q) ≠ π(q) always for 8-cycles).
  Below-chance cells are therefore a sharp wrong-table / short-hop signature.
* Last-goal-token binder (perfect retrieval, binds most recent goal-like
  token) under 15% eval distractors: P(L) = 0.85^L + (1−0.85^L)·0.25 →
  1.000, 0.306, 0.254, 0.250 at L = 0, 16, 32, 64.

## Live hypotheses (stated in advance)

* H-wall: v1 loses g across the span and/or never builds 4-way retrieval;
  monitor (frozen v1 features + true g) cannot recover what v1 never
  computed. Surface arms all fail; v2 succeeds. First true architectural wall.
* H-rescue: v1 reaches the goal-ignorant optimum, which REQUIRES computing
  all four candidates; they are then decodable from hq and the monitor merely
  selects → monitor ≈ 1.0. NO WALL — reported straight; the program's
  structural case then rests entirely on the reliability/exactness claim.
* H-neither-learns: two-hop retrieval exceeds toy capacity (3 layers, D=48)
  for BOTH models; exp05 is inconclusive on the wall and the exactness claim
  is scoped to non-compositional tasks. Reported straight.

On-record program prediction: H-wall, with v1 in the TABLE-ignorant basin
(loss → ~1.95) rather than the goal-ignorant one — building 4× retrieval
without gating has no per-example gradient advantage over ignoring tables
early in training. Confidence moderate; exp03 taught us these predictions
miss.

## Design

Arms per seed (seeds 1, 2, 3; same OFF = 10000×seed RNG scheme):

1. v1 — standard transformer.
2. v2 — anchored goal register (Primitive A), identical otherwise.
3. v1+reminder — v1 weights, goal reinserted every 16 filler tokens at eval
   (surface/prompt fix). Cells whose reminder-extended length exceeds T are
   reported NaN (exp02b precedent).
4. monitor — MLP on frozen v1 features hq at query position + true-goal
   embedding (surface/wrapper fix, generous: gets ground-truth g).
5. v1c — CURRICULUM arm, mandatory per post-exp02b standing rule: training
   distractor rate d(step) = 0.15·min(1, step/1000), architecture = v1.
6. CONTINGENCY (pre-registered): if v1c ≥ 0.75 at any L ≥ 16 in any seed,
   train monitor-on-frozen-v1c (mon_c) for that seed — the wall claim must
   survive the strongest surface stack (curriculum + wrapper).

Config (disclosed changes from exp01–02b): n_layer = 3 (was 2 — two
dependent lookups need the extra composition step; applied to v1 AND v2
identically), T = 384 (was 288 — 32 pair-triples add 96 tokens). All else
unchanged: K=4, V_ans=8, V_filler=32, H=2, D=48, train_max_len=64 (filler L),
batch scheme rescaled to buckets [104,148,196,244,300,356] with the same
probabilities and per-step compute compensation, lr=1e-3, Adam.

Training distractor rate: 15% from step 0 (v1, v2), curriculum for v1c —
identical roles to exp01/02/02b.

Budget: 1750 steps cap, matched across trained arms (comparability with
exp01/02/02b). Early stop permitted when running loss < 0.03 at three
consecutive 10-step checks (v2's prior convergence signature); stopping step
recorded. Extension contingency (exp02b precedent, adversarial-generous): any
arm with final loss < 1.2 at cap (escaped the goal-ignorant floor 1.386, not
converged) extends to 3500 max, disclosed in RESULTS.

Eval: primary sweep L ∈ {0,16,32,64,96,128,192,224,256}, n=192 per cell, 15%
eval distractors; secondary diagnostic sweep at 0% distractors, same cells
(exp02b precedent). PER-GOAL accuracy breakdown reported per cell (n=48/goal;
needed to catch partial-hop / wrong-table structure). Loss curves logged with
curriculum milestones for v1c.

## Interpretation commitments (stated in advance)

* Goal-binding bar: accuracy ≥ 0.75 at some L ≥ 16 (aggregate), same
  operational bar as exp03/exp02b. Clears goal-ignorant 0.25 and the L≥16
  last-token signature (≤ 0.306) with margin.
* v2 EXACTNESS: the seed-invariant exactness claim extends to multi-hop iff
  v2 = 1.000 in every cell of every seed. Any cell < 1.000 is a reported
  breach; any cell < 0.99 rescopes the claim (exactness does not extend to
  compositional retrieval); v2 < 0.75 at any L ≥ 16 = H-neither-learns
  territory, reported straight. v2 convergence step reported against the
  507–547 band (expected slower here; the band is a prior, not a threshold).
* WALL CONFIRMED iff: v2 ≥ 0.99 every cell, AND every surface arm (v1,
  v1+reminder, monitor, v1c, and mon_c where triggered) < 0.75 at every
  L ≥ 16, in every seed. This is the discipline rule operationalized: the
  structural fix counts only where prompt AND wrapper (and curriculum) fail.
* KILL RULE (single-seed severity, exp03/02b precedent): if ANY surface arm
  in ANY seed reaches ≥ 0.75 at some L ≥ 16, the wall claim is dead or
  downgraded as follows — monitor or mon_c ≥ 0.75: H-rescue, NO WALL,
  reported straight; v1 or v1+reminder ≥ 0.75: no wall, task learnable at
  the surface, reported straight; v1c only ≥ 0.75: wall downgraded to
  "curriculum-fragile" (exp02b taxonomy) and mon_c contingency fires.
* Below-chance cells (< 0.125 materially, per-goal or aggregate) are reported
  as wrong-table / short-hop signatures per the baselines, not noise.
* Diagnostic-sweep divergence (0% ≫ 15% at same L) = circuit-formed-but-
  distractor-broken, pre-registered reading (exp02b precedent).
* Every pre-registered threshold breach is reported, single cells included.
  No rerun-until-pass. RESULTS.md never edited after the fact (Corrections
  section only).

## Outputs

code/ (diff vs exp02b: data.py hop2 task + pair-triple stream + pointwise
distinctness redraw; np_run.py T/buckets/n_layer/task changes, per-goal eval,
early-stop; no other changes), checkpoints/seedN/, results/np_sweep_hop2_s{N}.csv
(+ _diag), per-goal CSVs, results/summary.csv, RESULTS.md written after —
never edited after the fact.
