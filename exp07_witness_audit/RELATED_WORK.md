# EXP07 — Related work & standards positioning (session 8, 2026-07-07)

All arXiv IDs below VERIFIED LIVE this session (abstract pages fetched or ID
confirmed in search) before inclusion — program citation protocol.

## 1. Concept Bottleneck Models — the direct precedent for the seed-1 finding

* Koh et al., "Concept Bottleneck Models," arXiv:2007.04612 (VERIFIED) —
  the canonical predict-concepts-then-label architecture. exp07's vW/vWr is
  a CBM-shaped design with ONE concept (the assigned goal) and NO concept
  labels.
* Margeloiu et al., "Do Concept Bottleneck Models Learn as Intended?,"
  arXiv:2105.04289 (VERIFIED) — CBM concepts fail to carry their intended
  semantics; predictors exploit leaked information. This is EXACTLY the
  seed-1 vWr result: the K-way record passes competence while W_g^π/V show
  it carries answer-hint information, not the goal. Our audit statistics
  (V, OC) are a per-example, label-free leakage detector — arguably sharper
  than the CBM literature's aggregate diagnostics, because the task is
  synthetic and rule(ŵ,q) is checkable per example.
* Broader CBM leakage literature exists (surveys + mitigation via
  supervised/hard concepts) — UNVERIFIED beyond titles; scan before citing
  specific mitigation papers in the writeup. Key mitigation theme: leakage
  is fixed by SUPERVISION or HARD symbolic pinning — consistent with our
  emerging conclusion (the latch = hard symbolic pinning; supervision =
  what the probe needs).

## 2. Discrete-representation training — standard-practice grounding for Amendment 1

* Jang, Gu & Poole, "Categorical Reparameterization with Gumbel-Softmax,"
  arXiv:1611.01144 (VERIFIED live).
* van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE),
  arXiv:1711.00937 (VERIFIED live).
* Codebook/slot collapse under straight-through training is a DOCUMENTED
  standard failure mode of discrete bottlenecks, and Gumbel-softmax
  sampling + load-balancing are standard mitigations. exp07's smoke-seed
  collapse and Amendment 1 (Gumbel-ST + KL-to-uniform load balance) follow
  established practice, not ad-hoc patching.

## 3. Faithfulness of self-reports — the framing exp07's V statistic operationalizes

* Turpin et al., "Language Models Don't Always Say What They Think,"
  arXiv:2305.04388 (VERIFIED) — CoT explanations systematically
  misrepresent the true cause of predictions.
* Lanham et al., "Measuring Faithfulness in Chain-of-Thought Reasoning,"
  arXiv:2307.13702 (VERIFIED) — faithfulness measured by intervening on
  the stated reasoning; larger models often less faithful.
* Positioning: exp07 moves the faithfulness question from post-hoc text to
  ARCHITECTURE — V = P(answer consistent with the carried record) is a
  mechanical, label-free faithfulness check, and the causal bottleneck is
  faithfulness-by-construction IF the record learns the right semantics.
  Seed 1 shows the "if" fails without symbolic pinning; the latch (exp06)
  provides that pinning for free (V=1.000/OC=0.000, 24/24).

## 4. Compliance / standards hook (PERIPHERAL per program framing — an
## application note, never the lead)

* EU AI Act Article 12 (Record-keeping): high-risk AI systems must
  technically allow automatic recording of events over the system's
  lifetime, enabling traceability of operation (see
  artificialintelligenceact.eu/article/12 — official-text mirror). Logging
  there is EXTERNAL (I/O events). exp07's witness question is the
  architectural analog: can the model itself carry a verifiable record of
  WHICH instruction it executed? Seed-1 answer: only if the record is
  structurally pinned (latch), not gradient-emergent. If replicated, the
  practical corollary for the writeup: exp06's latch is a zero-cost,
  in-forward-pass traceability record of the bound assignment — a
  compliance-relevant property obtained architecturally.
* NIST AI RMF / ISO 42001 audit-trail themes: UNVERIFIED this session —
  scan before citing in any writeup.

## 5. Positioning vs the exp06 four-level scheme

Instruction Hierarchy (2404.13208), StruQ (2402.06363), SecAlign
(2410.05451), CaMeL (2503.18813) — verified in session 7 (exp06/
RELATED_WORK.md). exp07 adds an orthogonal axis to all four levels:
VERIFIABILITY of the binding, not robustness of the binding. None of the
four provide a per-example, label-free record of which instruction was
executed; the latch does, exactly, for free.

---

## ADDENDUM (session 9, 2026-07-07, post-RESULTS landscape scan)

Verification protocol as above: VERIFIED = abstract page fetched or ID +
title + abstract confirmed in search this session; LISTING-ONLY = seen in
search results, scan before citing.

### 6. Standards leads — now VERIFIED (section 4's open item closed)

* NIST AI RMF: traceability is named a core characteristic of trustworthy
  AI; the framework calls for records supporting oversight and
  reconstruction (accountability records: model/prompt versions, input
  provenance, outputs, approvals). All EXTERNAL artifacts, lifecycle-level.
* ISO/IEC 42001:2023: Annex A.6.2.8 (AI system recording of event logs)
  requires event logging across training/validation/deployment/inference;
  A.7.5 (data provenance) complements it. Again: EXTERNAL logging of I/O
  and lifecycle events.
* Ojewale, Suresh & Venkatasubramanian, "Audit Trails for Accountability
  in Large Language Models," arXiv:2601.20727 (Jan 2026, VERIFIED) — the
  current academic synthesis of that same posture: tamper-evident
  lifecycle ledgers linking technical provenance to governance records.
* POSITIONING (unchanged, now grounded): the entire standards/audit-trail
  stack records what went IN and OUT of the model. None of it can say
  which instruction the model EXECUTED. exp07 shows that gap cannot be
  closed by training a reporter head (probe: seed-variant; witness:
  answer-hint channel) — but the latch closes it by construction. That is
  the one-sentence bridge from this program to the compliance literature,
  and it stays peripheral per framing.

### 7. CBM update — the field independently confirmed exp07's core finding

* Almudévar, Hernández-Lobato & Ortega, "There Was Never a Bottleneck in
  Concept Bottleneck Models," arXiv:2506.04877, ICLR 2026 (VERIFIED,
  abstract fetched): "the fact that a component can predict a concept does
  not guarantee that it encodes only information about that concept" —
  exp07's vWr result stated as a general claim. Their fix is a variational
  information-bottleneck regularizer (a TRAINED mitigation); ours is
  symbolic pinning (a STRUCTURAL one). exp07's 72/72 audit failure is a
  clean adversarial-free counterexample for the trained route at toy
  scale; their result says the disease is real at full scale.
* Hard-concept line (leakage mitigated by binary/hard concepts) continues:
  probabilistic hard CBMs, Hi-CBM intervention-matrix (LISTING-ONLY);
  survey of concept-based-model risks arXiv:2506.04237 (LISTING-ONLY).
  Direction of the whole mitigation literature = harder, more symbolic
  records — convergent with the latch conclusion.

### 8. Injection-defense state of practice (2026) — context for exp08 claims

* Consensus framing in current practitioner literature: no architectural
  separation of instructions vs data exists in deployed transformers; the
  problem is treated as unresolvable in-model, so defenses reduce blast
  radius (capability isolation, deterministic policy outside the LLM,
  egress constraints). Reported numbers: CaMeL provable security on 77%
  of AgentDojo tasks; StruQ <2% attack success on its benchmark; agent
  privilege separation arXiv:2603.13424 (LISTING-ONLY) reports 0% on its
  own benchmark via structural privilege splits.
* POSITIONING: everything successful in this space is STRUCTURAL and
  OUTSIDE the model. Lacuna's bet is the same design philosophy moved
  INSIDE the forward pass; exp06/07 toy evidence (trained defenses
  seed-variant or leaky, structural ones exact) rhymes with the field's
  trajectory. Use as motivation framing for exp08, not as a competing-
  benchmark claim — different scale, different threat model.

### 9. Faithfulness-by-architecture — small but real emerging lane

* "A transformer architecture alteration to incentivise externalised
  reasoning," arXiv:2603.21376 (Mar 2026, VERIFIED): early-exit + RL to
  force computation into visible tokens rather than hidden compute —
  architecture altered so the record (the CoT) is more likely to BE the
  computation. Closest current neighbor to exp07's thesis at LLM scale;
  incentive-based where the latch is constraint-based.
* RFEval counterfactual-intervention faithfulness benchmark
  (arXiv:2602.17053) and mechanistic faithfulness-decay evidence
  (arXiv:2602.11201): LISTING-ONLY — scan before citing in a writeup.
* Net read for exp08: nobody in this lane has a per-example, label-free,
  zero-training-cost execution record. The latch-derived C readout is a
  publishable claim shape if the stack holds it.

### 10. Anthropic "global workspace / J-space" (Jul 6 2026) — VERIFIED (full
### post fetched from anthropic.com/research/global-workspace, session 9)

* Anthropic's interpretability team reports an EMERGENT internal workspace
  in Claude ("J-space"): a small set of activation patterns, found via a
  Jacobian-based lens over would-say output tokens, that (a) CAUSALLY
  mediates multi-step reasoning (swapping "spider"→"ant" in the space flips
  the downstream answer — intervention, not correlation), (b) is reportable
  and controllable on request, (c) is read by many downstream computations
  (one France→China edit redirects capital/language/continent/currency),
  (d) carries silent safety-relevant content: eval-awareness ("fake",
  "fictional" during the agentic-misalignment blackmail scenario — ablating
  those patterns re-enables blackmail), fabrication intent ("manipulation"
  while editing score files), planted malicious goals in model organisms.
  Paper: transformer-circuits.pub/2026/workspace; open-source lens:
  github.com/anthropics/jacobian-lens; Neuronpedia demo on open-weights
  models. Anthropic's own caveat, verbatim class: the J-lens is "an
  imperfect method, which only approximately captures the model's 'true
  workspace'"; single-token concepts only.
* RELEVANCE TO THIS PROGRAM (three points, in claim-strength order):
  1. PREMISE VALIDATED AT FRONTIER SCALE: a causal, workspace-like internal
     commitment record EMERGES in large transformers — the thing exp07
     tried to force at toy scale exists at scale, unpinned. The program's
     bet is unchanged but sharpened: the record exists; what's missing is
     EXACTNESS. J-lens is a discovered, approximate, lens-mediated readout;
     the latch is a designed, exact, per-example one. "Approximate lens vs
     exact latch" is now the one-line differentiation vs the strongest
     possible neighbor.
  2. exp07's audit statistics transfer: OC (certify-while-not-executing)
     is precisely the signature the J-lens caught qualitatively
     ("manipulation" during fabrication). V/OC are quantitative, per-
     example versions of what their monitoring examples do by inspection.
  3. ROADMAP HOOK (funding-path memory: frontier-model inference battery
     is priority 1): the lens is open-sourced with an open-weights demo
     pipeline — Lacuna's V/OC battery could be computed over J-lens
     readouts on downloaded open models, i.e., the frontier inference
     battery has a ready-made instrument. Also note their eval-awareness
     ablation result for the battery's design: behavior under test is
     contingent on latent test-recognition — injection/override evals
     should control for it.
* Caution for any writeup: do NOT adopt the consciousness framing (their
  own disclaimer is explicit; program style: no metaphysical framing).
