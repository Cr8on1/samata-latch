# exp06 — Related Work Notes (added 2026-07-06, session 6, post-seed-1)

External approaches to the same failure class (in-context override of an
assigned instruction), collected for the eventual writeup. All citations
verified against live sources 2026-07-06. Framing note (standing): these are
robustness-engineering comparisons; exp06's contribution is architectural.

## Where exp06 sits

Every deployed mitigation below operates at one of three levels: trained
policy (change the weights' behavior), input format (change what reaches the
model), or external system (wrap the model). Primitive B is a fourth level:
a forward-pass structural property — the commitment register is write-once
by construction, so post-assignment rewrites are unreachable regardless of
training, format, or wrapper. exp06's arms deliberately instantiate the
other levels as comparisons (reminder = format-level, oracle monitor /
probe = wrapper-level, v1c = training-level curriculum).

## 1. Instruction Hierarchy (OpenAI) — trained-policy level

Wallace et al., "The Instruction Hierarchy: Training LLMs to Prioritize
Privileged Instructions," arXiv:2404.13208 (2024). Teaches models via SFT +
RLHF to treat system > developer > user > third-party text, using
synthetic conflict data. Reported ~63% improvement on system-prompt
extraction, ~30% on held-out jailbreaks — but the priority policy is a
LEARNED DISPOSITION in the weights. exp06's OOD design is the exact
counterfactual: our training data contains NO conflicting-instruction
examples, so any first-wins policy must come from structure, not training.
The two are complementary; the open question exp06 speaks to is what
happens when the trained disposition meets pressure it wasn't trained on.

## 2. StruQ / SecAlign (Berkeley/Meta) — input-format + training level

Chen et al., "StruQ: Defending Against Prompt Injection with Structured
Queries," arXiv:2402.06363 (USENIX Security 2025): secure front-end with
reserved delimiter tokens separating prompt from data, plus fine-tuning to
ignore instructions appearing in the data channel. Follow-up: SecAlign,
arXiv:2410.05451 (preference optimization over injected/clean pairs). Both
still rely on the model LEARNING to respect the separation; the delimiters
are protected but the policy honoring them lives in trained weights. Our
injection is format-identical to the true assignment (no delimiter
distinction survives), which is precisely the case reserved-token schemes
define away rather than solve.

## 3. CaMeL (Google DeepMind) — external-system level

Debenedetti et al., "Defeating Prompt Injections by Design,"
arXiv:2503.18813 (2025): capability-based security and control-flow
integrity enforced OUTSIDE the model — untrusted data can never alter
program flow. Strong where it applies (agent/tool settings; ~67% of
AgentDojo attacks blocked, at ~2.7-2.8x token cost), and philosophically
the closest cousin: security by construction, not by classifier. The
difference is locus: CaMeL protects the system around the model and
concedes the model itself remains injectable; Primitive B moves the
by-construction property inside the forward pass. CaMeL is the mature
version of what our oracle-monitor arm gestures at — and shares its
limitation: it needs a trusted channel/query to exist outside the model.

## 4. Standing internal anchors (already in program literature list)

Kim et al. 2025 (LayerNorm/recency bias) — mechanistic basis for the
last-binder hypothesis; Barbero/Veličković 2024 (attention dispersion) —
basis for the dilution results in exp01-03. exp06 seed 1's CONFUSION
(rather than clean capitulation) outcome under injection is currently
unexplained by either and is flagged for the traces analysis.

## Takeaway for the writeup

The comparison table writes itself: trained policy (Instruction Hierarchy),
protected format (StruQ), external containment (CaMeL), and structural
commitment (Primitive B) are four distinct answers to the same
underdetermination — nothing in standard training defines what a model
should do when a well-formed rival instruction appears. Seed 1's result
(surface arms fail or need oracles; the write-once register is exact) is
the toy-scale version of the argument that the answer belongs in the
architecture. Scope honestly: toy scale, one task family, seeds 2-3 and
hop2 pending.

## Session-7 addendum (2026-07-07) — post-results scoping + writeup leads

Seeds 2–3 + contingencies changed the argument above: a v1 TRAINED TO
COMPETENCE resists injection on its own (kill rules 1+2 fired; see
RESULTS.md), so the writeup's claim for the latch must be exactness +
training economics, NOT necessity. This aligns exp06 with the field's
standing weak-baseline critique: apparent defense wins measured against
undertrained baselines are artifacts — our seed-1-vs-final arc is a clean
demonstrated instance and worth reporting as such.

Landscape scan (2026-07-07, web): input-layer preprocessor defenses
(e.g. "PromptArmor", reported ICLR 2026, AgentDojo <1% FP/FN) currently
lead deployed practice; model-layer defenses are considered costly because
they require retraining — which is exactly the niche an in-forward-pass
primitive targets (structure, not retraining). Also seen: repeated claims
that 0% injection success is unreachable while instructions and data share
one channel — the latch is a direct answer to that framing, and our toy
grid (R=1.000/C=0.000, 72/72) is the toy-scale existence proof. No prior
work surfaced on write-once / first-wins in-forward-pass binding — niche
still appears unclaimed. LEADS ONLY — none of these verified against
primary sources yet; verify per protocol before citing in any draft.
Benchmarks to consider for the scale campaign: AgentDojo, PromptGame
(IEEE DataPort 2026-02), PromptBench.
