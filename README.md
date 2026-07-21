# The Samata Latch — Replication Package (Toy Arc)

Code, pre-registered specifications, and write-once results records for the
from-scratch toy experiments behind:

> Baxter, C. R. (2026). *The Samata Latch: Designed Internal State for
> In-Context Binding Integrity.* TICO AI LLC. Zenodo.
> DOI: [10.5281/zenodo.21464368](https://doi.org/10.5281/zenodo.21464368)

## What this is

This package contains two bodies of evidence:

**The constructive toy-scale result (exp01–exp08), fully reproducible here.**
A small transformer-class model, trained from scratch, with a write-once
"latch" that records the first binding and reads it back symbolically. The
full minimal stack (exp08) holds **zero breaches across 72 injected cells (a
fixed 24-cell grid per seed, three seeds), at 4.4–8.8× less training** than
the standard baseline needs to clear the same competence bar — while emitting
a free, label-free, per-example-verifiable record of the executed binding.

**The frontier-battery evidence (exp09), results only.** The per-cell
aggregate statistics and write-once adjudication record behind every
open-weights-model number in the paper (Pythia spine + Qwen3-8B base and
instruct, three framings). The battery itself — harness, item generator,
frozen items, transcripts — is withheld as a held-out benchmark; see
`exp09_scale_campaign/NOTE.md`.

Everything here is **numpy / CPU-friendly**. Each experiment retrains and
re-evaluates from scratch in minutes on a laptop.

The build-up is laid out experiment by experiment: the anchored goal register
(exp01–05), the write-once commitment latch (exp06), the witness/audit channel
(exp07), and the full minimal stack (exp08). Each folder carries its
pre-registered spec (written before results), its write-once results record —
including the **failed predictions and fired kill rules**, kept in on purpose —
and everything needed to reproduce it.

## Layout

    exp01_goal_persistence/    anchored register: ablation vs context growth
    exp02_seed_replication/    seed replication (6/6)
    exp02b_curriculum_control/ curriculum control (kill rule: learnability claim dies)
    exp03_hard_task/           hard-task inversion
    exp05_multihop/            multi-hop scoping (anchor ⊥ retrieval)
    exp06_friction_signal/     write-once latch under override pressure
    exp07_witness_audit/       witness channel; trained alternatives fail audit
    exp08_full_v2_toy/         full minimal stack, 72/72, 3/3 seeds
    exp09_scale_campaign/      frontier battery: results + adjudication ONLY (see NOTE.md)

Each folder: `EXPERIMENT_SPEC.md` (pre-registration), `RESULTS.md` (write-once
record, append-only Corrections), `code/`, `results/`. Read the spec first,
then the results.

## What is deliberately not here

The paper's **inference battery** — the harness used to measure last-wins
capitulation in production-trained open-weights models — is **not** included:
not the harness, not the item generator, not the frozen item packs, not the
raw transcripts. Its measured **results** are here (exp09_scale_campaign/,
aggregate statistics only — no item text); the instrument is not. Releasing
evaluation items causes benchmark contamination (future models train on them),
which destroys the instrument's value — the standard held-out-benchmark
practice. The battery is retained by TICO AI LLC (available under commercial
terms). The toy-scale code and results in this package stand on their own and
require nothing from the battery.

Toy-arc model checkpoints are omitted as trivially regenerable — every run here
reproduces from the released code.

## Running

    pip install -r <exp>/code/requirements.txt

Then run the `np_run*.py` entry point in each folder (flags documented in-file).

## Related work

The six diagnosis-side papers of the Lacuna series (Zenodo, CC BY 4.0) document
the vulnerability classes this program measures and addresses; their DOIs are
listed in the paper and cross-linked from its Zenodo record.

## License

Code (all `.py`, `.sh`, configuration): **MIT** — see `LICENSE`.
Documents (`EXPERIMENT_SPEC.md`, `RESULTS.md`, `RELATED_WORK.md`, this README)
and result data (CSV / JSON / PNG): **CC BY 4.0**, matching the Lacuna paper
series.

## Citation

Please cite the paper (DOI above). A `CITATION.cff` is included.
