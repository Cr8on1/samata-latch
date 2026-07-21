# exp09 — Frontier Battery: Results Only

This folder holds the **measured results and adjudication** for the paper's
inference battery on production-trained open-weights models — the evidence
behind every frontier-model number in the paper. It intentionally does **not**
contain the battery itself.

## Included here
- `results/*.csv` — per-cell aggregate statistics for all runs: six models
  (Pythia 410m / 1.4b / 2.8b / 6.9b spine; Qwen3-8B base and instruct), three
  framings (F1/F2/F3), across the pre-registered (L, φ, m) grid. Every column
  is a score or a configuration value (`R, C, X, delta, n, V, OC, Rw, Cw, dw`).
  **No item text appears in these files.**
- `RESULTS.md` — the write-once adjudication record: what was measured, the
  verdicts, the failed predictions, and the corrections.

## Deliberately withheld (held-out benchmark)
- the **item generator**, the **frozen item packs**, and the raw `.eval`
  **transcripts** — releasing evaluation items causes benchmark contamination
  (future models train on them), which destroys the instrument's value;
- the **battery harness** and the generator seed / item-keying scheme;
- the pre-registered **item-construction detail** — the method is described in
  the paper; the runnable recipe is not published here.

The numbers in `results/` stand on their own, the method is in the paper
(pre-registered), and the item bank stays private — standard practice for a
held-out benchmark. The battery is retained by TICO AI LLC and available for
evaluation under commercial terms.
