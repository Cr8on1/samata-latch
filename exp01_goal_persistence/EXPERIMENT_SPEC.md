# LACUNA V2 — Experiment #5: Goal Persistence Under Context Growth

A controlled ablation of **Primitive A** (the anchored goal channel). This is the
recommended first build: value-neutral, directly measurable, and it validates the
goal-persistence primitive the later toy arc builds on. [Public release note:
program-roadmap framing redacted; see PACKAGE_MANIFEST.]

## The claim under test

The requirement under test: *hold the assigned goal stable as context
accumulates.* A standard transformer reconstructs the goal from context on every
forward pass; as goal-irrelevant context grows, causal attention disperses and the
goal's influence at the query position decays. An **anchored goal channel** — a
persistent register re-injected into the residual stream at every layer, outside the
attention competition — should hold flat.

## Task (synthetic, fully controlled)

```
[GOAL] g   filler_1 ... filler_L   [QUERY] q   ->   a = rule(g, q)
```

* `g` selects a rule; `q` is the operand; the model predicts `a` at the query position.
* `--task shift`: `a = (q + g) mod V`  — goal is used **only at the end**.
* `--task hard` : `a = ((q + g)*(g+1)) mod V` — goal **gates intermediate computation**.
* `L` = filler length = goal→query distance. Sweeping `L` is the independent variable.

## The four arms (identical task, identical metric)

| arm | what it is | weight change |
|---|---|---|
| **v1** | standard transformer, goal only as a start token | trained |
| **v1 + reminder** | surface fix: re-insert `[GOAL] g` every `R` tokens | none (prompt only) |
| **v1 + monitor** | external wrapper: frozen v1 + bolt-on head given the goal (oracle) at the output | tiny head only |
| **v2 (anchor)** | internal anchored goal register, re-injected every layer | trained |

v1 and v2 are the **same class**, identical in size/data/training except the `anchor`
flag. That single structural difference is the ablation.

**Metric:** goal-conditioned accuracy vs. `L`. Chance = `1 / V_ans`.

## Pre-registered expectations (write these down before running)

Per the program's **discipline rule** — a structural fix only counts where a prompt
*and* a wrapper cannot match it — here is where each arm is expected to win or wall:

1. **v1** decays toward chance as `L` grows, and collapses past the training length.
2. **v1 + reminder** decays more slowly but still degrades *between* reminders and
   out of training range; it also spends context budget that scales with `L`.
3. **v1 + monitor** on `task=shift` is expected to be **competitive with v2** — and
   that is a legitimate, publishable finding: *for end-only goal use, an external
   wrapper suffices and you do not need architecture.* The honest result is the wall,
   not a foregone v2 win.
4. **The wall** is expected on `task=hard`: the goal must modulate *intermediate*
   computation, which an output-only wrapper cannot reach and a between-reminders gap
   cannot cover. v2 should stay flat while reminder/monitor fall off. That gap — if it
   appears and survives the controls — is the structural result.

If a reminder or wrapper fully matches v2 on `hard` too, that is a real finding that
Primitive A is not needed for this requirement. The experiment is built to be able to
lose.

## A caveat worth keeping honest

The framing "transformers structurally instantiate none of the five" overstates the
gap: model **weights** already are a persistent, context-resistant store of the goal
distribution. What they lack is an *addressable, per-instance* goal slot that is
(a) not reconstructed from context and (b) monitored for divergence. This experiment
tests exactly that narrower, defensible claim — not the strong one.

## Run it

```bash
pip install -r requirements.txt

python run.py smoke                 # ~30s CPU sanity run, end-to-end
python run.py all --task shift      # train v1+monitor+v2, sweep, plot
python run.py all --task hard       # the variant where the wall should appear
python smoke_test.py                # torch-free unit tests of the data logic
```

Outputs: `results/sweep_<task>.csv` and `results/sweep_<task>.png` (accuracy vs L,
with chance line and training-length marker).

### Sizing for a Legion 5 Pro (RTX 3070 Ti, 8 GB)

Defaults (`n_layer=4, n_embd=128, block_size=768`) are ~1–3 M params and train in
minutes on the 3070 Ti. Scale with `--n_layer --n_embd --steps --batch`. Keep
`block_size` above your largest `eval_lens` value. Add `--distractors` to put
plausible goal-like tokens in the filler stream (a harder, more realistic drift test).

## Files

* `data.py` — synthetic task, batch builders, reminder builder (pure numpy).
* `model.py` — nanoGPT + anchored goal channel (`anchor` flag) + external monitor.
* `run.py` — train / eval / all / smoke, CSV + plot.
* `smoke_test.py` — torch-free verification of the data/rule/reminder logic.

## Next primitives (after this validates)

* **B — pre-rational friction signal** (covers O): a divergence measure between the
  current semantic trajectory and the anchored reference that biases toward hold before
  the reasoning pass completes. See `Ontology_of_Friction`.
* **C — witness / meta-monitor head** (covers C): a lightweight head trained on a
  drift-detection objective (not next-token) that reads hidden states and gates output.
* **Corrigibility (§4b):** deliberately out of scope here. At this scale, use
  institutional enforcement (you control the update path). The privileged-update-pathway
  mechanism is an open problem and should not block this experiment.
