"""
exp06 hop2 battery: vB (latch+gate) on exp05's hop2 task, exact exp05 config
(n_layer=3, T=384, K=4, V_ans=8, pair-triple stream, distinct-candidate
redraw). Pre-registered in exp06 EXPERIMENT_SPEC.md ("hop2 battery").
Prediction on record: no rescue (vB chance-level like exp05's five arms).
Contingency: any cell >= 0.75 at L >= 16 -> headline; vB-latch (vBl) runs
for attribution.

Files: data_hop2.py = exp05 data.py verbatim; np_model.py = exp06 model
(mode system: latch reads first [GOAL] marker payload from the token
stream — no oracle channel; gate damps rival-goal-claiming tokens).

  python np_run_hop2B.py train --model vB|vBl --seconds 34 --seed N [--cap 1750]
  python np_run_hop2B.py eval --model vB|vBl [--lens 0,16,...] --seed N
  python np_run_hop2B.py status --seed N
"""
import argparse, csv, json, os, time
import numpy as np_plain
from data_hop2 import Vocab, make_batch
from np_model import init_params, loss_fn, grad_loss, adam_step, logits_at

CFG = dict(K=4, V_ans=8, V_filler=32, n_layer=3, H=2, D=48,
           T=384, train_max_len=64, batch=32, lr=1e-3, task="hop2")
STEP_CAP = 1750
SIZES = [104, 148, 196, 244, 300, 356]          # exp05 trained windows (seq = L+100)
SIZE_P = [0.35, 0.2, 0.15, 0.12, 0.1, 0.08]
SEED = 1; OFF = 10000; CK = "np_ckpt_hop2B_s1"
ARM_OFF = {"vB": 2, "vBl": 3}                   # exp06 offsets, disclosed

def set_seed(s):
    global SEED, OFF, CK
    SEED = int(s); OFF = 10000 * SEED; CK = f"np_ckpt_hop2B_s{SEED}"
    os.makedirs(CK, exist_ok=True)

def vocab(): return Vocab(K=CFG["K"], V_ans=CFG["V_ans"], V_filler=CFG["V_filler"])

def apos_of(x, PAD=2):
    return (x != PAD).argmax(1)

def _save(tag, p, m, v, step, losses):
    np_plain.savez(f"{CK}/{tag}.npz", step=step, losses=np_plain.array(losses),
                   **{"p_"+k: p[k] for k in p}, **{"m_"+k: m[k] for k in m},
                   **{"v_"+k: v[k] for k in v})

def _load(tag):
    f = f"{CK}/{tag}.npz"
    if not os.path.exists(f): return None
    z = np_plain.load(f)
    return ({k[2:]: z[k] for k in z.files if k.startswith("p_")},
            {k[2:]: z[k] for k in z.files if k.startswith("m_")},
            {k[2:]: z[k] for k in z.files if k.startswith("v_")},
            int(z["step"]), list(z["losses"]))

def _early_stop(losses):
    return len(losses) >= 3 and all(l < 0.03 for l in losses[-3:])

def train(tag, seconds, cap):
    vv = vocab(); ao = ARM_OFF[tag]
    st = _load(tag)
    if st is None:
        p = init_params(np_plain.random.default_rng(OFF + 42 + ao), vv.size,
                        vv.V_ans, vv.K, CFG["T"], CFG["n_layer"], CFG["H"],
                        CFG["D"], tag)
        m = {k: np_plain.zeros_like(p[k]) for k in p}
        v = {k: np_plain.zeros_like(p[k]) for k in p}
        step, losses = 0, []
        rng = np_plain.random.default_rng(OFF + ao)
    else:
        p, m, v, step, losses = st
        rng = np_plain.random.default_rng(OFF + 1000 + step + ao)
    if step >= cap or _early_stop(losses):
        print(json.dumps({"tag": tag, "step": step, "done": True,
                          "last_loss": round(losses[-1], 4) if losses else None}))
        return
    t0 = time.time()
    while time.time() - t0 < seconds and step < cap and not _early_stop(losses):
        L = int(rng.integers(0, CFG["train_max_len"] + 1))
        bs = max(int(rng.choice(SIZES, p=SIZE_P)), L + 100)   # exp05 _pick_bs
        b = max(8, int(CFG["batch"] * SIZES[0] / bs))
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], b, L, bs, distractors=0.15)
        ap = apos_of(x, vv.PAD)
        g = grad_loss(p, x, qpos, y, goal, ap, CFG["n_layer"], CFG["H"],
                      tag, vv.goal_lo, vv.K)
        step += 1
        p, m, v = adam_step(p, g, m, v, step, CFG["lr"])
        if step % 10 == 0:
            losses.append(float(loss_fn(p, x, qpos, y, goal, ap,
                                        CFG["n_layer"], CFG["H"], tag,
                                        vv.goal_lo, vv.K)))
    _save(tag, p, m, v, step, losses)
    print(json.dumps({"tag": tag, "step": step,
                      "done": step >= cap or _early_stop(losses),
                      "last_loss": round(losses[-1], 4) if losses else None}))

def fit(n):
    for b in SIZES:
        if b >= n: return b
    raise ValueError("too long")

def _acc(lg, y): return float((np_plain.asarray(lg).argmax(-1) == y).mean())

def evaluate(tag, lens, eval_n=192):
    """exp05 primary sweep convention: 15% distractors, balanced goals."""
    vv = vocab(); p = _load(tag)[0]
    rng = np_plain.random.default_rng(OFF + 999)
    rows = []
    for L in lens:
        bs = fit(L + 100)
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], eval_n, L, bs,
                                      distractors=0.15, balance_goals=True)
        ap = apos_of(x, vv.PAD)
        lg, _ = logits_at(p, x, qpos, None, ap, CFG["n_layer"], CFG["H"],
                          tag, vv.goal_lo, vv.K)
        acc = _acc(lg, y)
        rows.append([SEED, L, tag, acc, 1.0 / vv.V_ans, 0.25])
        print(f"L={L:4d} {tag}={acc:.3f}")
    path = f"results/np_sweep_hop2B_s{SEED}.csv"
    os.makedirs("results", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["seed", "L", "arm", "acc", "chance", "goal_ignorant_ceiling"])
        for r in rows: w.writerow(r)

def status():
    for tag in ("vB", "vBl"):
        st = _load(tag)
        if st: print(tag, "step", st[3], "loss", round(st[4][-1], 4) if st[4] else None)
        else: print(tag, "not started")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "status"])
    ap.add_argument("--model", default="vB")
    ap.add_argument("--seconds", type=float, default=34)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cap", type=int, default=STEP_CAP)
    ap.add_argument("--lens", type=str, default="")
    a = ap.parse_args()
    set_seed(a.seed)
    lens = [int(x) for x in a.lens.split(",")] if a.lens else [0, 16, 32, 64, 96, 128, 192, 224, 256]
    if a.cmd == "train": train(a.model, a.seconds, a.cap)
    elif a.cmd == "eval": evaluate(a.model, lens)
    else: status()
