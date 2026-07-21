"""
exp07 hop2 battery: vW (witness bottleneck) on exp05's hop2 task, exact
exp05 config (n_layer=3, T=384, K=4, V_ans=8, pair-triple stream,
distinct-candidate redraw). Pre-registered in exp07 EXPERIMENT_SPEC.md
("hop2 battery"). Prediction on record: NO RESCUE (vW chance-level 3/3 —
a witness records a binding; it cannot create retrieval). Contingency:
any cell >= 0.75 at L >= 16 -> headline; vWo runs on hop2 for attribution.

Witness columns reported on hop2 too: on hop2 the answer is not a function
of (g,q) (no memorizable table), so audit validity is computed per example
against the two-hop candidates: y_w = tabs[w][tabs[w][q]]. data_hop2.py is
exp05's file verbatim; candidates are reconstructed here at eval time.

Training mechanism = exp07 Amendment 1 exactly (Gumbel-ST, backward tau=1.0,
load-balance lambda=0.1); eval = deterministic hard argmax.

  python np_run_hop2W.py train --model vW|vWo --seconds 34 --seed N [--cap 1750]
  python np_run_hop2W.py fit_pi --seed N [--arms vW]
  python np_run_hop2W.py eval --model vW [--lens 0,16,...] --seed N
  python np_run_hop2W.py status --seed N
"""
import argparse, csv, json, os, time
import numpy as np_plain
import autograd.numpy as np
from data_hop2 import Vocab, make_batch, sample_tables, _stream
from np_model import (init_params, loss_fn_w, grad_loss_w, adam_step,
                      witness_logits_at)

CFG = dict(K=4, V_ans=8, V_filler=32, n_layer=3, H=2, D=48,
           T=384, train_max_len=64, batch=32, lr=1e-3, task="hop2")
STEP_CAP = 1750
SIZES = [104, 148, 196, 244, 300, 356]          # exp05 trained windows (seq = L+100)
SIZE_P = [0.35, 0.2, 0.15, 0.12, 0.1, 0.08]
SEED = 1; OFF = 10000; CK = "np_ckpt_hop2W_s1"
ARM_OFF = {"vW": 5, "vWo": 6}                   # exp07 offsets, disclosed
TAU = 1.0; LB_LAMBDA = 0.1                      # exp07 Amendment 1

def set_seed(s):
    global SEED, OFF, CK
    SEED = int(s); OFF = 10000 * SEED; CK = f"np_ckpt_hop2W_s{SEED}"
    os.makedirs(CK, exist_ok=True)

def vocab(): return Vocab(K=CFG["K"], V_ans=CFG["V_ans"], V_filler=CFG["V_filler"])

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
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], b, L, bs,
                                      distractors=0.15)
        gum = rng.gumbel(size=(x.shape[0], vv.K))
        g = grad_loss_w(p, x, qpos, y, CFG["n_layer"], CFG["H"], tag,
                        vv.goal_lo, vv.K, TAU, gum, LB_LAMBDA)
        step += 1
        p, m, v = adam_step(p, g, m, v, step, CFG["lr"])
        if step % 10 == 0:
            losses.append(float(loss_fn_w(p, x, qpos, y, CFG["n_layer"],
                                          CFG["H"], tag, vv.goal_lo, vv.K,
                                          TAU, gum, 0.0)))
    _save(tag, p, m, v, step, losses)
    print(json.dumps({"tag": tag, "step": step,
                      "done": step >= cap or _early_stop(losses),
                      "last_loss": round(losses[-1], 4) if losses else None}))

def fit(n):
    for b in SIZES:
        if b >= n: return b
    raise ValueError("too long")

def make_batch_cands(rng, vv, n, L, bs, rate=0.15, balance_goals=True):
    """make_example loop with the K two-hop candidates returned per example.
    Token construction identical to data_hop2.make_example."""
    xs, qs, ys, gs, cs = [], [], [], [], []
    for i in range(n):
        q, tabs = sample_tables(rng, vv.K, vv.V_ans)
        g = (i % vv.K) if balance_goals else int(rng.integers(vv.K))
        a = int(tabs[g][tabs[g][q]])
        cands = [int(t[t[q]]) for t in tabs]
        toks = [vv.GOAL_MARK, vv.goal_tok(g)]
        toks += _stream(rng, vv, tabs, L, rate, g=g)
        toks += [vv.QUERY_MARK, vv.ans_tok(q)]
        pad = bs - len(toks)
        xs.append([vv.PAD] * pad + toks); qs.append(pad + len(toks) - 1)
        ys.append(a); gs.append(g); cs.append(cands)
    return (np_plain.array(xs, dtype=np_plain.int64),
            np_plain.array(qs, dtype=np_plain.int64),
            np_plain.array(ys, dtype=np_plain.int64),
            np_plain.array(gs, dtype=np_plain.int64),
            np_plain.array(cs, dtype=np_plain.int64))

def _pi_path(): return f"results/pi_hop2W_s{SEED}.json"

def _pi_for(arm):
    if os.path.exists(_pi_path()):
        d = json.load(open(_pi_path()))
        if arm in d: return np_plain.array(d[arm]["pi"], dtype=np_plain.int64)
    print(f"WARN: pi not fitted for {arm} — identity used")
    return np_plain.arange(CFG["K"])

def _witness_hard(p, mode, x, qpos, vv):
    lg, lw, _ = witness_logits_at(p, x, qpos, CFG["n_layer"], CFG["H"], mode,
                                  vv.goal_lo, vv.K, 0.1, hard=True)
    return np_plain.asarray(lg).argmax(-1), np_plain.asarray(lw)

def fit_pi(arms=("vW",)):
    """exp07 procedure: best goal->slot permutation on clean L=16 (n=192),
    fit once per arm, frozen. Balanced goals (hop2 eval convention)."""
    from itertools import permutations
    vv = vocab()
    rng = np_plain.random.default_rng(OFF + 21)
    x, qpos, y, goal, cands = make_batch_cands(rng, vv, 192, 16, fit(116))
    out = json.load(open(_pi_path())) if os.path.exists(_pi_path()) else {}
    for arm in arms:
        st = _load(arm)
        if st is None:
            print(f"{arm}: no checkpoint"); continue
        _, lw = _witness_hard(st[0], arm, x, qpos, vv)
        wid = lw.argmax(-1)
        best, best_pi = -1.0, None
        for perm in permutations(range(vv.K)):
            pm = np_plain.array(perm)
            sc = float((pm[wid] == goal).mean())
            if sc > best: best, best_pi = sc, perm
        raw = float((wid == goal).mean())
        out[arm] = {"pi": list(best_pi), "match": best, "raw_match": raw,
                    "identity": list(best_pi) == list(range(vv.K))}
        print(f"{arm}: pi={best_pi} match={best:.3f} raw={raw:.3f}")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(_pi_path(), "w"), indent=1)

def evaluate(tag, lens, eval_n=192):
    """exp05 primary sweep convention (15% distractors, balanced goals) +
    witness audit columns. V/OC vs per-example two-hop candidates."""
    vv = vocab(); st = _load(tag); assert st, f"no checkpoint {tag}"
    p = st[0]
    pi = _pi_for(tag)
    rng = np_plain.random.default_rng(OFF + 999)
    rows = []
    for L in lens:
        bs = fit(L + 100)
        x, qpos, y, goal, cands = make_batch_cands(rng, vv, eval_n, L, bs)
        pred, lw = _witness_hard(p, tag, x, qpos, vv)
        w = pi[lw.argmax(-1)]
        idx = np_plain.arange(len(y))
        y_w = cands[idx, w]                      # rule(w, q) per example
        y_g = cands[idx, goal]                   # == y by construction
        acc = float((pred == y).mean())
        Wg_raw = float((lw.argmax(-1) == goal).mean())
        Wg = float((w == goal).mean())
        V = float((pred == y_w).mean())
        OC = float(((w == goal) & (pred != y_g)).mean())
        rows.append([SEED, L, tag, acc, Wg_raw, Wg, V, OC,
                     1.0 / vv.V_ans, 0.25])
        print(f"L={L:4d} {tag}={acc:.3f} Wg_raw={Wg_raw:.3f} Wg_pi={Wg:.3f} "
              f"V={V:.3f} OC={OC:.3f}")
    path = f"results/np_sweep_hop2W_s{SEED}.csv"
    os.makedirs("results", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["seed", "L", "arm", "acc", "Wg_raw", "Wg_pi",
                            "V", "OC", "chance", "goal_ignorant_ceiling"])
        for r in rows: w.writerow(r)

def status():
    for tag in ("vW", "vWo"):
        st = _load(tag)
        if st: print(tag, "step", st[3], "loss", round(st[4][-1], 4) if st[4] else None)
        else: print(tag, "not started")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "fit_pi", "eval", "status"])
    ap.add_argument("--model", default="vW")
    ap.add_argument("--seconds", type=float, default=34)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cap", type=int, default=STEP_CAP)
    ap.add_argument("--lens", type=str, default="")
    ap.add_argument("--arms", type=str, default="vW")
    a = ap.parse_args()
    set_seed(a.seed)
    lens = [int(x) for x in a.lens.split(",")] if a.lens else [0, 16, 32, 64, 96, 128, 192, 224, 256]
    if a.cmd == "train": train(a.model, a.seconds, a.cap)
    elif a.cmd == "fit_pi": fit_pi(tuple(a.arms.split(",")))
    elif a.cmd == "eval": evaluate(a.model, lens)
    else: status()
