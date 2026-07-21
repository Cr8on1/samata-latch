"""
exp08 chunked CPU runner (34s-burst discipline, checkpoint .npz).

  python np_run.py train --model vS|vSo|vSc|vSoc --seconds 34 --seed N [--cap 1750]
  python np_run.py eval        [--lens 0,16,32] [--dist 15|0] --seed N
  python np_run.py eval_inject [--lens 32,64] [--phis 0.25,end] [--arms vS,vSo] --seed N
  python np_run.py eval_degrade --seed N
  python np_run.py bytecheck --seed N --other <exp06 vBl.npz path>
  python np_run.py status --seed N

Audit columns (latch readout, zero parameters, NO pi anywhere — the record
is a token identity, exp08 spec): w_hat = token at apos+1.
  Wg = P(w_hat = g)   [analytic 1.0 — computed anyway, verifies the identity]
  V  = P(pred = rule(w_hat, q))  [label-free, per-example checkable]
  OC = P(w_hat = g AND pred != rule(g, q))
  Injected cells add Rw = P(w_hat=g), Cw = P(w_hat=g'), dw = Rw - Cw.

vS_degraded snapshot: at the first 10-step loss check with running loss in
[0.45, 0.70] (pre-registered band), the vS state is saved once per seed.
"""
import argparse, csv, json, os, time
import numpy as np_plain
import autograd.numpy as np
from data import Vocab, make_batch, make_batch_injected, rule_answer
from np_model import init_params, loss_fn, grad_loss, adam_step, logits_at, latch_ids

CFG = dict(K=4, V_ans=8, V_filler=32, n_layer=2, H=2, D=48,
           T=288, train_max_len=64, batch=32, lr=1e-3, task="hard")
STEP_CAP = 1750
SEED = 1; OFF = 10000; CK = "np_ckpt_hard_s1"
# vS gets exp06 vBl's offset (3) so init + fresh-start data stream are
# bit-identical to exp06 (H-byte-replication check, disclosed in spec).
ARM_OFF = {"vS": 3, "vSo": 5, "vSc": 6, "vSoc": 7}
CURR = dict(ramp_steps=1000, d_max=0.15)   # exp02b recipe (contingency arms)
DEG_BAND = (0.45, 0.70)                    # pre-registered snapshot band

def set_seed(s):
    global SEED, OFF, CK
    SEED = int(s); OFF = 10000 * SEED; CK = f"np_ckpt_{CFG['task']}_s{SEED}"
    os.makedirs(CK, exist_ok=True)

def vocab(): return Vocab(K=CFG["K"], V_ans=CFG["V_ans"], V_filler=CFG["V_filler"])

def apos_of(x, PAD=2):
    return (x != PAD).argmax(1)     # first non-pad = assignment marker

def _save(tag, p, m, v, step, losses):
    np_plain.savez(f"{CK}/{tag}.npz",
                   step=step, losses=np_plain.array(losses),
                   **{"p_"+k: p[k] for k in p},
                   **{"m_"+k: m[k] for k in m},
                   **{"v_"+k: v[k] for k in v})

def _load(tag):
    f = f"{CK}/{tag}.npz"
    if not os.path.exists(f): return None
    z = np_plain.load(f)
    p = {k[2:]: z[k] for k in z.files if k.startswith("p_")}
    m = {k[2:]: z[k] for k in z.files if k.startswith("m_")}
    v = {k[2:]: z[k] for k in z.files if k.startswith("v_")}
    return p, m, v, int(z["step"]), list(z["losses"])

def _early_stop(losses):
    return len(losses) >= 3 and all(l < 0.03 for l in losses[-3:])

def train(tag, seconds, cap):
    mode = {"vS": "vS", "vSc": "vS", "vSo": "vSo", "vSoc": "vSo"}[tag]
    curr = tag in ("vSc", "vSoc")
    vv = vocab()
    ao = ARM_OFF[tag]
    st = _load(tag)
    if st is None:
        p = init_params(np_plain.random.default_rng(OFF + 42 + ao), vv.size,
                        vv.V_ans, vv.K, CFG["T"], CFG["n_layer"], CFG["H"],
                        CFG["D"], mode)
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
        bs = int(rng.choice([68, 112, 160, 208, 256, 288],
                            p=[0.35, 0.2, 0.15, 0.12, 0.1, 0.08]))
        bs = max(bs, L + 4)
        b = max(8, int(CFG["batch"] * 68 / bs))
        dr = (CURR["d_max"] * min(1.0, step / CURR["ramp_steps"])) if curr else True
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], b, L, bs, distractors=dr)
        ap = apos_of(x, vv.PAD)
        g = grad_loss(p, x, qpos, y, goal, ap, CFG["n_layer"], CFG["H"],
                      mode, vv.goal_lo, vv.K)
        step += 1
        p, m, v = adam_step(p, g, m, v, step, CFG["lr"])
        if step % 10 == 0:
            l = float(loss_fn(p, x, qpos, y, goal, ap, CFG["n_layer"], CFG["H"],
                              mode, vv.goal_lo, vv.K))
            losses.append(l)
            if (tag == "vS" and DEG_BAND[0] <= l <= DEG_BAND[1]
                    and not os.path.exists(f"{CK}/vS_degraded.npz")):
                _save("vS_degraded", p, m, v, step, losses)
                print(json.dumps({"snapshot": "vS_degraded", "step": step,
                                  "loss": round(l, 4)}))
    _save(tag, p, m, v, step, losses)
    print(json.dumps({"tag": tag, "step": step, "done": step >= cap or _early_stop(losses),
                      "last_loss": round(losses[-1], 4) if losses else None}))

def _acc(lg, y): return float((np_plain.asarray(lg).argmax(-1) == y).mean())

SIZES = [68, 112, 160, 208, 256, 288]
def fit(n):
    for b in SIZES:
        if b >= n: return b
    raise ValueError("too long")

def _append_csv(path, header, rows):
    os.makedirs("results", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        for r in rows: w.writerow(r)

def _audit(pred, x, qpos, ap, goal, y, vv):
    """Latch-readout audit statistics. Returns Wg, V, OC (+ w_hat)."""
    idx = np_plain.arange(len(y))
    what = np_plain.asarray(x)[idx, np_plain.asarray(ap) + 1] - vv.goal_lo
    q = np_plain.asarray(x)[idx, qpos] - vv.ans_lo
    y_w = rule_answer(CFG["task"], what, q, vv.V_ans)
    Wg = float((what == goal).mean())
    V = float((pred == y_w).mean())
    OC = float(((what == goal) & (pred != y)).mean())
    return Wg, V, OC, what

def _arms_present(requested=None):
    arms = [a for a in ("vS", "vSo", "vSc", "vSoc") if _load(a)]
    if requested:
        arms = [a for a in arms if a in requested]
    return arms

MODE_OF = {"vS": "vS", "vSc": "vS", "vSo": "vSo", "vSoc": "vSo",
           "vS_degraded": "vS"}

def evaluate(eval_n=192, lens=(0, 16, 32, 64, 96, 128, 192, 224, 256), dist=15):
    vv = vocab()
    arms = _arms_present()
    ps = {a: _load(a)[0] for a in arms}
    rng = np_plain.random.default_rng(OFF + 999 + (0 if dist == 15 else 1))
    d = dist == 15
    rows = []
    hdr_arms = ["vS", "vSo", "vSc", "vSoc"]
    for L in lens:
        bs = fit(L + 4)
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], eval_n, L, bs, distractors=d)
        ap = apos_of(x, vv.PAD)
        row = [SEED, L]
        msg = f"L={L:4d}"
        cols = {}
        for a in hdr_arms:
            if a in ps:
                lg, _ = logits_at(ps[a], x, qpos, None, ap, CFG["n_layer"],
                                  CFG["H"], MODE_OF[a], vv.goal_lo, vv.K)
                pred = np_plain.asarray(lg).argmax(-1)
                acc = float((pred == y).mean())
                Wg, V, OC, _ = _audit(pred, x, qpos, ap, goal, y, vv)
                cols[a] = (acc, Wg, V, OC)
                msg += f" {a}={acc:.3f} V={V:.3f} OC={OC:.3f}"
            else:
                cols[a] = (float("nan"),) * 4
        for a in hdr_arms: row += list(cols[a])
        rows.append(row + [1.0 / vv.V_ans])
        print(msg)
    suf = "" if dist == 15 else "_diag"
    hdr = ["seed", "L"]
    for a in hdr_arms: hdr += [a, f"Wg_{a}", f"V_{a}", f"OC_{a}"]
    _append_csv(f"results/np_sweep_{CFG['task']}_s{SEED}{suf}.csv",
                hdr + ["chance"], rows)

def _rcx(pred, y, y_inj):
    R = float((pred == y).mean()); C = float((pred == y_inj).mean())
    return R, C, 1.0 - R - C

def eval_inject(eval_n=192, lens=(32, 64, 128, 192), phis=(0.25, 0.75, "end"),
                ms=(1, 4), arms=None):
    vv = vocab()
    arms = _arms_present(arms)
    ps = {a: _load(a)[0] for a in arms}
    rng = np_plain.random.default_rng(OFF + 5555)
    rows, pg_rows = [], []
    for L in lens:
        for phi in phis:
            for m in ms:
                bs = fit(L + 4)
                x, qpos, y, goal, ap, y_inj, ginj = make_batch_injected(
                    rng, vv, CFG["task"], eval_n, L, bs, phi, m, distractors=True)
                for a in arms:
                    lg, _ = logits_at(ps[a], x, qpos, None, ap, CFG["n_layer"],
                                      CFG["H"], MODE_OF[a], vv.goal_lo, vv.K)
                    pred = np_plain.asarray(lg).argmax(-1)
                    R, C, X = _rcx(pred, y, y_inj)
                    Wg, V, OC, what = _audit(pred, x, qpos, ap, goal, y, vv)
                    Rw = float((what == goal).mean())
                    Cw = float((what == ginj).mean())
                    rows.append([SEED, a, L, phi, m, R, C, X, R - C,
                                 Wg, V, OC, Rw, Cw, Rw - Cw, eval_n])
                    for g in range(CFG["K"]):
                        gi = goal == g
                        if gi.sum() == 0: continue
                        pg_rows.append([SEED, a, L, phi, m, g,
                                        float((pred[gi] == y[gi]).mean()),
                                        float((pred[gi] == y_inj[gi]).mean()),
                                        int(gi.sum())])
                print(f"cell L={L} phi={phi} m={m} done ({','.join(arms)})")
    _append_csv(f"results/np_inject_s{SEED}.csv",
                ["seed", "arm", "L", "phi", "m", "R", "C", "X", "delta",
                 "Wg", "V", "OC", "Rw", "Cw", "dw", "n"], rows)
    _append_csv(f"results/np_inject_pergoal_s{SEED}.csv",
                ["seed", "arm", "L", "phi", "m", "g", "R", "C", "n"], pg_rows)

def eval_degrade(eval_n=192):
    """Pre-registered battery: clean L in {16,64,192} + injected
    L in {64,192} x phi in {0.25,end} x m=4, arm vS_degraded."""
    vv = vocab()
    st = _load("vS_degraded")
    if st is None:
        print("no vS_degraded snapshot (band never hit — disclose)"); return
    p = st[0]
    rng = np_plain.random.default_rng(OFF + 4242)
    rows = []
    for L in (16, 64, 192):
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], eval_n, L,
                                      fit(L + 4), distractors=True)
        ap = apos_of(x, vv.PAD)
        lg, _ = logits_at(p, x, qpos, None, ap, CFG["n_layer"], CFG["H"],
                          "vS", vv.goal_lo, vv.K)
        pred = np_plain.asarray(lg).argmax(-1)
        acc = float((pred == y).mean())
        Wg, V, OC, _ = _audit(pred, x, qpos, ap, goal, y, vv)
        rows.append([SEED, "vS_degraded", "clean", L, "", "", acc, "",
                     Wg, V, OC, eval_n])
        print(f"clean L={L}: acc={acc:.3f} V={V:.3f} OC={OC:.3f}")
    for L in (64, 192):
        for phi in (0.25, "end"):
            x, qpos, y, goal, ap, y_inj, ginj = make_batch_injected(
                rng, vv, CFG["task"], eval_n, L, fit(L + 4), phi, 4,
                distractors=True)
            lg, _ = logits_at(p, x, qpos, None, ap, CFG["n_layer"], CFG["H"],
                              "vS", vv.goal_lo, vv.K)
            pred = np_plain.asarray(lg).argmax(-1)
            R, C, X = _rcx(pred, y, y_inj)
            Wg, V, OC, _ = _audit(pred, x, qpos, ap, goal, y, vv)
            rows.append([SEED, "vS_degraded", "inject", L, phi, 4, R, C,
                         Wg, V, OC, eval_n])
            print(f"inject L={L} phi={phi}: R={R:.3f} C={C:.3f} V={V:.3f} OC={OC:.3f}")
    _append_csv(f"results/np_degrade_s{SEED}.csv",
                ["seed", "arm", "kind", "L", "phi", "m", "R", "C",
                 "Wg", "V", "OC", "n"], rows)

def bytecheck(other):
    """H-byte-replication: sha256 of param arrays vs exp06 vBl checkpoint."""
    import hashlib
    st = _load("vS")
    if st is None: print("no vS checkpoint"); return
    z = np_plain.load(other)
    p6 = {k[2:]: z[k] for k in z.files if k.startswith("p_")}
    p8 = st[0]
    same_keys = set(p6) == set(p8)
    n_eq = sum(1 for k in p8 if k in p6 and np_plain.array_equal(p8[k], p6[k]))
    def h(d): return hashlib.sha256(
        b"".join(np_plain.ascontiguousarray(d[k]).tobytes() for k in sorted(d))).hexdigest()
    print(json.dumps({"keys_match": same_keys, "arrays_equal": f"{n_eq}/{len(p8)}",
                      "sha_vS": h(p8), "sha_other": h(p6),
                      "identical": same_keys and n_eq == len(p8),
                      "step_vS": st[3], "step_other": int(z["step"])}))

def status():
    for tag in ["vS", "vSo", "vSc", "vSoc", "vS_degraded"]:
        st = _load(tag)
        if st: print(tag, "step", st[3], "loss", round(st[4][-1], 4) if st[4] else None)
        else: print(tag, "not started")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "eval_inject",
                                    "eval_degrade", "bytecheck", "status"])
    ap.add_argument("--model", default="vS")
    ap.add_argument("--seconds", type=float, default=34)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--task", default="hard", choices=["shift", "hard"])
    ap.add_argument("--cap", type=int, default=STEP_CAP)
    ap.add_argument("--dist", type=int, default=15, choices=[0, 15])
    ap.add_argument("--lens", type=str, default="")
    ap.add_argument("--phis", type=str, default="")
    ap.add_argument("--arms", type=str, default="")
    ap.add_argument("--other", type=str, default="")
    a = ap.parse_args()
    CFG["task"] = a.task; set_seed(a.seed)
    lens = tuple(int(x) for x in a.lens.split(",")) if a.lens else None
    if a.cmd == "train": train(a.model, a.seconds, a.cap)
    elif a.cmd == "eval": evaluate(lens=lens or (0, 16, 32, 64, 96, 128, 192, 224, 256), dist=a.dist)
    elif a.cmd == "eval_inject":
        phis = tuple(float(x) if x != "end" else "end" for x in a.phis.split(",")) if a.phis else (0.25, 0.75, "end")
        eval_inject(lens=lens or (32, 64, 128, 192), phis=phis,
                    arms=tuple(a.arms.split(",")) if a.arms else None)
    elif a.cmd == "eval_degrade": eval_degrade()
    elif a.cmd == "bytecheck": bytecheck(a.other)
    else: status()
