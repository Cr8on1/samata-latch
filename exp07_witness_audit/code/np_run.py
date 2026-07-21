"""
exp07 chunked CPU runner (34s-burst discipline, checkpoint .npz).

  python np_run.py train --model vW|vWo|vWc --seconds 34 --seed N [--cap 1750]
  python np_run.py fit_pi --seed N [--arms vW,vWo,vW_degraded]
  python np_run.py eval        [--lens 0,16,32] [--dist 15|0] --seed N
  python np_run.py eval_inject [--lens 32,64] [--phis 0.25,0.75,end] --seed N
  python np_run.py eval_degrade --seed N [--cells clean|inject]
  python np_run.py status --seed N

Reused exp06 arms (v1, vB, mon, probe_v1, *_cap1750) are byte-verified
checkpoint COPIES placed in this experiment's CK dir — eval-only, never
retrained here.
"""
import argparse, csv, json, os, time
import numpy as np_plain
import autograd.numpy as np
from data import Vocab, make_batch, make_batch_injected, rule_answer
from np_model import (init_params, loss_fn, grad_loss, loss_fn_w, grad_loss_w,
                      adam_step, logits_at, witness_logits_at,
                      monitor_logits, probe_logits,
                      init_retro, retro_logits, loss_fn_retro, grad_retro)

CFG = dict(K=4, V_ans=8, V_filler=32, n_layer=2, H=2, D=48,
           T=288, train_max_len=64, batch=32, lr=1e-3, task="hard")
STEP_CAP = 1750
SEED = 1; OFF = 10000; CK = "np_ckpt_hard_s1"
ARM_OFF = {"vW": 5, "vWo": 6, "vWc": 7, "vWr": 8}    # disjoint from exp06's 0-4
CURR = dict(ramp_steps=1000, d_max=0.15)   # exp02b recipe (vWc contingency)
DEGRADE_LOSS = 0.6                         # pre-registered snapshot trigger

def set_seed(s):
    global SEED, OFF, CK
    SEED = int(s); OFF = 10000 * SEED; CK = f"np_ckpt_{CFG['task']}_s{SEED}"
    os.makedirs(CK, exist_ok=True)

def vocab(): return Vocab(K=CFG["K"], V_ans=CFG["V_ans"], V_filler=CFG["V_filler"])

def apos_of(x, PAD=2):
    return (x != PAD).argmax(1)

def tau_of(step):
    # Amendment 1: Gumbel straight-through commitment; backward softmax at
    # fixed tau=1.0 (the pre-registered 1.0->0.1 anneal saturated the witness
    # gradient in smoke; plain ST collapsed to a single slot).
    return 1.0

LB_LAMBDA = 0.1   # Amendment 1: load-balance KL(batch-mean soft || uniform)

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
    mode = "vW" if tag == "vWc" else tag
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
        dr = (CURR["d_max"] * min(1.0, step / CURR["ramp_steps"])) if tag == "vWc" else True
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], b, L, bs, distractors=dr)
        gum = rng.gumbel(size=(x.shape[0], vv.K))
        g = grad_loss_w(p, x, qpos, y, CFG["n_layer"], CFG["H"], mode,
                        vv.goal_lo, vv.K, tau_of(step), gum, LB_LAMBDA)
        step += 1
        p, m, v = adam_step(p, g, m, v, step, CFG["lr"])
        if step % 10 == 0:
            l = float(loss_fn_w(p, x, qpos, y, CFG["n_layer"], CFG["H"], mode,
                                vv.goal_lo, vv.K, tau_of(step), gum, 0.0))
            losses.append(l)
            if (tag == "vW" and l <= DEGRADE_LOSS
                    and not os.path.exists(f"{CK}/vW_degraded.npz")):
                _save("vW_degraded", p, m, v, step, losses)
                print(json.dumps({"snapshot": "vW_degraded", "step": step,
                                  "loss": round(l, 4)}))
    _save(tag, p, m, v, step, losses)
    print(json.dumps({"tag": tag, "step": step, "done": step >= cap or _early_stop(losses),
                      "last_loss": round(losses[-1], 4) if losses else None,
                      "tau": round(tau_of(step), 3)}))

def _frozen_feats(p1, x, qpos, ap, vv):
    _, hq = logits_at(p1, x, qpos, None, ap, CFG["n_layer"], CFG["H"],
                      "v1", vv.goal_lo, vv.K)
    hq = np_plain.asarray(hq)
    eq = np_plain.asarray(p1["tok"])[np_plain.asarray(x)[np_plain.arange(len(hq)), qpos]]
    Eg = np_plain.asarray(p1["tok"])[vv.goal_lo:vv.goal_lo + vv.K]
    return hq, eq, Eg

def train_retro(seconds, feat="v1", cap=STEP_CAP):
    """Amendment 2: witness bottleneck heads on FROZEN backbone features."""
    vv = vocab()
    st = _load(feat); assert st, f"need frozen backbone ckpt {feat} in {CK}"
    p1 = st[0]
    tag = "vWr" if feat == "v1" else f"vWr_{feat}"
    ao = ARM_OFF["vWr"]
    str_ = _load(tag)
    if str_ is None:
        rp = init_retro(np_plain.random.default_rng(OFF + 42 + ao), CFG["D"],
                        vv.K, vv.V_ans)
        m = {k: np_plain.zeros_like(rp[k]) for k in rp}
        v = {k: np_plain.zeros_like(rp[k]) for k in rp}
        step, losses = 0, []
        rng = np_plain.random.default_rng(OFF + ao)
    else:
        rp, m, v, step, losses = str_
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
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], b, L, bs, distractors=True)
        hq, eq, Eg = _frozen_feats(p1, x, qpos, apos_of(x, vv.PAD), vv)
        gum = rng.gumbel(size=(x.shape[0], vv.K))
        g = grad_retro(rp, hq, eq, Eg, y, tau_of(step), gum, LB_LAMBDA)
        step += 1
        rp, m, v = adam_step(rp, g, m, v, step, CFG["lr"])
        if step % 10 == 0:
            losses.append(float(loss_fn_retro(rp, hq, eq, Eg, y, tau_of(step), gum, 0.0)))
    _save(tag, rp, m, v, step, losses)
    print(json.dumps({"tag": tag, "step": step,
                      "done": step >= cap or _early_stop(losses),
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

# ---- witness-audit machinery ----
def _pi_path(): return f"results/pi_s{SEED}.json"

def _pi_for(arm):
    """Fitted goal-dictionary permutation for vW arms; identity for symbolic
    (latch) and label-trained (probe) readouts."""
    if arm.startswith("vW"):
        if os.path.exists(_pi_path()):
            d = json.load(open(_pi_path()))
            if arm in d: return np_plain.array(d[arm]["pi"], dtype=np_plain.int64)
        print(f"WARN: pi not fitted for {arm} — identity used")
    return np_plain.arange(CFG["K"])

def _witness_hard(p, mode, x, qpos, vv):
    lg, lw, hq = witness_logits_at(p, x, qpos, CFG["n_layer"], CFG["H"], mode,
                                   vv.goal_lo, vv.K, 0.1, hard=True)
    return (np_plain.asarray(lg).argmax(-1),
            np_plain.asarray(lw), np_plain.asarray(hq))

def _witness_soft_pred(p, mode, x, qpos, vv):
    lg, _, _ = witness_logits_at(p, x, qpos, CFG["n_layer"], CFG["H"], mode,
                                 vv.goal_lo, vv.K, 0.1, hard=False)
    return np_plain.asarray(lg).argmax(-1)

def _audit(wid_raw, pi, pred, y, goal, q, ginj=None):
    """Audit stats. wid_raw = raw slot ids; pi maps slot -> goal."""
    w = pi[wid_raw]
    Wg_raw = float((wid_raw == goal).mean())
    Wg = float((w == goal).mean())
    y_w = rule_answer(CFG["task"], w, q, CFG["V_ans"])
    V = float((pred == y_w).mean())
    OC = float(((w == goal) & (pred != y)).mean())
    Cw = float((w == ginj).mean()) if ginj is not None else float("nan")
    return Wg_raw, Wg, Cw, V, OC

def fit_pi(arms=("vW", "vWo")):
    """Fit the goal->slot dictionary ONCE per arm on clean L=16 (n=192),
    frozen thereafter (pre-registered). Brute force over K! permutations."""
    from itertools import permutations
    vv = vocab()
    rng = np_plain.random.default_rng(OFF + 21)
    x, qpos, y, goal = make_batch(rng, vv, CFG["task"], 192, 16, fit(20),
                                  distractors=True)
    out = json.load(open(_pi_path())) if os.path.exists(_pi_path()) else {}
    for arm in arms:
        st = _load(arm)
        if st is None:
            print(f"{arm}: no checkpoint"); continue
        if arm.startswith("vWr"):
            p1 = _load("v1_cap1750" if arm.endswith("cap1750") else "v1")
            assert p1, "frozen backbone missing for " + arm
            hqf, eqf, Egf = _frozen_feats(p1[0], x, qpos, apos_of(x, vv.PAD), vv)
            _, lw = retro_logits(st[0], hqf, eqf, Egf, 1.0, hard=True)
            lw = np_plain.asarray(lw)
        else:
            mode = "vW" if arm in ("vWc", "vW_degraded") else arm
            _, lw, _ = _witness_hard(st[0], mode, x, qpos, vv)
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

def evaluate(eval_n=192, lens=(0, 16, 32, 64, 96, 128, 192, 224, 256), dist=15):
    vv = vocab()
    arms_w = [(t, _load(t)) for t in ("vW", "vWo", "vWc")]
    arms_w = [(t, s) for t, s in arms_w if s]
    p1 = _load("v1"); pB = _load("vB"); mp = _load("mon"); pr1 = _load("probe_v1")
    rng = np_plain.random.default_rng(OFF + 999 + (0 if dist == 15 else 1))
    rows = []
    for L in lens:
        bs = fit(L + 4)
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], eval_n, L, bs,
                                      distractors=(dist == 15))
        ap = apos_of(x, vv.PAD); q = np_plain.asarray(x)[np_plain.arange(len(y)), qpos] - vv.ans_lo
        args = (CFG["n_layer"], CFG["H"])
        row = [SEED, L]; hdr = ["seed", "L"]
        for tag, st in arms_w:
            mode = "vW" if tag == "vWc" else tag
            pred, lw, _ = _witness_hard(st[0], mode, x, qpos, vv)
            sh = float((pred == _witness_soft_pred(st[0], mode, x, qpos, vv)).mean())
            Wg_raw, Wg, _, V, OC = _audit(lw.argmax(-1), _pi_for(tag), pred, y, goal, q)
            row += [_acc_pred(pred, y), Wg_raw, Wg, V, OC, sh]
            hdr += [f"{tag}", f"{tag}_Wg_raw", f"{tag}_Wg_pi", f"{tag}_V",
                    f"{tag}_OC", f"{tag}_softhard"]
        if p1:
            l1, hq1 = logits_at(p1[0], x, qpos, None, ap, *args, "v1", vv.goal_lo, vv.K)
            pred1 = np_plain.asarray(l1).argmax(-1)
            row += [_acc(l1, y)]; hdr += ["v1"]
            if pr1:
                wid = np_plain.asarray(probe_logits(pr1[0], np_plain.asarray(hq1))).argmax(-1)
                Wg_raw, Wg, _, V, OC = _audit(wid, _pi_for("probe"), pred1, y, goal, q)
                row += [Wg, V, OC]; hdr += ["probe_Wg", "probe_V", "probe_OC"]
            if mp:
                row += [_acc(monitor_logits(mp[0], np_plain.asarray(hq1), goal), y)]
                hdr += ["mon"]
            rr = _load("vWr")
            if rr:
                hqf, eqf, Egf = _frozen_feats(p1[0], x, qpos, ap, vv)
                lgr, lwr = retro_logits(rr[0], hqf, eqf, Egf, 1.0, hard=True)
                predr = np_plain.asarray(lgr).argmax(-1)
                Wg_raw, Wg, _, V, OC = _audit(np_plain.asarray(lwr).argmax(-1),
                                              _pi_for("vWr"), predr, y, goal, q)
                row += [_acc_pred(predr, y), Wg_raw, Wg, V, OC]
                hdr += ["vWr", "vWr_Wg_raw", "vWr_Wg_pi", "vWr_V", "vWr_OC"]
        if pB:
            lB, _ = logits_at(pB[0], x, qpos, None, ap, *args, "vB", vv.goal_lo, vv.K)
            predB = np_plain.asarray(lB).argmax(-1)
            wid = np_plain.asarray(x)[np_plain.arange(len(y)), ap + 1] - vv.goal_lo
            Wg_raw, Wg, _, V, OC = _audit(wid, _pi_for("latch"), predB, y, goal, q)
            row += [_acc(lB, y), Wg, V, OC]
            hdr += ["vB", "latch_Wg", "latch_V", "latch_OC"]
        rows.append(row)
        print(" ".join(f"{h}={v:.3f}" if isinstance(v, float) else f"{h}={v}"
                       for h, v in zip(hdr[1:], row[1:])))
    suf = "" if dist == 15 else "_diag"
    _append_csv(f"results/np_sweep_{CFG['task']}_s{SEED}{suf}.csv",
                hdr + ["chance"], [r + [1.0 / vv.V_ans] for r in rows])

def _acc_pred(pred, y): return float((pred == y).mean())

def _rcx(pred, y, y_inj):
    R = float((pred == y).mean()); C = float((pred == y_inj).mean())
    return R, C, 1.0 - R - C

def eval_inject(eval_n=192, lens=(32, 64, 128, 192), phis=(0.25, 0.75, "end"),
                ms=(1, 4), trace_n=32, only=None):
    # `only`: eval-plumbing filter (session 9) — emit rows for the named vW-arms
    # only, skipping the reused-arm blocks (their rows are already on file).
    # Data generation (rng) unchanged; statistics unchanged.
    vv = vocab()
    arms_w = [(t, _load(t)) for t in ("vW", "vWo", "vWc")]
    arms_w = [(t, s) for t, s in arms_w if s and (only is None or t in only)]
    p1 = _load("v1"); pB = _load("vB"); mp = _load("mon"); pr1 = _load("probe_v1")
    rng = np_plain.random.default_rng(OFF + 5555)
    args = (CFG["n_layer"], CFG["H"])
    rows = []
    tr_path = f"results/witness_traces_s{SEED}.npz"
    traces = dict(np_plain.load(tr_path)) if os.path.exists(tr_path) else {}
    HDR = ["seed", "arm", "L", "phi", "m", "R", "C", "X", "delta",
           "Wg_raw", "Rw_pi", "Cw_pi", "V", "OC", "softhard", "n"]
    for L in lens:
        for phi in phis:
            for m in ms:
                bs = fit(L + 4)
                x, qpos, y, goal, ap, y_inj, ginj = make_batch_injected(
                    rng, vv, CFG["task"], eval_n, L, bs, phi, m, distractors=True)
                q = np_plain.asarray(x)[np_plain.arange(len(y)), qpos] - vv.ans_lo
                for tag, st in arms_w:
                    mode = "vW" if tag == "vWc" else tag
                    pred, lw, _ = _witness_hard(st[0], mode, x, qpos, vv)
                    sh = float((pred == _witness_soft_pred(st[0], mode, x, qpos, vv)).mean())
                    R, C, X = _rcx(pred, y, y_inj)
                    Wg_raw, Rw, Cw, V, OC = _audit(lw.argmax(-1), _pi_for(tag),
                                                   pred, y, goal, q, ginj)
                    rows.append([SEED, tag, L, phi, m, R, C, X, R - C,
                                 Wg_raw, Rw, Cw, V, OC, sh, eval_n])
                    if tag == "vW":
                        traces[f"lw_L{L}_phi{phi}_m{m}"] = lw[:trace_n]
                    elif only is not None:
                        traces[f"lw_{tag}_L{L}_phi{phi}_m{m}"] = lw[:trace_n]
                if p1 and only is None:
                    l1, hq1 = logits_at(p1[0], x, qpos, None, ap, *args, "v1",
                                        vv.goal_lo, vv.K)
                    pred1 = np_plain.asarray(l1).argmax(-1)
                    R, C, X = _rcx(pred1, y, y_inj)
                    rows.append([SEED, "v1", L, phi, m, R, C, X, R - C]
                                + [float("nan")] * 5 + [eval_n])
                    if pr1:
                        wid = np_plain.asarray(probe_logits(pr1[0], np_plain.asarray(hq1))).argmax(-1)
                        Wg_raw, Rw, Cw, V, OC = _audit(wid, _pi_for("probe"),
                                                       pred1, y, goal, q, ginj)
                        rows.append([SEED, "probe_v1", L, phi, m] + [float("nan")] * 4
                                    + [Wg_raw, Rw, Cw, V, OC, float("nan"), eval_n])
                    if mp:
                        lmon = monitor_logits(mp[0], np_plain.asarray(hq1), goal)
                        R, C, X = _rcx(np_plain.asarray(lmon).argmax(-1), y, y_inj)
                        rows.append([SEED, "mon", L, phi, m, R, C, X, R - C]
                                    + [float("nan")] * 5 + [eval_n])
                    rr = _load("vWr")
                    if rr:
                        hqf, eqf, Egf = _frozen_feats(p1[0], x, qpos, ap, vv)
                        lgr, lwr = retro_logits(rr[0], hqf, eqf, Egf, 1.0, hard=True)
                        predr = np_plain.asarray(lgr).argmax(-1)
                        R, C, X = _rcx(predr, y, y_inj)
                        Wg_raw, Rw, Cw, V, OC = _audit(np_plain.asarray(lwr).argmax(-1),
                                                       _pi_for("vWr"), predr, y,
                                                       goal, q, ginj)
                        rows.append([SEED, "vWr", L, phi, m, R, C, X, R - C,
                                     Wg_raw, Rw, Cw, V, OC, float("nan"), eval_n])
                if pB and only is None:
                    lB, _ = logits_at(pB[0], x, qpos, None, ap, *args, "vB",
                                      vv.goal_lo, vv.K)
                    predB = np_plain.asarray(lB).argmax(-1)
                    R, C, X = _rcx(predB, y, y_inj)
                    wid = np_plain.asarray(x)[np_plain.arange(len(y)), ap + 1] - vv.goal_lo
                    Wg_raw, Rw, Cw, V, OC = _audit(wid, _pi_for("latch"),
                                                   predB, y, goal, q, ginj)
                    rows.append([SEED, "vB", L, phi, m, R, C, X, R - C,
                                 Wg_raw, Rw, Cw, V, OC, float("nan"), eval_n])
                print(f"cell L={L} phi={phi} m={m} done")
    os.makedirs("results", exist_ok=True)
    np_plain.savez(tr_path, **traces)
    _append_csv(f"results/np_inject_s{SEED}.csv", HDR, rows)

def eval_degrade(eval_n=192, cells="both", arms="all"):
    """Audit-honesty battery: probe on v1_cap1750 vs loss-matched vW_degraded.
    Pre-registered grid: clean L in {16,64,192}; inject L in {64,192} x
    phi in {0.25,end} x m=4."""
    vv = vocab()
    sel = None if arms == "all" else set(arms.split(","))
    def _on(name): return sel is None or name in sel
    pdW = _load("vW_degraded")
    p1d = _load("v1_cap1750"); prd = _load("probe_v1_cap1750")
    rrd = _load("vWr_v1_cap1750")
    args = (CFG["n_layer"], CFG["H"])
    rng = np_plain.random.default_rng(OFF + 8888)
    rows = []
    HDR = ["seed", "arm", "cell", "L", "phi", "m", "acc_or_R", "C",
           "Wg_raw", "Wg_pi", "Cw_pi", "V", "OC", "n"]
    def _cells_clean():
        for L in (16, 64, 192):
            x, qpos, y, goal = make_batch(rng, vv, CFG["task"], eval_n, L,
                                          fit(L + 4), distractors=True)
            yield "clean", L, "", "", x, qpos, y, goal, apos_of(x, vv.PAD), None, None
    def _cells_inj():
        for L in (64, 192):
            for phi in (0.25, "end"):
                x, qpos, y, goal, ap, y_inj, ginj = make_batch_injected(
                    rng, vv, CFG["task"], eval_n, L, fit(L + 4), phi, 4,
                    distractors=True)
                yield "inject", L, phi, 4, x, qpos, y, goal, ap, y_inj, ginj
    gens = []
    if cells in ("both", "clean"): gens.append(_cells_clean())
    if cells in ("both", "inject"): gens.append(_cells_inj())
    for gen in gens:
        for kind, L, phi, m, x, qpos, y, goal, ap, y_inj, ginj in gen:
            q = np_plain.asarray(x)[np_plain.arange(len(y)), qpos] - vv.ans_lo
            if pdW and _on("vW_degraded"):
                pred, lw, _ = _witness_hard(pdW[0], "vW", x, qpos, vv)
                Wg_raw, Wg, Cw, V, OC = _audit(lw.argmax(-1), _pi_for("vW_degraded"),
                                               pred, y, goal, q, ginj)
                R = _acc_pred(pred, y)
                C = float((pred == y_inj).mean()) if y_inj is not None else float("nan")
                rows.append([SEED, "vW_degraded", kind, L, phi, m, R, C,
                             Wg_raw, Wg, Cw, V, OC, eval_n])
            if p1d:
                l1, hq1 = logits_at(p1d[0], x, qpos, None, ap, *args, "v1",
                                    vv.goal_lo, vv.K)
                pred1 = np_plain.asarray(l1).argmax(-1)
                R = _acc_pred(pred1, y)
                C = float((pred1 == y_inj).mean()) if y_inj is not None else float("nan")
                if rrd and _on("vWr_v1_cap1750"):
                    hqf, eqf, Egf = _frozen_feats(p1d[0], x, qpos,
                                                  ap if ap is not None else apos_of(x, vv.PAD), vv)
                    lgr, lwr = retro_logits(rrd[0], hqf, eqf, Egf, 1.0, hard=True)
                    predr = np_plain.asarray(lgr).argmax(-1)
                    Wg_raw, Wg, Cw, V, OC = _audit(np_plain.asarray(lwr).argmax(-1),
                                                   _pi_for("vWr_v1_cap1750"), predr,
                                                   y, goal, q, ginj)
                    Rr = _acc_pred(predr, y)
                    Cr = float((predr == y_inj).mean()) if y_inj is not None else float("nan")
                    rows.append([SEED, "vWr_v1_cap1750", kind, L, phi, m, Rr, Cr,
                                 Wg_raw, Wg, Cw, V, OC, eval_n])
                if prd and _on("probe_v1_cap1750"):
                    wid = np_plain.asarray(probe_logits(prd[0], np_plain.asarray(hq1))).argmax(-1)
                    Wg_raw, Wg, Cw, V, OC = _audit(wid, _pi_for("probe"),
                                                   pred1, y, goal, q, ginj)
                    rows.append([SEED, "probe_v1_cap1750", kind, L, phi, m, R, C,
                                 Wg_raw, Wg, Cw, V, OC, eval_n])
            print(f"degrade cell {kind} L={L} phi={phi} done")
    _append_csv(f"results/np_degrade_s{SEED}.csv", HDR, rows)

def status():
    for tag in ["vW", "vWo", "vWc", "vWr", "vW_degraded", "v1", "v1_cap1750", "vB",
                "mon", "probe_v1", "probe_v1_cap1750"]:
        st = _load(tag)
        if st: print(tag, "step", st[3], "loss", round(st[4][-1], 4) if st[4] else None)
        else: print(tag, "not started")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "train_retro", "fit_pi", "eval", "eval_inject",
                                    "eval_degrade", "status"])
    ap.add_argument("--model", default="vW")
    ap.add_argument("--feat", default="v1")
    ap.add_argument("--seconds", type=float, default=34)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cap", type=int, default=STEP_CAP)
    ap.add_argument("--dist", type=int, default=15, choices=[0, 15])
    ap.add_argument("--lens", type=str, default="")
    ap.add_argument("--phis", type=str, default="")
    ap.add_argument("--arms", type=str, default="vW,vWo")
    ap.add_argument("--cells", type=str, default="both")
    ap.add_argument("--only", type=str, default="")
    a = ap.parse_args()
    set_seed(a.seed)
    lens = tuple(int(x) for x in a.lens.split(",")) if a.lens else None
    if a.cmd == "train": train(a.model, a.seconds, a.cap)
    elif a.cmd == "train_retro": train_retro(a.seconds, a.feat, a.cap)
    elif a.cmd == "fit_pi": fit_pi(tuple(a.arms.split(",")))
    elif a.cmd == "eval": evaluate(lens=lens or (0, 16, 32, 64, 96, 128, 192, 224, 256), dist=a.dist)
    elif a.cmd == "eval_inject":
        phis = tuple(float(x) if x != "end" else "end" for x in a.phis.split(",")) if a.phis else (0.25, 0.75, "end")
        eval_inject(lens=lens or (32, 64, 128, 192), phis=phis,
                    only=(tuple(a.only.split(",")) if a.only else None))
    elif a.cmd == "eval_degrade": eval_degrade(cells=a.cells, arms=(a.arms if a.arms != "vW,vWo" else "all"))
    else: status()
