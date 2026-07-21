"""
exp06 chunked CPU runner (34s-burst discipline, checkpoint .npz).

  python np_run.py train --model v1|v2|vB|vBl|v1c --seconds 34 --seed N [--cap 1750]
  python np_run.py train_monitor --feat v1|v1c --seconds 34 --seed N
  python np_run.py train_probe --feat v1|vB|v1c --seconds 34 --seed N
  python np_run.py eval        [--lens 0,16,32] [--dist 15|0] --seed N
  python np_run.py eval_inject [--lens 32,64]   --seed N
  python np_run.py eval_inject_rem --seed N        (position-controlled cells)
  python np_run.py status --seed N
"""
import argparse, csv, json, os, time
import numpy as np_plain
import autograd.numpy as np
from data import (Vocab, make_batch, make_batch_reminders, make_batch_injected,
                  make_batch_reminders_injected)
from np_model import (init_params, loss_fn, grad_loss, adam_step, logits_at,
                      init_monitor, monitor_logits, monitor_loss, grad_monitor,
                      init_probe, probe_logits, probe_loss, grad_probe,
                      friction_trace)

CFG = dict(K=4, V_ans=8, V_filler=32, n_layer=2, H=2, D=48,
           T=288, train_max_len=64, batch=32, lr=1e-3, task="hard",
           remind_every=16)
STEP_CAP = 1750
SEED = 1; OFF = 10000; CK = "np_ckpt_hard_s1"
ARM_OFF = {"v1": 0, "v2": 1, "vB": 2, "vBl": 3, "v1c": 4}
CURR = dict(ramp_steps=1000, d_max=0.15)   # exp02b pre-registered schedule (v1c contingency)

def set_seed(s):
    global SEED, OFF, CK
    SEED = int(s); OFF = 10000 * SEED; CK = f"np_ckpt_{CFG['task']}_s{SEED}"
    os.makedirs(CK, exist_ok=True)

def set_task(t): CFG["task"] = t

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
    mode = "v1" if tag == "v1c" else tag   # v1c = v1 architecture, curriculum distractors
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
        dr = (CURR["d_max"] * min(1.0, step / CURR["ramp_steps"])) if tag == "v1c" else True
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
    _save(tag, p, m, v, step, losses)
    print(json.dumps({"tag": tag, "step": step, "done": step >= cap or _early_stop(losses),
                      "last_loss": round(losses[-1], 4) if losses else None}))

def _feat(p, x, qpos, ap, mode, vv):
    _, hq = logits_at(p, x, qpos, None, ap, CFG["n_layer"], CFG["H"],
                      mode, vv.goal_lo, vv.K)
    return np_plain.asarray(hq)

def train_monitor(seconds, feat="v1"):
    vv = vocab()
    st = _load(feat); assert st, f"train {feat} first"
    p1 = st[0]
    fo = ARM_OFF[feat]                      # 0 for v1: legacy path byte-identical
    tag = "mon" if feat == "v1" else "mon_c"
    stm = _load(tag)
    if stm is None:
        mp = init_monitor(np_plain.random.default_rng(OFF + 9 + fo), vv.K, CFG["D"], vv.V_ans)
        m = {k: np_plain.zeros_like(mp[k]) for k in mp}
        v = {k: np_plain.zeros_like(mp[k]) for k in mp}
        step, losses = 0, []
        rng = np_plain.random.default_rng(OFF + 7 + fo)
    else:
        mp, m, v, step, losses = stm
        rng = np_plain.random.default_rng(OFF + 7000 + step + fo)
    t0 = time.time()
    while time.time() - t0 < seconds and step < STEP_CAP:
        L = int(rng.integers(0, CFG["train_max_len"] + 1))
        bs = int(rng.choice([68, 112, 160, 208, 256, 288],
                            p=[0.35, 0.2, 0.15, 0.12, 0.1, 0.08]))
        bs = max(bs, L + 4)
        b = max(8, int(CFG["batch"] * 68 / bs))
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], b, L, bs, distractors=True)
        hq = _feat(p1, x, qpos, apos_of(x, vv.PAD), "v1", vv)
        g = grad_monitor(mp, hq, goal, y)
        step += 1
        mp, m, v = adam_step(mp, g, m, v, step, CFG["lr"])
        if step % 10 == 0:
            losses.append(float(monitor_loss(mp, hq, goal, y)))
    _save(tag, mp, m, v, step, losses)
    print(json.dumps({"tag": tag, "step": step,
                      "last_loss": round(losses[-1], 4) if losses else None}))

def train_probe(feat, seconds):
    """Assigned-goal probe on frozen features (CLEAN data only, per spec)."""
    vv = vocab()
    st = _load(feat); assert st, f"train {feat} first"
    pf = st[0]
    tag = f"probe_{feat}"
    stp = _load(tag)
    if stp is None:
        pp = init_probe(np_plain.random.default_rng(OFF + 11 + ARM_OFF[feat]),
                        CFG["D"], vv.K)
        m = {k: np_plain.zeros_like(pp[k]) for k in pp}
        v = {k: np_plain.zeros_like(pp[k]) for k in pp}
        step, losses = 0, []
        rng = np_plain.random.default_rng(OFF + 13 + ARM_OFF[feat])
    else:
        pp, m, v, step, losses = stp
        rng = np_plain.random.default_rng(OFF + 13000 + step + ARM_OFF[feat])
    t0 = time.time()
    while time.time() - t0 < seconds and step < STEP_CAP:
        L = int(rng.integers(0, CFG["train_max_len"] + 1))
        bs = int(rng.choice([68, 112, 160, 208, 256, 288],
                            p=[0.35, 0.2, 0.15, 0.12, 0.1, 0.08]))
        bs = max(bs, L + 4)
        b = max(8, int(CFG["batch"] * 68 / bs))
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], b, L, bs, distractors=True)
        hq = _feat(pf, x, qpos, apos_of(x, vv.PAD),
                   "v1" if feat == "v1c" else feat, vv)
        g = grad_probe(pp, hq, goal)
        step += 1
        pp, m, v = adam_step(pp, g, m, v, step, CFG["lr"])
        if step % 10 == 0:
            losses.append(float(probe_loss(pp, hq, goal)))
    _save(tag, pp, m, v, step, losses)
    print(json.dumps({"tag": tag, "step": step,
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

def evaluate(eval_n=192, lens=(0, 16, 32, 64, 96, 128, 192, 224, 256), dist=15):
    vv = vocab()
    p1 = _load("v1")[0]; p2 = _load("v2")[0]
    pB = _load("vB")[0]; pBl = _load("vBl")[0]
    mp = _load("mon")[0]
    pr1 = _load("probe_v1"); prB = _load("probe_vB")
    pc = _load("v1c"); mpc = _load("mon_c"); prc = _load("probe_v1c")
    rng = np_plain.random.default_rng(OFF + 999 + (0 if dist == 15 else 1))
    d = dist == 15
    rows = []
    for L in lens:
        bs = fit(L + 4)
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], eval_n, L, bs, distractors=d)
        ap = apos_of(x, vv.PAD)
        args = (CFG["n_layer"], CFG["H"])
        l1, hq1 = logits_at(p1, x, qpos, None, ap, *args, "v1", vv.goal_lo, vv.K)
        l2, _ = logits_at(p2, x, qpos, goal, ap, *args, "v2", vv.goal_lo, vv.K)
        lB, hqB = logits_at(pB, x, qpos, None, ap, *args, "vB", vv.goal_lo, vv.K)
        lBl, _ = logits_at(pBl, x, qpos, None, ap, *args, "vBl", vv.goal_lo, vv.K)
        amon = _acc(monitor_logits(mp, np_plain.asarray(hq1), goal), y)
        ap1 = _acc(probe_logits(pr1[0], np_plain.asarray(hq1)), goal) if pr1 else float("nan")
        apB = _acc(probe_logits(prB[0], np_plain.asarray(hqB)), goal) if prB else float("nan")
        if pc:
            lc, hqc = logits_at(pc[0], x, qpos, None, ap, *args, "v1", vv.goal_lo, vv.K)
            ac = _acc(lc, y)
            amonc = _acc(monitor_logits(mpc[0], np_plain.asarray(hqc), goal), y) if mpc else float("nan")
            apc = _acc(probe_logits(prc[0], np_plain.asarray(hqc)), goal) if prc else float("nan")
        else:
            ac = amonc = apc = float("nan")
        try:
            rem_len = L + 4 + 2 * ((L - 1) // CFG["remind_every"] + 1 if L > 0 else 0)
            xr, qr, yr, gr = make_batch_reminders(rng, vv, CFG["task"], eval_n, L,
                                                  fit(rem_len), CFG["remind_every"],
                                                  distractors=d)
            if xr.shape[1] > CFG["T"]: raise ValueError("too long")
            lr, _ = logits_at(p1, xr, qr, None, apos_of(xr, vv.PAD), *args,
                              "v1", vv.goal_lo, vv.K)
            ar = _acc(lr, yr)
        except ValueError:
            ar = float("nan")
        row = [SEED, L, _acc(l1, y), ar, amon, _acc(l2, y), _acc(lB, y),
               _acc(lBl, y), ap1, apB, ac, amonc, apc]
        rows.append(row)
        print("L=%4d v1=%.3f rem=%.3f mon=%.3f v2=%.3f vB=%.3f vBl=%.3f pr1=%.3f prB=%.3f v1c=%.3f monc=%.3f prc=%.3f"
              % tuple(row[1:]))
    suf = "" if dist == 15 else "_diag"
    _append_csv(f"results/np_sweep_{CFG['task']}_s{SEED}{suf}.csv",
                ["seed", "L", "v1", "v1_reminder", "v1_monitor", "v2_anchor",
                 "vB", "vB_latch", "probe_v1", "probe_vB",
                 "v1c", "v1c_monitor", "probe_v1c", "chance"],
                [r + [1.0 / vv.V_ans] for r in rows])

def _rcx(lg, y, y_inj):
    pred = np_plain.asarray(lg).argmax(-1)
    R = float((pred == y).mean()); C = float((pred == y_inj).mean())
    return R, C, 1.0 - R - C

def _pergoal_rows(arm, L, phi, m, pred, y, y_inj, goal):
    out = []
    for g in range(CFG["K"]):
        idx = goal == g
        if idx.sum() == 0: continue
        out.append([SEED, arm, L, phi, m, g,
                    float((pred[idx] == y[idx]).mean()),
                    float((pred[idx] == y_inj[idx]).mean()), int(idx.sum())])
    return out

def eval_inject(eval_n=192, lens=(32, 64, 128, 192), phis=(0.25, 0.75, "end"),
                ms=(1, 4), trace_n=32):
    vv = vocab()
    p1 = _load("v1")[0]; p2 = _load("v2")[0]
    pB = _load("vB")[0]; pBl = _load("vBl")[0]
    mp = _load("mon")[0]
    pr1 = _load("probe_v1"); prB = _load("probe_vB")
    pc = _load("v1c"); mpc = _load("mon_c"); prc = _load("probe_v1c")
    rng = np_plain.random.default_rng(OFF + 5555)
    args = (CFG["n_layer"], CFG["H"])
    rows, pg_rows = [], []
    tr_path = f"results/traces_s{SEED}.npz"
    traces = dict(np_plain.load(tr_path)) if os.path.exists(tr_path) else {}
    for L in lens:
        for phi in phis:
            for m in ms:
                bs = fit(L + 4)
                x, qpos, y, goal, ap, y_inj, ginj = make_batch_injected(
                    rng, vv, CFG["task"], eval_n, L, bs, phi, m, distractors=True)
                cell = []
                l1, hq1 = logits_at(p1, x, qpos, None, ap, *args, "v1", vv.goal_lo, vv.K)
                cell.append(("v1", l1))
                l2, _ = logits_at(p2, x, qpos, goal, ap, *args, "v2", vv.goal_lo, vv.K)
                cell.append(("v2", l2))
                lB, hqB = logits_at(pB, x, qpos, None, ap, *args, "vB", vv.goal_lo, vv.K)
                cell.append(("vB", lB))
                lBl, _ = logits_at(pBl, x, qpos, None, ap, *args, "vBl", vv.goal_lo, vv.K)
                cell.append(("vBl", lBl))
                hqc = None
                if pc:
                    lc, hqc = logits_at(pc[0], x, qpos, None, ap, *args, "v1",
                                        vv.goal_lo, vv.K)
                    cell.append(("v1c", lc))
                cell.append(("mon", monitor_logits(mp, np_plain.asarray(hq1), goal)))
                if pc and mpc:
                    cell.append(("mon_c", monitor_logits(mpc[0], np_plain.asarray(hqc), goal)))
                for arm, lg in cell:
                    R, C, X = _rcx(lg, y, y_inj)
                    rows.append([SEED, arm, L, phi, m, R, C, X, R - C, eval_n])
                    pg_rows += _pergoal_rows(arm, L, phi, m,
                                             np_plain.asarray(lg).argmax(-1),
                                             y, y_inj, goal)
                # probes: R-analog = recover assigned g, C-analog = injected g'
                probe_set = [("probe_v1", pr1, hq1), ("probe_vB", prB, hqB)]
                if pc and prc: probe_set.append(("probe_v1c", prc, hqc))
                for nm, pr, hh in probe_set:
                    if pr is None: continue
                    pred = np_plain.asarray(probe_logits(pr[0], np_plain.asarray(hh))).argmax(-1)
                    R = float((pred == goal).mean()); C = float((pred == ginj).mean())
                    rows.append([SEED, nm, L, phi, m, R, C, 1 - R - C, R - C, eval_n])
                # reminder arm on the same grid geometry
                try:
                    rem_len = L + 4 + 2 * ((L - 1) // CFG["remind_every"] + 1) + 2
                    xr, qr, yr, gr, apr, yir, gir = make_batch_reminders_injected(
                        rng, vv, CFG["task"], eval_n, L, fit(rem_len),
                        CFG["remind_every"], phi, m, distractors=True)
                    if xr.shape[1] > CFG["T"]: raise ValueError
                    lr, _ = logits_at(p1, xr, qr, None, apr, *args, "v1",
                                      vv.goal_lo, vv.K)
                    R, C, X = _rcx(lr, yr, yir)
                    rows.append([SEED, "reminder", L, phi, m, R, C, X, R - C, eval_n])
                except ValueError:
                    rows.append([SEED, "reminder", L, phi, m] + [float("nan")] * 4 + [eval_n])
                f, s = friction_trace(pB, x[:trace_n], ap[:trace_n], vv.goal_lo, vv.K)
                traces[f"f_L{L}_phi{phi}_m{m}"] = np_plain.asarray(f)
                traces[f"s_L{L}_phi{phi}_m{m}"] = np_plain.asarray(s)
                print(f"cell L={L} phi={phi} m={m} done")
    os.makedirs("results", exist_ok=True)
    np_plain.savez(tr_path, **traces)
    _append_csv(f"results/np_inject_s{SEED}.csv",
                ["seed", "arm", "L", "phi", "m", "R", "C", "X", "delta", "n"], rows)
    _append_csv(f"results/np_inject_pergoal_s{SEED}.csv",
                ["seed", "arm", "L", "phi", "m", "g", "R", "C", "n"], pg_rows)

def eval_inject_rem(eval_n=192, lens=(64, 128), m=4):
    """Position-controlled reminder cells: injection after the last reminder
    vs covered by a following reminder."""
    vv = vocab()
    p1 = _load("v1")[0]
    rng = np_plain.random.default_rng(OFF + 7777)
    rows = []
    for L in lens:
        for mode in ("after_last", "covered"):
            rem_len = L + 4 + 2 * ((L - 1) // CFG["remind_every"] + 1) + 2
            xr, qr, yr, gr, apr, yir, gir = make_batch_reminders_injected(
                rng, vv, CFG["task"], eval_n, L, fit(rem_len),
                CFG["remind_every"], mode, m, distractors=True)
            lr, _ = logits_at(p1, xr, qr, None, apr, CFG["n_layer"], CFG["H"],
                              "v1", vv.goal_lo, vv.K)
            R, C, X = _rcx(lr, yr, yir)
            rows.append([SEED, f"reminder_{mode}", L, mode, m, R, C, X, R - C, eval_n])
            print(f"reminder {mode} L={L}: R={R:.3f} C={C:.3f}")
    _append_csv(f"results/np_inject_s{SEED}.csv",
                ["seed", "arm", "L", "phi", "m", "R", "C", "X", "delta", "n"], rows)

def status():
    for tag in ["v1", "v2", "vB", "vBl", "v1c", "mon", "mon_c",
                "probe_v1", "probe_vB", "probe_v1c"]:
        st = _load(tag)
        if st: print(tag, "step", st[3], "loss", round(st[4][-1], 4) if st[4] else None)
        else: print(tag, "not started")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "train_monitor", "train_probe",
                                    "eval", "eval_inject", "eval_inject_rem", "status"])
    ap.add_argument("--model", default="v1")
    ap.add_argument("--feat", default="v1")
    ap.add_argument("--seconds", type=float, default=34)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--task", default="hard", choices=["shift", "hard"])
    ap.add_argument("--cap", type=int, default=STEP_CAP)
    ap.add_argument("--dist", type=int, default=15, choices=[0, 15])
    ap.add_argument("--lens", type=str, default="")
    ap.add_argument("--phis", type=str, default="")
    a = ap.parse_args()
    set_task(a.task); set_seed(a.seed)
    lens = tuple(int(x) for x in a.lens.split(",")) if a.lens else None
    if a.cmd == "train": train(a.model, a.seconds, a.cap)
    elif a.cmd == "train_monitor": train_monitor(a.seconds, a.feat)
    elif a.cmd == "train_probe": train_probe(a.feat, a.seconds)
    elif a.cmd == "eval": evaluate(lens=lens or (0, 16, 32, 64, 96, 128, 192, 224, 256), dist=a.dist)
    elif a.cmd == "eval_inject":
        phis = tuple(float(x) if x != "end" else "end" for x in a.phis.split(",")) if a.phis else (0.25, 0.75, "end")
        eval_inject(lens=lens or (32, 64, 128, 192), phis=phis)
    elif a.cmd == "eval_inject_rem": eval_inject_rem(lens=lens or (64, 128))
    else: status()
