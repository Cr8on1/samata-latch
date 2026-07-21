"""
EXP05 chunked CPU runner (hop2 task). Trains in short bursts with .npz
checkpointing. Arms: v1, v2 (anchor), v1c (curriculum), mon (wrapper on
frozen v1), monc (contingency wrapper on frozen v1c).

  python np_run.py train --model v1|v2 --seconds 30 --seed N [--cap 1750]
  python np_run.py train_curr --seconds 30 --seed N [--cap 1750]
  python np_run.py train_monitor --features v1|v1c --seconds 30 --seed N
  python np_run.py eval --sweep primary|diag --lens 0,16,32 --seed N
  python np_run.py plot --seed N
  python np_run.py status --seed N
"""
import argparse, json, os, time
import numpy as np_plain
import autograd.numpy as np
from data import Vocab, make_batch, make_batch_reminders
from np_model import (init_params, loss_fn, grad_loss, adam_step, logits_at,
                      init_monitor, monitor_logits, monitor_loss, grad_monitor)

# ---- fixed experiment config (pre-registered in EXPERIMENT_SPEC.md) ----
CFG = dict(K=4, V_ans=8, V_filler=32, n_layer=3, H=2, D=48,
           T=384, train_max_len=64, batch=32, lr=1e-3, task="hop2",
           remind_every=16, cap=1750)
SIZES = [104, 148, 196, 244, 300, 356]          # trained window sizes (seq = L+100)
SIZE_P = [0.35, 0.2, 0.15, 0.12, 0.1, 0.08]
CURR = dict(ramp_steps=1000, d_max=0.15)        # exp02b-precedent schedule
EARLY = dict(thresh=0.03, checks=3, min_step=100)
SEED = 1; OFF = 10000; CK = "np_ckpt_hop2_s1"


def set_seed(s):
    global SEED, OFF, CK
    SEED = int(s); OFF = 10000 * SEED; CK = f"np_ckpt_{CFG['task']}_s{SEED}"
    os.makedirs(CK, exist_ok=True)


def vocab(): return Vocab(K=CFG["K"], V_ans=CFG["V_ans"], V_filler=CFG["V_filler"])


def _save(tag, p, m, v, step, losses):
    np_plain.savez(f"{CK}/{tag}.npz",
                   step=step, losses=np_plain.array(losses),
                   **{"p_" + k: p[k] for k in p},
                   **{"m_" + k: m[k] for k in m},
                   **{"v_" + k: v[k] for k in v})


def _load(tag):
    f = f"{CK}/{tag}.npz"
    if not os.path.exists(f): return None
    z = np_plain.load(f)
    p = {k[2:]: z[k] for k in z.files if k.startswith("p_")}
    m = {k[2:]: z[k] for k in z.files if k.startswith("m_")}
    v = {k[2:]: z[k] for k in z.files if k.startswith("v_")}
    return p, m, v, int(z["step"]), list(z["losses"])


def _pick_bs(rng, L):
    bs = int(rng.choice(SIZES, p=SIZE_P))
    return max(bs, L + 100)                      # seq = L + 100 must fit


def _bcomp(bs): return max(8, int(CFG["batch"] * SIZES[0] / bs))


def _early_stop(losses, step):
    return (step >= EARLY["min_step"] and len(losses) >= EARLY["checks"]
            and all(l < EARLY["thresh"] for l in losses[-EARLY["checks"]:]))


def _drate_curr(step):
    return CURR["d_max"] * min(1.0, step / CURR["ramp_steps"])


def _train_loop(tag, anchor, seconds, cap, drate_fn):
    vv = vocab()
    st = _load(tag)
    if st is None:
        p = init_params(np_plain.random.default_rng(OFF + 42 + int(anchor)),
                        vv.size, vv.V_ans, vv.K, CFG["T"], CFG["n_layer"],
                        CFG["H"], CFG["D"], anchor)
        m = {k: np_plain.zeros_like(p[k]) for k in p}
        v = {k: np_plain.zeros_like(p[k]) for k in p}
        step, losses = 0, []
        rng = np_plain.random.default_rng(OFF + (1 if anchor else 0))
    else:
        p, m, v, step, losses = st
        rng = np_plain.random.default_rng(OFF + 1000 + step + (1 if anchor else 0))
    t0 = time.time(); done = "cap" if step >= cap else ""
    if _early_stop(losses, step): done = "early"
    while not done and time.time() - t0 < seconds:
        d = drate_fn(step)
        L = int(rng.integers(0, CFG["train_max_len"] + 1))
        bs = _pick_bs(rng, L)
        b = _bcomp(bs)
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], b, L, bs, distractors=d)
        g = grad_loss(p, x, qpos, y, goal, CFG["n_layer"], CFG["H"], anchor)
        step += 1
        p, m, v = adam_step(p, g, m, v, step, CFG["lr"])
        if step % 10 == 0:
            losses.append(float(loss_fn(p, x, qpos, y, goal,
                                        CFG["n_layer"], CFG["H"], anchor)))
            if _early_stop(losses, step): done = "early"
        if step >= cap: done = "cap"
    _save(tag, p, m, v, step, losses)
    print(json.dumps({"tag": tag, "seed": SEED, "step": step,
                      "done": done or "running",
                      "d_now": round(drate_fn(step), 4),
                      "last_loss": round(losses[-1], 4) if losses else None}))


def train(tag, seconds, cap):
    _train_loop(tag, anchor=(tag == "v2"), seconds=seconds, cap=cap,
                drate_fn=lambda s: 0.15)


def train_curr(seconds, cap):
    _train_loop("v1c", anchor=False, seconds=seconds, cap=cap,
                drate_fn=_drate_curr)


def train_monitor(features, seconds, cap):
    """Wrapper arm: MLP on frozen features of `features` model + true goal."""
    vv = vocab()
    st = _load(features); assert st, f"train {features} first"
    p1 = st[0]
    tag = "mon" if features == "v1" else "monc"
    stm = _load(tag)
    if stm is None:
        mp = init_monitor(np_plain.random.default_rng(OFF + 9), vv.K, CFG["D"], vv.V_ans)
        m = {k: np_plain.zeros_like(mp[k]) for k in mp}
        v = {k: np_plain.zeros_like(mp[k]) for k in mp}
        step, losses = 0, []
        rng = np_plain.random.default_rng(OFF + 7)
    else:
        mp, m, v, step, losses = stm
        rng = np_plain.random.default_rng(OFF + 7000 + step)
    t0 = time.time(); done = "cap" if step >= cap else ""
    if _early_stop(losses, step): done = "early"
    while not done and time.time() - t0 < seconds:
        L = int(rng.integers(0, CFG["train_max_len"] + 1))
        bs = _pick_bs(rng, L)
        b = _bcomp(bs)
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], b, L, bs, distractors=True)
        _, hq = logits_at(p1, x, qpos, None, CFG["n_layer"], CFG["H"], False)
        hq = np_plain.asarray(hq)
        g = grad_monitor(mp, hq, goal, y)
        step += 1
        mp, m, v = adam_step(mp, g, m, v, step, CFG["lr"])
        if step % 10 == 0:
            losses.append(float(monitor_loss(mp, hq, goal, y)))
            if _early_stop(losses, step): done = "early"
        if step >= cap: done = "cap"
    _save(tag, mp, m, v, step, losses)
    print(json.dumps({"tag": tag, "seed": SEED, "step": step,
                      "done": done or "running",
                      "last_loss": round(losses[-1], 4) if losses else None}))


def _acc(lg, y): return float((np_plain.asarray(lg).argmax(-1) == y).mean())


def _acc_pg(lg, y, goal, K):
    pred = np_plain.asarray(lg).argmax(-1)
    return [round(float((pred == y)[goal == g].mean()), 4) for g in range(K)]


def _fit(n):
    for b in SIZES:
        if b >= n: return b
    raise ValueError("too long")


def _append(path, header, row):
    new = not os.path.exists(path)
    import csv
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row)


def evaluate(sweep, lens, eval_n=192):
    vv = vocab(); K = vv.K
    drate = 0.15 if sweep == "primary" else 0.0
    p1 = _load("v1")[0]; p2 = _load("v2")[0]; mp = _load("mon")[0]
    stc = _load("v1c"); pc = stc[0] if stc else None
    stmc = _load("monc"); mpc = stmc[0] if stmc else None
    os.makedirs("results", exist_ok=True)
    suff = "" if sweep == "primary" else "_diag"
    csv_main = f"results/np_sweep_hop2{suff}_s{SEED}.csv"
    csv_pg = f"results/np_sweep_hop2_pergoal_s{SEED}.csv"
    hdr = ["L", "v1", "v1_reminder", "v1_monitor", "v2_anchor", "v1c", "monc", "chance"]
    hdr_pg = ["sweep", "L", "arm", "g0", "g1", "g2", "g3"]
    chance = 1.0 / vv.V_ans
    for L in lens:
        rng = np_plain.random.default_rng(OFF + 999 + 31 * L + (7 if sweep == "diag" else 0))
        bs = _fit(L + 100)
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], eval_n, L, bs,
                                      distractors=drate, balance_goals=True)
        l1, hq = logits_at(p1, x, qpos, None, CFG["n_layer"], CFG["H"], False)
        a1 = _acc(l1, y); pg1 = _acc_pg(l1, y, goal, K)
        lm = monitor_logits(mp, np_plain.asarray(hq), goal)
        am = _acc(lm, y); pgm = _acc_pg(lm, y, goal, K)
        l2, _ = logits_at(p2, x, qpos, goal, CFG["n_layer"], CFG["H"], True)
        a2 = _acc(l2, y); pg2 = _acc_pg(l2, y, goal, K)
        if pc is not None:
            lc, hqc = logits_at(pc, x, qpos, None, CFG["n_layer"], CFG["H"], False)
            ac = _acc(lc, y); pgc = _acc_pg(lc, y, goal, K)
            if mpc is not None:
                lmc = monitor_logits(mpc, np_plain.asarray(hqc), goal)
                amc = _acc(lmc, y); pgmc = _acc_pg(lmc, y, goal, K)
            else:
                amc, pgmc = float("nan"), None
        else:
            ac, pgc, amc, pgmc = float("nan"), None, float("nan"), None
        try:
            xr, qr, yr, gr = make_batch_reminders(rng, vv, CFG["task"], eval_n, L,
                                                  _fit(L + 100 + 2 * max(0, (L - 1) // CFG["remind_every"])),
                                                  CFG["remind_every"], distractors=drate,
                                                  balance_goals=True)
            if xr.shape[1] > CFG["T"]: raise ValueError("too long")
            lr, _ = logits_at(p1, xr, qr, None, CFG["n_layer"], CFG["H"], False)
            ar = _acc(lr, yr); pgr = _acc_pg(lr, yr, gr, K)
        except ValueError:
            ar, pgr = float("nan"), None
        _append(csv_main, hdr, [L, a1, ar, am, a2, ac, amc, chance])
        for arm, pg in [("v1", pg1), ("v1_reminder", pgr), ("v1_monitor", pgm),
                        ("v2_anchor", pg2), ("v1c", pgc), ("monc", pgmc)]:
            if pg is not None:
                _append(csv_pg, hdr_pg, [sweep, L, arm] + pg)
        print(f"[{sweep}] L={L:4d}  v1={a1:.3f}  remind={ar:.3f}  mon={am:.3f}  "
              f"v2={a2:.3f}  v1c={ac:.3f}  monc={amc:.3f}")


def plot():
    import csv
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print("plot skipped:", e); return
    for suff in ["", "_diag"]:
        f = f"results/np_sweep_hop2{suff}_s{SEED}.csv"
        if not os.path.exists(f): continue
        rows = list(csv.DictReader(open(f)))
        rows.sort(key=lambda r: int(r["L"]))
        Ls = [int(r["L"]) for r in rows]
        plt.figure()
        for col, lab in [("v1", "v1"), ("v1_reminder", "v1+reminder"),
                         ("v1_monitor", "v1+monitor"), ("v2_anchor", "v2 (anchor)"),
                         ("v1c", "v1 curriculum"), ("monc", "monitor on v1c")]:
            ys = [float(r[col]) if r[col] not in ("", "nan") else float("nan") for r in rows]
            if all(y != y for y in ys): continue
            plt.plot(Ls, ys, marker="o", label=lab)
        plt.axhline(0.25, ls="--", c="gray", label="goal-ignorant ceiling")
        plt.axhline(1 / 8, ls=":", c="gray", label="chance")
        plt.axvline(CFG["train_max_len"], ls=":", c="k", alpha=.5, label="train max len")
        plt.xlabel("filler length L"); plt.ylabel("goal-conditioned accuracy")
        plt.title(f"exp05 hop2 seed {SEED}"
                  f" ({'15%' if suff == '' else '0%'} eval distractors)")
        plt.ylim(-0.02, 1.02); plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(f.replace(".csv", ".png"), dpi=130)
        print("wrote", f.replace(".csv", ".png"))


def status():
    for tag in ["v1", "v2", "v1c", "mon", "monc"]:
        st = _load(tag)
        if st:
            _, _, _, step, losses = st
            print(tag, "step", step,
                  "loss", round(losses[-1], 4) if losses else None,
                  "early" if _early_stop(losses, step) else "")
        else:
            print(tag, "not started")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "train_curr", "train_monitor",
                                    "eval", "plot", "status"])
    ap.add_argument("--model", default="v1")
    ap.add_argument("--features", default="v1", choices=["v1", "v1c"])
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cap", type=int, default=CFG["cap"])
    ap.add_argument("--sweep", default="primary", choices=["primary", "diag"])
    ap.add_argument("--lens", default="0,16,32,64,96,128,192,224,256")
    a = ap.parse_args()
    set_seed(a.seed)
    if a.cmd == "train": train(a.model, a.seconds, a.cap)
    elif a.cmd == "train_curr": train_curr(a.seconds, a.cap)
    elif a.cmd == "train_monitor": train_monitor(a.features, a.seconds, a.cap)
    elif a.cmd == "eval": evaluate(a.sweep, [int(x) for x in a.lens.split(",")])
    elif a.cmd == "plot": plot()
    else: status()
