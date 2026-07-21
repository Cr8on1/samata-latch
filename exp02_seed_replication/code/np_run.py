"""
Chunked CPU runner: trains in short bursts (checkpointing to .npz) so it can
run inside limited shell calls. Same 4-arm experiment as run.py.

  python np_run.py train --model v1|v2 --seconds 30 --seed N
  python np_run.py train_monitor --seconds 30
  python np_run.py eval
  python np_run.py status
"""
import argparse, json, os, time
import numpy as np_plain
import autograd.numpy as np
from data import Vocab, make_batch, make_batch_reminders
from np_model import (init_params, loss_fn, grad_loss, adam_step, logits_at,
                      init_monitor, monitor_logits, monitor_loss, grad_monitor)

# ---- fixed experiment config (small enough for CPU) ----
CFG = dict(K=4, V_ans=8, V_filler=32, n_layer=2, H=2, D=48,
           T=288, train_max_len=64, batch=32, lr=1e-3, task="shift",
           remind_every=16)
SEED = 1; OFF = 10000; CK = "np_ckpt_s1"

def set_seed(s):
    global SEED, OFF, CK
    SEED = int(s); OFF = 10000 * SEED; CK = f"np_ckpt_s{SEED}"
    os.makedirs(CK, exist_ok=True)


def vocab(): return Vocab(K=CFG["K"], V_ans=CFG["V_ans"], V_filler=CFG["V_filler"])


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


def train(tag, seconds):
    anchor = (tag == "v2")
    vv = vocab(); rng = np_plain.random.default_rng(OFF + (0 if not anchor else 1))
    st = _load(tag)
    if st is None:
        p = init_params(np_plain.random.default_rng(OFF + 42 + anchor), vv.size,
                        vv.V_ans, vv.K, CFG["T"], CFG["n_layer"], CFG["H"],
                        CFG["D"], anchor)
        m = {k: np_plain.zeros_like(p[k]) for k in p}
        v = {k: np_plain.zeros_like(p[k]) for k in p}
        step, losses = 0, []
    else:
        p, m, v, step, losses = st
        # fast-forward rng to keep data stream deterministic-ish per restart
        rng = np_plain.random.default_rng(OFF + 1000 + step + (1 if anchor else 0))
    t0 = time.time()
    while time.time() - t0 < seconds:
        L = int(rng.integers(0, CFG["train_max_len"] + 1))
        bs = int(rng.choice([68, 112, 160, 208, 256, 288],
                            p=[0.35, 0.2, 0.15, 0.12, 0.1, 0.08]))
        bs = max(bs, L + 4)
        b = max(8, int(CFG["batch"] * 68 / bs))   # keep step cost roughly flat
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], b, L, bs, distractors=True)
        g = grad_loss(p, x, qpos, y, goal, CFG["n_layer"], CFG["H"], anchor)
        step += 1
        p, m, v = adam_step(p, g, m, v, step, CFG["lr"])
        if step % 10 == 0:
            l = float(loss_fn(p, x, qpos, y, goal, CFG["n_layer"], CFG["H"], anchor))
            losses.append(l)
    _save(tag, p, m, v, step, losses)
    last = losses[-1] if losses else float("nan")
    print(json.dumps({"tag": tag, "step": step, "last_loss": round(last, 4)}))


def train_monitor(seconds):
    vv = vocab()
    st = _load("v1"); assert st, "train v1 first"
    p1 = st[0]
    stm = _load("mon")
    rng = np_plain.random.default_rng(OFF + 7)
    if stm is None:
        mp = init_monitor(np_plain.random.default_rng(OFF + 9), vv.K, CFG["D"], vv.V_ans)
        m = {k: np_plain.zeros_like(mp[k]) for k in mp}
        v = {k: np_plain.zeros_like(mp[k]) for k in mp}
        step, losses = 0, []
    else:
        mp, m, v, step, losses = stm
        rng = np_plain.random.default_rng(OFF + 7000 + step)
    t0 = time.time()
    while time.time() - t0 < seconds:
        L = int(rng.integers(0, CFG["train_max_len"] + 1))
        bs = int(rng.choice([68, 112, 160, 208, 256, 288],
                            p=[0.35, 0.2, 0.15, 0.12, 0.1, 0.08]))
        bs = max(bs, L + 4)
        b = max(8, int(CFG["batch"] * 68 / bs))
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], b, L, bs, distractors=True)
        _, hq = logits_at(p1, x, qpos, None, CFG["n_layer"], CFG["H"], False)
        hq = np_plain.asarray(hq)          # frozen features
        g = grad_monitor(mp, hq, goal, y)
        step += 1
        mp, m, v = adam_step(mp, g, m, v, step, CFG["lr"])
        if step % 10 == 0:
            losses.append(float(monitor_loss(mp, hq, goal, y)))
    _save("mon", mp, m, v, step, losses)
    print(json.dumps({"tag": "mon", "step": step,
                      "last_loss": round(losses[-1] if losses else float('nan'), 4)}))


def _acc(lg, y): return float((lg.argmax(-1) == y).mean())


def evaluate(eval_n=192, lens=(0, 16, 32, 64, 96, 128, 192, 224, 256)):
    vv = vocab()
    p1 = _load("v1")[0]; p2 = _load("v2")[0]; mp = _load("mon")[0]
    rng = np_plain.random.default_rng(OFF + 999)
    chance = 1.0 / vv.V_ans
    rows = []
    SIZES = [68, 112, 160, 208, 256, 288]      # the exact trained window sizes
    def fit(n):
        for b in SIZES:
            if b >= n: return b
        raise ValueError("too long")
    for L in lens:
        bs = fit(L + 4)
        x, qpos, y, goal = make_batch(rng, vv, CFG["task"], eval_n, L, bs, distractors=True)
        l1, hq = logits_at(p1, x, qpos, None, CFG["n_layer"], CFG["H"], False)
        a1 = _acc(l1, y)
        amon = _acc(monitor_logits(mp, np_plain.asarray(hq), goal), y)
        l2, _ = logits_at(p2, x, qpos, goal, CFG["n_layer"], CFG["H"], True)
        a2 = _acc(l2, y)
        try:
            rem_len = L + 4 + 2 * ((L - 1) // CFG["remind_every"] + 1 if L > 0 else 0)
            bs_r = fit(rem_len)
            xr, qr, yr, gr = make_batch_reminders(rng, vv, CFG["task"], eval_n, L,
                                                   bs_r, CFG["remind_every"], distractors=True)
            if xr.shape[1] > CFG["T"]: raise ValueError("too long")
            lr, _ = logits_at(p1, xr, qr, None, CFG["n_layer"], CFG["H"], False)
            ar = _acc(lr, yr)
        except ValueError:
            ar = float("nan")
        rows.append([L, a1, ar, amon, a2])
        print(f"L={L:4d}  v1={a1:.3f}  remind={ar:.3f}  monitor={amon:.3f}  v2={a2:.3f}")
    os.makedirs("results", exist_ok=True)
    import csv
    with open(f"results/np_sweep_s{SEED}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["L", "v1", "v1_reminder", "v1_monitor", "v2_anchor", "chance"])
        for r in rows: w.writerow(r + [chance])
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        Ls = [r[0] for r in rows]
        for i, lab in enumerate(["v1", "v1+reminder", "v1+monitor", "v2 (anchor)"]):
            plt.plot(Ls, [r[i+1] for r in rows], marker="o", label=lab)
        plt.axhline(chance, ls="--", c="gray", label="chance")
        plt.axvline(CFG["train_max_len"], ls=":", c="k", alpha=.5, label="train max len")
        plt.xlabel("filler length L"); plt.ylabel("goal-conditioned accuracy")
        plt.title(f"Goal persistence under context growth (task={CFG['task']}, CPU build)")
        plt.ylim(0, 1.02); plt.legend(); plt.tight_layout()
        plt.savefig(f"results/np_sweep_s{SEED}.png", dpi=130)
        print(f"wrote results/np_sweep_s{SEED}.csv")
    except Exception as e:
        print("plot skipped:", e)


def status():
    for tag in ["v1", "v2", "mon"]:
        st = _load(tag)
        if st: print(tag, "step", st[3], "loss", round(st[4][-1], 4) if st[4] else None)
        else: print(tag, "not started")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "train_monitor", "eval", "status"])
    ap.add_argument("--model", default="v1")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    set_seed(a.seed)
    if a.cmd == "train": train(a.model, a.seconds)
    elif a.cmd == "train_monitor": train_monitor(a.seconds)
    elif a.cmd == "eval": evaluate()
    else: status()
