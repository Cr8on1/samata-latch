"""
LACUNA V2 - Experiment #5: goal persistence under context growth.

Controlled ablation of primitive A (anchored goal channel). Four arms,
identical task, measured as goal-conditioned accuracy vs. filler length L:

    v1                 standard transformer (goal only as a start token)
    v1 + reminder      surface fix: re-insert the goal every R tokens (no retrain)
    v1 + monitor       external wrapper: frozen v1 + bolt-on head given the goal
    v2 (anchor)        internal anchored goal channel, re-injected every layer

Discipline rule: primitive A only "counts" where the reminder AND the monitor
fail to match v2. Pre-registered expectation (see README):
  - task=shift (goal used only at the end): monitor is expected to be COMPETITIVE.
    That is a legitimate finding: for end-only goal use, a wrapper suffices.
  - task=hard  (goal gates intermediate computation): the wall should appear -
    reminder decays between reminders / out of range, monitor can't inject
    mid-computation, v2 stays flat. That gap is the structural result.

Usage:
    python run.py all          # train everything + sweep + plot (default tiny-ish)
    python run.py train
    python run.py eval
    python run.py smoke        # 30-second CPU sanity run
Flags: --task {shift,hard} --device {auto,cpu,cuda} plus sizes; see bottom.
"""

import argparse, os, csv, math
import numpy as np

CKPT = "checkpoints"
OUT = "results"


def get_device(pref):
    import torch
    if pref == "cpu": return "cpu"
    if pref == "cuda": return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build(cfg_args, anchor):
    from model import GPT, GPTConfig
    from data import Vocab
    v = Vocab(K=cfg_args.n_goals, V_ans=cfg_args.n_ans, V_filler=cfg_args.n_filler)
    gc = GPTConfig(vocab_size=v.size, n_ans=v.V_ans, n_goals=v.K,
                   block_size=cfg_args.block_size, n_layer=cfg_args.n_layer,
                   n_head=cfg_args.n_head, n_embd=cfg_args.n_embd, anchor=anchor)
    return GPT(gc), v


def _to(t, device):
    import torch
    return torch.from_numpy(t).to(device)


def train_model(args, anchor, tag):
    import torch
    from data import make_batch
    device = get_device(args.device)
    model, v = build(args, anchor)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed + (1 if anchor else 0))
    model.train()
    for step in range(args.steps):
        L = int(rng.integers(0, args.train_max_len + 1))
        x, qpos, y, goal = make_batch(rng, v, args.task, args.batch, L,
                                      args.block_size, distractors=args.distractors)
        x, qpos, y, goal = _to(x, device), _to(qpos, device), _to(y, device), _to(goal, device)
        logits, _ = model(x, qpos, goal=goal if anchor else None)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, args.steps // 8) == 0 or step == args.steps - 1:
            print(f"[{tag}] step {step:5d}  loss {loss.item():.4f}")
    os.makedirs(CKPT, exist_ok=True)
    torch.save(model.state_dict(), f"{CKPT}/{tag}.pt")
    print(f"[{tag}] params={model.num_params():,}  saved -> {CKPT}/{tag}.pt")
    return model, v


def train_monitor(args, v1_model, v):
    import torch
    from data import make_batch
    from model import ExternalMonitor
    device = get_device(args.device)
    mon = ExternalMonitor(args.n_embd, v.K, v.V_ans).to(device)
    opt = torch.optim.AdamW(mon.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed + 7)
    v1_model.eval()
    for step in range(args.steps):
        L = int(rng.integers(0, args.train_max_len + 1))
        x, qpos, y, goal = make_batch(rng, v, args.task, args.batch, L,
                                      args.block_size, distractors=args.distractors)
        x, qpos, y, goal = _to(x, device), _to(qpos, device), _to(y, device), _to(goal, device)
        with torch.no_grad():
            _, hq = v1_model(x, qpos)          # frozen v1 features
        logits = mon(hq, goal)                 # wrapper gets oracle goal
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    os.makedirs(CKPT, exist_ok=True)
    torch.save(mon.state_dict(), f"{CKPT}/monitor.pt")
    print(f"[monitor] trained on frozen v1 -> {CKPT}/monitor.pt")
    return mon


def _acc(logits, y):
    import torch
    return (logits.argmax(-1) == y).float().mean().item()


def eval_sweep(args):
    import torch
    from data import make_batch, make_batch_reminders
    from model import ExternalMonitor
    device = get_device(args.device)

    v1, v = build(args, anchor=False); v1.load_state_dict(torch.load(f"{CKPT}/v1.pt", map_location=device)); v1.to(device).eval()
    v2, _ = build(args, anchor=True);  v2.load_state_dict(torch.load(f"{CKPT}/v2.pt", map_location=device)); v2.to(device).eval()
    mon = ExternalMonitor(args.n_embd, v.K, v.V_ans).to(device)
    mon.load_state_dict(torch.load(f"{CKPT}/monitor.pt", map_location=device)); mon.eval()

    rng = np.random.default_rng(args.seed + 999)
    chance = 1.0 / v.V_ans
    lengths = [int(x) for x in args.eval_lens.split(",")]
    rows = []
    print(f"\nfiller_len | v1     v1+remind  v1+monitor  v2(anchor)   (chance={chance:.3f})")
    for L in lengths:
        # shared batch for v1 / monitor / v2; reminder needs its own (longer) build
        x, qpos, y, goal = make_batch(rng, v, args.task, args.eval_n, L, args.block_size,
                                      distractors=args.distractors)
        xt, qt, yt, gt = _to(x, device), _to(qpos, device), _to(y, device), _to(goal, device)
        with torch.no_grad():
            l_v1, hq = v1(xt, qt)
            a_v1 = _acc(l_v1, yt)
            a_mon = _acc(mon(hq, gt), yt)
            a_v2 = _acc(v2(xt, qt, goal=gt)[0], yt)
            # reminder arm (own batch; may need bigger block_size at large L)
            try:
                xr, qr, yr, gr = make_batch_reminders(rng, v, args.task, args.eval_n, L,
                                                       args.block_size, args.remind_every,
                                                       distractors=args.distractors)
                lr, _ = v1(_to(xr, device), _to(qr, device))
                a_rem = _acc(lr, _to(yr, device))
            except ValueError:
                a_rem = float("nan")   # reminders overflowed block_size at this L
        rows.append((L, a_v1, a_rem, a_mon, a_v2))
        print(f"{L:9d}  | {a_v1:.3f}   {a_rem:.3f}      {a_mon:.3f}       {a_v2:.3f}")

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/sweep_{args.task}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filler_len", "v1", "v1_reminder", "v1_monitor", "v2_anchor", "chance"])
        for L, a, b, c, d in rows: w.writerow([L, a, b, c, d, chance])
    print(f"\nwrote {OUT}/sweep_{args.task}.csv")
    _plot(rows, chance, args)


def _plot(rows, chance, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable, skipping plot: {e})"); return
    Ls = [r[0] for r in rows]
    labels = ["v1", "v1+reminder", "v1+monitor", "v2 (anchor)"]
    for i, lab in enumerate(labels):
        plt.plot(Ls, [r[i + 1] for r in rows], marker="o", label=lab)
    plt.axhline(chance, ls="--", c="gray", label="chance")
    plt.axvline(args.train_max_len, ls=":", c="k", alpha=.5, label="train max len")
    plt.xlabel("filler length L (goal->query distance)")
    plt.ylabel("goal-conditioned accuracy")
    plt.title(f"Goal persistence under context growth  (task={args.task})")
    plt.ylim(0, 1.02); plt.legend(); plt.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/sweep_{args.task}.png"; plt.savefig(p, dpi=130); print(f"wrote {p}")


def cmd_train(args):
    v1, v = train_model(args, anchor=False, tag="v1")
    train_monitor(args, v1, v)
    train_model(args, anchor=True, tag="v2")


def cmd_all(args):
    cmd_train(args); eval_sweep(args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "eval", "all", "smoke"])
    p.add_argument("--task", default="shift", choices=["shift", "hard"])
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    # task / vocab
    p.add_argument("--n_goals", type=int, default=8)
    p.add_argument("--n_ans", type=int, default=16)
    p.add_argument("--n_filler", type=int, default=64)
    p.add_argument("--distractors", action="store_true")
    # model
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--block_size", type=int, default=768)
    # train
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train_max_len", type=int, default=96)
    # eval
    p.add_argument("--eval_n", type=int, default=512)
    p.add_argument("--eval_lens", default="0,16,32,64,96,128,192,256,384")
    p.add_argument("--remind_every", type=int, default=24)
    args = p.parse_args()

    if args.cmd == "smoke":
        # tiny, fast, CPU-friendly end-to-end check
        args.device = "cpu"; args.steps = 60; args.batch = 16; args.n_embd = 32
        args.n_layer = 2; args.n_head = 2; args.block_size = 256; args.train_max_len = 48
        args.eval_n = 64; args.eval_lens = "0,32,96,192"
        cmd_all(args); return
    {"train": cmd_train, "eval": eval_sweep, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
