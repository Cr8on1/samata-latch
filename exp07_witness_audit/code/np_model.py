"""
LACUNA V2 exp07 model — numpy/autograd. Backbone + heads are exp06's file
verbatim (v1/v2/vB/vBl modes kept for eval-only reuse of exp06 checkpoints);
adds Primitive C (witness) modes:

  vW  : v1 backbone -> witness head at qpos (K logits) -> K-way commitment
        record (weight-TIED to the goal-token embedding rows) -> answer MLP
        on [e_q ; record] ONLY. Causal bottleneck: all context/goal
        information reaching the answer flows through the K-way record.
        Soft commitment in training (softmax/tau), HARD argmax at eval.
        No goal input, no goal labels anywhere.
  vWo : single-axis ablation — identical witness path, answer MLP on
        [e_q ; record ; h_q] (bottleneck opened).
"""
import autograd.numpy as np
from autograd import grad
from autograd.extend import primitive, defvjp

@primitive
def _stop_grad(x): return x
defvjp(_stop_grad, lambda ans, x: lambda g: np.zeros_like(x))


def init_params(rng, V, n_ans, K, T, L_layers, H, D, mode):
    def rnd(*s): return rng.standard_normal(s) * 0.02
    p = {"tok": rnd(V, D), "pos": rnd(T, D),
         "head_w": rnd(D, n_ans), "head_b": np.zeros(n_ans)}
    for l in range(L_layers):
        p[f"{l}.qkv_w"] = rnd(D, 3 * D); p[f"{l}.qkv_b"] = np.zeros(3 * D)
        p[f"{l}.proj_w"] = rnd(D, D);    p[f"{l}.proj_b"] = np.zeros(D)
        p[f"{l}.fc1_w"] = rnd(D, 4 * D); p[f"{l}.fc1_b"] = np.zeros(4 * D)
        p[f"{l}.fc2_w"] = rnd(4 * D, D); p[f"{l}.fc2_b"] = np.zeros(D)
        p[f"{l}.ln1_g"] = np.ones(D); p[f"{l}.ln1_b"] = np.zeros(D)
        p[f"{l}.ln2_g"] = np.ones(D); p[f"{l}.ln2_b"] = np.zeros(D)
    p["lnf_g"] = np.ones(D); p["lnf_b"] = np.zeros(D)
    if mode == "v2":
        p["goal_reg"] = rnd(K, D)
    if mode in ("vB", "vBl"):
        p["B_reg"] = rnd(K, D)
    if mode == "vB":
        p["Wg"] = rnd(D, K)
        p["gate_a"] = np.array(4.0)
        p["gate_beta"] = np.array(-0.35)
    if mode in ("vW", "vWo"):
        p["Ww"] = rnd(D, K)                    # witness head at qpos
        din = 2 * D if mode == "vW" else 3 * D
        p["aw1"] = rnd(din, 96); p["ab1"] = np.zeros(96)
        p["aw2"] = rnd(96, n_ans); p["ab2"] = np.zeros(n_ans)
        # record vectors are p["tok"][goal_lo:goal_lo+K] — weight tied, no
        # separate register (pins slot semantics architecturally).
    return p


def _ln(x, g, b):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x); return e / e.sum(-1, keepdims=True)


def _gelu(x): return 0.5 * x * (1 + np.tanh(0.79788456 * (x + 0.044715 * x ** 3)))


def _softplus(x): return np.log1p(np.exp(x))


def _latch_ids(x_ids, apos, goal_lo):
    B = x_ids.shape[0]
    return x_ids[np.arange(B), apos + 1] - goal_lo


def _friction(p, x_ids, lgoal, K):
    e = p["tok"][x_ids]
    ell = e @ p["Wg"]
    onehot = np.eye(K)[lgoal]
    own = (ell * onehot[:, None, :]).sum(-1)
    rival = (ell - 1e9 * onehot[:, None, :]).max(-1)
    f = rival - own
    s = 1.0 / (1.0 + np.exp(-(p["gate_a"] - _softplus(p["gate_beta"]) * f)))
    return f, s


def backbone(p, x_ids, goal, apos, n_layer, H, mode, goal_lo, K):
    B, T = x_ids.shape
    D = p["tok"].shape[1]
    h = p["tok"][x_ids] + p["pos"][:T][None, :, :]
    g_vec = None; s = None
    if mode == "v2":
        g_vec = p["goal_reg"][goal][:, None, :]
    elif mode in ("vB", "vBl"):
        lgoal = _latch_ids(x_ids, apos, goal_lo)
        g_vec = p["B_reg"][lgoal][:, None, :]
        if mode == "vB":
            _, s = _friction(p, x_ids, lgoal, K)
    mask = np.tril(np.ones((T, T)))
    for l in range(n_layer):
        if g_vec is not None:
            h = h + g_vec
        z = _ln(h, p[f"{l}.ln1_g"], p[f"{l}.ln1_b"])
        qkv = z @ p[f"{l}.qkv_w"] + p[f"{l}.qkv_b"]
        q, k, v = qkv[..., :D], qkv[..., D:2*D], qkv[..., 2*D:]
        if s is not None:
            v = v * s[:, :, None]
        d = D // H
        def split(t): return np.transpose(t.reshape(B, T, H, d), (0, 2, 1, 3))
        q, k, v = split(q), split(k), split(v)
        att = q @ np.transpose(k, (0, 1, 3, 2)) / np.sqrt(d)
        att = np.where(mask[None, None] == 0, -1e9, att)
        att = _softmax(att)
        y = np.transpose(att @ v, (0, 2, 1, 3)).reshape(B, T, D)
        h = h + y @ p[f"{l}.proj_w"] + p[f"{l}.proj_b"]
        z = _ln(h, p[f"{l}.ln2_g"], p[f"{l}.ln2_b"])
        h = h + _gelu(z @ p[f"{l}.fc1_w"] + p[f"{l}.fc1_b"]) @ p[f"{l}.fc2_w"] + p[f"{l}.fc2_b"]
    return _ln(h, p["lnf_g"], p["lnf_b"])


def logits_at(p, x_ids, qpos, goal, apos, n_layer, H, mode, goal_lo=0, K=4):
    h = backbone(p, x_ids, goal, apos, n_layer, H, mode, goal_lo, K)
    hq = h[np.arange(h.shape[0]), qpos]
    return hq @ p["head_w"] + p["head_b"], hq


# ---- Primitive C: witness bottleneck forward ----
def witness_logits_at(p, x_ids, qpos, n_layer, H, mode, goal_lo, K, tau,
                      hard=False, gum=None):
    """Returns (answer_logits, witness_logits lw, hq). mode in {vW, vWo}.
    Backbone runs as pure v1 (no registers). Gradient reaches the backbone
    ONLY through lw (vW) — the causal bottleneck. hard=True is eval-only
    (deterministic argmax record, no noise, no gradient). Training path
    (hard=False) = Gumbel straight-through: gum is (B,K) Gumbel(0,1) noise
    sampled by the caller; forward commits to argmax(lw+gum), backward flows
    through softmax((lw+gum)/tau). (Amendment 1: plain tau-anneal soft
    commitment starved the witness; plain ST collapsed to one slot.)"""
    B = x_ids.shape[0]; idx = np.arange(B)
    h = backbone(p, x_ids, None, None, n_layer, H, "v1", goal_lo, K)
    hq = h[idx, qpos]
    lw = hq @ p["Ww"]                              # (B,K)
    Eg = p["tok"][goal_lo:goal_lo + K]             # (K,D) weight-tied record
    if hard:
        import numpy as _n
        r = Eg[_n.asarray(lw).argmax(-1)]
    else:
        import numpy as _n
        z = lw + gum if gum is not None else lw
        w = _softmax(z / tau)
        onehot = np.eye(K)[_n.asarray(z).argmax(-1)]
        w = w + _stop_grad(onehot - w)
        r = w @ Eg
    eq = p["tok"][x_ids[idx, qpos]]                # raw query embedding
    parts = [eq, r] + ([hq] if mode == "vWo" else [])
    z = np.concatenate(parts, axis=-1)
    a1 = np.tanh(z @ p["aw1"] + p["ab1"])
    return a1 @ p["aw2"] + p["ab2"], lw, hq


def loss_fn_w(p, x_ids, qpos, y, n_layer, H, mode, goal_lo, K, tau,
              gum=None, lb=0.0):
    lg, lw, _ = witness_logits_at(p, x_ids, qpos, n_layer, H, mode,
                                  goal_lo, K, tau, hard=False, gum=gum)
    lg = lg - lg.max(-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    ce = -logp[np.arange(y.shape[0]), y].mean()
    if lb:
        z = lw + gum if gum is not None else lw
        pbar = _softmax(z / tau).mean(0)          # batch-mean soft commitment
        kl_u = np.log(K * 1.0) + (pbar * np.log(pbar + 1e-9)).sum()
        ce = ce + lb * kl_u                        # load balance to uniform
    return ce


def loss_fn(p, x_ids, qpos, y, goal, apos, n_layer, H, mode, goal_lo=0, K=4):
    lg, _ = logits_at(p, x_ids, qpos, goal, apos, n_layer, H, mode, goal_lo, K)
    lg = lg - lg.max(-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    return -logp[np.arange(y.shape[0]), y].mean()


def adam_step(p, g, m, v, t, lr):
    b1, b2, eps = 0.9, 0.999, 1e-8
    out = {}
    for k in p:
        m[k] = b1 * m[k] + (1 - b1) * g[k]
        v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
        mh = m[k] / (1 - b1 ** t); vh = v[k] / (1 - b2 ** t)
        out[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)
    return out, m, v


# ---- exp06 heads, kept verbatim for eval-only checkpoint reuse ----
def init_monitor(rng, K, D, n_ans, hidden=96):
    r = lambda *s: rng.standard_normal(s) * 0.05
    return {"gem": r(K, D),
            "w1": r(2 * D, hidden), "b1": np.zeros(hidden),
            "w2": r(hidden, hidden), "b2": np.zeros(hidden),
            "w3": r(hidden, n_ans), "b3": np.zeros(n_ans)}


def monitor_logits(mp, hq, goal):
    z = np.concatenate([hq, mp["gem"][goal]], axis=-1)
    h = np.tanh(z @ mp["w1"] + mp["b1"])
    h = np.tanh(h @ mp["w2"] + mp["b2"])
    return h @ mp["w3"] + mp["b3"]


def init_probe(rng, D, K, hidden=96):
    r = lambda *s: rng.standard_normal(s) * 0.05
    return {"w1": r(D, hidden), "b1": np.zeros(hidden),
            "w2": r(hidden, K), "b2": np.zeros(K)}


def probe_logits(pp, hq):
    h = np.tanh(hq @ pp["w1"] + pp["b1"])
    return h @ pp["w2"] + pp["b2"]


grad_loss = grad(loss_fn)
grad_loss_w = grad(loss_fn_w)


# ---- Amendment 2: RETROFIT witness (vWr) — bottleneck heads on a FROZEN,
# competent v1 backbone (exp06 checkpoint). Trainable: Ww + answer MLP only.
# Label-free (answer CE through the commitment), causal (answer sees goal
# info only via the K-way record), same Gumbel-ST commitment as vW.
def init_retro(rng, D, K, n_ans, hidden=96):
    r = lambda *s: rng.standard_normal(s) * 0.05
    return {"Ww": r(D, K),
            "aw1": r(2 * D, hidden), "ab1": np.zeros(hidden),
            "aw2": r(hidden, n_ans), "ab2": np.zeros(n_ans)}


def retro_logits(rp, hq, eq, Eg, tau, hard=False, gum=None):
    """hq/eq/Eg are FROZEN v1 features (query-pos state, raw query embedding,
    goal-token embedding rows). Returns (answer_logits, witness_logits)."""
    lw = hq @ rp["Ww"]
    if hard:
        import numpy as _n
        r = Eg[_n.asarray(lw).argmax(-1)]
    else:
        import numpy as _n
        z = lw + gum if gum is not None else lw
        w = _softmax(z / tau)
        onehot = np.eye(Eg.shape[0])[_n.asarray(z).argmax(-1)]
        w = w + _stop_grad(onehot - w)
        r = w @ Eg
    zin = np.concatenate([eq, r], axis=-1)
    a1 = np.tanh(zin @ rp["aw1"] + rp["ab1"])
    return a1 @ rp["aw2"] + rp["ab2"], lw


def loss_fn_retro(rp, hq, eq, Eg, y, tau, gum=None, lb=0.0):
    lg, lw = retro_logits(rp, hq, eq, Eg, tau, hard=False, gum=gum)
    lg = lg - lg.max(-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    ce = -logp[np.arange(y.shape[0]), y].mean()
    if lb:
        z = lw + gum if gum is not None else lw
        pbar = _softmax(z / tau).mean(0)
        K = Eg.shape[0]
        ce = ce + lb * (np.log(K * 1.0) + (pbar * np.log(pbar + 1e-9)).sum())
    return ce


grad_retro = grad(loss_fn_retro)
