"""
LACUNA V2 exp06 model — numpy/autograd, extends the exp03 build with
Primitive B (friction). Modes:

  v1  : standard tiny transformer (exp01-03 class)
  v2  : + anchored goal register from GROUND-TRUTH goal id (Primitive A,
        oracle out-of-band channel) — positive control
  vBl : + COMMITMENT LATCH only: register keyed by the payload token at the
        FIRST assignment marker (apos+1), read from the token stream, added
        to the residual stream every layer. Write-once by construction.
  vB  : vBl + FRICTION GATE: per-token goal-claim logits l_t = e_t @ Wg;
        friction signal f_t = max_{g != ghat} l_t[g] - l_t[ghat];
        gate s_t = sigmoid(a - softplus(beta) * f_t) multiplies position t's
        value vectors in every attention layer.

No ground-truth goal reaches vB/vBl at any point — the latched id is the
token the model can see at apos+1 in its own input.
"""
import autograd.numpy as np
from autograd import grad


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
        p["gate_a"] = np.array(4.0)        # gate open at init (sigmoid(4)=0.982)
        p["gate_beta"] = np.array(-0.35)   # softplus(-0.35) ~ 0.53
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
    """Latched goal id = payload token at the first assignment marker."""
    B = x_ids.shape[0]
    return x_ids[np.arange(B), apos + 1] - goal_lo


def _friction(p, x_ids, lgoal, K):
    """Friction signal f (B,T) and gate s (B,T) from token embeddings."""
    e = p["tok"][x_ids]                              # (B,T,D)
    ell = e @ p["Wg"]                                # (B,T,K)
    onehot = np.eye(K)[lgoal]                        # (B,K)
    own = (ell * onehot[:, None, :]).sum(-1)         # (B,T)
    rival = (ell - 1e9 * onehot[:, None, :]).max(-1) # (B,T)
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
            h = h + g_vec                       # re-inject every layer
        z = _ln(h, p[f"{l}.ln1_g"], p[f"{l}.ln1_b"])
        qkv = z @ p[f"{l}.qkv_w"] + p[f"{l}.qkv_b"]
        q, k, v = qkv[..., :D], qkv[..., D:2*D], qkv[..., 2*D:]
        if s is not None:
            v = v * s[:, :, None]               # friction-damped value rows
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


def loss_fn(p, x_ids, qpos, y, goal, apos, n_layer, H, mode, goal_lo=0, K=4):
    lg, _ = logits_at(p, x_ids, qpos, goal, apos, n_layer, H, mode, goal_lo, K)
    lg = lg - lg.max(-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    return -logp[np.arange(y.shape[0]), y].mean()


def friction_trace(p, x_ids, apos, goal_lo, K):
    """Diagnostic (plain forward): per-token friction signal f and gate s."""
    lgoal = _latch_ids(x_ids, apos, goal_lo)
    return _friction(p, x_ids, lgoal, K)


def adam_step(p, g, m, v, t, lr):
    b1, b2, eps = 0.9, 0.999, 1e-8
    out = {}
    for k in p:
        m[k] = b1 * m[k] + (1 - b1) * g[k]
        v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
        mh = m[k] / (1 - b1 ** t); vh = v[k] / (1 - b2 ** t)
        out[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)
    return out, m, v


# ---- external monitor (oracle wrapper baseline, exp03-identical) ----
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


def monitor_loss(mp, hq, goal, y):
    lg = monitor_logits(mp, hq, goal)
    lg = lg - lg.max(-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    return -logp[np.arange(y.shape[0]), y].mean()


# ---- assigned-goal probe (NON-oracle wrapper: frozen features -> g) ----
def init_probe(rng, D, K, hidden=96):
    r = lambda *s: rng.standard_normal(s) * 0.05
    return {"w1": r(D, hidden), "b1": np.zeros(hidden),
            "w2": r(hidden, K), "b2": np.zeros(K)}


def probe_logits(pp, hq):
    h = np.tanh(hq @ pp["w1"] + pp["b1"])
    return h @ pp["w2"] + pp["b2"]


def probe_loss(pp, hq, g):
    lg = probe_logits(pp, hq)
    lg = lg - lg.max(-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    return -logp[np.arange(g.shape[0]), g].mean()


grad_loss = grad(loss_fn)
grad_monitor = grad(monitor_loss)
grad_probe = grad(probe_loss)
