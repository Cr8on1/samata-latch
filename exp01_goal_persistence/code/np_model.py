"""
Pure numpy/autograd version of the LACUNA V2 ablation (no torch needed).
Same experiment as model.py/run.py, sized down to train on CPU.
v1: standard tiny transformer. v2: + anchored goal register injected at
every layer (primitive A). External monitor: frozen-v1 features + goal -> head.
"""
import autograd.numpy as np
from autograd import grad


def init_params(rng, V, n_ans, K, T, L_layers, H, D, anchor):
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
    if anchor:
        p["goal_reg"] = rnd(K, D)
    return p


def _ln(x, g, b):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x); return e / e.sum(-1, keepdims=True)


def _gelu(x): return 0.5 * x * (1 + np.tanh(0.79788456 * (x + 0.044715 * x ** 3)))


def backbone(p, x_ids, goal, n_layer, H, anchor):
    B, T = x_ids.shape
    D = p["tok"].shape[1]
    h = p["tok"][x_ids] + p["pos"][:T][None, :, :]
    g_vec = p["goal_reg"][goal][:, None, :] if anchor else None
    mask = np.tril(np.ones((T, T)))
    for l in range(n_layer):
        if anchor:
            h = h + g_vec                       # re-inject every layer
        z = _ln(h, p[f"{l}.ln1_g"], p[f"{l}.ln1_b"])
        qkv = z @ p[f"{l}.qkv_w"] + p[f"{l}.qkv_b"]
        q, k, v = qkv[..., :D], qkv[..., D:2*D], qkv[..., 2*D:]
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


def logits_at(p, x_ids, qpos, goal, n_layer, H, anchor):
    h = backbone(p, x_ids, goal, n_layer, H, anchor)
    hq = h[np.arange(h.shape[0]), qpos]
    return hq @ p["head_w"] + p["head_b"], hq


def loss_fn(p, x_ids, qpos, y, goal, n_layer, H, anchor):
    lg, _ = logits_at(p, x_ids, qpos, goal, n_layer, H, anchor)
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


# ---- external monitor (wrapper baseline) ----
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


grad_loss = grad(loss_fn)
grad_monitor = grad(monitor_loss)
