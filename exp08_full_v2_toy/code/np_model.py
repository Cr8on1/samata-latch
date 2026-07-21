"""
LACUNA V2 exp08 model — numpy/autograd. The stack (A + B-latch + C-readout)
and its single-axis ablation:

  vS  : v1 backbone + COMMITMENT LATCH register: keyed by the payload token
        at the FIRST assignment marker (apos+1), read from the token stream,
        re-injected into the residual stream at EVERY layer (A's discipline),
        write-once by construction (B's keying). Architecturally identical to
        exp06's vBl — no gate (dead weight, exp06). C is NOT in the forward
        pass: the witness is the symbolic latch readout w_hat = token at
        apos+1, computed at eval time with zero parameters.
  vSo : SINGLE-AXIS ablation — identical except the register is added ONCE,
        to the embedding stream before layer 0, and never re-injected.
        Tests whether A's per-layer re-injection earns its keep in the stack.

No ground-truth goal reaches vS/vSo at any point — the latched id is the
token the model can see at apos+1 in its own input.

Determinism note (H-byte-replication): init_params draws base params then
B_reg in exp06's exact order, so with exp06's rng offsets (ARM_OFF=3) vS
init is bit-identical to exp06 vBl init.
"""
import autograd.numpy as np
from autograd import grad

STACK_MODES = ("vS", "vSo")


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
    if mode in STACK_MODES:
        p["B_reg"] = rnd(K, D)
    return p


def _ln(x, g, b):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x); return e / e.sum(-1, keepdims=True)


def _gelu(x): return 0.5 * x * (1 + np.tanh(0.79788456 * (x + 0.044715 * x ** 3)))


def latch_ids(x_ids, apos, goal_lo):
    """Latched goal id = payload token at the first assignment marker.
    This IS the witness readout (C): symbolic, zero parameters."""
    B = x_ids.shape[0]
    return x_ids[np.arange(B), apos + 1] - goal_lo


def backbone(p, x_ids, goal, apos, n_layer, H, mode, goal_lo, K):
    B, T = x_ids.shape
    D = p["tok"].shape[1]
    h = p["tok"][x_ids] + p["pos"][:T][None, :, :]
    lgoal = latch_ids(x_ids, apos, goal_lo)
    g_vec = p["B_reg"][lgoal][:, None, :]
    if mode == "vSo":
        h = h + g_vec                           # once, at input — never again
    mask = np.tril(np.ones((T, T)))
    for l in range(n_layer):
        if mode != "vSo":
            h = h + g_vec                       # re-inject every layer (A)
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


def logits_at(p, x_ids, qpos, goal, apos, n_layer, H, mode, goal_lo=0, K=4):
    h = backbone(p, x_ids, goal, apos, n_layer, H, mode, goal_lo, K)
    hq = h[np.arange(h.shape[0]), qpos]
    return hq @ p["head_w"] + p["head_b"], hq


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


grad_loss = grad(loss_fn)
