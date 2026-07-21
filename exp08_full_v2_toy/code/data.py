"""
Synthetic 'goal persistence under context growth' task.

Sequence layout (single example):

    [GOAL] g  filler_1 ... filler_L  [QUERY] q   ->  target a

where:
    g in {0..K-1}           the assigned goal (selects a rule)
    filler_i                goal-irrelevant tokens (distractor context)
    q in {0..V_ans-1}       the query operand
    a = (q + g) mod V_ans   the goal-conditioned answer   (task='shift')

The model must (a) still know g after L filler tokens and (b) apply the
rule g to q. As L grows, a model that reconstructs g from context each
forward pass loses it (attention disperses); a model with an anchored
goal channel does not. That gap is the experiment.

Pure Python/numpy so it can be unit-tested without torch.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Vocab:
    K: int = 8          # number of goals
    V_ans: int = 16     # answer/operand alphabet size
    V_filler: int = 64  # filler alphabet size

    def __post_init__(self):
        self.GOAL_MARK = 0
        self.QUERY_MARK = 1
        self.PAD = 2
        base = 3
        self.goal_lo = base;                self.goal_hi = base + self.K
        self.ans_lo = self.goal_hi;         self.ans_hi = self.ans_lo + self.V_ans
        self.fill_lo = self.ans_hi;         self.fill_hi = self.fill_lo + self.V_filler
        self.size = self.fill_hi

    def goal_tok(self, g):  return self.goal_lo + g
    def ans_tok(self, a):   return self.ans_lo + a
    def fill_tok(self, f):  return self.fill_lo + f


def rule_answer(task, g, q, V_ans):
    """Deterministic goal-conditioned answer. Add variants here."""
    if task == "shift":
        return (q + g) % V_ans
    if task == "hard":
        # goal gates a two-step computation: harder for an end-only wrapper
        return ((q + g) * (g + 1)) % V_ans
    raise ValueError(task)


def _drate(distractors):
    """Distractor arg: False/None -> 0.0, True -> 0.15 (legacy), float -> rate.
    (exp02b recipe, ported for the v1c curriculum contingency. RNG draw
    pattern identical to the legacy path for True/False inputs.)"""
    if distractors is True:  return 0.15
    if not distractors:      return 0.0
    return float(distractors)


def make_example(rng, vocab, task, filler_len, distractors=False):
    """Return (tokens list, query_position index, target answer id 0..V_ans-1)."""
    v = vocab; rate = _drate(distractors)
    g = int(rng.integers(v.K))
    q = int(rng.integers(v.V_ans))
    a = rule_answer(task, g, q, v.V_ans)

    toks = [v.GOAL_MARK, v.goal_tok(g)]
    for _ in range(filler_len):
        if rate and rng.random() < rate:
            gd = int(rng.integers(v.K))
            toks.append(v.goal_tok(gd))          # plausible-but-wrong goal-like distractor
        else:
            toks.append(v.fill_tok(int(rng.integers(v.V_filler))))
    toks.append(v.QUERY_MARK)
    toks.append(v.ans_tok(q))                    # operand reuses answer alphabet
    query_pos = len(toks) - 1
    return toks, query_pos, a


def make_batch(rng, vocab, task, batch_size, filler_len, block_size,
               distractors=False):
    """
    Left-padded fixed-length batch so the query sits at a stable right edge and
    goal->query distance == filler_len regardless of padding.
    Returns x[B,T], qpos[B], y[B], goal[B].
    """
    v = vocab
    xs, qs, ys, gs = [], [], [], []
    for _ in range(batch_size):
        toks, qpos, a = make_example(rng, vocab, task, filler_len, distractors)
        g = toks[1] - v.goal_lo
        if len(toks) > block_size:
            raise ValueError(f"seq len {len(toks)} > block_size {block_size}; "
                             f"raise block_size or lower filler_len")
        pad = block_size - len(toks)
        row = [v.PAD] * pad + toks
        xs.append(row); qs.append(pad + qpos); ys.append(a); gs.append(g)
    return (np.array(xs, dtype=np.int64),
            np.array(qs, dtype=np.int64),
            np.array(ys, dtype=np.int64),
            np.array(gs, dtype=np.int64))


def goal_ids_array(vocab):
    """Token ids encoding each goal g=0..K-1 (for anchored-channel lookup)."""
    return np.array([vocab.goal_tok(g) for g in range(vocab.K)], dtype=np.int64)


def make_batch_reminders(rng, vocab, task, batch_size, filler_len, block_size,
                         remind_every, distractors=False):
    """
    SURFACE-FIX arm: identical task, but the goal [GOAL,g] is re-inserted into
    the filler stream every `remind_every` tokens. Same weights as v1 at eval;
    only the prompt changes. Shortens effective goal->query distance.
    Returns x[B,T], qpos[B], y[B], goal[B].
    """
    v = vocab
    xs, qs, ys, gs = [], [], [], []
    for _ in range(batch_size):
        g = int(rng.integers(v.K))
        q = int(rng.integers(v.V_ans))
        a = rule_answer(task, g, q, v.V_ans)
        toks = [v.GOAL_MARK, v.goal_tok(g)]
        for i in range(filler_len):
            if remind_every and i > 0 and i % remind_every == 0:
                toks += [v.GOAL_MARK, v.goal_tok(g)]        # reminder
            if distractors and rng.random() < 0.15:
                toks.append(v.goal_tok(int(rng.integers(v.K))))
            else:
                toks.append(v.fill_tok(int(rng.integers(v.V_filler))))
        toks += [v.QUERY_MARK, v.ans_tok(q)]
        qpos = len(toks) - 1
        if len(toks) > block_size:
            raise ValueError(f"reminder seq len {len(toks)} > block_size {block_size}")
        pad = block_size - len(toks)
        xs.append([v.PAD] * pad + toks); qs.append(pad + qpos); ys.append(a); gs.append(g)
    return (np.array(xs, dtype=np.int64), np.array(qs, dtype=np.int64),
            np.array(ys, dtype=np.int64), np.array(gs, dtype=np.int64))


# ======================= exp06 additions (injection eval) =======================

def _redraw_pair(rng, v, task):
    """Draw (g, g', q) with g' != g and rule(g',q) != rule(g,q) — exp05-style
    distinctness redraw (verified mean 1.2 draws, every pair has >=6/8 usable q)."""
    g = int(rng.integers(v.K))
    while True:
        gp = int(rng.integers(v.K)); q = int(rng.integers(v.V_ans))
        if gp != g and rule_answer(task, gp, q, v.V_ans) != rule_answer(task, g, q, v.V_ans):
            return g, gp, q


def _filler_stream(rng, v, filler_len, distractors, exempt):
    """Filler token list with 15% goal-token distractors; slots in `exempt` are
    left as None placeholders for the injection block."""
    out = []; rate = _drate(distractors)
    for i in range(filler_len):
        if i in exempt:
            out.append(None)
        elif rate and rng.random() < rate:
            out.append(v.goal_tok(int(rng.integers(v.K))))
        else:
            out.append(v.fill_tok(int(rng.integers(v.V_filler))))
    return out


def make_batch_injected(rng, vocab, task, batch_size, filler_len, block_size,
                        phi, m, distractors=True):
    """
    OOD pressure eval: a contiguous block of m x [GOAL_MARK, goal_tok(g')]
    OVERWRITES the 2m filler slots starting at floor(phi*(L-2m)) (phi='end'
    -> flush against the query). Sequence length identical to the clean cell
    at the same L. (g',q) redrawn until rule(g',q) != rule(g,q).
    Returns x, qpos, y, goal, apos, y_inj, ginj.
    """
    v = vocab
    L = filler_len; W = 2 * m
    assert L >= W, "filler too short for injection block"
    start = L - W if phi == "end" else int(phi * (L - W))
    xs, qs, ys, gs, aps, yis, gis = [], [], [], [], [], [], []
    for _ in range(batch_size):
        g, gp, q = _redraw_pair(rng, v, task)
        a = rule_answer(task, g, q, v.V_ans)
        ai = rule_answer(task, gp, q, v.V_ans)
        fill = _filler_stream(rng, v, L, distractors, set(range(start, start + W)))
        block = [v.GOAL_MARK, v.goal_tok(gp)] * m
        for j in range(W):
            fill[start + j] = block[j]
        toks = [v.GOAL_MARK, v.goal_tok(g)] + fill + [v.QUERY_MARK, v.ans_tok(q)]
        qpos = len(toks) - 1
        if len(toks) > block_size:
            raise ValueError("seq too long")
        pad = block_size - len(toks)
        xs.append([v.PAD] * pad + toks); qs.append(pad + qpos)
        ys.append(a); gs.append(g); aps.append(pad); yis.append(ai); gis.append(gp)
    import numpy as _np
    return (_np.array(xs, dtype=_np.int64), _np.array(qs, dtype=_np.int64),
            _np.array(ys, dtype=_np.int64), _np.array(gs, dtype=_np.int64),
            _np.array(aps, dtype=_np.int64), _np.array(yis, dtype=_np.int64),
            _np.array(gis, dtype=_np.int64))


def make_batch_reminders_injected(rng, vocab, task, batch_size, filler_len,
                                  block_size, remind_every, phi, m,
                                  distractors=True):
    """
    Reminder stream + injection. phi in {float, 'end', 'after_last', 'covered'}:
      float/'end'  -> block overwrites filler slots at the usual grid position
      'after_last' -> block overwrites the last 2m filler slots (after the
                      final reminder; no reminder post-dates the injection)
      'covered'    -> block overwrites the 2m slots immediately BEFORE the
                      reminder nearest mid-span (a reminder post-dates it)
    Returns x, qpos, y, goal, apos, y_inj, ginj.
    """
    v = vocab
    L = filler_len; W = 2 * m
    assert L >= W and L > remind_every
    if phi == "after_last":
        start = L - W
    elif phi == "covered":
        k = max(1, round((L / 2) / remind_every))
        start = k * remind_every - W
        assert start > 0
    elif phi == "end":
        start = L - W
    else:
        start = int(phi * (L - W))
    xs, qs, ys, gs, aps, yis, gis = [], [], [], [], [], [], []
    for _ in range(batch_size):
        g, gp, q = _redraw_pair(rng, v, task)
        a = rule_answer(task, g, q, v.V_ans)
        ai = rule_answer(task, gp, q, v.V_ans)
        fill = _filler_stream(rng, v, L, distractors, set(range(start, start + W)))
        block = [v.GOAL_MARK, v.goal_tok(gp)] * m
        for j in range(W):
            fill[start + j] = block[j]
        toks = [v.GOAL_MARK, v.goal_tok(g)]
        for i in range(L):
            if remind_every and i > 0 and i % remind_every == 0:
                toks += [v.GOAL_MARK, v.goal_tok(g)]        # reminder
            toks.append(fill[i])
        toks += [v.QUERY_MARK, v.ans_tok(q)]
        qpos = len(toks) - 1
        if len(toks) > block_size:
            raise ValueError("seq too long")
        pad = block_size - len(toks)
        xs.append([v.PAD] * pad + toks); qs.append(pad + qpos)
        ys.append(a); gs.append(g); aps.append(pad); yis.append(ai); gis.append(gp)
    import numpy as _np
    return (_np.array(xs, dtype=_np.int64), _np.array(qs, dtype=_np.int64),
            _np.array(ys, dtype=_np.int64), _np.array(gs, dtype=_np.int64),
            _np.array(aps, dtype=_np.int64), _np.array(yis, dtype=_np.int64),
            _np.array(gis, dtype=_np.int64))
