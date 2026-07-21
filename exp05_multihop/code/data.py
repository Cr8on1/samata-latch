"""
EXP05 task 'hop2': goal-gated two-hop retrieval from goal-selected tables.

    [GOAL] g | L filler tokens + 32 pair-triples [TBL_t] k pi_t(k), shuffled | [QUERY] q
    answer a = pi_g(pi_g(q))

Tables: four independent uniform single-8-cycles, REDRAWN until the four
candidates {pi_t^2(q)} are distinct at the queried q (pre-registered design
constraint; pins goal-ignorant ceiling to exactly 0.25, wrong-table reads
to 0). Answer is NOT a function of (g,q): no memorizable table exists.

Sequence length = L + 100. Pure numpy so it unit-tests without torch.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Vocab:
    K: int = 4          # number of goals == number of tables
    V_ans: int = 8      # answer/key alphabet (single alphabet, chainable)
    V_filler: int = 32  # filler alphabet size

    def __post_init__(self):
        self.GOAL_MARK = 0
        self.QUERY_MARK = 1
        self.PAD = 2
        self.tbl_lo = 3; self.tbl_hi = 3 + self.K          # [TBL_t] markers
        base = self.tbl_hi
        self.goal_lo = base;        self.goal_hi = base + self.K
        self.ans_lo = self.goal_hi; self.ans_hi = self.ans_lo + self.V_ans
        self.fill_lo = self.ans_hi; self.fill_hi = self.fill_lo + self.V_filler
        self.size = self.fill_hi

    def goal_tok(self, g): return self.goal_lo + g
    def ans_tok(self, a):  return self.ans_lo + a
    def fill_tok(self, f): return self.fill_lo + f
    def tbl_tok(self, t):  return self.tbl_lo + t


def _cycle(rng, V):
    """Uniform random single V-cycle permutation."""
    c = rng.permutation(V)
    p = np.empty(V, dtype=np.int64)
    p[c] = c[np.append(np.arange(1, V), 0)]
    return p


def sample_tables(rng, K, V_ans):
    """(q, tables): redraw tables until the K two-hop candidates at q are distinct."""
    q = int(rng.integers(V_ans))
    while True:
        tabs = [_cycle(rng, V_ans) for _ in range(K)]
        cands = [int(t[t[q]]) for t in tabs]
        if len(set(cands)) == K:
            return q, tabs


def _drate(distractors):
    if distractors is True:  return 0.15
    if not distractors:      return 0.0
    return float(distractors)


def _stream(rng, vocab, tabs, filler_len, rate, g=None, remind_every=0):
    """Shuffled filler+pair stream. If remind_every, reinsert [GOAL,g] before
    every remind_every-th filler token (exp02b reminder-arm convention)."""
    v = vocab
    items = [("f",)] * filler_len + [("p", t, k) for t in range(v.K)
                                     for k in range(v.V_ans)]
    order = rng.permutation(len(items))
    toks, nf = [], 0
    for i in order:
        it = items[i]
        if it[0] == "f":
            if remind_every and nf > 0 and nf % remind_every == 0:
                toks += [v.GOAL_MARK, v.goal_tok(g)]
            nf += 1
            if rate and rng.random() < rate:
                toks.append(v.goal_tok(int(rng.integers(v.K))))
            else:
                toks.append(v.fill_tok(int(rng.integers(v.V_filler))))
        else:
            _, t, k = it
            toks += [v.tbl_tok(t), v.ans_tok(k), v.ans_tok(int(tabs[t][k]))]
    return toks


def make_example(rng, vocab, task, filler_len, distractors=False, force_g=None,
                 remind_every=0):
    """Return (tokens, query_pos, target, goal)."""
    assert task == "hop2", task
    v = vocab; rate = _drate(distractors)
    q, tabs = sample_tables(rng, v.K, v.V_ans)
    g = int(rng.integers(v.K)) if force_g is None else int(force_g)
    a = int(tabs[g][tabs[g][q]])
    toks = [v.GOAL_MARK, v.goal_tok(g)]
    toks += _stream(rng, vocab, tabs, filler_len, rate,
                    g=g, remind_every=remind_every)
    toks += [v.QUERY_MARK, v.ans_tok(q)]
    return toks, len(toks) - 1, a, g


def make_batch(rng, vocab, task, batch_size, filler_len, block_size,
               distractors=False, balance_goals=False):
    """Left-padded fixed-length batch. Returns x[B,T], qpos[B], y[B], goal[B].
    balance_goals=True (eval): goal = i % K for exact per-goal cell counts."""
    v = vocab
    xs, qs, ys, gs = [], [], [], []
    for i in range(batch_size):
        fg = (i % v.K) if balance_goals else None
        toks, qpos, a, g = make_example(rng, vocab, task, filler_len,
                                        distractors, force_g=fg)
        if len(toks) > block_size:
            raise ValueError(f"seq len {len(toks)} > block_size {block_size}")
        pad = block_size - len(toks)
        xs.append([v.PAD] * pad + toks); qs.append(pad + qpos)
        ys.append(a); gs.append(g)
    return (np.array(xs, dtype=np.int64), np.array(qs, dtype=np.int64),
            np.array(ys, dtype=np.int64), np.array(gs, dtype=np.int64))


def make_batch_reminders(rng, vocab, task, batch_size, filler_len, block_size,
                         remind_every, distractors=False, balance_goals=False):
    """SURFACE-FIX arm: goal reinserted every remind_every filler tokens."""
    v = vocab
    xs, qs, ys, gs = [], [], [], []
    for i in range(batch_size):
        fg = (i % v.K) if balance_goals else None
        toks, qpos, a, g = make_example(rng, vocab, task, filler_len,
                                        distractors, force_g=fg,
                                        remind_every=remind_every)
        if len(toks) > block_size:
            raise ValueError(f"reminder seq len {len(toks)} > block_size {block_size}")
        pad = block_size - len(toks)
        xs.append([v.PAD] * pad + toks); qs.append(pad + qpos)
        ys.append(a); gs.append(g)
    return (np.array(xs, dtype=np.int64), np.array(qs, dtype=np.int64),
            np.array(ys, dtype=np.int64), np.array(gs, dtype=np.int64))
