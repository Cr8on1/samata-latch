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


def make_example(rng, vocab, task, filler_len, distractors=False):
    """Return (tokens list, query_position index, target answer id 0..V_ans-1)."""
    v = vocab
    g = int(rng.integers(v.K))
    q = int(rng.integers(v.V_ans))
    a = rule_answer(task, g, q, v.V_ans)

    toks = [v.GOAL_MARK, v.goal_tok(g)]
    for _ in range(filler_len):
        if distractors and rng.random() < 0.15:
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
