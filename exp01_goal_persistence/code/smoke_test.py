"""
Torch-free smoke test of the data + reminder + rule logic (runs anywhere).
For the full model/training smoke run:  python run.py smoke
"""
import numpy as np
from data import Vocab, make_batch, make_batch_reminders, rule_answer, goal_ids_array


def test_vocab_blocks():
    v = Vocab(K=8, V_ans=16, V_filler=64)
    # contiguous, non-overlapping id ranges, all < size
    assert v.goal_lo == 3 and v.goal_hi == 11
    assert v.ans_lo == 11 and v.ans_hi == 27
    assert v.fill_lo == 27 and v.fill_hi == 91 == v.size
    ids = set()
    for g in range(v.K): ids.add(v.goal_tok(g))
    for a in range(v.V_ans): ids.add(v.ans_tok(a))
    for f in range(v.V_filler): ids.add(v.fill_tok(f))
    ids |= {v.GOAL_MARK, v.QUERY_MARK, v.PAD}
    assert len(ids) == v.size, "vocab ids overlap or leave gaps"
    print("ok  vocab blocks contiguous & unique")


def test_targets_and_distance():
    v = Vocab(); rng = np.random.default_rng(0)
    for L in [0, 5, 40]:
        x, qpos, y, goal = make_batch(rng, v, "shift", 128, L, block_size=256)
        assert x.shape == (128, 256) and qpos.shape == (128,)
        for i in range(128):
            row = x[i]
            # query token is at qpos; goal token immediately after GOAL_MARK
            assert row[qpos[i]] >= v.ans_lo and row[qpos[i]] < v.ans_hi
            q = int(row[qpos[i]] - v.ans_lo)
            assert y[i] == rule_answer("shift", int(goal[i]), q, v.V_ans)
            # goal token sits exactly L+? before query: distance goal->query == L+2
            gpos = qpos[i] - (L + 2)     # [GOAL,g][L filler][QUERY,q]
            assert row[gpos] == v.goal_tok(int(goal[i]))
    print("ok  targets match rule; goal->query distance == L")


def test_hard_task_differs():
    v = Vocab(); 
    diff = sum(rule_answer("shift", g, q, v.V_ans) != rule_answer("hard", g, q, v.V_ans)
               for g in range(v.K) for q in range(v.V_ans))
    assert diff > 0, "hard task should differ from shift for some (g,q)"
    print(f"ok  hard task differs from shift on {diff}/{v.K*v.V_ans} (g,q) pairs")


def test_reminders_insert_and_fit():
    v = Vocab(); rng = np.random.default_rng(1)
    L, R, B, T = 60, 20, 32, 512
    x, qpos, y, goal = make_batch_reminders(rng, v, "shift", B, L, T, remind_every=R)
    for i in range(B):
        row = x[i]
        n_goalmark = int((row == v.GOAL_MARK).sum())
        expected = 1 + (L - 1) // R  # initial marker + one per crossed multiple of R
        assert n_goalmark == expected, f"got {n_goalmark} goal markers, expected {expected}"
        q = int(row[qpos[i]] - v.ans_lo)
        assert y[i] == rule_answer("shift", int(goal[i]), q, v.V_ans)
    print("ok  reminder arm inserts goal markers and fits block_size")


def test_accuracy_aggregation():
    # mimic run.py's per-length accuracy loop with fake predictions:
    # a model that 'forgets' with length should show a monotone-ish decay.
    rng = np.random.default_rng(2); Vn = 16
    lens = [0, 32, 96, 256]; accs = []
    for L in lens:
        keep = max(0.05, 1.0 - L / 300.0)          # synthetic forgetting curve
        y = rng.integers(Vn, size=2000)
        correct = rng.random(2000) < keep
        pred = np.where(correct, y, (y + 1) % Vn)
        accs.append(float((pred == y).mean()))
    assert accs[0] > accs[-1] and accs[-1] < 0.5, "decay curve sanity failed"
    print(f"ok  accuracy aggregation sane: {[round(a,3) for a in accs]} (chance={1/Vn:.3f})")


if __name__ == "__main__":
    test_vocab_blocks()
    test_targets_and_distance()
    test_hard_task_differs()
    test_reminders_insert_and_fit()
    test_accuracy_aggregation()
    print("\nALL TORCH-FREE SMOKE TESTS PASSED")
