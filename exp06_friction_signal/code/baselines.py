"""exp06 analytic baselines — program config K=4, V_ans=8, task hard.
Computed BEFORE thresholds (standing rule)."""
import numpy as np
K, V = 4, 8
rule = lambda g, q: ((q + g) * (g + 1)) % V

print("chance = 1/V =", 1/V)

# goal-ignorant ceiling (must reproduce exp03's 15/32 = 0.469)
tot = 0
for q in range(V):
    counts = np.zeros(V)
    for g in range(K): counts[rule(g, q)] += 1
    tot += counts.max() / K
print("goal-ignorant ceiling =", tot / V, "(exp03 registered 0.469)")

# collision structure over ordered pairs g != g'
per_pair = {}
for q in range(V):
    for g in range(K):
        for gp in range(K):
            if gp == g: continue
            per_pair.setdefault((g, gp), 0)
            per_pair[(g, gp)] += (rule(g, q) == rule(gp, q))
n_pairs = len(per_pair); coll = sum(per_pair.values())
print("P[collision] =", coll / (n_pairs * V))
for pair, c in sorted(per_pair.items()):
    print("  pair", pair, "collides at", c, "/", V, "q values")
bad = [p for p, c in per_pair.items() if c == V]
print("pairs with NO distinguishing q:", bad)

# redraw policy feasibility: fix g uniform; draw (g',q) until rule distinct
p_ok = 1 - coll / (n_pairs * V)
print("distinct-triple fraction =", p_ok, "; mean redraws ≈", 1/p_ok)

# per-g conditional: for each true g, #(g',q) with distinct answers
for g in range(K):
    ok = sum(V - per_pair[(g, gp)] for gp in range(K) if gp != g)
    print(f"  g={g}: {ok}/{(K-1)*V} usable (g',q) combos")

# MC: random-goal-follower (binds uniform goal token from context) under redraw
rng = np.random.default_rng(0); N = 400000
hitsR = hitsC = 0
for _ in range(N):
    g = rng.integers(K)
    while True:
        gp = rng.integers(K); q = rng.integers(V)
        if gp != g and rule(gp, q) != rule(g, q): break
    a = rule(rng.integers(K), q)
    hitsR += a == rule(g, q); hitsC += a == rule(gp, q)
print("random-goal-follower under redraw: R≈%.4f C≈%.4f" % (hitsR/N, hitsC/N))

# MC: q-marginal predictor (exp03 v1 failure mode) scored on injected cells
hitsR = hitsC = 0
marg = {}
for q in range(V):
    counts = np.zeros(V)
    for g in range(K): counts[rule(g, q)] += 1
    marg[q] = counts.argmax()
for _ in range(N):
    g = rng.integers(K)
    while True:
        gp = rng.integers(K); q = rng.integers(V)
        if gp != g and rule(gp, q) != rule(g, q): break
    a = marg[q]
    hitsR += a == rule(g, q); hitsC += a == rule(gp, q)
print("q-marginal predictor under redraw: R≈%.4f C≈%.4f" % (hitsR/N, hitsC/N))
