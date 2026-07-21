import numpy as np, time
V, K = 8, 4
def cyc(rng):
    c = rng.permutation(V); p = np.empty(V, int); p[c] = c[np.append(np.arange(1,V),0)]; return p
def sample_example(rng):
    q = int(rng.integers(V)); tries = 0
    while True:
        tries += 1
        tabs = [cyc(rng) for _ in range(K)]
        cands = [int(t[t[q]]) for t in tabs]
        if len(set(cands)) == K:
            return tabs, q, cands, tries
rng = np.random.default_rng(0)
t0 = time.time(); T = 5000; tot = 0
ti_hist = np.zeros(V)
for _ in range(T):
    tabs, q, cands, tr = sample_example(rng); tot += tr
    g = int(rng.integers(K)); ti_hist[cands[g]] += 1
dt = (time.time()-t0)/T*1000
print(f"mean tries: {tot/T:.2f}   gen time/example: {dt:.3f} ms")
print("goal-ignorant ceiling: 0.25 exact (distinct by construction); wrong-table acc: 0 exact")
print("answer==q ever possible: no (8-cycle squared has no fixed points)")
# table-ignorant ceiling under conditioning: distribution of a given (g,q)
print("answer dist given q (should be ~uniform over 7 non-q):")
print(np.round(ti_hist/T, 3))
for L in [0,16,32,64]:
    keep = 0.85**L
    print(f"last-goal-token signature L={L}: {keep + (1-keep)*0.25:.3f}")
print("loss floors: goal-ignorant ln4=1.3863; table-ignorant ~ln7=1.9459; chance ln8=2.0794")
