import numpy as np
rng = np.random.default_rng(0)
N, K, V = 400_000, 4, 8
# batch-sample K independent 8-cycles per example
c = np.argsort(rng.random((N, K, V)), axis=-1)          # random orderings
pi = np.empty((N, K, V), int)
np.put_along_axis(pi, c, np.take(c, (np.arange(V)+1) % V, axis=-1), axis=-1)
g = rng.integers(K, size=N); q = rng.integers(V, size=N)
idx = np.arange(N)
pi2 = np.take_along_axis(pi, np.take_along_axis(pi, np.broadcast_to(np.arange(V), (N,K,V)), -1), -1)  # not needed; do direct
hop1 = pi[idx, :, :][np.arange(N)[:,None], np.arange(K)[None,:], q[:,None]]   # pi_t(q)  [N,K]
cands = pi[np.arange(N)[:,None], np.arange(K)[None,:], hop1]                  # pi_t^2(q) [N,K]
a = cands[idx, g]
# goal-ignorant argmax ceiling (uniform tie-break among modal candidates)
onehot = (cands[:,:,None] == np.arange(V)[None,None,:]).sum(1)   # [N,V] counts
m = onehot.max(1)
acc_gi = ((onehot[idx, a] == m) / (onehot == m[:,None]).sum(1)).mean()
loss_gi = -np.log(onehot[idx, a] / K).mean()
acc_1hop = (hop1[idx, g] == a).mean()
gp = (g + 1 + rng.integers(K-1, size=N)) % K
acc_wrong = (cands[idx, gp] == a).mean()
coll = ((onehot > 1).any(1)).mean()
print("goal-ignorant argmax ceiling:", round(float(acc_gi), 4))
print("goal-ignorant optimal loss floor:", round(float(loss_gi), 4), "(ln4 =", round(np.log(4), 4), ")")
print("one-hop-on-correct-table acc:", float(acc_1hop))
print("wrong-table 2-hop acc:", round(float(acc_wrong), 4), "(1/7 =", round(1/7, 4), ")")
print("collision rate (any duplicate candidates):", round(float(coll), 4))
print("table-ignorant ceiling 1/7 =", round(1/7, 4), "; loss floor ln7 =", round(np.log(7), 4))
print("chance:", 1/8, "; loss ln8 =", round(np.log(8), 4))
for L in [0, 16, 32, 64]:
    keep = 0.85 ** L
    print(f"last-goal-token signature L={L}: {keep + (1-keep)*(0.25 + 0.75/7):.3f}")
