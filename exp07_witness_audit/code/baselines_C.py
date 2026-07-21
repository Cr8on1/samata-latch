"""exp07 analytic baselines — witness/audit floors. Program config K=4,
V_ans=8, task hard. Computed BEFORE thresholds (standing rule).
Cross-check anchors: goal-ignorant ceiling 0.469 (exp03/06), collision 1/6
(exp06). New floors: W_g, V (audit validity), OC (overclaim), injected-cell
witness stats Rw/Cw."""
import numpy as np
K, V = 4, 8
rule = lambda g, q: ((q + g) * (g + 1)) % V
rng = np.random.default_rng(0); N = 400000

# --- anchors (must reproduce exp06) ---
tot = 0
for q in range(V):
    c = np.zeros(V)
    for g in range(K): c[rule(g, q)] += 1
    tot += c.max() / K
print("ANCHOR goal-ignorant ceiling = %.4f (want 0.469)" % (tot / V))
coll = sum(rule(g,q)==rule(gp,q) for q in range(V) for g in range(K)
           for gp in range(K) if gp!=g)
p_coll = coll / (K*(K-1)*V)
print("ANCHOR P[collision] = %.4f (want 1/6 = 0.1667)" % p_coll)

# --- witness floors, clean cells ---
print("\nW_g chance (random witness) = 1/K =", 1/K)
print("V floor, random answer (any witness) = 1/V =", 1/V)
# V for competent answerer (ans = rule(g,q)) + RANDOM witness:
v_exact = 1/K + (1-1/K)*p_coll
print("V: competent answer + random witness = 1/K+(3/4)coll = %.4f" % v_exact)
h = sum(rule(rng.integers(K), q)==rule(g, q)
        for g,q in zip(rng.integers(K,size=N), rng.integers(V,size=N))
        ) if False else None
gs = rng.integers(K, size=N); qs = rng.integers(V, size=N)
ws = rng.integers(K, size=N)
ans = np.array([rule(g,q) for g,q in zip(gs,qs)])
wit = np.array([rule(w,q) for w,q in zip(ws,qs)])
print("  MC: %.4f" % np.mean(ans==wit))

# DEGENERATE self-consistent witness: constant w0, ans = rule(w0,q).
# V = 1.000 by construction; W_g = 1/K; what CLEAN ACCURACY does it buy?
print("\nDegenerate constant-witness (V=1.0, W_g=0.25) clean accuracy per w0:")
for w0 in range(K):
    acc = np.mean([rule(g,q)==rule(w0,q) for g in range(K) for q in range(V)])
    print("  w0=%d: acc = %.4f" % (w0, acc))
# => V alone is gameable; success criterion must be JOINT (V AND W_g AND
#    base competence). Base competence bar 0.75 already excludes these.

# --- overclaim floors (OC = P[w-hat = g AND ans != rule(g,q)]) ---
print("\nOC floors:")
print("  competent answerer, any witness: OC = 0 (ans always = rule(g,q))")
acc_conf = 0.40   # confusion-floor answerer (exp06: undertrained plateaus 0.35-0.44)
print("  assignment-tracking witness (w=g always) + confused answerer acc a:")
for a in (0.35, 0.40, 0.469):
    print("    a=%.3f -> OC = 1-a = %.3f" % (a, 1-a))
print("  random witness + confused answerer acc a: OC = (1/K)(1-a):")
for a in (0.35, 0.40, 0.469):
    print("    a=%.3f -> OC = %.4f" % (a, (1-a)/K))

# --- injected-cell witness floors (redraw policy: rule distinct) ---
print("\nInjected cells (g' != g, rule distinct; witness analogs of R/C):")
print("  random witness: Rw = Cw = 1/K = 0.25, Dw = 0")
# MC under redraw for a witness that picks a uniformly random goal TOKEN
# seen in context (2 distinct goal tokens present: g at first marker, g' in
# injection block) -> Rw = Cw = 0.5:
hits_g = hits_gp = 0
for _ in range(N//4):
    g = rng.integers(K)
    while True:
        gp = rng.integers(K); q = rng.integers(V)
        if gp != g and rule(gp,q) != rule(g,q): break
    w = g if rng.random() < 0.5 else gp
    hits_g += w==g; hits_gp += w==gp
print("  uniform-over-present-goals witness: Rw≈%.4f Cw≈%.4f (want 0.5/0.5)"
      % (hits_g/(N//4), hits_gp/(N//4)))
print("  first-binder witness: Rw=1, Cw=0. last-binder: Rw=0, Cw=1.")

# --- V under injection for the four canonical (answer, witness) pairings ---
print("\nV on injected cells (redraw guarantees rule(g,q) != rule(g',q)):")
print("  ans follows g,  w=g : V=1   | ans follows g,  w=g': V=0")
print("  ans follows g', w=g': V=1 (honest wrong-master record)")
print("  ans follows g', w=g : V=0 (overclaim signature)")
