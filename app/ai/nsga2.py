"""
NSGA-II (Deb et al., 2002) with constrained domination, SBX crossover and
polynomial mutation, plus knee-point selection on the Pareto front.
All objectives are minimised; constraints are g(x) <= 0.
"""
from __future__ import annotations

import numpy as np


def _dominates(fa, fb, ca, cb):
    if ca == 0 and cb == 0:
        return np.all(fa <= fb) and np.any(fa < fb)
    if ca == 0 and cb > 0:
        return True
    if ca > 0 and cb > 0:
        return ca < cb
    return False


def fast_nondominated_sort(F, CV):
    n = len(F)
    S = [[] for _ in range(n)]
    nd = np.zeros(n, dtype=int)
    fronts = [[]]
    # vectorised dominance matrix
    feas = CV <= 0
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dom_obj = le & lt
    D = np.zeros((n, n), dtype=bool)
    ff = feas[:, None] & feas[None, :]
    D[ff] = dom_obj[ff]
    D |= feas[:, None] & ~feas[None, :]
    inf = ~feas[:, None] & ~feas[None, :]
    D |= inf & (CV[:, None] < CV[None, :])
    for p in range(n):
        S[p] = list(np.nonzero(D[p])[0])
        nd[p] = int(D[:, p].sum())
        if nd[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                nd[q] -= 1
                if nd[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(F):
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        o = np.argsort(F[:, k])
        d[o[0]] = d[o[-1]] = np.inf
        rng = F[o[-1], k] - F[o[0], k]
        if rng <= 0:
            continue
        d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / rng
    return d


def sbx(p1, p2, lo, hi, eta, rng):
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[i] - p2[i]) < 1e-12:
            continue
        x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        u = rng.random()
        beta = 1.0 + 2.0 * (x1 - lo[i]) / (x2 - x1)
        alpha = 2.0 - beta ** -(eta + 1)
        bq = (u * alpha) ** (1 / (eta + 1)) if u <= 1 / alpha else (1 / (2 - u * alpha)) ** (1 / (eta + 1))
        y1 = 0.5 * ((x1 + x2) - bq * (x2 - x1))
        beta = 1.0 + 2.0 * (hi[i] - x2) / (x2 - x1)
        alpha = 2.0 - beta ** -(eta + 1)
        bq = (u * alpha) ** (1 / (eta + 1)) if u <= 1 / alpha else (1 / (2 - u * alpha)) ** (1 / (eta + 1))
        y2 = 0.5 * ((x1 + x2) + bq * (x2 - x1))
        y1, y2 = np.clip([y1, y2], lo[i], hi[i])
        if rng.random() < 0.5:
            c1[i], c2[i] = y2, y1
        else:
            c1[i], c2[i] = y1, y2
    return c1, c2


def poly_mut(x, lo, hi, eta, pm, rng):
    y = x.copy()
    for i in range(len(x)):
        if rng.random() > pm:
            continue
        d = hi[i] - lo[i]
        if d <= 0:
            continue
        u = rng.random()
        d1 = (y[i] - lo[i]) / d
        d2 = (hi[i] - y[i]) / d
        if u < 0.5:
            dq = (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** (1 / (eta + 1)) - 1
        else:
            dq = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)) ** (1 / (eta + 1))
        y[i] = np.clip(y[i] + dq * d, lo[i], hi[i])
    return y


def nsga2(evaluate, lo, hi, pop=100, gens=120, seed=1, eta_c=15, eta_m=20, pc=0.9, callback=None, init=None):
    """evaluate(X) -> (F (n,m), G (n,k)) ; returns dict with population, front, history."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    d = len(lo)
    u = (rng.permuted(np.tile(np.arange(pop), (d, 1)), axis=1).T + rng.random((pop, d))) / pop
    X = lo + u * (hi - lo)
    if init is not None:
        X[: len(init)] = init
    F, G = evaluate(X)
    CV = np.maximum(G, 0).sum(1)
    hist = []
    pm = 1.0 / d
    for gen in range(gens):
        fronts = fast_nondominated_sort(F, CV)
        rank = np.empty(pop, dtype=int)
        cd = np.empty(pop)
        for r, fr in enumerate(fronts):
            rank[fr] = r
            cd[fr] = crowding(F[fr])
        # tournament
        kids = []
        while len(kids) < pop:
            a, b = rng.integers(0, pop, 2)
            p1 = a if (rank[a] < rank[b] or (rank[a] == rank[b] and cd[a] > cd[b])) else b
            a, b = rng.integers(0, pop, 2)
            p2 = a if (rank[a] < rank[b] or (rank[a] == rank[b] and cd[a] > cd[b])) else b
            if rng.random() < pc:
                c1, c2 = sbx(X[p1], X[p2], lo, hi, eta_c, rng)
            else:
                c1, c2 = X[p1].copy(), X[p2].copy()
            kids.append(poly_mut(c1, lo, hi, eta_m, pm, rng))
            kids.append(poly_mut(c2, lo, hi, eta_m, pm, rng))
        Xk = np.array(kids[:pop])
        Fk, Gk = evaluate(Xk)
        CVk = np.maximum(Gk, 0).sum(1)
        XX = np.vstack([X, Xk])
        FF = np.vstack([F, Fk])
        GG = np.vstack([G, Gk])
        CC = np.concatenate([CV, CVk])
        fronts = fast_nondominated_sort(FF, CC)
        sel = []
        for fr in fronts:
            if len(sel) + len(fr) <= pop:
                sel.extend(fr)
            else:
                c = crowding(FF[fr])
                order = np.argsort(-c)
                sel.extend(list(np.array(fr)[order[: pop - len(sel)]]))
                break
        sel = np.array(sel)
        X, F, G, CV = XX[sel], FF[sel], GG[sel], CC[sel]
        feas = CV <= 0
        hist.append({"gen": gen, "n_feasible": int(feas.sum()),
                     "best": F[feas].min(0).tolist() if feas.any() else None})
        if callback:
            callback(gen, X, F, CV)
    fronts = fast_nondominated_sort(F, CV)
    f0 = [i for i in fronts[0] if CV[i] <= 0]
    return {"X": X, "F": F, "G": G, "CV": CV, "front": np.array(f0, dtype=int), "history": hist}


def knee_point(Fp):
    """Knee of a (minimisation) Pareto front: max distance from the hyperplane through
    the extreme points in normalised objective space; fallback = closest to ideal."""
    if len(Fp) == 0:
        return None
    if len(Fp) <= 2:
        return 0
    fmin, fmax = Fp.min(0), Fp.max(0)
    Z = (Fp - fmin) / np.maximum(fmax - fmin, 1e-12)
    m = Z.shape[1]
    ext = [int(np.argmax(Z[:, k] - 1e3 * np.delete(Z, k, axis=1).sum(1) * 0)) for k in range(m)]
    E = Z[ext]
    try:
        # hyperplane sum(a_i z_i) = 1 through extremes
        a = np.linalg.solve(E, np.ones(m))
        dist = (1 - Z @ a) / np.linalg.norm(a)
        if np.all(np.isfinite(dist)):
            return int(np.argmax(dist))
    except np.linalg.LinAlgError:
        pass
    return int(np.argmin(np.linalg.norm(Z, axis=1)))
