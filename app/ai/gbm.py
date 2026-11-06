"""
Histogram gradient-boosted regression trees in pure NumPy (squared-error loss).

Used for (a) the model-plant mismatch correction and (b) the soft sensors for
hot-metal silicon and temperature between casts.  Trees are stored as flat
arrays so models serialise to JSON without pickle.
"""
from __future__ import annotations

import numpy as np


class GBM:
    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=4, min_leaf=15, n_bins=32,
                 subsample=0.8, colsample=1.0, l2=1.0, seed=0):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.n_bins = n_bins
        self.subsample = subsample
        self.colsample = colsample
        self.l2 = l2
        self.seed = seed
        self.trees = []
        self.edges = None
        self.base = 0.0
        self.gain = None

    def _bin(self, X):
        Xb = np.empty(X.shape, dtype=np.int32)
        for j in range(X.shape[1]):
            Xb[:, j] = np.searchsorted(self.edges[j], X[:, j], side="right")
        return Xb

    def fit(self, X, y, Xv=None, yv=None, verbose=False):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, d = X.shape
        self.edges = []
        for j in range(d):
            q = np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.n_bins + 1)[1:-1]))
            self.edges.append(q)
        Xb = self._bin(X)
        self.base = float(y.mean())
        F = np.full(n, self.base)
        self.gain = np.zeros(d)
        self.trees = []
        self.curve = []
        Fv = None
        if Xv is not None:
            Xvb = self._bin(np.asarray(Xv, float))
            Fv = np.full(len(Xv), self.base)
        for m in range(self.n_estimators):
            r = y - F
            idx = rng.choice(n, int(self.subsample * n), replace=False) if self.subsample < 1 else np.arange(n)
            cols = rng.choice(d, max(1, int(self.colsample * d)), replace=False) if self.colsample < 1 else np.arange(d)
            tree = self._build(Xb[idx], r[idx], cols)
            self.trees.append(tree)
            F += self.lr * self._predict_tree(tree, Xb)
            if Fv is not None:
                Fv += self.lr * self._predict_tree(tree, Xvb)
                if m % 10 == 0 or m == self.n_estimators - 1:
                    self.curve.append({"iter": m, "train_rmse": float(np.sqrt(np.mean((y - F) ** 2))),
                                       "val_rmse": float(np.sqrt(np.mean((yv - Fv) ** 2)))})
        tot = self.gain.sum()
        self.importance = (self.gain / tot).tolist() if tot > 0 else [0.0] * d
        return self

    def _build(self, Xb, r, cols):
        feat, thr, left, right, val = [], [], [], [], []

        def new_node():
            feat.append(-1)
            thr.append(0)
            left.append(-1)
            right.append(-1)
            val.append(0.0)
            return len(feat) - 1

        root = new_node()
        stack = [(root, np.arange(len(r)), 0)]
        nb = self.n_bins + 1
        while stack:
            node, idx, depth = stack.pop()
            rr = r[idx]
            G = rr.sum()
            H = len(idx)
            val[node] = G / (H + self.l2)
            if depth >= self.max_depth or H < 2 * self.min_leaf:
                continue
            best = (0.0, None, None)
            base_score = G * G / (H + self.l2)
            for j in cols:
                b = Xb[idx, j]
                gs = np.bincount(b, weights=rr, minlength=nb)
                hs = np.bincount(b, minlength=nb).astype(float)
                gl = np.cumsum(gs)[:-1]
                hl = np.cumsum(hs)[:-1]
                gr = G - gl
                hr = H - hl
                ok = (hl >= self.min_leaf) & (hr >= self.min_leaf)
                if not ok.any():
                    continue
                score = np.where(ok, gl * gl / (hl + self.l2) + gr * gr / (hr + self.l2) - base_score, -np.inf)
                k = int(np.argmax(score))
                if score[k] > best[0]:
                    best = (float(score[k]), j, k)
            if best[1] is None:
                continue
            j, k = best[1], best[2]
            self.gain[j] += best[0]
            feat[node] = int(j)
            thr[node] = int(k)
            mask = Xb[idx, j] <= k
            l = new_node()
            rgt = new_node()
            left[node] = l
            right[node] = rgt
            stack.append((l, idx[mask], depth + 1))
            stack.append((rgt, idx[~mask], depth + 1))
        return {"f": np.array(feat), "t": np.array(thr), "l": np.array(left), "r": np.array(right), "v": np.array(val)}

    def _predict_tree(self, tr, Xb):
        node = np.zeros(len(Xb), dtype=int)
        for _ in range(self.max_depth + 1):
            f = tr["f"][node]
            leaf = f < 0
            if leaf.all():
                break
            go_left = np.where(leaf, True, Xb[np.arange(len(Xb)), np.maximum(f, 0)] <= tr["t"][node])
            nxt = np.where(go_left, tr["l"][node], tr["r"][node])
            node = np.where(leaf, node, nxt)
        return tr["v"][node]

    def predict(self, X):
        Xb = self._bin(np.atleast_2d(np.asarray(X, float)))
        F = np.full(len(Xb), self.base)
        for tr in self.trees:
            F += self.lr * self._predict_tree(tr, Xb)
        return F

    def to_dict(self):
        return {"base": self.base, "lr": self.lr, "max_depth": self.max_depth, "edges": [e.tolist() for e in self.edges],
                "trees": [{k: v.tolist() for k, v in t.items()} for t in self.trees],
                "importance": getattr(self, "importance", None)}

    @classmethod
    def from_dict(cls, d):
        m = cls(max_depth=d["max_depth"], learning_rate=d["lr"])
        m.base = d["base"]
        m.edges = [np.array(e) for e in d["edges"]]
        m.trees = [{k: np.array(v) for k, v in t.items()} for t in d["trees"]]
        m.importance = d.get("importance")
        return m
