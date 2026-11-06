"""
Dependency-free (NumPy) multilayer perceptron with Adam and an optional
physics-informed penalty on balance residuals  r = A(x) y + b(x).

Loss = MSE(standardised outputs) + lambda * mean( (r / r_scale)^2 )
The balance residual uses the *de-standardised* balance outputs, so the
gradient flows back through the output scaling.
"""
from __future__ import annotations

import json
import time

import numpy as np


class MLP:
    def __init__(self, n_in, n_out, hidden=(96, 96), seed=0, act="silu"):
        rng = np.random.default_rng(seed)
        sizes = [n_in] + list(hidden) + [n_out]
        self.W = [rng.normal(0, np.sqrt(2.0 / (a + b)), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.act = act
        self.xm = self.xs = self.ym = self.ys = None

    # activation
    def _a(self, z):
        if self.act == "tanh":
            return np.tanh(z)
        s = 1.0 / (1.0 + np.exp(-z))
        return z * s

    def _da(self, z):
        if self.act == "tanh":
            t = np.tanh(z)
            return 1 - t * t
        s = 1.0 / (1.0 + np.exp(-z))
        return s * (1 + z * (1 - s))

    def forward(self, Xs, keep=False):
        h = Xs
        cache = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            cache.append((h, z))
            h = self._a(z) if i < len(self.W) - 1 else z
        return (h, cache) if keep else h

    def predict(self, X):
        Xs = (np.atleast_2d(X) - self.xm) / self.xs
        return self.forward(Xs) * self.ys + self.ym

    def fit(self, X, Y, Xv=None, Yv=None, epochs=1500, lr=2e-3, batch=128, wd=1e-6,
            phys=None, lam=0.0, bal_idx=None, r_scale=None, patience=250, seed=0, verbose=False, log_every=100):
        """phys: callable(x_raw_batch) -> (A, b) with A (n, nb, nbal_outputs)."""
        rng = np.random.default_rng(seed)
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-9
        self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-9
        Xs = (X - self.xm) / self.xs
        Ys = (Y - self.ym) / self.ys
        if phys is not None and lam > 0:
            A_all, b_all = phys(X)
            bal_idx = np.asarray(bal_idx)
            rs = np.asarray(r_scale, dtype=float)
        mW = [np.zeros_like(w) for w in self.W]
        vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        best = (np.inf, None)
        wait = 0
        hist = []
        n = len(X)
        for ep in range(epochs):
            perm = rng.permutation(n)
            lr_ep = lr * (0.5 * (1 + np.cos(np.pi * min(ep / epochs, 1.0)))) + 1e-5
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                xb, yb = Xs[idx], Ys[idx]
                out, cache = self.forward(xb, keep=True)
                g = 2.0 * (out - yb) / out.size
                if phys is not None and lam > 0:
                    yr = out[:, bal_idx] * self.ys[bal_idx] + self.ym[bal_idx]
                    A = A_all[idx]
                    r = np.einsum("nky,ny->nk", A, yr) + b_all[idx]
                    rn = r / rs
                    gr = 2.0 * lam * rn / rs / (len(idx) * len(rs))            # dL/dr
                    gy = np.einsum("nk,nky->ny", gr, A)                        # dL/dyr
                    g[:, bal_idx] += gy * self.ys[bal_idx]
                # backprop
                t += 1
                for i in range(len(self.W) - 1, -1, -1):
                    h, z = cache[i]
                    if i < len(self.W) - 1:
                        g = g * self._da(z)
                    gW = h.T @ g + wd * self.W[i]
                    gb = g.sum(0)
                    g = g @ self.W[i].T
                    mW[i] = b1 * mW[i] + (1 - b1) * gW
                    vW[i] = b2 * vW[i] + (1 - b2) * gW * gW
                    mb[i] = b1 * mb[i] + (1 - b1) * gb
                    vb[i] = b2 * vb[i] + (1 - b2) * gb * gb
                    mh = mW[i] / (1 - b1 ** t)
                    vh = vW[i] / (1 - b2 ** t)
                    self.W[i] -= lr_ep * mh / (np.sqrt(vh) + eps)
                    self.b[i] -= lr_ep * (mb[i] / (1 - b1 ** t)) / (np.sqrt(vb[i] / (1 - b2 ** t)) + eps)
            if Xv is not None and (ep % 10 == 0 or ep == epochs - 1):
                pv = self.predict(Xv)
                vloss = float(np.mean(((pv - Yv) / self.ys) ** 2))
                tl = float(np.mean((self.forward(Xs) - Ys) ** 2))
                hist.append({"epoch": ep, "train_mse": tl, "val_mse": vloss})
                if vloss < best[0] - 1e-6:
                    best = (vloss, ([w.copy() for w in self.W], [b.copy() for b in self.b]))
                    wait = 0
                else:
                    wait += 10
                if verbose and ep % log_every == 0:
                    print(f"    ep {ep} train {tl:.5f} val {vloss:.5f}", flush=True)
                if wait > patience:
                    break
        if best[1] is not None:
            self.W, self.b = best[1]
        self.history = hist
        return self

    def to_dict(self):
        return {"W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b], "act": self.act,
                "xm": self.xm.tolist(), "xs": self.xs.tolist(), "ym": self.ym.tolist(), "ys": self.ys.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls.__new__(cls)
        m.W = [np.array(w) for w in d["W"]]
        m.b = [np.array(b) for b in d["b"]]
        m.act = d.get("act", "silu")
        m.xm, m.xs, m.ym, m.ys = (np.array(d[k]) for k in ("xm", "xs", "ym", "ys"))
        return m

    def save_npz(self, path):
        arrs = {f"W{i}": w for i, w in enumerate(self.W)}
        arrs.update({f"b{i}": b for i, b in enumerate(self.b)})
        arrs.update({"xm": self.xm, "xs": self.xs, "ym": self.ym, "ys": self.ys, "n": np.array(len(self.W))})
        np.savez_compressed(path, **arrs)

    @classmethod
    def load_npz(cls, path, act="silu"):
        d = np.load(path)
        m = cls.__new__(cls)
        n = int(d["n"])
        m.W = [d[f"W{i}"] for i in range(n)]
        m.b = [d[f"b{i}"] for i in range(n)]
        m.xm, m.xs, m.ym, m.ys = d["xm"], d["xs"], d["ym"], d["ys"]
        m.act = act
        return m


def r2_scores(Y, P):
    ss = ((Y - P) ** 2).sum(0)
    st = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    return 1 - ss / st
