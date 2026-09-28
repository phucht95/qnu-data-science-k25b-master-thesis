"""Additive PWL-LS-SVM with monotonicity constraints on selected inputs (Section 3.5.5).

For an input with knots t_1 < ... < t_K the contribution
    h(x) = a x + sum_j c_j (x - t_j)_+
is re-parametrized by its slopes s_0 = a, s_j = a + c_1 + ... + c_j on the K + 1 segments:
    h(x) = sum_j s_j G_j(x),   G_0 = min(x, t_1),  G_j = clip(x - t_j, 0, t_{j+1} - t_j),  G_K = (x - t_K)_+ .
The ridge penalty of PWL-LS-SVM on (a, c) becomes ||D s||^2 with D the first-difference matrix, so the
model and its regularization are unchanged; "non-decreasing" is the bound s >= 0 and "non-increasing"
is s <= 0.  The problem stays a convex (bounded least squares) problem.
"""

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import lsq_linear
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import unique_labels

from pwlsvm import PWLFeatures, PWLSVM


class MonotonePWLLSSVM(PWLSVM):
    """monotone: dict {input index: sign} or {input index: (sign, t0)}; None = no constraint.

    sign = +1 asks for a non-decreasing contribution, -1 for a non-increasing one.  With (sign, t0) the
    constraint applies only to the segments that reach beyond t0 (original units), i.e. to the part
    x(i) >= t0 of the axis; the segments to the left of t0 stay free.
    """

    def __init__(self, m_per_dim=10, C=1.0, knots="quantile", monotone=None):
        super().__init__(kind="additive", m_per_dim=m_per_dim, loss="squared", C=C, knots=knots)
        self.monotone = monotone

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
            raise ValueError("binary problems only")
        t = np.where(y == self.classes_[1], 1.0, -1.0)
        self.scaler_ = MinMaxScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.features_ = PWLFeatures("additive", self.m_per_dim, self.knots).fit(Xs)
        n = Xs.shape[1]
        ax, kn = self.features_.axis_, self.features_.knot_
        G, D, lb, ub, sizes = [], [], [], [], []
        for i in range(n):
            tk = kn[ax == i]                                  # sorted knots of input i
            K = tk.size
            x = Xs[:, i]
            g = np.empty((x.size, K + 1))
            if K == 0:
                g[:, 0] = x
            else:
                g[:, 0] = np.minimum(x, tk[0])
                for j in range(1, K):
                    g[:, j] = np.clip(x - tk[j - 1], 0.0, tk[j] - tk[j - 1])
                g[:, K] = np.maximum(0.0, x - tk[K - 1])
            G.append(g)
            D.append(np.eye(K + 1) - np.eye(K + 1, k=-1))
            sgn, mask = self.constraint_mask(i, tk)
            lb.append(np.where(mask & (sgn > 0), 0.0, -np.inf))
            ub.append(np.where(mask & (sgn < 0), 0.0, np.inf))
            sizes.append(K + 1)
        G = np.hstack(G); D = block_diag(*D)
        gbar, tbar = G.mean(axis=0), t.mean()
        A = np.vstack([np.sqrt(self.C) * (G - gbar), D])
        b = np.r_[np.sqrt(self.C) * (t - tbar), np.zeros(D.shape[0])]
        lb, ub = np.concatenate(lb), np.concatenate(ub)
        if np.isfinite(lb).any() or np.isfinite(ub).any():
            s = lsq_linear(A, b, bounds=(lb, ub), method="bvls", tol=1e-12).x
        else:
            s = np.linalg.lstsq(A, b, rcond=None)[0]
        # back to the coefficients (a_i, c_ij) of the usual feature map phi(x) = [x, (x(i) - t_ij)_+]
        a = np.empty(n); c = np.empty(kn.size)
        pos = np.cumsum([0] + sizes)
        for i in range(n):
            si = s[pos[i]:pos[i + 1]]
            a[i] = si[0]
            c[ax == i] = np.diff(si)
        self.coef_ = np.r_[a, c][None, :]
        self.intercept_ = np.array([tbar - gbar @ s])
        self.slopes_ = [s[pos[i]:pos[i + 1]] for i in range(n)]
        return self

    def constraint_mask(self, i, tk=None, monotone=None):
        """(sign, boolean mask over the K + 1 segments of input i that the constraint applies to)."""
        spec = ((self.monotone if monotone is None else monotone) or {}).get(i, 0)
        sgn, t0 = (spec if isinstance(spec, tuple) else (spec, -np.inf))
        if tk is None:
            tk = self.features_.knot_[self.features_.axis_ == i]
        right = np.r_[tk, np.inf]                                  # right end of every segment (scaled units)
        lo, hi = self.scaler_.data_min_[i], self.scaler_.data_max_[i]
        t0s = (t0 - lo) / (hi - lo) if np.isfinite(t0) and hi > lo else t0
        return sgn, (right > t0s + 1e-12) & (sgn != 0)
