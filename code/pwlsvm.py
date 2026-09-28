"""PWL-SVM: support vector machines with piecewise linear feature mappings.

Re-implementation of the method of

    X. Huang, S. Mehrkanoon, J.A.K. Suykens (2013),
    "Support vector machines with piecewise linear feature mapping",
    Neurocomputing 117, 118-127.

Equation numbers in the comments refer to that paper.

Feature mappings (all include the n coordinates x(1), ..., x(n) first, eq. 4):
    additive : max{0, x(i_m) - q_m}             (eq. 11), knots on a uniform or quantile grid
    hh       : max{0, p_m^T x + q_m}            (eq. 12), random hinging hyperplanes
    ghh      : max{0, p_m1^T x + q_m1, ...,
                      p_mn^T x + q_mn}          (eq. 13), generalized hinging hyperplanes

Training problems on top of phi(x):
    loss="hinge"   : PWL-C-SVM  (eq. 5-6)
    loss="squared" : PWL-LS-SVM (eq. 9-10); primal = ridge regression on +-1 labels.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted


# ----------------------------------------------------------------------------
# Feature mapping
# ----------------------------------------------------------------------------

def random_hyperplane(rng, lo, hi, max_tries=100):
    """Hyperplane p^T x + q = 0 through n points drawn uniformly from the box [lo, hi].

    Following Section 2.4 of the paper, p(1) is fixed to +1 or -1 with equal
    probability; the remaining n unknowns (p(2), ..., p(n), q) are obtained from
    the n linear equations p^T u_j + q = 0, j = 1..n. The system is nonsingular
    with probability one; a degenerate draw is simply repeated.
    """
    n = lo.size
    for _ in range(max_tries):
        pts = rng.uniform(lo, hi, size=(n, n))          # row j = point u_j
        s = rng.choice([-1.0, 1.0])
        A = np.hstack([pts[:, 1:], np.ones((n, 1))])
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        sol = np.linalg.solve(A, -s * pts[:, 0])
        return np.r_[s, sol[:-1]], sol[-1]
    raise RuntimeError("could not draw a nondegenerate hyperplane")


class PWLFeatures(TransformerMixin, BaseEstimator):
    """Piecewise linear feature mapping phi(x) = [x(1..n), h_1(x), ..., h_D(x)].

    Parameters
    ----------
    kind : {"additive", "hh", "ghh"}
    m_per_dim : float
        Number of features per input dimension, M / n (as in Table 1 of the paper).
        additive: every axis is cut into `m_per_dim` segments, giving
                  m_per_dim - 1 knots per axis, so M = m_per_dim * n.
        hh, ghh:  D = (m_per_dim - 1) * n hinge features, so again M = m_per_dim * n.
    knots : {"uniform", "quantile"}   (additive only)
        "uniform" places the knots at equal distances (the rule of the paper);
        "quantile" places them at empirical quantiles of the training data
        (duplicated knots of discrete variables are removed).
    normalize : bool   (hh, ghh only)
        Rescale every random hyperplane to ||p||_2 = 1 instead of |p(1)| = 1.
    random_state : int or None
    """

    def __init__(self, kind="hh", m_per_dim=10, knots="uniform", normalize=False,
                 random_state=None):
        self.kind = kind
        self.m_per_dim = m_per_dim
        self.knots = knots
        self.normalize = normalize
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[1]
        lo, hi = X.min(axis=0), X.max(axis=0)
        hi = np.where(hi > lo, hi, lo + 1.0)             # guard constant columns
        self.n_features_in_ = n
        if self.kind == "additive":
            S = int(self.m_per_dim)
            frac = np.arange(1, S) / S                      # interior knot positions
            axes, knots = [], []
            for i in range(n):
                if self.knots == "uniform":
                    q = lo[i] + (hi[i] - lo[i]) * frac
                elif self.knots == "quantile":
                    q = np.unique(np.quantile(X[:, i], frac))
                    q = q[(q > lo[i]) & (q < hi[i])]
                else:
                    raise ValueError(f"unknown knots {self.knots!r}")
                axes.append(np.full(q.size, i)); knots.append(q)
            self.axis_ = np.concatenate(axes).astype(int)
            self.knot_ = np.concatenate(knots)
        elif self.kind in ("hh", "ghh"):
            rng = np.random.default_rng(self.random_state)
            D = int(round((self.m_per_dim - 1) * n))
            pieces = 1 if self.kind == "hh" else n
            P = np.empty((D, pieces, n))
            q = np.empty((D, pieces))
            for m in range(D):
                for j in range(pieces):
                    P[m, j], q[m, j] = random_hyperplane(rng, lo, hi)
            if self.normalize and D > 0:
                scale = np.linalg.norm(P, axis=2)
                P, q = P / scale[..., None], q / scale
            self.P_, self.q_ = P, q
        else:
            raise ValueError(f"unknown kind {self.kind!r}")
        return self

    @property
    def n_hinges_(self):
        return self.knot_.size if self.kind == "additive" else self.P_.shape[0]

    def hinge_part(self, X):
        """Only the nonlinear part [h_1(x), ..., h_D(x)]."""
        check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        if self.kind == "additive":
            return np.maximum(0.0, X[:, self.axis_] - self.knot_)
        D, pieces, n = self.P_.shape
        if D == 0:                                        # M = n: the linear classifier
            return np.zeros((X.shape[0], 0))
        Z = X @ self.P_.reshape(D * pieces, n).T + self.q_.reshape(-1)
        return np.maximum(0.0, Z.reshape(X.shape[0], D, pieces).max(axis=2))

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.hstack([X, self.hinge_part(X)])

    def n_stored_parameters(self):
        """Real numbers that must be stored to evaluate phi (knots or hyperplanes)."""
        if self.kind == "additive":
            return self.knot_.size          # axis indices are integers, not counted
        return self.P_.size + self.q_.size

    def activation_pattern(self, X):
        """Integer code of the linear region that contains each row of X."""
        X = np.asarray(X, dtype=float)
        if self.kind == "additive":
            act = (X[:, self.axis_] > self.knot_).astype(np.int64)
        else:
            D, pieces, n = self.P_.shape
            Z = (X @ self.P_.reshape(D * pieces, n).T + self.q_.reshape(-1)).reshape(-1, D, pieces)
            best = Z.argmax(axis=2)
            active = Z.max(axis=2) > 0
            act = np.where(active, best + 1, 0)           # 0 = the zero piece
        _, codes = np.unique(act, axis=0, return_inverse=True)
        return codes.ravel()


# ----------------------------------------------------------------------------
# Classifier
# ----------------------------------------------------------------------------

class PWLSVM(ClassifierMixin, BaseEstimator):
    """SVM with a piecewise linear feature mapping (binary; multiclass via one-vs-rest).

    Parameters
    ----------
    kind, m_per_dim, knots, normalize, random_state : passed to PWLFeatures.
    loss : {"hinge", "squared"}
        "hinge"   -> PWL-C-SVM  (eq. 5); "squared" -> PWL-LS-SVM (eq. 9).
    C : float
        The regularization constant gamma of the paper.
    solver : {"libsvm", "liblinear"}  (only for loss="hinge")
        "libsvm" solves the dual (6) exactly with an unregularized bias;
        "liblinear" uses dual coordinate descent on explicit features (fast; it
        regularizes the bias slightly, controlled by intercept_scaling).
    scale : bool
        Rescale every input to [0, 1] first (needed for the grid/random parameters).
    """

    def __init__(self, kind="hh", m_per_dim=10, loss="hinge", C=1.0,
                 solver="liblinear", scale=True, knots="uniform", normalize=False,
                 random_state=None):
        self.kind = kind
        self.m_per_dim = m_per_dim
        self.loss = loss
        self.C = C
        self.solver = solver
        self.scale = scale
        self.knots = knots
        self.normalize = normalize
        self.random_state = random_state

    # -- helpers -------------------------------------------------------------
    def _phi(self, X):
        Xs = self.scaler_.transform(X) if self.scale else np.asarray(X, float)
        return self.features_.transform(Xs)

    @staticmethod
    def _ridge_primal(Phi, T, C):
        """min 1/2||w||^2 + C/2 * sum_k (t_k - w^T phi_k - w0)^2, bias unpenalized."""
        mu, tbar = Phi.mean(axis=0), T.mean(axis=0)
        Pc, Tc = Phi - mu, T - tbar
        A = Pc.T @ Pc + np.eye(Phi.shape[1]) / C
        W = np.linalg.solve(A, Pc.T @ Tc)
        return W, tbar - mu @ W

    # -- API -------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = unique_labels(y)
        self.scaler_ = MinMaxScaler().fit(X) if self.scale else None
        Xs = self.scaler_.transform(X) if self.scale else X
        self.features_ = PWLFeatures(self.kind, self.m_per_dim, self.knots, self.normalize,
                                     self.random_state).fit(Xs)
        Phi = self.features_.transform(Xs)
        K = len(self.classes_)
        if self.loss == "squared":
            Y = np.where(np.asarray(y)[:, None] == self.classes_[None, :], 1.0, -1.0)
            if K == 2:
                Y = Y[:, [1]]
            W, b = self._ridge_primal(Phi, Y, self.C)
            self.coef_, self.intercept_ = W.T, b
        elif self.loss == "hinge":
            if self.solver == "libsvm" and K == 2:
                svm = SVC(kernel="linear", C=self.C).fit(Phi, y)
                self.coef_, self.intercept_ = svm.coef_, svm.intercept_
                self.n_support_ = svm.n_support_
            else:
                svm = LinearSVC(loss="hinge", C=self.C, dual=True, max_iter=20000,
                                intercept_scaling=10.0, tol=1e-5).fit(Phi, y)
                self.coef_, self.intercept_ = svm.coef_, svm.intercept_
        else:
            raise ValueError(f"unknown loss {self.loss!r}")
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        s = self._phi(X) @ self.coef_.T + self.intercept_
        return s.ravel() if s.shape[1] == 1 else s

    def predict(self, X):
        s = self.decision_function(X)
        if s.ndim == 1:
            return self.classes_[(s > 0).astype(int)]
        return self.classes_[s.argmax(axis=1)]

    def n_stored_parameters(self):
        """Numbers needed at prediction time: feature parameters + (w, w0) + scaler."""
        n = self.features_.n_features_in_
        return (self.features_.n_stored_parameters() + self.coef_.size
                + np.size(self.intercept_) + (2 * n if self.scale else 0))

    # -- interpretation of the additive model --------------------------------
    def additive_components(self, X):
        """For kind="additive": contributions h_i(x(i)) of every input to the decision value.

        Returns an array of shape (len(X), n) whose rows sum to decision_function(X) - w0.
        """
        if self.kind != "additive" or self.coef_.shape[0] != 1:
            raise ValueError("only for binary additive models")
        Xs = self.scaler_.transform(X) if self.scale else np.asarray(X, float)
        w = self.coef_.ravel()
        n = self.features_.n_features_in_
        out = Xs * w[:n]
        H = self.features_.hinge_part(Xs) * w[n:]
        for i in range(n):
            out[:, i] += H[:, self.features_.axis_ == i].sum(axis=1)
        return out


# ----------------------------------------------------------------------------
# PWL-LS-SVM in dual form (eq. 10) -- used to verify the primal solver
# ----------------------------------------------------------------------------

def ls_svm_dual(Phi, y, C):
    """Solve [[0, y^T], [y, Omega + I/C]] [w0; alpha] = [0; 1] with Omega_kl = y_k y_l phi_k^T phi_l.

    Returns (w, w0) with w = sum_k alpha_k y_k phi_k.
    """
    y = np.asarray(y, dtype=float)
    N = y.size
    Omega = (y[:, None] * y[None, :]) * (Phi @ Phi.T)
    A = np.zeros((N + 1, N + 1))
    A[0, 1:], A[1:, 0] = y, y
    A[1:, 1:] = Omega + np.eye(N) / C
    rhs = np.r_[0.0, np.ones(N)]
    sol = np.linalg.solve(A, rhs)
    w0, alpha = sol[0], sol[1:]
    return Phi.T @ (alpha * y), w0


# ----------------------------------------------------------------------------
# Baseline: intersection kernel SVM (Section 3.3 of the paper)
# ----------------------------------------------------------------------------

def intersection_kernel(A, B, chunk=256):
    """k(x, z) = sum_i min{x(i), z(i)} (for nonnegative inputs)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    out = np.empty((A.shape[0], B.shape[0]))
    for s in range(0, A.shape[0], chunk):
        out[s:s + chunk] = np.minimum(A[s:s + chunk, None, :], B[None, :, :]).sum(axis=2)
    return out
