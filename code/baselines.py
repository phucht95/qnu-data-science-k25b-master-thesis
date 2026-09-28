"""Reference classifiers used in the comparisons of Chapter 3 (binary labels +-1)."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils.multiclass import unique_labels

from pwlsvm import intersection_kernel


class LSSVMRBF(ClassifierMixin, BaseEstimator):
    """LS-SVM classifier with RBF kernel, solved through the dual linear system (Suykens & Vandewalle, 1999)."""

    def __init__(self, C=1.0, gamma=1.0):
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        assert len(self.classes_) == 2
        self.scaler_ = MinMaxScaler().fit(X)
        Xs = self.scaler_.transform(X)
        t = np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0)
        N = t.size
        Omega = np.outer(t, t) * rbf_kernel(Xs, Xs, gamma=self.gamma)
        A = np.zeros((N + 1, N + 1))
        A[0, 1:], A[1:, 0] = t, t
        A[1:, 1:] = Omega + np.eye(N) / self.C
        sol = np.linalg.solve(A, np.r_[0.0, np.ones(N)])
        self.b_, self.coef_ = sol[0], sol[1:] * t        # alpha_k * y_k
        self.Xs_ = Xs
        return self

    def decision_function(self, X):
        return rbf_kernel(self.scaler_.transform(X), self.Xs_, gamma=self.gamma) @ self.coef_ + self.b_

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def n_stored_parameters(self):
        N, n = self.Xs_.shape
        return N * (n + 1) + 1 + 2 * n


class IKSVM(ClassifierMixin, BaseEstimator):
    """Intersection-kernel SVM, k(x, z) = sum_i min{x(i), z(i)} on inputs scaled to [0, 1]."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        self.scaler_ = MinMaxScaler().fit(X)
        Xs = np.clip(self.scaler_.transform(X), 0, None)
        self.svc_ = SVC(kernel="precomputed", C=self.C).fit(intersection_kernel(Xs, Xs), y)
        self.Xsv_ = Xs[self.svc_.support_]
        self.classes_ = self.svc_.classes_
        return self

    def decision_function(self, X):
        Xs = np.clip(self.scaler_.transform(X), 0, None)
        return intersection_kernel(Xs, self.Xsv_) @ self.svc_.dual_coef_.ravel() + self.svc_.intercept_[0]

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def n_stored_parameters(self):
        S, n = self.Xsv_.shape
        return S * (n + 1) + 1 + 2 * n
