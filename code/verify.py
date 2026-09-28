"""Section 3.1: numerical checks of the formulas of Chapter 2.

(1) PWL-LS-SVM: the primal solution (ridge regression on +-1 labels) coincides with the
    solution of the dual linear system (2.2.10).
(2) PWL-C-SVM: solving the dual with the finite-rank kernel K = Phi Phi^T and recovering
    w = sum_k alpha_k y_k phi(x_k) gives the same classifier as the primal solver.
"""

import json
import os

import numpy as np
from sklearn.svm import SVC

import datasets2d as d2
from pwlsvm import PWLFeatures, PWLSVM, ls_svm_dual

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

out = {}
Xtr, ytr, Xte, yte = d2.make("moons", 400, 400, seed=0)
for kind in ("additive", "hh", "ghh"):
    model = PWLSVM(kind=kind, m_per_dim=10, loss="squared", C=10.0, random_state=0).fit(Xtr, ytr)
    Phi = model._phi(Xtr)
    w_dual, b_dual = ls_svm_dual(Phi, ytr, 10.0)
    f_primal = model.decision_function(Xte)
    f_dual = model._phi(Xte) @ w_dual + b_dual
    out[f"ls_{kind}"] = dict(M=Phi.shape[1], dw=float(np.abs(w_dual - model.coef_.ravel()).max()),
                             db=float(abs(b_dual - model.intercept_[0])),
                             df=float(np.abs(f_primal - f_dual).max()))

    svm_lin = SVC(kernel="linear", C=10.0, tol=1e-6).fit(Phi, ytr)
    svm_pre = SVC(kernel="precomputed", C=10.0, tol=1e-6).fit(Phi @ Phi.T, ytr)
    w_rec = svm_pre.dual_coef_.ravel() @ Phi[svm_pre.support_]
    out[f"c_{kind}"] = dict(nsv=int(svm_pre.support_.size),
                            dw=float(np.abs(w_rec - svm_lin.coef_.ravel()).max()),
                            agree=float((np.sign(model._phi(Xte) @ w_rec + svm_pre.intercept_[0])
                                         == svm_lin.predict(model._phi(Xte))).mean()))
print(json.dumps(out, indent=1))
with open(os.path.join(RESULTS, "verify.json"), "w") as f:
    json.dump(out, f, indent=1)
