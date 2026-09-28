"""Section 3.3.2: ten benchmark data sets, nine methods, ten random splits.

Protocol: stratified 50/50 splits (Magic: 2000 training points, as in Huang et al. 2013),
10 repetitions; hyper-parameters chosen by stratified 10-fold cross-validation on the
training part; accuracy measured on the test part.
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from baselines import IKSVM, LSSVMRBF
from datasets_uci import DATA, load
from pwlsvm import PWLSVM

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
OUT = os.path.join(RESULTS, "benchmark.json")

C_GRID = [2.0 ** k for k in range(-5, 12, 2)]     # 2^-5, ..., 2^11
G_GRID = [2.0 ** k for k in range(-15, 4, 2)]     # 2^-15, ..., 2^3
A_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]            # L2 penalty of the MLP


def methods(n, seed):
    D = 9 * n                                     # hinge features of HH with M/n = 10
    return {
        "linear": (PWLSVM(kind="additive", m_per_dim=1, loss="hinge", solver="liblinear"),
                   {"C": C_GRID}),
        "rbf": (make_pipeline(MinMaxScaler(), SVC(kernel="rbf")),
                {"svc__C": C_GRID, "svc__gamma": G_GRID}),
        "lssvm": (LSSVMRBF(), {"C": C_GRID, "gamma": G_GRID}),
        "knn": (make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=1)), None),
        "iksvm": (IKSVM(), {"C": C_GRID}),
        "mlp": (make_pipeline(MinMaxScaler(), MLPClassifier(hidden_layer_sizes=(D,), activation="relu",
                                                            solver="lbfgs", max_iter=2000,
                                                            random_state=seed)),
                {"mlpclassifier__alpha": A_GRID}),
        "pwl_c": (PWLSVM(kind="additive", m_per_dim=10, loss="hinge", solver="liblinear"),
                  {"C": C_GRID}),
        "pwl_ls": (PWLSVM(kind="additive", m_per_dim=10, loss="squared"), {"C": C_GRID}),
        "pwl_hh": (PWLSVM(kind="hh", m_per_dim=10, loss="hinge", solver="liblinear",
                          random_state=seed), {"C": C_GRID}),
    }


def n_params(model, n):
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "n_stored_parameters"):
        return int(est.n_stored_parameters())
    if isinstance(est, SVC):
        return int(est.support_vectors_.size + est.dual_coef_.size + 1 + 2 * n)
    if isinstance(est, KNeighborsClassifier):
        return int(est.n_samples_fit_ * (n + 1) + 2 * n)
    if isinstance(est, MLPClassifier):
        return int(sum(c.size for c in est.coefs_) + sum(b.size for b in est.intercepts_) + 2 * n)
    return -1


def splits(name, X, y, n_rep):
    _, _, ntr = DATA[name]
    if ntr is None:
        return list(StratifiedShuffleSplit(n_splits=n_rep, test_size=0.5, random_state=2024).split(X, y))
    out = []
    for r in range(n_rep):
        tr, te = train_test_split(np.arange(len(y)), train_size=ntr, stratify=y, random_state=2024 + r)
        out.append((tr, te))
    return out


def run(names, n_rep=10):
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["data"], r["rep"], r["method"]) for r in rows}
    for name in names:
        X, y = load(name)
        n = X.shape[1]
        print(name, X.shape, flush=True)
        for rep, (tr, te) in enumerate(splits(name, X, y, n_rep)):
            for key, (est, grid) in methods(n, rep).items():
                if (name, rep, key) in done:
                    continue
                t0 = time.perf_counter()
                if grid is None:
                    model, best = est.fit(X[tr], y[tr]), {}
                else:
                    gs = GridSearchCV(est, grid, cv=StratifiedKFold(10, shuffle=True, random_state=rep),
                                      n_jobs=-1).fit(X[tr], y[tr])
                    model, best = gs.best_estimator_, gs.best_params_
                tune = time.perf_counter() - t0
                acc = float((model.predict(X[te]) == y[te]).mean())
                rows.append(dict(data=name, rep=rep, method=key, acc=acc, n=n, N=len(tr), Ntest=len(te),
                                 tune_time=tune, params=n_params(model, n),
                                 best={k: float(v) for k, v in best.items()}))
            print(" ", rep, {r["method"]: round(r["acc"], 3) for r in rows[-9:] if r["data"] == name},
                  flush=True)
            with open(OUT, "w") as f:
                json.dump(rows, f)
    return rows


if __name__ == "__main__":
    names = sys.argv[1:] or list(DATA)
    run(names)
