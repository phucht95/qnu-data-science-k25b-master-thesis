"""Section 3.3 (preview): six UCI data sets used in Table 1 of Huang et al. (2013).

Protocol (close to the paper, but repeated): 10 random stratified 50/50 splits; on every
training half the hyper-parameters are chosen by 10-fold cross-validation; accuracy is
measured on the other half. PWL-SVMs use the additive map (11) with M/n = 10, as in the paper.
"""

import json
import os
import time

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from baselines import IKSVM, LSSVMRBF
from pwlsvm import PWLSVM

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

DATA = {  # name: (OpenML id, positive label)
    "Pima": (37, "tested_positive"),
    "Breast": (15, "malignant"),
    "Haberman": (43, "2"),
    "Transfusion": (1464, "2"),
    "Ionosphere": (59, "g"),
    "Parkinsons": (1488, "2"),
}
# Test accuracies reported in Table 1 of Huang et al. (2013), one 50/50 split
PAPER = {
    "Pima":        dict(lssvm=0.768, rbf=0.732, knn=0.667, iksvm=0.717, pwl_ls=0.766, pwl_c=0.742),
    "Breast":      dict(lssvm=0.949, rbf=0.940, knn=0.603, iksvm=0.940, pwl_ls=0.960, pwl_c=0.957),
    "Haberman":    dict(lssvm=0.758, rbf=0.752, knn=0.673, iksvm=0.686, pwl_ls=0.758, pwl_c=0.765),
    "Transfusion": dict(lssvm=0.783, rbf=0.703, knn=0.757, iksvm=0.685, pwl_ls=0.759, pwl_c=0.751),
    "Ionosphere":  dict(lssvm=0.933, rbf=0.895, knn=0.857, iksvm=0.905, pwl_ls=0.829, pwl_c=0.857),
    "Parkinsons":  dict(lssvm=0.983, rbf=0.983, knn=0.845, iksvm=0.948, pwl_ls=1.000, pwl_c=1.000),
}

C_GRID = [2.0 ** k for k in range(-5, 12, 2)]     # 2^-5, ..., 2^11
G_GRID = [2.0 ** k for k in range(-15, 4, 2)]     # 2^-15, ..., 2^3 (LIBSVM guide)


def load(name):
    did, pos = DATA[name]
    d = fetch_openml(data_id=did, as_frame=True, parser="auto")
    X = d.data.apply(lambda c: c.astype(float))
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0].to_numpy()          # e.g. the constant 2nd column of Ionosphere
    y = np.where(d.target.astype(str).to_numpy() == pos, 1, -1)
    return X, y


def methods(seed):
    return {
        "linear": (make_pipeline(MinMaxScaler(), SVC(kernel="linear")), {"svc__C": C_GRID}),
        "rbf": (make_pipeline(MinMaxScaler(), SVC(kernel="rbf")),
                {"svc__C": C_GRID, "svc__gamma": G_GRID}),
        "lssvm": (LSSVMRBF(), {"C": C_GRID, "gamma": G_GRID}),
        "knn": (make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=1)), None),
        "iksvm": (IKSVM(), {"C": C_GRID}),
        "pwl_c": (PWLSVM(kind="additive", m_per_dim=10, loss="hinge", solver="libsvm",
                         random_state=seed), {"C": C_GRID}),
        "pwl_ls": (PWLSVM(kind="additive", m_per_dim=10, loss="squared", random_state=seed),
                   {"C": C_GRID}),
        # LIBSVM converges very slowly for HH features at large C; LIBLINEAR is used here
        "pwl_hh": (PWLSVM(kind="hh", m_per_dim=10, loss="hinge", solver="liblinear",
                          random_state=seed), {"C": C_GRID}),
    }


def n_params(model, n):
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "n_stored_parameters"):
        return int(est.n_stored_parameters())
    if isinstance(est, SVC):
        if est.kernel == "linear":
            return n + 1 + 2 * n
        return int(est.support_vectors_.size + est.dual_coef_.size + 1 + 2 * n)
    if isinstance(est, KNeighborsClassifier):
        return int(est.n_samples_fit_ * (n + 1) + 2 * n)
    return -1


def predict_time(model, X, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model.predict(X)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / len(X)


def run(n_rep=10):
    rows = []
    for name in DATA:
        X, y = load(name)
        n = X.shape[1]
        print(name, X.shape, flush=True)
        sss = StratifiedShuffleSplit(n_splits=n_rep, test_size=0.5, random_state=2024)
        for rep, (tr, te) in enumerate(sss.split(X, y)):
            for key, (est, grid) in methods(rep).items():
                t0 = time.perf_counter()
                if grid is None:
                    model, best = est.fit(X[tr], y[tr]), {}
                else:
                    gs = GridSearchCV(est, grid, cv=StratifiedKFold(10, shuffle=True, random_state=rep),
                                      n_jobs=-1).fit(X[tr], y[tr])
                    model, best = gs.best_estimator_, gs.best_params_
                acc = float((model.predict(X[te]) == y[te]).mean())
                rows.append(dict(data=name, rep=rep, method=key, acc=acc, n=n, N=len(tr),
                                 tune_time=time.perf_counter() - t0,
                                 pred_time=predict_time(model, X[te]),
                                 params=n_params(model, n), best=str(best)))
            print(" ", rep, {r["method"]: round(r["acc"], 3) for r in rows[-len(methods(0)):]}, flush=True)
        with open(os.path.join(RESULTS, "exp_uci_preview.json"), "w") as f:
            json.dump(rows, f, indent=1)
    return rows


def summarize(rows):
    keys = list(methods(0).keys())
    summ = {}
    for name in DATA:
        for k in keys:
            R = [r for r in rows if r["data"] == name and r["method"] == k]
            a = np.array([r["acc"] for r in R])
            summ[f"{name}|{k}"] = dict(mean=a.mean(), std=a.std(ddof=1),
                                       params=float(np.median([r["params"] for r in R])),
                                       pred_us=1e6 * float(np.median([r["pred_time"] for r in R])))
    with open(os.path.join(RESULTS, "exp_uci_preview_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    for name in DATA:
        print(name.ljust(12), " ".join(f"{k}={100*summ[f'{name}|{k}']['mean']:.1f}±{100*summ[f'{name}|{k}']['std']:.1f}"
                                       for k in keys))
    return summ


if __name__ == "__main__":
    summarize(run())
