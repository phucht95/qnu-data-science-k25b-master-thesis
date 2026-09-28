"""Section 3.4: training time, prediction time and model size on two larger data sets.

Single-threaded measurements (BLAS and OpenMP limited to one thread). Hyper-parameters are
fixed to values chosen beforehand by cross-validation on a subsample, so that only the cost
of one training run is measured. Prediction time: median over 7 repetitions of predicting
the whole test part, divided by its size.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

import credit_data
from datasets_uci import load
from exp_credit import n_params, scores
from pwlsvm import PWLSVM
from scorecard import credit_columns, make_scorecard, scorecard_params

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
C_GRID = [2.0 ** k for k in range(-7, 8, 2)]


def candidates(n, scorecard):
    return {
        "linear": (PWLSVM(kind="additive", m_per_dim=1, loss="hinge", solver="liblinear"), {"C": C_GRID}),
        "logit": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                  {"logisticregression__C": [1e-2, 1e-1, 1.0, 10.0]}),
        "logit_bin": (scorecard, {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0, 10.0]}),
        "pwl_c": (PWLSVM(kind="additive", m_per_dim=10, loss="hinge", solver="liblinear", knots="quantile"),
                  {"C": C_GRID}),
        "pwl_ls": (PWLSVM(kind="additive", m_per_dim=10, loss="squared", knots="quantile"), {"C": C_GRID}),
        "pwl_hh": (PWLSVM(kind="hh", m_per_dim=10, loss="squared", random_state=0), {"C": C_GRID}),
        "rbf": (make_pipeline(MinMaxScaler(), SVC(kernel="rbf")),
                {"svc__C": [0.25, 1.0, 4.0, 16.0, 64.0], "svc__gamma": [2.0 ** k for k in (-7, -5, -3, -1, 1)]}),
        "mlp": (make_pipeline(MinMaxScaler(), MLPClassifier(hidden_layer_sizes=(9 * n,), solver="lbfgs",
                                                            max_iter=500, random_state=0)),
                {"mlpclassifier__alpha": [1e-2, 1e-1, 1.0, 10.0]}),
        "hgb": (HistGradientBoostingClassifier(random_state=0), None),
    }


def measure(name, X, y, Xtr, ytr, Xte, yte, use_auc, scorecard):
    n = X.shape[1]
    sub = train_test_split(np.arange(len(ytr)), train_size=min(4000, len(ytr) - 1), stratify=ytr,
                           random_state=0)[0]
    rows = []
    for key, (est, grid) in candidates(n, scorecard).items():
        if grid is not None:
            gs = GridSearchCV(est, grid, cv=StratifiedKFold(3, shuffle=True, random_state=0),
                              scoring="roc_auc" if use_auc else "accuracy", n_jobs=1).fit(Xtr[sub], ytr[sub])
            est = est.set_params(**gs.best_params_)
        t0 = time.perf_counter(); est.fit(Xtr, ytr); t_fit = time.perf_counter() - t0
        ts = []
        for _ in range(7):
            t0 = time.perf_counter(); s = scores(est, Xte); ts.append(time.perf_counter() - t0)
        t_pred = float(np.median(ts)) / len(yte)
        acc = float((est.predict(Xte) == yte).mean())
        auc = float(roc_auc_score(yte, s))
        par = scorecard_params(est) if key == "logit_bin" else n_params(est, n)
        rows.append(dict(data=name, method=key, t_fit=t_fit, t_pred=t_pred, params=par,
                         acc=acc, auc=auc, Ntrain=len(ytr)))
        print(name, key, f"fit={t_fit:.2f}s pred={1e6 * t_pred:.2f}us params={rows[-1]['params']} "
                         f"acc={acc:.3f} auc={auc:.3f}", flush=True)
    return rows


def main():
    # timing needs an otherwise idle machine: wait for the parallel Spambase runs to finish
    here = os.path.dirname(os.path.abspath(__file__))
    for flag in ("results_spam.done", "results_extra.done"):      # extra: the short credit analyses
        while not os.path.exists(os.path.join(here, flag)):
            time.sleep(15)
    rows = []
    X, y = load("Magic")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
    rows += measure("Magic", X, y, Xtr, ytr, Xte, yte, use_auc=False,
                    scorecard=make_scorecard([], list(range(X.shape[1])), []))
    Xdf, yc, _ = credit_data.load()
    Xc = Xdf.to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.3, stratify=yc, random_state=0)
    rows += measure("Credit", Xc, yc, Xtr, ytr, Xte, yte, use_auc=True,
                    scorecard=make_scorecard(*credit_columns(list(Xdf.columns))))
    with open(os.path.join(RESULTS, "cost.json"), "w") as f:
        json.dump(rows, f)


if __name__ == "__main__":
    main()
