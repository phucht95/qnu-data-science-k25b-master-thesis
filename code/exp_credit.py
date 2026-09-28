"""Section 3.5: application to the default of credit card clients data (Yeh & Lien, 2009).

Protocol: 5 stratified 70/30 splits; hyper-parameters chosen by 5-fold cross-validation
on the training part with the area under the ROC curve (AUC) as criterion. For the
Gaussian-kernel SVM and the MLP, tuning uses a stratified subsample of the training part
(cost), and the final model is refitted on the whole training part.
Classification threshold: chosen on the training part so that the predicted default rate
equals the observed training default rate.
"""

import json
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

import credit_data
from pwlsvm import PWLSVM

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

C_GRID = [2.0 ** k for k in range(-9, 8, 2)]          # 2^-9, ..., 2^7
N_REP = 5


def scores(model, X):
    if isinstance(model, HistGradientBoostingClassifier) or (
            hasattr(model, "steps") and isinstance(model[-1], (MLPClassifier, LogisticRegression))):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def n_params(model, n):
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "n_stored_parameters"):
        return int(est.n_stored_parameters())
    if isinstance(est, SVC):
        return int(est.support_vectors_.size + est.dual_coef_.size + 1 + 2 * n)
    if isinstance(est, MLPClassifier):
        return int(sum(c.size for c in est.coefs_) + sum(b.size for b in est.intercepts_) + 2 * n)
    if isinstance(est, LogisticRegression):
        return int(est.coef_.size + 1 + 2 * n)
    if isinstance(est, HistGradientBoostingClassifier):
        # every tree node stores a threshold and a feature index / leaf value
        return int(sum(p.nodes.size * 2 for it in est._predictors for p in it))
    return -1


def methods(n, rep):
    return {
        "logit": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                  {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0, 10.0]}, None),
        "linear": (PWLSVM(kind="additive", m_per_dim=1, loss="hinge", solver="liblinear"),
                   {"C": C_GRID}, None),
        "pwl_ls_uni": (PWLSVM(kind="additive", m_per_dim=10, loss="squared", knots="uniform"),
                       {"C": C_GRID}, None),
        "pwl_ls_q": (PWLSVM(kind="additive", m_per_dim=10, loss="squared", knots="quantile"),
                     {"C": C_GRID}, None),
        "pwl_c_q": (PWLSVM(kind="additive", m_per_dim=10, loss="hinge", solver="liblinear",
                           knots="quantile"), {"C": C_GRID}, None),
        "pwl_ls_hh": (PWLSVM(kind="hh", m_per_dim=10, loss="squared", random_state=rep),
                      {"C": C_GRID}, None),
        "rbf": (make_pipeline(MinMaxScaler(), SVC(kernel="rbf")),
                {"svc__C": [0.25, 1.0, 4.0, 16.0], "svc__gamma": [2.0 ** k for k in (-7, -5, -3, -1)]}, 5000),
        "mlp": (make_pipeline(MinMaxScaler(), MLPClassifier(hidden_layer_sizes=(9 * n,), activation="relu",
                                                            solver="lbfgs", max_iter=500, random_state=rep)),
                {"mlpclassifier__alpha": [1e-2, 1e-1, 1.0, 10.0]}, 7000),
        "hgb": (HistGradientBoostingClassifier(random_state=rep), None, None),
    }


def rate_threshold(s_train, pi):
    return np.quantile(s_train, 1 - pi)


def run():
    Xdf, y, _ = credit_data.load()
    X = Xdf.to_numpy(); n = X.shape[1]
    rows, roc = [], {}
    sss = StratifiedShuffleSplit(n_splits=N_REP, test_size=0.3, random_state=2025)
    for rep, (tr, te) in enumerate(sss.split(X, y)):
        pi = (y[tr] == 1).mean()
        for key, (est, grid, sub) in methods(n, rep).items():
            t0 = time.perf_counter()
            if grid is None:
                model, best = est.fit(X[tr], y[tr]), {}
            else:
                idx = tr if sub is None else train_test_split(tr, train_size=sub, stratify=y[tr], random_state=rep)[0]
                folds = 5 if sub is None else 3
                gs = GridSearchCV(est, grid, cv=StratifiedKFold(folds, shuffle=True, random_state=rep),
                                  scoring="roc_auc", n_jobs=-1).fit(X[idx], y[idx])
                best = gs.best_params_
                model = est.set_params(**best)
            t_tune = time.perf_counter() - t0
            t0 = time.perf_counter(); model.fit(X[tr], y[tr]); t_fit = time.perf_counter() - t0
            s_tr = scores(model, X[tr])
            t0 = time.perf_counter(); s_te = scores(model, X[te]); t_pred = (time.perf_counter() - t0) / len(te)
            thr = rate_threshold(s_tr, pi)
            yhat = np.where(s_te > thr, 1, -1)
            rows.append(dict(rep=rep, method=key, auc=float(roc_auc_score(y[te], s_te)),
                             bacc=float(balanced_accuracy_score(y[te], yhat)),
                             f1=float(f1_score(y[te], yhat, pos_label=1)),
                             acc=float((yhat == y[te]).mean()),
                             t_tune=t_tune, t_fit=t_fit, t_pred=t_pred, params=n_params(model, n),
                             best={k: float(v) for k, v in best.items()}))
            if rep == 0:
                fpr, tpr, _ = roc_curve(y[te], s_te)
                keep = np.unique(np.r_[0, np.linspace(0, len(fpr) - 1, 400).astype(int), len(fpr) - 1])
                roc[key] = dict(fpr=fpr[keep].tolist(), tpr=tpr[keep].tolist())
            print(rep, key, {k: round(v, 4) for k, v in rows[-1].items() if k in ("auc", "bacc", "f1", "t_fit")},
                  flush=True)
        with open(os.path.join(RESULTS, "credit.json"), "w") as f:
            json.dump(dict(rows=rows, roc=roc), f)
    return rows


def interpret():
    """Shape functions of the additive PWL-LS-SVM (quantile knots) fitted on the first split."""
    Xdf, y, raw = credit_data.load()
    X = Xdf.to_numpy(); names = list(Xdf.columns)
    tr, te = next(StratifiedShuffleSplit(n_splits=N_REP, test_size=0.3, random_state=2025).split(X, y))
    res = json.load(open(os.path.join(RESULTS, "credit.json")))
    C = [r for r in res["rows"] if r["rep"] == 0 and r["method"] == "pwl_ls_q"][0]["best"]["C"]
    model = PWLSVM(kind="additive", m_per_dim=10, loss="squared", knots="quantile", C=C).fit(X[tr], y[tr])
    comp = model.additive_components(X[tr])
    mean_c = comp.mean(axis=0)
    out = dict(C=C, names=names, intercept=float(model.intercept_[0]), shapes={}, importance={})
    ref = np.median(X[tr], axis=0)
    for i, nm in enumerate(names):
        vals = np.unique(X[tr, i])
        if vals.size <= 15:                                   # discrete variable
            grid = vals
        else:
            lo, hi = np.quantile(X[tr, i], [0.005, 0.995])
            grid = np.linspace(lo, hi, 400)
        Z = np.tile(ref, (grid.size, 1)); Z[:, i] = grid
        h = model.additive_components(Z)[:, i] - mean_c[i]
        knots = model.scaler_.data_min_[i] + model.features_.knot_[model.features_.axis_ == i] * \
            (model.scaler_.data_max_[i] - model.scaler_.data_min_[i])
        out["shapes"][nm] = dict(x=grid.tolist(), h=h.tolist(), knots=knots.tolist(),
                                 q01=float(np.quantile(X[tr, i], 0.01)), q99=float(np.quantile(X[tr, i], 0.99)))
        out["importance"][nm] = float(np.std(comp[:, i]))
    # nominal inputs: importance of the whole group of indicators = std of their summed contribution
    groups = {"EDUCATION": ["EDU_GRAD", "EDU_UNIV", "EDU_HIGH"], "MARRIAGE": ["MARRIED", "SINGLE"], "SEX": ["FEMALE"]}
    out["importance_groups"] = {g: float(np.std(comp[:, [names.index(c) for c in cols]].sum(axis=1)))
                                for g, cols in groups.items()}
    # explain the test client with the largest decision value among actual defaults
    s = model.decision_function(X[te])
    k = te[np.argmax(np.where(y[te] == 1, s, -np.inf))]
    ck = model.additive_components(X[[k]])[0] - mean_c
    out["client"] = dict(index=int(k), score=float(model.decision_function(X[[k]])[0]),
                         base=float(model.intercept_[0] + mean_c.sum()),
                         contrib={nm: float(v) for nm, v in zip(names, ck)},
                         values={nm: float(raw.iloc[k][nm]) if nm in raw.columns else float(X[k, i])
                                 for i, nm in enumerate(names)})
    out["threshold"] = float(rate_threshold(model.decision_function(X[tr]), (y[tr] == 1).mean()))
    with open(os.path.join(RESULTS, "credit_interpret.json"), "w") as f:
        json.dump(out, f)
    print("interpretation saved; C =", C)


if __name__ == "__main__":
    run()
    interpret()
