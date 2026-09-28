"""Section 3.5: the classical scorecard (binning + logistic regression) on the credit data.

Same five splits, same selection criterion (5-fold cross-validated AUC on the training part) and same
threshold rule as exp_credit.py; results are stored separately in results/credit_scorecard.json.
"""

import json
import os
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit

import credit_data
from exp_credit import N_REP, RESULTS, rate_threshold
from scorecard import credit_columns, make_scorecard, scorecard_params

Xdf, y, _ = credit_data.load()
X = Xdf.to_numpy()
cols = credit_columns(list(Xdf.columns))
rows, roc = [], {}
for rep, (tr, te) in enumerate(StratifiedShuffleSplit(n_splits=N_REP, test_size=0.3, random_state=2025).split(X, y)):
    pi = (y[tr] == 1).mean()
    t0 = time.perf_counter()
    gs = GridSearchCV(make_scorecard(*cols), {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1.0, 10.0]},
                      cv=StratifiedKFold(5, shuffle=True, random_state=rep), scoring="roc_auc", n_jobs=2).fit(X[tr], y[tr])
    t_tune = time.perf_counter() - t0
    model = gs.best_estimator_
    s_tr = model.predict_proba(X[tr])[:, 1]
    t0 = time.perf_counter(); s_te = model.predict_proba(X[te])[:, 1]; t_pred = (time.perf_counter() - t0) / len(te)
    yhat = np.where(s_te > rate_threshold(s_tr, pi), 1, -1)
    rows.append(dict(rep=rep, method="logit_bin", auc=float(roc_auc_score(y[te], s_te)),
                     bacc=float(balanced_accuracy_score(y[te], yhat)), f1=float(f1_score(y[te], yhat, pos_label=1)),
                     acc=float((yhat == y[te]).mean()), t_tune=t_tune, t_pred=t_pred,
                     params=scorecard_params(model), best={k: float(v) for k, v in gs.best_params_.items()}))
    if rep == 0:
        fpr, tpr, _ = roc_curve(y[te], s_te)
        keep = np.unique(np.r_[0, np.linspace(0, len(fpr) - 1, 400).astype(int), len(fpr) - 1])
        roc["logit_bin"] = dict(fpr=fpr[keep].tolist(), tpr=tpr[keep].tolist())
    print(rep, {k: round(v, 4) for k, v in rows[-1].items() if k in ("auc", "bacc", "f1")}, rows[-1]["best"], flush=True)
json.dump(dict(rows=rows, roc=roc), open(os.path.join(RESULTS, "credit_scorecard.json"), "w"))
