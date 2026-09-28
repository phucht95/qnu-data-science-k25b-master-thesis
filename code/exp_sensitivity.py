"""Section 3.3.3: effect of the number of features M/n and of the parameter rules.

PWL-LS-SVM (ridge primal, fast) with four feature maps:
    additive-uniform (rule of the paper), additive-quantile,
    HH with |p(1)| = 1 (rule of the paper), HH with ||p||_2 = 1 (Remark 2.9).
Three data sets, 5 random splits (same protocol as exp_benchmark), gamma chosen by 5-fold CV.
"""

import json
import os

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from datasets_uci import load
from exp_benchmark import splits
from pwlsvm import PWLSVM

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
C_GRID = [2.0 ** k for k in range(-5, 12, 2)]
MPD = [1, 2, 3, 5, 10, 20]
VARIANTS = {
    "add_uni": dict(kind="additive", knots="uniform"),
    "add_q": dict(kind="additive", knots="quantile"),
    "hh": dict(kind="hh", normalize=False),
    "hh_norm": dict(kind="hh", normalize=True),
}


def run(names=("Ionosphere", "Magic", "Spambase"), n_rep=5):
    rows = []
    for name in names:
        X, y = load(name)
        for rep, (tr, te) in enumerate(splits(name, X, y, n_rep)):
            for key, kw in VARIANTS.items():
                for mpd in MPD:
                    est = PWLSVM(m_per_dim=mpd, loss="squared", random_state=rep, **kw)
                    gs = GridSearchCV(est, {"C": C_GRID}, cv=StratifiedKFold(5, shuffle=True, random_state=rep),
                                      n_jobs=-1).fit(X[tr], y[tr])
                    m = gs.best_estimator_
                    rows.append(dict(data=name, rep=rep, variant=key, mpd=mpd,
                                     acc=float((m.predict(X[te]) == y[te]).mean()),
                                     M=int(m.coef_.size), C=gs.best_params_["C"]))
            print(name, rep, flush=True)
        with open(os.path.join(RESULTS, "sensitivity.json"), "w") as f:
            json.dump(rows, f)
    return rows


if __name__ == "__main__":
    run()
