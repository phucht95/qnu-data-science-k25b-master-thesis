"""Check for Section 3.3.2: PWL-C-SVM with HH features normalised to ||p||_2 = 1 on Spambase.

Same splits, grid and cross-validation as exp_benchmark.py (5 repetitions).
"""
import json
import os

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from datasets_uci import load
from exp_benchmark import C_GRID, splits
from pwlsvm import PWLSVM

X, y = load("Spambase")
rows = []
for rep, (tr, te) in enumerate(splits("Spambase", X, y, 5)):
    est = PWLSVM(kind="hh", m_per_dim=10, loss="hinge", solver="liblinear", normalize=True, random_state=rep)
    gs = GridSearchCV(est, {"C": C_GRID}, cv=StratifiedKFold(10, shuffle=True, random_state=rep), n_jobs=-1).fit(X[tr], y[tr])
    rows.append(dict(rep=rep, acc=float((gs.predict(X[te]) == y[te]).mean()), C=gs.best_params_["C"]))
    print(rows[-1], flush=True)
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "spam_hhnorm.json"), "w"))
