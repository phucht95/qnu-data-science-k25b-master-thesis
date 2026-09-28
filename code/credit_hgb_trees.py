"""Number of trees of the boosted ensembles of exp_credit.py (early stopping is on by default)."""

import json
import os

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit

import credit_data
from exp_credit import N_REP, RESULTS

Xdf, y, _ = credit_data.load()
X = Xdf.to_numpy()
out = []
for rep, (tr, te) in enumerate(StratifiedShuffleSplit(n_splits=N_REP, test_size=0.3, random_state=2025).split(X, y)):
    m = HistGradientBoostingClassifier(random_state=rep).fit(X[tr], y[tr])
    out.append(dict(rep=rep, trees=int(m.n_iter_), params=int(sum(p.nodes.size * 2 for it in m._predictors for p in it))))
    print(out[-1], flush=True)
json.dump(out, open(os.path.join(RESULTS, "credit_hgb_trees.json"), "w"))
