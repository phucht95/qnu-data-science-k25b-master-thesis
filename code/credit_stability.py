"""Section 3.5.4: stability of the shape functions of the additive PWL-LS-SVM over the five splits.

For every split of exp_credit.py the model is refitted on the training part with the gamma chosen
there; the centred contributions h_i are evaluated on a common grid (0.5%-99.5% quantiles of the
whole data set, or the observed values for discrete variables).
"""

import json
import os

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import credit_data
from exp_credit import N_REP, RESULTS
from pwlsvm import PWLSVM

Xdf, y, _ = credit_data.load()
X = Xdf.to_numpy(); names = list(Xdf.columns)
res = json.load(open(os.path.join(RESULTS, "credit.json")))
grid = {}
for i, nm in enumerate(names):
    vals = np.unique(X[:, i])
    grid[nm] = vals if vals.size <= 15 else np.linspace(*np.quantile(X[:, i], [0.005, 0.995]), 400)

out = dict(names=names, grid={nm: g.tolist() for nm, g in grid.items()}, splits=[],
           q05={nm: float(np.quantile(X[:, i], 0.05)) for i, nm in enumerate(names)},
           q95={nm: float(np.quantile(X[:, i], 0.95)) for i, nm in enumerate(names)})
for rep, (tr, te) in enumerate(StratifiedShuffleSplit(n_splits=N_REP, test_size=0.3, random_state=2025).split(X, y)):
    C = next(r for r in res["rows"] if r["rep"] == rep and r["method"] == "pwl_ls_q")["best"]["C"]
    model = PWLSVM(kind="additive", m_per_dim=10, loss="squared", knots="quantile", C=C).fit(X[tr], y[tr])
    comp = model.additive_components(X[tr])
    mean_c = comp.mean(axis=0)
    ref = np.median(X[tr], axis=0)
    h = {}
    for i, nm in enumerate(names):
        Z = np.tile(ref, (grid[nm].size, 1)); Z[:, i] = grid[nm]
        h[nm] = (model.additive_components(Z)[:, i] - mean_c[i]).tolist()
    imp = {nm: float(np.std(comp[:, i])) for i, nm in enumerate(names)}
    out["splits"].append(dict(rep=rep, C=C, h=h, importance=imp))
    print(rep, "C =", C, "top-6:", sorted(imp, key=lambda k: -imp[k])[:6], flush=True)
bill = [names.index(f"BILL_AMT{k}") for k in range(1, 7)]
out["corr_bill_consecutive"] = [float(np.corrcoef(X[:, bill[k]], X[:, bill[k + 1]])[0, 1]) for k in range(5)]
json.dump(out, open(os.path.join(RESULTS, "credit_shapes_all.json"), "w"))
