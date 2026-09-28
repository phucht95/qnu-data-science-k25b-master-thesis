"""Section 3.5.5: additive PWL-LS-SVM with monotonicity constraints on the credit data.

Constraints (business knowledge): the contribution of every repayment status PAY_0, PAY_2..PAY_6 is
non-decreasing from the status 0 on (more months of delay never lower the risk; the codes -2, -1, 0 are
different "no delay" situations without a natural order and stay free) and that of the credit limit
LIMIT_BAL is non-increasing.  Same five splits, same gamma (chosen by cross-validation for the unconstrained model in
exp_credit.py) and same threshold rule as exp_credit.py.
"""

import json
import os

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

import credit_data
from exp_credit import N_REP, RESULTS, rate_threshold
from pwlsvm import PWLSVM
from pwlsvm_monotone import MonotonePWLLSSVM

PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
SHOW = ["PAY_0", "PAY_4", "LIMIT_BAL"]

Xdf, y, _ = credit_data.load()
X = Xdf.to_numpy(); names = list(Xdf.columns)
mono = {names.index(c): (+1, 0.0) for c in PAY}
mono[names.index("LIMIT_BAL")] = -1
res = json.load(open(os.path.join(RESULTS, "credit.json")))
rows, shapes, viol, check = [], {}, [], []
for rep, (tr, te) in enumerate(StratifiedShuffleSplit(n_splits=N_REP, test_size=0.3, random_state=2025).split(X, y)):
    C = next(r for r in res["rows"] if r["rep"] == rep and r["method"] == "pwl_ls_q")["best"]["C"]
    pi = (y[tr] == 1).mean()
    for key, m in (("free", MonotonePWLLSSVM(C=C)), ("monotone", MonotonePWLLSSVM(C=C, monotone=mono))):
        m.fit(X[tr], y[tr])
        s_tr, s_te = m.decision_function(X[tr]), m.decision_function(X[te])
        yhat = np.where(s_te > rate_threshold(s_tr, pi), 1, -1)
        rows.append(dict(rep=rep, method=key, C=C, auc=float(roc_auc_score(y[te], s_te)),
                         bacc=float(balanced_accuracy_score(y[te], yhat)),
                         f1=float(f1_score(y[te], yhat, pos_label=1)), params=int(m.n_stored_parameters())))
        if key == "free":        # same model as PWLSVM(...).fit: check, then count contradicting segments
            ridge = PWLSVM(kind="additive", m_per_dim=10, loss="squared", knots="quantile", C=C).fit(X[tr], y[tr])
            check.append(float(np.abs(m.coef_ - ridge.coef_).max()))
            bad, nseg = {}, {}
            for nm in PAY + ["LIMIT_BAL"]:
                i = names.index(nm)
                sgn, mask = m.constraint_mask(i, monotone=mono)
                bad[nm] = int(np.sum((np.asarray(m.slopes_[i]) * sgn < -1e-10) & mask))
                nseg[nm] = int(mask.sum())
            viol.append(dict(rep=rep, bad=bad, nseg=nseg))
        if rep == 0:
            comp = m.additive_components(X[tr]); mean_c = comp.mean(axis=0)
            ref = np.median(X[tr], axis=0)
            shapes[key] = {}
            for nm in SHOW:
                i = names.index(nm)
                vals = np.unique(X[tr, i])
                grid = vals if vals.size <= 15 else np.linspace(*np.quantile(X[tr, i], [0.005, 0.995]), 400)
                Z = np.tile(ref, (grid.size, 1)); Z[:, i] = grid
                knots = m.scaler_.data_min_[i] + m.features_.knot_[m.features_.axis_ == i] * \
                    (m.scaler_.data_max_[i] - m.scaler_.data_min_[i])
                shapes[key][nm] = dict(x=grid.tolist(), h=(m.additive_components(Z)[:, i] - mean_c[i]).tolist(),
                                       knots=knots.tolist())
        print(rep, key, {k: round(v, 4) for k, v in rows[-1].items() if k in ("auc", "bacc", "f1")}, flush=True)
    print("   violating segments of the free model:", viol[-1]["bad"], flush=True)
# rare values: share of all clients with a repayment status >= 3, default rate by value of PAY_0
p0 = X[:, names.index("PAY_0")]
json.dump(dict(rows=rows, shapes=shapes, violations=viol, max_coef_diff_free_vs_ridge=max(check),
               share_ge3={nm: float((X[:, names.index(nm)] >= 3).mean()) for nm in PAY},
               share_pay0_ge3=float((p0 >= 3).mean()),
               default_rate_pay0={str(int(v)): float((y[p0 == v] == 1).mean()) for v in np.unique(p0)},
               count_pay0={str(int(v)): int((p0 == v).sum()) for v in np.unique(p0)}),
          open(os.path.join(RESULTS, "credit_monotone.json"), "w"))
