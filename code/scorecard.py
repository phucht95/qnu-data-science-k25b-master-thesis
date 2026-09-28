"""Classical credit scorecard: logistic regression on discretized inputs (piecewise constant, additive).

Continuous inputs are cut into 10 bins at their empirical deciles, ordinal repayment statuses are
one-hot encoded value by value, indicators are passed through; a logistic regression with an L2
penalty is fitted on the resulting 0/1 design (the usual "binning + logistic regression" scorecard).
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


def make_scorecard(cat_idx, cont_idx, pass_idx, n_bins=10):
    parts = []
    if cat_idx:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_idx)))
    if cont_idx:
        parts.append(("cont", KBinsDiscretizer(n_bins=n_bins, encode="onehot", strategy="quantile"), list(cont_idx)))
    if pass_idx:
        parts.append(("pass", "passthrough", list(pass_idx)))
    return make_pipeline(ColumnTransformer(parts), LogisticRegression(max_iter=5000))


def credit_columns(names):
    pay = [names.index(c) for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]]
    cont = [names.index(c) for c in ["LIMIT_BAL", "AGE"] + [f"BILL_AMT{k}" for k in range(1, 7)]
            + [f"PAY_AMT{k}" for k in range(1, 7)]]
    rest = [i for i in range(len(names)) if i not in pay + cont]
    return pay, cont, rest


def scorecard_params(model):
    """Numbers stored at prediction time: interior bin edges, category values, coefficients, intercept."""
    ct, lr = model[0], model[-1]
    k = lr.coef_.size + lr.intercept_.size
    for name, tr, _ in ct.transformers_:
        if name == "cont":
            k += sum(max(e.size - 2, 0) for e in tr.bin_edges_)
        elif name == "cat":
            k += sum(c.size for c in tr.categories_)
    return int(k)
