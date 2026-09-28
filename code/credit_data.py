"""Default of credit card clients (Yeh & Lien, 2009): loading and preprocessing."""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# x1..x23 in the OpenML copy (id 42477) follow the order of the UCI description
RAW_NAMES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
             "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
             "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
             "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

LABEL_VI = {
    "LIMIT_BAL": "hạn mức tín dụng", "AGE": "tuổi",
    "PAY_0": "trạng thái trả nợ tháng 9", "PAY_2": "trạng thái trả nợ tháng 8",
    "PAY_3": "trạng thái trả nợ tháng 7", "PAY_4": "trạng thái trả nợ tháng 6",
    "PAY_5": "trạng thái trả nợ tháng 5", "PAY_6": "trạng thái trả nợ tháng 4",
    "BILL_AMT1": "dư nợ sao kê tháng 9", "PAY_AMT1": "số tiền đã trả tháng 9",
    "PAY_AMT2": "số tiền đã trả tháng 8",
}


def load_raw():
    d = fetch_openml(data_id=42477, as_frame=True, parser="auto")
    df = d.data.copy()
    df.columns = RAW_NAMES
    df = df.astype(float)
    y = np.where(d.target.astype(str).to_numpy() == "1", 1, -1)
    return df, y


def preprocess(df):
    """Numerical design matrix: undocumented category codes merged, nominal variables one-hot encoded.

    SEX: 1 male, 2 female -> indicator FEMALE.
    EDUCATION: 1 graduate school, 2 university, 3 high school, others (0, 4, 5, 6) -> "khác".
    MARRIAGE: 1 married, 2 single, others (0, 3) -> "khác".
    PAY_*: ordinal repayment status, kept as numbers (-2, -1, 0 = no delay; k = k months of delay).
    Monetary amounts and AGE: kept as numbers.
    """
    out = pd.DataFrame(index=df.index)
    out["LIMIT_BAL"] = df["LIMIT_BAL"]
    out["AGE"] = df["AGE"]
    for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        out[c] = df[c]
    for c in [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]:
        out[c] = df[c]
    out["FEMALE"] = (df["SEX"] == 2).astype(float)
    edu = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3]), 4)
    out["EDU_GRAD"] = (edu == 1).astype(float)
    out["EDU_UNIV"] = (edu == 2).astype(float)
    out["EDU_HIGH"] = (edu == 3).astype(float)
    mar = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2]), 3)
    out["MARRIED"] = (mar == 1).astype(float)
    out["SINGLE"] = (mar == 2).astype(float)
    return out


def load():
    df, y = load_raw()
    X = preprocess(df)
    return X, y, df
