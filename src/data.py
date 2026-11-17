"""Đọc dữ liệu Telco Customer Churn, làm sạch, chia Train/Val/Test (stratified) và tiền xử lý.

Quy ước chống rò rỉ dữ liệu (data leakage): OneHotEncoder/StandardScaler/SMOTE chỉ được fit trên TẬP TRAIN.
"""
import hashlib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_SCALED = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_raw(path, target="Churn", id_col="customerID", impute="zero"):
    df = pd.read_csv(path)
    n_blank = int((df["TotalCharges"].astype(str).str.strip() == "").sum()) if "TotalCharges" in df else 0
    if "TotalCharges" in df:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        if impute == "zero":      # khách mới (tenure=0) chưa phát sinh cước
            df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
        else:
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df[target] = df[target].map({"No": 0, "Yes": 1}).astype(int)
    ids = df[id_col].copy() if id_col in df else pd.Series(range(len(df)), name="id")
    X = df.drop(columns=[c for c in (target, id_col) if c in df])
    return df, X, df[target].copy(), ids, n_blank


def split_data(X, y, ids, test_size, val_size, seed):
    """val_size là tỷ lệ so với TOÀN BỘ dữ liệu (vd 0.1 => 10% tổng)."""
    X_tmp, X_te, y_tmp, y_te, id_tmp, id_te = train_test_split(
        X, y, ids, test_size=test_size, random_state=seed, stratify=y)
    if val_size and val_size > 0:
        rel = val_size / (1.0 - test_size)
        X_tr, X_va, y_tr, y_va, id_tr, id_va = train_test_split(
            X_tmp, y_tmp, id_tmp, test_size=rel, random_state=seed, stratify=y_tmp)
    else:
        X_tr, y_tr, id_tr = X_tmp, y_tmp, id_tmp
        X_va, y_va, id_va = X_tmp.iloc[:0], y_tmp.iloc[:0], id_tmp.iloc[:0]
    return (X_tr, y_tr, id_tr), (X_va, y_va, id_va), (X_te, y_te, id_te)


def build_preprocessor(X_train):
    cat_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    num_scaled = [c for c in NUMERIC_SCALED if c in X_train.columns]
    passthrough = [c for c in X_train.columns if c not in cat_cols + num_scaled]   # SeniorCitizen (0/1)
    ct = ColumnTransformer([
        ("num", StandardScaler(), num_scaled),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), cat_cols),
        ("bin", "passthrough", passthrough),
    ], verbose_feature_names_out=False)
    return ct, {"numeric_scaled": num_scaled, "categorical_onehot": cat_cols, "passthrough": passthrough}


def transform(ct, X):
    arr = ct.transform(X)
    return pd.DataFrame(arr, columns=ct.get_feature_names_out(), index=X.index).astype(float)


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_info(path, df, X, splits, pre_groups, n_blank, n_feat_encoded, feature_names, args):
    (_, ytr, _), (_, yva, _), (_, yte, _) = splits

    def cnt(y):
        n1 = int(y.sum()); n0 = int(len(y) - n1)
        return {"n": int(len(y)), "retained(0)": n0, "churned(1)": n1,
                "churn_rate": round(n1 / max(len(y), 1), 4)}

    return {
        "dataset_name": "Telco Customer Churn (IBM sample data / Kaggle)",
        "file": path,
        "file_md5": file_md5(path),
        "n_samples": int(len(df)),
        "n_raw_input_features": int(X.shape[1]),
        "n_features_after_encoding": int(n_feat_encoded),
        "feature_names_after_encoding": list(feature_names),
        "raw_feature_groups": pre_groups,
        "n_missing_TotalCharges_blank": n_blank,
        "imputation": args.impute,
        "class_distribution_total": cnt(df["Churn"]),
        "imbalance_ratio_retained_to_churn": round(float((df["Churn"] == 0).sum() / (df["Churn"] == 1).sum()), 4),
        "split": {"strategy": "stratified hold-out", "test_size": args.test_size, "val_size": args.val_size,
                  "random_state": args.random_state,
                  "train": cnt(ytr), "val": cnt(yva), "test": cnt(yte)},
        "leakage_policy": "encoder/scaler/SMOTE fit on train only",
    }
