# -*- coding: utf-8 -*-
"""Phần xử lý (không có giao diện) của ứng dụng Streamlit dự báo Churn + giải thích SHAP.

Dùng lại đúng mã nguồn của nhóm (thư mục src/): đọc dữ liệu, chia Train/Val/Test (70/10/20, seed 42),
tiền xử lý chỉ fit trên Train, XGBoost cùng siêu tham số mặc định như run_experiment.py,
ngưỡng chọn tối đa F1 trên tập Validation.
"""
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import data as D  # noqa: E402

SEED, TEST_SIZE, VAL_SIZE = 42, 0.2, 0.1
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
RISK_BINS = [-0.01, 0.3, 0.6, 1.01]
RISK_LABELS = ["Thấp", "Trung bình", "Cao"]
XGB_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, random_state=SEED, n_jobs=-1, tree_method="hist", eval_metric=["logloss", "error"])


@dataclass
class Bundle:
    df: pd.DataFrame
    Xraw: pd.DataFrame
    y: pd.Series
    raw_columns: list
    ct: object
    model: object
    explainer: object
    thr: float
    feature_names: list
    cat_options: dict
    num_ranges: dict
    X_te_raw: pd.DataFrame
    y_te: pd.Series
    ids_te: pd.Series
    Xte_enc: pd.DataFrame
    p_te: np.ndarray
    shap_te: np.ndarray
    base_value: float
    cid_pos: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


def risk_group(p):
    return pd.cut(pd.Series(np.atleast_1d(p)), RISK_BINS, labels=RISK_LABELS).astype(str).values


def best_f1_threshold(y_val, p_val):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_val, (p_val >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def load_bundle(data_path):
    """Huấn luyện XGBoost (khoảng vài giây) và tính SHAP cho toàn bộ tập Test."""
    from xgboost import XGBClassifier
    import shap

    df, Xraw, y, ids, _ = D.load_raw(data_path)
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, id_te) = D.split_data(Xraw, y, ids, TEST_SIZE, VAL_SIZE, SEED)
    ct, _ = D.build_preprocessor(X_tr)
    ct.fit(X_tr)   # chỉ fit trên tập Train (chống rò rỉ dữ liệu)
    Xtr, Xva, Xte = D.transform(ct, X_tr), D.transform(ct, X_va), D.transform(ct, X_te)

    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    model = XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
    model.fit(Xtr, y_tr)

    thr = best_f1_threshold(y_va, model.predict_proba(Xva)[:, 1])
    p_te = model.predict_proba(Xte)[:, 1]

    explainer = shap.TreeExplainer(model)
    ex = explainer(Xte)
    shap_te = np.asarray(ex.values)
    base_value = float(np.asarray(ex.base_values).reshape(-1)[0])

    cat_cols = [c for c in Xraw.columns if not pd.api.types.is_numeric_dtype(Xraw[c])]
    cat_options = {c: sorted(Xraw[c].dropna().unique().tolist()) for c in cat_cols}
    cat_options["SeniorCitizen"] = [0, 1]
    num_ranges = {c: (float(Xraw[c].min()), float(Xraw[c].max())) for c in NUMERIC}

    pred = (p_te >= thr).astype(int)
    metrics = {"Accuracy": accuracy_score(y_te, pred), "Precision": precision_score(y_te, pred, zero_division=0),
               "Recall": recall_score(y_te, pred, zero_division=0), "F1": f1_score(y_te, pred, zero_division=0),
               "ROC-AUC": roc_auc_score(y_te, p_te), "PR-AUC": average_precision_score(y_te, p_te)}
    ids_te = id_te.reset_index(drop=True)
    b = Bundle(df=df, Xraw=Xraw, y=y, raw_columns=list(Xraw.columns), ct=ct, model=model, explainer=explainer,
               thr=thr, feature_names=list(Xte.columns), cat_options=cat_options, num_ranges=num_ranges,
               X_te_raw=X_te.reset_index(drop=True), y_te=y_te.reset_index(drop=True), ids_te=ids_te,
               Xte_enc=Xte.reset_index(drop=True), p_te=p_te, shap_te=shap_te, base_value=base_value,
               metrics=metrics)
    b.cid_pos = {cid: i for i, cid in enumerate(ids_te)}
    return b


def predict_row(b, row):
    """row: dict {cột gốc: giá trị}. Trả về (xác suất churn, DataFrame đã mã hóa 1 dòng)."""
    X1 = pd.DataFrame([row], columns=b.raw_columns)
    for c in b.raw_columns:
        if c in b.num_ranges or c == "SeniorCitizen":
            X1[c] = pd.to_numeric(X1[c])
    enc = D.transform(b.ct, X1)
    return float(b.model.predict_proba(enc)[:, 1][0]), enc


def explain_row(b, enc, raw_row):
    """Trả về shap.Explanation 1 dòng, trong đó giá trị hiển thị của biến số là giá trị gốc (không chuẩn hóa)."""
    import shap
    ex = b.explainer(enc)
    data = enc.copy()
    for c in NUMERIC:
        if c in data.columns:
            data[c] = float(raw_row[c])
    return shap.Explanation(values=np.asarray(ex.values)[0], base_values=float(np.asarray(ex.base_values).reshape(-1)[0]),
                            data=data.values[0], feature_names=list(enc.columns))


def readable_value(feature, enc_value, raw_row):
    """Chuyển đặc trưng đã mã hóa sang dạng dễ đọc, ví dụ Contract_Month-to-month=1 -> 'Contract = Month-to-month'."""
    if feature in NUMERIC:
        v = float(raw_row[feature])
        return f"{v:g}"
    if feature == "SeniorCitizen":
        return "Có" if int(round(enc_value)) == 1 else "Không"
    if "_" in feature:
        col, cat = feature.split("_", 1)
        return f"{col} = {cat}" if round(enc_value) == 1 else f"{col} ≠ {cat}"
    return f"{enc_value:g}"


def top_factors(ex_row, raw_row, n=8):
    vals = np.asarray(ex_row.values)
    order = np.argsort(-np.abs(vals))[:n]
    rows = []
    for i in order:
        f = ex_row.feature_names[i]
        enc_v = float(ex_row.data[i])
        rows.append({"Đặc trưng": f, "Giá trị": readable_value(f, enc_v, raw_row), "Tác động SHAP": round(float(vals[i]), 4),
                     "Hướng": "Tăng nguy cơ rời bỏ" if vals[i] > 0 else "Giảm nguy cơ rời bỏ"})
    return pd.DataFrame(rows)


def test_table(b):
    """Bảng dự báo cho toàn bộ tập Test kèm 3 yếu tố làm tăng nguy cơ nhiều nhất."""
    names = np.array(b.feature_names)
    top3 = []
    for i in range(len(b.p_te)):
        idx = np.argsort(-b.shap_te[i])[:3]
        top3.append(", ".join(names[j] for j in idx if b.shap_te[i, j] > 0))
    out = pd.DataFrame({"customerID": b.ids_te.values, "Xác suất churn": np.round(b.p_te, 4),
                        "Nhóm rủi ro": risk_group(b.p_te), "Churn thực tế": b.y_te.values,
                        "Dự báo": (b.p_te >= b.thr).astype(int), "Yếu tố làm tăng nguy cơ": top3})
    out["Hợp đồng"] = b.X_te_raw["Contract"].values
    out["tenure"] = b.X_te_raw["tenure"].values
    out["MonthlyCharges"] = b.X_te_raw["MonthlyCharges"].values
    return out
