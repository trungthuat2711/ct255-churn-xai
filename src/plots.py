"""Toàn bộ biểu đồ. Mỗi hàm gọi logger.save_fig(...) nên tự đăng ký (file code:hàm:dòng) vào artifact_index.csv."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {0: "#2b7bba", 1: "#d1495b"}
LABEL = {0: "Retained (0)", 1: "Churned (1)"}


# =============================== EDA / HISTOGRAM ===============================
def eda_all(df_raw, out_dir, log, bins=30):
    d = os.path.join(out_dir, "eda")
    num = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df_raw]
    cats = [c for c in df_raw.columns if not pd.api.types.is_numeric_dtype(df_raw[c]) and c != 'customerID']

    # 1) Phân bố nhãn
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    vc = df_raw["Churn"].value_counts().sort_index()
    ax[0].bar([LABEL[i] for i in vc.index], vc.values, color=[PALETTE[i] for i in vc.index])
    for i, v in enumerate(vc.values):
        ax[0].text(i, v + 40, f"{v} ({v/len(df_raw):.1%})", ha="center")
    ax[0].set_title("Class distribution (Histogram of target)"); ax[0].set_ylabel("Customers")
    ax[1].pie(vc.values, labels=[LABEL[i] for i in vc.index], colors=[PALETTE[i] for i in vc.index], autopct="%1.1f%%")
    ax[1].set_title("Class ratio")
    log.save_fig(fig, os.path.join(d, "hist_class_distribution.png"), "Phân bố nhãn Churn")

    # 2) Histogram + KDE các thuộc tính số theo nhãn
    fig, axes = plt.subplots(1, len(num), figsize=(5.2 * len(num), 4))
    for ax, c in zip(np.atleast_1d(axes), num):
        sns.histplot(data=df_raw, x=c, hue="Churn", bins=bins, kde=True, stat="density", common_norm=False,
                     palette=PALETTE, alpha=0.45, ax=ax, element="step")
        ax.set_title(f"Histogram + KDE: {c}")
        leg = ax.get_legend()
        if leg:
            for t, k in zip(leg.get_texts(), [0, 1]):
                t.set_text(LABEL[k])
    log.save_fig(fig, os.path.join(d, "hist_numeric_by_churn.png"), "Histogram+KDE tenure/MonthlyCharges/TotalCharges theo nhãn")

    # 3) Histogram tổng (không tách nhãn) + thống kê
    fig, axes = plt.subplots(1, len(num), figsize=(5.2 * len(num), 4))
    for ax, c in zip(np.atleast_1d(axes), num):
        sns.histplot(df_raw[c], bins=bins, color="#555", ax=ax)
        ax.axvline(df_raw[c].mean(), color="r", ls="--", label=f"mean={df_raw[c].mean():.1f}")
        ax.axvline(df_raw[c].median(), color="g", ls=":", label=f"median={df_raw[c].median():.1f}")
        ax.set_title(f"Histogram: {c} (skew={df_raw[c].skew():.2f})"); ax.legend()
    log.save_fig(fig, os.path.join(d, "hist_numeric_overall.png"), "Histogram tổng thể các thuộc tính số + mean/median")

    # 4) Box plot
    fig, axes = plt.subplots(1, len(num), figsize=(5 * len(num), 4))
    for ax, c in zip(np.atleast_1d(axes), num):
        sns.boxplot(data=df_raw, x="Churn", y=c, palette=PALETTE, hue="Churn", legend=False, ax=ax)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Retained", "Churned"]); ax.set_title(f"Box plot: {c}")
    log.save_fig(fig, os.path.join(d, "box_numeric_by_churn.png"), "Box plot thuộc tính số theo nhãn")

    # 5) Histogram (bar) các thuộc tính phân loại theo nhãn
    ncol = 4
    nrow = int(np.ceil((len(cats)) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow))
    for ax, c in zip(axes.ravel(), cats):
        sns.countplot(data=df_raw, x=c, hue="Churn", palette=PALETTE, ax=ax)
        ax.set_title(c); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=25)
        if ax.get_legend():
            ax.get_legend().remove()
    for ax in axes.ravel()[len(cats):]:
        ax.axis("off")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=PALETTE[k]) for k in (0, 1)], [LABEL[0], LABEL[1]], loc="upper right")
    log.save_fig(fig, os.path.join(d, "hist_categorical_by_churn.png"), "Histogram (count) các thuộc tính phân loại theo nhãn")

    # 6) Tỷ lệ churn theo nhóm Contract / tenure bin
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cr = df_raw.groupby("Contract")["Churn"].mean().sort_values()
    ax[0].bar(cr.index, cr.values, color="#d1495b"); ax[0].set_title("Churn rate by Contract"); ax[0].set_ylabel("Churn rate")
    for i, v in enumerate(cr.values):
        ax[0].text(i, v + 0.01, f"{v:.1%}", ha="center")
    tb = pd.cut(df_raw["tenure"], [-1, 12, 24, 36, 48, 60, 72])
    tr = df_raw.groupby(tb, observed=True)["Churn"].mean()
    ax[1].bar([str(i) for i in tr.index], tr.values, color="#d1495b"); ax[1].set_title("Churn rate by tenure bin (months)")
    ax[1].tick_params(axis="x", rotation=20)
    log.save_fig(fig, os.path.join(d, "churn_rate_by_group.png"), "Tỷ lệ churn theo Contract và tenure")

    # 7) Heatmap tương quan
    tmp = df_raw[num + ["SeniorCitizen", "Churn"]].copy()
    fig, ax = plt.subplots(figsize=(6.2, 5))
    sns.heatmap(tmp.corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Spearman correlation (numeric + target)")
    log.save_fig(fig, os.path.join(d, "corr_heatmap.png"), "Ma trận tương quan Spearman")

    # 8) Bảng thống kê mô tả
    desc = df_raw[num].groupby(df_raw["Churn"]).describe().T
    log.save_csv(os.path.join(d, "numeric_describe_by_churn.csv"), desc, "Thống kê mô tả thuộc tính số theo nhãn", index=True)


# =============================== ĐÁNH GIÁ MÔ HÌNH ===============================
def learning_curves(h, name, out_dir, log):
    if not h or not h["x"]:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["x"], h["train_loss"], label="train loss")
    if h["val_loss"]:
        ax[0].plot(h["x"], h["val_loss"], label="val loss")
        b = int(np.argmin(h["val_loss"]))
        ax[0].axvline(h["x"][b], color="gray", ls=":", label=f"best val @ {h['x'][b]}")
    ax[0].set_xlabel(h["x_label"]); ax[0].set_ylabel("log-loss"); ax[0].set_title(f"{name}: loss"); ax[0].legend()
    ax[1].plot(h["x"], h["train_acc"], label="train acc")
    if h["val_acc"]:
        ax[1].plot(h["x"], h["val_acc"], label="val acc")
    ax[1].set_xlabel(h["x_label"]); ax[1].set_ylabel("accuracy"); ax[1].set_title(f"{name}: accuracy"); ax[1].legend()
    p = os.path.join(out_dir, "curve_loss_acc_per_epoch.png")
    log.save_fig(fig, p, "Loss/Accuracy theo epoch (hoặc vòng boosting/số cây)", name)
    return p


def confusion_matrix_plot(cm, name, split, out_dir, log):
    cm = np.array(cm)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, (mat, fmt, title) in zip(ax, [(cm, "d", "counts"), (cm / cm.sum(axis=1, keepdims=True), ".2%", "row-normalized")]):
        sns.heatmap(mat, annot=True, fmt=fmt, cmap="Blues", cbar=False, ax=a,
                    xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
        a.set_xlabel("Predicted"); a.set_ylabel("Actual"); a.set_title(f"{name} - {split} ({title})")
    log.save_fig(fig, os.path.join(out_dir, f"confusion_matrix_{split}.png"), f"Ma trận nhầm lẫn tập {split}", name)


def roc_pr_plot(data, name, out_dir, log):
    """data: dict split -> (y, p)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for split, (y, p) in data.items():
        if len(y) == 0:
            continue
        f, t, _ = roc_curve(y, p)
        from sklearn.metrics import auc, average_precision_score
        ax[0].plot(f, t, label=f"{split} AUC={auc(f, t):.3f}")
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, label=f"{split} AP={average_precision_score(y, p):.3f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title(f"{name}: ROC"); ax[0].legend()
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title(f"{name}: Precision-Recall"); ax[1].legend()
    log.save_fig(fig, os.path.join(out_dir, "roc_pr_curves.png"), "Đường cong ROC và Precision-Recall (train/val/test)", name)


def calibration_gain_lift(y, p, name, out_dir, log):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    fr, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax[0].plot(mp, fr, "o-", label=name); ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_xlabel("Mean predicted prob")
    ax[0].set_ylabel("Fraction of churners"); ax[0].set_title("Calibration (test)"); ax[0].legend()
    order = np.argsort(-p); ys = np.asarray(y)[order]
    cum = np.cumsum(ys) / ys.sum(); x = np.arange(1, len(ys) + 1) / len(ys)
    ax[1].plot(x, cum, label=name); ax[1].plot([0, 1], [0, 1], "k--", label="random")
    ax[1].set_xlabel("Fraction of customers contacted"); ax[1].set_ylabel("Fraction of churners captured")
    ax[1].set_title("Cumulative gains (test)"); ax[1].legend()
    dec = np.array_split(ys, 10)
    lifts = [d.mean() / ys.mean() for d in dec]
    ax[2].bar(range(1, 11), lifts, color="#d1495b"); ax[2].axhline(1, color="k", ls="--")
    ax[2].set_xlabel("Decile (1 = highest risk)"); ax[2].set_ylabel("Lift"); ax[2].set_title("Lift by decile (test)")
    log.save_fig(fig, os.path.join(out_dir, "calibration_gain_lift_test.png"), "Hiệu chuẩn, Gains và Lift theo decile (test)", name)


def threshold_curve(y, p, thr_used, name, out_dir, log):
    from sklearn.metrics import f1_score, precision_score, recall_score
    grid = np.linspace(0.05, 0.95, 91)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(grid, [precision_score(y, p >= t, zero_division=0) for t in grid], label="precision")
    ax.plot(grid, [recall_score(y, p >= t, zero_division=0) for t in grid], label="recall")
    ax.plot(grid, [f1_score(y, p >= t, zero_division=0) for t in grid], label="F1")
    ax.axvline(thr_used, color="k", ls=":", label=f"threshold={thr_used:.2f}")
    ax.set_xlabel("decision threshold"); ax.set_title(f"{name}: metrics vs threshold (test)"); ax.legend()
    log.save_fig(fig, os.path.join(out_dir, "threshold_analysis_test.png"), "Precision/Recall/F1 theo ngưỡng (test)", name)


def prob_histogram(y, p, name, out_dir, log, bins=30):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    y = np.asarray(y)
    for k in (0, 1):
        ax.hist(p[y == k], bins=bins, alpha=0.55, color=PALETTE[k], label=LABEL[k], density=True)
    ax.set_xlabel("Predicted churn probability"); ax.set_ylabel("density"); ax.set_title(f"{name}: histogram of predicted probabilities (test)")
    ax.legend()
    log.save_fig(fig, os.path.join(out_dir, "hist_pred_prob_test.png"), "Histogram xác suất dự báo theo nhãn thật (test)", name)


# =============================== SO SÁNH NHIỀU MÔ HÌNH ===============================
def compare_models(summary_df, curves, out_dir, log):
    """summary_df: hàng = mô hình, cột = độ đo (test)."""
    keys = ["accuracy", "f1", "recall_sensitivity_tpr", "precision", "cohen_kappa", "mcc", "roc_auc", "ap_churn(PR-AUC)"]
    keys = [k for k in keys if k in summary_df.columns]
    fig, ax = plt.subplots(figsize=(13, 5))
    summary_df[keys].plot(kind="bar", ax=ax, width=0.82)
    ax.set_title("Model comparison on TEST set"); ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15); ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    log.save_fig(fig, os.path.join(out_dir, "compare_models_metrics_test.png"), "So sánh các độ đo giữa các mô hình (test)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, (y, p) in curves.items():
        f, t, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
        ax[0].plot(f, t, label=name); ax[1].plot(rc, pr, label=name)
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_title("ROC (test)"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend(fontsize=8)
    ax[1].set_title("Precision-Recall (test)"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend(fontsize=8)
    log.save_fig(fig, os.path.join(out_dir, "compare_models_roc_pr_test.png"), "ROC và PR chồng nhiều mô hình (test)")
