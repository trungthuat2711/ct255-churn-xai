"""Giải thích mô hình bằng SHAP + các độ đo chất lượng giải thích (Fidelity, Stability, Sparsity, Consistency)."""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)
XAI_MODELS = ["xgboost", "rforest", "dtree", "logreg"]


def _explainer(name, model, X_bg):
    if name == "logreg":
        return shap.LinearExplainer(model, X_bg)
    return shap.TreeExplainer(model)


def _values(explainer, X):
    ex = explainer(X)
    v, base = ex.values, ex.base_values
    if v.ndim == 3:                       # RF/DT: (n, features, classes) -> lớp Churn
        v, base = v[:, :, 1], base[:, 1]
    return np.asarray(v), np.asarray(base).reshape(-1), ex


def _model_output(name, model, X):
    """Đầu ra cùng thang đo với SHAP: log-odds (xgboost, logreg) hoặc xác suất (rforest, dtree)."""
    if name == "xgboost":
        return model.predict(X, output_margin=True)
    if name == "logreg":
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def run_xai(name, model, Xtr, Xte, yte, ids_te, prob_te, out_dir, log, args):
    d = os.path.join(out_dir, "xai")
    os.makedirs(d, exist_ok=True)
    rng = np.random.RandomState(args.random_state)
    expl = _explainer(name, model, Xtr)
    vals, base, ex = _values(expl, Xte)
    feats = list(Xte.columns)
    res = {"explainer": type(expl).__name__, "output_scale": "log-odds" if name in ("xgboost", "logreg") else "probability"}

    # ---- Fidelity: base + sum(phi) so với đầu ra mô hình (tính cộng tính của SHAP) ----
    recon = base + vals.sum(axis=1)
    out = _model_output(name, model, Xte)
    err = np.abs(recon - out)
    res["fidelity_additivity_mae"] = float(err.mean())
    res["fidelity_additivity_max_err"] = float(err.max())
    res["fidelity_score(1-MAE)"] = float(1 - err.mean())
    res["fidelity_corr_with_model_output"] = float(np.corrcoef(recon, out)[0, 1])

    # ---- Global importance + Sparsity ----
    imp = np.abs(vals).mean(axis=0)
    imp_df = pd.DataFrame({"feature": feats, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False)
    imp_df["share"] = imp_df["mean_abs_shap"] / imp_df["mean_abs_shap"].sum()
    imp_df["cum_share"] = imp_df["share"].cumsum()
    res["sparsity_n_features_for_80pct_importance"] = int((imp_df["cum_share"] < 0.8).sum() + 1)
    res["top5_features"] = imp_df["feature"].head(5).tolist()
    log.save_csv(os.path.join(d, "shap_global_importance.csv"), imp_df, "Tầm quan trọng SHAP toàn cục (mean |SHAP|)", name)

    # ---- Consistency: SHAP importance vs importance nội tại của mô hình ----
    native = None
    if hasattr(model, "feature_importances_"):
        native = model.feature_importances_
    elif hasattr(model, "coef_"):
        native = np.abs(model.coef_.ravel())
    if native is not None:
        res["spearman_shap_vs_native_importance"] = float(spearmanr(imp, native)[0])

    # ---- Stability: nhiễu Gaussian lên các cột số đã chuẩn hoá ----
    n = min(args.xai_sample, len(Xte))
    idx = rng.choice(len(Xte), n, replace=False)
    Xs = Xte.iloc[idx].copy()
    num_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in Xs.columns]
    eps = np.zeros(Xs.shape)
    for c in num_cols:
        eps[:, Xs.columns.get_loc(c)] = rng.normal(0, args.xai_noise, n)
    Xn = Xs + eps
    v0 = vals[idx]
    v1, _, _ = _values(expl, Xn)
    dphi = np.linalg.norm(v0 - v1, axis=1)
    de = np.linalg.norm(eps, axis=1)
    res["stability_index_mean(||dphi||/||eps||)"] = float(np.mean(dphi / de))
    res["stability_median"] = float(np.median(dphi / de))
    cos = np.sum(v0 * v1, axis=1) / (np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1) + 1e-12)
    res["stability_cosine_similarity_mean"] = float(cos.mean())
    res["stability_noise_sigma"] = args.xai_noise
    res["stability_n_samples"] = int(n)

    # ---- Biểu đồ ----
    plt.figure(figsize=(9, 6))
    shap.summary_plot(vals, Xte, feature_names=feats, show=False, max_display=15)
    plt.title(f"SHAP summary (beeswarm) - {name}")
    log.save_fig(plt.gcf(), os.path.join(d, "shap_summary_beeswarm.png"), "SHAP Summary (beeswarm) toàn cục - tập test", name)

    plt.figure(figsize=(8, 5.5))
    shap.summary_plot(vals, Xte, feature_names=feats, plot_type="bar", show=False, max_display=15)
    plt.title(f"SHAP feature importance - {name}")
    log.save_fig(plt.gcf(), os.path.join(d, "shap_importance_bar.png"), "SHAP Feature Importance (bar)", name)

    i = int(args.xai_index) if args.xai_index is not None else 0
    exp_i = shap.Explanation(values=vals[i], base_values=base[i], data=Xte.iloc[i].values, feature_names=feats)
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(exp_i, show=False, max_display=12)
    log.save_fig(plt.gcf(), os.path.join(d, f"shap_waterfall_customer_{i}.png"),
                 f"SHAP Waterfall khách hàng test#{i} (ID={ids_te.iloc[i]}, p={prob_te[i]:.3f}, y={int(yte.iloc[i])})", name)
    # khách rủi ro cao nhất
    j = int(np.argmax(prob_te))
    exp_j = shap.Explanation(values=vals[j], base_values=base[j], data=Xte.iloc[j].values, feature_names=feats)
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(exp_j, show=False, max_display=12)
    log.save_fig(plt.gcf(), os.path.join(d, "shap_waterfall_highest_risk.png"),
                 f"SHAP Waterfall khách rủi ro cao nhất (test#{j}, ID={ids_te.iloc[j]}, p={prob_te[j]:.3f})", name)

    for f in imp_df["feature"].head(2):
        plt.figure(figsize=(6.5, 4.5))
        shap.dependence_plot(f, vals, Xte, feature_names=feats, show=False)
        log.save_fig(plt.gcf(), os.path.join(d, f"shap_dependence_{f}.png"), f"SHAP Dependence plot: {f}", name)

    # ---- Bảng cho BI Dashboard: xác suất + Top-3 yếu tố rủi ro ----
    top3 = []
    for r in range(len(Xte)):
        order = np.argsort(-vals[r])[:3]
        top3.append("; ".join(f"{feats[k]} ({vals[r, k]:+.3f})" for k in order if vals[r, k] > 0))
    bi = pd.DataFrame({"customerID": ids_te.values, "actual_churn": yte.values, "churn_probability": prob_te})
    bi["risk_group"] = pd.cut(bi["churn_probability"], [-0.01, 0.3, 0.6, 1.01], labels=["Low", "Medium", "High"])
    bi["top3_risk_drivers"] = top3
    log.save_csv(os.path.join(d, "bi_export_test.csv"), bi, "Bảng xuất cho BI Dashboard: p(churn), nhóm rủi ro, Top-3 driver (SHAP)", name)
    rg = bi.groupby("risk_group", observed=True)["actual_churn"].agg(["count", "mean"]).rename(columns={"mean": "actual_churn_rate"})
    res["risk_group_summary"] = rg.reset_index().to_dict("records")
    return res
