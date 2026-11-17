#!/usr/bin/env python3
"""Giao diện dòng lệnh chạy thực nghiệm Dự báo Churn + XAI (Telco Customer Churn).

Tham khảo argparse: https://www.geeksforgeeks.org/python/command-line-arguments-in-python/

Ví dụ:
  python run_experiment.py --model-name xgboost --run-xai --run-eda --tune-threshold
  python run_experiment.py --model-name all --imbalance class_weight --cv-folds 5 --run-xai --exp-name EXP_BASELINE
  python run_experiment.py --model-name xgboost --n-estimators 300 --max-depth 5 --learning-rate 0.03 --imbalance smote
"""
import argparse
import datetime
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

from src import data as D
from src import metrics as M
from src import models as MD
from src import plots as P
from src import xai as X
from src.logger import ExperimentLogger, banner, env_info, fmt_block, to_jsonable


# ============================ 1. GIAO DIỆN DÒNG LỆNH ============================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Dự báo Churn & Giải thích mô hình (XAI) - chạy thực nghiệm từ dòng lệnh",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("1. Dữ liệu")
    g.add_argument("--data-path", default="data/Telco-Customer-Churn.csv", help="Đường dẫn file CSV")
    g.add_argument("--target", default="Churn", help="Tên cột nhãn")
    g.add_argument("--id-col", default="customerID", help="Tên cột ID (bị loại khỏi đặc trưng)")
    g.add_argument("--test-size", type=float, default=0.2, help="Tỷ lệ tập Test (so với toàn bộ)")
    g.add_argument("--val-size", type=float, default=0.1, help="Tỷ lệ tập Validation (so với toàn bộ); 0 = không dùng")
    g.add_argument("--random-state", type=int, default=42, help="Seed để tái lập kết quả")
    g.add_argument("--impute", choices=["zero", "median"], default="zero", help="Cách điền TotalCharges bị trống")
    g.add_argument("--imbalance", choices=["none", "class_weight", "smote"], default="class_weight",
                   help="Xử lý mất cân bằng lớp (chỉ áp dụng trên tập train)")

    g = p.add_argument_group("2. Mô hình & siêu tham số")
    g.add_argument("--model-name", default="xgboost", choices=MD.MODEL_CHOICES + ["all"], help="Thuật toán; 'all' = chạy tất cả")
    g.add_argument("--n-estimators", type=int, default=200, help="Số cây (rforest/xgboost)")
    g.add_argument("--max-depth", type=int, default=4, help="Độ sâu tối đa (dtree/rforest/xgboost)")
    g.add_argument("--min-samples-leaf", type=int, default=5, help="Số mẫu tối thiểu mỗi lá (dtree/rforest)")
    g.add_argument("--learning-rate", type=float, default=0.05, help="Tốc độ học (chỉ xgboost)")
    g.add_argument("--mlp-lr", type=float, default=0.001, help="Tốc độ học Adam của MLP (learning_rate_init)")
    g.add_argument("--scale-pos-weight", type=float, default=None, help="XGBoost; mặc định tự tính N_neg/N_pos của tập train")
    g.add_argument("--subsample", type=float, default=0.8, help="XGBoost: tỷ lệ mẫu mỗi cây")
    g.add_argument("--colsample-bytree", type=float, default=0.8, help="XGBoost: tỷ lệ cột mỗi cây")
    g.add_argument("--reg-lambda", type=float, default=1.0, help="Điều hoà L2 (xgboost, mlp alpha = lambda*1e-3)")
    g.add_argument("--C", type=float, default=1.0, help="Nghịch đảo độ điều hoà (logreg, svm)")
    g.add_argument("--k-neighbors", type=int, default=15, help="k của k-NN")
    g.add_argument("--hidden-sizes", default="64,32", help="MLP: kích thước các lớp ẩn, vd 64,32")
    g.add_argument("--epochs", type=int, default=100, help="MLP: số epoch; logreg: số vòng lặp tối đa")
    g.add_argument("--batch-size", type=int, default=128, help="MLP: batch size")

    g = p.add_argument_group("3. Đánh giá")
    g.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng phân loại")
    g.add_argument("--tune-threshold", action="store_true", help="Chọn ngưỡng tối đa F1 trên tập Validation")
    g.add_argument("--cv-folds", type=int, default=5, help="Stratified K-Fold CV trên tập train (0 = tắt)")
    g.add_argument("--bins", type=int, default=30, help="Số bin của histogram")

    g = p.add_argument_group("4. Giải thích mô hình (XAI)")
    g.add_argument("--run-xai", action="store_true", help="Tính SHAP + độ đo Fidelity/Stability/Sparsity")
    g.add_argument("--xai-index", type=int, default=0, help="Chỉ số khách hàng test để vẽ waterfall")
    g.add_argument("--xai-sample", type=int, default=300, help="Số mẫu đo Stability")
    g.add_argument("--xai-noise", type=float, default=0.05, help="Độ lệch chuẩn nhiễu Gaussian khi đo Stability")

    g = p.add_argument_group("5. Thực thi & đầu ra")
    g.add_argument("--output-dir", default="outputs", help="Thư mục gốc lưu kết quả")
    g.add_argument("--exp-name", default=None, help="Tên thực nghiệm (mặc định: EXP_<thời gian>)")
    g.add_argument("--run-eda", action="store_true", help="Vẽ histogram/box/heatmap phân tích dữ liệu")
    g.add_argument("--save-model", action="store_true", help="Lưu mô hình (.joblib)")
    g.add_argument("--n-jobs", type=int, default=-1, help="Số luồng CPU")
    return p.parse_args(argv)


# ============================ 2. TIỆN ÍCH ============================
def predict_proba(model, X_):
    return model.predict_proba(X_)[:, 1] if len(X_) else np.array([])


def cv_evaluate(name, args, X_raw, y_raw, thr):
    """Stratified K-Fold trên tập train: mỗi fold fit lại bộ tiền xử lý (không rò rỉ)."""
    skf = StratifiedKFold(args.cv_folds, shuffle=True, random_state=args.random_state)
    rows = []
    for k, (a, b) in enumerate(skf.split(X_raw, y_raw), 1):
        ct, _ = D.build_preprocessor(X_raw.iloc[a])
        ct.fit(X_raw.iloc[a])
        Xa, Xb = D.transform(ct, X_raw.iloc[a]), D.transform(ct, X_raw.iloc[b])
        ya, yb = y_raw.iloc[a], y_raw.iloc[b]
        if args.imbalance == "smote":
            Xa, ya = SMOTE(random_state=args.random_state).fit_resample(Xa, ya)
        est = MD.build_estimator(name, args, ya)
        if name == "mlp":
            est.set_params(max_iter=args.epochs)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(Xa, ya)
        m = M.compute_all(yb, predict_proba(est, Xb), thr)
        rows.append({"fold": k, **{c: m[c] for c in M.SCALAR_KEYS}})
    df = pd.DataFrame(rows)
    stats = {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=1))} for c in M.SCALAR_KEYS}
    return df, stats


def metrics_table_text(per_split):
    keys = [k for k in M.SCALAR_KEYS if k in next(iter(per_split.values()))] + ["tn", "fp", "fn", "tp"]
    splits = list(per_split)
    lines = [f"{'metric':<28}" + "".join(f"{s:>12}" for s in splits)]
    for k in keys:
        lines.append(f"{k:<28}" + "".join(f"{per_split[s][k]:>12.4f}" if isinstance(per_split[s][k], float)
                                         else f"{per_split[s][k]:>12}" for s in splits))
    return "\n".join(lines)


# ============================ 3. CHẠY MỘT MÔ HÌNH ============================
def run_one(name, args, ctx, log):
    t_all = time.time()
    mdir = os.path.join(ctx["root"], name)
    os.makedirs(mdir, exist_ok=True)
    full = MD.MODEL_FULLNAME[name]
    Xtr, ytr, Xva, yva, Xte, yte = ctx["Xtr"], ctx["ytr"], ctx["Xva"], ctx["yva"], ctx["Xte"], ctx["yte"]
    Xfit, yfit = (ctx["Xtr_smote"], ctx["ytr_smote"]) if args.imbalance == "smote" else (Xtr, ytr)

    model = MD.build_estimator(name, args, yfit)
    model, hist, t_train = MD.fit_with_history(name, model, Xfit, yfit, Xva, yva, args)
    if name == "logreg":
        model.set_params(max_iter=args.epochs)
    t0 = time.time()
    p_tr, p_va, p_te = predict_proba(model, Xtr), predict_proba(model, Xva), predict_proba(model, Xte)
    t_infer = (time.time() - t0) / max(len(Xte), 1)

    thr = args.threshold
    thr_src = "CLI --threshold"
    if args.tune_threshold:
        thr = M.best_f1_threshold(yva if len(yva) else ytr, p_va if len(yva) else p_tr)
        thr_src = "tuned: max F1 on validation set"
    per_split = {"train": M.compute_all(ytr, p_tr, thr), "test": M.compute_all(yte, p_te, thr)}
    if len(yva):
        per_split = {"train": per_split["train"], "val": M.compute_all(yva, p_va, thr), "test": per_split["test"]}

    # --- Biểu đồ ---
    P.learning_curves(hist, full, mdir, log)
    for s in per_split:
        P.confusion_matrix_plot(per_split[s]["confusion_matrix"], full, s, mdir, log)
    rp = {"train": (ytr.values, p_tr), "test": (yte.values, p_te)}
    if len(yva):
        rp["val"] = (yva.values, p_va)
    P.roc_pr_plot(rp, full, mdir, log)
    P.calibration_gain_lift(yte.values, p_te, full, mdir, log)
    P.threshold_curve(yte.values, p_te, thr, full, mdir, log)
    P.prob_histogram(yte.values, p_te, full, mdir, log, args.bins)

    # --- CV ---
    cv_stats = None
    if args.cv_folds and args.cv_folds > 1:
        cv_df, cv_stats = cv_evaluate(name, args, ctx["X_tr_raw"], ytr, thr)
        log.save_csv(os.path.join(mdir, "cv_folds.csv"), cv_df, f"Kết quả {args.cv_folds}-fold CV từng fold", full)

    # --- XAI ---
    xai_res = None
    if args.run_xai:
        if name in X.XAI_MODELS:
            xai_res = X.run_xai(name, model, Xtr, Xte, yte, ctx["id_te"], p_te, mdir, log, args)
        else:
            xai_res = {"skipped": f"SHAP chưa cấu hình cho {full} (chỉ {X.XAI_MODELS})"}

    # --- Dự đoán + mô hình ---
    pred = pd.concat([
        pd.DataFrame({"customerID": ctx["id_tr"].values, "split": "train", "y_true": ytr.values, "prob": p_tr}),
        pd.DataFrame({"customerID": ctx["id_va"].values, "split": "val", "y_true": yva.values, "prob": p_va}),
        pd.DataFrame({"customerID": ctx["id_te"].values, "split": "test", "y_true": yte.values, "prob": p_te})])
    pred["y_pred"] = (pred["prob"] >= thr).astype(int)
    log.save_csv(os.path.join(mdir, "predictions.csv"), pred, "Xác suất & nhãn dự báo từng khách hàng (train/val/test)", full)
    if hist:
        log.save_csv(os.path.join(mdir, "history_per_epoch.csv"),
                     pd.DataFrame({"x": hist["x"], "train_loss": hist["train_loss"], "train_acc": hist["train_acc"],
                                   **({"val_loss": hist["val_loss"], "val_acc": hist["val_acc"]} if hist["val_loss"] else {})}),
                     f"Loss/Acc theo {hist['x_label']}", full)
    if args.save_model:
        joblib.dump({"model": model, "preprocessor": ctx["ct"], "threshold": thr}, os.path.join(mdir, "model.joblib"))
        log.add_artifact(os.path.join(mdir, "model.joblib"), "Mô hình + bộ tiền xử lý + ngưỡng", full, depth=1)

    # --- Gói kết quả ---
    arch = MD.describe_architecture(name, model, Xtr.shape[1])
    hyper = to_jsonable(model.get_params())
    res = {"model": name, "model_fullname": full, "architecture": arch, "hyperparameters": hyper,
           "threshold": {"value": thr, "source": thr_src},
           "timing": {"train_seconds": t_train, "inference_seconds_per_sample": t_infer, "total_seconds": time.time() - t_all},
           "metrics": per_split, "cross_validation": cv_stats, "xai": xai_res,
           "history_summary": ({"x_label": hist["x_label"], "n_points": len(hist["x"]),
                                "best_val_loss": (min(hist["val_loss"]) if hist["val_loss"] else None),
                                "best_val_at": (hist["x"][int(np.argmin(hist["val_loss"]))] if hist["val_loss"] else None),
                                "final_train_loss": hist["train_loss"][-1]} if hist else None)}

    # --- File log .txt đầy đủ cho mô hình này ---
    txt = [banner(f"MODEL LOG: {full}  |  {ctx['exp_id']}"),
           f"Lệnh chạy      : {log.command}", f"Thời điểm      : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
           banner("1. THÔNG TIN DỮ LIỆU", "-"), fmt_block("", ctx["dataset_info"]).replace("feature_names_after_encoding", "feature_names"),
           banner("2. KIẾN TRÚC MÔ HÌNH", "-"), fmt_block("", arch),
           banner("3. SIÊU THAM SỐ (estimator.get_params)", "-"), fmt_block("", hyper),
           banner("4. THAM SỐ DÒNG LỆNH (toàn bộ)", "-"), fmt_block("", vars(args)),
           banner(f"5. NGƯỠNG & THỜI GIAN", "-"), fmt_block("", {"threshold": thr, "source": thr_src, **res["timing"]}),
           banner("6. KẾT QUẢ TRAIN / VAL / TEST", "-"), metrics_table_text(per_split),
           banner("7. MA TRẬN NHẦM LẪN [[TN, FP],[FN, TP]]", "-")]
    for s in per_split:
        txt.append(f"{s}: {per_split[s]['confusion_matrix']}")
    if cv_stats:
        txt += [banner(f"8. {args.cv_folds}-FOLD CROSS-VALIDATION (trên tập train, mean ± std)", "-")]
        txt += [f"{k:<28}{v['mean']:.4f} ± {v['std']:.4f}" for k, v in cv_stats.items()]
    if hist:
        txt += [banner("9. TÓM TẮT QUÁ TRÌNH HUẤN LUYỆN", "-"), fmt_block("", res["history_summary"])]
    if xai_res:
        txt += [banner("10. ĐỘ ĐO XAI", "-"), fmt_block("", xai_res)]
    txt += [banner("11. ĐỘ ĐO KHÔNG ÁP DỤNG CHO BÀI TOÁN", "-"), fmt_block("", M.NOT_APPLICABLE)]
    log.write_text(os.path.join(mdir, "model_log.txt"), "\n".join(txt))
    log.add_artifact(os.path.join(mdir, "model_log.txt"), "Log đầy đủ: data, kiến trúc, siêu tham số, kết quả", full, depth=1)
    log.save_json(os.path.join(mdir, "results.json"), res, "Kết quả dạng JSON (máy đọc được)", full)

    print(f"[{full:<20}] test: acc={per_split['test']['accuracy']:.4f} f1={per_split['test']['f1']:.4f} "
          f"auc={per_split['test']['roc_auc']:.4f} mcc={per_split['test']['mcc']:.4f} (thr={thr:.2f}, {t_train:.1f}s)")
    return res, (yte.values, p_te)


# ============================ 4. MAIN ============================
def main():
    args = parse_args()
    np.random.seed(args.random_state)
    command = "python " + " ".join(sys.argv)
    exp_id = args.exp_name or f"EXP_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    root = os.path.join(args.output_dir, exp_id)
    log = ExperimentLogger(root, exp_id, command)
    print(f"[CLI] exp={exp_id} model={args.model_name} data={args.data_path} imbalance={args.imbalance} -> {root}")

    # ---- dữ liệu ----
    df, Xraw, y, ids, n_blank = D.load_raw(args.data_path, args.target, args.id_col, args.impute)
    (X_tr, y_tr, id_tr), (X_va, y_va, id_va), (X_te, y_te, id_te) = D.split_data(Xraw, y, ids, args.test_size, args.val_size, args.random_state)
    ct, groups = D.build_preprocessor(X_tr)
    ct.fit(X_tr)
    Xtr, Xte = D.transform(ct, X_tr), D.transform(ct, X_te)
    Xva = D.transform(ct, X_va) if len(X_va) else pd.DataFrame(columns=Xtr.columns)
    info = D.dataset_info(args.data_path, df, Xraw, ((X_tr, y_tr, id_tr), (X_va, y_va, id_va), (X_te, y_te, id_te)),
                          groups, n_blank, Xtr.shape[1], Xtr.columns, args)
    ctx = dict(root=root, exp_id=exp_id, ct=ct, Xtr=Xtr, Xva=Xva, Xte=Xte, ytr=y_tr, yva=y_va, yte=y_te,
               id_tr=id_tr, id_va=id_va, id_te=id_te, X_tr_raw=X_tr, dataset_info=info)
    if args.imbalance == "smote":
        ctx["Xtr_smote"], ctx["ytr_smote"] = SMOTE(random_state=args.random_state).fit_resample(Xtr, y_tr)
        info["smote_train_after"] = {"n": int(len(ctx["ytr_smote"])), "churned(1)": int(ctx["ytr_smote"].sum())}
    log.save_json(os.path.join(root, "dataset_info.json"), info, "Thông tin dữ liệu, chia tập, đặc trưng")
    if args.run_eda:
        P.eda_all(df, root, log, args.bins)

    # ---- chạy mô hình ----
    names = MD.MODEL_CHOICES if args.model_name == "all" else [args.model_name]
    results, curves = {}, {}
    for n in names:
        results[n], curves[n] = run_one(n, args, ctx, log)

    # ---- tổng hợp ----
    sdir = os.path.join(root, "summary")
    rows = []
    for n, r in results.items():
        te, tr = r["metrics"]["test"], r["metrics"]["train"]
        row = {"model": MD.MODEL_FULLNAME[n], "threshold": r["threshold"]["value"]}
        row.update({k: te[k] for k in M.SCALAR_KEYS})
        row.update({"train_accuracy": tr["accuracy"], "train_f1": tr["f1"], "train_roc_auc": tr["roc_auc"],
                    "gap_f1(train-test)": tr["f1"] - te["f1"], "gap_auc(train-test)": tr["roc_auc"] - te["roc_auc"],
                    "train_seconds": r["timing"]["train_seconds"]})
        if r["cross_validation"]:
            for k in ("accuracy", "f1", "roc_auc", "mcc"):
                row[f"cv_{k}_mean"] = r["cross_validation"][k]["mean"]; row[f"cv_{k}_std"] = r["cross_validation"][k]["std"]
        if r["xai"] and "fidelity_score(1-MAE)" in r["xai"]:
            row["xai_fidelity"] = r["xai"]["fidelity_score(1-MAE)"]; row["xai_stability_cos"] = r["xai"]["stability_cosine_similarity_mean"]
            row["xai_sparsity_k80"] = r["xai"]["sparsity_n_features_for_80pct_importance"]
        rows.append(row)
    summ = pd.DataFrame(rows)
    log.save_csv(os.path.join(sdir, "results_summary_test.csv"), summ, "Bảng tổng hợp độ đo tập TEST + CV + XAI của các mô hình")
    if len(results) > 1:
        P.compare_models(summ.set_index("model"), {MD.MODEL_FULLNAME[n]: c for n, c in curves.items()}, sdir, log)

    elapsed = (datetime.datetime.now() - log.start).total_seconds()
    head = [banner(f"EXPERIMENT LOG - {exp_id}"), f"Lệnh chạy : {command}", f"Bắt đầu   : {log.start:%Y-%m-%d %H:%M:%S}",
            f"Thời gian : {elapsed:.1f}s", banner("MÔI TRƯỜNG", "-"), fmt_block("", env_info()),
            banner("THAM SỐ DÒNG LỆNH", "-"), fmt_block("", vars(args)),
            banner("DỮ LIỆU", "-"), fmt_block("", {k: v for k, v in info.items() if k != "feature_names_after_encoding"}),
            banner("BẢNG TỔNG HỢP (TEST)", "-"), summ.round(4).to_string(index=False),
            banner("ĐỘ ĐO KHÔNG ÁP DỤNG", "-"), fmt_block("", M.NOT_APPLICABLE),
            banner("THƯ MỤC KẾT QUẢ", "-"), "Xem artifact_index.csv: mỗi hình/bảng ↔ file code:hàm:dòng sinh ra nó."]
    log.write_text(os.path.join(root, "experiment_log.txt"), "\n".join(head))
    log.add_artifact(os.path.join(root, "experiment_log.txt"), "Log tổng của lần chạy")
    log.save_json(os.path.join(root, "all_results.json"), {"args": vars(args), "env": env_info(), "dataset": info, "results": results},
                  "Toàn bộ kết quả dạng JSON")
    idx = log.write_artifact_index()
    log.append_global_registry(args.output_dir, f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | {exp_id} | {args.model_name} | {command}")
    print(f"[DONE] {len(log.artifacts)} artifacts -> {idx}")


if __name__ == "__main__":
    main()
