"""Bộ độ đo đánh giá cho bài toán phân loại nhị phân Churn (lớp dương = 1 = rời bỏ).

Độ đo trong yêu cầu nhưng KHÔNG áp dụng cho bài toán này (ghi rõ vào log để minh bạch):
  - CMC (Cumulative Match Characteristic): dành cho nhận dạng/re-identification (xếp hạng khớp danh tính).
  - Các độ đo ẩn tin (PSNR, SSIM, BER, payload/capacity, RS-analysis...): dành cho steganography/watermarking ảnh.
Độ đo có thể quy đổi cho bài toán nhị phân:
  - IoU (Jaccard) = TP/(TP+FP+FN) cho lớp Churn; mIoU = trung bình IoU 2 lớp.
  - mAP = trung bình Average Precision của 2 lớp (AP lớp Churn chính là PR-AUC).
"""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score, fbeta_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve)

NOT_APPLICABLE = {
    "CMC": "chỉ dùng cho bài toán nhận dạng/re-identification (xếp hạng khớp danh tính), không phải phân loại nhị phân Churn",
    "PSNR/SSIM/BER/Payload": "độ đo cho ẩn tin (steganography) trên ảnh/âm thanh, không liên quan dữ liệu bảng Churn",
    "IoU/mAP (detection)": "dạng detection (bounding box) không áp dụng; dùng IoU/AP phiên bản phân loại nhị phân",
}


def _safe_div(a, b):
    return float(a / b) if b else 0.0


def expected_calibration_error(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def lift_at(y, p, frac):
    k = max(1, int(round(len(y) * frac)))
    order = np.argsort(-p)[:k]
    prec_k = y[order].mean()
    base = y.mean()
    return float(prec_k / base) if base > 0 else 0.0, float(y[order].sum() / max(y.sum(), 1))


def compute_all(y_true, y_prob, thr=0.5):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    m = {"threshold": float(thr)}
    # --- Nhóm 1: độ đo theo ngưỡng ---
    m["accuracy"] = accuracy_score(y, pred)
    m["balanced_accuracy"] = balanced_accuracy_score(y, pred)
    m["precision"] = precision_score(y, pred, zero_division=0)
    m["recall_sensitivity_tpr"] = recall_score(y, pred, zero_division=0)
    m["specificity_tnr"] = _safe_div(tn, tn + fp)
    m["npv"] = _safe_div(tn, tn + fn)
    m["fpr"] = _safe_div(fp, fp + tn)
    m["fnr"] = _safe_div(fn, fn + tp)
    m["f1"] = f1_score(y, pred, zero_division=0)
    m["f2"] = fbeta_score(y, pred, beta=2, zero_division=0)
    m["f1_macro"] = f1_score(y, pred, average="macro", zero_division=0)
    m["f1_weighted"] = f1_score(y, pred, average="weighted", zero_division=0)
    m["g_mean"] = float(np.sqrt(m["recall_sensitivity_tpr"] * m["specificity_tnr"]))
    m["cohen_kappa"] = cohen_kappa_score(y, pred)
    m["mcc"] = matthews_corrcoef(y, pred)
    # --- Nhóm 2: IoU / mAP (quy đổi cho phân loại) ---
    m["iou_churn"] = _safe_div(tp, tp + fp + fn)
    m["iou_retained"] = _safe_div(tn, tn + fp + fn)
    m["mIoU"] = (m["iou_churn"] + m["iou_retained"]) / 2
    m["ap_churn(PR-AUC)"] = average_precision_score(y, p)
    m["ap_retained"] = average_precision_score(1 - y, 1 - p)
    m["mAP"] = (m["ap_churn(PR-AUC)"] + m["ap_retained"]) / 2
    # --- Nhóm 3: xếp hạng, không phụ thuộc ngưỡng ---
    if len(np.unique(y)) == 2:
        m["roc_auc"] = roc_auc_score(y, p)
        m["gini"] = 2 * m["roc_auc"] - 1
        fpr_, tpr_, _ = roc_curve(y, p)
        m["ks_statistic"] = float(np.max(tpr_ - fpr_))
    # --- Nhóm 4: xác suất / hiệu chuẩn ---
    m["log_loss"] = log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])
    m["brier_score"] = brier_score_loss(y, p)
    m["ece_10bins"] = expected_calibration_error(y, p)
    # --- Nhóm 5: độ đo kinh doanh (chiến dịch giữ chân top-k% khách rủi ro cao) ---
    for frac in (0.1, 0.2, 0.3):
        lift, rec = lift_at(y, p, frac)
        m[f"lift_top{int(frac*100)}pct"] = lift
        m[f"recall_top{int(frac*100)}pct"] = rec
    # --- Ma trận nhầm lẫn ---
    m["tn"], m["fp"], m["fn"], m["tp"] = tn, fp, fn, tp
    m["confusion_matrix"] = [[tn, fp], [fn, tp]]
    return m


def best_f1_threshold(y_val, p_val):
    """Chọn ngưỡng tối đa F1 trên tập VALIDATION (không dùng test để chọn ngưỡng)."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_val, (p_val >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


SCALAR_KEYS = ["accuracy", "balanced_accuracy", "precision", "recall_sensitivity_tpr", "specificity_tnr", "f1", "f2",
               "g_mean", "cohen_kappa", "mcc", "iou_churn", "mIoU", "ap_churn(PR-AUC)", "mAP", "roc_auc", "gini",
               "ks_statistic", "log_loss", "brier_score", "ece_10bins", "lift_top10pct", "recall_top20pct"]
