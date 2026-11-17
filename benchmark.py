#!/usr/bin/env python3
"""Đối sánh kết quả thực nghiệm với các nghiên cứu liên quan.

  python benchmark.py --results outputs/EXP_BASELINE_ALL/summary/results_summary_test.csv \
                      --related data/related_work.csv --out outputs/BENCHMARK
Các công bố được lấy từ data/related_work.csv (mỗi dòng có source_url + verified_from). Cột `comparable_with_ours`
cho biết số liệu có cùng dữ liệu/giao thức chia tập hay không; nếu 'unverified' thì chỉ mang tính tham khảo.
"""
import argparse
import datetime
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.logger import ExperimentLogger, banner, sha256_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results_summary_test.csv của lần chạy thực nghiệm")
    ap.add_argument("--related", default="data/related_work.csv", help="CSV các nghiên cứu liên quan")
    ap.add_argument("--out", default="outputs/BENCHMARK", help="Thư mục lưu kết quả đối sánh")
    ap.add_argument("--ours-label", default="Nghiên cứu này", help="Nhãn nhóm kết quả của nhóm")
    a = ap.parse_args()
    log = ExperimentLogger(a.out, "BENCHMARK", "python " + " ".join(sys.argv))

    ours = pd.read_csv(a.results)
    rel = pd.read_csv(a.related)
    rows = []
    for _, r in ours.iterrows():
        rows.append({"study": f"{a.ours_label} - {r['model']}", "year": 2026, "method": r["model"], "xai": "TreeSHAP/LinearSHAP" if r["model"] in
                     ("XGBoost", "Random Forest", "Decision Tree", "Logistic Regression") else "-",
                     "dataset": "Telco IBM 7.043 mẫu, test 20% stratified", "comparable": "yes (cùng giao thức)",
                     "accuracy": r["accuracy"], "precision": r["precision"], "recall": r["recall_sensitivity_tpr"],
                     "f1": r["f1"], "roc_auc": r["roc_auc"], "source": a.results})
    for _, r in rel.iterrows():
        rows.append({"study": r["reference"], "year": r["year"], "method": r["method"], "xai": r["xai"], "dataset": r["dataset"],
                     "comparable": r["comparable_with_ours"], "accuracy": r["accuracy"], "precision": r["precision"],
                     "recall": r["recall"], "f1": r["f1"], "roc_auc": r["roc_auc"], "source": r["source_url"]})
    tab = pd.DataFrame(rows)
    log.save_csv(os.path.join(a.out, "benchmark_table.csv"), tab, "Bảng đối sánh với nghiên cứu liên quan")

    num = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    show = tab.drop(columns=["source"]).copy()
    for c in num:
        show[c] = show[c].map(lambda v: "N/A" if pd.isna(v) else f"{v:.4f}")
    md = ["| " + " | ".join(show.columns) + " |", "|" + "---|" * len(show.columns)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in show.values]
    log.write_text(os.path.join(a.out, "benchmark_table.md"), "\n".join(md))
    log.add_artifact(os.path.join(a.out, "benchmark_table.md"), "Bảng đối sánh dạng Markdown")

    # biểu đồ: Accuracy & F1 (chỉ hiển thị giá trị có sẵn)
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(tab) + 2.5))
    y = np.arange(len(tab))
    for k, (c, col) in enumerate([("accuracy", "#2b7bba"), ("f1", "#d1495b")]):
        v = tab[c].astype(float).values
        ax.barh(y + (k - 0.5) * 0.38, np.nan_to_num(v), 0.38, color=col, label=c)
        for yi, vi in zip(y, v):
            if not np.isnan(vi):
                ax.text(vi + 0.005, yi + (k - 0.5) * 0.38, f"{vi:.3f}", va="center", fontsize=7)
    labels = [s if len(s) < 48 else s[:46] + "…" for s in tab["study"]]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1.05)
    ax.set_title("Benchmark: Accuracy & F1 (trống = bài gốc không công bố; 'unverified' = chưa xác nhận cùng dataset)")
    ax.legend(loc="lower right")
    log.save_fig(fig, os.path.join(a.out, "benchmark_chart.png"), "Biểu đồ đối sánh Accuracy/F1 với nghiên cứu liên quan")

    log.write_text(os.path.join(a.out, "benchmark_log.txt"),
                   banner("BENCHMARK LOG") + f"Lệnh: {log.command}\nThời điểm: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
                   f"results : {a.results}  sha256={sha256_of(a.results)}\nrelated : {a.related}  sha256={sha256_of(a.related)}\n"
                   + banner("BẢNG ĐỐI SÁNH", "-") + tab.drop(columns=["source"]).to_string(index=False)
                   + "\n\nLƯU Ý: các dòng 'unverified' chỉ là tham khảo; phải mở bài gốc xác nhận dataset, cách chia tập, ngưỡng trước khi kết luận hơn/kém.")
    log.add_artifact(os.path.join(a.out, "benchmark_log.txt"), "Log đối sánh")
    log.write_artifact_index()
    print(tab.drop(columns=["source", "dataset", "xai"]).to_string(index=False))


if __name__ == "__main__":
    main()
