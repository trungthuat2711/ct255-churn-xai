#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gộp kết quả nhiều seed (EXP_SEED1..N) thành bảng trung bình ± độ lệch chuẩn.

Chạy trong thư mục churn_xai:
    python aggregate_seeds.py
    python aggregate_seeds.py --pattern "outputs/EXP_SEED*" --out-dir outputs/seed_summary

Đầu ra (trong --out-dir):
    seed_summary_mean_std.csv   bảng dạng "0.7651 ± 0.0241" (dán vào báo cáo)
    seed_summary_numeric.csv    mean và std tách cột (để vẽ biểu đồ / tính toán)
    seed_summary.md             bảng Markdown
    seed_all_runs.csv           toàn bộ kết quả từng seed (minh chứng)
"""
import argparse
import glob
import os
import sys

import pandas as pd

# (tên cột trong results_summary_test.csv) -> (tên hiển thị)
METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall_sensitivity_tpr": "Recall",
    "f1": "F1",
    "cohen_kappa": "Kappa",
    "mcc": "MCC",
    "roc_auc": "ROC-AUC",
    "ap_churn(PR-AUC)": "PR-AUC",
}


def main():
    ap = argparse.ArgumentParser(description="Gộp kết quả nhiều seed: mean ± std")
    ap.add_argument("--pattern", default="outputs/EXP_SEED*", help="Mẫu tên thư mục các seed")
    ap.add_argument("--file", default="summary/results_summary_test.csv", help="File kết quả trong mỗi thư mục")
    ap.add_argument("--out-dir", default="outputs/seed_summary", help="Thư mục lưu bảng tổng hợp")
    ap.add_argument("--digits", type=int, default=4, help="Số chữ số thập phân")
    ap.add_argument("--sort-by", default="ROC-AUC", help="Độ đo dùng để sắp xếp (tên hiển thị)")
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(args.pattern) if os.path.isdir(d))
    frames = []
    for d in dirs:
        f = os.path.join(d, args.file)
        if os.path.exists(f):
            df = pd.read_csv(f)
            df.insert(0, "run", os.path.basename(d))
            frames.append(df)
        else:
            print(f"[bỏ qua] không thấy {f}")
    if not frames:
        sys.exit(f"[LỖI] Không tìm thấy kết quả với mẫu '{args.pattern}'")

    all_df = pd.concat(frames, ignore_index=True)
    n_runs = all_df["run"].nunique()

    cols = {k: v for k, v in METRICS.items() if k in all_df.columns}
    missing = [k for k in METRICS if k not in all_df.columns]
    if missing:
        print(f"[lưu ý] thiếu cột {missing}, bỏ qua")

    g = all_df.groupby("model")[list(cols)]
    mean = g.mean().rename(columns=cols)
    std = g.std(ddof=1).rename(columns=cols)  # độ lệch chuẩn mẫu

    # bảng "mean ± std"
    fmt = f"{{:.{args.digits}f}} ± {{:.{args.digits}f}}"
    text = mean.copy().astype(object)
    for c in mean.columns:
        for m in mean.index:
            text.loc[m, c] = fmt.format(mean.loc[m, c], std.loc[m, c])
    order = mean[args.sort_by].sort_values(ascending=False).index if args.sort_by in mean else mean.index
    text = text.loc[order]

    # bảng số
    numeric = pd.concat({"mean": mean, "std": std}, axis=1).loc[order]

    os.makedirs(args.out_dir, exist_ok=True)
    text.to_csv(os.path.join(args.out_dir, "seed_summary_mean_std.csv"), encoding="utf-8-sig")
    numeric.to_csv(os.path.join(args.out_dir, "seed_summary_numeric.csv"), encoding="utf-8-sig")
    all_df.to_csv(os.path.join(args.out_dir, "seed_all_runs.csv"), index=False, encoding="utf-8-sig")

    header = "| Mô hình | " + " | ".join(text.columns) + " |"
    sep = "|" + "---|" * (len(text.columns) + 1)
    rows = ["| " + m + " | " + " | ".join(text.loc[m]) + " |" for m in text.index]
    md = "\n".join([header, sep] + rows)
    note = (f"\n\n*Trung bình ± độ lệch chuẩn mẫu (ddof=1) trên {n_runs} lần chạy "
            f"({', '.join(sorted(all_df['run'].unique()))}); ngưỡng chọn trên tập Validation.*\n")
    with open(os.path.join(args.out_dir, "seed_summary.md"), "w", encoding="utf-8") as f:
        f.write(md + note)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"Đã gộp {n_runs} lần chạy từ: {', '.join(sorted(all_df['run'].unique()))}\n")
    print(text.to_string())
    print(f"\nĐã lưu 4 file vào: {args.out_dir}")


if __name__ == "__main__":
    main()
