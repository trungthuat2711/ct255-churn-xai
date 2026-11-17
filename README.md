# Dự báo Churn & XAI (Telco Customer Churn) – bộ thực nghiệm dòng lệnh

## Chạy nhanh
```bash
pip install -r requirements.txt
python run_experiment.py --model-name xgboost --run-xai --run-eda --tune-threshold      # 1 mô hình
python run_experiment.py --model-name all --run-xai --run-eda --cv-folds 5 --exp-name EXP_X   # 7 mô hình
python benchmark.py --results outputs/EXP_X/summary/results_summary_test.csv              # đối sánh nghiên cứu liên quan
sh run_all.sh                                                                              # tái lập toàn bộ kết quả
python run_experiment.py -h                                                                # xem mọi tham số
```

## Tham số quan trọng
| Nhóm | Cờ | Mặc định | Ý nghĩa |
|---|---|---|---|
| Dữ liệu | `--data-path` `--test-size` `--val-size` `--random-state` `--impute` | …/0.2/0.1/42/zero | nguồn, chia train/val/test (stratified), seed |
| Mất cân bằng | `--imbalance {none,class_weight,smote}` | class_weight | xử lý chỉ trên tập train |
| Mô hình | `--model-name {logreg,dtree,rforest,xgboost,mlp,svm,knn,all}` | xgboost | thuật toán |
| Siêu tham số | `--n-estimators --max-depth --learning-rate --scale-pos-weight --subsample --colsample-bytree --reg-lambda --C --k-neighbors --hidden-sizes --epochs --batch-size --mlp-lr` | xem `-h` | |
| Đánh giá | `--threshold` `--tune-threshold` `--cv-folds` `--bins` | 0.5 / off / 5 / 30 | ngưỡng chọn trên **validation** |
| XAI | `--run-xai --xai-index --xai-sample --xai-noise` | off | SHAP + Fidelity/Stability/Sparsity |
| Đầu ra | `--output-dir --exp-name --run-eda --save-model --n-jobs` | outputs | |

## Cấu trúc kết quả `outputs/<EXP>/`
- `experiment_log.txt` – lệnh chạy, môi trường, tham số, dữ liệu, bảng tổng hợp.
- `<model>/model_log.txt` – dữ liệu, **kiến trúc**, **siêu tham số**, kết quả train/val/test, ma trận nhầm lẫn, CV, XAI (+ `results.json`).
- `<model>/*.png|csv` – loss/acc theo epoch, ROC/PR, confusion matrix, calibration/gain/lift, SHAP, `predictions.csv`.
- `eda/` – histogram (+KDE), box plot, heatmap, tỷ lệ churn theo nhóm.
- `artifact_index.csv` – **mỗi hình/bảng ↔ `file code:hàm:dòng` sinh ra nó + lệnh chạy + SHA-256**. Dùng cột `generated_by`
  ghép với URL repo (vd `https://github.com/<user>/<repo>/blob/<commit>/src/plots.py#L<dòng>`) để gắn link vào báo cáo.
- `../all_experiments.txt` – nhật ký nối tiếp (append) mọi lần chạy.

## Độ đo không áp dụng cho bài toán này
CMC (nhận dạng/re-id) và các độ đo ẩn tin (PSNR, SSIM, BER, payload) không có ý nghĩa với dữ liệu bảng Churn; IoU và mAP được quy đổi
sang phân loại nhị phân (IoU = TP/(TP+FP+FN); mAP = trung bình AP hai lớp). Ghi rõ trong mọi `model_log.txt`.
