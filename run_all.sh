#!/bin/sh
# Tái lập toàn bộ thực nghiệm (thứ tự và lệnh đúng như đã chạy)
python run_experiment.py --model-name all --imbalance class_weight --run-xai --run-eda --cv-folds 5 --tune-threshold --save-model --exp-name EXP_BASELINE_ALL
python run_experiment.py --model-name all --imbalance none  --run-xai --cv-folds 5 --tune-threshold --exp-name EXP_IMBALANCE_none
python run_experiment.py --model-name all --imbalance smote --run-xai --cv-folds 5 --tune-threshold --exp-name EXP_IMBALANCE_smote
python benchmark.py --results outputs/EXP_BASELINE_ALL/summary/results_summary_test.csv
