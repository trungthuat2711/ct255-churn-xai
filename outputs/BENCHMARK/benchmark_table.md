| study | year | method | xai | dataset | comparable | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|
| Nghiên cứu này - Logistic Regression | 2026 | Logistic Regression | TreeSHAP/LinearSHAP | Telco IBM 7.043 mẫu, test 20% stratified | yes (cùng giao thức) | 0.7651 | 0.5449 | 0.6979 | 0.6120 | 0.8422 |
| Nghiên cứu này - Decision Tree | 2026 | Decision Tree | TreeSHAP/LinearSHAP | Telco IBM 7.043 mẫu, test 20% stratified | yes (cùng giao thức) | 0.7757 | 0.5644 | 0.6791 | 0.6165 | 0.8292 |
| Nghiên cứu này - Random Forest | 2026 | Random Forest | TreeSHAP/LinearSHAP | Telco IBM 7.043 mẫu, test 20% stratified | yes (cùng giao thức) | 0.7878 | 0.5974 | 0.6150 | 0.6061 | 0.8349 |
| Nghiên cứu này - XGBoost | 2026 | XGBoost | TreeSHAP/LinearSHAP | Telco IBM 7.043 mẫu, test 20% stratified | yes (cùng giao thức) | 0.7835 | 0.5808 | 0.6631 | 0.6192 | 0.8440 |
| Nghiên cứu này - MLP (Neural Net) | 2026 | MLP (Neural Net) | - | Telco IBM 7.043 mẫu, test 20% stratified | yes (cùng giao thức) | 0.7367 | 0.5027 | 0.7567 | 0.6041 | 0.8129 |
| Nghiên cứu này - SVM (RBF) | 2026 | SVM (RBF) | - | Telco IBM 7.043 mẫu, test 20% stratified | yes (cùng giao thức) | 0.7722 | 0.5529 | 0.7406 | 0.6331 | 0.8253 |
| Nghiên cứu này - k-NN | 2026 | k-NN | - | Telco IBM 7.043 mẫu, test 20% stratified | yes (cùng giao thức) | 0.7473 | 0.5188 | 0.6658 | 0.5831 | 0.8051 |
| Chang et al., Algorithms 17(6):231 (doi:10.3390/a17060231) | 2024 | Random Forest | LIME+SHAP | Telecom churn, 'large dataset' (chưa xác nhận là Telco IBM 7.043 dòng) | unverified | 0.9166 | 0.8220 | 0.8180 | N/A | N/A |
| Christopher & Anand, IC3SE 2024, pp.534-539 | 2024 | Logistic Regression (tốt nhất trong 11 mô hình) | none | Telecom churn (chưa xác nhận tập cụ thể) | unverified | 0.8176 | 0.6895 | 0.5657 | 0.6215 | N/A |