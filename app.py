# -*- coding: utf-8 -*-
"""Ứng dụng Streamlit: Dự báo Churn và giải thích mô hình (XAI) trên bộ Telco Customer Churn.

Chạy trong thư mục churn_xai/ (cạnh run_experiment.py và thư mục src/):
    pip install streamlit
    streamlit run app.py
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import app_logic as L

st.set_page_config(page_title="Dự báo Churn và XAI", page_icon="📉", layout="wide")


# ---------------------------------------------------------------- tải mô hình (cache)
@st.cache_resource(show_spinner=False)
def get_bundle(data_path):
    return L.load_bundle(data_path)


@st.cache_data(show_spinner=False)
def get_test_table(_b):
    return L.test_table(_b)


def differs(a, b):
    try:
        return not np.isclose(float(a), float(b))
    except (TypeError, ValueError):
        return str(a) != str(b)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------- trang 0: tổng quan
def page_overview(b):
    st.title("Dự báo khách hàng rời bỏ (Churn) và giải thích mô hình")
    st.write("Ứng dụng minh họa kết quả của đề tài trên bộ **Telco Customer Churn** (7,043 khách hàng). "
             "Mô hình là **XGBoost**, giải thích bằng **TreeSHAP**. Mô hình được huấn luyện lại khi mở ứng dụng "
             "(khoảng vài giây) với cùng cấu hình và cùng cách chia dữ liệu như chương trình `run_experiment.py`.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số khách hàng", f"{len(b.df):,}")
    c2.metric("Tỷ lệ rời bỏ", f"{b.y.mean():.1%}")
    c3.metric("Số đặc trưng sau mã hóa", len(b.feature_names))
    c4.metric("Ngưỡng quyết định", f"{b.thr:.2f}")
    st.subheader("Hiệu năng trên tập Test (1,409 khách hàng chưa dùng để huấn luyện)")
    m = pd.DataFrame({"Độ đo": list(b.metrics.keys()), "Giá trị": [round(v, 4) for v in b.metrics.values()]})
    st.dataframe(m, hide_index=True)
    st.subheader("Cách sử dụng")
    st.markdown(
        "- **Dự báo từng khách hàng:** chọn một khách hàng, xem xác suất rời bỏ và lý do (SHAP). Chỉnh thông tin để thử kịch bản (ví dụ đổi loại hợp đồng) và xem xác suất thay đổi thế nào.\n"
        "- **Bảng điều khiển BI:** tổng quan tỷ lệ rời bỏ, nhóm rủi ro, yếu tố quan trọng và danh sách khách hàng rủi ro cao.\n"
        "- **So sánh mô hình:** xem kết quả của 7 mô hình và các hình đã lưu trong thư mục `outputs/`.")
    st.info("Lưu ý: xác suất do mô hình dùng trọng số lớp nên thiên cao hơn tỷ lệ rời bỏ thực tế; "
            "nên dùng để xếp hạng mức rủi ro hơn là đọc như xác suất tuyệt đối.")


# ---------------------------------------------------------------- trang 1: dự báo từng khách hàng
def page_predict(b):
    import shap

    st.header("Dự báo và giải thích từng khách hàng")
    order = np.argsort(-b.p_te)
    labels = {b.ids_te[i]: f"{b.ids_te[i]} – xác suất {b.p_te[i]:.0%}" for i in order}
    cid = st.selectbox("Chọn khách hàng (tập Test, sắp xếp theo xác suất rời bỏ giảm dần)",
                       list(labels.keys()), format_func=lambda k: labels[k])
    pos = b.cid_pos[cid]
    orig = b.X_te_raw.iloc[pos].to_dict()
    p0 = float(b.p_te[pos])

    left, right = st.columns([1, 1])
    new = dict(orig)
    with left:
        st.subheader("Thông tin khách hàng")
        st.caption("Có thể chỉnh các giá trị để thử kịch bản “nếu thì”.")

        def choose(col):
            opts = b.cat_options[col]
            cur = orig[col]
            if col == "SeniorCitizen":
                cur = int(cur)
            idx = opts.index(cur) if cur in opts else 0
            return st.selectbox(col, opts, index=idx, key=f"{col}_{cid}")

        tab1, tab2, tab3 = st.tabs(["Nhân khẩu học", "Dịch vụ", "Tài khoản và cước phí"])
        with tab1:
            for c in ["gender", "SeniorCitizen", "Partner", "Dependents"]:
                new[c] = choose(c)
        with tab2:
            for c in ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                      "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
                new[c] = choose(c)
        with tab3:
            new["tenure"] = st.slider("tenure (số tháng gắn bó)", 0, 72, int(orig["tenure"]), key=f"tenure_{cid}")
            new["Contract"] = choose("Contract")
            new["PaperlessBilling"] = choose("PaperlessBilling")
            new["PaymentMethod"] = choose("PaymentMethod")
            lo, hi = b.num_ranges["MonthlyCharges"]
            new["MonthlyCharges"] = st.slider("MonthlyCharges (cước hàng tháng)", float(np.floor(lo)), float(np.ceil(hi)),
                                              float(orig["MonthlyCharges"]), step=0.05, key=f"mc_{cid}")
            new["TotalCharges"] = st.number_input("TotalCharges (tổng cước)", min_value=0.0,
                                                  value=float(orig["TotalCharges"]), step=1.0, key=f"tc_{cid}")
            auto = st.checkbox("Tự cập nhật TotalCharges = tenure × MonthlyCharges khi có thay đổi",
                               value=True, key=f"auto_{cid}")

    changed = any(differs(new[c], orig[c]) for c in b.raw_columns if c != "TotalCharges")
    if changed and auto:
        new["TotalCharges"] = round(float(new["tenure"]) * float(new["MonthlyCharges"]), 2)
    p, enc = L.predict_row(b, new)

    with right:
        st.subheader("Kết quả dự báo")
        c1, c2, c3 = st.columns(3)
        c1.metric("Xác suất rời bỏ", f"{p:.1%}",
                  delta=(f"{(p - p0) * 100:+.1f} điểm % so với ban đầu" if changed else None), delta_color="inverse")
        c2.metric("Nhóm rủi ro", L.risk_group(p)[0])
        c3.metric("Ngưỡng quyết định", f"{b.thr:.2f}")
        st.progress(float(min(max(p, 0.0), 1.0)))
        if p >= b.thr:
            st.error("Dự báo: khách hàng CÓ NGUY CƠ RỜI BỎ")
        else:
            st.success("Dự báo: khách hàng CÓ KHẢ NĂNG Ở LẠI")
        actual = int(b.y_te.iloc[pos])
        st.caption(f"Nhãn thực tế trong dữ liệu: {'đã rời bỏ' if actual else 'còn ở lại'} "
                   f"(xác suất ban đầu của mô hình: {p0:.1%}).")

        st.subheader("Vì sao mô hình dự báo như vậy? (SHAP)")
        ex_row = L.explain_row(b, enc, new)
        shap.plots.waterfall(ex_row, show=False, max_display=12)
        fig = plt.gcf()
        fig.set_size_inches(8, 5)
        show_fig(fig)
        st.caption("Thanh đỏ đẩy xác suất rời bỏ lên, thanh xanh kéo xuống. Đơn vị là log-odds (thang đo của XGBoost).")
        st.dataframe(L.top_factors(ex_row, new), hide_index=True)


# ---------------------------------------------------------------- trang 2: bảng điều khiển BI
def page_dashboard(b):
    st.header("Bảng điều khiển BI")
    T = get_test_table(b)
    st.caption("Số liệu dự báo tính trên tập Test (1,409 khách hàng chưa dùng để huấn luyện). "
               "Biểu đồ tỷ lệ rời bỏ theo hợp đồng và thời gian gắn bó tính trên toàn bộ dữ liệu.")

    high = T[T["Nhóm rủi ro"] == "Cao"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Khách hàng (Test)", f"{len(T):,}")
    k2.metric("Tỷ lệ rời bỏ thực tế", f"{T['Churn thực tế'].mean():.1%}")
    k3.metric("Khách rủi ro cao", f"{len(high):,}")
    k4.metric("Tỷ lệ rời bỏ thực tế ở nhóm cao", f"{high['Churn thực tế'].mean():.1%}" if len(high) else "–")

    st.subheader("Phân tích dữ liệu")
    c1, c2 = st.columns(2)
    with c1:
        r = b.df.groupby("Contract")["Churn"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(r.index, r.values)
        for i, v in enumerate(r.values):
            ax.text(i, v + 0.01, f"{v:.1%}", ha="center")
        ax.set_ylabel("Tỷ lệ rời bỏ")
        ax.set_title("Tỷ lệ rời bỏ theo loại hợp đồng")
        show_fig(fig)
    with c2:
        bins = pd.cut(b.df["tenure"], [-1, 12, 24, 36, 48, 60, 72])
        r = b.df.groupby(bins, observed=True)["Churn"].mean()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar([str(i) for i in r.index], r.values)
        ax.set_ylabel("Tỷ lệ rời bỏ")
        ax.set_title("Tỷ lệ rời bỏ theo thời gian gắn bó (tháng)")
        plt.xticks(rotation=20)
        show_fig(fig)

    st.subheader("Dự báo của mô hình")
    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.hist(T.loc[T["Churn thực tế"] == 0, "Xác suất churn"], bins=30, alpha=0.6, label="Ở lại")
        ax.hist(T.loc[T["Churn thực tế"] == 1, "Xác suất churn"], bins=30, alpha=0.6, label="Rời bỏ")
        ax.axvline(b.thr, linestyle="--", color="black", label=f"Ngưỡng {b.thr:.2f}")
        ax.set_xlabel("Xác suất rời bỏ dự báo")
        ax.set_ylabel("Số khách hàng")
        ax.legend()
        show_fig(fig)
    with c4:
        g = T.groupby("Nhóm rủi ro")["Churn thực tế"].agg(["count", "mean"]).reindex(L.RISK_LABELS)
        g.columns = ["Số khách hàng", "Tỷ lệ rời bỏ thực tế"]
        g["Tỷ lệ rời bỏ thực tế"] = (g["Tỷ lệ rời bỏ thực tế"] * 100).round(1).astype(str) + "%"
        st.write("**Nhóm rủi ro (thấp < 0.3, trung bình 0.3–0.6, cao > 0.6)**")
        st.dataframe(g)

    st.subheader("Yếu tố ảnh hưởng nhiều nhất (SHAP)")
    import shap
    t1, t2 = st.tabs(["Mức độ quan trọng", "Chiều tác động (beeswarm)"])
    with t1:
        imp = pd.Series(np.abs(b.shap_te).mean(axis=0), index=b.feature_names).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(imp.index, imp.values)
        ax.set_xlabel("Trung bình |SHAP|")
        show_fig(fig)
    with t2:
        shap.summary_plot(b.shap_te, b.Xte_enc, show=False, max_display=12)
        fig = plt.gcf()
        fig.set_size_inches(8, 5)
        show_fig(fig)

    st.subheader("Danh sách khách hàng cần chăm sóc")
    f1, f2, f3 = st.columns(3)
    groups = f1.multiselect("Nhóm rủi ro", L.RISK_LABELS, default=["Cao"])
    contracts = f2.multiselect("Loại hợp đồng", sorted(T["Hợp đồng"].unique()), default=sorted(T["Hợp đồng"].unique()))
    topn = f3.slider("Số dòng hiển thị", 10, 100, 20, step=10)
    view = T[T["Nhóm rủi ro"].isin(groups) & T["Hợp đồng"].isin(contracts)].sort_values("Xác suất churn", ascending=False)
    st.dataframe(view.head(topn)[["customerID", "Xác suất churn", "Nhóm rủi ro", "Hợp đồng", "tenure", "MonthlyCharges",
                                  "Yếu tố làm tăng nguy cơ", "Churn thực tế"]], hide_index=True)
    st.download_button("Tải toàn bộ bảng dự báo (CSV)", T.to_csv(index=False).encode("utf-8-sig"),
                       file_name="du_bao_churn_test.csv", mime="text/csv")


# ---------------------------------------------------------------- trang 3: so sánh mô hình
MAIN_COLS = ["model", "threshold", "accuracy", "precision", "recall_sensitivity_tpr", "f1", "cohen_kappa", "mcc",
             "roc_auc", "ap_churn(PR-AUC)"]
PRETTY = {"model": "Mô hình", "threshold": "Ngưỡng", "accuracy": "Accuracy", "precision": "Precision",
          "recall_sensitivity_tpr": "Recall", "f1": "F1", "cohen_kappa": "Kappa", "mcc": "MCC", "roc_auc": "ROC-AUC",
          "ap_churn(PR-AUC)": "PR-AUC"}


def page_compare(outdir):
    st.header("So sánh các mô hình")
    exps = sorted(d for d in glob.glob(os.path.join(outdir, "EXP_*"))
                  if os.path.exists(os.path.join(d, "summary", "results_summary_test.csv")))
    if not exps:
        st.warning(f"Không tìm thấy kết quả trong '{outdir}'. Hãy chạy run_experiment.py trước (hoặc đổi đường dẫn ở thanh bên).")
        return
    names = [os.path.basename(d) for d in exps]
    default = names.index("EXP_BASELINE_ALL") if "EXP_BASELINE_ALL" in names else 0
    pick = st.selectbox("Chọn lần chạy", names, index=default)
    d = exps[names.index(pick)]
    df = pd.read_csv(os.path.join(d, "summary", "results_summary_test.csv"))
    cols = [c for c in MAIN_COLS if c in df.columns]
    show = df[cols].rename(columns=PRETTY).copy()
    for c in show.columns[1:]:
        show[c] = show[c].round(4)
    st.subheader("Kết quả trên tập Test")
    st.dataframe(show, hide_index=True)

    metric = st.selectbox("Biểu đồ theo độ đo", [PRETTY[c] for c in cols[1:]], index=cols[1:].index("roc_auc") if "roc_auc" in cols else 0)
    s = show.set_index("Mô hình")[metric].sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(s.index, s.values)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:.4f}", va="center")
    ax.set_xlabel(metric)
    show_fig(fig)

    seed_csv = os.path.join(outdir, "seed_summary", "seed_summary_mean_std.csv")
    if os.path.exists(seed_csv):
        st.subheader("Trung bình ± độ lệch chuẩn trên nhiều seed")
        st.dataframe(pd.read_csv(seed_csv), hide_index=True)

    st.subheader("Hình đã lưu của lần chạy này")
    groups = {"So sánh mô hình": os.path.join(d, "summary", "*.png"),
              "XGBoost": os.path.join(d, "xgboost", "*.png"),
              "SHAP": os.path.join(d, "xgboost", "xai", "*.png"),
              "Phân tích dữ liệu": os.path.join(d, "eda", "*.png")}
    tabs = st.tabs(list(groups.keys()))
    for tab, pattern in zip(tabs, groups.values()):
        with tab:
            files = sorted(glob.glob(pattern))
            if not files:
                st.info("Không có hình trong thư mục này.")
            for i in range(0, len(files), 2):
                cols2 = st.columns(2)
                for col, f in zip(cols2, files[i:i + 2]):
                    col.image(f, caption=os.path.basename(f))


# ---------------------------------------------------------------- main
def main():
    st.sidebar.title("Dự báo Churn và XAI")
    page = st.sidebar.radio("Chọn trang", ["Tổng quan", "Dự báo từng khách hàng", "Bảng điều khiển BI", "So sánh mô hình"])
    data_path = st.sidebar.text_input("Đường dẫn dữ liệu", "data/Telco-Customer-Churn.csv")
    outdir = st.sidebar.text_input("Thư mục kết quả", "outputs")
    st.sidebar.caption("Mô hình: XGBoost + TreeSHAP. Chia dữ liệu 70/10/20, seed 42.")

    if page == "So sánh mô hình":
        page_compare(outdir)
        return
    if not os.path.exists(data_path):
        st.error(f"Không tìm thấy tệp dữ liệu: {data_path}")
        st.stop()
    try:
        with st.spinner("Đang huấn luyện mô hình và tính SHAP (khoảng vài giây)..."):
            b = get_bundle(data_path)
    except ImportError as e:
        st.error(f"Thiếu thư viện: {e}. Hãy chạy: pip install -r requirements.txt")
        st.stop()
    {"Tổng quan": page_overview, "Dự báo từng khách hàng": page_predict, "Bảng điều khiển BI": page_dashboard}[page](b)


main()
