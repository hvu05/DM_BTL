#!/usr/bin/env python3
"""Streamlit web dashboard for SOHAS Weapon Classification - Random Forest model.

    streamlit run app_rf.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from predict_rf import predict_bgr  # noqa: E402  (same preprocessing as training)

RF_DIR = ROOT / "outputs" / "rf"
SVM_DIR = ROOT / "outputs" / "svm"
METRIC_COLS = ["model", "branch", "accuracy", "balanced_accuracy", "macro_precision",
               "macro_recall", "macro_f1", "weighted_f1", "train_time_sec"]

st.set_page_config(
    page_title="SOHAS Weapon Classification - Random Forest Dashboard",
    page_icon="🌲",
    layout="wide",
)


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def show_image(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Chưa có file `{path.relative_to(ROOT)}`")


def main():
    st.title("🌲 SOHAS Weapon Classification - Random Forest Dashboard")
    st.markdown("Phân loại vũ khí và đồ vật bằng **HOG + Random Forest**")

    st.sidebar.header("Cấu hình")
    model_choice = st.sidebar.selectbox(
        "Chọn mô hình Random Forest:",
        [
            ("Random Forest - Direct resize", "direct"),
            ("Random Forest - Padding (giữ tỷ lệ)", "padding"),
        ],
        format_func=lambda x: x[0],
    )
    branch = model_choice[1]
    model_path = RF_DIR / f"rf_{branch}_model.joblib"
    if not model_path.exists():
        st.error(f"Không tìm thấy file mô hình tại `outputs/rf/{model_path.name}`. "
                 "Hãy chạy `scripts/train_rf.py` hoặc kernel Kaggle rồi chép thư mục `outputs/rf/` vào đây.")
        return
    model = load_model(str(model_path))
    st.sidebar.caption(f"Số cây: {len(model.estimators_)} · max_depth: {model.max_depth} · "
                       f"class_weight: {model.class_weight}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Thử nghiệm suy luận",
        "📊 Báo cáo đánh giá (Test Set)",
        "🧭 Giải thích mô hình",
        "⚠️ Phân tích lỗi (Error Analysis)",
    ])

    # TAB 1: Inference
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("1. Ảnh đầu vào")
            uploaded = st.file_uploader("Upload file ảnh (JPG, PNG):", type=["jpg", "jpeg", "png"])
            img_bgr = None
            if uploaded:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    st.error("Không đọc được file ảnh.")
                else:
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=uploaded.name,
                             use_container_width=True)
            else:
                st.info("Hãy upload ảnh để bắt đầu phân loại.")

        with col2:
            st.subheader("2. Kết quả nhận diện")
            if img_bgr is not None:
                with st.spinner("Đang trích xuất HOG & chạy Random Forest..."):
                    res = predict_bgr(img_bgr, model, branch)
                top = res["top_prediction"]
                if top["is_weapon"]:
                    st.error(f"🚨 **CẢNH BÁO: PHÁT HIỆN {top['name_vi'].upper()} ({top['class'].upper()})**")
                else:
                    st.success(f"✅ **AN TOÀN: {top['name_vi'].upper()} ({top['class'].upper()})**")
                c1, c2 = st.columns(2)
                c1.metric("Vật thể dự đoán", f"{top['class']} ({top['name_vi']})")
                c2.metric("Độ tin cậy", f"{top['prob'] * 100:.2f}%")
                st.write("**Xác suất các lớp (tỷ lệ phiếu bầu của các cây):**")
                chart_df = pd.DataFrame([
                    {"Lớp": f"{r['class']} ({r['name_vi']})", "Xác suất (%)": round(r["prob"] * 100, 2)}
                    for r in res["all_probabilities"]
                ]).set_index("Lớp")
                st.bar_chart(chart_df)
            else:
                st.info("Vui lòng chọn hoặc tải lên một ảnh để phân loại.")

    # TAB 2: Metrics
    with tab2:
        st.subheader("Hiệu năng Random Forest")
        rf_file = RF_DIR / "rf_metrics_summary.csv"
        if rf_file.exists():
            rf_df = pd.read_csv(rf_file)
            st.dataframe(rf_df, use_container_width=True)

            svm_file = SVM_DIR / "svm_metrics_summary.csv"
            if svm_file.exists():
                st.subheader("So sánh với SVM trên cùng tập Test")
                svm_df = pd.read_csv(svm_file)
                frames = [df[df["split"] == "test"][[c for c in METRIC_COLS if c in df.columns]]
                          for df in (svm_df, rf_df)]
                comp = pd.concat(frames, ignore_index=True).sort_values(["branch", "macro_f1"], ascending=[True, False])
                st.dataframe(comp, use_container_width=True)
                st.write("**Macro-F1 trên tập Test** (trung bình F1 của 6 lớp, mỗi lớp trọng số như nhau):")
                st.bar_chart(comp.assign(name=comp["model"] + " · " + comp["branch"]).set_index("name")[["macro_f1"]])
        else:
            st.warning("Chưa có file outputs/rf/rf_metrics_summary.csv")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Ma trận nhầm lẫn ({branch})**")
            show_image(RF_DIR / f"confusion_matrix_rf_{branch}.png")
        with c2:
            st.write(f"**F1 từng lớp, sắp theo số ảnh train ({branch})**")
            show_image(RF_DIR / f"per_class_f1_rf_{branch}.png")
        per_class_file = RF_DIR / "rf_per_class_report.csv"
        if per_class_file.exists():
            pc = pd.read_csv(per_class_file)
            st.dataframe(pc[(pc["branch"] == branch) & (pc["split"] == "test")], use_container_width=True)

        st.divider()
        st.subheader("Tinh chỉnh siêu tham số (5-fold Cross-Validation)")
        show_image(RF_DIR / f"val_curve_rf_{branch}.png",
                   "Macro-F1 theo số cây, độ sâu và trọng số lớp")
        tuning_file = RF_DIR / f"tuning_rf_{branch}.csv"
        if tuning_file.exists():
            st.dataframe(pd.read_csv(tuning_file).sort_values("cv_macro_f1_mean", ascending=False),
                         use_container_width=True)

    # TAB 3: Explainability
    with tab3:
        st.subheader("Vùng ảnh nào quan trọng với Random Forest?")
        st.markdown("Độ quan trọng (Gini importance) của 26.244 đặc trưng HOG được cộng lại theo "
                    "từng block HOG, tạo thành lưới 27×27 phủ ảnh 224×224. Vùng càng sáng, "
                    "các cây càng hay dùng đặc trưng ở vùng đó để tách các lớp.")
        c1, c2 = st.columns([3, 2])
        with c1:
            show_image(RF_DIR / f"feature_importance_map_rf_{branch}.png")
        with c2:
            top_file = RF_DIR / f"top_features_rf_{branch}.csv"
            if top_file.exists():
                st.write("**Top đặc trưng HOG quan trọng nhất**")
                st.dataframe(pd.read_csv(top_file).head(20), use_container_width=True)

    # TAB 4: Misclassifications
    with tab4:
        st.subheader("Phân tích lỗi nhầm lẫn (Misclassifications)")
        misc_file = RF_DIR / f"misclassifications_rf_{branch}.csv"
        if misc_file.exists():
            df_misc = pd.read_csv(misc_file)
            st.write(f"Tổng số mẫu đoán sai trên Test set: **{len(df_misc)} mẫu**")
            st.write("**Top các cặp lớp bị nhầm lẫn nhiều nhất:**")
            counts = (df_misc.groupby(["true_class", "predicted_class"]).size()
                      .reset_index(name="count").sort_values("count", ascending=False))
            st.dataframe(counts, use_container_width=True)
            st.write("**Các mẫu sai mà mô hình tự tin nhất** (xác suất dự đoán cao nhưng sai):")
            st.dataframe(df_misc, use_container_width=True)
        else:
            st.warning("Chưa có file misclassifications.")


if __name__ == "__main__":
    main()
