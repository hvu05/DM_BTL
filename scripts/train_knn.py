import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# ================= CẤU HÌNH =================
#BRANCH = "direct"  # Lần 1 chạy "direct", lần 2 đổi thành "padding"
BRANCH = "padding"
NPZ_PATH = Path(f"dataset/features/hog_{BRANCH}.npz")
OUTPUT_DIR = Path(f"outputs/knn_{BRANCH}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

# ================= 1. NẠP DỮ LIỆU =================
data = np.load(NPZ_PATH, allow_pickle=True)
X = data["X"]
y = data["y"]
split = data["split"]
class_names = [str(c) for c in data["class_names"]]

dev_mask = (split == "train") | (split == "val")
test_mask = split == "test"

X_dev, y_dev = X[dev_mask], y[dev_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Nhánh dữ liệu: {BRANCH}")
print(f"Tập phát triển (Train + Val cho 5-Fold CV): {X_dev.shape[0]} mẫu × {X_dev.shape[1]} đặc trưng")
print(f"Tập kiểm tra độc lập (Test): {X_test.shape[0]} mẫu")

# ================= 2. PIPELINE & K-FOLD CV =================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(random_state=SEED)),
    ("knn", KNeighborsClassifier(n_jobs=-1)),
])

param_grid = {
    "pca__n_components": [100, 200, 300],
    "knn__n_neighbors": [3, 5, 7, 11],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=1,
    verbose=2,
    refit=True,
)

t0 = time.perf_counter()
grid_search.fit(X_dev, y_dev)
cv_total_time = time.perf_counter() - t0

best_model = grid_search.best_estimator_
print("\n=== THAM SỐ TỐT NHẤT (5-FOLD CV) ===")
print(grid_search.best_params_)
print(f"Best CV Macro-F1: {grid_search.best_score_:.4f}")
print(f"Tổng thời gian chạy GridSearchCV: {cv_total_time:.2f} giây")

cv_results_df = pd.DataFrame(grid_search.cv_results_)[
    ["params", "mean_test_score", "std_test_score", "mean_fit_time", "mean_score_time", "rank_test_score"]
].sort_values("rank_test_score")
cv_results_df.to_csv(OUTPUT_DIR / "knn_cv_results.csv", index=False)

# ================= 3. ĐÁNH GIÁ TRÊN TẬP TEST =================
t_fit_start = time.perf_counter()
best_model.fit(X_dev, y_dev)
train_time = time.perf_counter() - t_fit_start

t_pred_start = time.perf_counter()
y_pred = best_model.predict(X_test)
pred_time = time.perf_counter() - t_pred_start

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")
macro_prec = precision_score(y_test, y_pred, average="macro")
macro_rec = recall_score(y_test, y_pred, average="macro")

summary = {
    "branch": BRANCH,
    "best_params": grid_search.best_params_,
    "cv_best_macro_f1": round(float(grid_search.best_score_), 4),
    "test_accuracy": round(float(acc), 4),
    "test_macro_f1": round(float(macro_f1), 4),
    "test_weighted_f1": round(float(weighted_f1), 4),
    "test_macro_precision": round(float(macro_prec), 4),
    "test_macro_recall": round(float(macro_rec), 4),
    "best_model_fit_time_sec": round(float(train_time), 3),
    "test_predict_time_sec": round(float(pred_time), 3),
    "ms_per_image_predict": round(float(pred_time / len(X_test) * 1000), 3),
}

with open(OUTPUT_DIR / "knn_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

joblib.dump(best_model, OUTPUT_DIR / f"knn_{BRANCH}_best.joblib")

print("\n=== KẾT QUẢ TRÊN TẬP TEST ===")
for k, v in summary.items():
    print(f"{k}: {v}")

report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(OUTPUT_DIR / "knn_classification_report.csv")
print("\n=== BÁO CÁO TỪNG LỚP ===")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

# ================= 4. VẼ MA TRẬN NHẦM LẪN =================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title(f"Confusion Matrix — KNN ({BRANCH}) | Acc={acc*100:.2f}%, Macro-F1={macro_f1:.4f}")
plt.xlabel("Nhãn dự đoán (Predicted)")
plt.ylabel("Nhãn thực tế (True)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "knn_confusion_matrix.png", dpi=200)
plt.close()
print(f"\nĐã lưu toàn bộ kết quả tại: {OUTPUT_DIR}")