import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ tệp CSV
dulieu = pd.read_csv("brute_force_dataset_realistic_corrected.csv")


# Tách dữ liệu thành đặc trưng (X) và nhãn (y)
X = dulieu.drop(columns=["Attack (0: No, 1: Yes)"])
y = dulieu["Attack (0: No, 1: Yes)"]

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Kiểm tra sự cân bằng của tập kiểm tra
print("Phân phối nhãn trong tập kiểm tra:")
print(y_test.value_counts())

# Áp dụng SMOTE để cân bằng dữ liệu huấn luyện
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Giảm overfitting bằng cách điều chỉnh siêu tham số
mo_hinh = RandomForestClassifier(n_estimators=50, max_depth=2, min_samples_split=10, min_samples_leaf=5, random_state=42)
mo_hinh.fit(X_train_resampled, y_train_resampled)

# Dự đoán trên tập kiểm tra
y_pred = mo_hinh.predict(X_test)
y_probs = mo_hinh.predict_proba(X_test)[:, 1]  # Xác suất dự đoán

# Đánh giá mô hình
do_chinh_xac = accuracy_score(y_test, y_pred)
bao_cao = classification_report(y_test, y_pred)
ma_tran_nham_lan = confusion_matrix(y_test, y_pred)

print(f"Độ chính xác: {do_chinh_xac:.4f}")
print("Báo cáo phân loại:")
print(bao_cao)
print("Ma trận nhầm lẫn:")
print(ma_tran_nham_lan)

# Vẽ biểu đồ ma trận nhầm lẫn
plt.figure(figsize=(6, 4))
sns.heatmap(ma_tran_nham_lan, annot=True, fmt="d", cmap="Blues", xticklabels=["Bình thường", "Tấn công"], yticklabels=["Bình thường", "Tấn công"])
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Ma trận nhầm lẫn")
plt.show()

# Vẽ biểu đồ ROC
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Vẽ biểu đồ phân phối xác suất dự đoán
plt.figure(figsize=(6, 4))
sns.histplot(y_probs, bins=30, kde=True, color="blue")
plt.xlabel("Xác suất dự đoán là tấn công")
plt.ylabel("Số lượng mẫu")
plt.title("Phân phối xác suất dự đoán")
plt.show()

# Vẽ biểu đồ histogram của độ lỗi (Residuals)
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.histplot(residuals, bins=20, kde=True, color='purple', alpha=0.7)
plt.xlabel("Sai số (Residuals)")
plt.ylabel("Số lượng mẫu")
plt.title("Biểu đồ phân phối sai số dự đoán")
plt.show()

# Lưu mô hình để sử dụng sau này
joblib.dump(mo_hinh, "brute_force_rf.pkl")
