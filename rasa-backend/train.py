import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# === 1. Đọc dữ liệu ===
file_path = 'C:/Users\Admin\PycharmProjects\chatlord\combined_data.csv'
df = pd.read_csv(file_path)

# === 2. Mã hóa biến phân loại ===
categorical_cols = ['cp', 'restecg', 'slope', 'thal']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# === 3. Chuẩn hóa biến số ===
numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# === 4. Tách biến độc lập và mục tiêu ===
X = df.drop('target', axis=1)
y = df['target']

# === 5. Chia tập huấn luyện và kiểm tra ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# === 6. Tạo pipeline với Logistic Regression ===
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    )
)

# === 7. Huấn luyện mô hình ===
model.fit(X_train, y_train)

# === 8. Dự đoán và đánh giá ===
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Lấy xác suất lớp 1 (target=1)
y_pred_class = model.predict(X_test)  # Dự đoán nhãn lớp (0/1)

# Metrics phân loại (truyền thống)
accuracy = accuracy_score(y_test, y_pred_class)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Metrics hồi quy (tính trên xác suất)
mae = mean_absolute_error(y_test, y_pred_proba)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba))
r2 = r2_score(y_test, y_pred_proba)

print("\n=== Đánh giá mô hình ===")
print("\nMetrics phân loại:")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

print("\nMetrics hồi quy (tính trên xác suất):")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# === 9. Tuning tham số ===
param_grid = {
    'logisticregression__C': [0.01, 0.1, 1, 10],
    'logisticregression__penalty': ['l1', 'l2'],
    'logisticregression__solver': ['liblinear']
}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='roc_auc',  # Vẫn ưu tiên tối ưu AUC-ROC
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# === 10. Mô hình tốt nhất ===
best_model = grid_search.best_estimator_
print("\nTham số tốt nhất:", grid_search.best_params_)

# Đánh giá lại
best_pred_proba = best_model.predict_proba(X_test)[:, 1]
best_mae = mean_absolute_error(y_test, best_pred_proba)
best_rmse = np.sqrt(mean_squared_error(y_test, best_pred_proba))
best_r2 = r2_score(y_test, best_pred_proba)

print("\n=== Đánh giá mô hình tốt nhất ===")
print(f"MAE: {best_mae:.4f}")
print(f"RMSE: {best_rmse:.4f}")
print(f"R2: {best_r2:.4f}")

# === 11. Lưu mô hình ===
model_path = 'C:/Users\Admin\PycharmProjects\chatlord\heart_disease_model_logistic.pkl'
joblib.dump(best_model, model_path)
print(f"\n✅ Mô hình đã lưu tại: {model_path}")