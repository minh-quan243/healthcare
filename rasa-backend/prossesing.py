import pandas as pd

# === 1. Đọc dữ liệu ===
file_path = 'C:/Users\Admin\PycharmProjects\chatlord\combined_data.csv'  # Cập nhật đường dẫn nếu cần
df = pd.read_csv(file_path)

# === 2. Thống kê mô tả cơ bản ===
print("\n=== Thống kê mô tả cơ bản ===")
print("5 dòng đầu tiên:")
print(df.head())

print("\nThông tin DataFrame:")
print(df.info())

print("\nThống kê tổng quan:")
print(df.describe(include='all'))

# === 3. Thống kê chi tiết các biến quan trọng ===
print("\n=== Thống kê chi tiết các biến quan trọng ===")
important_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']
available_vars = [col for col in important_vars if col in df.columns]

if len(available_vars) < len(important_vars):
    missing = set(important_vars) - set(df.columns)
    print(f"Cảnh báo: Thiếu các cột {missing}")

if available_vars:
    stats = pd.DataFrame({
        'Mean': df[available_vars].mean(),
        'Median': df[available_vars].median(),
        'Mode': df[available_vars].mode().iloc[0],
        'Variance': df[available_vars].var(),
        'Std Dev': df[available_vars].std(),
        'Min': df[available_vars].min(),
        'Max': df[available_vars].max(),
        'Skewness': df[available_vars].skew(),
        'Kurtosis': df[available_vars].kurt()
    })
    print(stats)
else:
    print("Không có biến quan trọng nào tồn tại trong DataFrame")

# === 4. Thống kê tần số cho biến phân loại ===
print("\n=== Tần số các biến phân loại ===")
categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
available_cats = [col for col in categorical_vars if col in df.columns]

if len(available_cats) < len(categorical_vars):
    missing = set(categorical_vars) - set(df.columns)
    print(f"Cảnh báo: Thiếu các cột phân loại {missing}")

for var in available_cats:
    print(f"\nPhân phối của {var}:")
    try:
        freq = df[var].value_counts(dropna=False)
        perc = df[var].value_counts(normalize=True, dropna=False) * 100
        print(pd.DataFrame({'Count': freq, 'Percentage': perc.round(2)}))

        if df[var].isnull().any():
            print(f"Lưu ý: Có {df[var].isnull().sum()} giá trị null")
    except Exception as e:
        print(f"Không thể phân tích cột {var}: {str(e)}")

# === 5. Kiểm tra giá trị đặc biệt ===
print("\n=== Kiểm tra giá trị đặc biệt ===")
print("Giá trị duy nhất trong các cột phân loại:")
for var in available_cats:
    print(f"{var}: {sorted(df[var].unique())}")

# === 6. Xử lý dữ liệu thiếu ===
print("\n=== Xử lý dữ liệu thiếu ===")
print("Trước xử lý:\n", df.isnull().sum())
if df.isnull().values.any():
    df.interpolate(method="linear", inplace=True)
    df.bfill(inplace=True)
print("Sau xử lý:\n", df.isnull().sum())

# === 7. Kiểm tra và loại bỏ dữ liệu trùng lặp ===
print("\n=== Kiểm tra dữ liệu trùng lặp ===")
print("Số dòng trùng lặp:", df.duplicated().sum())
if df.duplicated().sum() > 0:
    df.drop_duplicates(inplace=True)
    print("Đã loại bỏ các dòng trùng lặp.")
print("Số dòng trùng lặp sau xử lý:", df.duplicated().sum())

# === 8. Xử lý outliers ===
print("\n=== Xử lý Outliers ===")
numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    before_rows = df.shape[0]
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    after_rows = df.shape[0]
    print(f"{col}: Đã loại bỏ {before_rows - after_rows} dòng ngoài khoảng IQR")

# === 9. Lưu dữ liệu đã xử lý ===
print("\nData processing completed!")
output_path = 'C:/Users\Admin\PycharmProjects\chatlord\combined_data.csv'  # Cập nhật đường dẫn nếu cần
df.to_csv(output_path, index=False)
print(f"\n✅ Dữ liệu đã xử lý được lưu vào: {output_path}")
