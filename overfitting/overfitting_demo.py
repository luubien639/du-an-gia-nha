import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# 1. Đọc dữ liệu (File nằm cùng thư mục nên chỉ cần gọi tên file)
data = pd.read_csv("data.csv")

# CHÚ Ý: Đảm bảo file data.csv của bạn có các cột này. 
# Nếu data.csv dùng tên cột khác (như SquareMeters, Bedrooms, Price), hãy sửa lại cho khớp.
features = ["OverallQual", "GrLivArea", "GarageCars"]
X = data[features]
y = data["SalePrice"]

# 2. Cố tình lấy tập huấn luyện rất nhỏ (50 mẫu) để gây Overfitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=50, random_state=42
)

# 3. MÔ HÌNH 1: GÂY OVERFITTING (Đa thức bậc 5)
overfit_model = make_pipeline(
    PolynomialFeatures(degree=5), 
    StandardScaler(), 
    LinearRegression()
)
overfit_model.fit(X_train, y_train)

# 4. MÔ HÌNH 2: KHẮC PHỤC BẰNG RIDGE REGRESSION
ridge_model = make_pipeline(
    PolynomialFeatures(degree=5), 
    StandardScaler(), 
    Ridge(alpha=100.0) 
)
ridge_model.fit(X_train, y_train)

# 5. In kết quả
print("=== MÔ HÌNH 1: BỊ OVERFITTING ===")
print(f"R2 Train : {r2_score(y_train, overfit_model.predict(X_train)):.4f}")
print(f"R2 Test : {r2_score(y_test, overfit_model.predict(X_test)):.4f}\n")

print("=== MÔ HÌNH 2: ĐÃ KHẮC PHỤC (RIDGE) ===")
print(f"R2 Train : {r2_score(y_train, ridge_model.predict(X_train)):.4f}")
print(f"R2 Test : {r2_score(y_test, ridge_model.predict(X_test)):.4f}")