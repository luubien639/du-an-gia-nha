import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# 1. Đọc dữ liệu
data = pd.read_csv("data/train.csv")

print("5 dòng đầu tiên:")
print(data.head())


# 2. Chọn dữ liệu đầu vào
X = data[["OverallQual", "GrLivArea", "GarageCars"]]

# 3. Chọn giá nhà cần dự đoán
y = data["SalePrice"]


# 4. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 5. Tạo mô hình
model = LinearRegression()


# 6. Huấn luyện mô hình
model.fit(X_train, y_train)


# 7. Dự đoán
y_pred = model.predict(X_test)


# 8. Đánh giá mô hình
r2 = r2_score(y_test, y_pred)

print("R2 =", r2)


# 9. Thông tin mô hình
print("Các hệ số:")
print(model.coef_)

print("Intercept:")
print(model.intercept_)


# 10. Dự đoán một căn nhà mới
new_house = [[7, 1500, 2]]

predicted_price = model.predict(new_house)

print("Giá nhà dự đoán:", predicted_price[0])
