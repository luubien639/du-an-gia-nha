import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # 1. Đọc dữ liệu
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'house_prices.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {data_path}")
    
    df = pd.read_csv(data_path)
    print("--- Dữ liệu 5 dòng đầu tiên ---")
    print(df.head(), "\n")

    # 2. Tách features (X) và target (y)
    feature_cols = ['SquareMeters', 'Bedrooms', 'DistanceToCenterKM']
    target_col = 'Price'

    X = df[feature_cols]
    y = df[target_col]

    # 3. Chia tập huấn luyện và kiểm thử (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Huấn luyện mô hình Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Dự báo trên tập kiểm thử
    y_pred = model.predict(X_test)

    # 6. Đánh giá mô hình
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("--- Kết quả đánh giá trên tập kiểm thử ---")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}\n")

    print("--- Hệ số hồi quy ---")
    for col, coef in zip(feature_cols, model.coef_):
        print(f"Hệ số của {col}: {coef:.4f}")
    print(f"Hệ số chặn (Intercept): {model.intercept_:.4f}\n")

    # 7. Dự báo cho 1 căn nhà mới ví dụ (85m2, 3 phòng ngủ, cách trung tâm 5km)
    sample_house = pd.DataFrame([[85, 3, 5]], columns=feature_cols)
    pred_price = model.predict(sample_house)[0]
    print(f"Dự báo giá cho nhà (85m2, 3 phòng ngủ, 5km): {pred_price:.2f} triệu VNĐ")

    # 8. Lưu biểu đồ so sánh Actual vs Predicted
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, color='blue', edgecolors='k', alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Giá thực tế (Actual)')
    plt.ylabel('Giá dự báo (Predicted)')
    plt.title('Actual vs Predicted House Prices')
    plt.tight_layout()
    plt.savefig('result_plot.png')
    print("\nĐã lưu biểu đồ so sánh tại: result_plot.png")

if __name__ == '__main__':
    main()
