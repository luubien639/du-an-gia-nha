Dự đoán giá nhà bằng Hồi quy tuyến tính (Linear Regression)
Bài tập cài đặt thuật toán Hồi quy tuyến tính đa biến từ đầu bằng Gradient Descent (không dùng thư viện scikit-learn), áp dụng để dự đoán giá nhà dựa trên diện tích và số phòng ngủ.

Mô tả bài toán
Chương trình dự đoán giá nhà (Price) dựa trên 2 đặc trưng đầu vào:

Area: diện tích nhà
Bedrooms: số phòng ngủ
Mô hình có dạng:

Price = w1 * Area + w2 * Bedrooms + b
Trong đó w1, w2, b được học tự động từ dữ liệu bằng thuật toán Gradient Descent.

Dữ liệu
File: House_Price_Prediction_Dataset.csv
Nguồn: Kaggle
Số lượng: 2000 mẫu
Cột dữ liệu: Id, Area, Bedrooms, Price
Yêu cầu cài đặt
pip install pandas numpy
Cách chạy
Đảm bảo file House_Price_Prediction_Dataset.csv nằm cùng thư mục với file bai_tap_hoi_quy_tuyen_tinh.py.
Chạy chương trình:
python bai_tap_hoi_quy_tuyen_tinh.py
Chương trình sẽ hỏi nhập diện tích và số phòng ngủ để dự đoán giá nhà.
Các bước xử lý trong chương trình
Đọc dữ liệu từ file CSV bằng pandas.
Chia tập train/test theo tỉ lệ 80/20.
Chuẩn hóa dữ liệu (Feature Scaling, dùng Z-score) — vì Area và Bedrooms có thang giá trị chênh lệch lớn, cần chuẩn hóa để Gradient Descent hội tụ đúng.
Huấn luyện mô hình bằng Gradient Descent tự cài đặt (không dùng thư viện học máy).
Đánh giá mô hình trên tập test bằng các chỉ số MSE, RMSE, R².
Dự đoán giá nhà mới dựa trên diện tích và số phòng ngủ do người dùng nhập vào.
Kết quả
Chỉ số	Giá trị
MSE	~77,856,354,491.91
RMSE	~279,027.52
R²	~-0.0007
Nhận xét: hệ số R² gần bằng 0 cho thấy trong bộ dữ liệu này, Area và Bedrooms gần như không có quan hệ tuyến tính rõ ràng với Price — đây là đặc điểm của bộ dữ liệu (có vẻ được sinh ngẫu nhiên để luyện tập), không phải lỗi của thuật toán cài đặt.

Công nghệ sử dụng
Python 3
pandas — đọc và xử lý dữ liệu
NumPy — tính toán ma trận, vector hóa Gradient Descent
