import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

# Đọc dữ liệu từ file CSV và đặt tần suất cho chỉ số thời gian
df = pd.read_csv('data-dak-to_2003-2011.csv', parse_dates=['Time'])
df.set_index('Time', inplace=True)

# Kiểm tra và xử lý các giá trị NaN và vô hạn
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Chọn chuỗi thời gian cần dự báo (dòng chảy khu vực A)
endog = df['Dak_To_flow']

# Chọn các biến ngoại sinh (dòng chảy khu vực B và lượng mưa ở các khu vực A, B, C, D)
exog = df[['Dak_Mod_flow', 'Dak_Mod_rain', 'Dak_To_rain', 'Dak_Glei_rain', 'Sa_Thay_rain']]

# Tạo giao diện với Streamlit
st.title('Dự báo dòng chảy bằng mô hình SARIMAX')

# Chọn khoảng thời gian huấn luyện
start_date = st.date_input('Chọn ngày bắt đầu', value=pd.to_datetime('2003-06-01'))
end_date = st.date_input('Chọn ngày kết thúc', value=pd.to_datetime('2009-11-30'))

if start_date >= end_date:
    st.error('Ngày bắt đầu phải nhỏ hơn ngày kết thúc')
else:
    # Lọc dữ liệu theo khoảng thời gian được chọn
    train_endog = endog[start_date:end_date]
    train_exog = exog[start_date:end_date]

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
    train_size = int(len(train_endog) * 0.8)
    train_endog, test_endog = train_endog[:train_size], train_endog[train_size:]
    train_exog, test_exog = train_exog[:train_size], train_exog[train_size:]

    # Xác định các tham số của mô hình SARIMA
    p, d, q = 4, 1, 5
    P, D, Q, s = 2, 1, 2, 6  # Thiết lập chu kỳ mùa vụ là 6

    # Xây dựng mô hình SARIMAX trên tập huấn luyện
    warnings.filterwarnings("ignore")
    model = sm.tsa.SARIMAX(train_endog, exog=train_exog, order=(p, d, q), seasonal_order=(P, D, Q, s))
    results = model.fit()

    # Dự đoán trên tập kiểm tra
    predictions = results.predict(start=len(train_endog), end=len(train_endog) + len(test_endog) - 1, exog=test_exog)

    # Đánh giá độ chính xác của mô hình
    mse = mean_squared_error(test_endog, predictions)
    mae = mean_absolute_error(test_endog, predictions)
    st.write(f'Mean Squared Error: {mse}')
    st.write(f'Mean Absolute Error: {mae}')

    # Vẽ biểu đồ so sánh giá trị thực tế và giá trị dự đoán
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_endog.index, train_endog, label='Train')
    ax.plot(test_endog.index, test_endog, label='Test')
    ax.plot(test_endog.index, predictions, label='Predictions', color='red')
    ax.legend()
    st.pyplot(fig)

    # Thêm tính năng dự đoán cho khung giờ tiếp theo
    st.subheader('Dự báo cho khung giờ tiếp theo')
    forecast_steps = st.number_input('Nhập số khung giờ cần dự đoán (mỗi khung giờ cách nhau 6 tiếng):', min_value=1, max_value=2000, value=1)

    if st.button('Dự đoán'):
        future_exog = exog.iloc[-forecast_steps:]  # Sử dụng các giá trị ngoại sinh trong tương lai
        future_predictions = results.get_forecast(steps=forecast_steps, exog=future_exog)
        future_index = pd.date_range(start=test_endog.index[-1], periods=forecast_steps + 1, freq='6h')[1:]
        future_forecast = future_predictions.predicted_mean
        future_forecast.index = future_index

        # Vẽ biểu đồ dự đoán tương lai
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(endog.index, endog, label='Historical Data')
        ax.plot(future_forecast.index, future_forecast, label='Future Predictions', color='red')
        ax.legend()
        st.pyplot(fig)
