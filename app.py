import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

# Đọc dữ liệu
df = pd.read_csv('data-dak-to_2003-2009.csv', parse_dates=['Time'])
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

    # Xây dựng và huấn luyện mô hình SARIMAX
    warnings.filterwarnings("ignore")
    model = sm.tsa.SARIMAX(train_endog, exog=train_exog, order=(2, 1, 1), seasonal_order=(1, 1, 6, 6))
    model_fit = model.fit(disp=False)

    # Dự báo trên tập kiểm tra
    forecast = model_fit.forecast(steps=len(test_endog), exog=test_exog)

    # Đánh giá mô hình
    mse = mean_squared_error(test_endog, forecast)
    mae = mean_absolute_error(test_endog, forecast)
    st.write(f'Mean Squared Error: {mse}')
    st.write(f'Mean Absolute Error: {mae}')

    # Vẽ biểu đồ kết quả
    plt.figure(figsize=(12, 6))
    plt.plot(train_endog, label='Train')
    plt.plot(test_endog, label='Test')
    plt.plot(test_endog.index, forecast, label='Forecast')
    plt.legend()
    st.pyplot(plt)

    # Thêm tính năng dự đoán cho khung giờ tiếp theo
    st.subheader('Dự báo cho khung giờ tiếp theo')
    forecast_steps = st.number_input('Nhập số khung giờ cần dự đoán (mỗi khung giờ cách nhau 6 tiếng):', min_value=1, max_value=24, value=1)
    
    if st.button('Dự đoán'):
        future_forecast = model_fit.forecast(steps=forecast_steps, exog=test_exog[-forecast_steps:])
        future_dates = pd.date_range(start=test_endog.index[-1], periods=forecast_steps + 1, freq='6H')[1:]
        
        forecast_df = pd.DataFrame({'Dự báo': future_forecast}, index=future_dates)
        st.write(forecast_df)

        # Vẽ biểu đồ dự báo
        plt.figure(figsize=(12, 6))
        plt.plot(train_endog, label='Train')
        plt.plot(test_endog, label='Test')
        plt.plot(test_endog.index, forecast, label='Forecast')
        plt.plot(future_dates, future_forecast, label='Future Forecast', linestyle='--')
        plt.legend()
        st.pyplot(plt)
