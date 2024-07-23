import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Tạo giao diện với Streamlit
st.title('Dự báo dòng chảy bằng mô hình SARIMAX')

# Tải lên tệp CSV ban đầu
uploaded_file = st.file_uploader("Tải lên tệp CSV ban đầu", type=["csv"])

# Biến global lưu trữ dữ liệu
global df, endog, exog, train_endog, test_endog, train_exog, test_exog, model, results

# Biến để kiểm soát việc dự đoán
predict_clicked = False

if uploaded_file is not None:
    # Đọc dữ liệu từ file CSV và đặt tần suất cho chỉ số thời gian
    df = pd.read_csv(uploaded_file, parse_dates=['Time'])
    df.set_index('Time', inplace=True)

    # Hiển thị thông tin vận hành hồ cho dữ liệu ban đầu
    st.subheader('Thông tin vận hành hồ (dữ liệu ban đầu)')
    last_4_rows = df.tail(4).reset_index()  # Lấy 4 hàng cuối cùng và reset chỉ số
    last_4_rows = last_4_rows.rename(columns={
        'Time': 'Thời gian',
        'Dak_To_rain': 'Lượng mưa trạm Đăk Tô',
        'Dak_Mod_rain': 'Lượng mưa trạm Đăk Mod',
        'Dak_Glei_rain': 'Lượng mưa trạm Đăk Glei',
        'Sa_Thay_rain': 'Lượng mưa trạm Sa Thầy',
        'Dak_Mod_flow': 'Dòng chảy trạm Đăk Mod',
        'Dak_To_flow': 'Dòng chảy về hồ Pleikrong'
    })
    st.table(last_4_rows[['Thời gian', 'Lượng mưa trạm Đăk Tô', 'Lượng mưa trạm Đăk Mod', 'Lượng mưa trạm Đăk Glei', 'Lượng mưa trạm Sa Thầy', 'Dòng chảy trạm Đăk Mod', 'Dòng chảy về hồ Pleikrong']])
    # Kiểm tra và xử lý các giá trị NaN và vô hạn
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Chọn chuỗi thời gian cần dự báo (dòng chảy khu vực A)
    endog = df['Dak_To_flow']

    # Chọn các biến ngoại sinh (dòng chảy khu vực B và lượng mưa ở các khu vực A, B, C, D)
    exog = df[['Dak_Mod_flow', 'Dak_Mod_rain', 'Dak_To_rain', 'Dak_Glei_rain', 'Sa_Thay_rain']]

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
    train_size = int(len(endog) * 0.8)
    train_endog, test_endog = endog[:train_size], endog[train_size:]
    train_exog, test_exog = exog[:train_size], exog[train_size:]

    # Xác định các tham số của mô hình SARIMA
    p, d, q = 4, 1, 5
    P, D, Q, s = 2, 1, 2, 6  # Thiết lập chu kỳ mùa vụ là 6

    # Xây dựng mô hình SARIMAX trên toàn bộ dữ liệu ban đầu
    warnings.filterwarnings("ignore")
    model = sm.tsa.SARIMAX(endog, exog=exog, order=(p, d, q), seasonal_order=(P, D, Q, s))
    results = model.fit()

    # Nhập giá trị ngoại sinh cho mốc thời gian tiếp theo để dự đoán
    st.subheader('Nhập giá trị ngoại sinh cho khung giờ tiếp theo')
    input_data = {
        'Dak_Mod_flow': st.number_input('Dòng chảy trạm Đăk Mod', value=0.0),
        'Dak_Mod_rain': st.number_input('Lượng mưa trạm Đăk Mod', value=0.0),
        'Dak_To_rain': st.number_input('Lượng mưa trạm Đăk Tô', value=0.0),
        'Dak_Glei_rain': st.number_input('Lượng mưa trạm Đăk Glei', value=0.0),
        'Sa_Thay_rain': st.number_input('Lượng mưa trạm Sa Thầy', value=0.0)
    }

    # Thực hiện dự đoán cho khung giờ tiếp theo khi nhấn nút Dự đoán
    st.subheader('Dự báo cho khung giờ tiếp theo')
    if st.button('Dự đoán'):
        predict_clicked = True
        input_df = pd.DataFrame([input_data])
        future_predictions = results.get_forecast(steps=1, exog=input_df)
        future_forecast = future_predictions.predicted_mean

        # Hiển thị giá trị dự đoán
        st.write("Giá trị dự đoán cho khung giờ tiếp theo:")
        st.write(future_forecast.iloc[0])

        # Kiểm tra điều kiện cảnh báo
        last_value = endog.iloc[-1]  # Dòng chảy ở thời điểm cuối cùng trong dữ liệu
        threshold = last_value * 5.1  # Ngưỡng cảnh báo: ví dụ 110% so với dòng chảy cuối cùng

        if future_forecast.values[0] > threshold:
            st.warning('Cảnh báo: Dòng chảy nước dự đoán vượt ngưỡng so với thời gian phía trước.')
        else:
            st.info('Trạng thái bình thường: Dòng chảy nước dự đoán trong ngưỡng chấp nhận được.')

else:
    st.write("Vui lòng tải lên tệp CSV ban đầu để bắt đầu.")
