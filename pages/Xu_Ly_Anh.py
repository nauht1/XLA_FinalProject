import streamlit as st
from PIL import Image
import numpy as np
import cv2
import module.Chapter3 as c3
import module.Chapter4 as c4
import module.Chapter5 as c5
import module.Chapter9 as c9
import module.Logo as logo

st.set_page_config(page_title="Xử lý ảnh", page_icon="😃", layout="wide")

sidebar_selection_box = st.sidebar.selectbox(
    "Chọn chương",
    ("Chương 3", "Chương 4", "Chương 5", "Chương 9")
)

logo.add_logo()

st.markdown("# Xử lý ảnh")

# Các tùy chọn xử lý ảnh
c3_options = [
    "1. Negative",
    "2. Power",
    "3. Logarithm", 
    "4. PiecewiseLinear - Biến đổi tuyến tính", 
    "5. Histogram", 
    "6. HistEqual - Cân bằng histogram", 
    "7. HistEqualColor - Cân bằng histogram - ảnh màu", 
    "8. LocalHist", 
    "9. HistStat", 
    "10. BoxFilter - Làm mờ ảnh bằng Box", 
    "11. LowpassGauss - Lọc Gaussian thấp tần số để làm mờ ảnh", 
    "12. Threshold - Phân ngưỡng", 
    "13. MedianFilter - Làm mờ ảnh và loại bỏ nhiễu bằng Median", 
    "14. Sharpen - Tăng độ nét của ảnh bằng cách làm nổi bật các biên", 
    "15. Gradient - Tính gradient để phát hiện biên của đối tượng"
]

c4_options = [
    "1. Spectrum", 
    "2. FrequencyFilter", 
    "3. DrawNotchRejectFilter", 
    "4. RemoveMoire"
]

c5_options = [
    "1. CreateMotionNoise - Tạo nhiễu chuyển động", 
    "2. DenoiseMotion - Loại bỏ nhiễu", 
    "3. DenoisestMotion - Loại bỏ nhiễu chuyển động mạnh"
]

c9_options = [
    "1. ConnectedComponent - Đếm thành phần liên thông", 
    "2. CountRice - Đếm gạo"
]

# Tạo selection box để chọn chức năng xử lý ảnh
if (sidebar_selection_box == "Chương 3"):
    selected_option = st.selectbox("Chọn chức năng xử lý ảnh", c3_options)

elif (sidebar_selection_box == "Chương 4"):
    selected_option = st.selectbox("Chọn chức năng xử lý ảnh", c4_options)

elif (sidebar_selection_box == "Chương 5"):
    selected_option = st.selectbox("Chọn chức năng xử lý ảnh", c5_options)

elif (sidebar_selection_box == "Chương 9"):
    selected_option = st.selectbox("Chọn chức năng xử lý ảnh", c9_options)

# Nút Open Image để chọn ảnh và hiển thị bản xem trước
uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png", "tif"])
process_button = st.button("Xử lý")

L = 256

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Sử dụng cột để hiển thị ảnh gốc và ảnh đã xử lý trên cùng một hàng
    col1, col2 = st.columns(2)
    col1.image(img_array, caption='Ảnh gốc', width=400)

    # Nút "Xử lý" để xử lý ảnh
    if process_button:
        # Xử lý ảnh theo chức năng đã chọn
        processed_img = None
        if selected_option == c3_options[0]:
            processed_img = c3.Negative(img_array)
        
        elif selected_option == c3_options[1]:
            processed_img = c3.Power(img_array)

        elif selected_option == c3_options[2]:
            processed_img = c3.Logarithm(img_array)

        elif selected_option == c3_options[3]:
            processed_img = c3.PiecewiseLinear(img_array)

        elif selected_option == c3_options[4]:
            processed_img = c3.Histogram(img_array)

        elif selected_option == c3_options[5]:
            processed_img = c3.HistEqual(img_array)

        elif selected_option == c3_options[6]:
            processed_img = c3.HistEqualColor(img_array)

        elif selected_option == c3_options[7]:
            processed_img = c3.LocalHist(img_array)

        elif selected_option == c3_options[8]:
            processed_img = c3.HistStat(img_array)

        elif selected_option == c3_options[9]:
            processed_img = c3.BoxFilter(img_array)

        elif selected_option == c3_options[10]:
            processed_img = c3.LowpassGauss(img_array)

        elif selected_option == c3_options[11]:
            processed_img = c3.Threshold(img_array)

        elif selected_option == c3_options[12]:
            processed_img = c3.MedianFilter(img_array)

        elif selected_option == c3_options[13]:
            processed_img = c3.Sharpen(img_array)

        elif selected_option == c3_options[14]:
            processed_img = c3.Gradient(img_array)

        elif selected_option == c4_options[0]:
            processed_img = c4.Spectrum(img_array)

        elif selected_option == c4_options[1]:
            processed_img = c4.FrequencyFilter(img_array)

        elif selected_option == c4_options[2]:
            processed_img = c4.DrawNotchRejectFilter(img_array)

        elif selected_option == c4_options[3]:
            processed_img = c4.RemoveMoire(img_array)

        elif selected_option == c5_options[0]:
            processed_img = c5.CreateMotionNoise(img_array)

        elif selected_option == c5_options[1]:
            processed_img = c5.DenoiseMotion(img_array)

        elif selected_option == c5_options[2]:
            processed_img = c5.DenoisestMotion(img_array)

        elif selected_option == c9_options[0]:
            processed_img = c9.ConnectedComponent(img_array)
        
        elif selected_option == c9_options[1]:
            processed_img = c9.CountRice(img_array)

        # Hiển thị ảnh sau khi xử lý (nếu có)
        if processed_img is not None:
            col2.image(processed_img, caption='Ảnh đã xử lý', width=400)
else:
    if process_button:
        st.warning("Vui lòng chọn một ảnh trước khi nhấn 'Xử lý'.")
