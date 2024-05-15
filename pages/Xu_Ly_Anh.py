import streamlit as st
from PIL import Image
import numpy as np
import cv2
import module.Chapter3 as c3
import module.Chapter4 as c4
import module.Chapter5 as c5
import module.Chapter9 as c9

st.set_page_config(page_title="Xử lý ảnh", page_icon="😃", layout="wide")

st.markdown("# Xử lý ảnh")

# Các tùy chọn xử lý ảnh
options = [
    "Negative", 
    "Logarithm", 
    "PiecewiseLinear", 
    "Histogram", 
    "HistEqual", 
    "HistEqualColor", 
    "LocalHist", 
    "HistStat", 
    "BoxFilter", 
    "LowpassGauss", 
    "Threshold", 
    "MedianFilter", 
    "Sharpen", 
    "Gradient", 
    "Spectrum", 
    "FrequencyFilter", 
    "DrawNotchRejectFilter", 
    "RemoveMoire", 
    "CreateMotionNoise", 
    "DenoiseMotion", 
    "DenoisestMotion",
    "ConnectedComponent", 
    "CountRice"
]

# Tạo selection box để chọn chức năng xử lý ảnh
selected_option = st.selectbox("Chọn chức năng xử lý ảnh", options)

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
        if selected_option == "Negative":
            processed_img = c3.Negative(img_array)

        elif selected_option == "Logarithm":
            processed_img = c3.Logarithm(img_array)

        elif selected_option == "PiecewiseLinear":
            processed_img = c3.PiecewiseLinear(img_array)

        elif selected_option == "Histogram":
            processed_img = c3.Histogram(img_array)

        elif selected_option == "HistEqual":
            processed_img = c3.HistEqual(img_array)

        elif selected_option == "HistEqualColor":
            processed_img = c3.HistEqualColor(img_array)

        elif selected_option == "LocalHist":
            processed_img = c3.LocalHist(img_array)

        elif selected_option == "HistStat":
            processed_img = c3.HistStat(img_array)

        elif selected_option == "BoxFilter":
            processed_img = c3.BoxFilter(img_array)

        elif selected_option == "LowpassGauss":
            processed_img = c3.LowpassGauss(img_array)

        elif selected_option == "Threshold":
            processed_img = c3.Threshold(img_array)

        elif selected_option == "MedianFilter":
            processed_img = c3.MedianFilter(img_array)

        elif selected_option == "Sharpen":
            processed_img = c3.Sharpen(img_array)

        elif selected_option == "Gradient":
            processed_img = c3.Gradient(img_array)

        elif selected_option == "Spectrum":
            processed_img = c4.Spectrum(img_array)

        elif selected_option == "FrequencyFilter":
            processed_img = c4.FrequencyFilter(img_array)

        elif selected_option == "DrawNotchRejectFilter":
            processed_img = c4.DrawNotchRejectFilter(img_array)

        elif selected_option == "RemoveMoire":
            processed_img = c4.RemoveMoire(img_array)

        elif selected_option == "CreateMotionNoise":
            processed_img = c5.CreateMotionNoise(img_array)

        elif selected_option == "DenoiseMotion":
            processed_img = c5.DenoiseMotion(img_array)

        elif selected_option == "DenoisestMotion":
            processed_img = c5.DenoisestMotion(img_array)

        elif selected_option == "ConnectedComponent":
            processed_img = c9.ConnectedComponent(img_array)
        
        elif selected_option == "CountRice":
            processed_img = c9.CountRice(img_array)

        # Hiển thị ảnh sau khi xử lý (nếu có)
        if processed_img is not None:
            col2.image(processed_img, caption='Ảnh đã xử lý', width=400)
else:
    if process_button:
        st.warning("Vui lòng chọn một ảnh trước khi nhấn 'Xử lý'.")
