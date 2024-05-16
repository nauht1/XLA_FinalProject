import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile
import module.vehicleprocessing as detection
import module.Logo as logo
st.set_page_config(page_title="Nhận diện biển số xe", page_icon="🚗", layout="wide")

st.markdown("# Nhận diện biển số xe")
logo.add_logo()

uploaded_file = st.file_uploader("Tải lên hình ảnh hoặc video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        img = Image.open(uploaded_file)
        img_array = np.array(img)

        results = detection.yolo_predictions(img_array)
        
        st.image(results, caption='Kết quả nhận diện biển số xe', use_column_width=True)

    elif uploaded_file.type == "video/mp4":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = detection.yolo_predictions(frame)
            stframe.image(results, channels="BGR")
        
        cap.release()
        tfile.close()
