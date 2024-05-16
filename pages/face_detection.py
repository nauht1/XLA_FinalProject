import streamlit as st
import cv2 as cv
from PIL import Image
import time
import tempfile
from module.facedetection import detect_faces, visualize, load_models

st.set_page_config(page_title="Nhận diện khuôn mặt", page_icon="😃", layout="wide")
st.markdown("# Nhận diện khuôn mặt")

face_detection_model_path = './models/face_detection_yunet_2023mar.onnx'
face_recognition_model_path = './models/face_recognition_sface_2021dec.onnx'
svc_path = './models/svc.pkl'

try:
    detector, recognizer, svc, mydict = load_models(face_detection_model_path, face_recognition_model_path, svc_path)
except Exception as e:
    st.error(str(e))
    st.stop()

start_button = st.button("Turn On Webcam")
stop_button = st.button("Turn Off Webcam")
uploaded_video = st.file_uploader("Upload a Video", type=["mp4"])

col1, col2 = st.columns(2)
frame_placeholder_original = col1.empty()
frame_placeholder_processed = col2.empty()

webcam_active = False

if start_button:
    webcam_active = True

if stop_button:
    webcam_active = False

if webcam_active:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        st.write("Couldn't open webcam.")
    else:
        prev_time = time.time()
        while webcam_active:
            ret, frame = cap.read()
            if not ret:
                st.write("Couldn't retrieve frame.")
                break
            img_ori = Image.fromarray(frame[..., ::-1])
            frame_placeholder_original.image(img_ori, channels="RGB")
            
            faces, names = detect_faces(frame, detector, recognizer, svc, mydict)
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time
            visualize(frame, faces, names, fps)
            
            img = Image.fromarray(frame[..., ::-1])
            frame_placeholder_processed.image(img, channels="RGB")
        cap.release()
        cv.destroyAllWindows()

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    vid = cv.VideoCapture(tfile.name)
    if not vid.isOpened():
        st.write("Couldn't open video.")
    else:
        prev_time = time.time()
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break
            frame = cv.resize(frame, (720,1280))
            
            img_original = Image.fromarray(frame[..., ::-1])
            frame_placeholder_original.image(img_original, channels="RGB")
            
            faces, names = detect_faces(frame, detector, recognizer, svc, mydict)
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time
            visualize(frame, faces, names, fps)
            img_processed = Image.fromarray(frame[..., ::-1])
            frame_placeholder_processed.image(img_processed, channels="RGB")
    
            del frame, faces, names
        vid.release()
        cv.destroyAllWindows()
