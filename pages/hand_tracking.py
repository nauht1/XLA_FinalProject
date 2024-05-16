import cv2
import mediapipe as mp
import streamlit as st
from PIL import Image
import numpy as np
import module.Logo as logo

st.set_page_config(page_title="Đếm số ngón tay", page_icon="🙌", layout="wide")

st.markdown("# Đếm số ngón tay")

logo.add_logo()

mp_drawing_util = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_hand = mp.solutions.hands

hands = mp_hand.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def is_finger_up(hand_landmarks, base_id, tip_id):
    return hand_landmarks[tip_id].y < hand_landmarks[base_id].y

def count_fingers(hand_landmarks, handedness):
    fingers = [
        is_finger_up(hand_landmarks, 5, 8),   # ngón trỏ
        is_finger_up(hand_landmarks, 9, 12),  # ngón giữa
        is_finger_up(hand_landmarks, 13, 16), # ngón áp út
        is_finger_up(hand_landmarks, 17, 20)  # ngón út
    ]

    thumb_up = False
    if handedness == "Right":
        thumb_up = hand_landmarks[4].x < hand_landmarks[3].x
    else:
        thumb_up = hand_landmarks[4].x > hand_landmarks[3].x

    fingers.append(thumb_up)
    return sum(fingers)

run = st.checkbox('Turn on webcam')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame)

    count = 0
    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_drawing_util.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hand.HAND_CONNECTIONS,
                mp_drawing_style.get_default_hand_landmarks_style(),
                mp_drawing_style.get_default_hand_connections_style()
            )
            handedness = result.multi_handedness[idx].classification[0].label
            count += count_fingers(hand_landmarks.landmark, handedness)

    cv2.putText(frame, f'Fingers: {count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')

camera.release()
