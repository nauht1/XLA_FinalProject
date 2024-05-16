import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, models
import cv2
from PIL import Image
from streamlit import session_state as session
import module.Logo as logo

st.set_page_config(page_title="Nhận diện chữ số viết tay", page_icon="🔢", layout="wide")

st.markdown("# Nhận diện chữ số viết tay MNIST")

logo.add_logo()

OPTIMIZER = tf.keras.optimizers.Adam()

# load model
model_architecture = './mnist/digit_config.json'
model_weights = './mnist/digit.weights.h5'
model = models.model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
              metrics=["accuracy"])

# data: shuffled and split between train and test sets
(_, _), (X_test, _) = datasets.mnist.load_data()

# reshape
X_test = X_test.reshape((10000, 28, 28, 1))

def create_random_image():
    # generate 100 random integers between 0 and 9999
    index = np.random.randint(0, 9999, 100)
    sample = np.zeros((100, 28, 28, 1))
    for i in range(100):
        sample[i] = X_test[index[i]]

    image = np.zeros((280, 280), dtype=np.uint8)
    k = 0
    for i in range(10):
        for j in range(10):
            image[i*28:(i+1)*28, j*28:(j+1)*28] = sample[k, :, :, 0]
            k += 1
    color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(color_converted)
    return image_pil, sample

def predict_image(data):
    prediction = model.predict(data, verbose=0)
    result = prediction.argmax(axis=1)
    return result.reshape(10, 10)

if st.button('Tạo ảnh ngẫu nhiên'):
    session.current_image_pil, session.current_data = create_random_image()
    st.image(session.current_image_pil, caption='Ảnh ngẫu nhiên', width=400)

if 'current_image_pil' not in session:
    session.current_image_pil, session.current_data = create_random_image()
    st.image(session.current_image_pil, caption='Ảnh ngẫu nhiên', width=400)

if st.button('Dự đoán'):
    result = predict_image(session.current_data)
    st.image(session.current_image_pil, caption='Ảnh ngẫu nhiên', width=400)
    # Chuyển mảng kết quả thành chuỗi
    result_str = "\n".join([" ".join(map(str, row)) for row in result])
    # In chuỗi kết quả ra màn hình
    st.text(result_str)
