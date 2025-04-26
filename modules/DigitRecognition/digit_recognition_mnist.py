import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets, models, optimizers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import Sequential
import os
import numpy as np
import random

# Hàm tạo ảnh ngẫu nhiên
def tao_anh_ngau_nhien():
    if 'X_test' not in st.session_state:
        st.error("Dữ liệu X_test chưa được tải. Hãy thử lại sau.")
        return None, None

    image = np.zeros((10*28, 10*28), np.uint8)
    data = np.zeros((100, 28, 28, 1), np.uint8)

    for i in range(100):
        n = random.randint(0, 9999)
        sample = st.session_state.X_test[n]
        data[i] = st.session_state.X_test[n]
        x = i // 10
        y = i % 10
        image[x*28:(x+1)*28, y*28:(y+1)*28] = sample[:, :, 0]
    return image, data

# Load model và dữ liệu nếu chưa có
if 'is_load' not in st.session_state:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "models")
        model_architecture = os.path.join(model_dir, 'digit_config.json')
        model_weights = os.path.join(model_dir, 'digit_weight.h5')

        with open(model_architecture, 'r') as f:
            model_json = f.read()

        model = model_from_json(model_json, custom_objects={
            'Sequential': Sequential,
            'InputLayer': InputLayer,
            'Conv2D': Conv2D,
            'MaxPooling2D': MaxPooling2D,
            'Flatten': Flatten,
            'Dense': Dense
        })
        model.load_weights(model_weights)

        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
        st.session_state.model = model

        (_, _), (X_test, y_test) = datasets.mnist.load_data()
        X_test = X_test.reshape((10000, 28, 28, 1))
        st.session_state.X_test = X_test
        st.session_state.is_load = True
        print('Lần đầu load model và data')
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
else:
    print('Đã load model và data rồi')

st.markdown('<div class="center-text"><h2>✨ Nhận dạng chữ số (MNIST)</h2></div>', unsafe_allow_html=True)
st.markdown('<div class="center-text" style="margin-bottom: 15px">Ứng dụng sử dụng mô hình học sâu để nhận dạng 100 chữ số viết tay ngẫu nhiên từ tập MNIST</div>', unsafe_allow_html=True)

# Chia layout thành 2 cột
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Tạo ảnh")
    if st.button('Tạo ảnh'):
        if 'X_test' not in st.session_state:
            st.warning("Dữ liệu X_test chưa được tải. Hãy thử lại sau.")
        else:
            image, data = tao_anh_ngau_nhien()
            if image is not None:
                st.session_state.image = image
                st.session_state.data = data

    if 'image' in st.session_state:
        st.image(st.session_state.image, use_container_width=True)

with col2:
    st.subheader("Nhận dạng")
    if 'image' in st.session_state:
        if st.button('Nhận dạng'):
            data = st.session_state.data
            data = data / 255.0
            data = data.astype('float32')
            ket_qua = st.session_state.model.predict(data)
            s = ''
            for i, x in enumerate(ket_qua):
                s += f'<span style="font-size:32px; margin: 8px"><b>{np.argmax(x)}</b></span> '
                if (i + 1) % 10 == 0:
                    s += '<br>'
            st.markdown(s, unsafe_allow_html=True)
    else:
        st.info("Hãy tạo ảnh trước khi nhận dạng.")

