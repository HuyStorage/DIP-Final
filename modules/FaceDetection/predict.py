import numpy as np
import cv2
import joblib
import streamlit as st
import os
import tempfile
import io
from static.utils import *

@st.cache_resource(show_spinner=False)
def load_models():
    face_det_path = get_file_path("face_detection_yunet_2023mar.onnx")
    face_rec_path = get_file_path("face_recognition_sface_2021dec.onnx")
    svc_path = get_file_path("svc.pkl")

    recognizer = cv2.FaceRecognizerSF.create(face_rec_path, "")
    detector = cv2.FaceDetectorYN.create(
        face_det_path, "", (640, 640), 0.5, 0.3, 5000
    )
    detector.setInputSize([640, 640])
    svc_model = joblib.load(svc_path)

    return recognizer, detector, svc_model


recognizer, detector, svc = load_models()
mydict = ["HongNhung", "KhanhHuy", "KimLoi", "NhutAnh", "SyCuong"]


def checkValidFace(frame, face_box):
    face_align = recognizer.alignCrop(frame, face_box)
    face_feature = recognizer.feature(face_align)
    test_predict = svc.predict(face_feature)
    confidence = np.max(np.abs(svc.decision_function(face_feature)))
    return test_predict if confidence > 0.5 else None


def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for face in faces[1]:
            result = checkValidFace(input, face)
            color = (0, 255, 0) if result is not None else (0, 0, 255)
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(input, (coords[0], coords[1]),
                        (coords[0] + coords[2], coords[1] + coords[3]),
                        color, thickness)
            for i in range(5):
                cv2.circle(input, (coords[4+i*2], coords[5+i*2]), 2, (255, 255, 0), thickness)
    cv2.putText(input, f"FPS: {fps:.2f}", (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def process(capture, container, frame_skip=1):
    if isinstance(capture, int):
        cap = cv2.VideoCapture(capture)
    else:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(capture.read())
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)

    tm = cv2.TickMeter()
    cur_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = standardize_image(frame, (640, 640))
        tm.start()
        faces = detector.detect(frame)
        tm.stop()

        if cur_frame % frame_skip == 0:
            if faces[1] is not None:
                for face_box in faces[1]:
                    result = checkValidFace(frame, face_box)
                    name = mydict[result[0]] if result is not None else "Unknown"
                    color = (0, 255, 0) if result is not None else (0, 0, 255)
                    cv2.putText(frame, name,
                                (int(face_box[0]), int(face_box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            visualize(frame, faces, tm.getFPS())
            container.image(frame, channels="BGR")

        cur_frame += 1
    cap.release()
    if not isinstance(capture, int):
        os.unlink(tmp_path)


def app():
    def reset_display():
        uploaded_containter.empty()
        cam_container.empty()
        input_container.empty()
        video_container.empty()
        result_container.empty()
        img_container.empty()

    st.markdown("""
            <div class="center-text">
                <h2>✨ <span style="background: linear-gradient(90deg, #3f51b5, #2196f3);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-weight: bold;
                    margin-bottom: 20px;">Nhận dang khuôn mặt</span>
                </h2>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="center-text" style="margin-bottom: 15px">Nhận diện khuôn mặt từ webcam hoặc video, hiển thị tên đã huấn luyện hoặc "unknown".</div>', unsafe_allow_html=True)

    uploaded_containter = st.empty()
    cam_container = st.empty()
    cols = st.columns(2)
    input_container = cols[0].empty()
    video_container = cols[0].empty()
    result_container = cols[1].empty()
    img_container = cols[1].empty()

    selected_option = st.sidebar.selectbox(
        "Nguồn video",
        ["Realtime Video Capture", "Upload Video to Detect"],
        index=0,
    )

    if selected_option == "Realtime Video Capture":
        reset_display()
        process(0, cam_container)

    elif selected_option == "Upload Video to Detect":
        reset_display()
        uploaded_video = uploaded_containter.file_uploader(
            "Tải video lên", type=["mp4", "mov", "avi"]
        )
        default_image_path = "test/face_detection_video.mp4"
        if uploaded_video is None and default_image_path:
            with open(default_image_path, "rb") as f:
                uploaded_video = io.BytesIO(f.read()) 

        if uploaded_video:
            input_container.subheader("Input")
            video_container.video(uploaded_video)
            result_container.subheader("Kết quả")
            process(uploaded_video, img_container, frame_skip=5)


app()
