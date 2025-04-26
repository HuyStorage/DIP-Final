import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
from static.utils import *

# Constants
inpWidth, inpHeight = 640, 640
confThreshold, nmsThreshold = 0.5, 0.4
models = {
    "Động vật": ("animal_detection.onnx", "animal.txt"),
    "Trái cây": ("trai_cay.onnx", "trai_cay.txt")
}
selected_model = st.sidebar.selectbox("🔍 Chọn mô hình", list(models.keys()))
model_path, class_file = models[selected_model]
model = get_file_path(model_path)
filename_classes = get_file_path(class_file)

if "Net" not in st.session_state or st.session_state.get("current_model") != model_path:
    st.session_state["Net"] = cv2.dnn.readNet(model)
    st.session_state["current_model"] = model_path

with open(filename_classes, "rt") as f:
    classes = f.read().rstrip("\n").split("\n")

selected_classes = st.sidebar.multiselect("🎯 Chọn lớp cần nhận diện", classes, default=classes[:3])
selected_class_ids = [classes.index(c) for c in selected_classes]

def process_output(out, box_scale, selected_class_ids):
    detections = []
    for detection in out.transpose(1, 0):
        scores = detection[4:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        if confidence > confThreshold and classId in selected_class_ids:
            center_x, center_y, width, height = detection[:4]
            left = int((center_x - width / 2) * box_scale[0])
            top = int((center_y - height / 2) * box_scale[1])
            width = int(width * box_scale[0])
            height = int(height * box_scale[1])
            detections.append((classId, confidence, [left, top, width, height]))
    return detections

def postprocess(frame, outs, selected_class_ids):
    frameHeight, frameWidth = frame.shape[:2]
    box_scale = (frameWidth / inpWidth, frameHeight / inpHeight)

    detections = []
    for out in outs:
        detections.extend(process_output(out[0], box_scale, selected_class_ids))

    if detections:
        classIds, confidences, boxes = zip(*detections)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

        for i in indices.flatten():
            box = boxes[i]
            left, top, width, height = box
            drawPred(
                frame,
                classIds[i],
                confidences[i],
                left,
                top,
                left + width,
                top + height,
            )

def drawPred(frame, classId, conf, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
    label = f"{classes[classId]}: {conf:.2f}"
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(
        frame,
        (left, top - labelSize[1]),
        (left + labelSize[0], top + baseLine),
        (255, 255, 255),
        cv2.FILLED,
    )
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

def app():
    st.title("🍎​ Object Detection")
    st.write("This program detects 5 types of animals: capybara, hedgehog, kangaroo, panda, penguin.")
    st.sidebar.write("Browse an image, then click the 'Predict' button for detection.")

    img_file_buffer = st.file_uploader("Choose image", type=["bmp", "png", "jpg", "jpeg", "tif", "gif"])

    cols = st.columns(2)

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = standardize_image(frame, (inpWidth, inpHeight))

        with cols[0]:
            st.subheader("Input")
            st.image(frame, channels="BGR")

        with cols[1]:
            text_container = st.empty()
            img_container = st.empty()

        if st.sidebar.button("Predict"):
            blob = cv2.dnn.blobFromImage(
                frame, size=(inpWidth, inpHeight), swapRB=True, ddepth=cv2.CV_8U
            )
            st.session_state["Net"].setInput(blob, scalefactor=0.00392)
            outs = st.session_state["Net"].forward(
                st.session_state["Net"].getUnconnectedOutLayersNames()
            )
            postprocess(frame, outs, selected_class_ids)
            text_container.subheader("Result")
            img_container.image(frame, channels="BGR")

app()
