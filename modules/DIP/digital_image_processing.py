import streamlit as st
import cv2
from PIL import Image
from image_processing import Chapter03, Chapter04, Chapter09
from static.utils import *

inpWidth, inpHeight = 640, 640

chapter_options = {
    "Chương 3: Chỉnh sửa ảnh cơ bản": {
        "1. Negative": Chapter03.Negative,
        "2. Logarit": Chapter03.Logarit,
        "3. Power": Chapter03.Power,
        "4. Piecewise Linear": Chapter03.PiecewiseLinear,
        "5. Histogram": Chapter03.Histogram,
        "6. Hist Equal": Chapter03.HistEqual,
        "7. Hist Equal Color": Chapter03.HistEqualColor,
        "8. Local Hist": Chapter03.LocalHist,
        "9. Hist Stat": Chapter03.HistStat,
        "10. Box Filter": Chapter03.BoxFilter,
        "11. Lowpass Gauss": Chapter03.LowpassGauss,
        "12. Threshold": Chapter03.Threshold,
        "13. Median Filter": Chapter03.MedianFilter,
        "14. Sharpen": Chapter03.Sharpen,
        "15. Gradient": Chapter03.Gradient,
    },
    "Chương 4: Biến đổi ảnh": {
        "1. Spectrum": Chapter04.Spectrum,
        "2. Frequency Filter": Chapter04.FrequencyFilter,
        "3. Draw Notch Reject Filter": Chapter04.DrawNotchRejectFilter,
        "4. Remove Moire": Chapter04.RemoveMoire,
    },
    "Chương 9: Nhận dạng và phân loại ảnh": {
        "1. Connected Component": Chapter09.ConnectedComponent,
        "2. Count Rice": Chapter09.CountRice,
    }
}

def app():
    st.title("🖼️​ Image Processing")

    selected_chapter = st.sidebar.selectbox("📘 Chọn chương", list(chapter_options.keys()))
    lesson_options = list(chapter_options[selected_chapter].keys())
    selected_lesson = st.sidebar.selectbox("🧪 Chọn bài học", lesson_options)
    selected_function = chapter_options[selected_chapter][selected_lesson]

    upload_image = st.file_uploader(
        "Choose image", type=["bmp", "png", "jpg", "jpeg", "tif", "gif"]
    )
    cols = st.columns(2)
    with cols[0]:
        input_container = st.empty()
        imagein_container = st.empty()

    with cols[1]:
        result_container = st.empty()
        imageout_container = st.empty()

    if selected_lesson.endswith("Draw Notch Reject Filter"):
        if st.sidebar.button("Process"):
            result = selected_function()
            input_container.subheader("Result")
            imagein_container.image(
                standardize_image_gray(result, (inpWidth, inpHeight))
            )

    elif upload_image is not None:
        input_container.subheader("Input")
        image = Image.open(upload_image)
        frame = np.array(image)

        if selected_lesson.endswith("Hist Equal Color"):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = standardize_image(frame, (inpWidth, inpHeight))
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            process_img = frame
            frame = standardize_image_gray(frame, (inpWidth, inpHeight))

        imagein_container.image(frame)

        if st.sidebar.button("Process"):
            result_container.subheader("Result")
            if selected_lesson.endswith("Remove Moire"):
                result = selected_function(process_img)
                imageout_container.image(
                    standardize_image_gray(result, (inpWidth, inpHeight))
                )
            else:
                result = selected_function(frame)
                imageout_container.image(result)

app()
