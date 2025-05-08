import streamlit as st
import cv2
from PIL import Image
from image_processing import Chapter03, Chapter04, Chapter09
import io
from static.utils import *

inpWidth, inpHeight = 640, 640

chapter_options = {
        "Chương 3: Chuyển đổi cường độ và lọc không gian": {
        "1. Negative": {"function": Chapter03.Negative, "description": "Biến đổi âm bản cho ảnh xám bằng cách đảo ngược mức độ sáng, giúp chuyển các vùng sáng thành tối và ngược lại.", "image": "test/Chuong3/1_Negative_Image.tif"},
        "2. Negative Color": {"function": Chapter03.NegativeColor, "description": "Biến đổi âm bản cho ảnh màu bằng cách đảo ngược độ sáng của từng kênh màu.", "image": "test/Chuong3/2_Negative_Color.tif"},
        "3. Logarit": {"function": Chapter03.Logarit, "description": "Biến đổi ảnh xám bằng hàm logarit nhằm tăng cường độ sáng ở các vùng tối và nén các giá trị ở vùng sáng.", "image": "test/Chuong3/3_Logarit.tif"},
        "4. Gamma": {"function": Chapter03.Power, "description": "Biến đổi độ sáng ảnh bằng hàm mũ (gamma), dùng để làm sáng hoặc làm tối ảnh tùy theo giá trị gamma.", "image": "test/Chuong3/4_Gamma.tif"},
        "5. Piecewise Linear": {"function": Chapter03.PiecewiseLinear, "description": "Biến đổi ảnh xám theo dạng tuyến tính từng đoạn, dựa trên giá trị mức sáng nhỏ nhất và lớn nhất trong ảnh. Phương pháp này giúp tăng độ tương phản cục bộ bằng cách giãn hoặc nén các dải cường độ sáng cụ thể.", "image": "test/Chuong3/5_PiecewiseLinear.jpg"},
        "6. Histogram": {"function": Chapter03.Histogram, "description": "Vẽ biểu đồ mức xám để phân tích sự phân bố độ sáng của ảnh.", "image": "test/Chuong3/6_Histogram_1.tif"},
        "7. Hist Equal": {"function": Chapter03.HistEqual, "description": "Cân bằng histogram để tăng độ tương phản của ảnh bằng cách phân phối lại các mức sáng.", "image": "test/Chuong3/7_Histogram_Equal.png"},
        "8. Hist Equal Color": {"function": Chapter03.HistEqualColor, "description": "Cân bằng histogram riêng biệt cho từng kênh màu để tăng độ tương phản ảnh màu.", "image": "test/Chuong3/8_Histogram_Equal_Color.tif"},
        "9. Local Hist": {"function": Chapter03.LocalHist, "description": "Cân bằng histogram cục bộ theo từng vùng nhỏ để cải thiện độ tương phản tại các khu vực có ánh sáng kém.", "image": "test/Chuong3/9_Local_Histogram.tif"},
        "10. Hist Stat": {"function": Chapter03.HistStat, "description": "Tăng cường độ tương phản dựa trên thống kê cục bộ về trung bình và độ lệch chuẩn.", "image": "test/Chuong3/10_Histogram_statistic.tif"},
        "11. Smooth box": {"function": Chapter03.BoxFilter, "description": "Làm mịn ảnh bằng bộ lọc trung bình (box filter) giúp giảm nhiễu và làm mờ toàn cục.", "image": "test/Chuong3/11_Smooth_box.tif"},
        "12. Smooth gauss": {"function": Chapter03.LowpassGauss, "description": "Làm mịn ảnh bằng bộ lọc Gaussian giúp làm mờ ảnh một cách mượt mà hơn so với bộ lọc trung bình.", "image": "test/Chuong3/12_Smooth_gauss.tif"},
        "13. Median filter": {"function": Chapter03.MedianFilter, "description": "Lọc nhiễu ảnh bằng bộ lọc trung vị (median filter), hiệu quả đặc biệt trong việc loại bỏ nhiễu muối tiêu (salt-and-pepper noise) mà vẫn giữ được biên ảnh rõ nét.", "image": "test/Chuong3/13_Median_filter.tif"},
        "14. Sharpening": {"function": Chapter03.Sharpen, "description": "Làm sắc nét ảnh bằng cách sử dụng bộ lọc Laplacian để phát hiện biên và trừ đi phần biên này khỏi ảnh gốc, giúp tăng độ tương phản tại các đường biên.", "image": "test/Chuong3/14_Sharpening.tif"},
        "15. Sharpening mask": {"function": Chapter03.HistStat, "description": "Làm sắc nét ảnh bằng kỹ thuật mask sharpening. Hàm sử dụng bộ lọc Gaussian để làm mờ ảnh, sau đó tính phần sai khác (mask) giữa ảnh gốc và ảnh đã làm mờ. Cuối cùng, mask này được khuếch đại và cộng ngược lại vào ảnh gốc để tăng độ sắc nét.", "image": "test/Chuong3/15_Sharpening_mask.tif"},
        "16. Gradient": {"function": Chapter03.Gradient, "description": "Gradient Detection", "image": "test/Chuong3/16_Gradient.tif"}
    },
    "Chương 4: Lọc trong miền tần số": {
        "1. Spectrum": {"function": Chapter04.Spectrum, "description": "Tính toán và hiển thị phổ tần số của ảnh bằng cách sử dụng biến đổi Fourier.", "image": "test/Chuong4/1_Spectrum.tif"},
        "2. Remove moire": {"function": Chapter04.RemoveMoire, "description": "Khử nhiễu Moire bằng cách sử dụng biến đổi Fourier và lọc tần số.", "image": "test/Chuong4/2_Remove_moire.tif"},
        "3. Remove inter inference": {"function": Chapter04.RemoveInterInference, "description": "Khử nhiễu giao thoa bằng bộ lọc notch trong miền tần số.", "image": "test/Chuong4/3_Remove_interference.tif"},
        "4. Create motion": {"function": Chapter04.CreateMotion, "description": "Tạo hiệu ứng chuyển động.", "image": "test/Chuong4/4_Create_motion.tif"},
        "5. Demotion": {"function": Chapter04.Demotion, "description": "Khử nhiễu mờ chuyển động bằng bộ lọc tần số dựa trên mô hình chuyển động tuyến tính.", "image": "test/Chuong4/5_Demotion.tif"},
        "6. Demotion noise": {"function": Chapter04.DemotionNoise, "description": "Khử mờ chuyển động kết hợp lọc nhiễu bằng median blur và biến đổi Fourier.", "image": "test/Chuong4/6_Demotion_noise.tif"},
    },
    "Chương 9: Xử lý hình ảnh hình thái": {
        "1. Erosion": {"function": Chapter09.Erosion, "description": "Phép co ảnh giúp loại bỏ các chi tiết nhỏ và làm mờ các cạnh.", "image": "test/Chuong9/1_Erosion.tif"},
        "2. Dilation": {"function": Chapter09.Dilation, "description": "Phép phát hiện biên giúp xác định các đường biên của các đối tượng trong ảnh.", "image": "test/Chuong9/2_Dilation.tif"},
        "3. Boundary": {"function": Chapter09.BoundaryExtraction, "description": "Phép phát hiện biên giúp xác định các đường biên của các đối tượng trong ảnh.", "image": "test/Chuong9/3_Boundary.tif"},
        "4. Contour": {"function": Chapter09.Contour, "description": "Phép phát hiện biên giúp xác định các đường biên của các đối tượng trong ảnh.", "image": "test/Chuong9/6_Remove_Small_Rice.tif"}
    }
}

def app():
    st.markdown("""
                <div class="center-text">
                    <h2>✨ <span style="background: linear-gradient(90deg, #3f51b5, #2196f3);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        font-weight: bold;
                        margin-bottom: 20px;">Xử lý ảnh số</span>
                    </h2>
                </div>
            """, unsafe_allow_html=True)

    selected_chapter = st.sidebar.selectbox("📘 Chọn chương", list(chapter_options.keys()))
    lesson_options = list(chapter_options[selected_chapter].keys())
    selected_lesson = st.sidebar.selectbox("🧪 Chọn bài học", lesson_options)
    selected_function = chapter_options[selected_chapter][selected_lesson]["function"]
    default_image_path = get_path(chapter_options[selected_chapter][selected_lesson]["image"])

    st.markdown("Mô tả: " + chapter_options[selected_chapter][selected_lesson]["description"])

    upload_image = st.file_uploader(
        "Choose image", type=["bmp", "png", "jpg", "jpeg", "tif", "gif"]
    )
    # set default image if no image is uploaded
    if upload_image is None and default_image_path:
        with open(default_image_path, "rb") as f:
            upload_image = io.BytesIO(f.read()) 

    cols = st.columns(2)
    with cols[0]:
        input_container = st.empty()
        imagein_container = st.empty()

    with cols[1]:
        result_container = st.empty()
        imageout_container = st.empty()

    if upload_image is not None:
        input_container.subheader("Input")
        image = Image.open(upload_image)
        frame = np.array(image)

        imagein_container.image(frame)

        if st.sidebar.button("Process"):
            result_container.subheader("Result")
            try:
                result = selected_function(frame)
                imageout_container.image(result)
            except Exception as e:
                st.error(f"Lỗi khi xử lý ảnh: {e}")

app()
