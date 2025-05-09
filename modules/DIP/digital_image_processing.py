import streamlit as st
import cv2
from PIL import Image
from image_processing import Chapter03, Chapter04, Chapter09
import io
from static.utils import *

inpWidth, inpHeight = 640, 640

CHAPTER_OPTIONS = {
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
        "14. Sharpening": {"function": Chapter03.Sharpen, "description": "Làm sắc nét ảnh bằng cách sử dụng bộ lọc Laplacian để làm nổi bật các vùng biên, sau đó trừ thành phần biên này khỏi ảnh gốc để tăng độ tương phản tại các cạnh.", "image": "test/Chuong3/14_Sharpening.tif"},
        "15. Sharpening mask": {"function": Chapter03.SharpeningMask, "description": "Tăng độ sắc nét ảnh bằng kỹ thuật mask sharpening: làm mờ ảnh bằng bộ lọc Gaussian, sau đó khuếch đại phần sai khác giữa ảnh gốc và ảnh làm mờ để nhấn mạnh chi tiết và biên ảnh.", "image": "test/Chuong3/15_Sharpening_mask.tif"},
        "16. Gradient": {"function": Chapter03.Gradient, "description": "Gradient Detection", "image": "test/Chuong3/16_Gradient.tif"}
    },
    "Chương 4: Lọc trong miền tần số": {
        "1. Spectrum": {"function": Chapter04.Spectrum, "description": "Tính và hiển thị phổ biên độ (spectrum) của ảnh bằng biến đổi Fourier, giúp phân tích thành phần tần số và cấu trúc không gian trong ảnh.", "image": "test/Chuong4/1_Spectrum.tif"},
        "2. Remove moire": {"function": Chapter04.RemoveMoire, "description": "Khử nhiễu Moire bằng cách sử dụng biến đổi Fourier và lọc tần số.", "image": "test/Chuong4/2_Remove_moire.tif"},
        "3. Remove inter inference": {"function": Chapter04.RemoveInterInference, "description": "Khử nhiễu giao thoa bằng bộ lọc notch trong miền tần số.", "image": "test/Chuong4/3_Remove_interference.tif"},
        "4. Create motion": {"function": Chapter04.CreateMotion, "description": "Tạo hiệu ứng chuyển động.", "image": "test/Chuong4/4_Create_motion.tif"},
        "5. Demotion": {"function": Chapter04.Demotion, "description": "Khử nhiễu mờ chuyển động bằng bộ lọc tần số dựa trên mô hình chuyển động tuyến tính.", "image": "test/Chuong4/5_Demotion.tif"},
        "6. Demotion noise": {"function": Chapter04.DemotionNoise, "description": "Khử mờ chuyển động kết hợp lọc nhiễu bằng median blur và biến đổi Fourier.", "image": "test/Chuong4/6_Demotion_noise.tif"},
    },
    "Chương 9: Xử lý hình ảnh hình thái": {
        "1. Erosion": {"function": Chapter09.Erosion, "description": "Phép co ảnh giúp loại bỏ các chi tiết nhỏ và làm mờ các cạnh.", "image": "test/Chuong9/1_Erosion.tif"},
        "2. Dilation": {"function": Chapter09.Dilation, "description": "Áp dụng phép giãn ảnh (dilation) để mở rộng các vùng sáng, giúp khôi phục chi tiết bị mất và làm nổi bật các đối tượng", "image": "test/Chuong9/2_Dilation.tif"},
        "3. Boundary": {"function": Chapter09.BoundaryExtraction, "description": "Trích xuất đường biên của đối tượng bằng cách lấy hiệu giữa ảnh gốc và ảnh đã co (erode).", "image": "test/Chuong9/3_Boundary.tif"},
        "4. Contour": {"function": Chapter09.Contour, "description": "Tìm và vẽ đường viền bao quanh đối tượng, giúp làm nổi bật hình dạng và biên của vật thể bằng các đường nối liên tiếp.", "image": "test/Chuong9/4_Contour.tif"},
        "5. Connected Components": {"function": Chapter09.ConnectedComponents, "description": "Xác định và đếm được số lượng vùng đối tượng tách biệt trong ảnh.", "image": "test/Chuong9/5_Connected_Components.tif"},
        "6. Remove Small Rice": {"function": Chapter09.RemoveSmallRice, "description": "Loại bỏ các hạt gạo nhỏ và giữ lại những hạt lớn bằng cách sử dụng biến đổi hình thái và phân tích thành phần liên thông.", "image": "test/Chuong9/6_Remove_Small_Rice.tif"}
    }
}

COLOR_IMAGE_KEYS = ["2. Negative Color", "8. Hist Equal Color"]

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

    selected_chapter = st.sidebar.selectbox("📘 Chọn chương", list(CHAPTER_OPTIONS.keys()))
    lesson_options = list(CHAPTER_OPTIONS[selected_chapter].keys())
    selected_lesson = st.sidebar.selectbox("🧪 Chọn bài học", lesson_options)
    selected_item = CHAPTER_OPTIONS[selected_chapter][selected_lesson]
    selected_function = selected_item["function"]
    default_image_path = get_path(selected_item["image"])

    st.markdown("📖 **Mô tả:** " + selected_item["description"])

    upload_image = st.file_uploader(
        "📂 Chọn ảnh đầu vào", type=["bmp", "png", "jpg", "jpeg", "tif", "gif"]
    )

    # --- Xử lý ảnh bằng OpenCV ---
    frame = None
    read_flag = cv2.IMREAD_COLOR if selected_lesson in COLOR_IMAGE_KEYS else cv2.IMREAD_GRAYSCALE
    if upload_image is not None:
        file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, read_flag)
    elif default_image_path:
        frame = cv2.imread(default_image_path, read_flag)

    # --- Giao diện ---
    cols = st.columns(2)
    with cols[0]:
        input_container = st.empty()
        imagein_container = st.empty()
    with cols[1]:
        result_container = st.empty()
        imageout_container = st.empty()

    if frame is not None:
        input_container.subheader("🖼️ Ảnh gốc")
        if selected_lesson in COLOR_IMAGE_KEYS:
            imagein_container.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        else:
            imagein_container.image(frame, channels="GRAY")

        if st.sidebar.button("🚀 Xử lý"):
            result_container.subheader("🎯 Kết quả")
            try:
                result = selected_function(frame)
                imageout_container.image(result)
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý ảnh: {e}")

app()
