import streamlit as st
from static.utils import *
from static.functions_config import *

st.set_page_config(page_title="Data Manager", page_icon="✏️", layout="wide")

load_css("static/styles.css")

# --- Sidebar ---
with st.sidebar:
    logo_path = "assets/logo-hcmute.png"
    if os.path.exists(logo_path):
        logo_base64 = local_image_to_base64(logo_path)
        st.markdown(f"""
            <div style="text-align: center; padding: 10px 0;">
                <img src="data:image/jpg;base64,{logo_base64}" width="200">
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<h2 style="text-align:center;">DIGITAL IMAGE PROCESSING</h2>', unsafe_allow_html=True)

    main_options = [
        "📄 Trang giới thiệu",
        "Giải phương trình bậc 2",
        "Nhận dạng chữ số",
        "Nhận dạng khuôn mặt",
        "Nhận dạng đối tượng",
        "Xử lý ảnh số",
        "Nhận Diện Cử Chỉ Tay"
    ]
    selected_main = st.selectbox("📚 Chọn chức năng", main_options)

    # selected_chapter = None
    # if selected_main == "Xử lý ảnh số":
    #     selected_chapter = st.selectbox("📂 Chọn chương", [
    #         "Chương 3: Chỉnh sửa ảnh cơ bản",
    #         "Chương 4: Biến đổi ảnh",
    #         "Chương 9: Nhận dạng và phân loại ảnh"
    #     ])

# --- Main Content ---
if selected_main == "📄 Trang giới thiệu":
    st.markdown("""
        <div style="padding: 25px; border-radius: 10px; border: 1px solid #ddd;">
            <h2>📄 Thông tin đồ án môn học</h2>
            <h4>🧑‍🤝‍🧑 Thành viên nhóm:</h4>
            <ul style="font-size: 16px; line-height: 1.8; margin-left: 20px;">
                <strong> <li>👨‍💻 Phạm Khánh Huy - 22110336</li> </strong>
                <strong> <li>👨‍💻 Trang Kim Lợi - 22110371</li> </strong>
            </ul>
            <h4>📘 Môn học:</h4>
                <ul style="font-size: 16px; line-height: 1.8; margin-left: 20px;">
                    <strong> <li>DIGITAL IMAGE PROCESSING - XỬ LÝ ẢNH SỐ</li> </strong>
                </ul>
            <h4>👨‍🏫 Giảng viên hướng dẫn:</h4>
                <ul style="font-size: 16px; line-height: 1.8; margin-left: 20px;">
                    <strong> <li>👨‍💻 Th.S Trần Tiến Đức</li> </strong>
                </ul>
            <h4>🎯 Mục tiêu:</h4>
            <ul style="font-size: 16px; line-height: 1.7; margin-left: 20px;">
                <strong> <li>Tổng hợp các bài học xử lý ảnh thành một giao diện học tập dễ sử dụng.</li> </strong>
                <strong> <li>Ứng dụng các mô hình học sâu và xử lý ảnh truyền thống.</li> </strong>
            </ul>
            <p style="margin-top: 20px; font-style: italic; color: gray; text-align: center;">
                Cảm ơn thầy/cô đã theo dõi và góp ý cho đồ án của nhóm chúng em!
            </p>
        </div>
    """, unsafe_allow_html=True)

else:
    selected_key = next(k for k, v in FUNCTIONS.items() if v["label"] == selected_main)
    selected_func = FUNCTIONS[selected_key]

    file_path = selected_func["file"]
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as f:
            exec(f.read(), globals())
    else:
        st.error(f"Không tìm thấy file: {file_path}")

# elif selected_chapter:
#     selected_chapter_key = {
#         "Chương 3: Chỉnh sửa ảnh cơ bản": 3,
#         "Chương 4: Biến đổi ảnh": 4,
#         "Chương 9: Nhận dạng và phân loại ảnh": 9
#     }[selected_chapter]

#     selected_func = FUNCTIONS[5]
#     chapter_info = selected_func["children"][selected_chapter_key]
    
#     st.markdown(f"## 📚 {chapter_info['label']}")
#     st.markdown(chapter_info["description"])

#     func_objs = chapter_info["functions"]
#     numbered_labels = [f"{i + 1}. {func['label']}" for i, func in enumerate(func_objs.values())]
#     label_to_file = {f"{i + 1}. {func['label']}": func["file"] for i, func in enumerate(func_objs.values())}

#     selected_func_label = st.selectbox("🧪 Chọn bài học", numbered_labels)
#     file_path = label_to_file[selected_func_label]

#     if os.path.exists(file_path):
#         with open(file_path, encoding="utf-8") as f:
#             exec(f.read(), globals())
#     else:
#         st.error(f"Không tìm thấy file: {file_path}")

