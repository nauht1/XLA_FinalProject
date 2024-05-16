import streamlit as st
from st_pages import Page, show_pages, add_page_title
import module.Logo as logo

st.set_page_config(
    page_title="Welcome",
    page_icon="🏠",
    layout="wide"
)

logo.add_logo()

show_pages(
    [
        Page("main.py", "Home", "🏠"),
        Page("pages/face_detection.py", "Nhận diện khuôn mặt", "😃"),
        Page("pages/Xu_Ly_Anh.py", "Xử lý ảnh", "✅"),
        Page("pages/object_detection.py", "Nhận diện trái cây", "🍎"),
        Page("pages/nhan_dang_chu_so_viet_tay.py", "Nhận dạng chữ số viết tay", "🔢"),
        Page("pages/hand_tracking.py", "Đếm số ngón tay", "🙌")
    ]
)

st.write("# DIPR430685_23_2_04CLC - Xử lý ảnh👋")


st.markdown(
    """
    ### Ngô Minh Thuận 21110314
    ### Nguyễn Minh 21110242
"""
)

col1, col2, col3 = st.columns(3)
col1.image("./images/python_icon.svg", width = 300)
col2.image("./images/streamlit_icon.png", width = 300)
col3.image("./images/tensorflow_icon.png", width = 300)

st.rerun()