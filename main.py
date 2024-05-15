import streamlit as st
from st_pages import Page, show_pages, add_page_title

st.set_page_config(
    page_title="Welcome",
    page_icon="🏠",
    layout="wide"
)

show_pages(
    [
        Page("main.py", "Home", "🏠"),
        Page("pages/face_detection.py", "Nhận diện khuôn mặt", "😃"),
        Page("pages/Xu_Ly_Anh.py", "Xử lý ảnh", "✅"),
        Page("pages/object_detection.py", "Nhận diện trái cây", "🍎"),
        Page("pages/nhan_dang_chu_so_viet_tay.py", "Nhận dạng chữ số viết tay", "🔢")
    ]
)

st.write("# DIPR430685_23_2_04CLC - Xử lý ảnh👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    ### Ngô Minh Thuận 21110314
    ### Nguyễn Minh 21110242
"""
)

st.image("./images/welcome.jpg")
st.rerun()