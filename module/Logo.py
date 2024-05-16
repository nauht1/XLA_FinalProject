import streamlit as st

def add_logo():
    st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)
    st.sidebar.title("21110314 Ngô Minh Thuận")
    st.sidebar.title("21110242 Nguyễn Minh")
    with st.sidebar:
        st.image("./images/opencv_icon.png", width= 250)
