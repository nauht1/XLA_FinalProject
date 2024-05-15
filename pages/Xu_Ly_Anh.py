import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Xử lý ảnh", page_icon="😃", layout="wide")

st.markdown("# Xử lý ảnh")

# Các tùy chọn xử lý ảnh
options = [
    "Negative", "Logarithm", "PiecewiseLinear", "Histogram", "HistEqual", "HistEqualColor", 
    "LocalHist", "HistStat", "BoxFilter", "LowpassGauss", "Threshold", "MedianFilter", 
    "Sharpen", "Gradient", "Spectrum", "FrequencyFilter", "DrawNotchRejectFilter", 
    "RemoveMoire", "CreateMotionNoise", "DenoiseMotion", "DenoisestMotion", "Erosion", 
    "Dilation", "OpeningClosing", "Boundary", "HoleFilling", "HoleFillingMouse", 
    "ConnectedComponent", "CountRice"
]

# Tạo selection box để chọn chức năng xử lý ảnh
selected_option = st.selectbox("Chọn chức năng xử lý ảnh", options)

# Nút Open Image để chọn ảnh và hiển thị bản xem trước
uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png", "tif"])
process_button = st.button("Xử lý")

L = 256

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Sử dụng cột để hiển thị ảnh gốc và ảnh đã xử lý trên cùng một hàng
    col1, col2 = st.columns(2)
    col1.image(img_array, caption='Ảnh gốc', width=400)

    # Nút "Xử lý" để xử lý ảnh
    if process_button:
        ### Chapter 3
        def Negative(imgin):
            M, N = imgin.shape
            imgout = np.zeros((M,N), np.uint8)
            for x in range(0, M):
                for y in range(0, N):
                    r = imgin[x,y]
                    s = L-1-r
                    imgout[x,y] = s
            return imgout

        def Logarithm(imgin):
            M, N = imgin.shape
            imgout = np.zeros((M,N), np.uint8)
            c = (L-1)/np.log(L)
            for x in range(0, M):
                for y in range(0, N):
                    r = imgin[x,y]
                    if r == 0:
                        r = 1
                    s = c*np.log(1+r)
                    imgout[x,y] = np.uint8(s)
            return imgout

        def PiecewiseLinear(imgin):
            M, N = imgin.shape
            imgout = np.zeros((M,N), np.uint8)
            rmin, rmax, vi_tri_rmin, vi_tri_rmax = cv2.minMaxLoc(imgin)
            r1 = rmin
            s1 = 0
            r2 = rmax
            s2 = L-1
            for x in range(0, M):
                for y in range(0, N):
                    r = imgin[x,y]
                    if r < r1:
                        s = s1/r1*r
                    elif r < r2:
                        s = (s2-s1)/(r2-r1)*(r-r1) + s1
                    else:
                        s = (L-1-s2)/(L-1-r2)*(r-r2) + s2
                    imgout[x,y] = np.uint8(s)
            return imgout

        def Histogram(imgin):
            M, N = imgin.shape
            L = 256
            imgout = np.zeros((M, L), np.uint8) + 255
            h = np.zeros(L, np.int32)
            for x in range(0, M):
                for y in range(0, N):
                    r = imgin[x, y]
                    h[r] = h[r] + 1
            p = h / (M * N)
            scale = 2000
            for r in range(0, L):
                cv2.line(imgout, (r, M - 1), (r, M - 1 - int(scale * p[r])), (0, 0, 0))
            return imgout

        def HistEqual(imgin):
            M, N = imgin.shape
            imgout = np.zeros((M,N), np.uint8)
            h = np.zeros(L, np.int32)
            for x in range(0, M):
                for y in range(0, N):
                    r = imgin[x,y]
                    h[r] = h[r]+1
            p = h/(M*N)

            s = np.zeros(L, np.float64)
            for k in range(0, L):
                for j in range(0, k+1):
                    s[k] = s[k] + p[j]

            for x in range(0, M):
                for y in range(0, N):
                    r = imgin[x,y]
                    imgout[x,y] = np.uint8((L-1)*s[r])
            return imgout

        def HistEqualColor(imgin):
            B = imgin[:,:,0]
            G = imgin[:,:,1]
            R = imgin[:,:,2]
            B = cv2.equalizeHist(B)
            G = cv2.equalizeHist(G)
            R = cv2.equalizeHist(R)
            imgout = np.array([B, G, R])
            imgout = np.transpose(imgout, axes = [1,2,0]) 
            return imgout

        def LocalHist(imgin):
            M, N = imgin.shape
            imgout = np.zeros((M,N), np.uint8)
            m = 3
            n = 3
            w = np.zeros((m,n), np.uint8)
            a = m // 2
            b = n // 2
            for x in range(a, M-a):
                for y in range(b, N-b):
                    for s in range(-a, a+1):
                        for t in range(-b, b+1):
                            w[s+a,t+b] = imgin[x+s,y+t]
                    w = cv2.equalizeHist(w)
                    imgout[x,y] = w[a,b]
            return imgout

        def HistStat(imgin):
            M, N = imgin.shape
            imgout = np.zeros((M,N), np.uint8)
            m = 3
            n = 3
            w = np.zeros((m,n), np.uint8)
            a = m // 2
            b = n // 2
            mG, sigmaG = cv2.meanStdDev(imgin)
            C = 22.8
            k0 = 0.0
            k1 = 0.1
            k2 = 0.0
            k3 = 0.1
            for x in range(a, M-a):
                for y in range(b, N-b):
                    for s in range(-a, a+1):
                        for t in range(-b, b+1):
                            w[s+a,t+b] = imgin[x+s,y+t]
                    msxy, sigmasxy = cv2.meanStdDev(w)
                    r = imgin[x,y]
                    if (k0*mG <= msxy <= k1*mG) and (k2*sigmaG <= sigmasxy <= k3*sigmaG):
                        imgout[x,y] = np.uint8(C*r)
                    else:
                        imgout[x,y] = r
            return imgout

        def BoxFilter(imgin):
            m = 21
            n = 21
            w = np.ones((m,n))
            w = w/(m*n)
            imgout = cv2.filter2D(imgin,cv2.CV_8UC1,w)
            return imgout

        def LowpassGauss(imgin):
            imgout = cv2.GaussianBlur(imgin,(43,43),7.0)
            return imgout
        
        def Threshold(imgin):
            temp = cv2.blur(imgin, (15,15))
            retval, imgout = cv2.threshold(temp,64,255,cv2.THRESH_BINARY)
            return imgout

        def MedianFilter(imgin):
            M, N = imgin.shape
            imgout = np.zeros((M,N), np.uint8)
            m = 5
            n = 5
            w = np.zeros((m,n), np.uint8)
            a = m // 2
            b = n // 2
            for x in range(0, M):
                for y in range(0, N):
                    for s in range(-a, a+1):
                        for t in range(-b, b+1):
                            w[s+a,t+b] = imgin[(x+s)%M,(y+t)%N]
                    w_1D = np.reshape(w, (m*n,))
                    w_1D = np.sort(w_1D)
                    imgout[x,y] = w_1D[m*n//2]
            return imgout

        def Sharpen(imgin):
            # Đạo hàm cấp 2 của ảnh
            w = np.array([[1,1,1],[1,-8,1],[1,1,1]])
            temp = cv2.filter2D(imgin,cv2.CV_32FC1,w)

            # Hàm cv2.Laplacian chỉ tính đạo hàm cấp 2
            # cho bộ lọc có số -4 chính giữa
            imgout = imgin - temp
            imgout = np.clip(imgout, 0, L-1)
            imgout = imgout.astype(np.uint8)
            return imgout

        def Gradient(imgin):
            sobel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
            sobel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

            # Đạo hàm cấp 1 theo hướng x
            mygx = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_x)
            # Đạo hàm cấp 1 theo hướng y
            mygy = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_y)

            # Lưu ý: cv2.Sobel có hướng x nằm ngang
            # ngược lại với sách Digital Image Processing
            gx = cv2.Sobel(imgin,cv2.CV_32FC1, dx = 1, dy = 0)
            gy = cv2.Sobel(imgin,cv2.CV_32FC1, dx = 0, dy = 1)

            imgout = abs(gx) + abs(gy)
            imgout = np.clip(imgout, 0, L-1)
            imgout = imgout.astype(np.uint8)
            return imgout

        ###Chapter 4
        
        ###
        
        # Xử lý ảnh theo chức năng đã chọn
        processed_img = None
        if selected_option == "Negative":
            processed_img = Negative(img_array)
        elif selected_option == "Logarithm":
            processed_img = Logarithm(img_array)
        elif selected_option == "PiecewiseLinear":
            processed_img = PiecewiseLinear(img_array)
        elif selected_option == "Histogram":
            processed_img = Histogram(img_array)
        elif selected_option == "HistEqual":
            processed_img = HistEqual(img_array)
        elif selected_option == "HistEqualColor":
            processed_img = HistEqualColor(img_array)
        elif selected_option == "LocalHist":
            processed_img = LocalHist(img_array)
        elif selected_option == "HistStat":
            processed_img = HistStat(img_array)
        elif selected_option == "BoxFilter":
            processed_img = BoxFilter(img_array)
        elif selected_option == "LowpassGauss":
            processed_img = LowpassGauss(img_array)
        elif selected_option == "Threshold":
            processed_img = Threshold(img_array)
        elif selected_option == "MedianFilter":
            processed_img = MedianFilter(img_array)
        elif selected_option == "Sharpen":
            processed_img = Sharpen(img_array)
        elif selected_option == "Gradient":
            processed_img = Gradient(img_array)

        # Hiển thị ảnh sau khi xử lý (nếu có)
        if processed_img is not None:
            col2.image(processed_img, caption='Ảnh đã xử lý', width=400)
else:
    if process_button:
        st.warning("Vui lòng chọn một ảnh trước khi nhấn 'Xử lý'.")
