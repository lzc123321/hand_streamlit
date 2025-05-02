import streamlit as st
import cv2
import numpy as np

import torch
import tempfile
from PIL import Image
import torch
from yolov5.models.common import DetectMultiBackend
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

@st.cache
def load_model():
    # Model loading
    model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=True) # Can be 'yolov5n' - 'yolov5x6', or 'custom'
    return model

demo_img = "fire.9.png"
demo_video = "Fire_Video.mp4"

st.title('Fire Detection')
st.sidebar.title('App Mode')


app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['About App','Run on Image','Run on Video','Run on WebCam'])

if app_mode == 'About App':
    st.subheader("About")
    st.markdown("<h5>This is the Fire Detection App created with custom trained models using YoloV5</h5>",unsafe_allow_html=True)

    st.markdown("- <h5>Select the App Mode in the SideBar</h5>",unsafe_allow_html=True)
    #st.image("Images/first_1.png")
    st.markdown("- <h5>Upload the Image and Detect the Fires in Images</h5>",unsafe_allow_html=True)
    #st.image("Images/second_2.png")
    st.markdown("- <h5>Upload the Video and Detect the fires in Videos</h5>",unsafe_allow_html=True)
    #st.image("Images/third_3.png")
    st.markdown("- <h5>Live Detection</h5>",unsafe_allow_html=True)
    #st.image("Images/fourth_4.png")
    st.markdown("- <h5>Click Start to start the camera</h5>",unsafe_allow_html=True)
    st.markdown("- <h5>Click Stop to stop the camera</h5>",unsafe_allow_html=True)

    st.markdown("""
                ## Features
- Detect on Image
- Detect on Videos
- Live Detection
## Tech Stack
- Python
- PyTorch
- Python CV
- Streamlit
- YoloV5
## 🔗 Links
[![twitter](https://img.shields.io/badge/Github-1DA1F2?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AntroSafin)
""")


if app_mode == 'Run on Image':
    st.subheader("Detected Fire:")
    text = st.markdown("")

    st.sidebar.markdown("---")
    # Input for Image
    img_file = st.sidebar.file_uploader("Upload an Image",type=["jpg","jpeg","png"])
    if img_file:
        image = np.array(Image.open(img_file))
    else:
        image = np.array(Image.open(demo_img))

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Original Image**")
    st.sidebar.image(image)

    # predict the image
    model = load_model()
    results = model(image)
    length = len(results.xyxy[0])
    output = np.squeeze(results.render())
    text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>",unsafe_allow_html = True)
    st.subheader("Output Image")
    st.image(output,use_column_width=True)

if app_mode == 'Run on Video':
    st.subheader("Detected Fire:")
    text = st.markdown("")

    st.sidebar.markdown("---")

    st.subheader("Output")
    stframe = st.empty()

    #Input for Video
    video_file = st.sidebar.file_uploader("Upload a Video",type=['mp4','mov','avi','asf','m4v'])
    st.sidebar.markdown("---")
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file:
        vid = cv2.VideoCapture(demo_video)
        tffile.name = demo_video
    else:
        tffile.write(video_file.read())
        vid = cv2.VideoCapture(tffile.name)

    st.sidebar.markdown("**Input Video**")
    st.sidebar.video(tffile.name)

    # predict the video
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        model = load_model()
        results = model(frame)
        length = len(results.xyxy[0])
        output = np.squeeze(results.render())
        text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>",unsafe_allow_html = True)
        stframe.image(output)
str = ""
if app_mode == 'Run on WebCam':
    st.subheader("Detected Fire:")
    text = st.markdown("")

    st.sidebar.markdown("---")

    st.subheader("Output")
    stframe = st.empty()

    run = st.sidebar.button("Start")
    stop = st.sidebar.button("Stop")
    st.sidebar.markdown("---")

    cam = cv2.VideoCapture(0)
    # 创建一个空字典
    hand_type = {}
    # 计时器
    time_start = 0
    # 结果类别
    result_cls = ""
    if(run):
        while(True):
            if(stop):
                break
            ret,frame = cam.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            model = load_model()
            results = model(frame)
            length = len(results.xyxy[0])

            # 遍历检测结果
            for det in results.xyxy[0]:
                # 提取类别 ID
                class_id = int(det[5])
                confidence = float(det[4])
                # 过滤低置信度的检测结果
                # 获取类别名称
                class_name = results.names[class_id]

                # 打印类别名称
                print(f"检测到的类别: {class_name}")
                if confidence > 0.75:
                    hand_type[class_name] = hand_type.get(class_name, 0) + 1
                    time_start += 1
                    output = np.squeeze(results.render())
                    stframe.image(output)
                else:
                    stframe.image(frame)

                if time_start > 10:
                    # 获取字典中max键值对
                    max_key = max(hand_type, key=hand_type.get)
                    result_cls = max_key
                    # 打印最大键值对
                    print(f"最大类别: {max_key}, 数量: {hand_type[max_key]}")

                    time_start = 0
                    hand_type = {}
                text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>",unsafe_allow_html = True)
                text.write(f"<h1 style='text-align: center; color:red;'>{result_cls}</h1>", unsafe_allow_html=True)
            stframe.image(frame)
