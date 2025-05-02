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



@st.cache_resource
def load_model():
    # Model loading
    model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.pt", force_reload=True) # Can be 'yolov5n' - 'yolov5x6', or 'custom'
    return model

demo_img = "fire.png"
demo_video = "好运来.mp4"

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

# if app_mode == 'Run on Video':
#     st.subheader("Detected Fire:")
#     text = st.markdown("")
#
#     st.sidebar.markdown("---")
#
#     st.subheader("Output")
#     stframe = st.empty()
#
#     #Input for Video
#     video_file = st.sidebar.file_uploader("Upload a Video",type=['mp4','mov','avi','asf','m4v'])
#     st.sidebar.markdown("---")
#     tffile = tempfile.NamedTemporaryFile(delete=False)
#
#     if not video_file:
#         vid = cv2.VideoCapture(demo_video)
#         tffile.name = demo_video
#     else:
#         tffile.write(video_file.read())
#         vid = cv2.VideoCapture(tffile.name)
#
#     st.sidebar.markdown("**Input Video**")
#     st.sidebar.video(tffile.name)
#
#     # predict the video
#     while vid.isOpened():
#         ret, frame = vid.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#         model = load_model()
#         results = model(frame)
#         length = len(results.xyxy[0])
#         output = np.squeeze(results.render())
#         text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>",unsafe_allow_html = True)
#         stframe.image(output)
import cv2
import tempfile
import os
from datetime import datetime

if app_mode == 'Run on Video':
    st.subheader("Detected Fire:")
    text = st.markdown("")
    st.sidebar.markdown("---")
    st.subheader("Output")
    stframe = st.empty()

    # 创建输出目录
    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)

    # 生成带时间戳的输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"new_video_{timestamp}.mp4")

    # 视频输入处理
    video_file = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    st.sidebar.markdown("---")
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file:
        vid = cv2.VideoCapture(demo_video)
        tffile.name = demo_video
    else:
        tffile.write(video_file.read())
        vid = cv2.VideoCapture(tffile.name)

    # 获取原视频参数
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)

    # 初始化视频写入器（使用H.264编码）
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # 或使用 'avc1' 需要对应环境支持
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # 提前加载模型（优化性能）
    model = load_model()  # 将模型加载移出循环

    # 显示原视频
    st.sidebar.markdown("**Input Video**")
    st.sidebar.video(tffile.name)

    # 进度条
    progress_bar = st.progress(0)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # 处理视频帧
    frame_count = 0
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        # 处理帧
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)
        output = np.squeeze(results.render())

        # 写入处理后的帧到视频文件（需要转换回BGR）
        out.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

        # 更新界面显示
        length = len(results.xyxy[0])
        text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>", unsafe_allow_html=True)
        stframe.image(output)

        # 更新进度条
        frame_count += 1
        progress = min(frame_count / total_frames, 1.0)
        progress_bar.progress(progress)

    # 释放资源
    vid.release()
    out.release()
    cv2.destroyAllWindows()

    # 显示完成提示和下载按钮
    st.success(f"视频处理完成！保存路径: {output_filename}")

    # 新增播放控制组件 ----------------------------------------
    st.markdown("---")
    # 显示原视频

    video_file = open(output_filename, 'rb')
    video_bytes = video_file.read()

    # 本地视频
    st.video(video_bytes, format="mp4", start_time=0)

    # 进度条
    progress_bar = st.progress(0)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))


    with open(output_filename, "rb") as file:
        st.download_button(
            label="下载处理后的视频",
            data=file,
            file_name=os.path.basename(output_filename),
            mime="video/mp4"
        )

    # 清理临时文件
    os.unlink(tffile.name)
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
