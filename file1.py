import streamlit as st
import cv2
import numpy as np

import torch
import tempfile
from PIL import Image
import torch
from yolov5.models.common import DetectMultiBackend
from pydub import AudioSegment
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 获取音频文件的长度
def get_audio_length(file_path):
    audio = AudioSegment.from_wav(file_path)
    return audio.duration_seconds  # 返回音频的长度（秒）

@st.cache_resource
def load_model():
    # Model loading
    model = torch.hub.load("ultralytics/yolov5",
                           "custom", path="best_m.pt", force_reload=True,
                           autoshape=True)  # Can be 'yolov5n' - 'yolov5x6', or 'custom'
    return model


demo_img = "fire.png"
demo_video = "好运来.mp4"

st.title('Gesture Recognition')
st.logo("logo_w.png", size="large")
sidebar_markdown_title = '''
    <center>
        <h1>Gesture recognition</h1>
    </center>
    '''
sidebar_markdown = '''
    <div style="text-align: center;">
        <code> v0.2.1 </code>
    </div>
    
    <center>
    <a href="https://github.com/lzc123321/hand_streamlit">
    <img src = "https://cdn-icons-png.flaticon.com/512/733/733609.png" width="23"></img></a>

    <a href="mailto:kathrin.sessler@tum.de">
    <img src="https://cdn-icons-png.flaticon.com/512/646/646094.png" alt="email" width = "27" ></a>
    </center>
    <br> <!-- 添加空行 -->

    '''.format()
st.sidebar.image("3.png", use_container_width=True)
st.sidebar.markdown(sidebar_markdown_title, unsafe_allow_html=True)

st.sidebar.markdown(sidebar_markdown, unsafe_allow_html=True)
st.sidebar.image("2.png", use_container_width=True)

app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['About App', 'About models', 'Run on Image', 'Run on Video', 'Run on WebCam'])

if app_mode == 'About App':
    st.markdown("<h5>This is the Gesture Recognition App created with custom trained models using YoloV5</h5>",
                unsafe_allow_html=True)

    # st.markdown("- <h5>Select the App Mode in the SideBar</h5>",unsafe_allow_html=True)
    # st.markdown("- <h6>Upload the Image and Detect the Gesture in Images</h5>",unsafe_allow_html=True)
    # #st.image("mode2.png")
    # st.markdown("- <h6>Upload the Video and Detect the Gestures in Videos</h5>",unsafe_allow_html=True)
    # #st.image("Images/third_3.png")
    # st.markdown("- <h6>Live Detection</h5>",unsafe_allow_html=True)
    #st.image("Images/fourth_4.png")
    # st.markdown("- <h5>Click Start to start the camera</h5>",unsafe_allow_html=True)
    # st.markdown("- <h5>Click Stop to stop the camera</h5>",unsafe_allow_html=True)

    st.markdown("""
## ☑️Select the App Mode in the SideBar☑️
- Upload the Image and Detect the Gesture in Images Detection
- Upload the Video and Detect the Gestures in Videos Detection
- Live
## ⭐Features⭐
- Detect on Image
- Detect on Videos
- Live Detection
## 🧩Tech Stack🧩
- Python
- PyTorch
- Python CV
- Streamlit
- YoloV5
## 📧 Contact  📧 联系

""")
if app_mode == 'About models':
    st.markdown(
        "<h5>YoloV5 is a family of object detection architectures and models pretrained on the COCO dataset.</h5>",
        unsafe_allow_html=True)
    st.image("1.jpg", use_container_width=True)
    st.markdown("""

    ## Citations
    - HaGRIDv2: Nuzhdin A., Nagaev A., et al. "HaGRIDv2: 1M Images for Static and Dynamic Hand Gesture Recognition", arXiv:2412.01508 (2024). 
      Available: https://arxiv.org/abs/2412.01508

    - HaGRID: Kapitanov A., Kvanchiani K., et al. "HaGRID - HAnd Gesture Recognition Image Dataset", WACV 2024, pp. 4572-4581. 
      Available: https://openaccess.thecvf.com/content/WACV2024/html/Kapitanov_HaGRID_-_HAnd_Gesture_Recognition_Image_Dataset_WACV_2024_paper.html

    """)

if app_mode == 'Run on Image':
    st.subheader("Results")
    text = st.markdown("")

    st.sidebar.markdown("---")
    # Input for Image
    img_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if img_file:
        image = np.array(Image.open(img_file))
    else:
        image = np.array(Image.open(demo_img))

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Original Image**")
    st.sidebar.image(image)

    # predict the image
    model = load_model()
    results = model(image, augment=True)
    print(results.xyxy[0])
    length = len(results.xyxy[0])
    output = np.squeeze(results.render())
    text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>", unsafe_allow_html=True)
    st.subheader("Output Image")
    st.image(output, use_container_width=True)

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
    st.subheader("Detected Gesture:")
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
    # 添加开始按钮
    start_button = st.sidebar.button("开始")

    if video_file:
        if start_button:
            tffile.write(video_file.read())
            tffile.flush()
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
            st.sidebar.markdown("**输入视频**")
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
        else:
            st.sidebar.markdown("请按下开始按钮以加载模型并处理视频。")
    else:
        st.sidebar.markdown("请上传一个符合条件的视频文件。")
    # if video_file:
    #     vid = cv2.VideoCapture(demo_video)
    #     tffile.name = demo_video
    # else:
    #     tffile.write(video_file.read())
    #     vid = cv2.VideoCapture(tffile.name)
    #
    # # 获取原视频参数
    # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = vid.get(cv2.CAP_PROP_FPS)
    #
    # # 初始化视频写入器（使用H.264编码）
    # fourcc = cv2.VideoWriter_fourcc(*'H264')  # 或使用 'avc1' 需要对应环境支持
    # out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    #
    # # 提前加载模型（优化性能）
    # model = load_model()  # 将模型加载移出循环
    #
    # # 显示原视频
    # st.sidebar.markdown("**Input Video**")
    # st.sidebar.video(tffile.name)

    # # 进度条
    # progress_bar = st.progress(0)
    # total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    #
    # # 处理视频帧
    # frame_count = 0
    # while vid.isOpened():
    #     ret, frame = vid.read()
    #     if not ret:
    #         break
    #
    #     # 处理帧
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = model(frame)
    #     output = np.squeeze(results.render())
    #
    #     # 写入处理后的帧到视频文件（需要转换回BGR）
    #     out.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    #
    #     # 更新界面显示
    #     length = len(results.xyxy[0])
    #     text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>", unsafe_allow_html=True)
    #     stframe.image(output)
    #
    #     # 更新进度条
    #     frame_count += 1
    #     progress = min(frame_count / total_frames, 1.0)
    #     progress_bar.progress(progress)

    # # 释放资源
    # vid.release()
    # out.release()
    # cv2.destroyAllWindows()
    #
    # # 显示完成提示和下载按钮
    # st.success(f"视频处理完成！保存路径: {output_filename}")

    # # 新增播放控制组件 ----------------------------------------
    # st.markdown("---")
    # # 显示原视频
    #
    # video_file = open(output_filename, 'rb')
    # video_bytes = video_file.read()
    #
    # # 本地视频
    # st.video(video_bytes, format="mp4", start_time=0)
    #
    # # 进度条
    # progress_bar = st.progress(0)
    # total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    #
    # with open(output_filename, "rb") as file:
    #     st.download_button(
    #         label="下载处理后的视频",
    #         data=file,
    #         file_name=os.path.basename(output_filename),
    #         mime="video/mp4"
    #     )
    #
    # # 清理临时文件
    # os.unlink(tffile.name)
str1 = ""
class_name = "None"
import time
import shutil

if app_mode == 'Run on WebCam':
    st.subheader("Detected Gesture:")
    text = st.markdown("")

    st.sidebar.markdown("---")

    st.subheader("Output")
    stframe = st.empty()

    run, stop = st.sidebar.columns(2)

    # 在第一列放置 "Start" 按钮
    with run:
        run = st.button("🔛Start🔛", use_container_width=True)

    # 在第二列放置 "Stop" 按钮
    with stop:
        stop = st.button("⏹Stop⏹", use_container_width=True)
    st.sidebar.markdown("---")

    cam = cv2.VideoCapture(0)
    # 创建一个空字典
    hand_type = {}
    # 计时器
    time_start = 0
    # 结果类别
    result_cls = ""
    class_name_ti10 = ""
    if (run):
        while (True):
            if (stop):
                break
            # 初始化计时器和帧计数器
            start_time = time.time()
            frame_count = 0
            ret, frame = cam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            model = load_model()
            results = model(frame)
            # 更新帧计数器
            frame_count += 1

            # 计算并显示FPS
            elapsed_time = time.time() - start_time
            print(f"FPS: {frame_count / elapsed_time:.2f}")
            length = 0
            # html_content = f"""
            #         <div style='text-align: center;'>
            #             <h1 style='color: red;'>fps: {frame_count / elapsed_time:.2f}</h1>
            #         </div>
            # """
            # text.write(html_content, unsafe_allow_html=True)
            # 检查是否有检测到任何框
            if len(results.xyxy[0]) > 0:
                # 找到置信度最高的框
                highest_confidence = -1
                highest_confidence_box = None
                highest_cls = None

                for *box, conf, cls in results.xyxy[0]:
                    if conf > highest_confidence:
                        highest_confidence = conf
                        highest_confidence_box = box
                        highest_cls = cls

                # 确保置信度高于阈值
                if highest_confidence > 0.8:
                    time_start += 1
                    # 将最高置信度的框转换为整数坐标
                    x1, y1, x2, y2 = map(int, highest_confidence_box)
                    class_id = int(highest_cls)
                    class_name = results.names[class_id]
                    confidence = float(highest_confidence)

                    # 打印类别名称
                    print(f"检测到的类别: {class_name}")

                    # 在原始图像上绘制最高置信度的框
                    frame_with_box = frame.copy()
                    cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_with_box, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # 显示图像
                    stframe.image(frame_with_box)
                    if time_start > 10:
                        time_start = 0
                        class_name_ti10 = class_name
                        audio_container = st.container()  # 创建一个容器来显示音频
                        audio_path = f"voices_hands/{class_name_ti10}.wav"
                        # # 生成唯一的临时文件名
                        # timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                        # temp_audio_path = f"voices_hands/temp_{timestamp}.wav"

                        # # 复制原始音频文件
                        # shutil.copyfile(audio_path, temp_audio_path)
                        audio_bytes = open(audio_path, 'rb').read()
                        with audio_container:
                            # 获取音频文件的长度
                            #audio_length = get_audio_length(audio_bytes)
                            st.audio(audio_bytes,format="audio/wav", autoplay =True)
                        print(f"播放音频时间: {audio_bytes}")
                        time.sleep(5)  # 假设音频长度为2秒，等待音频播放完毕
                        audio_container.empty()  # 清空容器以移除音频组件
                        # if os.path.exists(temp_audio_path):
                        #     os.remove(temp_audio_path)
                else:
                    # 如果没有检测到高置信度的框，显示原始图像
                    stframe.image(frame)
            else:
                # 如果没有检测到任何框，显示原始图像
                stframe.image(frame)
            html_content = f"""
                                    <div style='text-align: center;'>
                                         <h1 style='color: red;'>fps: {frame_count / elapsed_time:.2f}</h1>
                                        <h1 style='color: red;'>{class_name_ti10}</h1>
                                    </div>
                                    """
            text.write(html_content, unsafe_allow_html=True)
            # # 遍历检测结果
            # for det in results.xyxy[0]:
            #     # 提取类别 ID
            #     class_id = int(det[5])
            #     confidence = float(det[4])
            #     # 过滤低置信度的检测结果
            #     # 获取类别名称
            #     class_name = results.names[class_id]
            #
            #     # 打印类别名称
            #     print(f"检测到的类别: {class_name}")
            #     if confidence > 0.7:
            #         length = len(results.xyxy[0])
            #         hand_type[class_name] = hand_type.get(class_name, 0) + 1
            #         time_start += 1
            #         output = np.squeeze(results.render())
            #         stframe.image(output)
            #     else:
            #         stframe.image(frame)
            #
            #     if time_start > 10:
            #         # 获取字典中max键值对
            #         max_key = max(hand_type, key=hand_type.get)
            #         result_cls = max_key
            #         # 打印最大键值对
            #         print(f"最大类别: {max_key}, 数量: {hand_type[max_key]}")
            #
            #         time_start = 0
            #         hand_type = {}
            #         break
            #
            #     html_content = f"""
            #     <div style='text-align: center;'>
            #         <h1 style='color: red;'>{length}</h1>
            #         <h1 style='color: red;'>{result_cls}</h1>
            #     </div>
            #     """
            #     text.write(html_content, unsafe_allow_html=True)
            # stframe.image(frame)

