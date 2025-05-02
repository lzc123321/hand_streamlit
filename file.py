from PIL.Image import Image
from ultralytics import solutions

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import yolov5
model = yolov5.load("best.pt")  # 加载 YOLOv5s 模型

import streamlit as st

# 设置标题和文件上传小部件
st.title("YOLOv5 目标检测")
uploaded_file = st.file_uploader("上传图像")

if uploaded_file is not None:
    image = Image.open(uploaded_file)  # 将上传的图像读入内存
    results = model(image)  # 对图像进行目标检测
    # 遍历检测结果并显示边界框和标签
    for result in results.pred:
        for box, cls, conf in zip(*result):
            x, y, w, h = box.int().tolist()
            st.write(f"边界框: ({x}, {y}, {w}, {h})")
            st.write(f"类别: {cls}")
            st.write(f"置信度: {conf}")
