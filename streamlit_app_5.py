import numpy as np
import streamlit as st
import cv2

st.title("Deep Learning Face Detection (OpenCV DNN)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    net = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 2)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Deep Learning Detection")
