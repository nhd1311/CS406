import streamlit as st
import cv2
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def predict_image(weights_path, img_array):
    """Dự đoán ảnh bằng YOLO."""
    # Load model
    model = YOLO(weights_path)

    # Convert BGR to RGB
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Inference
    start_time = time.time()
    prediction = model(img_bgr)
    end_time = time.time()

    # Calculate inference time
    inference_time = end_time - start_time

    # Draw bounding boxes and labels
    img_result = img_bgr.copy()
    for box in prediction[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        label = f" {class_id} ({confidence:.2f})"

        # Draw rectangle and label
        cv2.rectangle(img_result, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=1)
        cv2.putText(
            img_result, label, (x1, y2 + 10),  # Adjust text position below the bounding box
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.3,  # Reduced font scale for smaller text
            color=(255, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # Convert back to RGB for display
    img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
    return img_result, inference_time

# Streamlit App
st.title("YOLOv8 & YOLOv11 Image Prediction")
st.write("Upload an image and view predictions from both YOLO versions.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Define weights paths
weights_paths = {
    "YOLOv8": "D:/Study/CS406/Do_An/dataset/best_150epoch_v8l.pt",
    "YOLOv11": "D:/Study/CS406/Do_An/dataset/best_150epochs_v11l.pt",
}

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file)
    img_array = np.array(img)

    # Predict with YOLOv8
    yolov8_weights = weights_paths["YOLOv8"]
    yolov8_img, yolov8_time = predict_image(yolov8_weights, img_array)

    # Predict with YOLOv11
    yolov11_weights = weights_paths["YOLOv11"]
    yolov11_img, yolov11_time = predict_image(yolov11_weights, img_array)

    # Display results
    st.subheader("Original Image")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.subheader("Predicted Image with YOLOv8")
    st.image(yolov8_img, caption=f"Predicted with YOLOv8 in {yolov8_time:.2f} seconds", use_column_width=True)

    st.subheader("Predicted Image with YOLOv11")
    st.image(yolov11_img, caption=f"Predicted with YOLOv11 in {yolov11_time:.2f} seconds", use_column_width=True)
