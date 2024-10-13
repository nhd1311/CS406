### Gõ python -m streamlit run app.py ở terminal để chạy demo

import os
import cv2
import numpy as np
import streamlit as st
import pickle
import matplotlib.pyplot as plt

# Path to your dataset and test data
seg_path = './dataset/seg'
test_path = './dataset/seg_test'

# Helper functions
def calculate_histogram(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])
    cv2.normalize(hist_r, hist_r)
    cv2.normalize(hist_g, hist_g)
    cv2.normalize(hist_b, hist_b)
    hist = np.concatenate((hist_r, hist_g, hist_b)).flatten()
    return hist

def compare_hist(hist1, hist2):
    distance = np.sqrt(np.sum((hist1 - hist2) ** 2))
    return distance

def find_similar_images(hist_input, load_hist):
    results = []
    for hist in load_hist:
        dir_name = hist[0]
        hist_data = hist[1]
        distance = compare_hist(hist_input, hist_data)
        results.append((dir_name, distance))
    results.sort(key=lambda x: x[1])
    return results[:10]

# Load the precomputed histograms from the pickle file
pickle_path = 'features.pkl'
with open(pickle_path, 'rb') as f:
    load_hist = pickle.load(f)

# Streamlit interface
st.title("Image Similarity Finder")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Decode the uploaded image as a NumPy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Convert the file bytes into an OpenCV image
    image = cv2.imdecode(file_bytes, 1)  # 1 means reading the image in color mode

    # Check if the image was loaded successfully
    if image is None:
        st.error("Error loading the image. Please upload a valid image file.")
    else:
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', channels="BGR", use_column_width=True)

        # Calculate the histogram of the uploaded image
        hist_input = calculate_histogram(image)

        # Find similar images
        similar_images = find_similar_images(hist_input, load_hist)

        st.write("Top 10 similar images:")
        cols = st.columns(5)
        for i, (dir_name, distance) in enumerate(similar_images):
            img = cv2.imread(dir_name)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extract folder name and file name from the directory path
            folder_name = os.path.basename(os.path.dirname(dir_name))
            file_name = os.path.basename(dir_name)

            # Display the image and include the folder name in the caption
            cols[i % 5].image(img_rgb, caption=f"File: {folder_name}\nScore: {distance:.2f}")

            
