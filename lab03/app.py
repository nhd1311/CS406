import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = np.array(gray, dtype=float)  # Convert to float for better precision
    
    # Sobel
    sobelx = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    
    # Prewitt
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewittx = cv2.filter2D(gray_float, -1, kernelx)
    prewitty = cv2.filter2D(gray_float, -1, kernely)
    prewitt = cv2.magnitude(prewittx, prewitty)  # Ensure dtype is float for correct magnitude
    
    # Canny
    canny = cv2.Canny(gray, 100, 200)
    
    return sobel, prewitt, canny

def display_images(image):
    # Denoise the image
    denoised_image = denoise_image(image)
    
    # Sharpen the image
    sharpened_image = sharpen_image(image)
    
    # Edge detection
    sobel, prewitt, canny = edge_detection(image)
    
    # Convert to RGB for displaying in Streamlit (as Streamlit expects images in RGB format)
    denoised_image_rgb = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
    sharpened_image_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)
    original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display images using matplotlib (for grayscale)
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns
    
    ax[0, 0].imshow(original_image_rgb)
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(denoised_image_rgb)
    ax[0, 1].set_title("Denoised Image")
    ax[0, 1].axis("off")

    ax[0, 2].imshow(sharpened_image_rgb)
    ax[0, 2].set_title("Sharpened Image")
    ax[0, 2].axis("off")
    
    ax[1, 0].imshow(sobel, cmap='gray')
    ax[1, 0].set_title("Sobel Edge Detection")
    ax[1, 0].axis("off")
    
    ax[1, 1].imshow(prewitt, cmap='gray')
    ax[1, 1].set_title("Prewitt Edge Detection")
    ax[1, 1].axis("off")
    
    ax[1, 2].imshow(canny, cmap='gray')
    ax[1, 2].set_title("Canny Edge Detection")
    ax[1, 2].axis("off")
    
    st.pyplot(fig)

# Streamlit app code
st.title("Image Processing: Denoise, Sharpen & Edge Detection")

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

        # Display processed images
        st.header("Processed Images")
        display_images(image)
