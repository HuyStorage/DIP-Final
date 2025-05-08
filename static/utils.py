import cv2
import numpy as np
import os
import streamlit as st
import base64


def center_crop_resize(image, target_size):
    height, width = image.shape[:2]
    crop_size = min(height, width)
    y = (height - crop_size) // 2
    x = (width - crop_size) // 2
    cropped_image = image[y : y + crop_size, x : x + crop_size]
    resized_image = cv2.resize(cropped_image, target_size)
    return resized_image


def standardize_image(image, target_size):
    M, N, C = image.shape
    max_dim = max(M, N)
    image_out = np.zeros((max_dim, max_dim, C), np.uint8)
    image_out[:M, :N, :] = image[:, :, :]
    image_out = cv2.resize(image_out, target_size)
    return image_out


def standardize_image_gray(image, target_size):
    M, N = image.shape
    max_dim = max(M, N)
    image_out = np.zeros((max_dim, max_dim), np.uint8)
    image_out[:M, :N] = image[:, :]
    image_out = cv2.resize(image_out, target_size)
    return image_out

def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def local_image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
def get_file_path(file_name):
    static_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(static_dir, ".."))
    model_dir = os.path.join(base_dir, "models")
    return os.path.join(model_dir, file_name)

def get_path(file_name):
    static_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(static_dir, ".."))
    return os.path.join(base_dir, file_name)