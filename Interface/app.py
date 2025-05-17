import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- 📛 Suppress oneDNN warnings ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- ⚙️ Constants ---
IMG_SIZE = 128
MODEL_PATH = r"C:\Users\yuvip\OneDrive\Desktop\mini_project\models\forensic_model.h5"

# --- 🔧 Utility Functions ---
def generate_ela_image(image, quality=90):
    temp_filename = 'temp_ela.jpg'
    cv2.imwrite(temp_filename, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    ela_img = cv2.imread(temp_filename)
    ela_image = cv2.absdiff(image, ela_img)
    ela_image = cv2.normalize(ela_image.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return ela_image

def generate_fft_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    fft_image = cv2.cvtColor(normalized.astype('uint8'), cv2.COLOR_GRAY2BGR)
    return fft_image.astype('float32') / 255.0

def preprocess_image(file) -> tuple:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    rgb = image.astype('float32') / 255.0
    ela = generate_ela_image(image)
    fft = generate_fft_image(image)
    return rgb, ela, fft, image

# --- 🔍 Grad-CAM ---
def get_grad_cam(model, inputs, layer_name='rgb_conv3'):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([np.expand_dims(x, axis=0) for x in inputs])
        loss = predictions[:, 0]  # Only one output neuron for sigmoid

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-6)
    return heatmap.numpy()

def display_gradcam(rgb_img, heatmap):
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    rgb_img = (rgb_img * 255).astype('uint8')
    superimposed = cv2.addWeighted(rgb_img, 0.6, heatmap_color, 0.4, 0)
    return superimposed

# --- 🎯 Load Model ---
model = load_model(MODEL_PATH)

# --- 🖥️ Streamlit App ---
st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("🕵️‍♂️ Deepfake Image Detection")
st.write("Upload an image to detect whether it is **real** or **fake** using multi-modal analysis (RGB + ELA + FFT) and Grad-CAM explanation.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # --- Preprocess ---
    rgb, ela, fft, orig = preprocess_image(uploaded_file)

    # --- Prediction ---
    preds = model.predict([np.expand_dims(rgb, axis=0),
                           np.expand_dims(ela, axis=0),
                           np.expand_dims(fft, axis=0)])
    prob = float(preds[0][0])
    label = "Fake" if prob >= 0.5 else "Real"
    confidence = prob * 100 if label == "Fake" else (1 - prob) * 100

    st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
    st.markdown(f"### 🔎 Prediction: **{label}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # --- Grad-CAM ---
    heatmap = get_grad_cam(model, inputs=[rgb, ela, fft])
    gradcam_img = display_gradcam(rgb, heatmap)
    st.image(cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB), caption="Grad-CAM Explanation", use_column_width=True)
