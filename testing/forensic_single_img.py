import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

IMG_SIZE = 128

# --- Preprocessing Functions ---
def generate_ela_image(image, quality=90):
    temp_filename = 'temp_ela.jpg'
    cv2.imwrite(temp_filename, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    ela_img = cv2.imread(temp_filename)
    ela_image = cv2.absdiff(image, ela_img)
    ela_image = cv2.normalize(ela_image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return ela_image

def generate_fft_image(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    fft_image = cv2.cvtColor(magnitude_spectrum.astype('uint8'), cv2.COLOR_GRAY2BGR)
    return fft_image.astype('float32') / 255.0

def preprocess_single_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    rgb = image.astype('float32') / 255.0
    ela = generate_ela_image(image)
    fft = generate_fft_image(image)
    return np.expand_dims(rgb, axis=0), np.expand_dims(ela, axis=0), np.expand_dims(fft, axis=0)

# --- Grad-CAM ---
def get_grad_cam(model, inputs, class_index=0, layer_name='rgb_conv3'):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-6
    return heatmap.numpy()

def display_gradcam(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert BGR to RGB (for original image & overlay)
    original_img_rgb = cv2.cvtColor((original_img * 255).astype('uint8'), cv2.COLOR_BGR2RGB)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend original and heatmap
    superimposed = cv2.addWeighted(original_img_rgb, 0.6, heatmap_color_rgb, 0.4, 0)

    # Show with matplotlib
    plt.imshow(superimposed)
    plt.title("Grad-CAM Visualization")
    plt.axis('off')
    plt.show()


# --- Load model ---
model = load_model(r"C:\Users\yuvip\OneDrive\Desktop\mini_project\models\forensic_model.h5")

# --- Test a single image ---
image_path = r"C:\Users\yuvip\Downloads\WhatsApp Image 2025-04-24 at 11.38.26 AM.jpeg"
rgb, ela, fft = preprocess_single_image(image_path)

# Predict
prediction = model.predict([rgb, ela, fft])[0][0]
label = 'Fake' if prediction > 0.5 else 'Real'
print(f"Predicted Label: {label} (Confidence: {prediction:.2f})")
#print(prediction.shape)

# Grad-CAM
heatmap = get_grad_cam(model, [rgb, ela, fft], class_index=0, layer_name='rgb_pool3')
display_gradcam(rgb[0], heatmap)
