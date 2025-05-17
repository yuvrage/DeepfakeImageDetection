import os

# --- 🛑 Suppress TensorFlow warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN to avoid numerical diff warnings

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf

IMG_SIZE = 128

# --- 📸 Utility Functions ---

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
    fft_image = fft_image.astype('float32') / 255.0
    return fft_image

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    rgb = img.astype('float32') / 255.0
    ela = generate_ela_image(img)
    fft = generate_fft_image(img)
    return rgb, ela, fft

def load_test_data(test_folder):
    rgb_list, ela_list, fft_list, labels = [], [], [], []
    for label_name in ['real', 'fake']:
        label_path = os.path.join(test_folder, label_name)
        label = 0 if label_name == 'real' else 1
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            try:
                rgb, ela, fft = preprocess_image(img_path)
                rgb_list.append(rgb)
                ela_list.append(ela)
                fft_list.append(fft)
                labels.append(label)
            except Exception as e:
                print(f"Skipped {filename}: {e}")
    return np.array(rgb_list), np.array(ela_list), np.array(fft_list), np.array(labels)

# --- 🔍 Grad-CAM ---

def get_grad_cam(model, inputs, class_index, layer_name='rgb_conv3'):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([np.expand_dims(x, axis=0) for x in inputs])
        loss = predictions[:, class_index]

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
    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Grad-CAM")
    plt.show()

# --- 🧪 Run Test ---

# Load model
model = load_model(r"C:\Users\yuvip\OneDrive\Desktop\mini_project\models\forensic_model.h5")

# ✅ Fix Warning #3: Compile to restore metrics
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Load and preprocess test data
test_folder = r"C:\Users\yuvip\OneDrive\Desktop\Datasetchsvghalofhuoarfl\Test"
X_rgb, X_ela, X_fft, y_true = load_test_data(test_folder)

# Predict
preds = model.predict([X_rgb, X_ela, X_fft])
y_pred = (preds > 0.5).astype(int).flatten()

# Classification report
print("Classification Report:\n", classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Show Grad-CAM on a few test images
for i in range(3):
    heatmap = get_grad_cam(
        model,
        inputs=[X_rgb[i], X_ela[i], X_fft[i]],
        class_index=0,
        layer_name='rgb_conv3'
    )
    display_gradcam(X_rgb[i], heatmap)
