import streamlit as st

# Panggil set_page_config paling awal
st.set_page_config(page_title="Deteksi Pneumonia", page_icon="🩺", layout="centered")

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image, ImageDraw, ImageFont
import cv2

# Fungsi untuk memuat model dengan caching
@st.cache_resource
def load_keras_model(model_path):
    return load_model(model_path)

# Pastikan model h5 berada di direktori kerja yang sama
model_path = 'best_pneumonia_model_initial_labkom_VGG16.h5'
if not os.path.exists(model_path):
    st.error("Model file not found. Pastikan 'best_pneumonia_model_initial_labkom_VGG16.h5' ada di direktori kerja.")
    st.stop()

# Load model
model = load_keras_model(model_path)

# Parameter
img_height = 224
img_width = 224
last_conv_layer_name = 'block5_conv3'

# Fungsi untuk membuat heatmap Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = 0
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Fungsi prediksi dan pembuatan heatmap
def predict_and_generate_heatmap(img, model, last_conv_layer_name, preprocess_input, img_size=(img_height, img_width)):
    img_resized = img.resize(img_size)
    img_array = image.img_to_array(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_array_preprocessed = preprocess_input(img_array_expanded)

    preds = model.predict(img_array_preprocessed)
    prob = preds[0][0]
    label = 'Pneumonia' if prob > 0.5 else 'Normal'
    prob_percent = prob * 100 if prob > 0.5 else (1 - prob) * 100

    heatmap = make_gradcam_heatmap(img_array_preprocessed, model, last_conv_layer_name)

    original_width, original_height = img.size
    heatmap = cv2.resize(heatmap, (original_width, original_height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_array = np.array(img)
    superimposed_img = heatmap * 0.4 + original_array
    superimposed_img = Image.fromarray(np.uint8(superimposed_img))

    draw = ImageDraw.Draw(superimposed_img)
    font = ImageFont.load_default()
    text = f"Prediction: {label}, Probability: {prob_percent:.2f}%"
    draw.rectangle([(0, 0), (400, 20)], fill='black')
    draw.text((5, 5), text, fill='white', font=font)

    return superimposed_img, label, prob_percent

# Tampilan Halaman Utama
st.markdown("<h1 style='text-align: center; color: white;'>Deteksi Pneumonia dengan Grad-CAM</h1>", unsafe_allow_html=True)
st.write("### Unggah gambar X-Ray dada Anda untuk memprediksi apakah ada tanda-tanda **Pneumonia**.")

# Widget upload dan tombol prediksi
uploaded_file = st.file_uploader("Upload gambar (X-Ray):", type=["png", "jpg", "jpeg"])
predict_button = st.button("Predict")

if uploaded_file is not None and predict_button:
    # Lakukan prediksi dan buat heatmap
    img = Image.open(uploaded_file).convert("RGB")
    result_img, label, prob_percent = predict_and_generate_heatmap(img, model, last_conv_layer_name, preprocess_input)

    # Tampilkan hasil
    st.subheader(f"**Prediksi: {label}**")
    st.write(f"**Probabilitas: {prob_percent:.2f}%**")

    # Tampilkan gambar asli
    st.image(img, caption="Gambar Asli", use_column_width=True)

    # Tampilkan gambar dengan heatmap
    st.image(result_img, caption="Gambar dengan Heatmap Grad-CAM", use_column_width=True)
