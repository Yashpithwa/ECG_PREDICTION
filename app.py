# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from gradcam_utils import get_gradcam_heatmap, overlay_gradcam

# Load model
model = tf.keras.models.load_model("ecg_model.h5")
class_names = ["Normal", "Abnormal", "MIP", "MI"]

st.set_page_config(page_title="ECG Classification", layout="wide")
st.title("🩺 ECG Report Classification with Explainable AI")

uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    confidence = preds[0][pred_class] * 100

    st.markdown(f"### 🧾 Prediction: **{class_names[pred_class]}** ({confidence:.2f}%)")

    # Grad-CAM
    heatmap = get_gradcam_heatmap(model, img_array, model.layers[-5].name)
    gradcam_img = overlay_gradcam(uploaded_file.name, heatmap)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original ECG Image", use_column_width=True)
    with col2:
        st.image(gradcam_img, caption="Explainable AI (Grad-CAM)", use_column_width=True)
