import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

st.set_page_config(page_title="NeuroScan | Brain MRI Classifier", page_icon="🧠", layout="centered")

st.title("🧠 NeuroScan")
st.subheader("Brain MRI Image Classification using CNN")
st.write("Upload a brain MRI image to run the trained convolutional neural network.")

MODEL_PATH = Path("models/brain_tumor_cnn.keras")
CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
if model is None:
    st.warning("The trained model file is not included yet. Follow README.md to download the dataset and train the model.")
    st.stop()

uploaded = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded MRI image", use_container_width=True)
    img = image.resize((150, 150))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    st.success(f"Predicted class: **{CLASS_NAMES[idx]}**")
    st.write(f"Model score: **{probs[idx] * 100:.2f}%**")
    st.caption("Educational demonstration only. This model is not a medical diagnostic tool.")
    st.bar_chart({CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))})
