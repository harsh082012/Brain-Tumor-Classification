import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import gdown
import os

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# 🔽 Download model from Google Drive (only once)
@st.cache_resource
def load_my_model():
    file_id = "1nwjaUkV_w6S1UWMz0VbEr_GxgRFEAEZL"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "brain_tumor_final.keras"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    return load_model(output)

model = load_my_model()

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("🧠 Brain Tumor Detection")
st.write("Upload a Brain MRI image")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with st.spinner("🔍 Analyzing image..."):

        # Read image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # ✅ Resize FIRST
        image = image.resize((300, 300))
        img = np.array(image)

        # ✅ Validate image (no cv2 needed)
        gray = np.mean(img, axis=2)
        std_dev = np.std(gray)

        if std_dev < 10:
            st.error("❌ Not a valid MRI image")

        else:
            # Preprocess
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            # Predict
            pred = model.predict(img, verbose=0)
            class_idx = np.argmax(pred)
            confidence = float(np.max(pred))

            predicted_class = classes[class_idx]

            if confidence < 0.80:
                st.warning("⚠️ Uncertain prediction")
            else:
                st.success(f"Prediction: {predicted_class}")
                st.info(f"Confidence: {confidence*100:.2f}%")

            # Probabilities
            st.subheader("Class Probabilities")
            for i in range(len(classes)):
                st.write(f"{classes[i]}: {pred[0][i]*100:.2f}%")