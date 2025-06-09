import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

@st.cache_resource
def load_model_cached():
    return load_model("skin_lesion_model.h5")

model = load_model_cached()

label_map = {
    0: 'Melanocytic nevi (nv)',
    1: 'Melanoma (mel)',
    2: 'Benign keratosis-like lesions (bkl)',
    3: 'Basal cell carcinoma (bcc)',
    4: 'Actinic keratoses (akiec)',
    5: 'Vascular lesions (vasc)',
    6: 'Dermatofibroma (df)'
}

st.title("🩺 Skin Lesion Classifier 🩺")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and decode image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:
        # Show image preview
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        img_resized = cv2.resize(img, (64, 64))
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)

        # Predict
        prediction = model.predict(img_expanded)[0]
        predicted_index = int(np.argmax(prediction))
        predicted_label = label_map.get(predicted_index, "Unknown")
        confidence = float(prediction[predicted_index]) * 100

        # Show results
        st.success(f"🎯 **Predicted Class:** {predicted_label}")
        st.info(f" **Confidence:** {confidence:.2f}%")
    else:
        st.error("❌ Could not decode the uploaded image. Please upload a valid image file.")
