import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np

st.title("📸 Human Emotion Detector (Image Upload)")
st.write("Upload an image and let AI detect age, gender, race, and emotion.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy array
    img_array = np.array(image)

    # Analyze with DeepFace
    if st.button("Analyze"):
        result = DeepFace.analyze(
            img_path=img_array,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True
        )

        st.subheader("Results:")
        st.write(f"Age: {result[0]['age']}")
        st.write(f"Gender: {result[0]['dominant_gender']}")
        st.write(f"Race: {result[0]['dominant_race']}")
        st.write(f"Emotion: {result[0]['dominant_emotion']}")
