import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model('plant_disease.h5')

# Name of classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Set title of app
st.title('🌿 Plant Disease Prediction')
st.markdown("Upload an image of the plant leaf (.jpg or .jpeg only)")

# Upload image
plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
submit = st.button('Predict')

# On Predict button click
if submit:
    if plant_image is not None:
        filename = plant_image.name.lower()

        if filename.endswith(('.jpg', '.jpeg')):
            # Convert to OpenCV image
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Show the image
            st.image(opencv_image, channels="BGR", caption="Uploaded Leaf Image")
            st.write(f"Image shape: {opencv_image.shape}")

            # Resize and prepare for prediction
            opencv_image = cv2.resize(opencv_image, (256, 256))
            opencv_image = opencv_image.reshape(1, 256, 256, 3)

            # Make prediction
            Y_pred = model.predict(opencv_image)
            result = CLASS_NAMES[np.argmax(Y_pred)]

            # Show prediction
            st.title(f"This is a {result.split('-')[0]} leaf with {result.split('-')[1]}")
        else:
            st.error("Invalid file type. Please upload a .jpg or .jpeg image.")
    else:
        st.warning("Please upload an image before clicking Predict.")
