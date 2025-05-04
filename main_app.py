import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

model = load_model('C:\Users\Aryush\Desktop\PlantDiseasePrediction\plant_disease.h5')

#Name of class
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

#SETTING TITLE OF APP
st.title('Plant Disease Prediction')
st.markdown("Upload an image of the plant leaf")

#uploading the plant image
plant_image = st.file_uploader("Choose an image...", type = jpg)
submit = st.button('Predict')

#on predict button click
if submit:

    if plant_image is not None:

        #Convert the file to an opencv image 
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)


        #Displaying the image
        st.image(opencv_image, "BGR")
        st.write(opencv_image.shape)

        #Resizingthe image 
        opencv_image = cv2.resize(opencv_image, (256,256))

        #convertimage to 4 Dimension
        opencv_image.shape = (1,256,256,3)

        #Make prediction

        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This is" +result.split('-')[0]+ "leaf with " + result.split('-')[1]))