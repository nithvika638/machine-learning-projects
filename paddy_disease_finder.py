import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
st.title("paddy crop disease detection ")
st.write("Upload an image of rice to detect the disease ")
MODEL_PATH = "C:/Users/karur/Downloads/vgg16_rice_model.h5"
model = load_model(MODEL_PATH)
classes = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']
upld_img = st.file_uploader("Choose an image ", type=["jpg", "png", "jpeg"])
if upld_img is not None:
    img = Image.open(upld_img).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0  
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.write("processing.. !")
    predictions = model.predict(img_array)
    st.write(f"Prediction: {classes[np.argmax(predictions)]}")
    st.write(f"Confidence: {np.max(predictions):.2f}")


