import streamlit as st
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
st.title("paddy crop disease detection ")
st.write("Upload an image of rice to detect the disease ")
MODEL_PATH = "C:/Users/karur/Downloads/vgg16_rice_model.h5"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
model = tf.keras.models.load_model("your_large_model.h5")
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
    st.write(f"Confidence: {np.max(predictions)*100:.2f}")
    if classes[np.argmax(predictions)] == 'Bacterialblight' :
        st.write('''1. Bacterial Blight Caused by Bacteria - Xanthomonas oryzae What to Spray: Streptocycline (200 ppm) or Copper-based fungicides like Copper oxychloride.''')
        st.write('''Fertilizers to Use: Apply potassium-rich fertilizers and reduce nitrogen usage, as excess nitrogen increases bacterial activity.''')
        st.write('''Specific Precautions:''')
        st.write('''1. Drain excess water from the field to prevent further spread.''')
        st.write('''2. Remove and burn infected plants to stop the bacteria from spreading to healthy plants.''')
        st.write('''3. Avoid working in the field when plants are wet, as the bacteria spread through water droplets.''')
        st.write('''4. Use organic compost or farmyard manure to improve soil health and increase plant resistance.''')
        st.write('''5.Keep a gap between plants to allow proper airflow, which reduces bacterial growth.''')
    if classes[np.argmax(predictions)] == 'Blast' :
        st.write('''2. Blast (Caused by Fungus - Magnaporthe oryzae)''')
        st.write('''What to Spray: Tricyclazole (75 WP), Carbendazim, or Mancozeb.''')
        st.write('''Fertilizers to Use: Apply phosphorus and potassium-rich fertilizers while avoiding excess nitrogen.''')
        st.write('''Specific Precautions:''')
        st.write('''1. Remove infected leaves and plants to prevent the fungus from spreading.''')
        st.write('''2. Maintain proper spacing between plants for good airflow and reduce moisture buildup.''')
        st.write('''3. Avoid overwatering the field, as high moisture helps the fungus grow.''')
        st.write('''4. Use organic methods like applying neem cake or ash to control fungal growth naturally.''')
        st.write('''5. Rotate crops with non-rice plants to break the fungal cycle in the soil.''')
    if classes[np.argmax(predictions)] == 'Brownspot' :
        st.write('''Brown Spot (Caused by Fungus - Bipolaris oryzae)''')
        st.write('''What to Spray: Propiconazole, Mancozeb, or Hexaconazole.''')
        st.write('''Fertilizers to Use: Use potassium and zinc fertilizers to improve plant immunity.''')
        st.write('''Specific Precautions:''')
        st.write('''1. Improve soil drainage to prevent water stagnation, which helps the fungus grow.''')
        st.write('''2. Remove infected plants and burn them to avoid further spread.''')
        st.write('''3. Treat seeds with fungicide before replanting in the next season.''')
        st.write('''4. Apply compost or organic manure to improve soil health and boost plant strength.''')
        st.write('''5. Regularly monitor the field and immediately remove infected patches. ''')
    if classes[np.argmax(predictions)] == 'Tungro' :
        st.write('''Tungro Virus (Caused by Virus, Spread by Green Leafhopper)''')
        st.write('''What to Spray: Imidacloprid or Thiamethoxam to control the Green Leafhopper, which spreads the virus.''')
        st.write('''Fertilizers to Use: Balanced NPK fertilizers with extra potassium to increase plant strength.''')
        st.write('''Specific Precautions:''')
        st.write('''1.Remove and destroy infected plants to prevent the virus from spreading to healthy crops.''')
        st.write('''2.Control weeds and grasses around the field, as they serve as hiding spots for the virus-carrying insects.''')
        st.write('''3. Use yellow sticky traps or light traps to catch the Green Leafhopper.''')
        st.write('''4. Avoid planting near fields that are already infected.''')
        st.write('''5. Regularly spray insecticides to prevent further infection during the early crop stages.''')

