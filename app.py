import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
st.title("GreenBuddy: Tomato Leaf Disease Detector")
st.write("For best results, place ONE tomato leaf on a plain dark background, close up, in even daylight.")
@st.cache_resource
def load(): return tf.keras.models.load_model("greenbuddy_final.keras")
model = load()
class_names = ["Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot", "Tomato_healthy"]
tips = {"Bacterial_spot": "Remove infected leaves, avoid overhead watering, use copper-based spray.", "Early_blight": "Remove lower infected leaves, mulch the soil, apply a copper or organic fungicide.", "Late_blight": "Remove and bin infected plants quickly, improve airflow, apply fungicide early.", "Leaf_Mold": "Lower humidity, space plants out, water at the base, remove infected leaves.", "Septoria_leaf_spot": "Remove spotted leaves, avoid wetting foliage, rotate crops yearly.", "Tomato_healthy": "Plant looks healthy. Keep regular watering and 6+ hours of sunlight."}
f = st.file_uploader("Upload a tomato leaf photo", type=["jpg", "jpeg", "png"])
img = Image.open(f).convert("RGB") if f else None
p = model.predict(np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), 0), verbose=0)[0] if f else None
top = class_names[int(np.argmax(p))] if f else None
_ = st.image(img, width=300) if f else None
_ = st.subheader("Diagnosis: " + top.replace("_", " ") + " (" + str(round(float(p.max()) * 100, 1)) + "% confidence)") if f else None
_ = st.write("Care tip: " + tips[top]) if f else None
_ = st.bar_chart(dict(zip(class_names, [float(v) for v in p]))) if f else None
