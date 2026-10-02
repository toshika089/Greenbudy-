import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
st.set_page_config(page_title="GreenBuddy", page_icon="🌱", layout="wide")
@st.cache_resource
def load(): return tf.keras.models.load_model("greenbuddy_final.keras")
model = load()
class_names = ["Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot", "Tomato_healthy"]
tips = {"Bacterial_spot": "Remove infected leaves, avoid overhead watering, use copper-based spray.", "Early_blight": "Remove lower infected leaves, mulch the soil, apply a copper or organic fungicide.", "Late_blight": "Remove and bin infected plants quickly, improve airflow, apply fungicide early.", "Leaf_Mold": "Lower humidity, space plants out, water at the base, remove infected leaves.", "Septoria_leaf_spot": "Remove spotted leaves, avoid wetting foliage, rotate crops yearly.", "Tomato_healthy": "Plant looks healthy. Keep regular watering and 6+ hours of sunlight."}
st.title("🌱 GreenBuddy: Tomato Leaf Disease Detector")
st.caption("AI-powered diagnosis using a MobileNetV2 CNN (test accuracy 87.4%) · Class 12 AI Capstone")
st.sidebar.header("How to use")
st.sidebar.write("1. Place ONE tomato leaf on a plain dark background.\n\n2. Take a close, well-lit photo.\n\n3. Upload it and read the result.")
st.sidebar.header("Detects")
st.sidebar.write("Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Healthy")
st.sidebar.warning("Limitation: trained on plain-background leaf images, so cluttered photos may give wrong results. Not a substitute for expert advice.")
f = st.file_uploader("Upload a tomato leaf photo", type=["jpg", "jpeg", "png"])
img = Image.open(f).convert("RGB") if f else None
p = model.predict(np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), 0), verbose=0)[0] if f else None
top = class_names[int(np.argmax(p))] if f else None
conf = float(p.max()) * 100 if f else 0
c1, c2 = st.columns(2)
_ = c1.image(img, caption="Your photo", width=350) if f else None
_ = (c2.success if top == "Tomato_healthy" else c2.warning)("Diagnosis: " + top.replace("_", " ")) if f else None
_ = c2.metric("Confidence", str(round(conf, 1)) + "%") if f else None
_ = c2.info("Care tip: " + tips[top]) if f else None
_ = c2.error("Low confidence: please retake the photo on a plain dark background.") if f and conf < 50 else None
_ = c2.write("**Top 3 predictions**") if f else None
_ = [c2.progress(float(p[i]), text=class_names[i].replace("_", " ") + ": " + str(round(float(p[i]) * 100, 1)) + "%") for i in np.argsort(p)[::-1][:3]] if f else None
_ = st.bar_chart(dict(zip(class_names, [float(v) for v in p]))) if f else None
_ = st.info("Upload a leaf photo above to get a diagnosis.") if not f else None
