import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
st.set_page_config(page_title="GreenBuddy", page_icon="🌱", layout="wide")
@st.cache_resource
def load(): return tf.keras.models.load_model("greenbuddy_final.keras")
model = load()
class_names = ["Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot", "Tomato_healthy"]
tips = {"Bacterial_spot": "Remove infected leaves, avoid overhead watering, use copper-based spray.", "Early_blight": "Remove lower infected leaves, mulch the soil, apply a copper or organic fungicide.", "Late_blight": "Remove and bin infected plants quickly, improve airflow, apply fungicide early.", "Leaf_Mold": "Lower humidity, space plants out, water at the base, remove infected leaves.", "Septoria_leaf_spot": "Remove spotted leaves, avoid wetting foliage, rotate crops yearly.", "Tomato_healthy": "Plant looks healthy. Keep regular watering and 6+ hours of sunlight."}
looks = {"Bacterial_spot": "Small dark, water-soaked spots, often with a yellow halo.", "Early_blight": "Brown spots with ring patterns on older, lower leaves.", "Late_blight": "Large dark, water-soaked patches that spread fast in damp weather.", "Leaf_Mold": "Pale yellow spots on top and olive-gray mold underneath.", "Septoria_leaf_spot": "Many small round spots with gray centers and dark edges.", "Tomato_healthy": "Even green leaf with no spots."}
if "log" not in st.session_state: st.session_state.log = []
st.title("🌱 GreenBuddy: Tomato Leaf Disease Detector")
st.caption("AI-powered diagnosis using a MobileNetV2 CNN (test accuracy 87.4%) · Class 12 AI Capstone")
st.sidebar.header("How to use")
st.sidebar.write("1. Place ONE tomato leaf on a plain dark background.\n\n2. Take a close, well-lit photo.\n\n3. Upload it and read the result.")
lib = st.sidebar.expander("🌿 Plant Library: Tomato basics")
lib.markdown("**Sunlight:** 6-8 hours of direct sun daily.\n\n**Watering:** deep watering at the base, about 2-3 times a week (adjust to your weather). Keep soil evenly moist, not soggy. Water in the morning.\n\n**Soil:** loose, well-drained, rich in compost, pH 6.0-6.8.\n\n**Spacing:** 45-60 cm apart, with stakes or cages.\n\n**Fertilizer:** compost or a balanced fertilizer at planting, then one higher in potassium once flowers appear. Avoid too much nitrogen.\n\n**Temperature:** best between 18 and 29 °C.\n\n**Prevent disease:** rotate crops yearly, leave space for airflow, avoid wetting leaves, remove lower leaves touching the soil.")
dis = st.sidebar.expander("🦠 Common tomato diseases")
dis.markdown("**Early blight:** brown spots with rings on older, lower leaves.\n\n**Late blight:** large dark, water-soaked patches; spreads fast in damp weather.\n\n**Leaf mold:** pale yellow spots on top, olive-gray mold underneath.\n\n**Septoria leaf spot:** many small round spots with gray centers and dark edges.\n\n**Bacterial spot:** small dark, water-soaked spots, often with a yellow halo.\n\n**Healthy:** even green leaf, no spots.")
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
_ = c2.write("**What it looks like:** " + looks[top]) if f else None
_ = c2.info("Care tip: " + tips[top]) if f else None
_ = c2.error("Low confidence: please retake the photo on a plain dark background.") if f and conf < 50 else None
_ = c2.write("**Top 3 predictions**") if f else None
_ = [c2.progress(float(p[i]), text=class_names[i].replace("_", " ") + ": " + str(round(float(p[i]) * 100, 1)) + "%") for i in np.argsort(p)[::-1][:3]] if f else None
_ = st.bar_chart(dict(zip(class_names, [float(v) for v in p]))) if f else None
note = st.text_input("📝 Add a note about this leaf (press Enter), then tap Save") if f else ""
save = st.button("📌 Save to My Garden") if f else False
_ = st.session_state.log.append({"Leaf file": f.name, "Diagnosis": top.replace("_", " "), "Confidence %": round(conf, 1), "Note": note}) if save else None
_ = st.subheader("📒 My Garden log") if st.session_state.log else None
_ = st.dataframe(pd.DataFrame(st.session_state.log)) if st.session_state.log else None
_ = st.download_button("Download log (CSV)", pd.DataFrame(st.session_state.log).to_csv(index=False), "greenbuddy_log.csv") if st.session_state.log else None
_ = st.info("Upload a leaf photo above to get a diagnosis.") if not f else None
