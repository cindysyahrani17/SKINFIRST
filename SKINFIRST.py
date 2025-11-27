import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import os
from datetime import datetime
import time

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(page_title="SKINFIRST", page_icon="🩺", layout="wide")

# =======================
# LOAD MODEL
# =======================
MODEL_PATH = "skin10_mobilenet.keras"

@st.cache_resource
def load_skin_model():
    model = load_model(MODEL_PATH)
    return model

model = load_skin_model()

classes = ["Eksim", "Gigitan Serangga", "Jerawat",
           "Kandidiasis (Infeksi Jamur Candida)", "Kanker Kulit",
           "Keratosis Seboroik", "Kurap", "Psoriasis",
           "Tumor Jinak Kulit", "Vitiligo"]

# =======================
# IMAGE PREPROCESS
# =======================
def preprocess(img):
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# =======================
# HORIZONTAL NAVBAR
# =======================
selected = option_menu(
    menu_title=None,
    options=["Home", "Classification", "Kritik & Saran", "About Us"],
    icons=["house", "folder", "pencil", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",  # NAVBAR DI ATAS
    styles={
        "container": {"padding": "0!important", "background-color": "#FFF4E1"},
        "icon": {"color": "#5C4033", "font-size": "18px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "color": "#5C4033", "padding": "10px 20px"},
        "nav-link-selected": {"background-color": "#A3672E", "color": "white", "font-weight": "bold"},
    }
)

# =======================
# HOME PAGE
# =======================
if selected == "Home":
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');
    body {background-color: #FFF4E1; font-family: 'Poppins', sans-serif;}
    .centered {display: flex; flex-direction: column; justify-content: center; align-items: center; height: 80vh; text-align: center; overflow: hidden; position: relative;}
    .logo-text {font-size: 70px; font-weight: 800; background: linear-gradient(90deg, #A0522D, #D2B48C); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 3px; opacity: 0; animation: blurFadeIn 3s ease-in-out forwards; animation-delay: 1.5s;}
    .slogan {font-size: 22px; color: #5C4033; margin-top: 20px; font-weight: bold; opacity: 0; animation: blurFadeIn 3s ease-in-out forwards; animation-delay: 3.5s;}
    .welcome {font-size: 42px; font-weight: bold; color: #5C4033; animation: blurFadeIn 3s ease-in-out;}
    @keyframes blurFadeIn {0% {opacity:0; filter: blur(20px);} 100% {opacity:1; filter: blur(0);}}
    </style>
    """, unsafe_allow_html=True)

    container = st.empty()
    with container:
        st.markdown("""
        <div class="centered">
            <div class="welcome">Welcome 👋</div>
        </div>
        """, unsafe_allow_html=True)
    time.sleep(2)
    container.empty()
    st.markdown("""
    <div class="centered">
        <div class="logo-text">SKINFIRST 🩺</div>
        <p class="slogan">PROTECT YOUR SKIN FIRST</p>
    </div>
    """, unsafe_allow_html=True)

# =======================
# CLASSIFICATION PAGE
# =======================
elif selected == "Classification":
    st.title("🖼️ Upload Gambar Kulit Anda")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar Anda", use_column_width=True)
        img_array = preprocess(img)
        pred = model.predict(img_array)
        class_idx = np.argmax(pred)
        confidence = pred[0][class_idx]
        st.success(f"Prediksi: **{classes[class_idx]}**")
        st.info(f"Confidence: {confidence*100:.2f}%")

# =======================
# KRITIK & SARAN PAGE
# =======================
elif selected == "Kritik & Saran":
    st.title("📩 Kritik & Saran")
    feedback = st.text_area("Tulis kritik/saran Anda di sini...")
    if st.button("Kirim"):
        if feedback.strip() == "":
            st.warning("⚠️ Tolong isi dulu kritik/sarannya.")
        else:
            new_data = pd.DataFrame([{
                "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "feedback": feedback
            }])
            if os.path.exists("feedback.csv"):
                old_data = pd.read_csv("feedback.csv")
                all_data = pd.concat([old_data, new_data], ignore_index=True)
            else:
                all_data = new_data
            all_data.to_csv("feedback.csv", index=False)
            st.success("✅ Terima kasih! Kritik & saran Anda sudah tersimpan.")

# =======================
# ABOUT US PAGE
# =======================
elif selected == "About Us":
    st.title("ℹ️ About SKINFIRST")
    st.markdown("""
    **SKINFIRST** dibuat oleh tim PKM Universitas Diponegoro.  
    Sistem Deteksi Dini Penyakit Kulit Masyarakat Indonesia ❤️🩺👩‍⚕️👨‍⚕️  
    """)
