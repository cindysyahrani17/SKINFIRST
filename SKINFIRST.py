#SKINFIRST - Streamlit App
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="SKINFIRST", page_icon="💉", layout="wide")

MODEL_PATH = "D:/SEMESTER 3/PKM/skin10_mobilenet.keras"
CLASS_NAMES = [
    "Eksim", "Gigitan Serangga", "Jerawat", "Kandidiasis (Infeksi Jamur Candida)",
    "Kanker Kulit", "Keratosis Seboroik", "Kurap", "Psoriasis",
    "Tumor Jinak Kulit", "Vitiligo"
]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_skin_model():
    model = load_model(MODEL_PATH)
    return model

model = load_skin_model()

# =========================
# SIDEBAR MENU
# =========================
menu = ["Home", "Classification", "Kritik & Saran", "About Us"]
choice = st.sidebar.selectbox("Menu", menu)

# =========================
# HOME PAGE
# =========================
if choice == "Home":
    st.title("💉 SKINFIRST")
    st.markdown("""
    Selamat datang di SKINFIRST!  
    Aplikasi ini dapat membantu memprediksi penyakit kulit dari gambar.
    
    **Fitur:**  
    - Upload foto kulit
    - Prediksi penyakit kulit
    - Statistik probabilitas tiap kelas
    - Kritik & saran
    """)
    st.image("https://images.unsplash.com/photo-1588776814546-00c7967c5ff0?auto=format&fit=crop&w=800&q=80", use_column_width=True)

# =========================
# CLASSIFICATION PAGE
# =========================
elif choice == "Classification":
    st.title("📷 Skin Classification")
    uploaded_file = st.file_uploader("Upload gambar kulit", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar untuk prediksi", use_column_width=True)
        
        # Load image
        img = image.load_img(uploaded_file, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)/255.0
        
        # Prediction with spinner
        with st.spinner("Sedang memprediksi..."):
            preds = model.predict(x)
            class_idx = np.argmax(preds)
            confidence = preds[0][class_idx]
            st.success(f"Hasil Prediksi: **{CLASS_NAMES[class_idx]}** ({confidence*100:.2f}%)")
            
            # Probabilitas semua kelas
            st.subheader("Probabilitas tiap kelas:")
            prob_df = pd.DataFrame(preds, columns=CLASS_NAMES)
            st.bar_chart(prob_df.T)

# =========================
# KRITIK & SARAN PAGE
# =========================
elif choice == "Kritik & Saran":
    st.title("✍ Kritik & Saran")
    with st.form("feedback_form"):
        nama = st.text_input("Nama")
        kritik = st.text_area("Kritik / Saran")
        submitted = st.form_submit_button("Kirim")
        if submitted:
            df_path = "feedback.csv"
            feedback_df = pd.DataFrame([[nama, kritik]], columns=["Nama", "Kritik/Saran"])
            if os.path.exists(df_path):
                feedback_df.to_csv(df_path, mode='a', header=False, index=False)
            else:
                feedback_df.to_csv(df_path, index=False)
            st.success("Terima kasih atas feedbacknya!")

# =========================
# ABOUT US PAGE
# =========================
elif choice == "About Us":
    st.title("👨‍💻 About Us")
    st.markdown("""
    **SKINFIRST Team**  
    - Nama 1 (Developer)  
    - Nama 2 (Designer)  
    - Nama 3 (Tester)  

    **Kontak:**  
    - Email: skinfo_app@example.com  
    - GitHub: [SKINFIRST](https://github.com)
    
    **Disclaimer:**  
    Hasil prediksi hanya bersifat **informasi**, bukan pengganti konsultasi dokter.
    """)
