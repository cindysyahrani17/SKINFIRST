import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

# =======================
# Path model
# =======================
MODEL_PATH = "skin10_mobilenet.keras"
FILE_ID = "1G0X-463Ni0b1vYga_AhuKWTdQ47nkOwI"  # File ID Google Drive
URL = f"https://drive.google.com/uc?id={FILE_ID}&export=download"

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model, please wait...")
    gdown.download(URL, MODEL_PATH, quiet=False)
    st.success("Model downloaded!")

# =======================
# Load model
# =======================
@st.cache_resource
def load_skin_model():
    model = load_model(MODEL_PATH)
    return model

model = load_skin_model()

# =======================
# Judul Aplikasi
# =======================
st.title("SKINFIRST")
st.write("Aplikasi klasifikasi penyakit kulit berbasis AI")

# =======================
# Sidebar menu
# =======================
menu = ["Home", "Classification", "Kritik & Saran", "About Us"]
choice = st.sidebar.selectbox("Menu", menu)

# =======================
# Home
# =======================
if choice == "Home":
    st.header("Selamat Datang di SKINFIRST!")
    st.write("""
        SKINFIRST adalah aplikasi AI untuk membantu mengklasifikasi penyakit kulit.
        Upload foto kulit, dan sistem akan memprediksi kemungkinan penyakitnya.
    """)

# =======================
# Classification
# =======================
elif choice == "Classification":
    st.header("Upload Gambar Kulit Anda")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Gambar Anda', use_column_width=True)
        
        # Preprocessing
        img = image.load_img(uploaded_file, target_size=(224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Prediksi
        pred = model.predict(img_array)
        classes = [
            "Eksim", "Gigitan Serangga", "Jerawat",
            "Kandidiasis (Infeksi Jamur Candida)", "Kanker Kulit",
            "Keratosis Seboroik", "Kurap", "Psoriasis",
            "Tumor Jinak Kulit", "Vitiligo"
        ]
        class_idx = np.argmax(pred)
        confidence = pred[0][class_idx]
        
        st.write(f"Prediksi: **{classes[class_idx]}**")
        st.write(f"Confidence: {confidence*100:.2f}%")

# =======================
# Kritik & Saran
# =======================
elif choice == "Kritik & Saran":
    st.header("Kritik & Saran")
    feedback = st.text_area("Tulis kritik atau saran Anda di sini:")
    if st.button("Kirim"):
        st.success("Terima kasih atas kritik & saran Anda!")

# =======================
# About Us
# =======================
elif choice == "About Us":
    st.header("About Us")
    st.write("""
        SKINFIRST dibuat oleh tim PKM UNDIP. 
        Tujuannya adalah membantu masyarakat dalam mendeteksi penyakit kulit secara dini menggunakan AI.
    """)
