import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# =======================
# Path model
# =======================
MODEL_PATH = "skin10_mobilenet.keras"

@st.cache_resource
def load_skin_model():
    model = load_model(MODEL_PATH)
    return model

model = load_skin_model()

# =======================
# Sidebar menu
# =======================
menu = ["Home", "Classification", "Kritik & Saran", "About Us"]
choice = st.sidebar.selectbox("Menu", menu)

# =======================
# Home dengan animasi blur
# =======================
if choice == "Home":
    st.components.v1.html("""
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');
        body {
            background-color: #FFF4E1;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 90vh;
            font-family: 'Poppins', sans-serif;
            color: #333;
        }
        .container {
            text-align: center;
            animation: fadeIn 3s ease forwards;
        }
        .title {
            font-size: 5em;
            font-weight: 700;
            color: #FF8C42;
            opacity: 0;
            animation: blurFade 2s 0.5s forwards;
        }
        .subtitle {
            font-size: 1.5em;
            margin-top: 20px;
            opacity: 0;
            animation: blurFade 2s 2.5s forwards;
        }
        @keyframes blurFade {
            0% { opacity:0; filter: blur(10px); }
            100% { opacity:1; filter: blur(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
    </head>
    <body>
        <div class="container">
            <div class="title">SKINFIRST 🩺</div>
            <div class="subtitle">Aplikasi AI untuk membantu deteksi penyakit kulit secara dini</div>
        </div>
    </body>
    </html>
    """, height=600)

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
        classes = ["Eksim", "Gigitan Serangga", "Jerawat",
                   "Kandidiasis (Infeksi Jamur Candida)", "Kanker Kulit",
                   "Keratosis Seboroik", "Kurap", "Psoriasis",
                   "Tumor Jinak Kulit", "Vitiligo"]
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
        st.success("Terima kasih atas kritik & saran Anda! 😊")

# =======================
# About Us
# =======================
elif choice == "About Us":
    st.header("About Us")
    st.write("""
        SKINFIRST dibuat oleh tim PKM UNDIP.  
        Tujuannya membantu masyarakat mendeteksi penyakit kulit secara dini menggunakan AI.  
        ❤️👩‍⚕️👨‍⚕️
    """)
