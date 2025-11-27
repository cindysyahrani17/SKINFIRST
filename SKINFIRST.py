import streamlit as st
from streamlit.components.v1 import html
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

st.set_page_config(page_title="SKINFIRST", page_icon="🧴")

# =======================
# Path & Model
# =======================
MODEL_PATH = "skin10_mobilenet.keras"

@st.cache_resource
def load_skin_model():
    model = load_model(MODEL_PATH)
    return model

model = load_skin_model()

# =======================
# Sidebar Menu
# =======================
menu = ["Home", "Classification", "Kritik & Saran", "About Us"]
choice = st.sidebar.selectbox("Menu", menu)

# =======================
# Home with Transition
# =======================
if choice == "Home":
    st.header("")  # kosong dulu biar layout rapi

    # HTML + CSS + JS untuk animasi blur + fade in
    home_html = """
    <style>
        body {
            background-color: #fff5e6;
            font-family: 'Arial', sans-serif;
        }
        .container {
            text-align: center;
            margin-top: 100px;
        }
        h1 {
            font-size: 70px;
            color: #d2691e;
            filter: blur(10px);
            opacity: 0;
            transition: filter 1s ease, opacity 1s ease;
        }
        p {
            font-size: 24px;
            color: #555;
            filter: blur(5px);
            opacity: 0;
            transition: filter 1s ease, opacity 1s ease;
        }
    </style>

    <div class="container">
        <h1 id="title">SKINFIRST</h1>
        <p id="desc">Aplikasi AI untuk membantu mengklasifikasi penyakit kulit 💉🩺</p>
    </div>

    <script>
        // Step 1: muncul title
        setTimeout(() => {
            document.getElementById("title").style.filter = "blur(0px)";
            document.getElementById("title").style.opacity = "1";
        }, 500);

        // Step 2: muncul deskripsi setelah title muncul
        setTimeout(() => {
            document.getElementById("desc").style.filter = "blur(0px)";
            document.getElementById("desc").style.opacity = "1";
        }, 1500);
    </script>
    """

    html(home_html, height=400)

# =======================
# Classification
# =======================
elif choice == "Classification":
    st.header("Upload Gambar Kulit Anda")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Gambar Anda', use_column_width=True)
        
        img = image.load_img(uploaded_file, target_size=(224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
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
        st.success("Terima kasih atas kritik & saran Anda!")

# =======================
# About Us
# =======================
elif choice == "About Us":
    st.header("About Us")
    st.write("""
        SKINFIRST dibuat oleh tim PKM UNDIP.  
        Tujuannya adalah membantu masyarakat dalam mendeteksi penyakit kulit secara dini menggunakan AI. 💻🧴
    """)
