import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import os

# --- CSS Kustom untuk Meniru Tampilan Bootstrap ---
custom_css = """
<style>
    /* Mengimpor font Poppins */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

    /* Mengatur latar belakang utama dan font */
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #f0f2f5;
    }

    /* Menargetkan blok utama Streamlit untuk dijadikan "kartu" */
    .stApp > header {
        background-color: transparent;
    }

    .main .block-container {
        font-family: 'Poppins', sans-serif;
        max-width: 700px; /* Lebar kartu */
        margin: 2rem auto;
        padding: 2rem;
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    
    /* Styling untuk judul */
    .main .block-container h1 {
        font-weight: 600;
        color: #1d2129;
        text-align: center;
    }

    /* Styling untuk teks deskripsi */
    .main .block-container p {
        color: #606770;
        text-align: center;
    }
    
    /* Styling untuk subheader */
    .main .block-container h3 {
        text-align: center;
        font-weight: 500;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Styling untuk pembatas "ATAU" */
     .main .block-container .stDivider {
        margin: 2rem 0;
     }

</style>
"""

# Menyuntikkan CSS ke dalam aplikasi
st.markdown(custom_css, unsafe_allow_html=True)


# --- JUDUL APLIKASI (sekarang akan di-style oleh CSS di atas) ---
st.title("🌿 Sistem Deteksi Sampah")
st.write("<p>Unggah gambar atau gunakan kamera untuk mendeteksi jenis sampah.</p>", unsafe_allow_html=True)


# --- MEMUAT MODEL ---
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

try:
    model = load_model('model/best.pt')
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


# --- FUNGSI PROSES DETEKSI ---
def process_and_display(image):
    st.image(image, caption="Gambar Asli", use_container_width=True)
    results = model(image)
    annotated_image = results[0].plot()
    annotated_image_rgb = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    st.image(annotated_image_rgb, caption="Hasil Deteksi", use_container_width=True)


# --- ANTARMUKA APLIKASI ---
st.subheader("1. Unggah Gambar Anda")
uploaded_file = st.file_uploader(
    "Pilih file gambar...",
    type=['png', 'jpg', 'jpeg', 'webp'],
    label_visibility="collapsed"
)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    process_and_display(image)

st.divider()

st.subheader("2. Atau Gunakan Kamera Langsung")
camera_image = st.camera_input(
    "Ambil gambar dengan kamera",
    label_visibility="collapsed"
)
if camera_image is not None:
    image = Image.open(camera_image)
    process_and_display(image)