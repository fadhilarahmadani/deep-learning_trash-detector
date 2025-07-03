import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2  # FIX 1: Tambahkan import cv2
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Deteksi Sampah",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- JUDUL APLIKASI ---
st.title("🌿 Sistem Deteksi Sampah")
st.write("Unggah gambar atau gunakan kamera untuk mendeteksi jenis sampah secara real-time.")

# --- MEMUAT MODEL (dengan cache agar lebih cepat) ---
# Ganti 'best.pt' dengan path dan nama file model Anda
@st.cache_resource
def load_model(model_path):
    """Memuat model YOLO dari path yang diberikan."""
    model = YOLO(model_path)
    return model

# Panggil fungsi untuk memuat model
try:
    model = load_model('pt-model/best.pt')
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'pt-model/best.pt' ada. Error: {e}")
    st.stop()


# --- FUNGSI UNTUK PROSES DETEKSI ---
def process_and_display(image):
    """Melakukan deteksi pada gambar dan menampilkannya."""
    # Tampilkan gambar asli
    st.image(image, caption="Gambar Asli", use_container_width=True) # FIX 2: Ganti ke use_container_width

    # Lakukan deteksi
    results = model(image)
    
    # Dapatkan gambar hasil deteksi
    annotated_image = results[0].plot()
    
    # Konversi warna dari BGR (OpenCV) ke RGB (Pillow/Streamlit)
    annotated_image_rgb = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    
    # Tampilkan gambar hasil deteksi
    st.image(annotated_image_rgb, caption="Hasil Deteksi", use_container_width=True) # FIX 2: Ganti ke use_container_width


# --- ANTARMUKA UPLOAD GAMBAR ---
st.subheader("1. Unggah Gambar Anda")
uploaded_file = st.file_uploader(
    "Pilih file gambar...",
    type=['png', 'jpg', 'jpeg', 'webp'],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # Buka gambar menggunakan Pillow
    image = Image.open(uploaded_file)
    process_and_display(image)

st.divider()

# --- ANTARMUKA KAMERA ---
st.subheader("2. Atau Gunakan Kamera Langsung")
camera_image = st.camera_input(
    "Ambil gambar dengan kamera",
    label_visibility="collapsed"
)

if camera_image is not None:
    # Buka gambar dari kamera menggunakan Pillow
    image = Image.open(camera_image)
    process_and_display(image)