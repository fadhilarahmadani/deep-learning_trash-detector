import os
import cv2
from flask import Flask, render_template, request, Response, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- KONFIGURASI ---
# Tentukan path untuk folder upload dan hasil
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
# Pastikan folder-folder tersebut ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Tentukan format file gambar yang diizinkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Muat model YOLOv8 (ganti 'best.pt' dengan nama file model Anda)
try:
    model = YOLO('pt-model/model.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    # Anda bisa memutuskan untuk keluar dari aplikasi jika model gagal dimuat
    # exit()

# --- FUNGSI BANTUAN ---
def allowed_file(filename):
    """Memeriksa apakah ekstensi file diizinkan."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- RUTE-RUTE APLIKASI ---
@app.route('/')
def index():
    """Menampilkan halaman utama (index.html)."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Menangani upload file dan melakukan prediksi."""
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']

    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Lakukan deteksi pada gambar yang diunggah
        img = cv2.imread(filepath)
        results = model(img)
        
        # Render hasil dan simpan gambar baru
        results.render()
        output_filename = 'result_' + filename
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        cv2.imwrite(output_path, results.imgs[0])

        return render_template('result.html', filename=output_filename)

def generate_frames():
    """Menghasilkan frame video dari kamera untuk streaming."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Kamera tidak bisa dibuka.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Lakukan deteksi pada setiap frame
        results = model(frame)
        annotated_frame = results[0].plot() # Menggunakan .plot() yang lebih baru

        # Encode frame ke format JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        # Kirim frame sebagai respons HTTP
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    """Rute untuk video streaming."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_feed')
def camera_page():
    """Menampilkan halaman untuk deteksi via kamera."""
    return render_template('camera.html')

# --- MENJALANKAN APLIKASI ---
if __name__ == '__main__':
    app.run(debug=True)