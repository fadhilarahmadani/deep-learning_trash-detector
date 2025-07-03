from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import shutil
import cv2 # Import OpenCV

app = Flask(__name__)

# --- Konfigurasi dan Setup Awal (Sama seperti sebelumnya) ---
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# Buat folder jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load model YOLOv8
model = YOLO("/pt-model/model.pt")

# Inisialisasi kamera
# Angka 0 berarti menggunakan webcam default. Jika Anda punya banyak kamera, bisa jadi 1, 2, dst.
camera = cv2.VideoCapture(0)


# --- Fungsi untuk Streaming Kamera ---
def generate_frames():
    while True:
        # Baca frame dari kamera
        success, frame = camera.read()
        if not success:
            break
        else:
            # Jalankan deteksi YOLO pada frame
            # stream=True lebih efisien untuk video
            results = model.predict(frame, stream=True)

            # Loop melalui hasil deteksi dan gambar kotak prediksinya
            for r in results:
                # .plot() akan menggambar kotak deteksi langsung pada frame
                frame = r.plot()

            # Encode frame ke format JPEG
            ret, buffer = cv2.imencode('.jpg','.png', frame)
            frame_bytes = buffer.tobytes()

            # Yield frame dalam format http multipart
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# --- Rute Aplikasi ---

# Halaman utama untuk upload file
@app.route('/')
def index():
    return render_template('index.html')

# Rute untuk memproses file upload (Tidak diubah)
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    results = model.predict(source=upload_path, task='obb', save=True)
    result_dir = results[0].save_dir
    result_img_path = os.path.join(result_dir, filename)
    result_static_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    
    shutil.copy(result_img_path, result_static_path)
    
    return render_template('result.html', filename=filename)


# --- Rute BARU untuk Halaman Kamera dan Streaming ---

# Halaman yang akan menampilkan video stream dari kamera
@app.route('/camera_feed')
def camera_feed():
    return render_template('camera.html')

# Rute khusus yang akan menyediakan stream video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)