import numpy as np
import base64
import cv2
import os
import gdown
import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

# =========================================
# INIT APP
# =========================================
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# DOWNLOAD MODEL DARI GDRIVE
# =========================
MODEL_PATH = "model_sampah.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/file/d/1fs5cqFvZyXorbs6fZaxWvGOrKtFfodlb/view?usp=sharing"
    gdown.download(url, MODEL_PATH, quiet=False)
    
# =========================================
# LOAD MODEL
# =========================================
model = load_model('model_sampah.h5', compile=False)

# =========================================
# FUNCTION PREPROCESS (SAMA DENGAN COLAB)
# =========================================
def preprocess_image(img):
    # OpenCV BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize sama seperti training
    img = cv2.resize(img, (150,150))

    # Normalisasi
    img = img.astype("float32") / 255.0

    # Expand dim
    img = np.expand_dims(img, axis=0)

    return img

# =========================================
# FUNCTION PREDIKSI
# =========================================
def predict_image(img):
    processed = preprocess_image(img)

    pred = model.predict(processed)[0][0]

    print("DEBUG NILAI PRED:", pred)  # penting buat cek

    if pred > 0.5:
        return "Anorganik (R)", float(pred)
    else:
        return "Organik (O)", float(1 - pred)

# =========================================
# ROUTE HOME
# =========================================
@app.route('/')
def index():
    return render_template('index.html')

# =========================================
# ROUTE WEBCAM
# =========================================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data['image']

    # decode base64
    img_bytes = base64.b64decode(img_data.split(',')[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    label, conf = predict_image(img)

    return jsonify({
        'label': label,
        'confidence': conf
    })

# =========================================
# ROUTE UPLOAD
# =========================================
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        label, conf = predict_image(img)

        return jsonify({
            'label': label,
            'confidence': conf,
            'img_path': filepath
        })

    return jsonify({'error': 'No file'})

# =========================================
# RUN
# =========================================
#if __name__ == '__main__':
    #app.run(debug=True)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)