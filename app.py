import os
from flask import Flask, jsonify, request, render_template
import numpy as np
import cv2

app = Flask(__name__)

static_folder = os.path.join(os.getcwd(), 'static')
if not os.path.exists(static_folder):
    os.makedirs(static_folder)

model_filename = 'C:/Users/Vincent Vigonte/BSCS/GitHub/HandWrittenNumbersRecognition/models/knn_model.xml'
knn = cv2.ml.KNearest_load(model_filename)

def preprocess_image(image_data, target_size=(28, 28)):
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Image not found!")
        return None

    original_image_path = os.path.join(static_folder, 'original_image.png')
    cv2.imwrite(original_image_path, img)

    if len(img.shape) == 3 and img.shape[2] == 4:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img_rgb[img[:, :, 3] == 0] = [255, 255, 255]
    else:
        img_rgb = img

    if len(img_rgb.shape) == 3:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_rgb

    _, img_bin = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)

    processed_image_path = os.path.join(static_folder, 'processed_image.png')
    cv2.imwrite(processed_image_path, img_bin)

    return resize_with_aspect_ratio(img_bin, target_size)

def resize_with_aspect_ratio(image, target_size=(28, 28)):
    h, w = image.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    canvas = np.full(target_size, 255, dtype=np.uint8)
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    return canvas

@app.route('/')
def index():
    return render_template('style.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_data = file.read()
    processed_image = preprocess_image(image_data)

    if processed_image is None:
        return jsonify({'error': 'Error processing image'}), 500

    processed_image_flat = processed_image.flatten().astype(np.float32) / 255.0

    ret, result, neighbors, dist = knn.findNearest(np.array([processed_image_flat]), k=5)
    predicted_label = int(result[0][0]) 
    
    return jsonify({
        'prediction': str(predicted_label),
        'original_image': '/static/original_image.png',
        'processed_image': '/static/processed_image.png'
    })

if __name__ == '__main__':
    app.run(debug=True)
