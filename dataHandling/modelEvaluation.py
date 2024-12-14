import cv2
import numpy as np

model_filename = 'C:/Users/Vincent Vigonte/BSCS/GitHub/HandWrittenNumbersRecognition/models/knn_model.xml'
knn = cv2.ml.KNearest_load(model_filename)

def preprocess_image(image_path, target_size=(28, 28)):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) 
    if img is None:
        print("Image not found!")
        return None

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

image_path = 'C:/Users/Vincent Vigonte/BSCS/GitHub/HandWrittenNumbersRecognition/imgTest/4.png'

processed_image = preprocess_image(image_path)

if processed_image is not None:
    processed_image_flat = processed_image.flatten().astype(np.float32) / 255.0 
    
    ret, results, neighbors, dist = knn.findNearest(np.array([processed_image_flat]), k=5)
    predicted_digit = int(results[0][0])
    print(f"Predicted digit: {predicted_digit}")

 
    input_digit = 4
    print(f"Input digit: {input_digit}")
    accuracy = 100 if predicted_digit == input_digit else 0
    print(f"Prediction accuracy: {accuracy}%")
