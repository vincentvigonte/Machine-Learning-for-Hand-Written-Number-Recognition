import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

dataset_path = "C:/Users/Vincent Vigonte/BSCS/GitHub/HandWrittenNumbersRecognition/dataset"
processed_path = "processedData" 

def load_images_from_directory(base_path, target_size=(28, 28), output_dir="processed"):
    images = []
    labels = []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label in range(10):
        folder_path = os.path.join(base_path, str(label))
        output_folder = os.path.join(output_dir, str(label))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        if not os.path.exists(folder_path):
            continue

        image_files = os.listdir(folder_path)
        print(f"Found {len(image_files)} images in folder {folder_path}")

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) 
            if img is None:
                continue

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

            if img_bin.shape != target_size:
                processed_img = resize_with_aspect_ratio(img_bin, target_size)
            else:
                processed_img = img_bin 

            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, processed_img)

            images.append(processed_img.flatten())
            labels.append(label)
    
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

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

def normalize_data(images):
    return images / 255.0

def main():
    print("Loading and preprocessing dataset...")
    images, labels = load_images_from_directory(dataset_path, output_dir=processed_path)
    images = normalize_data(images)
    print(f"Loaded {len(images)} images and saved to {processed_path}/.")

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    print("Data saved successfully in .npy format!")

if __name__ == "__main__":
    main()
