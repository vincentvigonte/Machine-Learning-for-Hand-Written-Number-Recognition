import cv2
import numpy as np
import os
import joblib

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print("Training the k-NN classifier...")
knn = cv2.ml.KNearest_create()
knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
print("Training complete.")

model_filename = 'models/knn_model.xml'
os.makedirs('models', exist_ok=True)
knn.save(model_filename)
print(f"Model saved to {model_filename}")

print("Testing the model...")
ret, results, neighbors, dist = knn.findNearest(X_test, k=5)

accuracy = np.mean(results.flatten() == y_test) * 100
print(f"Model accuracy: {accuracy:.2f}%")
