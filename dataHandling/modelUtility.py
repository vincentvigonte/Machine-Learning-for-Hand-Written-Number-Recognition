import numpy as np
import matplotlib.pyplot as plt

X_train = np.load('C:/Users/Vincent Vigonte/BSCS/GitHub/HandWrittenNumbersRecognition/dataset/X_train.npy')
X_test = np.load('C:/Users/Vincent Vigonte/BSCS/GitHub/HandWrittenNumbersRecognition/dataset/X_test.npy')
y_train = np.load('C:/Users/Vincent Vigonte/BSCS/GitHub/HandWrittenNumbersRecognition/dataset/y_train.npy')
y_test = np.load('C:/Users/Vincent Vigonte/BSCS/GitHub/HandWrittenNumbersRecognition/dataset/y_test.npy')

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

print(f"Sample image shape: {X_train[0].shape}")
print(f"Sample label: {y_train[0]}")

sample_image = X_train[9].reshape(28, 28)
plt.imshow(sample_image, cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()
