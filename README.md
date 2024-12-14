## Machine Learning for Hand-Written Numbers Recognition

The **Machine Learning for Hand-Written Numbers Recognition** project allows users to recognize hand-written numbers using a k-Nearest Neighbors (k-NN) model. With a user-friendly web interface, users can draw digits on a canvas, and the system predicts the number based on the trained model.

## Features
- **Image Preprocessing**:
  - Removes noise from uploaded images.
  - Resizes images to a standard 28x28 size.
  - Converts images to grayscale for model compatibility.

- **Data Splitting**:
  - Divides data into 80% for training and 20% for testing.

- **Machine Learning Model**:
  - Trains and utilizes a k-NN model for digit recognition.

- **Web Application**:
  - Users can draw numbers on a canvas.
  - Saves the drawn image as a PNG file and predicts the digit.

## How to Use

# Prerequisites
- Python 3.x installed on your machine.
- Basic knowledge of Python and web applications.
  
# Dataset
https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist 

# Installation Steps
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/vincentvigonte/Machine-Learning-for-Hand-Written-Numbers-Recognition.git
   cd Machine-Learning-for-Hand-Written-Numbers-Recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

4. Open your web browser and navigate to `http://127.0.0.1:5000/` to use the application.

## Key Functionalities

# Backend (Flask):
- Processes uploaded images by:
  - Converting them to grayscale.
  - Resizing and normalizing them.
  - Preparing them for prediction using the k-NN model.

- Predicts the digit using the trained model and returns the result.

# Frontend:
- **Drawing Canvas**: Users can draw numbers directly on the web application.
- **Responsive Design**: Built with Tailwind CSS for usability on all devices.

# Future Enhancements
- Support for more advanced models and algorithms.
- Deployment to cloud platforms for wider accessibility.
- Improved frontend design with additional features for users.

# Acknowledgments
This project leverages OpenCV for image processing and Flask for backend deployment. Special thanks to the contributors and the open-source community for their support.

# License
This project is licensed under the MIT License. See the LICENSE file for details.
