# Hand-Written-Recognition-By-Ml
Handwritten Recognition using Machine Learning This project uses machine learning models to detect and convert handwritten text into digital form with high accuracy. It learns from various handwriting styles to improve recognition and performance.

# 🧠 Handwritten Digit Recognition using CNN (TensorFlow)

## 📌 Overview
This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. The model is trained to classify digits (0–9) with high accuracy.

---

## 🚀 Features
- CNN architecture for image recognition
- Automatic feature extraction from handwritten digit images
- High accuracy on MNIST dataset
- Visualization of predictions

---

## 🧪 Dataset
- **Training Images:** 60,000
- **Testing Images:** 10,000
- **Image Size:** 28x28 grayscale
- **Labels:** Digits 0–9

---

## 🏗️ Model Architecture
| Layer Type   | Description                    |
|--------------|-------------------------------|
| Conv2D       | Extracts features from image  |
| MaxPooling2D | Reduces spatial dimensions    |
| Flatten      | Converts 2D features to 1D    |
| Dense Layer  | Learns classification patterns|
| Dropout      | Prevents overfitting          |
| Softmax      | Outputs probability for digits|

---

## 📦 Installation
```bash
pip install tensorflow numpy matplotlib
