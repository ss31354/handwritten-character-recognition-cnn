# Handwritten Character Recognition using CNN

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to recognize handwritten digits and characters.

The model is trained using the **EMNIST dataset** and deployed with a **Flask web application** that allows users to upload handwritten images and receive predictions.

---

## Features

- CNN model trained on EMNIST dataset
- Predict handwritten digits and characters
- Flask web interface
- Image preprocessing pipeline
- Canvas where you draw to predict the digit/letter


---

## Technologies

- Python
- PyTorch
- Flask
- NumPy
- OpenCV
- Matplotlib

---

## Installation

Clone the repository:
git clone https://github.com/ss31354/handwritten-character-recognition-cnn.git


cd handwritten-character-recognition-cnn

Install dependencies:


pip install -r requirements.txt

## Run the Application


cd app


python app.py


Then open:


http://127.0.0.1:5000


Upload a handwritten character image to see the prediction or draw on the canvas to let it guess for you and show you the probability of the guess.



---

## Author

Salman Syed
