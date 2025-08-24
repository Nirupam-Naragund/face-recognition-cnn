# Face Recognition CNN 🎭

A deep learning project for **real-time face recognition** using a Convolutional Neural Network (CNN) built on **TensorFlow/Keras** and **OpenCV**.  

This project can:
- Train a CNN model on custom face datasets
- Recognize faces in real-time from a webcam
- Output the person’s name on the video feed

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Nirupam-Naragund/face-recognition-cnn.git
cd face-recognition-cnn
```
### 2. Create a virtual environment (Python 3.11 recommended)
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Linux/Mac
```
### 3. Install dependencies

```bash
pip install --upgrade pip
pip install tensorflow keras opencv-python numpy pillow scikit-learn matplotlib
```

### 4. Dataset Preparation

Inside the data/ folder, create subfolders for each person you want to train on:

```
data/
│── Brad Pitt/
│    ├── img1.jpg
│    ├── img2.png
│── Angelina Jolie/
     ├── img1.jpg
     ├── img2.png
```

### 5. Train the model

```bash
python train.py
```

### 6.Recognize 

```bash
python recognize.py
```



⚠️ Notes

Make sure your webcam access is enabled in Windows Privacy Settings.

More training images per person → better accuracy.

Works best with well-lit, frontal face images.

