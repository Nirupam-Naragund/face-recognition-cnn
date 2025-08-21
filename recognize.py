import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

IMG_SIZE = (160, 160)
model = tf.keras.models.load_model("model/face_cnn.h5")

# Load class indices
import json
with open("class_indices.json") as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
frame_count = 0
MAX_FRAMES = 10  # Set your desired frame limit
CONF_THRESHOLD = 0.5  # You can adjust this value

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    recognized = False  # Track if any face is recognized

    for (x,y,w,h) in faces:
        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, IMG_SIZE)
        roi = img_to_array(roi)/255.0
        roi = np.expand_dims(roi, axis=0)

        pred = model.predict(roi)[0]
        label = class_labels[np.argmax(pred)]
        conf = np.max(pred)

        if conf >= CONF_THRESHOLD:
            print(f"Recognized: {label} ({conf:.2f})")
            recognized = True
        else:
            print("Not recognized")

        text = f"{label} ({conf:.2f})" if conf >= CONF_THRESHOLD else "Not recognized"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    if not recognized and len(faces) == 0:
        print("No face detected")

    cv2.imshow("Face Recognition", frame)

    frame_count += 1
    if frame_count >= MAX_FRAMES or cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
