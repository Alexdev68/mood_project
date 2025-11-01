from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from database import init_db, save_prediction, get_all_predictions

app = Flask(__name__)

# Load trained CNN model
model = load_model("cnn_emotion_model.h5")

# Emotion labels (adjust if your model uses a different order)
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Upload directory
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize DB
init_db()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        name = request.form.get("name")
        image = request.files["image"]

        if not name or not image:
            result = "Please provide both name and image."
            return render_template("index.html", result=result)

        path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(path)

        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = face_roi / 255.0
                face_roi = np.expand_dims(face_roi, axis=(0, -1))

                preds = model.predict(face_roi)
                emotion = emotion_labels[np.argmax(preds)]
                result = emotion
                break
        else:
            emotion = "No face detected"
            result = emotion

        # Save to database
        save_prediction(name, image.filename, result)

    return render_template("index.html", result=result)

@app.route("/history")
def history():
    records = get_all_predictions()
    return render_template("history.html", records=records)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
