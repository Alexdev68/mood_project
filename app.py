from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from database import init_db, save_prediction, get_all_predictions

app = Flask(__name__)

# Load trained CNN model
model = load_model("face_emotion_model.h5")

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
    """Image upload and emotion detection."""
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


# ======== LIVE DETECTION FEATURE ========

def generate_frames(user_name="Anonymous"):
    """Stream live webcam frames with emotion detection."""
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi / 255.0
            face_roi = np.expand_dims(face_roi, axis=(0, -1))

            preds = model.predict(face_roi)
            emotion = emotion_labels[np.argmax(preds)]

            # Draw results on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Save one frame every 30 frames (roughly 1 second)
            if frame_count % 30 == 0:
                filename = f"live_{frame_count}.jpg"
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                cv2.imwrite(save_path, frame)
                save_prediction(user_name, filename, emotion)

        frame_count += 1

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route("/live")
def live():
    """Live emotion detection using webcam."""
    user_name = request.args.get("name", "Anonymous")
    return Response(generate_frames(user_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/history")
def history():
    """View saved predictions."""
    records = get_all_predictions()
    return render_template("history.html", records=records)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
