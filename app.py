from flask import Flask, render_template, request, Response, redirect, url_for
import sqlite3
import cv2
import numpy as np
import os
import tensorflow as tf
from database import init_db, save_prediction, get_all_predictions

app = Flask(__name__)

# ======== Load TensorFlow Lite Model ========
interpreter = tf.lite.Interpreter(model_path="face_emotion_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Emotion labels (must match dataset order)
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Uploads folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database
init_db()


def predict_emotion(face):
    """Run emotion prediction using the TFLite model (RGB input)."""
    try:
        face = cv2.resize(face, (96, 96))
        face = face / 255.0
        face = np.expand_dims(face, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], face)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])

        conf = np.max(preds)
        if conf < 0.35:  # slightly higher threshold to reduce false predictions
            return "Uncertain"
        return emotion_labels[np.argmax(preds)]
    except Exception as e:
        print("Prediction error:", e)
        return "Error"


@app.route("/", methods=["GET", "POST"])
def index():
    """Image upload and emotion detection."""
    result = None
    if request.method == "POST":
        name = request.form.get("name")
        image = request.files.get("image")

        if not name or not image:
            return render_template("index.html", result="Please provide both name and image.")

        path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(path)

        img = cv2.imread(path)
        if img is None:
            return render_template("index.html", result="Invalid image uploaded.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_roi = img[y:y+h, x:x+w]  # use color ROI (RGB expected)
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                emotion = predict_emotion(face_roi)
                result = emotion
                break
        else:
            result = "No face detected"

        if result not in ["No face detected", "Error"]:
            save_prediction(name, image.filename, result)

    return render_template("index.html", result=result)


def generate_frames(user_name="Anonymous"):
    """Stream live webcam frames with emotion detection."""
    cap = cv2.VideoCapture(0)
    frame_count = 0

    if not cap.isOpened():
        print("⚠️ Unable to access webcam.")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                emotion = predict_emotion(face_roi)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Save one frame every ~1 second
                if frame_count % 30 == 0 and emotion not in ["Uncertain", "Error"]:
                    filename = f"live_{frame_count}.jpg"
                    save_path = os.path.join(UPLOAD_FOLDER, filename)
                    cv2.imwrite(save_path, frame)
                    save_prediction(user_name, filename, emotion)

            frame_count += 1
            if frame_count > 3000:
                frame_count = 0  # prevent overflow

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()


@app.route("/live")
def live():
    user_name = request.args.get("name", "Anonymous")
    return Response(generate_frames(user_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/history")
def history():
    records = get_all_predictions()
    return render_template("history.html", records=records)


@app.route("/clear_history", methods=["POST"])
def clear_history():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return redirect(url_for("history"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
