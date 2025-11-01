# 🧠 Mood Detector — Emotion Recognition Web App

A deep learning–powered **emotion recognition system** that detects human moods from facial images using **CNN (Convolutional Neural Networks)**.  
The project includes a Flask web interface for image upload, real-time emotion prediction, and automatic storage of results in a local database.

---

## 🚀 Features

- 📸 Upload an image to detect emotions  
- 🧠 CNN model (trained on FER2013 dataset or custom dataset)  
- 💾 Automatically stores:
  - User name  
  - Uploaded image  
  - Model prediction (detected mood)  
- 🌐 Works both online and offline  
- 📊 History page showing previous results  
- 🧱 Built with Flask, TensorFlow/Keras, and SQLite  

---

## 🧩 Tech Stack

| Category | Technology |
|-----------|-------------|
| **Frontend** | HTML5, CSS3 (Jinja templates) |
| **Backend** | Python (Flask) |
| **Machine Learning** | TensorFlow / Keras (MobileNetV2 Transfer Learning) |
| **Database** | SQLite3 |
| **Deployment** | Render |

---

## 🗂 Project Structure

```
mood_project/
│
├── app.py                  # Flask web server
├── database.py             # Database functions (SQLite)
├── cnn_emotion_model.h5    # Trained CNN model
├── requirements.txt        # Dependencies
├── Procfile                # Render startup instruction
├── render.yaml             # Optional Render configuration
│
├── templates/
│   ├── index.html          # Main upload page
│   └── history.html        # Displays stored records
│
├── static/
│   └── uploads/            # Uploaded user images
│
└── mood_data.db            # Auto-generated SQLite database
```

---

## ⚙️ Installation (Local Setup)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Alexdev68/mood_project.git
cd mood_project
```

### 2️⃣ Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Flask app
```bash
python app.py
```

Then visit: 👉 [http://localhost:5000](http://localhost:5000)

---

## ☁️ Deployment (Render)

1. Push your project to your GitHub repository.  
2. Go to [Render.com](https://render.com).  
3. Create a new **Web Service** and connect your repo.  
4. Render auto-detects Flask if:
   - `requirements.txt` is present  
   - `Procfile` contains:  
     ```
     web: gunicorn app:app
     ```
5. Once deployed, your app will be live online 🚀

---

## 🧠 Model Information

- **Architecture:** CNN (Convolutional Neural Network)
- **Base:** MobileNetV2 (Transfer Learning)
- **Input:** 48x48 grayscale or RGB images
- **Output:** Emotion classes (happy, sad, angry, neutral, surprise, disgust, fear)
- **Accuracy:** ~70–80% (depends on dataset quality)

---

## 📸 Sample Workflow

1. User uploads an image  
2. The CNN model predicts the emotion  
3. Result (username, image, emotion) is stored in `mood_data.db`  
4. The user can view all previous detections on the **History** page  

---

## 🤝 Contributing

Pull requests are welcome!  
For major changes, please open an issue first to discuss what you would like to improve.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 💡 Acknowledgments

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [TensorFlow](https://www.tensorflow.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Render](https://render.com/)

---

### 👨‍💻 Author

**Anachebe Ikechukwu**  
💬 _“Turning emotions into data, one face at a time.”_  
📧 **anachebeikechukwu68@gmail.com**  
🔗 [GitHub](https://github.com/Alexdev68)
