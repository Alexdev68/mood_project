import sqlite3

DB_NAME = "emotion_app.db"

def init_db():
    """Create database and table if not exists."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            result TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(name, image_path, result):
    """Save a prediction record."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO users (name, image_path, result)
        VALUES (?, ?, ?)
    """, (name, image_path, result))
    conn.commit()
    conn.close()

def get_all_predictions():
    """Retrieve all records."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name, image_path, result, timestamp FROM users ORDER BY timestamp DESC")
    records = cursor.fetchall()
    conn.close()
    return records
