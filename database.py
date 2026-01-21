import sqlite3
from datetime import datetime
import json
import numpy as np

DB_NAME = "data/face_tracker.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    # Enable foreign keys
    c.execute("PRAGMA foreign_keys = ON;")

    # Table: Videos
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT UNIQUE NOT NULL,
            processed_date TEXT
        )
    ''')

    # Table: Persons
    # Embedding is stored as a JSON list or blob. Here we use JSON string for simplicity.
    c.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            embedding_json TEXT
        )
    ''')

    # Table: Appearances
    c.execute('''
        CREATE TABLE IF NOT EXISTS appearances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            person_id INTEGER,
            timestamp REAL,
            image_path TEXT,
            FOREIGN KEY(video_id) REFERENCES videos(id),
            FOREIGN KEY(person_id) REFERENCES persons(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def add_video(filepath):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT OR IGNORE INTO videos (filepath, processed_date) VALUES (?, ?)", 
                  (filepath, datetime.now().isoformat()))
        conn.commit()
        # Get ID
        c.execute("SELECT id FROM videos WHERE filepath = ?", (filepath,))
        return c.fetchone()['id']
    finally:
        conn.close()

def create_person(embedding):
    """
    Creates a new person profile. 
    embedding: numpy array or list of 128 floats
    """
    conn = get_connection()
    c = conn.cursor()
    emb_json = json.dumps(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
    c.execute("INSERT INTO persons (name, embedding_json) VALUES (?, ?)", ("Unknown", emb_json))
    pid = c.lastrowid
    # Update name to 'Person {id}' by default
    c.execute("UPDATE persons SET name = ? WHERE id = ?", (f"Person {pid}", pid))
    conn.commit()
    conn.close()
    return pid

def log_appearance(video_id, person_id, timestamp, image_path):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO appearances (video_id, person_id, timestamp, image_path)
        VALUES (?, ?, ?, ?)
    ''', (video_id, person_id, timestamp, image_path))
    conn.commit()
    conn.close()

def get_all_persons():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, embedding_json FROM persons")
    return c.fetchall()
