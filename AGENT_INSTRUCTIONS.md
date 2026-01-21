# Project Overview: Video Face Tracker at Scale

## Usage Context for AI Agents

This repository is designed to be a scalable system for tracking individuals across multiple video sources.

**The User's End Goal:**

- Track **10,000+ distinct individuals**.
- Process **1,000+ separate video incidents**.
- Link identities across these videos (e.g., "Person A in Video 1 is the same as Person X in Video 50").
- Create comprehensive profiles for each individual including face images and appearance timestamps.

## Current Architecture

- **Language**: Python 3.9+
- **Face Detection**: `face_recognition` (dlib HOG/CNN) & `opencv-python`.
- **Storage**: `sqlite3` (schema in `database.py`).
- **Clustering**: `scikit-learn` DBSCAN for grouping faces within a video.

## Scaling Instructions

If you are an AI agent picking this up for *optimization* or *scaling*:

1. **Database**: The current SQLite DB may eventually need migration to a Vector Database (e.g., Milvus, Pinecone, or pgvector) to handle 10k+ embedding comparisons efficiently.
2. **Detection**: `face_recognition` is accurate but slow on CPU. Consider upgrading to `InsightFace` (ONNX) for faster inference or implementing GPU acceleration.
3. **Clustering**: Naive DBSCAN on 10,000 vectors might be slow. Consider hierarchical clustering or index-based retrieval.

## Key Files

- `main.py`: Entry point.
- `processor.py`: Frame extraction and encoding.
- `profiler.py`: Logic for "Who is this?" (Matching/Creating Profiles).
