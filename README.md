# 🧠 FaceTrack AI

**Face Recognition Attendance System & Smart Curriculum Tracker (Prototype)**

A prototype combining **Face Detection**, **Face Recognition**, and **Behavioral Analysis** to automate attendance and track classroom engagement using AI and computer vision.

---

## ✨ Features

- 🔒 **Face Detection & Recognition** — Using InsightFace (ArcFace) with real-time webcam feed
- 🧠 **Sentiment & Emotion Analysis** — Understand student reactions via DeepFace
- 📈 **Engagement Tracking** — Monitor attention span, expressions, and interaction levels
- 🖼️ **Live Camera Feed** — Real-time detection overlay with bounding boxes and labels

---

## 📁 Project Structure

```
facetrack-ai/
├── scripts/
│   ├── realtime_face_match.py        # Real-time face matching (single reference)
│   ├── image_pair_comparison.py      # Compare two images for same person
│   ├── multi_face_recognition.py     # Multi-person recognition from database
│   └── face_recognition_emotion.py   # Face recognition + emotion detection
├── data/
│   └── known_faces/                  # Database of known face images
├── samples/                          # Sample test images
├── demos/                            # Demo screenshots and videos
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Webcam (for real-time scripts)
- GPU recommended (but CPU works with `ctx_id=-1`)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd facetrack-ai

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Adding Known Faces

Place face images in the `data/known_faces/` directory. Each image should:
- Contain a single, clearly visible face
- Be named after the person (e.g., `john.jpeg`, `jane.png`)

---

## 📜 Scripts

### 1. `realtime_face_match.py`
Captures webcam video and matches detected faces against a single reference image.

```bash
python scripts/realtime_face_match.py
```

### 2. `image_pair_comparison.py`
Compares two static images to determine if they contain the same person.

```bash
python scripts/image_pair_comparison.py
```

### 3. `multi_face_recognition.py`
Real-time multi-person recognition using a database of known faces from `data/known_faces/`.

```bash
python scripts/multi_face_recognition.py
```

### 4. `face_recognition_emotion.py`
Combines face recognition with emotion/sentiment analysis using DeepFace.

```bash
python scripts/face_recognition_emotion.py
```

> **Note:** Press `q` to quit any real-time webcam script.

---

## ⚠️ Current Status

- ✅ Functional Prototype
- ⚠️ Uses pre-built libraries for rapid prototyping
- 🚧 Not optimized for deployment or real-world performance
- 🧪 Experimental features like behavior and efficiency tracking

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| [OpenCV](https://opencv.org/) | Image processing & webcam capture |
| [InsightFace](https://github.com/deepinsight/insightface) | Face detection & recognition (ArcFace) |
| [DeepFace](https://github.com/serengil/deepface) | Emotion & sentiment analysis |
| [scikit-learn](https://scikit-learn.org/) | Cosine similarity computation |
| [NumPy](https://numpy.org/) | Numerical operations |
