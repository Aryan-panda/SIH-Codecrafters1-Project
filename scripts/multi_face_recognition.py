# Real-time Face Recognition using InsightFace and OpenCV
# This script captures video from the webcam, detects faces, and recognizes them by
# comparing against a database of known faces stored in a directory.

import cv2
import os
import numpy as np
from numpy.linalg import norm
import insightface
from insightface.app import FaceAnalysis

# Initialize InsightFace (face detection + recognition)
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 uses GPU if available, -1 for CPU

# Directory containing known faces
known_faces_dir = 'data/known_faces'

# Store embeddings and corresponding names
known_embeddings = []
known_names = []

print("[INFO] Loading known faces...")

for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(known_faces_dir, filename)
        img = cv2.imread(path)

        faces = app.get(img)
        if faces:
            embedding = faces[0].embedding
            name = os.path.splitext(filename)[0]
            known_embeddings.append(embedding)
            known_names.append(name)
            print(f"  ✅ Loaded: {name}")
        else:
            print(f"  ⚠️ No face found in: {filename}")

print(f"[INFO] Loaded {len(known_names)} known faces.")

# Cosine similarity-based recognition
def recognize_face(embedding, threshold=0.45):  # Recommended threshold range: 0.4–0.6
    if not known_embeddings:
        return "Unknown", 0.0

    similarities = [np.dot(embedding, db_emb) / (norm(embedding) * norm(db_emb)) for db_emb in known_embeddings]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score > threshold:
        return known_names[best_idx], best_score
    else:
        return "Unknown", best_score

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame.")
        break

    # Detect faces in the current frame
    faces = app.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        embedding = face.embedding

        name, score = recognize_face(embedding, threshold=0.45)

        # Draw bounding box
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Display name and score
        label = f"{name} ({score:.2f})" if name != "Unknown" else "Unknown"
        cv2.putText(frame, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow("InsightFace Recognition", frame)

    # Exit loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("[INFO] Program terminated.")
