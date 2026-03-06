# Sentiment analysis and face recognition using InsightFace and DeepFace


import cv2
import os
import numpy as np
from numpy.linalg import norm
import insightface
from insightface.app import FaceAnalysis
from deepface import DeepFace

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for GPU, -1 for CPU

# Load known faces
known_faces_dir = 'data/known_faces'
known_embeddings = []
known_names = []

print("[INFO] Loading known faces...")

for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(known_faces_dir, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        faces = app.get(img)
        if faces:
            embedding = faces[0].embedding
            name = os.path.splitext(filename)[0]
            known_embeddings.append(embedding)
            known_names.append(name)
            print(f"✅ Loaded: {name}")
        else:
            print(f"⚠️ No face found in: {filename}")

# Face recognition function
def recognize_face(embedding, threshold=0.45):
    if not known_embeddings:
        return "Unknown", 0.0
    similarities = [np.dot(embedding, db_emb) / (norm(embedding) * norm(db_emb))
                    for db_emb in known_embeddings]
    best_idx = int(np.argmax(similarities))
    best_score = similarities[best_idx]
    if best_score > threshold:
        return known_names[best_idx], best_score
    return "Unknown", best_score

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        embedding = face.embedding

        # Recognize face
        name, score = recognize_face(embedding)

        # Crop face region for emotion analysis
        x1, y1, x2, y2 = box
        cropped_face = frame[y1:y2, x1:x2]

        # Emotion analysis
        try:
            analysis = DeepFace.analyze(cropped_face, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
        except:
            emotion = "N/A"

        # Draw results
        label = f"{name} ({score:.2f}) | {emotion}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Face Recognition + Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
