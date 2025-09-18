# Real-Time Face Recognition using InsightFace and OpenCV
# This script captures video from the webcam, detects faces, and compares them to a reference image
# to identify if the person matches the known individual.

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare the face recognition model
app = FaceAnalysis(name='buffalo_l')  # You can try 'buffalo_m' or 'buffalo_s' for smaller models
app.prepare(ctx_id=0)  # Set to -1 for CPU, 0 for GPU

# Load reference image (known person)
ref_img = cv2.imread("Tanmay1.jpeg")  # Replace with your reference image path
ref_faces = app.get(ref_img)

if len(ref_faces) == 0:
    print("No face detected in the reference image.")
    exit()

ref_embedding = ref_faces[0].embedding  # Get embedding of the known person

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        # Draw bounding box
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Compare with reference embedding
        similarity = cosine_similarity([ref_embedding], [face.embedding])[0][0]

        label = f"Similarity: {similarity:.2f}"
        if similarity > 0.6:
            label = "MATCH "
        else:
            label = "NO MATCH "

        # Display label
        cv2.putText(frame, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Face Recognition", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
