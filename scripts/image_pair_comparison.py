# Face Recognition using InsightFace
# This script compares two images to determine if they contain the same person using face embeddings.

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the face analysis app
app = FaceAnalysis(name='buffalo_l')  # uses a default model
app.prepare(ctx_id=0)  # use 0 for GPU, -1 for CPU

# Load and process two images
img1 = cv2.imread("samples/sample_tanmay_1.jpeg")
img2 = cv2.imread("samples/sample_tanmay_2.jpeg")

# Detect faces and get embeddings
faces1 = app.get(img1)
faces2 = app.get(img2)

# Check if face detected in both
if len(faces1) == 0 or len(faces2) == 0:
    print("Face not detected in one or both images.")
    exit()

# Get embeddings
embedding1 = faces1[0].embedding
embedding2 = faces2[0].embedding

# Compare embeddings using cosine similarity
similarity = cosine_similarity([embedding1], [embedding2])[0][0]

print(f"Similarity score: {similarity}")

# Set a threshold for recognition (typically around 0.4 to 0.6)
if similarity > 0.6:
    print("Same person ✅")
else:
    print("Different persons ❌")
