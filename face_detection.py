import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm

KNOWN_DIR = "task5/images/"
THRESHOLD = 0.5
FRAME_SKIP = 5  

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


known_embeddings, known_names = [], []

for file in os.listdir(KNOWN_DIR):
    img = cv2.imread(os.path.join(KNOWN_DIR, file))
    if img is None:
        continue

    faces = app.get(img)

    if faces:
        known_embeddings.append(faces[0].embedding)
        known_names.append(os.path.splitext(file)[0])

print(f"Loaded {len(known_names)} known faces")


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


cap = cv2.VideoCapture(0)
frame_count = 0
last_faces = []   

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        faces = app.get(frame)
        last_faces = faces

    else:
        faces = last_faces

    for face in faces:
        bbox = face.bbox.astype(int)
        emb = face.embedding

        similarities = [
            cosine_similarity(emb, known_emb)
            for known_emb in known_embeddings
        ]

        if similarities:
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            name = known_names[best_idx] if best_score >= THRESHOLD else "Unknown"

        else:
            name = "Unknown"
            best_score = 0

        cv2.rectangle( frame,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(0, 255, 0),2)

        cv2.putText( frame,f"{name} ({best_score:.2f})",(bbox[0], bbox[1] - 10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2 )

    cv2.imshow("ArcFace Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
