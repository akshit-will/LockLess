import cv2
import sys, os
sys.path.append(os.path.abspath("LockLess/src"))

from src.biometric.face_detection import FaceDetector
detector = FaceDetector(backend="opencv")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect_faces(frame)

    print("Faces detected:", len(faces))  # 👈 IMPORTANT

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()