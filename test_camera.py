from src.biometric.face_detection import FaceDetector
import cv2

detector = FaceDetector(backend="opencv")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Detect faces on smaller frame
    faces = detector.detect_faces(small_frame)

    # Scale faces back to original size
    scaled_faces = []
    for (x, y, w, h) in faces:
        scaled_faces.append((x * 2, y * 2, w * 2, h * 2))

    # Draw on original frame
    frame = detector.visualize_detections(frame, scaled_faces)

    # SHOW WINDOW (correct indentation)
    cv2.imshow("Face Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()