# detectors/eye.py
import cv2

def detect_eyes(image_path, face_cascade_path, eye_cascade_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found or cannot be loaded.")

    face_classifier = cv2.CascadeClassifier(face_cascade_path)
    eye_classifier = cv2.CascadeClassifier(eye_cascade_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

    return image
