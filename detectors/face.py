import cv2

def detect_faces(image_path, cascade_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found or cannot be loaded.")

    face_classifier = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)

    return image
