# detectors/pedestrian.py
import cv2

def detect_pedestrians(image_path, cascade_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found or cannot be loaded.")

    classifier = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bodies = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))

    for (x, y, w, h) in bodies:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return image
